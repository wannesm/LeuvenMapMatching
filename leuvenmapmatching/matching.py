# encoding: utf-8
"""
leuvenmapmatching.matching
~~~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2015-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
from __future__ import print_function

import math
import sys
import logging
import time
from collections import OrderedDict, defaultdict, namedtuple
from typing import Optional, Set

import numpy as np
from scipy.stats import halfnorm, norm

from . import util
from .util import Segment, approx_equal


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")
approx_value = 0.0000000001
ema_const = namedtuple('EMAConst', ['prev', 'cur'])(0.9, 0.1)
default_label_width = 25


class Matching(object):
    """Matching object that represents a node in the Viterbi lattice.

    To enable alternative emission and transition functions overwrite the following functions:
    - `logprob_trans(self, next_label=None, next_latlon=None)`
    - `logprob_obs(self, dist, edge_m, edge_o)`
    - `logprob_obs_ne(self, dist, edge_m, edge_o)`

    If a deeper change is required, overwrite the `next` function.
    """
    __slots__ = ['matcher', 'edge_m', 'edge_o', 'logprob', 'logprobema', 'obs', 'obs_ne', 'dist_obs',
                 'prev', 'prev_other', 'stop', 'length']

    def next(self, edge_m: Segment, edge_o: Segment, obs: int=0, obs_ne: int=0, only_edges=False):
        """Create a next Matching object with this Matching object as the previous one."""
        new_stop = False
        if edge_m.is_point() and edge_o.is_point():
            # node to node
            dist = self.matcher.map.distance(edge_m.p1, edge_o.p1)
            proj_m = edge_m.p1
        elif edge_m.is_point() and not edge_o.is_point():
            # node to edge
            dist, proj_o, t_o = self.matcher.map.distance_point_to_segment(edge_m.p1, edge_o.p1, edge_o.p2)
            proj_m = edge_m.p1
            edge_o.pi = proj_o
            edge_o.ti = t_o
        elif not edge_m.is_point() and edge_o.is_point():
            # edge to node
            dist, proj_m, t_m = self.matcher.map.distance_point_to_segment(edge_o.p1, edge_m.p1, edge_m.p2)
            if only_edges and (approx_equal(t_m, 0.0) or approx_equal(t_m, 1.0)):
                if __debug__ and logger.isEnabledFor(logging.DEBUG):
                    new_stop = True
                else:
                    return None
            edge_m.pi = proj_m
            edge_m.ti = t_m
        elif not edge_m.is_point() and not edge_o.is_point():
            # edge to edge
            dist, proj_m, proj_o, t_m, t_o = self.matcher.map.distance_segment_to_segment(edge_m.p1, edge_m.p2,
                                                                                          edge_o.p1, edge_o.p2)
            edge_m.pi = proj_m
            edge_m.ti = t_m
            edge_o.pi = proj_o
            edge_o.ti = t_o
        else:
            raise Exception(f"Should not happen")

        logprob_trans = self.logprob_trans(edge_m.label, proj_m)
        if obs_ne == 0:
            logprob_obs = self.logprob_obs(dist, edge_m, edge_o)
        else:
            logprob_obs = self.logprob_obs_ne(dist, edge_m, edge_o)
        new_logprob_delta = logprob_trans + logprob_obs
        if edge_m.ti < self.edge_m.ti:
            # This node would imply going back on the edge between observations
            new_logprob_delta + math.log(0.9999)  # slight preference to avoid going back
        new_logprob = self.logprob + new_logprob_delta
        new_logprobema = ema_const.cur * new_logprob_delta + ema_const.prev * self.logprobema
        new_stop |= self.matcher.do_stop(new_logprobema, dist)
        if not new_stop or (__debug__ and logger.isEnabledFor(logging.DEBUG)):
            m_next = self.__class__(self.matcher, edge_m, edge_o,
                                    logprob=new_logprob, logprobema=new_logprobema,
                                    obs=obs, obs_ne=obs_ne, prev={self}, dist_obs=dist,
                                    stop=new_stop, length=self.length + 1)
            return m_next
        else:
            return None

    def __init__(self, matcher: 'Matcher', edge_m: Segment, edge_o: Segment,
                 logprob=-np.inf, logprobema=-np.inf, dist_obs: float=0.0,
                 obs: int=0, obs_ne: int=0,
                 prev: Optional[Set['Matching']]=None, stop: bool=False, length: int=1):
        self.edge_m: Segment = edge_m
        self.edge_o: Segment = edge_o
        self.logprob: float = logprob        # max probability
        self.logprobema: float = logprobema  # exponential moving average log probability
        self.obs: int = obs  # reference to path entry index (observation)
        self.obs_ne: int = obs_ne  # number of non-emitting states for this observation
        self.dist_obs: float = dist_obs  # Distance between map point and observation
        self.prev: Set[Matching] = set() if prev is None else prev  # Previous best matching objects
        self.prev_other: Set[Matching] = set()  # Previous matching objects with lower logprob
        self.stop: bool = stop
        self.length: int = length
        self.matcher: Matcher = matcher

    def update(self, m_next):
        """Update the current entry if the new matching object is better.

        :param m_next: The new matching object representing the same node in the lattice.
        :return: True if the current object is replaced, False otherwise
        """
        if self.length != m_next.length:
            slogprob_norm = self.logprob / self.length
            nlogprob_norm = m_next.logprob / m_next.length
        else:
            slogprob_norm = self.logprob
            nlogprob_norm = m_next.logprob
        if (self.stop == m_next.stop and slogprob_norm < nlogprob_norm) or (self.stop and not m_next.stop):
            self._update_inner(m_next)
            return True
        elif abs(slogprob_norm - nlogprob_norm) < approx_value and self.stop == m_next.stop:
            self.prev.update(m_next.prev)
            self.stop = m_next.stop
            return False
        else:
            self.prev_other.update(m_next.prev)
            return False

    def _update_inner(self, m_other: 'Matching'):
        self.edge_m = m_other.edge_m
        self.edge_o = m_other.edge_o
        self.logprob = m_other.logprob
        self.logprobema = m_other.logprobema
        self.dist_obs = m_other.dist_obs
        self.prev_other.update(self.prev)
        self.prev = m_other.prev
        self.stop = m_other.stop
        self.length = m_other.length

    def __str__(self, label_width=None):
        stop = "x" if self.stop else ""
        if label_width is None:
            label_width = default_label_width
        repr_tmpl = "{:<2} | {:<"+str(label_width)+"} | {:10.5f} | {:10.5f} | {:10.5f} | {:<3} | {:10.5f} | {}"
        return repr_tmpl.format(stop, self.label, self.logprob, self.logprob / self.length, self.logprobema, self.obs,
                                self.dist_obs, ",".join([str(prev.label) for prev in self.prev]))

    @staticmethod
    def repr_header(label_width=None):
        if label_width is None:
            label_width = default_label_width
        repr_tmpl = "{:<2} | {:<"+str(label_width)+"} | {:<10} | {:<10} | {:<10} | " + \
                    "{:<3} | {:<10} | {:<"+str(label_width)+"} |"
        return repr_tmpl.format("", "", "lg(Pr)", "nlg(Pr)", "slg(Pr)", "obs", "d(obs)", "prev")

    @staticmethod
    def repr_static(fields, label_width=None):
        if label_width is None:
            label_width = default_label_width
        default_fields = ["", "", float('nan'), float('nan'), float('nan'), "", float('nan'), "", ""]
        repr_tmpl = "{:<2} | {:<" + str(label_width) + "} | {:10.5f} | {:10.5f} | {:10.5f} | " + \
                    "{:<3} | {:10.5f} | {:<" + str(label_width) + "} |"
        if len(fields) < 8:
            fields = list(fields) + default_fields[len(fields):]
        return repr_tmpl.format(*fields)

    @property
    def label(self):
        if self.edge_m.p2 is None:
            return "{}---{}-{}".format(self.edge_m.l1, self.obs, self.obs_ne)
        else:
            return "{}-{}-{}-{}".format(self.edge_m.l1, self.edge_m.l2, self.obs, self.obs_ne)

    @property
    def cname(self):
        if self.edge_m.l2 is None:
            return "{}_{}_{}".format(self.edge_m.l1, self.obs, self.obs_ne)
        else:
            return "{}_{}_{}_{}".format(self.edge_m.l1, self.edge_m.l2, self.obs, self.obs_ne)

    @property
    def key(self):
        """Key that indicates the node or edge, observation and non-emitting step.
        This is the unique key that is used in the lattice.
        """
        if self.edge_m.l2 is None:
            return tuple([self.edge_m.l1, self.obs, self.obs_ne])
        else:
            return tuple([self.edge_m.l1, self.edge_m.l2, self.obs, self.obs_ne])

    @property
    def shortkey(self):
        """Key that indicates the node or edge. Irrespective of the current observation."""
        if self.edge_m.l2 is None:
            return self.edge_m.l1
        else:
            return tuple([self.edge_m.l1, self.edge_m.l2])

    def __hash__(self):
        return self.cname.__hash__()

    def logprob_trans(self, next_label=None, next_latlon=None):
        """Properties about the movement and the transition probability starting from this state.
        """
        return 0  # All probabilities are 1 (thus technically not a distribution)

    def logprob_obs(self, dist, edge_m, edge_o):
        """Emission probability for emitting states."""
        return self.matcher.logprob_obs(dist)

    def logprob_obs_ne(self, dist, edge_m,edge_o):
        """Emission probability for non-emitting states."""
        return self.matcher.logprob_obs_ne(dist)


class MatchingSpeed(Matching):
    __slots__ = ['avg_dist', 'var_dist', 'dist_mov']  # Additional fields

    def __init__(self, *args, **kwargs):
        """Properties about the movement and the transition probability.

        The transition probability takes into account the average distance travelled to
        compute the probability for the next projected observation.

        TODO: It would be better to combine projection and expected next location.
        TODO: This could also include expected movement vector instead of only distance
        """
        super().__init__(*args, **kwargs)
        self.avg_dist: float = 0.0
        self.var_dist: float = 1.0
        self.dist_mov: float = 0
        if len(self.prev) != 0:
            m_prev = list(self.prev)[0]  # type: MatchingSpeed
            dist = 1  # TODO
            # dist = matcher.map.distance(m.proj_obs, next_latlon)
            self.avg_dist = 0.8 * m_prev.avg_dist + 0.2 * dist
            self.var_dist = 0.8 * m_prev.var_dist + 0.2 * (dist - m_prev.avg_dist) ** 2

    def logprob_trans(self, next_label=None, next_latlon=None):
        dist = self.matcher.map.distance(self.edge_o.pi, next_latlon)
        stddev = 1 if self.var_dist == 0 else math.sqrt(self.var_dist)
        logprob_trans = norm.logpdf(dist, loc=self.avg_dist, scale=stddev) + math.log(stddev) + math.log(2)
        # logger.debug("logprob_trans = {} = f({}, {}, {})".format(self.logprob_trans, dist, self.avg_dist, stddev))
        return logprob_trans

    # def __format__(self, format_spec):
    #     return self.avg_dist.__format__(format_spec)

    @staticmethod
    def repr_header(label_width=None):
        res = Matching.repr_header(label_width)
        res += f"{'d(mov)':<5} |"
        return res

    def __str__(self, label_width=None):
        res = super().__str__(label_width)
        res += f"{self.dist_mov:<5.2f}"
        return res


class Matcher:

    def __init__(self, map_con, obs_noise=1, max_dist_init=None, max_dist=None, min_prob_norm=None,
                 non_emitting_states=False, max_lattice_width=None,
                 only_edges=False, obs_noise_ne=None, matching=Matching):
        """Initialize a matcher for map matching.

        Distances are in meters when using latitude-longitude.

        :param map_con: Map object to connect to map database
        :param obs_noise: Standard deviation of noise
        :param obs_noise_ne: Standard deviation of noise for non-emitting states (is set to obs_noise if not give)
        :param max_dist_init: Maximum distance from start location (if not given, uses max_dist)
        :param max_dist: Maximum distance from path (this is a hard cut, min_prob_norm should be better)
        :param min_prob_norm: Minimum normalized probability of observations (ema)
        :param non_emitting_states: Allow non-emitting states. A non-emitting state is a state that is
            not associated with an observation. Here we assume it can be associated with a location in between
            two observations to allow for pruning. It is advised to set min_prob_norm and/or max_dist to avoid
            visiting all possible nodes in the graph.
        :param max_lattice_width: Restrict the lattice (or possible candidate states per observation) to this value.
            If there are more possible next states, the states with the best likelihood so far are selected.
        :param only_edges: Do not include nodes as states, only edges.
        :param matching: Matching type
        """
        self.map = map_con
        if max_dist:
            self.max_dist = max_dist
        else:
            self.max_dist = np.inf
        if max_dist_init:
            self.max_dist_init = max_dist_init
        else:
            self.max_dist_init = self.max_dist
        if min_prob_norm:
            self.min_logprob_norm = math.log(min_prob_norm)
        else:
            self.min_logprob_norm = -np.inf
        logger.debug(f"Matcher.min_logprob_norm = {self.min_logprob_norm}, Matcher.max_dist = {self.max_dist}")
        self.obs_noise = obs_noise
        if obs_noise_ne is None:
            self.obs_noise_ne = obs_noise
        else:
            self.obs_noise_ne = obs_noise_ne

        self.path = None
        self.lattice = None  # dict[idx,dict[label,Matching]]
        self.lattice_best = None
        self.node_path = None
        self.obs_noise_dist = halfnorm(scale=self.obs_noise)
        self.obs_noise_dist_ne = halfnorm(scale=self.obs_noise_ne)
        # normalize to max 1 to simulate a prob instead of density
        self.obs_noise_logint = math.log(self.obs_noise * math.sqrt(2 * math.pi) / 2)
        self.obs_noise_logint_ne = math.log(self.obs_noise_ne * math.sqrt(2 * math.pi) / 2)
        self.matching = matching
        self.non_emitting_states = non_emitting_states
        self.non_emitting_states_maxnb = 100
        self.max_lattice_width = max_lattice_width
        self.only_edges = only_edges

    def logprob_obs(self, dist):
        result = self.obs_noise_dist.logpdf(dist) + self.obs_noise_logint
        # print("logprob_obs: {} -> {:.5f} = {:.5f}".format(dist, result, math.exp(result)))
        return result

    def logprob_obs_ne(self, dist):
        result = self.obs_noise_dist_ne.logpdf(dist) + self.obs_noise_logint_ne
        # print("logprob_obs: {} -> {:.5f} = {:.5f}".format(dist, result, math.exp(result)))
        return result

    def _init_lattice(self, path):
        self.lattice = dict()
        for idx in range(len(path)):
            self.lattice[idx] = dict()
        self.lattice_best = []

    def match_gpx(self, gpx_file, unique=True):
        """Map matching from a gpx file"""
        path = util.gpx_to_path(gpx_file)
        return self.match(path, unique=unique)

    def do_stop(self, logprob_norm, dist):
        if logprob_norm < self.min_logprob_norm:
            logger.debug(f"Stopped trace because norm(log(Pr)) is too small: {logprob_norm} < {self.min_logprob_norm}")
            return True
        if dist > self.max_dist:
            logger.debug(f"Stopped trace because distance is too large: {dist} > {self.max_dist}")
            return True
        return False

    def _insert(self, m_next):
        if m_next.key in self.lattice[m_next.obs]:
            self.lattice[m_next.obs][m_next.key].update(m_next)
        else:
            self.lattice[m_next.obs][m_next.key] = m_next
        return self.lattice[m_next.obs][m_next.key]

    def match(self, path, unique=True, tqdm=None):
        """Dynamic Programming based (HMM-like) map matcher.

        :param path: list[Union[tuple[lat, lon], tuple[lat, lon, time]]
        :param unique: Only retain unique nodes in the sequence (avoid repetitions)
        :param tqdm: Use a tqdm progress reporter (default is None)
        :return: Sequence of [node, observation, observation-iterations] and
            [segment-node, segment-node, observation, observation-iterations]
        :rtype: list[Union[tuple[str,int,int],tuple[str,str,int,int]]]
        """
        if __debug__:
            logger.debug("Start matching path of length {}".format(len(path)))

        # Initialisation
        self.path = path
        self._create_start_nodes()

        # Start iterating over observations 1..end
        t_start = time.time()
        iterator = range(1, len(path))
        if tqdm:
            iterator = tqdm(iterator)
        early_stop_idx = None
        for obs_idx in iterator:
            if __debug__:
                logger.debug("--- obs {} --- {} ---".format(obs_idx, path[obs_idx]))
            cnt_lat_size_not_zero = False
            for m_tmp in self.lattice[obs_idx - 1].values():
                if not m_tmp.stop:
                    cnt_lat_size_not_zero = True
                    break
            # if len(self.lattice[obs_idx - 1]) == 0:
            if not cnt_lat_size_not_zero:
                if __debug__:
                    logger.debug("No solutions found anymore")
                early_stop_idx = obs_idx
                logger.info(f'Stopped early at observation {early_stop_idx}')
                break
            self._match_states(obs_idx, path)
            if self.non_emitting_states:
                # Fill in non-emitting states between previous and current observation
                self._match_non_emitting_states(obs_idx - 1, path)
            if __debug__:
                if logger.isEnabledFor(logging.DEBUG):
                    self.print_lattice(obs_idx=obs_idx, label_width=default_label_width)

        t_delta = time.time() - t_start
        logger.info("Build lattice in {} seconds".format(t_delta))

        # Backtrack to find best path
        if early_stop_idx:
            if early_stop_idx <= 1:
                return None
            start_idx = early_stop_idx - 2
        else:
            start_idx = len(path) - 1
        node_path = self._build_node_path(start_idx, unique)
        return node_path

    def _create_start_nodes(self):
        # Initialisation on first observation
        t_start = time.time()
        self._init_lattice(self.path)
        nodes = self.map.nodes_closeto(self.path[0], max_dist=self.max_dist_init)
        if __debug__:
            logger.debug("--- obs {} --- {} ---".format(0, self.path[0]))
        t_delta = time.time() - t_start
        logger.info("Initialized map with {} starting points in {} seconds".format(len(nodes), t_delta))
        if len(nodes) == 0:
            logger.info(f'Stopped early at observation 0'
                        f', no starting points x found for which '
                        f'|x - ({self.path[0][0]:.2f},{self.path[0][1]:.2f})| < {self.max_dist_init}')
            self.node_path = []
            return self.node_path
        if __debug__:
            logger.debug(self.matching.repr_header())
        logprob_init = 0  # math.log(1.0/len(nodes))
        if self.only_edges:
            # Search for nearby edges
            # TODO: Search for nearby edges, not first nodes
            for dist_obs, label, loc in nodes:
                # TODO: update dist_obs to be distance to line
                nbrs = self.map.nodes_nbrto(label)
                for nbr_label, nbr_loc in nbrs:
                    logprob = logprob_init + self.logprob_obs(dist_obs)
                    new_stop = self.do_stop(logprob, dist_obs)
                    if not new_stop or logger.isEnabledFor(logging.DEBUG):
                        edge_m = Segment(label, loc, nbr_label, nbr_loc, loc, 0.0)
                        edge_o = Segment(f"O{0}", self.path[0])
                        m_next = self.matching(self, edge_m=edge_m, edge_o=edge_o, logprob=logprob, logprobema=logprob,
                                               dist_obs=dist_obs, obs=0, stop=new_stop)
                        if label in self.lattice[0]:
                            self.lattice[0][label].update(m_next)
                        else:
                            self.lattice[0][label] = m_next
                        if __debug__:
                            logger.debug(str(m_next))
        else:
            # Search for nearby nodes
            for dist_obs, label, loc in nodes:
                logprob = logprob_init + self.logprob_obs(dist_obs)
                new_stop = self.do_stop(logprob, dist_obs)
                if not new_stop or logger.isEnabledFor(logging.DEBUG):
                    edge_m = Segment(label, loc)
                    edge_o = Segment(f"O{0}", self.path[0])
                    m_next = self.matching(self, edge_m=edge_m, edge_o=edge_o, logprob=logprob, logprobema=logprob,
                                           dist_obs=dist_obs, obs=0, stop=new_stop)
                    self.lattice[0][label] = m_next
                    if __debug__:
                        logger.debug(str(m_next))
        if self.max_lattice_width:
            self._prune_lattice(0)
            # if self.non_emitting_states:
            #     self._match_non_emitting_states(0, path)

    def _match_states(self, obs_idx, path):
        """

        :param obs_idx:
        :param path:
        :return: True is new states have been found, False otherwise.
        """
        prev_lattice = self.lattice[obs_idx - 1].values()
        count = 0
        for m in prev_lattice:  # type: Matching
            if m.stop:
                continue
            count += 1
            if m.edge_m.is_point():
                # == Move to neighbour from node ==
                nbrs = self.map.nodes_nbrto(m.edge_m.l1)
                # print("Neighbours for {}: {}".format(m, nbrs))
                if nbrs is None:
                    if __debug__:
                        logger.debug("No neighbours found for node {}".format(m.edge_m.l1))
                    continue
                if __debug__:
                    logger.debug("   Move to {} neighbours from node {}".format(len(nbrs), m.edge_m.l1))
                    logger.debug(m.repr_header())
                for nbr_label, nbr_loc in nbrs:
                    # === Move from node to node (or stay on node) ===
                    if not self.only_edges:
                        edge_m = Segment(nbr_label, nbr_loc)
                        edge_o = Segment(f"O{obs_idx}", path[obs_idx])
                        m_next = m.next(edge_m, edge_o, obs=obs_idx)
                        if m_next is not None:
                            self._insert(m_next)
                            if __debug__:
                                logger.debug(str(m_next))

                    # === Move from node to edge ===
                    if m.edge_m.l1 != nbr_label:
                        edge_m = Segment(m.edge_m.l1, m.edge_m.p1, nbr_label, nbr_loc)
                        edge_o = Segment(f"O{obs_idx}", path[obs_idx])
                        m_next = m.next(edge_m, edge_o, obs=obs_idx)
                        if m_next is not None:
                            self._insert(m_next)
                            if __debug__:
                                logger.debug(str(m_next))
                    else:
                        if __debug__:
                            logger.debug(self.matching.repr_static(('x', f'{nbr_label}-{nbr_label} <')))

            else:
                # == Move to neighbour from edge ==
                if __debug__:
                    logger.debug("   Move to neighbour from edge {}".format(m.key))
                    logger.debug(m.repr_header())

                # === Stay on edge ===
                edge_m = Segment(m.edge_m.l1, m.edge_m.p1, m.edge_m.l2, m.edge_m.p2)
                edge_o = Segment(f"O{obs_idx}", path[obs_idx])
                m_next = m.next(edge_m, edge_o, obs=obs_idx)
                if m_next is not None:
                    self._insert(m_next)
                    if __debug__:
                        logger.debug(str(m_next))

                # === Move from edge to node ===
                if not self.only_edges:
                    edge_m = Segment(m.edge_m.l2, m.edge_m.p2)
                    edge_o = Segment(f"O{obs_idx}", path[obs_idx])
                    m_next = m.next(edge_m, edge_o, obs=obs_idx)
                    if m_next is not None:
                        self._insert(m_next)
                        if __debug__:
                            logger.debug(str(m_next))

                else:
                    # === Move from edge to next edge ===
                    nbrs = self.map.nodes_nbrto(m.edge_m.l2)
                    if nbrs is None:
                        if __debug__:
                            logger.debug(f"No neighbours found for node {m.edge_m.l2}")
                        continue
                    for nbr_label, nbr_loc in nbrs:
                        if m.edge_m.l2 != nbr_label and m.edge_m.l1 != nbr_label:
                            edge_m = Segment(m.edge_m.l2, m.edge_m.p2, nbr_label, nbr_loc)
                            edge_o = Segment(f"O{obs_idx}", path[obs_idx])
                            m_next = m.next(edge_m, edge_o, obs=obs_idx)
                            if m_next is not None:
                                self._insert(m_next)
                                if __debug__:
                                    logger.debug(str(m_next))
        if self.max_lattice_width:
            self._prune_lattice(obs_idx)
        if count == 0:
            if __debug__:
                logger.debug("No active solution found anymore")
            return False
        return True

    def _match_non_emitting_states(self, obs_idx, path):
        """Match sequences of nodes that all refer to the same observation at obs_idx.

        Assumptions:
        This method assumes that the lattice is filled up for both obs_idx and obs_idx + 1.

        :param obs_idx: Index of the first observation used (the second will be obs_idx + 1)
        :param path: List of all observations
        :return: None
        """
        obs = path[obs_idx]
        if obs_idx < len(path) - 1:
            obs_next = path[obs_idx + 1]
        else:
            obs_next = None
        # The current states are the current observation's states
        cur_lattice = dict((m.key, m) for m in self.lattice[obs_idx].values())
        lattice_toinsert = list()
        # The current best states are the next observation's states if you would ignore non-emitting states
        lattice_best = dict((m.shortkey, m) for m in self.lattice[obs_idx + 1].values())
        # cur_lattice = set(self.lattice[obs_idx].values())
        nb_ne = 0
        while len(cur_lattice) > 0 and nb_ne < self.non_emitting_states_maxnb:
            nb_ne += 1
            if __debug__:
                logger.debug("--- obs {}:{} --- {} - {} ---".format(obs_idx, nb_ne, obs, obs_next))
            cur_lattice = self._match_non_emitting_states_inner(cur_lattice, obs_idx, obs, obs_next, nb_ne,
                                                                lattice_toinsert, lattice_best)
            # Link to next observation
            self._match_non_emitting_states_end(cur_lattice, obs_idx + 1, obs_next,
                                                lattice_toinsert, lattice_best)
            if self.max_lattice_width is not None and len(cur_lattice) > self.max_lattice_width:
                ms = sorted(cur_lattice.values(), key=lambda t: t.logprob, reverse=True)
                cur_lattice = dict()
                for m in ms[:self.max_lattice_width]:
                    cur_lattice[m.key] = m
        # logger.info('Used {} levels of non-emitting states'.format(nb_ne))
        for m in lattice_toinsert:
            self._insert(m)

    def _node_in_prev_ne(self, m_next, label):
        # for m in itertools.chain(m_next.prev, m_next.prev_other):
        for m in m_next.prev:
            if m.obs != m_next.obs:
                return False
            assert(m_next.obs_ne != m.obs_ne)
            # print('prev', m.shortkey, 'checking for ', label)
            if label == m.shortkey:
                return True
            if m.obs_ne == 0:
                return False
            if self._node_in_prev_ne(m, label):
                return True
        return False

    @staticmethod
    def _insert_tmp(m_next, lattice):
        if m_next.key in lattice:
            return lattice[m_next.key].update(m_next)
        else:
            lattice[m_next.key] = m_next
            return True

    def _match_non_emitting_states_inner(self, cur_lattice, obs_idx, obs, obs_next, nb_ne,
                                         lattice_toinsert, lattice_best):
        cur_lattice_new = dict()
        for m in cur_lattice.values():  # type: Matching
            if m.stop:
                continue
            # == Move to neighbour edge from edge ==
            if m.edge_m.l2 is not None and self.only_edges:
                nbrs = self.map.nodes_nbrto(m.edge_m.l2)
                # print("Neighbours for {}: {}".format(m, nbrs))
                if nbrs is None:
                    if __debug__:
                        logger.debug(f"No neighbours found for node {m.edge_m.l2} ({m.label}, non-emitting)")
                    continue
                if __debug__:
                    logger.debug(f"   Move to {len(nbrs)} neighbours from node {m.edge_m.l2} ({m.label}, non-emitting)")
                    logger.debug(m.repr_header())
                for nbr_label, nbr_loc in nbrs:
                    if self._node_in_prev_ne(m, nbr_label):
                        if __debug__:
                            logger.debug(self.matching.repr_static(('x', '{} <'.format(nbr_label))))
                        continue
                    # === Move to next edge ===
                    if m.edge_m.l2 != nbr_label and m.edge_m.l1 != nbr_label:
                        edge_m = Segment(m.edge_m.l2, m.edge_m.p2, nbr_label, nbr_loc)
                        edge_o = Segment(f"O{obs_idx}", obs, f"O{obs_idx+1}", obs_next)
                        m_next = m.next(edge_m, edge_o, obs=obs_idx, obs_ne=nb_ne)
                        if m_next is not None:
                            if m_next.key in cur_lattice_new:
                                cur_lattice_new[m_next.key].update(m_next)
                            else:
                                if m_next.shortkey in lattice_best:
                                    # if m_next.logprob > lattice_best[m_next.shortkey].logprob:
                                    if m_next.dist_obs < lattice_best[m_next.shortkey].dist_obs:
                                        cur_lattice_new[m_next.key] = m_next
                                        lattice_best[m_next.shortkey] = m_next
                                        lattice_toinsert.append(m_next)
                                    elif __debug__ and logger.isEnabledFor(logging.DEBUG):
                                        m_next.stop = True
                                        self._insert(m_next)
                                else:
                                    cur_lattice_new[m_next.key] = m_next
                                    lattice_best[m_next.shortkey] = m_next
                                    lattice_toinsert.append(m_next)
                            # cur_lattice_new.add(m_next)
                            if __debug__:
                                logger.debug(str(m_next))
                    else:
                        if __debug__:
                            logger.debug(self.matching.repr_static(('x', f'{m.edge_m.l2}-{nbr_label} <')))
            # == Move to neighbour node from node==
            if m.edge_m.l2 is None and not self.only_edges:
                cur_node = m.edge_m.l1
                nbrs = self.map.nodes_nbrto(cur_node)
                if nbrs is None:
                    if __debug__:
                        logger.debug(
                            f"No neighbours found for node {cur_node} ({m.label}, non-emitting)")
                    continue
                if __debug__:
                    logger.debug(
                        f"   Move to {len(nbrs)} neighbours from node {cur_node} ({m.label}, non-emitting)")
                    logger.debug(m.repr_header())
                for nbr_label, nbr_loc in nbrs:
                    if self._node_in_prev_ne(m, nbr_label):
                        if __debug__:
                            logger.debug(self.matching.repr_static(('x', '{} <'.format(nbr_label))))
                        continue
                    # === Move to next node ===
                    if m.edge_m.l1 != nbr_label:
                        edge_m = Segment(nbr_label, nbr_loc)
                        edge_o = Segment(f"O{obs_idx}", obs, f"O{obs_idx+1}", obs_next)
                        m_next = m.next(edge_m, edge_o, obs=obs_idx, obs_ne=nb_ne)
                        if m_next is not None:
                            if m_next.key in cur_lattice_new:
                                cur_lattice_new[m_next.key].update(m_next)
                            else:
                                if m_next.shortkey in lattice_best:
                                    # if m_next.logprob > lattice_best[m_next.shortkey].logprob:
                                    if m_next.dist_obs < lattice_best[m_next.shortkey].dist_obs:
                                        cur_lattice_new[m_next.key] = m_next
                                        lattice_best[m_next.shortkey] = m_next
                                        lattice_toinsert.append(m_next)
                                    elif __debug__ and logger.isEnabledFor(logging.DEBUG):
                                        m_next.stop = True
                                        self._insert(m_next)
                                else:
                                    cur_lattice_new[m_next.key] = m_next
                                    lattice_best[m_next.shortkey] = m_next
                                    lattice_toinsert.append(m_next)
                            # cur_lattice_new.add(m_next)
                            if __debug__:
                                logger.debug(str(m_next))
                    else:
                        if __debug__:
                            logger.debug(
                                self.matching.repr_static(('x', f'{m.edge_m.l2}-{nbr_label} <')))
            else:
                if __debug__:
                    logger.debug(f"   Ignoring, non-emitting is only node-node or edge-edge")
        return cur_lattice_new

    def _match_non_emitting_states_end(self, cur_lattice, obs_idx, obs_next,
                                       lattice_toinsert, lattice_best):
        for m in cur_lattice.values():  # type: Matching
            if m.stop:
                continue
            if m.edge_m.l2 is not None:
                # Move to neighbour edge from edge
                nbrs = self.map.nodes_nbrto(m.edge_m.l2)
                # print("Neighbours for {}: {}".format(m, nbrs))
                if nbrs is None:
                    if __debug__:
                        logger.debug("No neighbours found for node {}".format(m.edge_m.l2, m.label))
                    continue
                if __debug__:
                    logger.debug(f"   Move to {len(nbrs)} neighbours from node {m.edge_m.l2} "
                                 f"({m.label}, non-emitting->emitting)")
                    logger.debug(m.repr_header())
                for nbr_label, nbr_loc in nbrs:
                    if self._node_in_prev_ne(m, nbr_label):
                        if __debug__:
                            logger.debug(self.matching.repr_static(('x', '{} <'.format(nbr_label))))
                        continue
                    # Move to next edge
                    if m.edge_m.l1 != nbr_label and m.edge_m.l2 != nbr_label:
                        edge_m = Segment(m.edge_m.l2, m.edge_m.p2, nbr_label, nbr_loc)
                        edge_o = Segment(f"O{obs_idx+1}", obs_next)
                        m_next = m.next(edge_m, edge_o, obs=obs_idx)
                        if m_next is not None:
                            self._insert(m_next)
                            if m_next.shortkey in lattice_best:
                                # if m_next.logprob > lattice_best[m_next.shortkey].logprob:
                                if m_next.dist_obs < lattice_best[m_next.shortkey].dist_obs:
                                    lattice_best[m_next.shortkey] = m_next
                                    lattice_toinsert.append(m_next)
                                elif __debug__ and logger.isEnabledFor(logging.DEBUG):
                                    m_next.stop = True
                                    self._insert(m_next)
                            else:
                                lattice_best[m_next.shortkey] = m_next
                                lattice_toinsert.append(m_next)
                            if __debug__:
                                logger.debug(str(m_next))
                    else:
                        if __debug__:
                            logger.debug(self.matching.repr_static(('x', '{} <'.format(nbr_label))))
            else:  # m.edge_m.l2 is None:
                # Move to neighbour node from node
                cur_node = m.edge_m.l1
                nbrs = self.map.nodes_nbrto(cur_node)
                # print("Neighbours for {}: {}".format(m, nbrs))
                if nbrs is None:
                    if __debug__:
                        logger.debug("No neighbours found for node {}".format(cur_node, m.label))
                    continue
                if __debug__:
                    logger.debug(f"   Move to {len(nbrs)} neighbours from node {cur_node} "
                                 f"({m.label}, non-emitting->emitting)")
                    logger.debug(m.repr_header())
                for nbr_label, nbr_loc in nbrs:
                    if self._node_in_prev_ne(m, nbr_label):
                        if __debug__:
                            logger.debug(self.matching.repr_static(('x', '{} <'.format(nbr_label))))
                        continue
                    # Move to next edge
                    if m.edge_m.l1 != nbr_label:
                        edge_m = Segment(nbr_label, nbr_loc)
                        edge_o = Segment(f"O{obs_idx+1}", obs_next)
                        m_next = m.next(edge_m, edge_o, obs=obs_idx)
                        if m_next is not None:
                            self._insert(m_next)
                            if m_next.shortkey in lattice_best:
                                # if m_next.logprob > lattice_best[m_next.shortkey].logprob:
                                if m_next.dist_obs < lattice_best[m_next.shortkey].dist_obs:
                                    lattice_best[m_next.shortkey] = m_next
                                    lattice_toinsert.append(m_next)
                                elif __debug__ and logger.isEnabledFor(logging.DEBUG):
                                    m_next.stop = True
                                    self._insert(m_next)
                            else:
                                lattice_best[m_next.shortkey] = m_next
                                lattice_toinsert.append(m_next)
                            if __debug__:
                                logger.debug(str(m_next))
                    else:
                        if __debug__:
                            logger.debug(self.matching.repr_static(('x', '{} <'.format(nbr_label))))

    def _prune_lattice(self, obs_idx):
        logger.debug('Prune lattice[{}] from {} to {}'
                     .format(obs_idx, len(self.lattice[obs_idx]), self.max_lattice_width))
        if len(self.lattice[obs_idx]) <= self.max_lattice_width:
            return
        top_m = [m for m in self.lattice[obs_idx].values() if not m.stop]
        top_m.sort(key=lambda m: m.logprobema, reverse=True)
        self.lattice[obs_idx] = dict((m.key, m) for m in top_m[:self.max_lattice_width])
        self._cleanup_lattice(obs_idx)
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            for m in self.lattice[obs_idx].values():
                logger.debug(str(m))

    def _cleanup_lattice(self, obs_idx):
        """Remove all lattice entries that cannot be part of a backtracking path starting at obs_idx."""
        if obs_idx <= 1:
            return
        all_prev = set()
        for m in self.lattice[obs_idx].values():
            all_prev.update(m.prev)
        lattice_col = self.lattice[obs_idx - 1]
        for label, m in lattice_col.items():
            if m not in all_prev:
                del lattice_col[label]


    def _build_node_path(self, start_idx, unique=True):
        node_max = None
        for m in self.lattice[start_idx].values():
            if node_max is None or m.logprob > node_max.logprob:
                node_max = m
        if node_max is None:
            logger.info("Did not find a matching node for path point at index {}".format(start_idx))
            return None
        logger.debug("Start: {}".format(node_max))
        node_path_rev = [node_max.shortkey]
        self.lattice_best.append(node_max)
        # for obs_idx in reversed(range(start_idx)):
        while len(node_max.prev) > 0:
            node_max_last = node_max
            node_max: Optional[Matching] = None
            if __debug__:
                logger.debug("Prev ({}):".format(node_max_last.obs))
            for prev_m in node_max_last.prev:
                logger.debug(prev_m)
                # if prev_m.obs != obs_idx:
                #     raise Exception("Previous matching node index {} and path index do not match {}."
                #                     .format(prev_m.obs, obs_idx))
                if prev_m is not None and (node_max is None or prev_m.logprob > node_max.logprob):
                    node_max = prev_m
            if node_max is None:
                logger.info("Did not find a matching node for path point at index {}".format(node_max_last.obs))
                return None
            logger.debug("Max ({}): {}".format(node_max_last.obs, node_max))
            node_path_rev.append(node_max.shortkey)
            self.lattice_best.append(node_max)

        self.lattice_best = list(reversed(self.lattice_best))
        if unique:
            self.node_path = []
            prev_node = None
            for node in reversed(node_path_rev):
                if node != prev_node:
                    self.node_path.append(node)
                    prev_node = node
        else:
            self.node_path = list(reversed(node_path_rev))
        return self.node_path

    def print_lattice(self, file=None, obs_idx=None, label_width=None):
        if file is None:
            file = sys.stdout
        # print("Lattice:", file=file)
        if obs_idx is not None:
            idxs = [obs_idx]
        else:
            idxs = range(len(self.lattice))
        for idx in idxs:
            if len(self.lattice[idx]) > 0:
                if label_width is None:
                    label_width = 0
                    for m in self.lattice[idx].values():
                        label_width = max(label_width, len(str(m.label)))
                print("--- obs {} ---".format(idx), file=file)
                print(self.matching.repr_header(label_width=label_width), file=file)
                for m in sorted(self.lattice[idx].values(), key=lambda t: str(t.label)):
                    print(m.__str__(label_width=label_width), file=file)

    def lattice_dot(self, file=None):
        if file is None:
            file = sys.stdout
        print('digraph lattice {', file=file)
        print('\trankdir=LR;', file=file)
        for idx in range(len(self.lattice)):
            if len(self.lattice[idx]) == 0:
                continue
            cnames = [(m.obs_ne, m.cname, m.stop) for m in self.lattice[idx].values()]
            cnames.sort()
            cur_obs_ne = -1
            print('\t{\n\t\trank=same; ', file=file)
            for obs_ne, cname, stop, in cnames:
                if obs_ne != cur_obs_ne:
                    if cur_obs_ne != -1:
                        print('\t};\n\t{\n\t\trank=same; ', file=file)
                    cur_obs_ne = obs_ne
                if stop:
                    options = 'label="{} x",color=gray,fontcolor=gray'.format(cname)
                else:
                    options = 'label="{} ."'.format(cname)
                print('\t\t{} [{}];'.format(cname, options), file=file)
            print('\t};', file=file)
        for idx in range(len(self.lattice)):
            if len(self.lattice[idx]) == 0:
                continue
            for m in self.lattice[idx].values():
                for mp in m.prev:
                    if m.stop:
                        options = ',color=gray,fontcolor=gray'
                    else:
                        options = ''
                    print(f'\t {mp.cname} -> {m.cname} [label="{m.logprob}"{options}];', file=file)
                for mp in m.prev_other:
                    if m.stop:
                        options = ',color=gray,fontcolor=gray'
                    else:
                        options = ''
                    print(f'\t {mp.cname} -> {m.cname} [color=gray,label="{m.logprob}"{options}];', file=file)
        print('}', file=file)

    def print_lattice_stats(self, file=None, verbose=False):
        if file is None:
            file = sys.stdout
        print("Stats lattice", file=file)
        print("-------------", file=file)
        stats = OrderedDict()
        stats["nbr levels"] = len(self.lattice)
        total_nodes = 0
        max_nodes = 0
        min_nodes = 9999999
        sizes = []
        for idx in range(len(self.lattice)):
            level = self.lattice[idx]
            # stats["#nodes[{}]".format(idx)] = len(level)
            sizes.append(len(level))
            total_nodes += len(level)
            if len(level) < min_nodes:
                min_nodes = len(level)
            if len(level) > max_nodes:
                max_nodes = len(level)
        stats["nbr lattice"] = total_nodes
        if verbose:
            stats["nbr lattice[level]"] = ", ".join([str(s) for s in sizes])
        stats["avg lattice[level]"] = total_nodes/len(self.lattice)
        stats["min lattice[level]"] = min_nodes
        stats["max lattice[level]"] = max_nodes
        if len(self.lattice_best) > 0:
            stats["avg obs distance"] = np.mean([m.dist_obs for m in self.lattice_best])
            stats["last logprob"] = self.lattice_best[-1].logprob
            stats["last length"] = self.lattice_best[-1].length
            stats["last norm logprob"] = self.lattice_best[-1].logprob / self.lattice_best[-1].length
            if verbose:
                stats["best logprob"] = ", ".join(["{:.3f}".format(m.logprob) for m in self.lattice_best])
                stats["best norm logprob"] = \
                    ", ".join(["{:.3f}".format(m.logprob/m.length) for i, m in enumerate(self.lattice_best)])
                stats["best norm prob"] = \
                    ", ".join(["{:.3f}".format(math.exp(m.logprob/m.length)) for i, m in enumerate(self.lattice_best)])
        for key, val in stats.items():
            print("{:<24} : {}".format(key, val), file=file)

    def node_counts(self):
        counts = defaultdict(lambda: 0)
        for level in self.lattice.values():
            for m in level.values():
                counts[m.label] += 1
        return counts

    @property
    def path_pred(self):
        return self.node_path

    @property
    def path_pred_onlynodes(self):
        if self.node_path is None or len(self.node_path) == 0:
            return []
        nodes = []
        if type(self.node_path[0]) is tuple:
            nodes.append(self.node_path[0][0])
            nodes.append(self.node_path[0][1])
            prev_node = self.node_path[0][1]
        else:
            nodes.append(self.node_path[0])
            prev_node = self.node_path[0]
        prev_state = None
        for state in self.node_path[1:]:
            if state == prev_state:
                continue
            if type(state) is not tuple:
                if state != prev_node:
                    nodes.append(state)
                    prev_node = state
            elif type(state) is tuple:
                if state[0] == prev_node:
                    if state[1] != prev_node:
                        nodes.append(state[1])
                        prev_node = state[1]
                elif state[1] == prev_node:
                    if state[0] != prev_node:
                        nodes.append(state[0])
                        prev_node = state[0]
                else:
                    raise Exception(f"No node in edge {state} is previous node {prev_node}")
            else:
                raise Exception(f"Unknown type of state: {state} ({type(state)})")
            prev_state = state
        return nodes
