# encoding: utf-8
"""
leuvenmapmatching.matcher.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Base Matcher and Matching classes.

This a generic base class to be used by matchers. This class itself
does not implement a working matcher.

:author: Wannes Meert
:copyright: Copyright 2015-2021 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
from __future__ import print_function

import math
import sys
import logging
import time
from collections import OrderedDict, defaultdict, namedtuple
from itertools import islice
from typing import List, Tuple, Dict, Any, Optional, Set

import numpy as np

from ..util.segment import Segment
from ..util import approx_equal, approx_leq


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")
approx_value = 0.0000000001
ema_const = namedtuple('EMAConst', ['prev', 'cur'])(0.7, 0.3)
default_label_width = 25


class BaseMatching(object):
    """Matching object that represents a node in the Viterbi lattice."""
    __slots__ = ['matcher', 'edge_m', 'edge_o',
                 'logprob', 'logprobema', 'logprobe', 'logprobne',
                 'obs', 'obs_ne', 'dist_obs',
                 'prev', 'prev_other', 'stop', 'length', 'delayed']

    def __init__(self, matcher: 'BaseMatcher', edge_m: Segment, edge_o: Segment,
                 logprob=-np.inf, logprobema=-np.inf, logprobe=-np.inf, logprobne=-np.inf,
                 dist_obs: float = 0.0, obs: int = 0, obs_ne: int = 0,
                 prev: Optional[Set['BaseMatching']] = None, stop: bool = False, length: int = 1,
                 delayed: int = 0, **_kwargs):
        """

        :param matcher: Reference to the Matcher used to generate this matching object.
        :param edge_m: Segment in the given graph (thus line between two nodes in the graph).
        :param edge_o: Segment in the given observations (thus line in between two observations).
        :param logprob: Log probability of this matching.
        :param logprobema: Exponential Mean Average of Log probability.
        :param logprobe: Emitting
        :param logprobne: Non-emitting
        :param dist_obs: Distance between map point and observation
        :param obs: Reference to path entry index (observation)
        :param obs_ne: Number of non-emitting states for this observation
        :param prev: Previous best matching objects
        :param stop: Stop after this matching (e.g. because probability is too low)
        :param length: Lenght of current matching sequence through lattice.
        :param delayed: This matching is temporarily stopped if >0 (e.g. to first explore better options).
        :param dist_m: Distance over graph
        :param dist_o: Distance over observations
        :param _kwargs:
        """
        self.edge_m: Segment = edge_m
        self.edge_o: Segment = edge_o
        self.logprob: float = logprob        # max probability
        self.logprobe: float = logprobe      # Emitting
        self.logprobne: float = logprobne    # Non-emitting
        self.logprobema: float = logprobema  # exponential moving average log probability  # TODO: Not used anymore?
        self.obs: int = obs  # reference to path entry index (observation)
        self.obs_ne: int = obs_ne  # number of non-emitting states for this observation
        self.dist_obs: float = dist_obs  # Distance between map point and observation
        self.prev: Set[BaseMatching] = set() if prev is None else prev  # Previous best matching objects
        self.prev_other: Set[BaseMatching] = set()  # Previous matching objects with lower logprob
        self.stop: bool = stop
        self.length: int = length
        self.delayed: int = delayed
        self.matcher: BaseMatcher = matcher

    @property
    def prune_value(self):
        """Pruning the lattice (e.g. to delay) is based on this key."""
        return self.logprob
        # return self.logprobema

    def next(self, edge_m: Segment, edge_o: Segment, obs: int = 0, obs_ne: int = 0):
        """Create a next lattice Matching object with this Matching object as the previous one in the lattice."""
        new_stop = False
        if edge_m.is_point() and edge_o.is_point():
            # node to node
            dist = self.matcher.map.distance(edge_m.p1, edge_o.p1)
            # proj_m = edge_m.p1
            # proj_o = edge_o.pi
        elif edge_m.is_point() and not edge_o.is_point():
            # node to edge
            dist, proj_o, t_o = self.matcher.map.distance_point_to_segment(edge_m.p1, edge_o.p1, edge_o.p2)
            # proj_m = edge_m.p1
            edge_o.pi = proj_o
            edge_o.ti = t_o
        elif not edge_m.is_point() and edge_o.is_point():
            # edge to node
            dist, proj_m, t_m = self.matcher.map.distance_point_to_segment(edge_o.p1, edge_m.p1, edge_m.p2)
            if not self.matcher.only_edges and (approx_equal(t_m, 0.0) or approx_equal(t_m, 1.0)):
                if __debug__ and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"   | Stopped trace: Too close to end, {t_m}")
                    new_stop = True
                else:
                    return None
            edge_m.pi = proj_m
            edge_m.ti = t_m
            # proj_o = edge_o.pi
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

        logprob_trans, props_trans = self.matcher.logprob_trans(self, edge_m, edge_o,
                                                                is_prev_ne=(self.obs_ne != 0),
                                                                is_next_ne=(obs_ne != 0))
        logprob_obs, props_obs = self.matcher.logprob_obs(dist, self, edge_m, edge_o,
                                                          is_ne=(obs_ne != 0))
        if __debug__ and logprob_trans > 0:
            raise Exception(f"logprob_trans = {logprob_trans} > 0")
        if __debug__ and logprob_obs > 0:
            raise Exception(f"logprob_obs = {logprob_obs} > 0")
        new_logprob_delta = logprob_trans + logprob_obs
        if obs_ne == 0:
            new_logprobe = self.logprob + new_logprob_delta
            new_logprobne = 0
            new_logprob = new_logprobe
            new_length = self.length + 1
        else:
            # Non-emitting states require normalisation
            # "* e^(ne_length_factor_log)" or "- ne_length_factor_log" for every step to a non-emitting
            # state to prefer shorter paths
            new_logprobe = self.logprobe + self.matcher.ne_length_factor_log
            # The obvious choice would be average to compensate for that non-emitting states
            # create different path lengths between emitting nodes.
            # We use min() as it is a monotonic function, in contrast with an average
            new_logprobne = min(self.logprobne, new_logprob_delta)
            new_logprob = new_logprobe + new_logprobne
            # Alternative approach with an average
            # new_logprobne = self.logprobne + new_logprob_delta
            # "+ 1" to punish non-emitting states a bit less. Otherwise it would be
            # similar to (Pr_tr*Pr_obs)**2, which punishes just one non-emitting state too much.
            # new_logprob = new_logprobe + new_logprobne / (obs_ne + 1)
            new_length = self.length
        new_logprobema = ema_const.cur * new_logprob_delta + ema_const.prev * self.logprobema
        new_stop |= self.matcher.do_stop(new_logprob / new_length, dist, logprob_trans, logprob_obs)
        if __debug__ and new_logprob > self.logprob:
            raise Exception(f"Expecting a monotonic probability, "
                            f"new_logprob = {new_logprob} > logprob = {self.logprob}")
        if not new_stop or (__debug__ and logger.isEnabledFor(logging.DEBUG)):
            m_next = self.__class__(self.matcher, edge_m, edge_o,
                                    logprob=new_logprob, logprobne=new_logprobne,
                                    logprobe=new_logprobe, logprobema=new_logprobema,
                                    obs=obs, obs_ne=obs_ne, prev={self}, dist_obs=dist,
                                    stop=new_stop, length=new_length, delayed=self.delayed,
                                    **props_trans, **props_obs)
            return m_next
        else:
            return None

    @classmethod
    def first(cls, logprob_init, edge_m, edge_o, matcher, dist_obs):
        """Create an initial lattice Matching object."""
        logprob_obs, props_obs = matcher.logprob_obs(dist_obs, None, edge_m, edge_o)
        logprob = logprob_init + logprob_obs
        new_stop = matcher.do_stop(logprob, dist_obs, logprob_init, logprob_obs)
        if not new_stop or logger.isEnabledFor(logging.DEBUG):
            m_next = cls(matcher, edge_m=edge_m, edge_o=edge_o,
                         logprob=logprob, logprobema=logprob, logprobe=logprob, logprobne=0,
                         dist_obs=dist_obs, obs=0, stop=new_stop, **props_obs)
            return m_next
        else:
            return None

    def update(self, m_next):
        """Update the current entry if the new matching object for this state is better.

        :param m_next: The new matching object representing the same node in the lattice.
        :return: True if the current object is replaced, False otherwise
        """
        # if self.length != m_next.length:
        #     slogprob_norm = self.logprob / self.length
        #     nlogprob_norm = m_next.logprob / m_next.length
        # else:
        #     slogprob_norm = self.logprob
        #     nlogprob_norm = m_next.logprob
        # if (self.stop == m_next.stop and slogprob_norm < nlogprob_norm) or (self.stop and not m_next.stop):
        #     self._update_inner(m_next)
        #     return True
        # elif abs(slogprob_norm - nlogprob_norm) < approx_value and self.stop == m_next.stop:
        #     self.prev.update(m_next.prev)
        #     self.stop = m_next.stop
        #     return False
        assert self.length == m_next.length
        if (self.stop and not m_next.stop) \
                or (self.stop == m_next.stop and self.logprob < m_next.logprob):
            self._update_inner(m_next)
            return True
        else:
            self.prev_other.update(m_next.prev)
            return False

    def _update_inner(self, m_other: 'BaseMatching'):
        self.edge_m = m_other.edge_m
        self.edge_o = m_other.edge_o
        self.logprob = m_other.logprob
        self.logprobe = m_other.logprobe
        self.logprobne = m_other.logprobne
        self.logprobema = m_other.logprobema
        self.dist_obs = m_other.dist_obs
        self.obs = m_other.obs
        self.obs_ne = m_other.obs_ne
        self.prev_other.update(self.prev)  # Do we use this?
        self.prev = m_other.prev
        self.stop = m_other.stop
        self.delayed = m_other.delayed
        self.length = m_other.length

    def is_nonemitting(self):
        return self.obs_ne != 0

    def is_emitting(self):
        return self.obs_ne == 0

    def last_emitting_logprob(self):
        if self.is_emitting():
            return self.logprob
        elif self.prev is None or len(self.prev) == 0:
            return 0
        else:
            return next(iter(self.prev)).last_emitting_logprob()

    def __str__(self, label_width=None):
        stop = ''
        if self.stop:
            stop = 'x'
        else:
            stop = f'{self.delayed}'
        if label_width is None:
            label_width = default_label_width
        repr_tmpl = "{:<2} | {:<"+str(label_width)+"} | {:10.5f} | {:10.5f} | {:10.5f} | {:10.5f} | " +\
                    "{:<3} | {:10.5f} | {:<" + str(label_width) + "} |"
        return repr_tmpl.format(stop, self.label, self.logprob, self.logprob / self.length,
                                self.logprobema, self.logprobne, self.obs,
                                self.dist_obs, ",".join([str(prev.label) for prev in self.prev]))

    def __repr__(self):
        return "Matching<"+str(self.label)+">"

    @staticmethod
    def repr_header(label_width=None, stop=""):
        if label_width is None:
            label_width = default_label_width
        repr_tmpl = "{:<2} | {:<"+str(label_width)+"} | {:<10} | {:<10} | {:<10} | {:<10} | " + \
                    "{:<3} | {:<10} | {:<"+str(label_width)+"} |"
        return repr_tmpl.format(stop, "", "lg(Pr)", "nlg(Pr)", "slg(Pr)", "lg(Pr-ne)", "obs", "d(obs)", "prev")

    @staticmethod
    def repr_static(fields, label_width=None):
        if label_width is None:
            label_width = default_label_width
        default_fields = ["", "", float('nan'), float('nan'), float('nan'), float('nan'), "", float('nan'), "", ""]
        repr_tmpl = "{:<2} | {:<" + str(label_width) + "} | {:10.5f} | {:10.5f} | {:10.5f} | {:10.5f} | " + \
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

    @property
    def nodes(self):
        if self.edge_m.l2 is None:
            return [self.edge_m.l1]
        else:
            return [self.edge_m.l1, self.edge_m.l2]

    def __hash__(self):
        return self.cname.__hash__()

    def __lt__(self, o):
        return self.logprob < o.logprob

    def __le__(self, o):
        return self.logprob <= o.logprob

    def __eq__(self, o):
        return self.logprob == o.logprob

    def __ne__(self, o):
        return self.logprob != o.logprob

    def __ge__(self, o):
        return self.logprob >= o.logprob

    def __gt__(self, o):
        return self.logprob > o.logprob


class LatticeColumn:

    def __init__(self, obs_idx):
        # 0 = obs, >0 = non-emitting between this obs and next
        self.obs_idx = obs_idx
        self.o = []  # type list[dict[label,Matching]]

    def __contains__(self, item):
        for c in self.o:
            if item in c:
                return True
        return False

    def __len__(self):
        return len(self.o)

    def set_delayed(self, delayed):
        """Update all delayed values."""
        for c in self.o:
            for m in c.values():
                m.delayed = delayed

    def dict(self, obs_ne=None):
        if obs_ne is None:
            raise AttributeError('obs_ne should be value')
        while obs_ne >= len(self.o):
            self.o.append({})
        return self.o[obs_ne]

    def values_all(self):
        """All matches for the emitting layer and all non-emitting layers."""
        values = set()
        for o in self.o:
            values.update(o.values())
        return values

    def values(self, obs_ne=None):
        if obs_ne is None:
            raise AttributeError('obs_ne should be value')
        if len(self.o) <= obs_ne:
            return []
        return self.o[obs_ne].values()

    def upsert(self, matching):
        # type: (BaseMatching) -> None
        if matching is None:
            return None
        while matching.obs_ne >= len(self.o):
            self.o.append({})
        c = self.o[matching.obs_ne]
        if matching.key in c:
            other_matching = c[matching.key]  # type: BaseMatching
            other_matching.update(matching)
        else:
            c[matching.key] = matching
        return c[matching.key]

    def prune(self, obs_ne, max_lattice_width, expand_upto, prune_thr=None):
        """Prune given column in the lattice to fit in max_lattice_width.
        Also ignore all matchings with a probability lower than prune_thr. These are
        matchings that are worse than the matchings at the next observation that are
        retained after pruning.

        :param obs_ne:
        :param max_lattice_width:
        :param expand_upto: The current expand level
        :return:
        """
        cur_lattice = [m for m in self.values(obs_ne) if not m.stop]
        if __debug__:
            logger.debug('Prune lattice[{},{}] from {} to {}, with prune thr {}'
                         .format(self.obs_idx, obs_ne,
                                 len([m for m in cur_lattice if not m.stop and m.delayed == expand_upto]),
                                 max_lattice_width, prune_thr))
            cnt_pruned = 0
        if max_lattice_width is not None and len(cur_lattice) > max_lattice_width:
            ms = sorted(cur_lattice, key=lambda t: t.prune_value, reverse=True)
            cur_width = max_lattice_width
            m_last = ms[cur_width - 1]
            # Extend current width if next pruned matching has same logprob as last kept matching
            # This increases the lattice width but otherwise the algorithm depends on the
            # order of edges/nodes and is not deterministic.
            while cur_width < len(ms) and ms[cur_width].prune_value == m_last.prune_value:
                m_last = ms[cur_width]
                cur_width += 1
            if prune_thr is not None:
                while cur_width > 0 and ms[cur_width - 1].prune_value < prune_thr:
                    cur_width -= 1
            for m in ms[:cur_width]:  # type: BaseMatching
                if m.delayed > expand_upto:
                    m.delayed = expand_upto  # expand now
            for m in ms[cur_width:]:
                if m.delayed <= expand_upto:
                    if __debug__:
                        cnt_pruned += 1
                    m.delayed = expand_upto + 1  # expand later
            if cur_width > 0:
                prune_thr = ms[cur_width - 1].prune_value
        if __debug__:
            logger.debug(f'Pruned {cnt_pruned} matchings, return {prune_thr}')
        return prune_thr


class BaseMatcher:

    def __init__(self, map_con, obs_noise=1, max_dist_init=None, max_dist=None, min_prob_norm=None,
                 non_emitting_states=True, max_lattice_width=None,
                 only_edges=True, obs_noise_ne=None, matching=BaseMatching,
                 non_emitting_length_factor=0.75, **kwargs):
        """Initialize a matcher for map matching.

        This a generic base class to be used by matchers. This class itself
        does not implement a working matcher.

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
        :param max_lattice_width: Only continue from a limited number of states (thus locations) for a given observation.
            This possibly speeds up the matching by a lot.
            If there are more possible next states, the states with the best likelihood so far are selected.
            The other states are 'delayed'. If the matching is continued later with a larger value using
            `increase_max_lattice_width`, the algorithms continuous from these delayed states.
        :param only_edges: Do not include nodes as states, only edges. This is the typical setting for HMM methods.
        :param matching: Matching type
        :param non_emitting_length_factor: Reduce the probability of a sequence of non-emitting states the longer it
            is. This can be used to prefer shorter paths. This is separate from the transition probabilities because
            transition probabilities are averaged for non-emitting states and thus the length is also averaged out.

        To define a custom transition and/or emission probability distribtion, overwrite the following functions:

        - :meth:`logprob_trans`
        - :meth:`logprob_obs`

        """
        self.map = map_con  # type: BaseMap
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
        self.lattice = None  # type: Optional[dict[int,LatticeColumn]]
        # Best path through lattice:
        self.lattice_best = None  # type: Optional[list[BaseMatching]]
        self.node_path = None  # type: Optional[list[str]]
        self.matching = matching
        self.non_emitting_states = non_emitting_states  # type: bool
        self.non_emitting_states_maxnb = 100
        self.max_lattice_width = max_lattice_width  # type: Optional[int]
        self.only_edges = only_edges  # type: bool
        self.expand_now = 0  # all m.delayed <= expand_upto will be expanded
        self.early_stop_idx = None

        # Penalties
        self.ne_length_factor_log = math.log(non_emitting_length_factor)

    def logprob_trans(self, prev_m, edge_m, edge_o,
                      is_prev_ne=False, is_next_ne=False):
        # type: (BaseMatcher, BaseMatching, Segment, Segment, bool, bool) -> Tuple[float, Dict[str, Any]]
        """Transition probability.

        Note: In contrast with a regular HMM, this cannot be a probability density function, it needs
              to be a proper probability (thus values between 0.0 and 1.0).

        :return: probability, properties that are passed to the matching object
        """
        return 0, {}  # All probabilities are 1 (thus technically not a distribution)

    def logprob_obs(self, dist, prev_m, new_edge_m, new_edge_o, is_ne=False):
        """Emission probability.

        Note: In contrast with a regular HMM, this cannot be a probability density function, it needs
              to be a proper probability (thus values between 0.0 and 1.0).

        :return: probability, properties that are passed to the matching object
        """
        return 0, {}

    def match_gpx(self, gpx_file, unique=True):
        """Map matching from a gpx file"""
        from ..util.gpx import gpx_to_path
        path = gpx_to_path(gpx_file)
        return self.match(path, unique=unique)

    def do_stop(self, logprob_norm, dist, logprob_trans, logprob_obs):
        if logprob_norm < self.min_logprob_norm:
            logger.debug(f"   | Stopped trace: norm(log(Pr)) too small: {logprob_norm} < {self.min_logprob_norm}"
                         f"  -- lPr_t = {logprob_trans:.3f}, lPr_o = {logprob_obs:.3f}")
            return True
        if dist > self.max_dist:
            logger.debug(f"   | Stopped trace: distance too large: {dist} > {self.max_dist}")
            return True
        return False

    def _insert(self, m_next):
        return self.lattice[m_next.obs].upsert(m_next)

    def match(self, path, unique=False, tqdm=None, expand=False):
        """Dynamic Programming based (HMM-like) map matcher.

        If the matcher fails to match the entire path, the last matched index is returned.
        This index can be used to run the matcher again from that observation onwards.

        :param path: list[Union[tuple[lat, lon], tuple[lat, lon, time]]
        :param unique: Only retain unique nodes in the sequence (avoid repetitions)
        :param tqdm: Use a tqdm progress reporter (default is None)
        :param expand: Expand the current lattice (delayed matches)
        :return: Tuple of (List of BaseMatching, index of last observation that was matched)
        """
        if __debug__:
            logger.debug("Start matching path of length {}".format(len(path)))

        # Initialisation
        if expand:
            self.expand_now += 1
            if self.path != path:
                is_path_extended = True
                if len(path) > len(self.path):
                    for pi, spi in zip(path, self.path):
                        if pi != spi:
                            is_path_extended = False
                            break
                else:
                    is_path_extended = False
                if is_path_extended:
                    self.lattice[len(self.path) - 1].set_delayed(self.expand_now)
                    for obs_idx in range(len(self.path), len(path)):
                        if obs_idx not in self.lattice:
                            self.lattice[obs_idx] = LatticeColumn(obs_idx)
                    self.path = path
                else:
                    raise Exception(f'Cannot expand for a new path, should be the same path (or an extension).')
        else:
            self.path = path
            self.expand_now = 0

        nb_start_nodes = self._create_start_nodes(use_edges=self.only_edges)
        if nb_start_nodes == 0:
            self.lattice_best = []
            return [], 0
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            self.print_lattice(obs_idx=0, label_width=default_label_width, debug=True)

        # Start iterating over observations 1..end
        t_start = time.time()
        iterator = range(1, len(path))
        if tqdm:
            iterator = tqdm(iterator)
        self.early_stop_idx = None
        for obs_idx in iterator:
            if __debug__:
                logger.debug("--- obs {} --- {} ---".format(obs_idx, self.path[obs_idx]))
            # check if early stopping has occured
            cnt_lat_size_not_zero = False
            for m_tmp in self.lattice[obs_idx - 1].values(0):
                if not m_tmp.stop:
                    cnt_lat_size_not_zero = True
                    break
            # if len(self.lattice[obs_idx - 1]) == 0:
            if not cnt_lat_size_not_zero:
                if __debug__:
                    logger.debug("No solutions found anymore")
                self.early_stop_idx = obs_idx - 1
                logger.info(f'Stopped early at observation {self.early_stop_idx}')
                break
            # Expand matches
            self._match_states(obs_idx)
            if self.non_emitting_states:
                # Fill in non-emitting states between previous and current observation
                self._match_non_emitting_states(obs_idx - 1, expand=expand)
            if self.max_lattice_width:
                # Prune again if non_emitting_states reactives matches from match_states
                self.lattice[obs_idx].prune(0, self.max_lattice_width, self.expand_now)
            if __debug__ and logger.isEnabledFor(logging.DEBUG):
                self.print_lattice(obs_idx=obs_idx, label_width=default_label_width, debug=True)
                logger.debug(f"--- end obs {obs_idx} ---")

        t_delta = time.time() - t_start
        logger.info("--- end ---")
        logger.info("Build lattice in {} seconds".format(t_delta))

        # Backtrack to find best path
        if not self.early_stop_idx:
            one_no_stop = False
            for m in self.lattice[len(path) - 1].values_all():  # todo: could be values(0) ?
                if not m.stop:
                    one_no_stop = True
                    break
            if not one_no_stop:
                self.early_stop_idx = len(path) - 1
        if self.early_stop_idx is not None:
            if self.early_stop_idx == 0:
                self.lattice_best = []
                return [], 0
            start_idx = self.early_stop_idx - 1
        else:
            start_idx = len(self.path) - 1
        node_path = self._build_node_path(start_idx, unique)
        return node_path, start_idx

    def _skip_ne_states(self, prev_m):
        # type: (BaseMatcher, BaseMatching) -> bool
        return False

    def _create_start_nodes(self, use_edges=True):
        """Find those nodes that are close to the first point in the path.

        :return: Number of created start points.
        """
        # Initialisation on first observation
        if self.expand_now > 0:
            # No need to search for new points, only activate delayed matches
            self.lattice[0].prune(0, self.max_lattice_width, self.expand_now)
            return len(self.lattice[0])

        t_start = time.time()
        self.lattice = dict()
        for obs_idx in range(len(self.path)):
            self.lattice[obs_idx] = LatticeColumn(obs_idx)

        if use_edges:
            nodes = self.map.edges_closeto(self.path[0], max_dist=self.max_dist_init)
        else:
            nodes = self.map.nodes_closeto(self.path[0], max_dist=self.max_dist_init)
        if __debug__:
            logger.debug("--- obs {} --- {} ---".format(0, self.path[0]))
        t_delta = time.time() - t_start
        logger.info("Initialized lattice with {} starting points in {} seconds".format(len(nodes), t_delta))
        if len(nodes) == 0:
            logger.info(f'Stopped early at observation 0'
                        f', no starting points/edges x found for which '
                        f'|x - ({self.path[0][0]:.2f},{self.path[0][1]:.2f})| < {self.max_dist_init}')
            return 0
        if __debug__:
            logger.debug(self.matching.repr_header())
        logprob_init = 0  # math.log(1.0/len(nodes))
        if use_edges:
            # Search for nearby edges
            for dist_obs, label1, loc1, label2, loc2, pi, ti in nodes:
                if label2 == label1:
                    continue
                edge_m = Segment(label1, loc1, label2, loc2, pi, ti)
                edge_o = Segment(f"O{0}", self.path[0])
                m_next = self.matching.first(logprob_init, edge_m, edge_o, self, dist_obs)
                if m_next is not None:
                    self.lattice[0].upsert(m_next)
                    if __debug__:
                        logger.debug(str(m_next))
        else:
            # Search for nearby nodes
            for dist_obs, label, loc in nodes:
                edge_m = Segment(label, loc)
                edge_o = Segment(f"O{0}", self.path[0])
                m_next = self.matching.first(logprob_init, edge_m, edge_o, self, dist_obs)
                if m_next is not None:
                    self.lattice[0].upsert(m_next)
                    if __debug__:
                        logger.debug(str(m_next))
        if self.max_lattice_width:
            self.lattice[0].prune(0, max_lattice_width=self.max_lattice_width, expand_upto=self.expand_now)
            # if self.non_emitting_states:
            #     self._match_non_emitting_states(0, path)
        return len(self.lattice[0])

    def increase_delayed(self, expand_from=None):
        if expand_from is None:
            expand_from = self.expand_now + 1
        for col in self.lattice.values():
            for colo in col.o:
                for m in colo.values():
                    if m.delayed >= expand_from:
                        m.delayed += 1

    def _match_states(self, obs_idx, prev_lattice=None, max_dist=None, inc_delayed=False):
        """Match states

        :param obs_idx:
        :param prev_lattice: Start from this list instead of the previous
            column in the lattice
        :param max_dist: Use map.*_closeto instead of map.*_nbrto
        :param inc_delayed: Increase delayed property when new state is created
        :return: True is new states have been found, False otherwise.
        """
        if prev_lattice is None:
            prev_lattice = [m for m in self.lattice[obs_idx - 1].values(0) if not m.stop and m.delayed == self.expand_now]
        count = 0
        for m in prev_lattice:  # type: BaseMatching
            if m.stop:
                assert False  # should not happen
                continue
            count += 1
            if m.edge_m.is_point():
                # == Move to neighbour from node ==
                if max_dist is None:
                    nbrs = self.map.nodes_nbrto(m.edge_m.l1)
                else:
                    nbrs = self.map.nodes_closeto(m.edge_m.p1, max_dist=max_dist)
                # print("Neighbours for {}: {}".format(m, nbrs))
                if nbrs is None:
                    if __debug__:
                        logger.debug("No neighbours found for node {}".format(m.edge_m.l1))
                    continue
                if __debug__:
                    logger.debug("   + Move to {} neighbours from node {}".format(len(nbrs), m.edge_m.l1))
                    logger.debug(m.repr_header())
                for nbr_label, nbr_loc in nbrs:
                    # === Move from node to node (or stay on node) ===
                    if not self.only_edges:
                        edge_m = Segment(nbr_label, nbr_loc)
                        edge_o = Segment(f"O{obs_idx}", self.path[obs_idx])
                        m_next = m.next(edge_m, edge_o, obs=obs_idx)
                        if m_next is not None:
                            if inc_delayed:
                                m_next.delayed += 1
                            self._insert(m_next)
                            if __debug__:
                                logger.debug(str(m_next))

                    # === Move from node to edge ===
                    if m.edge_m.l1 != nbr_label:
                        edge_m = Segment(m.edge_m.l1, m.edge_m.p1, nbr_label, nbr_loc)
                        edge_o = Segment(f"O{obs_idx}", self.path[obs_idx])
                        m_next = m.next(edge_m, edge_o, obs=obs_idx)
                        if m_next is not None:
                            if inc_delayed:
                                m_next.delayed += 1
                            self._insert(m_next)
                            if __debug__:
                                logger.debug(str(m_next))
                    else:
                        if __debug__:
                            logger.debug(self.matching.repr_static(('x', f'{nbr_label}-{nbr_label} < self-loop')))

            else:
                # == Move to neighbour from edge ==
                if __debug__:
                    logger.debug("   + Move to neighbour from edge {}".format(m.label))
                    logger.debug(m.repr_header())

                # === Stay on edge ===
                edge_m = Segment(m.edge_m.l1, m.edge_m.p1, m.edge_m.l2, m.edge_m.p2)
                edge_o = Segment(f"O{obs_idx}", self.path[obs_idx])
                m_next = m.next(edge_m, edge_o, obs=obs_idx)
                if m_next is not None:
                    if inc_delayed:
                        m_next.delayed += 1
                    self._insert(m_next)
                    if __debug__:
                        logger.debug(str(m_next))

                # === Move from edge to node ===
                if not self.only_edges:
                    edge_m = Segment(m.edge_m.l2, m.edge_m.p2)
                    edge_o = Segment(f"O{obs_idx}", self.path[obs_idx])
                    m_next = m.next(edge_m, edge_o, obs=obs_idx)
                    if m_next is not None:
                        if inc_delayed:
                            m_next.delayed += 1
                        self._insert(m_next)
                        if __debug__:
                            logger.debug(str(m_next))

                else:
                    # === Move from edge to next edge ===
                    if max_dist is None:
                        nbrs = self.map.edges_nbrto((m.edge_m.l1, m.edge_m.l2))  # type: list
                    else:
                        nbrs = [(l1, p1, l2, p2) for _, l1, p1, l2, p2, _, _
                                in self.map.edges_closeto(m.edge_m.pi, max_dist=max_dist)]
                    if nbrs is None or len(nbrs) == 0:
                        if __debug__:
                            logger.debug(f"No neighbours found for edge {m.edge_m.label}")
                        continue
                    for nbr_label1, nbr_loc1, nbr_label2, nbr_loc2 in nbrs:
                        # same edge is different action, opposite edge should be allowed to return in a one-way street
                        if m.edge_m.l2 != nbr_label2 and m.edge_m.l1 != nbr_label1:
                            edge_m = Segment(nbr_label1, nbr_loc1, nbr_label2, nbr_loc2)
                            edge_o = Segment(f"O{obs_idx}", self.path[obs_idx])
                            m_next = m.next(edge_m, edge_o, obs=obs_idx)
                            if m_next is not None:
                                if inc_delayed:
                                    m_next.delayed += 1
                                self._insert(m_next)
                                if __debug__:
                                    mstr = str(m_next)
                                    logger.debug(mstr)
        if self.max_lattice_width:
            self.lattice[obs_idx].prune(0, self.max_lattice_width, self.expand_now)
        if count == 0:
            if __debug__:
                logger.debug("No active solution found anymore")
            return False
        return True

    def _match_non_emitting_states(self, obs_idx, expand=False):
        """Match sequences of nodes that all refer to the same observation at obs_idx.

        Assumptions:
        This method assumes that the lattice is filled up for both obs_idx and obs_idx + 1.

        :param obs_idx: Index of the first observation used (the second will be obs_idx + 1)
        :return: None
        """
        obs = self.path[obs_idx]
        if obs_idx < len(self.path) - 1:
            obs_next = self.path[obs_idx + 1]
        else:
            obs_next = None
        # The current states are the current observation's states
        if expand:
            cur_lattice = dict((m.key, m) for m in self.lattice[obs_idx].values(0) if not m.stop and m.delayed == self.expand_now)
        else:
            cur_lattice = dict((m.key, m) for m in self.lattice[obs_idx].values(0) if not (m.stop or m.delayed > 0))
        lattice_toinsert = list()
        # The current best states are the next observation's states if you would ignore non-emitting states
        lattice_best = dict((m.shortkey, m)
                            for m in self.lattice[obs_idx + 1].values(0) if not m.stop)
        lattice_ne = set(m.shortkey
                         for m in self.lattice[obs_idx + 1].values(0) if not m.stop and self._skip_ne_states(m))
        # cur_lattice = set(self.lattice[obs_idx].values())
        nb_ne = 0
        prune_thr = None
        while len(cur_lattice) > 0 and nb_ne < self.non_emitting_states_maxnb:
            nb_ne += 1
            if __debug__:
                logger.debug("--- obs {}:{} --- {} - {} ---".format(obs_idx, nb_ne, obs, obs_next))
            cur_lattice = self._match_non_emitting_states_inner(cur_lattice, obs_idx, obs, obs_next, nb_ne,
                                                                lattice_best, lattice_ne)
            if self.max_lattice_width is not None:
                self.lattice[obs_idx].prune(nb_ne, self.max_lattice_width, self.expand_now, prune_thr)
            # Link to next observation
            self._match_non_emitting_states_end(cur_lattice, obs_idx + 1, obs_next,
                                                lattice_best, expand=expand)
            if self.max_lattice_width is not None:
                prune_thr = self.lattice[obs_idx + 1].prune(0, self.max_lattice_width, self.expand_now, None)
        if self.max_lattice_width is not None:
            self.lattice[obs_idx + 1].prune(0, self.max_lattice_width, self.expand_now, None)
        # logger.info('Used {} levels of non-emitting states'.format(nb_ne))
        # for m in lattice_toinsert:
        #     self._insert(m)

    def _node_in_prev_ne(self, m_next, label):
        """Is the given node already visited in the chain of non-emitting states.

        :param m_next:
        :param label: Node label
        :return: True or False
        """
        # for m in itertools.chain(m_next.prev, m_next.prev_other):
        for m in m_next.prev:  # type: BaseMatching
            if m.obs != m_next.obs:
                return False
            assert(m_next.obs_ne != m.obs_ne)
            # print('prev', m.shortkey, 'checking for ', label)
            # if label == m.shortkey:
            if label in m.nodes:
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
                                         lattice_best, lattice_ne):
        # cur_lattice_new = dict()
        cur_lattice_new = self.lattice[obs_idx].dict(nb_ne)
        for m in cur_lattice.values():  # type: BaseMatching
            if m.stop or m.delayed != self.expand_now:
                continue
            if m.shortkey in lattice_ne:
                logger.debug(f"Skip non-emitting states from {m.label}, already visited")
                continue
            # == Move to neighbour edge from edge ==
            if m.edge_m.l2 is not None and self.only_edges:
                nbrs = self.map.edges_nbrto((m.edge_m.l1, m.edge_m.l2))
                # print("Neighbours for {}: {}".format(m, nbrs))
                if nbrs is None or len(nbrs) == 0:
                    if __debug__:
                        logger.debug(f"No neighbours found for edge {m.edge_m.label} ({m.label}, non-emitting)")
                    continue
                for nbr_label1, nbr_loc1, nbr_label2, nbr_loc2 in nbrs:
                    if self._node_in_prev_ne(m, nbr_label2):
                        if __debug__:
                            logger.debug(self.matching.repr_static(('x', '{} < node in prev ne'.format(nbr_label2))))
                        continue
                    # === Move to next edge ===
                    if m.edge_m.l2 != nbr_label2 and m.edge_m.l1 != nbr_label2:
                        edge_m = Segment(nbr_label1, nbr_loc1, nbr_label2, nbr_loc2)
                        edge_o = Segment(f"O{obs_idx}", obs, f"O{obs_idx+1}", obs_next)
                        m_next = m.next(edge_m, edge_o, obs=obs_idx, obs_ne=nb_ne)
                        if m_next is not None:
                            if m_next.key in cur_lattice_new:
                                if m_next.shortkey in lattice_best:
                                    if approx_leq(m_next.dist_obs, lattice_best[m_next.shortkey].dist_obs):
                                        cur_lattice_new[m_next.key].update(m_next)
                                    else:
                                        m_next.stop = True
                                        if __debug__ and logger.isEnabledFor(logging.DEBUG):
                                            logger.debug(f"   | Stopped trace: distance larger than best for key {m_next.shortkey}: "
                                                         f"{m_next.dist_obs} > {lattice_best[m_next.shortkey].dist_obs}")
                                else:
                                    cur_lattice_new[m_next.key].update(m_next)
                            else:
                                if m_next.shortkey in lattice_best:
                                    # if m_next.logprob > lattice_best[m_next.shortkey].logprob:
                                    if approx_leq(m_next.dist_obs, lattice_best[m_next.shortkey].dist_obs):
                                        cur_lattice_new[m_next.key] = m_next
                                        # lattice_best[m_next.shortkey] = m_next
                                        # lattice_toinsert.append(m_next)
                                    else:
                                        if __debug__ and logger.isEnabledFor(logging.DEBUG):
                                            logger.debug(f"   | Stopped trace: distance larger than best for key {m_next.shortkey}: "
                                                         f"{m_next.dist_obs} > {lattice_best[m_next.shortkey].dist_obs}")
                                        m_next.stop = True
                                else:
                                    cur_lattice_new[m_next.key] = m_next
                                    # lattice_best[m_next.shortkey] = m_next
                                    # lattice_toinsert.append(m_next)
                            # cur_lattice_new.add(m_next)
                            if __debug__:
                                logger.debug(str(m_next))
                    else:
                        if __debug__:
                            logger.debug(self.matching.repr_static(('x', f'{nbr_label1}-{nbr_label2} < goes back (ne)')))
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
                        f"   + Move to {len(nbrs)} neighbours from node {cur_node} ({m.label}, non-emitting)")
                    logger.debug(m.repr_header())
                for nbr_label, nbr_loc in nbrs:
                    # print(f"self._node_in_prev_ne({m.label}, {nbr_label}) = {self._node_in_prev_ne(m, nbr_label)}")
                    if self._node_in_prev_ne(m, nbr_label):
                        if __debug__:
                            logger.debug(self.matching.repr_static(('x', '{} < node in prev ne'.format(nbr_label))))
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
                                        # lattice_toinsert.append(m_next)
                                    elif __debug__ and logger.isEnabledFor(logging.DEBUG):
                                        m_next.stop = True
                                        cur_lattice_new[m_next.key] = m_next
                                        # lattice_toinsert.append(m_next)
                                else:
                                    cur_lattice_new[m_next.key] = m_next
                                    lattice_best[m_next.shortkey] = m_next
                                    # lattice_toinsert.append(m_next)
                            # cur_lattice_new.add(m_next)
                            if __debug__:
                                logger.debug(str(m_next))
                    else:
                        if __debug__:
                            logger.debug(f"x  | {m.edge_m.l1}-{nbr_label} < self-loop")

        return cur_lattice_new

    def _match_non_emitting_states_end(self, cur_lattice, obs_idx, obs_next,
                                       lattice_best, expand=False):
        for m in cur_lattice.values():  # type: BaseMatching
            if m.stop or m.delayed > self.expand_now:
                continue
            if m.edge_m.l2 is not None:
                # Move to neighbour edge from edge
                nbrs = self.map.edges_nbrto((m.edge_m.l1, m.edge_m.l2))
                # print("Neighbours for {}: {}".format(m, nbrs))
                if nbrs is None or len(nbrs) == 0:
                    if __debug__:
                        logger.debug("No neighbours found for edge {} ({})".format(m.edge_m.label, m.label))
                    continue
                if __debug__:
                    logger.debug(f"   + Move to {len(nbrs)} neighbours from edge {m.edge_m.label} "
                                 f"({m.label}, non-emitting->emitting)")
                    logger.debug(m.repr_header())
                for nbr_label1, nbr_loc1, nbr_label2, nbr_loc2 in nbrs:
                    if self._node_in_prev_ne(m, nbr_label2):
                        if __debug__:
                            logger.debug(self.matching.repr_static(('x', '{} < node in prev ne'.format(nbr_label2))))
                        continue
                    # Move to next edge
                    if m.edge_m.l1 != nbr_label2 and m.edge_m.l2 != nbr_label2:
                        edge_m = Segment(nbr_label1, nbr_loc1, nbr_label2, nbr_loc2)
                        edge_o = Segment(f"O{obs_idx+1}", obs_next)
                        m_next = m.next(edge_m, edge_o, obs=obs_idx)
                        if m_next is not None:
                            if m_next.shortkey in lattice_best:
                                # if m_next.dist_obs < lattice_best[m_next.shortkey].dist_obs:
                                if m_next.logprob > lattice_best[m_next.shortkey].logprob:
                                    lattice_best[m_next.shortkey] = m_next
                                    # lattice_toinsert.append(m_next)
                                    self.lattice[obs_idx].upsert(m_next)
                                elif __debug__ and logger.isEnabledFor(logging.DEBUG):
                                    m_next.stop = True
                                    # lattice_toinsert.append(m_next)
                                    self.lattice[obs_idx].upsert(m_next)
                            else:
                                lattice_best[m_next.shortkey] = m_next
                                # lattice_toinsert.append(m_next)
                                self.lattice[obs_idx].upsert(m_next)
                            if __debug__:
                                logger.debug(str(m_next))
                    else:
                        if __debug__:
                            logger.debug(self.matching.repr_static(('x', '{} < going back'.format(nbr_label2))))
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
                    logger.debug(f"   + Move to {len(nbrs)} neighbours from node {cur_node} "
                                 f"({m.label}, non-emitting->emitting)")
                    logger.debug(m.repr_header())
                for nbr_label, nbr_loc in nbrs:
                    if self._node_in_prev_ne(m, nbr_label):
                        if __debug__:
                            logger.debug(self.matching.repr_static(('x', '{} < node in prev ne'.format(nbr_label))))
                        continue
                    # Move to next node
                    if m.edge_m.l1 != nbr_label:
                        # edge_m = Segment(m.edge_m.l1, m.edge_m.p1, nbr_label, nbr_loc)
                        edge_m = Segment(nbr_label, nbr_loc)
                        edge_o = Segment(f"O{obs_idx+1}", obs_next)
                        m_next = m.next(edge_m, edge_o, obs=obs_idx)
                        if m_next is not None:
                            if m_next.shortkey in lattice_best:
                                # if m_next.dist_obs < lattice_best[m_next.shortkey].dist_obs:
                                if m_next.logprob > lattice_best[m_next.shortkey].logprob:
                                    lattice_best[m_next.shortkey] = m_next
                                    # lattice_toinsert.append(m_next)
                                    self.lattice[obs_idx].upsert(m_next)
                                elif __debug__ and logger.isEnabledFor(logging.DEBUG):
                                    m_next.stop = True
                                    # lattice_toinsert.append(m_next)
                                    self.lattice[obs_idx].upsert(m_next)
                            else:
                                lattice_best[m_next.shortkey] = m_next
                                # lattice_toinsert.append(m_next)
                                self.lattice[obs_idx].upsert(m_next)
                            if __debug__:
                                logger.debug(str(m_next))
                    else:
                        if __debug__:
                            logger.debug(self.matching.repr_static(('x', '{} < self-loop'.format(nbr_label))))

    def get_matching(self, identifier=None):
        m = None  # type: Optional[BaseMatching]
        if isinstance(identifier, BaseMatching):
            m = identifier
        elif identifier is None:
            col = self.lattice[len(self.lattice) - 1]
            for curm in col.values_all():
                if m is None or curm.logprob > m.logprob:
                    m = curm
        elif type(identifier) is int:
            # If integer, search for the best matching at this index in the lattice
            for cur_m in self.lattice[identifier].values_all():  # type:BaseMatching
                if not cur_m.stop and (m is None or cur_m.logprob > m.logprob):
                    m = cur_m
        elif type(identifier) is str:
            # If string, try to parse identifier
            parts = identifier.split('-')
            idx, ne, key = None, None, None
            if len(parts) == 4:
                nodea, nodeb, idx, ne = [int(part) for part in parts]
                key = (nodea, nodeb, idx, ne)
                col = self.lattice[idx]  # type: LatticeColumn
                col_ne = col.o[ne]
                m = col_ne[key]
            elif len(parts) == 3:
                node, idx, ne = [int(part) for part in parts]
                key = (node, idx, ne)
                col = self.lattice[idx]  # type: LatticeColumn
                col_ne = col.o[ne]
                m = col_ne[key]
            elif len(parts) == 1:
                m = None
                l1 = int(parts[0])
                for l in self.lattice.values():  # type: LatticeColumn
                    for curm in l.values_all():
                        if (curm.edge_m.l1 == l1 or curm.edge_m.l2 == l1) and \
                                (m is None or curm.logprob > m.logprob):
                            m = curm
            else:
                raise AttributeError(f'Unknown string format for matching. '
                                     'Expects <node>-<idx>-<ne> or <node>-<node>-<idx>-<ne>.')

        return m

    def get_matching_path(self, start_m):
        """List of Matching objects that end in the given Matching object."""
        start_m = self.get_matching(start_m)
        return self._build_matching_path(start_m)

    def get_node_path(self, start_m, only_nodes=False):
        """List of node/edge names that end in the given Matching object."""
        path = self.get_matching_path(start_m)
        node_path = [m.shortkey for m in path]
        if only_nodes:
            node_path = self.node_path_to_only_nodes(node_path)
        return node_path

    def get_path(self, only_nodes=True, allow_jumps=False, only_closest=True):
        """A list with all the nodes (no edges) the matched path passes through."""
        if only_nodes is False:
            return self.node_path
        if self.node_path is None or len(self.node_path) == 0:
            return []
        path = self.node_path_to_only_nodes(self.node_path, allow_jumps=allow_jumps)
        if only_closest:
            m = self.lattice_best[0]
            if m.edge_m.ti > 0.5:
                path.pop(0)
        return path

    def node_path_to_only_nodes(self, path, allow_jumps=False):
        """Path of nodes and edges to only nodes.

        :param path: List of node names or edges as (node name, node name)
        :param allow_jumps: Allow a path over edges that are not connected.
            This occurs when matches are added without an edge, for example,
            when searching for edges in the distance neighborhood instead in
            the graph.
        :return: List of node names
        """
        nodes = []
        prev_state = path[0]
        if type(prev_state) is tuple:
            nodes.append(prev_state[0])
            nodes.append(prev_state[1])
            prev_node = prev_state[1]
        else:
            nodes.append(prev_state)
            prev_node = prev_state
        for state in path[1:]:
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
                elif not allow_jumps:
                    raise Exception(f"State {state} does not have as previous node {prev_node}")
                else:
                    nodes.append(state[0])
                    nodes.append(state[1])
                    prev_node = state[1]
            else:
                raise Exception(f"Unknown type of state: {state} ({type(state)})")
            prev_state = state
        return nodes

    def _build_matching_path(self, start_m, max_depth=None):
        lattice_best = []
        node_max = start_m
        cur_depth = 0
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug(self.matching.repr_header(stop="             "))
        logger.debug("Start ({}): {}".format(node_max.obs, node_max))
        lattice_best.append(node_max)
        if node_max.is_emitting():
            cur_depth += 1
        # for obs_idx in reversed(range(start_idx)):
        if max_depth is None:
            max_depth = len(self.lattice) + 1
        while cur_depth < max_depth and len(node_max.prev) > 0:
            node_max_last = node_max
            node_max: Optional[BaseMatching] = None
            for prev_m in node_max_last.prev:
                if prev_m is not None and (node_max is None or prev_m.logprob > node_max.logprob):
                    node_max = prev_m
            if node_max is None:
                logger.error("Did not find a matching node for path point at index {}. ".format(node_max_last.obs) +
                             "Stopped building path.")
                break
            logger.debug("Max   ({}): {}".format(node_max.obs, node_max))
            lattice_best.append(node_max)
            if node_max.is_emitting():
                cur_depth += 1
        lattice_best = list(reversed(lattice_best))
        return lattice_best

    def _build_node_path(self, start_idx, unique=True, max_depth=None, last_is_e=False):
        """Build the path from the lattice.

        :param start_idx:
        :param unique:
        :param max_depth:
        :param last_is_e: Last matched lattice node should be an emitting state.
            In case the matching stops early, the longest path can be in between two observations
            and thus be a nonemitting state (which by definition has a lower probability than the
            last emitting state). If this argument is set to true, the longer match is preferred.
        :return:
        """
        node_max = None
        node_max_ne = 0
        if last_is_e:
            for m in self.lattice[start_idx].values_all():  # type:BaseMatching
                if not m.stop and (node_max is None or m.logprob > node_max.logprob):
                    node_max = m
        else:
            for m in self.lattice[start_idx].values_all():  # type:BaseMatching
                if not m.stop and (node_max is None or m.obs_ne > node_max_ne or m.logprob > node_max.logprob):
                    node_max_ne = m.obs_ne
                    node_max = m
        if node_max is None:
            logger.error("Did not find a matching node for path point at index {}".format(start_idx))
            return None

        self.lattice_best = self._build_matching_path(node_max, max_depth)

        node_path = [m.shortkey for m in self.lattice_best]
        if unique:
            self.node_path = []
            prev_node = None
            for node in node_path:
                if node != prev_node:
                    self.node_path.append(node)
                    prev_node = node
        else:
            self.node_path = node_path
        return self.node_path

    def increase_max_lattice_width(self, max_lattice_width, unique=False, tqdm=None):
        self.max_lattice_width = max_lattice_width
        return self.match(self.path, unique=unique, tqdm=tqdm, expand=True)

    def continue_with_distance(self, from_matches=None, k=2, nb_obs=2, max_dist=None):
        """Continue the matcher but ignore edges and allow jumps
        to nearby edged.

        :param from_matches: Search in the neigborhood of these matches
        :param k: If from_matches is not given, the k best matches are used
            in the last nb_obs observations since last early_stop_idx
        :praram nb_obs: If from_matches is not given, the k best matches are used
            in the last nb_obs observations since last early_stop_idx
        :param max_dist: Add edges that are maximally max_dist away from the previous
            match. If none, self.max_dist * 3 is used.
        """
        if from_matches is None:
            from_matches = self.best_last_matches(k=k, nb_obs=nb_obs)
        self.increase_delayed()
        if max_dist is None:
            max_dist = self.max_dist * 3
        for obs_idx, cur_matches in from_matches.items():
            self._match_states(obs_idx, prev_lattice=cur_matches, max_dist=max_dist, inc_delayed=True)

    def path_bb(self):
        """Get boundig box of matched path (if it exists, otherwise return None)."""
        path = self.path
        plat, plon = islice(zip(*path), 2)
        lat_min, lat_max = min(plat), max(plat)
        lon_min, lon_max = min(plon), max(plon)
        bb = lat_min, lon_min, lat_max, lon_max
        return bb

    def print_lattice(self, file=None, obs_idx=None, obs_ne=0, label_width=None, debug=False):
        if debug:
            xprint = logger.debug
        else:
            if file is None:
                file = sys.stdout
            xprint = lambda arg: print(arg, file=file)
        # print("Lattice:", file=file)
        if obs_idx is not None:
            idxs = [obs_idx]
        else:
            idxs = range(len(self.lattice))
        for idx in idxs:
            if len(self.lattice[idx]) > 0:
                if label_width is None:
                    label_width = 0
                    for m in self.lattice[idx].values(obs_ne):
                        label_width = max(label_width, len(str(m.label)))
                xprint("--- obs {} ---".format(idx))
                xprint(self.matching.repr_header(label_width=label_width))
                for m in sorted(self.lattice[idx].values(obs_ne), key=lambda t: str(t.label)):
                    xprint(m.__str__(label_width=label_width))

    def lattice_dot(self, file=None, precision=None, render=False):
        """Write the lattice as a Graphviz DOT file.

        :param file: File object to print to. Prints to stdout if None.
        :param precision: Precision of (log) probabilities.
        :param render: Try to render the generated Graphviz file.
        """
        if file is None:
            file = sys.stdout
        if precision is None:
            prfmt = ''
        else:
            prfmt = f'.{precision}f'
        print('digraph lattice {', file=file)
        print('\trankdir=LR;', file=file)
        # Vertices
        for idx_ob in range(len(self.lattice)):
            col = self.lattice[idx_ob]
            for idx_ne in range(len(col)):
                ms = col.values(idx_ne)
                if len(ms) == 0:
                    continue
                cnames = [(m.obs_ne, m.cname, m.stop, m.delayed) for m in ms]
                cnames.sort()
                cur_obs_ne = -1
                print('\t{\n\t\trank=same; ', file=file)
                for obs_ne, cname, stop, delayed in cnames:
                    if obs_ne != cur_obs_ne:
                        if cur_obs_ne != -1:
                            print('\t};\n\t{\n\t\trank=same; ', file=file)
                        cur_obs_ne = obs_ne
                    if stop:
                        options = 'label="{} x",color=gray,fontcolor=gray'.format(cname)
                    elif delayed > self.expand_now:
                        options = 'label="{} d{}",color=gray,fontcolor=gray'.format(cname, delayed)
                    elif self.expand_now != 0:
                        options = 'label="{} d{}"'.format(cname, delayed)
                    else:
                        options = 'label="{}  "'.format(cname)
                    print('\t\t{} [{}];'.format(cname, options), file=file)
                print('\t};', file=file)
        # Edges
        for idx_ob in range(len(self.lattice)):
            col = self.lattice[idx_ob]
            for idx_ne in range(len(col)):
                ms = col.values(idx_ne)
                if len(ms) == 0:
                    continue
                for m in ms:
                    for mp in m.prev:
                        if m.stop or m.delayed > self.expand_now:
                            options = ',color=gray,fontcolor=gray'
                        else:
                            options = ''
                        print(f'\t {mp.cname} -> {m.cname} [label="{m.logprob:{prfmt}}"{options}];', file=file)
                    for mp in m.prev_other:
                        if m.stop or m.delayed > self.expand_now:
                            options = ',color=gray,fontcolor=gray'
                        else:
                            options = ''
                        print(f'\t {mp.cname} -> {m.cname} [color=gray,label="{m.logprob:{prfmt}}"{options}];', file=file)
        print('}', file=file)
        if render and file is not None:
            import subprocess as sp
            from pathlib import Path
            from io import TextIOWrapper
            if isinstance(file, Path):
                fn = str(file.canonical())
            elif isinstance(file, TextIOWrapper):
                file.flush()
                fn = file.name
            else:
                fn = str(file)
            cmd = ['dot', '-Tpdf', '-O', fn]
            logger.debug(' '.join(cmd))
            sp.call(cmd)

    def print_lattice_stats(self, file=None, verbose=False):
        if file is None:
            file = sys.stdout
        print("Stats lattice", file=file)
        print("-------------", file=file)
        stats = OrderedDict()
        stats["nbr levels"] = len(self.lattice) if self.lattice else "?"
        total_nodes = 0
        max_nodes = 0
        min_nodes = 9999999
        if self.lattice:
            sizes = []
            for idx in range(len(self.lattice)):
                level = self.lattice[idx].values(0)
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
        if self.lattice_best and len(self.lattice_best) > 0:
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
        if self.lattice is None:
            return None
        counts = defaultdict(lambda: 0)
        for level in self.lattice.values():
            for m in level.values_all():
                counts[m.label] += 1
        return counts

    def inspect_early_stopping(self):
        """Analyze the lattice and try to find most plausible reason why the
        matching stopped early and print to stdout."""
        if self.early_stop_idx is None:
            print("No early stopping.")
            return
        col = self.lattice[self.early_stop_idx - 1]
        print("The last matched nodes or edges were:")
        first_row = True
        ignore = set()
        for ne_i in range(len(col.o) - 1, -1, -1):
            for v in col.o[ne_i].values():
                if v.key not in ignore:
                    if first_row:
                        print(v.repr_header())
                        first_row = False
                    print(v)
                ignore.update(r.key for r in v.prev)

    def best_last_matches(self, k=1, nb_obs=3):
        """Return the k best last matches.

        :param k: Number of best matches to keep for an observation
        :param nb_obs: How many last matched observations to consider
        """
        import heapq
        if self.early_stop_idx is None:
            col_idx = len(self.lattice) - 1
        else:
            col_idx = self.early_stop_idx - 1
        hh = []
        obs_cnt = 0
        while col_idx >= 0 and obs_cnt < nb_obs:
            h = []
            col = self.lattice[col_idx]
            col_oneselected = False
            for ne_i in range(len(col.o) - 1, -1, -1):
                for v in col.o[ne_i].values():
                    if v.stop:
                        continue
                    if len(h) < k:
                        heapq.heappush(h, (v.logprob, v))
                        col_oneselected = True
                    elif v.logprob > h[0][0]:
                        heapq.heappop(h)
                        heapq.heappush(h, (v.logprob, v))
                        col_oneselected = True
            hh.extend(h)
            if col_oneselected is False:
                print(f'break in {col_idx=}')
                break
            col_idx -= 1
            obs_cnt += 1
        result = defaultdict(list)
        for m in hh:
            m = m[1]
            result[m.obs + 1].append(m)
        # return [m[1] for m in hh]
        return result

    def copy_lastinterface(self, nb_interfaces=1):
        """Copy the current matcher and keep the last interface as the start point.

        This method allows you to perform incremental matching without keeping the entire
        lattice in memory.

        You need to run :meth:`match_incremental` on this object to continue from the existing
        (partial) lattice. Otherwise, if you use :meth:`match`, it will be overwritten.

        Open question, if there is no need to keep track of older lattices, it will probably
        be more efficient to clear the older parts of the interface instead of copying the newer
        parts.

        :param nb_interfaces: Nb of interfaces (columns in lattice) to keep. Default is 1, the last one.
        :return: new Matcher object
        """
        matcher = self.__class__(self.map, obs_noise=self.obs_noise, max_dist_init=self.max_dist_init,
                                 max_dist=self.max_dist, min_prob_norm=self.min_logprob_norm,
                                 non_emitting_states=self.non_emitting_states,
                                 max_lattice_width=self.max_lattice_width, only_edges=self.only_edges,
                                 obs_noise_ne=self.obs_noise_ne, matching=self.matching,
                                 avoid_goingback=self.avoid_goingback,
                                 non_emitting_length_factor=math.exp(self.ne_length_factor_log))
        matcher.lattice = []
        matcher.path = []
        for int_i in range(len(self.lattice) - nb_interfaces, len(self.lattice)):
            matcher.lattice.append(self.lattice[int_i])
            matcher.path.append(self.path[int_i])
        return matcher

    @property
    def path_pred(self):
        """The matched path, both nodes and/or edges (depending on your settings)."""
        return self.node_path

    @property
    def path_pred_onlynodes(self):
        """A list with all the nodes (no edges) the matched path passes through."""
        return self.get_path(only_nodes=True, allow_jumps=False)

    @property
    def path_pred_onlynodes_withjumps(self):
        """A list with all the nodes (no edges) the matched path passes through."""
        return self.get_path(only_nodes=True, allow_jumps=True)

    def path_pred_distance(self):
        """Total distance of the matched path."""
        if self.lattice_best is None:
            return None
        if len(self.lattice_best) == 1:
            return 0
        dist = 0
        m_prev = self.lattice_best[0]
        for idx, m in enumerate(self.lattice_best[1:]):
            if m_prev.edge_m.label != m.edge_m.label and m_prev.edge_m.l2 == m.edge_m.l1:
                # Go over the connection between two edges to compute the distance
                cdist = self.map.distance(m_prev.edge_m.pi, m_prev.edge_m.p2)
                cdist += self.map.distance(m_prev.edge_m.p2, m.edge_m.pi)
            else:
                cdist = self.map.distance(m_prev.edge_m.pi, m.edge_m.pi)
            dist += cdist
            m_prev = m
        return dist

    def path_distance(self):
        """Total distance of the observations."""
        if self.lattice_best is None:
            return None
        if len(self.lattice_best) == 1:
            return 0
        dist = 0
        m_prev = self.lattice_best[0]
        for m in self.lattice_best[1:]:
            dist += self.map.distance(m_prev.edge_o.pi, m.edge_o.pi)
            m_prev = m
        return dist

    def path_all_distances(self):
        """Return a list of all distances between observed trace and map.

        One entry for each point in the map and point in the trace that are mapped to each other.
        In case non-emitting nodes are used, extra entries can be present where a point in the trace
        or a point in the map is mapped to a segment.
        """
        path = self.lattice_best
        dists = [m.dist_obs for m in path]
        return dists
