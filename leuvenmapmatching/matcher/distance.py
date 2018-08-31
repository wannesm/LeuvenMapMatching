# encoding: utf-8
"""
leuvenmapmatching.matcher.distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import logging
import math

from .base import BaseMatching, BaseMatcher

logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


class DistanceMatching(BaseMatching):
    __slots__ = ['d_s', 'd_o', 'lpe', 'lpt']  # Additional fields

    def __init__(self, *args, d_s=0.0, d_o=0.0, lpe=0.0, lpt=0.0, **kwargs):
        """

        :param args: Arguments for BaseMatching
        :param d_s: Distance between two (interpolated) states
        :param d_o: Distance between two (interpolated) observations
        :param lpe: Log probability of emission
        :param lpt: Log probablity of transition
        :param kwargs: Arguments for BaseMatching
        """
        super().__init__(*args, **kwargs)
        self.d_o: float = d_o
        self.d_s: float = d_s
        self.lpe: float = lpe
        self.lpt: float = lpt

    def _update_inner(self, m_other):
        # type: (DistanceMatching, DistanceMatching) -> None
        super()._update_inner(m_other)
        self.d_s = m_other.d_s
        self.d_o = m_other.d_o
        self.lpe = m_other.lpe
        self.lpt = m_other.lpt

    @staticmethod
    def repr_header(label_width=None, stop=""):
        res = BaseMatching.repr_header(label_width)
        res += f" {'dt(o)':<6} | {'dt(s)':<6} |"
        if logger.isEnabledFor(logging.DEBUG):
            res += f" {'lg(Pr-t)':<9} | {'lg(Pr-e)':<9} |"
        return res

    def __str__(self, label_width=None):
        res = super().__str__(label_width)
        res += f" {self.d_o:>6.2f} | {self.d_s:>6.2f} |"
        if logger.isEnabledFor(logging.DEBUG):
            res += f" {self.lpt:>9.2f} | {self.lpe:>9.2f} |"
        return res


class DistanceMatcher(BaseMatcher):
    """
    Take distance between observations vs states into account. Inspired on the
    method presented in:

        P. Newson and J. Krumm. Hidden markov map matching through noise and sparseness.
        In Proceedings of the 17th ACM SIGSPATIAL international conference on advances
        in geographic information systems, pages 336–343. ACM, 2009.
    """

    def __init__(self, *args, **kwargs):
        """Map Matching that takes into account the distance on the map wrt distance between the
        observations.

        :param map_con: Map object to connect to map database
        :param obs_noise: Standard deviation of noise
        :param obs_noise_ne: Standard deviation of noise for non-emitting states (is set to obs_noise if not given)
        :param max_dist_init: Maximum distance from start location (if not given, uses max_dist)
        :param max_dist: Maximum distance from path (this is a hard cut, min_prob_norm should be better)
        :param min_prob_norm: Minimum normalized probability of observations (ema)
        :param non_emitting_states: Allow non-emitting states. A non-emitting state is a state that is
            not associated with an observation. Here we assume it can be associated with a location in between
            two observations to allow for pruning. It is advised to set min_prob_norm and/or max_dist to avoid
            visiting all possible nodes in the graph.
        :param non_emitting_length_factor: Reduce the probability of a sequence of non-emitting states the longer it
            is. This can be used to prefer shorter paths. This is separate from the transition probabilities because
            transition probabilities are averaged for non-emitting states and thus the length is also averaged out.
        :param max_lattice_width: Restrict the lattice (or possible candidate states per observation) to this value.
            If there are more possible next states, the states with the best likelihood so far are selected.

        :param dist_noise: Standard deviation of difference between distance between states and distance
            between observatoins. If not given, set to obs_noise
        :param dist_noise_ne: If not given, set to dist_noise
        :param restrained_ne: Avoid non-emitting states if the distance between states and between
            observations is close to each other.
        :param avoid_goingback: If true, the probability is lowered for a transition that returns back to a
            previous edges or returns to a position on an edge.

        :param args: Arguments for BaseMatcher
        :param kwargs: Arguments for BaseMatcher
        """

        if not kwargs.get("only_edges", True):
            logger.warning("The MatcherDistance method only works on edges as states. Nodes have been disabled.")
        kwargs["only_edges"] = True
        if "matching" not in kwargs:
            kwargs["matching"] = DistanceMatching
        super().__init__(*args, **kwargs)
        self.use_original = kwargs.get('use_original', False)

        # if not use_original, the following value for beta gives a prob of 0.5 at dist=x_half:
        # beta = np.sqrt(np.power(x_half, 2) / (np.log(2)*2))
        self.dist_noise = kwargs.get('dist_noise', self.obs_noise)
        self.dist_noise_ne = kwargs.get('dist_noise_ne', self.dist_noise)
        self.beta = 2 * self.dist_noise**2
        self.beta_ne = 2 * self.dist_noise_ne ** 2

        self.sigma = 2 * self.obs_noise**2
        self.sigma_ne = 2 * self.obs_noise_ne ** 2

        self.restrained_ne = kwargs.get('restrained_ne', True)
        self.restrained_ne_thr = 1.25  # Threshold
        self.exact_dt_s = True  # Newson and Krumm is 'True'

        self.avoid_goingback = kwargs.get('avoid_goingback', True)
        self.gobackonedge_factor_log = math.log(0.5)
        self.gobacktoedge_factor_log = math.log(0.5)

        self.notconnectededges_factor_log = math.log(0.5)

    def logprob_trans(self, prev_m: DistanceMatching, edge_m, edge_o,
                      is_prev_ne=False, is_next_ne=False):
        """Transition probability.

        :math:`P(dt) = exp(-d_t^2 / (2 * dist_{noise}^2))`

        with :math:`d_t = |d_s - d_o|,
        d_s = |loc_{prev\_state} - loc_{cur\_state}|,
        d_o = |loc_{prev\_obs} - loc_{cur\_obs}|`

        This function is more tolerant for low values. The intuition is that values under a certain
        distance should all be close to probability 1.0.

        Note: We could also smooth the distance between observations to handle outliers better.

        :param prev_m:
        :param edge_m:
        :param edge_o:
        :param is_prev_ne:
        :param is_next_ne:
        :return:
        """
        d_z = self.map.distance(prev_m.edge_o.pi, edge_o.pi)
        if ((not self.exact_dt_s) or
            prev_m.edge_m.label == edge_m.label or  # On same edge
            prev_m.edge_m.l2 != edge_m.l1):  # Edges are not connected
            d_x = self.map.distance(prev_m.edge_m.pi, edge_m.pi)
        else:
            d_x = self.map.distance(prev_m.edge_m.pi, prev_m.edge_m.p2) + self.map.distance(prev_m.edge_m.p2, edge_m.pi)
        d_t = abs(d_z - d_x)
        # p_dt = 1 / beta * math.exp(-d_t / beta)
        if is_prev_ne or is_next_ne:
            beta = self.beta_ne
        else:
            beta = self.beta
        logprob = -d_t**2 / beta

        # Penalties
        if prev_m.edge_m.label == edge_m.label:
            # Staying in same state
            if self.avoid_goingback and edge_m.key == prev_m.edge_m.key and edge_m.ti < prev_m.edge_m.ti:
                # Going back on edge
                logprob += self.gobackonedge_factor_log  # Prefer not going back
        else:
            # Moving states
            if prev_m.edge_m.l2 != edge_m.l1:
                # We are moving between states that represent edges that are not connected through a node
                logprob += self.notconnectededges_factor_log
            elif self.avoid_goingback:
                # Goin back on state
                going_back = False
                for m in prev_m.prev:
                    if edge_m.label == m.edge_m.label:
                        going_back = True
                        break
                if going_back:
                    logprob += self.gobacktoedge_factor_log  # prefer not going back

        props = {
            'd_o': d_z,
            'd_s': d_x,
            'lpt': logprob
        }
        return logprob, props

    def logprob_obs(self, dist, prev_m, new_edge_m, new_edge_o, is_ne=False):
        """Emission probability for emitting states.

        :math:`P(dt) = exp(-d_o^2 / (2 * obs_{noise}^2))`

        with :math:`d_o = |loc_{state} - loc_{obs}|`

        """
        if is_ne:
            sigma = self.sigma_ne
        else:
            sigma = self.sigma
        result = -dist**2 / sigma
        props = {
            'lpe': result
        }
        return result, props

    def _skip_ne_states(self, next_ne_m):
        # type: (DistanceMatcher, DistanceMatching) -> bool
        # Skip searching for non-emitting states when the distances between nodes
        # on the map are similar to the distances between the observation
        if not self.restrained_ne:
            return False
        if next_ne_m.d_s > 0:
            factor = (next_ne_m.d_o + next_ne_m.dist_obs) / next_ne_m.d_s
        else:
            factor = 0
        if factor < self.restrained_ne_thr:
            logger.debug(f"Skip non-emitting states to {next_ne_m.label}: {factor} < {self.restrained_ne_thr}")
            return True
        return False
