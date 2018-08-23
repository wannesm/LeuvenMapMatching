# encoding: utf-8
"""
leuvenmapmatching.matcher.distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
from scipy.stats import norm
import math
import logging

from .base import BaseMatching, BaseMatcher

logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


class DistanceMatching(BaseMatching):
    __slots__ = ['dist_betw_obs', 'dist_betw_states', 'lgprd']  # Additional fields

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_betw_obs: float = 1.0
        self.dist_betw_states: float = 1.0
        self.lgprd: float = 0.0
        if len(self.prev) != 0:
            m_prev = list(self.prev)[0]  # type: DistanceMatching
            if m_prev.edge_m.label == self.edge_m.label:
                self.dist_betw_states = self.matcher.map.distance(m_prev.edge_m.pi, self.edge_m.pi)
            else:
                self.dist_betw_states = self.matcher.map.distance(m_prev.edge_m.pi, m_prev.edge_m.p2) + \
                                        self.matcher.map.distance(m_prev.edge_m.p2, self.edge_m.pi)
            self.dist_betw_obs = self.matcher.map.distance(m_prev.edge_o.pi, self.edge_o.pi)
            if logger.isEnabledFor(logging.DEBUG):
                d_t = abs(self.dist_betw_states - self.dist_betw_obs)
                beta = 1 / 6
                icp_dt = math.exp(-d_t / beta)
                try:
                    licp_dt = math.log(icp_dt)
                except ValueError:
                    licp_dt = float('-inf')
                self.lgprd = licp_dt

    def _update_inner(self, m_other: 'DistanceMatching'):
        super()._update_inner(m_other)
        self.dist_betw_states = m_other.dist_betw_states
        self.dist_betw_obs = m_other.dist_betw_obs
        self.lgprd = m_other.lgprd

    @staticmethod
    def repr_header(label_width=None, stop=""):
        res = BaseMatching.repr_header(label_width)
        res += f" {'d(o)':<5} | {'d(s)':<5} |"
        if logger.isEnabledFor(logging.DEBUG):
            res += f" {'lg(Pr-d)':<5}"
        return res

    def __str__(self, label_width=None):
        res = super().__str__(label_width)
        res += f" {self.dist_betw_obs:<5.2f} | {self.dist_betw_states:<5.2f} |"
        if logger.isEnabledFor(logging.DEBUG):
            res += f" {self.lgprd:5.2f} |"
        return res


class DistanceMatcher(BaseMatcher):
    """
    Take distance between observations vs states into account. Based on the
    method presented in:

        P. Newson and J. Krumm. Hidden markov map matching through noise and sparseness.
        In Proceedings of the 17th ACM SIGSPATIAL international conference on advances
        in geographic information systems, pages 336–343. ACM, 2009.

    Two important differences:

    * Newson and Krumm use shortest path to handle situations where the distances between
      observations are larger than distances between nodes in the graph. The LeuvenMapMatching
      toolbox uses non-emitting states to handle this. We thus do not implement the shortest
      path algorithm in this class.
    * Transition and emission probability are transformed from densities to probababilities by
      taking the 1 - CDF instead of the PDF.


    Newson and Krumm defaults:

    - max_dist = 200 m
    - obs_noise = 4.07 m
    - beta = 1/6
    - only_edges = True
    """

    def __init__(self, *args, **kwargs):
        """

        :param beta: Default is 1/6
        :param beta_ne: Default is beta
        :param args: Arguments for BaseMatcher
        :param kwargs: Arguments for BaseMatcher
        """

        if not kwargs.get("only_edges", True):
            logger.warning("The MatcherDistance method only works on edges as states. Nodes have been disabled.")
        kwargs["only_edges"] = True
        if "matching" not in kwargs:
            kwargs["matching"] = DistanceMatching
        super().__init__(*args, **kwargs)
        self.beta = kwargs.get('beta', 1/6)
        self.beta_ne = kwargs.get('beta_ne', self.beta)
        self.obs_noise_dist = norm(scale=self.obs_noise)
        self.obs_noise_dist_ne = norm(scale=self.obs_noise_ne)
        self.ne_thr = 1.25

    def logprob_trans(self, prev_m: DistanceMatching, edge_m, edge_o,
                      is_prev_ne=False, is_next_ne=False):
        """Transition probability.

        Original PDF:
        p(dt) = 1 / beta * e^(-dt / beta)
        with beta = 1/6

        Transformed to probability:
        P(dt) = p(d > dt) = e^(-dt / beta)

        Main difference with Newson and Krumm: we know all points are connected thus do not compute the
        shortest path but the distance between two points.

        :param prev_m:
        :param edge_m:
        :param edge_o:
        :param is_prev_ne:
        :param is_next_ne:
        :return:
        """
        d_z = self.map.distance(prev_m.edge_o.pi, edge_o.pi)
        if prev_m.edge_m.label == edge_m.label:
            d_x = self.map.distance(prev_m.edge_m.pi, edge_m.pi)
        else:
            d_x = self.map.distance(prev_m.edge_m.pi, prev_m.edge_m.p2) + self.map.distance(prev_m.edge_m.p2, edge_m.pi)
        d_t = abs(d_z - d_x)
        # p_dt = 1 / beta * math.exp(-d_t / beta)
        if is_prev_ne or is_next_ne:
            beta = self.beta_ne
        else:
            beta = self.beta
        icp_dt = math.exp(-d_t / beta)
        try:
            licp_dt = math.log(icp_dt)
        except ValueError:
            licp_dt = float('-inf')
        return licp_dt

    def logprob_obs(self, dist, prev_m, new_edge_m, new_edge_o, is_ne=False):
        """Emission probability for emitting states.

        Original pdf:
        p(d) = N(0, sigma)
        with sigma = 4.07m

        Transformed to probability:
        P(d) = 2 * (1 - p(d > D)) = 2 * (1 - cdf)

        """
        if is_ne:
            result = 2 * (1 - self.obs_noise_dist_ne.cdf(dist))
        else:
            result = 2 * (1 - self.obs_noise_dist.cdf(dist))
        if result == 0:
            return -float("inf")
        return math.log(result)

    def _skip_ne_states(self, obs_idx):
        # Skip searching for non-emitting states when the distances between nodes
        # on the map are similar to the distances between the observation
        min_ne_factor = self.ne_thr * 2
        for m in self.lattice[obs_idx].values():
            if m.dist_betw_states > 0:
                min_ne_factor = min(min_ne_factor, m.dist_betw_obs / m.dist_betw_states)
        if min_ne_factor < self.ne_thr:
            logger.debug(f"Skip non-emitting states between {obs_idx - 1} and {obs_idx}, "
                         f"{min_ne_factor} < {self.ne_thr}")
            return True
        return False
