# encoding: utf-8
"""
leuvenmapmatching.matching_distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Take distance between observations vs states into account. Based on the
method presented in Newson and Krumm (2009).

:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
from .matching import Matching, Matcher
from scipy.stats import norm
import math
import logging


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


class MatchingDistance(Matching):
    __slots__ = ['dist_betw_obs', 'dist_betw_states']  # Additional fields

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if logger.isEnabledFor(logging.DEBUG):
            self.dist_betw_obs: float = 1.0
            self.dist_betw_states: float = 1.0
            if len(self.prev) != 0:
                m_prev = list(self.prev)[0]  # type: MatchingDistance
                if m_prev.edge_m.label == self.label:
                    self.dist_betw_states = self.matcher.map.distance(m_prev.edge_m.pi, self.edge_m.pi)
                else:
                    self.dist_betw_states = self.matcher.map.distance(m_prev.edge_m.pi, m_prev.edge_m.p2) +\
                                            self.matcher.map.distance(m_prev.edge_m.p2, self.edge_m.pi)
                self.dist_betw_obs = self.matcher.map.distance(m_prev.edge_o.pi, self.edge_o.pi)

    @staticmethod
    def repr_header(label_width=None, stop=""):
        res = Matching.repr_header(label_width)
        if logger.isEnabledFor(logging.DEBUG):
            res += f" {'d(o)':<5} | {'d(s)':<5} |"
        return res

    def __str__(self, label_width=None):
        res = super().__str__(label_width)
        if logger.isEnabledFor(logging.DEBUG):
            res += f" {self.dist_betw_obs:<5.2f} | {self.dist_betw_states:<5.2f} |"
        return res


class MatcherDistance(Matcher):

    def __init__(self, *args, **kwargs):
        """
        Newson and Krumm defaults:

        - max_dist = 200 m
        - obs_noise = 4.07 m
        - only_edges = True
        - non_emitting_states = False

        Transition and emission probability are transformed from densities to probababilities by
        taking the 1 - CDF instead of the PDF.

        """
        if not kwargs.get("only_edges", True):
            logger.warning("The MatcherDistance method only works on edges as states. Nodes have been disabled.")
        kwargs["only_edges"] = True
        if "matching" not in kwargs:
            kwargs["matching"] = MatchingDistance
        super().__init__(*args, **kwargs)
        self.obs_noise_dist = norm(scale=self.obs_noise)
        self.obs_noise_dist_ne = norm(scale=self.obs_noise_ne)

    def logprob_trans(self, prev_m: MatchingDistance, edge_m, edge_o):
        """Transition probability.

        Original PDF:
        p(dt) = 1 / beta * e^(-dt / beta)

        Transformed to probability:
        P(dt) = p(d > dt) = e^(-dt / beta)

        Main difference with Newson and Krumm: we know all points are connected thus do not compute the
        shortest path but the distance between two points.

        :param prev_m:
        :param edge_m:
        :param edge_o:
        :return:
        """
        d_z = self.map.distance(prev_m.edge_o.pi, edge_o.pi)
        if prev_m.edge_m.label == edge_m.label:
            d_x = self.map.distance(prev_m.edge_m.pi, edge_m.pi)
        else:
            d_x = self.map.distance(prev_m.edge_m.pi, prev_m.edge_m.p2) + self.map.distance(prev_m.edge_m.p2, edge_m.pi)
        d_t = abs(d_z - d_x)
        beta = 1 / 6
        # p_dt = 1 / beta * math.exp(-d_t / beta)
        icp_dt = math.exp(-d_t / beta)
        licp_dt = math.log(icp_dt)
        return licp_dt

    def logprob_obs(self, dist, prev_m, new_edge_m, new_edge_o):
        """Emission probability for emitting states."""
        print(dist)
        result = 2 * (1 - self.obs_noise_dist.cdf(dist))
        if result == 0:
            return -float("inf")
        return math.log(result)

    def logprob_obs_ne(self, dist, prev_m, new_edge_m, new_edge_o):
        """Emission probability for non-emitting states."""
        result = 2 * (1 - self.obs_noise_dist_ne.cdf(dist))
        if result == 0:
            return -float("inf")
        return math.log(result)
