# encoding: utf-8
"""
leuvenmapmatching.matching_distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        with beta = 1/6

        Transformed to probability:
        P(dt) = p(d > dt) = e^(-dt / beta)

        Main difference with Newson and Krumm: we know all points are connected thus do not compute the
        shortest path but the distance between two points.

        :param prev_m:
        :param edge_m:
        :param edge_o:
        :return:
        """
        if prev_m.label == "A-B-3-0":
            print("XXX")
        d_z = self.map.distance(prev_m.edge_o.pi, edge_o.pi)
        if prev_m.edge_m.label == edge_m.label:
            print(f"d_x(1), {prev_m.edge_m}, {edge_m}")
            d_x = self.map.distance(prev_m.edge_m.pi, edge_m.pi)
        else:
            print(f"d_x(2), {prev_m.edge_m}, {edge_m}")
            print(f"self.map.distance({prev_m.edge_m.pi}, {prev_m.edge_m.p2}) = {self.map.distance(prev_m.edge_m.pi, prev_m.edge_m.p2)}")
            print(f"self.map.distance({prev_m.edge_m.p2}, {edge_m.pi}) = {self.map.distance(prev_m.edge_m.p2, edge_m.pi)}")
            print(f"obs: {prev_m.edge_o.pi}, {edge_o.pi}")
            d_x = self.map.distance(prev_m.edge_m.pi, prev_m.edge_m.p2) + self.map.distance(prev_m.edge_m.p2, edge_m.pi)
        d_t = abs(d_z - d_x)
        beta = 1 / 6
        # p_dt = 1 / beta * math.exp(-d_t / beta)
        icp_dt = math.exp(-d_t / beta)
        try:
            licp_dt = math.log(icp_dt)
        except ValueError:
            licp_dt = float('-inf')
        return licp_dt

    def logprob_obs(self, dist, prev_m, new_edge_m, new_edge_o):
        """Emission probability for emitting states.

        Original pdf:
        p(d) = N(0, sigma)
        with sigma = 4.07m

        Transformed to probability:
        P(d) = 2 * (1 - p(d > D)) = 2 * (1 - cdf)

        """
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
