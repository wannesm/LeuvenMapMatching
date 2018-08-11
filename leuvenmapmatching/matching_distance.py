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


class MatchingDistance(Matching):
    __slots__ = ['dist_betw_obs', 'dist_betw_states']  # Additional fields

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_betw_obs: float = 0.0
        self.dist_betw_states: float = 1.0
        if len(self.prev) != 0:
            m_prev = list(self.prev)[0]  # type: MatchingDistance
            self.dist_betw_states = self.matcher.map.distance(m_prev.edge_m.pi, self.edge_m.pi)
            self.dist_betw_obs = self.matcher.map.distance(m_prev.edge_o.pi, self.edge_o.pi)

    @staticmethod
    def repr_header(label_width=None, stop=""):
        res = Matching.repr_header(label_width)
        res += f"{'d(o)':<5} | {'d(s)':<5} |"
        return res

    def __str__(self, label_width=None):
        res = super().__str__(label_width)
        res += f"{self.dist_betw_obs:<5.2f} | {self.dist_betw_states:<5.2f} |"
        return res


class MatcherDistance(Matcher):

    def __init__(self, *args, **kwargs):
        """
        Newson and Krumm defaults:

        - max_dist = 200 m
        - obs_noise = 4.07 m
        - only_edges = True
        - non_emitting_states = False

        """
        super().__init__(*args, **kwargs)
        self.obs_noise_dist = norm(scale=self.obs_noise)
        self.obs_noise_dist_ne = norm(scale=self.obs_noise_ne)

    def logprob_trans(self, prev_m: MatchingDistance, next_label=None, next_pos=None, next_obs=None):
        """Transition probability.

        P(dt) = 1 / beta * e^(-dt / beta)

        Main difference with Newson and Krumm: we know all points are connected thus do not compute the
        shortest path but the distance between two points.

        :param prev_m:
        :param next_label:
        :param next_pos:
        :param next_obs:
        :return:
        """
        d_z = self.map.distance(prev_m.edge_o.pi, next_obs)
        d_x = self.map.distance(prev_m.edge_m.pi, next_pos)
        d_t = abs(d_z - d_x)
        beta = 1 / 6
        p_dt = 1 / beta * math.exp(-d_t / beta)
        return p_dt

    def logprob_obs(self, dist, prev_m, new_edge_m, new_edge_o):
        """Emission probability for emitting states."""
        result = self.obs_noise_dist.logpdf(dist)
        return result

    def logprob_obs_ne(self, dist, prev_m, new_edge_m, new_edge_o):
        """Emission probability for non-emitting states."""
        result = self.obs_noise_dist_ne.logpdf(dist)
        return result
