# encoding: utf-8
"""
leuvenmapmatching.matcher.newsonkrumm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods similar to Newson Krumm 2009 for comparison purposes.

    P. Newson and J. Krumm. Hidden markov map matching through noise and sparseness.
    In Proceedings of the 17th ACM SIGSPATIAL international conference on advances
    in geographic information systems, pages 336–343. ACM, 2009.



:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
from scipy.stats import norm
import math
import logging

from .base import BaseMatching, BaseMatcher

logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


class NewsonKrummMatching(BaseMatching):
    __slots__ = ['d_s', 'd_o', 'lpe', 'lpt']  # Additional fields

    def __init__(self, *args, d_s=1.0, d_o=1.0, lpe=0.0, lpt=0.0, **kwargs):
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
        # type: (NewsonKrummMatching, NewsonKrummMatching) -> None
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


class NewsonKrummMatcher(BaseMatcher):
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
            kwargs["matching"] = NewsonKrummMatching
        super().__init__(*args, **kwargs)

        # if not use_original, the following value for beta gives a prob of 0.5 at dist=x_half:
        # beta = np.sqrt(np.power(x_half, 2) / (np.log(2)*2))
        self.beta = kwargs.get('beta', 1/6)
        self.beta_ne = kwargs.get('beta_ne', self.beta)

        self.obs_noise_dist = norm(scale=self.obs_noise)
        self.obs_noise_dist_ne = norm(scale=self.obs_noise_ne)
        self.ne_thr = 1.25

    def logprob_trans(self, prev_m: NewsonKrummMatching, edge_m, edge_o,
                      is_prev_ne=False, is_next_ne=False):
        """Transition probability.

        Main difference with Newson and Krumm: we know all points are connected thus do not compute the
        shortest path but the distance between two points.

        Original PDF:
        p(dt) = 1 / beta * e^(-dt / beta)
        with beta = 1/6

        Transformed to probability:
        P(dt) = p(d > dt) = e^(-dt / beta)

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
        # icp_dt = math.exp(-d_t / beta)
        # try:
        #     licp_dt = math.log(icp_dt)
        # except ValueError:
        #     licp_dt = float('-inf')
        licp_dt = -d_t / beta
        props = {
            'd_o': d_z,
            'd_s': d_x,
            'lpt': licp_dt
        }
        return licp_dt, props

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
        try:
            result = math.log(result)
        except ValueError:
            result = -float("inf")
        props = {
            'lpe': result
        }
        return result, props
