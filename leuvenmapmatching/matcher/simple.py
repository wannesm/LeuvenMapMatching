# encoding: utf-8
"""
leuvenmapmatching.matcher.simple
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2015-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import math
from scipy.stats import halfnorm, norm

from .base import BaseMatcher, BaseMatching
from ..util.segment import Segment


class SimpleMatching(BaseMatching):
    pass


class SimpleMatcher(BaseMatcher):

    def __init__(self, *args, **kwargs):
        """

        :param avoid_goingback: Change the transition probability to be lower for the direction the path is coming
            from.
        :param kwargs: Arguments passed to :class:`BaseMatcher`.
        """
        if "matching" not in kwargs:
            kwargs['matching'] = SimpleMatching
        super().__init__(*args, **kwargs)

        self.obs_noise_dist = halfnorm(scale=self.obs_noise)
        self.obs_noise_dist_ne = halfnorm(scale=self.obs_noise_ne)
        # normalize to max 1 to simulate a prob instead of density
        self.obs_noise_logint = math.log(self.obs_noise * math.sqrt(2 * math.pi) / 2)
        self.obs_noise_logint_ne = math.log(self.obs_noise_ne * math.sqrt(2 * math.pi) / 2)

        # Transition probability is divided (in logprob_trans) by this factor if we move back on the
        # current edge.
        self.avoid_goingback = kwargs.get('avoid_goingback', True)
        self.gobackonedge_factor_log = math.log(0.99)
        # Transition probability is divided (in logprob_trans) by this factor if the next state is
        # also the previous state, thus if we go back
        self.gobacktoedge_factor_log = math.log(0.5)
        # Transition probability is divided (in logprob_trans) by this factor if a transition is made
        self.transition_factor = math.log(0.9)

    def logprob_trans(self, prev_m:BaseMatching, edge_m:Segment, edge_o:Segment,
                      is_prev_ne=False, is_next_ne=False):
        """Transition probability.

        Note: In contrast with a regular HMM, this cannot be a probability density function, it needs
              to be a proper probability (thus values between 0.0 and 1.0).
        """
        logprob = 0
        if prev_m.edge_m.label == edge_m.label:
            # Staying in same state
            if self.avoid_goingback and edge_m.key == prev_m.edge_m.key and edge_m.ti < prev_m.edge_m.ti:
                # Going back on edge
                logprob += self.gobackonedge_factor_log  # prefer not going back
        else:
            # Moving states
            logprob += self.transition_factor
            if self.avoid_goingback:
                # Goin back on state
                going_back = False
                for m in prev_m.prev:
                    if edge_m.label == m.edge_m.label:
                        going_back = True
                        break
                if going_back:
                    logprob += self.gobacktoedge_factor_log  # prefer not going back
        return logprob, {}  # All probabilities are 1 (thus technically not a distribution)

    def logprob_obs(self, dist, prev_m, new_edge_m, new_edge_o, is_ne=False):
        """Emission probability.

        Note: In contrast with a regular HMM, this cannot be a probability density function, it needs
              to be a proper probability (thus values between 0.0 and 1.0).
        """
        if is_ne:
            result = self.obs_noise_dist_ne.logpdf(dist) + self.obs_noise_logint_ne
        else:
            result = self.obs_noise_dist.logpdf(dist) + self.obs_noise_logint
        # print("logprob_obs: {} -> {:.5f} = {:.5f}".format(dist, result, math.exp(result)))
        return result, {}
