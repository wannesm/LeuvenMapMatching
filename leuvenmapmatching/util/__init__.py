# encoding: utf-8
"""
leuvenmapmatching.util
~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2015-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""

# No automatic loading to avoid dependency on packages such as nvector and gpxpy if not used.


def approx_equal(a, b, rtol=1e-05, atol=1e-08):
    return abs(a - b) <= (atol + rtol * abs(b))


def approx_leq(a, b, rtol=1e-05, atol=1e-08):
    return (b - a) <= (atol + rtol * abs(b))
