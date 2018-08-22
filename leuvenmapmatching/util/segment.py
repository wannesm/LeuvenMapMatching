# encoding: utf-8
"""
leuvenmapmatching.util.segment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2015-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import logging


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


class Segment(object):
    """Segment and interpolated point"""
    __slots__ = ["l1", "p1", "l2", "p2", "_pi", "_ti"]

    def __init__(self, l1, p1, l2=None, p2=None, pi=None, ti=None):
        self.l1 = l1  # Start of segment, label
        self.p1 = p1  # point
        self.l2 = l2  # End of segment, if None the segment is a point
        self.p2 = p2
        self.pi = pi  # Interpolated point
        self.ti = ti  # Position on segment p1-p2

    @property
    def label(self):
        if self.l2 is None:
            return self.l1
        return f"{self.l1}-{self.l2}"

    @property
    def key(self):
        if self.l2 is None:
            return self.l1
        return f"{self.l1}-{self.l2}"

    @property
    def pi(self):
        if self.p2 is None:
            return self.p1
        return self._pi

    @pi.setter
    def pi(self, value):
        if value is not None and len(value) > 2:
            self._pi = tuple(value[:2])
        else:
            self._pi = value

    @property
    def ti(self):
        if self.p2 is None:
            return 0
        return self._ti

    @ti.setter
    def ti(self, value):
        self._ti = value

    def is_point(self):
        return self.p2 is None

    def last_point(self):
        if self.p2 is None:
            return self.p1
        return self.p2

    def loc_to_str(self):
        if self.p2 is None:
            return f"{self.p1}"
        if self._pi is not None:
            return f"{self.p1}-{self.pi}/{self.ti}-{self.p2}"
        return f"{self.p1}-{self.p2}"

    def __str__(self):
        if self.p2 is None:
            return f"{self.l1}"
        if self._pi is not None:
            return f"{self.l1}-i-{self.l2}"
        return f"{self.l1}-{self.l2}"

    def __repr__(self):
        return self.__str__()
