# encoding: utf-8
"""
leuvenmapmatching.map.base
~~~~~~~~~~~~~~~~~~~~~~~~~~

Base Map class.

To be used in a Matcher object, the following functions need to be defined:

- ``edges_closeto``
- ``nodes_closeto``
- ``nodes_nbrto``
- ``edges_nbrto``

For visualiation purposes the following methods need to be implemented:

- ``bb``
- ``labels``
- ``size``
- ``coordinates``
- ``node_coordinates``
- ``all_edges``
- ``all_nodes``

:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""

from abc import abstractmethod
import logging
MYPY = False
if MYPY:
    from typing import Tuple, Union, List
    LabelType = Union[int, str]
    LocType = Tuple[float, float]


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


class BaseMap(object):
    """Abstract class for a Map."""

    def __init__(self, name, use_latlon=True):
        """Simple database wrapper/stub."""
        self.name = name
        self._use_latlon = None
        self.distance = None
        self.distance_point_to_segment = None
        self.distance_segment_to_segment = None
        self.box_around_point = None
        self.use_latlon = use_latlon

    @property
    def use_latlon(self):
        return self._use_latlon

    @use_latlon.setter
    def use_latlon(self, value):
        self._use_latlon = value
        if self._use_latlon:
            from ..util import dist_latlon as dist_lib
        else:
            from ..util import dist_euclidean as dist_lib
        self.distance = dist_lib.distance
        self.distance_point_to_segment = dist_lib.distance_point_to_segment
        self.distance_segment_to_segment = dist_lib.distance_segment_to_segment
        self.box_around_point = dist_lib.box_around_point
        self.lines_parallel = dist_lib.lines_parallel

    @abstractmethod
    def bb(self):
        """Bounding box.

        :return: (lat_min, lon_min, lat_max, lon_max)
        """

    @abstractmethod
    def labels(self):
        """Labels of all nodes."""

    @abstractmethod
    def size(self):
        """Number of nodes."""

    @abstractmethod
    def node_coordinates(self, node_key):
        """Coordinates for given node key."""

    @abstractmethod
    def edges_closeto(self, loc, max_dist=None, max_elmt=None):
        """Find edges close to a certain location.

        :param loc: Latitude, Longitude
        :param max_dist: Maximal distance that returned nodes can be from lat-lon
        :param max_elmt: Maximal number of elements returned after sorting according to distance.
        :return: list[tuple[dist, label, loc]]
        """
        return None

    @abstractmethod
    def nodes_closeto(self, loc, max_dist=None, max_elmt=None):
        """Find nodes close to a certain location.

        :param loc: Latitude, Longitude
        :param max_dist: Maximal distance that returned nodes can be from lat-lon
        :param max_elmt: Maximal number of elements returned after sorting according to distance.
        :return: list[tuple[dist, label, loc]]
        """
        return None

    @abstractmethod
    def nodes_nbrto(self, node):
        # type: (BaseMap, LabelType) -> List[Tuple[LabelType, LocType]]
        """Return all nodes that are linked to ``node``.

        :param node: Node identifier
        :return: list[tuple[label, loc]]
        """
        return []

    def edges_nbrto(self, edge):
        # type: (BaseMap, Tuple[LabelType, LabelType]) -> List[Tuple[LabelType, LocType, LabelType, LocType]]
        """Return all edges that are linked to ``edge``.

        Defaults to ``nodes_nbrto``.

        :param edge: Edge identifier
        :return: list[tuple[label1, label2, loc1, loc2]]
        """
        results = []
        l1, l2 = edge
        p2 = self.node_coordinates(l2)
        for l3, p3 in self.nodes_nbrto(l2):
            results.append((l2, p2, l3, p3))
        return results

    @abstractmethod
    def all_nodes(self, bb=None):
        """All node keys and coordinates.

        :return: [(key, (lat, lon))]
        """

    @abstractmethod
    def all_edges(self, bb=None):
        """All edges.

        :return: [(key_a, loc_a, key_b, loc_b)]
        """
