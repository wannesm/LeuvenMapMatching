# encoding: utf-8
"""
leuvenmapmatching.map
~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""

from abc import abstractmethod
import logging


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


class Map(object):
    """Abstract class for a Map."""

    def __init__(self, use_latlon=True):
        """Simple database wrapper/stub."""
        self._use_latlon = None
        self.distance = None
        self.distance_point_to_segment = None
        self.distance_segment_to_segment = None
        self.use_latlon = use_latlon

    @property
    def use_latlon(self):
        return self._use_latlon

    @use_latlon.setter
    def use_latlon(self, value):
        self._use_latlon = value
        if self._use_latlon:
            from ..util import dist_latlon
            self.distance = dist_latlon.distance
            self.distance_point_to_segment = dist_latlon.distance_point_to_segment
            self.distance_segment_to_segment = dist_latlon.distance_segment_to_segment
        else:
            from ..util import dist_euclidean
            self.distance = dist_euclidean.distance
            self.distance_point_to_segment = dist_euclidean.distance_point_to_segment
            self.distance_segment_to_segment = dist_euclidean.distance_segment_to_segment

    @abstractmethod
    def get_graph(self):
        """Return the full (or cached part of the) graph.

        :return: dict[label, ((lat, lon), list[label_nbr_out], list[label_nbr_in)]
        """

    @abstractmethod
    def preload_nodes(self, path, dist):
        """Preload all nodes that are within a certain range from a given path.

        This avoids the repeated querying of the database.
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
        """Return all nodes that are linked to `node` and are closer than `max_dist` to `loc_obs`

        :param node: Node identifier
        :return: list[tuple[label, loc]]
        """
        return None
