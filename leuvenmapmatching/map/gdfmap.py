# encoding: utf-8
"""
leuvenmapmatching.map.gdfmap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Map based on the GeoPandas GeoDataFrame.

:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import logging
from . import Map
import pandas as pd
import geopandas as gp
from shapely.geometry import Point
import pyproj
from functools import partial


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


class GDFMap(Map):
    def __init__(self, use_latlon):
        """
        In-memory representation of a map based on a GeoDataFrame.
        """
        super(GDFMap, self).__init__(use_latlon=use_latlon)
        self.graph = dict()
        self.nodes = None
        if use_latlon:
            self.crs_in = {'init': 'epsg:4326'}   # GPS
            self.crs_out = {'init': 'epsg:3395'}  # Mercator projection
            proj_in = pyproj.Proj(self.crs_in, preserve_units=True)
            proj_out = pyproj.Proj(self.crs_out, preserve_units=True)
            self._latlon2yx = partial(pyproj.transform, proj_in, proj_out)
            self._yx2latlon = partial(pyproj.transform, proj_out, proj_in)
        else:
            self.crs_in = None
            self.crs_out = None
            self._latlon2yx = lambda lat, lon: (lat, lon)
            self._yx2latlon = lambda x, y: (x, y)
        print("DO NOT USE THIS CLASS. IT IS UNDER CONSTRUCTION")

    def get_graph(self):
        return self.graph

    def add_node(self, node, loc):
        """
        :param node: label
        :param loc: (lat, lon) or (y, x)
        """
        if node in self.graph:
            if self.graph[node][0] is None:
                self.graph[node] = (loc, self.graph[node][1], self.graph[node][2])
        else:
            self.graph[node] = (loc, [], [])

    def add_edge(self, node_a, node_b):
        if node_a not in self.graph:
            raise ValueError(f"Add {node_a} first as node")
        if node_b not in self.graph:
            raise ValueError(f"Add {node_b} first as node")
        if node_b not in self.graph[node_a][1]:
            self.graph[node_a][1].append(node_b)
        if node_a not in self.graph[node_b][2]:
            self.graph[node_b][2].append(node_a)

    def all_edges(self):
        for key_a, (loc_a, nbrs, _) in self.graph.items():
            if loc_a is not None:
                for nbr in nbrs:
                    try:
                        loc_b, _, _ = self.graph[nbr]
                        if loc_b is not None:
                            yield (key_a, loc_a, nbr, loc_b)
                    except KeyError:
                        # print("Node not found: {}".format(nbr))
                        pass

    def all_nodes(self):
        for key_a, (loc_a, nbrs, _) in self.graph.items():
            if loc_a is not None:
                yield key_a, loc_a

    def purge(self):
        cnt_noloc = 0
        cnt_noedges = 0
        remove = []
        for node in self.graph.keys():
            if self.graph[node][0] is None:
                cnt_noloc += 1
                remove.append(node)
                # print("No location for node {}".format(node))
            elif len(self.graph[node][1]) == 0 and len(self.graph[node][1]) == 0:
                cnt_noedges += 1
                remove.append(node)
        for node in remove:
            del self.graph[node]
        print("Removed {} nodes without location".format(cnt_noloc))
        print("Removed {} nodes without edges".format(cnt_noedges))

    def prepare_index(self, force=False):
        if self.nodes is not None and not force:
            return

        if self.use_latlon:
            lats, lons, labels = [], [], []
            for label, data in self.graph.items():
                labels.append(label)
                lats.append(data[0][0])
                lons.append(data[0][1])
            df = pd.DataFrame({'label': labels, 'lat': lats, 'lon': lons})
            df['Coordinates'] = list(zip(df.lon, df.lat))
            df['Coordinates'] = df['Coordinates'].apply(Point)
            self.nodes = gp.GeoDataFrame(df, geometry='Coordinates', crs=self.crs_in)

        else:
            ys, xs, labels = [], [], []
            for label, data in self.graph.items():
                labels.append(label)
                ys.append(data[0][0])
                xs.append(data[0][1])
            df = pd.DataFrame({'label': labels, 'y': ys, 'x': xs})
            df['Coordinates'] = list(zip(df.x, df.y))
            df['Coordinates'] = df['Coordinates'].apply(Point)
            self.nodes = gp.GeoDataFrame(df, geometry='Coordinates', crs=self.crs_in)

    def to_xy(self):
        if not self.use_latlon:
            return self
        self.prepare_index()
        map = GDFMap(use_latlon=False)
        map.graph = self.graph
        map.nodes = self.nodes.to_crs(self.crs_out, inplace=True)
        map.nodes['lat'] = map.nodes.apply(lambda row: row['geometry'].x, axis=1)
        map.nodes['lon'] = map.nodes.apply(lambda row: row['geometry'].y, axis=1)
        for label, row in map.graph.items():
            point = map.nodes[label]['geometry']
            row[0] = (point.y, point.x)
        return map

    def latlon2xy(self, lat, lon):
        x, y = self._latlon2yx(lon, lat)  # x, y
        return y, x

    def xy2latlon(self, x, y):
        lon, lat = self._yx2latlon(y, x)
        return lat, lon

    def preload_nodes(self, path, dist):
        pass

    def nodes_closeto(self, loc, max_dist=None, max_elmt=None):
        self.prepare_index()
        lat, lon = loc
        if max_dist is not None:
            nodes = self.nodes.cx[lat - max_dist: lat + max_dist, lon - max_dist: lon + max_dist]
        else:
            nodes = self.nodes
        dists = nodes.distance(Point(lon, lat)).sort_values(inplace=True)
        if max_elmt is not None:
            dists = dists.iloc[:max_elmt]
        results = []
        for label, dist in dists.items():
            point = nodes[label]['geometry']
            results.append((dist, label, (point.y, point.x)))
        return results

    def nodes_nbrto(self, node):
        results = []
        if node not in self.graph:
            return results
        loc_node, nbrs, _ = self.graph[node]
        for nbr_label in nbrs + [node]:
            try:
                loc_nbr = self.graph[nbr_label][0]
                if loc_nbr is not None:
                    results.append((nbr_label, loc_nbr))
            except KeyError:
                pass
        return results

    def print_stats(self):
        print("Graph\n-----")
        print("Nodes: {}".format(len(self.graph)))

    def __str__(self):
        s = ""
        for label, (loc, nbrs, _) in self.graph.items():
            s += f"{label:<10} - ({loc[0]:10.4f}, {loc[1]:10.4f})\n"
        return s
