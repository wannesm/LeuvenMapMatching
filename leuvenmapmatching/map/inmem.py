# encoding: utf-8
"""
leuvenmapmatching.map.inmem
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple in-memory map representation. Not suited for production purposes.
Write your own map class that connects to your map (e.g. a database instance).

:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import logging
import time
from pathlib import Path
import pickle
from .base import BaseMap
try:
    import rtree
except ImportError:
    rtree = None
try:
    import pyproj
except ImportError:
    pyproj = None
from functools import partial


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


class InMemMap(BaseMap):
    def __init__(self, name, use_latlon=True, use_rtree=False, index_edges=False,
                 crs_lonlat=None, crs_xy=None, graph=None, dir=None, deserializing=False):
        """
        In-memory representation of a map.
        """
        super(InMemMap, self).__init__(name, use_latlon=use_latlon)
        self.dir = Path(".") if dir is None else Path(dir)
        self.index_edges = index_edges
        self.graph = dict() if graph is None else graph
        self.rtree = None
        self.use_rtree = use_rtree
        if self.use_rtree:
            self.setup_index(deserializing=deserializing)

        self.crs_lonlat = {'init': 'epsg:4326'} if crs_lonlat is None else crs_lonlat  # GPS
        self.crs_xy = {'init': 'epsg:3395'} if crs_xy is None else crs_xy  # Mercator projection
        if pyproj:
            proj_lonlat = pyproj.Proj(self.crs_lonlat, preserve_units=True)
            proj_xy = pyproj.Proj(self.crs_xy, preserve_units=True)
            self.lonlat2xy = partial(pyproj.transform, proj_lonlat, proj_xy)
            self.xy2lonlat = partial(pyproj.transform, proj_xy, proj_lonlat)
        else:
            def pyproj_notfound(*_args, **_kwargs):
                raise Exception("pyproj package not found")
            self.lonlat2xy = pyproj_notfound
            self.xy2lonlat = pyproj_notfound

    def serialize(self):
        data = {
            "name": self.name,
            "graph": self.graph,
            "use_latlon": self.use_latlon,
            "use_rtree": self.use_rtree,
            "index_edges": self.index_edges,
            "crs_lonlat": self.crs_lonlat,
            "crs_xy": self.crs_xy
        }
        if self.dir is not None:
            data["dir"] = self.dir
        return data

    @classmethod
    def deserialize(cls, data):
        nmap = cls(data["name"], dir=data.get("dir", None),
                   use_latlon=data["use_latlon"], use_rtree=data["use_rtree"],
                   index_edges=data["index_edges"],
                   crs_lonlat=data["crs_lonlat"], crs_xy=data["crs_xy"],
                   graph=data["graph"], deserializing=True)
        return nmap

    def dump(self):
        """Serialize map using pickle.

        All files will be saved to the `dir` directory using the `name` as filename.
        """
        if self.dir is None:
            logger.error(f"No directory set where to save (see InMemMap.__init__)")
            return
        filename = self.dir / (self.name + ".pkl")
        with filename.open("wb") as ofile:
            pickle.dump(self.serialize(), ofile)
        logger.debug(f"Saved map to {filename}")
        if self.rtree:
            rtree_fn = self.rtree_fn()
            if rtree_fn is not None:
                self.rtree.close()
                self.rtree = rtree.index.Index(str(rtree_fn))

    @classmethod
    def from_pickle(cls, filename):
        """Deserialize map using pickle to the given filename."""
        filename = Path(filename)
        with filename.open("rb") as ifile:
            data = pickle.load(ifile)
        nmap = cls.deserialize(data)
        return nmap

    def get_graph(self):
        return self.graph

    def bb(self):
        """Bounding box.

        :return: (lat_min, lon_min, lat_max, lon_max)
        """
        if self.use_rtree:
            lat_min, lon_min, lat_max, lon_max = self.rtree.bounds
        else:
            glat, glon = zip(*[t[0] for t in self.graph.values()])
            lat_min, lat_max = min(glat), max(glat)
            lon_min, lon_max = min(glon), max(glon)
        return lat_min, lon_min, lat_max, lon_max

    def labels(self):
        return self.graph.keys()

    def size(self):
        return len(self.graph)

    def node_coordinates(self, node_key):
        """Get the coordinates of the given node.

        :param node_key: Node label/key
        :return: (lat, lon)
        """
        return self.graph[node_key][0]

    def add_node(self, node, loc):
        """
        :param node: label
        :param loc: (lat, lon) or (y, x)
        """
        if node in self.graph:
            if self.graph[node][0] is None:
                self.graph[node] = (loc, self.graph[node][1])
        else:
            self.graph[node] = (loc, [])
        if self.use_rtree and self.rtree:
            self.rtree.insert(node, (loc[0], loc[1], loc[0], loc[1]))

    def del_node(self, node):
        if node not in self.graph:
            return
        if self.rtree:
            data = self.graph[node]
            loc = data[0]
            self.rtree.delete(node, (loc[0], loc[1], loc[0], loc[1]))
        del self.graph[node]

    def add_edge(self, node_a, node_b):
        if node_a not in self.graph:
            raise ValueError(f"Add {node_a} first as node")
        if node_b not in self.graph:
            raise ValueError(f"Add {node_b} first as node")
        if node_b not in self.graph[node_a][1]:
            self.graph[node_a][1].append(node_b)

    def all_edges(self, bb=None):
        if bb is None or self.rtree is None:
            keyvals = self.graph.items()
        else:
            node_idxs = self.rtree.intersection(bb)
            keyvals = ((idx, self.graph[idx]) for idx in node_idxs)
        for key_a, (loc_a, nbrs) in keyvals:
            if loc_a is not None:
                for nbr in nbrs:
                    try:
                        loc_b, _ = self.graph[nbr]
                        if loc_b is not None:
                            yield (key_a, loc_a, nbr, loc_b)
                    except KeyError:
                        # print("Node not found: {}".format(nbr))
                        pass

    def all_nodes(self, bb=None):
        if bb is None or self.rtree is None:
            keyvals = self.graph.items()
        else:
            node_idxs = self.rtree.intersection(bb)
            keyvals = ((idx, self.graph[idx]) for idx in node_idxs)
        for key_a, (loc_a, nbrs) in keyvals:
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
            elif len(self.graph[node][1]) == 0:
                cnt_noedges += 1
                remove.append(node)
        for node in remove:
            self.del_node(node)
        logger.debug("Removed {} nodes without location".format(cnt_noloc))
        logger.debug("Removed {} nodes without edges".format(cnt_noedges))

    def rtree_size(self):
        bb = self.rtree.bounds
        if bb[0] < bb[2] and bb[1] < bb[3]:
            rtree_size = self.rtree.count(bb)
        else:
            rtree_size = 0
        return rtree_size

    def rtree_fn(self):
        rtree_fn = None
        if self.dir is not None:
            rtree_fn = self.dir / self.name
        return rtree_fn

    def setup_index(self, force=False, deserializing=False):
        if not self.use_rtree:
            return
        if self.rtree is not None and not force:
            return
        if rtree is None:
            raise Exception("rtree package not found")

        rtree_fn = self.rtree_fn()
        args = []

        if self.graph and (not deserializing or rtree_fn is None or not rtree_fn.exists()):
            if self.index_edges:
                logger.debug("Index edges")
                def generator_function():
                    for label, data in self.graph.items():
                        lat_min, lon_min = data[0]
                        lat_max, lon_max = lat_min, lon_min
                        for idx in data[1]:
                            olat, olon = self.graph[idx][0]
                            lat_min = min(lat_min, olat)
                            lat_max = max(lat_max, olat)
                            lon_min = min(lon_min, olon)
                            lon_max = max(lon_max, olon)
                        yield (label, (lat_min, lon_min, lat_max, lon_max), None)
            else:
                def generator_function():
                    for label, data in self.graph.items():
                        lat, lon = data[0]
                        yield (label, (lat, lon, lat, lon), None)
            args.append(generator_function())

        t_start = time.time()
        if self.dir is not None:
            props = rtree.index.Property()
            # if force:
            #     props.overwrite = True
            logger.debug(f"Creating new file-based rtree index ({rtree_fn}) ...")
            args.insert(0, str(rtree_fn))
        elif deserializing:
            raise Exception("Cannot deserialize, no directory given")
        else:
            logger.debug("Creating new in-memory rtree index ...")
        self.rtree = rtree.index.Index(*args)
        t_delta = time.time() - t_start
        logger.debug(f"... done: rtree size = {self.rtree_size()}, time = {t_delta} sec")

    def fill_index(self):
        if not self.use_rtree or self.rtree is None:
            return

        for label, data in self.graph.items():
            loc = data[0]
            self.rtree.insert(label, (loc[1], loc[0], loc[1], loc[0]))
        logger.debug(f"After filling rtree, size = {self.rtree_size()}")

    def to_xy(self, name=None, use_rtree=None):
        """Create a map that uses a projected XY representation on which Euclidean distances
        can be used.
        """
        if not self.use_latlon:
            return self
        if name is None:
            name = self.name + "_xy"
        if use_rtree is None:
            use_rtree = self.use_rtree

        ngraph = dict()
        for label, row in self.graph.items():
            lat, lon = row[0]
            x, y = self.lonlat2xy(lon, lat)
            ngraph[label] = ((y, x), row[1])
        nmap = self.__class__(name, dir=self.dir, graph=ngraph,
                              use_latlon=False, use_rtree=use_rtree, index_edges=self.index_edges,
                              crs_xy=self.crs_xy, crs_lonlat=self.crs_lonlat)

        return nmap

    def latlon2xy(self, lat, lon):
        x, y = self.lonlat2xy(lon, lat)
        return x, y

    def latlon2yx(self, lat, lon):
        x, y = self.lonlat2xy(lon, lat)
        return y, x

    def xy2latlon(self, x, y):
        lon, lat = self.xy2lonlat(x, y)
        return lat, lon

    def yx2latlon(self, y, x):
        lon, lat = self.xy2lonlat(x, y)
        return lat, lon

    def preload_nodes(self, path, dist):
        pass

    def nodes_closeto(self, loc, max_dist=None, max_elmt=None):
        t_start = time.time()
        lat, lon = loc[:2]
        if self.rtree is not None and max_dist is not None:
            bb = (lat - max_dist, lon - max_dist,  # y_min, x_min
                  lat + max_dist, lon + max_dist)  # y_max, x_max
            logger.debug(f"Search closeby nodes to {loc}, bb={bb}")
            nodes = self.rtree.intersection(bb)
        else:
            logger.warning("Searching closeby nodes with linear search, use an index and set max_dist")
            nodes = self.graph.keys()
        t_delta_search = time.time() - t_start
        t_start = time.time()
        results = []
        for label in nodes:
            oloc, nbrs = self.graph[label]
            dist = self.distance(loc, oloc)
            if dist < max_dist:
                results.append((dist, label, oloc))
        results.sort()
        t_delta_dist = time.time() - t_start
        logger.debug(f"Found {len(results)} closeby nodes "
                     f"in {t_delta_search} sec and computed distances in {t_delta_dist} sec")
        if max_elmt is not None:
            results = results[:max_elmt]
        return results

    def edges_closeto(self, loc, max_dist=None, max_elmt=None):
        t_start = time.time()
        lat, lon = loc[:2]
        if self.rtree is not None and max_dist is not None and self.index_edges:
            bb = (lat - max_dist, lon - max_dist,  # y_min, x_min
                  lat + max_dist, lon + max_dist)  # y_max, x_max
            logger.debug(f"Search closeby edges to {loc}, bb={bb}")
            nodes = self.rtree.intersection(bb)
        else:
            if self.rtree is not None and max_dist is not None and not self.index_edges:
                logger.warning("Index is built for nodes, not for edges, set the index_edges argument to true")
            logger.warning("Searching closeby nodes with linear search, use an index and set max_dist")
            nodes = self.graph.keys()
        t_delta_search = time.time() - t_start
        t_start = time.time()
        results = []
        for label in nodes:
            oloc, nbrs = self.graph[label]
            for nbr in nbrs:
                nbr_data = self.graph[nbr]
                dist, pi, ti = self.distance_point_to_segment(loc, oloc, nbr_data[0])
                if dist < max_dist:
                    results.append((dist, label, oloc, nbr, nbr_data[0], pi, ti))
        results.sort()
        t_delta_dist = time.time() - t_start
        logger.debug(f"Found {len(results)} closeby edges "
                     f"in {t_delta_search} sec and computed distances in {t_delta_dist} sec")
        if max_elmt is not None:
            results = results[:max_elmt]
        return results

    def nodes_nbrto(self, node):
        results = []
        if node not in self.graph:
            return results
        loc_node, nbrs = self.graph[node]
        for nbr_label in nbrs + [node]:
            try:
                loc_nbr = self.graph[nbr_label][0]
                if loc_nbr is not None:
                    results.append((nbr_label, loc_nbr))
            except KeyError:
                pass
        return results

    def find_duplicates(self, func=None):
        """Find entries with identical locations."""
        cnt = 0
        for label, data in self.graph.items():
            lat, lon = data[0]
            idxs = list(self.rtree.nearest((lat, lon, lat, lon), num_results=1))
            idxs.remove(label)
            if len(idxs) > 0:
                logger.info(f"Found doubles for {label}: {idxs}")
                if func:
                    func(label, idxs)
        logger.info(f"Found {cnt} doubles")

    def print_stats(self):
        print("Graph\n-----")
        print("Nodes: {}".format(len(self.graph)))

    def __str__(self):
        # s = ""
        # for label, (loc, nbrs, _) in self.graph.items():
        #     s += f"{label:<10} - ({loc[0]:10.4f}, {loc[1]:10.4f})\n"
        # return s
        return f"InMemMap({self.name}, size={self.size()})"
