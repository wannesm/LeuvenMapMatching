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
from functools import partial
try:
    import rtree
except ImportError:
    rtree = None
try:
    import pyproj
except ImportError:
    pyproj = None
try:
    import tqdm
except ImportError:
    tqdm = None
MYPY = False
if MYPY:
    from typing import Optional, Set, Tuple, Dict, Union
    LabelType = Union[int, str]
    LocType = Tuple[float, float]
    EdgeType = Tuple[LabelType, LabelType]


from .base import BaseMap


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


class InMemMap(BaseMap):
    def __init__(self, name, use_latlon=True, use_rtree=False, index_edges=False,
                 crs_lonlat=None, crs_xy=None, graph=None, dir=None, deserializing=False):
        """In-memory representation of a map.

        This is a simple database-like object to perform experiments with map matching.
        For production purposes it is recommended to use your own derived
        class (e.g. to connect to your database instance).

        This class supports:
        - Indexing using rtrees to allow for fast searching of points on the map.
          When using the rtree index, only integer numbers are allowed as node labels.
        - Serializing to write and read from files.
        - Projecting points to a different frame (e.g. GPS to Lambert)

        :param name: Map name (mandatory)
        :param use_latlon: The locations represent latitude-longitude pairs, otherwise y-x coordinates
            are assumed.
        :param use_rtree: Build an rtree index to quickly search for locations.
        :param index_edges: Build an index for the edges in the map instead of the vertices.
        :param crs_lonlat: Coordinate reference system for the latitude-longitude coordinates.
        :param crs_xy: Coordiante reference system for the y-x coordinates.
        :param graph: Initial graph of form Dict[label, Tuple[Tuple[y,x], List[neighbor]]]]
        :param dir: Directory where to serialize to. If given, the rtree index structure will be written
            to a file immediately.
        :param deserializing: Internal variable to indicate that the object is being build from a file.
        """
        super(InMemMap, self).__init__(name, use_latlon=use_latlon)
        self.dir = None if dir is None else Path(dir)
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

        self.linked_edges = None  # type: Optional[Dict[EdgeType, Set[Tuple[EdgeType]]]]

    def serialize(self):
        """Create a serializable data structure."""
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
        """Create a new instance from a dictionary."""
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

    def bb(self):
        """Bounding box.

        :return: (lat_min, lon_min, lat_max, lon_max) or (y_min, x_min, y_max, x_max)
        """
        if self.use_rtree:
            lat_min, lon_min, lat_max, lon_max = self.rtree.bounds
        else:
            glat, glon = zip(*[t[0] for t in self.graph.values()])
            lat_min, lat_max = min(glat), max(glat)
            lon_min, lon_max = min(glon), max(glon)
        return lat_min, lon_min, lat_max, lon_max

    def labels(self):
        """All labels."""
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
        """Add new node to the map.

        :param node: label
        :param loc: (lat, lon) or (y, x)
        """
        if node in self.graph:
            if self.graph[node][0] is None:
                self.graph[node] = (loc, self.graph[node][1])
        else:
            self.graph[node] = (loc, [])
        if self.use_rtree and self.rtree:
            if type(node) is not int:
                raise Exception(f"Rtree index only supports integer keys for vertices")
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
        """Add new edge to the map.

        :param node_a: Label for the node that is the start of the edge
        :param node_b: Label for the node that is the end of the edge
        """
        if node_a not in self.graph:
            raise ValueError(f"Add {node_a} first as node")
        if node_b not in self.graph:
            raise ValueError(f"Add {node_b} first as node")
        if node_b not in self.graph[node_a][1]:
            self.graph[node_a][1].append(node_b)

    def _items_in_bb(self, bb):
        if self.rtree is not None:
            node_idxs = self.rtree.intersection(bb)
            for key in node_idxs:
                yield (key, self.graph[key])
        else:
            lat_min, lon_min, lat_max, lon_max = bb
            for key, value in self.graph.items():
                ((lat, lon), nbrs) = value
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                    yield (key, value)

    def all_edges(self, bb=None):
        """Return all edges.

        :param bb: Bounding box
        :return: (key_a, loc_a, nbr, loc_b)
        """
        if bb is None:
            keyvals = self.graph.items()
        else:
            keyvals = self._items_in_bb(bb)
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
        """Return all nodes.

        :param bb: Bounding box
        :return:
        """
        if bb is None:
            keyvals = self.graph.items()
        else:
            keyvals = self._items_in_bb(bb)
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

        if self.graph and len(self.graph) > 0 and (not deserializing or rtree_fn is None or not rtree_fn.exists()):
            if self.index_edges:
                logger.debug("Generator to index edges")

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
                        if type(label) is not int:
                            raise Exception(f"Rtree index only supports integer keys for vertices")
                        yield (label, (lat_min, lon_min, lat_max, lon_max), None)
            else:
                logger.debug("Generator to index nodes")

                def generator_function():
                    for label, data in self.graph.items():
                        lat, lon = data[0]
                        yield (label, (lat, lon, lat, lon), None)
            args.append(generator_function())

        t_start = time.time()
        if self.dir is not None:
            # props = rtree.index.Property()
            # if force:
            #     props.overwrite = True
            logger.debug(f"Creating new file-based rtree index ({rtree_fn}) ...")
            args.insert(0, str(rtree_fn))
        elif deserializing:
            raise Exception("Cannot deserialize, no directory given")
        else:
            logger.debug(f"Creating new in-memory rtree index (args={args}) ...")
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

    def nodes_closeto(self, loc, max_dist=None, max_elmt=None):
        """Return all nodes close to the given location.

        :param loc: Location
        :param max_dist: Maximal distance from the location
        :param max_elmt: Return only the most nearby nodes
        """
        t_start = time.time()
        lat, lon = loc[:2]
        lat_b, lon_l, lat_t, lon_r = self.box_around_point((lat, lon), max_dist)
        bb = (lat_b, lon_l,  # y_min, x_min
              lat_t, lon_r)  # y_max, x_max
        if self.rtree is not None and max_dist is not None:
            logger.debug(f"Search closeby nodes to {loc}, bb={bb}")
            nodes = self.rtree.intersection(bb)
        else:
            logger.warning("Searching closeby nodes with linear search, use an index and set max_dist")
            if max_dist is not None:
                nodes = (key for key, _ in self._items_in_bb(self.box_around_point((lat, lon), max_dist)))
            else:
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
        """Return all nodes that are on an edge that is close to the given location.

        :param loc: Location
        :param max_dist: Maximal distance from the location
        :param max_elmt: Return only the most nearby nodes
        """
        t_start = time.time()
        lat, lon = loc[:2]
        if self.rtree is not None and max_dist is not None and self.index_edges:
            lat_b, lon_l, lat_t, lon_r = self.box_around_point((lat, lon), max_dist)
            bb = (lat_b, lon_l,  # y_min, x_min
                  lat_t, lon_r)  # y_max, x_max
            logger.debug(f"Search closeby edges to {loc}, bb={bb}")
            nodes = self.rtree.intersection(bb)
        else:
            if self.rtree is not None and max_dist is not None and not self.index_edges:
                logger.warning("Index is built for nodes, not for edges, set the index_edges argument to true")
            logger.warning("Searching closeby nodes with linear search, use an index and set max_dist")
            if max_dist is not None:
                bb = self.box_around_point((lat, lon), max_dist)
                nodes = (key for key, _ in self._items_in_bb(bb))
            else:
                nodes = self.graph.keys()
        t_delta_search = time.time() - t_start
        t_start = time.time()
        results = []
        for label in nodes:
            oloc, nbrs = self.graph[label]
            for nbr in nbrs:
                nbr_data = self.graph[nbr]
                dist, pi, ti = self.distance_point_to_segment(loc, oloc, nbr_data[0])
                # print(f"label={label}/{oloc}, nbr={nbr}/{nbr_data[0]}   -- loc={loc}  -> {dist}, {pi}, {ti}")
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

    def edges_nbrto(self, edge):
        results = []
        l1, l2 = edge
        p1 = self.node_coordinates(l1)
        p2 = self.node_coordinates(l2)
        # Edges that connect at end of this edge
        for l3, p3 in self.nodes_nbrto(l2):
            results.append((l2, p2, l3, p3))
        # Edges that are in parallel and close
        if self.linked_edges:
            for (l3, l4) in self.linked_edges.get(edge, []):
                p3 = self.node_coordinates(l3)
                p4 = self.node_coordinates(l4)
                results.append((l3, p3, l4, p4))
        return results

    def find_duplicates(self, func=None):
        """Find entries with identical locations."""
        cnt = 0
        for label, data in self.graph.items():
            lat, lon = data[0]
            idxs = list(self.rtree.nearest((lat, lon, lat, lon), num_results=1))
            idxs.remove(label)
            if len(idxs) > 0:
                # logger.info(f"Found doubles for {label}: {idxs}")
                if func:
                    func(label, idxs)
        logger.info(f"Found {cnt} doubles")

    def connect_parallelroads(self, dist=0.5, bb=None):
        if self.rtree is None or not self.index_edges:
            logger.error("Finding parallel roads requires and edge-based index")
            return
        self.linked_edges = {}
        it = self.all_edges(bb=bb)
        if tqdm:
            it = tqdm.tqdm(list(it))
        for key_a, loc_a, key_b, loc_b in it:
            bb2 = [min(loc_a[0], loc_b[0]), min(loc_a[1], loc_b[1]),
                   max(loc_a[0], loc_b[0]), max(loc_a[1], loc_b[1])]
            for key_c, loc_c, key_d, loc_d in self.all_edges(bb=bb2):
                if key_a == key_c or key_a == key_d or key_b == key_c or key_b == key_d:
                    continue
                # print(f"Test: ({key_a},{key_b}) - ({key_c},{key_d})")
                if self.lines_parallel(loc_a, loc_b, loc_c, loc_d, d=dist):
                    # print(f"Parallel: ({key_a},{key_b}) - ({key_c},{key_d})")
                    key = (key_a, key_b)
                    if key in self.linked_edges:
                        self.linked_edges[key].add((key_c, key_d))
                    else:
                        self.linked_edges[key] = {(key_c, key_d)}
        logger.debug(f"Linked {len(self.linked_edges)} edges")


    def print_stats(self):
        print("Graph\n-----")
        print("Nodes: {}".format(len(self.graph)))

    def __str__(self):
        # s = ""
        # for label, (loc, nbrs, _) in self.graph.items():
        #     s += f"{label:<10} - ({loc[0]:10.4f}, {loc[1]:10.4f})\n"
        # return s
        return f"InMemMap({self.name}, size={self.size()})"
