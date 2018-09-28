# encoding: utf-8
"""
leuvenmapmatching.map.sqlite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Map representation based on a sqlite database. Not optimized for production purposes.

:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import sqlite3
import tempfile
import shutil
import logging
import time
from pathlib import Path
import pickle
from functools import partial
try:
    import pyproj
except ImportError:
    pyproj = None
try:
    import tqdm
except ImportError:
    tqdm = None


from .base import BaseMap


MYPY = False
if MYPY:
    from typing import Optional, Set, Tuple, Dict, Union
    LabelType = Union[int, str]
    LocType = Tuple[float, float]
    EdgeType = Tuple[LabelType, LabelType]


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


class SqliteMap(BaseMap):
    def __init__(self, name, use_latlon=True,
                 crs_lonlat=None, crs_xy=None, dir=None, deserializing=False):
        """Store a map as a SQLite instance.

        This class supports:
        - Indexing using rtrees to allow for fast searching of points on the map.
          When using the rtree index, only integer numbers are allowed as node labels.
        - Serializing to write and read from files.
        - Projecting points to a different frame (e.g. GPS to Lambert)

        :param name: Name of database file
        :param use_latlon: The locations represent latitude-longitude pairs, otherwise y-x coordinates
            are assumed.
        :param crs_lonlat: Coordinate reference system for the latitude-longitude coordinates.
        :param crs_xy: Coordiante reference system for the y-x coordinates.
        :param dir: Directory where to serialize to. If not given, a temporary location will be used.
        :param deserializing: Internal variable to indicate that the object is being build from a file.
        """
        super(SqliteMap, self).__init__(name, use_latlon=use_latlon)
        self.dir = Path(tempfile.gettempdir()) if dir is None else Path(dir)
        name = Path(name)
        suffix = name.suffix
        if suffix == '':
            name = name.with_suffix('.sqlite')
        self.db_fn = self.dir / name
        if deserializing and not self.db_fn.exists():
            raise Exception(f"File not found: {self.db_fn}")
        logger.debug(f"Opening database: {self.db_fn}")
        self.db = sqlite3.connect(str(self.db_fn))
        self.crs_lonlat = crs_lonlat
        self.crs_xy = crs_xy
        self.use_latlon = use_latlon

        if deserializing:
            self.read_properties()
        else:
            self.create_db()

        if self.crs_lonlat is None:
            self.crs_lonlat = {'init': 'epsg:4326'}  # GPS
        if self.crs_xy is None:
            self.crs_xy = {'init': 'epsg:3395'}  # Mercator projection

        self.save_properties()

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

    def read_properties(self):
        c = self.db.cursor()
        for row in c.execute("SELECT key, value FROM properties;"):
            key, value = row[0], pickle.loads(row[1])
            self.__dict__[key] = value

    def save_properties(self):
        c = self.db.cursor()
        q = "INSERT INTO properties (key, value) VALUES (?, ?)"
        v = [('name', pickle.dumps(self.name)),
             ('use_latlon', pickle.dumps(self.use_latlon)),
             ('crs_lonlat', pickle.dumps(self.crs_lonlat)),
             ('crs_xy', pickle.dumps(self.crs_xy))]
        c.executemany(q, v)
        self.db.commit()

    def create_db(self):
        c = self.db.cursor()
        c.execute("DROP INDEX IF EXISTS edges_from_index")
        c.execute("DROP INDEX IF EXISTS close_edges_index")
        c.execute("DROP TABLE IF EXISTS nodes_index")
        c.execute("DROP TABLE IF EXISTS nodes")
        c.execute("DROP TABLE IF EXISTS edges_index")
        c.execute("DROP TABLE IF EXISTS edges")
        c.execute("DROP TABLE IF EXISTS close_edges")
        c.execute("DROP TABLE IF EXISTS properties")
        self.db.commit()

        # Create tables
        q = ("CREATE VIRTUAL TABLE nodes_index USING rtree(\n"
             "id,              -- Integer primary key\n"
             "minX, maxX,      -- Minimum and maximum X coordinate\n"
             "minY, maxY       -- Minimum and maximum Y coordinate\n"
             ")")
        c.execute(q)
        q = ("CREATE TABLE nodes(\n"
             "id INTEGER PRIMARY KEY,\n"
             "x REAL,\n"
             "y REAL\n"
             ")")
        c.execute(q)
        q = ("CREATE VIRTUAL TABLE edges_index USING rtree(\n"
             "id,              -- Integer primary key\n"
             "minX, maxX,      -- Minimum and maximum X coordinate\n"
             "minY, maxY       -- Minimum and maximum Y coordinate\n"
             ")")
        c.execute(q)
        q = ("CREATE TABLE edges(\n"
             "id INTEGER PRIMARY KEY,\n"
             "path INTEGER,\n"  # Not necessarily unique, a pathway id can consist of multiple edges
             "id1 INTEGER,\n"  # node 1
             "id2 INTEGER\n"  # node 2
             ")")
        c.execute(q)
        q = ("CREATE TABLE close_edges(\n"
             "id1 INTEGER,\n"  # edge 1
             "id2 INTEGER\n"  # edge 2
             ")")
        c.execute(q)
        q = ("CREATE TABLE properties(\n"
             "key TEXT,\n"
             "value BLOB\n"
             ")")
        c.execute(q)
        q = "CREATE INDEX edges_from_index ON edges(id1)"
        c.execute(q)
        q = "CREATE INDEX close_edges_index ON close_edges(id1)"
        c.execute(q)
        self.db.commit()

    @classmethod
    def from_file(cls, filename):
        """Read from an existing file."""
        filename = Path(filename)
        nmap = cls(filename.name, dir=filename.parent)
        return nmap

    def bb(self):
        """Bounding box.

        :return: (lat_min, lon_min, lat_max, lon_max) or (y_min, x_min, y_max, x_max)
        """
        c = self.db.cursor()
        c.execute('SELECT min(minX), max(minX), min(maxX), max(maxX) FROM nodes_index;')
        lon_min, lon_max, lat_min, lat_max = c.fetchone()
        return lat_min, lon_min, lat_max, lon_max

    def labels(self):
        """All labels."""
        c = self.db.cursor()
        c.execute('SELECT id FROM nodes;')
        result = [row[0] for row in c.fetchall()]
        return result

    def size(self):
        c = self.db.cursor()
        c.execute('SELECT count(*) FROM nodes')
        result = c.fetchone()[0]
        return result

    def node_coordinates(self, node_key):
        """Get the coordinates of the given node.

        :param node_key: Node label/key
        :return: (lat, lon)
        """
        c = self.db.cursor()
        c.execute('SELECT y, x FROM nodes WHERE id = ?', (node_key, ))
        result = c.fetchone()
        return result

    def add_node(self, node, loc):
        """Add new node to the map.

        :param node: label
        :param loc: (lat, lon) or (y, x)
        """
        c = self.db.cursor()
        lat, lon = loc
        q = "INSERT INTO nodes_index VALUES(?, ?, ?, ?, ?)"
        c.execute(q, (node, lon, lon, lat, lat))
        c.execute("SELECT last_insert_rowid()")
        rowid = c.fetchone()[0]
        q = "INSERT INTO nodes VALUES(?, ?, ?)"
        c.execute(q, (rowid, lon, lat))
        self.db.commit()

    def add_nodes(self, nodes):
        """Add list of nodes to database.

        :param nodes: List[Tuple[node_key, Tuple[lat, lon]]]
        """
        c = self.db.cursor()

        def get_node_index():
            for key, (lat, lon) in nodes:
                yield key, lon, lon, lat, lat

        q = "INSERT INTO nodes_index VALUES(?, ?, ?, ?, ?)"
        c.executemany(q, get_node_index())

        def get_node_vals():
            for key, (lat, lon) in nodes:
                yield key, lon, lat

        q = "INSERT INTO nodes VALUES(?, ?, ?)"
        c.executemany(q, get_node_vals())
        self.db.commit()

    def del_node(self, node):
        raise Exception("TODO")

    def add_edge(self, node_a, node_b, loc_a=None, loc_b=None, path=None, no_index=False):
        """Add new edge to the map.

        :param node_a: Label for the node that is the start of the edge
        :param node_b: Label for the node that is the end of the edge
        """
        c = self.db.cursor()
        eid = (node_a, node_b).__hash__()
        c.execute('INSERT INTO edges(id, path, id1, id2) VALUES (?, ?, ?, ?)', (eid, path, node_a, node_b))
        # c.execute('SELECT last_insert_rowid();')
        # eid = c.fetchone()[0]

        if not no_index:
            if loc_a is None:
                c.execute('SELECT y, x FROM nodes WHERE id = ?;', (node_a, ))
                loc_a = c.fetchone()
            if loc_b is None:
                c.execute('SELECT y, x FROM nodes WHERE id = ?;', (node_b, ))
                loc_b = c.fetchone()
            lat1, lon1 = loc_a
            lat2, lon2 = loc_b
            if lat1 > lat2:
                lat1, lat2 = lat2, lat1
            if lon1 > lon2:
                lon1, lon2 = lon2, lon1
            c.execute('INSERT INTO edges_index(id, minX, maxX, minY, maxY) VALUES (?, ?, ?, ?, ?)',
                      (eid, lon1, lon2, lat1, lat2))

        self.db.commit()

    def add_edges(self, edges, no_index=False):
        """Add list of nodes to database.

        :param edges: List[Tuple[node_key, node_key]]
        """
        c = self.db.cursor()

        def get_edge():
            for row in edges:
                if len(row) == 2:
                    key_a, key_b = row
                    path = None
                else:
                    key_a, key_b, path = row
                eid = (key_a, key_b).__hash__()
                yield eid, path, key_a, key_b

        q = "INSERT INTO edges(id, path, id1, id2) VALUES(?, ?, ?, ?);"
        c.executemany(q, get_edge())
        self.db.commit()

        if not no_index:
            self.reindex_edges()

    def reindex_edges(self):
        logger.debug("Reindexing edges ...")
        t_start = time.time()
        c = self.db.cursor()
        c2 = self.db.cursor()
        c.execute('DELETE FROM edges_index;')
        q = ('SELECT e.id, MIN(n1.x,n2.x) AS minX, MAX(n1.x,n2.x) AS x, '
             'MIN(n1.y,n2.y) AS y, MAX(n1.y,n2.y) AS maxY '
             'FROM edges e '
             'LEFT JOIN nodes n1 ON n1.id = e.id1 '
             'LEFT JOIN nodes n2 ON n2.id = e.id2')
        cnt = 0
        for row in c.execute(q):
            # Contained in query
            c2.execute('INSERT INTO edges_index(id, minX, maxX, minY, maxY) VALUES (?, ?, ?, ?, ?)', row)
            cnt += 1
        self.db.commit()
        t_delta = time.time() - t_start
        logger.debug(f"... done, #rows = {cnt}, time = {t_delta} sec")

    def all_edges(self, bb=None):
        """Return all edges.

        :param bb: Bounding box
        :return: (key_a, loc_a, nbr, loc_b)
        """
        c = self.db.cursor()
        q = 'SELECT e.id1, e.id2, n1.x AS n1x, n2.x AS n2x, n1.y AS n1y, n2.y AS n2y ' + \
            'FROM edges e, edges_index ei ' + \
            'LEFT JOIN nodes n1 ON n1.id = e.id1 ' + \
            'LEFT JOIN nodes n2 ON n2.id = e.id2 ' + \
            'WHERE ei.id == e.id'
        if bb:
            min_y, min_x, max_y, max_x = bb
            # Intersecting with query
            q += ' AND ei.maxX >= ? AND ei.minX <= ? AND ei.maxY >= ? AND ei.minY <= ?'
            c.execute(q, (min_x, max_x, min_y, max_y))
        else:
            c.execute(q)

        for row in c.fetchall():
            key_a, key_b, lon_a, lon_b, lat_a, lat_b = row
            yield key_a, (lat_a, lon_a), key_b, (lat_b, lon_b)

    def all_nodes(self, bb=None):
        """Return all nodes.

        :param bb: Bounding box (minY, minX, maxY, maxX)
        :return:
        """
        c = self.db.cursor()
        q = ('SELECT n.id, n.x, n.y '
             'FROM nodes n, nodes_index ni '
             'WHERE n.id = ni.id ')
        if bb:
            minY, minX, maxY, maxX = bb
            q += 'AND ni.minX >= ? AND ni.maxX <= ? AND ni.minY >= ? AND ni.maxY <= ?'
            c.execute(q, (minX, maxX, minY, maxY))
        else:
            c.execute(q)

        for row in c.fetchall():
            key_a, lon_a, lat_a = row
            yield key_a, (lat_a, lon_a)

    def purge(self):
        pass

    def to_xy(self, name=None):
        """Create a map that uses a projected XY representation on which Euclidean distances
        can be used.
        """
        if not self.use_latlon:
            return self
        if name is None:
            name = self.name + "_xy"

        logger.debug("Start transformation ...")
        t_start = time.time()
        nmap = self.__class__(name, dir=self.dir, use_latlon=self.use_latlon,
                              crs_xy=self.crs_xy, crs_lonlat=self.crs_lonlat)

        raise Exception("to implement")

        t_delta = time.time() - t_start
        logger.debug(f"... done: rtree size = {self.rtree_size()}, time = {t_delta} sec")

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
        nodes = self.all_nodes(bb=bb)
        t_delta_search = time.time() - t_start
        t_start = time.time()
        results = []
        for key_o, loc_o in nodes:
            dist = self.distance(loc, loc_o)
            if dist < max_dist:
                results.append((dist, key_o, loc_o))
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
        lat_b, lon_l, lat_t, lon_r = self.box_around_point((lat, lon), max_dist)
        bb = (lat_b, lon_l,  # y_min, x_min
              lat_t, lon_r)  # y_max, x_max
        nodes = self.all_edges(bb=bb)
        t_delta_search = time.time() - t_start
        t_start = time.time()
        results = []
        for key_a, loc_a, key_b, loc_b, key_p in nodes:
            dist, pi, ti = self.distance_point_to_segment(loc, loc_a, loc_b)
            if dist < max_dist:
                results.append((dist, key_a, loc_a, key_b, loc_b, pi, ti, key_p))
        results.sort()
        t_delta_dist = time.time() - t_start
        logger.debug(f"Found {len(results)} closeby edges "
                     f"in {t_delta_search} sec and computed distances in {t_delta_dist} sec")
        if max_elmt is not None:
            results = results[:max_elmt]
        return results

    def nodes_nbrto(self, node):
        c = self.db.cursor()
        q = ('SELECT e.id2, n2.y, n2.x FROM edges e '
             'INNER JOIN nodes n2 ON n2.id = e.id2 '
             'WHERE e.id1 = ?')
        results = []
        for nbr_label, nbr_lat, nbr_lon in c.execute(q, (node, )):
            results.append((nbr_label, (nbr_lat, nbr_lon)))
        return results

    def edges_nbrto(self, edge):
        l1, l2 = edge
        c = self.db.cursor()
        c.execute('SELECT n.y, n.x FROM nodes n WHERE id = ?', (l2, ))
        p2 = c.fetchone()
        results = []
        # Edges that connect at end of this edge
        for l3, p3 in self.nodes_nbrto(l2):
            results.append((l2, p2, l3, p3))
        # Edges that are in parallel and close
        edge_id = edge.__hash__()
        q = ('SELECT e.id1, e.id2, n1.y, n1.x, n2.y, n2.x FROM close_edges ce '
             'INNER JOIN edges e ON e.id = ce.id2 '
             'INNER JOIN nodes n1 ON n1.id = e.id1 '
             'INNER JOIN nodes n2 ON n2.id = e.id2 '
             'WHERE ce.id1 = ?')
        for l3, l4, p3lat, p3lon, p4lat, p4lon in c.execute(q, (edge_id,)):
            results.append((l3, (p3lat, p3lon), l4, (p4lat, p4lon)))
        return results

    def find_duplicates(self, func=None):
        """Find entries with identical locations."""
        c = self.db.cursor()
        logger.debug('Find duplicates ...')
        t_start = time.time()
        cnt = 0
        q = ('select count(*)as qty, group_concat(id) '
             'from nodes '
             'group by y, x '
             'having qty > 1 ')
        for ncnt, idxs in c.execute(q):
            func(idxs)
        t_delta = time.time() - t_start
        logger.info(f"Found {cnt} doubles, time: {t_delta} seconds")

    def connect_parallelroads(self, dist=0.5, bb=None):
        c = self.db.cursor()
        it = self.all_edges(bb=bb)
        if tqdm:
            it = tqdm.tqdm(list(it))
        cnt = 0
        for key_a, loc_a, key_b, loc_b in it:
            e_id1 = (key_a, key_b).__hash__()
            bb2 = [min(loc_a[0], loc_b[0]), min(loc_a[1], loc_b[1]),
                   max(loc_a[0], loc_b[0]), max(loc_a[1], loc_b[1])]
            for key_c, loc_c, key_d, loc_d in self.all_edges(bb=bb2):
                e_id2 = (key_c, key_d).__hash__()
                if key_a == key_c or key_a == key_d or key_b == key_c or key_b == key_d:
                    continue
                # print(f"Test: ({key_a},{key_b}) - ({key_c},{key_d})")
                if self.lines_parallel(loc_a, loc_b, loc_c, loc_d, d=dist):
                    # print(f"Parallel: ({key_a},{key_b}) - ({key_c},{key_d})")
                    c.execute('INSERT INTO close_edges(id1, id2) VALUES (?, ?)', (e_id1, e_id2))
                    c.execute('INSERT INTO close_edges(id1, id2) VALUES (?, ?)', (e_id2, e_id1))
                    cnt += 1
        logger.debug(f"Linked {cnt} edges")
        self.db.commit()

    def print_stats(self):
        print("Graph\n-----")
        print("Nodes: {}".format(len(self.graph)))

    def __str__(self):
        # s = ""
        # for label, (loc, nbrs, _) in self.graph.items():
        #     s += f"{label:<10} - ({loc[0]:10.4f}, {loc[1]:10.4f})\n"
        # return s
        c = self.db.cursor()
        c.execute("select sqlite_version()")
        row = c.fetchone()
        version = row[0]
        return f"SqliteMap({self.name}, size={self.size()}, version={version})"
