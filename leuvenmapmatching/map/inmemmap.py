# encoding: utf-8
"""
leuvenmapmatching.map.inmemmap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import logging
from . import Map


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")


class InMemMap(Map):
    def __init__(self, graph=None, use_latlon=True):
        """
        Naive in-memory representation of a map (only for testing purposes, too many linear operations).

        :param graph: dict[name_key, list[(lat, lon), list[neighbours_out], list[neighbours_in]
        """
        super(InMemMap, self).__init__(use_latlon=use_latlon)
        self.graph = dict()
        if graph is not None:
            for label, loc, nbrs_out in graph:
                self.graph[label] = (loc, nbrs_out, [])
            for label, _, nbrs_out in graph:
                for nbr in nbrs_out:
                    self.graph[nbr][2].append(label)

    def get_graph(self):
        return self.graph

    def add_node(self, node, loc):
        if node in self.graph:
            if self.graph[node][0] is None:
                self.graph[node] = (loc, self.graph[node][1], self.graph[node][2])
        else:
            self.graph[node] = (loc, [], [])

    def add_edge(self, node_a, loc_a, node_b, loc_b):
        self.add_node(node_a, loc_a)
        self.add_node(node_b, loc_b)
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

    def preload_nodes(self, path, dist):
        pass

    def nodes_closeto(self, loc, max_dist=None, max_elmt=None):
        logger.warning("You are using a naive map representation. Searching nodes can be slow.")
        # TODO: Add http://toblerity.org/rtree/ to speed up searching
        results = []
        for label, (loc2, nbrs, _) in self.graph.items():
            if loc2 is None:
                continue
            lat2, lon2 = loc2
            dist = self.distance(loc, loc2)
            # print(f"{label}: {dist}")
            if max_dist is None or dist <= max_dist:
                results.append((dist, label, (lat2, lon2)))
        results.sort(key=lambda t: t[0])
        if max_elmt is not None:
            results = results[:max_elmt]
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
