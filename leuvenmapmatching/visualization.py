# encoding: utf-8
"""
leuvenmapmatching.visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2015-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import math
import logging
from itertools import islice

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import smopy


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")
match_color = mcolors.CSS4_COLORS['green']
match_ne_color = mcolors.CSS4_COLORS['olive']
lattice_color = mcolors.CSS4_COLORS['magenta']
# nodes_color = mcolors.CSS4_COLORS['orange']
nodes_color = mcolors.CSS4_COLORS['black']
# path_color = mcolors.CSS4_COLORS['blue']
path_color = mcolors.CSS4_COLORS['blue']


def plot_map(map_con, path=None, nodes=None, counts=None, ax=None, use_osm=False, z=None, bb=None,
             show_labels=False, matcher=None, show_graph=True, zoom_path=False, show_lattice=False,
             show_matching=False, filename=None, linewidth=2, coord_trans=None):
    """Plot the db/graph and optionally include the observed path and inferred nodes.

    :param map_con: Map
    :param path: list[(lat, lon)]
    :param nodes: list[str]
    :param counts: Number of visits of a node in the lattice. dict[label, int]
    :param ax: Matplotlib axis
    :param use_osm: Use OpenStreetMap layer, the points should be latitude-longitude pairs.
    :param matcher: Matcher object (overrules given path, nodes and counts)
    :param filename: File to write image to
    :param show_graph: Plot the vertices and edges in the graph
    :return: None
    """
    if matcher is not None:
        path = matcher.path
        counts = matcher.node_counts()
        nodes = matcher.path_pred_onlynodes
        lat_nodes = matcher.lattice_best
        if coord_trans:
            path = [coord_trans(lat, lon) for lat, lon in path]
    else:
        lat_nodes = None
    lat_min, lon_min, lat_max, lon_max = map_con.bb()
    if coord_trans:
        lat_min, lon_min = coord_trans(lat_min, lon_min)
        lat_max, lon_max = coord_trans(lat_max, lon_max)
    if path:
        if type(zoom_path) is slice:
            plat, plon = islice(zip(*path[zoom_path]), 2)
            lat_min, lat_max = min(plat), max(plat)
            lon_min, lon_max = min(plon), max(plon)
        elif zoom_path:
            plat, plon = islice(zip(*path), 2)
            lat_min, lat_max = min(plat), max(plat)
            lon_min, lon_max = min(plon), max(plon)
        else:
            plat, plon = islice(zip(*path), 2)
            lat_min, lat_max = min(lat_min, min(plat)), max(lat_max, max(plat))
            lon_min, lon_max = min(lon_min, min(plon)), max(lon_max, max(plon))
        logger.debug("Setting bounding box to path")
    else:
        plat, plon = None, None

    width = 10

    if use_osm:
        from .util import dist_latlon
        project = dist_latlon.project
        if bb is None:
            bb = [lat_min, lon_min, lat_max, lon_max]
            logger.debug("bb = [{}, {}, {}, {}]".format(*bb))
        if z is None:
            z = 18
        m = smopy.Map(bb, z=z, ax=ax)
        to_pixels = m.to_pixels
        x_max, y_max = to_pixels(lat_max, lon_max)
        x_min, y_min = to_pixels(lat_min, lon_min)
        height = width / abs(x_max - x_min) * abs(y_max - y_min)
        if ax is None:
            ax = m.show_mpl(figsize=(width, height))
        else:
            ax = m.show_mpl(ax=ax)
        fig = None

    else:
        from .util import dist_euclidean
        project = dist_euclidean.project

        lat_max += (lat_max - lat_min) * 0.1
        lon_min -= (lon_max - lon_min) * 0.1
        lat_min -= (lat_max - lat_min) * 0.1
        lon_max += (lon_max - lon_min) * 0.1

        def to_pixels(lat, lon=None):
            if lon is None:
                lat, lon = lat[0], lat[1]
            return lon, lat
        # to_pixels = lambda lat_, lon_: (lon_, lat_)
        if bb is None:
            x_max, y_max = to_pixels(lat_max, lon_max)
            x_min, y_min = to_pixels(lat_min, lon_min)
        else:
            x_max, y_max = to_pixels(bb[2], bb[3])
            x_min, y_min = to_pixels(bb[0], bb[1])
        height = width / abs(lon_max - lon_min) * abs(lat_max - lat_min)
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(width, height))
        else:
            fig = None
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

    if counts is None:
        node_sizes = [10] * map_con.size()
    else:
        node_sizes = [counts[label]*100+5 for label in map_con.labels()]

    if show_graph:
        logger.debug('Plot vertices ...')
        gx, gy = zip(*[to_pixels(t) for t in map_con.coordinates()])
        ax.scatter(gx, gy, s=node_sizes, alpha=0.4)
        if show_labels:
            for key, coord in map_con.all_nodes():
                key = str(key)
                if type(show_labels) is int:
                    key = key[-show_labels:]
                ax.annotate(key, xy=to_pixels(coord))

        logger.debug('Plot lines ...')
        for _, loc_a, _, loc_b in map_con.all_edges():
            x_a, y_a = to_pixels(*loc_a)
            x_b, y_b = to_pixels(*loc_b)
            ax.plot([x_a, x_b], [y_a, y_b], 'k', linewidth=0.3)

    # for label, ((lat, lon), nbrs, _) in graph.items():
    #     ax.annotate(label, xy=(lon, lat))
    #     for nbr in nbrs:
    #         lat2, lon2 = graph[nbr][0]
    #         plt.plot((lon, lon2), (lat, lat2), 'k', linewidth=0.3)

    if show_lattice:
        if matcher is None:
            logger.warning("Matcher needs to be passed to show lattice. Not showing lattice.")
        plot_lattice(ax, to_pixels, matcher)

    if path:
        logger.debug('Plot path ...')
        px, py = zip(*[to_pixels(p[:2]) for p in path])
        ax.plot(px, py, 'o-', linewidth=linewidth, markersize=linewidth * 2, alpha=0.75,
                linestyle="--", color=path_color)
        if show_labels:
            for li, (lx, ly) in enumerate(zip(px, py)):
                # ax.text(lx, ly, f"O{li}", color=path_color)
                ax.annotate(f"O{li}", xy=(lx, ly), color=path_color)

    if nodes:
        logger.debug('Plot nodes ...')
        xs, ys = [], []
        for node in nodes:
            if type(node) == tuple and len(node) == 3:
                lat, lon = map_con.node_coordinates(node[0])
                if coord_trans:
                    lat, lon = coord_trans(lat, lon)
                x, y = to_pixels(lat, lon)
                xs.append(x)
                ys.append(y)
            elif type(node) == str or type(node) == int:
                lat, lon = map_con.node_coordinates(node)
                if coord_trans:
                    lat, lon = coord_trans(lat, lon)
                x, y = to_pixels(lat, lon)
                xs.append(x)
                ys.append(y)
        ax.plot(xs, ys, 'o-', linewidth=linewidth, markersize=linewidth * 2, alpha=0.75,
                linestyle="--", color=nodes_color)

    if matcher and show_matching:
        logger.debug('Plot matching path-nodes (using matcher) ...')
        for idx, m in enumerate(lat_nodes):
            lat, lon = m.edge_m.pi[:2]
            lat2, lon2 = m.edge_o.pi[:2]
            if coord_trans:
                lat, lon = coord_trans(lat, lon)
                lat2, lon2 = coord_trans(lat2, lon2)
            x, y = to_pixels(lat, lon)
            x2, y2 = to_pixels(lat2, lon2)
            if m.edge_o.is_point():
                ax.plot((x, x2), (y, y2), '-', color=match_color, linewidth=linewidth, alpha=0.75)
            else:
                ax.plot((x, x2), (y, y2), '-', color=match_ne_color, linewidth=linewidth, alpha=0.75)
            # ax.plot((x, x2), (y, y2), '-', color=match_color, linewidth=10, alpha=0.1)
            # if show_labels:
            #     ax.annotate(str(m.obs), xy=(x, y))
    elif path and nodes and len(path) == len(nodes) and show_matching:
        logger.debug('Plot matching path-nodes (using sequence of nodes) ...')
        for idx, (loc, node) in enumerate(zip(path, nodes)):
            x, y = to_pixels(*loc)
            if type(node) == tuple and (len(node) == 4 or len(node) == 2):
                latlon2, latlon3 = map_con.node_coordinates(node[0]), map_con.node_coordinates(node[1])
                if coord_trans:
                    latlon2 = coord_trans(*latlon2)
                    latlon3 = coord_trans(*latlon3)
                latlon4, _ = project(latlon2, latlon3, loc)
                x4, y4 = to_pixels(*latlon4)
                ax.plot((x, x4), (y, y4), '-', color=match_color, linewidth=linewidth, alpha=0.75)
            elif type(node) == tuple and len(node) == 3:
                lat2, lon2 = map_con.node_coordinates(node[0])
                if coord_trans:
                    lat2, lon2 = coord_trans(lat2, lon2)
                x2, y2 = to_pixels(lat2, lon2)
                ax.plot((x, x2), (y, y2), '-', color=match_color, linewidth=linewidth, alpha=0.75)
            elif type(node) == str or type(node) == int:
                lat2, lon2 = map_con.node_coordinates(node[0])
                if coord_trans:
                    lat2, lon2 = coord_trans(lat2, lon2)
                x2, y2 = to_pixels(lat2, lon2)
                ax.plot((x, x2), (y, y2), '-', color=match_color, linewidth=linewidth, alpha=0.75)
            else:
                raise Exception('Unknown node type: {} ({})'.format(node, type(node)))
            # if show_labels:
            #     ax.annotate(str(idx), xy=(x, y))
    if map_con.use_latlon:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    ax.axis('equal')
    ax.set_aspect('equal')
    if filename is not None:
        plt.savefig(filename)
        if fig is not None:
            plt.close(fig)
            fig = None
            ax = None
    return fig, ax


def plot_lattice(ax, to_pixels, matcher):
    for idx in range(len(matcher.lattice)):
        if len(matcher.lattice[idx]) == 0:
            continue
        for m in matcher.lattice[idx].values():
            for mp in m.prev:
                if m.stop:
                    alpha = 0.1
                    linewidth = 1
                else:
                    alpha = 0.3
                    linewidth = 3
                if mp.edge_m.p2 is None:
                    prv = mp.edge_m.p1
                else:
                    prv = mp.edge_m.p2
                nxt = m.edge_m.p1
                x1, y1 = to_pixels(*prv)
                x2, y2 = to_pixels(*nxt)
                ax.plot((x1, x2), (y1, y2), '.-', color=lattice_color, linewidth=linewidth, alpha=alpha)
                if m.edge_m.p2 is not None:
                    x1, y1 = to_pixels(*m.edge_m.p1)
                    x2, y2 = to_pixels(*m.edge_m.p2)
                    ax.plot((x1, x2), (y1, y2), '.-', color=lattice_color, linewidth=linewidth, alpha=alpha)


def plot_obs_noise_dist(matcher):
    """Plot the expected noise of an observation distribution.

    :param matcher: Matcher
    :return:
    """
    x = np.linspace(matcher.obs_noise_dist.ppf(0.01), matcher.obs_noise_dist.ppf(0.999), 100)
    y = matcher.obs_noise_dist.pdf(x) * math.exp(matcher.obs_noise_logint)
    plt.plot(x, y)
    plt.xlabel("Distance")
    plt.ylabel("Probability")
    plt.axvline(x=matcher.obs_noise, color='red', alpha=0.7)
    plt.annotate("Observation noise stddev", xy=(matcher.obs_noise, 0))
