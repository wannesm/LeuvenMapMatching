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

from . import util


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")
match_color = mcolors.CSS4_COLORS['olive']
lattice_color = mcolors.CSS4_COLORS['magenta']
# nodes_color = mcolors.CSS4_COLORS['orange']
nodes_color = mcolors.CSS4_COLORS['black']
# path_color = mcolors.CSS4_COLORS['blue']
path_color = mcolors.CSS4_COLORS['blue']


def plot_map(map_con, path=None, nodes=None, counts=None, ax=None, use_osm=False, z=None, bb=None,
             show_labels=False, matcher=None, hide_graph=False, zoom_path=False, show_lattice=False,
             show_matching=False, filename=None, linewidth=2):
    """Plot the db/graph and optionally include the observed path and inferred nodes.

    :param map_con: Map
    :param path: list[(lat, lon)]
    :param nodes: list[str]
    :param counts: Number of visits of a node in the lattice. dict[label, int]
    :param ax: Matplotlib axis
    :param use_osm: Use OpenStreetMap layer, the points should be latitude-longitude pairs.
    :param matcher: Matcher object (overrules given path, nodes and counts)
    :param filename: File to write image to
    :return: None
    """
    if matcher is not None:
        path = matcher.path
        counts = matcher.node_counts()
        nodes = matcher.path_pred_onlynodes
        lat_nodes = matcher.lattice_best
    else:
        lat_nodes = None
    graph = map_con.get_graph()
    glat, glon = zip(*[t[0] for t in graph.values()])
    lat_min, lat_max = min(glat), max(glat)
    lon_min, lon_max = min(glon), max(glon)
    if path:
        plat, plon = islice(zip(*path), 2)
        if zoom_path:
            lat_min, lat_max = min(plat), max(plat)
            lon_min, lon_max = min(plon), max(plon)
        else:
            lat_min, lat_max = min(lat_min, min(plat)), max(lat_max, max(plat))
            lon_min, lon_max = min(lon_min, min(plon)), max(lon_max, max(plon))
        logger.debug("Setting bounding box to path")
    else:
        plat, plon = None, None

    width = 10

    if use_osm:
        if bb is None:
            bb = [lat_min, lon_min, lat_max, lon_max]
            # logger.debug("bb = [{}, {}, {}, {}]".format(*bb))
        if z is None:
            z = 18
        m = smopy.Map(bb, z=z, ax=ax)
        coord_trans = m.to_pixels
        x_max, y_max = coord_trans(lat_max, lon_max)
        x_min, y_min = coord_trans(lat_min, lon_min)
        height = width / abs(x_max - x_min) * abs(y_max - y_min)
        if ax is None:
            ax = m.show_mpl(figsize=(width, height))
        else:
            ax = m.show_mpl(ax=ax)
        fig = None

    else:
        lat_max += (lat_max - lat_min) * 0.1
        lon_min -= (lon_max - lon_min) * 0.1
        lat_min -= (lat_max - lat_min) * 0.1
        lon_max += (lon_max - lon_min) * 0.1

        def coord_trans(lat, lon=None):
            if lon is None:
                lat, lon = lat[0], lat[1]
            return lon, lat
        # coord_trans = lambda lat_, lon_: (lon_, lat_)
        if bb is None:
            x_max, y_max = coord_trans(lat_max, lon_max)
            x_min, y_min = coord_trans(lat_min, lon_min)
        else:
            x_max, y_max = coord_trans(bb[2], bb[3])
            x_min, y_min = coord_trans(bb[0], bb[1])
        height = width / abs(lon_max - lon_min) * abs(lat_max - lat_min)
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(width, height))
        else:
            fig = None
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

    if counts is None:
        node_sizes = [10]*len(graph)
    else:
        node_sizes = [counts[label]*100+5 for label in graph.keys()]

    if not hide_graph:
        logger.debug('Plot vertices ...')
        gx, gy = zip(*[coord_trans(t[0]) for t in graph.values()])
        ax.scatter(gx, gy, s=node_sizes, alpha=0.4)
        if show_labels:
            for key, t in graph.items():
                ax.annotate(str(key), xy=coord_trans(t[0]))

        logger.debug('Plot lines ...')
        for _, loc_a, _, loc_b in map_con.all_edges():
            x_a, y_a = coord_trans(*loc_a)
            x_b, y_b = coord_trans(*loc_b)
            ax.plot([x_a, x_b], [y_a, y_b], 'k', linewidth=0.3)

    # for label, ((lat, lon), nbrs, _) in graph.items():
    #     ax.annotate(label, xy=(lon, lat))
    #     for nbr in nbrs:
    #         lat2, lon2 = graph[nbr][0]
    #         plt.plot((lon, lon2), (lat, lat2), 'k', linewidth=0.3)

    if show_lattice:
        if matcher is None:
            raise Exception("matcher needs to be passed to show lattice")
        plot_lattice(ax, coord_trans, matcher)

    if path:
        logger.debug('Plot path ...')
        px, py = zip(*[coord_trans(p) for p in path])
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
                x, y = coord_trans(*graph[node[0]][0])
                xs.append(x)
                ys.append(y)
            elif type(node) == str or type(node) == int:
                x, y = coord_trans(*graph[node][0])
                xs.append(x)
                ys.append(y)
        ax.plot(xs, ys, linewidth=linewidth, alpha=0.75, color=nodes_color)

    if matcher and show_matching:
        logger.debug('Plot matching path-nodes (using matcher) ...')
        for idx, m in enumerate(lat_nodes):
            x, y = coord_trans(*m.edge_m.pi[:2])
            x2, y2 = coord_trans(*m.edge_o.pi[:2])
            ax.plot((x, x2), (y, y2), '-', color=match_color, linewidth=linewidth, alpha=0.75)
            # ax.plot((x, x2), (y, y2), '-', color=match_color, linewidth=10, alpha=0.1)
            # if show_labels:
            #     ax.annotate(str(m.obs), xy=(x, y))
    elif path and nodes and len(path) == len(nodes) and show_matching:
        logger.debug('Plot matching path-nodes (using sequence of nodes) ...')
        for idx, (loc, node) in enumerate(zip(path, nodes)):
            x, y = coord_trans(*loc)
            if type(node) == tuple and (len(node) == 4 or len(node) == 2):
                latlon2, latlon3 = graph[node[0]][0], graph[node[1]][0]
                latlon4, _ = util.project(latlon2, latlon3, loc)
                x4, y4 = coord_trans(*latlon4)
                ax.plot((x, x4), (y, y4), '-', color=match_color, linewidth=linewidth, alpha=0.75)
            elif type(node) == tuple and len(node) == 3:
                x2, y2 = coord_trans(*graph[node[0]][0])
                ax.plot((x, x2), (y, y2), '-', color=match_color, linewidth=linewidth, alpha=0.75)
            elif type(node) == str or type(node) == int:
                x2, y2 = coord_trans(*graph[node][0])
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
    if filename is not None:
        plt.savefig(filename)
        if fig is not None:
            plt.close(fig)
            fig = None
            ax = None
    return fig, ax


def plot_lattice(ax, coord_trans, matcher):
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
                if mp.loc2 is None:
                    prv = mp.loc1
                else:
                    prv = mp.loc2
                nxt = m.loc1
                x1, y1 = coord_trans(*prv)
                x2, y2 = coord_trans(*nxt)
                ax.plot((x1, x2), (y1, y2), '.-', color=lattice_color, linewidth=linewidth, alpha=alpha)
                if m.loc2 is not None:
                    x1, y1 = coord_trans(*m.loc1)
                    x2, y2 = coord_trans(*m.loc2)
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
