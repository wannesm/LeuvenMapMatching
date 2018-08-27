#!/usr/bin/env python3
# encoding: utf-8
"""
tests.test_nonemitting_circle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import sys, os
import logging
import math
from pathlib import Path
import numpy as np

try:
    import leuvenmapmatching as mm
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
    import leuvenmapmatching as mm
from leuvenmapmatching.matcher.simple import SimpleMatcher
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap


directory = None


def setup_map(disconnect=True):
    theta = np.linspace(0, 2 * math.pi, 4 * 5 + 1)[:-1]

    ox = 0.1 + np.cos(theta * 0.95)
    oy = np.sin(theta * 1)
    path1 = list(zip(ox, oy))  # all observations
    path2 = [(x, y) for x, y in zip(ox, oy) if x > -0.60]

    nx = np.cos(theta)
    ny = np.sin(theta)
    nl = [f"N{i}" for i in range(len(nx))]
    graph = {}
    for i, (x, y, l) in enumerate(zip(nx, ny, nl)):
        if disconnect:
            edges = []
            if i != len(nx) - 1:
                edges.append(nl[(i + 1) % len(nl)])
            if i != 0:
                edges.append(nl[(i - 1) % len(nl)])
        else:
            edges = [nl[(i - 1) % len(nl)], nl[(i + 1) % len(nl)]]
        graph[l] = ((x, y), edges)
    graph["M"] = ((0, 0), ["N5", "N15"])
    graph["N5"][1].append("M")
    graph["N15"][1].append("M")
    print(graph)

    path_sol = nl
    if not disconnect:
        path_sol += ["N0"]

    mapdb = InMemMap("map", graph=graph, use_latlon=False)
    return mapdb, path1, path2, path_sol


def visualize_map():
    if directory is None:
        return
    import matplotlib.pyplot as plt
    import leuvenmapmatching.visualization as mm_vis

    mapdb, path1, path2, path_sol = setup_map()

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 5))
    mm_vis.plot_map(mapdb, path=path1, ax=ax, show_labels=True)
    fig.savefig(str(directory / 'test_nonemitting_circle_map_path1.png'))
    plt.close(fig)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    mm_vis.plot_map(mapdb, path=path2, ax=ax, show_labels=True)
    fig.savefig(str(directory / 'test_nonemitting_circle_map_path2.png'))
    plt.close(fig)


def visualize_path(matcher, mapdb, name="test"):
    import matplotlib.pyplot as plt
    from leuvenmapmatching import visualization as mmviz
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    mmviz.plot_map(mapdb, matcher=matcher, ax=ax,
                   show_labels=True, show_matching=True, show_graph=True,
                   linewidth=2)
    fn = directory / f"test_nonemitting_circle_{name}_map.png"
    fig.savefig(str(fn))
    plt.close(fig)
    print(f"saved to {fn}")


def test_path1():
    mapdb, path1, path2, path_sol = setup_map()
    matcher = SimpleMatcher(mapdb, max_dist_init=1, min_prob_norm=0.5, obs_noise=0.5,
                            non_emitting_states=True)
    matcher.match(path1, unique=True)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        visualize_path(matcher, mapdb, name="testpath1")
    assert path_pred == path_sol, f"Nodes not equal:\n{path_pred}\n{path_sol}"


def test_path1_dist():
    mapdb, path1, path2, path_sol = setup_map()
    matcher = DistanceMatcher(mapdb, max_dist_init=1, min_prob_norm=0.8, obs_noise=0.5,
                              non_emitting_states=True)
    matcher.match(path1, unique=True)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        visualize_path(matcher, mapdb, name="test_path1_dist")
    assert path_pred == path_sol, f"Nodes not equal:\n{path_pred}\n{path_sol}"


def test_path2():
    mapdb, path1, path2, _ = setup_map()
    path_sol = [f"N{i}" for i in range(20)]
    matcher = SimpleMatcher(mapdb, max_dist_init=0.2, min_prob_norm=0.1,
                                  obs_noise=0.1, obs_noise_ne=1,
                                  non_emitting_states=True, only_edges=True)
    path_pred = matcher.match(path2, unique=True)
    print(path_pred)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        matcher.print_lattice_stats(verbose=True)
        matcher.print_lattice()
        print("Best path through lattice:")
        for m in matcher.lattice_best:
            print(m)
        visualize_path(matcher, mapdb, name="testpath2")
    assert path_pred == path_sol, f"Nodes not equal:\n{path_pred}\n{path_sol}"


def test_path2_dist():
    mapdb, path1, path2, _ = setup_map()
    path_sol = [f"N{i}" for i in range(20)]
    matcher = DistanceMatcher(mapdb, max_dist_init=0.2, min_prob_norm=0.1,
                              obs_noise=0.1, obs_noise_ne=1,
                              non_emitting_states=True)
    matcher.match(path2)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        visualize_path(matcher, mapdb, name="test_path2_dist")
    assert path_pred == path_sol, f"Nodes not equal:\n{path_pred}\n{path_sol}"


if __name__ == "__main__":
    # mm.matching.logger.setLevel(logging.INFO)
    mm.logger.setLevel(logging.DEBUG)
    mm.logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    # visualize_map()
    test_path1()
    # test_path1_dist()
    # test_path2()
    # test_path2_dist()
