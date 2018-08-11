#!/usr/bin/env python3
# encoding: utf-8
"""
tests.test_bugs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import os
import sys
import logging
from pathlib import Path
import leuvenmapmatching as mm
from leuvenmapmatching.map.inmemmap import InMemMap


directory = None


def test_bug1():
    dist = 10
    nb_steps = 20

    map_con = InMemMap("map", graph={
        "A":  ((1, dist), ["B"]),
        "B":  ((2, dist), ["A", "C", "CC"]),
        "C":  ((3, 0), ["B", "D"]),
        "D":  ((4 + dist, 0), ["C", "E"]),
        "CC": ((3, 2 * dist), ["B", "DD"]),
        "DD": ((4 + dist, 2 * dist), ["CC", "E"]),
        "E":  ((5 + dist, dist), ["F", "D", "DD"]),
        "F":  ((6 + dist, dist), ["E", ]),

    }, use_latlon=False)

    i = 10
    path = [(1.1,      2*dist*i/nb_steps),
            (2.1,      2*dist*i/nb_steps),
            (5.1+dist, 2*dist*i/nb_steps),
            (6.1+dist, 2*dist*i/nb_steps)
            # (1, len*i/nb_steps),
            # (2, len*i/nb_steps),
            # (3, len*i/nb_steps)
            ]

    matcher = mm.matching.Matcher(map_con, max_dist=dist + 1, obs_noise=dist + 1, min_prob_norm=None,
                                  non_emitting_states=True)

    nodes = matcher.match(path, unique=False)
    print("Solution: ", nodes)
    if directory:
        import leuvenmapmatching.visualization as mm_vis
        matcher.print_lattice()
        matcher.print_lattice_stats()
        mm_vis.plot_map(map_con, path=path, nodes=nodes, counts=matcher.node_counts(),
                        show_labels=True, filename=str(directory / "test_bugs_1.png"))


if __name__ == "__main__":
    mm.matching.logger.setLevel(logging.DEBUG)
    mm.matching.logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    test_bug1()
