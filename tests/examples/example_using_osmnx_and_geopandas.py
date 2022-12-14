import os
import sys
import logging
from pathlib import Path

this_path = Path(os.path.realpath(__file__)).parent.parent / "rsrc" / "path_latlon"
assert(this_path.exists())
path_to_mytrack_gpx = this_path / "route.gpx"
assert(path_to_mytrack_gpx.exists())

import leuvenmapmatching as mm
from leuvenmapmatching.map.inmem import InMemMap


def run():
    # Start example
    import osmnx as ox

    # Select map (all, drive, walk, ...)
    graph = ox.graph_from_place('Leuven, Belgium', network_type='all', simplify=False)
    graph_proj = ox.project_graph(graph)

    # Create GeoDataFrames
    # Approach 1: translate map to graph
    # DistanceMatcher uses edges, thus build index based on edges
    map_con = InMemMap("myosm", use_latlon=True, use_rtree=True, index_edges=True)
    nodes_proj, edges_proj = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True)
    for nid, row in nodes_proj[['x', 'y']].iterrows():
        map_con.add_node(nid, (row['x'], row['y']))
    for eid, _ in edges_proj.iterrows():
        map_con.add_edge(eid[0], eid[1])

    # Approach 2: use a specific projection
    map_con = InMemMap("myosm", use_latlon=True, use_rtree=True, index_edges=True)
    nodes_proj, edges_proj = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True)
    nodes_proj = nodes_proj.to_crs("EPSG:3395")
    # edges_proj = edges_proj.to_crs("EPSG:3395")
    for nid, row in nodes_proj.iterrows():
        map_con.add_node(nid, (row['lat'], row['lon']))
    # We can also extract edges also directly from networkx graph
    for nid1, nid2, _ in graph.edges:
        map_con.add_edge(nid1, nid2)

    # Perform matching
    from leuvenmapmatching.util.gpx import gpx_to_path
    from leuvenmapmatching.matcher.distance import DistanceMatcher

    track = gpx_to_path(path_to_mytrack_gpx)
    matcher = DistanceMatcher(map_con,
                             max_dist=100, max_dist_init=50,  # meter
                             non_emitting_length_factor=0.75,
                             obs_noise=50, obs_noise_ne=75,  # meter
                             dist_noise=50,  # meter
                             non_emitting_states=True,
                             max_lattice_width=5)
    states, lastidx = matcher.match(track)
    print(states)

    # End example

    # import leuvenmapmatching.visualization as mm_viz
    # import matplotlib as mpl
    # mpl.use('MacOSX')
    # mm_viz.plot_map(map_con, matcher=matcher, use_osm=True,
    #                 zoom_path=True, show_graph=True,
    #                 filename=Path(os.environ.get('TESTDIR', Path(__file__).parent)) / "example.png")


if __name__ == "__main__":
    mm.logger.setLevel(logging.INFO)
    mm.logger.addHandler(logging.StreamHandler(sys.stdout))
    run()
