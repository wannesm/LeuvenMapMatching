Map from OpenStreetMap
======================

You can download a graph for map-matching from the OpenStreetMap.org service.
Multiple methods exists, we illustrate two.

Using requests, osmread and gpx
-------------------------------

You can perform map matching on a OpenStreetMap database by combing ``leuvenmapmatching``
with the packages ``requests``, ``osmread`` and ``gpx``.

Download a map as XML
~~~~~~~~~~~~~~~~~~~~~

You can use the overpass-api.de service:

.. code-block:: python

    from pathlib import Path
    import requests
    xml_file = Path(".") / "osm.xml"
    url = 'http://overpass-api.de/api/map?bbox=4.694933,50.870047,4.709256000000001,50.879628'
    r = requests.get(url, stream=True)
    with xml_file.open('wb') as ofile:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                ofile.write(chunk)


Create graph using osmread
~~~~~~~~~~~~~~~~~~~~~~~~~~

Once we have a file containing the region we are interested in, we can select the roads we want to use
to create a graph from. In this case we focus on 'ways' with a 'highway' tag. Those represent a variety
of roads. For a more detailed filtering look at the
`possible values of the highway tag <https://wiki.openstreetmap.org/wiki/Key:highway>`_.

.. code-block:: python

    from leuvenmapmatching.map.inmem import InMemMap
    import osmread

    map_con = InMemMap("myosm", use_latlon=True, use_rtree=True, index_edges=True)
    for entity in osmread.parse_file(str(xml_file)):
        if isinstance(entity, osmread.Way) and 'highway' in entity.tags:
            for node_a, node_b in zip(entity.nodes, entity.nodes[1:]):
                map_con.add_edge(node_a, node_b)
                # Some roads are one-way. We'll add both directions.
                map_con.add_edge(node_b, node_a)
        if isinstance(entity, osmread.Node):
            map_con.add_node(entity.id, (entity.lat, entity.lon))
    map_con.purge()


Note that ``InMemMap`` is a simple container for a map. It is recommended to use
your own optimized connecter to your map dataset.

If you want to allow transitions that are not following the exact road segments you can inherit from the ``Map``
class and define a new class with your own transitions.
The transitions are defined using the ``nodes_nbrto`` and ``edges_nbrt`` methods.


Perform map matching on an OpenStreetMap database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create a list of latitude-longitude coordinates manually. Or read a gpx file.

.. code-block:: python

    from leuvenmapmatching.util.gpx import gpx_to_path

    track = gpx_to_path("mytrack.gpx")
    matcher = DistanceMatcher(map_con,
                             max_dist=100, max_dist_init=25,  # meter
                             min_prob_norm=0.001,
                             non_emitting_length_factor=0.75,
                             obs_noise=50, obs_noise_ne=75,  # meter
                             dist_noise=50,  # meter
                             non_emitting_states=True)
    states, lastidx = matcher.match(track)


Using osmnx and geopandas
-------------------------

Another great library to interact with OpenStreetMap data is the `osmnx <https://github.com/gboeing/osmnx>`_ package.
The osmnx package can retrieve relevant data automatically, for example when given a name of a region.
This package is build on top of the `geopandas <http://geopandas.org>`_ package.

.. code-block:: python

    import osmnx
    graph = ox.graph_from_place('Leuven, Belgium', network_type='drive')
    graph_proj = ox.project_graph(graph)
    
    # Create GeoDataFrames
    # Approach 1
    nodes_proj, edges_proj = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True)
    for nid, row in nodes_proj[['x', 'y']].iterrows():
        map_con.add_node(nid, (row['x'], row['y']))
    for nid, row in edges_proj[['u', 'v']].iterrows():
        map_con.add_edge(row['u'], row['v'])
    
    # Approach 2
    nodes, edges = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True)
    
    nodes_proj = nodes.to_crs("EPSG:3395")
    edges_proj = edges.to_crs("EPSG:3395")
    
    for nid, row in nodes_proj.iterrows():
        map_con.add_node(nid, (row['lat'], row['lon']))
    
    # adding edges using networkx graph
    for nid1, nid2, _ in graph.edges:
        map_con.add_edge(nid1, nid2)


