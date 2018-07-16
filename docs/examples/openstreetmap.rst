Map from OpenStreetMap
======================

You can download a graph for map-matching from the OpenStreetMap.org service.
Multiple methods exists, we illustrate two.


Download a map as XML
---------------------

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


Create graph using osmread and pyproj
-------------------------------------

Once we have a file containing the region we are interested in, we can select the roads we want to use
to create a graph from. In this case we focus on 'ways' with a 'highway' tag. Those represent a variety
of roads. For a more detailed filtering look at the
`possible values of the highway tag <https://wiki.openstreetmap.org/wiki/Key:highway>`_.

It is also recommended to project the latitude-longitude coordinates to an Euclidean space.
In the example below this is achieved using the projection available in the utils, which is based on the
`pyproj <https://jswhit.github.io/pyproj/>`_ package.
But any other projection can be used by using the pyproj package directly.

.. code-block:: python
    import leuvenmapmatching as mm
    from leuvenmapmatching.util.projections import latlon2grs80
    import osmread
    map_con = mm.map.InMemMap(use_latlon=use_latlon)
    for entity in osmread.parse_file(str(xml_file)):
        if isinstance(entity, osmread.Way) and 'highway' in entity.tags:
            for node_a, node_b in zip(entity.nodes, entity.nodes[1:]):
                map_con.add_edge(node_a, None, node_b, None)
                # Some roads are one-way. We'll add both directions.
                map_con.add_edge(node_b, None, node_a, None)
        if isinstance(entity, osmread.Node):
            lat, lon = list(latlon2grs80([(entity.lat, entity.lon)]))[0]
            map_con.add_node(entity.id, (lat, lon))
    map_con.purge()


Create graph using osmnx and geopandas
--------------------------------------

Another great library to interact with OpenStreetMap data is the `osmnx <https://github.com/gboeing/osmnx>`_ package.
The osmnx package can retrieve relevant data automatically, for example when given a name of a region.
This package is build on top of the `geopandas <http://geopandas.org>`_ package that also includes routines to
perform projections.

.. code-block:: python
    import osmnx
    graph = ox.graph_from_place('Leuven, Belgium', network_type='drive')
    graph_proj = ox.project_graph(graph)
    # Create GeoDataFrames
    nodes_proj, edges_proj = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True)
    for nid, row in edges_proj[['u', 'v']].iterrows():
        map_cont.add_edge(row['u'], None, row['v'], None)
    for nid, row in nodes_proj[['x', 'y']].iterrows()
        map_con.add_node(nid, (row['x'], row['y']))


The projections can also be achieved directly on the GeoDataFrame:

.. code-block:: python
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    nodes.crs = {'init': 'epsg:4326'}  # WGS 84, System used in GPS
    nodes_proj = nodes.to_crs({'init': 'epsg:3395'})  # Mercator projection
    edges.crs = {'init': 'epsg:4326'}
    edges_proj = nodes.to_crs({'init': 'epsg:3395'})

When projecting both the map and the track you want to match, make sure to use the same projection.
