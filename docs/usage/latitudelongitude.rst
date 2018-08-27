Dealing with Latitude-Longitude
===============================

The toolbox can deal with latitude-longitude coordinates directly.
Map matching, however, requires a lot of repeated computations between points and latitude-longitude
computations will be more expensive than Euclidean distances.

There are three different options how you can handle latitude-longitude coordinates:

Option 1: Use Latitude-Longitude directly
-----------------------------------------

Set the ``use_latlon`` flag in the :class:`Map` to true.

For example to read in an OpenStreetMap file directly to a :class:`InMemMap` object:

.. code-block:: python

    from leuvenmapmatching.map.inmem import InMemMap

    map_con = InMemMap("myosm", use_latlon=True)

    for entity in osmread.parse_file(osm_fn):
        if isinstance(entity, osmread.Way) and 'highway' in entity.tags:
            for node_a, node_b in zip(entity.nodes, entity.nodes[1:]):
                map_con.add_edge(node_a, node_b)
                map_con.add_edge(node_b, node_a)
        if isinstance(entity, osmread.Node):
            map_con.add_node(entity.id, (entity.lat, entity.lon))
    map_con.purge()


Option 2: Project Latitude-Longitude to X-Y
-------------------------------------------

Latitude-Longitude coordinates can be transformed two a frame with two orthogonal axis.

.. code-block:: python

   from leuvenmapmatching.map.inmem import InMemMap

   map_con_latlon = InMemMap("myosm", use_latlon=True)
   # Add edges/nodes
   map_con_xy = map_con_latlon.to_xy()

   route_latlon = []
   # Add GPS locations
   route_xy = [map_con_xy.latlon2yx(latlon) for latlon in route_latlon]


This can also be done directly using the `pyproj <https://github.com/jswhit/pyproj>`_ toolbox.
For example, using the Lambert Conformal projection to project the route GPS coordinates:

.. code-block:: python

   import pyproj

   route = [(4.67878,50.864),(4.68054,50.86381),(4.68098,50.86332),(4.68129,50.86303),(4.6817,50.86284),
            (4.68277,50.86371),(4.68894,50.86895),(4.69344,50.86987),(4.69354,50.86992),(4.69427,50.87157),
            (4.69643,50.87315),(4.69768,50.87552),(4.6997,50.87828)]
   lon_0, lat_0 = route[0]
   proj = pyproj.Proj(f"+proj=merc +ellps=GRS80 +units=m +lon_0={lon_0} +lat_0={lat_0} +lat_ts={lat_0} +no_defs")
   xs, ys = [], []
   for lon, lat in route:
       x, y = proj(lon, lat)
       xs.append(x)
       ys.append(y)


Notice that the pyproj package uses the convention to express coordinates as x-y which is
longitude-latitude because it is defined this way in the CRS definitions while the Leuven.MapMatching
toolbox follows the ISO 6709 standard and expresses coordinates as latitude-longitude. If you
want ``pyproj`` to use latitude-longitude you can use set the
`axisswap option <https://proj4.org/operations/conversions/axisswap.html>`_.

If you want to define both the from and to projections:

.. code-block:: python

   import pyproj

   route = [(4.67878,50.864),(4.68054,50.86381),(4.68098,50.86332),(4.68129,50.86303),(4.6817,50.86284),
            (4.68277,50.86371),(4.68894,50.86895),(4.69344,50.86987),(4.69354,50.86992),(4.69427,50.87157),
            (4.69643,50.87315),(4.69768,50.87552),(4.6997,50.87828)]
   p1 = pyproj.Proj(proj='latlon', datum='WGS84')
   p2 = pyproj.Proj(proj='utm', datum='WGS84')
   xs, ys = [], []
   for lon, lat in route:
       x, y = pyproj.transform(lon, lat)
       xs.append(x)
       ys.append(y)


Option 3: Use Latitude-Longitude as if they are X-Y points
----------------------------------------------------------

A naive solution would be to use latitude-longitude coordinate pairs as if they are X-Y coordinates.
For small distances, far away from the poles and not crossing the dateline, this option might work.
But it is not adviced.

For example, for long distances the error is quite large. In the image beneath, the blue line is the computation
of the intersection using latitude-longitude while the red line is the intersection using Eucludean distances.

.. figure:: https://people.cs.kuleuven.be/wannes.meert/leuvenmapmatching/latlon_mismatch_1.png?v=1
   :alt: Latitude-Longitude mismatch

.. figure:: https://people.cs.kuleuven.be/wannes.meert/leuvenmapmatching/latlon_mismatch_2.png?v=1
   :alt: Latitude-Longitude mismatch detail
