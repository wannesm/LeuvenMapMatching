Dealing with Latitude-Longitude
===============================

The toolbox can deal with latitude-longitude coordinates and uses the nvector package for this.
Map matching, however, requires a lot of repeated computations between points and latitude-longitude
computations can be rather expensive compared to simple Euclidean distances.
Therefore, it is recommended to project your latitude-longitude coordinates first to an x-y plane and
use these coordinates for map matching.

There are three options:

Option 1: Project Latitude-Longitude to X-Y
-------------------------------------------

This is the recommended approach.

Latitude-Longitude coordinates can be transformed two a frame with two orthogonal axis.
For example, using the Lambert Conformal projection. When using the `pyproj` toolbox this can be done as follows:

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


Or if you want to define from and to projections:

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


Option 2: Use Latitude-Longitude directly
-----------------------------------------

Set the `use_latlon` flag in the :class:`Map` to true. The Leuven.MapMatching toolbox will use
the `nvector` package to compute distance between latitude-longitude coordinates.

For example to read in an OpenStreetMap file directly to a :class:`InMemMap` object:

.. code-block:: python

    map_con = mm.map.InMemMap(use_latlon=True)
    for entity in osmread.parse_file(osm_fn):
        if isinstance(entity, osmread.Way) and 'highway' in entity.tags:
            for node_a, node_b in zip(entity.nodes, entity.nodes[1:]):
                map_con.add_edge(node_a, None, node_b, None)
                map_con.add_edge(node_b, None, node_a, None)
        if isinstance(entity, osmread.Node):
            map_con.add_node(entity.id, (entity.lat, entity.lon))
    map_con.purge()


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

