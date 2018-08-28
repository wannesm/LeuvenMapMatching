Visualisation
=============

To inspect the results, a plotting function is included.

Simple plotting
---------------

To plot the graph in a matplotlib figure use:

.. code-block:: python

    from leuvenmapmatching import visualization as mmviz
    mmviz.plot_map(map_con, matcher=matcher,
                   show_labels=True, show_matching=True, show_graph=True,
                   filename="my_plot.png")

This will result in the following figure:

.. figure:: https://people.cs.kuleuven.be/wannes.meert/leuvenmapmatching/plot1.png?v=1
   :alt: Plot1

You can also define your own figure by passing a matplotlib axis object:

.. code-block:: python

    fig, ax = plt.subplots(1, 1)
    mmviz.plot_map(map_con, matcher=matcher,
                   ax=ax,
                   show_labels=True, show_matching=True, show_graph=True,
                   filename="my_plot.png")


Plotting with an OpenStreetMap background
-----------------------------------------

The plotting function also supports a link with the ``smopy`` package.
Set the ``use_osm`` argument to true and pass a map that is defined with
latitude-longitude (thus ``use_latlon=True``).

You can set ``zoom_path`` to true to only see the relevant part and not the
entire map that is available in the map. Alternatively you can also set the
bounding box manually using the ``bb`` argument.

.. code-block:: python

    mm_viz.plot_map(map_con, matcher=matcher,
                    use_osm=True, zoom_path=True,
                    show_labels=False, show_matching=True, show_graph=False,
                    filename="my_osm_plot.png")


This will result in the following figure:

.. figure:: https://people.cs.kuleuven.be/wannes.meert/leuvenmapmatching/plot2.png?v=1
   :alt: Plot2

Or when some GPS points are missing in the track, the matching is more
visible as the matched route deviates from the straight line between two
GPS points:

.. figure:: https://people.cs.kuleuven.be/wannes.meert/leuvenmapmatching/plot3.png?v=1
   :alt: Plot3
