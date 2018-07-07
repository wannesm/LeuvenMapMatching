.. Leuven.MapMatching documentation master file, created by
   sphinx-quickstart on Sat Apr 14 23:24:31 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Leuven.MapMatching's documentation!
==============================================

Align a trace of coordinates (e.g. GPS measurements) to a map of road segments.

The matching is based on a Hidden Markov Model (HMM) with non-emitting
states. The model can deal with missing data and you can plug in custom
transition and emission probability distributions.

Reference:

   Meert Wannes, Mathias Verbeke, "HMM with Non-Emitting States for Map Matching",
   European Conference on Data Analysis (ECDA), Paderborn, Germany, 2018.


.. figure:: http://people.cs.kuleuven.be/wannes.meert/dtaimapmatching/example1.png
   :alt: example


.. toctree::
   :maxdepth: 2
   :caption: Contents:


.. toctree::
   :caption: Usage


   usage/installation


.. toctree::
   :caption: Examples


   examples/introduction


.. toctree::
   :caption: Classes

   classes/Matching
   classes/Matcher


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
