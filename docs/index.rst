.. ClusterStack documentation master file, created by
   sphinx-quickstart on Mon Aug 27 14:27:12 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ClusterStack: models for lensing signals from cross-correlation measurements
============================================================================

ClusterStack generates models for lensing signals from cross-correlation measurements.  The user
must supply callable functions for the `\DeltaSigma` model as well as the mass-concentration
relation; prescriptions for scatter are included, and the ClusterStack objects generate Monte Carlo
realizations of the scatter, which converge well as long as the parameter space is well-sampled.

ClusterStack objects, and this package in general, also include convenience functions to allow
sampling via the package `emcee` (url; Foreman_Mackey et al.)

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   classes



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
