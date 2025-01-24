Simulations
========================================================================================

Introduction
-------------------------------------------------------------------------------

.. figure:: _static/tdastro-intro.png
   :class: no-scaled-link
   :scale: 80 %
   :align: center
   :alt: TDAstro simulation components

   TDAstro simulation components

The main simulation components in TDAstro include:

* ``PhysicalModel`` defines the properties of the time-domain source, which can 
  also include a host-galaxy model.
* ``EffectModel`` describes how the SED changes (e.g. redshifting, lensing, dust)
* ``PassbandGroup`` contains the filter information of the telescope
* ``Opsim`` contains the survey information such as survey strategy and observing
  conditions.

We can generate random realizations of the ``PhysicalModel`` and produce realistic
light curves using the above components.

Defining a model with a DAG
-------------------------------------------------------------------------------

A hierarchical model can be visualized by a Directed Acyclic Graph (DAG). In TDAstro,
we can define a ``PhysicalModel`` based on a DAG. For example, a simplified SNIa
model with a host component can have the following DAG:

.. figure:: _static/dag-model.png
   :class: no-scaled-link
   :scale: 80 %
   :align: center
   :alt: An example DAG for a SNIa model

   An example DAG for a SNIa model

Examples
-------------------------------------------------------------------------------

After loading the necessary information (such as ``PassbandGroup`` and ``Opsim``),
and defining the ``PhysicalModel``, we can generate light curves with realistic
cadence and noise.

.. figure:: _static/lightcurves.png
   :class: no-scaled-link
   :scale: 80 %
   :align: center
   :alt: Simulated light curves of SNIa from LSST

   Simulated light curves of SNIa from LSST

See our selection of :doc:`tutorial notebooks <notebooks>` for further examples.