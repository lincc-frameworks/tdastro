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

* A statistical simulation step where the hyperparameters of the model are drawn
  from one or more prior distributions.
* ``PhysicalModel`` defines the properties of the time-domain source, which can 
  also include a host-galaxy model, and is used to generate the noise-free light curves.
* ``Opsim`` contains the survey information such as survey strategy and observing
  conditions. It is used to specify the observing times and bands.
* A set of predefined effects, such as dust extinction and detector noise, are applied
   to the noise-free light curves to produce realistic light curves.
* The ``PassbandGroup`` contains the filter information of the telescope and is used
  to calculate the fluxes in each band.

Defining a parameterized model
-------------------------------------------------------------------------------

The core idea behind TDAstro is that we want to generate light curves from parameterized models
of physical objects. The ``PhysicalModel`` class defines the structure for modeling physical object.
New object types are derived from the ``PhysicalModel`` base class and implement a ``compute_flux()``
function that generates the noise-free flux densities given information about the times, wavelengths,
and model parameters (called graph_state). 

.. code-block:: python
    def compute_flux(self, times, wavelengths, graph_state, **kwargs):

A user using a particular physical model only needs to understand what parameters the model has
and how they are set. A user creating a new physical model additionally needs to know how the noise-free
flux density values are generated from those parameters.

The parameters are defined by a hierarchical model can be visualized by a Directed Acyclic Graph (DAG).
This means that the parameters to our physical model, such as a type Ia supernova, can themselves be sampled
based on distributions of hyperparameters. For example, a simplified SNIa model with a host component
can have the following DAG:

.. figure:: _static/dag-model.png
   :class: no-scaled-link
   :scale: 80 %
   :align: center
   :alt: An example DAG for a SNIa model

   An example DAG for a SNIa model

In this example, the parameter ``c`` is drawn from a predefined distribution, while the parameter ``x1``
is drawn from a distribution that is itself parameterized by the ``host_mass`` parameter. TDAstro handles
the sequential processing of the graph so that all parameters are consistently sampled for each object.

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