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
* A model that defines the properties of the time-domain source, which can 
  also include a host-galaxy model, and is used to generate the noise-free light curves.
* ``Opsim`` contains the survey information such as survey strategy and observing
  conditions. It is used to specify the observing times and bands.
* A set of predefined effects, such as dust extinction and detector noise, are applied to
  the noise-free light curves to produce realistic light curves.
* The ``PassbandGroup`` contains the filter information of the telescope and is used
  to calculate the fluxes in each band.

See the :doc:`Glossary <glossary>` for definitions of key terms, such as
*GraphState*, *Node*, *Parameter*, *ParameterizedNode*, *PhysicalModel*, and *Source*.

Defining a parameterized model
-------------------------------------------------------------------------------

The core idea behind TDAstro is that we want to generate light curves from parameterized models
of physical objects. The ``PhysicalModel`` class defines the structure for modeling physical objects.
New object types are derived from the ``PhysicalModel`` base class and implement a ``compute_sed()``
function that generates the noise-free flux densities in the object's rest frame given information about
the times, wavelengths, and model parameters (called graph_state). Both the times and wavelengths are
converted to account for redshift before being passed to the ``compute_sed()`` function.

.. code-block:: python

    def compute_sed(self, times, wavelengths, graph_state, **kwargs):

A user of a particular physical model only needs to understand what parameters the model has
and how they are set. A user creating a new physical model additionally needs to know how the noise-free,
rest frame flux density values are generated from those parameters.

The parameters that are defined by a hierarchical model can be visualized by a Directed Acyclic Graph (DAG).
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

See the :doc:`Introduction notebook<notebooks/introduction>` for details on how to
define the parameter DAG.


Generating light curves
-------------------------------------------------------------------------------

Light curves are generated with a multiple step process. First, the object's parameter DAG is sampled
to get concrete values for each parameter in the model. This combination of parameters is call the graph
state (and stored in a ``GraphState`` object), because it represents the sampled state of the DAG.

Next, the ``OpSim`` is used to determine at what times and in which bands the object will be evaluated.
These times and wavelengths are based into the object's ``evaluate_sed()`` function along with the graph state.
The ``evaluate_sed()`` function handles the mechanics of the simulation, such as applying redshifts to both the
times and wavelengths before calling the ``compute_sed()``.

.. figure:: _static/compute_sed.png
   :class: no-scaled-link
   :scale: 80 %
   :align: center
   :alt: An example of the compute_sed function

   An example of the compute_sed function

Additional effects can be applied to the noise-free light curves to produce more realistic light curves.
The effects are applied in two batches. Rest frame effects are applied to the flux densities in the frame.
The flux densities are then converted to the observer frame where the observer frame effects are applied.

Finally, the raw flux densities are are converted into the magnitudes observed in each band using the
``PassbandGroup``.


Generating band flux curves
-------------------------------------------------------------------------------

All sources provide a helper function, ``evaluate_band_fluxes()``, that wraps the combination of
evaluation and integration with the passbands. This function takes the passband information,
a list of times, and a list of filter names. It returns the band flux at each of those times
in each of the filters.

.. figure:: _static/GetBandFluxes.png
   :class: no-scaled-link
   :scale: 80 %
   :align: center
   :alt: An example of the evaluate_band_fluxes function

   An example of the evaluate_band_fluxes function

In addition to being a convenient helper function, generating the data at the band flux level allows
certain models to skip SED generation. In particular a ``BandfluxModel`` is a subclass of the ``PhysicalModel``
whose computation is only defined at the band flux level. An example of this are models of empirically
fit light curves, such as those from LCLIB. Since we do not have the underlying SEDs for these types of models,
so we can only work with them at the band flux level. See the
:doc:`lightcurve source <notebooks/lightcurve_source_demo>` for an example of this type of model.

**Note** that most models in TDAstro operate at the SED level and we *strongly* encourage new models to
produce SEDs where possible. Working at the finer grained level allows more comprehensive and accurate
simulations, such as accounting for wavelength and time compression due to redshift. The models that generate
band fluxes directly will not account for all of these factors.


Examples
-------------------------------------------------------------------------------

After loading the necessary information (such as ``PassbandGroup`` and ``Opsim``),
and defining the physical model for our source, we can generate light curves with realistic
cadence and noise.

.. figure:: _static/lightcurves.png
   :class: no-scaled-link
   :scale: 80 %
   :align: center
   :alt: Simulated light curves of SNIa from LSST

   Simulated light curves of SNIa from LSST

See our selection of :doc:`tutorial notebooks <notebooks>` for further examples.