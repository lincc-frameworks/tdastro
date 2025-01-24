
.. tdastro documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TDAstro
========================================================================================

Time-Domain Forward-Modeling for the Rubin Era
-------------------------------------------------------------------------------

Realistic light curve simulations are essential to many time-domain problems. 
simulations are needed to evaluate observing strategy, characterize biases, 
and test pipelines. The need for a flexible, scalable, and user-friendly time-domain
simulation software has increased as the new survey telescopes get ready for their
first lights. TDAstro aims to provide such software for the time domain community.

.. figure:: _static/tdastro-intro.png
   :class: no-scaled-link
   :scale: 80 %
   :align: center
   :alt: TDAstro simulation components

   TDAstro simulation components

The main simulation components in TDAstro include:

* ``PhysicalModel`` defines the properties of the time-domain source, which can 
  also include a host-galaxy model.
* ``PassbandGroup`` contains the filter information of the telescope
* ``Opsim`` contains the survey information such as survey strategy and observing
  conditions.

We can generate random realizations of the ``PhysicalModel`` and produce realistic
light curves using the above components.

Getting Started
-------------------------------------------------------------------------------

.. code-block:: bash

   >> pip install tdastro

See our selection of :doc:`tutorial notebooks <notebooks>` for usage examples.

If you are interested in installing from source, or contributing to the package,
see the :doc:`contribution guide <contributing>`.

.. toctree::
   :hidden:

   Home page <self>
   Simulations <simulations>
   Notebooks <notebooks>
   API Reference <autoapi/index>
   Contribution Guide <contributing>

Acknowledgements
-------------------------------------------------------------------------------

This project is supported by Schmidt Sciences.