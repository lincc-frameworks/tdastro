
.. tdastro documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TDAstro
========================================================================================

Time-Domain Forward-Modeling for the Rubin Era
-------------------------------------------------------------------------------

Realistic light curve simulations are essential to many time-domain problems. 
Simulations are needed to evaluate observing strategy, characterize biases, 
and test pipelines. TDAstro aims to provide a flexible, scalable, and user-friendly
time-domain simulation software with realistic effects and survey strategies.

.. figure:: _static/tdastro-intro.png
   :class: no-scaled-link
   :scale: 80 %
   :align: center
   :alt: TDAstro simulation components


The main simulation components in TDAstro include:

* A statistical simulation step where the hyperparameters of the model are drawn
  from one or more prior distributions. This can include existing packages such as
  `pzflow <https://pzflow.readthedocs.io/en/latest/>`_ or custom distributions.
* A model that defines the properties of the time-domain light source, which can
  also include a host-galaxy model, and is used to generate the noise-free light curves.
* ``ObsTable`` contains the survey information such as survey strategy and observing
  conditions. It is used to specify the observing times and bands.
* A set of predefined effects, such as dust extinction and detector noise, are applied to
  the noise-free light curves to produce realistic light curves.
* The ``PassbandGroup`` contains the filter information of the telescope and is used
  to calculate the fluxes in each band.

TDAstro can generate numerous random realizations of the parameters for the physical model 
(``SEDModel`` or ``BandfluxModel``), and then apply the effects to these realizations. The ``ObsTable``
component is used to produce realistic light curves using the above components.  See the
:doc:`simulations <simulations>` page for a more detailed description of the process.

For an overview of the package, we recommend starting with the notebooks in the "Getting Started"
section of the :doc:`notebooks page <notebooks>`. The :doc:`glossary <glossary>` provides definitions of
key terms, such as *GraphState*, *Node*, *Parameter*, *ParameterizedNode*, *BasePhysicalModel*,
*BandfluxModel*, and *SEDModel*.

The `full source code <https://github.com/lincc-frameworks/tdastro>`_ is available on GitHub.


Getting Started
-------------------------------------------------------------------------------

You can install TDAstro from PyPI with pip or from conda-forge with conda. We recommend using a dedicated environment.

.. tab-set::
   :sync-group: packagemanager

   .. tab-item:: pip
      :sync: pip

      .. code-block:: bash

         # Create a virtual environment
         python3 -m venv ~/envs/tdastro
         # Activate it
         source ~/envs/tdastro/bin/activate
         # Install from PyPI
         python -m pip install tdastro

   .. tab-item:: conda
      :sync: conda

      .. code-block:: bash

         # Create a virtual environment
         conda create -p tdastro python=3.12
         # Activate it
         conda activate tdastro
         # Install from conda-forge channel
         conda install conda-forge::tdastro

Since TDAstro relies on a large number of existing packages, not all of the packages
are installed in the default configuration. For example the microlensing (`VBMicrolensing`),
pzflow (`pzflow`), and sncosmo (`sncosmo`) packages are not included by default. If you try to
import a module that is not installed, TDAstro will raise an `ImportError` with information on which
packages you need to install. You will need to install these manually. You can also install most
optional dependencies with:

.. tab-set::
   :sync-group: packagemanager

   .. tab-item:: pip
      :sync: pip

      .. code-block:: bash

         python -m pip install 'tdastro[all]'

   .. tab-item:: conda
      :sync: conda

      .. code-block:: bash

         conda install conda-forge::tdastro conda-forge::pzflow conda-forge::sncosmo

See our selection of :doc:`tutorial notebooks <notebooks>` for usage examples.
We recommend starting with the :doc:`introduction notebook <notebooks/introduction>`
to get a high level overview.

If you are interested in installing from `source <https://github.com/lincc-frameworks/tdastro>`_,
or contributing to the package, see the :doc:`contribution guide <contributing>`.

Running TDAstro in a Jupyter Notebook
-------------------------------------------------------------------------------

Note that to run the TDAstro within a notebook, your notebook kernel will need to have TDAstro installed.
If you are using JupyterLab, you can install the kernel with the following commands
(using the Terminal program):

.. code-block:: bash

   >> python3 -m pip install ipykernel
   >> python3 -m ipykernel install --user --name=tdastro

You will need to restart your notebook server to see the new kernel.


Acknowledgements
-------------------------------------------------------------------------------

This project is supported by Schmidt Sciences.


.. toctree::
   :hidden:

   Home page <self>
   Simulations <simulations>
   Notebooks <notebooks>
   API Reference <autoapi/index>
   Glossary <glossary>
   Contribution Guide <contributing>
   Citations <citations>
