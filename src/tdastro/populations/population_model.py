"""The base population models."""

import numpy as np

from tdastro.base_models import ParameterizedNode
from tdastro.sources.physical_model import PhysicalModel


class PopulationModel(ParameterizedNode):
    """A model of a population of PhysicalModels.

    Attributes
    ----------
    num_sources : `int`
        The number of different sources in the population.
    sources : `list`
        A list of sources from which to draw.
    """

    def __init__(self, rng=None, **kwargs):
        super().__init__(**kwargs)
        self.num_sources = 0
        self.sources = []

    def add_source(self, new_source, **kwargs):
        """Add a new source to the population.

        Parameters
        ----------
        new_source : `PhysicalModel`
            A source from the population.
        **kwargs : `dict`, optional
           Any additional keyword arguments.
        """
        if not isinstance(new_source, PhysicalModel):
            raise ValueError("All sources must be PhysicalModels")
        self.sources.append(new_source)
        self.num_sources += 1

    def draw_source(self):
        """Sample a single source from the population.

        Returns
        -------
        source : `PhysicalModel`
            A source from the population.
        """
        raise NotImplementedError()

    def evaluate(self, samples, times, wavelengths, **kwargs):
        """Draw observations from a single (randomly sampled) source.

        Parameters
        ----------
        samples : `int`
            The number of sources to samples.
        times : `numpy.ndarray`
            A length T array of timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        results : `numpy.ndarray`
            A shape (samples, T, N) matrix of SED values.
        """
        if samples <= 0:
            raise ValueError("The number of samples must be > 0.")

        results = []
        for _ in range(samples):
            source = self.draw_source()
            state = source.sample_parameters()
            object_fluxes = source.evaluate(times, wavelengths, state, **kwargs)
            results.append(object_fluxes)
        return np.array(results)
