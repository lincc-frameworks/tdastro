import numpy as np

from tdastro.base_models import PopulationModel


class FixedPopulation(PopulationModel):
    """A population with a predefined, fixed probability of sampling each source.

    Attributes
    ----------
    probs : `numpy.ndarray`
        The probability of drawing each type of source.
    _raw_rates : `numpy.ndarray`
        An array of floats that provides the base sampling rate for each
        type. This is normalized into a probability distributions so
        [100, 200, 200] -> [0.2, 0.4, 0.4].
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.probs = np.array([])
        self._raw_rates = np.array([])

    def __str__(self):
        """Return the string representation of the model."""
        return f"FixedPopulation({self.probability})"

    def _update_probabilities(self):
        """Update the probability array."""
        self.probs = self._raw_rates / np.sum(self._raw_rates)

    def add_source(self, new_source, rate, **kwargs):
        """Add a new source to the population.

        Parameters
        ----------
        new_source : `PhysicalModel`
            A source from the population.
        rate : `float`
            A numerical rate for drawing the object.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Raises
        ------
        ``ValueError`` if the rate <= 0.0.
        """
        if rate <= 0.0:
            raise ValueError(f"Expected positive rate. Found {rate}.")
        super().add_source(new_source, **kwargs)

        self._raw_rates = np.append(self._raw_rates, rate)
        self._update_probabilities()

    def change_rate(self, source_index, rate, **kwargs):
        """Add a new source to the population.

        Parameters
        ----------
        source_index : `int`
            The index of the source whose rate is changing.
        rate : `float`
            A numerical rate for drawing the object.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Raises
        ------
        ``ValueError`` if the rate <= 0.0.
        """
        if rate <= 0.0:
            raise ValueError(f"Expected positive rate. Found {rate}.")
        self._raw_rates[source_index] = rate
        self._update_probabilities()

    def draw_source(self):
        """Sample a single source from the population.

        Returns
        -------
        source : `PhysicalModel`
            A source from the population.
        """
        index = self._rng.choice(np.arange(0, self.num_sources), p=self.probs)
        return self.sources[index]
