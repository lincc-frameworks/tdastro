import random

from tdastro.populations.population_model import PopulationModel


class FixedPopulation(PopulationModel):
    """A population with a predefined, fixed probability of sampling each source.

    Attributes
    ----------
    weights : `numpy.ndarray`
        An array of floats that provides the base sampling rate for each
        type. This is normalized into a probability distributions so
        [100, 200, 200] -> [0.2, 0.4, 0.4].
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weights = []

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
        self.weights.append(rate)

    def change_rate(self, source_index, rate, **kwargs):
        """Change rate of a source.

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
        self.weights[source_index] = rate

    def draw_source(self):
        """Sample a single source from the population.

        Returns
        -------
        source : `PhysicalModel`
            A source from the population.
        """
        return random.choices(self.sources, weights=self.weights)[0]
