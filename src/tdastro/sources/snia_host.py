from tdastro.sources.physical_model import PhysicalModel

class SNIaHost(PhysicalModel):
    """A static source.

    Attributes
    ----------
    radius_std : `float`
        The standard deviation of the brightness as we move away
        from the galaxy's center (in degrees).
    brightness : `float`
        The inherent brightness at the center of the galaxy.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("hostmass", required=True, **kwargs)