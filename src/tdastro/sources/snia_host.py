from tdastro.sources.physical_model import PhysicalModel


class SNIaHost(PhysicalModel):
    """A SN Ia host galaxy model with a hostmass parameter, more to be added.

    Parameterized values include:
      * dec - The object's declination in degrees. [from PhysicalModel]
      * distance - The object's luminosity distance in pc. [from PhysicalModel]
      * hostmass - The hostmass value.
      * ra - The object's right ascension in degrees. [from PhysicalModel]
      * redshift - The object's redshift. [from PhysicalModel]
      * t0 - The t0 of the zero phase, date. [from PhysicalModel]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("hostmass", **kwargs)
