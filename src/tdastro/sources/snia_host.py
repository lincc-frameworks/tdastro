from tdastro.sources.physical_model import PhysicalModel


class SNIaHost(PhysicalModel):
    """
    A SN Ia host galaxy model with a hostmass parameter, more to be added.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("hostmass", **kwargs)
