import numpy as np

from dustmaps.sfd import SFDQuery

from tdastro.effects.dust import DustExtinction
from tdastro.sources.static_source import StaticSource


def test_sfd_dust_extinction(tdastro_data_dir):
    """Test that we can sample and create a DustExtinction object."""
    dust_map = SFDQuery()
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    wavelengths = np.array([800.0, 900.0, 1000.0, 1000.0, 900.0])

    # Create a model without any dust.
    model_clean = StaticSource(brightness=100.0, ra=0.0, dec=40.0)
    fluxes_clean = model_clean.evaluate(times, wavelengths)
    assert len(fluxes_clean) == 5
    assert np.all(fluxes_clean == 100.0)

    # Create a model with ccm89 extinction.
    model_ccm89 = StaticSource(brightness=100.0, ra=0.0, dec=40.0)
    model_ccm89.add_effect(DustExtinction(dust_map, "ccm89", r_v=3.1))
    fluxes_ccm98 = model_ccm89.evaluate(times, wavelengths)
    assert len(fluxes_ccm98) == 5
    assert np.all(fluxes_ccm98 < 100.0)

    # Create a model with ccm89 extinction.
    model_fitzpatrick99 = StaticSource(brightness=100.0, ra=0.0, dec=40.0)
    model_fitzpatrick99.add_effect(DustExtinction(dust_map, "fitzpatrick99"))
    fluxes_fitzpatrick99 = model_fitzpatrick99.evaluate(times, wavelengths)
    assert len(fluxes_fitzpatrick99) == 5
    assert np.all(fluxes_fitzpatrick99 < 100.0)
