"""Benchmarks for core TDAstro functionality.

To manually run the benchmarks use: asv run

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

from pathlib import Path

import numpy as np
from astropy import units as u
from tdastro.astro_utils.passbands import PassbandGroup
from tdastro.astro_utils.snia_utils import DistModFromRedshift, HostmassX1Func, X0FromDistMod
from tdastro.astro_utils.unit_utils import fnu_to_flam
from tdastro.base_models import FunctionNode
from tdastro.effects.white_noise import WhiteNoise
from tdastro.math_nodes.np_random import NumpyRandomFunc
from tdastro.sources.basic_sources import LinearWavelengthSource, StepSource
from tdastro.sources.lightcurve_source import LightcurveSource
from tdastro.sources.sncomso_models import SncosmoWrapperModel

# ASV runs from copy of the project (benchmarks/env/....). So we load the
# data files based off the current file location instead.
_PROJECT_BASE_DIR = Path(__file__).parent.parent
_TEST_DATA_DIR = _PROJECT_BASE_DIR / "tests" / "tdastro" / "data"


def _load_test_passbands():
    """Load passbands to use in various benchmarks."""
    passbands_dir = _TEST_DATA_DIR / "passbands"
    passbands = PassbandGroup(
        given_passbands=[
            {
                "filter_name": "g",
                "survey": "LSST",
                "table_path": passbands_dir / "LSST" / "g.dat",
            },
            {
                "filter_name": "r",
                "survey": "LSST",
                "table_path": passbands_dir / "LSST" / "r.dat",
            },
        ],
        survey="LSST",
        units="nm",
        trim_quantile=0.001,
        delta_wave=1,
    )
    return passbands


class TimeSuite:
    """A suite of timing functions."""

    def setup(self):
        """Set up items that will be used in multiple tests."""
        # Preload the passbands for tests that use them.
        self.passbands = _load_test_passbands()

        # Create a model we can use in tests.
        self.redshift = 0.1
        self.hostmass = 8.0
        self.distmod_func = DistModFromRedshift(self.redshift, H0=73.0, Omega_m=0.3)
        self.x1_func = HostmassX1Func(self.hostmass)
        self.x0_func = X0FromDistMod(
            distmod=self.distmod_func,
            x1=self.x1_func,
            c=0.0,
            alpha=0.14,
            beta=3.1,
            m_abs=-19.3,
        )

        self.salt3_model = SncosmoWrapperModel(
            "salt3",
            t0=0.0,
            x0=self.x0_func,
            x1=self.x1_func,
            c=0.0,
            ra=0.0,
            dec=0.0,
            redshift=self.redshift,
        )

        # A simple LinearWavelengthSource that we can use in tests.
        self.linear_source = LinearWavelengthSource(linear_base=1.0, linear_scale=0.1)

        # Create samples that we can use in tests.
        self.times = np.arange(-20.0, 50.0, 0.5)
        self.wavelengths = self.passbands.waves
        self.filter_options = ["LSST_g", "LSST_r"]
        self.filters = np.array([self.filter_options[i % 2] for i in range(len(self.times))])

        self.graph_state = self.salt3_model.sample_parameters()
        self.fluxes = self.salt3_model.evaluate(self.times, self.wavelengths, graph_state=self.graph_state)

        self.white_noise = WhiteNoise(white_noise_sigma=0.1)

    def time_chained_evaluate(self):
        """Time the generation of random numbers with an numpy generation node."""

        def _add_func(a, b):
            return a + b

        # Generate a starting mean and scale from uniform distributions. Use those to
        # generate a sample from the normal distribution. Then shift that sample by -5.0.
        loc_node = NumpyRandomFunc("uniform", low=10.0, high=20.0)
        scale_node = NumpyRandomFunc("uniform", low=0.5, high=1.0)
        norm_node = NumpyRandomFunc("normal", loc=loc_node, scale=scale_node)
        val_node = FunctionNode(_add_func, a=norm_node, b=-5.0)

        # Generate 100,000 samples.
        _ = val_node.sample_parameters(num_samples=100_000)

    def time_apply_white_noise(self):
        """Time the application of white noise to a sample."""
        _ = self.white_noise.apply(self.fluxes, white_noise_sigma=0.1)

    def time_make_x1_from_hostmass(self):
        """Time the creation of the X1 function."""
        _ = HostmassX1Func(self.hostmass)

    def time_sample_x1_from_hostmass(self):
        """Time the computation of the X1 function."""
        _ = self.x1_func.sample_parameters()

    def time_sample_x0_from_distmod(self):
        """Time the computation of the X0 function."""
        _ = self.x0_func.sample_parameters()

    def time_make_and_evaluate_step_source(self):
        """Time creating and evaluating a StepSource."""
        model = StepSource(brightness=100.0, t0=2.0, t1=5.0)
        state = model.sample_parameters()
        times = np.arange(0.0, 10.0, 0.05)
        wavelengths = np.arange(1000.0, 2000.0, 5.0)
        _ = model.evaluate(times, wavelengths, state)

    def time_make_simple_linear_wavelength_source(self):
        """Time creating a simple LinearWavelengthSource."""
        _ = LinearWavelengthSource(linear_base=1.0, linear_scale=0.1)

    def time_evaluate_simple_linear_wavelength_source(self):
        """Time evaluating a simple LinearWavelengthSource."""
        _ = self.linear_source.evaluate(self.times, self.wavelengths)

    def time_make_new_salt3_model(self):
        """Time creating a new SALT3 model."""
        _ = SncosmoWrapperModel(
            "salt3",
            t0=0.0,
            x0=self.x0_func,
            x1=self.x1_func,
            c=0.0,
            ra=0.0,
            dec=0.0,
            redshift=self.redshift,
        )

    def time_evaluate_salt3_model(self):
        """Time querying a predefined salt3 model."""
        _ = self.salt3_model.evaluate(
            self.times,
            self.wavelengths,
            graph_state=self.graph_state,
        )

    def time_load_passbands(self):
        """Time loading the passbands from files."""
        _ = _load_test_passbands()

    def time_apply_passbands(self):
        """Time applying the (already loaded) passbands to flux."""
        _ = self.passbands.fluxes_to_bandfluxes(self.fluxes)

    def time_evaluate_salt3_passbands(self):
        """Time evaluate the SALT3 model at the passband level."""
        _ = self.salt3_model.get_band_fluxes(
            self.passbands,
            self.times,
            self.filters,
            self.graph_state,
        )

    def time_fnu_to_flam(self):
        """Time the fnu_to_flam function."""
        _ = fnu_to_flam(
            self.fluxes,
            self.wavelengths,
            wave_unit=u.AA,
            flam_unit=u.erg / u.second / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )

    def time_lightcurve_source(self):
        """Time the creation and query of a LightcurveSource."""
        lc_times = np.linspace(0.0, 6 * np.pi, 100)
        g_gluxes = np.sin(lc_times) + 3.0
        r_gluxes = np.cos(lc_times) + 5.0

        lightcurves = {
            "g": np.column_stack((lc_times, g_gluxes)),
            "r": np.column_stack((lc_times, r_gluxes)),
        }
        lc_source = LightcurveSource(
            lightcurves,
            self.passbands,
            lc_t0=0.0,
            periodic=True,
            baseline=None,
            t0=0.0,
        )

        # Sample the lightcurve source to ensure it works.
        _ = lc_source.evaluate(self.times, self.wavelengths)
