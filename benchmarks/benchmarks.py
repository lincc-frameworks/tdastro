"""Benchmarks for core TDAstro functionality.

To manually run the benchmarks use: asv run

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

import numpy as np
import tdastro
from astropy import units as u
from tdastro.astro_utils.passbands import PassbandGroup
from tdastro.astro_utils.snia_utils import DistModFromRedshift, HostmassX1Func, X0FromDistMod
from tdastro.astro_utils.unit_utils import fnu_to_flam
from tdastro.base_models import FunctionNode
from tdastro.math_nodes.np_random import NumpyRandomFunc
from tdastro.sources.sncomso_models import SncosmoWrapperModel


def _load_test_passbands():
    """Load passbands to use in various benchmarks."""
    passbands_dir = tdastro._TDASTRO_TEST_DATA_DIR / "passbands"
    passbands = PassbandGroup(
        passband_parameters=[
            {
                "filter_name": "g",
                "table_path": passbands_dir / "LSST" / "g.dat",
            },
            {
                "filter_name": "r",
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

        # Create samples that we can use in tests.
        self.times = np.arange(-20.0, 50.0, 0.5)
        self.wavelengths = self.passbands.waves
        self.filter_options = ["LSST_g", "LSST_r"]
        self.filters = np.array([self.filter_options[i % 2] for i in range(len(self.times))])

        self.graph_state = self.salt3_model.sample_parameters()
        self.fluxes = self.salt3_model.evaluate(self.times, self.wavelengths, graph_state=self.graph_state)

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

    def time_make_x1_from_hostmass(self):
        """Time the creation of the X1 function."""
        _ = HostmassX1Func(self.hostmass)

    def time_sample_x1_from_hostmass(self):
        """Time the computation of the X1 function."""
        _ = self.x1_func.sample_parameters()

    def time_sample_x0_from_distmod(self):
        """Time the computation of the X0 function."""
        _ = self.x0_func.sample_parameters()

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
