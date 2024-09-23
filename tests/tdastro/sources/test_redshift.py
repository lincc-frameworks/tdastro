import numpy as np
from tdastro.sources.step_source import StepSource


def test_redshift_values() -> None:
    """Test that we correctly calculate redshifted values."""
    times = np.linspace(0, 100, 1000)
    wavelengths = np.array([100.0, 200.0, 300.0])
    t0 = 10.0
    t1 = 30.0
    brightness = 50.0

    for redshift in [0.0, 0.5, 2.0, 3.0, 30.0]:
        model_redshift = StepSource(brightness=brightness, t0=t0, t1=t1, redshift=redshift)
        values_redshift = model_redshift.evaluate(times, wavelengths)

        for i, time in enumerate(times):
            if t0 <= time and time <= (t1 - t0) * (1 + redshift) + t0:
                assert np.all(values_redshift[i] == brightness / (1 + redshift))
            else:
                assert np.all(values_redshift[i] == 0.0)
