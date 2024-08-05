import numpy as np
from tdastro.effects.redshift import Redshift
from tdastro.sources.step_source import StepSource


def get_no_effect_and_redshifted_values(times, wavelengths, t0, t1, brightness, redshift) -> tuple:
    """Get the values for a source with no effects and a redshifted source."""
    model_no_effects = StepSource(brightness=brightness, t0=t0, t1=t1)
    model_redshift = StepSource(brightness=brightness, t0=t0, t1=t1, redshift=redshift)
    model_redshift.add_effect(Redshift(redshift=model_redshift, t0=model_redshift))

    values_no_effects = model_no_effects.evaluate(times, wavelengths)
    values_redshift = model_redshift.evaluate(times, wavelengths)

    # Check shape of output is as expected
    assert values_no_effects.shape == (len(times), len(wavelengths))
    assert values_redshift.shape == (len(times), len(wavelengths))

    return values_no_effects, values_redshift


def test_redshift() -> None:
    """Test that we can create a Redshift object and it gives us values as expected."""
    times = np.array([1, 2, 3, 5, 10])
    wavelengths = np.array([100.0, 200.0, 300.0])
    t0 = 1.0
    t1 = 2.0
    brightness = 15.0
    redshift = 1.0

    # Get the values for a redshifted step source, and a step source with no effects for comparison
    (values_no_effects, values_redshift) = get_no_effect_and_redshifted_values(
        times, wavelengths, t0, t1, brightness, redshift
    )

    # Check that the step source activates within the correct time range:
    # For values_no_effects, the activated values are in the range [t0, t1]
    for i, time in enumerate(times):
        if t0 <= time and time <= t1:
            assert np.all(values_no_effects[i] == brightness)
        else:
            assert np.all(values_no_effects[i] == 0.0)

    # With redshift = 1.0, the activated values are *observed* in the range [(t0-t0)*(1+redshift)+t0,
    # (t1-t0*(1+redshift+t0] (the first term reduces to t0). Also, the values are scaled by a factor
    # of (1+redshift).
    for i, time in enumerate(times):
        if t0 <= time and time <= (t1 - t0) * (1 + redshift) + t0:
            assert np.all(values_redshift[i] == brightness / (1 + redshift))
        else:
            assert np.all(values_redshift[i] == 0.0)


def test_other_redshift_values() -> None:
    """Test that we can create a Redshift object with various other redshift values."""
    times = np.linspace(0, 100, 1000)
    wavelengths = np.array([100.0, 200.0, 300.0])
    t0 = 10.0
    t1 = 30.0
    brightness = 50.0

    for redshift in [0.0, 0.5, 2.0, 3.0, 30.0]:
        model_redshift = StepSource(brightness=brightness, t0=t0, t1=t1, redshift=redshift)
        model_redshift.add_effect(Redshift(redshift=model_redshift.redshift, t0=model_redshift.t0))
        values_redshift = model_redshift.evaluate(times, wavelengths)

        for i, time in enumerate(times):
            if t0 <= time and time <= (t1 - t0) * (1 + redshift) + t0:
                assert np.all(values_redshift[i] == brightness / (1 + redshift))
            else:
                assert np.all(values_redshift[i] == 0.0)
