import os

from tdastro.sources.salt2 import SALT2JaxModel


def test_load_salt2_model(test_data_dir):
    """Test loading a Salt2 object from a file."""
    dir_name = os.path.join(test_data_dir, "truncated-salt2-h17")
    model = SALT2JaxModel(
        x0=0.5,
        x1=0.2,
        c=1.0,
        model_dir=dir_name,
        m0_filename="fake_salt2_template_0.dat",
        m1_filename="fake_salt2_template_1.dat",
        cl_filename="salt2_color_correction.dat",
    )

    assert model._color_law is not None
    assert model._m0_model is not None
    assert model._m1_model is not None
