from tdastro.sources.snia_host import SNIaHost


def test_snia_host():
    """
    Test that we can initialize a host with a hostmass parameter
    """

    host = SNIaHost(hostmass=10.0)
    state = host.sample_parameters()
    assert host.get_param(state, "hostmass") == 10.0
