from tdastro.math_nodes.single_value_node import SingleVariableNode


def test_single_variable_node():
    """Test that we can create and query a SingleVariableNode."""
    node = SingleVariableNode("A", 10.0)
    assert str(node) == "SingleVariableNode"

    state = node.sample_parameters()
    assert node.get_param(state, "A") == 10
