import math

import jax
import pytest
from tdastro.math_nodes.basic_math_node import BasicMathNode
from tdastro.math_nodes.single_value_node import SingleVariableNode


def test_basic_math_node():
    """Test that we can perform computations via a BasicMathNode."""
    node_a = SingleVariableNode("a", 10.0)
    node_b = SingleVariableNode("b", -5.0)
    node = BasicMathNode("a + b", a=node_a.a, b=node_b.b, node_label="test", backend="math")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 5.0

    # Try with a math function.
    node_c = SingleVariableNode("c", 1000.0)
    node = BasicMathNode("a + b - log10(c)", a=10.0, b=5.0, c=node_c.c, node_label="test", backend="math")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 12.0

    # Try with a second math function.
    node = BasicMathNode(
        "sqrt(a) + b - log10(c)", a=16.0, b=4.0, c=node_c.c, node_label="test", backend="math"
    )
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 5.0

    # Test that we can reproduce the power function.
    node_d = SingleVariableNode("d", 5.0)
    node = BasicMathNode("a ** b", a=node_d.d, b=2.5, node_label="test", backend="math")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == pytest.approx(math.pow(5.0, 2.5))


def test_basic_math_node_numpy():
    """Test that we can perform computations via a BasicMathNode."""
    node_a = SingleVariableNode("a", 10.0)
    node_b = SingleVariableNode("b", -5.0)
    node = BasicMathNode("a + b", a=node_a.a, b=node_b.b, node_label="test", backend="numpy")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 5.0

    # Try with a math function.
    node_c = SingleVariableNode("c", 1000.0)
    node = BasicMathNode("a + b - log10(c)", a=10.0, b=5.0, c=node_c.c, node_label="test", backend="numpy")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 12.0

    # Try with a second math function.
    node = BasicMathNode(
        "sqrt(a) + b - log10(c)", a=16.0, b=4.0, c=node_c.c, node_label="test", backend="numpy"
    )
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 5.0

    # Test that we can reproduce the power function.
    node_d = SingleVariableNode("d", 5.0)
    node = BasicMathNode("a ** b", a=node_d.d, b=2.5, node_label="test", backend="math")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == pytest.approx(math.pow(5.0, 2.5))


def test_basic_math_node_jax():
    """Test that we can perform computations via a BasicMathNode."""
    node_a = SingleVariableNode("a", 10.0)
    node_b = SingleVariableNode("b", -5.0)
    node = BasicMathNode("a + b", a=node_a.a, b=node_b.b, node_label="test", backend="jax")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 5.0

    # Try with a math function.
    node_c = SingleVariableNode("c", 1000.0)
    node = BasicMathNode("a + b - log10(c)", a=10.0, b=5.0, c=node_c.c, node_label="test", backend="jax")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 12.0

    # Try with a second math function.
    node = BasicMathNode(
        "sqrt(a) + b - log10(c)", a=16.0, b=4.0, c=node_c.c, node_label="test", backend="jax"
    )
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 5.0

    # Test that we can reproduce the power function.
    node_d = SingleVariableNode("d", 5.0)
    node = BasicMathNode("a ** b", a=node_d.d, b=2.5, node_label="test", backend="math")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == pytest.approx(math.pow(5.0, 2.5))


def test_basic_math_node_autodiff_jax():
    """Test that we can do auto-differentiation with JAX."""
    node_a = SingleVariableNode("a", 16.0, node_label="a_node")
    node_b = SingleVariableNode("b", 1000.0, node_label="b_node")

    # Create a basic math function and create tghe pytree.
    node = BasicMathNode(
        "sqrt(a) + 1.0 - log10(b)", a=node_a.a, b=node_b.b, node_label="diff_test", backend="jax"
    )
    state = node.sample_parameters()
    pytree = node.build_pytree(state)

    gr_func = jax.value_and_grad(node.resample_and_compute)
    values, gradients = gr_func(pytree)
    assert values == 2.0
    assert gradients["a_node"]["a"] > 0.0
    assert gradients["b_node"]["b"] < 0.0
