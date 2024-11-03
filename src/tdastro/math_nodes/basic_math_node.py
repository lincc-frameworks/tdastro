"""Nodes that perform basic math operations."""

import ast
import math

import jax.numpy as jnp
import numpy as np

from tdastro.base_models import FunctionNode


class BasicMathNode(FunctionNode):
    """A node that evaluates basic mathematical functions.

    Attributes
    ----------
    expression : `str`
        The expression to evaluate.
    backend : `str`
        The math libary to use. Must be one of: math, numpy, or jax.
    tree : `ast.*`
        The root node of the parsed syntax tree.

    Parameters
    ----------
    expression : `str`
        The expression to evaluate.
    backend : `str`
        The math libary to use. Must be one of: math, numpy, or jax.
    node_label : `str`, optional
        An identifier (or name) for the current node.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    # A list of supported Python operations. Used to prevent eval from
    # running arbitrary python expressions.
    _supported_ast_nodes = (
        ast.Module,  # Top level object when parsed as exec.
        ast.Expression,  # Top level object when parsed as eval.
        ast.Expr,  # Math expressions.
        ast.Constant,  # Constant values.
        ast.Name,  # A named variable or function.
        ast.BinOp,  # Binary operations
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UnaryOp,  # Uninary operations
        ast.UAdd,
        ast.USub,
        ast.Invert,
        # Call functions (but we do NOT include ast.Call because
        # we need to special case that).
        ast.Load,
        ast.Store,
    )

    # A very limited set of math operations that are supported
    # in all of the backends.
    _math_funcs = set(
        [
            "abs",
            "ceil",
            "cos",
            "cosh",
            "degrees",
            "exp",
            "floor",
            "log",
            "log10",
            "log2",
            "radians",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
        ]
    )

    def __init__(self, expression, backend="numpy", node_label=None, **kwargs):
        self.expression = expression
        self.backend = backend

        # Check that all the functions are supported.
        if backend == "math":
            supported_funcs = dir(math)
            supported_funcs.append("abs")
        elif backend == "jax":
            supported_funcs = dir(jnp)
        elif backend == "numpy":
            supported_funcs = dir(np)
        else:
            raise ValueError(f"Unsupported math backend {backend}")

        for fn_name in self._math_funcs:
            if fn_name not in supported_funcs:
                raise ValueError(f"Function {fn_name} is not supported by {backend}.")

        # Check the expression is pure math and translate it into the correct backend.
        self._compile()

        # Create a function from the expression. Note the expression has
        # already been sanitized and validated via _compile().
        def eval_func(**kwargs):
            return eval(self.expression, globals(), kwargs)

        super().__init__(eval_func, node_label=node_label, **kwargs)

    def __call__(self, **kwargs):
        """Evaluate thge"""
        return eval(self.expression, globals(), kwargs)

    def _compile(self, **kwargs):
        """Compile a python expression that consists of only basic math.

        Parameters
        ----------
        **kwargs : `dict`, optional
            Any additional keyword arguments, including the variable
            assignments.

        Returns
        -------
        tree : `ast.*`
            The root node of the parsed syntax tree.
        """
        tree = ast.parse(self.expression)

        # Walk the tree and confirm that it only contains the basic math.
        for node in ast.walk(tree):
            if isinstance(node, self._supported_ast_nodes):
                # Nothing to do, this is a valid operation for the ast.
                continue
            elif isinstance(node, (ast.FunctionType, ast.Call)):
                func_name = node.func.id
                if func_name not in self._math_funcs:
                    raise ValueError("Unsupported function {func_name}.")

                if self.backend == "numpy":
                    node.func.id = f"np.{func_name}"
                elif self.backend == "jax":
                    node.func.id = f"jnp.{func_name}"
                elif self.backend == "math" and "func_name" != "abs":
                    node.func.id = f"math.{func_name}"
            else:
                raise ValueError(f"Invalid part of expression {type(node)}")

        # Convert the expression back into a string.
        self.expression = ast.unparse(tree)
