"""Classes to wrap functions to allow users to pass around functions
with partially specified arguments.
"""

import copy


class TDFunc:
    """A class to wrap functions to pass around functions with default
    argument settings, arguments from kwargs, and (optionally)
    arguments that are from the fcalling object.

    The object stores default arguments for the function, which does not
    need to include all the function's parameters. Any parameters not
    included in the default list must be specified as part of the kwargs.

    Attributes
    ----------
    func : `function` or `method`
        The function to call during an evaluation.
    default_args : `dict`
        A dictionary of default arguments to pass to the function.
        This does not need to include all the arguments.
    object_args : `list`, optional
        Arguments that are provided by attributes of the calling object.
    """

    def __init__(self, func, default_args=None, object_args=None, **kwargs):
        self.func = func
        self.object_args = object_args
        self.default_args = {}

        # The default arguments are the union of the default_args parameter
        # and the remaining kwargs.
        if default_args is not None:
            self.default_args = default_args
        if kwargs:
            for key, value in kwargs.items():
                self.default_args[key] = value

    def __str__(self):
        """Return the string representation of the function."""
        return f"TDFunc({self.func.name})"

    def __call__(self, calling_object=None, **kwargs):
        """Execute the wrapped function.

        Parameters
        ----------
        calling_object : any, optional
            The object that called the function.
        **kwargs : `dict`, optional
            Additional function arguments.
        """
        # Start with the default arguments. We make a copy so we can modify the dictionary.
        args = copy.copy(self.default_args)

        # If there are arguments to get from the calling object, set those.
        if self.object_args is not None and len(self.object_args) > 0:
            if calling_object is None:
                raise ValueError(f"Calling object needed for parameters: {self.object_args}")
            for arg_name in self.object_args:
                args[arg_name] = getattr(calling_object, arg_name)

        # Set any last arguments from the kwargs (overwriting previous settings).
        args.update(kwargs)

        # Call the function with all the parameters.
        return self.func(**args)
