"""Utilities to wrap functions for inclusion in an evaluation graph."""

import copy


class TDFunc:
    """A class to wrap functions and their argument settings.

    Attributes
    ----------
    func : `function` or `method`
        The function to call during an evaluation.
    default_args : `dict`
        A dictionary of default arguments to pass to the function. Assembled
        from the ```default_args`` parameter and additional ``kwargs``.
    setter_functions : `dict`
        A dictionary mapping arguments names to functions, methods, or
        TDFunc objects used to set that argument. These are evaluated dynamically
        each time.

    Examples
    --------
    my_func = TDFunc(random.randint, a=1, b=10)
    value1 = my_func()      # Sample from default range
    value2 = my_func(b=20)  # Sample from extended range

    Note
    ----
    All the function's parameters that will be used need to be specified
    in either the default_args dict, object_args list, or as a kwarg in the
    constructor. Arguments cannot be first given during function call.
    For example the following will fail (because b is not defined in the
    constructor):

    my_func = TDFunc(random.randint, a=1)
    value1 = my_func(b=10.0)
    """

    def __init__(self, func, **kwargs):
        self.func = func
        self.default_args = {}
        self.setter_functions = {}

        for key, value in kwargs.items():
            if callable(value):
                self.default_args[key] = None
                self.setter_functions[key] = value
            else:
                self.default_args[key] = value

    def __str__(self):
        """Return the string representation of the function."""
        return f"TDFunc({self.func.name})"

    def __call__(self, **kwargs):
        """Execute the wrapped function.

        Parameters
        ----------
        **kwargs : `dict`, optional
            Additional function arguments.
        """
        # Start with the default arguments. We make a copy so we can modify the dictionary.
        args = copy.copy(self.default_args)

        # If there are arguments to get from the calling functions, set those.
        for key, value in self.setter_functions.items():
            if isinstance(value, TDFunc):
                args[key] = value(**kwargs)
            else:
                args[key] = value()

        # Set any last arguments from the kwargs (overwriting previous settings).
        # Only use known kwargs.
        for key, value in kwargs.items():
            if key in self.default_args:
                args[key] = value

        # Call the function with all the parameters.
        return self.func(**args)
