#!/usr/bin/env python

import argparse


def parse_args(args):
    """Parse the command line arguments.

    Parameters
    ----------
    args : `list` of `str`
        The command line arguments.

    Returns
    -------
    `argparse.Namespace`
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Create a truncated version of a SALT2 table.")
    parser.add_argument("--input", default="", help="The input SALT2 data file.")
    parser.add_argument("--output", default="", help="The file name for the output data.")
    parser.add_argument("--t_min", default=-100.0, help="The minimum time to include.")
    parser.add_argument("--t_max", default=100.0, help="The maxmimum time to include.")
    parser.add_argument("--w_min", default=0.0, help="The minimum wavelength to include.")
    parser.add_argument("--w_max", default=25000.0, help="The maxmimum wavelength to include.")
    return parser.parse_args(args)


def main(args=None):
    """Subsample an OpSim file and save it to a new file.

    Parameters
    ----------
    args : `list` of `str`, optional
        The command line arguments.
    """
    args = parse_args(args)
    if args.input == args.output:
        raise ValueError("Cannot overwrite file.")
    times = (float(args.t_min), float(args.t_max))
    waves = (float(args.w_min), float(args.w_max))

    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Time range: {times}")
    print(f"Wavelength range: {waves}")

    with open(args.input, "r") as input_file:
        with open(args.output, "w") as output_file:
            for line in input_file:
                tokens = line.split()
                if (times[0] <= float(tokens[0]) <= times[1]) and (waves[0] <= float(tokens[1]) <= waves[1]):
                    output_file.write(line)


if __name__ == "__main__":
    main()
