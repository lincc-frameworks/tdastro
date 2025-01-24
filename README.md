# TDAstro

Time-Domain Forward-Modeling for the Rubin Era

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/tdastro?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/tdastro/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/tdastro/smoke-test.yml)](https://github.com/lincc-frameworks/tdastro/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/lincc-frameworks/tdastro/branch/main/graph/badge.svg)](https://codecov.io/gh/lincc-frameworks/tdastro)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/tdastro/asv-main.yml?label=benchmarks)](https://lincc-frameworks.github.io/tdastro/)
[![Read the Docs](https://img.shields.io/readthedocs/tdastro)](https://tdastro.readthedocs.io/)

## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```
>> conda create env -n <env_name> python=3.10
>> conda activate <env_name>
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> pip install -e .'[dev]'
>> pre-commit install
>> conda install pandoc
```

Notes:
1. The single quotes around `'[dev]'` may not be required for your operating system.
2. `pre-commit install` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on 
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)

## Acknowledgements

This project is supported by Schmidt Sciences.