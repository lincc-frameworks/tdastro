# TDAstro

Time-Domain Forward-Modeling for the Rubin Era

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/tdastro?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/tdastro/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/tdastro/smoke-test.yml)](https://github.com/lincc-frameworks/tdastro/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/lincc-frameworks/tdastro/branch/main/graph/badge.svg)](https://codecov.io/gh/lincc-frameworks/tdastro)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/tdastro/asv-main.yml?label=benchmarks)](https://lincc-frameworks.github.io/tdastro/)
[![Read the Docs](https://img.shields.io/readthedocs/tdastro)](https://tdastro.readthedocs.io/)


## Introduction

Realistic light curve simulations are essential to many time-domain problems. 
Simulations are needed to evaluate observing strategy, characterize biases, 
and test pipelines. TDAstro aims to provide a flexible, scalable, and user-friendly
time-domain simulation software with realistic effects and survey strategies.

The software package consists of multiple stages:
1. A flexible framework for consistently sampling model parameters (and hyper-parameters),
2. Realistic models of time varying phenomena (such as supernovae and AGNs),
3. Effect models (such as dust extinction), and
4. Survey characteristics (such as cadence, filters, and noise).

For an overview of the package, we recommend starting with introduction notebook 
(at `notebooks/introduction.ipynb`).


## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment such as `venv`

```
>> python3 -m venv ~/envs/tdastro
>> source ~/envs/tdastro/bin/activate
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> pip install -e .'[dev]'
>> pre-commit install
```

Notes:
1. The single quotes around `'[dev]'` may not be required for your operating system.
2. `pre-commit install` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on 
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)

## Acknowledgements

This project is supported by Schmidt Sciences.