## Motivation

Trying to understand how to de-embed 2-port fixtures from a network.

## Details

This repository contains Python functions for cascading and de-embedding networks using the scikit-rf library. The main functions provided are:

- `cascade`: Cascade a multi-port DUT (device under test) network with fixtures connected to each port.
- `deembed`: De-embed a multi-port network from a set of fixtures connected to each port.
- `plot_networks`: Plot the magnitude of S-parameters for a list of networks.

## Installation

This repo is using poetry for dependencies management.

1. [install poetry](https://python-poetry.org/docs/#installation)
2. run `poetry install`
3. run the code with `python3 main.py`
