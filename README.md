![PyPI - Version](https://img.shields.io/pypi/v/experiment-design)
[![tests](https://github.com/canbooo/experimental-design/actions/workflows/tests.yml/badge.svg)](https://github.com/canbooo/experimental-design/actions/workflows/tests.yml)
![coverage-badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/canbooo/6e8333ea5cf30b2e8c83d55eee51525b/raw/coverage_token.json)
# `experiment-design`: Tools to create and extend experiment plans

Documentation is under construction. You can install using

`pip install experiment-design`

See  [demos](./demos) for example usage.

## Create and extend Latin hypercube designs (LHD)

### LHD extension by doubling the sample size ([source](https://github.com/canbooo/experiment-design/blob/main/demos/lhs_sampling_extension_by_doubling.py))
![LHS extension by doubling](docs/source/images/lhs_extension_by_doubling.gif)

### LHD extension by adding a single sample at a time ([source](https://github.com/canbooo/experiment-design/blob/main/demos/lhs_sampling_extension_constant.py))
![LHS extension one by one](docs/source/images/lhs_extension_by_constant.gif)

### Extending an existing sampling locally following LHD rules ([source](https://github.com/canbooo/experiment-design/blob/main/demos/lhs_sampling_local_extension.py))
![Local LHS extension](docs/source/images/lhs_extension_local.gif)

## Orthogonal designs with any[^1] distribution ([source](https://github.com/canbooo/experiment-design/blob/main/demos/orthogonal_sampling_extension_by_doubling.py))
![OS extension by doubling](docs/source/images/os_extension_by_doubling.gif)

[^1]: As long as it is supported by `scipy.stats`

## Create and extend correlated designs ([source](https://github.com/canbooo/experiment-design/blob/main/demos/lhs_sampling_nonzero_correlation.py))

![Correlated LHS](docs/source/images/lhs_correlation.gif)
