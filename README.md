[![tests](https://github.com/canbooo/experimental-design/actions/workflows/tests.yml/badge.svg)](https://github.com/canbooo/experimental-design/actions/workflows/tests.yml)

# `experiment-design`: Tools to create and extend experiment plans

Documentation is under construction. You can install using

`pip install experiment-design`

See  [demos](./demos) for example usage.

## Create and extend Latin hypercube designs (LHD)

### LHD extension by doubling the sample size ([source](https://github.com/canbooo/experiment-design/blob/main/demos/lhs_sampling_extension_by_doubling.py))
![LHS extension by doubling](./media/lhs_extension_by_doubling.gif)

### LHD extension by adding a single sample at a time ([source](https://github.com/canbooo/experiment-design/blob/main/demos/lhs_sampling_extension_constant.py))
![LHS extension one by one](./media/lhs_extension_by_constant.gif)

### Extending an existing sampling locally following LHD rules ([source](https://github.com/canbooo/experiment-design/blob/main/demos/lhs_sampling_local_extension.py))
![Local LHS extension](./media/lhs_extension_local.gif)

## Orthogonal designs with any[^1] distribution ([source](https://github.com/canbooo/experiment-design/blob/main/demos/orthogonal_sampling_extension_by_doubling.py))
![OS extension by doubling](./media/os_extension_by_doubling.gif)

[^1]: As long as it is supported by `scipy.stats`

## Create and extend correlated designs ([source](https://github.com/canbooo/experiment-design/blob/main/demos/lhs_sampling_nonzero_correlation.py))

![Correlated LHS](./media/lhs_correlation.gif)
