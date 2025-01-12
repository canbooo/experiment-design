![PyPI - Python Version](https://img.shields.io/pypi/pyversions/experiment-design)
![PyPI - Version](https://img.shields.io/pypi/v/experiment-design)
[![tests](https://github.com/canbooo/experimental-design/actions/workflows/tests.yml/badge.svg)](https://github.com/canbooo/experimental-design/actions/workflows/tests.yml)
[![codecov](https://codecov.io/github/canbooo/experiment-design/graph/badge.svg?token=S5XHYYL1U9)](https://codecov.io/github/canbooo/experiment-design)
![Code style ruff](https://img.shields.io/badge/style-ruff-41B5BE?style=flat)
# `experiment-design`: Tools to create and extend experiment plans

`experiment-design` allows you to create high quality designs of experiment with just a few lines
of code. Additionally, it allows you to extend the designs of experiments...
<p float="left">
    <img src="docs/source/images/lhs_extension_by_doubling.gif" alt="Image: Latin hypercube sampling extension by doubling" width="200">
    <img src="docs/source/images/lhs_extension_by_constant.gif" alt="Image: Latin hypercube sampling extension using one sample at a time" width="200">
    <img src="docs/source/images/lhs_extension_local.gif" alt="Image: Local Latin hypercube extension" width="200">
</p>

... create and optimize orthogonal sampling designs with any[^1] distribution

<img src="docs/source/images/os_extension_by_doubling.gif" alt="Image: Orthogonal sampling creation and extension with any distribution" width="200">

...and easily simulate correlated variables.

<img src="docs/source/images/lhs_correlation.gif" alt="Image: Latin hypercube sampling with correlated variables" width="200">

There is even more! See the [documentation](https://experiment-design.readthedocs.io) for more details and
especially the section "[Why should you use `experiment-design`?](https://experiment-design.readthedocs.io/en/latest/orthogonal_sampling.html#why-should-you-use-experiment-design)"

Also, see [demos](./demos) to understand how the images above were created.

[^1]: As long as it is supported by `scipy.stats`

## Install
`experiment-design` can be installed easily from PyPI using

`pip install experiment-design`

## Cite

You can use the zenodo DOI to cite the code, but I would appreciate you citing either of the following
publications to cite the methods:

- Journal paper about locally extending experiment designs for adaptive sampling:
```latex
@Article{Bogoclu2021,
  title       = {Local {L}atin hypercube refinement for multi-objective design uncertainty optimization},
  author      = {Can Bogoclu and Tamara Nestorovi{\'c} and Dirk Roos},
  journal     = {Applied Soft Computing},
  year        = {2021},
  arxiv       = {2108.08890},
  doi         = {10.1016/j.asoc.2021.107807},
  pdf         = {https://www.sciencedirect.com/science/article/abs/pii/S1568494621007286},
}
```
- PhD thesis:
```latex
@phdthesis{Bogoclu2022,
  title       = {Local {L}atin hypercube refinement for uncertainty quantification and optimization: {A}ccelerating the surrogate-based solutions using adaptive sampling},
  author      = {Bogoclu, Can},
  school      = {Ruhr-Universit\"{a}t Bochum},
  type         = {PhD thesis},
  year        = {2022},
  doi         = {10.13154/294-9143},
  pdf         = {https://d-nb.info/1268193348/34},
}
```
