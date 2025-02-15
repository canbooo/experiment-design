import abc

import diversipy
import numpy as np
import pyDOE
import pyDOE2
import pyDOE3
import pyLHD
from doepy import build as doepy_build

from experiment_design import ParameterSpace


class ExternalDesignerAdapter(abc.ABC):
    """
    The adapter is used to integrate external tools to the benchmark. Specifically, a uniform LHS within the bounds
    [0,1]^d is generated with any tool that does not natively support orthogonal sampling. The values of this LHS is
    interpreted as probabilities and mapped to the original ParameterSpace using the inverse of the CDF function,
    also known as the inverse transform sampling. Even if the external algorithm support having non-unit bounds,
    our usage of inverse transform sampling is justified as long as we use identical bounds for every variable since
    the optimal setting for [0, 1]^d will be the same as in [a, b]^d in the uniform space.
    """

    @abc.abstractmethod
    def _generate_probabilities(self, dimensions: int, sample_size: int) -> np.ndarray:
        raise NotImplementedError

    def design(self, space: ParameterSpace, sample_size: int) -> np.ndarray:
        probabilities = self._generate_probabilities(space.dimensions, sample_size)
        return space.value_of(probabilities)


class PyDOEAdapter(ExternalDesignerAdapter):
    def _generate_probabilities(self, dimensions: int, sample_size: int) -> np.ndarray:
        return pyDOE.lhs(dimensions, sample_size, criterion="maximin")


class PyDOE2Adapter(ExternalDesignerAdapter):
    def _generate_probabilities(self, dimensions: int, sample_size: int) -> np.ndarray:
        return pyDOE2.lhs(dimensions, sample_size, criterion="corr")


class PyDOE3Adapter(ExternalDesignerAdapter):
    def _generate_probabilities(self, dimensions: int, sample_size: int) -> np.ndarray:
        return pyDOE3.lhs(dimensions, sample_size, criterion="lhsmu")


class DiversiPyAdapter(ExternalDesignerAdapter):
    def _generate_probabilities(self, dimensions: int, sample_size: int) -> np.ndarray:
        ids = diversipy.cube.improved_latin_design(sample_size, dimensions)
        return (ids + 0.5) / sample_size  # central design


class DOEPyAdapter(ExternalDesignerAdapter):
    def _generate_probabilities(self, dimensions: int, sample_size: int) -> np.ndarray:
        space = {f"x{k}": [0.0, 1.0] for k in range(dimensions)}
        # DOEPy seems to have at least two problems:
        # 1. build.space_filling_lhs errors out due to unresolved reference transform_spread_out
        # 2. lhs accepts probability distribution but does not do anything with them
        return doepy_build.lhs(space, num_samples=sample_size).values


class PyLHDAdapter(ExternalDesignerAdapter):
    def _generate_probabilities(self, dimensions: int, sample_size: int) -> np.ndarray:
        ids = pyLHD.maximinLHD((sample_size, dimensions))
        return (ids + 0.5) / sample_size  # central design
