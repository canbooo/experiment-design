from dataclasses import dataclass, field

import numpy as np

# Following is ugly, but it is scipy's fault for not exposing rv_frozen
# noinspection PyProtectedMember
from scipy.stats._distn_infrastructure import rv_frozen

from experiment_design import variable


@dataclass
class DesignSpace:
    """A container of multiple variables defining a design space."""

    variables: (
        list[
            variable.Variable | variable.ContinuousVariable | variable.DiscreteVariable
        ]
        | list[rv_frozen]
    )
    _lower_bound: np.ndarray = field(init=False, repr=False, default=None)
    _upper_bound: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if isinstance(self.variables[0], rv_frozen):
            self.variables = variable.create_variables_from_distributions(
                self.variables
            )
        lower, upper = [], []
        for var in self.variables:
            lower.append(var.finite_lower_bound)
            upper.append(var.finite_upper_bound)
        self._lower_bound = np.array(lower)
        self._upper_bound = np.array(upper)

    def _map_by(self, attribute: str, values: np.ndarray) -> np.ndarray:
        if len(values.shape) != 2:
            values = values.reshape((-1, len(self.variables)))
        results = np.zeros(values.shape)
        for i_dim, design_variable in enumerate(self.variables):
            results[:, i_dim] = getattr(design_variable, attribute)(values[:, i_dim])
        return results

    def value_of(self, probabilities: np.ndarray) -> np.ndarray:
        """Given an array of probabilities return the corresponding values using the inverse cdf."""
        return self._map_by("value_of", probabilities)

    def cdf_of(self, values: np.ndarray) -> np.ndarray:
        """Given an array of values return the probability using the cdf."""
        return self._map_by("cdf_of", values)

    @property
    def lower_bound(self) -> np.ndarray:
        """Finite lower bound of the space."""
        return self._lower_bound

    @property
    def upper_bound(self) -> np.ndarray:
        """Finite upper bound of the space."""
        return self._upper_bound

    @property
    def dimensions(self) -> int:
        """Size of the space, i.e. the number of dimensions."""
        return len(self.variables)

    def __len__(self):
        """Size of the space, i.e. the number of dimensions."""
        return self.dimensions


VariableCollection = list[variable.Variable] | DesignSpace
