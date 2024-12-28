from dataclasses import dataclass, field

import numpy as np

# Following is ugly, but it is scipy's fault for not exposing rv_frozen
# noinspection PyProtectedMember
from scipy.stats._distn_infrastructure import rv_frozen

from experiment_design import variable


@dataclass
class ParameterSpace:
    """A container of multiple variables defining a parameter space.

    :param variables: list of variables or marginal distributions that define the marginal parameters
    :param correlation: A float or asymmetric matrix with shape (len(variables), len(variables)), representing the
        linear dependency between the dimensions. If a float is passed, all non-diagonal entries of the unit matrix will
        be set to this value.
    """

    variables: (
        list[
            variable.Variable | variable.ContinuousVariable | variable.DiscreteVariable
        ]
        | list[rv_frozen]
    )
    correlation: float | np.ndarray | None = None
    _lower_bound: np.ndarray = field(init=False, repr=False, default=None)
    _upper_bound: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if isinstance(self.variables[0], rv_frozen):
            self.variables = variable.create_variables_from_distributions(
                self.variables
            )

        if self.correlation is None:
            self.correlation = 0
        self.correlation = create_correlation_matrix(self.correlation, self.dimensions)
        if not (
            self.correlation.shape[0] == self.correlation.shape[1] == self.dimensions
        ):
            raise ValueError(
                f"Inconsistent shapes: {self.dimensions} does not match "
                f"{self.correlation.shape}"
            )
        if np.max(np.abs(self.correlation)) > 1:
            raise ValueError("Correlations should be in the interval [-1,1].")

        lower, upper = [], []
        for var in self.variables:
            lower.append(var.finite_lower_bound)
            upper.append(var.finite_upper_bound)
        self._lower_bound = np.array(lower)
        self._upper_bound = np.array(upper)

    def _map_by(self, attribute: str, values: np.ndarray) -> np.ndarray:
        if len(values.shape) != 2:
            values = values.reshape((-1, self.dimensions))
        results = np.zeros(values.shape)
        for i_dim, design_variable in enumerate(self.variables):
            results[:, i_dim] = getattr(design_variable, attribute)(values[:, i_dim])
        return results

    def value_of(self, probabilities: np.ndarray) -> np.ndarray:
        """Given an array of probabilities return the corresponding values using the inverse marginal cdf.
        Ignores correlation!
        """
        return self._map_by("value_of", probabilities)

    def cdf_of(self, values: np.ndarray) -> np.ndarray:
        """Given an array of values return the marginal probability using the cdf.
        Ignores correlation!
        """
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


VariableCollection = list[rv_frozen] | list[variable.Variable] | ParameterSpace


def create_correlation_matrix(
    target_correlation: float | np.ndarray = 0.0,
    num_variables: int | None = None,
) -> np.ndarray:
    """Create a correlation matrix from the target correlation in case it is a float"""
    if not np.isscalar(target_correlation):
        return target_correlation
    if not num_variables:
        raise ValueError(
            "num_variables have to be passed if the target_correlation is a scalar."
        )
    return (
        np.eye(num_variables) * (1 - target_correlation)
        + np.ones((num_variables, num_variables)) * target_correlation
    )
