from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import randint

# Following is ugly, but it is scipy's fault for not exposing rv_frozen
# noinspection PyProtectedMember
from scipy.stats._distn_infrastructure import rv_frozen

from experiment_design import variable
from experiment_design.variable import ContinuousVariable, DiscreteVariable


@dataclass
class ParameterSpace:
    """A container of multiple variables defining a parameter space.

    :param variables: List of variables or marginal distributions that define the marginal parameters
    :param correlation: A float or asymmetric matrix with shape (len(variables), len(variables)), representing the
        linear dependency between the dimensions. If a float is passed, all non-diagonal entries of the unit matrix will
        be set to this value.
    :param infinite_bound_probability_tolerance: If the variable is unbounded, this will be used to extract finite bounds
        as described in lower_bound and upper_bound descriptions. (Default: 1e-6)
    """

    variables: (
        list[
            variable.Variable | variable.ContinuousVariable | variable.DiscreteVariable
        ]
        | list[rv_frozen]
    )
    correlation: float | np.ndarray | None = None
    infinite_bound_probability_tolerance: float = 1e-6

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
            lower.append(
                var.finite_lower_bound(self.infinite_bound_probability_tolerance)
            )
            upper.append(
                var.finite_upper_bound(self.infinite_bound_probability_tolerance)
            )
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
        """
        Given an array of marginal probabilities, return the corresponding values with shape probabilities.shape using
        the inverse marginal CDF.

        Since it operates on the marginal variables, correlation does not have an effect.
        """
        return self._map_by("value_of", probabilities)

    def cdf_of(self, values: np.ndarray) -> np.ndarray:
        """
        Given an array of marginal values return the marginal probabilities with shape values.shape using the CDF.

        Since it operates on the marginal variables, correlation does not have an effect.
        """
        return self._map_by("cdf_of", values)

    @property
    def lower_bound(self) -> np.ndarray:
        """
        Lower bound values of the space with shape (self.dimensions, ).

        variable.finite_lower_bound is used to provide finite bounds even for unbounded variables
        """
        return self._lower_bound

    @property
    def upper_bound(self) -> np.ndarray:
        """
        Upper bound of the space with shape (self.dimensions, ).

        variable.finite_upper_bound is used to provide finite bounds even for unbounded variables
        """
        return self._upper_bound

    @property
    def dimensions(self) -> int:
        """Size of the space, i.e. the number of variables."""
        return len(self.variables)


VariableCollection = list[rv_frozen] | list[variable.Variable] | ParameterSpace


def create_correlation_matrix(
    target_correlation: float | np.ndarray = 0.0,
    num_variables: int | None = None,
) -> np.ndarray:
    """
    Create a correlation matrix from the target correlation in case it is a float

    :meta private:
    """
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


def create_discrete_uniform_space(
    discrete_sets: list[list[int | float]],
) -> ParameterSpace:
    """
    Given sets of possible values, create corresponding discrete variables with equal probability of each value.

    :param discrete_sets: List of possible values for each variable
    :return: Parameter space consisting of discrete uniform variables with the same size as the discrete_sets
    """
    variables = []
    for discrete_set in discrete_sets:
        n_values = len(discrete_set)
        if n_values < 2:
            raise ValueError("At least two values are required for discrete variables")
        # In the following, it is OK and even advantageous to have a mutable
        # default argument as a very rare occasion. Therefore, we disable inspection.
        # noinspection PyDefaultArgument
        variables.append(
            DiscreteVariable(
                distribution=randint(0, n_values),
                # Don't forget to bind the discrete_set below either by
                # defining a kwarg as done here, or by generating in another
                # scope, e.g. function. Otherwise, the last value of discrete_sets
                # i.e. the last entry of discrete_sets will be used for all converters
                # Check https://stackoverflow.com/questions/19837486/lambda-in-a-loop
                # for a description as this is expected python behaviour.
                value_mapper=lambda x, values=sorted(discrete_set): values[int(x)],
                inverse_value_mapper=lambda x,
                values=sorted(discrete_set): values.index(x),
            )
        )
    return ParameterSpace(variables)


def create_continuous_uniform_space(
    lower_bounds: Sequence[float], upper_bounds: Sequence[float]
) -> ParameterSpace:
    """
    Given lower and upper bounds, create uniformly distributed variables.

    :param lower_bounds: Array with shape (n_dim,) representing the lower bounds of the uniform variables
    :param upper_bounds: Array with shape (n_dim,) representing the upper bounds of the uniform variables
    :return: Parameter space consisting of continuous uniform variables with the same size as the passed bounds
    """
    if len(lower_bounds) != len(upper_bounds):
        raise ValueError(
            "Number of lower bounds has to be equal to the number of upper bounds"
        )
    variables = []
    for lower, upper in zip(lower_bounds, upper_bounds):
        variables.append(ContinuousVariable(lower_bound=lower, upper_bound=upper))
    return ParameterSpace(variables)
