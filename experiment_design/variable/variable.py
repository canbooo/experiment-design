from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np
from scipy.stats import rv_continuous, rv_discrete, uniform

# Following is ugly, but it is scipy's fault for not exposing rv_frozen
# noinspection PyProtectedMember
from scipy.stats._distn_infrastructure import rv_frozen


def _is_frozen_discrete(dist: Any) -> bool:
    """Check if dist is a rv_frozen_discrete instance"""
    return isinstance(dist, rv_frozen) and isinstance(dist.dist, rv_discrete)


def _is_frozen_continuous(dist: Any) -> bool:
    """Check if dist is a rv_frozen_continuous instance"""
    return isinstance(dist, rv_frozen) and isinstance(dist.dist, rv_continuous)


def _change_field_representation(
    dataclass_instance: dataclass, representations_to_change: dict[str, Any]
) -> str:
    """Just like the default __repr__ but supports reformatting and replacing some values."""
    final = []
    for current_field in dataclass_instance.__dataclass_fields__.values():
        if not current_field.repr:
            continue
        name = current_field.name
        value = representations_to_change.get(
            name, dataclass_instance.__getattribute__(name)
        )
        final.append(f"{name}={value}")
    return f"{dataclass_instance.__class__.__name__}({', '.join(final)})"


def _create_distribution_representation(distribution: rv_frozen) -> str:
    """Create a readable representation of rv_frozen instances"""
    args = ", ".join([str(a) for a in distribution.args])
    kwargs = ", ".join([f"{k}={v}" for k, v in distribution.kwds.items()])
    params = [a for a in [args, kwargs] if a]
    return f"{distribution.dist.name}({', '.join(params)})"


@dataclass
class ContinuousVariable:
    """
    A variable with continuous distribution

    :param distribution: rv_frozen instance representing the distribution. If None (default), it will be set to uniform
        between the passed lower_bound and upper_bound
    :param lower_bound: Lower bound for the variable. If None (default), left support boundary of the distribution will
        be used in case the distribution is bounded. Otherwise, distribution.ppf(infinite_bound_probability_tolerance)
        will be used.
    :param upper_bound: Upper bound for the variable. If None (default), right support boundary of the distribution will
        be used in case the distribution is bounded. Otherwise, distribution.ppf(1 - infinite_bound_probability_tolerance)
        will be used.
    """

    distribution: rv_frozen | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        if self.distribution is None and None in [self.lower_bound, self.upper_bound]:
            raise ValueError(
                "Either the distribution or both "
                "lower_bound and upper_bound have to be set."
            )
        if self.distribution is None:
            self.distribution = uniform(
                self.lower_bound, self.upper_bound - self.lower_bound
            )
        if (
            None not in [self.lower_bound, self.upper_bound]
            and self.lower_bound >= self.upper_bound
        ):
            raise ValueError("lower_bound has to be smaller than upper_bound")
        if not _is_frozen_continuous(self.distribution):
            raise ValueError("Only frozen continuous distributions are supported.")

    def value_of(self, probability: float | np.ndarray) -> float | np.ndarray:
        """Given a probability or an array of probabilities return the corresponding value(s) using the inverse |CDF|."""
        values = self.distribution.ppf(probability)
        if self.upper_bound is not None or self.lower_bound is not None:
            return np.clip(values, self.lower_bound, self.upper_bound)
        return values

    def cdf_of(self, value: float | np.ndarray) -> float | np.ndarray:
        """Given a value or an array of values return the probability using the |CDF|."""
        return self.distribution.cdf(value)

    def finite_lower_bound(
        self, infinite_bound_probability_tolerance: float = 1e-6
    ) -> float:
        """
        Provide a finite lower bound of the variable even if it was not provided by the user.

        :param infinite_bound_probability_tolerance: If the variable is unbounded and no explicit lower_bound was
            passed, this will be used to extract finite bounds as described in lower_bound and upper_bound descriptions.
            (Default: 1e-6)
        """
        if self.lower_bound is not None:
            return self.lower_bound
        value = self.value_of(0.0)
        if np.isfinite(value):
            return value
        return self.value_of(infinite_bound_probability_tolerance)

    def finite_upper_bound(
        self, infinite_bound_probability_tolerance: float = 1e-6
    ) -> float:
        """Provide a finite upper bound of the variable even if it was not provided by the user.

        :param infinite_bound_probability_tolerance: If the variable is unbounded and no explicit lower_bound was
            passed, this will be used to extract finite bounds as described in lower_bound and upper_bound descriptions.
            (Default: 1e-6)

        """
        if self.upper_bound is not None:
            return self.upper_bound
        value = self.value_of(1.0)
        if np.isfinite(value):
            return value
        return self.value_of(1 - infinite_bound_probability_tolerance)

    def __repr__(self) -> str:
        distribution_representation = _create_distribution_representation(
            self.distribution
        )
        return _change_field_representation(
            self, {"distribution": distribution_representation}
        )


@dataclass
class DiscreteVariable:
    """
    A variable with discrete distribution

    :param distribution: rv_frozen instance representing the distribution. If None (default), it will be set to uniform between
        the passed lower_bound and upper_bound
    :param value_mapper: Given an integer, i.e. an ordinal encoding, this is expected to return the corresponding
        discrete value of the underlying set of possible values. (Default: lambda x: x)
    :param inverse_value_mapper: Given a discrete value, this is expected to return the corresponding integer value,
        i.e. ordinal encoding. (Default: lambda x: x)
    """

    distribution: rv_frozen
    value_mapper: Callable[[float], float | int] = lambda x: x
    inverse_value_mapper: Callable[[float, int], float] = lambda x: x

    def __post_init__(self) -> None:
        if not _is_frozen_discrete(self.distribution):
            raise ValueError("Only frozen discrete distributions are supported.")
        self.value_mapper = np.vectorize(self.value_mapper)
        self.inverse_value_mapper = np.vectorize(self.inverse_value_mapper)

    def value_of(self, probability: float | np.ndarray) -> float | np.ndarray:
        """Given a probability or an array of probabilities return the corresponding value(s) using the inverse cdf."""
        values = self.distribution.ppf(probability)
        return self.value_mapper(values)

    def cdf_of(self, values: float | np.ndarray) -> float | np.ndarray:
        """Given a value or an array of values return the probability using the cdf."""
        return self.distribution.cdf(self.inverse_value_mapper(values))

    def finite_lower_bound(
        self, infinite_bound_probability_tolerance: float = 1e-6
    ) -> float:
        """
        Provide a finite lower bound of the variable even if it was not provided by the user.

        :param infinite_bound_probability_tolerance: If the variable is unbounded and no explicit lower_bound was
            passed, this will be used to extract finite bounds as described in lower_bound and upper_bound descriptions.
            (Default: 1e-6)
        """
        support = self.distribution.support()
        if np.isfinite(support[0]):
            return self.value_mapper(support[0])
        return self.value_of(infinite_bound_probability_tolerance)

    def finite_upper_bound(
        self, infinite_bound_probability_tolerance: float = 1e-6
    ) -> float:
        """
        Provide a finite upper bound of the variable even if it was not provided by the user.

        :param infinite_bound_probability_tolerance: If the variable is unbounded and no explicit lower_bound was
            passed, this will be used to extract finite bounds as described in lower_bound and upper_bound descriptions.
            (Default: 1e-6)
        """
        support = self.distribution.support()
        if np.isfinite(support[-1]):
            return self.value_mapper(support[1])
        return self.value_of(1 - infinite_bound_probability_tolerance)

    def __repr__(self) -> str:
        distribution_representation = _create_distribution_representation(
            self.distribution
        )
        return _change_field_representation(
            self, {"distribution": distribution_representation}
        )


class Variable(Protocol):
    """A protocol to represent the expected methods of valid Variable objects"""

    @property
    def distribution(self) -> rv_frozen:
        """Distribution of the variable"""

    def value_of(self, probability: float | np.ndarray) -> float | np.ndarray:
        """Given a probability or an array of probabilities return the corresponding value(s) using the inverse cdf."""

    def cdf_of(self, value: float | np.ndarray) -> float | np.ndarray:
        """Given a value or an array of values return the probability using the cdf."""

    def finite_lower_bound(
        self, infinite_bound_probability_tolerance: float = 1e-6
    ) -> float:
        """
        Provide a finite upper bound of the variable even if it was not provided by the user.

        :param infinite_bound_probability_tolerance: If the variable is unbounded and no explicit lower_bound was
            passed, this will be used to extract finite bounds as described in lower_bound and upper_bound descriptions.
            (Default: 1e-6)
        """

    def finite_upper_bound(
        self, infinite_bound_probability_tolerance: float = 1e-6
    ) -> float:
        """
        Provide a finite upper bound of the variable even if it was not provided by the user.

        :param infinite_bound_probability_tolerance: If the variable is unbounded and no explicit lower_bound was
            passed, this will be used to extract finite bounds as described in lower_bound and upper_bound descriptions.
            (Default: 1e-6)
        """


def create_variables_from_distributions(
    distributions: list[rv_frozen],
) -> list[ContinuousVariable | DiscreteVariable]:
    """
    Given a list of distributions, create the corresponding continuous or discrete variables.

    :param distributions: Frozen scipy distributions each representing a marginal variable
    :return: List of variables according to the passed distributions
    """
    variables = []
    for dist in distributions:
        if _is_frozen_discrete(dist):
            variables.append(DiscreteVariable(distribution=dist))
        elif _is_frozen_continuous(dist):
            variables.append(ContinuousVariable(distribution=dist))
        else:
            raise ValueError(
                f"Each distribution must be a frozen discrete or continuous type, got {type(dist)}"
            )
    return variables
