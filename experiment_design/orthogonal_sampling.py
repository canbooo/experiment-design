from functools import partial
from typing import Callable

import numpy as np
from scipy.stats import uniform

from experiment_design.covariance_modification import iman_connover_transformation
from experiment_design.experiment_designer import ExperimentDesigner
from experiment_design.optimize import (
    random_search,
    simulated_annealing_by_perturbation,
)
from experiment_design.scorers import Scorer, ScorerFactory, select_local
from experiment_design.variable import ParameterSpace


class OrthogonalSamplingDesigner(ExperimentDesigner):
    """
    Create or extend an orthogonal sampling design. Orthogonal sampling design partitions the design space into
    bins of equal marginal probability and places samples such that each bin is only filled once for each dimension.
    If all variables are uniform, orthogonal sampling becomes a Latin hypercube sampling.

    :param inter_bin_randomness: Controls the randomness of placed points between the bin bounds. Specifically, 0. means that
        the points are placed at the center of each bin, whereas 1. leads to a random point placement within the bounds.
        Any other fractions leads to a random placement within that fraction of the bin bounds in each dimension.

    :param non_occupied_bins: Only relevant for extending the design, i.e. if old points are provided, and if the constraint
        regarding the number of occupation of each bin has to be violated. True means that each bin is occupied at least
        once for each dimension, although some bins might be occupied more often. Otherwise, each bin is occupied once
        or less often, leading to empty bins in some cases.
    :param scorer_factory: A factory that creates scorers for the given variables, sample_size and in the cast of an
        extension, old sampling points. If not passed, a default one will be created, that evaluates the maximum
        correlation error and minimum pairwise distance. See `experiment_design.scorers.create_default_scorer_factory`
        for more details.
    """

    def __init__(
        self,
        inter_bin_randomness: float = 1.0,
        non_occupied_bins: bool = False,
        scorer_factory: ScorerFactory | None = None,
    ) -> None:
        self.inter_bin_randomness = inter_bin_randomness
        if non_occupied_bins:
            self.empty_size_check = np.min
        else:
            self.empty_size_check = np.max
        super(OrthogonalSamplingDesigner, self).__init__(scorer_factory=scorer_factory)

    def _create(
        self,
        space: ParameterSpace,
        sample_size: int,
        scorer: Scorer,
        initial_steps: int,
        final_steps: int,
        verbose: int,
    ) -> np.ndarray:
        if (initial_steps + final_steps) <= 2:
            # Enable faster use cases:
            return create_orthogonal_design(
                space=space,
                sample_size=sample_size,
                inter_bin_randomness=self.inter_bin_randomness,
            )

        if verbose:
            print("Creating an initial design")
        doe = random_search(
            creator=partial(
                create_orthogonal_design,
                space=space,
                sample_size=sample_size,
                inter_bin_randomness=self.inter_bin_randomness,
            ),
            scorer=scorer,
            steps=initial_steps,
            verbose=verbose,
        )
        if verbose:
            print("Optimizing the initial design")
        return simulated_annealing_by_perturbation(
            doe, scorer, steps=final_steps, verbose=verbose
        )

    def _extend(
        self,
        old_sample: np.ndarray,
        space: ParameterSpace,
        sample_size: int,
        scorer: Scorer,
        initial_steps: int,
        final_steps: int,
        verbose: int,
    ) -> np.ndarray:
        local_doe = select_local(old_sample, space)
        probabilities = space.cdf_of(local_doe)
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError(
                "Non-finite probability encountered. Please check the distributions."
            )

        bins_per_dimension = sample_size + local_doe.shape[0]

        empty = _find_sufficient_empty_bins(
            probabilities,
            bins_per_dimension,
            sample_size,
            empty_size_check=self.empty_size_check,
        )
        if verbose:
            print("Creating candidate points to extend the design")
        new_sample = random_search(
            creator=partial(
                _create_candidates_from,
                empty_bins=empty,
                variables=space,
                sample_size=sample_size,
                inter_bin_randomness=self.inter_bin_randomness,
            ),
            scorer=scorer,
            steps=initial_steps,
            verbose=verbose,
        )
        if verbose:
            print("Optimizing candidate points to extend the design")
        return simulated_annealing_by_perturbation(
            new_sample, scorer, steps=final_steps
        )


def create_orthogonal_design(
    space: ParameterSpace,
    sample_size: int,
    inter_bin_randomness: float = 1.0,
) -> np.ndarray:
    """Create an orthogonal design without any optimization."""
    # Sometimes, we may randomly generate probabilities with
    # singular correlation matrices. Try 3 times to avoid issue until we give up
    error_text = ""
    for k in range(3):
        probabilities = create_lhd_probabilities(
            len(space), sample_size, inter_bin_randomness=inter_bin_randomness
        )
        doe = space.value_of(probabilities)
        try:
            return iman_connover_transformation(doe, space.correlation)
        except np.linalg.LinAlgError as exc:
            error_text = str(exc)
            pass
    raise np.linalg.LinAlgError(error_text)


def create_lhd_probabilities(
    num_variables: int,
    sample_size: int,
    inter_bin_randomness: float = 1.0,
) -> np.ndarray:
    """Create probabilities for a Latin hypercube design."""
    if not 0.0 <= inter_bin_randomness <= 1.0:
        raise ValueError(
            f"inter_bin_randomness has to be between 0 and 1, got {inter_bin_randomness}"
        )
    doe = uniform.rvs(size=(sample_size, num_variables))
    doe = (np.argsort(doe, axis=0) + 0.5) / sample_size
    if inter_bin_randomness == 0.0:
        return doe
    delta = inter_bin_randomness / sample_size
    return doe + uniform(-delta / 2, delta).rvs(size=(sample_size, num_variables))


def _find_sufficient_empty_bins(
    probabilities: np.ndarray,
    bins_per_dimension: int,
    required_sample_size: int,
    empty_size_check: Callable[[np.ndarray], float] = np.max,
) -> np.ndarray:
    empty = _find_empty_bins(probabilities, bins_per_dimension)
    cols = np.where(empty)[1]
    while (
        empty_size_check(np.unique(cols, return_counts=True)[1]) < required_sample_size
    ):
        bins_per_dimension += 1
        empty = _find_empty_bins(probabilities, bins_per_dimension)
        cols = np.where(empty)[1]
    return empty


def _find_empty_bins(probabilities: np.ndarray, bins_per_dimension: int) -> np.ndarray:
    """
    Find empty bins on an orthogonal sampling grid given the probabilities.

    :param probabilities: Array of cdf values of the observed points.
    :param bins_per_dimension: Determines the size of the grid to be tested.
    :return: Boolean array of empty bins with shape=(n_bins, n_dims).
    """
    empty_bins = np.ones((bins_per_dimension, probabilities.shape[1]), dtype=bool)
    edges = np.arange(bins_per_dimension + 1) / bins_per_dimension
    edges = edges.reshape((-1, 1))
    for i_dim in range(probabilities.shape[1]):
        condition = np.logical_and(
            probabilities[:, i_dim] >= edges[:-1], probabilities[:, i_dim] < edges[1:]
        )
        empty_bins[:, i_dim] = np.logical_not(condition.any(1))
    return empty_bins


def _create_candidates_from(
    empty_bins: np.ndarray,
    variables: ParameterSpace,
    sample_size: int,
    inter_bin_randomness: float = 1.0,
) -> np.ndarray:
    if not 0.0 <= inter_bin_randomness <= 1.0:
        raise ValueError(
            f"inter_bin_randomness has to be between 0 and 1, got {inter_bin_randomness}"
        )
    empty_rows, empty_cols = np.where(empty_bins)
    bins_per_dimension, dimensions = empty_bins.shape
    delta = 1 / bins_per_dimension
    probabilities = np.empty((sample_size, dimensions))
    for i_dim in range(dimensions):
        values = empty_rows[empty_cols == i_dim]
        np.random.shuffle(values)
        diff = sample_size - values.size
        if diff < 0:
            values = values[:sample_size]
        elif diff > 0:
            available = [idx for idx in range(bins_per_dimension) if idx not in values]
            extra = np.random.choice(available, diff, replace=False)
            values = np.append(extra, values)
        probabilities[:, i_dim] = values * delta + delta / 2
    if inter_bin_randomness > 0.0:
        delta *= inter_bin_randomness
        probabilities += uniform(-delta / 2, delta).rvs(size=(sample_size, dimensions))
    return variables.value_of(probabilities)
