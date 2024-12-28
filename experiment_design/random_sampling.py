import logging
from functools import partial

import numpy as np
from scipy.stats import uniform

from experiment_design.covariance_modification import (
    iman_connover_transformation,
    second_moment_transformation,
)
from experiment_design.experiment_designer import ExperimentDesigner
from experiment_design.optimize import random_search
from experiment_design.scorers import Scorer
from experiment_design.variable import ParameterSpace, VariableCollection
from experiment_design.variable.space import create_correlation_matrix


class RandomSamplingDesigner(ExperimentDesigner):
    """Create or extend a design of experiments (DoE) by randomly sampling from the variable distributions.

    :param exact_correlation: If True, the correlation matrix of the resulting design will match the target correlation
    exactly using a second moment transformation. This may lead variables with finite bounds to generate values that are
    out of bounds. Otherwise, Iman-Connover method will be used, where the values will be kept as is for each variable
    as they are generated from the marginal distribution. This may lead to some imprecision of the correlation matrix.

    """

    def __init__(
        self,
        exact_correlation: bool = False,
    ) -> None:
        self.exact_correlation = exact_correlation
        super(RandomSamplingDesigner, self).__init__()

    def _create(
        self,
        space: ParameterSpace,
        sample_size: int,
        scorer: Scorer,
        initial_steps: int,
        final_steps: int,
        verbose: int,
    ) -> np.ndarray:
        steps = initial_steps + final_steps
        return random_search(
            creator=partial(
                sample_from,
                space,
                sample_size,
                self.exact_correlation,
            ),
            scorer=scorer,
            steps=steps,
            verbose=verbose,
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
        steps = initial_steps + final_steps
        logging.warning(
            "If the design space changes, "
            "random sampling may not handle correlation modification properly!"
        )
        return random_search(
            creator=partial(
                sample_from,
                space,
                sample_size,
                self.exact_correlation,
            ),
            scorer=scorer,
            steps=steps,
            verbose=verbose,
        )


def sample_from(
    space: ParameterSpace,
    sample_size: int,
    exact_correlation: bool = False,
) -> np.ndarray:
    """
    Sample from the distributions of the variables.

    :param space: Determines the dimensions of the resulting sample.
    :param sample_size: the number of points to be created.
    :param exact_correlation: If True, second moment transformation will be used, which may not respect the finite
    bounds of the marginal distributions. Otherwise, Iman-Connover method will be used, which may yield imprecise
    correlation matrices.
    :return: Sample matrix with shape (len(variables), samples_size).
    """

    # Sometimes, we may randomly generate probabilities with
    # singular correlation matrices. Try 3 times to avoid issue until we give up
    error_text = ""
    transformer = (
        second_moment_transformation
        if exact_correlation
        else iman_connover_transformation
    )
    for k in range(3):
        doe = uniform(0, 1).rvs((sample_size, len(space)))

        doe = space.value_of(doe)

        if (
            np.isclose(space.correlation == np.eye(len(space)))
            and not exact_correlation
        ):
            return doe
        try:
            return transformer(doe, space.correlation)
        except np.linalg.LinAlgError as exc:
            error_text = str(exc)
            pass
    raise np.linalg.LinAlgError(error_text)
