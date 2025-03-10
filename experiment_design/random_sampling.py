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
from experiment_design.scorers import Scorer, ScorerFactory
from experiment_design.variable import ParameterSpace


class RandomSamplingDesigner(ExperimentDesigner):
    """
    Create or extend a |DoE| by randomly sampling from the variable distributions.

    :param exact_correlation: If True, the correlation matrix of the resulting design will match the target correlation
        exactly using a second moment transformation. This may lead variables with finite bounds to generate values that
        are out of bounds. Otherwise, Iman-Connover method will be used, where the values will be kept as is for each
        variable as they are generated from the marginal distribution. This may lead to some imprecision of the
        correlation matrix.
    :param scorer_factory: A factory that creates scorers for the given variables, sample_size and in the cast of an
        extension, old sampling points. If not passed, a default one will be created, that evaluates the maximum
        correlation error and minimum pairwise distance.See
        `experiment_design.scorers.create_default_scorer_factory <#experiment_design.scorers.create_default_scorer_factory>`_
        for more details.

    Examples
    --------
    >>> from experiment_design import create_continuous_uniform_space, RandomSamplingDesigner
    >>> space = create_continuous_uniform_space([-2., -2.], [2., 2.])
    >>> designer = RandomSamplingDesigner()
    >>> doe1 = designer.design(space, sample_size=20)
    >>> doe1.shape
    (20, 2)
    >>> doe2 = designer.design(space, sample_Size=4, old_sample=doe1)
    >>> doe2.shape
    (4, 2)
    """

    def __init__(
        self,
        exact_correlation: bool = False,
        scorer_factory: ScorerFactory | None = None,
    ) -> None:
        self.exact_correlation = exact_correlation
        super(RandomSamplingDesigner, self).__init__(scorer_factory=scorer_factory)

    def _create(
        self,
        space: ParameterSpace,
        sample_size: int,
        scorer: Scorer,
        initial_steps: int,
        final_steps: int,
    ) -> np.ndarray:
        steps = initial_steps + final_steps
        if steps <= 1:
            return sample_from(space, sample_size, self.exact_correlation)
        return random_search(
            creator=partial(
                sample_from,
                space,
                sample_size,
                self.exact_correlation,
            ),
            scorer=scorer,
            steps=steps,
        )

    def _extend(
        self,
        old_sample: np.ndarray,
        space: ParameterSpace,
        sample_size: int,
        scorer: Scorer,
        initial_steps: int,
        final_steps: int,
    ) -> np.ndarray:
        steps = initial_steps + final_steps
        if steps <= 1:
            return sample_from(space, sample_size, self.exact_correlation)
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
        )


def sample_from(
    space: ParameterSpace,
    sample_size: int,
    exact_correlation: bool = False,
) -> np.ndarray:
    """
    Sample from the distributions of the variables.

    :param space: Determines the dimensions of the resulting sample.
    :param sample_size: The number of points to be created.
    :param exact_correlation: If True, second moment transformation will be used, which may not respect the finite
    bounds of the marginal distributions. Otherwise, Iman-Connover method will be used, which may yield imprecise
    correlation matrices.
    :return: Sample matrix with shape (len(variables), samples_size).

    :meta private:
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
        doe = uniform(0, 1).rvs((sample_size, space.dimensions))

        doe = space.value_of(doe)

        if (
            np.isclose(space.correlation, np.eye(space.dimensions)).all()
            and not exact_correlation
        ):
            return doe
        try:
            return transformer(doe, space.correlation)
        except np.linalg.LinAlgError as exc:
            error_text = str(exc)
            pass
    raise np.linalg.LinAlgError(error_text)
