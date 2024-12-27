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
from experiment_design.scorers import Scorer, create_correlation_matrix
from experiment_design.variable import DesignSpace, VariableCollection


class RandomSamplingDesigner(ExperimentDesigner):
    """Create or extend a design of experiments (DoE) by randomly sampling from the variable distributions.

    :param target_correlation: A float or a symmetric matrix with shape (len(variables), len(variables)), representing
    the linear dependency between the dimensions. If a float is passed, all non-diagonal entries of the unit matrix will
    be set to this value.
    :param exact_correlation: If True, the correlation matrix of the resulting design will match the target correlation
    exactly using a second moment transformation. This may lead variables with finite bounds to generate values that are
    out of bounds. Otherwise, Iman-Connover method will be used, where the values will be kept as is for each variable
    as they are generated from the marginal distribution. This may lead to some imprecision of the correlation matrix.

    """

    def __init__(
        self,
        target_correlation: np.ndarray | float = 0.0,
        exact_correlation: bool = False,
    ) -> None:
        self.target_correlation = target_correlation
        self.exact_correlation = exact_correlation
        super(RandomSamplingDesigner, self).__init__()

    def _create(
        self,
        variables: DesignSpace,
        sample_size: int,
        scorer: Scorer,
        initial_steps: int,
        final_steps: int,
        verbose: int,
    ) -> np.ndarray:
        steps = initial_steps + final_steps
        target_correlation = create_correlation_matrix(self.target_correlation)
        return random_search(
            creator=partial(
                sample_from,
                variables,
                sample_size,
                target_correlation,
                self.exact_correlation,
            ),
            scorer=scorer,
            steps=steps,
            verbose=verbose,
        )

    def _extend(
        self,
        old_sample: np.ndarray,
        variables: DesignSpace,
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
        target_correlation = create_correlation_matrix(self.target_correlation)
        return random_search(
            creator=partial(
                sample_from,
                variables,
                sample_size,
                target_correlation,
                self.exact_correlation,
            ),
            scorer=scorer,
            steps=steps,
            verbose=verbose,
        )


def sample_from(
    variables: VariableCollection,
    sample_size: int,
    target_correlation: np.ndarray | None = None,
    exact_correlation: bool = False,
) -> np.ndarray:
    """
    Sample from the distributions of the variables.

    :param variables: Determines the dimensions of the resulting sample.
    :param sample_size: the number of points to be created.
    :param target_correlation: A symmetric matrix with shape (len(variables), len(variables)), representing the linear
    :param exact_correlation: If True, second moment transformation will be used, which may not respect the finite
    bounds of the marginal distributions. Otherwise, Iman-Connover method will be used, which may yield imprecise
    correlation matrices.
    :return: Sample matrix with shape (len(variables), samples_size).
    """
    if not isinstance(variables, DesignSpace):
        variables = DesignSpace(variables)
        # Sometimes, we may randomly generate probabilities with
        # singular correlation matrices. Try 3 times to avoid issue until we give up
    error_text = ""
    transformer = (
        second_moment_transformation
        if exact_correlation
        else iman_connover_transformation
    )
    for k in range(3):
        doe = uniform(0, 1).rvs((sample_size, len(variables)))
        if not isinstance(variables, DesignSpace):
            variables = DesignSpace(variables)
        doe = variables.value_of(doe)
        if target_correlation is None:
            target_correlation = np.eye(len(variables))
        if (
            np.all(target_correlation == np.eye(len(variables)))
            and not exact_correlation
        ):
            return doe
        try:
            return transformer(doe, target_correlation)
        except np.linalg.LinAlgError as exc:
            error_text = str(exc)
            pass
    raise np.linalg.LinAlgError(error_text)
