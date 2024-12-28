import numpy as np
from scipy.linalg import solve_triangular


def iman_connover_transformation(
    doe: np.ndarray,
    target_correlation: np.ndarray,
    means: np.ndarray | None = None,
    standard_deviations: np.ndarray | None = None,
) -> np.ndarray:
    """Rearrange the values of doe to reduce correlation error while keeping the Latin hypercube constraint"""
    # See Chapter 4.3.2 of
    # Local Latin Hypercube Refinement for Uncertainty Quantification and Optimization, Can Bogoclu, (2022)
    transformed = second_moment_transformation(
        doe, target_correlation, means, standard_deviations
    )
    order = np.argsort(np.argsort(transformed, axis=0), axis=0)
    return np.take_along_axis(np.sort(doe, axis=0), order, axis=0)


def second_moment_transformation(
    doe: np.ndarray,
    target_correlation: np.ndarray,
    means: np.ndarray | None = None,
    standard_deviations: np.ndarray | None = None,
) -> np.ndarray:
    """Second-moment transformation for achieving the target covariance"""
    if means is None:
        means = np.mean(doe, axis=0)
    if standard_deviations is None:
        standard_deviations = np.std(doe, axis=0, keepdims=True)
        standard_deviations = standard_deviations.reshape((1, -1))
    target_covariance = (
        standard_deviations.T.dot(standard_deviations) * target_correlation
    )
    target_cov_upper = np.linalg.cholesky(
        target_covariance
    ).T  # convert to covariance before Cholesky
    cur_cov_upper = np.linalg.cholesky(np.cov(doe, rowvar=False)).T
    inv_cov_upper = solve_triangular(cur_cov_upper, target_cov_upper)
    return (doe - means).dot(inv_cov_upper) + means
