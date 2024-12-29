import numpy as np
from scipy.linalg import solve_triangular


def iman_connover_transformation(
    doe: np.ndarray,
    target_correlation: np.ndarray,
    means: np.ndarray | None = None,
    standard_deviations: np.ndarray | None = None,
) -> np.ndarray:
    """Rearrange the values of doe to reduce correlation error while keeping the Latin hypercube constraint

    R. L. Iman and W. J. Conover (1982). “A distribution-free approach to inducing rank correlation among input
    variables”

    C. Bogoclu (2022). "Local Latin Hypercube Refinement for Uncertainty Quantification and Optimization" Chapter 4.3.2
    https://hss-opus.ub.ruhr-uni-bochum.de/opus4/frontdoor/deliver/index/docId/9143/file/diss.pdf
    """
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
    jitter: float = 1e-6,
) -> np.ndarray:
    """Second-moment transformation for achieving the target covariance"""
    if means is None:
        means = np.mean(doe, axis=0)
    if standard_deviations is None:
        standard_deviations = np.std(doe, axis=0, keepdims=True)
        standard_deviations = standard_deviations.reshape((1, -1))
    target_covariance = (
        standard_deviations.T.dot(standard_deviations) * target_correlation
    )  # convert to covariance before Cholesky
    try:
        target_cov_upper = np.linalg.cholesky(target_covariance).T
    except np.linalg.LinAlgError:
        target_cov_upper = np.linalg.cholesky(
            target_covariance + np.eye(target_covariance.shape[0]) * jitter
        ).T
    cur_cov = np.cov(doe, rowvar=False)
    try:
        cur_cov_upper = np.linalg.cholesky(cur_cov).T
    except np.linalg.LinAlgError:
        cur_cov_upper = np.linalg.cholesky(
            cur_cov + np.eye(cur_cov.shape[0]) * jitter
        ).T

    inv_cov_upper = solve_triangular(cur_cov_upper, target_cov_upper)
    return (doe - means).dot(inv_cov_upper) + means
