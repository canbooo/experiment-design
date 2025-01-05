import numpy as np
from scipy.linalg import solve_triangular


def iman_connover_transformation(
    doe: np.ndarray,
    target_correlation: np.ndarray,
    means: np.ndarray | None = None,
    standard_deviations: np.ndarray | None = None,
) -> np.ndarray:
    """
    Rearrange the values of doe to reduce correlation error while adhering to any marginal constraints of the values
    such as an |LHS|

    :param doe: Array with shape (n_sample, n_dim) representing the initial |DoE| with arbitrary
        correlation.
    :param target_correlation: Symmetric positive definite correlation matrix with shape (n_dim, n_dim) representing
        the desired correlation between variables
    :param means: Array with shape (n_dim,) representing the means of the marginal distributions. If None, it will be
        inferred from doe
    :param standard_deviations: Array with shape (n_dim,) representing the standard deviations of the marginal
        distributions. If None, it will be inferred from doe
    :return: New |DoE| with the same shape and values as doe but smaller correlation error wrt. target_correlation

    References
    ----------
    R.L. Iman and W.J. Conover (1982). “`A distribution-free approach to inducing rank correlation among input variables
    <https://www.researchgate.net/publication/243048186_A_Distribution-Free_Approach_to_Inducing_Rank_Correlation_Among_Input_Variates>`_”


    C. Bogoclu (2022). "`Local Latin Hypercube Refinement for Uncertainty Quantification and Optimization
    <https://hss-opus.ub.ruhr-uni-bochum.de/opus4/frontdoor/deliver/index/docId/9143/file/diss.pdf>`_" Chapter 4.3.2

    Examples
    --------
    >>> from experiment_design.covariance_modification import iman_connover_transformation
    >>> import numpy as np
    >>> from scipy import stats
    >>> np.random.seed(1337)
    >>> samples = stats.randint(0, 100).rvs((30, 2))
    >>> correlation_error = np.max(np.abs(np.corrcoef(samples, rowvar=False) - np.eye(2)))
    >>> new_samples = iman_connover_transformation(samples, np.eye(2))
    >>> np.max(np.abs(np.corrcoef(new_samples, rowvar=False) - np.eye(2))) < correlation_error
    True
    >>> sorted(samples[:, 0]) == sorted(new_samples[:, 0])
    True
    >>> sorted(samples[:, 1]) == sorted(new_samples[:, 1])
    True
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
    """
    Second-moment transformation for achieving the target covariance

    :param doe: Array with shape (n_sample, n_dim) representing the initial design of experiment with arbitrary
        correlation.
    :param target_correlation: Symmetric positive definite correlation matrix with shape (n_dim, n_dim) representing
        the desired correlation between variables
    :param means: Array with shape (n_dim,) representing the means of the marginal distributions. If None, it will be
        inferred from doe
    :param standard_deviations: Array with shape (n_dim,) representing the standard deviations of the marginal
        distributions. If None, it will be inferred from doe
    :param jitter: A small positive constant that will be added to the diagonal of the covariance matrix in case
        it is positive semi-definite to enable Cholesky decomposition.
    :return: New |DoE| with the same shape but different values as doe, that matches the target_correlation exactly.


    Examples
    --------
    >>> from experiment_design.covariance_modification import iman_connover_transformation
    >>> import numpy as np
    >>> from scipy import stats
    >>> np.random.seed(1337)
    >>> samples = stats.norm.rvs(size=(50, 2))
    >>> new_samples = iman_connover_transformation(samples, np.eye(2))
    >>> correlation_error = np.max(np.abs(np.corrcoef(new_samples, rowvar=False) - np.eye(2)))
    >>> bool(np.isclose(correlation_error, 0, atol=1e-6))
    True

    """
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
