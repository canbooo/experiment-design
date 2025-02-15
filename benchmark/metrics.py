from typing import Protocol

import numpy as np
from scipy.spatial.distance import pdist


class DoEMetric(Protocol):
    def __call__(self, doe: np.ndarray) -> float: ...


def max_correlation_error(doe: np.ndarray) -> float:
    """Assumes that the target correlation is 0 for all variable pairs"""
    corr = np.corrcoef(doe, rowvar=False)
    return float(np.max(np.abs(corr - np.eye(doe.shape[1]))))


def mean_correlation_error(doe: np.ndarray) -> float:
    """Assumes that the target correlation is 0 for all variable pairs"""
    corr = np.corrcoef(doe, rowvar=False)
    return float(np.mean(np.abs(corr - np.eye(doe.shape[1]))))


def min_pairwise_distance(doe: np.ndarray) -> float:
    return float(np.min(pdist(doe)))


def average_inverse_distance(doe: np.ndarray) -> float:
    return float(np.mean((1 / pdist(doe)) ** (doe.shape[1] + 1)))
