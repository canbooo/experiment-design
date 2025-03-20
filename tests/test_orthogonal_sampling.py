import itertools

import numpy as np
import pytest
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

from experiment_design import orthogonal_sampling as module_under_test
from experiment_design import variable


@pytest.fixture(params=[0, 0.5])
def current_correlation_matrix(request) -> np.ndarray:
    corr = request.param
    corr_mat = np.eye(2)
    corr_mat[0, 1] = corr_mat[1, 0] = corr
    return corr_mat


@pytest.fixture
def distributions_to_test() -> list[rv_frozen]:
    return [stats.norm(5, 10), stats.uniform(-1, 1), stats.lognorm(0.3)]


class TestOrthogonalSamplingDesigner:
    @staticmethod
    def get_instance():
        return module_under_test.OrthogonalSamplingDesigner()

    def test_design(self, distributions_to_test, current_correlation_matrix) -> None:
        np.random.seed(1337)
        for dist1, dist2 in itertools.combinations(distributions_to_test, 2):
            space = variable.ParameterSpace(
                variables=[dist1, dist2], correlation=current_correlation_matrix
            )
            instance = self.get_instance()
            doe = instance.design(space, 256)
            assert np.isclose(
                np.corrcoef(doe, rowvar=False), current_correlation_matrix, atol=5e-2
            ).all()
            doe2 = instance.design(space, 256, old_sample=doe)
            assert np.isclose(
                np.corrcoef(doe2, rowvar=False), current_correlation_matrix, atol=5e-2
            ).all()
            doe3 = np.vstack((doe, doe2))
            assert np.isclose(
                np.corrcoef(doe3, rowvar=False), current_correlation_matrix, atol=5e-2
            ).all()

    def test_design_1d(self) -> None:
        space = variable.ParameterSpace([stats.norm()])
        instance = self.get_instance()
        doe = instance.design(space, sample_size=256)
        assert doe.shape == (256, 1)
