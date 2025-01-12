import numpy as np
import pytest
from scipy import stats

import experiment_design.variable.space as module_under_test


@pytest.fixture(
    params=(
        [stats.uniform(-2, 4) for _ in range(2)],
        [stats.uniform(-2, 4), stats.bernoulli(0.8)],
        [stats.bernoulli(0.8) for _ in range(2)],
    ),
    ids=("Continuous", "Mixed", "Discrete"),
)
def design_space(request) -> module_under_test.ParameterSpace:
    return module_under_test.ParameterSpace(request.param)


class TestParameterSpace:
    def test_value_of(self, design_space: module_under_test.ParameterSpace):
        probabilities = np.array([[0.1, 0.1], [0.5, 0.6], [0.9, 0.8]])
        if isinstance(design_space.variables[0], module_under_test.DiscreteVariable):
            # Both are discrete
            expected = np.array([[0, 0], [1, 1], [1, 1]])
        elif isinstance(design_space.variables[1], module_under_test.DiscreteVariable):
            # Mixed case
            expected = np.array([[-1.6, 0], [0, 1], [1.6, 1]])
        else:
            # Both are continuous
            expected = np.array([[-1.6, -1.6], [0, 0.4], [1.6, 1.2]])
        assert np.all(np.isclose(design_space.value_of(probabilities), expected))

    def test_design_space_size_getters(
        self, design_space: module_under_test.ParameterSpace
    ):
        assert design_space.dimensions == 2

    def test_design_space_lower_bound(
        self, design_space: module_under_test.ParameterSpace
    ):
        if isinstance(design_space.variables[0], module_under_test.DiscreteVariable):
            # Both are discrete
            expected = np.array([0, 0])
        elif isinstance(design_space.variables[1], module_under_test.DiscreteVariable):
            # Mixed case
            expected = np.array([-2, 0])
        else:
            # Both are continuous
            expected = np.array([-2, -2])
        assert all(design_space.lower_bound == expected)

    def test_design_space_upper_bound(
        self, design_space: module_under_test.ParameterSpace
    ):
        if isinstance(design_space.variables[0], module_under_test.DiscreteVariable):
            # Both are discrete
            expected = np.array([1, 1])
        elif isinstance(design_space.variables[1], module_under_test.DiscreteVariable):
            # Mixed case
            expected = np.array([2, 1])
        else:
            # Both are continuous
            expected = np.array([2, 2])
        assert np.all(design_space.upper_bound == expected)
