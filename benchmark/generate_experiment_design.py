import functools
import os
from typing import Protocol

import joblib
import numpy as np
from scipy import stats

# noinspection PyProtectedMember
from scipy.stats._distn_infrastructure import rv_frozen

from benchmark.external_adapter import (
    DiversiPyAdapter,
    DOEPyAdapter,
    PyDOE2Adapter,
    PyDOE3Adapter,
    PyDOEAdapter,
    PyLHDAdapter,
)
from experiment_design import OrthogonalSamplingDesigner, ParameterSpace


class DoEGenerator(Protocol):
    def __call__(
        self, space: ParameterSpace, sample_size: int, **kwargs
    ) -> np.ndarray: ...


def generate_does(
    random_seed: int,
    algorithm: DoEGenerator,
    dimension_sample_combinations: tuple[int, int],
    base_distribution: rv_frozen,
    save_prefix: str,
    target_folder: str,
) -> None:
    results = {}
    save_name = f"{save_prefix}_seed{random_seed}.npz"
    save_path = os.path.join(target_folder, save_name)
    for dimension, sample in dimension_sample_combinations:
        space = ParameterSpace([base_distribution for _ in range(dimension)])
        np.random.seed(random_seed)
        try:
            doe = algorithm(space, sample)
        except ValueError:
            continue
        results[f"dim{dimension}_size{sample}"] = doe
        np.savez_compressed(save_path, **results)


DISTRIBUTIONS = [
    stats.uniform(loc=0, scale=1),  # [0, 1]
    stats.norm(loc=0.5, scale=1 / 6.180464612335626),  # [0, 1] > 95 %
    stats.lognorm(0.448605225, scale=0.25),  # [0, 1] > 95%
    stats.gumbel_r(loc=-2.81, scale=1.13),  # [-5, 5] > 95 %
]


DIMENSION_SAMPLE_COMBINATIONS = [
    (2, 32),
    (2, 64),
    (2, 96),
    (2, 128),
    (3, 32),
    (3, 64),
    (3, 96),
    (3, 128),
    (4, 32),
    (4, 64),
    (4, 96),
    (4, 128),
    (5, 32),
    (5, 64),
    (5, 96),
    (5, 128),
    (10, 64),
    (10, 96),
    (10, 128),
    (10, 256),
    (15, 64),
    (15, 96),
    (15, 128),
    (15, 256),
    (20, 64),
    (20, 96),
    (20, 128),
    (20, 256),
    (25, 64),
    (25, 96),
    (25, 128),
    (25, 256),
    (50, 128),
    (50, 256),
    (50, 512),
    (75, 128),
    (75, 256),
    (75, 512),
    (100, 128),
    (100, 256),
    (100, 512),
]

# np.random.seed(1337)
# SEEDS = stats.randint(0, 2**32 - 1).rvs(64)
SEEDS = [
    1125387415,
    2407456957,
    681542492,
    913057000,
    1194544295,
    2332513753,
    1972751015,
    145906010,
    1378686834,
    1010987090,
    2226480212,
    2543106120,
    1125036297,
    1438929286,
    4192254390,
    1161710938,
    3147414551,
    1320499864,
    495099009,
    3446149915,
    1659038813,
    408656646,
    2699392022,
    1559675250,
    537119682,
    3183238761,
    4224309096,
    686974155,
    1903636289,
    3705043004,
    3391127255,
    3392574099,
    3410713326,
    22946129,
    1551606639,
    1991039579,
    1787152819,
    1636677433,
    2509369539,
    2235531668,
    3264912904,
    1588932168,
    806630973,
    2042842695,
    1237668474,
    3709011803,
    2878568073,
    1739551428,
    2145972959,
    1891940321,
    766946636,
    604963718,
    1774428356,
    3522650100,
    855537010,
    3855394293,
    2283631534,
    3987478263,
    3575004948,
    2660935186,
    795646746,
    4112099011,
    4111826537,
    3783868066,
]

NATIVE_ORTHOGONAL_SAMPLERS = ["experiment_design"]

if __name__ == "__main__":
    target_dir = "../benchmark_results/doe"
    os.makedirs(target_dir, exist_ok=True)
    designers = {
        "pyDOE": PyDOEAdapter(),
        "pyDOE2": PyDOE2Adapter(),
        "pyDOE3": PyDOE3Adapter(),
        "diversipy": DiversiPyAdapter(),
        "doepy": DOEPyAdapter(),
        "pyLHD": PyLHDAdapter(),
        "experiment_design": OrthogonalSamplingDesigner(),
    }
    for prefix, designer in designers.items():
        for dist in DISTRIBUTIONS:
            if prefix not in NATIVE_ORTHOGONAL_SAMPLERS and dist.dist.name != "uniform":
                # Since we map only using the probabilities, no need to generate these results
                # multiple times
                continue
            save_dir = os.path.join(target_dir, prefix)
            os.makedirs(save_dir, exist_ok=True)
            job = functools.partial(
                generate_does,
                algorithm=designer.design,
                dimension_sample_combinations=DIMENSION_SAMPLE_COMBINATIONS,
                base_distribution=dist,
                save_prefix=dist.dist.name,
                target_folder=save_dir,
            )
            with joblib.Parallel(n_jobs=16, verbose=10) as para:
                _ = para(joblib.delayed(job)(seed) for seed in SEEDS)
