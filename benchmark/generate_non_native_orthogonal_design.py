import numpy as np

# noinspection PyProtectedMember
from scipy.stats._distn_infrastructure import rv_frozen

from experiment_design import ParameterSpace


def orthogonal_designs_from(lhs_path: str, distribution: rv_frozen) -> None:
    probabilities: dict[str, np.ndarray] = dict(np.load(lhs_path))
    results = {}
    for key, probs in probabilities.items():
        space = ParameterSpace([distribution for _ in range(probs.shape[1])])
        results[key] = space.value_of(probs)
    save_path = lhs_path.replace("uniform_", distribution.dist.name + "_")
    np.savez_compressed(save_path, **results)


if __name__ == "__main__":
    import glob
    import os

    from benchmark.generate_experiment_design import (
        DISTRIBUTIONS,
        NATIVE_ORTHOGONAL_SAMPLERS,
    )

    for algo_dir in glob.glob("../benchmark_results/doe/*"):
        algo_name = os.path.basename(algo_dir)
        if algo_name in NATIVE_ORTHOGONAL_SAMPLERS:
            continue
        lhs_paths = list(glob.glob(algo_dir + "/*"))
        for dist in DISTRIBUTIONS:
            if dist.dist.name == "uniform":
                continue
            for path in lhs_paths:
                orthogonal_designs_from(path, dist)
                print("Creating", dist.dist.name, "from", path, "for", algo_name)
