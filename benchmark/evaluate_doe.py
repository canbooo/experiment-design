import glob
import os

import joblib
import numpy as np
import pandas as pd

from benchmark import metrics
from benchmark.generate_experiment_design import DIMENSION_SAMPLE_COMBINATIONS


def evaluate_doe_metrics(
    does: list[np.ndarray], doe_metrics: dict[str, metrics.DoEMetric]
) -> list[dict[str, float]]:
    return [
        {name: metric(doe) for name, metric in doe_metrics.items()}
        for doe in does
        if doe is not None
    ]


def load_does(
    algorithm_name: str, target_folder: str = "../benchmark_results/doe"
) -> dict[str, list[np.lib.npyio.NpzFile]]:
    target_pattern = os.path.join(target_folder, f"{algorithm_name}/*")
    does = {}
    for file_name in reversed(glob.glob(target_pattern)):
        key = os.path.basename(file_name).split("_seed")[0]
        does.setdefault(key, []).append(dict(np.load(file_name)))
    return does


def compute_all_metrics(
    result_key: str, does: list[np.lib.npyio.NpzFile], dimension: int, sample: int
) -> pd.DataFrame | None:
    cur_dim_sample_does = [
        cur_doe.get(f"dim{dimension}_size{sample}") for cur_doe in does
    ]
    cur_dim_sample_does = [doe for doe in cur_dim_sample_does if does is not None]
    if not cur_dim_sample_does:
        return None
    doe_metric_results = pd.DataFrame(
        evaluate_doe_metrics(cur_dim_sample_does, all_doe_metrics)
    )
    doe_metric_results["doe_name"] = result_key
    doe_metric_results["dimension"] = dimension
    doe_metric_results["sample"] = sample
    return doe_metric_results


if __name__ == "__main__":
    library_names = [
        "diversipy",
        "doepy",
        "experiment_design",
        "pyDOE",
        "pyDOE2",
        "pyDOE3",
        "pyLHD",
    ]
    for library_name in library_names:
        print("Processing", library_name, "results")
        all_does = load_does(library_name)

        all_doe_metrics = {
            "Max. Correlation Error": metrics.max_correlation_error,
            "Mean Correlation Error": metrics.mean_correlation_error,
            "Min. Pairwise Distance": metrics.min_pairwise_distance,
            "Inv. Avg. Distance": metrics.average_inverse_distance,
        }

        with joblib.Parallel(n_jobs=24, verbose=10) as para:
            all_results = para(
                joblib.delayed(compute_all_metrics)(key, cur_does, dimension, sample)
                for key, cur_does in all_does.items()
                for dimension, sample in DIMENSION_SAMPLE_COMBINATIONS
            )

        all_results = pd.concat(all_results, axis=0, ignore_index=True)

        all_results.to_csv(f"../benchmark_results/{library_name}_doe_metrics.csv")
