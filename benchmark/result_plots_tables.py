import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_doe_result_df(algorithms: list[str]) -> pd.DataFrame:
    all_results = []
    for algo in algorithms:
        all_results.append(pd.read_csv(f"../benchmark_results/{algo}_doe_metrics.csv"))
        all_results[-1]["algorithm"] = algo
    joined_df = pd.concat(all_results, axis=0)
    return joined_df.rename({"doe_name": "doe_type"}, axis=1)


if __name__ == "__main__":
    df = make_doe_result_df(
        [
            "diversipy",
            "doepy",
            "experiment_design",
            "pyDOE",
            "pyDOE2",
            "pyDOE3",
            "pyLHD",
        ]
    )
    for metric_name in [
        "Max. Correlation Error",
        "Mean Correlation Error",
        "Min. Pairwise Distance",
        "Inv. Avg. Distance",
    ]:
        plt.figure(figsize=(12, 7))
        plt.title("All distributions")
        sns.lineplot(df, x="dimension", y=metric_name, hue="algorithm")
        if metric_name == "Inv. Avg. Distance":
            plt.yscale("log")

        plt.figure(figsize=(12, 7))
        plt.title("All distributions")
        sns.lineplot(df, x="sample", y=metric_name, hue="algorithm")
        if metric_name == "Inv. Avg. Distance":
            plt.yscale("log")

    uniform_df = df[df.doe_type.str.endswith("uniform")]
    for metric_name in [
        "Max. Correlation Error",
        "Mean Correlation Error",
        "Min. Pairwise Distance",
        "Inv. Avg. Distance",
    ]:
        plt.figure(figsize=(12, 7))
        plt.title("Uniform distribution")
        sns.lineplot(uniform_df, x="dimension", y=metric_name, hue="algorithm")
        if metric_name == "Inv. Avg. Distance":
            plt.yscale("log")

        plt.figure(figsize=(12, 7))
        plt.title("Uniform distribution")
        sns.lineplot(uniform_df, x="sample", y=metric_name, hue="algorithm")
        if metric_name == "Inv. Avg. Distance":
            plt.yscale("log")

    plt.show()
