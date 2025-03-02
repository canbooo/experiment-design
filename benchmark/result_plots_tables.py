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


def to_filename(text: str) -> str:
    text = text.replace(" ", "_")
    return "".join([t.lower() for t in text if t.isalnum() or t == "_"])


def plot_results(
    metric_df: pd.DataFrame,
    x_name: str,
    y_name: str,
    title: str,
    use_log_scale: bool = False,
) -> None:
    plt.figure(figsize=(12, 7))
    plt.title(title)
    sns.lineplot(metric_df, x=x_name, y=y_name, hue="algorithm")
    if use_log_scale:
        plt.yscale("log")
    save_name = "-".join([to_filename(y_name), to_filename(x_name), to_filename(title)])
    plt.savefig(f"../docs/source/images/benchmark/{save_name}.png", bbox_inches="tight")


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

    metric_names = [
        "Max. Correlation Error",
        "Mean Correlation Error",
        "Min. Pairwise Distance",
        "Inv. Avg. Distance",
    ]

    for metric_name in metric_names:
        log_scale = metric_name == "Inv. Avg. Distance"
        plot_results(
            df,
            x_name="dimension",
            y_name=metric_name,
            title="All distributions",
            use_log_scale=log_scale,
        )
        plot_results(
            df,
            x_name="sample",
            y_name=metric_name,
            title="All distributions",
            use_log_scale=log_scale,
        )

    uniform_df = df[df.doe_type.str.endswith("uniform")]
    for metric_name in metric_names:
        log_scale = metric_name == "Inv. Avg. Distance"
        plot_results(
            uniform_df,
            x_name="dimension",
            y_name=metric_name,
            title="Uniform distribution",
            use_log_scale=log_scale,
        )
        plot_results(
            uniform_df,
            x_name="sample",
            y_name=metric_name,
            title="Uniform distribution",
            use_log_scale=log_scale,
        )

    plt.show()
