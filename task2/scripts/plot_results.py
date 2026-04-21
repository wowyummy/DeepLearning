import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from rcnn_experiment.utils.io import ensure_dir, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot key metrics for report-ready figures.")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more metrics.json files.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    metric_names = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1", "FPS"]
    labels = []
    values = {name: [] for name in metric_names}
    for path in args.inputs:
        metrics = load_json(path)
        model_name = metrics.get("model_family") or "model"
        experiment_name = metrics.get("experiment_name") or metrics.get("split")
        labels.append(f"{model_name}-{experiment_name}")
        for name in metric_names:
            values[name].append(metrics.get(name, 0.0))

    for name in metric_names:
        plt.figure(figsize=(8, 4.5))
        plt.bar(labels, values[name], color="#3c6e71")
        plt.title(name)
        plt.ylabel(name)
        plt.xticks(rotation=20)
        plt.tight_layout()
        safe_name = name.replace("@", "_at_").replace(":", "_").replace(".", "_").replace("/", "_")
        plt.savefig(Path(output_dir) / f"{safe_name}.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
