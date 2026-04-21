import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from rcnn_experiment.utils.io import ensure_dir, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge metrics.json files into CSV and Markdown tables.")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more metrics.json files.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.inputs:
        metrics = load_json(path)
        rows.append(
            {
                "experiment": metrics.get("experiment_name"),
                "model_family": metrics.get("model_family"),
                "dataset": metrics.get("dataset_name"),
                "split": metrics.get("split"),
                "mAP@0.5": metrics.get("mAP@0.5"),
                "mAP@0.5:0.95": metrics.get("mAP@0.5:0.95"),
                "Precision": metrics.get("Precision"),
                "Recall": metrics.get("Recall"),
                "F1": metrics.get("F1"),
                "FPS": metrics.get("FPS"),
                "num_detections": metrics.get("num_detections"),
                "score_threshold": metrics.get("score_threshold"),
                "nms_threshold": metrics.get("nms_threshold"),
                "metrics_path": str(Path(path).resolve()),
            }
        )
    df = pd.DataFrame(rows)
    ensure_dir(Path(args.output_csv).parent)
    ensure_dir(Path(args.output_md).parent)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    Path(args.output_md).write_text(df.to_markdown(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()
