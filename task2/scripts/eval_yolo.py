import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcnn_experiment.engine.infer_yolo import infer_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference and evaluate with the project's unified metrics.")
    parser.add_argument("--annotations", required=True, help="Unified evaluation annotation JSON.")
    parser.add_argument("--model-path", required=True, help="Path to trained YOLO weights.")
    parser.add_argument("--output-dir", required=True, help="Directory for detections and metrics.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.7)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer_yolo(
        annotations_path=args.annotations,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
