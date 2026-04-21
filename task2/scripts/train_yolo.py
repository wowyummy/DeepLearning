import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcnn_experiment.engine.train_yolo import train_yolo_model
from rcnn_experiment.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO detector with the project's unified annotation format.")
    parser.add_argument("--train-annotations", required=True, help="Unified training annotation JSON.")
    parser.add_argument("--val-annotations", help="Optional unified validation annotation JSON.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model checkpoint or model name.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = train_yolo_model(
        train_annotations_path=args.train_annotations,
        val_annotations_path=args.val_annotations,
        output_dir=args.output_dir,
        model=args.model,
        device=args.device,
        img_size=args.img_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        amp=args.amp,
    )
    save_json(stats, f"{args.output_dir}/train_stats.json")


if __name__ == "__main__":
    main()
