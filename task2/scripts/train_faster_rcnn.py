import argparse

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from rcnn_experiment.engine.train_faster_rcnn import train_faster_rcnn_model
from rcnn_experiment.utils.io import save_json
from rcnn_experiment.utils.runtime import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a teaching-oriented Faster R-CNN detector.")
    parser.add_argument("--train-annotations", required=True, help="Unified training annotation JSON.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    stats = train_faster_rcnn_model(
        train_annotations_path=args.train_annotations,
        output_dir=args.output_dir,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
    )
    stats["seed"] = args.seed
    save_json(vars(args), f"{args.output_dir}/run_config.json")
    save_json(stats, f"{args.output_dir}/train_stats.json")


if __name__ == "__main__":
    main()
