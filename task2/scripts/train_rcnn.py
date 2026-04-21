import argparse

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from rcnn_experiment.engine.train_rcnn import train_rcnn_model
from rcnn_experiment.utils.io import save_json
from rcnn_experiment.utils.runtime import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a teaching-oriented R-CNN classifier.")
    parser.add_argument("--train-annotations", required=True, help="Unified training annotation JSON.")
    parser.add_argument("--proposal-dir", required=True, help="Directory containing per-image proposal npz files.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--pos-iou", type=float, default=0.5)
    parser.add_argument("--neg-iou", type=float, default=0.3)
    parser.add_argument("--max-pos-per-image", type=int, default=32)
    parser.add_argument("--max-neg-per-image", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    stats = train_rcnn_model(
        train_annotations_path=args.train_annotations,
        proposal_dir=args.proposal_dir,
        output_dir=args.output_dir,
        backbone=args.backbone,
        device=args.device,
        pos_iou=args.pos_iou,
        neg_iou=args.neg_iou,
        max_pos_per_image=args.max_pos_per_image,
        max_neg_per_image=args.max_neg_per_image,
        batch_size=args.batch_size,
    )
    stats["seed"] = args.seed
    save_json(vars(args), f"{args.output_dir}/run_config.json")
    save_json(stats, f"{args.output_dir}/train_stats.json")


if __name__ == "__main__":
    main()
