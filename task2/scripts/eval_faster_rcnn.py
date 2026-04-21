import argparse

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from rcnn_experiment.engine.infer_faster_rcnn import infer_faster_rcnn
from rcnn_experiment.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and evaluate the teaching-oriented Faster R-CNN model.")
    parser.add_argument("--annotations", required=True, help="Unified evaluation annotation JSON.")
    parser.add_argument("--model-path", required=True, help="Path to trained faster_rcnn_model.pth.")
    parser.add_argument("--output-dir", required=True, help="Directory for detections and metrics.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_json(vars(args), f"{args.output_dir}/eval_config.json")
    infer_faster_rcnn(
        annotations_path=args.annotations,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
