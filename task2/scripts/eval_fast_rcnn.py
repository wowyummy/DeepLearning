import argparse

from rcnn_experiment.engine.infer_fast_rcnn import infer_fast_rcnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and evaluate the teaching-oriented Fast R-CNN model.")
    parser.add_argument("--annotations", required=True, help="Unified evaluation annotation JSON.")
    parser.add_argument("--proposal-dir", required=True, help="Directory containing proposal npz files.")
    parser.add_argument("--model-path", required=True, help="Path to trained fast_rcnn_model.pth.")
    parser.add_argument("--output-dir", required=True, help="Directory for detections and metrics.")
    parser.add_argument("--image-root", help="Optional image directory used when annotation file paths are invalid on this machine.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--max-proposals", type=int, default=500)
    parser.add_argument("--proposal-batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer_fast_rcnn(
        annotations_path=args.annotations,
        proposal_dir=args.proposal_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        image_root=args.image_root,
        device=args.device,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        max_proposals=args.max_proposals,
        proposal_batch_size=args.proposal_batch_size,
    )


if __name__ == "__main__":
    main()
