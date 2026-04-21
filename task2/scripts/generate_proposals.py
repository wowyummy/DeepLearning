import argparse

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from rcnn_experiment.engine.proposals import generate_from_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Selective Search proposals for a unified dataset JSON.")
    parser.add_argument("--annotations", required=True, help="Unified annotation JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory to save per-image proposals.")
    parser.add_argument("--mode", default="fast", choices=["fast", "quality"])
    parser.add_argument("--max-proposals", type=int, default=1500)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_from_annotations(
        annotations_path=args.annotations,
        output_dir=args.output_dir,
        mode=args.mode,
        max_proposals=args.max_proposals,
        min_size=args.min_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
