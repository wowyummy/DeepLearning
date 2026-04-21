import argparse

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from rcnn_experiment.data.adapters import prepare_coco, prepare_voc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VOC/COCO annotations into a unified JSON format.")
    parser.add_argument("--dataset", required=True, choices=["voc", "coco"])
    parser.add_argument("--root", required=True, help="Dataset root directory.")
    parser.add_argument("--split", required=True, help="VOC split or COCO split name.")
    parser.add_argument("--output", required=True, help="Output unified annotation JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "voc":
        prepare_voc(root=args.root, split=args.split, output=args.output)
    else:
        prepare_coco(root=args.root, split=args.split, output=args.output)


if __name__ == "__main__":
    main()
