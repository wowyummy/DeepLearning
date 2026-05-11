import argparse
import csv
import os

import matplotlib.pyplot as plt


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_train_metrics(csv_path, output_dir):
    rows = read_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    test_psnr = [float(row["test_psnr"]) for row in rows]
    test_ssim = [float(row["test_ssim"]) for row in rows]

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5), dpi=160)
    plt.plot(epochs, train_loss, color="#1f77b4", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("SRCNN Training Loss")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_loss_curve.png"))
    plt.close()

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=160)
    ax1.plot(epochs, test_psnr, color="#d62728", linewidth=1.8, label="PSNR")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("PSNR (dB)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(epochs, test_ssim, color="#2ca02c", linewidth=1.8, label="SSIM")
    ax2.set_ylabel("SSIM", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    plt.title("SRCNN Validation Metrics")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_metrics_curve.png"))
    plt.close(fig)


def plot_test_metrics(csv_path, output_dir):
    rows = [row for row in read_csv(csv_path) if row.get("file_name") != "Average"]
    if not rows:
        raise ValueError(f"No per-image rows found in {csv_path}")

    names = [row["file_name"] for row in rows]
    psnr_key = "srcnn_psnr" if "srcnn_psnr" in rows[0] else "psnr"
    ssim_key = "srcnn_ssim" if "srcnn_ssim" in rows[0] else "ssim"
    psnr = [float(row[psnr_key]) for row in rows]
    ssim = [float(row[ssim_key]) for row in rows]

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(max(8, len(names) * 1.1), 5), dpi=160)
    plt.bar(names, psnr, color="#4c78a8")
    plt.xlabel("Image")
    plt.ylabel("PSNR (dB)")
    plt.title("SRCNN Test PSNR")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_psnr_bar.png"))
    plt.close()

    plt.figure(figsize=(max(8, len(names) * 1.1), 5), dpi=160)
    plt.bar(names, ssim, color="#59a14f")
    plt.xlabel("Image")
    plt.ylabel("SSIM")
    plt.title("SRCNN Test SSIM")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_ssim_bar.png"))
    plt.close()

    if "bicubic_psnr" in rows[0] and "bicubic_ssim" in rows[0]:
        bicubic_psnr = [float(row["bicubic_psnr"]) for row in rows]
        bicubic_ssim = [float(row["bicubic_ssim"]) for row in rows]
        x = range(len(names))
        width = 0.38

        plt.figure(figsize=(max(8, len(names) * 1.25), 5), dpi=160)
        plt.bar([i - width / 2 for i in x], bicubic_psnr, width=width, label="Bicubic", color="#9ecae1")
        plt.bar([i + width / 2 for i in x], psnr, width=width, label="SRCNN", color="#4c78a8")
        plt.xlabel("Image")
        plt.ylabel("PSNR (dB)")
        plt.title("Bicubic vs SRCNN PSNR")
        plt.xticks(list(x), names, rotation=35, ha="right")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bicubic_srcnn_psnr_compare.png"))
        plt.close()

        plt.figure(figsize=(max(8, len(names) * 1.25), 5), dpi=160)
        plt.bar([i - width / 2 for i in x], bicubic_ssim, width=width, label="Bicubic", color="#a1d99b")
        plt.bar([i + width / 2 for i in x], ssim, width=width, label="SRCNN", color="#59a14f")
        plt.xlabel("Image")
        plt.ylabel("SSIM")
        plt.title("Bicubic vs SRCNN SSIM")
        plt.xticks(list(x), names, rotation=35, ha="right")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bicubic_srcnn_ssim_compare.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot SRCNN experiment metrics for reports.")
    parser.add_argument("--train_csv", type=str, default="", help="Path to train_metrics.csv.")
    parser.add_argument("--test_csv", type=str, default="", help="Path to test_metrics.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save figures.")
    args = parser.parse_args()

    if args.train_csv:
        plot_train_metrics(args.train_csv, args.output_dir)
    if args.test_csv:
        plot_test_metrics(args.test_csv, args.output_dir)

    print(f"Metric figures save to `{args.output_dir}`")


if __name__ == "__main__":
    main()
