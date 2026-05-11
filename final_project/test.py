# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import csv

import cv2
import numpy as np
import torch
from natsort import natsorted

import config
import imgproc
from image_quality_assessment import PSNR, SSIM
from model import SRCNN


def main() -> None:
    ensure_test_config()

    # Initialize the super-resolution model
    model = SRCNN().to(device=config.device, memory_format=torch.channels_last)
    print("Build SRCNN model successfully.")

    # Load the super-resolution model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load SRCNN model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    results_dir = config.sr_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    comparison_dir = os.path.join(results_dir, "comparisons")
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)

    # Start the verification mode of the model.
    model.eval()
    # Initialize the sharpness evaluation function
    psnr = PSNR(config.upscale_factor, False)
    ssim = SSIM(config.upscale_factor, False)

    # Set the sharpness evaluation function calculation device to the specified model
    psnr = psnr.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    ssim = ssim.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

    # Initialize IQA metrics
    bicubic_psnr_metrics = 0.0
    bicubic_ssim_metrics = 0.0
    srcnn_psnr_metrics = 0.0
    srcnn_ssim_metrics = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)
    image_metrics = []

    for index in range(total_files):
        lr_image_path = os.path.join(config.lr_dir, file_names[index])
        sr_image_path = os.path.join(config.sr_dir, file_names[index])
        hr_image_path = os.path.join(config.hr_dir, file_names[index])
        comparison_image_path = os.path.join(comparison_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        # Read LR image and HR image
        lr_image = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        hr_image = cv2.imread(hr_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

        lr_image = imgproc.image_resize(lr_image, 1 / config.upscale_factor)
        lr_image = imgproc.image_resize(lr_image, config.upscale_factor)

        # Get Y channel image data
        lr_y_image = imgproc.bgr2ycbcr(lr_image, True)
        hr_y_image = imgproc.bgr2ycbcr(hr_image, True)

        # Get Cb Cr image data from hr image
        hr_ycbcr_image = imgproc.bgr2ycbcr(hr_image, False)
        _, hr_cb_image, hr_cr_image = cv2.split(hr_ycbcr_image)

        # Convert RGB channel image format data to Tensor channel image format data
        lr_y_tensor = imgproc.image2tensor(lr_y_image, False, False).unsqueeze_(0)
        hr_y_tensor = imgproc.image2tensor(hr_y_image, False, False).unsqueeze_(0)

        # Transfer Tensor channel image format data to CUDA device
        lr_y_tensor = lr_y_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        hr_y_tensor = hr_y_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_y_tensor = model(lr_y_tensor).clamp_(0, 1.0)

        # Save image
        sr_y_image = imgproc.tensor2image(sr_y_tensor, False, True)
        sr_y_image = sr_y_image.astype(np.float32) / 255.0
        sr_ycbcr_image = cv2.merge([sr_y_image, hr_cb_image, hr_cr_image])
        sr_image = imgproc.ycbcr2bgr(sr_ycbcr_image)
        cv2.imwrite(sr_image_path, sr_image * 255.0)

        # Cal IQA metrics
        bicubic_psnr = psnr(lr_y_tensor, hr_y_tensor).item()
        bicubic_ssim = ssim(lr_y_tensor, hr_y_tensor).item()
        srcnn_psnr = psnr(sr_y_tensor, hr_y_tensor).item()
        srcnn_ssim = ssim(sr_y_tensor, hr_y_tensor).item()
        bicubic_psnr_metrics += bicubic_psnr
        bicubic_ssim_metrics += bicubic_ssim
        srcnn_psnr_metrics += srcnn_psnr
        srcnn_ssim_metrics += srcnn_ssim
        image_metrics.append([file_names[index],
                              f"{bicubic_psnr:.6f}", f"{bicubic_ssim:.6f}",
                              f"{srcnn_psnr:.6f}", f"{srcnn_ssim:.6f}",
                              f"{srcnn_psnr - bicubic_psnr:.6f}", f"{srcnn_ssim - bicubic_ssim:.6f}",
                              sr_image_path, comparison_image_path])

        save_comparison_image(lr_image, sr_image, hr_image, comparison_image_path)

    # Calculate the average value of the sharpness evaluation index,
    # and all index range values are cut according to the following values
    # PSNR range value is 0~100
    # SSIM range value is 0~1
    avg_bicubic_ssim = min(1, bicubic_ssim_metrics / total_files)
    avg_bicubic_psnr = min(100, bicubic_psnr_metrics / total_files)
    avg_srcnn_ssim = min(1, srcnn_ssim_metrics / total_files)
    avg_srcnn_psnr = min(100, srcnn_psnr_metrics / total_files)

    print(f"Bicubic PSNR: {avg_bicubic_psnr:4.2f} dB\n"
          f"Bicubic SSIM: {avg_bicubic_ssim:4.4f} u\n"
          f"SRCNN PSNR: {avg_srcnn_psnr:4.2f} dB\n"
          f"SRCNN SSIM: {avg_srcnn_ssim:4.4f} u\n"
          f"PSNR gain: {avg_srcnn_psnr - avg_bicubic_psnr:+.2f} dB\n"
          f"SSIM gain: {avg_srcnn_ssim - avg_bicubic_ssim:+.4f} u")

    metrics_path = os.path.join(results_dir, "test_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "bicubic_psnr", "bicubic_ssim", "srcnn_psnr", "srcnn_ssim",
                         "psnr_gain", "ssim_gain", "sr_image_path", "comparison_image_path"])
        writer.writerows(image_metrics)
        writer.writerow(["Average",
                         f"{avg_bicubic_psnr:.6f}", f"{avg_bicubic_ssim:.6f}",
                         f"{avg_srcnn_psnr:.6f}", f"{avg_srcnn_ssim:.6f}",
                         f"{avg_srcnn_psnr - avg_bicubic_psnr:.6f}",
                         f"{avg_srcnn_ssim - avg_bicubic_ssim:.6f}", "", ""])
    print(f"Per-image metrics save to `{metrics_path}`")
    print(f"Comparison images save to `{comparison_dir}`")


def save_comparison_image(lr_image: np.ndarray, sr_image: np.ndarray, hr_image: np.ndarray, output_path: str) -> None:
    """Save a LR/SR/HR visual comparison image for course reports."""
    lr_u8 = to_uint8_bgr(lr_image)
    sr_u8 = to_uint8_bgr(sr_image)
    hr_u8 = to_uint8_bgr(hr_image)

    target_size = (hr_u8.shape[1], hr_u8.shape[0])
    lr_u8 = cv2.resize(lr_u8, target_size, interpolation=cv2.INTER_CUBIC)
    sr_u8 = cv2.resize(sr_u8, target_size, interpolation=cv2.INTER_CUBIC)

    panels = [add_label(lr_u8, "Bicubic input"), add_label(sr_u8, "SRCNN output"), add_label(hr_u8, "Ground truth")]
    separator = np.full((target_size[1], 8, 3), 255, dtype=np.uint8)
    comparison = np.concatenate([panels[0], separator, panels[1], separator, panels[2]], axis=1)
    cv2.imwrite(output_path, comparison)


def to_uint8_bgr(image: np.ndarray) -> np.ndarray:
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (220, 30), (0, 0, 0), -1)
    cv2.putText(labeled, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return labeled


def ensure_test_config() -> None:
    if not hasattr(config, "lr_dir"):
        config.lr_dir = "./data/Set5/GTmod12"
    if not hasattr(config, "hr_dir"):
        config.hr_dir = "./data/Set5/GTmod12"
    if not hasattr(config, "sr_dir"):
        config.sr_dir = os.path.join("results", "test", config.exp_name)
    if not hasattr(config, "model_path"):
        config.model_path = os.path.join("results", config.exp_name, "best.pth.tar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test SRCNN and compare it with Bicubic baseline.")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--upscale_factor", type=int, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--lr_dir", type=str, default=None)
    parser.add_argument("--hr_dir", type=str, default=None)
    parser.add_argument("--sr_dir", type=str, default=None)
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    if args.exp_name is not None and args.sr_dir is None:
        config.sr_dir = os.path.join("results", "test", config.exp_name)


if __name__ == "__main__":
    apply_args(parse_args())
    main()
