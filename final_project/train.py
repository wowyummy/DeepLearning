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
# ============================================================================
import argparse
import csv
import os
import shutil
import time
from enum import Enum

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader

import config
from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from model import SRCNN


def main() -> None:
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0

    train_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully")

    model = build_model()
    print("Build SRCNN model successfully.")

    pixel_criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(model)
    print("Define all optimizer functions successfully.")

    print("Check whether the pretrained model is restored...")
    if config.resume:
        # Load checkpoint model
        checkpoint = torch.load(config.resume, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        best_ssim = checkpoint["best_ssim"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Loaded pretrained model weights.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    save_config_snapshot(results_dir)

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    psnr_model = PSNR(config.upscale_factor, False)
    ssim_model = SSIM(config.upscale_factor, False)

    # Transfer the IQA model to the specified device
    psnr_model = psnr_model.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    ssim_model = ssim_model.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

    best_epoch = checkpoint.get("best_epoch", 0) if config.resume else 0
    last_epoch = start_epoch

    for epoch in range(start_epoch, config.epochs):
        last_epoch = epoch + 1
        train_loss = train(model, train_prefetcher, pixel_criterion, optimizer, epoch, scaler)
        psnr, ssim = validate(model, test_prefetcher, psnr_model, ssim_model, "Test")
        print("\n")

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr and ssim > best_ssim
        if is_best:
            best_epoch = epoch + 1
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        torch.save({"epoch": epoch + 1,
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "best_epoch": best_epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()},
                   os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"))
        if is_best:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "best.pth.tar"))
        if (epoch + 1) == config.epochs:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "last.pth.tar"))

        metrics_path = os.path.join(results_dir, "train_metrics.csv")
        append_metrics_csv(metrics_path,
                           ["epoch", "train_loss", "test_psnr", "test_ssim", "best_psnr", "best_ssim", "best_epoch", "is_best"],
                           [epoch + 1, f"{train_loss:.8f}", f"{psnr:.6f}", f"{ssim:.6f}",
                            f"{best_psnr:.6f}", f"{best_ssim:.6f}", best_epoch, int(is_best)])

    save_summary(results_dir, samples_dir, last_epoch, best_epoch, best_psnr, best_ssim)


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(config.train_image_dir, config.image_size, config.upscale_factor, "Train")
    test_datasets = TestImageDataset(config.test_lr_image_dir, config.test_hr_image_dir, config.upscale_factor)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    return train_prefetcher, test_prefetcher


def build_model() -> nn.Module:
    model = SRCNN()
    model = model.to(device=config.device, memory_format=torch.channels_last)

    return model


def define_loss() -> nn.MSELoss:
    pixel_criterion = nn.MSELoss()
    pixel_criterion = pixel_criterion.to(device=config.device, memory_format=torch.channels_last)

    return pixel_criterion


def define_optimizer(model) -> optim.SGD:
    optimizer = optim.SGD([{"params": model.features.parameters()},
                           {"params": model.map.parameters()},
                           {"params": model.reconstruction.parameters(), "lr": config.model_lr * 0.1}],
                          lr=config.model_lr,
                          momentum=config.model_momentum,
                          weight_decay=config.model_weight_decay,
                          nesterov=config.model_nesterov)

    return optimizer


def train(model: nn.Module,
          train_prefetcher: CUDAPrefetcher,
          pixel_criterion: nn.MSELoss,
          optimizer: optim.SGD,
          epoch: int,
          scaler: amp.GradScaler) -> float:
    """Training main program

    Args:
        model (nn.Module): the generator model in the generative network
        train_prefetcher (CUDAPrefetcher): training dataset iterator
        pixel_criterion (nn.MSELoss): Calculate the pixel difference between real and fake samples
        optimizer (optim.SGD): optimizer for optimizing generator models in generative networks
        epoch (int): number of training epochs during training the generative network
        scaler (amp.GradScaler): Mixed precision training function

    """
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        lr = batch_data["lr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        hr = batch_data["hr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

        # Initialize the generator gradient
        model.zero_grad()

        # Initialize generator gradients
        model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            sr = model(lr)
            loss = pixel_criterion(sr, hr)

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # Statistical loss value for terminal data output
        losses.update(loss.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % config.print_frequency == 0:
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1

    return losses.avg


def validate(model: nn.Module,
             data_prefetcher: CUDAPrefetcher,
             psnr_model: nn.Module,
             ssim_model: nn.Module,
             mode: str) -> [float, float]:
    """Test main program

    Args:
        model (nn.Module): generator model in adversarial networks
        data_prefetcher (CUDAPrefetcher): test dataset iterator
        psnr_model (nn.Module): The model used to calculate the PSNR function
        ssim_model (nn.Module): The model used to compute the SSIM function
        mode (str): test validation dataset accuracy or test dataset accuracy

    """
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes], prefix=f"{mode}: ")

    # Put the adversarial network model in validation mode
    model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer the in-memory data to the CUDA device to speed up the test
            lr = batch_data["lr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
            hr = batch_data["hr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

            # Use the generator model to generate a fake sample
            with amp.autocast():
                sr = model(lr)

            # Statistical loss value for terminal data output
            psnr = psnr_model(sr, hr)
            ssim = ssim_model(sr, hr)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % max(1, batches // 5) == 0:
                progress.display(batch_index)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

        # print metrics
        progress.display_summary()

        if mode != "Valid" and mode != "Test":
            raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

        return psnres.avg, ssimes.avg


def append_metrics_csv(path: str, header: list[str], row: list) -> None:
    """Append one row to a CSV file and create the header when needed."""
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def save_config_snapshot(results_dir: str) -> None:
    config_items = [
        ("exp_name", config.exp_name),
        ("upscale_factor", config.upscale_factor),
        ("device", str(config.device)),
        ("train_image_dir", config.train_image_dir),
        ("test_lr_image_dir", config.test_lr_image_dir),
        ("test_hr_image_dir", config.test_hr_image_dir),
        ("image_size", config.image_size),
        ("batch_size", config.batch_size),
        ("num_workers", config.num_workers),
        ("resume", config.resume),
        ("epochs", config.epochs),
        ("model_lr", config.model_lr),
        ("model_momentum", config.model_momentum),
        ("model_weight_decay", config.model_weight_decay),
        ("model_nesterov", config.model_nesterov),
        ("print_frequency", config.print_frequency),
        ("optimizer", "SGD"),
        ("loss", "MSELoss"),
    ]
    with open(os.path.join(results_dir, "config_snapshot.txt"), "w", encoding="utf-8") as f:
        for key, value in config_items:
            f.write(f"{key} = {value}\n")


def save_summary(results_dir: str, samples_dir: str, last_epoch: int, best_epoch: int, best_psnr: float, best_ssim: float) -> None:
    with open(os.path.join(results_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Experiment: {config.exp_name}\n")
        f.write(f"Last epoch: {last_epoch}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best PSNR: {best_psnr:.6f}\n")
        f.write(f"Best SSIM: {best_ssim:.6f}\n")
        f.write(f"Best checkpoint: {os.path.join(results_dir, 'best.pth.tar')}\n")
        f.write(f"Last checkpoint: {os.path.join(results_dir, 'last.pth.tar')}\n")
        f.write(f"Epoch checkpoints: {samples_dir}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SRCNN.")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--upscale_factor", type=int, default=None)
    parser.add_argument("--train_image_dir", type=str, default=None)
    parser.add_argument("--test_lr_image_dir", type=str, default=None)
    parser.add_argument("--test_hr_image_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--model_lr", type=float, default=None)
    parser.add_argument("--print_frequency", type=int, default=None)
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    apply_args(parse_args())
    main()
