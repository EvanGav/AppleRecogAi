"""
train.py
--------
Command-line entry point for training the apple detector.

Usage
-----
    python train.py --data_dir ./detection --epochs 20 --batch_size 4

This script wires together the concrete implementations of BaseDetector and
BaseDataLoader and hands them to AppleDetectorTrainer.  Changing detector or
dataset only requires changing the imports here — nothing else.
"""

import argparse
import json
from pathlib import Path

from data.MinneAppleDataLoader import MinneAppleDataLoader
from model.SSDMobileNetDetector import SSDMobileNetDetector
from model.AppleDetectorTrainer import AppleDetectorTrainer


def ParseArgs() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the MinnEapple apple detector."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("detection"),
        help="Root directory of the extracted MinnEapple dataset (default: ./detection)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Total number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Mini-batch size (default: 4)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Folder where weights and logs are saved (default: ./output)",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.35,
        help="Confidence threshold at inference (default: 0.35)",
    )
    return parser.parse_args()


def Main() -> None:
    """Main training entry point."""
    args = ParseArgs()

    print("=" * 60)
    print("  🍎  Apple Detector — Training")
    print("=" * 60)
    print(f"  Data dir     : {args.data_dir}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Conf thresh  : {args.conf_threshold}")
    print("=" * 60 + "\n")

    # ── Build concrete components ──────────────────────────────────────────────

    # 1. Data loader (ISP: provides only what Trainer needs)
    data_loader = MinneAppleDataLoader(
        data_dir   = args.data_dir,
        batch_size = args.batch_size,
    )

    # 2. Detector (OCP: swap with any BaseDetector subclass)
    detector = SSDMobileNetDetector(
        conf_threshold    = args.conf_threshold,
        nms_iou_threshold = 0.25,
        max_detections    = 100,
    )

    # 3. Trainer (DIP: depends only on abstractions)
    trainer = AppleDetectorTrainer(
        detector    = detector,
        data_loader = data_loader,
        output_dir  = args.output_dir,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    results = trainer.PrepareAndTrain(epochs=args.epochs)

    # Persist training history for the web dashboard
    history_path = args.output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(results["history"], f, indent=2)
    print(f"\nTraining history saved → {history_path}")

    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        # per_image_AP is a list of floats — fine for JSON
        json.dump(
            {"mAP": results["metrics"]["mAP"]},
            f,
            indent=2,
        )
    print(f"Metrics saved          → {metrics_path}")
    print("\n✅  Training complete.")


if __name__ == "__main__":
    Main()
