#!/usr/bin/env python3
"""
cadet_atr experiment runner.

Usage:
    python run_experiment.py --mode baseline_only
    python run_experiment.py --mode adapt --strategy histogram    --checkpoint checkpoints/baseline_best.pt
    python run_experiment.py --mode adapt --strategy domain_random
    python run_experiment.py --mode adapt --strategy finetune    --checkpoint checkpoints/domain_random_best.pt
    python run_experiment.py --mode adapt --strategy dann        --checkpoint checkpoints/domain_random_best.pt
    python run_experiment.py --mode gap_only  --checkpoint checkpoints/dann_best.pt
    python run_experiment.py --mode full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Resolve paths so the script works regardless of CWD
_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from utils.config import cfg
from models.convnext import build_model, get_backbone
from data.dataset import make_loaders, generate_placeholder_synthetic
from training.trainer import Trainer
from evaluation.evaluator import measure_domain_gap, load_model, accuracy


def _ensure_data():
    """Create placeholder synthetic data if the data directory is empty."""
    synth_dir = Path(cfg.data_root) / "synthetic"
    if not synth_dir.exists() or not list(synth_dir.rglob("*.png")):
        print("[run] No synthetic data found — generating placeholders...")
        generate_placeholder_synthetic(cfg.data_root, n_per_class=40)


def run_baseline(verbose: bool = True) -> str:
    """Train ConvNeXt on synthetic data only. Returns checkpoint path."""
    _ensure_data()
    train_loader, val_loader = make_loaders()
    model   = build_model(cfg.model_name, cfg.num_classes, pretrained=cfg.pretrained)
    trainer = Trainer(model, verbose=verbose)
    ckpt    = trainer.fit(train_loader, val_loader,
                          ckpt_path=str(Path(cfg.ckpt_dir) / "baseline_best.pt"))
    print(f"[run] Baseline checkpoint → {ckpt}")
    return ckpt


def run_adapt_strategy(
    strategy: str,
    checkpoint: str,
    verbose: bool = True,
) -> str:
    """Run a single adaptation strategy on top of an existing checkpoint."""
    _ensure_data()

    if strategy == "histogram":
        from adaptation.strategies import build_reference_histogram, apply_histogram_matching
        train_loader, val_loader = make_loaders()
        model = build_model(cfg.model_name, cfg.num_classes, pretrained=False)
        if checkpoint:
            model = load_model(model, checkpoint)

        # Build reference histogram from val data (approximates real distribution)
        print("[histogram] Building reference histogram from val data...")
        ref_cdf = build_reference_histogram(val_loader)

        # Evaluate on synthetic data with matched histogram
        from data.dataset import SyntheticIRDataset
        import torch
        from torch.utils.data import DataLoader

        class HistMatchedDataset:
            def __init__(self, ds, src_cdf, tgt_cdf):
                self.ds = ds; self.src = src_cdf; self.tgt = tgt_cdf
            def __len__(self): return len(self.ds)
            def __getitem__(self, idx):
                img, label = self.ds[idx]
                img = apply_histogram_matching(img.unsqueeze(0), self.src, self.tgt).squeeze(0)
                return img, label

        from data.dataset import SyntheticIRDataset
        synth_path = Path(cfg.data_root) / "synthetic"
        ds = SyntheticIRDataset(str(synth_path), split="val")

        # Build synthetic histogram
        synth_cdf = build_reference_histogram(val_loader)
        matched_ds = HistMatchedDataset(ds, synth_cdf, ref_cdf)
        matched_loader = DataLoader(matched_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

        acc = accuracy(model, matched_loader)
        print(f"[histogram] Val accuracy after histogram matching: {acc:.3f}")
        ckpt_out = str(Path(cfg.ckpt_dir) / "histogram_best.pt")
        torch.save({"model_state": model.state_dict(), "val_acc": acc}, ckpt_out)
        return ckpt_out

    elif strategy == "domain_random":
        from adaptation.strategies import BackgroundSwapDataset
        from data.dataset import SyntheticIRDataset
        from torch.utils.data import DataLoader

        synth_path = Path(cfg.data_root) / "synthetic"
        train_ds = SyntheticIRDataset(str(synth_path), split="train")
        val_ds   = SyntheticIRDataset(str(synth_path), split="val")

        # Use synthetic images as the background pool (no real data needed)
        aug_ds   = BackgroundSwapDataset(train_ds, bg_pool=val_ds, swap_prob=0.4)
        train_loader = DataLoader(aug_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
        val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

        model = build_model(cfg.model_name, cfg.num_classes, pretrained=cfg.pretrained)
        if checkpoint:
            model = load_model(model, checkpoint)

        trainer = Trainer(model, epochs=max(10, cfg.epochs // 3), verbose=verbose)
        ckpt_out = trainer.fit(train_loader, val_loader,
                               ckpt_path=str(Path(cfg.ckpt_dir) / "domain_random_best.pt"))
        return ckpt_out

    elif strategy == "finetune":
        from adaptation.strategies import RealDataFinetuner

        # Use val synthetic as "real" substitute when no real data is available
        _train_loader, val_loader = make_loaders()
        real_dir = Path(cfg.data_root) / "real"
        if real_dir.exists() and list(real_dir.rglob("*.png")) + list(real_dir.rglob("*.jpg")):
            from data.dataset import make_real_loader
            real_train_loader = make_real_loader(augment=True)
            real_val_loader   = make_real_loader(augment=False)
        else:
            print("[finetune] No real data found — using synthetic val as proxy.")
            real_train_loader = val_loader
            real_val_loader   = val_loader

        model = build_model(cfg.model_name, cfg.num_classes, pretrained=False)
        if checkpoint:
            model = load_model(model, checkpoint)

        finetuner = RealDataFinetuner(model)
        ckpt_out  = finetuner.finetune(real_train_loader, real_val_loader, mode="full",
                                       ckpt_path=str(Path(cfg.ckpt_dir) / "finetune_best.pt"),
                                       verbose=verbose)
        return ckpt_out

    elif strategy == "dann":
        from adaptation.strategies import DANNModel, DANNTrainer

        train_loader, val_loader = make_loaders()

        # Target domain: val synthetic or real if available
        real_dir = Path(cfg.data_root) / "real"
        if real_dir.exists() and list(real_dir.rglob("*.png")) + list(real_dir.rglob("*.jpg")):
            from data.dataset import make_real_loader
            tgt_loader = make_real_loader(augment=True)
        else:
            print("[dann] No real data — using val synthetic as target domain.")
            tgt_loader = val_loader

        backbone = build_model(cfg.model_name, cfg.num_classes, pretrained=False)
        if checkpoint:
            backbone = load_model(backbone, checkpoint)

        dann = DANNModel(backbone.features, num_classes=cfg.num_classes, lam=cfg.dann_lambda)
        # Copy task head weights from pretrained model
        dann.task_head.weight.data = backbone.classifier[2].weight.data.clone()
        dann.task_head.bias.data   = backbone.classifier[2].bias.data.clone()

        trainer  = DANNTrainer(dann)
        ckpt_out = trainer.train(train_loader, tgt_loader, val_loader,
                                 ckpt_path=str(Path(cfg.ckpt_dir) / "dann_best.pt"),
                                 verbose=verbose)
        return ckpt_out

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         "Choose from: histogram, domain_random, finetune, dann")


def run_gap_only(checkpoint: str) -> dict:
    """Measure domain gap for a saved checkpoint."""
    _ensure_data()
    _, val_loader = make_loaders()
    model = build_model(cfg.model_name, cfg.num_classes, pretrained=False)
    model = load_model(model, checkpoint)

    real_dir = Path(cfg.data_root) / "real"
    if real_dir.exists() and list(real_dir.rglob("*.png")) + list(real_dir.rglob("*.jpg")):
        from data.dataset import make_real_loader
        real_loader = make_real_loader()
    else:
        print("[gap] No real data — using val synthetic for both halves (gap will be ~0).")
        real_loader = val_loader

    return measure_domain_gap(model, val_loader, real_loader)


def run_full_pipeline(verbose: bool = True) -> None:
    """Run all strategies sequentially: baseline → domain_random → finetune → dann."""
    print("\n" + "=" * 60)
    print("  cadet_atr full pipeline")
    print("=" * 60)

    print("\n[Stage 1] Baseline training on synthetic data")
    ckpt_baseline = run_baseline(verbose)

    print("\n[Stage 2] Domain randomisation")
    ckpt_dr = run_adapt_strategy("domain_random", ckpt_baseline, verbose)

    print("\n[Stage 3] Fine-tuning")
    ckpt_ft = run_adapt_strategy("finetune", ckpt_dr, verbose)

    print("\n[Stage 4] DANN")
    ckpt_dann = run_adapt_strategy("dann", ckpt_ft, verbose)

    print("\n[Final] Domain gap measurement")
    run_gap_only(ckpt_dann)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="cadet_atr domain adaptation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", required=True,
                        choices=["baseline_only", "adapt", "gap_only", "full"])
    parser.add_argument("--strategy",   default="dann",
                        choices=["histogram", "domain_random", "finetune", "dann"])
    parser.add_argument("--checkpoint", default="",  metavar="PATH",
                        help="Starting checkpoint for adapt / gap_only modes")
    parser.add_argument("--data",  default=cfg.data_root, metavar="PATH",
                        help=f"Data root (default: {cfg.data_root})")
    parser.add_argument("--ckpts", default=cfg.ckpt_dir,  metavar="PATH",
                        help=f"Checkpoint dir (default: {cfg.ckpt_dir})")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-epoch output")
    args = parser.parse_args()

    cfg.data_root = args.data
    cfg.ckpt_dir  = args.ckpts

    if args.mode == "baseline_only":
        run_baseline(verbose=not args.quiet)

    elif args.mode == "adapt":
        if not args.checkpoint and args.strategy not in ("domain_random",):
            parser.error(f"--checkpoint required for strategy '{args.strategy}'")
        run_adapt_strategy(args.strategy, args.checkpoint, verbose=not args.quiet)

    elif args.mode == "gap_only":
        if not args.checkpoint:
            parser.error("--checkpoint required for gap_only mode")
        run_gap_only(args.checkpoint)

    elif args.mode == "full":
        run_full_pipeline(verbose=not args.quiet)


if __name__ == "__main__":
    main()
