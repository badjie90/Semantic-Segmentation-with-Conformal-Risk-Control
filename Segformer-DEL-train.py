"""
SegFormer DEL design-time pipeline.

Implements the manuscript methodology:
1. Multiclass task-specific baseline training with per-pixel CE.
2. Evidential fine-tuning with CE + EDL + L2-SP catastrophic-forgetting
   regularization and staged freezing/unfreezing.
3. Export train/calibration/test splits for predicted-class conformal runtime.
"""

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort # inport this library using "pip install onnxruntime numpy Pillow"
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor


def default_project_root() -> str:
    candidates = [
        "/mnt/nvme1n1/bbadjie/FCT-PROJECT-2025/LISBON-PHASE-1",
        "/home/bbadjie/FCT-PROJECT-2025/LISBON-PHASE-1",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


@dataclass(frozen=True)
class Config:
    project_root: str = default_project_root()
    image_dir: str = ""
    mask_dir: str = ""
    output_dir: str = ""
    split_dir: str = ""

    base_model: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    num_classes: int = 3
    class_names: Tuple[str, ...] = ("path", "object", "background")
    image_size: int = 256
    batch_size: int = 2
    num_workers: int = 4
    random_state: int = 42
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    baseline_epochs: int = 40
    edl_epochs: int = 30
    lr: float = 3e-5
    weight_decay: float = 0.01
    lambda_uq: float = 0.5
    lambda_cf: float = 1e-4
    beta_anneal_epochs: int = 10

    freeze_backbone_epochs: int = 5
    unfreeze_all_epoch: int = 15
    train_ratio: float = 0.70
    calib_ratio: float = 0.15
    test_ratio: float = 0.15
    eps: float = 1e-8

    def resolved(self) -> "Config":
        root = self.project_root
        return Config(**{
            **asdict(self),
            "image_dir": self.image_dir or os.path.join(root, "PREPROCESSED-DATA", "combined_images"),
            "mask_dir": self.mask_dir or os.path.join(root, "PREPROCESSED-DATA", "combined_masks"),
            "output_dir": self.output_dir or os.path.join(root, "TRAINING", "SEGFORMER", "DEL", "FOCAL", "Training-outputs"),
            "split_dir": self.split_dir or os.path.join(root, "TRAINING", "SEGFORMER", "DEL", "FOCAL", "splits"),
        })


class SegDataset(Dataset):
    def __init__(self, items: List[Dict[str, str]], processor: SegformerImageProcessor, cfg: Config):
        self.items = items
        self.processor = processor
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img = Image.open(self.items[idx]["image"]).convert("RGB")
        mask = Image.open(self.items[idx]["mask"])
        encoded = self.processor(
            images=img,
            segmentation_maps=mask,
            return_tensors="pt",
            size={"height": self.cfg.image_size, "width": self.cfg.image_size},
        )
        labels = encoded["labels"].squeeze(0).long().clamp(0, self.cfg.num_classes - 1)
        return encoded["pixel_values"].squeeze(0), labels


class SegFormerEDL(nn.Module):
    def __init__(self, cfg: Config, pretrained: bool = True):
        super().__init__()
        id2label = {i: cfg.class_names[i] for i in range(cfg.num_classes)}
        label2id = {v: k for k, v in id2label.items()}
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                cfg.base_model,
                num_labels=cfg.num_classes,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
        else:
            model_cfg = SegformerConfig(
                num_labels=cfg.num_classes,
                id2label=id2label,
                label2id=label2id,
                depths=[1, 1, 1, 1],
                hidden_sizes=[8, 16, 32, 64],
                decoder_hidden_size=32,
                num_attention_heads=[1, 2, 4, 8],
            )
            self.model = SegformerForSemanticSegmentation(model_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).logits


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def require_onnx_package() -> None:
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("ONNX export requested, but the 'onnx' Python package is not installed. Install it before full training: pip install onnx") from exc


def collect_pairs(image_dir: str, mask_dir: str) -> Tuple[List[str], List[str]]:
    images = sorted([str(Path(image_dir) / f) for f in os.listdir(image_dir)])
    masks = sorted([str(Path(mask_dir) / f) for f in os.listdir(mask_dir)])
    if len(images) != len(masks):
        raise ValueError(f"Image/mask count mismatch: {len(images)} images vs {len(masks)} masks.")
    return images, masks


def split_data(cfg: Config) -> Dict[str, List[Dict[str, str]]]:
    images, masks = collect_pairs(cfg.image_dir, cfg.mask_dir)
    holdout = cfg.calib_ratio + cfg.test_ratio
    tr_i, ho_i, tr_m, ho_m = train_test_split(images, masks, test_size=holdout, random_state=cfg.random_state)
    cal_i, te_i, cal_m, te_m = train_test_split(
        ho_i, ho_m, test_size=cfg.test_ratio / holdout, random_state=cfg.random_state
    )
    split = {
        "train": [{"image": i, "mask": m} for i, m in zip(tr_i, tr_m)],
        "calib": [{"image": i, "mask": m} for i, m in zip(cal_i, cal_m)],
        "val": [{"image": i, "mask": m} for i, m in zip(cal_i, cal_m)],
        "test": [{"image": i, "mask": m} for i, m in zip(te_i, te_m)],
    }
    write_json(os.path.join(cfg.split_dir, "split.json"), split)
    return split


def make_loader(items: List[Dict[str, str]], processor: SegformerImageProcessor, cfg: Config, shuffle: bool) -> DataLoader:
    return DataLoader(
        SegDataset(items, processor, cfg),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def beta_at_epoch(epoch_index: int, anneal_epochs: int) -> float:
    if anneal_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch_index + 1) / float(anneal_epochs))


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    sum_alpha = alpha.sum(dim=1)
    c = alpha.shape[1]
    ln_b_alpha = torch.lgamma(alpha).sum(dim=1) - torch.lgamma(sum_alpha)
    ln_b_uniform = -math.lgamma(c)
    return ln_b_uniform - ln_b_alpha + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(sum_alpha).unsqueeze(1))).sum(dim=1)


def edl_loss(logits: torch.Tensor, targets: torch.Tensor, epoch_index: int, cfg: Config) -> Tuple[torch.Tensor, Dict[str, float]]:
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    alpha = F.softplus(logits) + 1.0
    strength = alpha.sum(dim=1, keepdim=True).clamp_min(cfg.eps)
    probs = alpha / strength
    one_hot = F.one_hot(targets, cfg.num_classes).permute(0, 3, 1, 2).float()
    mse = ((one_hot - probs) ** 2).sum(dim=1)
    var = (alpha * (strength - alpha) / (strength * strength * (strength + 1.0))).sum(dim=1)
    base = (mse + var).mean()
    tilde_alpha = one_hot + (1.0 - one_hot) * alpha
    kl = dirichlet_kl_to_uniform(tilde_alpha).mean()
    beta = beta_at_epoch(epoch_index, cfg.beta_anneal_epochs)
    return base + beta * kl, {"edl_base": float(base.item()), "kl": float(kl.item()), "beta": beta}


def snapshot_shared_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def l2sp_penalty(model: nn.Module, theta0: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    penalty = torch.zeros((), device=device)
    for name, p in model.named_parameters():
        if name in theta0 and p.requires_grad and p.shape == theta0[name].shape:
            penalty = penalty + torch.sum((p - theta0[name].to(device)) ** 2)
    return penalty


def set_requires_grad(params: Iterable[nn.Parameter], value: bool) -> None:
    for p in params:
        p.requires_grad = value


def configure_trainable_parameters(model: SegFormerEDL, epoch_index: int, cfg: Config) -> str:
    set_requires_grad(model.parameters(), True)
    if epoch_index < cfg.freeze_backbone_epochs:
        set_requires_grad(model.model.segformer.parameters(), False)
        set_requires_grad(model.model.decode_head.parameters(), True)
        return "decode_head_only"
    if epoch_index < cfg.unfreeze_all_epoch:
        set_requires_grad(model.model.segformer.parameters(), False)
        try:
            set_requires_grad(model.model.segformer.encoder.block[-1].parameters(), True)
            return "decode_head_plus_last_encoder_stage"
        except Exception:
            return "decode_head_only"
    set_requires_grad(model.parameters(), True)
    return "all"


@torch.no_grad()
def eval_path_iou(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    tp = fp = fn = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = F.interpolate(model(x), size=y.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1)
        gt = y == 0
        pr = pred == 0
        tp += int((gt & pr).sum().item())
        fp += int((~gt & pr).sum().item())
        fn += int((gt & ~pr).sum().item())
    return tp / max(1, tp + fp + fn)


def train_baseline(model: SegFormerEDL, train_loader: DataLoader, calib_loader: DataLoader, cfg: Config, device: torch.device) -> str:
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.baseline_epochs)
    best_iou = -1.0
    best_path = os.path.join(cfg.output_dir, "baseline_theta0_best_segformer.pth")
    for epoch in range(cfg.baseline_epochs):
        model.train()
        total = 0.0
        for x, y in tqdm(train_loader, desc=f"[baseline CE][{epoch + 1}/{cfg.baseline_epochs}]"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = F.interpolate(model(x), size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        scheduler.step()
        path_iou = eval_path_iou(model, calib_loader, device)
        print(f"[baseline] epoch={epoch + 1} loss={total / max(1, len(train_loader)):.4f} calib_path_iou={path_iou:.4f}")
        if path_iou > best_iou:
            best_iou = path_iou
            torch.save(model.state_dict(), best_path)
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "baseline_theta0_final_segformer.pth"))
    return best_path


def train_edl(model: SegFormerEDL, train_loader: DataLoader, calib_loader: DataLoader, theta0: Dict[str, torch.Tensor], cfg: Config, device: torch.device) -> str:
    criterion = nn.CrossEntropyLoss()
    best_iou = -1.0
    best_path = os.path.join(cfg.output_dir, "evidential_theta_best_segformer.pth")
    current_mode = None
    optimizer = None
    scheduler = None
    for epoch in range(cfg.edl_epochs):
        mode = configure_trainable_parameters(model, epoch, cfg)
        if mode != current_mode or optimizer is None:
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.edl_epochs - epoch))
            current_mode = mode
            print(f"[EDL] optimizer reset, trainable mode: {mode}")
        model.train()
        totals = {"loss": 0.0, "ce": 0.0, "edl": 0.0, "cf": 0.0, "kl": 0.0}
        beta = beta_at_epoch(epoch, cfg.beta_anneal_epochs)
        for x, y in tqdm(train_loader, desc=f"[EDL fine-tune][{epoch + 1}/{cfg.edl_epochs}]"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = F.interpolate(model(x), size=y.shape[-2:], mode="bilinear", align_corners=False)
            ce = criterion(logits, y)
            uq, comps = edl_loss(logits, y, epoch, cfg)
            cf = l2sp_penalty(model, theta0, device)
            loss = ce + cfg.lambda_uq * uq + cfg.lambda_cf * cf
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["ce"] += float(ce.item())
            totals["edl"] += float(uq.item())
            totals["cf"] += float(cf.item())
            totals["kl"] += comps["kl"]
        scheduler.step()
        path_iou = eval_path_iou(model, calib_loader, device)
        denom = max(1, len(train_loader))
        print(
            f"[EDL] epoch={epoch + 1} mode={mode} loss={totals['loss'] / denom:.4f} "
            f"ce={totals['ce'] / denom:.4f} edl={totals['edl'] / denom:.4f} "
            f"kl={totals['kl'] / denom:.4f} beta={beta:.3f} cf={totals['cf'] / denom:.2f} "
            f"calib_path_iou={path_iou:.4f}"
        )
        if path_iou > best_iou:
            best_iou = path_iou
            torch.save(model.state_dict(), best_path)
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "evidential_theta_final_segformer.pth"))
    return best_path


def export_onnx(model: nn.Module, cfg: Config, device: torch.device, out_path: str) -> str:
    model.eval()
    dummy = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Methodology-matched SegFormer DEL design-time training.")
    p.add_argument("--project-root", default=Config.project_root)
    p.add_argument("--image-dir", default="")
    p.add_argument("--mask-dir", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--split-dir", default="")
    p.add_argument("--device", default=Config.device)
    p.add_argument("--baseline-epochs", type=int, default=Config.baseline_epochs)
    p.add_argument("--edl-epochs", type=int, default=Config.edl_epochs)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--num-workers", type=int, default=Config.num_workers)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    p.add_argument("--lambda-uq", type=float, default=Config.lambda_uq)
    p.add_argument("--lambda-cf", type=float, default=Config.lambda_cf)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def run_smoke_test(cfg: Config, device: torch.device) -> None:
    print("Running SegFormer training smoke test...")
    smoke_cfg = Config(**{**asdict(cfg), "image_size": 64})
    model = SegFormerEDL(smoke_cfg, pretrained=False).to(device)
    x = torch.randn(1, 3, smoke_cfg.image_size, smoke_cfg.image_size, device=device)
    y = torch.randint(0, smoke_cfg.num_classes, (1, smoke_cfg.image_size, smoke_cfg.image_size), device=device)
    logits = F.interpolate(model(x), size=y.shape[-2:], mode="bilinear", align_corners=False)
    ce = nn.CrossEntropyLoss()(logits, y)
    uq, comps = edl_loss(logits, y, 0, smoke_cfg)
    theta0 = snapshot_shared_parameters(model)
    cf = l2sp_penalty(model, theta0, device)
    mode = configure_trainable_parameters(model, 0, smoke_cfg)
    print(
        "Smoke test OK: "
        f"logits={tuple(logits.shape)}, ce={float(ce.item()):.4f}, "
        f"edl={float(uq.item()):.4f}, kl={comps['kl']:.4f}, "
        f"cf={float(cf.item()):.4f}, trainable_mode={mode}"
    )


def main() -> None:
    args = parse_args()
    cfg = Config(
        project_root=args.project_root,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        split_dir=args.split_dir,
        device=args.device,
        baseline_epochs=args.baseline_epochs,
        edl_epochs=args.edl_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_uq=args.lambda_uq,
        lambda_cf=args.lambda_cf,
    ).resolved()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.split_dir, exist_ok=True)
    seed_everything(cfg.random_state)
    device = torch.device(cfg.device if torch.cuda.is_available() or "cuda" not in cfg.device else "cpu")
    if args.smoke_test:
        run_smoke_test(cfg, device)
        return

    require_onnx_package()
    split = split_data(cfg)
    processor = SegformerImageProcessor.from_pretrained(cfg.base_model)
    train_loader = make_loader(split["train"], processor, cfg, shuffle=True)
    calib_loader = make_loader(split["calib"], processor, cfg, shuffle=False)
    model = SegFormerEDL(cfg, pretrained=True).to(device)
    baseline_path = train_baseline(model, train_loader, calib_loader, cfg, device)
    model.load_state_dict(torch.load(baseline_path, map_location=device), strict=True)
    theta0 = snapshot_shared_parameters(model)
    evidential_path = train_edl(model, train_loader, calib_loader, theta0, cfg, device)
    model.load_state_dict(torch.load(evidential_path, map_location=device), strict=True)
    onnx_path = export_onnx(model, cfg, device, os.path.join(cfg.output_dir, "evidential_theta_best_segformer_del.onnx"))
    write_json(os.path.join(cfg.output_dir, "training_summary.json"), {
        "methodology": "SegFormer CE baseline followed by EDL fine-tuning with masked KL, beta annealing, L2-SP, staged unfreezing, and split export.",
        "config": asdict(cfg),
        "split_file": os.path.join(cfg.split_dir, "split.json"),
        "baseline_theta0_best": baseline_path,
        "evidential_theta_best": evidential_path,
        "evidential_theta_best_onnx": onnx_path,
    })
    print("Design-time training complete.")
    print(f"Baseline:   {baseline_path}")
    print(f"Evidential: {evidential_path}")
    print(f"ONNX:       {onnx_path}")


if __name__ == "__main__":
    main()
