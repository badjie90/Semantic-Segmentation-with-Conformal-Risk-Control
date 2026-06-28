"""
DINOv3/DINOv2 DEL design-time pipeline.


1. Train a task-specific multiclass segmentation baseline with CE + weight decay.
2. Fine-tune the same model with CE + evidential Dirichlet loss + L2-SP
   catastrophic-forgetting regularization, using staged freezing/unfreezing.
3. Save train/calibration/test splits for split-conformal calibration.
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
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort # inport this library using "pip install onnxruntime numpy Pillow"
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


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

    backbone_name: str = "vit_base_patch14_dinov2"
    feature_channels: int = 768
    pretrained_backbone: bool = True

    num_classes: int = 3
    class_names: Tuple[str, ...] = ("path", "object", "background")
    image_size: int = 518
    batch_size: int = 2
    num_workers: int = 4
    random_state: int = 42
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    baseline_epochs: int = 40
    edl_epochs: int = 30
    lr: float = 3e-5
    weight_decay: float = 1e-4

    lambda_uq: float = 0.5
    lambda_cf: float = 1e-4
    beta_anneal_epochs: int = 10

    freeze_backbone_epochs: int = 5
    unfreeze_last_blocks_epoch: int = 5
    unfreeze_all_epoch: int = 15
    unfreeze_last_blocks: int = 4

    train_ratio: float = 0.70
    calib_ratio: float = 0.15
    test_ratio: float = 0.15
    eps: float = 1e-8

    def resolved(self) -> "Config":
        root = self.project_root
        image_dir = self.image_dir or os.path.join(root, "PREPROCESSED-DATA", "combined_images")
        mask_dir = self.mask_dir or os.path.join(root, "PREPROCESSED-DATA", "combined_masks")
        output_dir = self.output_dir or os.path.join(root, "TRAINING", "DINOv3", "DEL", "Training-outputs")
        split_dir = self.split_dir or os.path.join(root, "TRAINING", "DINOv3", "DEL", "splits")
        return Config(**{**asdict(self), "image_dir": image_dir, "mask_dir": mask_dir, "output_dir": output_dir, "split_dir": split_dir})


class SegDataset(Dataset):
    def __init__(self, images: List[str], masks: List[str], cfg: Config):
        self.images = images
        self.masks = masks
        self.num_classes = cfg.num_classes
        self.tf_img = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
        ])
        self.tf_mask = transforms.Resize((cfg.image_size, cfg.image_size), interpolation=Image.NEAREST)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.tf_img(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(self.tf_mask(Image.open(self.masks[idx])), dtype=np.int64)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = torch.from_numpy(mask).clamp(0, self.num_classes - 1)
        return img, mask


class DinoSeg(nn.Module):
    def __init__(self, cfg: Config, pretrained: bool):
        super().__init__()
        self.backbone = timm.create_model(cfg.backbone_name, pretrained=pretrained, features_only=True)
        self.head = nn.Conv2d(cfg.feature_channels, cfg.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)[-1]
        logits = self.head(feats)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


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
    image_paths = sorted([str(Path(image_dir) / f) for f in os.listdir(image_dir)])
    mask_paths = sorted([str(Path(mask_dir) / f) for f in os.listdir(mask_dir)])
    if len(image_paths) != len(mask_paths):
        raise ValueError(f"Image/mask count mismatch: {len(image_paths)} images vs {len(mask_paths)} masks.")
    return image_paths, mask_paths


def split_data(cfg: Config) -> Dict[str, List[str]]:
    images, masks = collect_pairs(cfg.image_dir, cfg.mask_dir)
    holdout = cfg.calib_ratio + cfg.test_ratio
    tr_i, ho_i, tr_m, ho_m = train_test_split(images, masks, test_size=holdout, random_state=cfg.random_state)
    test_fraction_of_holdout = cfg.test_ratio / holdout
    cal_i, te_i, cal_m, te_m = train_test_split(
        ho_i, ho_m, test_size=test_fraction_of_holdout, random_state=cfg.random_state
    )
    split = {
        "train_imgs": tr_i,
        "train_masks": tr_m,
        "calib_imgs": cal_i,
        "calib_masks": cal_m,
        "test_imgs": te_i,
        "test_masks": te_m,
    }
    write_json(os.path.join(cfg.split_dir, "split.json"), split)
    write_json(os.path.join(cfg.split_dir, "train_val_split.json"), {
        "train_imgs": tr_i,
        "train_masks": tr_m,
        "val_imgs": cal_i,
        "val_masks": cal_m,
    })
    write_json(os.path.join(cfg.split_dir, "test_split.json"), {"images": te_i, "masks": te_m})
    return split


def make_loader(images: List[str], masks: List[str], cfg: Config, shuffle: bool) -> DataLoader:
    return DataLoader(
        SegDataset(images, masks, cfg),
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
    num_classes = alpha.shape[1]
    ln_b_alpha = torch.lgamma(alpha).sum(dim=1) - torch.lgamma(sum_alpha)
    ln_b_uniform = -math.lgamma(num_classes)
    digamma_sum = torch.digamma(sum_alpha).unsqueeze(1)
    kl = ln_b_uniform - ln_b_alpha
    kl = kl + ((alpha - 1.0) * (torch.digamma(alpha) - digamma_sum)).sum(dim=1)
    return kl


def edl_loss(logits: torch.Tensor, targets: torch.Tensor, epoch_index: int, cfg: Config) -> Tuple[torch.Tensor, Dict[str, float]]:
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True).clamp_min(cfg.eps)
    probs = alpha / strength
    one_hot = F.one_hot(targets, cfg.num_classes).permute(0, 3, 1, 2).float()

    mse = ((one_hot - probs) ** 2).sum(dim=1)
    var = (alpha * (strength - alpha) / (strength * strength * (strength + 1.0))).sum(dim=1)
    base = (mse + var).mean()

    tilde_alpha = one_hot + (1.0 - one_hot) * alpha
    kl = dirichlet_kl_to_uniform(tilde_alpha).mean()
    beta = beta_at_epoch(epoch_index, cfg.beta_anneal_epochs)
    loss = base + beta * kl
    return loss, {"edl_base": float(base.item()), "kl": float(kl.item()), "beta": beta}


def snapshot_shared_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def l2sp_penalty(model: nn.Module, theta0: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    penalty = torch.zeros((), device=device)
    for name, p in model.named_parameters():
        if name in theta0 and p.requires_grad and p.shape == theta0[name].shape:
            penalty = penalty + torch.sum((p - theta0[name].to(device)) ** 2)
    return penalty


def set_requires_grad(params: Iterable[nn.Parameter], requires_grad: bool) -> None:
    for p in params:
        p.requires_grad = requires_grad


def configure_trainable_parameters(model: DinoSeg, epoch_index: int, cfg: Config) -> str:
    set_requires_grad(model.parameters(), True)
    if epoch_index < cfg.freeze_backbone_epochs:
        set_requires_grad(model.backbone.parameters(), False)
        set_requires_grad(model.head.parameters(), True)
        return "head_only"

    if epoch_index < cfg.unfreeze_all_epoch:
        set_requires_grad(model.backbone.parameters(), False)
        set_requires_grad(model.head.parameters(), True)
        blocks = getattr(model.backbone, "blocks", None)
        if blocks is None and hasattr(model.backbone, "model"):
            blocks = getattr(model.backbone.model, "blocks", None)
        if blocks is not None:
            for block in list(blocks)[-cfg.unfreeze_last_blocks:]:
                set_requires_grad(block.parameters(), True)
            return f"head_plus_last_{cfg.unfreeze_last_blocks}_blocks"
        set_requires_grad(model.backbone.parameters(), True)
        return "all_no_block_api"

    set_requires_grad(model.parameters(), True)
    return "all"


def pixel_iou(model: nn.Module, loader: DataLoader, class_id: int, device: torch.device) -> float:
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(dim=1)
            gt_c = y == class_id
            pr_c = pred == class_id
            tp += int((gt_c & pr_c).sum().item())
            fp += int((~gt_c & pr_c).sum().item())
            fn += int((gt_c & ~pr_c).sum().item())
    return tp / max(1, tp + fp + fn)


def train_baseline(model: DinoSeg, train_loader: DataLoader, calib_loader: DataLoader, cfg: Config, device: torch.device) -> str:
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.baseline_epochs)
    best_iou = -1.0
    best_path = os.path.join(cfg.output_dir, "baseline_theta0_best.pth")

    for epoch in range(cfg.baseline_epochs):
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"[baseline CE][{epoch + 1}/{cfg.baseline_epochs}]"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        scheduler.step()

        path_iou = pixel_iou(model, calib_loader, class_id=0, device=device)
        print(f"[baseline] epoch={epoch + 1} loss={running / max(1, len(train_loader)):.4f} calib_path_iou={path_iou:.4f}")
        if path_iou > best_iou:
            best_iou = path_iou
            torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "baseline_theta0_final.pth"))
    print(f"Best baseline theta0 path IoU: {best_iou:.4f}")
    return best_path


def train_edl(model: DinoSeg, train_loader: DataLoader, calib_loader: DataLoader, theta0: Dict[str, torch.Tensor], cfg: Config, device: torch.device) -> str:
    criterion = nn.CrossEntropyLoss()
    best_iou = -1.0
    best_path = os.path.join(cfg.output_dir, "evidential_theta_best.pth")
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
            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            ce = criterion(logits, y)
            uq, comps = edl_loss(logits, y, epoch, cfg)
            cf = l2sp_penalty(model, theta0, device)
            loss = ce + cfg.lambda_uq * uq + cfg.lambda_cf * cf
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.item())
            totals["ce"] += float(ce.item())
            totals["edl"] += float(uq.item())
            totals["cf"] += float(cf.item())
            totals["kl"] += comps["kl"]

        scheduler.step()
        path_iou = pixel_iou(model, calib_loader, class_id=0, device=device)
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

    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "evidential_theta_final.pth"))
    print(f"Best evidential theta path IoU: {best_iou:.4f}")
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
    p = argparse.ArgumentParser(description="DINO DEL design-time training.")
    p.add_argument("--project-root", default=Config.project_root)
    p.add_argument("--image-dir", default="path to your preprocessed images directory")
    p.add_argument("--mask-dir", default="path to your masks directory")
    p.add_argument("--output-dir", default="path to your output directory")
    p.add_argument("--split-dir", default="path to your data split directory")
    p.add_argument("--device", default=Config.device)
    p.add_argument("--baseline-epochs", type=int, default=Config.baseline_epochs)
    p.add_argument("--edl-epochs", type=int, default=Config.edl_epochs)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--num-workers", type=int, default=Config.num_workers)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    p.add_argument("--lambda-uq", type=float, default=Config.lambda_uq)
    p.add_argument("--lambda-cf", type=float, default=Config.lambda_cf)
    p.add_argument("--smoke-test", action="store_true", help="Run a tiny synthetic check without loading data or pretrained weights.")
    return p.parse_args()


def run_smoke_test(cfg: Config, device: torch.device) -> None:
    print("Running training smoke test...")
    smoke_cfg = Config(**{**asdict(cfg), "pretrained_backbone": False})
    model = DinoSeg(smoke_cfg, pretrained=False).to(device)
    model.eval()

    x = torch.randn(1, 3, smoke_cfg.image_size, smoke_cfg.image_size, device=device)
    y = torch.randint(0, smoke_cfg.num_classes, (1, smoke_cfg.image_size, smoke_cfg.image_size), device=device)
    with torch.no_grad():
        logits = model(x)
    expected_shape = (1, smoke_cfg.num_classes, smoke_cfg.image_size, smoke_cfg.image_size)
    if logits.shape != expected_shape:
        raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")

    ce = nn.CrossEntropyLoss()(logits, y)
    uq, comps = edl_loss(logits, y, epoch_index=0, cfg=smoke_cfg)
    theta0 = snapshot_shared_parameters(model)
    cf = l2sp_penalty(model, theta0, device)
    mode = configure_trainable_parameters(model, epoch_index=0, cfg=smoke_cfg)

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
    train_loader = make_loader(split["train_imgs"], split["train_masks"], cfg, shuffle=True)
    calib_loader = make_loader(split["calib_imgs"], split["calib_masks"], cfg, shuffle=False)

    model = DinoSeg(cfg, pretrained=cfg.pretrained_backbone).to(device)
    baseline_path = train_baseline(model, train_loader, calib_loader, cfg, device)

    model.load_state_dict(torch.load(baseline_path, map_location=device))
    theta0 = snapshot_shared_parameters(model)
    evidential_path = train_edl(model, train_loader, calib_loader, theta0, cfg, device)
    model.load_state_dict(torch.load(evidential_path, map_location=device), strict=True)
    onnx_path = export_onnx(model, cfg, device, os.path.join(cfg.output_dir, "evidential_theta_best_dinov3_del.onnx"))

    summary = {
        "methodology": "CE baseline followed by EDL fine-tuning with masked KL, beta annealing, L2-SP, staged unfreezing, and split conformal calibration split export.",
        "config": asdict(cfg),
        "split_file": os.path.join(cfg.split_dir, "split.json"),
        "baseline_theta0_best": baseline_path,
        "evidential_theta_best": evidential_path,
        "evidential_theta_best_onnx": onnx_path,
        "losses": {
            "baseline": "CE + lambda_reg * weight_decay",
            "fine_tune": "CE + lambda_uq * (MSE + variance + beta(epoch)*KL(Dir(tilde_alpha)||Dir(1))) + lambda_cf * ||theta-theta0||_2^2",
        },
    }
    write_json(os.path.join(cfg.output_dir, "training_summary.json"), summary)
    print("Design-time training complete.")
    print(f"Baseline:   {baseline_path}")
    print(f"Evidential: {evidential_path}")
    print(f"ONNX:       {onnx_path}")


if __name__ == "__main__":
    main()
