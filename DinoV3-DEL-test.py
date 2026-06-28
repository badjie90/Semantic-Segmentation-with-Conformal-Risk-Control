"""
DINOv3/DINOv2 DEL run-time evaluation and conformal selective prediction.


- Dirichlet predictions alpha = softplus(logits) + 1.
- Mean probabilities, total entropy, aleatoric expected entropy, and epistemic MI.
- Class-conditional split-conformal thresholds grouped by predicted class.
- Selective output accepts a pixel iff s(u) <= tau_{hat_y(u)}(alpha).
"""

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", f"matplotlib-cache-{os.getuid()}"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix
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


def write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


@dataclass(frozen=True)
class Config:
    project_root: str = default_project_root()
    split_json: str = ""
    model_path: str = ""
    output_dir: str = ""

    backbone_name: str = "vit_base_patch14_dinov2"
    feature_channels: int = 768
    num_classes: int = 3
    class_names: Tuple[str, ...] = ("path", "object", "background")
    image_size: int = 518
    batch_size: int = 2
    num_workers: int = 4
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    alpha_values: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)
    score_names: Tuple[str, ...] = ("aleatoric", "epistemic")
    combine_rule: str = "and"

    max_pixels_metrics: int = 500_000
    rng_seed: int = 42
    eps: float = 1e-8

    def resolved(self) -> "Config":
        root = self.project_root
        split_json = self.split_json or os.path.join(root, "TRAINING", "DINOv3", "DEL", "splits", "split.json")
        model_path = self.model_path or os.path.join(root, "TRAINING", "DINOv3", "DEL", "Training-outputs", "evidential_theta_best.pth")
        output_dir = self.output_dir or os.path.join(root, "TRAINING", "DINOv3", "DEL", "Test-outputs")
        return Config(**{**asdict(self), "split_json": split_json, "model_path": model_path, "output_dir": output_dir})


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
        return img, torch.from_numpy(mask).clamp(0, self.num_classes - 1)


class DinoSeg(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.backbone = timm.create_model(cfg.backbone_name, pretrained=False, features_only=True)
        self.head = nn.Conv2d(cfg.feature_channels, cfg.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)[-1]
        logits = self.head(feats)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


@torch.no_grad()
def dirichlet_outputs(logits: torch.Tensor, eps: float) -> Dict[str, torch.Tensor]:
    alpha = F.softplus(logits) + 1.0
    alpha = alpha.clamp_min(eps)
    strength = alpha.sum(dim=1, keepdim=True).clamp_min(eps)
    probs = alpha / strength
    total = -(probs.clamp_min(eps) * torch.log(probs.clamp_min(eps))).sum(dim=1)

    alea = torch.digamma(strength + 1.0).squeeze(1)
    alea = alea - ((alpha / strength) * torch.digamma(alpha + 1.0)).sum(dim=1)
    epi = (total - alea).clamp_min(0.0)
    conf, preds = probs.max(dim=1)
    return {
        "alpha": alpha,
        "probs": probs,
        "preds": preds,
        "conf": conf,
        "total": total,
        "aleatoric": alea,
        "epistemic": epi,
    }


def make_loader(images: List[str], masks: List[str], cfg: Config) -> DataLoader:
    return DataLoader(
        SegDataset(images, masks, cfg),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def collect_pixels(model: nn.Module, loader: DataLoader, cfg: Config, device: torch.device, desc: str) -> Dict[str, np.ndarray]:
    arrays: Dict[str, List[np.ndarray]] = {
        "y": [], "pred": [], "conf": [], "total": [], "aleatoric": [], "epistemic": [], "probs": []
    }
    model.eval()
    for x, y in tqdm(loader, desc=desc):
        x = x.to(device, non_blocking=True)
        out = dirichlet_outputs(model(x), cfg.eps)
        arrays["y"].append(y.numpy().reshape(-1))
        arrays["pred"].append(out["preds"].cpu().numpy().reshape(-1))
        arrays["conf"].append(out["conf"].cpu().numpy().reshape(-1))
        arrays["total"].append(out["total"].cpu().numpy().reshape(-1))
        arrays["aleatoric"].append(out["aleatoric"].cpu().numpy().reshape(-1))
        arrays["epistemic"].append(out["epistemic"].cpu().numpy().reshape(-1))
        arrays["probs"].append(out["probs"].cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, cfg.num_classes))

    return {
        key: np.concatenate(vals, axis=0) if key == "probs" else np.concatenate(vals).reshape(-1)
        for key, vals in arrays.items()
    }


def sample_pixels(data: Dict[str, np.ndarray], max_pixels: int, seed: int) -> Dict[str, np.ndarray]:
    n = len(data["y"])
    if n <= max_pixels:
        return data
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_pixels, replace=False)
    return {k: v[idx] for k, v in data.items()}


def ece(conf: np.ndarray, pred: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    val = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            val += abs((pred[mask] == y[mask]).mean() - conf[mask].mean()) * (mask.mean())
    return float(val)


def aurc(conf: np.ndarray, pred: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    order = np.argsort(-conf)
    errors = (pred[order] != y[order]).astype(np.float64)
    risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    coverage = np.arange(1, len(errors) + 1) / len(errors)
    return float(np.trapz(risk, coverage))


def class_conditional_thresholds(
    pred: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    num_classes: int,
    alpha: float,
) -> Dict[int, float]:
    thresholds: Dict[int, float] = {}
    err = (pred != y).astype(np.int64)
    for c in range(num_classes):
        mask = pred == c
        if not mask.any():
            thresholds[c] = float("-inf")
            continue
        sc = scores[mask].astype(np.float64)
        ec = err[mask].astype(np.int64)
        order = np.argsort(sc)
        sc_sorted = sc[order]
        ec_sorted = ec[order]
        retained = np.arange(1, len(sc_sorted) + 1)
        risk = np.cumsum(ec_sorted) / retained
        ok = np.where(risk <= alpha)[0]
        thresholds[c] = float(sc_sorted[ok[-1]]) if ok.size else float("-inf")
    return thresholds


def accepted_mask(pred: np.ndarray, scores: Dict[str, np.ndarray], thresholds: Dict[str, Dict[int, float]], cfg: Config) -> np.ndarray:
    per_score_accept = []
    for score_name in cfg.score_names:
        tau = thresholds[score_name]
        score = scores[score_name]
        acc = np.zeros_like(pred, dtype=bool)
        for c in range(cfg.num_classes):
            m = pred == c
            acc[m] = score[m] <= tau[c]
        per_score_accept.append(acc)
    if cfg.combine_rule == "or":
        return np.logical_or.reduce(per_score_accept)
    return np.logical_and.reduce(per_score_accept)


def selective_metrics(data: Dict[str, np.ndarray], thresholds: Dict[str, Dict[int, float]], cfg: Config) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
    acc = accepted_mask(
        data["pred"],
        {"aleatoric": data["aleatoric"], "epistemic": data["epistemic"], "total": data["total"]},
        thresholds,
        cfg,
    )
    errors = data["pred"] != data["y"]
    coverage = float(acc.mean()) if len(acc) else 0.0
    risk = float(errors[acc].mean()) if acc.any() else 0.0

    pc_cov: Dict[str, float] = {}
    pc_risk: Dict[str, float] = {}
    for c, name in enumerate(cfg.class_names):
        m = data["pred"] == c
        pc_cov[name] = float(acc[m].mean()) if m.any() else 0.0
        pc_risk[name] = float(errors[m & acc].mean()) if (m & acc).any() else 0.0
    return coverage, risk, pc_cov, pc_risk


def save_core_metrics(data: Dict[str, np.ndarray], cfg: Config, metric_dir: str) -> None:
    y, pred, conf = data["y"], data["pred"], data["conf"]
    cm = confusion_matrix(y, pred, labels=list(range(cfg.num_classes)))
    rows = []
    fw_num = 0.0
    fw_den = 0.0
    for c, name in enumerate(cfg.class_names):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        iou = tp / max(1, tp + fp + fn)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        ap = average_precision_score((y == c).astype(int), data["probs"][:, c]) if len(y) else 0.0
        support = cm[c, :].sum()
        fw_num += support * iou
        fw_den += support
        rows.append([name, iou, precision, recall, f1, ap, support])

    pd.DataFrame(rows, columns=["Class", "IoU", "Precision", "Recall", "F1", "AP", "Support"]).to_csv(
        os.path.join(metric_dir, "per_class_segmentation_metrics.csv"), index=False
    )

    correct = pred == y
    with open(os.path.join(metric_dir, "dinov3_del_test_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["accuracy", float(correct.mean()) if len(y) else 0.0])
        w.writerow(["ece", ece(conf, pred, y)])
        w.writerow(["aurc", aurc(conf, pred, y)])
        w.writerow(["fw_iou", fw_num / max(1.0, fw_den)])
        for score_name in ("total", "aleatoric", "epistemic"):
            score = data[score_name]
            w.writerow([f"{score_name}_mean", float(score.mean()) if len(score) else 0.0])
            w.writerow([f"{score_name}_correct", float(score[correct].mean()) if correct.any() else 0.0])
            w.writerow([f"{score_name}_wrong", float(score[~correct].mean()) if (~correct).any() else 0.0])


def save_plots(test_data: Dict[str, np.ndarray], conformal_df: pd.DataFrame, cfg: Config, plot_dir: str) -> None:
    y, pred = test_data["y"], test_data["pred"]
    correct = pred == y
    for score_name in ("aleatoric", "epistemic", "total"):
        plt.figure(figsize=(7, 5))
        plt.hist(test_data[score_name][correct], bins=50, alpha=0.6, label="correct")
        plt.hist(test_data[score_name][~correct], bins=50, alpha=0.6, label="wrong")
        plt.xlabel(score_name)
        plt.ylabel("pixels")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{score_name}_vs_error.png"), dpi=150)
        plt.close()

    if not conformal_df.empty:
        plt.figure(figsize=(7, 5))
        plt.plot(conformal_df["overall_coverage"], conformal_df["overall_risk"], "o-")
        for a in conformal_df["alpha"]:
            plt.axhline(float(a), linestyle=":", alpha=0.4)
        plt.xlabel("coverage")
        plt.ylabel("accepted-pixel risk")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "conformal_risk_coverage.png"), dpi=150)
        plt.close()

        for score_name in cfg.score_names:
            plt.figure(figsize=(8, 5))
            for cls in cfg.class_names:
                plt.plot(conformal_df["alpha"], conformal_df[f"{cls}_{score_name}_tau"], "o-", label=cls)
            plt.xlabel("target risk alpha")
            plt.ylabel(f"{score_name} threshold")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"conformal_thresholds_{score_name}.png"), dpi=150)
            plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=" DINO DEL runtime evaluation.")
    p.add_argument("--project-root", default=Config.project_root)
    p.add_argument("--split-json", default="path to your test split json file")
    p.add_argument("--model-path", default="path to your trained model checkpoint")
    p.add_argument("--output-dir", default="path to your output directory for metrics and plots")
    p.add_argument("--device", default=Config.device)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--num-workers", type=int, default=Config.num_workers)
    p.add_argument("--alphas", nargs="+", type=float, default=list(Config.alpha_values))
    p.add_argument("--scores", nargs="+", choices=["aleatoric", "epistemic", "total"], default=list(Config.score_names))
    p.add_argument("--combine-rule", choices=["and", "or"], default=Config.combine_rule)
    p.add_argument("--max-pixels-metrics", type=int, default=Config.max_pixels_metrics)
    p.add_argument("--smoke-test", action="store_true", help="Run a tiny synthetic check without loading data or checkpoints.")
    return p.parse_args()


def run_smoke_test(cfg: Config, device: torch.device) -> None:
    print("Running runtime smoke test...")
    model = DinoSeg(cfg).to(device)
    model.eval()

    with torch.no_grad():
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device)
        out = dirichlet_outputs(model(x), cfg.eps)

    n = 128
    rng = np.random.default_rng(cfg.rng_seed)
    y = rng.integers(0, cfg.num_classes, size=n)
    pred = rng.integers(0, cfg.num_classes, size=n)
    aleatoric = rng.random(n)
    epistemic = rng.random(n) * 0.2
    total = aleatoric + epistemic
    thresholds = {
        "aleatoric": class_conditional_thresholds(pred, y, aleatoric, cfg.num_classes, alpha=0.10),
        "epistemic": class_conditional_thresholds(pred, y, epistemic, cfg.num_classes, alpha=0.10),
    }
    data = {
        "y": y,
        "pred": pred,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "total": total,
    }
    coverage, risk, _, _ = selective_metrics(data, thresholds, cfg)

    required = {"alpha", "probs", "preds", "conf", "total", "aleatoric", "epistemic"}
    missing = required.difference(out.keys())
    if missing:
        raise RuntimeError(f"Missing Dirichlet outputs: {sorted(missing)}")

    print(
        "Smoke test OK: "
        f"probs={tuple(out['probs'].shape)}, total={tuple(out['total'].shape)}, "
        f"coverage={coverage:.4f}, risk={risk:.4f}"
    )


def main() -> None:
    args = parse_args()
    cfg = Config(
        project_root=args.project_root,
        split_json=args.split_json,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        alpha_values=tuple(args.alphas),
        score_names=tuple(args.scores),
        combine_rule=args.combine_rule,
        max_pixels_metrics=args.max_pixels_metrics,
    ).resolved()

    device = torch.device(cfg.device if torch.cuda.is_available() or "cuda" not in cfg.device else "cpu")
    if args.smoke_test:
        run_smoke_test(cfg, device)
        return

    metric_dir = os.path.join(cfg.output_dir, "metrics")
    plot_dir = os.path.join(cfg.output_dir, "plots")
    os.makedirs(metric_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    with open(cfg.split_json, "r") as f:
        split = json.load(f)
    calib_imgs = split.get("calib_imgs", split.get("val_imgs"))
    calib_masks = split.get("calib_masks", split.get("val_masks"))
    test_imgs = split.get("test_imgs", split.get("images"))
    test_masks = split.get("test_masks", split.get("masks"))
    if test_imgs is None:
        test_json = os.path.join(Path(cfg.split_json).parent, "test_split.json")
        with open(test_json, "r") as f:
            test_split = json.load(f)
        test_imgs, test_masks = test_split["images"], test_split["masks"]

    model = DinoSeg(cfg).to(device)
    model.load_state_dict(torch.load(cfg.model_path, map_location=device), strict=True)

    calib_data = collect_pixels(model, make_loader(calib_imgs, calib_masks, cfg), cfg, device, "Collecting calibration pixels")
    test_data_full = collect_pixels(model, make_loader(test_imgs, test_masks, cfg), cfg, device, "Collecting test pixels")
    test_data = sample_pixels(test_data_full, cfg.max_pixels_metrics, cfg.rng_seed)

    save_core_metrics(test_data, cfg, metric_dir)

    rows = []
    for alpha in cfg.alpha_values:
        thresholds: Dict[str, Dict[int, float]] = {}
        for score_name in cfg.score_names:
            thresholds[score_name] = class_conditional_thresholds(
                pred=calib_data["pred"],
                y=calib_data["y"],
                scores=calib_data[score_name],
                num_classes=cfg.num_classes,
                alpha=alpha,
            )

        coverage, risk, pc_cov, pc_risk = selective_metrics(test_data_full, thresholds, cfg)
        row = {"alpha": alpha, "combine_rule": cfg.combine_rule, "overall_coverage": coverage, "overall_risk": risk}
        for c, cls in enumerate(cfg.class_names):
            row[f"{cls}_coverage"] = pc_cov[cls]
            row[f"{cls}_risk"] = pc_risk[cls]
            for score_name in cfg.score_names:
                row[f"{cls}_{score_name}_tau"] = thresholds[score_name][c]
        rows.append(row)
        print(f"alpha={alpha:.3f} coverage={coverage:.4f} risk={risk:.4f}")

    conformal_df = pd.DataFrame(rows)
    conformal_csv = os.path.join(metric_dir, "class_conditional_conformal_metrics.csv")
    conformal_df.to_csv(conformal_csv, index=False)
    save_plots(test_data, conformal_df, cfg, plot_dir)

    write_json(os.path.join(metric_dir, "runtime_summary.json"), {
        "methodology": "Predicted-class conditional split conformal selective prediction.",
        "config": asdict(cfg),
        "threshold_rule": "tau_c(alpha) is the largest uncertainty threshold with empirical retained-pixel risk <= alpha among calibration pixels predicted as class c.",
        "selective_rule": "accept iff every selected score passes its class-specific threshold for predicted class hat_y, or any score passes when combine_rule='or'.",
    })

    print("Runtime evaluation complete.")
    print(f"Metrics: {metric_dir}")
    print(f"Plots:   {plot_dir}")
    print(f"Conformal CSV: {conformal_csv}")


if __name__ == "__main__":
    main()
