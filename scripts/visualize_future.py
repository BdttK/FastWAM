"""Visualize / score a FastWAM checkpoint's imagined future against ground truth.

`infer_action` (the path used by `scripts/fastwam_serve.py`) never runs the video
denoiser, so it cannot show a future.  This script calls `model.infer_joint`,
which jointly denoises the video latents and the action chunk and then VAE-decodes
the video.  For each sampled validation clip it writes:

    <out>/sample_XXXX/pred.mp4              imagined future (9 frames, uncond)
    <out>/sample_XXXX/gt.mp4                ground-truth future from the dataset
    <out>/sample_XXXX/side_by_side.mp4      GT | PRED, one frame per row-column
    <out>/sample_XXXX/input.png             conditioning first frame
    <out>/metrics.json                      per-frame PSNR + action MSE

Typical use (from the FastWAM repo root, on the machine that holds the val data):

    python scripts/visualize_future.py task=cobotmagic \
      +VIS.ckpt=checkpoints/fastwam_cobotmagic_20261120/model.pt \
      +VIS.num_samples=8

If the val data lives somewhere else than configs/data/*.yaml says:

    python scripts/visualize_future.py task=cobotmagic \
      +VIS.ckpt=/path/model.pt \
      +VIS.dataset_dirs=['/data/cobotmagic_Real_items_handover_val'] \
      +VIS.val_set_proportion=0.0 \
      +VIS.num_samples=10

To compare checkpoints, run the script twice with different `+VIS.ckpt` /
`+VIS.output_dir` and compare `metrics.json` (mean_future_psnr and
mean_action_mse).  Compare checkpoints from the *same* fine-tuning run (e.g.
different saved steps); the released RobotWin checkpoint has different
action/proprio dims, so its heads will not load into this 32-dim model
(`load_checkpoint` is strict=False) and it is not a fair baseline.

Notes
-----
* The fine-tuned uncond model was trained with `action_conditioned=false`, so the
  imagined video is the model's most likely continuation given first frame +
  text + proprio; it is NOT the future induced by the actions that will be
  executed.  That is expected for Fast-WAM's "first-frame" mode.
* Frame 0 of the prediction is a VAE round-trip of the input frame.  The reported
  `future_psnr` ignores it and only averages frames 1..T-1.
* Nothing is written into the checkpoint directory; outputs go to
  `evaluate_results/future_video/<task>/` unless `+VIS.output_dir=...` is given.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fastwam.utils.config_resolvers import register_default_resolvers  # noqa: E402
from fastwam.utils.video_io import save_mp4  # noqa: E402

register_default_resolvers()
logger = logging.getLogger(__name__)


def _vis_config(cfg: DictConfig) -> DictConfig:
    defaults = OmegaConf.create(
        {
            "ckpt": None,
            "device": None,
            "dataset_stats_path": None,
            "dataset_dirs": None,
            "val_set_proportion": None,
            "output_dir": None,
            "num_samples": 8,
            "indices": None,
            "use_text_cache": False,
            "num_inference_steps": None,
            "sigma_shift": None,
            "seed": 0,
            "fps": 4,
            "tiled": False,
        }
    )
    return OmegaConf.merge(defaults, cfg.get("VIS", {}))


def _resolve_dataset_stats_path(vis: DictConfig) -> Path:
    if vis.dataset_stats_path:
        path = Path(str(vis.dataset_stats_path))
    elif vis.ckpt:
        path = Path(str(vis.ckpt)).parent / "dataset_stats.json"
    else:
        raise ValueError(
            "Pass +VIS.ckpt=<model.pt> (dataset_stats.json is looked up next to it) "
            "or +VIS.dataset_stats_path=<dataset_stats.json>."
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"dataset_stats.json not found at {path}. "
            "Pass +VIS.dataset_stats_path=/path/to/dataset_stats.json."
        )
    return path


def _build_val_dataset(cfg: DictConfig, vis: DictConfig, stats_path: Path):
    ds_cfg = OmegaConf.to_container(cfg.data.val, resolve=True)
    if vis.dataset_dirs is not None:
        ds_cfg["dataset_dirs"] = list(vis.dataset_dirs)
    if vis.val_set_proportion is not None:
        ds_cfg["val_set_proportion"] = float(vis.val_set_proportion)
    # The val split is not the training split, so stats must be supplied.
    ds_cfg["pretrained_norm_stats"] = str(stats_path)
    ds_cfg["is_training_set"] = False
    ds_cfg["use_text_embed_cache"] = bool(vis.use_text_cache)
    ds = instantiate(OmegaConf.create(ds_cfg))
    logger.info(
        "val dataset: %d clips | dirs=%s | stats=%s",
        len(ds),
        ds_cfg["dataset_dirs"],
        stats_path,
    )
    return ds


def _tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    """[C,H,W] in [-1, 1] -> uint8 RGB PIL image."""
    arr = frame.detach().float().clamp(-1, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(((arr + 1.0) * 127.5).astype(np.uint8))


def _psnr(gt: Image.Image, pred: Image.Image) -> float:
    gt_arr = np.asarray(gt.convert("RGB"), dtype=np.float32)
    pred_arr = np.asarray(pred.convert("RGB"), dtype=np.float32)
    if gt_arr.shape != pred_arr.shape:
        gt = gt.resize(pred.size, resample=Image.BILINEAR)
        gt_arr = np.asarray(gt.convert("RGB"), dtype=np.float32)
    mse = float(np.mean((gt_arr - pred_arr) ** 2))
    return 10.0 * np.log10((255.0 * 255.0) / max(mse, 1e-8))


def _label(frame: Image.Image, text: str) -> Image.Image:
    canvas = frame.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=max(14, frame.width // 16))
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    pad = 4
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.rectangle([0, 0, bbox[2] - bbox[0] + 2 * pad, bbox[3] - bbox[1] + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return canvas


def _pick_indices(num_available: int, vis: DictConfig) -> list[int]:
    if vis.indices is not None:
        indices = [int(i) for i in vis.indices]
        bad = [i for i in indices if i < 0 or i >= num_available]
        if bad:
            raise IndexError(f"VIS.indices out of range [0,{num_available}): {bad}")
        return indices
    num_samples = max(1, min(int(vis.num_samples), num_available))
    picked = np.linspace(0, num_available - 1, num_samples).round().astype(int)
    return sorted(set(int(i) for i in picked))


@torch.no_grad()
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    vis = _vis_config(cfg)

    if not vis.ckpt:
        raise ValueError("Pass the checkpoint with +VIS.ckpt=/path/to/model.pt")

    stats_path = _resolve_dataset_stats_path(vis)
    device = str(vis.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else (
        torch.float16 if cfg.mixed_precision == "fp16" else torch.float32
    )

    task_name = str(HydraConfig.get().runtime.choices.get("task", "task"))
    out_dir = Path(str(vis.output_dir)) if vis.output_dir else (
        Path("evaluate_results") / "future_video" / task_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # The task config disables the text encoder for cached-embedding training runs;
    # visualization needs it unless we feed the cached context from the dataset.
    use_text_cache = bool(vis.use_text_cache)
    cfg.model.load_text_encoder = not use_text_cache
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.load_checkpoint(str(vis.ckpt))
    model.eval()
    logger.info(
        "loaded %s | device=%s | dtype=%s | text_encoder=%s (cache=%s)",
        vis.ckpt,
        device,
        model_dtype,
        model.text_encoder is not None,
        use_text_cache,
    )

    ds = _build_val_dataset(cfg, vis, stats_path)
    indices = _pick_indices(len(ds), vis)

    num_inference_steps = int(
        vis.num_inference_steps
        if vis.num_inference_steps is not None
        else cfg.get("eval_num_inference_steps", 10)
    )
    sigma_shift = None if vis.sigma_shift is None else float(vis.sigma_shift)
    seed = None if vis.seed is None else int(vis.seed)
    logger.info(
        "sampling %d clips | steps=%d | sigma_shift=%s | seed=%s",
        len(indices),
        num_inference_steps,
        sigma_shift,
        seed,
    )

    all_records = []
    for rank, idx in enumerate(indices):
        sample = ds[idx]
        video = torch.as_tensor(sample["video"]).float()  # [C, T_video, H, W] in [-1, 1]
        if video.ndim != 4:
            raise ValueError(f"Expected sample['video'] [C,T,H,W], got {tuple(video.shape)}")
        num_video_frames = int(video.shape[1])
        action = torch.as_tensor(sample["action"]).float()  # [T_action, D], normalized
        proprio = torch.as_tensor(sample["proprio"])[0].float()  # [D], normalized
        prompt = None if use_text_cache else str(sample["prompt"])
        context = sample.get("context") if use_text_cache else None
        context_mask = sample.get("context_mask") if use_text_cache else None

        infer_kwargs = {
            "prompt": prompt,
            "input_image": video[:, 0],
            "num_video_frames": num_video_frames,
            "action_horizon": int(action.shape[0]),
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "num_inference_steps": num_inference_steps,
            "sigma_shift": sigma_shift,
            "seed": seed,
            "rand_device": "cpu",
            "tiled": bool(vis.tiled),
            # Do not run the action-only path again; we only need the video here.
            "test_action_with_infer_action": False,
        }
        pred = model.infer_joint(**infer_kwargs)
        pred_frames = pred["video"]
        if len(pred_frames) != num_video_frames:
            raise RuntimeError(
                f"infer_joint returned {len(pred_frames)} frames, expected {num_video_frames}"
            )

        gt_frames = [_tensor_frame_to_pil(video[:, t]) for t in range(num_video_frames)]
        frame_psnr = [_psnr(gt_frames[t], pred_frames[t]) for t in range(num_video_frames)]
        future_psnr = float(np.mean(frame_psnr[1:])) if num_video_frames > 1 else float(frame_psnr[0])

        pred_action = pred["action"].float()
        action_mse = float(((pred_action - action) ** 2).mean().item())

        sample_dir = out_dir / f"sample_{rank:02d}_idx{idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        input_image = _tensor_frame_to_pil(video[:, 0])
        input_image.save(sample_dir / "input.png")

        save_mp4(gt_frames, str(sample_dir / "gt.mp4"), fps=int(vis.fps))
        save_mp4(pred_frames, str(sample_dir / "pred.mp4"), fps=int(vis.fps))
        side_by_side = []
        for t, (gt_frame, pred_frame) in enumerate(zip(gt_frames, pred_frames)):
            left = _label(gt_frame, f"GT t+{t}")
            right = _label(pred_frame, f"PRED t+{t}")
            side_by_side.append(
                Image.new("RGB", (left.width + right.width, left.height))
            )
            side_by_side[-1].paste(left, (0, 0))
            side_by_side[-1].paste(right, (left.width, 0))
        save_mp4(side_by_side, str(sample_dir / "side_by_side.mp4"), fps=int(vis.fps))

        record = {
            "sample_rank": rank,
            "dataset_index": idx,
            "prompt": sample.get("prompt"),
            "num_video_frames": num_video_frames,
            "num_action_steps": int(action.shape[0]),
            "frame_psnr": [float(v) for v in frame_psnr],
            "future_psnr": future_psnr,
            "action_mse": action_mse,
        }
        all_records.append(record)
        logger.info(
            "[%d/%d] idx=%d future_psnr=%.2f dB (frame0=%.2f) action_mse=%.4f -> %s",
            rank + 1,
            len(indices),
            idx,
            future_psnr,
            frame_psnr[0],
            action_mse,
            sample_dir,
        )

    summary = {
        "checkpoint": str(vis.ckpt),
        "dataset_stats": str(stats_path),
        "task": task_name,
        "num_samples": len(all_records),
        "num_inference_steps": num_inference_steps,
        "sigma_shift": sigma_shift,
        "seed": seed,
        "mean_future_psnr": float(np.mean([r["future_psnr"] for r in all_records])),
        "std_future_psnr": float(np.std([r["future_psnr"] for r in all_records])),
        "mean_action_mse": float(np.mean([r["action_mse"] for r in all_records])),
        "samples": all_records,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "done | mean future PSNR = %.2f dB (+/- %.2f) | mean action MSE = %.4f | %s",
        summary["mean_future_psnr"],
        summary["std_future_psnr"],
        summary["mean_action_mse"],
        out_dir / "metrics.json",
    )


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def _entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    _entry()
