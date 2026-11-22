#!/usr/bin/env python
"""Serve a NATIVE FastWAM checkpoint over WebSocket for the Piper client.

Loads a native FastWAM ``.pt`` checkpoint (from ``train_zero1.sh task=cobotmagic``)
and serves action inference over a msgpack WebSocket, so the 5B video/action
model runs on the GPU machine while the Piper host streams observations.

This uses the **native** FastWAM action-inference path (``model.infer_action``,
first-frame / Fast-WAM mode) — no lerobot dependency. The 3 cameras are
concatenated into the native ``robotwin`` grid ``[384, 320, 3]``:
cam_high on top (256x320), cam_left_wrist | cam_right_wrist on the bottom
(128x160 each).

Message protocol (msgpack + msgpack-numpy):

    client -> server  {"type": "reset"}
    server -> client  {"ok": true}

    client -> server  {"type": "select_action",
                       "images": {"cam_high": uint8 (H,W,3),
                                  "cam_left_wrist": uint8 (H,W,3),
                                  "cam_right_wrist": uint8 (H,W,3)},
                       "state":  float32 (32,),
                       "task":   str}
    server -> client  {"action": float32 (32,)}   # one denormalized action step

Usage (GPU machine, FastWAM env):
    python scripts/fastwam_serve.py \
        --ckpt runs/cobotmagic/<run_id>/checkpoints/weights/<step>.pt \
        --stats runs/cobotmagic/<run_id>/dataset_stats.json \
        --port 8001

Optional future-imagination dump: `infer_action` never runs the video denoiser,
so add `--visualize-future` to also call `infer_joint` and write the imagined
next ~1s (9 frames) as MP4. This adds one video-denoising pass per
`--future-every` chunks; use e.g. `--future-every 5` to keep latency low:

    python scripts/fastwam_serve.py ... \
        --visualize-future --future-dir ./evaluate_results/live_future \
        --future-every 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import msgpack
import msgpack_numpy
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.video_io import save_mp4

msgpack_numpy.patch()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fastwam_serve")


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Resize an (H, W, 3) uint8 RGB image to (W, H), preserving uint8."""
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    return np.asarray(pil.resize(size_wh, resample=Image.BILINEAR), dtype=np.uint8)


def _compose_config(task: str):
    """Compose `train.yaml` with `task=<name>` (gives model + data/processor config)."""
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        return compose(config_name="train", overrides=[f"task={task}"])


class NativeFastWAMPolicy:
    """Wraps the native FastWAM model for per-step action inference (Piper)."""

    def __init__(
        self,
        cfg,
        checkpoint_path: str,
        dataset_stats_path: str,
        device: str,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: float | None,
        seed: int | None,
        text_cfg_scale: float,
        negative_prompt: str,
        visualize_future: bool = False,
        future_dir: str | None = None,
        future_every: int = 1,
        future_num_inference_steps: int | None = None,
        future_fps: int = 8,
    ):
        model_cfg = cfg.model
        model_cfg.load_text_encoder = True
        model_dtype = torch.bfloat16

        logger.info("building native FastWAM model ...")
        self.model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        logger.info("loading FastWAM processor + norm stats ...")
        self.processor = instantiate(cfg.data.train.processor).eval()
        stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)

        self.pending: deque[np.ndarray] = deque()

        self.visualize_future = bool(visualize_future)
        self.future_dir = Path(future_dir) if future_dir else None
        self.future_every = max(1, int(future_every))
        self.future_num_inference_steps = (
            int(num_inference_steps)
            if future_num_inference_steps is None
            else int(future_num_inference_steps)
        )
        self.future_fps = int(future_fps)
        self._chunk_index = 0
        # 33 frames / ratio 4 -> 9 video frames, matching the training config.
        self.num_video_frames = (
            (int(cfg.data.train.num_frames) - 1)
            // int(cfg.data.train.action_video_freq_ratio)
            + 1
        )
        if self.visualize_future:
            if self.future_dir is None:
                raise ValueError("`visualize_future=True` requires a future_dir.")
            self.future_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "future visualization ON | dir=%s | every=%d chunks | video_frames=%d | video_steps=%d",
                self.future_dir,
                self.future_every,
                self.num_video_frames,
                self.future_num_inference_steps,
            )

        logger.info("native FastWAM ready | ckpt=%s | horizon=%d | replan=%d",
                    checkpoint_path, self.action_horizon, self.replan_steps)

    # ------------------------------------------------------------------ #
    # Model-side transforms
    # ------------------------------------------------------------------ #
    def _build_grid(self, images: dict[str, np.ndarray]) -> torch.Tensor:
        """3 cameras -> native robotwin grid [1, 3, 384, 320] in [-1, 1]."""
        head = _resize_rgb(images["cam_high"], (320, 256))          # 256x320
        left = _resize_rgb(images["cam_left_wrist"], (160, 128))    # 128x160
        right = _resize_rgb(images["cam_right_wrist"], (160, 128))  # 128x160
        bottom = np.concatenate([left, right], axis=1)              # 128x320
        grid = np.concatenate([head, bottom], axis=0)               # 384x320x3

        tensor = torch.from_numpy(grid).permute(2, 0, 1).unsqueeze(0)
        return (tensor.to(device=self.model.device, dtype=self.model.torch_dtype) * (2.0 / 255.0)) - 1.0

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_key = self.processor.shape_meta["state"][0]["key"]
        batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        action_key = self.processor.shape_meta["action"][0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()  # [B, T, D]

    def _infer_chunk(self, images: dict[str, np.ndarray], state: np.ndarray, task: str) -> np.ndarray:
        image_tensor = self._build_grid(images)
        proprio = self._normalize_state(state)
        prompt = DEFAULT_PROMPT.format(task=task)

        with torch.no_grad():
            pred = self.model.infer_action(
                prompt=prompt,
                input_image=image_tensor,
                action_horizon=self.action_horizon,
                proprio=proprio,
                negative_prompt=self.negative_prompt,
                text_cfg_scale=self.text_cfg_scale,
                num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift,
                seed=self.seed,
                rand_device="cpu",
            )
        self._maybe_save_future(image_tensor, proprio, prompt)
        self._chunk_index += 1
        return self._denormalize_action(pred["action"])[0]  # [T, D]

    def _maybe_save_future(self, image_tensor: torch.Tensor, proprio: torch.Tensor, prompt: str) -> None:
        """Run the joint video+action denoiser and dump the imagined future.

        `infer_action` itself is action-only: it pre-fills the video KV cache
        with the current frame and never denoises future latents.  This helper
        is the only place the future is actually generated, and it is therefore
        opt-in -- it costs one extra (small-canvas) video diffusion pass.
        """
        if not self.visualize_future:
            return
        if self._chunk_index % self.future_every != 0:
            return

        with torch.no_grad():
            out = self.model.infer_joint(
                prompt=prompt,
                input_image=image_tensor,
                num_video_frames=self.num_video_frames,
                action_horizon=self.action_horizon,
                proprio=proprio,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.future_num_inference_steps,
                sigma_shift=self.sigma_shift,
                seed=self.seed,
                rand_device="cpu",
                # Control comes from infer_action; skip its duplicate action pass.
                test_action_with_infer_action=False,
            )

        stem = self.future_dir / f"chunk_{self._chunk_index:06d}"
        save_mp4(out["video"], str(stem.with_suffix(".mp4")), fps=self.future_fps)

        # Save the conditioning frame next to the video for reference.
        frame = image_tensor[0].detach().float().clamp(-1, 1).permute(1, 2, 0).cpu().numpy()
        Image.fromarray(((frame + 1.0) * 127.5).astype(np.uint8)).save(stem.parent / f"{stem.name}_input.png")

        logger.info("saved imagined future: %s.mp4", stem)

    # ------------------------------------------------------------------ #
    # Policy API (matches the client protocol)
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.pending.clear()

    def select_action(self, images: dict[str, np.ndarray], state: np.ndarray, task: str) -> np.ndarray:
        if not self.pending:
            chunk = self._infer_chunk(images, state, task)
            for i in range(min(self.replan_steps, chunk.shape[0])):
                self.pending.append(np.asarray(chunk[i], dtype=np.float32))
        return self.pending.popleft()


def make_handler(policy: NativeFastWAMPolicy):
    def handler(ws) -> None:
        try:
            while True:
                msg = msgpack.unpackb(ws.recv(), raw=False)
                t = msg.get("type")

                if t == "reset":
                    policy.reset()
                    ws.send(msgpack.packb({"ok": True}))

                elif t == "select_action":
                    action = policy.select_action(msg["images"], msg["state"], msg["task"])
                    ws.send(msgpack.packb({"action": action}))

                else:
                    ws.send(msgpack.packb({"error": f"unknown message type {t!r}"}))
        except ConnectionClosed:
            logger.info("client disconnected")

    return handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="native FastWAM .pt checkpoint")
    p.add_argument("--stats", required=True, help="dataset_stats.json from the training run")
    p.add_argument("--task", default="cobotmagic", help="task config name (default cobotmagic)")
    p.add_argument("--device", default=None, help="e.g. cuda:0 (default: cuda if available)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--action-horizon", type=int, default=None, help="default: num_frames-1 from config")
    p.add_argument("--replan-steps", type=int, default=8)
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--sigma-shift", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--text-cfg-scale", type=float, default=1.0)
    p.add_argument("--negative-prompt", default="")
    p.add_argument(
        "--visualize-future",
        action="store_true",
        help="also run infer_joint and dump the imagined future MP4s",
    )
    p.add_argument(
        "--future-dir",
        default="./evaluate_results/live_future",
        help="directory for imagined-future videos (default: %(default)s)",
    )
    p.add_argument(
        "--future-every",
        type=int,
        default=1,
        help="dump a future every N action chunks (default: %(default)s)",
    )
    p.add_argument(
        "--future-num-inference-steps",
        type=int,
        default=None,
        help="video denoising steps for the future dump (default: same as --num-inference-steps)",
    )
    p.add_argument("--future-fps", type=int, default=8)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = _compose_config(args.task)
    action_horizon = args.action_horizon or int(cfg.data.train.num_frames) - 1

    policy = NativeFastWAMPolicy(
        cfg=cfg,
        checkpoint_path=args.ckpt,
        dataset_stats_path=args.stats,
        device=device,
        action_horizon=action_horizon,
        replan_steps=args.replan_steps,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        text_cfg_scale=args.text_cfg_scale,
        negative_prompt=args.negative_prompt,
        visualize_future=args.visualize_future,
        future_dir=args.future_dir,
        future_every=args.future_every,
        future_num_inference_steps=args.future_num_inference_steps,
        future_fps=args.future_fps,
    )

    # Clients stream raw camera frames, which easily exceed the 1 MiB default.
    with serve(make_handler(policy), args.host, args.port, max_size=None) as server:
        logger.info("serving native FastWAM on %s:%d", args.host, args.port)
        server.serve_forever()


if __name__ == "__main__":
    main()
