#!/usr/bin/env python
"""Serve a LeRobot FastWAM policy over WebSocket for the Piper client.

Loads a FastWAM checkpoint via ``lerobot.policies.fastwam.FastWAMPolicy`` and
serves ``select_action`` / ``reset`` over a msgpack WebSocket, so the model can
run on the GPU machine while the Piper host streams observations remotely.

Message protocol (msgpack + msgpack-numpy):

    client -> server  {"type": "reset"}
    server -> client  {"ok": true}

    client -> server  {"type": "select_action",
                       "input_image": uint8 (C, H, 3W),   # 3 cams width-concatenated
                       "state":       float32 (32,),
                       "task":        str,
                       "seed":        int}
    server -> client  {"action": float32 (32,)}

Dependencies (FastWAM env):
    pip install "lerobot[fastwam]@git+https://github.com/huggingface/lerobot.git" \
                websockets msgpack msgpack-numpy

NOTE: this loads a **LeRobot-format** FastWAM checkpoint (``FastWAMPolicy.from_pretrained``),
i.e. one produced by ``lerobot-train --policy.type=fastwam`` (or a HF repo like
``ZibinDong/fastwam_robotwin_uncond_3cam_384``). A native ``yuantianyuan01/FastWAM``
``.pt`` checkpoint is a different format and needs conversion first.

Usage (GPU machine):
    python scripts/fastwam_serve.py --ckpt /path/to/fastwam_checkpoint --port 8001
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

import msgpack
import msgpack_numpy
from websockets.sync.server import serve

msgpack_numpy.patch()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fastwam_serve")


def load_policy(ckpt: str, device: str):
    try:
        from lerobot.policies.fastwam import FastWAMPolicy
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "LeRobot (with the [fastwam] extra) is required. Install it with: "
            "pip install 'lerobot[fastwam]@git+https://github.com/huggingface/lerobot.git'"
        ) from exc

    logger.info("loading FastWAM checkpoint from %s ...", ckpt)
    policy = FastWAMPolicy.from_pretrained(ckpt)
    policy.eval()
    policy.to(device)
    return policy


def make_handler(policy, device: str):
    """Return the per-connection WebSocket handler (one connection = one robot)."""

    def handler(ws) -> None:
        while True:
            msg = msgpack.unpackb(ws.recv(), raw=False)
            t = msg.get("type")

            if t == "reset":
                policy.reset()
                ws.send(msgpack.packb({"ok": True}))

            elif t == "select_action":
                # uint8 (C,H,W) -> float32 [0,1]; the policy resizes + rescales internally.
                img = torch.from_numpy(msg["input_image"].astype(np.float32)) / 255.0
                state = torch.from_numpy(msg["state"].astype(np.float32)).unsqueeze(0)

                batch = {
                    "input_image": img.to(device),
                    "observation.state": state.to(device),
                    "task": msg["task"],
                    "seed": int(msg.get("seed", 0)),
                }
                with torch.no_grad():
                    action = policy.select_action(batch)  # (1, action_dim)
                ws.send(msgpack.packb({"action": action[0].detach().cpu().numpy().astype(np.float32)}))

            else:
                ws.send(msgpack.packb({"error": f"unknown message type {t!r}"}))

    return handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="LeRobot-format FastWAM checkpoint path / HF repo id")
    p.add_argument("--device", default=None, help="e.g. cuda:0 (default: cuda if available)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.ckpt, device)
    logger.info("FastWAM policy loaded on %s", device)

    with serve(make_handler(policy, device), args.host, args.port) as server:
        logger.info("serving on %s:%d", args.host, args.port)
        server.serve_forever()


if __name__ == "__main__":
    main()
