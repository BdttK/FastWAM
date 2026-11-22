# FastWAM 双臂 Piper 真机推理操作指南

用微调后的 FastWAM(5B 世界模型)做 CobotMagic 双臂 Piper 的真机推理。远程推理架构:
**GPU 服务器跑模型,Piper 主机只负责「采相机 + 读状态 + 发动作」**,两者靠 WebSocket 通信。

```
台式机 = Piper 主机(部署侧)                服务器 = GPU 推理机(模型侧)
─────────────────────────────            ─────────────────────────────
3×RealSense + 2×Piper 双臂                 FastWAM fork(BdttK/FastWAM)
  │ 观测(3 相机 + 32 维 state)            ──► fastwam_serve.py :8001
  │ ─────────── WebSocket ────────────────► 推理
  │ 动作(32 维) ◄───────────────────────────
  ▼ rollout_fastwam.py 抽关节+夹爪 → 驱动 Piper
```

**分工一句话**:台式机**不需要**模型权重;服务器**不需要**相机和机器人硬件。

---

## 一、从 HuggingFace 下载

### ① 微调权重(私有仓库,约 11.2 GB)

> 仓库:`Bdttk/cobotmagic-items-handover-fastwam-20261120`(私有),revision 固定为:
> `c527d568ea43205ea84ebc2c4f9e9cc0417cd56b`
> 内容 = `model.pt`(完整 MoT 权重)+ `dataset_stats.json`(归一化)+ `config.yaml`

```bash
huggingface-cli download Bdttk/cobotmagic-items-handover-fastwam-20261120 \
  --revision c527d568ea43205ea84ebc2c4f9e9cc0417cd56b \
  --local-dir <fastwam_repo>/checkpoints/fastwam_cobotmagic_400 \
  --token $HF_TOKEN
```

### ② Wan 基础组件(公开,serve 启动时自动下载)

`fastwam_serve.py` 启动时会自动下载以下组件到 `checkpoints/`(需要服务器有外网):

- Wan 2.2 **VAE**
- **umT5-XXL 文本编码器**(约 11 GB)
- umT5 分词器

下载位置 / 来源由环境变量控制:

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"   # 默认 ./checkpoints
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface            # 国内可改 modelscope
```

### ③ ActionDiT backbone(一次性预生成)

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

### ❌ 明确不需要的

| 不要 | 原因 |
|---|---|
| 训练 optimizer state / trainer_state | 部署包本来就不含 |
| 数据集 `RoboSynChallenge/...` | 推理只要 `dataset_stats.json` |
| `pi05_base` 等 openpi 权重 | 那是 openpi 的,与 FastWAM 无关 |

---

## 二、部署命令行

### 服务器(GPU 推理机)

```bash
cd <fastwam_repo>

python scripts/fastwam_serve.py \
  --ckpt <fastwam_repo>/checkpoints/fastwam_cobotmagic_400/model.pt \
  --stats <fastwam_repo>/checkpoints/fastwam_cobotmagic_400/dataset_stats.json \
  --port 8001
```

- `--ckpt` 指向下载的 `model.pt`,`--stats` 指向 `dataset_stats.json`。
- 看到 `serving native FastWAM on 0.0.0.0:8001` 说明起来了。
- **显存**:FastWAM 是 5B 模型,bf16 推理建议 **≥24GB 卡(3090/4090)**,40GB 更稳。
- **防火墙**:确保 Piper 主机能连到服务器 `8001` 端口。

### 台式机(Piper 主机)

部署客户端位于 **FastWAM-RL 仓库的 `deployment/` 目录**(`rollout_fastwam.py` + 硬件驱动),台式机需要装:

```bash
pip install -e <fastwam_repo> 2>/dev/null || true   # 若台式机需要 openpi-client 以外的依赖
pip install piper_sdk
pip install python-can
pip install pyrealsense2        # 或 opencv-python,按相机 backend
```

**① 先无真机冒烟(不碰机器人/相机)**

```bash
cd <FastWAM-RL_repo>
python deployment/rollout_fastwam.py --host <服务器IP> --dry-run
```

**② 真机**(先改好 `deployment/configs/cobotmagic_real.yaml`,急停装好、慢速)

```bash
python deployment/rollout_fastwam.py \
  --host <服务器IP> \
  --config deployment/configs/cobotmagic_real.yaml
```

---

## 三、关键约定与提醒

| 项 | FastWAM |
|---|---|
| 动作维度 | **32**(完整数据集维度,客户端 `extract_joint_command` 抽 14 维关节+夹爪) |
| 观测状态 | 32 维,**ee 位姿需要真实 FK**(FastWAM 真的用 ee) |
| 相机 | 3 路分开发,serve 拼成 `[384,320]` 网格(cam_high 上 + 左右腕下) |
| 策略 | **有状态**:chunking 在 `select_action` 内部,客户端每步调 `select_action` |
| 端口 | 8001(openpi/pi0.5 用 8000) |
| 控制频率 | 对齐数据集 10Hz |

上真机前必查:

1. 关节「绝对 vs 增量」——先发一帧固定观测,打印动作和当前关节对比。
2. FK 的 `rot6d` 约定必须和数据集转换脚本一致(见 `robots/cobotmagic_piper.py` 的 TODO)。
3. `--stats` 必须用**自己训练 run 的** `dataset_stats.json`,不是 robotwin 那份。



me:
cd /root/gk/FastWAM
conda activate fastwam
python scripts/fastwam_serve.py \
    --ckpt checkpoints/fastwam_cobotmagic_20261120/model.pt \
    --stats checkpoints/fastwam_cobotmagic_20261120/dataset_stats.json \ --task cobotmagic \
    --device cuda:0 \
    --host 0.0.0.0 --port 8001 \
    --num-inference-steps 10 \
    --replan-steps 8


zhuji:
ssh -o ProxyCommand='nc -X connect -x 127.0.0.1:7897 %h %p' -p 10500 -N \
  -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 \
  -o ServerAliveInterval=15 root@js4.blockelite.cn 


cd /home/agilex/gk/FastWAM-RL
python deployment/rollout_fastwam.py --host 127.0.0.1 --port 8001 \
  --config deployment/configs/cobotmagic_real.yaml \
  --episodes 1 --steps 0 --control-hz 10