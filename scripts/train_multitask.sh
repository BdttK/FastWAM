#!/usr/bin/env bash
# Launch FastWAM multi-task fine-tuning (experiment 8).
#
# Usage:
#   bash scripts/train_multitask.sh "0,1,2,3,4,5,6,7,8,9,10" [extra hydra overrides...]
#
# The first argument is the comma-separated 0-indexed task list (see
# merge_multitask.py TASK_IDS). It is validated against the split manifest; run
# openpi/scripts/merge_multitask.py with the same --tasks first.
set -euo pipefail

TASKS="${1:?usage: train_multitask.sh <comma-separated task ids> [extra overrides...]}"
shift || true

W=/efs/shicheng/siyuan/robot-ws
MANIFEST="$W/data/RoboSynChallenge/cobotmagic_Real_multitask_split_manifest.json"
OUT="$W/checkpoints/fastwam_cobotmagic_multitask_20000_20261128"

source "$W/runtime/phase5_env.sh"
exec 9>"$W/runtime/fastwam_phase8.lock"
flock -n 9 || { echo "another FastWAM run holds the lock"; exit 20; }
cd "$W/repos/FastWAM"

if [[ -f "$OUT/TRAINING_COMPLETE" ]]; then
  echo "ALREADY_COMPLETE"; exit 22
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo 'GPU_BUSY: refusing to start over another GPU process'; exit 21
fi

# Validate --tasks against the manifest produced by merge_multitask.py.
python3 - "$TASKS" "$MANIFEST" <<'PY'
import json, sys
tasks = sorted(set(int(x) for x in sys.argv[1].split(",") if x.strip() != ""))
man = json.load(open(sys.argv[2]))
man_tasks = sorted(t["id"] for t in man["tasks"])
assert tasks == man_tasks, (
    f"--tasks {tasks} does not match manifest tasks {man_tasks}. "
    "Re-run merge_multitask.py with the same --tasks."
)
print(f"manifest OK: {len(tasks)} tasks, train={man['datasets']['train']['frames']} frames, val={man['datasets']['val']['frames']} frames")
PY

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RUN_ID=cobotmagic-multitask-20000 WANDB_RUN_ID=fwcm-mt-20000 WANDB_RESUME="${WANDB_RESUME:-allow}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

# Note: no data.*.pretrained_norm_stats override -- stats are computed from the
# merged train set on first run (configs/data/cobotmagic_multitask.yaml has
# pretrained_norm_stats: null), then the val set reuses <output_dir>/dataset_stats.json.
bash scripts/train_zero1.sh 8 task=cobotmagic_multitask \
  "output_dir=$OUT" \
  model.skip_dit_load_from_pretrain=true model.action_dit_pretrained_path=null \
  model.redirect_common_files=false model.mot_checkpoint_mixed_attn=false \
  batch_size=2 gradient_accumulation_steps=8 num_workers=4 \
  max_steps=20000 \
  wandb.enabled=true wandb.project=fast-wam wandb.workspace=2046065740-cuhksz \
  wandb.name=cobotmagic-multitask-20000-20261128 "$@"

touch "$OUT/TRAINING_COMPLETE"
