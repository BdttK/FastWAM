"""Fixed 80-clip held-out validation for FastWAM (phase 7 / 20k retrain).

Protocol identical to experiment 5 so curves stay comparable:
  * 10 held-out episodes x 8 non-tail windows = 80 clips, same windows every run.
  * Deterministic noise: seed = 420000 + frame_index, no image augmentation.
  * Records `validation_history.jsonl` and `best_checkpoint.json`, and saves the
    current-best deploy weights as `best_step_<step>.pt` (previous best is removed
    to bound disk).

No early stopping: training runs to `max_steps`; the best point is chosen
afterwards from the recorded curve.
"""
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def evaluate_fixed(self):
    model = self.accelerator.unwrap_model(self.model)
    previous = model.dit.training
    model.eval()
    bounds = self.val_dataset.lerobot_dataset.episode_data_index
    indices = []
    for start, end in zip(bounds['from'], bounds['to']):
        # Avoid tail replication for selection; same eight windows per episode.
        indices.extend(np.linspace(int(start), max(int(start), int(end)-33), 8, dtype=int).tolist())
    assert len(indices) == 80 and len(set(indices)) == 80
    rank = self.accelerator.process_index
    world = self.accelerator.num_processes
    totals = torch.zeros(2, device=self.accelerator.device, dtype=torch.float64)
    details = {}
    np_state, py_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=[self.accelerator.device]):
            for idx in indices[rank::world]:
                seed = 420000 + idx
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                sample = self._to_batched_eval_sample(self.val_dataset[idx])
                with self.accelerator.autocast():
                    loss, parts = model.training_loss(sample)
                value = float(loss.detach().float().item())
                if not math.isfinite(value):
                    raise FloatingPointError(f'Nonfinite validation loss at step {self.global_step}, frame {idx}')
                totals[0] += value
                totals[1] += 1
                for key, val in (parts or {}).items():
                    val = float(val.detach().float().item()) if isinstance(val, torch.Tensor) else float(val)
                    details[key] = details.get(key, 0.0) + val
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        if previous:
            self._apply_dit_only_train_mode(model)
    totals = self.accelerator.reduce(totals, reduction='sum')
    result = {'val_loss': float((totals[0]/totals[1]).item()), 'clips': int(totals[1].item())}
    for key in sorted(details):
        value = torch.tensor(details[key], device=self.accelerator.device, dtype=torch.float64)
        result[key] = float((self.accelerator.reduce(value, reduction='sum')/totals[1]).item())
    folder = Path(self.output_dir)
    bestfile = folder/'best_checkpoint.json'
    best = json.loads(bestfile.read_text()) if bestfile.exists() else {'validation_loss': float('inf'), 'step': 0}
    improved = result['val_loss'] < best['validation_loss']
    self.accelerator.wait_for_everyone()
    if self.accelerator.is_main_process:
        record = dict(step=self.global_step, **result, improved=improved)
        with (folder/'validation_history.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        if improved:
            path = Path(self.weights_dir)/f'best_step_{self.global_step:06d}.pt'
            temp = path.with_suffix('.pt.incomplete')
            model.save_checkpoint(str(temp), optimizer=None, step=self.global_step)
            os.replace(temp, path)
            # Phase 7: keep only the current-best deploy weight to bound disk.
            prev_weights = best.get('weights_path')
            if prev_weights and Path(prev_weights).exists() and Path(prev_weights).resolve() != path.resolve():
                Path(prev_weights).unlink()
            best = dict(step=self.global_step, validation_loss=result['val_loss'], weights_path=str(path),
                        dataset_stats=str(folder/'dataset_stats.json'),
                        protocol='fixed_80_clips_seed_420000_plus_frame_index',
                        validation_indices=indices)
            temporary = bestfile.with_suffix('.tmp')
            temporary.write_text(json.dumps(best, indent=2)+'\n')
            os.replace(temporary, bestfile)
        print('[fixed_validation] '+json.dumps(record), flush=True)
    self.accelerator.wait_for_everyone()
    best = json.loads(bestfile.read_text())
    return result
