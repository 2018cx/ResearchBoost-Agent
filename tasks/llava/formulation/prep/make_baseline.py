"""Write ws/task/baseline.md from the frozen reference timing, the noise-floor run and the profile output.
make_baseline.py  (run after `formulate.sh timing` and a reference profile in task/profile)"""
import os
import json
import statistics
from pathlib import Path

WS = Path(os.environ['WORK']) / 'llava' / 'ws'
T = WS / 'task/timing'
P = WS / 'task/profile'


def read(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


a, b = read(T / 'time_ref_a.json'), read(T / 'time_ref_b.json')
cfg = read(WS / 'bench/bench.json')
prof_md = sorted(P.glob('profile_*.md'))
lines = ['# Baseline (reference, 4 × H20, upstream recipe unchanged)', '']
if a and b:
    ms = [a['ms_per_step'], b['ms_per_step']]
    ref = statistics.fmean(ms)
    lines += [
        f'- Workload: {cfg["per_rank_batch"]} per GPU × gradient accumulation {cfg["grad_accum"]} × {cfg["world_size"]} GPUs = global '
        f'{cfg["per_rank_batch"] * cfg["grad_accum"] * cfg["world_size"]}, model_max_length 2048, bf16 + tf32, '
        f'ZeRO-3, flash-attn 2, gradient checkpointing',
        f'- Timing plan: {cfg["timing"]["warmup"]} warmup steps, {cfg["timing"]["repeats"]} windows of '
        f'{cfg["timing"]["window"]} steps each',
        f'- **{ref:.0f} ms per step** (two reference runs {ms[0]:.0f} / {ms[1]:.0f} ms, relative gap '
        f'{abs(ms[0] - ms[1]) / ref:.2%})',
        f'- Throughput {a["samples_per_s"]:.2f} samples/s (all 4 GPUs)',
        f'- Peak memory {max(a["peak_memory_allocated_gib"], b["peak_memory_allocated_gib"]):.1f} GiB allocated / '
        f'{max(a["peak_memory_reserved_gib"], b["peak_memory_reserved_gib"]):.1f} GiB reserved (largest GPU)',
        f'- {a["startup_to_first_step_s"]:.0f} s from startup to the first step (loading the 7B weights into ZeRO-3 + CLIP + projector)',
        f'- Median GPU utilization inside the windows {a.get("gpu_util_median")}%, average power of the 4 GPUs {a.get("avg_power_w_all_gpus")} W',
        f'- Windows of the two reference runs (s): {a["window_seconds"]} / {b["window_seconds"]}',
        '',
    ]
for md in prof_md:
    lines += [f'## Profile summary ({md.name})', '', md.read_text().strip(), '']
(WS / 'task/baseline.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines[:14]))
