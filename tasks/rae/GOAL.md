# Goal

*Translated from the Chinese original the agent received.*

Speed up RAE Stage-1 training to **more than 4× the fp32 reference**, while passing the numerics checks below.

- Training code: `rae/`, upstream bytetriper/RAE at commit a4d18c4; this is where you make changes. The entry point is `src/train_stage1.py`, the configuration `configs/stage1/training/DINOv2-B_decXL.yaml`.
- The model has three parts: a frozen DINOv2-B encoder, the ViT-XL decoder being trained (with EMA), and a DINO-S/8 discriminator (GAN). The losses are L1, LPIPS and the adversarial loss.
- Reference: `targets/rae/reference`, the same commit unchanged, fp32, with PyTorch's default math-mode switches. The reference's weighted step time is about 1835 ms/step (2 GPUs, batch 64 per GPU).

## What counts as reaching the goal (the single criterion)

```
python bench/run.py check --tree rae --config <switch file> --out <output dir>
```

It writes `<output dir>/check.json`. **The goal is reached when `status` is `PASS` and `performance.speedup` ≥ 4.0.**

`check` applies the gates below in order and stops at the first that fails:

0. **Coverage and structure**:
   - everything in `main()` outside the per-step loop, and `parse_args()`, must be identical to the reference, compared by AST;
   - the first two batches from data loading must be bitwise identical;
   - two very short cases are recorded (one reconstruction step, one adversarial step), and their exact items (phase, call counts, random numbers, learning rate, cross-GPU sync) must match.
1. **Benefit**: reference–candidate–reference timing in three independent processes (64 per GPU, the three training phases weighted 6:2:8); the speedup must exceed max(1.02, 1 + 2 × relative gap between the two reference runs). `performance.speedup` is this weighted speedup.
2. **Bitwise**: if every tensor of the two gate-0 steps is bitwise identical, the candidate passes directly.
3. **Statistical**: if bitwise is not achieved, run paired-seed training trajectories of 300 steps and look at the loss at 5 endpoints:
   - start with 3 seeds: if any endpoint deviates by more than δ (twice the reference's own seed-to-seed fluctuation, frozen) and significantly, it fails; if all are within δ, it passes;
   - otherwise extend to 6 seeds and decide again.

Two other modes:
- `--quick`: gate 0 only, results in a minute or two;
- `--skip-timing`: skip timing and judge numerics only.

Time on its own with `python bench/run.py time --tree rae --config <switch file> --out <dir>`; for Nsight Systems use `python bench/run.py profile --tree rae --config <switch file> --phase all --out <dir>` (run `source $WORK/opt/nsight/env.sh` first). `bench/README.md` has the details.

## Constraints

- Do not change:
  - the parts of `main()` in `rae/src/train_stage1.py` outside the per-step loop, or `parse_args()`. Anything that has to happen during setup goes into the modules they call;
  - the batch of 64 per GPU, the data, the schedule or the hyperparameters.
- Switching the whole training step to bf16/fp16 (global autocast or equivalent) is not allowed. Local precision changes are allowed but must pass the checks.
- The candidate's switches (environment variables) go in the switch file, in the format `{"environment": {"X": "1"}}`. Variables the bench manages itself may not be set: `TMPDIR`, compile cache directories, `CUDA_VISIBLE_DEVICES` and so on.
- `bench/`, `numerics/`, `targets/`, `task/`, `../assets/` and `../cache/` are read-only. Data and weights cannot be downloaded.

## Environment

- 2 H200-class GPUs (sm_90, about 143 GB each); the GPU lock is `/tmp/rae_gpu01.lock`:
  - `bench/run.py` takes the lock itself; do not wrap it in flock;
  - wrap any other command that uses the GPUs as `flock /tmp/rae_gpu01.lock <command>`.
- Python: `$ASSETS/env/bin/python` (torch 2.8.0+cu128). Do not install packages into this environment; create a venv in the workspace if needed.
- No network (only the model API is reachable), no Nsight Compute. Space in `/tmp` is limited; delete what you no longer need.
- Start long jobs with `setsid nohup ... &` and wait for them inside the session.
