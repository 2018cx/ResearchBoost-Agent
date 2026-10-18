# Speedup ceiling within the rules (for people only; never part of any message to Codex)

Global batch 128, about 850 tokens per sample on average (including 576 image tokens) ≈ 109k tokens per step; 6 × 7e9 × 1.09e5 × 4/3 (recomputation) ≈ 6.1e15 FLOP.
21.0 s / 4 GPUs ≈ 73 TFLOPS per GPU, against an H20 bf16 peak of 148 → already about 50% MFU. Add about 26% communication and the extra third of forward
work from gradient checkpointing: even with communication at 0 and no recomputation, the ceiling is about 1.75–1.9×. So 2× sits essentially at the ceiling, and 1.5× is a realistic finish line.
This reasoning is "the answer" and must not appear in any message to Codex (hotspot shares are visible in its own profiles, so those are not a leak).

## Recomputed (the 1.75–1.9× estimate above was wrong; 2.0297× disproved it)

**What was wrong with the old estimate**: it used the rough 6N model at 73 TFLOPS per GPU, and it assumed that (a) the 37% padding waste stays and (b) gradient
checkpointing recomputes everything. The candidate removed both, so it crossed my line. Lesson: a ceiling estimate must treat what the formulator allows to change
as changeable too.

**New model** (the real configuration + the 128 real samples of the first step):
- 32 layers, hidden 4096, intermediate 11008, vocab 32000 → 6.476 G linear-layer parameters + 131 M head;
- 128 samples with 34665 valid tokens, each image placeholder expanded by 575 → **108265 tokens per step**;
- matmul FLOP per token = fwd 13.21 + bwd 26.43 + recomputation (32 − keep)/32 × 13.21; keeping 17 layers = **45.71 GFLOP**;
- **4.949 PFLOP** per step.

**Two independent measurements agree (so the model is right)**:
- reference 20934 ms, computing on 172400 padded slots with full recomputation of all 32 layers → 9.11 PFLOP → **108.8 TFLOPS per GPU = 73.5% MFU**;
- candidate r58 10314 ms → 4.949 PFLOP → **120.0 TFLOPS per GPU = 81% MFU** (H20 bf16 nominal 148 TFLOPS).

**Ceiling within bf16**:
- 3× = 6978 ms per step needs **177 TFLOPS per GPU = 120% of peak** → arithmetically impossible;
- even keeping all 32 layers' activations with zero recomputation (memory far beyond 95 GiB, not feasible): 4.292 PFLOP → 154 TFLOPS per GPU = **104% of peak**, still impossible;
- 100% MFU + zero recomputation = **2.89×** (absolute ceiling); 85–90% MFU + 17 retained layers ≈ **2.25–2.4×** (realistic ceiling).
- So r58's 2.0297× is already within 10–15% of the realistic ceiling. What is left to squeeze: pushing the number of recomputed layers further down (memory-bound),
  overlapping the 5.1% ReduceScatter, index select/func at 3.3%, and pushing GEMMs from 81% MFU towards 85–90%.

**The only thing that raises the ceiling is FP8** (H20 FP8 peak 296 TFLOPS; 3× needs only 60% of the FP8 peak). The owner was asked and
**chose not to change the precision rule**: line 184 of the task statement, "training may not be lowered to FP8 / INT8 / FP16", stays as is; the main target is **2.5×**,
with a wrap-up floor of **2.0297×** (no regression allowed).

None of this derivation **may appear in any message to Codex**, nor in ws/ (hotspot shares are visible in its own profiles, so those are not a leak; the ceiling numbers count as the answer).
