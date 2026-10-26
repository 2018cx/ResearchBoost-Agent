# MLLM

[← All results](README.md) · [ResearchBoost-Agent](../../README.md#results)

3 projects · [LLaVA](#faster-llava) · [BAGEL](#faster-bagel) · [R3](#faster-r3)

## Faster-LLaVA

**LLaVA-1.5-7B stage-2 fine-tuning**

| Item | Result |
|---|---|
| Speedup | **2.39×** |
| Hardware | 4 × H20 |
| Measurement scope | Stage-2 fine-tuning |
| Validation | Statistical equivalence |



### Further results

After one epoch of fine-tuning, scores on four benchmarks differ from the released official weights by **−0.50 to +0.44 points**.

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-LLaVA) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-LLaVA/blob/main/docs/h20-results.md)

## Faster-BAGEL

**7B-MoT SFT · BF16 compute, no FP8**

| Item | Result |
|---|---|
| Speedup | **1.82×** |
| Hardware | 8 × Hopper 80GB |
| Measurement scope | 1.4367 → 0.7873 s/step |
| Validation | Statistical equivalence |

Validated over 2,000 steps × 4 seeds, with 2σ trajectory checks and held-out VLM cross-entropy evaluation.

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-BAGEL) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-BAGEL/blob/main/docs/judge.md)

## Faster-R3

**BAGEL 7B-MoT Tree-RL**

| Item | Result |
|---|---|
| Speedup | **2.08×** |
| Hardware | 8 × H100 80GB |
| Measurement scope | 24.72 → 11.87 s/RL step · rollout, GenEval reward service and optimizer |
| Validation | Bitwise identical · 14 deterministic steps on 8 ranks |

Measured over 16 steps, with 10 warmup steps and three paired timing windows. Recorded training items match bitwise over 14 deterministic steps on all eight ranks.

### Further results

The **2.08×** result includes GenEval reward-service improvements. With the optimized reward service on both sides, the training code alone achieves **1.88×**.

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-R3) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-R3/blob/main/docs/judge.md)
