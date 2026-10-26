# LLM

[← All results](README.md) · [ResearchBoost-Agent](../../README.md#results)

2 projects · [NanoChat](#faster-nanochat) · [YOCO](#faster-yoco)

## Faster-NanoChat

**D24 (~1.384B) · Tensorwise FP8 training**

| Item | Result |
|---|---|
| Speedup | **1.0744×** |
| Hardware | 8 × H20 |
| Measurement scope | 3,358.4 → 3,126.0 ms/update |
| Validation | N1 statistical equivalence |

Steady-state timing after compilation and warmup; six-seed short-trajectory check passed.

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-NanoChat) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-NanoChat/blob/main/docs/training.md)

## Faster-YOCO

**YOCO-3B pretraining · BF16, no FP8**

| Item | Result |
|---|---|
| Speedup | **2.12×** |
| Hardware | 8 × H100 80GB |
| Measurement scope | Training update speedup |
| Validation | Statistical equivalence |

A 20,000-update run achieved 1.91× end-to-end speedup (5,345 → 2,799 s), including startup and saves.

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-YOCO) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-YOCO/blob/main/docs/judge.md)
