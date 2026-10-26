# Image Generation

[← All results](README.md) · [ResearchBoost-Agent](../../README.md#results)

3 projects · [RAE](#faster-rae) · [DMD2](#faster-dmd) · [JiT](#faster-jit)

## Faster-RAE

**Stage-1 decoder + GAN training**

| Item | Result |
|---|---|
| Speedup | **5.32×** |
| Hardware | 2 × H200 |
| Measurement scope | Decoder + GAN step time |
| Validation | Statistical equivalence |



### Further results

A later H200 benchmark reports **5.41×**. On **8 × H20** at global batch 512, step timing improves **3.22×**; a completed 16-epoch run achieves rFID **0.541**, compared with **0.569** for the public checkpoint.

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-RAE) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-RAE/blob/main/docs/h20-results.md)

## Faster-DMD

**DMD2 one-step SD1.5 distillation + GAN**

| Item | Result |
|---|---|
| Speedup | **1.72×** |
| Hardware | 4 × H20 |
| Measurement scope | Distillation + GAN |
| Validation | Statistical equivalence |



### Further results

A 2,000-iteration continuation on **8 × H20**, global batch 2,048, achieves **1.67×** speedup: 16.39 → 9.79 h of logged iteration time, excluding checkpoint writes. FID-30k is **8.7030**, compared with baseline **8.6536** (seed 10).

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-DMD) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-DMD/blob/main/docs/h20-results.md)

## Faster-JiT

**JiT-B/16 · ImageNet 256 training**

| Item | Result |
|---|---|
| Speedup | **1.56×** |
| Hardware | 2 × H200-class |
| Measurement scope | 100.02 → 64.04 ms/step |
| Validation | Statistical equivalence |

A separate 40-epoch run on 8 × H100 achieved FID-50K 39.24, compared with baseline 39.07 (one seed per arm).

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-JiT) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-JiT/blob/main/docs/training.md#validation)
