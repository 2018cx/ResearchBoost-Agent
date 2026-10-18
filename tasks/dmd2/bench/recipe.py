"""The frozen workload: upstream's SDv1.5 DMD2 recipe (the published FID 8.35 configuration), with only the
single-machine 4-GPU change.

Upstream: experiments/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_
noode_resume_fixdata.sh, launched with `torchrun --nnodes 8 --nproc_per_node=8 main/train_sd.py` (DDP, no FSDP),
resuming from the published 39k from-scratch checkpoint. It is the SDv1.5 row of the DMD2 model zoo with the best
FID (8.35).

Here: one machine, 4 GPUs, same launcher (torchrun) and the same DDP setup. Per GPU everything is as upstream:
batch 32, 512x512 (latent 64), fp32 weights with bf16 autocast, gradient checkpointing, one-step generator
(conditioning timestep 999), a generator update every 10th iteration, the GAN classifier head on clean images with
diffusion-GAN noising, guidance scale 1.75. The global batch is 128 instead of 2048: upstream train_sd.py asserts
gradient_accumulation_steps == 1, so the batch cannot be accumulated without changing the training code.

Why SDv1.5 and not the SDXL 4-step configuration: upstream's SDXL recipe trains under FSDP, and that path does not
run on this machine on any installable torch -- upstream's pinned torch 2.0.1 crashes in flash-attention's backward
on H20 (sm_90, "CUDA error: an illegal instruction was encountered"), torch 2.1.2 crashes inside cuBLAS on a bf16
GEMM shape of SDXL's cross-attention, and torch >= 2.1 FSDP rejects the recipe's requires_grad toggling with
"Invalid: ``_post_backward_hook_state``" (upstream issue #41, unresolved). The SDv1.5 configuration uses DDP, which
has none of these problems, and keeps upstream's own parallel setup for this recipe.

Bench-managed arguments (fixture and scratch paths, checkpointing off): see `argv()`.
"""
from pathlib import Path

EXPERIMENT = 'laion6.25_sd_baseline_8node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_resume'
ENTRY = 'main/train_sd.py'
UPDATE_RATIO = 10           # --dfake_gen_update_ratio: a generator update every 10th iteration
WANDB_ITERS = 50
WORLD_SIZE = 4

# upstream argument list, in upstream order; {output}, {prompts}, {latents}, {ckpt} are filled in by argv()
UPSTREAM = [
    '--generator_lr', '5e-7',
    '--guidance_lr', '5e-7',
    '--train_iters', '100000000',
    '--output_path', '{output}',
    '--batch_size', '32',
    '--grid_size', '2',
    '--initialie_generator', '--log_iters', '1000',
    '--resolution', '512',
    '--latent_resolution', '64',
    '--seed', '10',
    '--real_guidance_scale', '1.75',
    '--fake_guidance_scale', '1.0',
    '--max_grad_norm', '10.0',
    '--model_id', 'stable-diffusion-v1-5/stable-diffusion-v1-5',
    '--train_prompt_path', '{prompts}',
    '--wandb_iters', str(WANDB_ITERS),
    '--wandb_entity', 'bench',
    '--wandb_project', 'dmd2',
    '--wandb_name', EXPERIMENT,
    '--use_fp16',
    '--log_loss',
    '--dfake_gen_update_ratio', str(UPDATE_RATIO),
    '--gradient_checkpointing',
    '--cls_on_clean_image',
    '--gen_cls_loss',
    '--gen_cls_loss_weight', '1e-3',
    '--guidance_cls_loss_weight', '1e-2',
    '--diffusion_gan',
    '--diffusion_gan_max_timestep', '1000',
    '--ckpt_only_path', '{ckpt}',
    '--real_image_path', '{latents}',
]
# upstream's model_id is "runwayml/stable-diffusion-v1-5", which no longer exists on the hub; the files come from
# the identical community mirror stable-diffusion-v1-5/stable-diffusion-v1-5 (same sha256 for unet, text encoder
# and VAE), which is also what `model_id` points at here.
UPSTREAM_MODEL_ID = 'runwayml/stable-diffusion-v1-5'


def argv(scratch, prompts, latents, ckpt, seed=None, warmup_step=None, iters=None, batch_size=None):
    """train_sd.py arguments for one run.

    scratch: a fresh directory for the run's output and log directories (upstream's defaults point at
    /mnt/localssd). Added by the bench: --log_path/--cache_dir inside scratch and --no_save (upstream saves a
    checkpoint every 1000 iterations, outside every window measured here).
    seed / warmup_step / iters / batch_size: numerics cases and trajectories only; timing runs and the bitwise
    cases use upstream's values (seed 10, warmup 500, batch 32). The statistical gate runs with a smaller per-GPU
    batch so that a few hundred paired iterations fit in the schedule; reference and candidate use the same one."""
    scratch = Path(scratch)
    out = [a.format(output=scratch / 'output', prompts=prompts, latents=latents, ckpt=ckpt) for a in UPSTREAM]
    out += ['--log_path', str(scratch / 'log'), '--cache_dir', str(scratch / 'cache'), '--no_save']
    if seed is not None:
        out[out.index('--seed') + 1] = str(seed)
    if warmup_step is not None:
        out += ['--warmup_step', str(warmup_step)]
    if iters is not None:
        out[out.index('--train_iters') + 1] = str(iters)
    if batch_size is not None:
        out[out.index('--batch_size') + 1] = str(batch_size)
    return out
