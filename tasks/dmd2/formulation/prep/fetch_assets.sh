#!/bin/bash
# Public weights, data and resume checkpoint for the DMD2 SDv1.5 bench, plus the Nsight Systems CLI.
#   ASSETS       the task's assets directory (written here; mounted read-only in the sandbox later)
#   HF_ENDPOINT  Hugging Face endpoint or mirror (default https://huggingface.co)
set -euo pipefail
A=${ASSETS:?set ASSETS}; HUB=${HF_ENDPOINT:-https://huggingface.co}
P="python3 $(cd "$(dirname "$0")" && pwd)/pget.py"
SD=$A/hf/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5; REV=451f4fe16113bff5a5d2269ed5ad43b0592e9a14
HF=$HUB/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/$REV
CN=laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch_fid9.28_checkpoint_model_039000
mkdir -p $SD/blobs $SD/snapshots/$REV $A/data/raw $A/data15 $A/ckpt15/$CN $A/nsight
mkdir -p $SD/refs && printf %s $REV > $SD/refs/main   # no trailing newline: huggingface_hub 0.22 does not strip it
small() {  # path
  mkdir -p $SD/snapshots/$REV/$(dirname $1)
  [ -s $SD/snapshots/$REV/$1 ] || curl -sSL -o $SD/snapshots/$REV/$1 $HF/$1
}
lfs() {    # path sha256
  mkdir -p $SD/snapshots/$REV/$(dirname $1)
  if [ ! -f $SD/blobs/$2 ]; then
    $P $HF/$1 $SD/blobs/$2.dl --threads 32 --chunk 16
    echo "$2  $SD/blobs/$2.dl" | sha256sum -c -
    mv $SD/blobs/$2.dl $SD/blobs/$2
  fi
  ln -sfn $SD/blobs/$2 $SD/snapshots/$REV/$1
}
echo "== sd15 config files"
for f in model_index.json scheduler/scheduler_config.json tokenizer/merges.txt tokenizer/special_tokens_map.json \
         tokenizer/tokenizer_config.json tokenizer/vocab.json text_encoder/config.json unet/config.json vae/config.json; do small $f; done
echo "== sd15 weights"
lfs unet/diffusion_pytorch_model.safetensors 19da7aaa4b880e59d56843f1fcb4dd9b599c28a1d9d9af7c1143057c8ffae9f1
lfs text_encoder/model.safetensors d008943c017f0092921106440254dbbe00b6a285f7883ec8ba160c3faad88334
lfs vae/diffusion_pytorch_model.safetensors a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c
echo "== dmd2 prompts, sd15 latents and the 39k resume checkpoint"
D=$HUB/tianweiy/DMD2/resolve/main
C=model/sdv1.5/$CN
$P $D/data/laion/captions_laion_score6.25.pkl $A/data/raw/captions_laion_score6.25.pkl --parts 16 &
$P $D/data/laion_vae_latents/sd_vae_latents_laion_500k_lmdb.zip $A/data15/sd_vae_latents_laion_500k_lmdb.zip --threads 40 --chunk 32 &
$P $D/$C/pytorch_model.bin $A/ckpt15/$CN/pytorch_model.bin --threads 24 --chunk 16 &
$P $D/$C/pytorch_model_1.bin $A/ckpt15/$CN/pytorch_model_1.bin --threads 24 --chunk 16 &
wait
echo "== nsight systems CLI"
if ! ls $A/nsight/opt/nvidia/nsight-systems/*/bin/nsys >/dev/null 2>&1; then
  DEB=nsight-systems-2025.6.3_2025.6.3.541-1_amd64.deb
  $P https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/$DEB $A/$DEB --parts 16
  dpkg -x $A/$DEB $A/nsight && rm -f $A/$DEB
fi
echo FETCH_DONE
