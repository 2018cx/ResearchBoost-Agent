#!/bin/bash
# Public weights and the instruction json into $ASSETS. Pinned revisions; sha256 checked (LFS files).
#   ASSETS       the task's assets directory
#   HF_ENDPOINT  Hugging Face endpoint or mirror (default https://huggingface.co)
set -euo pipefail
A=${ASSETS:?set ASSETS}; M=$A/models; D=$A/data; H=${HF_ENDPOINT:-https://huggingface.co}
PDL="python3 $(cd "$(dirname "$0")" && pwd)/pdl.py"
small() { mkdir -p "$2"; [ -s "$2/$(basename $1)" ] || curl -sS -L --retry 5 -o "$2/$(basename $1)" "$1"; }
V=lmsys/vicuna-7b-v1.5; VR=3321f76e3f527bd14065daf69dad9344000a201d
C=openai/clip-vit-large-patch14-336; CR=ce19dc912ca5cd21c8a653c79e251e808ccabcd1
P=liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5; PR=5414da88308e4287a29f2e9609256458afb0a981
J=liuhaotian/LLaVA-Instruct-150K; JR=9d451dc7629cfe0469f6ae4432b765cd603d5fcb
echo "== small"
for f in config.json generation_config.json pytorch_model.bin.index.json special_tokens_map.json tokenizer_config.json tokenizer.model; do small $H/$V/resolve/$VR/$f $M/vicuna-7b-v1.5; done
for f in config.json preprocessor_config.json merges.txt special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json; do small $H/$C/resolve/$CR/$f $M/clip-vit-large-patch14-336; done
for f in config.json mm_projector.bin trainer_state.json; do small $H/$P/resolve/$PR/$f $M/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5; done
echo "== json"
$PDL $H/datasets/$J/resolve/$JR/llava_v1_5_mix665k.json $D/llava_v1_5_mix665k.json --parts 8 --sha256 ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8
echo "== clip"
$PDL $H/$C/resolve/$CR/pytorch_model.bin $M/clip-vit-large-patch14-336/pytorch_model.bin --parts 12
echo "== vicuna"
$PDL $H/$V/resolve/$VR/pytorch_model-00002-of-00002.bin $M/vicuna-7b-v1.5/pytorch_model-00002-of-00002.bin --parts 12
$PDL $H/$V/resolve/$VR/pytorch_model-00001-of-00002.bin $M/vicuna-7b-v1.5/pytorch_model-00001-of-00002.bin --parts 16
echo "== sha"
cd $M && find . -type f ! -name '*.ok' | sort | xargs sha256sum > $A/models.sha256
cat $A/models.sha256
echo "== DL_DONE"
