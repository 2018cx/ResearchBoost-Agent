#!/bin/bash
# Fixture for the bench (run after fetch_assets.sh): upstream's full training prompt file and upstream's own SDv1.5 real-image latent LMDB
# (tianweiy/DMD2 data/laion_vae_latents/sd_vae_latents_laion_500k_lmdb.zip), unpacked as is.
set -euo pipefail
A=${ASSETS:?set ASSETS}
cp -n $A/data/raw/captions_laion_score6.25.pkl $A/data/captions_laion_score6.25.pkl 2>/dev/null || true
if [ ! -f $A/data15/sd_vae_latents_laion_500k_lmdb/data.mdb ]; then
  cd $A/data15 && unzip -q -o sd_vae_latents_laion_500k_lmdb.zip && rm -f sd_vae_latents_laion_500k_lmdb.zip
fi
ls -la $A/data15/sd_vae_latents_laion_500k_lmdb $A/ckpt15
sha256sum $A/ckpt15/*.bin > $A/ckpt15/sha256.txt
cat $A/ckpt15/sha256.txt
echo "== FIXTURE_DONE"
