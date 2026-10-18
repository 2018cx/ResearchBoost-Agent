#!/bin/bash
# apex with its CUDA/C++ extensions (pip --config-settings --build-option is ignored by the pyproject build,
# so the wheel is built directly with setup.py).
set -euo pipefail
# Run after install_extras.sh has put the apex source under $ASSETS/build/apex.
A=${ASSETS:?set ASSETS}; E=$A/env
export PATH=$E/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export MAX_JOBS=32 TORCH_CUDA_ARCH_LIST=9.0 NVCC_APPEND_FLAGS="--threads 4"
cd $A/build/apex
$E/bin/pip uninstall -y apex >/dev/null 2>&1 || true
rm -rf build dist
$E/bin/python setup.py bdist_wheel --cpp_ext --cuda_ext
$E/bin/pip install --no-deps --force-reinstall dist/apex-*.whl
$E/bin/python -c "import fused_adam_cuda, fused_layer_norm_cuda; print(\"apex cuda ext ok\")"
echo APEX_DONE
