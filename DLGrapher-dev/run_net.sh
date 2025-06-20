#!/bin/sh

CUDA_VERSION="12.1"
CONDA_ROOT=$(dirname "$(dirname "$(which conda)")")

if command -v module > /dev/null
then
    module try-load cuda"$CUDA_VERSION"/toolkit
fi

cd "$(dirname "$0")" &&
. "$CONDA_ROOT"/etc/profile.d/conda.sh &&
conda activate .env/ &&
nvidia-smi &&
cd src &&
python main.py \
    dataset=net \
    +experiment=net \
    ++dataset.name="$1" \
    ++train.batch_size="$2" \
    ++general.wandb=disabled
