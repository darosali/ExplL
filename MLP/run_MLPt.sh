#!/bin/bash
#SBATCH --job-name=mlp
#SBATCH --output=run_mlpT_output.txt
#SBATCH --time=20:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --mem=256G

module load Python/3.11.5-GCCcore-13.2.0
module load PyTorch/2.5.0-foss-2023b-CUDA-12.4.0
module load scikit-learn/1.3.2-gfbf-2023b
pip install --user pyarrow
pip install --upgrade polars
pip install --user shap matplotlib
pip install --upgrade pyarrow

HIDDEN_LAYERS="128 32"
BATCH_SIZE=32
EPOCHS=5
LEARNING_RATE=0.0001
WEIGHT_DECAY=0.01
ACTIVATION_FN="ReLU"

python /home/darosali/BP/MLP/FD_MLP.py \
    --hidden_layers $HIDDEN_LAYERS \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --activation_fn $ACTIVATION_FN
