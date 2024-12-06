#!/bin/bash
#SBATCH --job-name=run_xgb_job
#SBATCH --output=run_xgb_output.txt
#SBATCH --time=20:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --mem=64G

module load polars/0.15.6-foss-2022a
pip install --user imbalanced-learn
pip install --user shap xgboost
python /home/darosali/BP/FD.py \
    --tuning $1 \
    --learning_rate $2 \
    --n_estimators $3 \
    --max_depth $4 \
    --min_child_weight $5 \
    --subsample $6 \
    --colsample_bytree $7 \
    --gamma $8
