#!/bin/bash

# P4PxGMVAE training script

set -e

# config
INPUT_DATA="data/CLUESImmVar_nonorm.V6.h5ad"
PROCESSED_DIR="processed_data"
MODELS_DIR="models"
LOGS_DIR="logs"
EXP_NAME="p4pxgmvae"

# create directories
mkdir -p $PROCESSED_DIR
mkdir -p $MODELS_DIR
mkdir -p $LOGS_DIR

# check input data
if [ ! -f "$INPUT_DATA" ]; then
    echo "error: input data: $INPUT_DATA"
    exit 1
fi

echo "config:"
echo "  input: $INPUT_DATA"
echo "  processed data: $PROCESSED_DIR"
echo "  models: $MODELS_DIR"
echo "  logs: $LOGS_DIR"
echo "  experiment: $EXP_NAME"
echo

# data preprocessing
python scripts/1_data_preprocessing.py \
    --input $INPUT_DATA \
    --output $PROCESSED_DIR \
    --subsample 0.01 \
    --seed 42 \
    --min_cells 5 \
    --min_genes 200 \
    --max_genes 5000 \
    --mt_threshold 0.2

# GMVAE training
python scripts/2_train_gmvae.py \
    --data_dir $PROCESSED_DIR \
    --output $MODELS_DIR/gmvae_model.pth \
    --epochs 4000 \
    --batch_size 37 \
    --learning_rate 6e-6 \
    --latent_dim 32 \
    --hidden_dim 128 \
    --n_layers 2 \
    --zinb_loss \
    --seed 42 \
    --log_dir $LOGS_DIR/gmvae

# P4P classifier
python scripts/3_train_classifier.py \
    --data_dir $PROCESSED_DIR \
    --gmvae_model $MODELS_DIR/gmvae_model.pth \
    --output $MODELS_DIR/p4p_classifier.pth \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --n_proto 8 \
    --z_dim 32 \
    --h_dim 128 \
    --n_layers 2 \
    --lambda_1 0 \
    --lambda_2 0 \
    --lambda_3 0 \
    --lambda_4 0 \
    --lambda_5 0 \
    --lambda_6 1 \
    --test_step 1 \
    --split_ratio 0.7 0.15 0.15 \
    --seed 42 \
    --exp_str $EXP_NAME \
    --log_dir $LOGS_DIR/p4p \
    --device cpu
