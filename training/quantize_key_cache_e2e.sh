#!/bin/bash

# E2E-CommVQ: End-to-End training script for Key cache quantization
# This script trains the codebook using gradient descent instead of EM algorithm

# Default parameters
EPOCHS=100
LR=0.001
BATCH_SIZE=256
NUM_LAYERS=32

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --layer)
            SINGLE_LAYER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "===== E2E-CommVQ Training Configuration ====="
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Batch size: $BATCH_SIZE"
echo "Number of layers: $NUM_LAYERS"
echo "============================================="

# Train single layer or all layers
if [ -n "$SINGLE_LAYER" ]; then
    echo "Training layer $SINGLE_LAYER with E2E method..."
    python quantize_key_cache.py $SINGLE_LAYER \
        --training_method e2e \
        --epochs $EPOCHS \
        --lr $LR \
        --batch_size $BATCH_SIZE
else
    # Train all layers
    for ((layer=0; layer<$NUM_LAYERS; layer++)); do
        echo "Training layer $layer/$NUM_LAYERS with E2E method..."
        python quantize_key_cache.py $layer \
            --training_method e2e \
            --epochs $EPOCHS \
            --lr $LR \
            --batch_size $BATCH_SIZE
        
        if [ $? -ne 0 ]; then
            echo "Error training layer $layer. Stopping."
            exit 1
        fi
    done
fi

echo "E2E training completed successfully!"

