#!/bin/bash

# E2E-CommVQ: End-to-End Key Codebook Training Script
# This script trains the Key codebook using gradient descent on a pretrained model

set -e  # Exit on error

# Default parameters
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="HuggingFaceFW/fineweb-edu"
OUTPUT_DIR="codebook_e2e_12bits_128group_21residuals"
QUANT_BITS=2  # 2-bit quantization (12 total bits: 6+6)
EPOCHS=100
LR=0.001
BATCH_SIZE=256
NUM_LAYERS=32
NUM_SAMPLES=1000000  # Number of samples to collect for training
MAX_SEQ_LENGTH=8192

# Parse command line arguments
function usage() {
    cat << EOF
Usage: bash train_e2e_key_codebook.sh [OPTIONS]

E2E Key Codebook Training Options:
  -m, --model MODEL          Model path or HuggingFace model name
                             (default: meta-llama/Llama-3.1-8B-Instruct)
  -d, --dataset DATASET      Dataset path or HuggingFace dataset name
                             (default: HuggingFaceFW/fineweb-edu)
  -o, --output_dir DIR       Output directory for codebook
                             (default: codebook_e2e_12bits_128group_21residuals)
  -b, --quant_bits BITS      Quantization bits (1 or 2)
                             (default: 2)
  --epochs EPOCHS            Number of training epochs
                             (default: 100)
  --lr LR                    Learning rate
                             (default: 0.001)
  --batch_size SIZE          Batch size
                             (default: 256)
  --num_layers N             Number of layers to train
                             (default: 32)
  --num_samples N            Number of samples to collect
                             (default: 1000000)
  --layer LAYER_ID           Train only specific layer (optional)
  -h, --help                 Show this help message

Examples:
  # Train all layers with default settings
  bash train_e2e_key_codebook.sh

  # Train with custom model and output directory
  bash train_e2e_key_codebook.sh -m /path/to/model -o my_codebook

  # Train single layer with custom hyperparameters
  bash train_e2e_key_codebook.sh --layer 0 --epochs 200 --lr 0.0005

  # Train 1-bit quantization
  bash train_e2e_key_codebook.sh -b 1 -o codebook_e2e_6bits_128group_21residuals
EOF
}

SINGLE_LAYER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--quant_bits)
            QUANT_BITS="$2"
            shift 2
            ;;
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
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --layer)
            SINGLE_LAYER="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

echo "========================================"
echo "E2E-CommVQ Key Codebook Training"
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output Directory: $OUTPUT_DIR"
echo "Quantization Bits: $QUANT_BITS"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "Batch Size: $BATCH_SIZE"
echo "Max Sequence Length: $MAX_SEQ_LENGTH"
echo "Number of Samples: $NUM_SAMPLES"

if [ -n "$SINGLE_LAYER" ]; then
    echo "Training Layer: $SINGLE_LAYER"
else
    echo "Training Layers: 0 to $((NUM_LAYERS-1))"
fi
echo "========================================"

# Step 1: Collect KV cache using the model
echo ""
echo "[Step 1/3] Collecting KV cache from model..."
echo "This will run the model on $NUM_SAMPLES samples and save KV cache to data/key/"

python collect_kv_for_e2e.py \
    --model_path "$MODEL" \
    --dataset "$DATASET" \
    --output_dir "data/key" \
    --num_samples "$NUM_SAMPLES" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --quant_bits "$QUANT_BITS"

if [ $? -ne 0 ]; then
    echo "Error: KV collection failed!"
    exit 1
fi

echo "[Step 1/3] ✓ KV cache collection completed"

# Step 2: Compute scaling factors
echo ""
echo "[Step 2/3] Computing scaling factors..."

python make_scale.py

if [ $? -ne 0 ]; then
    echo "Error: Scaling factor computation failed!"
    exit 1
fi

echo "[Step 2/3] ✓ Scaling factors computed"

# Step 3: Train codebook with E2E method
echo ""
echo "[Step 3/3] Training Key codebook with E2E method..."

mkdir -p "$OUTPUT_DIR"

if [ -n "$SINGLE_LAYER" ]; then
    # Train single layer
    echo "Training layer $SINGLE_LAYER..."
    python quantize_key_cache.py "$SINGLE_LAYER" \
        --training_method e2e \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch_size "$BATCH_SIZE"
    
    # Move to output directory
    mv "codebook_12bits_128group_21residuals/${SINGLE_LAYER}"_*.pt "$OUTPUT_DIR/" 2>/dev/null || true
else
    # Train all layers
    for ((layer=0; layer<NUM_LAYERS; layer++)); do
        echo "Training layer $layer/$((NUM_LAYERS-1))..."
        
        python quantize_key_cache.py "$layer" \
            --training_method e2e \
            --epochs "$EPOCHS" \
            --lr "$LR" \
            --batch_size "$BATCH_SIZE"
        
        if [ $? -ne 0 ]; then
            echo "Error: Training failed for layer $layer!"
            exit 1
        fi
        
        # Move completed layer to output directory
        mv "codebook_12bits_128group_21residuals/${layer}"_*.pt "$OUTPUT_DIR/" 2>/dev/null || true
    done
fi

echo "[Step 3/3] ✓ Key codebook training completed"

# Summary
echo ""
echo "========================================"
echo "E2E Training Completed Successfully!"
echo "========================================"
echo "Codebook saved to: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "1. Train Value codebook (if needed):"
echo "   bash finetune/llama3.1_8b_int2.sh"
echo ""
echo "2. Evaluate the model:"
echo "   cd evaluation/longbench"
echo "   python pred.py --model /path/to/model"
echo "========================================"

