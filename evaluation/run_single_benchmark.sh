#!/bin/bash

################################################################################
# 단일 벤치마크 평가 스크립트
# 
# 사용법:
#   bash run_single_benchmark.sh <benchmark> <model_path>
#
# 벤치마크 옵션:
#   - longbench      : LongBench (16 datasets)
#   - niah           : Needle in a Haystack
#   - memory         : Memory Measurement
#   - infinitebench  : InfiniteBench Passkey
#   - all            : 모든 벤치마크 실행
#
# 예시:
#   bash run_single_benchmark.sh longbench /path/to/model
#   bash run_single_benchmark.sh all meta-llama/Llama-3.1-8B-Instruct
################################################################################

set -e

BENCHMARK=${1:-"all"}
MODEL_PATH=${2:-"meta-llama/Llama-3.1-8B-Instruct"}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_${BENCHMARK}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo "Running: ${BENCHMARK}"
echo "Model:   ${MODEL_PATH}"
echo "Results: ${RESULTS_DIR}"
echo "============================================================"
echo ""

run_longbench() {
    echo "[LongBench] Starting evaluation..."
    cd longbench
    
    # Add model to config if it's a custom path
    if [[ "${MODEL_PATH}" != "meta-llama/"* ]] && [[ "${MODEL_PATH}" != "mistralai/"* ]]; then
        echo "  Adding custom model to config..."
        python << EOF
import json
with open('config/model2path.json', 'r') as f:
    config = json.load(f)
config['custom_model'] = '${MODEL_PATH}'
with open('config/model2path.json', 'w') as f:
    json.dump(config, f, indent=4)
EOF
        MODEL_NAME="custom_model"
    else
        MODEL_NAME="llama"
    fi
    
    echo "  Running pred.py (progress shown in real-time)..."
    python pred.py --model ${MODEL_NAME} 2>&1 | tee "../${RESULTS_DIR}/longbench.log"
    
    echo "  Computing scores..."
    python eval.py 2>&1 | tee "../${RESULTS_DIR}/longbench_scores.txt"
    
    echo "  ✓ LongBench completed"
    echo "  Results: ${RESULTS_DIR}/longbench_scores.txt"
    cd ..
}

run_niah() {
    echo "[NIAH] Starting evaluation..."
    cd niah
    
    echo "  Testing context lengths: 32K, 48K, 64K"
    python run_needle_in_haystack.py \
        --model_name "${MODEL_PATH}" \
        --attn_implementation sdpa \
        --s_len 32000 \
        --e_len 64000 \
        --step 16000 \
        2>&1 | tee "../${RESULTS_DIR}/niah.log"
    
    echo "  ✓ NIAH completed"
    echo "  Results: ${RESULTS_DIR}/niah.log"
    cd ..
}

run_memory() {
    echo "[Memory] Starting evaluation..."
    cd memory_measurement
    
    echo "  Measuring memory usage and throughput..."
    CUDA_VISIBLE_DEVICES=0 python eval_memory.py \
        --model_name "${MODEL_PATH}" \
        --attn_implementation sdpa \
        2>&1 | tee "../${RESULTS_DIR}/memory.log"
    
    echo "  ✓ Memory Measurement completed"
    echo "  Results: ${RESULTS_DIR}/memory.log"
    cd ..
}

run_infinitebench() {
    echo "[InfiniteBench] Starting evaluation (Passkey)..."
    cd infiniteBench/src
    
    echo "  Testing 100 samples..."
    CUDA_VISIBLE_DEVICES=0 python eval_commvq.py \
        --model_name custom \
        --model_path "${MODEL_PATH}" \
        --task passkey \
        --start_idx 0 \
        --stop_idx 100 \
        2>&1 | tee "../../${RESULTS_DIR}/infinitebench.log"
    
    echo "  ✓ InfiniteBench completed"
    echo "  Results: ${RESULTS_DIR}/infinitebench.log"
    cd ../..
}

# 벤치마크 실행
case ${BENCHMARK} in
    longbench)
        run_longbench
        ;;
    niah)
        run_niah
        ;;
    memory)
        run_memory
        ;;
    infinitebench)
        run_infinitebench
        ;;
    all)
        echo "Running all benchmarks..."
        run_longbench
        run_niah
        run_memory
        run_infinitebench
        echo ""
        echo "✅ All benchmarks completed!"
        ;;
    *)
        echo "Error: Unknown benchmark '${BENCHMARK}'"
        echo "Available: longbench, niah, memory, infinitebench, all"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "✅ Evaluation completed!"
echo "Results saved to: ${RESULTS_DIR}/"
echo "============================================================"

