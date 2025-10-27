#!/bin/bash

################################################################################
# CommVQ vs E2E-CommVQ 모델 비교 평가 스크립트
# 
# 사용법:
#   bash compare_models.sh <original_model_path> <e2e_model_path>
#
# 예시:
#   bash compare_models.sh meta-llama/Llama-3.1-8B-Instruct /path/to/e2e/model
################################################################################

set -e  # 오류 발생 시 중단

# ============================================================
# 설정
# ============================================================

# 모델 경로
ORIGINAL_MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
E2E_MODEL=${2:-"../training/finetune/output/llama3.1_8b_int1"}

# 결과 저장 디렉토리
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_comparison_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# 로그 파일
LOG_FILE="${RESULTS_DIR}/comparison.log"

echo "============================================================" | tee -a "${LOG_FILE}"
echo "CommVQ Model Comparison Evaluation" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Original Model: ${ORIGINAL_MODEL}" | tee -a "${LOG_FILE}"
echo "E2E Model:      ${E2E_MODEL}" | tee -a "${LOG_FILE}"
echo "Results Dir:    ${RESULTS_DIR}" | tee -a "${LOG_FILE}"
echo "Started at:     $(date)" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# ============================================================
# 1. LongBench 평가
# ============================================================

echo "[1/4] Running LongBench evaluation..." | tee -a "${LOG_FILE}"
echo "------------------------------------------------------" | tee -a "${LOG_FILE}"

cd longbench

# Original model
echo "  [1.1] Evaluating Original CommVQ model..." | tee -a "../${LOG_FILE}"
echo "        (Progress will be shown in real-time)" | tee -a "../${LOG_FILE}"
python pred.py --model llama 2>&1 | tee "../${RESULTS_DIR}/longbench_original.log"
echo "  ✓ Original model done" | tee -a "../${LOG_FILE}"

# E2E model - model2path.json에 추가 필요
echo "  [1.2] Adding E2E model to config..." | tee -a "../${LOG_FILE}"
python << EOF
import json
with open('config/model2path.json', 'r') as f:
    config = json.load(f)
config['llama-e2e'] = '${E2E_MODEL}'
with open('config/model2path.json', 'w') as f:
    json.dump(config, f, indent=4)
EOF

echo "  [1.3] Evaluating E2E model..." | tee -a "../${LOG_FILE}"
echo "        (Progress will be shown in real-time)" | tee -a "../${LOG_FILE}"
python pred.py --model llama-e2e 2>&1 | tee "../${RESULTS_DIR}/longbench_e2e.log"
echo "  ✓ E2E model done" | tee -a "../${LOG_FILE}"

# 결과 평가
echo "  [1.4] Computing scores..." | tee -a "../${LOG_FILE}"
python eval.py 2>&1 | tee "../${RESULTS_DIR}/longbench_scores.txt"
echo "  ✓ LongBench completed" | tee -a "../${LOG_FILE}"
echo "" | tee -a "../${LOG_FILE}"

cd ..

# ============================================================
# 2. NIAH (Needle in a Haystack) 평가
# ============================================================

echo "[2/4] Running NIAH evaluation..." | tee -a "${LOG_FILE}"
echo "------------------------------------------------------" | tee -a "${LOG_FILE}"

cd niah

# Original model
echo "  [2.1] Evaluating Original CommVQ model..." | tee -a "../${LOG_FILE}"
echo "        (Testing context lengths: 32K, 48K, 64K)" | tee -a "../${LOG_FILE}"
python run_needle_in_haystack.py \
    --model_name "${ORIGINAL_MODEL}" \
    --attn_implementation sdpa \
    --s_len 32000 \
    --e_len 64000 \
    --step 16000 \
    2>&1 | tee "../${RESULTS_DIR}/niah_original.log"
echo "  ✓ Original model done" | tee -a "../${LOG_FILE}"

# E2E model
echo "  [2.2] Evaluating E2E model..." | tee -a "../${LOG_FILE}"
echo "        (Testing context lengths: 32K, 48K, 64K)" | tee -a "../${LOG_FILE}"
python run_needle_in_haystack.py \
    --model_name "${E2E_MODEL}" \
    --attn_implementation sdpa \
    --s_len 32000 \
    --e_len 64000 \
    --step 16000 \
    2>&1 | tee "../${RESULTS_DIR}/niah_e2e.log"
echo "  ✓ E2E model done" | tee -a "../${LOG_FILE}"
echo "  ✓ NIAH completed" | tee -a "../${LOG_FILE}"
echo "" | tee -a "../${LOG_FILE}"

cd ..

# ============================================================
# 3. Memory Measurement 평가
# ============================================================

echo "[3/4] Running Memory Measurement..." | tee -a "${LOG_FILE}"
echo "------------------------------------------------------" | tee -a "${LOG_FILE}"

cd memory_measurement

# Original model
echo "  [3.1] Evaluating Original CommVQ model..." | tee -a "../${LOG_FILE}"
echo "        (Measuring memory usage and throughput)" | tee -a "../${LOG_FILE}"
CUDA_VISIBLE_DEVICES=0 python eval_memory.py \
    --model_name "${ORIGINAL_MODEL}" \
    --attn_implementation sdpa \
    2>&1 | tee "../${RESULTS_DIR}/memory_original.log"
echo "  ✓ Original model done" | tee -a "../${LOG_FILE}"

# E2E model
echo "  [3.2] Evaluating E2E model..." | tee -a "../${LOG_FILE}"
echo "        (Measuring memory usage and throughput)" | tee -a "../${LOG_FILE}"
CUDA_VISIBLE_DEVICES=0 python eval_memory.py \
    --model_name "${E2E_MODEL}" \
    --attn_implementation sdpa \
    2>&1 | tee "../${RESULTS_DIR}/memory_e2e.log"
echo "  ✓ E2E model done" | tee -a "../${LOG_FILE}"
echo "  ✓ Memory Measurement completed" | tee -a "../${LOG_FILE}"
echo "" | tee -a "../${LOG_FILE}"

cd ..

# ============================================================
# 4. InfiniteBench 평가 (Passkey만 - 가장 대표적)
# ============================================================

echo "[4/4] Running InfiniteBench (Passkey)..." | tee -a "${LOG_FILE}"
echo "------------------------------------------------------" | tee -a "${LOG_FILE}"

cd infiniteBench/src

# Original model
echo "  [4.1] Evaluating Original CommVQ model..." | tee -a "../../${LOG_FILE}"
echo "        (Testing Passkey task: 0-100 samples)" | tee -a "../../${LOG_FILE}"
CUDA_VISIBLE_DEVICES=0 python eval_commvq.py \
    --model_name original \
    --model_path "${ORIGINAL_MODEL}" \
    --task passkey \
    --start_idx 0 \
    --stop_idx 100 \
    2>&1 | tee "../../${RESULTS_DIR}/infinitebench_original.log"
echo "  ✓ Original model done" | tee -a "../../${LOG_FILE}"

# E2E model
echo "  [4.2] Evaluating E2E model..." | tee -a "../../${LOG_FILE}"
echo "        (Testing Passkey task: 0-100 samples)" | tee -a "../../${LOG_FILE}"
CUDA_VISIBLE_DEVICES=0 python eval_commvq.py \
    --model_name e2e \
    --model_path "${E2E_MODEL}" \
    --task passkey \
    --start_idx 0 \
    --stop_idx 100 \
    2>&1 | tee "../../${RESULTS_DIR}/infinitebench_e2e.log"
echo "  ✓ E2E model done" | tee -a "../../${LOG_FILE}"
echo "  ✓ InfiniteBench completed" | tee -a "../../${LOG_FILE}"
echo "" | tee -a "../../${LOG_FILE}"

cd ../..

# ============================================================
# 5. 결과 요약
# ============================================================

echo "============================================================" | tee -a "${LOG_FILE}"
echo "Evaluation Completed!" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Finished at: $(date)" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Results saved to: ${RESULTS_DIR}/" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Summary of results:" | tee -a "${LOG_FILE}"
echo "  - LongBench:        ${RESULTS_DIR}/longbench_scores.txt" | tee -a "${LOG_FILE}"
echo "  - NIAH:             ${RESULTS_DIR}/niah_*.log" | tee -a "${LOG_FILE}"
echo "  - Memory:           ${RESULTS_DIR}/memory_*.log" | tee -a "${LOG_FILE}"
echo "  - InfiniteBench:    ${RESULTS_DIR}/infinitebench_*.log" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# 간단한 결과 요약 생성
cat > "${RESULTS_DIR}/SUMMARY.txt" << EOF
CommVQ vs E2E-CommVQ Comparison Results
========================================

Generated: $(date)

Models Compared:
  - Original: ${ORIGINAL_MODEL}
  - E2E:      ${E2E_MODEL}

Benchmarks Evaluated:
  1. LongBench (16 datasets)
  2. NIAH (Needle in a Haystack)
  3. Memory Measurement
  4. InfiniteBench (Passkey)

Detailed Results:
  See individual log files in this directory.

To view LongBench scores:
  cat longbench_scores.txt

To view all logs:
  ls -lh *.log
EOF

echo "✅ All evaluations completed successfully!" | tee -a "${LOG_FILE}"
echo "📊 Check ${RESULTS_DIR}/SUMMARY.txt for quick summary" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

