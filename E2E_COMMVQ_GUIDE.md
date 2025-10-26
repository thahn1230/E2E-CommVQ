# E2E-CommVQ: End-to-End Differentiable Training Guide

## Overview

E2E-CommVQ는 CommVQ의 불안정한 EM(Expectation-Maximization) 알고리즘을 대체하는 **End-to-End 미분 가능한** VQ(Vector Quantization) 학습 방식입니다.

### 핵심 특징

✅ **완전히 미분 가능한 학습**: Straight-Through Estimator(STE)를 사용하여 gradient-based optimization  
✅ **기존 평가 스크립트와 100% 호환**: EM 방식과 동일한 저장 형식 사용  
✅ **더 안정적인 학습**: EM 알고리즘의 local minima 문제 해결  
✅ **RoPE-Commutative 구조 유지**: CommVQ의 핵심 아키텍처 보존  

---

## Architecture

### E2E Key Compressor

```
Input (keys) 
    ↓
Normalization (learnable scale + L2 norm)
    ↓
Encoder Network (LayerNorm → Linear → GELU → Linear)
    ↓
Logits [batch, seq, groups, residuals, codebook_size]
    ↓
Gumbel-Softmax (differentiable sampling)
    ↓
Codebook Lookup (RoPE-Commutative structure)
    ↓
Residual Quantization (21 residuals)
    ↓
Quantized Output
```

### Loss Function

```python
Total Loss = Reconstruction Loss + 0.25 × Commitment Loss

where:
  Reconstruction Loss = MSE(original, quantized)
  Commitment Loss = MSE(residual after quantization)
```

---

## Installation

E2E-CommVQ는 기존 CommVQ 환경에서 바로 작동합니다:

```bash
conda activate commvq
# 추가 설치 필요 없음 - 모든 의존성이 이미 설치되어 있습니다
```

---

## Usage

### 1. EM 방식 (기존)

```bash
cd training

# Step 1: KV 캐시 수집
bash collect_kv.sh

# Step 2: Scaling factors 준비
python make_scale.py

# Step 3: Key 코드북 학습 (EM)
bash quantize_key_cache.sh

# Step 4: Value 코드북 학습
bash finetune/llama3.1_8b_int1.sh
```

### 2. E2E 방식 (새로운 방법)

```bash
cd training

# Step 1-2: 동일 (KV 캐시 수집 및 scaling factors)
bash collect_kv.sh
python make_scale.py

# Step 3: Key 코드북 학습 (E2E) ⭐ NEW
bash quantize_key_cache_e2e.sh

# 또는 단일 레이어 학습:
python quantize_key_cache.py 0 --training_method e2e --epochs 100 --lr 0.001 --batch_size 256

# Step 4: Value 코드북 학습 (동일)
bash finetune/llama3.1_8b_int1.sh
```

### 3. 커스텀 하이퍼파라미터

```bash
# E2E 학습 파라미터 조정
python quantize_key_cache.py <layer_idx> \
    --training_method e2e \
    --epochs 200 \           # 에폭 수 (기본: 100)
    --lr 0.0005 \            # 학습률 (기본: 0.001)
    --batch_size 512         # 배치 크기 (기본: 256)

# 모든 레이어 학습
bash quantize_key_cache_e2e.sh \
    --epochs 150 \
    --lr 0.0005 \
    --batch_size 512 \
    --num_layers 32

# 특정 레이어만 학습
bash quantize_key_cache_e2e.sh --layer 5 --epochs 100
```

---

## Compatibility Test

E2E로 학습된 모델이 기존 평가 스크립트와 호환되는지 테스트:

```bash
cd training
python test_e2e_compatibility.py
```

이 테스트는 다음을 검증합니다:
- ✅ 저장 형식이 EM과 동일
- ✅ theta 파라미터 shape 일치
- ✅ clustering_centers shape 일치
- ✅ residual 수 일치
- ✅ forward pass 정상 작동

---

## Training Details

### EM vs E2E 비교

| 측면 | EM (기존) | E2E (새로운) |
|------|-----------|--------------|
| **알고리즘** | Expectation-Maximization | Gradient Descent |
| **최적화** | 반복적 재할당 | End-to-end backprop |
| **미분 가능성** | ❌ 불가능 | ✅ 완전히 가능 |
| **학습 안정성** | 중간 (local minima) | 높음 (SGD + scheduler) |
| **학습 속도** | ~100 steps | ~100 epochs |
| **메모리 사용** | 낮음 | 중간 (gradient 저장) |
| **저장 형식** | `{layer}_{group}.pt` | 동일 |

### Hyperparameter 가이드

**Epochs**
- 기본: 100
- 빠른 프로토타입: 50
- 고품질: 200-300

**Learning Rate**
- 기본: 0.001
- 큰 모델: 0.0005
- 작은 모델: 0.002
- Scheduler: CosineAnnealing

**Batch Size**
- 기본: 256
- GPU 메모리 부족 시: 128
- 충분한 메모리: 512-1024

**Commitment Loss Weight**
- 기본: 0.25
- Codebook utilization 낮을 때: 0.5
- Reconstruction 우선: 0.1

---

## File Structure

### 저장 형식 (EM과 동일)

E2E로 학습된 코드북은 다음 형식으로 저장됩니다:

```
codebook_12bits_128group_21residuals/
├── 000_0.pt  # Layer 0, Group 0
├── 000_1.pt  # Layer 0, Group 1
├── ...
└── 031_7.pt  # Layer 31, Group 7
```

각 파일 구조:
```python
{
    "steps": [  # 21 residuals
        {
            "mse_loss": tensor(float),
            "theta": tensor([128, 64]),           # RoPE parameters
            "clustering_centers": tensor([4096, 128])  # Codebook entries
        },
        # ... 20 more residuals
    ],
    "final_error": float
}
```

### 코드 구조

```
CommVQ/
├── commvq/
│   ├── compress_training.py      ⭐ E2E KeyCompressor 추가
│   ├── compress_evaluation.py    (수정 불필요)
│   └── modeling_llama_*.py       (수정 불필요)
├── training/
│   ├── quantize_key_cache.py     ⭐ E2E 모드 추가
│   ├── quantize_key_cache_e2e.sh ⭐ NEW
│   └── test_e2e_compatibility.py ⭐ NEW
└── evaluation/                    (수정 불필요)
```

---

## Implementation Details

### Straight-Through Estimator (STE)

E2E-CommVQ는 미분 불가능한 argmax 연산을 우회하기 위해 STE를 사용합니다:

```python
# Forward pass: hard assignment
indices_hard = gumbel_softmax(logits, tau=1.0, hard=True)

# Backward pass: soft gradients
indices_soft = gumbel_softmax(logits, tau=1.0, hard=False)

# STE: forward hard, backward soft
indices = indices_hard.detach() + (indices_soft - indices_soft.detach())
```

### RoPE-Commutative Structure

E2E 학습에서도 CommVQ의 RoPE-Commutative 구조를 유지합니다:

```python
# Learnable parameters: theta = [x, y]
codebook_params: [num_groups, num_residuals, codebook_size_half, group_size//2, 2]

# Transformation matrix T (고정)
T: [2*codebook_size, 2*codebook_size_half]

# Codebook centers = T @ theta
clustering_centers = T @ theta
```

이 구조는 EM 방식과 정확히 동일하여, 추론 시 최적화된 디코딩이 가능합니다.

### Residual Quantization

21개의 residual을 순차적으로 처리:

```python
for r in range(21):
    # 1. Encode: predict codebook index
    logits = encoder(residual)
    
    # 2. Sample: differentiable with Gumbel-Softmax
    indices = gumbel_softmax(logits)
    
    # 3. Lookup: get codebook vector
    quantized = codebook[indices]
    
    # 4. Update residual
    residual = residual - quantized
```

---

## Evaluation

E2E로 학습된 모델은 기존 평가 스크립트를 **수정 없이** 사용할 수 있습니다:

```bash
# Longbench
cd evaluation/longbench
python pred.py --model $CHECKPOINT
python eval.py --model $RESULT_DIR

# InfiniteBench
cd evaluation/infiniteBench/src
bash run_passkey.sh
python compute_scores.py --task all --model_name commvq

# NIAH
cd evaluation/niah
bash run.sh $CHECKPOINT
```

평가 스크립트는 `compress_evaluation.KeyCompressor`를 사용하여 코드북을 로드하며, E2E와 EM 방식의 저장 형식이 동일하므로 자동으로 작동합니다.

---

## Troubleshooting

### 1. Out of Memory (OOM)

**증상**: CUDA out of memory 에러  
**해결**:
```bash
# Batch size 줄이기
python quantize_key_cache.py 0 --training_method e2e --batch_size 128

# 또는 gradient accumulation 추가 (코드 수정 필요)
```

### 2. 학습 불안정

**증상**: Loss가 발산하거나 NaN  
**해결**:
```bash
# Learning rate 줄이기
python quantize_key_cache.py 0 --training_method e2e --lr 0.0005

# Gradient clipping은 이미 구현됨 (max_norm=1.0)
```

### 3. Codebook Collapse

**증상**: 모든 입력이 같은 코드북 엔트리로 매핑  
**해결**:
- Commitment loss weight 증가: `loss = recon_loss + 0.5 * commitment_loss`
- 초기화 개선: `codebook_params.normal_(0, 0.05)` (코드 수정)

### 4. 저장 형식 불일치

**증상**: 평가 스크립트 로드 실패  
**해결**:
```bash
# 호환성 테스트 실행
python test_e2e_compatibility.py

# 파일 구조 확인
ls -la codebook_12bits_128group_21residuals/
```

---

## Performance Tips

1. **Warm-up**: 초기 몇 epoch는 낮은 learning rate 사용
2. **Data Augmentation**: 입력에 small noise 추가하여 robustness 향상
3. **Curriculum Learning**: 간단한 데이터부터 시작
4. **Multi-GPU**: DataParallel 또는 DistributedDataParallel 사용
5. **Mixed Precision**: `torch.cuda.amp` 사용하여 메모리 절약

---

## Citation

E2E-CommVQ를 사용하는 경우, 원본 CommVQ 논문을 인용해주세요:

```bibtex
@inproceedings{li2025commvq,
  title = {CommVQ: Commutative Vector Quantization for KV Cache Compression},
  author = {Junyan Li and Yang Zhang and Muhammad Yusuf Hassan and Talha Chafekar and Tianle Cai and Zhile Ren and Pengsheng Guo and Binazir Karimzadeh and Colorado J Reed and Chong Wang and Chuang Gan},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year = {2025}
}
```

---

## Contributing

E2E-CommVQ는 CommVQ 프로젝트의 확장입니다. 개선 사항이나 버그 리포트는 환영합니다!

---

## License

CommVQ의 원본 라이선스를 따릅니다.

