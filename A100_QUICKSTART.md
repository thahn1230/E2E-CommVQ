# A100 서버 E2E-CommVQ 빠른 시작 가이드

## 현재 상황 ✅

- ✅ Step 1 완료: KV cache 수집 완료 (320 files)
- ⏳ Step 2 대기 중: Scaling factors 계산 필요
- ⏳ Step 3 대기 중: E2E 학습 필요

---

## 다음 단계 실행

### Step 2: Scaling Factors 계산

```bash
cd /home/ieslab/taehyun/E2E-CommVQ/training

# Scaling factors 계산 (5-10분 소요)
python make_scale.py
```

**예상 출력**:
```
============================================================
Computing scaling factors for E2E-CommVQ
============================================================

Processing 64 layer components (key + value)...
Processing key_000: 10 files
  ✓ key_000: torch.Size([9934820, 1024])
Processing key_001: 10 files
  ✓ key_001: torch.Size([9934820, 1024])
...
⚠️  No files found for value_000, skipping...
⚠️  No files found for value_001, skipping...
...

============================================================
✓ Processed: 32 components
  Skipped: 32 components (no data)
  Key layers: 32
  Value layers: 0
✓ Saved to: data/learnable_scale.pt
============================================================
```

**확인**:
```bash
# Scaling factors가 생성되었는지 확인
ls -lh data/learnable_scale.pt
python -c "import torch; d=torch.load('data/learnable_scale.pt'); print('Keys:', list(d.keys())); print('Key layers:', len(d['key']))"
```

---

### Step 3: E2E 학습 시작

#### 옵션 1: 단일 레이어 빠른 테스트 (권장 🌟)

```bash
# Layer 0만 학습 (15분 소요)
python quantize_key_cache.py 0 \
    --training_method e2e \
    --epochs 50 \
    --lr 0.001 \
    --batch_size 256
```

**진행 상황**:
```
Layer 0, Epoch 1/50
100%|████████████| 38/38 [00:15<00:00]
Loss: 0.0234, Recon: 0.0189, Commit: 0.0180
Saved best model with loss 0.023400

Layer 0, Epoch 2/50
...
```

#### 옵션 2: 모든 레이어 학습

```bash
# 전체 자동화 스크립트 (8-10시간 소요)
bash train_e2e_key_codebook.sh \
    --num_samples 10000 \
    --epochs 50 \
    --lr 0.001 \
    --batch_size 256
```

**또는 수동으로 각 레이어**:
```bash
# 각 레이어를 순차적으로 학습
for layer in {0..31}; do
    echo "Training layer $layer..."
    python quantize_key_cache.py $layer \
        --training_method e2e \
        --epochs 50 \
        --lr 0.001 \
        --batch_size 256
done
```

---

## 결과 확인

### 학습된 코드북 확인

```bash
# 생성된 코드북 파일 확인
ls -lh codebook_12bits_128group_21residuals/

# Layer 0 코드북 내용 확인
python -c "
import torch
data = torch.load('codebook_12bits_128group_21residuals/000_0.pt')
print('Keys:', list(data.keys()))
print('Steps:', len(data['steps']))
print('Final error:', data['final_error'])
"
```

### 평가 준비

학습이 완료되면:

```bash
# 코드북을 적절한 위치로 복사 (필요 시)
mkdir -p ../evaluation/codebooks/e2e_10k_50epochs
cp -r codebook_12bits_128group_21residuals/* ../evaluation/codebooks/e2e_10k_50epochs/

# 평가 스크립트 실행
cd ../evaluation/longbench
CUDA_VISIBLE_DEVICES=0 python pred.py \
    --model ../../Llama-3.1-8B-Instruct-CommVQ-E2E
```

---

## 문제 해결

### 오류: "No files found for key_XXX"

```bash
# 데이터 확인
python check_data.py

# KV cache 재수집 (필요 시)
python collect_kv_for_e2e.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset HuggingFaceFW/fineweb-edu \
    --output_dir data/key \
    --num_samples 10000
```

### 오류: "CUDA out of memory"

```bash
# 배치 크기 줄이기
python quantize_key_cache.py 0 \
    --training_method e2e \
    --epochs 50 \
    --lr 0.001 \
    --batch_size 128  # ← 256에서 128로 감소
```

### 학습 중단 후 재시작

```bash
# 이미 학습된 레이어는 건너뛰고 계속
for layer in {5..31}; do  # Layer 0-4는 완료된 상태
    python quantize_key_cache.py $layer \
        --training_method e2e \
        --epochs 50 \
        --lr 0.001 \
        --batch_size 256
done
```

---

## 예상 시간표 (10,000 샘플 기준)

| 단계 | 소요 시간 | 상태 |
|------|----------|------|
| Step 1: KV 수집 | ~30분 | ✅ 완료 |
| Step 2: Scaling | ~10분 | ⏳ 대기 |
| Step 3a: Layer 0 | ~15분 | ⏳ 대기 |
| Step 3b: 전체 32 layers | ~8시간 | ⏳ 대기 |

---

## 다음 실행 명령

```bash
cd /home/ieslab/taehyun/E2E-CommVQ/training

# 1. Scaling factors 계산
python make_scale.py

# 2. Layer 0 테스트 (빠른 검증)
python quantize_key_cache.py 0 \
    --training_method e2e \
    --epochs 50 \
    --lr 0.001 \
    --batch_size 256

# 3. 결과가 좋으면 전체 학습
bash train_e2e_key_codebook.sh --epochs 50
```

**지금 바로 실행**: `python make_scale.py` 🚀

