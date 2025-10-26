# E2E-CommVQ Training Flow Guide

## 🎯 EM vs E2E: 핵심 차이점

| 단계 | EM (기존) | E2E (새로운) |
|------|-----------|--------------|
| **KV 수집** | collect_kv.sh<br>(modeling_llama_collect_kv) | collect_kv_for_e2e.py<br>(베이스 모델 직접 사용) |
| **Scaling** | make_scale.py | make_scale.py (동일) |
| **Key 학습** | quantize_key_cache.py<br>(EM 알고리즘) | quantize_key_cache.py<br>(--training_method e2e) |
| **출력 디렉토리** | codebook_12bits_... | codebook_e2e_12bits_... |
| **Value 학습** | finetune.py | finetune.py (동일) |

---

## 🚀 E2E Training: 완전 자동화 (권장)

### 기본 사용법

```bash
cd training

# 모든 것을 한 번에 실행
bash train_e2e_key_codebook.sh
```

이 스크립트는 자동으로:
1. ✅ KV cache 수집 (1M 샘플)
2. ✅ Scaling factors 계산
3. ✅ 모든 레이어(0-31) Key 코드북 E2E 학습
4. ✅ 별도 디렉토리에 저장 (`codebook_e2e_12bits_128group_21residuals/`)

### 커스터마이징

```bash
# 커스텀 모델 사용
bash train_e2e_key_codebook.sh \
    --model /path/to/your/model \
    --output_dir my_custom_codebook

# 하이퍼파라미터 조정
bash train_e2e_key_codebook.sh \
    --epochs 200 \
    --lr 0.0005 \
    --batch_size 512

# 1-bit 양자화
bash train_e2e_key_codebook.sh \
    --quant_bits 1 \
    --output_dir codebook_e2e_6bits_128group_21residuals

# 단일 레이어만 학습 (빠른 테스트)
bash train_e2e_key_codebook.sh \
    --layer 0 \
    --epochs 50

# 샘플 수 조정
bash train_e2e_key_codebook.sh \
    --num_samples 500000  # 50만 샘플만 사용
```

### 전체 옵션

```bash
bash train_e2e_key_codebook.sh --help
```

---

## 🔧 E2E Training: 수동 단계별 실행

더 세밀한 제어가 필요하다면:

### Step 1: KV Cache 수집

```bash
cd training

python collect_kv_for_e2e.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset HuggingFaceFW/fineweb-edu \
    --output_dir data/key \
    --num_samples 1000000 \
    --max_seq_length 8192 \
    --quant_bits 2
```

**주의**: 
- ✅ 베이스 모델 사용 (CommVQ 학습 전)
- ✅ 충분한 샘플 수집 (최소 100K, 권장 1M)
- ✅ `data/key/` 디렉토리에 저장

### Step 2: Scaling Factors 계산

```bash
python make_scale.py
```

**출력**: `data/learnable_scale.pt`

### Step 3: Key Codebook E2E 학습

#### 옵션 A: 모든 레이어 (자동 스크립트)

```bash
bash quantize_key_cache_e2e.sh --epochs 100 --lr 0.001
```

#### 옵션 B: 단일 레이어 (수동)

```bash
# Layer 0
python quantize_key_cache.py 0 \
    --training_method e2e \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 256

# Layer 1
python quantize_key_cache.py 1 \
    --training_method e2e \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 256

# ... (repeat for layers 2-31)
```

**출력**: `codebook_12bits_128group_21residuals/000_*.pt`, `001_*.pt`, ...

#### 옵션 C: 별도 디렉토리로 이동

```bash
# 학습 후
mkdir -p codebook_e2e_12bits_128group_21residuals
mv codebook_12bits_128group_21residuals/*.pt codebook_e2e_12bits_128group_21residuals/
```

### Step 4: Value Codebook 학습 (선택)

Value는 E2E로 이미 학습되어 있으므로, 원한다면 기존 방식 사용:

```bash
bash finetune/llama3.1_8b_int2.sh
```

---

## 📁 디렉토리 구조

### EM 방식 (기존)
```
training/
├── data/
│   ├── key/                    # collect_kv.sh로 수집
│   └── learnable_scale.pt
├── codebook_12bits_128group_21residuals/  # EM 학습 결과
│   ├── 000_0.pt
│   ├── 000_1.pt
│   └── ...
```

### E2E 방식 (새로운)
```
training/
├── data/
│   ├── key/                    # collect_kv_for_e2e.py로 수집
│   └── learnable_scale.pt
├── codebook_e2e_12bits_128group_21residuals/  # E2E 학습 결과 ⭐
│   ├── 000_0.pt
│   ├── 000_1.pt
│   └── ...
```

**중요**: E2E와 EM 코드북은 **별도 디렉토리**에 저장!

---

## ⚙️ 하이퍼파라미터 가이드

### 기본 설정 (권장)
```bash
--epochs 100
--lr 0.001
--batch_size 256
--num_samples 1000000
```

### 빠른 프로토타입
```bash
--epochs 50
--lr 0.002
--batch_size 128
--num_samples 100000
--layer 0  # 단일 레이어만
```

### 고품질 학습
```bash
--epochs 200
--lr 0.0005
--batch_size 512
--num_samples 2000000
```

### GPU 메모리 부족 시
```bash
--batch_size 128  # 또는 64
--num_samples 500000
```

---

## 🔄 E2E vs EM 선택 가이드

### E2E 사용 권장:
- ✅ 더 안정적인 학습 원함
- ✅ Local minima 문제 해결하고 싶음
- ✅ 하이퍼파라미터 튜닝 용이성
- ✅ End-to-end 최적화

### EM 사용 권장:
- ✅ 원 논문과 완전히 동일한 재현
- ✅ 더 빠른 학습 (100 steps vs 100 epochs)
- ✅ 메모리 효율적

---

## 🧪 평가 (E2E vs EM 비교)

### E2E 모델 평가

```bash
cd evaluation/longbench

# E2E 코드북 경로를 모델 config에 설정 후
CUDA_VISIBLE_DEVICES=0 python pred.py --model /path/to/model
```

### 코드북 교체 방법

```bash
# 1. E2E 코드북을 모델 디렉토리로 복사
cp -r codebook_e2e_12bits_128group_21residuals/* \
      /path/to/model/Llama-3.1-8B-Instruct-CommVQ-2bit-codebook/

# 2. 평가 실행
cd evaluation/longbench
CUDA_VISIBLE_DEVICES=0 python pred.py --model /path/to/model
```

---

## ❓ FAQ

### Q1: E2E와 EM 코드북을 같은 디렉토리에 저장해도 되나요?
**A**: ❌ 안 됩니다! 파일 이름이 같아서 덮어씌워집니다. 반드시 별도 디렉토리 사용:
- EM: `codebook_12bits_128group_21residuals/`
- E2E: `codebook_e2e_12bits_128group_21residuals/`

### Q2: collect_kv.sh를 E2E에도 사용할 수 있나요?
**A**: ❌ 안 됩니다! E2E는 `collect_kv_for_e2e.py`를 사용해야 합니다.
- EM: modeling_llama_collect_kv (학습 중)
- E2E: 베이스 모델 직접 사용

### Q3: E2E 학습에 얼마나 걸리나요?
**A**: 
- KV 수집: ~2-4시간 (1M 샘플, A100)
- Layer당 학습: ~30분-1시간 (100 epochs, A100)
- 전체 32 layers: ~16-32시간

### Q4: 중간에 중단되면 어떻게 하나요?
**A**: 레이어별로 저장되므로, 이미 완료된 레이어는 건너뛰고 나머지만 학습:
```bash
# Layer 10부터 재개
for layer in {10..31}; do
    python quantize_key_cache.py $layer --training_method e2e ...
done
```

### Q5: E2E 결과가 EM보다 나쁘면?
**A**: 하이퍼파라미터 조정 시도:
1. Learning rate 줄이기: `--lr 0.0005`
2. Epochs 늘리기: `--epochs 200`
3. Batch size 늘리기: `--batch_size 512`
4. Commitment loss weight 조정 (코드 수정 필요)

---

## 📊 예상 결과

### 학습 곡선 (Layer 0 예시)
```
Epoch 1/100 - Loss: 0.5234, Recon: 0.4891, Commit: 0.1372
Epoch 10/100 - Loss: 0.2156, Recon: 0.1912, Commit: 0.0976
Epoch 50/100 - Loss: 0.0834, Recon: 0.0712, Commit: 0.0488
Epoch 100/100 - Loss: 0.0567, Recon: 0.0491, Commit: 0.0304
✓ Saved best model with loss 0.0567
```

### 저장 파일
```
codebook_e2e_12bits_128group_21residuals/
├── 000_0.pt  (Layer 0, Group 0: ~2MB)
├── 000_1.pt  (Layer 0, Group 1: ~2MB)
├── ...
├── 031_7.pt  (Layer 31, Group 7: ~2MB)
└── (Total: ~512MB for all 256 files)
```

---

## 🎓 완전한 E2E 예시

```bash
#!/bin/bash
# complete_e2e_training.sh

cd training

# 1. 자동화 스크립트 사용 (권장)
bash train_e2e_key_codebook.sh \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset HuggingFaceFW/fineweb-edu \
    --output_dir codebook_e2e_12bits_128group_21residuals \
    --quant_bits 2 \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 256 \
    --num_samples 1000000

# 2. (선택) Value 학습
bash finetune/llama3.1_8b_int2.sh

# 3. 평가
cd ../evaluation/longbench
CUDA_VISIBLE_DEVICES=0 python pred.py --model /path/to/model

echo "✓ E2E Training Pipeline Completed!"
```

---

## 🏆 성공 체크리스트

- [ ] `collect_kv_for_e2e.py`로 KV cache 수집 완료
- [ ] `data/key/`에 파일 생성 확인
- [ ] `make_scale.py`로 scaling factors 계산
- [ ] `data/learnable_scale.pt` 존재 확인
- [ ] E2E 학습 완료 (모든 레이어)
- [ ] `codebook_e2e_12bits_128group_21residuals/`에 256개 파일 확인
- [ ] 평가 스크립트 정상 실행
- [ ] 성능 결과 기록

---

**다음 단계**: [E2E_COMMVQ_GUIDE.md](E2E_COMMVQ_GUIDE.md)에서 더 자세한 정보 확인

