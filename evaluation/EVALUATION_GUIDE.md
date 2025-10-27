# CommVQ 평가 가이드

이 가이드는 CommVQ 모델을 평가하는 방법을 설명합니다.

---

## 📊 사용 가능한 벤치마크

### 1. **LongBench**
- **설명**: 16개의 long-context 이해 데이터셋
- **측정 지표**: F1, Rouge, Accuracy 등
- **실행 시간**: ~2-4시간

### 2. **NIAH (Needle in a Haystack)**
- **설명**: 긴 문맥에서 특정 정보 검색 능력 테스트
- **측정 지표**: Accuracy at different depths and lengths
- **실행 시간**: ~1-2시간

### 3. **Memory Measurement**
- **설명**: 메모리 사용량 및 처리량 측정
- **측정 지표**: Memory usage, throughput
- **실행 시간**: ~30분

### 4. **InfiniteBench**
- **설명**: 매우 긴 문맥 (100K+ tokens) 처리 능력
- **서브태스크**: passkey, kv_retrieval, longbook_qa 등
- **실행 시간**: ~2-3시간 (전체), ~30분 (passkey만)

---

## 🚀 빠른 시작

### 방법 1: 모델 비교 (추천)

**원본 CommVQ vs E2E 학습 모델 비교:**

```bash
cd evaluation

# 기본 사용 (자동으로 모델 경로 감지)
bash compare_models.sh

# 또는 명시적으로 지정
bash compare_models.sh \
    meta-llama/Llama-3.1-8B-Instruct \
    ../training/finetune/output/llama3.1_8b_int1
```

**결과:**
- `results_comparison_YYYYMMDD_HHMMSS/` 디렉토리에 모든 결과 저장
- 각 벤치마크별로 original vs e2e 비교 로그
- `SUMMARY.txt`에 요약 제공

---

### 방법 2: 단일 벤치마크 실행

**특정 벤치마크만 실행:**

```bash
cd evaluation

# LongBench만 실행
bash run_single_benchmark.sh longbench /path/to/model

# NIAH만 실행
bash run_single_benchmark.sh niah /path/to/model

# Memory Measurement만 실행
bash run_single_benchmark.sh memory /path/to/model

# InfiniteBench만 실행
bash run_single_benchmark.sh infinitebench /path/to/model

# 모든 벤치마크 실행
bash run_single_benchmark.sh all /path/to/model
```

---

## 📝 개별 벤치마크 실행 (수동)

### LongBench

```bash
cd evaluation/longbench

# 1. config/model2path.json에 모델 추가
# (또는 스크립트가 자동으로 처리)

# 2. 예측 생성
python pred.py --model llama

# 3. 점수 계산
python eval.py
```

**결과 위치:** `pred/` 및 `pred_e/` 디렉토리

---

### NIAH (Needle in a Haystack)

```bash
cd evaluation/niah

python run_needle_in_haystack.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --attn_implementation sdpa \
    --s_len 32000 \
    --e_len 64000 \
    --step 16000
```

**파라미터:**
- `--s_len`: 시작 길이 (토큰)
- `--e_len`: 종료 길이 (토큰)
- `--step`: 증가 단위

**결과 위치:** `results/` 디렉토리

---

### Memory Measurement

```bash
cd evaluation/memory_measurement

CUDA_VISIBLE_DEVICES=0 python eval_memory.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --attn_implementation sdpa
```

**결과:** 터미널에 직접 출력 (메모리 사용량, 처리 시간 등)

---

### InfiniteBench

```bash
cd evaluation/infiniteBench/src

# Passkey 태스크
CUDA_VISIBLE_DEVICES=0 python eval_commvq.py \
    --model_name llama \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --task passkey \
    --start_idx 0 \
    --stop_idx 100

# 다른 태스크들:
# - kv_retrieval
# - longbook_qa_eng
# - longbook_sum_eng
# - longdialogue_qa_eng
# - math_find
# - number_string
# - code_debug
```

**결과 위치:** `results/` 디렉토리

---

## 🔧 문제 해결

### 1. GLIBC 오류
```
Error: GLIBC_2.32 not found
```

**해결:** 이미 모든 스크립트에서 flash-attn을 비활성화하고 SDPA를 사용하도록 수정되었습니다.

---

### 2. GPU 메모리 부족
```
RuntimeError: CUDA out of memory
```

**해결:**
```bash
# 단일 GPU만 사용
CUDA_VISIBLE_DEVICES=0 python ...

# 또는 batch size 줄이기 (코드 수정 필요)
```

---

### 3. 모델 경로 오류
```
Error: Model not found
```

**해결:**
- HuggingFace 모델: `meta-llama/Llama-3.1-8B-Instruct`
- 로컬 모델: 절대 경로 사용 `/home/.../model`
- Fine-tuned 모델: `../training/finetune/output/llama3.1_8b_int1`

---

## 📊 결과 해석

### LongBench 점수

```
Dataset             | Score (F1/Accuracy)
--------------------|--------------------
narrativeqa         | 23.45
qasper              | 43.21
multifieldqa_en     | 52.34
...
Average             | 42.87
```

**좋은 점수:**
- Average > 40: 우수
- Average > 35: 양호
- Average < 30: 개선 필요

---

### NIAH 결과

**시각화:**
```bash
cd evaluation/niah
python viz.py  # 히트맵 생성
```

**해석:**
- Depth 0-100%, Length 32K-64K에서의 정확도
- 높을수록 좋음 (100% = 완벽)

---

### Memory 결과

```
Average Memory: 12.34 GB
Peak Memory: 15.67 GB
Throughput: 123.45 tokens/sec
```

**비교:**
- CommVQ는 baseline 대비 ~50% 메모리 절감
- Throughput 10-20% 향상

---

## 🎯 추천 평가 순서

### 빠른 검증 (30분-1시간)
```bash
# 1. Memory measurement (가장 빠름)
bash run_single_benchmark.sh memory /path/to/model

# 2. InfiniteBench Passkey (대표적)
bash run_single_benchmark.sh infinitebench /path/to/model
```

### 전체 평가 (4-8시간)
```bash
# 모든 벤치마크 실행
bash compare_models.sh \
    meta-llama/Llama-3.1-8B-Instruct \
    ../training/finetune/output/llama3.1_8b_int1
```

---

## 📈 모델 비교 체크리스트

평가 완료 후 확인 사항:

- [ ] LongBench 평균 점수가 baseline 유지 또는 향상
- [ ] NIAH에서 긴 문맥에서도 높은 정확도
- [ ] Memory 사용량 감소 (50% 목표)
- [ ] Throughput 유지 또는 향상
- [ ] InfiniteBench Passkey > 90% accuracy

---

## 💡 팁

### 백그라운드 실행

```bash
# nohup으로 실행
nohup bash compare_models.sh > eval.log 2>&1 &

# 진행 상황 확인
tail -f eval.log

# 프로세스 확인
ps aux | grep python
```

### GPU 사용률 모니터링

```bash
# 별도 터미널에서
watch -n 1 nvidia-smi
```

### 결과 백업

```bash
# 결과 압축
tar -czf results_$(date +%Y%m%d).tar.gz results_*

# 다른 서버로 전송
scp results_*.tar.gz user@server:/path/
```

---

## 📚 참고 문서

- **LongBench**: https://github.com/THUDM/LongBench
- **InfiniteBench**: https://github.com/OpenBMB/InfiniteBench
- **NIAH**: https://github.com/FranxYao/Long-Context-Data-Engineering
- **CommVQ Paper**: arXiv:2506.18879v1

---

## 🆘 도움이 필요하신가요?

문제가 발생하면:
1. 로그 파일 확인: `results_*/comparison.log`
2. GPU 메모리 확인: `nvidia-smi`
3. 오류 메시지 전체 복사

