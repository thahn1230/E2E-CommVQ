# tmux를 사용한 CommVQ 평가 가이드

tmux를 사용하면 **실시간 진행 상황**을 보면서 평가를 실행하고, 터미널을 종료해도 계속 실행할 수 있습니다.

---

## 🎯 기본 사용법

### 1. tmux 세션 시작

```bash
# A100 서버에서
cd /home/ieslab/taehyun/E2E-CommVQ/evaluation

# 새 tmux 세션 생성
tmux new -s eval

# 또는 이름 없이
tmux
```

### 2. 평가 실행 (실시간 출력 확인)

```bash
# 전체 비교 평가
bash compare_models.sh \
    meta-llama/Llama-3.1-8B-Instruct \
    ../training/finetune/output/llama3.1_8b_int1

# 또는 단일 벤치마크
bash run_single_benchmark.sh longbench /path/to/model
```

**진행 상황이 실시간으로 보입니다:**
```
[1/4] Running LongBench evaluation...
------------------------------------------------------
  [1.1] Evaluating Original CommVQ model...
        (Progress will be shown in real-time)
Generating predictions: 100%|████████| 16/16 [1:23:45<00:00, 313.47s/it]
  ✓ Original model done
  [1.2] Adding E2E model to config...
  [1.3] Evaluating E2E model...
        (Progress will be shown in real-time)
Generating predictions:  25%|██▌     | 4/16 [21:15<1:03:45, 318.75s/it]
```

### 3. tmux 세션 나가기 (평가는 계속 실행)

```bash
# Ctrl+B, 그 다음 D 키
# 또는
tmux detach
```

터미널을 종료해도 평가는 계속 실행됩니다!

---

## 📊 실행 중인 평가 확인

### 다시 연결하기

```bash
# SSH로 다시 A100 서버 접속 후

# 세션 목록 확인
tmux ls

# 세션에 다시 연결
tmux attach -t eval

# 또는 가장 최근 세션
tmux attach
```

### 여러 패널로 모니터링

```bash
# tmux 세션 안에서
# Ctrl+B, 그 다음 "  (수평 분할)
# 또는
# Ctrl+B, 그 다음 %  (수직 분할)

# 패널 간 이동: Ctrl+B, 그 다음 화살표 키
```

**예시 레이아웃:**
```
┌─────────────────────────┬──────────────────────┐
│                         │                      │
│  평가 스크립트 실행      │  nvidia-smi          │
│  (실시간 진행 상황)      │  (GPU 사용률)         │
│                         │                      │
│                         │                      │
├─────────────────────────┴──────────────────────┤
│                                                │
│  tail -f results_*/comparison.log              │
│  (로그 파일 실시간 확인)                         │
│                                                │
└────────────────────────────────────────────────┘
```

**설정 방법:**
```bash
# tmux 세션 안에서

# 1. 평가 스크립트 실행
bash compare_models.sh ...

# 2. 수평 분할 (Ctrl+B, ")
# 3. nvidia-smi 실행
watch -n 1 nvidia-smi

# 4. 원래 패널로 이동 (Ctrl+B, 화살표↑)
# 5. 수직 분할 (Ctrl+B, %)
# 6. 로그 확인
tail -f results_comparison_*/comparison.log
```

---

## 🎨 유용한 tmux 명령어

### 기본 단축키
- `Ctrl+B, D` : Detach (세션 나가기, 계속 실행)
- `Ctrl+B, "` : 수평 분할
- `Ctrl+B, %` : 수직 분할
- `Ctrl+B, 화살표` : 패널 이동
- `Ctrl+B, X` : 현재 패널 닫기
- `Ctrl+B, [` : 스크롤 모드 (q로 종료)

### 세션 관리
```bash
# 새 세션 생성
tmux new -s <name>

# 세션 목록
tmux ls

# 세션 연결
tmux attach -t <name>

# 세션 종료
tmux kill-session -t <name>

# 모든 세션 종료
tmux kill-server
```

---

## 💡 실전 예제

### 예제 1: 전체 평가 실행 및 모니터링

```bash
# 1. tmux 시작
tmux new -s eval_full

# 2. 평가 스크립트 실행
cd /home/ieslab/taehyun/E2E-CommVQ/evaluation
bash compare_models.sh \
    meta-llama/Llama-3.1-8B-Instruct \
    ../training/finetune/output/llama3.1_8b_int1

# 3. Ctrl+B, " (수평 분할)

# 4. GPU 모니터링
watch -n 1 nvidia-smi

# 5. Ctrl+B, D (세션 나가기)

# 나중에 다시 확인
tmux attach -t eval_full
```

### 예제 2: 단계별 진행 (추천)

```bash
# tmux 시작
tmux new -s eval_step

cd /home/ieslab/taehyun/E2E-CommVQ/evaluation

# 1단계: 빠른 검증
bash run_single_benchmark.sh memory ../training/finetune/output/llama3.1_8b_int1

# 결과 확인 후 2단계
bash run_single_benchmark.sh infinitebench ../training/finetune/output/llama3.1_8b_int1

# 만족스러우면 3단계: 전체 평가
bash run_single_benchmark.sh all ../training/finetune/output/llama3.1_8b_int1
```

### 예제 3: 여러 모델 동시 평가 (고급)

```bash
# tmux 시작
tmux new -s eval_multi

# 창 1: 모델 A 평가
bash run_single_benchmark.sh longbench /path/to/model_a

# 새 창 생성 (Ctrl+B, C)
# 창 2: 모델 B 평가
bash run_single_benchmark.sh longbench /path/to/model_b

# 창 전환 (Ctrl+B, N: 다음 창, Ctrl+B, P: 이전 창)
```

---

## 📝 진행 상황 확인 방법

### 방법 1: 직접 확인
```bash
# tmux 세션에 연결
tmux attach -t eval

# 실시간으로 tqdm 진행 바와 로그 확인
```

### 방법 2: 로그 파일
```bash
# 별도 터미널에서
cd /home/ieslab/taehyun/E2E-CommVQ/evaluation
tail -f results_comparison_*/comparison.log
```

### 방법 3: GPU 사용률
```bash
# 평가가 실행 중이면 GPU 사용률이 높음
nvidia-smi
```

---

## 🔍 예상 출력 예시

### LongBench 진행 중
```
[1/4] Running LongBench evaluation...
------------------------------------------------------
  [1.1] Evaluating Original CommVQ model...
        (Progress will be shown in real-time)

Loading model: 100%|████████| 4/4 [00:10<00:00]
Processing dataset narrativeqa: 100%|████████| 200/200 [05:23<00:00,  1.62s/it]
Processing dataset qasper: 100%|████████| 200/200 [06:15<00:00,  1.88s/it]
Processing dataset multifieldqa_en: 100%|████████| 150/150 [04:45<00:00,  1.90s/it]
...
Overall progress: 50%|█████     | 8/16 [41:23<41:23, 310.47s/it]
```

### NIAH 진행 중
```
[2/4] Running NIAH evaluation...
------------------------------------------------------
  [2.1] Evaluating Original CommVQ model...
        (Testing context lengths: 32K, 48K, 64K)

Testing depth: 0%, length: 32000
Testing depth: 10%, length: 32000
Testing depth: 20%, length: 32000
...
Progress: 33%|███▍      | 1/3 lengths [15:23<30:46, 923.17s/it]
```

### Memory 진행 중
```
[3/4] Running Memory Measurement...
------------------------------------------------------
  [3.1] Evaluating Original CommVQ model...
        (Measuring memory usage and throughput)

Loading model...
Warming up...
Measuring memory: 100%|████████| 20/20 [02:15<00:00,  6.78s/it]
Average Memory: 12.34 GB
Peak Memory: 15.67 GB
Throughput: 123.45 tokens/sec
```

---

## 🚨 문제 해결

### 1. tmux 세션을 찾을 수 없음
```bash
# 모든 세션 확인
tmux ls

# 실행 중인 프로세스 확인
ps aux | grep compare_models
ps aux | grep python
```

### 2. 세션이 응답 없음
```bash
# 다른 터미널에서 확인
tmux attach -t eval

# 정말 응답이 없으면 강제 종료
tmux kill-session -t eval
```

### 3. GPU 메모리 부족
```bash
# tmux 세션 안에서 Ctrl+C로 중단
# GPU 메모리 정리
nvidia-smi

# 단일 벤치마크만 실행
bash run_single_benchmark.sh memory /path/to/model
```

---

## ⚡ 프로 팁

### 1. 자동 저장 설정
```bash
# ~/.tmux.conf 추가
set -g history-limit 50000
set -g mouse on
```

### 2. 여러 단계 자동화
```bash
# 스크립트 작성
cat > run_evaluation_pipeline.sh << 'EOF'
#!/bin/bash
cd /home/ieslab/taehyun/E2E-CommVQ/evaluation

echo "Step 1: Memory (5min)"
bash run_single_benchmark.sh memory ../training/finetune/output/llama3.1_8b_int1

echo "Step 2: InfiniteBench (30min)"
bash run_single_benchmark.sh infinitebench ../training/finetune/output/llama3.1_8b_int1

echo "Step 3: Full evaluation (4-8h)"
bash compare_models.sh \
    meta-llama/Llama-3.1-8B-Instruct \
    ../training/finetune/output/llama3.1_8b_int1
EOF

chmod +x run_evaluation_pipeline.sh

# tmux에서 실행
tmux new -s eval
./run_evaluation_pipeline.sh
```

### 3. 알림 추가
```bash
# 평가 완료 시 알림
bash compare_models.sh ... && echo "✅ Evaluation completed!" | mail -s "CommVQ Eval Done" your@email.com
```

---

## 📚 요약

### tmux 사용의 장점:
- ✅ **실시간 진행 상황** 확인 (tqdm 진행 바, 로그 등)
- ✅ **SSH 연결 끊어도 계속 실행**
- ✅ **여러 패널로 동시 모니터링** (평가 + GPU + 로그)
- ✅ **언제든 다시 연결** 가능

### 기본 워크플로우:
```bash
# 1. tmux 시작
tmux new -s eval

# 2. 평가 실행
bash compare_models.sh ...

# 3. 세션 나가기 (계속 실행)
Ctrl+B, D

# 4. 나중에 확인
tmux attach -t eval
```

**지금 바로 시도해보세요!** 🚀

