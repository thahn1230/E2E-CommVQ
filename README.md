# CommVQ

## Training

```python
cd training
pip install -e ../transformers_collect_kv
python collect_kv.py
python make_scale.py
pip install -e ../transformers_train
bash finetune/llama3.1_8b_int1.sh
```

## Evaluation

### Longbench

```python
cd evaluation/longbench
pip install -e ../../transformers_infer
python pred.py --model $CHECKPOINT
python eval.py --model $RESULT_DIR
```

### Infinitebench

First, install our modified transformers library to support our method:

```python
pip install -e ../../transformers_infer
```

Then, follow [Infinitebench](https://github.com/OpenBMB/InfiniteBench) repo to perform the evaluation. There is no need to change the code, all you need to do is installing our modified transformers library then you should be good to go.

### NIAH

```python
cd evaluation/niah
pip install -e ../../transformers_infer
bash run.sh $CHECKPOINT
```

### Memory Measurement

We implement some Triton kernels to optimize the memory usage to achieve a real memory saving using our method. It is still under active development and currently only LLaMA-3.1 8B model is supported.

```python
cd evaluation/memory_measurement
pip install -e ../../transformers_triton_infer
bash eval_memory.sh $CHECKPOINT
```

