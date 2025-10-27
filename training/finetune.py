# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
try:
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    zero = None
    ZeroParamStatus = None
import transformers
from transformers import Trainer, GPTQConfig
try:
    from transformers import deepspeed as transformers_deepspeed
except ImportError:
    transformers_deepspeed = None
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from commvq.compress_training import KeyCompressor, ValueCompressor
from commvq.modeling_llama_training import LlamaForCausalLM
import numpy as np
import warnings
warnings.filterwarnings("ignore")


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    freeze_model: bool = field(default=False)
    is_stage1: bool = field(default=False)
    is_stage2: bool = field(default=False)
    use_mse_loss: bool = field(default=True)
    use_vq_loss: bool = field(default=False)
    freeze_vq_model: bool = field(default=False)
    n_e: int = field(default=1024)
    stage1_droppath: float = field(default=0.0)
    weight_vq_loss: float = field(default=None)
    quant_bits: float = field(default=2)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    max_steps: int = field(
        default=-1,
        metadata={"help": "Total number of training steps to perform."}
    )
    max_seq_length: int = field(
        default=4096,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    dispatch_batches: bool = field(
        default=False,
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm."},
    )
    resume_from_checkpoint: bool = field(
        default=False,
        metadata={"help": "Resume from the checkpoint."},
    )
    num_train_epochs: int = field(
        default=-1,
        metadata={"help": "Total number of training epochs to perform."}
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Use packing or not."}
    )


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    use_lora: bool = field(
        default=False,
        metadata={"help": "Use LoRA for training."},
    )


def maybe_zero_3(param):
    if DEEPSPEED_AVAILABLE and hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if DEEPSPEED_AVAILABLE and transformers_deepspeed and transformers_deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)



ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n{output}"
    ),
}

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'text': prompt_format.format(**example)}



def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or (DEEPSPEED_AVAILABLE and transformers_deepspeed and transformers_deepspeed.is_deepspeed_zero3_enabled()):
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )

    is_chat_model = 'chat' in model_args.model_name_or_path.lower()
    if (
            lora_args.use_lora
            and not lora_args.q_lora
            and DEEPSPEED_AVAILABLE and transformers_deepspeed and transformers_deepspeed.is_deepspeed_zero3_enabled()
            and not is_chat_model
    ):
        raise RuntimeError("ZeRO3 is incompatible with LoRA when finetuning on base model.")

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    config.use_cache = False
    config.is_stage1 = model_args.is_stage1
    config.is_stage2 = model_args.is_stage2
    config.use_mse_loss = model_args.use_mse_loss
    config.use_vq_loss = model_args.use_vq_loss
    config.n_e = model_args.n_e
    config.stage1_droppath = model_args.stage1_droppath
    config.weight_vq_loss = model_args.weight_vq_loss
    config.quant_bits = model_args.quant_bits

    # Load model and tokenizer
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if "output" not in model_args.model_name_or_path:
        if model_args.is_stage1:
            learnable_scale_values = torch.load("data/learnable_scale.pt", map_location="cpu")["value"]
            for i in range(len(model.model.layers)):
                feat_dim = model.config.hidden_size // model.config.num_attention_heads * model.config.num_key_value_heads
                key_vq_model = KeyCompressor(feat_dim=feat_dim, layer_idx=i, quant_bits=config.quant_bits)
                value_vq_model = ValueCompressor(
                    quant_bits=config.quant_bits,
                    feat_dim=feat_dim,
                )
                model.model.layers[i].self_attn.key_vq_model = key_vq_model
                model.model.layers[i].self_attn.value_vq_model = value_vq_model
                value_mean = learnable_scale_values[i].to(torch.bfloat16)
                model.model.layers[i].self_attn.value_vq_model.learnable_scale.data = value_mean
        elif not model_args.is_stage2 and not model_args.is_stage1:
            for i in range(len(model.model.layers)):
                model.model.layers[i].self_attn.key_vq_model = None
                model.model.layers[i].self_attn.value_vq_model = None
    if model_args.is_stage1:
        # freeze parameter with no "vq_model" in the name
        for name, param in model.named_parameters():
            if "value_vq_model" not in name:
                param.requires_grad = False
            if "learnable_scale" in name:
                param.requires_grad = False
    elif model_args.is_stage2 and model_args.freeze_vq_model:
        raise NotImplementedError
        # freeze parameter with "vq_model" in the name
        for name, param in model.named_parameters():
            if "vq_model" in name:
                param.requires_grad = False
        # for i in range(len(model.model.layers)):
        #     model.model.layers[i].self_attn.key_vq_model.eval()
        #     model.model.layers[i].self_attn.value_vq_model.eval()
        #     assert not model.model.layers[i].self_attn.key_vq_model.training
        #     assert not model.model.layers[i].self_attn.value_vq_model.training
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if lora_args.use_lora:
        if lora_args.q_lora or is_chat_model:
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    rank0_print("Loading data...")
    if "alpaca-cleaned" in data_args.data_path:
        train_dataset = load_dataset(data_args.data_path, split="train")
        train_dataset = train_dataset.map(extract_alpaca_dataset, remove_columns=['instruction', "output", "input"])
        response_template = "### Response:\n"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
        model.tokenizer = tokenizer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            args=training_args,
            data_collator=collator,
            packing=False,
            dataset_text_field='text',
        )
    elif "filtered_dataset" in data_args.data_path:
        from datasets import load_from_disk
        train_dataset = load_from_disk(data_args.data_path)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            args=training_args,
            dataset_text_field="text",
            packing=False,
        )
    else:
        train_dataset = load_dataset(data_args.data_path, name="default", split="train", streaming=True)
        # RuntimeError: You can't use batches of different size with `dispatch_batches=True` or when using an `IterableDataset`.either pass `dispatch_batches=False` and have each process fetch its own batch  or pass `split_batches=True`. By doing so, the main process will fetch a full batch and slice it into `num_processes` batches for each process.
        # https://discuss.huggingface.co/t/how-to-handle-iterabledataset-with-huggingface-trainer-and-num-workers-in-ddp-setup/78459
        # Start trainner
        if training_args.packing:
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                args=training_args,
                dataset_text_field="text",
                packing=True,
            )
        else:
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                args=training_args,
                dataset_text_field="text",
                packing=False,
            )
    trainer.use_lora = lora_args.use_lora

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()
