"""
E2E-CommVQ: KV Cache Collection Script

This script collects KV cache from a pretrained model for E2E training.
Unlike EM training which uses a trained model, E2E directly uses the base model.
"""

import argparse
import os
import sys

# Disable flash-attn import to avoid GLIBC errors
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
sys.modules['flash_attn'] = None
sys.modules['flash_attn.bert_padding'] = None
sys.modules['flash_attn.flash_attn_interface'] = None

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Collect KV cache for E2E training')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pretrained model or HuggingFace model name')
    parser.add_argument('--dataset', type=str, default='HuggingFaceFW/fineweb-edu',
                       help='Dataset to use for KV collection')
    parser.add_argument('--output_dir', type=str, default='data/key',
                       help='Directory to save collected KV cache')
    parser.add_argument('--num_samples', type=int, default=1000000,
                       help='Number of samples to process')
    parser.add_argument('--max_seq_length', type=int, default=8192,
                       help='Maximum sequence length')
    parser.add_argument('--quant_bits', type=int, default=2,
                       help='Quantization bits (1 or 2)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing')
    return parser.parse_args()


def collect_kv_cache(args):
    """Collect KV cache from model"""
    
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Load model WITHOUT CommVQ (we want the base model's KV cache)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # Use SDPA instead of flash_attention_2 for GLIBC compatibility
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    print(f"Loading dataset {args.dataset}...")
    if args.dataset.startswith("HuggingFaceFW/"):
        dataset = load_dataset(args.dataset, split='train', streaming=True)
    else:
        dataset = load_dataset(args.dataset, split='train')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get number of layers
    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")
    
    # Initialize storage for each layer
    kv_caches = {layer_idx: [] for layer_idx in range(num_layers)}
    
    print(f"Collecting KV cache from {args.num_samples} samples...")
    
    samples_processed = 0
    tokens_per_sample = args.max_seq_length
    
    with torch.no_grad():
        for example in tqdm(dataset, total=args.num_samples):
            if samples_processed >= args.num_samples:
                break
            
            # Get text
            text = example.get('text', '')
            if not text:
                continue
            
            # Tokenize
            inputs = tokenizer(
                text,
                max_length=args.max_seq_length,
                truncation=True,
                return_tensors='pt'
            ).to(model.device)
            
            if inputs.input_ids.shape[1] < 100:  # Skip very short sequences
                continue
            
            # Forward pass to get KV cache
            outputs = model(
                **inputs,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True
            )
            
            # Extract key cache from past_key_values
            # past_key_values format: tuple of (key, value) for each layer
            # key shape: [batch, num_heads, seq_len, head_dim]
            past_key_values = outputs.past_key_values
            
            for layer_idx in range(num_layers):
                key_cache = past_key_values[layer_idx][0]  # Get key (not value)
                
                # Convert to [seq_len, num_heads * head_dim] format
                # key_cache: [1, num_heads, seq_len, head_dim]
                key_cache = key_cache.squeeze(0)  # [num_heads, seq_len, head_dim]
                key_cache = key_cache.permute(1, 0, 2)  # [seq_len, num_heads, head_dim]
                key_cache = key_cache.flatten(1)  # [seq_len, num_heads * head_dim]
                
                kv_caches[layer_idx].append(key_cache.cpu())
            
            samples_processed += 1
            
            # Save periodically to avoid memory issues
            if samples_processed % 1000 == 0:
                print(f"Processed {samples_processed} samples, saving intermediate results...")
                for layer_idx in range(num_layers):
                    if len(kv_caches[layer_idx]) > 0:
                        # Concatenate and save
                        layer_cache = torch.cat(kv_caches[layer_idx], dim=0)
                        
                        # Save to file
                        filename = os.path.join(
                            args.output_dir,
                            f"{str(layer_idx).zfill(3)}_{samples_processed//1000}.pt"
                        )
                        torch.save(layer_cache.unsqueeze(0), filename)
                        
                        # Clear memory
                        kv_caches[layer_idx] = []
                
                torch.cuda.empty_cache()
    
    # Save remaining data
    print("Saving final results...")
    for layer_idx in range(num_layers):
        if len(kv_caches[layer_idx]) > 0:
            layer_cache = torch.cat(kv_caches[layer_idx], dim=0)
            filename = os.path.join(
                args.output_dir,
                f"{str(layer_idx).zfill(3)}_final.pt"
            )
            torch.save(layer_cache.unsqueeze(0), filename)
    
    print(f"✓ KV cache collection completed!")
    print(f"  Processed {samples_processed} samples")
    print(f"  Saved to {args.output_dir}/")


if __name__ == "__main__":
    args = parse_args()
    collect_kv_cache(args)

