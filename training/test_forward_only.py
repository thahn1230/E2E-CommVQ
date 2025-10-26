#!/usr/bin/env python3
"""
Test only forward pass without backward to isolate the issue
"""

import torch
import glob
import sys
import os
sys.path.append('..')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from commvq.compress_training import KeyCompressor

# Constants
N_BITS = 12
M_GROUP = 128
RESIDUAL_NUM = 21
KV_DIM = 1024

def test_forward_only():
    print("="*60)
    print("Testing Forward Pass Only")
    print("="*60)
    
    layer_idx = 0
    
    # Load small data
    print(f"\nLoading data...")
    all_tensors = []
    ALL_FILES = glob.glob(f"data/key/{str(layer_idx).zfill(3)}_*.pt")
    ALL_FILES.sort()
    for file in ALL_FILES[:1]:  # Only 1 file
        tensor = torch.load(file, map_location=torch.device('cpu'))
        all_tensors.append(tensor)
    all_tensors = torch.cat(all_tensors, dim=1).squeeze().float()
    
    # Very small subset
    N = 128
    all_tensors = all_tensors[:N].cuda()
    all_tensors = all_tensors.view(N, 8, 2, 64).transpose(2, 3).flatten(1)
    tensors_norm = all_tensors.norm(dim=1, keepdim=True)
    
    print(f"Data shape: {all_tensors.shape}")
    
    # Initialize model
    print(f"\nInitializing model...")
    model = KeyCompressor(
        feat_dim=KV_DIM,
        layer_idx=layer_idx,
        quant_bits=N_BITS // 6,
        num_residuals=RESIDUAL_NUM,
        group_size=M_GROUP
    ).cuda()
    
    model.train()
    
    # Test with single batch
    batch_size = 16
    batch_x = all_tensors[:batch_size]
    batch_norm = tensors_norm[:batch_size]
    
    print(f"\nTest batch shapes:")
    print(f"  batch_x: {batch_x.shape}")
    print(f"  batch_norm: {batch_norm.shape}")
    
    # Test forward pass step by step
    print(f"\n--- Forward Pass ---")
    
    try:
        # Add batch dimension
        batch_x_input = batch_x.unsqueeze(0)  # [1, 16, 1024]
        print(f"✓ Input prepared: {batch_x_input.shape}")
        
        # Forward
        print(f"Starting encode...")
        with torch.no_grad():  # NO GRADIENTS - just forward
            quantized_x, prescale, commitment_loss = model.encode(batch_x_input, training_method='e2e')
        
        print(f"✓ Forward pass completed!")
        print(f"  quantized_x: {quantized_x.shape}")
        print(f"  prescale: {prescale.shape}")
        print(f"  commitment_loss: {commitment_loss.item():.6f}")
        
        # Check for NaN
        if torch.isnan(quantized_x).any():
            print(f"✗ NaN detected in quantized_x")
        else:
            print(f"✓ No NaN in output")
        
        print(f"\n{'='*60}")
        print(f"✓ Forward-only test PASSED!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Error in forward pass:")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        
        print(f"\nModel info:")
        print(f"  codebook_params shape: {model.codebook_params.shape}")
        print(f"  codebook_size: {model.codebook_size}")
        print(f"  num_residuals: {model.num_residuals}")
        print(f"  num_groups: {model.num_groups}")

if __name__ == "__main__":
    test_forward_only()

