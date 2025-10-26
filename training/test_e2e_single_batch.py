#!/usr/bin/env python3
"""
Test E2E training with a single batch for debugging
"""

import torch
import torch.nn.functional as F
import glob
import sys
sys.path.append('..')

from commvq.compress_training import KeyCompressor

# Constants (from quantize_key_cache.py)
N_BITS = 12
M_GROUP = 128
RESIDUAL_NUM = 21
KV_DIM = 1024
CLUSTERING_CENTER_NUM = 4096

def test_single_batch():
    print("="*60)
    print("Testing E2E Training with Single Batch")
    print("="*60)
    
    layer_idx = 0
    
    # Load data
    print(f"\nLoading data for layer {layer_idx}...")
    all_tensors = []
    ALL_FILES = glob.glob(f"data/key/{str(layer_idx).zfill(3)}_*.pt")
    ALL_FILES.sort()
    for file in ALL_FILES[:2]:  # Only load first 2 files for quick test
        tensor = torch.load(file, map_location=torch.device('cpu'))
        all_tensors.append(tensor)
    all_tensors = torch.cat(all_tensors, dim=1).squeeze().float()
    
    # Prepare data
    N = min(1024, all_tensors.shape[0])  # Only 1024 samples for test
    all_tensors = all_tensors[:N].cuda()
    all_tensors = all_tensors.view(N, 8, 2, 64).transpose(2, 3).flatten(1)
    tensors_norm = all_tensors.norm(dim=1, keepdim=True)
    
    print(f"Data shape: {all_tensors.shape}")
    print(f"Norm shape: {tensors_norm.shape}")
    
    # Initialize model
    print(f"\nInitializing model...")
    model = KeyCompressor(
        feat_dim=KV_DIM,
        layer_idx=layer_idx,
        quant_bits=N_BITS // 6,
        num_residuals=RESIDUAL_NUM,
        group_size=M_GROUP
    ).cuda()
    
    print(f"Model parameters:")
    print(f"  feat_dim: {model.feat_dim}")
    print(f"  num_groups: {model.num_groups}")
    print(f"  group_size: {model.group_size}")
    print(f"  codebook_size: {model.codebook_size}")
    print(f"  num_residuals: {model.num_residuals}")
    
    # Test with single batch
    print(f"\nTesting forward pass with batch_size=16...")
    batch_size = 16
    batch_x = all_tensors[:batch_size]  # [16, 1024]
    batch_norm = tensors_norm[:batch_size]  # [16, 1]
    
    print(f"Input shapes:")
    print(f"  batch_x: {batch_x.shape}")
    print(f"  batch_norm: {batch_norm.shape}")
    
    # Add batch dimension
    batch_x_input = batch_x.unsqueeze(0)  # [1, 16, 1024]
    print(f"  batch_x_input: {batch_x_input.shape}")
    
    # Forward pass
    print(f"\nForward pass...")
    try:
        model.train()
        quantized_x, prescale, commitment_loss = model.encode(batch_x_input, training_method='e2e')
        
        print(f"✓ Forward pass successful!")
        print(f"Output shapes:")
        print(f"  quantized_x: {quantized_x.shape}")
        print(f"  prescale: {prescale.shape}")
        print(f"  commitment_loss: {commitment_loss.item():.6f}")
        
        # Test loss computation
        print(f"\nTesting loss computation...")
        quantized_x = quantized_x.squeeze(0)  # [16, 1024]
        batch_x_normed = batch_x / batch_norm
        quantized_x_norm = torch.norm(quantized_x, dim=1, keepdim=True)
        quantized_x_normed = quantized_x / (quantized_x_norm + 1e-6)
        
        recon_loss = F.mse_loss(quantized_x_normed, batch_x_normed)
        loss = recon_loss + 0.25 * commitment_loss
        
        print(f"✓ Loss computation successful!")
        print(f"  recon_loss: {recon_loss.item():.6f}")
        print(f"  total_loss: {loss.item():.6f}")
        
        # Test backward
        print(f"\nTesting backward pass...")
        loss.backward()
        print(f"✓ Backward pass successful!")
        
        print(f"\n{'='*60}")
        print(f"✓ All tests passed!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Error occurred:")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        
        # Print shapes for debugging
        print(f"\nDebugging info:")
        print(f"  Model encoder output dim: {model.intermediate_dim}")
        print(f"  Expected encoder output: {model.num_groups * model.num_residuals * model.codebook_size}")

if __name__ == "__main__":
    # Set environment for better error messages
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    test_single_batch()

