#!/usr/bin/env python3
"""
Test E2E training with smaller dataset to isolate the issue
"""

import torch
import torch.nn.functional as F
import glob
import sys
import os
sys.path.append('..')

from commvq.compress_training import KeyCompressor
from tqdm import tqdm

# Set for better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Constants
N_BITS = 12
M_GROUP = 128
RESIDUAL_NUM = 21
KV_DIM = 1024
CLUSTERING_CENTER_NUM = 4096

def train_small_dataset():
    print("="*60)
    print("E2E Training with Small Dataset")
    print("="*60)
    
    layer_idx = 0
    
    # Load SMALL dataset
    print(f"\nLoading data for layer {layer_idx}...")
    all_tensors = []
    ALL_FILES = glob.glob(f"data/key/{str(layer_idx).zfill(3)}_*.pt")
    ALL_FILES.sort()
    for file in ALL_FILES[:2]:  # Only first 2 files
        tensor = torch.load(file, map_location=torch.device('cpu'))
        all_tensors.append(tensor)
    all_tensors = torch.cat(all_tensors, dim=1).squeeze().float()
    
    # Prepare data - use SMALL subset
    N = 4096  # Only 4096 samples instead of 1M+
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
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare DataLoader (CPU data)
    all_tensors_cpu = all_tensors.cpu()
    tensors_norm_cpu = tensors_norm.cpu()
    dataset = torch.utils.data.TensorDataset(all_tensors_cpu, tensors_norm_cpu)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)
    
    print(f"DataLoader batches: {len(dataloader)}")
    
    # Train for 3 epochs
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (batch_x, batch_norm) in enumerate(pbar):
            batch_x = batch_x.cuda()
            batch_norm = batch_norm.cuda()
            
            optimizer.zero_grad()
            
            # Forward
            batch_x_input = batch_x.unsqueeze(0)
            quantized_x, prescale, commitment_loss = model.encode(batch_x_input, training_method='e2e')
            
            # Loss
            quantized_x = quantized_x.squeeze(0)
            batch_x_normed = batch_x / batch_norm
            quantized_x_norm = torch.norm(quantized_x, dim=1, keepdim=True)
            quantized_x_normed = quantized_x / (quantized_x_norm + 1e-6)
            
            recon_loss = F.mse_loss(quantized_x_normed, batch_x_normed)
            loss = recon_loss + 0.25 * commitment_loss
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nBatch {batch_idx}: NaN/Inf loss detected!")
                print(f"  recon_loss: {recon_loss.item()}")
                print(f"  commitment_loss: {commitment_loss.item()}")
                continue
            
            # Backward
            loss.backward()
            
            # Manual gradient clipping (safer)
            max_norm = 1.0
            total_norm = 0.0
            parameters = [p for p in model.parameters() if p.grad is not None]
            
            # Check gradients and calculate norm
            has_nan_grad = False
            for p in parameters:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    has_nan_grad = True
                    break
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            
            if has_nan_grad or total_norm > 1e6:
                print(f"\nBatch {batch_idx}: Invalid gradients detected, skipping")
                optimizer.zero_grad()
                continue
            
            total_norm = total_norm ** 0.5
            
            # Clip manually
            clip_coef = max_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                for p in parameters:
                    p.grad.data.mul_(clip_coef)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'recon': f'{recon_loss.item():.6f}',
                'commit': f'{commitment_loss.item():.6f}'
            })
            
            # Memory cleanup every 10 batches
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.6f}")
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("✓ Training completed successfully!")
    print("="*60)

if __name__ == "__main__":
    train_small_dataset()

