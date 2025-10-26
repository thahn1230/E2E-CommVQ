#!/usr/bin/env python3
"""
Debug script to check what data files exist and their format
"""

import os
import glob
import torch

def check_data_structure():
    print("=" * 60)
    print("Checking E2E-CommVQ Data Structure")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists("data"):
        print("❌ ERROR: data/ directory does not exist!")
        print("   Please run Step 1 (KV cache collection) first")
        return False
    
    print("✓ data/ directory exists")
    
    # Check subdirectories
    subdirs = ["key", "value"]
    for subdir in subdirs:
        path = f"data/{subdir}"
        if os.path.exists(path):
            files = glob.glob(f"{path}/*.pt")
            print(f"✓ data/{subdir}/ exists with {len(files)} files")
            
            if len(files) > 0:
                # Show first few file names
                print(f"  Sample files:")
                for f in sorted(files)[:5]:
                    basename = os.path.basename(f)
                    size = os.path.getsize(f) / (1024**2)  # MB
                    print(f"    - {basename} ({size:.1f} MB)")
                
                if len(files) > 5:
                    print(f"    ... and {len(files) - 5} more")
                
                # Try to load first file and check format
                try:
                    first_file = sorted(files)[0]
                    data = torch.load(first_file, map_location='cpu')
                    print(f"  Data format: {type(data)}")
                    
                    if isinstance(data, torch.Tensor):
                        print(f"  Tensor shape: {data.shape}")
                    elif isinstance(data, list):
                        print(f"  List length: {len(data)}")
                        if len(data) > 0:
                            print(f"  First element shape: {data[0].shape}")
                    elif isinstance(data, dict):
                        print(f"  Dict keys: {list(data.keys())}")
                
                except Exception as e:
                    print(f"  ⚠️  Could not load file: {e}")
        else:
            print(f"❌ data/{subdir}/ does not exist")
    
    # Check for scaling factors
    scale_file = "data/learnable_scale.pt"
    if os.path.exists(scale_file):
        print(f"✓ {scale_file} exists")
        try:
            scale_data = torch.load(scale_file, map_location='cpu')
            print(f"  Keys: {list(scale_data.keys())}")
            if 'key' in scale_data:
                print(f"  Key layers: {len(scale_data['key'])} layers")
            if 'value' in scale_data:
                print(f"  Value layers: {len(scale_data['value'])} layers")
        except Exception as e:
            print(f"  ⚠️  Could not load: {e}")
    else:
        print(f"❌ {scale_file} does not exist")
        print("   Run Step 2 (make_scale.py) to create it")
    
    print("=" * 60)
    
    # Summary
    key_files = glob.glob("data/key/*.pt")
    if len(key_files) > 0:
        print(f"\n✓ Data collection appears complete!")
        print(f"  {len(key_files)} key cache files found")
        return True
    else:
        print(f"\n❌ No key cache files found!")
        print("   Please run: python collect_kv_for_e2e.py ...")
        return False

if __name__ == "__main__":
    check_data_structure()

