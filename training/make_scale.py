import numpy as np
import glob
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
N = 32 # Number of layers in LLM

def handle_one(prefix):
    x1, idx = prefix.split("_")
    files = glob.glob(f"data/{x1}/{idx}_*.pt")
    
    # Check if files exist
    if len(files) == 0:
        print(f"⚠️  No files found for {prefix} (data/{x1}/{idx}_*.pt), skipping...")
        return None
    
    print(f"Processing {prefix}: {len(files)} files")
    
    # sample 1024 files
    # files = np.random.choice(files, 1024, replace=False)
    tensors = []
    for file in tqdm(sorted(files), desc=f"{prefix}"):
        try:
            t = torch.load(file, map_location="cpu")[0]
            if t.isnan().any():
                print(f"  ⚠️  NaN in {file}, skipping")
                continue
            if t.sum() == 0:
                print(f"  ⚠️  Zero tensor in {file}, skipping")
                continue
            tensors.append(t)
        except Exception as e:
            print(f"  ⚠️  Error loading {file}: {e}")
    
    if len(tensors) == 0:
        print(f"⚠️  No valid tensors for {prefix}, skipping...")
        return None
    
    tensors = torch.cat(tensors, axis=0)
    tqdm.write(f"  ✓ {prefix}: {tensors.shape}")
    mean = torch.mean(torch.abs(tensors), axis=0).cpu()
    return mean



if __name__ == "__main__":
    print("=" * 60)
    print("Computing scaling factors for E2E-CommVQ")
    print("=" * 60)
    
    prefix = []
    for i in range(N):
        prefix.append(f"key_{i:03d}")
        prefix.append(f"value_{i:03d}")
    
    print(f"\nProcessing {len(prefix)} layer components (key + value)...")
    means = process_map(handle_one, prefix, max_workers=8)
    
    learnable_scale = {
        "key": {},
        "value": {}
    }
    
    skipped = 0
    for m, p in zip(means, prefix):
        if m is None:
            skipped += 1
            continue
        x1, idx = p.split("_")
        idx = int(idx)
        learnable_scale[x1][idx] = m
    
    print("\n" + "=" * 60)
    print(f"✓ Processed: {len(means) - skipped} components")
    print(f"  Skipped: {skipped} components (no data)")
    print(f"  Key layers: {len(learnable_scale['key'])}")
    print(f"  Value layers: {len(learnable_scale['value'])}")
    
    torch.save(learnable_scale, f"data/learnable_scale.pt")
    print(f"✓ Saved to: data/learnable_scale.pt")
    print("=" * 60)
