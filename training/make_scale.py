import numpy as np
import glob
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
# model = "llama2-7b-32k"
# model = "Llama-3.1-70B-Instruct"
# model = "Qwen2.5-7B-Instruct"

def handle_one(prefix):
    x1, idx = prefix.split("_")
    files = glob.glob(f"data/{x1}/{idx}_*.pt")
    # sample 1024 files
    # files = np.random.choice(files, 1024, replace=False)
    tensors = []
    for file in tqdm(files):
        try:
            t = torch.load(file, map_location="cpu")[0]
            if t.isnan().any():
                print(f"Error: {file}")
                continue
            if t.sum() == 0:
                print(f"Error: {file}")
                continue
            tensors.append(t)
        except:
            print(f"Error: {file}")
    tensors = torch.cat(tensors, axis=0)
    tqdm.write(f"{tensors.shape}")
    mean = torch.mean(torch.abs(tensors), axis=0).cpu()
    return mean



if __name__ == "__main__":
    prefix = []
    for i in range(28):
        prefix.append(f"key_{i:03d}")
        prefix.append(f"value_{i:03d}")
    means = process_map(handle_one, prefix, max_workers=8)
    learnable_scale = {
        "key": {},
        "value": {}
    }
    for m, p in zip(means, prefix):
        x1, idx = p.split("_")
        idx = int(idx)
        learnable_scale[x1][idx] = m
    torch.save(learnable_scale, f"data/learnable_scale.pt")
    print("Done")
