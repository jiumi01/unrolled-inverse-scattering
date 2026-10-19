import torch
sd = torch.load("results/ckpt_constrained_K8.pt", map_location="cpu")
for k, v in sd.items():
    print(k, tuple(v.shape) if hasattr(v, "shape") else v)
