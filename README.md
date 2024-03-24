# MLX models

Purpose of this repo:

- Learn MLX
- Load timm weights directly to MLX without PyTorch

For example, download `.safetensors` weights

```bash
wget https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/resolve/main/model.safetensors?download=true -O model.safetensors
```

Load weights directly to MLX model

```python
from mlx_models import ViT

model = ViT(embed_dim=192, depth=12, num_heads=3, num_classes=1000)
model.load_weights("model.safetensors")
```

This is possible because:

- timm stores weights in safetensors now and MLX can load safetensors directly.
- From what I have seen so far, all weights shape in MLX is the same as PyTorch, except Conv2d (`(out_channels, kH, kW, in_channels)` in MLX vs `(out_channels, in_channels, kH, kW)` in PyTorch). This can be overcame by overloading `mlx.nn.Module.load_weights()` and transpose the weights accordingly.

Some inference benchmarks on MacBook Air M1. (PyTorch is in eager mode, since I found `torch.compile()` is slower).

Model | MLX `0.8.0` throughput (it/s) | PyTorch `2.2.1` throughput (it/s)
------|-------------------------------|--------------------------
vit_tiny_patch16_224.augreg_in21k_ft_in1k | 182.56 | 73.26
vit_base_patch16_224.orig_in21k_ft_in1k | 20.81 | 11.90

ViT TODO:

- [ ] Create model from timm's model tag string. This requires inferring model config from weights shape (full model config is not available in HF page)
- [ ] Training-related stuff: weights init, various DropOut types.
