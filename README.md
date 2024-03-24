# MLX models

Purpose of this repo:

- Learn MLX
- Load timm weights directly to MLX without PyTorch

Usage

```python
import mlx.core as mx
import mlx_models

model = mlx_models.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
model.eval()

model(mx.random.uniform(shape=(1, 224, 224, 3)))  # output shape: (1, 1000)
```

This is possible because:

- timm stores weights in safetensors now and MLX can load safetensors directly.
- From what I have seen so far, all weights shape in MLX is the same as PyTorch, except Conv2d (`(out_channels, kH, kW, in_channels)` in MLX vs `(out_channels, in_channels, kH, kW)` in PyTorch). This can be overcame by overloading `mlx.nn.Module.load_weights()` and transpose the weights accordingly.

Some inference benchmarks on MacBook Air M1.

Model | MLX `0.8.0` throughput (it/s) | PyTorch `2.2.1` (`cpu`) throughput (it/s) | PyTorch `2.2.1` (`mps`) throughput (it/s)
------|-------------------------------|-------------------------------------------|------------------------------------------
vit_tiny_patch16_224.augreg_in21k_ft_in1k | 187.92 | 77.77 | 68.58
vit_base_patch16_224.orig_in21k_ft_in1k | 20.84 | 11.93 | 26.26

ViT TODO:

- [ ] Training-related stuff: weights init, various DropOut types.
- [ ] Support quirky ViT-variants: CLIP/OpenCLIP, SigLIP, GAP/fc_norm
