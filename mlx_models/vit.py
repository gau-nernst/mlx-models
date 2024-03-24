from functools import partial

import mlx.core as mx
from mlx import nn


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)

    def __call__(self, x):
        return self.proj(x).flatten(-3, -2)  # NHWC -> NLC


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        N, L, D = x.shape
        qkv = self.qkv(x).reshape(N, L, 3 * self.num_heads, self.head_dim).swapaxes(1, 2)
        q, k, v = qkv.split(3, axis=1)

        attn = mx.softmax(self.scale * q @ k.swapaxes(-2, -1), axis=-1)
        x = attn @ v

        x = x.swapaxes(1, 2).reshape(N, L, D)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class ViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        class_token: bool = True,
        fc_norm: bool | None = None,
        norm_layer: type[nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        assert global_pool in ("avg", "token")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        self.in_chans = in_chans
        self.global_pool = global_pool
        self.num_prefix_tokens = 1 if class_token else 0

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token = mx.zeros((1, 1, embed_dim))

        n_patches = (img_size // patch_size) ** 2 + 1
        self.pos_embed = mx.zeros((1, n_patches, embed_dim))
        self.blocks = [
            ViTBlock(embed_dim, num_heads, mlp_ratio, norm_layer=norm_layer, act_layer=act_layer) for _ in range(depth)
        ]
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        x = self.patch_embed(x)
        cls_token = mx.repeat(self.cls_token, x.shape[0], axis=0)
        x = mx.concatenate([cls_token, x], axis=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(axis=1)
        elif self.global_pool == "token":
            x = x[:, 0]

        return self.head(self.fc_norm(x))

    def load_weights(self, file_or_weights: str | list[tuple[str, mx.array]], strict: bool = True) -> None:
        weights = file_or_weights
        if isinstance(weights, str):
            weights = list(mx.load(weights).items())

        # patch PyTorch Conv2d weights
        for i, (k, v) in enumerate(weights):
            if k == "patch_embed.proj.weight" and v.shape[1] == self.in_chans:
                weights[i] = (k, v.transpose(0, 2, 3, 1))

        return super().load_weights(weights, strict)
