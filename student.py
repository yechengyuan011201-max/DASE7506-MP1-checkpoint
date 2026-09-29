"""Student implementation for the MP1 project."""

import torch
from torch import nn
from torch.nn import functional as F


class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=256, base=10000.0):
        super().__init__()

        assert head_dim % 2 == 0

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2).float() / head_dim
            )
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)

        self.register_buffer(
            "cos_cached",
            freqs.cos()[None, None, :, :],
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            freqs.sin()[None, None, :, :],
            persistent=False,
        )

    def rotate(self, x, cos, sin):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        return torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        ).flatten(-2)

    def forward(self, q, k):
        length = q.shape[-2]

        cos = self.cos_cached[:, :, :length, :].to(
            device=q.device,
            dtype=q.dtype,
        )

        sin = self.sin_cached[:, :, :length, :].to(
            device=q.device,
            dtype=q.dtype,
        )

        q = self.rotate(q, cos, sin)
        k = self.rotate(k, cos, sin)

        return q, k


class SwiGLU(nn.Module):
    def __init__(self, width=128, hidden=341):
        super().__init__()
        self.gate = nn.Linear(width, hidden)
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class StudentBlock(nn.Module):
    def __init__(self, width=128, heads=4, hidden=341, context=256):
        super().__init__()

        self.heads = heads

        assert width % heads == 0

        head_dim = width // heads

        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)

        # QK-Norm:
        # Normalize Q and K independently within each attention head.
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

        self.rope = RoPE(
            head_dim=head_dim,
            max_seq_len=context,
        )

        self.mlp = SwiGLU(width, hidden=hidden)

    def forward(self, x):
        batch, length, width = x.shape

        q, k, v = self.qkv(self.norm1(x)).view(
            batch,
            length,
            3,
            self.heads,
            width // self.heads,
        ).permute(2, 0, 3, 1, 4)

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE after QK normalization
        q, k = self.rope(q, k)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )

        x = x + self.proj(
            attended.transpose(1, 2).reshape(
                batch,
                length,
                width,
            )
        )

        return x + self.mlp(self.norm2(x))


class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = dict(config)
        self.context = config["context"]

        width = config["width"]
        hidden = config.get("hidden", 341)

        self.token = nn.Embedding(
            config["vocab"],
            width,
        )

        self.blocks = nn.ModuleList([
            StudentBlock(
                width=width,
                heads=config["heads"],
                hidden=hidden,
                context=self.context,
            )
            for _ in range(config["depth"])
        ])

        self.norm = nn.LayerNorm(width)

        self.head = nn.Linear(
            width,
            config["vocab"],
            bias=False,
        )

        self.apply(self.initialize)

        # Weight tying
        self.head.weight = self.token.weight

    def initialize(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                std=0.02,
            )

    def features(self, ids):
        x = self.token(ids)

        for block in self.blocks:
            x = block(x)

        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(
            self(ids).float(),
            dim=-1,
        )


def build_model(config):
    return StudentGPT(config)
