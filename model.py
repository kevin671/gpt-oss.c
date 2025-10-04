from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 2880
    n_layers: int = 38
    n_experts: int = 128
    experts_per_token: int = 4
    n_heads: int = 64
    n_kv_heads: int = 8
    vocab_size: int = 201088
    hidden_dim: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    sliding_window: int = 128
    context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class MoEFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_experts: int, k: int, swiglu_limit: float):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Linear(dim, num_experts)
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim * 2, dim))
        self.b1 = nn.Parameter(torch.empty(num_experts, hidden_dim * 2))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.b2 = nn.Parameter(torch.empty(num_experts, dim))
        self.swiglu_limit = swiglu_limit

    def forward(self, x):  # this for token wise
        g = self.gate(x)
        experts = torch.topk(g, self.k, dim=-1)
        expert_indices = experts.indices  # (B, k)
        expert_weights = F.softmax(experts.values, dim=-1)  # (B, k)

        w1 = self.w1[expert_indices, ...]
        b1 = self.b1[expert_indices, ...]
        w2 = self.w2[expert_indices, ...]
        b2 = self.b2[expert_indices, ...]

        h = torch.einsum("beck,bk->bec", w1, x) + b1
        h_glu, h_linear = h[..., ::2], h[..., 1::2]
        h_glu = torch.clamp(h_glu, max=self.swiglu_limit)
        h_linear = torch.clamp(h_linear, min=-self.swiglu_limit, max=self.swiglu_limit)
        h = (h_glu * torch.sigmoid(1.702 * h_glu)) * (h_linear + 1)

        h = torch.einsum("beck,bek->bec", w2, h) + b2

        h = torch.einsum("bec,be->bc", h, expert_weights)
        return h


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim)
        self.sliding_window = args.sliding_window if layer_idx % 2 == 0 else 0
        self.sinks = nn.Parameter(torch.empty(args.n_heads))
        self.rope = None  # no parameters

    def forward(self, x):
        pass


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, layer_id)
        self.feed_forward = MoEFeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            num_experts=args.n_experts,
            k=args.experts_per_token,
            swiglu_limit=args.swiglu_limit,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))  # maybe wrong
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

    def forward(self, x):
        x = self.tok_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output(x)
        return x

    def from_checkpoint(self, checkpoint_path: str):
        pass

    @torch.no_grad()
    def generate(self, prompt_tokens, temperature, max_new_tokens):
        pass
