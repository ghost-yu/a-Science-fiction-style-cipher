from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class Config:
    vocab_size: int = 8232
    block_size: int = 512
    layers: int = 6
    heads: int = 8
    width: int = 512
    dropout: float = 0.1

class Attention(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.qkv = nn.Linear(c.width, 3 * c.width)
        self.proj = nn.Linear(c.width, c.width)
        self.heads = c.heads
        self.dropout = c.dropout
        self.resid_dropout = nn.Dropout(c.dropout)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.resid_dropout(self.proj(y))

class Block(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.n1 = nn.LayerNorm(c.width)
        self.attn = Attention(c)
        self.n2 = nn.LayerNorm(c.width)
        self.mlp = nn.Sequential(nn.Linear(c.width, 4 * c.width), nn.GELU(approximate='tanh'), nn.Linear(4 * c.width, c.width), nn.Dropout(c.dropout))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))

class CharTransformer(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.config = c
        self.token = nn.Embedding(c.vocab_size, c.width)
        self.position = nn.Embedding(c.block_size, c.width)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.Sequential(*[Block(c) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.width)
        self.head = nn.Linear(c.width, c.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids, targets=None):
        _, t = ids.shape
        x = self.drop(self.token(ids) + self.position(torch.arange(t, device=ids.device)))
        x = self.norm(self.blocks(x))
        logits = self.head(x if targets is not None else x[:, -1:])
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return (logits, loss)

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature=0.8, top_k=30, eos_id=2, forbidden=(0, 1, 3)):
        for _ in range(max_new_tokens):
            logits, _ = self(ids[:, -self.config.block_size:])
            logits = logits[:, -1, :].float() / temperature
            logits[:, list(forbidden)] = -float('inf')
            if top_k:
                cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, -float('inf'))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            ids = torch.cat((ids, nxt), dim=1)
            if int(nxt[0, 0]) == eos_id:
                break
        return ids
