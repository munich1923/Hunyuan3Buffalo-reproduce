"""Padding-mask extension of the pinned DiT; parameter keys/shapes stay intact."""
import torch
from torch import nn
import torch.nn.functional as F

from hy3dshape.models.denoisers.hunyuandit import CrossAttention, HunYuanDiTBlock, HunYuanDiTPlain


def validate_mask(context, mask):
    if mask.dtype != torch.bool or mask.shape != context.shape[:2] or mask.device != context.device:
        raise ValueError("main_mask must be bool [B,L] on the context device")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Each condition needs at least one valid token")


class MaskedCrossAttention(CrossAttention):
    def __init__(self, original):
        nn.Module.__init__(self)
        if original.with_dca:
            raise ValueError("The reviewed text experiment does not use decoupled attention")
        # Adopt the original modules without allocating parameters or consuming RNG.
        for name, module in original._modules.items():
            self.add_module(name, module)
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.with_dca = False

    def forward(self, x, y, attention_mask=None):
        if attention_mask is None:
            return super().forward(x, y)
        validate_mask(y, attention_mask)
        b, length, _ = x.shape
        # Padding never contributes, including through projection biases.
        y = y.masked_fill(~attention_mask.unsqueeze(-1), 0)
        q = self.to_q(x).view(b, length, self.num_heads, self.head_dim)
        # Preserve the pinned upstream projection packing exactly; changing the
        # head layout here would change the pretrained model's computation.
        kv = torch.cat((self.to_k(y), self.to_v(y)), dim=-1)
        kv = kv.view(1, -1, self.num_heads, self.head_dim * 2)
        k, v = torch.split(kv, self.head_dim, dim=-1)
        k = k.reshape(b, y.shape[1], self.num_heads, self.head_dim)
        v = v.reshape(b, y.shape[1], self.num_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        # SDPA bool True means allowed. Leave backend selection to PyTorch so
        # masked calls can use the math fallback if fused kernels do not support it.
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attention_mask[:, None, None, :],
        ).transpose(1, 2).reshape(b, length, -1)
        return self.out_proj(output)


class MaskedDiTBlock(HunYuanDiTBlock):
    def __init__(self, original):
        nn.Module.__init__(self)
        for name, module in original._modules.items():
            self.add_module(name, MaskedCrossAttention(module) if name == "attn2" else module)
        self.skip_linear = original.skip_linear
        self.timested_modulate = original.timested_modulate
        self.use_moe = original.use_moe

    def forward(self, x, c=None, text_states=None, skip_value=None, attention_mask=None):
        if attention_mask is None:
            return super().forward(x, c, text_states, skip_value)
        if self.skip_linear is not None:
            x = self.skip_norm(self.skip_linear(torch.cat([skip_value, x], dim=-1)))
        if self.timested_modulate:
            x = x + self.default_modulation(c).unsqueeze(dim=1)
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), text_states, attention_mask=attention_mask)
        mlp_inputs = self.norm3(x)
        return x + (self.moe(mlp_inputs) if self.use_moe else self.mlp(mlp_inputs))


class MaskedHunYuanDiTPlain(HunYuanDiTPlain):
    """Official weights and computation, with an explicit condition-mask path."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_attention_pooling or self.with_decoupled_ca:
            raise ValueError("Text stage2 requires no attention pooling or decoupled condition")
        self.blocks = nn.ModuleList([MaskedDiTBlock(block) for block in self.blocks])

    def forward(self, x, t, contexts, **kwargs):
        mask = contexts.get("main_mask")
        if mask is None:
            return super().forward(x, t, contexts, **kwargs)
        cond = contexts["main"]
        validate_mask(cond, mask)
        c = self.t_embedder(t, condition=kwargs.get("guidance_cond"))
        x = self.x_embedder(x)
        if self.use_pos_emb:
            x = x + self.pos_embed.to(x.dtype)
        x = torch.cat([c, x], dim=1)
        skip_values = []
        for layer, block in enumerate(self.blocks):
            skip = None if layer <= self.depth // 2 else skip_values.pop()
            x = block(x, c, cond, skip_value=skip, attention_mask=mask)
            if layer < self.depth // 2:
                skip_values.append(x)
        return self.final_layer(x)
