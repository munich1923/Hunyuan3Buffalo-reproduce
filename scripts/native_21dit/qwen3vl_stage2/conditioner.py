"""Replace DINO with frozen text-only Qwen plus a trainable tokenwise MLP."""
from pathlib import Path

import torch
from torch import nn

from .text import MODEL_REVISION, TASK_PREFIX, messages_for


class _FrozenBF16(nn.Module):
    """Keep backbone dtypes when the official pipeline/Lightning casts its parent.

    Probe the requested device with an empty tensor, then move the original
    tensor without first casting it through FP16 (which could lose information).
    Preserve FP32 rotary buffers as well as BF16 model parameters.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    def _apply(self, fn, recurse=True):
        def device_only(tensor):
            probe = fn(torch.empty(0, device=tensor.device, dtype=tensor.dtype))
            return tensor.to(device=probe.device)
        return super()._apply(device_only, recurse=recurse)


class Qwen3VLTextConditioner(nn.Module):
    def __init__(self, model_path, revision=MODEL_REVISION, task_prefix=TASK_PREFIX,
                 max_length=2048, drop_ratio=0.1, disable_drop=True,
                 local_files_only=True, backbone=None, tokenizer=None):
        super().__init__()
        if backbone is None or tokenizer is None:
            if not Path(model_path).is_dir():
                raise FileNotFoundError("Provision the pinned Qwen snapshot on H100: " + str(model_path))
            from .contracts import qwen_files
            qwen_files(model_path, revision)
            from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, revision=revision, local_files_only=local_files_only)
            backbone = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, revision=revision, local_files_only=local_files_only,
                dtype=torch.bfloat16, attn_implementation="sdpa")
        if backbone.config.text_config.hidden_size != 4096:
            raise ValueError("Expected Qwen3-VL-8B text hidden_size=4096")
        if any(p.is_floating_point() and p.dtype != torch.bfloat16 for p in backbone.parameters()):
            raise ValueError("Qwen weights must be BF16 before wrapping")
        self.backbone = _FrozenBF16(backbone)
        self.tokenizer = tokenizer
        # Left padding supports causal language generation in the separate probe.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The pinned Qwen tokenizer must define a pad token")
        self.connector = nn.Sequential(nn.Linear(4096, 1024), nn.GELU(), nn.Linear(1024, 1024))
        self.task_prefix = task_prefix
        self.max_length = int(max_length)
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        self.drop_ratio = float(drop_ratio)
        self.disable_drop = bool(disable_drop)
        self.last_token_count = None
        self.last_attention_mask = None

    @property
    def vlm(self):
        return self.backbone.model

    def tokenize_text(self, text):
        texts = [text] if isinstance(text, str) else text
        if not isinstance(texts, (list, tuple)) or not texts:
            raise ValueError("Expected a nonempty batch of caption strings")
        messages = [messages_for(item, self.task_prefix) for item in texts]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, padding=True, truncation=False,
            add_generation_prompt=True, return_dict=True, return_tensors="pt")
        allowed = {"input_ids", "attention_mask"}
        if set(inputs) - allowed:
            raise ValueError("Unexpected non-text tokenizer inputs")
        if "attention_mask" not in inputs:
            raise ValueError("Batched text tokenization must return its padding mask")
        ids, valid = inputs["input_ids"], inputs["attention_mask"]
        if ids.ndim != 2 or ids.shape[0] != len(texts) or valid.shape != ids.shape:
            raise ValueError("Tokenizer returned inconsistent batch/mask shapes")
        if not bool(((valid == 0) | (valid == 1)).all()) or not bool(valid.bool().any(dim=1).all()):
            raise ValueError("Every caption must contain valid tokens and a binary mask")
        if ids.shape[1] > self.max_length:
            raise ValueError("Caption exceeds max_length; review it instead of silently truncating")
        # Reject multimodal placeholders even if present in a caption string.
        image_ids = [getattr(self.vlm.config, name, None) for name in
                     ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")]
        if any(token is not None and bool((inputs["input_ids"] == token).any()) for token in image_ids):
            raise ValueError("Image/video tokens are forbidden in text-only stage2")
        device = next(self.vlm.parameters()).device
        return {key: value.to(device) for key, value in inputs.items()}

    def encode_text(self, text):
        inputs = self.tokenize_text(text)
        self.backbone.eval()
        # no_grad applies ONLY to Qwen; the connector must retain its graph.
        with torch.no_grad():
            hidden = self.vlm.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
        valid = inputs["attention_mask"].bool()
        hidden = hidden.detach().masked_fill(~valid.unsqueeze(-1), 0)
        if hidden.shape[:2] != valid.shape or not bool(torch.isfinite(hidden).all()):
            raise RuntimeError("Empty/nonfinite Qwen text hidden state")
        self.last_token_count = hidden.shape[1]
        self.last_attention_mask = valid.detach().clone()
        return hidden, valid

    def forward(self, image=None, text=None, mask=None, **kwargs):
        if image is not None or mask is not None or kwargs:
            raise ValueError("Only text is allowed in this conditioner")
        hidden, valid = self.encode_text(text)
        context = self.connector(hidden.to(self.connector[0].weight))
        context = context.masked_fill(~valid.unsqueeze(-1), 0)
        if not self.disable_drop:
            keep = torch.rand((context.shape[0], 1, 1), device=context.device) > self.drop_ratio
            context = context * keep
        if not bool(torch.isfinite(context).all()):
            raise RuntimeError("Nonfinite connector output")
        return {"main": context, "main_mask": valid}

    def unconditional_embedding(self, batch_size, **kwargs):
        if kwargs or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Expected a positive unconditional batch size")
        if self.last_attention_mask is None or self.last_attention_mask.shape[0] not in (1, batch_size):
            raise ValueError("Encode a matching text batch first to determine unconditional masks")
        weight = self.connector[0].weight
        return {"main": weight.new_zeros((batch_size, self.last_token_count, 1024)),
                "main_mask": self.last_attention_mask.to(weight.device).expand(batch_size, -1).clone()}
