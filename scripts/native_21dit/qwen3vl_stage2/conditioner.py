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
        self.connector = nn.Sequential(nn.Linear(4096, 1024), nn.GELU(), nn.Linear(1024, 1024))
        self.task_prefix = task_prefix
        self.max_length = int(max_length)
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        self.drop_ratio = float(drop_ratio)
        self.disable_drop = bool(disable_drop)
        self.last_token_count = None

    @property
    def vlm(self):
        return self.backbone.model

    def tokenize_text(self, text):
        if not isinstance(text, str):
            if not isinstance(text, (list, tuple)) or len(text) != 1:
                raise ValueError("Text-only stage2 requires microbatch=1")
            text = text[0]
        inputs = self.tokenizer.apply_chat_template(
            messages_for(text, self.task_prefix), tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt")
        allowed = {"input_ids", "attention_mask"}
        if set(inputs) - allowed:
            raise ValueError("Unexpected non-text tokenizer inputs")
        if inputs["input_ids"].shape[1] > self.max_length:
            raise ValueError("Caption exceeds max_length; review it instead of silently truncating")
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
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
        hidden = hidden[:, inputs["attention_mask"][0].bool(), :].detach()
        if hidden.shape[1] == 0 or not bool(torch.isfinite(hidden).all()):
            raise RuntimeError("Empty/nonfinite Qwen text hidden state")
        self.last_token_count = hidden.shape[1]
        return hidden

    def forward(self, image=None, text=None, mask=None, **kwargs):
        if image is not None or mask is not None or kwargs:
            raise ValueError("Only text is allowed in this conditioner")
        hidden = self.encode_text(text)
        context = self.connector(hidden.to(self.connector[0].weight))
        if not self.disable_drop:
            keep = torch.rand((context.shape[0], 1, 1), device=context.device) > self.drop_ratio
            context = context * keep
        if not bool(torch.isfinite(context).all()):
            raise RuntimeError("Nonfinite connector output")
        return {"main": context}

    def unconditional_embedding(self, batch_size, **kwargs):
        if kwargs or batch_size != 1 or self.last_token_count is None:
            raise ValueError("Encode one text first to determine unconditional token length")
        weight = self.connector[0].weight
        return {"main": weight.new_zeros((batch_size, self.last_token_count, 1024))}
