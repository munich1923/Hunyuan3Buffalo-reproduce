"""Instantiate only the VAE half needed by a stage; never construct an encoder for sampling."""
from stage2_runtime import path
import torch
from torch import nn
from hy3dshape.models.autoencoders.model import (
    VectsetVAE, FourierEmbedder, PointCrossAttentionEncoder,
    Transformer, CrossAttentionDecoder, smart_load_model)


class PosteriorEncoder(nn.Module):
    def __init__(self,p):
        super().__init__()
        self.fourier_embedder=FourierEmbedder(num_freqs=p.get("num_freqs",8),
                                             include_pi=p.get("include_pi",True))
        self.encoder=PointCrossAttentionEncoder(
            fourier_embedder=self.fourier_embedder,num_latents=p["num_latents"],
            downsample_ratio=p.get("downsample_ratio",20),pc_size=p.get("pc_size",5120),
            pc_sharpedge_size=p.get("pc_sharpedge_size",5120),point_feats=p.get("point_feats",3),
            width=p["width"],heads=p["heads"],layers=p.get("num_encoder_layers",8),
            qkv_bias=p.get("qkv_bias",True),use_ln_post=p.get("use_ln_post",True),
            qk_norm=p.get("qk_norm",False))
        self.pre_kl=nn.Linear(p["width"],p["embed_dim"]*2)
        self.scale_factor=p.get("scale_factor",1.0)


class GeometryDecoder(VectsetVAE):
    def __init__(self,p):
        super().__init__()
        self.fourier_embedder=FourierEmbedder(num_freqs=p.get("num_freqs",8),
                                             include_pi=p.get("include_pi",True))
        self.post_kl=nn.Linear(p["embed_dim"],p["width"])
        self.transformer=Transformer(n_ctx=p["num_latents"],width=p["width"],
            layers=p["num_decoder_layers"],heads=p["heads"],qkv_bias=p.get("qkv_bias",True),
            qk_norm=p.get("qk_norm",False),drop_path_rate=p.get("drop_path_rate",0.))
        down=p.get("geo_decoder_downsample_ratio",1)
        self.geo_decoder=CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder,out_channels=1,
            num_latents=p["num_latents"],mlp_expand_ratio=p.get("geo_decoder_mlp_expand_ratio",4),
            downsample_ratio=down,enable_ln_post=p.get("geo_decoder_ln_post",True),
            width=p["width"]//down,heads=p["heads"]//down,qkv_bias=p.get("qkv_bias",True),
            qk_norm=p.get("qk_norm",False),label_type=p.get("label_type","binary"))
        self.scale_factor=p.get("scale_factor",1.0)
        self.latent_shape=(p["num_latents"],p["embed_dim"])

    def decode(self,latent):
        return self.transformer(self.post_kl(latent))


def load_part(cfg,part,device="cuda"):
    import yaml
    config_file,checkpoint=smart_load_model(str(path(cfg["models"]["shape"])),
        subfolder="hunyuan3d-vae-v2-1",use_safetensors=False,variant="fp16")
    params=yaml.safe_load(path(config_file).read_text())["params"]
    model=(PosteriorEncoder if part=="encoder" else GeometryDecoder)(params)
    # mmap maps the monolithic checkpoint; only selected state tensors are copied into modules.
    state=torch.load(path(checkpoint),map_location="cpu",weights_only=True,mmap=True)
    expected=set(model.state_dict())
    selected={key:value for key,value in state.items() if key in expected}
    model.load_state_dict(selected,strict=True)
    model.requires_grad_(False)
    return model.to(device=device,dtype=torch.float32).eval()
