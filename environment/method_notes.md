# Method choices and deviations

- Geometry only; source perception uses four neutral multiview renders, not Buffalo's point-cloud VecSet/Q-Former.
- Qwen2.5-VL-7B-Instruct last decoder-layer contextual hidden states, no generated answer text.
- Upstream DiT checkpoint condition space is 1024 channels and 1370 tokens; Qwen hidden width is 3584.
- Connector: 3584 -> 1024 -> 1024 with GELU, fixed pooling to 1370 tokens.
- Source and target are VAE-encoded in shared coordinates. Source latent is fixed during denoising and concatenated along token axis. Only noisy target token predictions are used.
- DiT flow convention is noise at t=0 and target at t=1; velocity target is target minus noise.
- Shape checkpoint includes conditioner and VAE state; separate VAE checkpoint is also retained for explicit encoding verification. DINO need not be fetched separately if the packaged conditioner state loads completely.
- Shape-only requirements derive from upstream versions; Paint, Gradio, rembg, bpy and custom rasterizers are excluded because they are outside scope.
- Nano3D v1 release contains add and replace, not direct removal. Use add only, then swap source/target to create removal supervision, with explicit instruction "Remove the following added parts: ...". Store inversion provenance; original and reverse stay in the same split.
- 1024 training entries include both directions of 512 underlying paired assets; validation 128 entries likewise uses 64 disjoint pairs. Report both counts, not 1024 independent geometries.
- Dataset and benchmark currently advertise MIT in their cards; retain cards and provenance.
- All mirrors retain official repository identity and pinned revision. Large payloads must pass expected LFS SHA-256 and size checks when supplied by the API.
- No official Buffalo score comparison until its exact evaluation protocol has been recovered.
