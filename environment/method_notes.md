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

- All edited-model sampling uses 50-step forward Euler without classifier-free guidance; native image-to-shape baseline uses the official default guidance 5.0. This is a documented difference, not a claim of exact inference parity.
- LoRA weights are optimized in FP32 while frozen base weights and activations use BF16.
- VAE source reconstruction gate uses F-score at 0.05 >= 0.5 in the shared 1.5 source extent frame; 8-sample gate requires a 10% reduction of deterministic-noise training loss. These are diagnostic stop rules, not final success criteria.

- Ninja is 1.13.0 (manylinux2014): original 1.11.1.1 installs and runs but pip 26.2 rejects its old manylinux1 platform tag. No upstream source is patched.

- Headless NVIDIA EGL is selected only within experiment subprocesses; inherited global Mesa software variables are cleared locally. Saved RGBA renders use a white RGB background plus depth-derived alpha for the native pipeline.
- VAE surface sampling now seeds both NumPy and Torch per sample/role. Reverse tasks undergo their own shared-frame QC.
- Synthetic interface diagnostics are separate from reconstruction gates on cleaned public data. Low synthetic reconstruction scores are retained and trigger precision/domain diagnostics, not a claim of trained editing.

- Formal VAE cache and edited-model evaluation decoder use FP32 weights/inputs with CUDA FP16 autocast; pure FP16 synthetic box failures are retained separately. Neutral renders v2 use material0.6, ambient0.15, directional1.3; all-black EGL frames retry or fail.
- Agent visual QC reviewed 46 candidate pairs, excluded16, retained30 reviewed pairs in the final training split. Remaining 482 training pairs have automated geometric QC only.

- Formal connector/LoRA initial and final validation losses use all128 held-out entries. The 20-step trend/checkpoint selection uses a fixed16-entry probe; reports distinguish these counts. The preliminary8-pair fitting diagnostic used only16 validation entries.

- Published manifests are authoritative for reruns. Visual-review application snapshots initial splits and restores untouched validation/test snapshots. Candidate enumeration is sorted for fresh deterministic cleaning; rerunning preparation must not silently resplit already-reviewed assets.

- The untrained-connector evaluation loads connector_initial.pt saved immediately before formal optimization, ensuring the random baseline uses the same initial MLP rather than a new RNG draw after loading VAE.

- Benchmark iteration alternates add/remove and covers every object at the first seed before later seeds. Original indices and cyclic shuffled instructions remain unchanged. This limits operation/order bias if the time cap produces partial coverage.

- Controlled comparisons use deterministic sorted object IDs for bootstrap resampling, report separate add/remove deltas and ground-truth-assisted surface proxies, and include a scoreboard restricted to jointly valid meshes in every mode.
