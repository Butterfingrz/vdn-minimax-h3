# The diffusers component

Each checkpoint on the Hub carries a `diffusers/` directory, so the VDN-H3 transformer loads as a component of diffusers' `ModularPipeline`. This page covers its options and the matching flags of [src/inference/infer_diffusers.py](../src/inference/infer_diffusers.py). The basic usage is in the [README](../README.md#inference-with-diffusers); this repository's own, faster stack is in [inference.md](inference.md).

## What gets loaded

`diffusers/` holds a `config.json` and one file of remote code, `modeling_vdn_h3.py`, and no weights. Loading assembles three things that live elsewhere in the same Hub repository: the 66 GB MiniMax-H3 transformer under `h3-base/`, the checkpoint's linear branch, and its LoRA adapters, which are merged into the base weights. The base loads in bf16.

The pipeline index names the 8-step model. The 50-step model is selected per component:

```python
pipe.load_components(trust_remote_code=True, torch_dtype=torch.bfloat16,
                     subfolder={"transformer": "stage-b-step-2000/diffusers"})
out = pipe(prompt="a prompt", num_frames=345, num_inference_steps=51, ...)
```

`load_components` turns a component's load error into a warning and leaves `pipe.transformer` as `None`. The traceback printed with that warning names the cause.

## Steps and frames

`num_inference_steps` counts sigma grid points, the terminal 0 included, so it is one more than the number of model evaluations: 9 for the 8-step model, 51 for the 50-step one. The script's `--steps` takes model evaluations and adds the one.

`num_frames` runs from 120 to 360, 5 to 15 seconds at 24 fps, and is snapped up to the next 17n+5. 345 frames are the 14.4 seconds every reported number uses.

## Offloading

The README snippet offloads everything but the transformer. The 62 GB Qwen3-VL text encoder streams onto the GPU one layer at a time (`leaf_level`), and each decoder comes in whole through accelerate's hook while it runs. The transformer stays on the GPU, or streams in one block at a time:

```python
apply_group_offloading(pipe.transformer, onload_device="cuda", offload_type="block_level",
                       num_blocks_per_group=1, use_stream=True)
```

The transformer can be offloaded per model or per block, but not per leaf: its fused kernels read a child module's weights without calling the child, so a leaf's hook never fires.

Measured on an H200 at 345 frames and 8 model evaluations, the streamed rows with the CUDA allocator capped at 24 GB:

| Transformer | Seconds per evaluation | Peak GPU memory |
|---|---:|---:|
| bf16, on the GPU | 14.3 | 85 to 103 GB, by prompt length |
| fp8, on the GPU | 12.1 | 65 GB |
| bf16, streamed per block | 16.2 | 22 GB |
| fp8, streamed per block | 13.1 | 20 GB |

The decode phase peaks at 19 GB: the 9.8 GB video VAE plus the 345 decoded frames.

## Window softmax backend

The component's window softmax runs on FlexAttention, which is what fits a 24 GB card. `softmax_backend` selects the decomposition this repository's own scripts use instead. It is faster on B200, and its transformer peak when streamed is 28 to 30 GB.

```python
pipe.load_components(trust_remote_code=True, torch_dtype=torch.bfloat16,
                     softmax_backend={"transformer": "decomposed"})
```

Both backends pick their kernel by the card:

| Card | `flex` | `decomposed` |
|---|---|---|
| Hopper, data-center Blackwell (sm90, sm100, sm110) | FlexAttention's Flash backend (FA4) | FA4's varlen kernel, cuDNN SDPA |
| Ampere, Ada, consumer Blackwell (sm8x, sm120) | FlexAttention's Triton kernel | PyTorch's `varlen_attn`, SDPA's own kernels |

## Quantization

Quantization is torchao's (`pip install torchao`) and runs last, after the LoRA merge, because a quantized weight cannot take the merge. `quantization_config` therefore accepts a `TorchAoConfig` only. Any other backend raises within a second, before the 66 GB load, and so does `fp8=True` together with a config.

### fp8

```python
pipe.load_components(trust_remote_code=True, torch_dtype=torch.bfloat16,
                     fp8={"transformer": True})
```

`fp8`, or `--fp8` for the script, is a preset `TorchAoConfig`: every Linear at least 4096 wide on both sides goes to fp8 e4m3 with dynamic activation scales. That is 363 Linears: the qkv and output projections, the feed-forward pair and the linear branch's readout. Narrow Linears stay in bf16. Scales are per row on sm90 and per tensor on sm100 and above, which is where each card's fast GEMM is. The transformer's weights drop from 62 GB to 45.

Each quantized Linear calls one shared `torch.compile`d `F.linear`, so the first evaluation compiles. The quantized weights are ordinary parameters, and diffusers' group offloading streams them as they are.

fp8 needs compute capability 9.0 or above. It changes the sample: the same seed gives a different video than bf16, of the same quality.

### A config of your own

```python
from diffusers import TorchAoConfig
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow

pipe.load_components(
    trust_remote_code=True, torch_dtype=torch.bfloat16,
    quantization_config={"transformer": TorchAoConfig(
        Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))},
)
```

A config without a `modules_to_not_convert` of its own leaves the same narrow Linears in bf16. A list of your own is applied as given, plus the modules the model keeps in fp32. A Float8 dynamic-activation config that sets no `activation_value_lb` gets the preset's floor on the activation amax. The linear branch skips the anchor frames, so rows of exact zeros reach its readout, and an unfloored per-row scale quantizes them to NaN.

### int8

`Int8DynamicActivationInt8WeightConfig` runs from Ampere up. On an H200 it renders at bf16 speed with fp8's memory: 14.7 to 15.0 seconds per evaluation and 67 to 77 GB with the transformer on the GPU. Its first evaluation compiles for 100 to 180 seconds.

`Int8WeightOnlyConfig` loads and renders, an order of magnitude slower: torchao's weight-only int8 has no fast path under the compiled `F.linear`.

## The script

`infer_diffusers.py` runs the README snippet and shares no code with the repository's own stack.

| Flag | Meaning |
|---|---|
| `prompt` | the text; defaults to the text `prompts/example_0.pt` was encoded from |
| `--out` | the mp4, default `results/diffusers.mp4` |
| `--steps` | model evaluations, default 8 |
| `--frames` | default 345 |
| `--transformer` | a checkpoint's `diffusers/` subfolder, e.g. `stage-b-step-2000/diffusers` with `--steps 50` |
| `--first`, `--last` | keyframes: `--first` alone is I2VA, `--last` alone is L2VA, both is FL2VA |
| `--fp8` | the fp8 preset |
| `--offload_dit` | stream the transformer per block, for a 24 GB card |
| `--seed`, `--device` | default 0 and `cuda` |
