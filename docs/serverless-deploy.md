# Serving `finch-slm-fine-tuning-call-04-gate-smollm3-3b` with vLLM on Runpod Serverless

Deployment guide for the merged SmolLM3-3B fine-tune hosted at
`calcworks/finch-slm-fine-tuning-call-04-gate-smollm3-3b`.

> **Prerequisite:** the model repo must exist on the Hub before you deploy. See
> [Appendix A](#appendix-a-pushing-the-model) if the push hasn't been completed yet.

---

## What you're deploying

These are **full merged weights**, not a LoRA adapter — vLLM loads them like any other
model, with no `--enable-lora` flag and no adapter path.

| Property | Value |
| --- | --- |
| Architecture | `SmolLM3ForCausalLM` |
| Parameters | ~3B (36 layers, hidden 2048, intermediate 11008) |
| Attention | GQA — 16 query heads, 4 KV heads, head_dim 128 |
| Weights dtype | `bfloat16` (~6.15 GB on disk, single shard) |
| Native context | 65,536 tokens (`max_position_embeddings`) |
| Vocab | 128,256 (tied embeddings) |
| EOS / pad token | `<\|im_end\|>` (id 128012) / id 128004 |
| Chat template | Bundled as `chat_template.jinja` (ChatML-style, supports think / no-think) |
| Repo visibility | **Private** — requires an HF token at load time |

---

## Requirements

* A [Runpod account](https://docs.runpod.io/accounts-billing/manage-accounts) and
  [API key](https://docs.runpod.io/get-started/api-keys).
* A **Hugging Face token with read access** to `calcworks/*`. The repo is private, so the
  worker cannot download the weights without one.
  Create it at <https://huggingface.co/settings/tokens> → fine-grained → *Read access to
  contents of all repos under your personal namespace*.

---

## GPU sizing

Weights are ~6.2 GB. KV cache for this model costs roughly **72 KiB per token**
(2 × 36 layers × 4 KV heads × 128 head_dim × 2 bytes), i.e. ~1.2 GB per 16K-token sequence.

| GPU | VRAM | Verdict |
| --- | --- | --- |
| A4000 / RTX 4000 Ada | 16 GB | Works. Good for `MAX_MODEL_LEN=16384` and modest concurrency. |
| **L4 / A5000 / RTX 4090** | **24 GB** | **Recommended.** Comfortable headroom for 32K context + batching. |
| L40S / A100 | 48–80 GB | Only if you need the full 65K context at high concurrency. |

Don't pay for multi-GPU — a 3B model has no need for tensor parallelism.

---

## Step 1: Deploy the worker

1. Open the [vLLM worker](https://console.runpod.io/hub/runpod-workers/worker-vllm) in the
   Runpod Hub and click **Deploy**. **Use the latest worker version** — SmolLM3 support
   requires a recent vLLM (see [Troubleshooting](#troubleshooting)).
2. In the **Model** field, enter:
   ```
   calcworks/finch-slm-fine-tuning-call-04-gate-smollm3-3b
   ```
3. Enter your **Hugging Face token** in the token field. This is mandatory here — the repo
   is private and the download will 401 without it.
4. Click **Advanced** to expand the vLLM settings and set:
   * **Max Model Length**: `16384` (a safe start; raise toward 65536 only with a bigger GPU)
   * **Dtype**: `bfloat16`
5. Pick a **24 GB GPU** per the table above, then click **Next** → **Create Endpoint**.

Initialization takes several minutes while Runpod pulls the ~6.2 GB of weights.

## Step 2: Set environment variables

Go to **Manage → Edit Endpoint → Public Environment Variables** and confirm these. Set
`HF_TOKEN` as a **secret**, not a public variable.

| Variable | Value | Why |
| --- | --- | --- |
| `HF_TOKEN` | `hf_...` | **Required** — private repo. Store as a Runpod secret. |
| `DTYPE` | `bfloat16` | **Set explicitly.** This config was written by transformers v5, which renamed `torch_dtype` → `dtype`; some vLLM builds won't find the old key and will fall back to `float32`, doubling VRAM. |
| `MAX_MODEL_LEN` | `16384` | Caps KV cache. Raise only if VRAM allows. |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Leave ~10% headroom; bump to `0.95` if you need more KV cache. |
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | `finch-gate-smollm3` *(optional)* | Shorter `model` string in OpenAI-style requests. |

## Step 3: Note your endpoint ID

Copy the **Endpoint ID** from the endpoint detail page — every request URL needs it.

---

## Step 4: Send a test request

### Runpod native API

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
     -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "input": {
         "prompt": "Hello World",
         "sampling_params": {"max_tokens": 256, "temperature": 0.6, "top_p": 0.95}
       }
     }'
```

The temperature/top-p above match the values baked into the model's `generation_config.json`.

### OpenAI-compatible API (recommended)

This route applies the bundled chat template for you, so you don't have to hand-format
`<|im_start|>` turns.

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1/chat/completions" \
     -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "calcworks/finch-slm-fine-tuning-call-04-gate-smollm3-3b",
       "messages": [{"role": "user", "content": "Hello World"}],
       "max_tokens": 256,
       "temperature": 0.6,
       "top_p": 0.95
     }'
```

### Python (OpenAI client)

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_RUNPOD_API_KEY",
    base_url="https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1",
)

resp = client.chat.completions.create(
    model="calcworks/finch-slm-fine-tuning-call-04-gate-smollm3-3b",
    messages=[{"role": "user", "content": "Hello World"}],
    max_tokens=256,
    temperature=0.6,
    top_p=0.95,
)
print(resp.choices[0].message.content)
```

---

## Reasoning ("thinking") mode

SmolLM3 is a hybrid reasoning model, and this fine-tune keeps that behavior — by default
it emits a `<think>...</think>` block before its answer. Verified locally on the merged
weights:

```
<think>
Okay, so I need to explain what a gate is in a neural network...
</think>
```

Your client must handle this. Options:

* **Strip it** — parse out everything up to and including `</think>`.
* **Disable it** — put `/no_think` in the system message, which the bundled template
  understands:
  ```json
  "messages": [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": "Hello World"}
  ]
  ```
* **Force it on** — use `/think` in the system message.

Budget `max_tokens` accordingly: with thinking enabled, reasoning tokens are counted, so a
256-token cap can be consumed entirely by the `<think>` block, truncating the actual answer.

---

## Troubleshooting

**Worker fails at config load / unknown key `rope_parameters` or `layer_types`**
This config was serialized by transformers 5.14.1, which uses newer key names than
transformers 4.x. Fix by using the newest vLLM worker image. If you're pinned to an older
one, re-save the config with the legacy keys (`torch_dtype`, top-level `rope_theta`) and
push that.

**`401 Unauthorized` / "Repository not found" during download**
The `HF_TOKEN` is missing, expired, or lacks read scope on `calcworks/*`. Note that a
fine-grained token with *no* permissions checked will still download public models but
fails on this private repo — verify with:
```bash
hf auth whoami
```

**Unknown architecture `SmolLM3ForCausalLM`**
The worker's vLLM is too old. Redeploy on the latest version.

**Out of memory at startup**
Lower `MAX_MODEL_LEN` (try `8192`), then `GPU_MEMORY_UTILIZATION` (try `0.85`), or move to
a 24 GB GPU. Also confirm `DTYPE=bfloat16` — a silent `float32` fallback doubles the
weight footprint to ~12 GB.

**Responses never stop / trailing special tokens**
Generation should stop on `<|im_end|>` (id 128012). If it doesn't, add
`"stop": ["<|im_end|>"]` to your request.

**Cold starts are slow**
Every scale-from-zero re-downloads ~6.2 GB. Set **Active Workers** to 1 to keep one warm,
or enable FlashBoot on the endpoint.

---

## Appendix A: Pushing the model

If `calcworks/finch-slm-fine-tuning-call-04-gate-smollm3-3b` doesn't exist yet, the merged
weights are in `./smollm3-merged` and the upload script is `./push_to_hf.py`. It needs an
`HF_TOKEN` with **write** access to the `calcworks` namespace:

```bash
export HF_TOKEN=hf_...   # write-scoped
python push_to_hf.py
```

To regenerate the merged weights from the LoRA adapter, run `./merge_lora.py`.
