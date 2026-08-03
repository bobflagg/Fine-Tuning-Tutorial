"""Merge the SmolLM3-3B LoRA adapter into the base model and save the full weights."""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "HuggingFaceTB/SmolLM3-3B"
ADAPTER = "/workspace/project/smollm3-adapter"
OUT = "/workspace/project/smollm3-merged"

print("loading base model...", flush=True)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")

print("applying adapter...", flush=True)
model = PeftModel.from_pretrained(base, ADAPTER)

print("merging...", flush=True)
model = model.merge_and_unload()

print("saving weights ->", OUT, flush=True)
model.save_pretrained(OUT, safe_serialization=True)

# tokenizer from the fine-tune (falls back to the base tokenizer if unusable)
try:
    tok = AutoTokenizer.from_pretrained(ADAPTER)
except Exception as e:  # noqa: BLE001
    print("adapter tokenizer failed (%s), using base tokenizer" % e, flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
tok.save_pretrained(OUT)

print("done", flush=True)
