"""
Fine-tune HuggingFaceTB/SmolLM3-3B on the same text information-extraction
task as finetune_gemma4_ner_extraction.py / finetune_phi4_ner_extraction.py
(NER -> structured JSON), tracked locally with trackio, with the same
custom evaluation metric computed BEFORE and DURING/AFTER fine-tuning.

Task, dataset, prompt, and metric logic are shared with the other two
scripts via ie_common.py -- this file only contains what's specific to
SmolLM3.

SmolLM3-3B is a plain `SmolLM3ForCausalLM` (confirmed via its config.json:
architectures=["SmolLM3ForCausalLM"], model_type="smollm3"; and via its
actual modeling code in transformers, which defines separate
q_proj/k_proj/v_proj/o_proj and gate_proj/up_proj/down_proj -- the same
naming Llama/Mistral/Qwen use). Apache 2.0, 3B params, purely text, no
custom layer wrappers -- so like Phi-4-mini, NONE of the Gemma-4-specific
day-zero workarounds are needed here: no layer patch, no tower exclusion,
no token_type_ids/mm_token_type_ids collator, no trust_remote_code.

One thing that IS specific to this model: SmolLM3 has a dual-mode chat
template with an extended "thinking" (chain-of-thought) mode, toggled via
`enable_thinking=True/False` on apply_chat_template (or a "/think" /
"/no_think" string in the system prompt, which overrides the kwarg). We
pass enable_thinking=False everywhere in ie_common.py and below, so the
model answers directly with JSON instead of spending our tight
max_new_tokens budget on a reasoning trace first. If you want to compare
against thinking-mode quality later, that's the flag to flip -- just budget
for much longer generations if you do.

Requires transformers>=4.53 (when SmolLM3 support landed).

Install:
    pip install -U "transformers>=4.53" trl peft datasets trackio torch \
        accelerate bitsandbytes
"""

import json

import torch
import trackio
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from ie_common import (
    ENTITY_TYPES,
    ExtractionEvalCallback,
    build_prompt_messages,
    evaluate_extraction,
    load_conll2003,
    to_example,
)

# --------------------------------------------------------------------------
# 0. Config
# --------------------------------------------------------------------------
MODEL = "HuggingFaceTB/SmolLM3-3B"
USE_4BIT = True                          # optional here (3B fits comfortably either way)

TRAIN_SAMPLES = 800
EVAL_SAMPLES = 150

num_train_epochs = 6

# With per_device_train_batch_size=2 x gradient_accumulation_steps=4 = an
# effective batch size of 8, one epoch here is TRAIN_SAMPLES/8 = 100 steps.
# This runs the custom eval every N steps IN ADDITION to every epoch, for a
# smoother trackio curve. Lower = smoother curve but more wall-clock time
# spent generating instead of training -- tune to taste.
EVAL_EVERY_N_STEPS = 25


PROJECT = "ner-extraction-conll2003"
RUN_NAME = "smollm3"

trackio.init(
    project=PROJECT,
    name=RUN_NAME,
    config={
        "model": MODEL,
        "dataset": "eriktks/conll2003",
        "task": "named-entity extraction to JSON",
        "train_samples": TRAIN_SAMPLES,
        "eval_samples": EVAL_SAMPLES,
        "eval_every_n_steps": EVAL_EVERY_N_STEPS,
        "entity_types": ENTITY_TYPES,
        "learning_rate": 2e-4,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "quant": "nf4-4bit" if USE_4BIT else "none",
        "enable_thinking": False,
    },
)

# --------------------------------------------------------------------------
# 1. Model + tokenizer (plain text-only causal LM, no special handling needed)
# --------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

quantization_config = None
if USE_4BIT:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=quantization_config,
    device_map="auto",
    attn_implementation="sdpa",
)
model.config.use_cache = False
model.gradient_checkpointing_enable()
if USE_4BIT:
    model = prepare_model_for_kbit_training(model)

# SmolLM3 uses standard, separate attention/MLP projections (confirmed
# against its modeling code) -- same naming as Gemma's text decoder, but no
# exclude_modules needed since there are no vision/audio towers to avoid.
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# --------------------------------------------------------------------------
# 2. Data: CoNLL-2003 BIO tags -> {"PER": [...], "ORG": [...], ...} JSON
#    (loading + conversion logic lives in ie_common.py, shared across all
#    three fine-tuning scripts)
# --------------------------------------------------------------------------
raw = load_conll2003()
train_rows = [to_example(r) for r in raw["train"].select(range(TRAIN_SAMPLES))]
eval_rows = [to_example(r) for r in raw["validation"].select(range(EVAL_SAMPLES))]

train_texts = [
    tokenizer.apply_chat_template(
        build_prompt_messages(ex["sentence"], json.dumps(ex["entities"], ensure_ascii=False)),
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,  # keep training targets = plain JSON, no chain-of-thought preamble
    )
    for ex in train_rows
]
train_dataset = Dataset.from_dict({"text": train_texts})

# --------------------------------------------------------------------------
# 3. Baseline: evaluate the un-tuned instruct model before any fine-tuning
#    (evaluate_extraction() in ie_common.py already passes enable_thinking=False)
# --------------------------------------------------------------------------
print("Evaluating BASE model (before fine-tuning)...")
baseline_metrics, _ = evaluate_extraction(model, tokenizer, eval_rows, label="baseline")
trackio.log(baseline_metrics, step=0)
print(json.dumps(baseline_metrics, indent=2))

# --------------------------------------------------------------------------
# 4. LoRA fine-tuning with trackio-tracked training
#    (no custom collator needed -- SmolLM3ForCausalLM's forward signature is
#    the standard input_ids/attention_mask/labels, so SFTTrainer's default
#    collator is fine as-is)
# --------------------------------------------------------------------------
training_args = SFTConfig(
    output_dir="./smollm3-ner-lora",
    dataset_text_field="text",
    max_length=512,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=num_train_epochs,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="no",
    bf16=torch.cuda.is_available(),
    report_to="trackio",
    run_name=RUN_NAME,
    project=PROJECT,                   # matches trackio.init(project=...) above
    trackio_space_id=None,             # explicit: keep everything local, no HF Space sync
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=lora_config,
    callbacks=[ExtractionEvalCallback(tokenizer, eval_rows, eval_steps=EVAL_EVERY_N_STEPS)],
)

trainer.train()

# --------------------------------------------------------------------------
# 5. Final evaluation + before/after comparison
# --------------------------------------------------------------------------
print("Evaluating FINE-TUNED model (after fine-tuning)...")
final_metrics, _ = evaluate_extraction(model, tokenizer, eval_rows, label="final")

# Same trackio run-lifecycle note as the other two scripts: SFTTrainer's
# TrackioCallback closes the run as soon as trainer.train() returns, so
# reopen it (same project + name, resume="must") before logging again.
trackio.init(project=PROJECT, name=RUN_NAME, resume="must")
trackio.log(final_metrics, step=trainer.state.global_step)

print("\n=== Before vs. after fine-tuning (held-out eval set) ===")
print(f"{'metric':<22}{'before':>10}{'after':>10}{'delta':>10}")
for etype in ENTITY_TYPES + ["micro"]:
    b = baseline_metrics[f"baseline/f1_{etype}"]
    a = final_metrics[f"final/f1_{etype}"]
    print(f"f1_{etype:<19}{b:>10.3f}{a:>10.3f}{a - b:>+10.3f}")
b_valid = baseline_metrics["baseline/json_valid_rate"]
a_valid = final_metrics["final/json_valid_rate"]
print(f"{'json_valid_rate':<22}{b_valid:>10.1%}{a_valid:>10.1%}{a_valid - b_valid:>+10.1%}")

# --------------------------------------------------------------------------
# 6. Save the adapter
# --------------------------------------------------------------------------
print("Saving model...")
trainer.model.save_pretrained("./lora_adapter_smollm3")
tokenizer.save_pretrained("./lora_adapter_smollm3")
print("Model saved to ./lora_adapter_smollm3")

try:
    trackio.finish()
except RuntimeError:
    pass  # run was already finished by the Trainer's TrackioCallback

# To inspect results: run `trackio show` in a terminal in this directory --
# it opens a local Gradio dashboard reading from
# ~/.cache/huggingface/trackio/ (no data ever leaves your machine).
