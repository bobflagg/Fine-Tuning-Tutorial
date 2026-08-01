"""
Fine-tune Qwen/Qwen3-4B-Instruct-2507 on the same text information-extraction
task as the other three scripts (NER -> structured JSON), tracked locally
with trackio, with the same custom evaluation metric computed BEFORE and
DURING/AFTER fine-tuning.

Task, dataset, prompt, and metric logic are shared with the other three
scripts via ie_common.py -- this file only contains what's specific to Qwen3.

NOTE on model choice: the newer Qwen3.5-4B was considered first (it's the
current generation, similar size), but turned out to be a considerably
riskier pick on inspection -- its config.json shows it's natively
multimodal (vision_config/image_token_id/video_token_id baked in, class
Qwen3_5ForConditionalGeneration) AND it uses a hybrid attention design
where 3 of every 4 layers are a linear-attention variant (GatedDeltaNet,
with in_proj_qkv/in_proj_z/in_proj_b/in_proj_a instead of the usual
q/k/v/o projections) with regular full attention only every 4th layer.
There's no settled community consensus yet on how to LoRA-target that
correctly, and it needs a very recent transformers (v5+). We went with the
previous-generation Qwen3-4B-Instruct-2507 instead: confirmed via its own
config.json to be a plain `Qwen3ForCausalLM` -- no vision/video config at
all, standard GQA, same q_proj/k_proj/v_proj/o_proj + gate_proj/up_proj/
down_proj naming as Llama/Mistral/SmolLM3. Same ~4B size class (4.0B
total / 3.6B non-embedding), Apache 2.0, and -- per its own model card --
"supports only non-thinking mode", so there's no thinking-mode toggle to
worry about here at all (unlike SmolLM3, where we had to pass
enable_thinking=False explicitly).

Like Phi-4-mini and SmolLM3, NONE of the Gemma-4-specific day-zero
workarounds are needed: no custom layer patch, no tower exclusion, no
token_type_ids/mm_token_type_ids collator, no trust_remote_code. A known
Unsloth bug (unslothai/unsloth#3501) affects 8-bit loading of this model
family under torch.compile -- irrelevant here since we use 4-bit (nf4) via
plain transformers+peft, which is confirmed unaffected.

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
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
USE_4BIT = True                          # optional here (4B fits comfortably either way)

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
RUN_NAME = "qwen3"


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
    # No trust_remote_code: Qwen3ForCausalLM is natively built into
    # transformers -- confirmed via config.json ("architectures":
    # ["Qwen3ForCausalLM"], "model_type": "qwen3"). Setting it would only
    # risk the same version-skew crash we hit with Phi-4-mini's bundled
    # remote code (see finetune_phi4_ner_extraction.py).
)
model.config.use_cache = False
model.gradient_checkpointing_enable()
if USE_4BIT:
    model = prepare_model_for_kbit_training(model)

# Qwen3 uses standard, separate attention/MLP projections -- same naming as
# Llama/Mistral/SmolLM3. No exclude_modules needed: no vision/audio towers.
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
#    four fine-tuning scripts)
# --------------------------------------------------------------------------
raw = load_conll2003()
train_rows = [to_example(r) for r in raw["train"].select(range(TRAIN_SAMPLES))]
eval_rows = [to_example(r) for r in raw["validation"].select(range(EVAL_SAMPLES))]

train_texts = [
    tokenizer.apply_chat_template(
        build_prompt_messages(ex["sentence"], json.dumps(ex["entities"], ensure_ascii=False)),
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,  # no-op here (this checkpoint only supports non-thinking mode
                                 # anyway); kept for consistency with finetune_smollm3_ner_extraction.py
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
#    (no custom collator needed -- Qwen3ForCausalLM's forward signature is
#    the standard input_ids/attention_mask/labels, so SFTTrainer's default
#    collator is fine as-is)
# --------------------------------------------------------------------------
training_args = SFTConfig(
    output_dir="./qwen3-ner-lora",
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
    project=PROJECT,                   # matches trackio.init(project=...) above
    run_name=RUN_NAME,                 # matches trackio.init(name=...) above -- see note below
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

# Same trackio run-lifecycle note as the other scripts: SFTTrainer's
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
trainer.model.save_pretrained("./lora_adapter_qwen3")
tokenizer.save_pretrained("./lora_adapter_qwen3")
print("Model saved to ./lora_adapter_qwen3")

try:
    trackio.finish()
except RuntimeError:
    pass  # run was already finished by the Trainer's TrackioCallback

# To inspect results: run `trackio show` in a terminal in this directory --
# it opens a local Gradio dashboard reading from
# ~/.cache/huggingface/trackio/ (no data ever leaves your machine).
