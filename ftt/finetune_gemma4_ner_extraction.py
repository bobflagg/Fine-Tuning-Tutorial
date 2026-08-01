"""
Fine-tune Gemma 4 (E2B-it) on a text information-extraction task (Named Entity
Recognition -> structured JSON), tracked locally with trackio, with a custom
evaluation metric (per-type + micro precision/recall/F1 and JSON-validity
rate) computed BEFORE and DURING/AFTER fine-tuning so you can see the metric
actually move.

Task, dataset, prompt, and metric logic are shared with
finetune_phi4_ner_extraction.py via ie_common.py -- this file only contains
what's specific to Gemma 4.

Dataset: eriktks/conll2003. This is text-only -- no images/audio are ever
passed as input anywhere in this script. Note, though, that the checkpoint
itself is a single unified multimodal model: loading it via
AutoModelForCausalLM picks the causal-LM forward/generate path, but the
vision_tower/audio_tower/multi_modal_projector submodules are still present
in memory (they are NOT stripped out). That matters for LoRA targeting
below -- see the "Gemma 4 day-zero compatibility" section.

Known Gemma 4 day-zero issues this script works around (as of mid-2026,
freshly reported against peft/transformers -- see
https://github.com/huggingface/peft/issues/3129 and
https://huggingface.co/google/gemma-4-31B/discussions/3):
  1. PEFT's LoRA doesn't recognize Gemma 4's custom `Gemma4ClippableLinear`
     layer wrapper as a LoRA-able module -> we monkey-patch it to subclass
     nn.Linear before the model is loaded.
  2. Because target_modules matches by name across the WHOLE model, plain
     ["q_proj","k_proj","v_proj","o_proj"] also matches the vision/audio
     towers' attention projections -> we explicitly exclude those towers.
  3. Gemma 4's forward pass expects `token_type_ids` and `mm_token_type_ids`
     even for text-only batches -> we use a custom data collator that fills
     both with zeros.
None of this is needed for microsoft/Phi-4-mini-instruct (plain Phi3ForCausalLM,
no vision/audio, standard collator) -- see finetune_phi4_ner_extraction.py.

Tracking: trackio, kept fully local (no Hugging Face Space sync -- we never
pass a `space_id` / `trackio_space_id`). After running, view the dashboard
with:

    trackio show

Install:
    pip install -U "transformers>=4.57" trl peft datasets trackio torch \
        accelerate bitsandbytes
"""

import json
from dataclasses import dataclass

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
MODEL = "google/gemma-4-E2B-it"          # smallest Gemma 4 checkpoint, text+image+audio,
                                          # but we only ever feed it text below
USE_4BIT = True                          # QLoRA -- turn off if you have a big GPU

TRAIN_SAMPLES = 800
EVAL_SAMPLES = 150

num_train_epochs = 6

# With per_device_train_batch_size=2 x gradient_accumulation_steps=4 = an
# effective batch size of 8, one epoch here is TRAIN_SAMPLES/8 = 100 steps --
# so "once per epoch" and "once every 100 steps" were the same thing. This
# runs the custom eval (full generation pass over eval_rows) every N steps
# IN ADDITION to every epoch, for a smoother trackio curve. Lower = smoother
# curve but more wall-clock time spent generating instead of training --
# tune to taste.
EVAL_EVERY_N_STEPS = 20

PROJECT = "ner-extraction-conll2003"
RUN_NAME = "gemma4"

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
# Gemma 4 day-zero compatibility patch #1: PEFT doesn't yet know how to
# attach LoRA to Gemma 4's custom `Gemma4ClippableLinear` layer (it wraps
# nn.Linear/Linear4bit for optional input/output clamping rather than being
# one), so `get_peft_model` raises:
#   ValueError: Target module Gemma4ClippableLinear(...) is not supported.
# Monkey-patch it to be a real nn.Linear subclass. MUST run before
# AutoModelForCausalLM.from_pretrained() below.
# See https://github.com/huggingface/peft/issues/3129
# --------------------------------------------------------------------------
try:
    from transformers.models.gemma4 import modeling_gemma4

    class _PatchedClippableLinear(torch.nn.Linear):
        def __init__(self, config, in_features, out_features):
            torch.nn.Linear.__init__(self, in_features, out_features, bias=False)
            self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
            if self.use_clipped_linears:
                self.register_buffer("input_min", torch.tensor(-float("inf")))
                self.register_buffer("input_max", torch.tensor(float("inf")))
                self.register_buffer("output_min", torch.tensor(-float("inf")))
                self.register_buffer("output_max", torch.tensor(float("inf")))

        def forward(self, x):
            if self.use_clipped_linears:
                x = torch.clamp(x, self.input_min, self.input_max)
            out = torch.nn.Linear.forward(self, x)
            if self.use_clipped_linears:
                out = torch.clamp(out, self.output_min, self.output_max)
            return out

    modeling_gemma4.Gemma4ClippableLinear = _PatchedClippableLinear
    print("Applied Gemma4ClippableLinear -> nn.Linear patch (peft compatibility).")
except ImportError as e:
    print(
        f"WARNING: could not apply the Gemma4ClippableLinear patch ({e}). "
        "If LoRA setup fails with 'Target module ... is not supported', see "
        "https://github.com/huggingface/peft/issues/3129 -- your transformers "
        "version may lay out the gemma4 modeling module differently, or the "
        "issue may already be fixed upstream (in which case this is harmless)."
    )

# --------------------------------------------------------------------------
# 1. Model + tokenizer (text-only: AutoModelForCausalLM, no processor)
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

# Configure LoRA (same attention projections as the Gemma 4 GQA architecture).
# target_modules matches by NAME across the whole model, and the vision and
# audio towers reuse the same q_proj/k_proj/v_proj/o_proj naming convention
# for their own internal attention -- exclude_modules keeps LoRA on the text
# decoder only (day-zero compatibility fix #2, see module docstring).
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    exclude_modules=["vision_tower", "audio_tower", "multi_modal_projector"],
    task_type="CAUSAL_LM",
)

# --------------------------------------------------------------------------
# 2. Data: CoNLL-2003 BIO tags -> {"PER": [...], "ORG": [...], ...} JSON
#    (loading + conversion logic lives in ie_common.py, shared with Phi-4)
# --------------------------------------------------------------------------
raw = load_conll2003()
train_rows = [to_example(r) for r in raw["train"].select(range(TRAIN_SAMPLES))]
eval_rows = [to_example(r) for r in raw["validation"].select(range(EVAL_SAMPLES))]

train_texts = [
    tokenizer.apply_chat_template(
        build_prompt_messages(ex["sentence"], json.dumps(ex["entities"], ensure_ascii=False)),
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,  # no-op here (SmolLM3-specific flag); kept for consistency with
                                 # finetune_smollm3_ner_extraction.py, silently ignored otherwise
    )
    for ex in train_rows
]
train_dataset = Dataset.from_dict({"text": train_texts})

# --------------------------------------------------------------------------
# 3. Baseline: evaluate the un-tuned instruct model before any fine-tuning
# --------------------------------------------------------------------------
print("Evaluating BASE model (before fine-tuning)...")
baseline_metrics, _ = evaluate_extraction(model, tokenizer, eval_rows, label="baseline")
trackio.log(baseline_metrics, step=0)
print(json.dumps(baseline_metrics, indent=2))

# --------------------------------------------------------------------------
# Gemma 4 day-zero compatibility patch #3: the forward pass expects
# `token_type_ids` and `mm_token_type_ids` even for text-only batches. The
# default SFT collator doesn't produce them, so training crashes with a
# missing-argument error. Fill both with zeros (i.e. "no multimodal
# segments") for every position.
# --------------------------------------------------------------------------
@dataclass
class TextOnlyGemma4Collator:
    tokenizer: object

    def __call__(self, features):
        input_ids_list = [f["input_ids"] for f in features]
        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, token_type_ids, mm_token_type_ids, labels = [], [], [], [], []
        for f in features:
            ids = list(f["input_ids"])
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            token_type_ids.append([0] * max_len)
            mm_token_type_ids.append([0] * max_len)
            lbls = list(f.get("labels", ids)) + [-100] * pad_len
            labels.append(lbls)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "mm_token_type_ids": torch.tensor(mm_token_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# --------------------------------------------------------------------------
# 4. LoRA fine-tuning with trackio-tracked training
# --------------------------------------------------------------------------
training_args = SFTConfig(
    output_dir="./gemma4-ner-lora",
    dataset_text_field="text",
    max_length=512,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=num_train_epochs,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="no",
    bf16=torch.cuda.is_available(),
    remove_unused_columns=False,       # needed so our custom collator sees raw columns (day-zero fix #3)
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
    data_collator=TextOnlyGemma4Collator(tokenizer),
    callbacks=[ExtractionEvalCallback(tokenizer, eval_rows, eval_steps=EVAL_EVERY_N_STEPS)],
)

trainer.train()

# --------------------------------------------------------------------------
# 5. Final evaluation + before/after comparison
# --------------------------------------------------------------------------
print("Evaluating FINE-TUNED model (after fine-tuning)...")
final_metrics, _ = evaluate_extraction(model, tokenizer, eval_rows, label="final")

# SFTTrainer's built-in TrackioCallback calls trackio.finish() as soon as
# trainer.train() returns (on_train_end), which clears the global run
# context -- a plain trackio.log() here would raise "Call trackio.init()
# before trackio.log()". Reopen the SAME run (same project + name) with
# resume="must" so this final data point lands on the same chart as the
# per-epoch curve from ExtractionEvalCallback, instead of logging blind.
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
trainer.model.save_pretrained("./lora_adapter_gemma4")
tokenizer.save_pretrained("./lora_adapter_gemma4")
print("Model saved to ./lora_adapter_gemma4")

try:
    trackio.finish()
except RuntimeError:
    pass  # run was already finished by the Trainer's TrackioCallback

# To inspect results: run `trackio show` in a terminal in this directory --
# it opens a local Gradio dashboard reading from
# ~/.cache/huggingface/trackio/ (no data ever leaves your machine).
