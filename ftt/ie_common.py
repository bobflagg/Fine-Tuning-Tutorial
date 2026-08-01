"""
Shared, model-agnostic pieces for the NER -> JSON information-extraction
fine-tuning tutorials (Gemma 4, Phi-4-mini-instruct, and anything else with
a chat-template tokenizer + .generate()):

  - CoNLL-2003 loading + BIO -> JSON conversion
  - the chat-template prompt builder
  - the custom evaluation metric (per-type + micro P/R/F1, JSON-validity rate)
  - a TrainerCallback that logs that metric to trackio every epoch

Nothing in this file is specific to any one model architecture. Import from
here in the per-model driver scripts (finetune_gemma4_ner_extraction.py,
finetune_phi4_ner_extraction.py, eval_only.py) instead of duplicating it --
that way the task/metric logic only needs fixing in one place.
"""

import json
import re
from collections import Counter, defaultdict

import torch
import trackio
from datasets import load_dataset
from transformers import TrainerCallback

ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]

# Standard CoNLL-2003 BIO tag scheme. The Hub's auto-converted Parquet export
# (see load_conll2003() below) doesn't carry the original ClassLabel
# metadata, so this is hardcoded rather than read from dataset.features.
CONLL_ID2LABEL = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

SYSTEM_PROMPT = (
    "You are an information extraction system. Given a sentence, extract all "
    "named entities and return ONLY a JSON object with exactly these keys: "
    '"PER", "ORG", "LOC", "MISC". Each key maps to a list of the exact entity '
    "strings found in the sentence, in the order they appear. If a type has "
    "no entities, use an empty list. Do not include any text outside the "
    "JSON object."
)


def load_conll2003():
    """Load eriktks/conll2003 straight from its Hub-hosted Parquet export,
    bypassing the repo's .py loading script. `datasets>=4.0` removed script
    execution entirely (RuntimeError: "Dataset scripts are no longer
    supported"), so a plain `load_dataset("eriktks/conll2003")` fails on
    current installs -- see
    https://huggingface.co/datasets/eriktks/conll2003/discussions/13.

    The repo has a Hub-generated `refs/convert/parquet` branch with
    `conll2003/{train,validation,test}/0000.parquet` (verified directly by
    browsing that branch); we fetch those files by URL, which works
    regardless of `datasets` version and doesn't touch the deprecated
    script at all.

    Fallback if this ever breaks (dataset re-hosted, files reshaped):
        pip install "datasets<4.0"
        load_dataset("eriktks/conll2003", trust_remote_code=True)
    """
    base = (
        "https://huggingface.co/datasets/eriktks/conll2003/resolve/"
        "refs%2Fconvert%2Fparquet/conll2003"
    )
    data_files = {
        "train": f"{base}/train/0000.parquet",
        "validation": f"{base}/validation/0000.parquet",
        "test": f"{base}/test/0000.parquet",
    }
    return load_dataset("parquet", data_files=data_files)


def bio_to_entities(tokens, tag_ids, id2label=CONLL_ID2LABEL):
    """Decode a BIO tag sequence into {type: [entity strings]}."""
    entities = defaultdict(list)
    current_type, current_tokens = None, []
    for token, tag_id in zip(tokens, tag_ids):
        tag = id2label[tag_id]
        if tag == "O":
            if current_type:
                entities[current_type].append(" ".join(current_tokens))
                current_type, current_tokens = None, []
            continue
        prefix, etype = tag.split("-")
        if prefix == "B" or etype != current_type:
            if current_type:
                entities[current_type].append(" ".join(current_tokens))
            current_type, current_tokens = etype, [token]
        else:
            current_tokens.append(token)
    if current_type:
        entities[current_type].append(" ".join(current_tokens))
    # always include all four keys, even if empty, so the model learns a fixed schema
    return {etype: entities.get(etype, []) for etype in ENTITY_TYPES}


def to_example(row, id2label=CONLL_ID2LABEL):
    sentence = " ".join(row["tokens"])
    entities = bio_to_entities(row["tokens"], row["ner_tags"], id2label)
    return {"sentence": sentence, "entities": entities}


def build_prompt_messages(sentence, answer=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sentence},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def normalize_set(items):
    if not isinstance(items, list):
        return set()
    return {s.strip().lower() for s in items if isinstance(s, str) and s.strip()}


@torch.no_grad()
def evaluate_extraction(model, tokenizer, examples, label="eval", max_new_tokens=150, batch_size=8):
    """Generate on every example, parse JSON, and score entity-level P/R/F1.

    Returns (metrics, predictions): `metrics` is a flat dict of
    precision/recall/F1 per entity type plus micro-averages and the
    JSON-validity rate; `predictions` is a list of per-example dicts
    (sentence/gold/raw generation/parsed JSON) useful for eyeballing.
    """
    was_training = model.training
    model.eval()
    tp, fp, fn = Counter(), Counter(), Counter()
    valid_json = 0
    predictions = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_prompt_messages(ex["sentence"]),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # SmolLM3-specific (disables its chain-of-thought mode so
                                         # it doesn't burn max_new_tokens on reasoning instead of
                                         # JSON); silently ignored by tokenizers that don't use it
            )
            for ex in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_texts = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        for ex, gen in zip(batch, gen_texts):
            pred = extract_json(gen)
            predictions.append({"sentence": ex["sentence"], "gold": ex["entities"], "raw": gen, "parsed": pred})
            if isinstance(pred, dict):
                valid_json += 1
            else:
                pred = {}
            for etype in ENTITY_TYPES:
                pred_set = normalize_set(pred.get(etype, []))
                gold_set = normalize_set(ex["entities"].get(etype, []))
                tp[etype] += len(pred_set & gold_set)
                fp[etype] += len(pred_set - gold_set)
                fn[etype] += len(gold_set - pred_set)

    metrics = {}
    total_tp = total_fp = total_fn = 0
    for etype in ENTITY_TYPES:
        p = tp[etype] / (tp[etype] + fp[etype]) if (tp[etype] + fp[etype]) else 0.0
        r = tp[etype] / (tp[etype] + fn[etype]) if (tp[etype] + fn[etype]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        metrics[f"{label}/precision_{etype}"] = p
        metrics[f"{label}/recall_{etype}"] = r
        metrics[f"{label}/f1_{etype}"] = f1
        total_tp += tp[etype]
        total_fp += fp[etype]
        total_fn += fn[etype]

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    metrics[f"{label}/precision_micro"] = micro_p
    metrics[f"{label}/recall_micro"] = micro_r
    metrics[f"{label}/f1_micro"] = micro_f1
    metrics[f"{label}/json_valid_rate"] = valid_json / len(examples) if examples else 0.0

    if was_training:
        model.train()
    return metrics, predictions


class ExtractionEvalCallback(TrainerCallback):
    """Runs the custom metric on the held-out eval set and logs it to
    trackio, so the dashboard shows the F1 curve climbing over the course of
    training (not just a single before/after snapshot).

    By default this only fires at the end of every epoch (on_epoch_end),
    which -- with small datasets -- can be much coarser than it looks: e.g.
    TRAIN_SAMPLES=800 with an effective batch size of 8 is exactly 100 steps
    per epoch, so "once per epoch" and "once every 100 steps" are the same
    thing. Pass `eval_steps=N` to ALSO run the eval every N training steps,
    independent of epoch boundaries, for a smoother curve. This re-runs
    generation over the whole eval set each time, which is real wall-clock
    cost -- don't set N too low relative to your eval set size/max_new_tokens
    unless you're prepared for evaluation to dominate training time.
    """

    def __init__(self, tokenizer, eval_examples, eval_steps=None):
        self.tokenizer = tokenizer
        self.eval_examples = eval_examples
        self.eval_steps = eval_steps

    def _run_eval(self, model, global_step, epoch=None):
        metrics, _ = evaluate_extraction(model, self.tokenizer, self.eval_examples, label="eval")
        trackio.log(metrics, step=global_step)
        where = f"epoch {epoch:.2f}" if epoch is not None else f"step {global_step}"
        print(
            f"[{where}] micro F1={metrics['eval/f1_micro']:.3f} "
            f"json_valid={metrics['eval/json_valid_rate']:.1%}"
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self.eval_steps and state.global_step > 0 and state.global_step % self.eval_steps == 0:
            self._run_eval(kwargs["model"], state.global_step)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        # Skip if on_step_end already logged this exact step, so we don't
        # double-log (and double-generate) the same data point.
        if not (self.eval_steps and state.global_step % self.eval_steps == 0):
            self._run_eval(kwargs["model"], state.global_step, epoch=state.epoch)
        return control
