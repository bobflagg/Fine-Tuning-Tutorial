"""
Standalone sanity-check script: loads a BASE instruct model and the
CoNLL-2003 eval split, runs the custom extraction metric shared with the
fine-tuning scripts (ie_common.py), and prints the results plus a handful of
example predictions next to their gold labels.

Works for any causal LM with a chat-template tokenizer -- pass --model to
pick which one. Defaults to the Gemma 4 checkpoint used in
finetune_gemma4_ner_extraction.py; pass --model microsoft/Phi-4-mini-instruct
or --model HuggingFaceTB/SmolLM3-3B or --model Qwen/Qwen3-4B-Instruct-2507
to check the models used in finetune_phi4_ner_extraction.py /
finetune_smollm3_ner_extraction.py / finetune_qwen3_ner_extraction.py
instead. Unlike the fine-tuning scripts, no model-specific patching is
needed here at all -- those workarounds (see finetune_gemma4_ner_extraction.py's
docstring) are only triggered by PEFT's LoRA injection, not by plain
.generate(). (SmolLM3's chain-of-thought "thinking" mode is disabled via
enable_thinking=False inside evaluate_extraction() in ie_common.py, which
applies regardless of which --model you pass.)

Use this BEFORE running the full fine-tuning script to confirm:
  - the model/tokenizer download and load correctly
  - the dataset converts into the expected JSON schema
  - generation + JSON parsing actually work end to end
  - what the *un-tuned* model's extraction quality looks like

No training happens here, and nothing is logged to trackio -- this is just a
quick, cheap smoke test. Run with a small --eval-samples first (default
below), then bump it up once you've confirmed everything works.

Usage:
    python eval_only.py                                          # Gemma 4 E2B-it
    python eval_only.py --model microsoft/Phi-4-mini-instruct     # Phi-4-mini
    python eval_only.py --model HuggingFaceTB/SmolLM3-3B          # SmolLM3
    python eval_only.py --model Qwen/Qwen3-4B-Instruct-2507       # Qwen3-4B
    python eval_only.py --model <any-hf-model-id> --eval-samples 100 --no-4bit

Install:
    pip install -U "transformers>=4.49" trl peft datasets torch accelerate bitsandbytes
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ie_common import evaluate_extraction, load_conll2003, to_example


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default="google/gemma-4-E2B-it",
        help="HF model id to evaluate (default: google/gemma-4-E2B-it). "
        "Try microsoft/Phi-4-mini-instruct, HuggingFaceTB/SmolLM3-3B, or Qwen/Qwen3-4B-Instruct-2507 too.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=20,
        help="Number of CoNLL-2003 validation examples to evaluate on (small by default -- this is a smoke test).",
    )
    parser.add_argument(
        "--num-examples-to-print",
        type=int,
        default=5,
        help="How many individual predictions to print for eyeballing.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (loads in bf16/fp16 instead).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_4bit = not args.no_4bit

    print(f"CUDA available: {torch.cuda.is_available()}")

    print(f"Loading dataset (eriktks/conll2003, {args.eval_samples} validation examples)...")
    raw = load_conll2003()
    eval_rows = [to_example(r) for r in raw["validation"].select(range(args.eval_samples))]
    print(f"Example converted row: {eval_rows[0]}")

    print(f"Loading tokenizer + model ({args.model}, 4-bit={use_4bit})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation="sdpa",
        # Deliberately NOT passing trust_remote_code=True: all three models we
        # target (Gemma 4, Phi-4-mini, SmolLM3) have their architectures built
        # natively into transformers ("architectures" in their config.json
        # resolves to a class transformers already ships). Setting
        # trust_remote_code=True instead makes transformers download and exec
        # the repo's OWN bundled modeling_*.py -- which can be pinned against
        # an older transformers API than what you have installed. That's
        # exactly what broke Phi-4-mini here: its bundled modeling_phi3.py
        # imports `LossKwargs` from transformers.utils, which newer
        # transformers releases removed/renamed, so the remote-code path
        # crashes with an ImportError that the *built-in* Phi3 implementation
        # doesn't have. If you add a model that genuinely needs remote code,
        # pass trust_remote_code=True only for that one call.
    )

    print(f"Running baseline extraction eval on {args.eval_samples} examples...")
    metrics, predictions = evaluate_extraction(model, tokenizer, eval_rows, label="baseline")

    print("\n=== Baseline metrics (untuned model) ===")
    print(json.dumps(metrics, indent=2))

    print(f"\n=== First {args.num_examples_to_print} example predictions vs. gold ===")
    for ex in predictions[: args.num_examples_to_print]:
        print("-" * 70)
        print(f"Sentence: {ex['sentence']}")
        print(f"Gold:     {json.dumps(ex['gold'], ensure_ascii=False)}")
        print(f"Parsed:   {json.dumps(ex['parsed'], ensure_ascii=False) if ex['parsed'] is not None else 'INVALID JSON'}")
        print(f"Raw gen:  {ex['raw']!r}")


if __name__ == "__main__":
    main()
