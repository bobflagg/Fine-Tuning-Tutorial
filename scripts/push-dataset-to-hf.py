from datasets import load_dataset

REPO_ID = "calcworks/finch-slm-fine-tuning-call-04-gate"
DATA_DIR = "ft/call04/gate/data"

ds = load_dataset(
    "json",
    data_files={
        "train": f"{DATA_DIR}/gate_sft_train.jsonl",
        "validation": f"{DATA_DIR}/gate_sft_val.jsonl",
        "test": f"{DATA_DIR}/gate_test_eval.jsonl",
    },
)

print(ds)

ds.push_to_hub(REPO_ID, private=True)
print(f"Pushed to https://huggingface.co/datasets/{REPO_ID}")
