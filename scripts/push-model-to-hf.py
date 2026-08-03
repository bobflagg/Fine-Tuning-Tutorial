"""Push the merged SmolLM3-3B model to the Hugging Face Hub (private repo)."""

from huggingface_hub import HfApi

REPO = "calcworks/finch-slm-fine-tuning-call-04-gate-smollm3-3b"
FOLDER = "/workspace/project/smollm3-merged"

api = HfApi()
url = api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
print("repo:", url, flush=True)

api.upload_folder(
    repo_id=REPO,
    folder_path=FOLDER,
    repo_type="model",
    commit_message="Add SmolLM3-3B with LoRA adapter merged in",
)
print("upload complete", flush=True)

for f in sorted(api.list_repo_files(REPO)):
    print(" ", f)
