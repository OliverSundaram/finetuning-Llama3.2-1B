from huggingface_hub import HfApi

api = HfApi()
for name in ["5k", "10k", "20k"]:
    repo_id = f"OliverSundaram/Llama-3.2-1B-MathCodeInstruct-{name}"
    local_folder = f"Llama-3.2-1B-MathCodeInstruct-{name}/outputs/llama-3.2-1b-{name}"
    api.upload_folder(folder_path=local_folder, repo_id=repo_id, repo_type="model")
    print(f"Updated {repo_id}")