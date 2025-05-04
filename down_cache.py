from huggingface_hub import snapshot_download

# Download and cache the repo
cache_path = snapshot_download("stable-diffusion-v1-5/stable-diffusion-inpainting")

# print(f"Repository is cached at: {cache_path}")
