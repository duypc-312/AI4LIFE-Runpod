from huggingface_hub import hf_hub_download
import zipfile
import os
import sys

# === Config ===
REPO_ID = "PhanDuy/SD1.5_inpaint_triton"     # HF repo ID
FILENAME = "ai4life_triton_1.zip"               # The zip file you uploaded earlier
REPO_TYPE = "model"                        # or "dataset"
UNZIP_DIR = "model_repo"            # Where to unzip the file

# === Step 1: Download zip file from the repo ===
local_zip_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    repo_type=REPO_TYPE
)
print(f"Downloaded to: {local_zip_path}")

# # === Step 2: Unzip ===
os.makedirs(UNZIP_DIR, exist_ok=True)

with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
    zip_ref.extractall(UNZIP_DIR)
    print(f"Unzipped to: {UNZIP_DIR}")

def verify_model_folder(models_root: str, model_name: str):
    """
    Checks that under `models_root` there is a subfolder named `model_name`.
    Exits with code 0 if OK, or prints an error and exits code 1 otherwise.
    """
    model_path = os.path.join(models_root, model_name)
    if os.path.isdir(model_path):
        print(f"✅ Found model folder: {model_path}")
        return True
    else:
        print(f"❌ ERROR: Expected folder not found: {model_path}", file=sys.stderr)
        print(f"Contents of {models_root}: {os.listdir(models_root)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    MODELS_ROOT = "/app/model_repo"  # or wherever you unzip into
    MODEL_NAME  = "BLS"

    verify_model_folder(MODELS_ROOT, MODEL_NAME)
