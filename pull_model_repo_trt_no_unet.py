from huggingface_hub import hf_hub_download
import zipfile
import os

# === Config ===
REPO_ID = "PhanDuy/SD1.5_inpaint_triton"     # HF repo ID
FILENAME = "ai4life_triton_trt_no_unet.zip"               # The zip file you uploaded earlier
REPO_TYPE = "model"                        # or "dataset"
UNZIP_DIR = "models"            # Where to unzip the file

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
