#!/usr/bin/env bash
set -euo pipefail

# ensure we run from the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔧 Building TensorRT engine with Docker..."
docker run --gpus device=0 --rm \
  -v "/app/unet_opt:/models" \
  nvcr.io/nvidia/tensorrt:24.04-py3 \
  trtexec \
    --onnx=/models/model.onnx \
    --shapes=sample:1x9x64x64,timestep:1,encoder_hidden_state:1x256x768 \
    --builderOptimizationLevel=2 \
    --saveEngine=/models/unet_ne.plan \
    --fp16 --int8

echo "📂 Moving the generated plan file into models/more_repo/unet/1/"
mv "/app/unet_opt/unet_ne.plan" "models/unet/1/unet_ne.plan"

echo "✅ All done."

