#!/usr/bin/env python3
#
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# [License text omitted for brevity]
import io

import argparse
import asyncio
import json
import sys
import gc
import time
import nest_asyncio
nest_asyncio.apply()  # Allow nested event loops
import requests
import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import *
import runpod
import os
from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput
import os
import base64
from io import BytesIO
from runpod.serverless.utils.rp_cleanup import clean
# (Your original functions and classes remain unchanged.)
def wait_for_triton(timeout=360):
    """Polls Triton's health endpoint until ready or timeout reached."""
    start_time = time.time()
    url = "http://localhost:7000/v2/health/ready"
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print("Triton is ready!")
                return
        except Exception as e:
            pass
        print("Waiting for Triton to be ready...")
        time.sleep(2)
    raise Exception(f"Triton server not ready after {timeout} seconds.")


# --- User‑configurable settings ---
SERVER_URL           = "localhost:8001"
MODEL_NAME           = "BLS"

def load_and_preprocess(image_ne, ori):
    if ori == 1:
        img = image_ne.convert("RGB")
        # 2) Turn into a (H, W, 3) uint8 NumPy array
        arr = np.array(img, dtype=np.float32)
        return arr
    else:
        img = image_ne.convert("L")
        # 2) Turn into a (H, W, 3) uint8 NumPy array
        arr = np.array(img, dtype=np.float32)
        return arr 


# Create FastAPI app and define an endpoint
app = FastAPI()

@app.post("/inference")
async def inference_endpoint(event: dict):
    # Extract inputs
    tritonclient = grpcclient.InferenceServerClient(url=SERVER_URL, verbose=False)
    
    init_images = event["input"].get("ori_iamge")
    mask_base64 = event["input"].get("mask")
    guidanscale = float(event["input"].get("scale"))
    num_infer_step = int(event["input"].get("step"))
    
    # Validate inputs
    if not init_images or not mask_base64:
        return {"error": "Both original image and mask are required"}
    
    # Decode original image (first in the list)
    original_image = Image.open(BytesIO(base64.b64decode(init_images)))
    
    # Decode mask image
    mask_image = Image.open(BytesIO(base64.b64decode(mask_base64)))

    # 2) Load and preprocess images
    orig = load_and_preprocess(original_image, 1)  # shape (1, H, W, 3)
    mask = load_and_preprocess(mask_image, 0)  # shape (1, H, W, 3)
    # orig = np.expand_dims(orig, 0)   # (1, H, W, 3)
    # mask = np.expand_dims(mask, 0) 

    # 3) Prepare scalar inputs as 1-D arrays ([batch])
    gs_array    = np.array([guidanscale], dtype=np.float32)  # shape (1,)
    steps_array= np.array([num_infer_step], dtype=np.int8)   # shape (1,)
    # gs_array = np.expand_dims(gs_array, 0)   # (1, H, W, 3)
    # steps_array = np.expand_dims(steps_array, 0)   # (1, H, W, 3)

    # 4) Build Triton InferInput objects
    inp_orig = InferInput("ORIGINAL_IMAGE",       orig.shape,     "FP32")
    inp_orig.set_data_from_numpy(orig)

    inp_mask = InferInput("MASK_IMAGE",           mask.shape,     "FP32")
    inp_mask.set_data_from_numpy(mask)

    inp_gs   = InferInput("GUIDANCE_SCALE",       gs_array.shape, "FP32")
    inp_gs.set_data_from_numpy(gs_array)

    inp_steps= InferInput("NUM_INFERENCE_STEPS",  steps_array.shape, "INT8")
    inp_steps.set_data_from_numpy(steps_array)

    # 5) Specify the output you want
    outputs = [InferRequestedOutput("IMAGES")]

    # 6) Perform inference
    response = tritonclient.infer(
        model_name=MODEL_NAME,
        inputs=[inp_orig, inp_mask, inp_gs, inp_steps],
        outputs=outputs
    )

    # 7) Retrieve and print the result
    result = response.as_numpy("IMAGES")
    print("result",result.shape)
    
    image_array = result.astype(np.uint8)
    # 3) Make a PIL Image and save
    img = Image.fromarray(image_array, mode="RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)  # rewind to the start

    # 3) Base64-encode the bytes
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    buffer.close()
    del img, image_array, result, orig, mask
    gc.collect()
    return {"result_image": img_b64}

# Use TestClient to wrap the FastAPI app as a simple handler for Runpod.
client = TestClient(app)

def handler(event):
    # Forward the event to the FastAPI endpoint via TestClient.
    response = client.post("/inference", json=event)
    clean()
    return response.json()

if __name__ == '__main__':
    try:
        runpod.serverless.start({"handler": handler})
    except SystemExit as e:
        if e.code != 0:
            raise
