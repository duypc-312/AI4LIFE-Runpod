#!/bin/bash
echo "=============================START============================="
tritonserver --model-repository=/app/model_repo --http-port 7000 --grpc-port 8001 --metrics-port 8002 &
TRITON_PID=$!
echo "Started Triton Server with PID ${TRITON_PID}"


echo "Waiting for Triton to be ready..."
until [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:7000/v2/health/ready)" -eq 200 ]; do

    echo "Triton not ready yet, sleeping..."
    sleep 1
done
echo "Triton is ready!"

# sleep 400

python3 -u runpod_handler.py &
RUNPOD_PID=$!
echo "Started Runpod handler with PID ${RUNPOD_PID}"

wait -n

exit $?
echo "=============================END============================="
