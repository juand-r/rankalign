#!/usr/bin/env python3
"""Create a TAUR RunPod H100-SXM pod for s13 epoch0 eval.

Uses the verified recipe in notes/runpod-pod-creation-recipe.md.
torch>=2.5 base image is mandatory for gemma-4 (see docs/gemma4-31b-pod-setup.md).

Usage: _create_s13_eval_pod.py <pod-name>
"""
import os
import sys
import requests
from key_handler.key_handler import KeyHandler

KeyHandler.set_env_key()
API_KEY = os.environ["RUNPOD_API_KEY_TAUR"]

IMAGE = "runpod/pytorch:0.7.0-cu1241-torch251-ubuntu2204"  # torch 2.5.1+cu124
GPU_TYPE = "NVIDIA H100 80GB HBM3"  # SXM — NVL has a safetensors CUDA-map bug for PEFT eval

MUTATION = """
mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) { id name desiredStatus }
}
"""


def create_pod(name: str) -> str:
    variables = {
        "input": {
            "name": name,
            "imageName": IMAGE,
            "gpuTypeId": GPU_TYPE,
            "gpuCount": 1,
            "containerDiskInGb": 50,
            "volumeInGb": 100,
            "volumeMountPath": "/workspace",
            "ports": "22/tcp",
            "startSsh": True,
            "startJupyter": False,
            "cloudType": "SECURE",
        }
    }
    resp = requests.post(
        "https://api.runpod.io/graphql",
        json={"query": MUTATION, "variables": variables},
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        timeout=30,
    )
    data = resp.json()
    pod = data.get("data", {}).get("podFindAndDeployOnDemand")
    if not pod:
        raise RuntimeError(f"Pod creation failed: {data}")
    print(f"Created: id={pod['id']} name={pod['name']} status={pod['desiredStatus']}")
    return pod["id"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: _create_s13_eval_pod.py <pod-name>")
    create_pod(sys.argv[1])
