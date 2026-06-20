"""
System Checker for YOLOv8 + UNet Medical Imaging Project
--------------------------------------------------------
This script checks:
- Python version
- Installed packages and versions
- GPU availability (CUDA, PyTorch, TorchVision)
- Pre-requisite libraries for this project

Run:
    python system_check.py
"""

import sys
import importlib
import subprocess
import torch

# List of key packages and recommended versions
REQUIRED_LIBS = {
    "torch": "2.7.1",
    "torchvision": "0.15.2",
    "ultralytics": "8.4.65",
    "pandas": "2.1.2",
    "openpyxl": "3.1.2",
    "numpy": None,
    "matplotlib": None,
    "opencv-python": None,
    "tqdm": None,
}

def check_package(pkg_name, required_version=None):
    try:
        pkg = importlib.import_module(pkg_name)
        version = getattr(pkg, "__version__", "Unknown")
        if required_version:
            status = "OK" if version == required_version else f"Installed ({version}), Recommended ({required_version})"
        else:
            status = f"Installed ({version})"
        return True, version, status
    except ImportError:
        return False, None, "Not installed"

def main():
    print("="*60)
    print("SYSTEM CHECK FOR YOLOv8 + UNet PROJECT")
    print("="*60)
    print(f"Python version: {sys.version}")
    print("-"*60)
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if gpu_available else 0
    cuda_version = torch.version.cuda
    print(f"GPU available: {gpu_available}")
    if gpu_available:
        print(f"Number of GPUs: {n_gpus}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"CUDA version: {cuda_version}")
    print("-"*60)
    
    # Check packages
    print("Checking required packages:")
    for pkg, ver in REQUIRED_LIBS.items():
        installed, version, status = check_package(pkg, ver)
        print(f"{pkg:<20}: {status}")
    
    print("="*60)
    print("SYSTEM CHECK COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()