#!/usr/bin/env python3
"""
CUDA/cuDNN initialization fix.
This module MUST be imported before any CUDA libraries to properly set up cuDNN.
"""

import os
import logging

logger = logging.getLogger(__name__)

def setup_cuda_environment():
    """Setup CUDA and cuDNN environment variables for Windows."""

    if os.name != 'nt':  # Only needed on Windows
        return True

    cuda_version = "13.0"
    cudnn_path = fr"C:\Program Files\NVIDIA\CUDNN\v9.13"
    cuda_toolkit_path = fr"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{cuda_version}"

    # Set environment variables
    env_vars = {
        "CUDA_PATH": cuda_toolkit_path,
        "CUDNN_PATH": cudnn_path,
        "TORCH_CUDNN_V8_API_ENABLED": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }

    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.debug(f"Set {key}={value}")

    # Add paths to PATH
    path_additions = [
        fr"{cudnn_path}\bin\{cuda_version}",
        fr"{cudnn_path}\bin",
        fr"{cuda_toolkit_path}\bin",
    ]

    current_path = os.environ.get("PATH", "")
    path_updated = False

    for path_add in path_additions:
        if os.path.exists(path_add) and path_add not in current_path:
            os.environ["PATH"] = path_add + os.pathsep + current_path
            current_path = os.environ["PATH"]
            path_updated = True
            logger.debug(f"Added {path_add} to PATH")

    if path_updated:
        logger.info("CUDA environment configured successfully")

    return True

def test_cuda_availability():
    """Test if CUDA is available and working."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            logger.info(f"CUDA available: {device_count} device(s), primary: {device_name}")
            return True
        else:
            logger.info("CUDA not available")
            return False
    except ImportError:
        logger.debug("PyTorch not installed, cannot test CUDA")
        return False
    except Exception as e:
        logger.warning(f"CUDA test failed: {e}")
        return False

# Initialize CUDA environment when this module is imported
setup_cuda_environment()