"""Environment Validation Script for AI Dehazing Project.

Verifies the installed versions of Python, PyTorch, OpenCV, NumPy, and checks
CUDA / GPU acceleration availability alongside host CPU information.
"""

import os
import platform
import sys
from typing import Dict, Any


def get_cpu_info() -> str:
    """Retrieve readable CPU processor information."""
    processor = platform.processor()
    cores = os.cpu_count()
    if processor and cores:
        return f"{processor} ({cores} logical cores)"
    elif cores:
        return f"{cores} logical cores"
    return "Unknown CPU"


def validate_environment() -> Dict[str, Any]:
    """Inspect and validate the runtime environment.

    Returns:
        dict: Diagnostics dictionary containing check statuses and version strings.
    """
    results: Dict[str, Any] = {
        "python": {"ok": False, "version": sys.version.split()[0]},
        "pytorch": {"ok": False, "version": "Not installed"},
        "opencv": {"ok": False, "version": "Not installed"},
        "numpy": {"ok": False, "version": "Not installed"},
        "cuda_available": False,
        "gpu_name": "N/A",
        "cpu_info": get_cpu_info(),
    }

    # Python version check (requires Python 3.8+)
    if sys.version_info >= (3, 8):
        results["python"]["ok"] = True

    # PyTorch check
    try:
        import torch
        results["pytorch"]["ok"] = True
        results["pytorch"]["version"] = torch.__version__
        if torch.cuda.is_available():
            results["cuda_available"] = True
            results["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    # OpenCV check
    try:
        import cv2
        results["opencv"]["ok"] = True
        results["opencv"]["version"] = cv2.__version__
    except ImportError:
        pass

    # NumPy check
    try:
        import numpy as np
        results["numpy"]["ok"] = True
        results["numpy"]["version"] = np.__version__
    except ImportError:
        pass

    return results


def print_environment_report(results: Dict[str, Any]) -> bool:
    """Print formatted environment report matching project specifications.

    Args:
        results: Dictionary returned by validate_environment().

    Returns:
        bool: True if all core requirements (Python, PyTorch, OpenCV, NumPy) pass.
    """
    print("========================================")
    print("AI DEHAZING PROJECT ENVIRONMENT CHECK")
    print("========================================")
    print()

    py_ok = "OK" if results["python"]["ok"] else "FAIL"
    print(f"Python: {py_ok} ({results['python']['version']})")

    torch_ok = "OK" if results["pytorch"]["ok"] else "FAIL"
    print(f"PyTorch: {torch_ok} ({results['pytorch']['version']})")

    cv_ok = "OK" if results["opencv"]["ok"] else "FAIL"
    print(f"OpenCV: {cv_ok} ({results['opencv']['version']})")

    np_ok = "OK" if results["numpy"]["ok"] else "FAIL"
    print(f"NumPy: {np_ok} ({results['numpy']['version']})")

    print()
    print("Compute Device:")
    if results["cuda_available"]:
        print("    CUDA: Available (YES)")
        print(f"    GPU: {results['gpu_name']}")
    else:
        print("    CUDA: Not available (Using CPU fallback - Non-fatal)")
        print(f"    CPU: {results['cpu_info']}")

    print()
    all_core_ok = (
        results["python"]["ok"]
        and results["pytorch"]["ok"]
        and results["opencv"]["ok"]
        and results["numpy"]["ok"]
    )

    if all_core_ok:
        print("Environment validation completed: SUCCESS")
    else:
        print("Environment validation completed: WARNING - Some dependencies are missing.")
    print("========================================")

    return all_core_ok


if __name__ == "__main__":
    env_results = validate_environment()
    success = print_environment_report(env_results)
    sys.exit(0 if success else 1)
