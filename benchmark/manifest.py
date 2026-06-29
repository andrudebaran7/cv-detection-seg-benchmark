from __future__ import annotations

import datetime
import platform

import psutil


def _torch_info() -> tuple[str, bool, str]:
    try:
        import torch
    except Exception:
        return ("not-installed", False, "")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    return (torch.__version__, torch.cuda.is_available(), gpu)


def build_manifest(device: str) -> dict:
    torch_version, cuda_available, gpu = _torch_info()
    return {
        "device": device,
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu": gpu,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "python": platform.python_version(),
        "os": platform.platform(),
        "measured_at": datetime.date.today().isoformat(),
    }
