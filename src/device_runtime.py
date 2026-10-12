"""Explicit accelerator checks; no silent CPU fallback for CUDA requests."""
import torch

def require_device(device):
    kind=torch.device(device).type
    if kind=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select the multimodal-gpu interpreter in VS Code')
    if kind=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    return torch.device(device)

def synchronize(device):
    kind=torch.device(device).type
    if kind=='cuda':torch.cuda.synchronize(device)
    elif kind=='mps':torch.mps.synchronize()

def memory_stats(device):
    kind=torch.device(device).type
    if kind=='cuda':
        return {'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(device),'gpu':torch.cuda.get_device_name(device)}
    if kind=='mps':
        return {'mps_current_allocated_bytes':torch.mps.current_allocated_memory(),'mps_driver_allocated_bytes':torch.mps.driver_allocated_memory()}
    return {}
