import sys
out = []
try:
    import torch
    out.append(f"PyTorch {torch.__version__}, CUDA build: {getattr(torch.version,'cuda',None)}")
    out.append(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        out.append(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
        free,total = torch.cuda.mem_get_info()
        out.append(f"VRAM free/total: {free/1024**3:.2f}/{total/1024**3:.2f} GiB")
except Exception as e:
    out.append(f"torch import failed: {e}")

try:
    import torch_directml as dml
    out.append(f"DirectML available: True | device: {dml.device()}")
except Exception:
    out.append("DirectML available: False")

print("\n".join(out))
