# gpu_capacity_smi.py
# Works even if PyTorch has no CUDA. Uses nvidia-smi (if NVIDIA) or Windows WMI as fallback.

import shutil, subprocess, re, math

def human_gib(x_bytes): return f"{x_bytes/(1024**3):.2f} GiB"

def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except Exception:
        return None

def estimate_table(free_bytes):
    # Very rough upper bounds ignoring activations:
    modes = [
        ("Adam FP32 (~16 B/param)", 16),
        ("Adam AMP  (~14 B/param)", 14),
        ("SGD  FP32 (~12 B/param)", 12),
        ("Weights-only FP16 (~2 B)", 2),
    ]
    lines = []
    for label, bpp in modes:
        params = int(free_bytes // bpp)
        lines.append(f"{label:<28} : {params/1e6:,.0f} M params")
    return "\n".join(lines)

print("="*80)
print("GPU capacity estimate (no CUDA needed)")
print("="*80)

free_bytes = None
total_bytes = None
gpu_name = "Unknown GPU"

# Try NVIDIA SMI first
if shutil.which("nvidia-smi"):
    q = "name,memory.total,memory.free,driver_version,compute_cap"
    out = run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    if out:
        line = out.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpu_name = parts[0]
            total_mib = float(parts[1])
            free_mib  = float(parts[2])
            total_bytes = int(total_mib * 1024 * 1024)
            free_bytes  = int(free_mib  * 1024 * 1024)

# Fallback: Windows WMI (total only)
if total_bytes is None:
    out = run(["wmic","path","win32_VideoController","get","Name,AdapterRAM"])
    if out:
        lines = [l.strip() for l in out.splitlines() if l.strip() and "AdapterRAM" not in l]
        # Take first line with a number
        for l in lines:
            m = re.search(r"(.+?)\s+(\d+)$", l)
            if m:
                gpu_name = m.group(1).strip()
                total_bytes = int(m.group(2))
                # Assume 85% free as a conservative **guess** (no processes using VRAM heavily)
                free_bytes = int(total_bytes * 0.85)
                break

if total_bytes is None:
    print("No GPU info found.\n- If you have NVIDIA, install the full driver, then ensure `nvidia-smi` works.")
    print("- Otherwise, run: dxdiag /t \"%TEMP%\\dx.txt\" and check Display Memory (VRAM).")
else:
    print(f"GPU: {gpu_name}")
    print(f"Total VRAM : {human_gib(total_bytes)}")
    if free_bytes is not None:
        print(f"Free (est.): {human_gib(free_bytes)}")
        usable = int(free_bytes * 0.90)  # keep ~10% safety margin
        print(f"Usable (~90% of free): {human_gib(usable)}\n")
        print("Rule-of-thumb parameter upper bounds (ignore activations):")
        print(estimate_table(usable))
        print("\nNote: training activations can consume 30–60%+ of VRAM.")
    else:
        print("Could not determine free VRAM; estimates need free VRAM to be accurate.")
