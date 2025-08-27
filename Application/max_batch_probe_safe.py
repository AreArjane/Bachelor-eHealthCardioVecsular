# max_batch_probe_safe.py
import argparse, time, torch
from torchvision.models import resnet18, resnet50

def build(model): return {"resnet18": resnet18, "resnet50": resnet50}[model]()

def try_step(m, B, C, H, W, device, amp, step_timeout):
    x = torch.randn(B, C, H, W, device=device)
    opt = torch.optim.SGD(m.parameters(), 0.01, 0.9)
    opt.zero_grad(set_to_none=True)
    start = time.time()
    try:
        with torch.amp.autocast(device_type="cuda", enabled=amp):
            y = m(x).sum()
        scaler.scale(y).backward()
        scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
    except RuntimeError as e:
        del x, opt
        torch.cuda.empty_cache()
        msg = str(e).lower()
        if "out of memory" in msg or "cuda error" in msg:
            return False, time.time() - start, "oom"
        # re-raise unexpected errors
        raise
    dur = time.time() - start
    del x, y, opt
    torch.cuda.empty_cache()
    if dur > step_timeout:
        return False, dur, "timeout"
    return True, dur, "ok"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18", choices=["resnet18","resnet50"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--time-budget", type=float, default=10.0, help="total seconds to spend probing")
    ap.add_argument("--step-timeout", type=float, default=3.0, help="max seconds for a single step")
    ap.add_argument("--max-batch", type=int, default=4096)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    m = build(args.model).to(device).train()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    H = W = args.size
    C = args.channels

    # Exponential grow to find an upper bound quickly, with time guard
    B = 1
    best_ok = 0
    upper_fail = None
    t0 = time.time()

    while True:
        if time.time() - t0 > args.time_budget:
            break
        if B > args.max_batch:
            upper_fail = args.max_batch
            break
        ok, dur, why = try_step(m, B, C, H, W, device, args.amp, args.step_timeout)
        if ok:
            best_ok = B
            B *= 2
        else:
            upper_fail = B
            break

    # Binary search between best_ok and upper_fail (if we found a failing bound)
    if upper_fail is not None and best_ok < upper_fail:
        lo, hi = best_ok + 1, upper_fail - 1
        while lo <= hi and (time.time() - t0) < args.time_budget:
            mid = (lo + hi) // 2
            ok, dur, why = try_step(m, mid, C, H, W, device, args.amp, args.step_timeout)
            if ok:
                best_ok = mid
                lo = mid + 1
            else:
                hi = mid - 1

    free, total = torch.cuda.mem_get_info()
    print(f"GPU free/total now: {free/1024**3:.2f}/{total/1024**3:.2f} GiB")
    print(f"Max batch for {args.model}@{H} (AMP={args.amp}) = {best_ok}")
