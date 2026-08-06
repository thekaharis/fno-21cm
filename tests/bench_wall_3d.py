"""Cost of the 3-D chamfer distance at real lightcone-cube size."""
import os, sys, time
sys.path.insert(0, "/pfs/10/work/hd_id260-fno_training/fno-21cm")
os.chdir("/pfs/10/work/hd_id260-fno_training/fno-21cm")
import torch
from losses import ExponentialWallDistance

D, H, W = 140, 140, 256
for B in (1, 2):
    y = (torch.rand(B, 1, D, H, W, device="cuda") > 0.5).float()
    p = torch.rand(B, 1, D, H, W, device="cuda", requires_grad=True)
    for cap in (8, 16, 32):
        e = ExponentialWallDistance(scale=16.0, cap=cap)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(3):
            e(p, y).backward()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / 3
        mem = torch.cuda.max_memory_allocated() / 2**30
        print(f"B={B} cap={cap:2d}: {dt*1000:7.1f} ms/step  peak {mem:5.2f} GiB")
        torch.cuda.reset_peak_memory_stats()
