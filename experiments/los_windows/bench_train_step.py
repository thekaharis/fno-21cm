"""Time a full training step (fwd+bwd+optimizer) per model on a real native window."""
import argparse, json, sys, time, torch
sys.path.insert(0, ".")
from fno_multifield import read_json, prepared_dataset, window_configuration
from dataset.fields import FieldRegistry, FieldMapping
from modeling import ModelConfig
from multifield_model import MultiFieldModel
prep=read_json("experiments/los_windows/preparation_xhi_2000.json")
reg=FieldRegistry.from_dict(prep["registry"])
mapping=FieldMapping.create(["density"],["neutral_fraction"],prep["conditioning"],reg)
ds,rows,reg=prepared_dataset(prep,mapping)
a=argparse.Namespace(sampling="contiguous",window_size=256,window_halo=32,windows_per_cone=4,
                     context_factor=4,context_xy=4,context_features=8)
wc=window_configuration(a,ds); s=ds.window(int(rows["train"][0]),0,wc)
x=s["x"][None].cuda(); y=s["y"][None].cuda()
for tag in sys.argv[1:]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m=MultiFieldModel(ModelConfig.from_dict(json.load(open(f"experiments/los_windows/model_{tag}.json"))),
                      ds.in_channels,mapping,reg,wc).cuda()
    opt=torch.optim.Adam(m.parameters(),lr=1e-4)
    def step():
        opt.zero_grad(); loss=((m(x)-y)**2).mean(); loss.backward(); opt.step()
    try:
        for _ in range(2): step()
        torch.cuda.synchronize(); t=[]
        for _ in range(5):
            t0=time.perf_counter(); step(); torch.cuda.synchronize(); t.append(time.perf_counter()-t0)
        t.sort(); med=t[2]
        print(f"{tag:22s} train step {med:6.3f} s  -> 6400 windows = {6400*med/3600:5.2f} h/epoch  "
              f"peak {torch.cuda.max_memory_allocated()/2**30:5.1f} GiB", flush=True)
    except Exception as e:
        print(f"{tag:22s} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
    del m, opt
ds.close()
