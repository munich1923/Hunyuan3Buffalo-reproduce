from stage2_runtime import *
from prepare_stage2 import extract_sources
p=parser(__doc__);args=p.parse_args();cfg=config(args.config)
receipts=manifest_root(cfg)/"downloads"
cfg["dataset"]["shards"]=[n for n in cfg["dataset"]["shards"] if (receipts/f"part_{n:03d}.tar.gz.json").exists()]
with lock("environment/stage2_extract.lock"):
    extract_sources(cfg,path(cfg["dataset"]["base"]))
print("VERIFIED_EXTRACTION_DONE",len(cfg["dataset"]["shards"]),flush=True)
