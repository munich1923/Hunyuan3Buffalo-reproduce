from stage2_runtime import *
from prepare_stage2 import check_candidate
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def main():
    cfg=config("configs/stage2_nano10k.yaml")
    root=manifest_root(cfg);qc=root/"qc"
    existing=sorted(qc.glob("*.json"))[:4]
    assert len(existing)==4
    ids={p.stem for p in existing}
    rows=[r for r in read_lines(path(cfg["dataset"]["base"])/"v1_100k/info/v1_100k_info.jsonl") if r["uid"] in ids]
    tasks=[(cfg,path(cfg["dataset"]["base"])/"stage2_sources",qc,r) for r in sorted(rows,key=lambda x:x["uid"])]
    serial=[check_candidate(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context("spawn")) as pool:
        parallel=list(pool.map(check_candidate,tasks))
    assert serial==parallel
    print("PARALLEL_QC_MATCHES_SERIAL",len(serial),flush=True)
if __name__=="__main__":main()
