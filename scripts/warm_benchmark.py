import json
from prepare_dataset import prepare_bench
from remote_runtime import ROOT
records=prepare_bench()
out=ROOT/"assets/manifests";out.mkdir(parents=True,exist_ok=True)
(out/"benchmark_downloaded.jsonl").write_text("".join(json.dumps(x)+"\n" for x in records))
print("BENCHMARK DOWNLOAD VERIFIED",len(records),flush=True)
