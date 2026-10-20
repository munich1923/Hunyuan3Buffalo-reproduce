"""Pinned Nano3D source-only acquisition, resumable CPU QC and immutable splits."""
from stage2_runtime import *
import re
import shutil
import tarfile
import urllib.parse
import urllib.request
from collections import Counter


def download_command(partial, url):
    return ["aria2c", "--async-dns=false", "--disable-ipv6=true", "--continue=true",
            "--allow-overwrite=true", "--auto-file-renaming=false", "--file-allocation=none",
            "--max-connection-per-server=1", "--split=1", "--max-tries=3",
            "--retry-wait=30", "--connect-timeout=20", "--timeout=60",
            "--lowest-speed-limit=64K", "--summary-interval=30", "--console-log-level=warn",
            "--dir", str(partial.parent), "--out", partial.name, url]


def download_rate(log_file, offset):
    """Last measured network rate for this attempt, never sparse file length."""
    with log_file.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(offset, stream.tell() - 16384))
        text = stream.read().decode(errors="replace")
    rates = re.findall(r"DL:([0-9.]+)([KMGT]?i?B)", text)
    if not rates:
        return None
    value, unit = rates[-1]
    powers = {"B": 0, "KiB": 1, "MiB": 2, "GiB": 3, "TiB": 4,
              "KB": 1, "MB": 2, "GB": 3, "TB": 4}
    return float(value) * (1024 ** powers[unit])


def download_archive(cfg, name, partial, size, receipts):
    log_file = receipts / (name.rsplit("/", 1)[-1] + ".log")
    # Keep aria2 control files: file length alone can include sparse holes.
    for endpoint in ["https://hf-mirror.com", "https://huggingface.co"]:
        base_url = f"{endpoint}/datasets/{cfg['dataset']['repo']}/resolve/{cfg['dataset']['revision']}/{name}"
        for attempt in range(8):
            url = base_url + "?download=true&stage2_resume=" + str(time.time_ns())
            append(receipts / "events.jsonl", {"event": "download_attempt", "file": name,
                   "endpoint": endpoint, "attempt": attempt + 1, "policy": "low-speed-recovery-v3"})
            with log_file.open("a") as log:
                log.write(f"\nSTAGE2_RESUME_V3 endpoint={endpoint} attempt={attempt + 1}\n")
                log.flush()
                offset = log.tell()
                slow_since = None
                process = subprocess.Popen(download_command(partial, url), stdout=log,
                                           stderr=subprocess.STDOUT)
                try:
                    while True:
                        try:
                            code = process.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            stat = partial.stat() if partial.exists() else None
                            rate = download_rate(log_file, offset)
                            now = time.monotonic()
                            slow_since = (slow_since if slow_since is not None else now) if rate is not None and rate < 128 * 1024 else None
                            append(receipts / "events.jsonl", {"event": "download_heartbeat",
                                   "file": name, "pid": process.pid, "expected_bytes": size,
                                   "download_Bps": rate,
                                   "file_bytes": stat.st_size if stat else 0,
                                   "allocated_bytes": stat.st_blocks * 512 if stat else 0})
                            if slow_since is not None and now - slow_since >= 120:
                                append(receipts / "events.jsonl", {"event": "download_slow_restart",
                                       "file": name, "download_Bps": rate, "slow_seconds": now - slow_since})
                                code = 75
                                break
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try: process.wait(timeout=15)
                        except subprocess.TimeoutExpired: process.kill(); process.wait()
            if code == 0 and partial.exists() and partial.stat().st_size == size:
                return
            delay = min(60 * 2 ** attempt, 300)
            append(receipts / "events.jsonl", {"event": "download_backoff", "file": name,
                   "endpoint": endpoint, "exit_code": code, "seconds": delay})
            print("DOWNLOAD_RETRY", name, "exit", code, "backoff", delay, flush=True)
            if attempt < 7:
                time.sleep(delay)
    raise RuntimeError("Download failed; partial and transfer logs retained: " + name)


def acquire(cfg, offline=False):
    base = path(cfg["dataset"]["base"])
    meta = read_json("environment/manifest-yejunliang23--Nano3D-Edit-100k.json")
    if meta["sha"] != cfg["dataset"]["revision"]:
        raise RuntimeError("Pinned dataset manifest mismatch")
    items = {item["rfilename"]:item for item in meta["siblings"]}
    receipts = manifest_root(cfg)/"downloads"
    receipts.mkdir(parents=True, exist_ok=True)
    def acquire_shard(number):
        name = f"v1_100k/editing_assets/part_{number:03d}.tar.gz"
        item = items[name]
        target = path(base/name)
        receipt = receipts/(target.name+".json")
        expected = item["lfs"]["sha256"]
        size = item["size"]
        if target.exists() and receipt.exists():
            old = read_json(receipt)
            stat = target.stat()
            if (old.get("sha256")==expected and old.get("bytes")==stat.st_size==size
                    and old.get("mtime_ns")==stat.st_mtime_ns):
                return
        if not target.exists():
            if offline: raise RuntimeError("Missing archive: "+name)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name+".partial")
            download_archive(cfg, name, partial, size, receipts)
            if digest(partial)!=expected:
                raise RuntimeError("Archive checksum mismatch: "+name)
            os.replace(partial,target)
        if target.stat().st_size!=size or digest(target)!=expected:
            raise RuntimeError("Archive integrity failed: "+name)
        atomic_json(receipt,{"file":name,"revision":meta["sha"],"sha256":expected,
                             "bytes":size,"mtime_ns":target.stat().st_mtime_ns,**clock()})
        print("VERIFIED",name,flush=True)
    # Serialize transfers to reduce resolver/CDN rate-limit pressure.
    for number in cfg["dataset"]["shards"]:
        acquire_shard(number)
    return base


def extract_sources(cfg, base):
    destination = path(base/"stage2_sources")
    selected = set(cfg["dataset"]["shards"])
    for number in cfg["dataset"]["shards"]:
        receipt = manifest_root(cfg)/"extraction"/f"{number:03d}.json"
        if receipt.exists():
            old = read_json(receipt)
            if len(old["files"])==1000 and all(path(f).is_file() for f in old["files"]):
                continue
        archive = base/f"v1_100k/editing_assets/part_{number:03d}.tar.gz"
        found = {}
        with tarfile.open(archive, "r|gz") as stream:
            for member in stream:
                name = Path(member.name)
                if not member.isfile() or name.name!="src_mesh.glb": continue
                uid = next((p for p in name.parts if re.fullmatch(r"uid_\d+",p)),None)
                if uid is None or int(uid[4:])//1000!=number:
                    raise ValueError("Unexpected source member: "+member.name)
                if uid in found or member.size<=0 or member.size>512*1024**2:
                    raise ValueError("Invalid/duplicate source member: "+member.name)
                target = path(destination/uid/"src_mesh.glb")
                target.parent.mkdir(parents=True,exist_ok=True)
                payload = stream.extractfile(member)
                temp = target.with_name(target.name+".tmp")
                with temp.open("wb") as out:
                    shutil.copyfileobj(payload,out,1024**2)
                if temp.stat().st_size!=member.size:
                    raise ValueError("Short extraction: "+uid)
                os.replace(temp,target)
                found[uid]=str(target)
        if len(found)!=1000:
            raise RuntimeError(f"Expected 1000 source meshes in shard {number}: {len(found)}")
        atomic_json(receipt,{"files":list(found.values()),"archive_sha256":digest(archive)})
        print("EXTRACTED_SOURCE_ONLY",number,len(found),flush=True)
    return destination


def prepare(cfg, offline=False):
    root = manifest_root(cfg)
    root.mkdir(parents=True,exist_ok=True)
    if (root/"ready.json").exists():
        for split in cfg["dataset"]["counts"]: manifests(cfg,split)
        print("Immutable manifests already prepared",flush=True)
        return
    base = acquire(cfg,offline)
    destination = extract_sources(cfg,base)
    from stage2_data import load_mesh,normalize,geometry_hash,stratify,protocol
    metadata = read_lines(base/"v1_100k/info/v1_100k_info.jsonl")
    if len(metadata)!=100000 or len({r["uid"] for r in metadata})!=100000:
        raise RuntimeError("Expected 100000 unique original objects")
    counts = Counter(row["category"].strip() for row in metadata)
    candidates = [row for row in metadata if int(row["uid"][4:])//1000
                  in cfg["dataset"]["shards"]]
    accepted, rejected, seen = [], [], {}
    qc_dir = root/"qc"
    qc_dir.mkdir(parents=True,exist_ok=True)
    for index, row in enumerate(sorted(candidates,key=lambda r:r["uid"])):
        uid=row["uid"]
        file=path(destination/uid/"src_mesh.glb")
        record_file=qc_dir/(uid+".json")
        sha=digest(file)
        contract=fingerprint({"mesh":sha,"version":cfg["dataset"]["geometry_version"],
                              "extent":cfg["dataset"]["normalized_extent"]})
        if record_file.exists() and read_json(record_file).get("qc_contract")==contract:
            result=read_json(record_file)
        else:
            try:
                if not isinstance(row.get("prompt"),str) or not row["prompt"].strip():
                    raise ValueError("Empty original prompt")
                mesh=load_mesh(file)
                normalized, center, scale=normalize(mesh,cfg["dataset"]["normalized_extent"])
                result={"accepted":True,"uid":uid,"prompt":row["prompt"],
                        "category":row["category"].strip(),"mesh":str(file),
                        "mesh_sha256":sha,"geometry_hash":geometry_hash(mesh),
                        "normalization":{"center":center,"scale":scale,
                          "version":cfg["dataset"]["geometry_version"],
                          "extent":cfg["dataset"]["normalized_extent"]},
                        "surface_seed":seed(cfg["seed"],"surface",uid),
                        "vertices":len(mesh.vertices),"faces":len(mesh.faces)}
            except Exception as exc:
                result={"accepted":False,"uid":uid,"reason":repr(exc)}
            result["qc_contract"]=contract
            atomic_json(record_file,result)
        if not result["accepted"]:
            rejected.append(result)
        elif result["geometry_hash"] in seen:
            rejected.append({"uid":uid,"reason":"duplicate_geometry",
                             "representative":seen[result["geometry_hash"]]})
        else:
            seen[result["geometry_hash"]]=uid
            accepted.append(result)
        if (index+1)%100==0:
            print("QC",index+1,"accepted",len(accepted),"rejected",len(rejected),flush=True)
    atomic_lines(root/"accepted.jsonl",accepted)
    atomic_lines(root/"rejected.jsonl",rejected)
    atomic_json(root/"qc_report.json",{"candidate_count":len(candidates),
       "accepted":len(accepted),"rejected":len(rejected),"full_category_counts":dict(counts),
       "accepted_category_counts":dict(Counter(r["category"] for r in accepted))})
    splits, allocations=stratify(accepted,counts,cfg["dataset"]["counts"],cfg["seed"])
    for split,rows in splits.items(): atomic_lines(root/(split+".jsonl"),rows)
    evaluation=protocol(splits,cfg)
    atomic_json(root/"evaluation_protocol.json",evaluation)
    by_uid={row["uid"]:row for group in splits.values() for row in group}
    pair_review={"protocol_fingerprint":evaluation["fingerprint"],"reviewer":None}
    for split in ["val","test"]:
        pair_review[split]=[{**pair,"prompt":by_uid[pair["uid"]]["prompt"],
                           "wrong_prompt":by_uid[pair["wrong_uid"]]["prompt"],
                           "geometrically_distinguishable":None,"notes":None}
                          for pair in evaluation[split+"_pairs"]]
    atomic_json(root/"condition_pair_review.json",pair_review)
    ready={"dataset_contract":dataset_contract(cfg),"sampling":"17-shard-pool-stratified",
           "allocations":allocations,"split_sha256":{s:digest(root/(s+".jsonl")) for s in splits},
           "protocol_sha256":digest(root/"evaluation_protocol.json"),**clock()}
    atomic_json(root/"ready.json",ready)
    print("READY",canonical(ready),flush=True)


def main():
    p=parser(__doc__)
    p.add_argument("--offline",action="store_true",help="Fail rather than download absent archives")
    p.add_argument("--inventory-only",action="store_true")
    args=p.parse_args(); cfg=config(args.config)
    with lock("environment/stage2_prepare.lock"):
        if args.inventory_only:
            meta=read_json("environment/manifest-yejunliang23--Nano3D-Edit-100k.json")
            names={f"v1_100k/editing_assets/part_{n:03d}.tar.gz" for n in cfg["dataset"]["shards"]}
            items=[x for x in meta["siblings"] if x["rfilename"] in names]
            report={"expected_bytes":sum(x["size"] for x in items),"shards":len(items),
                    "existing_bytes":sum(path(cfg["dataset"]["base"]+"/"+x["rfilename"]).stat().st_size
                       for x in items if path(cfg["dataset"]["base"]+"/"+x["rfilename"]).exists())}
            atomic_json(manifest_root(cfg)/"inventory.json",report); print(report)
        else: prepare(cfg,args.offline)

if __name__=="__main__": main()
