"""CPU geometry and sampling contracts. No imports from the editing data pipeline."""
from collections import Counter, defaultdict
import hashlib
import random
import numpy as np
import trimesh
from stage2_runtime import seed, fingerprint


def quotas(counts, total):
    size = sum(counts.values())
    exact = {key: value * total / size for key, value in sorted(counts.items())}
    result = {key: int(value) for key, value in exact.items()}
    order = sorted(result, key=lambda key: (-(exact[key]-result[key]), key))
    for key in order[:total-sum(result.values())]:
        result[key] += 1
    return result


def load_mesh(file):
    obj = trimesh.load(file, process=False)
    if isinstance(obj, trimesh.Scene):
        parts = []
        for node in obj.graph.nodes_geometry:
            transform, geometry = obj.graph[node]
            part = obj.geometry[geometry].copy()
            if isinstance(part, trimesh.Trimesh):
                part.apply_transform(transform)
                parts.append(part)
        if not parts:
            raise ValueError("No triangle geometry")
        obj = trimesh.util.concatenate(parts)
    if not isinstance(obj, trimesh.Trimesh) or len(obj.vertices)<4 or len(obj.faces)<4:
        raise ValueError("Empty/tiny mesh")
    if not np.isfinite(obj.vertices).all() or not np.isfinite(obj.face_normals).all():
        raise ValueError("Nonfinite coordinates/normals")
    if obj.faces.min()<0 or obj.faces.max()>=len(obj.vertices):
        raise ValueError("Invalid face index")
    if not np.isfinite(obj.area) or obj.area<=1e-10:
        raise ValueError("Degenerate area")
    return obj


def normalize(mesh, extent=1.5):
    center = mesh.bounds.mean(axis=0)
    longest = np.ptp(mesh.vertices, axis=0).max()
    if not np.isfinite(longest) or longest <= 1e-8:
        raise ValueError("Degenerate bounds")
    result = mesh.copy()
    result.vertices = (result.vertices-center) * (extent/longest)
    return result, center.tolist(), float(extent/longest)


def geometry_hash(mesh):
    # Conservative identity grouping, invariant to vertex order/translation/uniform scale.
    vertices = mesh.vertices-mesh.bounds.mean(axis=0)
    vertices /= np.ptp(mesh.vertices, axis=0).max()
    vertices = np.unique(np.round(vertices, 5), axis=0).astype("<f8")
    return hashlib.sha256(vertices.tobytes()).hexdigest()


def surface(mesh, count, random_seed):
    state = np.random.get_state()
    try:
        np.random.seed(random_seed)
        points, faces = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    normals = mesh.face_normals[faces]
    normals = normals / np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    return np.concatenate([points,normals,np.zeros((count,1))],axis=1).astype("float32")


def stratify(rows, full_counts, sizes, master):
    by_category = defaultdict(list)
    for row in sorted(rows, key=lambda row: row["uid"]):
        by_category[row["category"]].append(row)
    allocations = {split: quotas(full_counts, size) for split,size in sizes.items()}
    deficits = {}
    for category in full_counts:
        required = sum(q[category] for q in allocations.values())
        if len(by_category[category]) < required:
            deficits[category] = {"available":len(by_category[category]),"required":required}
    if deficits:
        raise ValueError("Category deficits: "+str(deficits))
    output = {split: [] for split in sizes}
    for category, group in sorted(by_category.items()):
        random.Random(seed(master, "split", category)).shuffle(group)
        offset = 0
        for split in sizes:
            amount = allocations[split][category]
            output[split].extend({**r,"split":split} for r in group[offset:offset+amount])
            offset += amount
    for split, group in output.items():
        random.Random(seed(master, "manifest_order", split)).shuffle(group)
    return output, allocations


def subset(rows, count, master, name):
    categories = Counter(row["category"] for row in rows)
    allocation = quotas(categories, count)
    result = []
    for category in sorted(categories):
        group = [row for row in rows if row["category"]==category]
        random.Random(seed(master, name, category)).shuffle(group)
        result += group[:allocation[category]]
    random.Random(seed(master, name, "order")).shuffle(result)
    return [row["uid"] for row in result]


def protocol(splits, cfg):
    e, master = cfg["evaluation"], cfg["seed"]
    out = {
        "version":1,
        "train_monitor":subset(splits["train"],e["monitor_train"],master,"train_monitor"),
        "val_monitor":subset(splits["val"],e["monitor_val"],master,"val_monitor"),
        "vae":[],
        "note":"Semantic judgments require recorded review; never inferred from Chamfer or loss.",
    }
    combined = splits["train"]+splits["val"]
    for category in sorted({r["category"] for r in combined}):
        group = [r for r in combined if r["category"]==category]
        random.Random(seed(master,"vae",category)).shuffle(group)
        out["vae"] += [r["uid"] for r in group[:e["vae_per_category"]]]
    for split in ["val","test"]:
        ids = subset(splits[split],e["heldout_count"],master,split+"_heldout")
        out[split+"_heldout"] = ids
        lookup = {r["uid"]:r for r in splits[split]}
        pairs = []
        for kind, amount in [("cross",e["pairs_cross"]),("same",e["pairs_same"])]:
            candidates = ids.copy()
            random.Random(seed(master,split,kind)).shuffle(candidates)
            used = set()
            for uid in candidates:
                alternatives = [other for other in ids if other!=uid and
                    ((lookup[other]["category"]==lookup[uid]["category"])==(kind=="same"))]
                if not alternatives: continue
                other = random.Random(seed(master,split,kind,uid)).choice(alternatives)
                if uid in used: continue
                used.add(uid)
                pairs.append({"uid":uid,"wrong_uid":other,"kind":kind,
                              "review_confirmed_geometrically_distinguishable":False})
                if len(used)==amount: break
            if len(used)!=amount: raise ValueError("Insufficient condition pairs")
        out[split+"_pairs"] = pairs
    out["empty_val"] = out["val_heldout"][:e["empty_diagnostics"]]
    monitor = out["val_monitor"]
    out["monitor_pairs"] = [{"uid":uid,"wrong_uid":monitor[(i+1)%len(monitor)]}
                            for i,uid in enumerate(monitor[:e["monitor_wrong"]])]
    out["fingerprint"] = fingerprint(out)
    return out
