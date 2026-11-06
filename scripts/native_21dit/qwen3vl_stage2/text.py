"""Caption identity, independent randomness and the reviewable task prefix."""
import hashlib
import json
import random
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
TASK_PREFIX = "Create a 3D object that faithfully matches the following description:"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_captions(metadata_path, rows, metadata_revision):
    """Join original metadata/candidates.jsonl by exact UID, never dedup aliases."""
    wanted = {row["uid"]: row for row in rows}
    found = {}
    for record in jsonl(metadata_path):
        uid = record["uid"]
        if uid not in wanted:
            continue
        row = wanted[uid]
        if uid in found:
            raise ValueError("Duplicate metadata UID: " + uid)
        if any(record[key] != row[key] for key in ("source", "raw_sha256")):
            raise ValueError("Caption/geometry identity differs: " + uid)
        if record.get("metadata_revision") != metadata_revision:
            raise ValueError("Metadata revision differs: " + uid)
        captions = record.get("captions")
        if not isinstance(captions, list) or any(not isinstance(t, str) for t in captions):
            raise ValueError("Expected original caption list: " + uid)
        indices = [i for i, text in enumerate(captions) if text.strip()]
        if not indices:
            raise ValueError("No nonempty original caption: " + uid)
        found[uid] = {"captions": captions, "nonempty_caption_indices": indices}
    missing = wanted.keys() - found.keys()
    if missing:
        raise ValueError("Missing same-UID metadata captions: " + ", ".join(sorted(missing)[:8]))
    return found


class CaptionSampler:
    """Uniform per-occurrence choice; preserve duplicates and original strings."""
    def __init__(self, seed=20261027):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

    def choose(self, record):
        index = self.rng.choice(record["nonempty_caption_indices"])
        text = record["captions"][index]
        return {"text": text, "caption_index": index,
                "caption_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}

    def state_dict(self):
        return {"seed": self.seed, "rng": self.rng.getstate()}

    def load_state_dict(self, state):
        if state["seed"] != self.seed:
            raise ValueError("Caption seed differs from checkpoint")
        self.rng.setstate(state["rng"])


def messages_for(text, task_prefix=TASK_PREFIX):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Expected a nonempty caption string")
    content = task_prefix + "\n" + text if task_prefix else text
    return [{"role": "user", "content": content}]
