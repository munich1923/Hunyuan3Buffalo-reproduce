"""Pair existing ABO/HSSD splits with same-UID original images; CPU only.

Never creates geometry, changes coordinates, or starts model/GPU work.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / 'assets/native21_abo_hssd_single_image'
REVIEW = ROOT / 'results/native21_abo_hssd_interface_20261102'
SPLITS = ('train', 'val', 'test')
COORDINATES = 'official_surface_npz_unit_box_no_extra_transform'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def file_record(path):
    st = path.stat()
    return dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns, sha256=sha(path))


def resolve(path):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Path outside project workspace: ' + str(path))
    return path


def validate_pair(geometry, image, cache_root):
    """Full payload check once; reusable only with matching bytes/mtime and contract."""
    import numpy as np
    from PIL import Image
    uid = geometry['uid']
    if image['uid'] != uid or image['representative_uid'] != uid:
        raise ValueError('Same-UID representative pairing required')
    if image['split'] != geometry['split'] or image['raw_sha256'] != geometry['raw_sha256'] or image['source'] != geometry['source']:
        raise ValueError('Image/geometry identity or split mismatch')
    receipt_path = resolve(geometry['preprocessing_receipt'])
    if resolve(image['preprocessing_receipt']) != receipt_path:
        raise ValueError('Image/geometry receipt mismatch')
    receipt = read(receipt_path)
    if sha(receipt_path) != geometry['preprocessing_receipt_sha256'] or sha(receipt_path) != image['preprocessing_receipt_sha256']:
        raise ValueError('Geometry receipt SHA mismatch')
    if receipt['contract']['uid'] != uid or receipt['contract']['raw_sha256'] != geometry['raw_sha256']:
        raise ValueError('Geometry receipt identity mismatch')
    surface = receipt_path.parent / (uid + '_surface.npz')
    picture = resolve(image['image_path'])
    if picture.parent.name != uid or picture.name != image['selected']['filename']:
        raise ValueError('Selected image path mismatch')
    image_receipt_path = picture.parent / 'image_complete.json'
    image_receipt = read(image_receipt_path)
    if image_receipt['uid'] != uid:
        raise ValueError('Image receipt identity mismatch')
    sr = receipt['files'][surface.name]
    ir = image_receipt['files'][picture.name]
    if ir['sha256'] != image['image_sha256'] or ir['sha256'] != image['selected']['sha256']:
        raise ValueError('Selected image hash mismatch')
    observed = {k: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                for k, p in [('surface', surface), ('image', picture)]}
    for key, rec in [('surface', sr), ('image', ir)]:
        if observed[key]['bytes'] != rec['bytes']:
            raise ValueError('Changed payload size: ' + key)
    contract = dict(uid=uid, split=geometry['split'], source=geometry['source'], raw_sha256=geometry['raw_sha256'],
                    geometry_receipt_sha256=geometry['preprocessing_receipt_sha256'],
                    image_receipt_sha256=sha(image_receipt_path), surface_sha256=sr['sha256'], image_sha256=ir['sha256'],
                    observed=observed, coordinate_rule=COORDINATES, validator_sha256=sha(__file__))
    cache = cache_root / (uid + '.json')
    if cache.exists():
        previous = read(cache)
        if previous.get('contract') == contract:
            return previous['row']
    with np.load(surface, allow_pickle=False) as z:
        if set(z.files) != {'random_surface', 'sharp_surface'}:
            raise ValueError('Invalid official surface keys')
        for key in z.files:
            a = z[key]
            if a.shape != (124928, 6) or a.dtype != np.float16 or not np.isfinite(a).all():
                raise ValueError('Invalid/nonfinite surface: ' + key)
            if np.any(np.linalg.norm(a[:, 3:6].astype(np.float32), axis=1) <= 1e-8):
                raise ValueError('Zero surface normal: ' + key)
    if sha(surface) != sr['sha256'] or sha(picture) != ir['sha256']:
        raise ValueError('Payload SHA mismatch')
    with Image.open(picture) as im:
        im.load()
        if im.mode not in ('RGB', 'RGBA') or im.size != (1024, 1024):
            raise ValueError('Unexpected original image mode/size')
        if im.mode == 'RGBA' and im.getchannel('A').getextrema()[1] == 0:
            raise ValueError('Empty alpha mask')
        mode, size = im.mode, list(im.size)
    row = dict(uid=uid, split=geometry['split'], source=geometry['source'], raw_sha256=geometry['raw_sha256'],
               surface=str(surface), sha256=sr['sha256'], size=observed['surface']['bytes'],
               mtime_ns=observed['surface']['mtime_ns'], image=str(picture), image_sha256=ir['sha256'],
               image_size_bytes=observed['image']['bytes'], image_mtime_ns=observed['image']['mtime_ns'],
               image_mode=mode, image_dimensions=size, geometry_receipt=str(receipt_path),
               geometry_receipt_sha256=geometry['preprocessing_receipt_sha256'],
               image_receipt=str(image_receipt_path), image_receipt_sha256=contract['image_receipt_sha256'],
               coordinate_rule=COORDINATES, camera_alignment_to_vae=image.get('camera_alignment_to_vae', 'not established'))
    write(cache, dict(contract=contract, row=row))
    return row


def validate_index(root, full_hash=False, splits=SPLITS):
    """Validate immutable manifests and selected payloads. Does not write files."""
    root = resolve(root)
    ready = read(root / 'ready.json')
    if not ready['passed'] or ready['dataset_kind'] != 'abo_hssd' or ready['coordinate_rule'] != COORDINATES:
        raise ValueError('Invalid paired-index readiness')
    for filename, expected in ready['files'].items():
        if sha(root / filename) != expected:
            raise ValueError('Changed paired-index file: ' + filename)
    for filename, expected in ready['source_files'].items():
        if sha(resolve(filename)) != expected:
            raise ValueError('Changed original source manifest: ' + filename)
    all_uids = set()
    for split in SPLITS:
        rows = jsonl(root / (split + '.jsonl'))
        if len(rows) != ready['counts'][split]:
            raise ValueError('Paired-index count mismatch: ' + split)
        for row in rows:
            if row['uid'] in all_uids or row['split'] != split or row['coordinate_rule'] != COORDINATES:
                raise ValueError('Duplicate UID, split or coordinates mismatch')
            all_uids.add(row['uid'])
            if split not in splits:
                continue
            for key, size_key, mtime_key, hash_key in [
                ('surface', 'size', 'mtime_ns', 'sha256'),
                ('image', 'image_size_bytes', 'image_mtime_ns', 'image_sha256')]:
                path = resolve(row[key]);st = path.stat()
                if st.st_size != row[size_key] or st.st_mtime_ns != row[mtime_key]:
                    raise ValueError('Changed accepted payload: ' + str(path))
                if full_hash and sha(path) != row[hash_key]:
                    raise ValueError('Changed accepted payload SHA: ' + str(path))
            for path_key, hash_key in [('geometry_receipt', 'geometry_receipt_sha256'), ('image_receipt', 'image_receipt_sha256')]:
                if sha(resolve(row[path_key])) != row[hash_key]:
                    raise ValueError('Changed accepted receipt: ' + row['uid'])
    return ready


def build(output=DEFAULT_OUTPUT, workers=2):
    output = resolve(output)
    if (output / 'ready.json').exists():
        return validate_index(output, full_hash=True)
    if output.exists():
        raise RuntimeError('Output exists without final readiness; inspect before replacing')
    m = ROOT / 'assets/abo_hssd_v1/manifests/abo_hssd_v1'
    images_root = ROOT / 'assets/abo_hssd_v1/renders_original'
    migration = read(images_root / 'migration_complete.json')
    if not migration['passed'] or migration['verified'] != migration['expected']:
        raise RuntimeError('Original image migration not complete')
    source_ready = read(m / 'ready.json')
    sources = {str(m / 'ready.json'): sha(m / 'ready.json'),
               str(images_root / 'paired_index.jsonl'): sha(images_root / 'paired_index.jsonl'),
               str(images_root / 'migration_complete.json'): sha(images_root / 'migration_complete.json')}
    if sources[str(images_root / 'paired_index.jsonl')] != migration['paired_index_sha256']:
        raise RuntimeError('Migrated image index changed')
    images = jsonl(images_root / 'paired_index.jsonl')
    by_uid = {row['uid']: row for row in images}
    if len(by_uid) != len(images):
        raise RuntimeError('Duplicate image UIDs')
    geometry = []
    for split in SPLITS:
        path = m / (split + '.jsonl');sources[str(path)] = sha(path)
        if sources[str(path)] != source_ready['files'][path.name]:
            raise RuntimeError('Original geometry split changed: ' + split)
        group = jsonl(path)
        if any(row['split'] != split for row in group):
            raise RuntimeError('Original split mismatch')
        geometry.extend(group)
    if len({g['uid'] for g in geometry}) != len(geometry):
        raise RuntimeError('Original geometry splits overlap')
    cache_root = REVIEW / 'pair_validation';cache_root.mkdir(parents=True, exist_ok=True)
    accepted = {s: [] for s in SPLITS};rejected = [];start = time.monotonic()
    def one(g):
        try:
            if g['uid'] not in by_uid:
                return g, None, 'missing_same_uid_image'
            return g, validate_pair(g, by_uid[g['uid']], cache_root), None
        except (ValueError, OSError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as e:
            return g, None, type(e).__name__ + ': ' + str(e)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for number, (g, row, error) in enumerate(pool.map(one, geometry), 1):
            if error:
                rejected.append(dict(uid=g['uid'], split=g['split'], source=g['source'], reason=error))
            else:
                accepted[g['split']].append(row)
            if number % 200 == 0 or number == len(geometry):
                progress = dict(scanned=number, total=len(geometry), accepted=sum(map(len, accepted.values())),
                                rejected=len(rejected), seconds=time.monotonic()-start, gpu_used=False)
                write(REVIEW / 'index_progress.json', progress)
                print(json.dumps(progress), flush=True)
    tmp = output.with_name(output.name + '.building');tmp.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = sorted(accepted[split], key=lambda row: row['uid'])
        (tmp / (split + '.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    (tmp / 'rejected.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rejected))
    counts = {s: len(accepted[s]) for s in SPLITS}
    summary = dict(original_counts={s: sum(g['split'] == s for g in geometry) for s in SPLITS},
                   counts=counts, rejected=len(rejected), reasons=dict(__import__('collections').Counter(x['reason'] for x in rejected)),
                   sources={s: dict(__import__('collections').Counter(x['source'] for x in accepted[s])) for s in SPLITS},
                   aliases_used=False, coordinate_rule=COORDINATES, gpu_used=False, geometry_recomputed=False)
    write(tmp / 'summary.json', summary)
    if not all(counts[split] > 0 for split in SPLITS):
        raise RuntimeError('Every fixed split must contain at least one paired object')
    ready = dict(schema_version=1, passed=True, dataset_kind='abo_hssd', counts=counts, count=counts['train'],
                 rejected_count=sum(x['split']=='train' for x in rejected), total_rejected=len(rejected),
                 coordinate_rule=COORDINATES, source_files=sources,
                 files={p.name: sha(p) for p in sorted(tmp.iterdir()) if p.is_file()},
                 upstream_commit=read(ROOT / 'configs/abo_hssd_v1.json')['upstream_commit'],
                 image_preprocessing='RGB/RGBA white composite, padding1.15, resize518',
                 validation='all paired original NPZ arrays finite; image decode and payload hashes checked',
                 builder_sha256=sha(__file__), gpu_used=False, training_started=False)
    write(tmp / 'ready.json', ready)
    os.rename(tmp, output)
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        result = validate_index(args.output, full_hash=True)
    else:
        result = build(args.output, args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
