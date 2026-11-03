"""Shared immutable-manifest helpers for ABO/HSSD preparation."""
from pathlib import Path

from core import ROOT, dirs, fp, read, readlines, sha


def validate_record(folder, record):
    for name, item in record['files'].items():
        file = folder / name
        if not file.is_file() or file.stat().st_size != item['bytes']:
            raise RuntimeError('missing/size changed: ' + str(file))
        if file.stat().st_mtime_ns != item['mtime_ns'] and sha(file) != item['sha256']:
            raise RuntimeError('checksum changed: ' + str(file))


def geometry_row(row, config):
    assets, _ = dirs(config)
    uid = row['uid']
    folder = assets / 'processed_pilot' / uid / 'upstream_geometry_v1'
    if not folder.exists():
        folder = assets / 'processed' / uid / 'upstream_geometry_v1'
    receipt = folder / 'complete.json'
    if not receipt.exists():
        return None

    record = read(receipt)
    validate_record(folder, record)
    if record['contract']['raw_sha256'] != row['raw_sha256']:
        raise RuntimeError('raw geometry identity mismatch')

    def relative_file(suffix):
        return str((folder / (uid + suffix)).relative_to(ROOT))

    def file_sha(suffix):
        return record['files'][uid + suffix]['sha256']

    return {
        **row,
        'mesh': relative_file('_vae.obj'),
        'mesh_sha256': file_sha('_vae.obj'),
        'watertight_mesh': relative_file('_watertight.obj'),
        'watertight_sha256': file_sha('_watertight.obj'),
        'converted_mesh': str((folder / 'converted/mesh.ply').relative_to(ROOT)),
        'converted_mesh_sha256': record['files']['converted/mesh.ply']['sha256'],
        'surface_points': relative_file('_vae_points.npy'),
        'surface_points_sha256': file_sha('_vae_points.npy'),
        'random_surface': relative_file('_vae_random_surface.npy'),
        'random_surface_sha256': file_sha('_vae_random_surface.npy'),
        'coordinate_transform': read(folder / (uid + '_coordinate_transform.json')),
        'preprocessing_receipt': str(receipt.relative_to(ROOT)),
        'preprocessing_receipt_sha256': sha(receipt),
        'quality': record['quality'],
    }


def manifests(config):
    assets, _ = dirs(config)
    root = assets / 'manifests/abo_hssd_v1'
    ready = read(root / 'ready.json')
    if ready['config_contract'] != manifest_contract(config):
        raise RuntimeError('manifest/config fingerprint mismatch')
    for name, expected in ready['files'].items():
        if sha(root / name) != expected:
            raise RuntimeError('immutable manifest changed: ' + name)
    return {split: readlines(root / (split + '.jsonl'))
            for split in ('train', 'val', 'test')}, ready


def manifest_contract(config):
    keys = ('seed', 'upstream_commit', 'metadata_revision', 'trellis_commit',
            'sources', 'split_ratios', 'caption_policy', 'caption_selection',
            'evaluation_caption', 'geometry')
    return fp({key: config[key] for key in keys})
