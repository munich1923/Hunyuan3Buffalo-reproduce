"""Source and runtime contracts for the official ABO/HSSD training path."""
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_settings(path):
    path = Path(path).resolve()
    config = read(path)
    assert config['root'] == str(ROOT) and config['gpu_index'] == 1
    assert config['gpu_uuid'] == 'GPU-47f8e8ad-bf0a-79d9-2e95-645bf2ee0609'
    assert isinstance(config['max_updates'], int) and config['max_updates'] > 0
    assert config['microbatch'] == 1 and config['accumulation'] == 8
    assert config['checkpoint_every'] == 3000
    config['_settings_path'] = str(path)
    return config


def pretrained_files(config):
    """Resolve pretrained files from the official training YAML."""
    import yaml

    training = yaml.safe_load(Path(config['training_yaml']).read_text())
    model = training['model']['params']
    expected = {
        'first_stage_config': 'hy3dshape.models.autoencoders.ShapeVAE',
        'denoiser_cfg': 'hy3dshape.models.denoisers.hunyuandit.HunYuanDiTPlain',
        'cond_stage_config': 'hy3dshape.models.conditioner.SingleImageEncoder',
    }
    for key, target in expected.items():
        if model[key]['target'] != target:
            raise RuntimeError('Expected official model target ' + key)

    vae_root = Path(model['first_stage_config']['from_pretrained'])
    dit_root = Path(model['denoiser_cfg']['from_pretrained'])
    if not vae_root.is_absolute() or not dit_root.is_absolute():
        raise RuntimeError('Expected absolute local pretrained model roots')
    dino = model['cond_stage_config']['params']['main_image_encoder']
    if dino['type'] != 'DinoImageEncoder' or 'config' in dino['kwargs']:
        raise RuntimeError('Expected official DINO version loading')
    dino_root = Path(dino['kwargs']['version'])
    if not dino_root.is_absolute():
        raise RuntimeError('Expected absolute local DINO version path')

    return {
        'model_weights': dit_root / 'hunyuan3d-dit-v2-1/model.fp16.ckpt',
        'model_config': dit_root / 'hunyuan3d-dit-v2-1/config.yaml',
        'vae_weights': vae_root / 'hunyuan3d-vae-v2-1/model.fp16.ckpt',
        'vae_config': vae_root / 'hunyuan3d-vae-v2-1/config.yaml',
        'dino_weights': dino_root / 'model.safetensors',
        'dino_config': dino_root / 'config.json',
    }


def verify_pretrained(config):
    files = pretrained_files(config)
    for key in ('model_weights', 'vae_weights'):
        if files[key].resolve() != Path(config[key]).resolve():
            raise RuntimeError('YAML/settings pretrained path mismatch ' + key)
    expected_hashes = {
        'model_weights': 'model_sha256',
        'vae_weights': 'vae_sha256',
        'dino_weights': 'dino_sha256',
        'dino_config': 'dino_config_sha256',
    }
    for key, config_key in expected_hashes.items():
        if sha(files[key]) != config[config_key]:
            raise RuntimeError('Pretrained file SHA changed: ' + key)
    for key in ('model_config', 'vae_config'):
        if not files[key].is_file():
            raise RuntimeError('Missing official model config ' + key)
    return files


def verify_upstream(config):
    upstream = ROOT / 'src/Hunyuan3D-2.1'
    commit = subprocess.check_output(
        ['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True
    ).strip()
    if commit != config['upstream_commit']:
        raise RuntimeError('Upstream Hunyuan3D-2.1 commit mismatch')
    if subprocess.check_output(
            ['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Upstream Hunyuan3D-2.1 submodule is dirty')
    sampler = upstream / 'hy3dshape/tools/watertight/watertight_and_sample.py'
    if sha(sampler) != config['watertight_script_sha256']:
        raise RuntimeError('Official surface sampler SHA mismatch')


def integration_contract(config):
    if config.get('dataset_kind') != 'abo_hssd':
        raise RuntimeError('This project only supports the ABO/HSSD dataset contract')

    from scripts.native_21dit.native21_official.prepare_abo_hssd_index import validate_index

    index_root = Path(config['paired_index_root'])
    ready = validate_index(index_root, full_hash=False, splits=())
    verify_upstream(config)
    pretrained = verify_pretrained(config)
    return {
        'dataset_kind': 'abo_hssd',
        'settings_sha256': sha(config['_settings_path']),
        'training_yaml_sha256': sha(config['training_yaml']),
        'scripts': {
            path.name: sha(path)
            for path in sorted((ROOT / 'scripts/native_21dit/native21_official').glob('*.py'))
        },
        'upstream_commit': config['upstream_commit'],
        'paired_ready_sha256': sha(index_root / 'ready.json'),
        'paired_files': ready['files'],
        'source_files': ready['source_files'],
        'coordinate_rule': ready['coordinate_rule'],
        'accepted_count': ready['count'],
        'pretrained_files': {key: sha(path) for key, path in pretrained.items()},
        'image_adapter_sha256': sha(ROOT / 'scripts/native_21dit/native21_official/data.py'),
    }
