#!/usr/bin/env python3
"""CPU-only, streaming verification of the recorded external model and dataset."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_KEY = '/data/model/Qwen3.8-27B-Channel-INT8-w8a8'
DEFAULT_MANIFEST = (ROOT / 'result/evidence/official-alignment-20261017'
                    '/community-official-20261017-c/model-sha256.json')
DATASET_SHA256 = '35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4'


def hash_file(path):
    """Read in bounded blocks; never load the model or dataset into memory."""
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def model_entries(manifest):
    content = manifest.read_bytes()
    entries = json.loads(content)[MODEL_KEY]
    if not isinstance(entries, dict) or len(entries) != 31:
        raise ValueError('Expected exactly 31 recorded INT8 model files')
    for relative, expected in entries.items():
        rel = Path(relative)
        if rel.is_absolute() or not rel.parts or '..' in rel.parts:
            raise ValueError(f'Unsafe model manifest path: {relative}')
        if (not isinstance(expected, dict)
                or type(expected.get('bytes')) is not int or expected['bytes'] < 0
                or not isinstance(expected.get('sha256'), str)
                or re.fullmatch('[0-9a-f]{64}', expected['sha256']) is None):
            raise ValueError(f'Invalid size/SHA256 entry: {relative}')
    return entries, hashlib.sha256(content).hexdigest()


def check_assets(model_path, dataset_path, manifest):
    report = {
        'status': 'EXTERNAL_ASSET_CHECK_FAILED',
        'scope': ('CPU byte-count and SHA256 identity check of the 31 manifest-listed '
                  'INT8 model/config/tokenizer files and the fixed ShareGPT file; '
                  'unlisted model files, quantization provenance, official asset permission, '
                  'GPU runtime, performance and accuracy are not verified'),
        'manifest': {'path': str(manifest), 'model_key': MODEL_KEY,
                     'sha256': None, 'default_recorded_manifest': manifest == DEFAULT_MANIFEST.resolve()},
        'model': {'path': str(model_path), 'expected_files': 31, 'checked_files': 0,
                  'matched_files': 0, 'expected_bytes': None, 'files': []},
        'dataset': {'path': str(dataset_path), 'expected_sha256': DATASET_SHA256,
                    'bytes': None, 'sha256': None, 'status': 'NOT_CHECKED'},
        'errors': [],
    }
    try:
        entries, digest = model_entries(manifest)
        report['manifest']['sha256'] = digest
        report['model']['expected_bytes'] = sum(v['bytes'] for v in entries.values())
    except (OSError, ValueError, KeyError, TypeError) as error:
        report['errors'].append({'asset': 'manifest', 'error': str(error)})
        return report

    for relative, expected in sorted(entries.items()):
        row = {'file': relative, 'expected_bytes': expected['bytes'],
               'expected_sha256': expected['sha256'], 'status': 'FAIL'}
        try:
            asset = model_path / relative
            if not asset.resolve().is_relative_to(model_path):
                raise ValueError('Model file resolves outside the selected model directory')
            if not asset.is_file():
                raise ValueError('Missing model file or not a regular file')
            size, digest = hash_file(asset)
            row.update(bytes=size, sha256=digest)
            report['model']['checked_files'] += 1
            if size != expected['bytes'] or digest != expected['sha256']:
                raise ValueError('Model byte count or SHA256 mismatch')
            row['status'] = 'PASS'
            report['model']['matched_files'] += 1
        except (OSError, ValueError) as error:
            row['error'] = str(error)
            report['errors'].append({'asset': 'model', 'file': relative, 'error': str(error)})
        report['model']['files'].append(row)

    try:
        if not dataset_path.is_file():
            raise ValueError('Missing ShareGPT file or not a regular file')
        size, digest = hash_file(dataset_path)
        report['dataset'].update(bytes=size, sha256=digest)
        if digest != DATASET_SHA256:
            raise ValueError('ShareGPT SHA256 mismatch')
        report['dataset']['status'] = 'PASS'
    except (OSError, ValueError) as error:
        report['dataset']['status'] = 'FAIL'
        report['dataset']['error'] = str(error)
        report['errors'].append({'asset': 'dataset', 'error': str(error)})

    if not report['errors']:
        report['status'] = 'EXTERNAL_ASSET_IDENTITY_PASS'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path,
                        default=Path(os.environ.get('MODEL_PATH', MODEL_KEY)))
    parser.add_argument('--dataset-path', type=Path,
                        default=Path(os.environ.get('DATASET_PATH',
                            '/data/wangzy/datasets/ShareGPT_V3_unfiltered_cleaned_split.json')))
    parser.add_argument('--model-manifest', type=Path, default=DEFAULT_MANIFEST,
                        help='Recorded manifest; an explicit override is identified in the report')
    parser.add_argument('--output', type=Path,
                        help='Optional report in an existing directory; never overwrite')
    args = parser.parse_args()
    model_path = args.model_path.resolve()
    dataset_path = args.dataset_path.resolve()
    manifest = args.model_manifest.resolve()
    if args.output:
        output = args.output.resolve()
        if args.output.exists() or args.output.is_symlink():
            parser.error(f'Refusing to overwrite output: {args.output}')
        if output.is_relative_to(model_path) or output in (dataset_path, manifest):
            parser.error('Report output must be outside the model directory and input assets')
        if not output.parent.is_dir():
            parser.error(f'Output parent directory does not exist: {output.parent}')
    report = check_assets(model_path, dataset_path, manifest)
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        try:
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(text)
        except OSError as error:
            parser.error(f'Cannot create report: {error}')
    print(text, end='')
    return 0 if report['status'] == 'EXTERNAL_ASSET_IDENTITY_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
