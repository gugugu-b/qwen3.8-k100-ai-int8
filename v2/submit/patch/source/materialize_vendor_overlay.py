#!/usr/bin/env python3
"""Export exact vendor input/runtime LongKV source and diffs without importing Torch.

Run inside the declared vendor image. This writes only a NEW output directory;
it neither imports candidate sitecustomize nor edits the installed framework.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-packages', type=Path,
                        default=Path('/usr/local/lib/python3.10/dist-packages'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[2]
    anchors = json.loads((Path(__file__).parent / 'longkv-anchors.json').read_text())
    vendor = args.site_packages / 'flash_attn/flash_attn_triton_mqa_gqa.py'
    raw = vendor.read_text()
    states = [(raw.count(anchors['_old_' + name]), raw.count(anchors['_new_' + name]))
              for name in ('start', 'guard')]
    if states == [(1, 0), (1, 0)]:
        modified = raw
        for name in ('start', 'guard'):
            modified = modified.replace(anchors['_old_' + name], anchors['_new_' + name])
        status = 'EXACT_TWO_ANCHOR_EDITS'
    elif states == [(0, 1), (0, 1)]:
        # The measured overlay itself accepts this existing fixed source.
        modified = raw
        status = 'VENDOR_SOURCE_ALREADY_HAS_BOTH_FIXED_ANCHORS'
    else:
        raise SystemExit('Vendor LongKV source differs from both known anchor states: ' + repr(states))
    overlay = package / 'patch/runtime/direct_register_custom_op.py'
    registration = args.site_packages / 'lightop/_lmslim_native/vllm_compat/direct_register_custom_op.py'
    original_registration = registration.read_text()
    new_registration = overlay.read_text()
    outputs = {
        'vendor/flash_attn_triton_mqa_gqa.py': raw,
        'runtime/flash_attn_triton_mqa_gqa.py': modified,
        'longkv-full-source.patch': ''.join(difflib.unified_diff(
            raw.splitlines(True), modified.splitlines(True),
            fromfile='a/flash_attn/flash_attn_triton_mqa_gqa.py',
            tofile='b/flash_attn/flash_attn_triton_mqa_gqa.py')),
        'vendor/direct_register_custom_op.py': original_registration,
        'runtime/direct_register_custom_op.py': new_registration,
        'lightop-full-source.patch': ''.join(difflib.unified_diff(
            original_registration.splitlines(True), new_registration.splitlines(True),
            fromfile='a/lightop/_lmslim_native/vllm_compat/direct_register_custom_op.py',
            tofile='b/lightop/_lmslim_native/vllm_compat/direct_register_custom_op.py')),
    }
    # Check every input before creating output, and never overwrite old evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for name, text in outputs.items():
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        manifest[name] = {'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    result = {'status': status, 'vendor_file': str(vendor),
              'source_is_runtime_equivalent_to': 'patch/sitecustomize.py LongKV _raw transformation',
              'jit_configuration': {'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 0,
                                    'PRE_LOAD_V': False, 'num_stages': 1, 'num_warps': 8},
              'shared_installation_modified': False, 'compiled': False, 'files': manifest}
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
