#!/usr/bin/env python3
"""Offline stdlib-only audit of delivered patch sources; does not import GPU code."""
import argparse
import ast
import difflib
import hashlib
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diff(old, new, before, after):
    return ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                      fromfile=before, tofile=after))


def literal_assignments(path, names):
    found = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    assert set(found) == set(names), ('missing assignments', path, names)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args()
    root = args.package.resolve()
    patch = root / 'patch'
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    manifest = json.loads((patch / 'bundle-manifest.json').read_text())
    for name, entry in manifest['files'].items():
        path = root / name
        check('sha256:' + name, path.is_file() and path.stat().st_size == entry['bytes']
              and sha(path) == entry['sha256'])
    for name in manifest['files']:
        if name.endswith('.py'):
            ast.parse((root / name).read_text(), filename=name)
    check('python_sources_parse', True)
    base = (patch / 'baseline-v1/sitecustomize.py').read_text()
    candidate = (patch / 'sitecustomize.py').read_text()
    check('candidate_preserves_v1_full_prefix', candidate.startswith(base))
    marker = '# Independent communication item; unchanged tile base.\n'
    tile = candidate[:candidate.index(marker)]
    expected = diff(base, tile, 'a/sitecustomize.py', 'b/sitecustomize.py') + diff(
        '', (patch / 'triton_tiles.json').read_text(), '/dev/null', 'b/triton_tiles.json')
    check('v2_tile_diff_exact', expected == (patch / 'diffs/0002-v2-triton-tiles.patch').read_text())
    expected = diff(tile, candidate, 'a/sitecustomize.py', 'b/sitecustomize.py') + diff(
        '', (patch / 'native_bf16_ar_guard.py').read_text(), '/dev/null', 'b/native_bf16_ar_guard.py')
    check('v2_native_guard_diff_exact', expected == (patch / 'diffs/0003-v2-native-bf16-guard.patch').read_text())
    for name in ('k100_int8_gemv_unified.hip', 'k100_int8_gemv_unified.so',
                 'build_unified.py', 'patch_dup_reg_guard.py',
                 'libtorch_c_dlpack_addon_torch210-rocm.so'):
        check('inherited_unchanged:' + name, sha(patch / name) == sha(patch / 'baseline-v1' / name))
    rows = json.loads((patch / 'triton_tiles.json').read_text())
    check('18_unique_tiles_split_k_1', len(rows) == 18
          and len({(r['M'], r['K'], r['N']) for r in rows}) == 18
          and all(r['config']['SPLIT_K'] == 1 for r in rows))
    replacements = literal_assignments(patch / 'patch_dup_reg_guard.py', ('old', 'new'))
    overlay = (patch / 'runtime/direct_register_custom_op.py').read_text()
    check('dup_reg_overlay_exact_new_block', overlay.count(replacements['new']) == 1)
    original = overlay.replace(replacements['new'], replacements['old'])
    reconstructed = patch / 'source/reconstructed-stock/lightop/_lmslim_native/vllm_compat/direct_register_custom_op.py'
    check('dup_reg_preimage_explicit_reconstruction', reconstructed.read_text() == original)
    expected = diff(original, overlay,
                    'a/lightop/_lmslim_native/vllm_compat/direct_register_custom_op.py',
                    'b/lightop/_lmslim_native/vllm_compat/direct_register_custom_op.py')
    check('dup_reg_diff_exact', expected == (patch / 'diffs/0001-lightop-duplicate-registration.patch').read_text())
    anchors = literal_assignments(patch / 'baseline-v1/sitecustomize.py',
                                  ('_old_start', '_new_start', '_old_guard', '_new_guard'))
    check('longkv_anchors_from_actual_source', anchors == json.loads((patch / 'source/longkv-anchors.json').read_text()))
    expected = diff(anchors['_old_start'], anchors['_new_start'],
                    'vendor:start_m-anchor', 'runtime:start_m-anchor') + diff(
        anchors['_old_guard'], anchors['_new_guard'], 'vendor:guard-anchor', 'runtime:guard-anchor')
    check('longkv_anchor_diff_exact', expected == (patch / 'diffs/0004-inherited-longkv-anchor-edits.diff').read_text())
    exported = patch / 'source/fixed-image-export'
    materialized = exported / 'materialized'
    exported_manifest = json.loads((materialized / 'manifest.json').read_text())
    for name, entry in exported_manifest['files'].items():
        path = materialized / name
        check('fixed_image_export_sha:' + name, sha(path) == entry['sha256']
              and path.stat().st_size == entry['bytes'])
    environment = json.loads((exported / 'export-environment.json').read_text())
    check('fixed_image_export_no_gpu', environment['devices_mounted'] is False)
    check('fixed_image_export_image_identity', environment['image_id'] ==
          'sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad')
    check('actual_lightop_stock_matches_reconstructed_preimage',
          (materialized / 'vendor/direct_register_custom_op.py').read_text() == original)
    check('actual_lightop_runtime_matches_delivered_overlay',
          (materialized / 'runtime/direct_register_custom_op.py').read_text() == overlay)
    check('actual_lightop_diff_matches_delivered_diff',
          (materialized / 'lightop-full-source.patch').read_text() ==
          (patch / 'diffs/0001-lightop-duplicate-registration.patch').read_text())
    vendor_longkv = (materialized / 'vendor/flash_attn_triton_mqa_gqa.py').read_text()
    runtime_longkv = vendor_longkv
    for key in ('start', 'guard'):
        check('actual_longkv_unique_old_anchor:' + key,
              runtime_longkv.count(anchors['_old_' + key]) == 1)
        runtime_longkv = runtime_longkv.replace(anchors['_old_' + key], anchors['_new_' + key])
    check('actual_longkv_runtime_matches_source_transformation', runtime_longkv ==
          (materialized / 'runtime/flash_attn_triton_mqa_gqa.py').read_text())
    expected = diff(vendor_longkv, runtime_longkv,
                    'a/flash_attn/flash_attn_triton_mqa_gqa.py',
                    'b/flash_attn/flash_attn_triton_mqa_gqa.py')
    check('actual_longkv_full_diff_exact', expected ==
          (materialized / 'longkv-full-source.patch').read_text() ==
          (patch / 'diffs/0005-inherited-longkv-full-source.patch').read_text())
    tvm = exported / 'tvm-ffi-source'
    tvm_manifest = json.loads((tvm / 'manifest.json').read_text())
    check('tvm_fixed_image_identity', tvm_manifest['image_id'] == environment['image_id'])
    for name, entry in tvm_manifest['files'].items():
        check('tvm_export_sha:' + name, sha(tvm / name) == entry['sha256']
              and (tvm / name).stat().st_size == entry['bytes'])
    tvm_cpp = literal_assignments(tvm / 'utils/_build_optional_torch_c_dlpack.py', ('cpp_source',))
    check('tvm_cpp_exact_ast_literal_extraction', (tvm / 'addon.cc').read_text() == tvm_cpp['cpp_source'])
    licenses = exported / 'licenses'
    license_manifest = json.loads((licenses / 'manifest.json').read_text())
    for name, entry in license_manifest['files'].items():
        check('tvm_license_sha:' + name, sha(licenses / name) == entry['sha256']
              and (licenses / name).stat().st_size == entry['bytes'])
    license_environment = json.loads((licenses / 'export-environment.json').read_text())
    check('tvm_license_export_no_gpu', license_environment['devices_mounted'] is False
          and license_manifest['gpu_used'] is False and license_manifest['compiled'] is False)
    check('tvm_license_export_same_image', license_environment['image_id'] == environment['image_id'])
    native = literal_assignments(patch / 'native_bf16_ar_guard.py', ('SOURCE_SHA', 'BINARY_SHA'))
    check('native_ar_vendor_python_matches_guard', native['SOURCE_SHA'] == sha(
        patch / 'source/vendor-reference/sglang/srt/distributed/device_communicators/custom_all_reduce.py'))
    binary = json.loads((patch / 'source/vendor-reference/sgl_kernel/binary-identity.json').read_text())
    check('native_ar_vendor_binary_identity_matches_guard', native['BINARY_SHA'] == binary['sha256'])
    check('tvm_cache_same_as_measured_copy', sha(patch / 'runtime/tvm-ffi/libtorch_c_dlpack_addon_torch210-rocm.so')
          == sha(patch / 'libtorch_c_dlpack_addon_torch210-rocm.so'))
    result = {'status': 'PASS', 'scope': 'Offline source/manifest/diff consistency only',
              'checks': len(checks), 'files': len(manifest['files']),
              'gpu_used': False, 'rebuild_executed': False,
              'remaining_source_gaps': manifest['upstream_source_gaps'],
              'runtime_or_accuracy_certification': False}
    if args.json_output:
        if args.json_output.exists():
            raise FileExistsError('Refusing to overwrite report: ' + str(args.json_output))
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'FAIL', 'error': str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
