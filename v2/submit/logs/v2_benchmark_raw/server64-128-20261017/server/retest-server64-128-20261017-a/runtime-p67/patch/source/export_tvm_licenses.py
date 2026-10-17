#!/usr/bin/env python3
"""Read only TVM FFI package copyright and license files from a fixed image."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-packages', type=Path,
                        default=Path('/usr/local/lib/python3.10/dist-packages'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.site_packages
    locations = [root / 'tvm_ffi'] + sorted(root.glob('*tvm*ffi*.dist-info'))
    files = []
    for location in locations:
        if not location.is_dir():
            continue
        for path in sorted(location.rglob('*')):
            name = path.name.lower()
            if (path.is_file() and not path.is_symlink()
                    and (name.startswith(('license', 'notice', 'copying'))
                         or 'license' in name)):
                files.append(path)
    if not files:
        raise SystemExit('No matching TVM FFI licenses were found')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for source in files:
        relative = source.relative_to(root)
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[str(relative)] = {'bytes': target.stat().st_size,
                                  'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                                  'original_path': str(source)}
    result = {'status': 'LICENSES_EXPORTED_WITHOUT_IMPORT_OR_COMPILATION',
              'searched': [str(p) for p in locations], 'files': manifest,
              'gpu_used': False, 'compiled': False}
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
