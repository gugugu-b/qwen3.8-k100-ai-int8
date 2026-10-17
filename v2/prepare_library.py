#!/usr/bin/env python3
"""Prepare v2's exact rocBLAS library from the repository's existing v1 archive."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile

V2 = Path(__file__).resolve().parent
ARCHIVE_SHA = '60c1992063a4101ba7c55450304963485abed405e451fb40c1810485f34a9c6e'
DIRECTORY = 'library_gpu4_int8tuned_stock_clean'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path,
                   default=V2.parent / 'v1/submit1005/config' / (DIRECTORY + '.tar.gz'))
    a = p.parse_args()
    config = V2 / 'submit/config'
    expected = json.loads((config / 'library-sha256.json').read_text())
    output = config / DIRECTORY
    if output.exists():
        actual = {str(f.relative_to(output)) for f in output.rglob('*') if f.is_file()}
        assert actual == set(expected), 'Existing library file coverage differs'
        for rel, digest in expected.items():
            f = output / rel
            assert not f.is_symlink() and sha(f) == digest, rel
        print(json.dumps({'status': 'LIBRARY_SHA256_PASS', 'files': len(expected),
                          'action': 'EXISTING_DIRECTORY_VERIFIED', 'directory': str(output)}))
        return
    assert a.archive.is_file(), 'Missing archive; clone the repository including v1/'
    assert sha(a.archive) == ARCHIVE_SHA, 'Archive SHA256 differs'
    with tempfile.TemporaryDirectory(prefix='.library-stage-', dir=config) as temp:
        stage = Path(temp) / DIRECTORY
        stage.mkdir()
        seen = set()
        total = 0
        ignored = 0
        with tarfile.open(a.archive, 'r:gz') as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                assert not PurePosixPath(member.name).is_absolute() and '..' not in parts
                if any(part in ('.DS_Store', '__MACOSX') or part.startswith('._') for part in parts):
                    ignored += 1
                    continue
                assert parts and parts[0] == DIRECTORY, member.name
                if member.isdir():
                    continue
                assert member.isfile(), 'Only regular library files are accepted'
                rel = PurePosixPath(member.name).relative_to(DIRECTORY).as_posix()
                assert rel in expected and rel not in seen, rel
                target = stage / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                size = 0
                with archive.extractfile(member) as source, target.open('xb') as dest:
                    for block in iter(lambda: source.read(1 << 20), b''):
                        h.update(block)
                        dest.write(block)
                        size += len(block)
                assert size == member.size and h.hexdigest() == expected[rel], rel
                target.chmod(0o644)
                seen.add(rel)
                total += size
        assert seen == set(expected), 'Archive library coverage differs'
        stage.rename(output)
    print(json.dumps({'status': 'LIBRARY_SHA256_PASS', 'files': len(seen),
                      'bytes': total, 'archive_sha256': ARCHIVE_SHA,
                      'ignored_apple_metadata': ignored, 'directory': str(output)}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (AssertionError, OSError, ValueError, tarfile.TarError) as e:
        print('LIBRARY_PREPARATION_FAILED: ' + str(e), file=sys.stderr)
        raise SystemExit(1)
