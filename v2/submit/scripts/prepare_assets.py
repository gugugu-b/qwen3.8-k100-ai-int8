#!/usr/bin/env python3
"""Copy the exact measured library on the server; no network or GPU operations."""
import argparse
import shutil
from pathlib import Path
from verify_package import ROOT,verify_library
p=argparse.ArgumentParser()
p.add_argument('--library-source',type=Path,required=True)
a=p.parse_args()
src=a.library_source.resolve(); dst=ROOT/'config/library_gpu4_int8tuned_stock_clean'
count=verify_library(src)
if dst.exists():raise SystemExit(f'Refusing overwrite: {dst}; use LIBRARY_DIR to mount an existing verified library')
shutil.copytree(src,dst,symlinks=False)
verify_library(dst)
print(f'Copied and SHA-verified all {count} library files to {dst}')
