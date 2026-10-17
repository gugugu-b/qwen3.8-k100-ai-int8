#!/usr/bin/env python3
"""Build script for unified multi-M INT8 GEMV extension on K100-AI."""

import os
from pathlib import Path
from torch.utils.cpp_extension import load

os.environ["CXX"] = "hipcc"
current_dir = Path(__file__).parent.resolve()
hip_file = str(current_dir / "k100_int8_gemv_unified.hip")
build_dir = "/tmp/q38_build_unified_gemv"
os.makedirs(build_dir, exist_ok=True)

print(f"Compiling {hip_file} into {build_dir}...")
mod = load(
    name="k100_int8_gemv_unified",
    sources=[hip_file],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx928"],
    build_directory=build_dir,
    verbose=True,
)
print("Build successful!")

# Also copy to permanent target location
target_dir = "/data/wangzy/native_ext"
os.makedirs(target_dir, exist_ok=True)
so_files = list(Path(build_dir).glob("*.so"))
if so_files:
    import shutil
    target_so = f"{target_dir}/k100_int8_gemv_unified.so"
    shutil.copyfile(str(so_files[0]), target_so)
    print(f"Copied {so_files[0]} -> {target_so}")
