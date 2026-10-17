#!/usr/bin/env python3
"""Rebuild from supplied HIP source into a NEW directory in the vendor image."""
import argparse
import os
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
os.environ['CXX']='/opt/dtk/bin/hipcc'
from torch.utils.cpp_extension import load
root=Path(__file__).resolve().parents[1]
load(name='k100_int8_gemv_unified',sources=[str(root/'patch/k100_int8_gemv_unified.hip')],
     extra_cuda_cflags=['-O3','--offload-arch=gfx928'],build_directory=str(a.output.resolve()),verbose=True)
print('Rebuilt artifact requires numerical/Graph/performance revalidation; measured binary was not overwritten.')
