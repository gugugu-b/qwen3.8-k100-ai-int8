#!/usr/bin/env python3
"""Guarded installed BF16 copy-in TP2 reduction gate; no model or timing claim."""
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def save(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    rank = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 2 and rank in (0, 1)
    assert os.environ['HIP_VISIBLE_DEVICES'] == '4,5'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '4,5'
    assert os.environ['SGLANG_USE_1STAGE_ALLREDUCE'] == '1'
    assert torch.cuda.device_count() == 2
    torch.cuda.set_device(rank)
    output = Path(os.environ['AR_PROBE_OUTPUT']) / f'rank-{rank}'
    output.mkdir(exist_ok=False)
    dist.init_process_group('gloo', timeout=timedelta(seconds=90))
    rccl = dist.new_group([0, 1], backend='nccl', timeout=timedelta(seconds=90))
    from sglang.srt.distributed.device_communicators import custom_all_reduce as source
    from sglang.srt.distributed.device_communicators import custom_all_reduce_ops as ops

    # Use the installed public dispatch, without changing topology detection.
    from native_bf16_ar_guard import install
    install()
    cls = source.dispatch_custom_allreduce('native')
    assert cls is source.CustomAllreduce
    ca = cls(dist.group.WORLD, torch.device('cuda', rank))
    binary = Path('/usr/local/lib/python3.10/dist-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so')
    save(output / 'environment.json', dict(rank=rank, physical_device=rank+4, tp=2,
        backend='native', deterministic_one_stage=ca.use_amd_deterministic_impl,
        disabled=ca.disabled, full_nvlink=ca.full_nvlink,
        device_properties=str(torch.cuda.get_device_properties(rank)),
        torch_version=torch.__version__, hip_version=torch.version.hip,
        source_file=source.__file__,
        source_sha256=hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        max_size=ca.max_size, topology_override=False, quantization=False,
        formal_acceptance=False, actual_model_graph_gate='REVIEW_REQUIRED'))
    assert not ca.disabled and ca.use_amd_deterministic_impl
    assert ca.full_nvlink is False, 'Retain HCU topology detection result'
    rows = []
    for dtype in (torch.bfloat16,):
        for m in (1, 2, 4, 8, 16, 32, 64, 256, 1024):
            tag = f'{str(dtype).split(".")[-1]}-m{m}'
            inp = torch.empty((m, 5120), device='cuda', dtype=dtype)
            assert ca.should_custom_ar(inp)

            def fill(seed, mode):
                gen = torch.Generator().manual_seed(seed)
                a = torch.randn((m, 5120), generator=gen).to(dtype)
                b = torch.randn((m, 5120), generator=gen).to(dtype)
                if mode == 'zero':
                    a.zero_(); b.zero_()
                elif mode == 'cancellation':
                    a.mul_(256); b.copy_(-a); b[:, 1::2].mul_(0.5)
                inp.copy_((a if rank == 0 else b).to(inp.device))
                expected = (a.float()+b.float()).to(dtype).to(inp.device)
                return a, b, expected

            def check(actual, reference, expected, a, b, phase, index):
                actual_cpu = actual.detach().cpu()
                reference_cpu = reference.detach().cpu()
                expected_cpu = expected.detach().cpu()
                finite = torch.isfinite(actual_cpu)
                delta = (actual_cpu.float()-reference_cpu.float()).abs()
                equal = torch.equal(actual_cpu, reference_cpu) and torch.equal(actual_cpu, expected_cpu)
                value = dict(phase=phase, index=index, equal=bool(equal),
                    elements=inp.numel(), finite_count=int(finite.sum()),
                    nonfinite_count=int((~finite).sum()),
                    mismatch_vs_rccl=int((actual_cpu != reference_cpu).sum()),
                    mismatch_vs_explicit_sum=int((actual_cpu != expected_cpu).sum()),
                    max_abs=float(delta.max()) if bool(torch.isfinite(delta).all()) else None)
                if not equal:
                    tensor_path = output / f'{tag}-{phase}-{index}-failure-tensors.pt'
                    assert not tensor_path.exists()
                    torch.save(dict(input_rank0=a, input_rank1=b,
                        actual=actual_cpu, rccl_reference=reference_cpu,
                        explicit_reference=expected_cpu, rank=rank,
                        dtype=str(dtype), m=m, phase=phase, index=index), tensor_path)
                    value['failure_tensor_sha256'] = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
                save(output / f'{tag}-{phase}-{index}.json', value)
                assert equal, ('native one-stage mismatch', rank, dtype, m, phase, index)
                return value

            eager = []
            poisoned = []
            for i, mode in enumerate(('random', 'zero', 'cancellation')):
                a, b, expected = fill(20260912+i, mode)
                reference = inp.clone()
                dist.all_reduce(reference, group=rccl)
                actual = torch.full_like(inp, float('nan'))
                # Exact installed _all_reduce_impl dispatch, with explicit NaN
                # output to prove full writes rather than allocator contents.
                if inp.numel()*inp.element_size() < ca.max_size:
                    reg_buffer = ca.buffer.view(dtype)[:inp.numel()]
                    ops.deterministic_all_reduce_unreg(ca._ptr, inp, reg_buffer, actual)
                else:
                    ca.register_buffer(inp)
                    ops.deterministic_all_reduce_reg(ca._ptr, inp, actual)
                torch.cuda.synchronize()
                poisoned.append(check(actual, reference, expected, a, b, 'poisoned', i))
                actual_public = ca.custom_all_reduce(inp)
                assert actual_public is not None
                torch.cuda.synchronize()
                eager.append(check(actual_public, reference, expected, a, b, 'public-eager', i))
            fill(20260912, 'random')
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            dist.barrier()
            with ca.capture():
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        ca.custom_all_reduce(inp)
                stream.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    value = ca.custom_all_reduce(inp)
            torch.cuda.current_stream().wait_stream(stream)
            graph_checks = []
            for i, mode in enumerate(('random', 'zero', 'cancellation')):
                a, b, expected = fill(20261010+i, mode)
                reference = inp.clone()
                dist.all_reduce(reference, group=rccl)
                value.fill_(float('nan'))
                torch.cuda.synchronize()
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize()
                graph_checks.append(check(value, reference, expected, a, b, 'changed-graph', i))
            row = dict(status='PASS', dtype=str(dtype), m=m, hidden=5120,
                bytes=inp.numel()*inp.element_size(),
                input_path='copy-in' if inp.numel()*inp.element_size() < ca.max_size else 'registered',
                poisoned_full_write_checks=poisoned, public_eager_checks=eager,
                changed_input_graph_checks=graph_checks)
            save(output / f'{tag}.json', row)
            rows.append(row)
            print(json.dumps(dict(rank=rank, dtype=str(dtype), m=m, status='PASS')), flush=True)
            del graph, value, inp, reference, expected, actual, actual_public
        assert not ca.should_custom_ar(torch.empty(3, device='cuda', dtype=dtype))
        assert not ca.should_custom_ar(torch.empty((8, 16), device='cuda', dtype=dtype)[:, ::2])
    # Explicit route exclusions: caller keeps its unchanged RCCL fallback.
    excluded = []
    for m in (1, 2, 4, 8, 16, 32, 64, 256, 1024):
        value = torch.zeros((m, 5120), device='cuda', dtype=torch.float32)
        assert not ca.should_custom_ar(value) and ca.custom_all_reduce(value) is None
        excluded.append(dict(dtype='torch.float32', m=m, bytes=value.numel()*value.element_size(), native_called=False))
    value = torch.zeros((2048, 5120), device='cuda', dtype=torch.bfloat16)
    assert value.numel()*value.element_size() >= ca.max_size
    assert not ca.should_custom_ar(value) and ca.custom_all_reduce(value) is None
    excluded.append(dict(dtype='torch.bfloat16', m=2048, bytes=value.numel()*value.element_size(), native_called=False))
    save(output/'excluded-routes.json', dict(status='ROUTE_EXCLUSIONS_PASS', rows=excluded,
        fallback='Unchanged caller RCCL/PyNccl', fallback_graph_retested=False))
    dist.barrier()
    save(output / 'summary.json', dict(status='NUMERICAL_GRAPH_PASS', rank=rank,
        physical_device=rank+4, tp=2, backend='native', deterministic_one_stage=True,
        cases=len(rows), excluded_routes=len(excluded), poisoned_full_write_checks=3*len(rows),
        public_eager_checks=3*len(rows), changed_graph_checks=3*len(rows), rows=rows,
        full_nvlink=ca.full_nvlink, topology_override=False, quantization=False,
        formal_acceptance=False, actual_model_graph_gate='REVIEW_REQUIRED'))
    ca.close()
    dist.destroy_process_group(rccl)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
