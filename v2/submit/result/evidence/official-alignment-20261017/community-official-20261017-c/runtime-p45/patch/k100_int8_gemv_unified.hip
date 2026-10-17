#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

// High-Performance Unified Multi-M (M in [1, 2, 3, 4]) INT8 GEMV for Qwen3.8-27B on K100-AI (gfx928)
// Covers:
// 1. Column-parallel: K=5120 -> N (qkv_proj N=8192/14336, gate_up_proj N=17408/34816)
// 2. Row-parallel:    K=8704 -> N=5120 (down_proj), K=3072 -> N=5120 (out_proj)

namespace {
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
union WaveBufferResource {
  i32x4 content;
  struct { uint64_t address; uint32_t range; uint32_t config; } fields;
};

__device__ __forceinline__ i32x4 make_res(const void* p, uint32_t bytes) {
  WaveBufferResource r{};
  uint64_t a = (uint64_t)p;
  r.content[0] = __builtin_amdgcn_readfirstlane((uint32_t)a);
  r.content[1] = __builtin_amdgcn_readfirstlane((uint32_t)(a >> 32));
  r.content[2] = __builtin_amdgcn_readfirstlane(bytes);
  r.content[3] = 0x00020000;
  return r.content;
}

__device__ __forceinline__ i32x4 ld4(i32x4 r, uint32_t off) {
  i32x4 v;
  asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen;\n\t"
               : "=v"(v) : "v"(off), "s"(r) : "memory");
  return v;
}

#define WV(N) __device__ __forceinline__ void wait##N(){asm volatile("s_waitcnt vmcnt(" #N ")":::"memory");}
WV(0) WV(1) WV(2) WV(3) WV(4)
#undef WV

__device__ __forceinline__ int dot4x4(i32x4 x, i32x4 w, int a) {
  a = __builtin_amdgcn_sdot4((int)x[0], (int)w[0], a, false);
  a = __builtin_amdgcn_sdot4((int)x[1], (int)w[1], a, false);
  a = __builtin_amdgcn_sdot4((int)x[2], (int)w[2], a, false);
  a = __builtin_amdgcn_sdot4((int)x[3], (int)w[3], a, false);
  return a;
}

// ---------------------------------------------------------------------------
// 1. Column-parallel K=5120 Kernel
// ---------------------------------------------------------------------------
template<int M_BATCH, int WAVES>
__global__ __launch_bounds__(WAVES * 64) void gemv_k5120_multi_m_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ w,
    const float* __restrict__ xs,
    const float* __restrict__ ws,
    hip_bfloat16* __restrict__ out,
    int nout) {
  constexpr int K = 5120;
  constexpr int V = K / 16;
  __shared__ __align__(16) int8_t sx[M_BATCH * K];

  const int tid = threadIdx.x;
  auto* sxv = reinterpret_cast<i32x4*>(sx);
  const auto* xgv = reinterpret_cast<const i32x4*>(x);

  constexpr int TOTAL_V = M_BATCH * V;
#pragma unroll 2
  for (int i = tid; i < TOTAL_V; i += WAVES * 64) {
    sxv[i] = xgv[i];
  }
  __syncthreads();

  const int lane = tid & 63;
  const int wave = tid >> 6;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= nout) return;

  const i32x4 wr = make_res(w + (int64_t)n * K, K);
  auto off = [&](int s) -> uint32_t { return uint32_t((lane + s * 64) * 16); };

  i32x4 wv[5];
#pragma unroll
  for (int s = 0; s < 5; ++s) wv[s] = ld4(wr, off(s));

  int a0[M_BATCH] = {0};
  int a1[M_BATCH] = {0};

  wait4();
#pragma unroll
  for (int m = 0; m < M_BATCH; ++m) {
    i32x4 xm = sxv[m * V + lane + 0 * 64];
    a0[m] = dot4x4(xm, wv[0], a0[m]);
  }

  wait3();
#pragma unroll
  for (int m = 0; m < M_BATCH; ++m) {
    i32x4 xm = sxv[m * V + lane + 1 * 64];
    a1[m] = dot4x4(xm, wv[1], a1[m]);
  }

  wait2();
#pragma unroll
  for (int m = 0; m < M_BATCH; ++m) {
    i32x4 xm = sxv[m * V + lane + 2 * 64];
    a0[m] = dot4x4(xm, wv[2], a0[m]);
  }

  wait1();
#pragma unroll
  for (int m = 0; m < M_BATCH; ++m) {
    i32x4 xm = sxv[m * V + lane + 3 * 64];
    a1[m] = dot4x4(xm, wv[3], a1[m]);
  }

  wait0();
#pragma unroll
  for (int m = 0; m < M_BATCH; ++m) {
    i32x4 xm = sxv[m * V + lane + 4 * 64];
    a0[m] = dot4x4(xm, wv[4], a0[m]);
  }

  const float w_scale = ws[n];
#pragma unroll
  for (int m = 0; m < M_BATCH; ++m) {
    int acc = a0[m] + a1[m];
#pragma unroll
    for (int sh = 32; sh > 0; sh >>= 1) acc += __shfl_down(acc, sh, 64);
    if (lane == 0) {
      out[m * nout + n] = hip_bfloat16((float)acc * xs[m] * w_scale);
    }
  }
}

// ---------------------------------------------------------------------------
// 2. Row-parallel K=8704 & K=3072 Kernel
// ---------------------------------------------------------------------------
template<int K, int M_BATCH, int WAVES>
__global__ __launch_bounds__(WAVES * 64) void gemv_row_multi_m_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ w,
    const float* __restrict__ xs,
    const float* __restrict__ ws,
    hip_bfloat16* __restrict__ out,
    int nout) {
  static_assert(K == 8704 || K == 3072);
  constexpr int V = K / 16;
  __shared__ __align__(16) int8_t sx[M_BATCH * K];

  const int tid = threadIdx.x;
  auto* sxv = reinterpret_cast<i32x4*>(sx);
  const auto* xgv = reinterpret_cast<const i32x4*>(x);

  constexpr int TOTAL_V = M_BATCH * V;
#pragma unroll 2
  for (int i = tid; i < TOTAL_V; i += WAVES * 64) {
    sxv[i] = xgv[i];
  }
  __syncthreads();

  const int lane = tid & 63;
  const int wave = tid >> 6;
  const int n = blockIdx.x * WAVES + wave;
  if (n >= nout) return;

  const i32x4 wr = make_res(w + (int64_t)n * K, K);
  i32x4 z{};
  int a0[M_BATCH] = {0};
  int a1[M_BATCH] = {0};

  if constexpr (K == 8704) {
    // 8704 / 16 = 544 vectors = 8 full stripes (64) + 32 lanes.
    i32x4 w0 = ld4(wr, uint32_t((lane + 0 * 64) * 16));
    i32x4 w1 = ld4(wr, uint32_t((lane + 1 * 64) * 16));
    i32x4 w2 = ld4(wr, uint32_t((lane + 2 * 64) * 16));
    i32x4 w3 = ld4(wr, uint32_t((lane + 3 * 64) * 16));

    wait3();
#pragma unroll
    for (int m = 0; m < M_BATCH; ++m) a0[m] = dot4x4(sxv[m * V + lane + 0 * 64], w0, a0[m]);

    wait2();
#pragma unroll
    for (int m = 0; m < M_BATCH; ++m) a1[m] = dot4x4(sxv[m * V + lane + 1 * 64], w1, a1[m]);

    wait1();
#pragma unroll
    for (int m = 0; m < M_BATCH; ++m) a0[m] = dot4x4(sxv[m * V + lane + 2 * 64], w2, a0[m]);

    wait0();
#pragma unroll
    for (int m = 0; m < M_BATCH; ++m) a1[m] = dot4x4(sxv[m * V + lane + 3 * 64], w3, a1[m]);

    i32x4 w4 = ld4(wr, uint32_t((lane + 4 * 64) * 16));
    i32x4 w5 = ld4(wr, uint32_t((lane + 5 * 64) * 16));
    i32x4 w6 = ld4(wr, uint32_t((lane + 6 * 64) * 16));
    i32x4 w7 = ld4(wr, uint32_t((lane + 7 * 64) * 16));

    wait3();
#pragma unroll
    for (int m = 0; m < M_BATCH; ++m) a0[m] = dot4x4(sxv[m * V + lane + 4 * 64], w4, a0[m]);

    wait2();
#pragma unroll
    for (int m = 0; m < M_BATCH; ++m) a1[m] = dot4x4(sxv[m * V + lane + 5 * 64], w5, a1[m]);

    wait1();
#pragma unroll
    for (int m = 0; m < M_BATCH; ++m) a0[m] = dot4x4(sxv[m * V + lane + 6 * 64], w6, a0[m]);

    wait0();
#pragma unroll
    for (int m = 0; m < M_BATCH; ++m) a1[m] = dot4x4(sxv[m * V + lane + 7 * 64], w7, a1[m]);

    i32x4 w8 = z;
    if (lane < 32) w8 = ld4(wr, uint32_t((lane + 8 * 64) * 16));
    wait0();
    if (lane < 32) {
#pragma unroll
      for (int m = 0; m < M_BATCH; ++m) a0[m] = dot4x4(sxv[m * V + lane + 8 * 64], w8, a0[m]);
    }
  } else {
    // 3072 / 16 = 192 vectors = exactly 3 full stripes of 64 lanes.
    i32x4 w0 = ld4(wr, uint32_t((lane + 0 * 64) * 16));
    i32x4 w1 = ld4(wr, uint32_t((lane + 1 * 64) * 16));
    i32x4 w2 = ld4(wr, uint32_t((lane + 2 * 64) * 16));

    wait2();
#pragma unroll
    for (int m = 0; m < M_BATCH; ++m) a0[m] = dot4x4(sxv[m * V + lane + 0 * 64], w0, a0[m]);

    wait1();
#pragma unroll
    for (int m = 0; m < M_BATCH; ++m) a1[m] = dot4x4(sxv[m * V + lane + 1 * 64], w1, a1[m]);

    wait0();
#pragma unroll
    for (int m = 0; m < M_BATCH; ++m) a0[m] = dot4x4(sxv[m * V + lane + 2 * 64], w2, a0[m]);
  }

  const float w_scale = ws[n];
#pragma unroll
  for (int m = 0; m < M_BATCH; ++m) {
    int acc = a0[m] + a1[m];
#pragma unroll
    for (int sh = 32; sh > 0; sh >>= 1) acc += __shfl_down(acc, sh, 64);
    if (lane == 0) {
      out[m * nout + n] = hip_bfloat16((float)acc * xs[m] * w_scale);
    }
  }
}

// ---------------------------------------------------------------------------
// Dispatchers
// ---------------------------------------------------------------------------
template<int M_BATCH, int WAVES>
void launch_k5120(const int8_t* x, const int8_t* w, const float* xs, const float* ws,
                  hip_bfloat16* out, int n, hipStream_t s) {
  dim3 b(WAVES * 64), g((n + WAVES - 1) / WAVES);
  hipLaunchKernelGGL((gemv_k5120_multi_m_kernel<M_BATCH, WAVES>), g, b, 0, s, x, w, xs, ws, out, n);
}

template<int K, int M_BATCH, int WAVES>
void launch_row(const int8_t* x, const int8_t* w, const float* xs, const float* ws,
                hip_bfloat16* out, int n, hipStream_t s) {
  dim3 b(WAVES * 64), g((n + WAVES - 1) / WAVES);
  hipLaunchKernelGGL((gemv_row_multi_m_kernel<K, M_BATCH, WAVES>), g, b, 0, s, x, w, xs, ws, out, n);
}

} // namespace

torch::Tensor gemv_k5120(torch::Tensor x, torch::Tensor w, torch::Tensor xs,
                         torch::Tensor ws, int64_t waves) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "cuda tensors required");
  TORCH_CHECK(x.scalar_type() == torch::kInt8 && w.scalar_type() == torch::kInt8, "int8 required");
  int m = x.size(0);
  int k = x.size(1);
  int n = w.size(0);
  TORCH_CHECK(k == 5120 && w.size(1) == 5120, "K must be 5120");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "contiguous required");
  TORCH_CHECK(xs.scalar_type() == torch::kFloat32 && ws.scalar_type() == torch::kFloat32, "scale f32 required");

  auto o = torch::empty({m, n}, x.options().dtype(torch::kBFloat16));
  auto s = c10::hip::getCurrentHIPStream();

  const int8_t* xp = x.data_ptr<int8_t>();
  const int8_t* wp = w.data_ptr<int8_t>();
  const float* xsp = xs.data_ptr<float>();
  const float* wsp = ws.data_ptr<float>();
  auto* op = (hip_bfloat16*)o.data_ptr<at::BFloat16>();

  if (m == 1) {
    if (waves == 2) launch_k5120<1, 2>(xp, wp, xsp, wsp, op, n, s);
    else launch_k5120<1, 4>(xp, wp, xsp, wsp, op, n, s);
  } else if (m == 2) {
    if (waves == 2) launch_k5120<2, 2>(xp, wp, xsp, wsp, op, n, s);
    else launch_k5120<2, 4>(xp, wp, xsp, wsp, op, n, s);
  } else if (m == 3) {
    if (waves == 2) launch_k5120<3, 2>(xp, wp, xsp, wsp, op, n, s);
    else launch_k5120<3, 4>(xp, wp, xsp, wsp, op, n, s);
  } else if (m == 4) {
    if (waves == 2) launch_k5120<4, 2>(xp, wp, xsp, wsp, op, n, s);
    else launch_k5120<4, 4>(xp, wp, xsp, wsp, op, n, s);
  } else {
    TORCH_CHECK(false, "Unsupported M batch for gemv_k5120: only M=1, 2, 3, 4 supported");
  }
  return o;
}

torch::Tensor gemv_row(torch::Tensor x, torch::Tensor w, torch::Tensor xs,
                       torch::Tensor ws, int64_t waves) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "cuda tensors required");
  TORCH_CHECK(x.scalar_type() == torch::kInt8 && w.scalar_type() == torch::kInt8, "int8 required");
  int m = x.size(0);
  int k = x.size(1);
  int n = w.size(0);
  TORCH_CHECK((k == 8704 || k == 3072) && w.size(1) == k, "K must be 8704 or 3072");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "contiguous required");
  TORCH_CHECK(xs.scalar_type() == torch::kFloat32 && ws.scalar_type() == torch::kFloat32, "scale f32 required");

  auto o = torch::empty({m, n}, x.options().dtype(torch::kBFloat16));
  auto s = c10::hip::getCurrentHIPStream();

  const int8_t* xp = x.data_ptr<int8_t>();
  const int8_t* wp = w.data_ptr<int8_t>();
  const float* xsp = xs.data_ptr<float>();
  const float* wsp = ws.data_ptr<float>();
  auto* op = (hip_bfloat16*)o.data_ptr<at::BFloat16>();

  if (k == 8704) {
    if (m == 1) {
      if (waves == 2) launch_row<8704, 1, 2>(xp, wp, xsp, wsp, op, n, s);
      else launch_row<8704, 1, 4>(xp, wp, xsp, wsp, op, n, s);
    } else if (m == 2) {
      if (waves == 2) launch_row<8704, 2, 2>(xp, wp, xsp, wsp, op, n, s);
      else launch_row<8704, 2, 4>(xp, wp, xsp, wsp, op, n, s);
    } else if (m == 3) {
      if (waves == 2) launch_row<8704, 3, 2>(xp, wp, xsp, wsp, op, n, s);
      else launch_row<8704, 3, 4>(xp, wp, xsp, wsp, op, n, s);
    } else if (m == 4) {
      if (waves == 2) launch_row<8704, 4, 2>(xp, wp, xsp, wsp, op, n, s);
      else launch_row<8704, 4, 4>(xp, wp, xsp, wsp, op, n, s);
    } else {
      TORCH_CHECK(false, "Unsupported M batch for gemv_row K=8704: only M=1, 2, 3, 4 supported");
    }
  } else {
    if (m == 1) {
      if (waves == 2) launch_row<3072, 1, 2>(xp, wp, xsp, wsp, op, n, s);
      else launch_row<3072, 1, 4>(xp, wp, xsp, wsp, op, n, s);
    } else if (m == 2) {
      if (waves == 2) launch_row<3072, 2, 2>(xp, wp, xsp, wsp, op, n, s);
      else launch_row<3072, 2, 4>(xp, wp, xsp, wsp, op, n, s);
    } else if (m == 3) {
      if (waves == 2) launch_row<3072, 3, 2>(xp, wp, xsp, wsp, op, n, s);
      else launch_row<3072, 3, 4>(xp, wp, xsp, wsp, op, n, s);
    } else if (m == 4) {
      if (waves == 2) launch_row<3072, 4, 2>(xp, wp, xsp, wsp, op, n, s);
      else launch_row<3072, 4, 4>(xp, wp, xsp, wsp, op, n, s);
    } else {
      TORCH_CHECK(false, "Unsupported M batch for gemv_row K=3072: only M=1, 2, 3, 4 supported");
    }
  }
  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv_k5120", &gemv_k5120, "K100-AI Unified Multi-M K5120 INT8 GEMV",
        py::arg("x"), py::arg("w"), py::arg("xs"), py::arg("ws"), py::arg("waves") = 4);
  m.def("gemv_row", &gemv_row, "K100-AI Unified Multi-M Row (K8704/K3072) INT8 GEMV",
        py::arg("x"), py::arg("w"), py::arg("xs"), py::arg("ws"), py::arg("waves") = 4);
}
