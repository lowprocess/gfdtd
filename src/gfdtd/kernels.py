"""
The GPU kernels (CUDA C; CuPy compiles the same source for ROCm through hiprtc). The update maths is in
core/fdtd_core.h, shared with the CPU core; here are the kernels around it: one thread per node in a grid-stride
loop for the H and E updates (on the compressed operator: codes, tables and line arrays, no coefficient arrays),
the energy as a block reduction, and the sparse kernels on flat indices into the (3, Nx, Ny, Nz) arrays.

    source('unsigned short')        # the module's source, for uint16 or uint32 (unsigned int) codes
"""
import os

NAMES = ('update_h', 'update_e', 'energy', 'mur_pre', 'mur_post', 'excite', 'assign', 'par_l', 'series_rlc', 'probe',
         'pml_e', 'pml_h', 'step_h', 'step_e', 'step_post', 'dump')
SLAB = 6                # the PML kernels' slab table: start, m, axis, s0, cnt, offset of b / c, per slab

HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core', 'fdtd_core.h')

KERNELS = r'''
#define GF_LOOP(m) for (idx_t t = blockIdx.x * (idx_t)blockDim.x + threadIdx.x; t < (m); t += (idx_t)gridDim.x * blockDim.x)
// the node of t, in 32-bit arithmetic (the backend keeps 3 nx ny nz below 2^31): 64-bit division is emulated
#define GF_IJK const unsigned u_ = (unsigned)t, r_ = u_ / (unsigned)D.nz; const int k = (int)(u_ - r_ * (unsigned)D.nz); \
    const int i = (int)(r_ / (unsigned)D.ny), j = (int)(r_ - (unsigned)i * (unsigned)D.ny);

extern "C" __global__ void update_h(const float* __restrict__ V, float* __restrict__ I, const float* __restrict__ G,
                                    int nx, int ny, int nz, float kh)
{
    const Dims D = {nx, ny, nz};
    GF_LOOP((idx_t)D.nx * D.ny * D.nz) { GF_IJK gf_h_node(i, j, k, t, V, I, G, D, kh); }
}

extern "C" __global__ void update_e(float* __restrict__ V, const float* __restrict__ I, const CODE_T* __restrict__ code,
                                    const float* tvv, const float* ts, const int* tpole, const float* tkb,
                                    const float* tka, const float* th, int P, const int* __restrict__ slot,
                                    float* __restrict__ pol, const float* __restrict__ G, int nx, int ny, int nz, float ke, float dt)
{
    const Dims D = {nx, ny, nz};
    GF_LOOP((idx_t)D.nx * D.ny * D.nz) {
        GF_IJK gf_e_node<CODE_T>(i, j, k, t, V, I, code, tvv, ts, tpole, tkb, tka, th, P, slot, pol, G, D, ke, dt);
    }
}

extern "C" __global__ void energy(const float* V, const float* I, const CODE_T* code, const float* ts, const float* G,
                                  int nx, int ny, int nz, float ke, float kh, double* out)
{
    const Dims D = {nx, ny, nz};
    __shared__ double part[256];
    double e = 0.0;
    GF_LOOP((idx_t)D.nx * D.ny * D.nz) { GF_IJK e += gf_energy_node<CODE_T>(i, j, k, t, V, I, code, ts, G, D, ke, kh); }
    part[threadIdx.x] = e;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, part[0]);
}

// all PML slabs in one launch: the slab table desc (SLAB values each) maps a thread's index g to its slab;
// psi (2 m per slab, from 2 start), k1 / k2 (m per slab, from start) and b / c (cnt per slab) concatenated
#define GF_SLAB                                                                                         \
    int q = 0;                                                                                          \
    while (q < ns - 1 && g >= desc[(q + 1) * 6]) ++q;                                                   \
    const idx_t* d = desc + q * 6;                                                                      \
    const idx_t st = d[0], m = d[1], e = g - st;                                                        \
    const int a = (int)d[2], s0 = (int)d[3], cnt = (int)d[4];                                           \
    const float *bq = b + d[5], *cq = c + d[5];

extern "C" __global__ void pml_e(float* V, const float* I, float* psi, const float* b, const float* c, const float* k1,
                                 const float* k2, int nx, int ny, int nz, const idx_t* desc, int ns, idx_t M)
{
    const Dims D = {nx, ny, nz};
    for (idx_t g = blockIdx.x * (idx_t)blockDim.x + threadIdx.x; g < M; g += (idx_t)gridDim.x * blockDim.x) {
        GF_SLAB
        gf_pml_e(e, m, V, I, psi + 2 * st, bq, cq, k1 + st, k2 + st, D, a, s0, cnt);
    }
}

extern "C" __global__ void pml_h(const float* V, float* I, float* psi, const float* b, const float* c, const float* k1,
                                 const float* k2, int nx, int ny, int nz, const idx_t* desc, int ns, idx_t M)
{
    const Dims D = {nx, ny, nz};
    for (idx_t g = blockIdx.x * (idx_t)blockDim.x + threadIdx.x; g < M; g += (idx_t)gridDim.x * blockDim.x) {
        GF_SLAB
        gf_pml_h(e, m, V, I, psi + 2 * st, bq, cq, k1 + st, k2 + st, D, a, s0, cnt);
    }
}

extern "C" __global__ void mur_pre(const float* V, const idx_t* i0, const idx_t* i1, float* o0, float* o1, idx_t m)
{
    idx_t e = blockIdx.x * (idx_t)blockDim.x + threadIdx.x;
    if (e < m) { o0[e] = V[i0[e]]; o1[e] = V[i1[e]]; }
}

extern "C" __global__ void mur_post(float* V, const idx_t* i0, const idx_t* i1, const float* k,
                                    const float* o0, const float* o1, idx_t m)
{
    idx_t e = blockIdx.x * (idx_t)blockDim.x + threadIdx.x;
    if (e < m) V[i0[e]] = o1[e] + k[e] * (V[i1[e]] - o0[e]);
}

extern "C" __global__ void excite(float* V, const idx_t* idx, const float* amp, float s, idx_t m)
{
    idx_t e = blockIdx.x * (idx_t)blockDim.x + threadIdx.x;
    if (e < m) atomicAdd(&V[idx[e]], amp[e] * s);
}

extern "C" __global__ void assign(float* V, const idx_t* idx, const float* amp, float s, idx_t m)
{
    idx_t e = blockIdx.x * (idx_t)blockDim.x + threadIdx.x;
    if (e < m) V[idx[e]] = amp[e] * s;
}

extern "C" __global__ void par_l(float* V, const float* vi, const idx_t* idx, const float* k, float* IL, idx_t m)
{
    idx_t e = blockIdx.x * (idx_t)blockDim.x + threadIdx.x;
    if (e < m) {
        const idx_t q = idx[e];
        const float v = V[q] - vi[e] * IL[e];
        V[q] = v;
        IL[e] += k[e] * v;
    }
}

extern "C" __global__ void series_rlc(float* V, const float* vi, const idx_t* idx, const float* a, const float* b,
                                      const float* c, float* Ib, float* q, float dt, idx_t m)
{
    idx_t e = blockIdx.x * (idx_t)blockDim.x + threadIdx.x;
    if (e < m) {
        const idx_t p = idx[e];
        const float i = Ib[e];
        const float v = V[p] - vi[e] * i;
        V[p] = v;
        const float qq = q[e] + dt * i;
        q[e] = qq;
        Ib[e] = a[e] * i + b[e] * v - c[e] * qq;
    }
}

extern "C" __global__ void probe(const float* F, const idx_t* idx, const float* w, const idx_t* pid, float* row,
                                 idx_t m)
{
    idx_t e = blockIdx.x * (idx_t)blockDim.x + threadIdx.x;
    if (e < m) atomicAdd(&row[pid[e]], w[e] * F[idx[e]]);
}

// ---- one timestep in three launches ----------------------------------------------------------------------------
// step_h: H, its PML terms, and (indices past the nodes) the Mur copies of V before the E update
extern "C" __global__ void step_h(const float* __restrict__ V, float* __restrict__ I, const float* __restrict__ G,
                                  int nx, int ny, int nz, float kh, float* psi, const float* b, const float* c,
                                  const idx_t* desc, int ns, const int* box, const CODE_T* code, const float* ts,
                                  const idx_t* m0, const idx_t* m1, float* o0, float* o1, idx_t mm)
{
    const Dims D = {nx, ny, nz};
    const idx_t n = (idx_t)nx * ny * nz;
    GF_LOOP(n + mm) {
        if (t < n) {
            GF_IJK
            gf_h_node(i, j, k, t, V, I, G, D, kh);
            if (ns)
                gf_pml_node<false, CODE_T>(i, j, k, t, (float*)V, I, psi, b, c, desc, ns, box, code, ts, G, D, kh);
        } else {
            const idx_t e = t - n;
            o0[e] = V[m0[e]];
            o1[e] = V[m1[e]];
        }
    }
}

// step_e: E with the Debye poles, then its PML terms
extern "C" __global__ void step_e(float* __restrict__ V, const float* __restrict__ I, const CODE_T* __restrict__ code,
                                  const float* tvv, const float* ts, const int* tpole, const float* tkb,
                                  const float* tka, const float* th, int P, const int* __restrict__ slot,
                                  float* __restrict__ pol, const float* __restrict__ G, int nx, int ny, int nz,
                                  float ke, float dt, float* psi, const float* b, const float* c, const idx_t* desc,
                                  int ns, const int* box)
{
    const Dims D = {nx, ny, nz};
    GF_LOOP((idx_t)nx * ny * nz) {
        GF_IJK
        gf_e_node<CODE_T>(i, j, k, t, V, I, code, tvv, ts, tpole, tkb, tka, th, P, slot, pol, G, D, ke, dt);
        if (ns)
            gf_pml_node<true, CODE_T>(i, j, k, t, V, (float*)I, psi, b, c, desc, ns, box, code, ts, G, D, ke);
    }
}

// step_post: the excitation, the lumped branches and the Mur update, one index range each (disjoint edges:
// checked when the backend is set up)
extern "C" __global__ void step_post(float* V, const idx_t* ex_idx, const float* ex_amp, float sig, idx_t n_ex,
                                     const float* pl_vi, const idx_t* pl_idx, const float* pl_k, float* pl_I, idx_t n_pl,
                                     const float* se_vi, const idx_t* se_idx, const float* se_a, const float* se_b,
                                     const float* se_c, float* se_I, float* se_q, float dt, idx_t n_se,
                                     const idx_t* m0, const idx_t* m1, const float* mk, const float* o0,
                                     const float* o1, idx_t n_mu)
{
    GF_LOOP(n_ex + n_pl + n_se + n_mu) {
        idx_t e = t;
        if (e < n_ex) { atomicAdd(&V[ex_idx[e]], ex_amp[e] * sig); continue; }
        e -= n_ex;
        if (e < n_pl) {
            const idx_t q = pl_idx[e];
            const float v = V[q] - pl_vi[e] * pl_I[e];
            V[q] = v;
            pl_I[e] += pl_k[e] * v;
            continue;
        }
        e -= n_pl;
        if (e < n_se) {
            const idx_t p = se_idx[e];
            const float i = se_I[e];
            const float v = V[p] - se_vi[e] * i;
            V[p] = v;
            const float qq = se_q[e] + dt * i;
            se_q[e] = qq;
            se_I[e] = se_a[e] * i + se_b[e] * v - se_c[e] * qq;
            continue;
        }
        e -= n_se;
        V[m0[e]] = o1[e] + mk[e] * (V[m1[e]] - o0[e]);
    }
}

// ---- field dumps: one region's points interpolated (dumps.Region) and, in the frequency domain, one DFT term ---
// field 0 E, 1 H, 2 J (rot H); cell: cell centres (fixed weights, zero outside), else nodes (the edges that exist)
__device__ inline float gf_fE(const float* V, const float* G, Dims D, int a, int i, int j, int k)
{
    const int ii[3] = {i, j, k}, nn[3] = {D.nx, D.ny, D.nz};
    for (int b = 0; b < 3; ++b)
        if (ii[b] < 0 || ii[b] > nn[b] - (b == a ? 2 : 1)) return 0.f;
    const idx_t n = (idx_t)D.nx * D.ny * D.nz;
    return V[a * n + ((idx_t)i * D.ny + j) * D.nz + k] * GF_G(G, D, 1, a, ii[a]);
}

__device__ inline float gf_fH(const float* I, const float* G, Dims D, int a, int i, int j, int k)
{
    const int ii[3] = {i, j, k}, nn[3] = {D.nx, D.ny, D.nz};
    for (int b = 0; b < 3; ++b)
        if (ii[b] < 0 || ii[b] > nn[b] - (b == a ? 1 : 2)) return 0.f;
    const idx_t n = (idx_t)D.nx * D.ny * D.nz;
    return I[a * n + ((idx_t)i * D.ny + j) * D.nz + k] * GF_G(G, D, 3, a, ii[a]);
}

__device__ inline float gf_fJ(const float* I, const float* G, Dims D, int a, int i, int j, int k)
{
    const int ii[3] = {i, j, k}, nn[3] = {D.nx, D.ny, D.nz};
    for (int q = 0; q < 3; ++q)
        if (ii[q] < 0 || ii[q] > nn[q] - 1) return 0.f;
    const int b = (a + 1) % 3, c = (a + 2) % 3;
    const idx_t n = (idx_t)D.nx * D.ny * D.nz, t = ((idx_t)i * D.ny + j) * D.nz + k;
    const idx_t st[3] = {(idx_t)D.ny * D.nz, (idx_t)D.nz, 1};
    const float* Ib = I + b * n;
    const float* Ic = I + c * n;
    const float cur = Ic[t] - (ii[b] > 0 ? Ic[t - st[b]] : 0.f) - Ib[t] + (ii[c] > 0 ? Ib[t - st[c]] : 0.f);
    return cur * GF_G(G, D, 3, b, ii[b]) * GF_G(G, D, 3, c, ii[c]);
}

__device__ inline float gf_dump_f(int field, const float* V, const float* I, const float* G, Dims D, int a, int i,
                                  int j, int k)
{
    return field == 0 ? gf_fE(V, G, D, a, i, j, k) : (field == 1 ? gf_fH(I, G, D, a, i, j, k) : gf_fJ(I, G, D, a, i, j, k));
}

// out: nf == 0: the values (np, 3); else acc (nf, np, 3) complex, += value * exp(-j w t)
extern "C" __global__ void dump(const float* V, const float* I, const float* G, int nx, int ny, int nz, int field,
                                int cell, int lx, int ly, int lz, int sx, int sy, int sz, int nf, const float* omega,
                                float tm, float* out)
{
    const Dims D = {nx, ny, nz};
    const idx_t np_ = (idx_t)sx * sy * sz;
    GF_LOOP(np_ * 3) {
        const int a = (int)(t % 3);
        const idx_t p = t / 3;
        const int pk = (int)(p % sz), pj = (int)((p / sz) % sy), pi = (int)(p / ((idx_t)sz * sy));
        const int i = lx + pi, j = ly + pj, k = lz + pk;
        const int nn[3] = {nx, ny, nz};
        float sum = 0.f, cnt = 0.f;
        // the axes it averages along: E / J, cells: the two across a; nodes: a. H: the other way round
        const bool along_a = (field == 1) != (cell == 0);
        int ax[2], na = 0;
        for (int b = 0; b < 3; ++b)
            if ((b == a) == along_a) ax[na++] = b;
        const int nterm = 1 << na;
        for (int q = 0; q < nterm; ++q) {
            int ii[3] = {i, j, k};
            for (int r = 0; r < na; ++r)
                ii[ax[r]] += ((q >> r) & 1) ? (cell ? 1 : 0) : (cell ? 0 : -1);
            const float f = gf_dump_f(field, V, I, G, D, a, ii[0], ii[1], ii[2]);
            sum += f;
            if (!cell) {                                // nodes: count the edges that exist
                bool ok = true;
                for (int b = 0; b < 3; ++b) {
                    const int hi = field == 1 ? (b == a ? nn[b] - 1 : nn[b] - 2) : (b == a ? nn[b] - 2 : nn[b] - 1);
                    if (ii[b] < 0 || ii[b] > hi) ok = false;
                }
                cnt += ok ? 1.f : 0.f;
            }
        }
        const float v = cell ? sum / (float)nterm : sum / fmaxf(cnt, 1.f);
        if (nf == 0) {
            out[p * 3 + a] = v;
        } else {
            for (int f = 0; f < nf; ++f) {
                float sn, cs;
                sincosf(-omega[f] * tm, &sn, &cs);
                const idx_t o = ((f * np_ + p) * 3 + a) * 2;
                out[o] += v * cs;
                out[o + 1] += v * sn;
            }
        }
    }
}
'''


def source(code_t='unsigned short'):
    with open(HEADER) as fh:
        head = fh.read()
    return (f"#define GF_DEV __device__ inline\n#define GF_ADD(p, v) atomicAdd((p), (v))\n#define CODE_T {code_t}\n" +
            head + KERNELS)
