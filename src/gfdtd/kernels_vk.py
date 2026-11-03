"""
The Vulkan kernels (Slang, compiled to SPIR-V by slangpy): kernels.py's, kernel for kernel, around the same update
maths (core/fdtd_core.h, read by Slang as it is). The differences are Vulkan's:

- the model's arrays are reached through one struct in device memory (Model: their device addresses and the sizes),
  so a launch sets a pointer and a few values, not twenty arguments (each costs host time in slangpy);
- a launch's threads cover its range in a grid-stride loop (`stride`: the threads launched), the node index in 32
  bits as on CUDA (the backend keeps 3 nx ny nz below 2^31);
- the energy: a double per workgroup into part[], their sum on the host (no float64 atomics on RDNA's Vulkan), and
  the lumped branches' energy in part[nblk] by the first thread;
- probe() adds the voltage and the current probes in one launch.

    source('uint16_t')          # the module's source, for uint8_t / uint16_t / uint32_t codes
    MODEL                       # Model's fields in order: (name, kind), kind 'p' a pointer, 'i' int32, 'f' float32
"""


from .kernels import HEADER

NAMES = ('update_h', 'update_e', 'energy', 'step_h', 'step_e', 'step_post', 'pml_e', 'pml_h', 'mur_pre', 'mur_post',
         'excite', 'assign', 'par_l', 'series_rlc', 'probe', 'dump', 'copy_words')

# the struct the kernels read the model through: pointers first (8 bytes each), then 32-bit values, so the layout
# is the same in every Vulkan layout rule (and numpy's, packed)
MODEL = [(n, 'p') for n in (
    'V', 'I', 'G', 'code', 'tvv', 'ts', 'tpole', 'tkb', 'tka', 'th', 'slot', 'pol',
    'psiH', 'bH', 'cH', 'k1H', 'k2H', 'descH', 'psiE', 'bE', 'cE', 'k1E', 'k2E', 'descE', 'boxH', 'boxE',
    'ex_idx', 'ex_amp', 'pl_vi', 'pl_idx', 'pl_k', 'pl_I', 'pl_L',
    'se_vi', 'se_idx', 'se_a', 'se_b', 'se_c', 'se_I', 'se_q', 'se_L', 'se_Cinv',
    'm0', 'm1', 'mk', 'o0', 'o1', 'part', 'buf', 'pv_idx', 'pv_w', 'pv_pid', 'pi_idx', 'pi_w', 'pi_pid')] + \
    [(n, 'i') for n in ('nx', 'ny', 'nz', 'P', 'ns', 'n_ex', 'n_pl', 'n_se', 'n_mu', 'n_pv', 'n_pi', 'np_', 'pml_M',
                         'pad')] + \
    [(n, 'f') for n in ('kh', 'ke', 'dt', 'dtf')]

_TYPES = {'V': 'float', 'I': 'float', 'G': 'float', 'code': 'C', 'tpole': 'int', 'slot': 'int', 'descH': 'idx_t',
          'descE': 'idx_t', 'boxH': 'int', 'boxE': 'int', 'ex_idx': 'idx_t', 'pl_idx': 'idx_t', 'se_idx': 'idx_t',
          'm0': 'idx_t', 'm1': 'idx_t', 'part': 'double', 'pl_L': 'double', 'se_L': 'double', 'se_Cinv': 'double',
          'pv_idx': 'idx_t', 'pv_pid': 'idx_t', 'pi_idx': 'idx_t', 'pi_pid': 'idx_t'}


def _model_struct():
    lines = []
    for n, k in MODEL:
        lines.append(f"    {_TYPES.get(n, 'float')}* {n};" if k == 'p' else f"    {'int' if k == 'i' else 'float'} {n};")
    return "struct Model {\n" + "\n".join(lines) + "\n};\n"


KERNELS = r'''
uniform Model* M;
uniform uint stride;                    // the threads launched: the grid-stride loops' step

// per launch (the kernels that take them)
uniform int ns_;                        // step_h / step_e: PML slabs to run in the kernel (0: split, run apart)
uniform float s_;                       // excite / assign / step_post: the signal's value
uniform int n_ex_;                      // step_post: the excitation's entries (0 past the signal's end)
uniform float* F_;                      // excite / assign: the field (V or I)
uniform idx_t* idx_;
uniform float* amp_;
uniform uint m_;
uniform uint slot_;                     // probe: the buffer row
uniform uint nblk_;                     // energy: the workgroups launched (part[nblk_]: the lumped branches')
// dump: one region's points
uniform int field_, cell_, lx_, ly_, lz_, sx_, sy_, sz_, nf_;
uniform float* omega_;
uniform float tm_;
uniform float* dst_;

#define GF_LOOP(m) for (uint t = tid.x; t < (uint)(m); t += stride)
#define GF_DIMS const Dims D = {M->nx, M->ny, M->nz}; const uint nn_ = (uint)M->nx * (uint)M->ny * (uint)M->nz;
#define GF_IJK const uint r_ = t / (uint)D.nz; const int k = (int)(t - r_ * (uint)D.nz); \
    const int i = (int)(r_ / (uint)D.ny), j = (int)(r_ - (uint)i * (uint)D.ny);

[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void update_h(uint3 tid: SV_DispatchThreadID)
{
    GF_DIMS
    GF_LOOP(nn_) { GF_IJK gf_h_node(i, j, k, t, M->V, M->I, M->G, D, M->kh); }
}

[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void update_e(uint3 tid: SV_DispatchThreadID)
{
    GF_DIMS
    GF_LOOP(nn_) {
        GF_IJK gf_e_node(i, j, k, t, M->V, M->I, M->code, M->tvv, M->ts, M->tpole, M->tkb, M->tka, M->th, M->P, M->slot,
                         M->pol, M->G, D, M->ke, M->dt);
    }
}

groupshared double gf_part[GF_BLOCK];

[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void energy(uint3 tid: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    GF_DIMS
    double e = 0.0;
    GF_LOOP(nn_) { GF_IJK e += gf_energy_node(i, j, k, t, M->V, M->I, M->code, M->ts, M->G, D, M->ke, M->kh); }
    gf_part[lid.x] = e;
    GroupMemoryBarrierWithGroupSync();
    for (uint w = GF_BLOCK / 2; w > 0; w >>= 1) {
        if (lid.x < w) gf_part[lid.x] += gf_part[lid.x + w];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid.x == 0) M->part[gid.x] = gf_part[0];
    if (tid.x == 0) {                   // the lumped branches: L I^2 in the inductors, q^2 / C in the capacitors
        double el = 0.0;
        for (int q = 0; q < M->n_pl; ++q) { const double c = M->pl_I[q]; el += M->pl_L[q] * c * c; }
        for (int q = 0; q < M->n_se; ++q) {
            const double c = M->se_I[q], qq = M->se_q[q];
            el += M->se_L[q] * c * c + M->se_Cinv[q] * qq * qq;
        }
        M->part[nblk_] = el;
    }
}

// one timestep in three launches, as kernels.py's step_h / step_e / step_post
[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void step_h(uint3 tid: SV_DispatchThreadID)
{
    GF_DIMS
    GF_LOOP(nn_ + (uint)M->n_mu) {
        if (t < nn_) {
            GF_IJK
            gf_h_node(i, j, k, t, M->V, M->I, M->G, D, M->kh);
            if (ns_ != 0)
                gf_pml_node(false, i, j, k, t, M->V, M->I, M->psiH, M->bH, M->cH, M->descH, ns_, M->boxH, M->code,
                            M->ts, M->G, D, M->kh);
        } else {
            const uint e = t - nn_;
            M->o0[e] = M->V[M->m0[e]];
            M->o1[e] = M->V[M->m1[e]];
        }
    }
}

[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void step_e(uint3 tid: SV_DispatchThreadID)
{
    GF_DIMS
    GF_LOOP(nn_) {
        GF_IJK
        gf_e_node(i, j, k, t, M->V, M->I, M->code, M->tvv, M->ts, M->tpole, M->tkb, M->tka, M->th, M->P, M->slot,
                  M->pol, M->G, D, M->ke, M->dt);
        if (ns_ != 0)
            gf_pml_node(true, i, j, k, t, M->V, M->I, M->psiE, M->bE, M->cE, M->descE, ns_, M->boxE, M->code, M->ts,
                        M->G, D, M->ke);
    }
}

// the excitation, the lumped branches and the Mur update, one index range each (disjoint edges: the backend checks)
[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void step_post(uint3 tid: SV_DispatchThreadID)
{
    float* V = M->V;
    GF_LOOP(n_ex_ + M->n_pl + M->n_se + M->n_mu) {
        uint e = t;
        if (e < (uint)n_ex_) { GF_ADD(V + M->ex_idx[e], M->ex_amp[e] * s_); continue; }
        e -= (uint)n_ex_;
        if (e < (uint)M->n_pl) {
            const idx_t q = M->pl_idx[e];
            const float v = V[q] - M->pl_vi[e] * M->pl_I[e];
            V[q] = v;
            M->pl_I[e] += M->pl_k[e] * v;
            continue;
        }
        e -= (uint)M->n_pl;
        if (e < (uint)M->n_se) {
            const idx_t p = M->se_idx[e];
            const float i = M->se_I[e];
            const float v = V[p] - M->se_vi[e] * i;
            V[p] = v;
            const float qq = M->se_q[e] + M->dtf * i;
            M->se_q[e] = qq;
            M->se_I[e] = M->se_a[e] * i + M->se_b[e] * v - M->se_c[e] * qq;
            continue;
        }
        e -= (uint)M->n_se;
        V[M->m0[e]] = M->o1[e] + M->mk[e] * (V[M->m1[e]] - M->o0[e]);
    }
}

// all PML slabs in one launch (the split mode): the slab table maps a thread's index g to its slab
#define GF_SLAB(desc, b, c)                                                                             \
    int q = 0;                                                                                          \
    while (q < M->ns - 1 && (idx_t)g >= desc[(q + 1) * 6]) ++q;                                         \
    idx_t* d = desc + q * 6;                                                                            \
    const idx_t st = d[0], m = d[1], e = (idx_t)g - st;                                                 \
    const int a = (int)d[2], s0 = (int)d[3], cnt = (int)d[4];                                           \
    float* bq = b + d[5];                                                                               \
    float* cq = c + d[5];

[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void pml_e(uint3 tid: SV_DispatchThreadID)
{
    GF_DIMS
    for (uint g = tid.x; g < (uint)M->pml_M; g += stride) {
        GF_SLAB(M->descE, M->bE, M->cE)
        gf_pml_e(e, m, M->V, M->I, M->psiE + 2 * st, bq, cq, M->k1E + st, M->k2E + st, D, a, s0, cnt);
    }
}

[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void pml_h(uint3 tid: SV_DispatchThreadID)
{
    GF_DIMS
    for (uint g = tid.x; g < (uint)M->pml_M; g += stride) {
        GF_SLAB(M->descH, M->bH, M->cH)
        gf_pml_h(e, m, M->V, M->I, M->psiH + 2 * st, bq, cq, M->k1H + st, M->k2H + st, D, a, s0, cnt);
    }
}

[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void mur_pre(uint3 tid: SV_DispatchThreadID)
{
    GF_LOOP(M->n_mu) { M->o0[t] = M->V[M->m0[t]]; M->o1[t] = M->V[M->m1[t]]; }
}

[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void mur_post(uint3 tid: SV_DispatchThreadID)
{
    GF_LOOP(M->n_mu) { M->V[M->m0[t]] = M->o1[t] + M->mk[t] * (M->V[M->m1[t]] - M->o0[t]); }
}

[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void excite(uint3 tid: SV_DispatchThreadID)
{
    GF_LOOP(m_) { GF_ADD(F_ + idx_[t], amp_[t] * s_); }
}

[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void assign(uint3 tid: SV_DispatchThreadID)
{
    GF_LOOP(m_) { F_[idx_[t]] = amp_[t] * s_; }
}

[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void par_l(uint3 tid: SV_DispatchThreadID)
{
    GF_LOOP(M->n_pl) {
        const idx_t q = M->pl_idx[t];
        const float v = M->V[q] - M->pl_vi[t] * M->pl_I[t];
        M->V[q] = v;
        M->pl_I[t] += M->pl_k[t] * v;
    }
}

[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void series_rlc(uint3 tid: SV_DispatchThreadID)
{
    GF_LOOP(M->n_se) {
        const idx_t p = M->se_idx[t];
        const float i = M->se_I[t];
        const float v = M->V[p] - M->se_vi[t] * i;
        M->V[p] = v;
        const float qq = M->se_q[t] + M->dtf * i;
        M->se_q[t] = qq;
        M->se_I[t] = M->se_a[t] * i + M->se_b[t] * v - M->se_c[t] * qq;
    }
}

// one sample: the voltage probes' terms, then the current probes', added into buffer row slot_ (zeroed by the host)
[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void probe(uint3 tid: SV_DispatchThreadID)
{
    float* row = M->buf + (idx_t)slot_ * M->np_;
    GF_LOOP(M->n_pv + M->n_pi) {
        if (t < (uint)M->n_pv)
            GF_ADD(row + M->pv_pid[t], M->pv_w[t] * M->V[M->pv_idx[t]]);
        else {
            const uint e = t - (uint)M->n_pv;
            GF_ADD(row + M->pi_pid[e], M->pi_w[e] * M->I[M->pi_idx[e]]);
        }
    }
}

// a device array to mapped host memory (emsdev.vulkan.HostBuffer): reads at PCIe speed, where slangpy's read-back
// buffers are uncached memory on some drivers
uniform uint* src_w;
uniform uint* dst_w;
uniform uint nw_;

[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void copy_words(uint3 tid: SV_DispatchThreadID)
{
    GF_LOOP(nw_) { dst_w[t] = src_w[t]; }
}

// ---- field dumps (kernels.py's dump): field 0 E, 1 H, 2 J (rot H); cell: cell centres, else nodes ----------
float gf_fE(float* V, float* G, Dims D, int a, int i, int j, int k)
{
    int ii[3] = {i, j, k}, nn[3] = {D.nx, D.ny, D.nz};
    for (int b = 0; b < 3; ++b)
        if (ii[b] < 0 || ii[b] > nn[b] - (b == a ? 2 : 1)) return 0.f;
    const idx_t n = (idx_t)D.nx * D.ny * D.nz;
    return V[a * n + ((idx_t)i * D.ny + j) * D.nz + k] * GF_G(G, D, 1, a, ii[a]);
}

float gf_fH(float* I, float* G, Dims D, int a, int i, int j, int k)
{
    int ii[3] = {i, j, k}, nn[3] = {D.nx, D.ny, D.nz};
    for (int b = 0; b < 3; ++b)
        if (ii[b] < 0 || ii[b] > nn[b] - (b == a ? 1 : 2)) return 0.f;
    const idx_t n = (idx_t)D.nx * D.ny * D.nz;
    return I[a * n + ((idx_t)i * D.ny + j) * D.nz + k] * GF_G(G, D, 3, a, ii[a]);
}

float gf_fJ(float* I, float* G, Dims D, int a, int i, int j, int k)
{
    int ii[3] = {i, j, k}, nn[3] = {D.nx, D.ny, D.nz};
    for (int q = 0; q < 3; ++q)
        if (ii[q] < 0 || ii[q] > nn[q] - 1) return 0.f;
    const int b = (a + 1) % 3, c = (a + 2) % 3;
    const idx_t n = (idx_t)D.nx * D.ny * D.nz, t = ((idx_t)i * D.ny + j) * D.nz + k;
    idx_t st[3] = {(idx_t)D.ny * D.nz, (idx_t)D.nz, 1};
    float* Ib = I + b * n;
    float* Ic = I + c * n;
    const float cur = Ic[t] - (ii[b] > 0 ? Ic[t - st[b]] : 0.f) - Ib[t] + (ii[c] > 0 ? Ib[t - st[c]] : 0.f);
    return cur * GF_G(G, D, 3, b, ii[b]) * GF_G(G, D, 3, c, ii[c]);
}

float gf_dump_f(int field, float* V, float* I, float* G, Dims D, int a, int i, int j, int k)
{
    return field == 0 ? gf_fE(V, G, D, a, i, j, k) : (field == 1 ? gf_fH(I, G, D, a, i, j, k) : gf_fJ(I, G, D, a, i, j, k));
}

// dst_: nf_ == 0: the values (np, 3); else acc (nf, np, 3) complex, += value * exp(-j w t)
[shader("compute")] [numthreads(GF_BLOCK, 1, 1)]
void dump(uint3 tid: SV_DispatchThreadID)
{
    GF_DIMS
    const uint np_ = (uint)sx_ * (uint)sy_ * (uint)sz_;
    GF_LOOP(np_ * 3) {
        const int a = (int)(t % 3);
        const uint p = t / 3;
        const int pk = (int)(p % (uint)sz_), pj = (int)((p / (uint)sz_) % (uint)sy_), pi = (int)(p / ((uint)sz_ * (uint)sy_));
        const int i = lx_ + pi, j = ly_ + pj, k = lz_ + pk;
        int nn[3] = {D.nx, D.ny, D.nz};
        float sum = 0.f, cnt = 0.f;
        const bool along_a = (field_ == 1) != (cell_ == 0);
        int ax[2] = {0, 0};
        int na = 0;
        for (int b = 0; b < 3; ++b)
            if ((b == a) == along_a) ax[na++] = b;
        const int nterm = 1 << na;
        for (int q = 0; q < nterm; ++q) {
            int ii[3] = {i, j, k};
            for (int r = 0; r < na; ++r)
                ii[ax[r]] += ((q >> r) & 1) != 0 ? (cell_ != 0 ? 1 : 0) : (cell_ != 0 ? 0 : -1);
            sum += gf_dump_f(field_, M->V, M->I, M->G, D, a, ii[0], ii[1], ii[2]);
            if (cell_ == 0) {
                bool ok = true;
                for (int b = 0; b < 3; ++b) {
                    const int hi = field_ == 1 ? (b == a ? nn[b] - 1 : nn[b] - 2) : (b == a ? nn[b] - 2 : nn[b] - 1);
                    if (ii[b] < 0 || ii[b] > hi) ok = false;
                }
                cnt += ok ? 1.f : 0.f;
            }
        }
        const float v = cell_ != 0 ? sum / (float)nterm : sum / max(cnt, 1.f);
        if (nf_ == 0) {
            dst_[p * 3 + a] = v;
        } else {
            for (int f = 0; f < nf_; ++f) {
                float sn, cs;
                sincos(-omega_[f] * tm_, sn, cs);
                const int64_t o = (((int64_t)f * np_ + p) * 3 + a) * 2;
                dst_[o] += v * cs;
                dst_[o + 1] += v * sn;
            }
        }
    }
}
'''


def source(code_t='uint16_t', block=128, idx='int'):
    """The module: CODE_T codes, `block` threads a workgroup, `idx` the index type (int: the index arrays int32)."""
    with open(HEADER, encoding='utf-8') as fh:
        head = fh.read()
    pre = (f"#define GF_DEV\n#define GF_GPU 1\n#define GF_BLOCK {int(block)}\n#define CODE_T {code_t}\n"
           f"#define GF_IDX {idx}\n#define GF_ADD(p, v) InterlockedAdd(*(p), (v))\n")
    return pre + head + "\n" + _model_struct() + KERNELS



