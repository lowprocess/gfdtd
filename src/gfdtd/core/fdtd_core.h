// gfdtd's update maths, one node at a time: shared by the GPU kernels (kernels.py: CUDA / HIP through nvrtc /
// hiprtc) and the CPU core (gfdtd_core.cpp: g++ with OpenMP), so the two run the same arithmetic.
//
// Fields V, I: float32 (3, nx, ny, nz), z fastest. The compressed operator (operator.Compressed): per node a
// code into tables vv / s / pole of its three E edges ([code * 3 + a]); vi = ke * d_a / (dual_b dual_c) * s and iv = kh * dual_a / (d_b d_c) from the
// line arrays G = [d | 1/d | dual | 1/dual], each [x | y | z]. Debye pole currents (scaled, i~) in pol
// (nslot, 3, P), the node's row in slot (-1: none).
#ifndef GF_DEV
#define GF_DEV inline
#endif

typedef long long idx_t;

// a PML term added to a field: the GPU runs all slabs at once and the slabs meet at the corners, so atomically
#ifndef GF_ADD
#define GF_ADD(p, v) (*(p) += (v))
#endif

struct Dims {
    int nx, ny, nz;
};

// G[kind][axis][index]: kind 0 d, 1 1/d, 2 dual, 3 1/dual
#define GF_G(G, D, kind, ax, idx) \
    G[(kind) * (D.nx + D.ny + D.nz) + ((ax) == 0 ? 0 : ((ax) == 1 ? D.nx : D.nx + D.ny)) + (idx)]

GF_DEV void gf_h_node(int i, int j, int k, idx_t t, const float* V, float* I, const float* G, Dims D, float kh)
{
    const idx_t n = (idx_t)D.nx * D.ny * D.nz, sy = D.nz, sx = (idx_t)D.ny * D.nz;
    const float *Vx = V, *Vy = V + n, *Vz = V + 2 * n;
    const float vx = Vx[t], vy = Vy[t], vz = Vz[t];
    if (j < D.ny - 1 && k < D.nz - 1)
        I[t] -= kh * GF_G(G, D, 2, 0, i) * GF_G(G, D, 1, 1, j) * GF_G(G, D, 1, 2, k) *
                (Vz[t + sy] - vz - Vy[t + 1] + vy);
    if (i < D.nx - 1 && k < D.nz - 1)
        I[n + t] -= kh * GF_G(G, D, 2, 1, j) * GF_G(G, D, 1, 2, k) * GF_G(G, D, 1, 0, i) *
                    (Vx[t + 1] - vx - Vz[t + sx] + vz);
    if (i < D.nx - 1 && j < D.ny - 1)
        I[2 * n + t] -= kh * GF_G(G, D, 2, 2, k) * GF_G(G, D, 1, 0, i) * GF_G(G, D, 1, 1, j) *
                        (Vy[t + sx] - vy - Vx[t + sy] + vx);
}

// one E component: the loop sum c, the coefficients by code, the Debye poles (folded into this update)
template <typename C>
GF_DEV void gf_e_comp(int a, idx_t t, float c, float geo, float* V, const C* code, const float* tvv, const float* ts,
                      const int* tpole, const float* tkb, const float* tka, const float* th, int P, const int* slot,
                      float* pol, float dt, idx_t n)
{
    const idx_t q = (idx_t)code[t] * 3 + a;
    const float s = ts[q], v0 = V[a * n + t];
    float v = tvv[q] * v0 + geo * s * c;
    const int pr = tpole[q];
    if (pr >= 0) {
        float* ip = pol + ((idx_t)slot[t] * 3 + a) * P;
        const float* kb = tkb + (idx_t)pr * P;
        float sum = 0.f;
        for (int p = 0; p < P; ++p)
            sum += th[p] * ip[p];
        v -= dt * s * sum;
        for (int p = 0; p < P; ++p)
            ip[p] = tka[p] * ip[p] + kb[p] * (v - v0);
    }
    V[a * n + t] = v;
}

template <typename C>
GF_DEV void gf_e_node(int i, int j, int k, idx_t t, float* V, const float* I, const C* code, const float* tvv,
                      const float* ts, const int* tpole, const float* tkb, const float* tka, const float* th, int P,
                      const int* slot, float* pol, const float* G, Dims D, float ke, float dt)
{
    const idx_t n = (idx_t)D.nx * D.ny * D.nz, sy = D.nz, sx = (idx_t)D.ny * D.nz;
    const float *Ix = I, *Iy = I + n, *Iz = I + 2 * n;
    const float ix = Ix[t], iy = Iy[t], iz = Iz[t];
    if (i < D.nx - 1) {
        const float c = iz - (j > 0 ? Iz[t - sy] : 0.f) - iy + (k > 0 ? Iy[t - 1] : 0.f);
        const float geo = ke * GF_G(G, D, 0, 0, i) * GF_G(G, D, 3, 1, j) * GF_G(G, D, 3, 2, k);
        gf_e_comp<C>(0, t, c, geo, V, code, tvv, ts, tpole, tkb, tka, th, P, slot, pol, dt, n);
    }
    if (j < D.ny - 1) {
        const float c = ix - (k > 0 ? Ix[t - 1] : 0.f) - iz + (i > 0 ? Iz[t - sx] : 0.f);
        const float geo = ke * GF_G(G, D, 0, 1, j) * GF_G(G, D, 3, 2, k) * GF_G(G, D, 3, 0, i);
        gf_e_comp<C>(1, t, c, geo, V, code, tvv, ts, tpole, tkb, tka, th, P, slot, pol, dt, n);
    }
    if (k < D.nz - 1) {
        const float c = iy - (i > 0 ? Iy[t - sx] : 0.f) - ix + (j > 0 ? Ix[t - sy] : 0.f);
        const float geo = ke * GF_G(G, D, 0, 2, k) * GF_G(G, D, 3, 0, i) * GF_G(G, D, 3, 1, j);
        gf_e_comp<C>(2, t, c, geo, V, code, tvv, ts, tpole, tkb, tka, th, P, slot, pol, dt, n);
    }
}

// the field energy of one node (both fields, without the dt / 2): V^2 / vi + I^2 / iv
template <typename C>
GF_DEV double gf_energy_node(int i, int j, int k, idx_t t, const float* V, const float* I, const C* code,
                             const float* ts, const float* G, Dims D, float ke, float kh)
{
    const idx_t n = (idx_t)D.nx * D.ny * D.nz;
    const int ii[3] = {i, j, k};
    double e = 0.0;
    for (int a = 0; a < 3; ++a) {
        const int b = (a + 1) % 3, c = (a + 2) % 3;
        const double vi = (double)ke * GF_G(G, D, 0, a, ii[a]) * GF_G(G, D, 3, b, ii[b]) * GF_G(G, D, 3, c, ii[c]) *
                          ts[(idx_t)code[t] * 3 + a];
        const double iv = (double)kh * GF_G(G, D, 2, a, ii[a]) * GF_G(G, D, 1, b, ii[b]) * GF_G(G, D, 1, c, ii[c]);
        const double v = V[a * n + t], cur = I[a * n + t];
        if (vi > 0) e += v * v / vi;
        if (iv > 0) e += cur * cur / iv;
    }
    return e;
}

// one PML slab node (cnt lines along axis a from s0, the slab-local index e): psi update and its terms on the
// two tangential components; k1, k2: their vi (E) or iv (H), slab-local
#define GF_PML_DECODE                                                                                   \
    const idx_t n = (idx_t)D.nx * D.ny * D.nz;                                                          \
    const int L1 = a == 1 ? cnt : D.ny, L2 = a == 2 ? cnt : D.nz;                                       \
    const idx_t sa = a == 0 ? (idx_t)D.ny * D.nz : (a == 1 ? (idx_t)D.nz : (idx_t)1);                   \
    const int na = a == 0 ? D.nx : (a == 1 ? D.ny : D.nz);                                              \
    const idx_t o1 = (idx_t)((a + 1) % 3) * n, o2 = (idx_t)((a + 2) % 3) * n;                           \
    int k = (int)(e % L2);                                                                              \
    const idx_t r = e / L2;                                                                             \
    int j = (int)(r % L1), i = (int)(r / L1);                                                           \
    const int la = a == 0 ? i : (a == 1 ? j : k);                                                       \
    if (a == 0) i += s0; else if (a == 1) j += s0; else k += s0;                                        \
    const int ia = la + s0;                                                                             \
    const idx_t t = ((idx_t)i * D.ny + j) * D.nz + k;                                                   \
    const float bb = b[la], cc = c[la];                                                                 \
    (void)na;

GF_DEV void gf_pml_e(idx_t e, idx_t m, float* V, const float* I, float* psi, const float* b, const float* c,
                     const float* k1, const float* k2, Dims D, int a, int s0, int cnt)
{
    GF_PML_DECODE
    const float d2 = I[o2 + t] - (ia > 0 ? I[o2 + t - sa] : 0.f);
    const float p1 = bb * psi[e] + cc * d2;
    psi[e] = p1;
    GF_ADD(&V[o1 + t], -k1[e] * p1);
    const float d1 = I[o1 + t] - (ia > 0 ? I[o1 + t - sa] : 0.f);
    const float p2 = bb * psi[m + e] + cc * d1;
    psi[m + e] = p2;
    GF_ADD(&V[o2 + t], k2[e] * p2);
}

GF_DEV void gf_pml_h(idx_t e, idx_t m, const float* V, float* I, float* psi, const float* b, const float* c,
                     const float* k1, const float* k2, Dims D, int a, int s0, int cnt)
{
    GF_PML_DECODE
    const float d2 = (ia < na - 1 ? V[o2 + t + sa] : 0.f) - V[o2 + t];
    const float p1 = bb * psi[e] + cc * d2;
    psi[e] = p1;
    GF_ADD(&I[o1 + t], k1[e] * p1);
    const float d1 = (ia < na - 1 ? V[o1 + t + sa] : 0.f) - V[o1 + t];
    const float p2 = bb * psi[m + e] + cc * d1;
    psi[m + e] = p2;
    GF_ADD(&I[o2 + t], -k2[e] * p2);
}

// ---- CPU rows: one (i, j) line along z with the boundary tests out of the loops, so that they vectorise --------
GF_DEV void gf_h_row(int i, int j, const float* V, float* I, const float* G, Dims D, float kh)
{
    const idx_t n = (idx_t)D.nx * D.ny * D.nz, sy = D.nz, sx = (idx_t)D.ny * D.nz, t0 = ((idx_t)i * D.ny + j) * D.nz;
    const float *Vx = V + t0, *Vy = V + n + t0, *Vz = V + 2 * n + t0;
    float *Ix = I + t0, *Iy = I + n + t0, *Iz = I + 2 * n + t0;
    const float* idz = &GF_G(G, D, 1, 2, 0);
    const float* dualz = &GF_G(G, D, 2, 2, 0);
    const int kz = D.nz - 1;
    const bool jy = j < D.ny - 1, ix = i < D.nx - 1;
    if (jy) {
        const float f = kh * GF_G(G, D, 2, 0, i) * GF_G(G, D, 1, 1, j);
#pragma omp simd
        for (int k = 0; k < kz; ++k)
            Ix[k] -= f * idz[k] * (Vz[k + sy] - Vz[k] - Vy[k + 1] + Vy[k]);
    }
    if (ix) {
        const float f = kh * GF_G(G, D, 2, 1, j) * GF_G(G, D, 1, 0, i);
#pragma omp simd
        for (int k = 0; k < kz; ++k)
            Iy[k] -= f * idz[k] * (Vx[k + 1] - Vx[k] - Vz[k + sx] + Vz[k]);
    }
    if (ix && jy) {
        const float f = kh * GF_G(G, D, 1, 0, i) * GF_G(G, D, 1, 1, j);
#pragma omp simd
        for (int k = 0; k < D.nz; ++k)
            Iz[k] -= f * dualz[k] * (Vy[k + sx] - Vy[k] - Vx[k + sy] + Vx[k]);
    }
}

// one E component along a row of kn nodes: c = Pp - Pm - Qp + Qm (four rows, the caller's: neighbour rows, a row
// shifted along z, or zeros), then vv V + vi c vectorised, then the Debye poles of the nodes that have them
template <typename C>
GF_DEV void gf_e_row_comp(int a, idx_t t0, int kn, float* Vr, const float* Pp, const float* Pm, const float* Qp,
                          const float* Qm, const float* geo_k, float f, const C* code, const float* tvv,
                          const float* ts, const int* tpole, const float* tkb, const float* tka, const float* th,
                          int P, const int* slot, float* pol, float dt, idx_t n, float* old, bool poles)
{
    const C* cd = code + t0;
#pragma omp simd
    for (int k = 0; k < kn; ++k) {
        const float v0 = Vr[k];
        const idx_t q = (idx_t)cd[k] * 3 + a;
        old[k] = v0;
        Vr[k] = tvv[q] * v0 + f * geo_k[k] * ts[q] * (Pp[k] - Pm[k] - Qp[k] + Qm[k]);
    }
    if (!poles)
        return;
    for (int k = 0; k < kn; ++k) {
        const idx_t q = (idx_t)cd[k] * 3 + a;
        const int pr = tpole[q];
        if (pr < 0)
            continue;
        const float s = ts[q];
        float* ip = pol + ((idx_t)slot[t0 + k] * 3 + a) * P;
        const float* kb = tkb + (idx_t)pr * P;
        float sum = 0.f;
        for (int p = 0; p < P; ++p)
            sum += th[p] * ip[p];
        const float v = Vr[k] - dt * s * sum;
        Vr[k] = v;
        for (int p = 0; p < P; ++p)
            ip[p] = tka[p] * ip[p] + kb[p] * (v - old[k]);
    }
}

// the E update of one (i, j) row; scratch: 4 nz floats per thread (zeros, the z-shifted rows, old V)
template <typename C>
GF_DEV void gf_e_row(int i, int j, float* V, const float* I, const C* code, const float* tvv, const float* ts,
                     const int* tpole, const float* tkb, const float* tka, const float* th, int P, const int* slot,
                     float* pol, const float* G, Dims D, float ke, float dt, float* scratch, bool poles)
{
    const idx_t n = (idx_t)D.nx * D.ny * D.nz, sy = D.nz, sx = (idx_t)D.ny * D.nz, t0 = ((idx_t)i * D.ny + j) * D.nz;
    const int nz = D.nz;
    const float *Ix = I + t0, *Iy = I + n + t0, *Iz = I + 2 * n + t0;
    float *zero = scratch, *shift = scratch + nz, *old = scratch + 2 * nz;
    const float* dz = &GF_G(G, D, 0, 2, 0);
    const float* iduz = &GF_G(G, D, 3, 2, 0);
    auto shifted = [&](const float* r) {             // r[k - 1], 0 at k = 0
        shift[0] = 0.f;
        for (int k = 1; k < nz; ++k)
            shift[k] = r[k - 1];
        return (const float*)shift;
    };
    if (i < D.nx - 1) {                              // Vx: c = Iz - Iz[j-1] - Iy + Iy[k-1]
        const float f = ke * GF_G(G, D, 0, 0, i) * GF_G(G, D, 3, 1, j);
        gf_e_row_comp<C>(0, t0, nz, V + t0, Iz, j > 0 ? Iz - sy : zero, Iy, shifted(Iy), iduz, f, code, tvv, ts,
                         tpole, tkb, tka, th, P, slot, pol, dt, n, old, poles);
    }
    if (j < D.ny - 1) {                              // Vy: c = Ix - Ix[k-1] - Iz + Iz[i-1]
        const float f = ke * GF_G(G, D, 0, 1, j) * GF_G(G, D, 3, 0, i);
        gf_e_row_comp<C>(1, t0, nz, V + n + t0, Ix, shifted(Ix), Iz, i > 0 ? Iz - sx : zero, iduz, f, code, tvv,
                         ts, tpole, tkb, tka, th, P, slot, pol, dt, n, old, poles);
    }
    {                                                // Vz (k < nz - 1): c = Iy - Iy[i-1] - Ix + Ix[j-1]
        const float f = ke * GF_G(G, D, 3, 0, i) * GF_G(G, D, 3, 1, j);
        gf_e_row_comp<C>(2, t0, nz - 1, V + 2 * n + t0, Iy, i > 0 ? Iy - sx : zero, Ix, j > 0 ? Ix - sy : zero, dz,
                         f, code, tvv, ts, tpole, tkb, tka, th, P, slot, pol, dt, n, old, poles);
    }
}

// ---- the PML folded into the H / E node updates (GPU): the node's terms of every slab it is in ---------------
// desc: per slab SLAB values (start of its local index range, m, axis, s0, cnt, offset of b / c); psi: the slabs'
// psi concatenated (2 m each, from 2 start). The node's own iv / vi from the line arrays (and its code).
GF_DEV float gf_iv(int a, const int* ii, const float* G, Dims D, float kh)
{
    const int b = (a + 1) % 3, c = (a + 2) % 3;
    return kh * GF_G(G, D, 2, a, ii[a]) * GF_G(G, D, 1, b, ii[b]) * GF_G(G, D, 1, c, ii[c]);
}

GF_DEV float gf_vi(int a, const int* ii, const float* G, Dims D, float ke, float s)
{
    const int b = (a + 1) % 3, c = (a + 2) % 3;
    return ke * GF_G(G, D, 0, a, ii[a]) * GF_G(G, D, 3, b, ii[b]) * GF_G(G, D, 3, c, ii[c]) * s;
}

template <bool E, typename C>
GF_DEV void gf_pml_node(int i, int j, int k, idx_t t, float* V, float* I, float* psi, const float* b, const float* c,
                        const idx_t* desc, int ns, const int* box, const C* code, const float* ts, const float* G,
                        Dims D, float kf)
{
    // box: the PML-free interior, lines [box[a], box[3 + a]) on each axis: most nodes leave here
    if (i >= box[0] && i < box[3] && j >= box[1] && j < box[4] && k >= box[2] && k < box[5])
        return;
    const idx_t n = (idx_t)D.nx * D.ny * D.nz;
    const int ii[3] = {i, j, k}, nn[3] = {D.nx, D.ny, D.nz};
    for (int q = 0; q < ns; ++q) {
        const idx_t* d = desc + q * 6;
        const int a = (int)d[2], s0 = (int)d[3], cnt = (int)d[4];
        const int ia = ii[a];
        if (ia < s0 || ia >= s0 + cnt)
            continue;
        const int la = ia - s0, L1 = a == 1 ? cnt : D.ny, L2 = a == 2 ? cnt : D.nz;
        const int li = a == 0 ? la : i, lj = a == 1 ? la : j, lk = a == 2 ? la : k;
        const idx_t e = ((idx_t)li * L1 + lj) * L2 + lk, m = d[1];
        float* p = psi + 2 * d[0];
        const float bb = b[d[5] + la], cc = c[d[5] + la];
        const idx_t sa = a == 0 ? (idx_t)D.ny * D.nz : (a == 1 ? (idx_t)D.nz : (idx_t)1);
        const int t1 = (a + 1) % 3, t2 = (a + 2) % 3;
        const idx_t o1 = (idx_t)t1 * n, o2 = (idx_t)t2 * n;
        if (E) {                                        // backward differences of I; V += vi (-psi1, +psi2)
            const float d2 = I[o2 + t] - (ia > 0 ? I[o2 + t - sa] : 0.f);
            const float d1 = I[o1 + t] - (ia > 0 ? I[o1 + t - sa] : 0.f);
            const float p1 = bb * p[e] + cc * d2, p2 = bb * p[m + e] + cc * d1;
            p[e] = p1;
            p[m + e] = p2;
            V[o1 + t] -= gf_vi(t1, ii, G, D, kf, ts[(idx_t)code[t] * 3 + t1]) * p1;
            V[o2 + t] += gf_vi(t2, ii, G, D, kf, ts[(idx_t)code[t] * 3 + t2]) * p2;
        } else {                                        // forward differences of V; I += iv (psi1, -psi2)
            const float d2 = (ia < nn[a] - 1 ? V[o2 + t + sa] : 0.f) - V[o2 + t];
            const float d1 = (ia < nn[a] - 1 ? V[o1 + t + sa] : 0.f) - V[o1 + t];
            const float p1 = bb * p[e] + cc * d2, p2 = bb * p[m + e] + cc * d1;
            p[e] = p1;
            p[m + e] = p2;
            I[o1 + t] += gf_iv(t1, ii, G, D, kf) * p1;
            I[o2 + t] -= gf_iv(t2, ii, G, D, kf) * p2;
        }
    }
}
