// gfdtd's CPU core: the H / E updates, the energy and the PML slabs of core/fdtd_core.h, over OpenMP threads (x and
// y across the threads, z innermost). Built by core/build.py into gfdtd._core; engine.CoreBackend drives it.
#include <cstdint>
#include <omp.h>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "fdtd_core.h"

#if defined(__SSE__) || defined(__x86_64__)
#include <xmmintrin.h>
// denormals flushed to zero in the core's threads while it runs (the GPU does this anyway): the fields grow from
// zero, and the tiny values ahead of a wave ran the first few thousand steps 5-10x slower; restored afterwards
struct FlushDenormals {
    unsigned old = _mm_getcsr();
    FlushDenormals() { _mm_setcsr(old | 0x8040); }
    ~FlushDenormals() { _mm_setcsr(old); }
};
#else
struct FlushDenormals {};
#endif

namespace py = pybind11;
template <typename T> using arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T> static T* ptr(arr<T>& a) { return a.mutable_data(); }

static void update_h(arr<float> V, arr<float> I, arr<float> G, int nx, int ny, int nz, float kh)
{
    const Dims D{nx, ny, nz};
    const float* v = ptr(V); float* cur = ptr(I); const float* g = ptr(G);
    py::gil_scoped_release nogil;
#pragma omp parallel
    {
        FlushDenormals ftz;
#pragma omp for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            gf_h_row(i, j, v, cur, g, D, kh);
    }
}

template <typename C>
static void update_e(arr<float> V, arr<float> I, arr<C> code, arr<float> tvv, arr<float> ts, arr<int32_t> tpole,
                     arr<float> tkb, arr<float> tka, arr<float> th, int P, arr<int32_t> slot, arr<float> pol,
                     arr<float> G, int nx, int ny, int nz, float ke, float dt)
{
    const Dims D{nx, ny, nz};
    float* v = ptr(V); const float* cur = ptr(I); const C* cd = ptr(code);
    const float *vv = ptr(tvv), *s = ptr(ts), *kb = ptr(tkb), *ka = ptr(tka), *h = ptr(th), *g = ptr(G);
    const int *tp = ptr(tpole), *sl = ptr(slot);
    float* pl = ptr(pol);
    const bool poles = P > 0;
    py::gil_scoped_release nogil;
#pragma omp parallel
    {
        FlushDenormals ftz;
        std::vector<float> scratch(3 * (size_t)nz, 0.f);
#pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                gf_e_row<C>(i, j, v, cur, cd, vv, s, tp, kb, ka, h, P, sl, pl, g, D, ke, dt, scratch.data(), poles);
    }
}

static void mur_pre(arr<float> V, arr<int64_t> i0, arr<int64_t> i1, arr<float> o0, arr<float> o1)
{
    const float* v = ptr(V); const int64_t *a = ptr(i0), *b = ptr(i1); float *p = ptr(o0), *q = ptr(o1);
    const idx_t m = i0.size();
    py::gil_scoped_release nogil;
#pragma omp parallel
    {
        FlushDenormals ftz;
#pragma omp for schedule(static)
    for (idx_t e = 0; e < m; ++e) { p[e] = v[a[e]]; q[e] = v[b[e]]; }
    }
}

static void mur_post(arr<float> V, arr<int64_t> i0, arr<int64_t> i1, arr<float> k, arr<float> o0, arr<float> o1)
{
    float* v = ptr(V); const int64_t *a = ptr(i0), *b = ptr(i1); const float *kk = ptr(k), *p = ptr(o0), *q = ptr(o1);
    const idx_t m = i0.size();
    py::gil_scoped_release nogil;
#pragma omp parallel
    {
        FlushDenormals ftz;
#pragma omp for schedule(static)
    for (idx_t e = 0; e < m; ++e) v[a[e]] = q[e] + kk[e] * (v[b[e]] - p[e]);
    }
}

template <typename C>
static double energy(arr<float> V, arr<float> I, arr<C> code, arr<float> ts, arr<float> G, int nx, int ny, int nz,
                     float ke, float kh)
{
    const Dims D{nx, ny, nz};
    const float *v = ptr(V), *cur = ptr(I), *s = ptr(ts), *g = ptr(G);
    const C* cd = ptr(code);
    double e = 0.0;
    py::gil_scoped_release nogil;
#pragma omp parallel reduction(+ : e)
    {
        FlushDenormals ftz;
#pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j) {
                const idx_t t0 = ((idx_t)i * ny + j) * nz;
                for (int k = 0; k < nz; ++k)
                    e += gf_energy_node<C>(i, j, k, t0 + k, v, cur, cd, s, g, D, ke, kh);
            }
    }
    return e;
}

// one PML slab (cnt lines along axis a from s0): rows along z, the same arithmetic as gf_pml_e / gf_pml_h, so
// that the rows vectorise (psi, k1, k2 in the slab-local order of the GPU kernels)
template <bool E>
static void pml(arr<float> V, arr<float> I, arr<float> psi, arr<float> b, arr<float> c, arr<float> k1, arr<float> k2,
                int nx, int ny, int nz, int a, int s0, int cnt)
{
    float *v = ptr(V), *cur = ptr(I), *ps = ptr(psi);
    const float *bb = ptr(b), *cc = ptr(c), *q1 = ptr(k1), *q2 = ptr(k2);
    const idx_t n = (idx_t)nx * ny * nz;
    const int L0 = a == 0 ? cnt : nx, L1 = a == 1 ? cnt : ny, L2 = a == 2 ? cnt : nz;
    const idx_t m = (idx_t)L0 * L1 * L2;
    const idx_t sa = a == 0 ? (idx_t)ny * nz : (a == 1 ? (idx_t)nz : (idx_t)1);
    const int na = a == 0 ? nx : (a == 1 ? ny : nz);
    const idx_t o1 = (idx_t)((a + 1) % 3) * n, o2 = (idx_t)((a + 2) % 3) * n;
    // E reads I (and writes V), H reads V (and writes I): the field read and the field written
    const float* R = E ? cur : v;
    float* W = E ? v : cur;
    py::gil_scoped_release nogil;
#pragma omp parallel
    {
        FlushDenormals ftz;
#pragma omp for collapse(2) schedule(static)
        for (int li = 0; li < L0; ++li)
            for (int lj = 0; lj < L1; ++lj) {
                const int i = li + (a == 0 ? s0 : 0), j = lj + (a == 1 ? s0 : 0), k0 = a == 2 ? s0 : 0;
                const idx_t e0 = ((idx_t)li * L1 + lj) * L2, t0 = ((idx_t)i * ny + j) * nz + k0;
                const int la0 = a == 0 ? li : (a == 1 ? lj : 0);
                float *p1 = ps + e0, *p2 = ps + m + e0;
                const float *r1 = R + o1 + t0, *r2 = R + o2 + t0, *f1 = q1 + e0, *f2 = q2 + e0;
                float *w1 = W + o1 + t0, *w2 = W + o2 + t0;
#pragma omp simd
                for (int lk = 0; lk < L2; ++lk) {
                    const int la = a == 2 ? lk : la0, ia = la + s0;
                    float d2, d1;
                    if (E) {                                // backward differences, I[-1] = 0
                        d2 = r2[lk] - (ia > 0 ? r2[lk - sa] : 0.f);
                        d1 = r1[lk] - (ia > 0 ? r1[lk - sa] : 0.f);
                    } else {                                // forward differences, V[n] = 0
                        d2 = (ia < na - 1 ? r2[lk + sa] : 0.f) - r2[lk];
                        d1 = (ia < na - 1 ? r1[lk + sa] : 0.f) - r1[lk];
                    }
                    const float x1 = bb[la] * p1[lk] + cc[la] * d2, x2 = bb[la] * p2[lk] + cc[la] * d1;
                    p1[lk] = x1;
                    p2[lk] = x2;
                    if (E) { w1[lk] -= f1[lk] * x1; w2[lk] += f2[lk] * x2; }
                    else { w1[lk] += f1[lk] * x1; w2[lk] -= f2[lk] * x2; }
                }
            }
    }
}

PYBIND11_MODULE(_core, m)
{
    m.doc() = "gfdtd's CPU core (OpenMP): the updates of core/fdtd_core.h";
    m.def("update_h", &update_h);
    m.def("update_e", &update_e<uint8_t>);
    m.def("update_e", &update_e<uint16_t>);
    m.def("update_e", &update_e<uint32_t>);
    m.def("energy", &energy<uint8_t>);
    m.def("energy", &energy<uint16_t>);
    m.def("energy", &energy<uint32_t>);
    m.def("mur_pre", &mur_pre);
    m.def("mur_post", &mur_post);
    m.def("pml_e", &pml<true>);
    m.def("pml_h", &pml<false>);
    m.def("set_threads", [](int n) { omp_set_num_threads(n); });
    m.def("threads", []() {
        int n = 1;
#pragma omp parallel
#pragma omp single
        n = omp_get_num_threads();
        return n;
    });
}
