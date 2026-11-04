"""
The time loop: H and E updates on a backend, the extensions (PML slabs, Mur boundaries, Debye poles, soft E
excitations, lumped branches), probes, field dumps, the energy end criterion (the field's and the lumped branches'), and openEMS's outputs: its status
lines on stdout, its probe files (port_ut_1 ...) and its dump files in the run folder.

    Engine(op, backend='auto').run(path, nr_ts, end_crit)

Backends: cupy (the GPU: CUDA or ROCm), vulkan (the GPU through Vulkan and slangpy: AMD on Windows, any Vulkan GPU)
and core (the compiled CPU core, gfdtd._core: OpenMP) run the compressed operator (operator.compress: material codes
and line arrays in place of the coefficient arrays) through the same update code (core/fdtd_core.h); numpy runs the
dense coefficient arrays, slowly: the reference.

A file ABORT in the run folder ends the run as with openEMS (kicad2ems's port watch and Stop use it).
"""
import os, time

import numpy as np
from emsdev import gpu as devices                                    # the GPU's probe, shared with mom25d
from emsdev.cpu import core_threads                                  # noqa: F401  (gfdtd.engine.core_threads)

from . import kernels
from .dumps import Dumps

STATUS_S = 1.0          # wall seconds between status lines
CONCURRENT_CELLS = 1.2e6    # GPU runs at once: up to this many cells in all (GFDTD_CONCURRENT_CELLS sets it)
CHECK_STEPS = 100       # timesteps between energy / ABORT checks (one host sync each)
CHUNK = 512             # probe samples kept on the device before they are written out


def _gpu_probes():
    """(name, probe) for each GPU backend, in emsdev's 'auto' order: Vulkan first, CuPy first only on CUDA."""
    probes = dict(cupy=lambda: devices.cupy_device(0), vulkan=lambda: devices.vulkan_device(need=VULKAN_NEEDS))
    return [(n, probes[n]) for n in devices.auto_order()]


def gpu_info():
    """(usable, one line): is there a GPU to run on, through the backend 'gpu' picks (gpu_backend)?"""
    why = []
    for _, probe in _gpu_probes():
        try:
            return True, probe().describe()
        except devices.Unavailable as e:                            # no CuPy / slangpy, no driver, a broken install
            why.append(str(e))
    return False, "no GPU: " + "; ".join(why)


VULKAN_NEEDS = ('bda', 'int64', 'float64', 'atomic_add_f32')       # the Slang kernels' features (emsdev.vulkan)


def backend_info(name):
    """One line: what backend `name` (as resolve_backend gives it) runs on."""
    try:
        if name == 'cupy':
            return devices.cupy_device(0).describe()
        if name == 'vulkan':
            return devices.vulkan_device(need=VULKAN_NEEDS).describe()
    except devices.Unavailable as e:
        return f"no GPU: {e}"
    return core_info()[1] if name == 'core' else 'numpy (slow)'


def gpu_backend():
    """The GPU backend 'gpu' means here, or None: Vulkan (slangpy) first, CuPy first only where it runs on CUDA
    (emsdev.gpu.auto_order)."""
    for name, probe in _gpu_probes():
        try:
            probe()
            return name
        except devices.Unavailable:
            pass
    return None


def core_info():
    """(usable, one line): is the compiled CPU core built?"""
    try:
        from . import _core                                         # noqa: F401  (built?)
        n = os.environ.get('OMP_NUM_THREADS') or core_threads()
        return True, f"compiled CPU core, {n} threads"
    except ImportError as e:
        return False, f"no compiled CPU core (python -m gfdtd.core.build): {e}"


def concurrent_runs(cells, runs):
    """How many runs of a model of `cells` to have on the GPU at once. Small models leave it idle between kernel
    launches, and two to four runs side by side fill it. That holds while their fields fit in its cache together:
    on an RX 7800 XT (64 MB Infinity Cache) 4 runs of 0.29 M cells gave 1.46 times one run's throughput and 2 of
    0.59 M cells 1.16, but 2 of 1.48 M cells 0.64 (they thrash the cache). Hence up to CONCURRENT_CELLS in all."""
    limit = float(os.environ.get('GFDTD_CONCURRENT_CELLS') or CONCURRENT_CELLS)
    return max(1, min(int(runs), int(limit // max(cells, 1))))


def resolve_backend(name='auto'):
    """'cupy', 'vulkan', 'core' or 'numpy' for a backend name: auto (the GPU, else the core, else numpy), gpu (Vulkan's
    GPU, else CuPy's; CuPy's first on CUDA), cupy (cuda, rocm, hip), vulkan (vk), cpu (the core, else numpy), core, numpy.
    GFDTD_BACKEND overrides it."""
    name = devices.preference(name, 'GFDTD_BACKEND', dict(cuda='cupy', rocm='cupy', hip='cupy', vk='vulkan'))
    if name in ('gpu', 'auto'):
        gpu = gpu_backend()
        if gpu or name == 'gpu':
            return gpu or 'vulkan'                                  # gpu without one: the Vulkan error says why
        name = 'cpu'
    if name == 'cpu':
        name = 'core' if core_info()[0] else 'numpy'
    if name not in ('cupy', 'vulkan', 'core', 'numpy'):
        raise ValueError(f"gfdtd: backend {name!r}: auto, gpu (vulkan, cupy), cpu (core, numpy)")
    return name


class NumpyBackend:
    """The dense reference: coefficient arrays vv, vi, iv, numpy slices."""
    name = 'numpy'
    dense = True

    def __init__(self, op, dtype=np.float32):
        self.op, self.dt = op, dtype
        self.xp = np
        self.V = np.zeros((3,) + op.shape, dtype)
        self.I = np.zeros((3,) + op.shape, dtype)
        self.vv, self.vi, self.iv = (np.ascontiguousarray(a, dtype) for a in (op.vv, op.vi, op.iv))
        self._sparse()

    def _sparse(self):
        """The sparse extensions, the probes, the PML slabs (their vi / iv slab-local) and the Debye entries."""
        xp, op, dt = self.xp, self.op, self.dt
        vi_at = lambda flat: xp.asarray(np.asarray(op.vi.reshape(-1)[flat], dt))
        self.exc_idx, self.exc_amp = xp.asarray(op.exc[0]), xp.asarray(op.exc[1], dt)
        self.more = [dict(g, idx=xp.asarray(g['idx']), amp=xp.asarray(g['amp'], dt))
                     for g in getattr(op, 'exc_more', [])]
        if op.mur is not None:
            self.m0, self.m1, self.mk = xp.asarray(op.mur[0]), xp.asarray(op.mur[1]), xp.asarray(op.mur[2], dt)
            self.o0 = xp.zeros(len(op.mur[0]), dt)
            self.o1 = xp.zeros(len(op.mur[0]), dt)
        if op.par_L:
            idx, L = zip(*op.par_L)
            idx = np.array(idx, np.int64)
            self.pl_idx, self.pl_k = xp.asarray(idx), xp.asarray(op.dt / np.array(L), dt)
            self.pl_vi = vi_at(idx)
            self.pl_I = xp.zeros(len(idx), dt)
            self.pl_L = xp.asarray(np.array(L, float))
        if op.series:
            idx, R, L, Cs = zip(*op.series)
            idx = np.array(idx, np.int64)
            R, L = np.array(R, float), np.array(L, float)
            Cinv = np.array([0.0 if c is None else 1.0 / c for c in Cs])
            den = L / op.dt + R / 2
            if np.any(den <= 0):
                raise ValueError("gfdtd: a series lumped element needs R or L")
            self.se_idx, self.se_vi = xp.asarray(idx), vi_at(idx)
            self.se_a, self.se_b, self.se_c = (xp.asarray(v, dt) for v in ((L / op.dt - R / 2) / den, 1 / den,
                                                                            Cinv / den))
            self.se_I = xp.zeros(len(idx), dt)
            self.se_q = xp.zeros(len(idx), dt)
            self.se_L, self.se_Cinv = xp.asarray(L), xp.asarray(Cinv)
            self.dtf = dt(op.dt)
        vt = [p for p in op.probes if p['field'] == 'V']
        it = [p for p in op.probes if p['field'] == 'I']
        self.order = vt + it                                        # buffer columns: voltage probes, then current

        def terms(ps, off):
            if not ps:
                return None
            idx = np.concatenate([p['idx'] for p in ps])
            w = np.concatenate([p['w'] for p in ps])
            pid = np.concatenate([np.full(len(p['idx']), off + n, np.int64) for n, p in enumerate(ps)])
            return xp.asarray(idx), xp.asarray(w, dt), xp.asarray(pid)
        self.pv, self.pi = terms(vt, 0), terms(it, len(vt))
        self.np_ = len(self.order)
        self.buf = xp.zeros((CHUNK, max(self.np_, 1)), dt)
        self.pml = []
        for sl in op.pml:
            a, n = sl['axis'], sl['n']
            t1, t2 = (a + 1) % 3, (a + 2) % 3
            local = lambda F, s0: np.ascontiguousarray(np.take(F, range(s0, s0 + n), axis=a), dt).ravel()
            m = n * int(np.prod(op.shape)) // op.shape[a]
            self.pml.append(dict(sl, psiE=xp.zeros(2 * m, dt), psiH=xp.zeros(2 * m, dt), m=m,
                                 kE1=xp.asarray(local(op.vi[t1], sl['e0'])), kE2=xp.asarray(local(op.vi[t2], sl['e0'])),
                                 kH1=xp.asarray(local(op.iv[t1], sl['h0'])), kH2=xp.asarray(local(op.iv[t2], sl['h0'])),
                                 **{k: xp.asarray(sl[k], dt) for k in ('bE', 'cE', 'bH', 'cH')}))
        if self.dense and op.debye is not None:
            idx, ka, kb, sg = op.debye
            self.db_idx = xp.asarray(idx)
            self.db_ka, self.db_kb, self.db_h, self.db_sg = (xp.asarray(v, dt) for v in (ka, kb, 0.5 * (1 + ka), sg))
            self.db_vi = vi_at(idx)
            self.db_i = xp.zeros(len(idx), dt)

    # ---- the updates ---------------------------------------------------------------------------------------
    def step(self, sig=None, n=None):
        """One timestep: H, its PML, the H excitations, E (with the Debye poles), its PML, the excitation (sig: the
        signal's value, None past its end) and the delayed ones (n: the step), the lumped branches and Mur."""
        self.update_H()
        self.pml_h()
        if self.more and n is not None:
            self.excite_more('I', n)
        self.mur_pre()
        self.update_E()
        self.pml_e()
        if sig is not None:
            self.excite(sig)
        if self.more and n is not None:
            self.excite_more('V', n)
        self.lumped()
        self.mur_post()

    def excite_more(self, field, n):
        """The excitation groups on field 'V' or 'I' at step n: each its delay late, on the H signal (half a step
        later) for 'I'; a hard group sets its edges, a soft one adds to them."""
        sig = self.op.signal_h if field == 'I' else self.op.signal
        F = self.I if field == 'I' else self.V
        for g in self.more:
            if g['field'] != field:
                continue
            k = n - g['delay']
            val = float(sig[k]) if 0 <= k < len(sig) else 0.0
            if g['hard']:
                self.assign(F, g['idx'], g['amp'], val)
            elif val:
                self.add(F, g['idx'], g['amp'], val)

    def add(self, F, idx, amp, val):
        np.add.at(F.reshape(-1), idx, amp * self.dt(val))

    def assign(self, F, idx, amp, val):
        F.reshape(-1)[idx] = amp * self.dt(val)

    def update_H(self):
        (Vx, Vy, Vz), (Ix, Iy, Iz), (ax, ay, az) = self.V, self.I, self.iv
        Ix[:, :-1, :-1] -= ax[:, :-1, :-1] * (Vz[:, 1:, :-1] - Vz[:, :-1, :-1] - Vy[:, :-1, 1:] + Vy[:, :-1, :-1])
        Iy[:-1, :, :-1] -= ay[:-1, :, :-1] * (Vx[:-1, :, 1:] - Vx[:-1, :, :-1] - Vz[1:, :, :-1] + Vz[:-1, :, :-1])
        Iz[:-1, :-1, :] -= az[:-1, :-1, :] * (Vy[1:, :-1, :] - Vy[:-1, :-1, :] - Vx[:-1, 1:, :] + Vx[:-1, :-1, :])

    def update_E(self):
        """The E update with the poles folded in: V' = vv V + vi (c - sum h i), i' = ka i + kb (V' - sg V) (sg 1:
        Debye, -1: a conducting sheet's R-L branch)."""
        v = self.V.reshape(-1)
        if self.op.debye is not None:
            old = v[self.db_idx]
        (Ix, Iy, Iz) = self.I
        c = Iz - Iy
        c[:, 1:, :] -= Iz[:, :-1, :]
        c[:, :, 1:] += Iy[:, :, :-1]
        self.V[0] *= self.vv[0]
        self.V[0] += self.vi[0] * c
        c = Ix - Iz
        c[:, :, 1:] -= Ix[:, :, :-1]
        c[1:, :, :] += Iz[:-1, :, :]
        self.V[1] *= self.vv[1]
        self.V[1] += self.vi[1] * c
        c = Iy - Ix
        c[1:, :, :] -= Iy[:-1, :, :]
        c[:, 1:, :] += Ix[:, :-1, :]
        self.V[2] *= self.vv[2]
        self.V[2] += self.vi[2] * c
        if self.op.debye is not None:
            np.subtract.at(v, self.db_idx, self.db_vi * self.db_h * self.db_i)
            self.db_i[:] = self.db_ka * self.db_i + self.db_kb * (v[self.db_idx] - self.db_sg * old)

    def _pml(self, field):
        """Each slab's psi update and terms (numpy): the slab's axis moved first."""
        for sl in self.pml:
            a, n = sl['axis'], sl['n']
            t1, t2 = (a + 1) % 3, (a + 2) % 3
            s0 = sl[field.lower() + '0']
            b_, c_ = sl['b' + field].reshape(-1, 1, 1), sl['c' + field].reshape(-1, 1, 1)
            mv = lambda F: np.moveaxis(F, a, 0)
            psi = sl['psi' + field].reshape((2, n) + tuple(m for b, m in enumerate(self.op.shape) if b != a))
            if field == 'E':
                V1, V2, I1, I2, k1, k2 = mv(self.V[t1]), mv(self.V[t2]), mv(self.I[t1]), mv(self.I[t2]), \
                    mv(self.vi[t1]), mv(self.vi[t2])

                def diff(F):                                            # backward, I[-1] = 0
                    d = F[s0:s0 + n].copy()
                    if s0 > 0:
                        d -= F[s0 - 1:s0 + n - 1]
                    else:
                        d[1:] -= F[:n - 1]
                    return d
                psi[0] = b_ * psi[0] + c_ * diff(I2)
                V1[s0:s0 + n] -= k1[s0:s0 + n] * psi[0]
                psi[1] = b_ * psi[1] + c_ * diff(I1)
                V2[s0:s0 + n] += k2[s0:s0 + n] * psi[1]
            else:
                I1, I2, V1, V2, k1, k2 = mv(self.I[t1]), mv(self.I[t2]), mv(self.V[t1]), mv(self.V[t2]), \
                    mv(self.iv[t1]), mv(self.iv[t2])
                N = V1.shape[0]

                def diff(F):                                            # forward, V[N] = 0
                    d = -F[s0:s0 + n]
                    if s0 + n < N:
                        d = d + F[s0 + 1:s0 + n + 1]
                    else:
                        d[:-1] += F[s0 + 1:s0 + n]
                    return d
                psi[0] = b_ * psi[0] + c_ * diff(V2)
                I1[s0:s0 + n] += k1[s0:s0 + n] * psi[0]
                psi[1] = b_ * psi[1] + c_ * diff(V1)
                I2[s0:s0 + n] -= k2[s0:s0 + n] * psi[1]

    def pml_h(self):
        self._pml('H')

    def pml_e(self):
        self._pml('E')

    def mur_pre(self):
        if self.op.mur is not None:
            v = self.V.reshape(-1)
            self.o0[:] = v[self.m0]
            self.o1[:] = v[self.m1]

    def mur_post(self):
        if self.op.mur is not None:
            v = self.V.reshape(-1)
            v[self.m0] = self.o1 + self.mk * (v[self.m1] - self.o0)

    def excite(self, s):
        if len(self.exc_idx):
            self.add(self.V, self.exc_idx, self.exc_amp, s)

    def lumped(self):
        v = self.V.reshape(-1)
        if self.op.par_L:
            v[self.pl_idx] -= self.pl_vi * self.pl_I
            self.pl_I += self.pl_k * v[self.pl_idx]
        if self.op.series:
            v[self.se_idx] -= self.se_vi * self.se_I
            self.se_q += self.dtf * self.se_I
            self.se_I[:] = self.se_a * self.se_I + self.se_b * v[self.se_idx] - self.se_c * self.se_q

    def probe(self, slot):
        row = self.buf[slot]
        row[:] = 0
        for t, F in ((self.pv, self.V), (self.pi, self.I)):
            if t is not None:
                idx, w, pid = t
                np.add.at(row, pid, w * F.reshape(-1)[idx])

    def energy(self):
        """The energy (J) the end criterion watches: the field's and the lumped branches' (one host sync)."""
        return float(self.field_energy() + self.lumped_energy())

    def field_energy(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            e = np.where(self.vi > 0, self.V * self.V / self.vi, 0).sum(dtype=np.float64)
            e += np.where(self.iv > 0, self.I * self.I / self.iv, 0).sum(dtype=np.float64)
        return e * self.op.dt / 2

    def lumped_energy(self):
        """The energy in the lumped branches (J, a scalar on the backend's device): L I^2 / 2 in the inductors,
        q^2 / 2C in the series capacitors. Not in the field: a lumped LC filter rings in its branches."""
        e = 0.0
        if self.op.par_L:
            i = self.pl_I.astype(np.float64)
            e = e + self.pl_L @ (i * i)
        if self.op.series:
            i, q = self.se_I.astype(np.float64), self.se_q.astype(np.float64)
            e = e + self.se_L @ (i * i) + self.se_Cinv @ (q * q)
        return e / 2

    def fetch(self, n):
        return np.array(self.buf[:n], float)

    def sync(self):
        pass

    def to_host(self, a):
        """A backend array as numpy."""
        return np.asarray(a)

    def fields(self):
        """(V, I) as arrays of xp (the dumps without a kernel compute on them)."""
        return self.V, self.I

    def zeros(self, shape, dtype=np.float32):
        return self.xp.zeros(shape, dtype)

    def up(self, a, dtype=None):
        return self.xp.asarray(a, dtype)


class _Compressed:
    """Setup shared by the backends on the compressed operator: its tables on the device (xp)."""
    dense = False

    def _tables(self):
        from .operator import compress
        xp, op = self.xp, self.op
        self.cpr = c = compress(op)
        f32 = lambda a: xp.asarray(np.ascontiguousarray(a, np.float32))
        self.code = xp.asarray(c.code)
        self.tvv, self.ts, self.G = f32(c.vv.ravel()), f32(c.s.ravel()), f32(c.G)
        self.tpole = xp.asarray(np.ascontiguousarray(c.pole.ravel(), np.int32))
        self.P = int(c.P)
        self.tkb = f32(c.kbt.reshape(-1) if c.kbt.size else np.zeros(1))
        self.tka, self.th = f32(c.ka if c.P else np.zeros(1)), f32(c.h if c.P else np.zeros(1))
        self.slot = xp.asarray(c.slot)
        self.pol = xp.zeros(max(1, c.nslot * 3 * c.P), np.float32)
        self.ke, self.kh, self.dtf32 = np.float32(c.ke), np.float32(c.kh), np.float32(op.dt)
        self.idims = tuple(int(v) for v in op.shape)


class CoreBackend(_Compressed, NumpyBackend):
    """The compiled CPU core (gfdtd._core, OpenMP) on the compressed operator; the sparse extensions as numpy."""
    name = 'core'

    def __init__(self, op, dtype=np.float32):
        from . import _core
        if dtype != np.float32:
            raise ValueError("gfdtd: the CPU core is float32")
        self.core, self.op, self.dt, self.xp = _core, op, dtype, np
        if not os.environ.get('OMP_NUM_THREADS'):
            _core.set_threads(core_threads())
        self.V = np.zeros((3,) + op.shape, dtype)
        self.I = np.zeros((3,) + op.shape, dtype)
        self._tables()
        self._sparse()

    def update_H(self):
        self.core.update_h(self.V, self.I, self.G, *self.idims, self.kh)

    def update_E(self):
        self.core.update_e(self.V, self.I, self.code, self.tvv, self.ts, self.tpole, self.tkb, self.tka, self.th,
                           self.P, self.slot, self.pol, self.G, *self.idims, self.ke, self.dtf32)

    def _pml(self, f):
        update = self.core.pml_e if f == 'E' else self.core.pml_h
        for sl in self.pml:
            update(self.V, self.I, sl['psi' + f], sl['b' + f], sl['c' + f], sl[f'k{f}1'], sl[f'k{f}2'], *self.idims,
                   sl['axis'], sl[f.lower() + '0'], sl['n'])

    def mur_pre(self):
        if self.op.mur is not None:
            self.core.mur_pre(self.V, self.m0, self.m1, self.o0, self.o1)

    def mur_post(self):
        if self.op.mur is not None:
            self.core.mur_post(self.V, self.m0, self.m1, self.mk, self.o0, self.o1)

    def field_energy(self):
        return self.core.energy(self.V, self.I, self.code, self.ts, self.G, *self.idims, self.ke, self.kh) * \
            self.op.dt / 2


class CupyBackend(_Compressed, NumpyBackend):
    """The GPU (CuPy: CUDA or ROCm) on the compressed operator."""
    name = 'cupy'
    BLOCK = 128

    def __init__(self, op, dtype=np.float32):
        import cupy as cp
        if dtype != np.float32:
            raise ValueError("gfdtd: the GPU kernels are float32")
        self.op, self.dt, self.xp, self.cp = op, dtype, cp, cp
        self.V = cp.zeros((3,) + op.shape, dtype)
        self.I = cp.zeros((3,) + op.shape, dtype)
        self._tables()
        ctype = {np.dtype(np.uint8): 'unsigned char', np.dtype(np.uint16): 'unsigned short'}.get(self.code.dtype,
                                                                                                    'unsigned int')
        self.mod = cp.RawModule(code=kernels.source(ctype), options=('-std=c++17',))
        self.k = {n: self.mod.get_function(n) for n in kernels.NAMES}
        nx, ny, nz = op.shape
        self.n = nx * ny * nz
        if 3 * self.n >= 2 ** 31:                                   # the kernels decode nodes in 32-bit arithmetic
            raise ValueError(f"gfdtd: {self.n / 1e6:.0f} M nodes - the GPU kernels take up to 715 M")
        self.dims = (np.int32(nx), np.int32(ny), np.int32(nz))
        self.grid = self._blocks(self.n)
        self._sparse()
        self._pml_all()
        self._fused()
        self.e_out = cp.zeros(1, np.float64)

    def _blocks(self, m, most=65535 * 8):
        """The launch grid for m threads (grid-stride kernels: at most `most` blocks)."""
        return (min((m + self.BLOCK - 1) // self.BLOCK, most),)

    def _fused(self):
        """The arguments of the three-launch step (step_h, step_e, step_post). The post kernel runs its excitation,
        lumped and Mur entries at once, so it needs their edges disjoint (and Mur's inner edges apart from the
        others); where they are not, the step runs the separate kernels."""
        cp, op = self.cp, self.op
        z64, z32 = cp.zeros(1, np.int64), cp.zeros(1, np.float32)
        get = lambda name, d: getattr(self, name, d)
        self.n_ex = len(self.exc_idx)
        self.n_pl = len(self.pl_idx) if op.par_L else 0
        self.n_se = len(self.se_idx) if op.series else 0
        self.n_mu = len(self.m0) if op.mur is not None else 0
        self.fused = _post_disjoint(op)
        pE, pH = getattr(self, 'pmlE', None), getattr(self, 'pmlH', None)
        self.ns = np.int32(len(self.pml))
        self.hE = (pE['psi'], pE['b'], pE['c'], pE['desc']) if pE else (z32, z32, z32, z64)
        self.hH = (pH['psi'], pH['b'], pH['c'], pH['desc']) if pH else (z32, z32, z32, z64)

        def box(key):                                               # the PML-free interior: lines [lo, hi) per axis
            lo, hi = [0, 0, 0], list(op.shape)
            for sl in self.pml:
                a, s0 = sl['axis'], sl[key]
                if sl['hi']:
                    hi[a] = min(hi[a], s0)
                else:
                    lo[a] = max(lo[a], s0 + sl['n'])
            return cp.asarray(np.array(lo + hi, np.int32))
        self.boxE, self.boxH = box('e0'), box('h0')
        self.mu = (get('m0', z64), get('m1', z64), get('mk', z32), get('o0', z32), get('o1', z32))
        self.pl = (get('pl_vi', z32), get('pl_idx', z64), get('pl_k', z32), get('pl_I', z32))
        self.se = (get('se_vi', z32), get('se_idx', z64), get('se_a', z32), get('se_b', z32), get('se_c', z32),
                   get('se_I', z32), get('se_q', z32))
        self.n_post = self.n_ex + self.n_pl + self.n_se + self.n_mu
        self.mode = 'fused' if self.fused else 'separate'
        if os.environ.get('GFDTD_STEP') in ('fused', 'split', 'separate'):         # for checks: one way only
            self.mode = os.environ['GFDTD_STEP'] if self.fused else 'separate'
        elif self.fused and self.pml:
            self.mode = self._pick_mode()

    def _pick_mode(self, seconds=0.01, rounds=3):
        """PML in the H / E kernels ('fused') or as its own launch per field ('split'): whichever runs faster here
        (the best of a few rounds of `seconds` each, the modes alternating). In-kernel PML won on small models
        (fewer launches) and on large ones (no second pass over the slabs), and lost at ~1 M cells, where the fields
        sit in the GPU's cache and the bigger kernels' occupancy counts more. The trial steps change nothing: the
        fields are still zero and nothing excites them."""
        best = {}
        for _ in range(rounds):
            for mode in ('fused', 'split'):
                self.mode = mode
                self.step(None)
                self.sync()
                n, t0 = 0, time.perf_counter()
                while True:
                    for _ in range(10):
                        self.step(None)
                    n += 10
                    self.sync()
                    dt = time.perf_counter() - t0
                    if dt >= seconds:
                        break
                best[mode] = min(best.get(mode, dt / n), dt / n)
        return min(best, key=best.get)

    def step(self, sig=None, n=None):
        """One timestep in three launches: H with its PML and the Mur copies, E with the Debye poles and its PML,
        then the excitation, lumped branches and Mur update together (or separately: see _fused). The H excitations
        and the delayed ones (n: the step) run between them."""
        if self.mode == 'separate':
            return NumpyBackend.step(self, sig, n)
        split = self.mode == 'split'
        ns = np.int32(0) if split else self.ns
        psi, b, c, desc = self.hH
        m0, m1, mk, o0, o1 = self.mu
        self.k['step_h'](self._blocks(self.n + self.n_mu), (self.BLOCK,), (self.V, self.I, self.G, *self.dims, self.kh, psi, b, c,
                                                                  desc, ns, self.boxH, self.code, self.ts, m0, m1,
                                                                  o0, o1, np.int64(self.n_mu)))
        if split:
            self._pml('H')
        if self.more and n is not None:
            self.excite_more('I', n)
        psi, b, c, desc = self.hE
        self.k['step_e'](self.grid, (self.BLOCK,), (self.V, self.I, self.code, self.tvv, self.ts, self.tpole, self.tkb,
                                                    self.tka, self.th, np.int32(self.P), self.slot, self.pol, self.G,
                                                    *self.dims, self.ke, self.dtf32, psi, b, c, desc, ns,
                                                    self.boxE))
        if split:
            self._pml('E')
        if self.more and n is not None:
            self.excite_more('V', n)
        n_ex = self.n_ex if sig is not None else 0
        m = n_ex + self.n_pl + self.n_se + self.n_mu
        if m:
            self.k['step_post'](self._blocks(m), (self.BLOCK,), (
                self.V, self.exc_idx, self.exc_amp, np.float32(sig or 0.0), np.int64(n_ex),
                *self.pl, np.int64(self.n_pl), *self.se, self.dtf if self.n_se else np.float32(0),
                np.int64(self.n_se), m0, m1, mk, o0, o1, np.int64(self.n_mu)))

    def _launch(self, name, n, *args):
        if n:
            self.k[name](((n + self.BLOCK - 1) // self.BLOCK,), (self.BLOCK,), args)

    def update_H(self):
        self.k['update_h'](self.grid, (self.BLOCK,), (self.V, self.I, self.G, *self.dims, self.kh))

    def update_E(self):
        self.k['update_e'](self.grid, (self.BLOCK,), (self.V, self.I, self.code, self.tvv, self.ts, self.tpole,
                                                      self.tkb, self.tka, self.th, np.int32(self.P), self.slot,
                                                      self.pol, self.G, *self.dims, self.ke, self.dtf32))

    def _pml_all(self):
        """The slabs concatenated for the one-launch PML kernels: per field psi, b, c, k1, k2 and the slab table."""
        cp = self.cp
        self.pml_M = sum(sl['m'] for sl in self.pml)
        if not self.pml:
            return
        desc, st, bo = [], 0, 0
        for sl in self.pml:
            desc.append([st, sl['m'], sl['axis'], 0, sl['n'], bo])
            st, bo = st + sl['m'], bo + sl['n']
        cat = lambda k: cp.concatenate([sl[k] for sl in self.pml])
        self.pmlE = dict(psi=cat('psiE'), b=cat('bE'), c=cat('cE'), k1=cat('kE1'), k2=cat('kE2'),
                         desc=self._desc(desc, 'e0'))
        self.pmlH = dict(psi=cat('psiH'), b=cat('bH'), c=cat('cH'), k1=cat('kH1'), k2=cat('kH2'),
                         desc=self._desc(desc, 'h0'))

    def _desc(self, desc, key):
        d = np.array(desc, np.int64)
        d[:, 3] = [sl[key] for sl in self.pml]
        return self.cp.asarray(d.ravel())

    def _pml(self, field):
        if not self.pml_M:
            return
        p = self.pmlE if field == 'E' else self.pmlH
        self.k['pml_e' if field == 'E' else 'pml_h'](self._blocks(self.pml_M), (self.BLOCK,), (
            self.V, self.I, p['psi'], p['b'], p['c'], p['k1'], p['k2'], *self.dims, p['desc'], np.int32(len(self.pml)),
            np.int64(self.pml_M)))

    def mur_pre(self):
        if self.op.mur is not None:
            self._launch('mur_pre', len(self.m0), self.V, self.m0, self.m1, self.o0, self.o1, np.int64(len(self.m0)))

    def mur_post(self):
        if self.op.mur is not None:
            self._launch('mur_post', len(self.m0), self.V, self.m0, self.m1, self.mk, self.o0, self.o1,
                         np.int64(len(self.m0)))

    def add(self, F, idx, amp, val):
        self._launch('excite', len(idx), F, idx, amp, np.float32(val), np.int64(len(idx)))

    def assign(self, F, idx, amp, val):
        self._launch('assign', len(idx), F, idx, amp, np.float32(val), np.int64(len(idx)))

    def lumped(self):
        if self.op.par_L:
            n = len(self.pl_idx)
            self._launch('par_l', n, self.V, self.pl_vi, self.pl_idx, self.pl_k, self.pl_I, np.int64(n))
        if self.op.series:
            n = len(self.se_idx)
            self._launch('series_rlc', n, self.V, self.se_vi, self.se_idx, self.se_a, self.se_b, self.se_c,
                         self.se_I, self.se_q, self.dtf, np.int64(n))

    def probe(self, slot):
        row = self.buf[slot]
        row.fill(0)
        for t, F in ((self.pv, self.V), (self.pi, self.I)):
            if t is not None:
                idx, w, pid = t
                self._launch('probe', len(idx), F, idx, w, pid, row, np.int64(len(idx)))

    def field_energy(self):
        self.e_out.fill(0)
        self.k['energy'](self._blocks(self.n, 4096), (self.BLOCK,), (self.V, self.I, self.code, self.ts, self.G, *self.dims, self.ke, self.kh,
                                            self.e_out))
        return self.e_out[0] * (self.op.dt / 2)

    def fetch(self, n):
        return self.cp.asnumpy(self.buf[:n]).astype(float)

    def sync(self):
        self.cp.cuda.get_current_stream().synchronize()

    def to_host(self, a):
        return self.cp.asnumpy(a)

    def dump_kernel(self, region, out, nf, t):
        m = int(np.prod(region.shape)) * 3
        self.k['dump'](((m + self.BLOCK - 1) // self.BLOCK,), (self.BLOCK,), (self.V, self.I, self.G, *self.dims,
                                                                              *region.args, np.int32(nf),
                                                                              region.omega, np.float32(t), out))


class _Dev:
    """A device array of the Vulkan backend: its buffer, address and length (len() as a numpy array's)."""
    __slots__ = ('buf', 'addr', 'n', 'dtype', 'shape')

    def __init__(self, buf, n, dtype, shape=None):
        self.buf, self.addr, self.n, self.dtype = buf, buf.device_address, int(n), np.dtype(dtype)
        self.shape = shape if shape is not None else (self.n,)

    def __len__(self):
        return self.n


class VulkanBackend(_Compressed, NumpyBackend):
    """The GPU through Vulkan (slangpy: Slang kernels, kernels_vk.py) on the compressed operator: AMD on Windows,
    where CuPy has no ROCm build, and any other Vulkan GPU. The same steps as the CuPy backend (fused / split /
    separate), recorded into command buffers that are submitted every SUBMIT launches and at each host read, so
    the host records while the GPU runs."""
    name = 'vulkan'
    BLOCK = 64                  # threads a workgroup: 64 ran 1-4 % faster than 128 / 256 on an RX 7800 XT
    SUBMIT = 96                 # launches per command buffer, at most
    FIRST = 6                   # the first after a sync: the GPU starts while the host records the rest (it is
                                # idle after each sync, the energy check's every CHECK_STEPS)
    NBLK = 1024                 # the energy kernel's workgroups
    IDX = np.int32              # the kernels' index type (kernels_vk.source idx: int); GFDTD_VK_IDX=64 for int64

    def __init__(self, op, dtype=np.float32):
        from emsdev import vulkan
        from . import kernels_vk
        if dtype != np.float32:
            raise ValueError("gfdtd: the GPU kernels are float32")
        try:
            vulkan.device(need=VULKAN_NEEDS)
            self.dev = vulkan.slang_device()
        except vulkan.Unavailable as e:
            raise RuntimeError(f"gfdtd: no Vulkan GPU: {e}")
        import slangpy as spy
        self.spy, self.op, self.dt, self.xp = spy, op, dtype, np
        nx, ny, nz = op.shape
        self.n = nx * ny * nz
        if 3 * self.n >= 2 ** 31:
            raise ValueError(f"gfdtd: {self.n / 1e6:.0f} M nodes - the GPU kernels take up to 715 M")
        self._keep, self.enc, self.launches, self.batch = [], None, 0, self.FIRST
        self.usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | \
            spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
        self.max_groups = int(self.dev.info.limits.max_compute_dispatch_thread_groups[0])
        self._tables()                                              # host arrays (xp = numpy), uploaded below
        self._sparse()
        ctype = {np.dtype(np.uint8): 'uint8_t', np.dtype(np.uint16): 'uint16_t'}.get(np.dtype(self.code.dtype),
                                                                                    'uint32_t')
        if os.environ.get('GFDTD_VK_IDX') == '64':
            self.IDX = np.int64
        self.BLOCK = int(os.environ.get('GFDTD_VK_BLOCK') or self.BLOCK)
        self.k = _vk_kernels(self.dev, ctype, self.BLOCK, 'int64_t' if self.IDX == np.int64 else 'int')
        f = {}                                                      # Model's fields: addresses, then values
        self.V = self.zeros((3,) + op.shape)
        self.I = self.zeros((3,) + op.shape)
        f.update(V=self.V, I=self.I, G=self.up(self.G), code=self.up(self.code), tvv=self.up(self.tvv),
                 ts=self.up(self.ts), tpole=self.up(self.tpole), tkb=self.up(self.tkb), tka=self.up(self.tka),
                 th=self.up(self.th), slot=self.up(self.slot), pol=self.up(self.pol))
        self._pml_all(f)
        self.exc_idx, self.exc_amp = self.up(self.exc_idx, self.IDX), self.up(self.exc_amp)
        f.update(ex_idx=self.exc_idx, ex_amp=self.exc_amp)
        for g in self.more:
            g['idx'], g['amp'] = self.up(g['idx'], self.IDX), self.up(g['amp'])
        if op.par_L:
            f.update(pl_vi=self.up(self.pl_vi), pl_idx=self.up(self.pl_idx, self.IDX), pl_k=self.up(self.pl_k),
                     pl_I=self.up(self.pl_I), pl_L=self.up(self.pl_L, np.float64))
        if op.series:
            f.update(se_vi=self.up(self.se_vi), se_idx=self.up(self.se_idx, self.IDX), se_a=self.up(self.se_a),
                     se_b=self.up(self.se_b), se_c=self.up(self.se_c), se_I=self.up(self.se_I),
                     se_q=self.up(self.se_q), se_L=self.up(self.se_L, np.float64),
                     se_Cinv=self.up(self.se_Cinv, np.float64))
        if op.mur is not None:
            f.update(m0=self.up(self.m0, self.IDX), m1=self.up(self.m1, self.IDX), mk=self.up(self.mk),
                     o0=self.up(self.o0), o1=self.up(self.o1))
        for key, t in (('pv', self.pv), ('pi', self.pi)):
            if t is not None:
                f.update({key + '_idx': self.up(t[0], self.IDX), key + '_w': self.up(t[1]),
                          key + '_pid': self.up(t[2], self.IDX)})
        from emsdev.vulkan import HostBuffer
        self.hpart = HostBuffer(self.dev, 8 * (self.NBLK + 1))    # the energy's partial sums, written there
        self._hosts = [self.hpart]
        self.buf = self.zeros(self.buf.shape)
        f.update(part=_Addr(self.hpart.address), buf=self.buf)
        self.n_ex = len(op.exc[0])
        self.n_pl = len(op.par_L) if op.par_L else 0
        self.n_se = len(op.series) if op.series else 0
        self.n_mu = len(op.mur[0]) if op.mur is not None else 0
        vals = dict(nx=nx, ny=ny, nz=nz, P=self.P, ns=len(self.pml), n_ex=self.n_ex, n_pl=self.n_pl, n_se=self.n_se,
                    n_mu=self.n_mu, n_pv=len(self.pv[0]) if self.pv is not None else 0,
                    n_pi=len(self.pi[0]) if self.pi is not None else 0, np_=self.buf.shape[1],
                    pml_M=self.pml_M, pad=0, kh=float(self.kh), ke=float(self.ke), dt=float(self.dtf32),
                    dtf=float(getattr(self, 'dtf', 0.0)))
        rec = np.zeros(1, _model_dtype(kernels_vk.MODEL))
        for name, kind in kernels_vk.MODEL:
            rec[name] = f[name].addr if kind == 'p' and name in f else (0 if kind == 'p' else vals[name])
        self.model = self.up(rec.view(np.uint8))
        self.fused = _post_disjoint(op)
        self.mode = 'fused' if self.fused else 'separate'
        if os.environ.get('GFDTD_STEP') in ('fused', 'split', 'separate'):
            self.mode = os.environ['GFDTD_STEP'] if self.fused else 'separate'
        elif self.fused and self.pml:
            self.mode = self._pick_mode()

    # ---- device memory -----------------------------------------------------------------------------------------
    def _buffer(self, nbytes, data=None):
        size = max(-(-int(nbytes) // 16) * 16, 16)                  # whole 16 bytes, the data padded to them
        if data is not None and data.nbytes != size:
            pad = np.zeros(size, np.uint8)
            pad[:data.nbytes] = np.ascontiguousarray(data).view(np.uint8).ravel()
            data = pad
        try:
            b = self.dev.create_buffer(size=size, usage=self.usage)
        except Exception as e:                                      # past the driver's largest buffer, out of memory
            raise MemoryError(f"gfdtd: a {nbytes / 1e9:.2f} GB GPU buffer: {e}")
        if data is not None:
            b.copy_from_numpy(data)
        self._keep.append(b)
        return b

    def up(self, a, dtype=None):
        """A host array on the device (a copy)."""
        a = np.ascontiguousarray(np.asarray(a) if dtype is None else np.asarray(a, dtype))
        if a.size == 0:
            a = np.zeros(4, a.dtype)
            return _Dev(self._buffer(16, a), 0, a.dtype)
        return _Dev(self._buffer(a.nbytes, a), a.size, a.dtype, a.shape)

    def zeros(self, shape, dtype=np.float32):
        shape = (shape,) if np.isscalar(shape) else tuple(shape)
        n = int(np.prod(shape))
        d = _Dev(self._buffer(n * np.dtype(dtype).itemsize), n, dtype, shape)
        self._clear(d)
        return d

    def _clear(self, d):
        self._encoder().clear_buffer(d.buf)

    def to_host(self, d):
        """A device array as numpy (a host read: everything recorded so far runs first). Through mapped host memory,
        filled by a kernel (copy_words)."""
        if not isinstance(d, _Dev):
            return np.asarray(d)
        nbytes = d.n * d.dtype.itemsize
        h = self._host_stage(nbytes)
        self._launch('copy_words', -(-nbytes // 4), src_w=d.addr, dst_w=h.address, nw_=-(-nbytes // 4))
        self.sync()
        return h.array[:nbytes].view(d.dtype).reshape(d.shape).copy()

    def _host_stage(self, nbytes):
        from emsdev.vulkan import HostBuffer
        h = getattr(self, '_stage', None)
        if h is None or h.nbytes < nbytes:
            if h is not None:
                h.free()
            h = self._stage = HostBuffer(self.dev, max(nbytes, 1 << 16))
            self._hosts.append(h)
        return h

    def fields(self):
        """(V, I) as numpy, for what has no kernel (dumps of Jc)."""
        return self.to_host(self.V), self.to_host(self.I)

    # ---- launches --------------------------------------------------------------------------------------------
    def _encoder(self):
        if self.enc is None:
            self.enc = self.dev.create_command_encoder()
        return self.enc

    def _launch(self, name, m, **vars):
        """Kernel `name` over m items (a grid-stride loop), recorded after a barrier on what came before."""
        m = int(m)
        if m <= 0:
            return
        groups = min((m + self.BLOCK - 1) // self.BLOCK, self.max_groups)
        enc = self._encoder()
        p = enc.begin_compute_pass()                                # the pass and cursor directly: half the host
        cur = self.spy.ShaderCursor(p.bind_pipeline(self.k[name]))  # time of ComputeKernel.dispatch(vars=...)
        cur['M'] = self.model.addr
        cur['stride'] = groups * self.BLOCK
        for key, v in vars.items():
            cur[key] = v
        p.dispatch_compute([groups, 1, 1])
        p.end()
        enc.global_barrier()                                        # the kernels reach memory by address: no
                                                                    # resource tracking, so a barrier each
        self.launches += 1
        if self.launches >= self.batch:
            self._submit()
            self.batch = min(2 * self.batch, self.SUBMIT)

    def _submit(self):
        if self.enc is not None:
            self.dev.submit_command_buffer(self.enc.finish())
            self.enc, self.launches = None, 0

    def sync(self):
        self._submit()
        self.dev.wait_for_idle()
        self.batch = self.FIRST

    # ---- the step --------------------------------------------------------------------------------------------
    _pick_mode = CupyBackend._pick_mode

    def step(self, sig=None, n=None):
        """One timestep in three launches (as CupyBackend.step), or the separate kernels."""
        if self.mode == 'separate':
            return NumpyBackend.step(self, sig, n)
        split = self.mode == 'split'
        ns = 0 if split else len(self.pml)
        self._launch('step_h', self.n + self.n_mu, ns_=ns)
        if split:
            self._pml('H')
        if self.more and n is not None:
            self.excite_more('I', n)
        self._launch('step_e', self.n, ns_=ns)
        if split:
            self._pml('E')
        if self.more and n is not None:
            self.excite_more('V', n)
        n_ex = self.n_ex if sig is not None else 0
        self._launch('step_post', n_ex + self.n_pl + self.n_se + self.n_mu, n_ex_=n_ex, s_=float(sig or 0.0))

    def _pml_all(self, f):
        """The slabs concatenated for the one-launch PML kernels (as CupyBackend._pml_all), into Model's fields."""
        self.pml_M = sum(sl['m'] for sl in self.pml)
        if not self.pml:
            return
        desc, st, bo = [], 0, 0
        for sl in self.pml:
            desc.append([st, sl['m'], sl['axis'], 0, sl['n'], bo])
            st, bo = st + sl['m'], bo + sl['n']
        for F in 'EH':
            d = np.array(desc, self.IDX)
            d[:, 3] = [sl[F.lower() + '0'] for sl in self.pml]
            cat = lambda k: np.concatenate([sl[k] for sl in self.pml])
            f.update({'psi' + F: self.zeros(2 * self.pml_M), 'b' + F: self.up(cat('b' + F)),
                      'c' + F: self.up(cat('c' + F)), 'k1' + F: self.up(cat(f'k{F}1')),
                      'k2' + F: self.up(cat(f'k{F}2')), 'desc' + F: self.up(d.ravel(), self.IDX)})
            lo, hi = [0, 0, 0], list(self.op.shape)                 # the PML-free interior: lines [lo, hi) per axis
            for sl in self.pml:
                a, s0 = sl['axis'], sl[F.lower() + '0']
                if sl['hi']:
                    hi[a] = min(hi[a], s0)
                else:
                    lo[a] = max(lo[a], s0 + sl['n'])
            f['box' + F] = self.up(np.array(lo + hi, np.int32))

    def _pml(self, field):
        self._launch('pml_e' if field == 'E' else 'pml_h', self.pml_M)

    def update_H(self):
        self._launch('update_h', self.n)

    def update_E(self):
        self._launch('update_e', self.n)

    def mur_pre(self):
        self._launch('mur_pre', self.n_mu)

    def mur_post(self):
        self._launch('mur_post', self.n_mu)

    def excite(self, s):
        self.add(self.V, self.exc_idx, self.exc_amp, s)

    def add(self, F, idx, amp, val):
        self._launch('excite', len(idx), F_=F.addr, idx_=idx.addr, amp_=amp.addr, s_=float(val), m_=len(idx))

    def assign(self, F, idx, amp, val):
        self._launch('assign', len(idx), F_=F.addr, idx_=idx.addr, amp_=amp.addr, s_=float(val), m_=len(idx))

    def lumped(self):
        self._launch('par_l', self.n_pl)
        self._launch('series_rlc', self.n_se)

    def probe(self, slot):
        """One sample into buffer row slot (rows are zeroed after each fetch)."""
        self._launch('probe', self.np_terms, slot_=int(slot))

    @property
    def np_terms(self):
        return (len(self.pv[0]) if self.pv is not None else 0) + (len(self.pi[0]) if self.pi is not None else 0)

    def energy(self):
        """The field's and the lumped branches' energy (one host read)."""
        self._launch('energy', self.NBLK * self.BLOCK, nblk_=self.NBLK)
        self.sync()
        p = self.hpart.view(np.float64, self.NBLK + 1)
        return float(p[:self.NBLK].sum() * (self.op.dt / 2) + p[self.NBLK] / 2)

    def fetch(self, n):
        rows = self.to_host(self.buf)[:n].astype(float)
        self._clear(self.buf)
        return rows

    # ---- dumps (dumps.Region on the GPU) ---------------------------------------------------------------------
    def dump_kernel(self, region, out, nf, t):
        a = region.args
        self._launch('dump', int(np.prod(region.shape)) * 3, field_=int(a[0]), cell_=int(a[1]), lx_=int(a[2]),
                     ly_=int(a[3]), lz_=int(a[4]), sx_=int(a[5]), sy_=int(a[6]), sz_=int(a[7]), nf_=int(nf),
                     omega_=region.omega.addr, tm_=float(t), dst_=out.addr)


class _Addr:
    """A device address as Model's fields take it (mapped host memory)."""
    def __init__(self, addr):
        self.addr = addr


def _model_dtype(fields):
    return np.dtype([(n, '<u8' if k == 'p' else ('<i4' if k == 'i' else '<f4')) for n, k in fields])


_vk_local = None


def _vk_kernels(dev, ctype, block, idx='int'):
    """The Vulkan kernels' pipelines for a device and code type, compiled once per thread (as the device: emsdev's
    slang_device is one per thread; slangpy keeps the SPIR-V on disk too). Kept thread-local, so that a thread's
    device and pipelines go with it."""
    global _vk_local
    import threading
    from . import kernels_vk
    if _vk_local is None:
        _vk_local = threading.local()
    cache = getattr(_vk_local, 'cache', None)
    if cache is None:
        cache = _vk_local.cache = {}
    key = (id(dev), ctype, block, idx)
    if key not in cache:
        mod = dev.load_module_from_source(f"gfdtd_{ctype}_{block}_{idx}", kernels_vk.source(ctype, block, idx))
        cache[key] = (dev, {n: dev.create_compute_pipeline(dev.link_program([mod], [mod.entry_point(n)]))
                            for n in kernels_vk.NAMES})
    return cache[key][1]


def _post_disjoint(op):
    """Whether the post kernel may run the excitation, lumped and Mur entries at once: their edges disjoint (and
    Mur's inner edges apart from the others)."""
    sets = [np.asarray(op.exc[0]), np.array([e for e, _ in op.par_L], np.int64),
            np.array([e[0] for e in op.series], np.int64)]
    mur = [op.mur[0], op.mur[1]] if op.mur is not None else []
    flat = np.concatenate([np.unique(x) for x in sets] + [np.unique(np.concatenate(mur))] if mur else
                          [np.unique(x) for x in sets])
    dup = len(np.unique(flat)) < len(flat) or len(set(e for e, _ in op.par_L)) < len(op.par_L) or \
        len(set(e[0] for e in op.series)) < len(op.series)
    return not dup


def make_backend(op, backend='auto', dtype=np.float32):
    name = resolve_backend(backend)
    return {'cupy': CupyBackend, 'vulkan': VulkanBackend, 'core': CoreBackend, 'numpy': NumpyBackend}[name](op, dtype)


def _fmt_t(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s" if s >= 3600 else (
        f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s")


class Engine:
    def __init__(self, op, backend='auto', dtype=np.float32):
        self.op = op
        self.be = make_backend(op, backend, dtype)

    def run(self, path, nr_ts, end_crit, verbose=1, out=print):
        """Run up to nr_ts timesteps in folder `path` (probe files written there), ending when the energy (the
        field's and the lumped branches', backend.energy) falls below end_crit times its peak after the excitation. Returns dict(steps, seconds, reason, db)."""
        op, be = self.op, self.be
        dt, sig, sub = op.dt, op.signal, op.sub
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'et'), 'w', encoding='utf-8') as fh:
            fh.write("% time-domain excitation signal by gfdtd\n% t/s\tvoltage\n")
            fh.writelines(f"{n * dt:.12g}\t{v:.12g}\n" for n, v in enumerate(sig))
        files = []
        for p in be.order:                                          # the probe files, with openEMS's headers
            fh = open(os.path.join(path, p['name']), 'w', encoding='utf-8')
            files.append(fh)
            what, u = 'voltage' if p['field'] == 'V' else 'current', p['unit']
            fh.write(f"% time-domain {what} integration by gfdtd @{time.ctime()}\n")
            for end, (xyz, ijk) in (('start', p['start']), ('stop', p['stop'])):
                fh.write(f"% {end}-coordinates: ({','.join(f'{v * u:g}' for v in xyz)}) m -> [{','.join(map(str, ijk))}]\n")
            fh.write(f"% t/s\t{what}\n")
        dumps = Dumps(op, be, path, op.notes)
        pending = []                                                # (step, slot) samples in the device buffer

        def flush():
            if not pending:
                return
            rows = be.fetch(len(pending))
            for c, (p, fh) in enumerate(zip(be.order, files)):
                half = 0.5 if p['field'] == 'I' else 0.0            # I sits half a step before V (its first
                                                                    # sample at -dt / 2: readers take dt from t1 - t0)
                fh.writelines(f"{(st - half) * dt:.12g}\t{rows[r, c]:.12g}\n" for r, st in enumerate(pending))
                fh.flush()
            pending.clear()

        be.probe(0)
        pending.append(0)
        if dumps:
            dumps.sample(0)
        abort = os.path.join(path, 'ABORT')
        t0 = last = time.time()
        emax, e, reason, s, last_s = 0.0, 0.0, 'steps', 0, 0
        nexc = len(sig)
        nr_ts = int(nr_ts)
        try:
            while s < nr_ts:
                s += 1
                be.step(sig[s] if s < nexc else None, s)
                if s % sub == 0:
                    be.probe(len(pending))
                    pending.append(s)
                    if dumps:
                        dumps.sample(s)
                    if len(pending) == CHUNK:
                        flush()
                if s % CHECK_STEPS == 0 or s == nr_ts:
                    e = be.energy()
                    emax = max(emax, e)
                    now = time.time()
                    if verbose and now - last >= STATUS_S:
                        flush()
                        db = 10 * np.log10(e / emax) if emax > 0 and e > 0 else float('-inf')
                        rate = op.cells * (s - last_s) / (now - last) / 1e6
                        out(f"[@{_fmt_t(now - t0):>10}] Timestep: {s:>12d} || Speed: {rate:6.1f} MC/s "
                            f"({(now - last) / max(s - last_s, 1):.3e} s/TS) || Energy: ~{e:.2e} ({db:.2f}dB)")
                        last, last_s = now, s
                    if not np.isfinite(e):
                        reason = 'unstable'
                        break
                    if s > nexc + getattr(op, 'delay_max', 0) and emax > 0 and e < end_crit * emax:
                        reason = 'end'
                        break
                    if os.path.exists(abort):
                        reason = 'abort'
                        break
            be.sync()
            flush()
            written = dumps.close()
        finally:
            for fh in files:
                fh.close()
        sec = time.time() - t0
        db = 10 * np.log10(e / emax) if emax > 0 and e > 0 else float('-inf')
        if reason == 'end':
            out(f"RunFDTD: end-criteria of {10 * np.log10(end_crit):.2f}dB reached after {s} timesteps ({db:.2f}dB)")
        elif reason == 'abort':
            out(f"RunFDTD: aborted (ABORT file) after {s} timesteps ({db:.2f}dB)")
        elif reason == 'unstable':
            out(f"RunFDTD: WARNING: the fields grew without bound after {s} timesteps - unstable")
        else:
            out(f"RunFDTD: max. number of timesteps ({nr_ts}) reached ({db:.2f}dB)")
        out(f"Time for {s} iterations with {op.cells:.2f} cells : {sec:.2f} sec")
        out(f"Speed: {op.cells * s / max(sec, 1e-9) / 1e6:.2f} MCells/s ")
        return dict(steps=s, seconds=sec, reason=reason, db=db, dumps=written)
