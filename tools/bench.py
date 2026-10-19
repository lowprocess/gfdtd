"""
Hot-loop benchmark of gfdtd's engine: the time per timestep of each operation, by backend and model size.

    python tools/bench_gfdtd.py [--backends cupy,core,numpy] [--sizes 0.1,1,4,16] [--steps 200] [--bc PML_8]

The model is a microstrip on a Debye substrate (one pole), two ports and a series RLC part, with PML (or --bc) on
five sides and PEC below: about N million cells, x : y : z = 4 : 2 : 1. "wall" is the engine's own step (the GPU's:
three fused launches) with a probe sample every 25 steps, synchronised once at the end. The breakdown below it
times each operation run on its own (the GPU: events around each kernel, which add a little each).
"""
import argparse, contextlib, io, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from gfdtd import ContinuousStructure, openEMS
from gfdtd.operator import build
from gfdtd.engine import make_backend, resolve_backend

OPS = ['update_H', 'pml_h', 'mur_pre', 'update_E', 'pml_e', 'excite', 'lumped', 'mur_post', 'probe']


def model(n, bc='PML_8'):
    F = openEMS(NrTS=1000, EndCriteria=1e-6)
    F.SetGaussExcite(5e9, 5e9)
    F.SetBoundaryCond([bc] * 4 + ['PEC', bc])
    C = ContinuousStructure()
    F.SetCSX(C)
    g = C.GetGrid()
    g.SetDeltaUnit(1e-3)
    s = (n * 1e6 / 8) ** (1 / 3)
    nx, ny, nz = int(4 * s), int(2 * s), int(s)
    nsub = max(4, nz // 4)
    g.SetLines('x', np.linspace(-20, 20, nx + 1))
    g.SetLines('y', np.linspace(-10, 10, ny + 1))
    g.SetLines('z', np.r_[np.linspace(0, 1, nsub + 1), np.linspace(1, 10, nz - nsub + 1)[1:]])
    m = C.AddDebyeMaterial('sub', order=1, epsilon=3.0, kappa=1e-3)
    m.SetDispersiveMaterialProperty(order=0, eps_delta=1.0, eps_relax=30e-12)
    m.AddBox([-15, -8, 0], [15, 8, 1], priority=1)
    C.AddMetal('cu').AddBox([-15, -1, 1], [15, 1, 1], priority=10)
    F.AddLumpedPort(1, 50, [-15, -1, 0], [-15, 1, 1], 'z', excite=1, priority=5)
    F.AddLumpedPort(2, 50, [15, -1, 0], [15, 1, 1], 'z', excite=0, priority=5)
    C.AddLumpedElement('rlc', ny='z', caps=True, R=10, L=1e-9, C=1e-12, LEtype=1).AddBox([0, 3, 0], [0, 3.5, 1],
                                                                                           priority=5)
    with contextlib.redirect_stdout(io.StringIO()):
        return build(C, F.settings())


def _call(n, f):
    return f(0) if n == 'probe' else (f(0.5) if n == 'excite' else f())


def bench(op, backend, steps):
    """({op: s per step}, wall s per step) of one backend."""
    be = make_backend(op, backend)
    calls = [(n, getattr(be, n)) for n in OPS]
    for _ in range(3):                                              # warm up: kernel compilation, caches
        for n, f in calls:
            _call(n, f)
    be.sync()
    t = dict.fromkeys(OPS, 0.0)
    if be.name == 'cupy':
        import cupy as cp
        ev = [cp.cuda.Event() for _ in range(len(OPS) + 1)]
        for _ in range(steps):
            ev[0].record()
            for i, (n, f) in enumerate(calls):
                _call(n, f)
                ev[i + 1].record()
            ev[-1].synchronize()
            for i, n in enumerate(OPS):
                t[n] += cp.cuda.get_elapsed_time(ev[i], ev[i + 1]) * 1e-3
    else:
        for _ in range(steps):
            for n, f in calls:
                a = time.perf_counter()
                _call(n, f)
                t[n] += time.perf_counter() - a
    be.sync()
    for _ in range(3):
        be.step(0.5)
    be.sync()
    t0 = time.perf_counter()
    for s in range(steps):
        be.step(0.5)
        if s % 25 == 0:
            be.probe(0)
    be.sync()
    return {n: v / steps for n, v in t.items()}, (time.perf_counter() - t0) / steps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backends', default='cupy,core,numpy')
    ap.add_argument('--sizes', default='0.1,1,4')
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--bc', default='PML_8')
    a = ap.parse_args()
    for n in [float(v) for v in a.sizes.split(',')]:
        op = model(n, a.bc)
        for b in a.backends.split(','):
            try:
                if resolve_backend(b) != b:
                    print(f"\n{b}: not available")
                    continue
                per, wall = bench(op, b, max(5, a.steps // (20 if b == 'numpy' else 1)))
            except Exception as e:                                  # no GPU, no core
                print(f"\n{b}: {type(e).__name__}: {e}")
                continue
            print(f"\n{b}: {op.cells / 1e6:.2f} M cells {op.shape}: {wall * 1e3:.3f} ms/step, "
                  f"{op.cells / wall / 1e6:.0f} M cells/s")
            tot = sum(per.values()) or 1
            for k in OPS:
                if per[k] > 0.005 * tot:
                    print(f"   {k:10s} {per[k] * 1e6:10.1f} us  {per[k] / tot * 100:5.1f} %")


if __name__ == '__main__':
    main()
