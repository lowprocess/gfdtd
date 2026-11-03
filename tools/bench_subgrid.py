"""
Local mesh refinement, compared on one model (gfdtd.experimental.subgrid: numpy, float64, 1 thread):

    python tools/bench_subgrid.py accuracy [--size 1] [--ns 15]   # resonances and cost of four meshes
    python tools/bench_subgrid.py stability [--steps 60000]         # late-time energy of the couplings

accuracy: a closed PEC cavity (24 x 16 x 8 mm, x size in plan) with a square 1 x 1 mm post from the floor to
1/3 mm below the ceiling: a re-entrant, capacitively loaded resonator, whose gap sets the lowest mode. Meshes:

    fine      uniform 1/3 mm everywhere (the reference)
    coarse    uniform 1 mm; the gap can only be 1 mm
    graded    1/3 mm across the post and the gap, growing by 1.5 a cell to 1 mm (gfdtd's kind of mesh)
    subgrid   uniform 1 mm with a 7 x 7 x 8 mm box around the post refined 3:1 (adjoint coupling, one dt)

A soft Ez source and an Ez probe sit in the coarse region, so the subgrid's interface lies between them and the
post. The resonances are the peaks of the probe's spectrum (Hann window after the pulse, parabolic peak), each
matched to the nearest reference peak.

stability: a 20 x 16 x 12 mm cavity with a 6 x 6 x 4 mm box inside it (all six faces are interfaces), with and
without a 1 mm PEC cube in the box. Lossless and closed: after the pulse the energy should stay constant.
"""
import argparse, os, sys, time, warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np

from gfdtd.experimental.subgrid import Grid, Subgridded, SubgriddedOverlap, graded_lines

MM = 1e-3
TAU = 25e-12
T0 = 4 * TAU
GAP = 1 / 3


def pulse(t):
    """A Gaussian derivative (no DC), peak near 9 GHz, usable to ~20 GHz."""
    u = (t - T0) / TAU
    return -u * np.exp(-u * u)


def u_lines(lo, hi, h):
    return np.linspace(lo, hi, int(round((hi - lo) / h)) + 1) * MM


def model(size=1):
    """The cavity's plan (A, B), height, the post's corner, source and probe points (mm)."""
    A, B, H = 24 * size, 16 * size, 8
    px, py = A // 2, B // 2
    return dict(A=A, B=B, H=H, post=((px, py, 0), (px + 1, py + 1, H - GAP)),
                src=(4, 4, 3.5), prb=(A - 4, B - 4, 4.5), box=((px - 3, px + 4), (py - 3, py + 4), (0, H)),
                fine=((px - 1, px + 2), (py - 1, py + 2), (H - 1, H)))


def meshes(size=1):
    """{name: (solver, grid holding the source and probe)}"""
    m = model(size)
    A, B, H = m['A'], m['B'], m['H']
    lo, hi = (np.array(p) * MM for p in m['post'])
    out = {}
    g = Grid(u_lines(0, A, 1 / 3), u_lines(0, B, 1 / 3), u_lines(0, H, 1 / 3))
    g.add_pec_box(lo, hi)
    out['fine'] = (g, g)
    g = Grid(u_lines(0, A, 1), u_lines(0, B, 1), u_lines(0, H, 1))
    g.add_pec_box(lo, hi - np.array((0, 0, 1 - GAP)) * MM)          # the nearest a 1 mm mesh gets: a 1 mm gap
    out['coarse'] = (g, g)
    (fx0, fx1), (fy0, fy1), (fz0, fz1) = m['fine']
    g = Grid(graded_lines(0, A, fx0, fx1, 1 / 3, 1) * MM, graded_lines(0, B, fy0, fy1, 1 / 3, 1) * MM,
             graded_lines(0, H, fz0, fz1, 1 / 3, 1) * MM)
    g.add_pec_box(lo, hi)
    out['graded'] = (g, g)
    s = Subgridded(u_lines(0, A, 1), u_lines(0, B, 1), u_lines(0, H, 1), m['box'])
    s.fine.add_pec_box(lo, hi)
    out['subgrid'] = (s, s.coarse)
    return out, m


def peaks(t, v, fmin=1e9, fmax=20e9):
    """The clear peaks of v's spectrum (Hz), from the samples after the pulse."""
    keep = t > T0 + 4 * TAU
    t, v = t[keep], v[keep]
    v = (v - v.mean()) * np.hanning(len(v))
    dt = t[1] - t[0]
    N = 1 << 22
    S = np.abs(np.fft.rfft(v, N))
    f = np.fft.rfftfreq(N, dt)
    df = f[1] - f[0]
    w = int(3 / (t[-1] - t[0]) / df)                                # a peak is the largest within +- 3 / T
    band = np.where((f > fmin) & (f < fmax))[0]
    top = S[band].max()
    out = []
    for k in band[1:-1]:
        if S[k] < 1e-4 * top or S[k] < S[max(k - w, 0):k + w + 1].max():
            continue
        a, b, c = np.log(S[k - 1]), np.log(S[k]), np.log(S[k + 1])
        out.append(f[k] + 0.5 * (a - c) / (a - 2 * b + c) * df)
    return np.array(out)


def step_fn(solver):
    return solver.step if hasattr(solver, 'step') else (lambda src: (solver.update_H(), solver.update_E(),
                                                                      src(solver) if src else None))


def run(solver, grid, m, t_end):
    si, pi = grid.nearest(2, np.array(m['src']) * MM), grid.nearest(2, np.array(m['prb']) * MM)
    steps = int(np.ceil(t_end / solver.dt))
    rec = np.zeros(steps)
    step = step_fn(solver)
    a = time.perf_counter()
    for n in range(steps):
        tn = (n + 0.5) * solver.dt
        step(lambda g, tn=tn: g.Ez.__setitem__(si, g.Ez[si] + pulse(tn)))
        rec[n] = grid.Ez[pi]
    return np.arange(steps) * solver.dt, rec, steps, time.perf_counter() - a


def accuracy(t_end, size=1, nmodes=3):
    ms, m = meshes(size)
    res = {}
    print(f"cavity {m['A']} x {m['B']} x {m['H']} mm, {t_end * 1e9:g} ns", flush=True)
    for name, (solver, grid) in ms.items():
        t, v, steps, wall = run(solver, grid, m, t_end)
        res[name] = dict(f=peaks(t, v), steps=steps, wall=wall, cells=solver.cells, dt=solver.dt)
        print(f"  {name:8s} {solver.cells:7d} cells  dt {solver.dt * 1e12:6.3f} ps  {steps:6d} steps  {wall:7.1f} s"
              f"  f = " + ', '.join(f"{x / 1e9:.4f}" for x in res[name]['f'][:6]) + " GHz", flush=True)
    ref = res['fine']['f'][:nmodes]
    print(f"\n| mesh | cells | dt (ps) | steps | cell-updates | wall (s) | " +
          ' | '.join(f"f{i + 1} = {f / 1e9:.3f} GHz" for i, f in enumerate(ref)) + ' |')
    print('|---' * (6 + len(ref)) + '|')
    for name, r in res.items():
        errs = []
        for f in ref:
            near = r['f'][np.argmin(np.abs(r['f'] - f))] if len(r['f']) else np.nan
            errs.append(f"{(near - f) / f * 100:+.2f} %" if abs(near - f) < 0.25 * f else "missing")
        print(f"| {name} | {r['cells']:,} | {r['dt'] * 1e12:.3f} | {r['steps']:,} | "
              f"{r['cells'] * r['steps'] / 1e6:,.0f} M | {r['wall']:.0f} | " + ' | '.join(errs) + ' |', flush=True)
    return res


def stability(steps, every=250):
    """Each coupling's energy after the pulse, relative to its value just after the pulse, until `steps` (fine
    steps; a subcycled run takes a third as many coarse ones) or until the energy grows a million times."""
    warnings.simplefilter('ignore')
    lines = (u_lines(0, 20, 1), u_lines(0, 16, 1), u_lines(0, 12, 1))
    box = ((7, 13), (5, 11), (4, 8))
    cube = (np.array((9.5, 7.5, 5.5)) * MM, np.array((10.5, 8.5, 6.5)) * MM)

    def make(kind, with_cube):
        if kind == 'coarse alone':
            g = Grid(*lines, dt=Subgridded(*lines, box).dt)                 # the subgrid's step, for a like test
            return g, g
        s = (SubgriddedOverlap(*lines, box) if kind == 'overlap' else
             Subgridded(*lines, box, subcycle=(kind == 'adjoint, subcycled 3:1')))
        if with_cube:
            s.fine.add_pec_box(*cube)
        return s, s.coarse

    print("| coupling | box contents | steps run | W(end) / W(after pulse) | max / min | outcome | wall (s) |")
    print("|---|---|---|---|---|---|---|")
    for kind, with_cube in (('coarse alone', False), ('overlap', False), ('adjoint, subcycled 3:1', False),
                            ('adjoint, one dt', False), ('adjoint, one dt', True)):
        s, g = make(kind, with_cube)
        si = g.nearest(2, np.array((3, 3, 5.5)) * MM)
        n_steps = steps // 3 if getattr(s, 'subcycle', False) or kind == 'overlap' else steps
        step = step_fn(s)
        t_off = T0 + 6 * TAU
        W, w0, outcome = [], None, 'bounded'
        a = time.perf_counter()
        for n in range(n_steps):
            tn = (n + 0.5) * s.dt
            step((lambda g, tn=tn: g.Ez.__setitem__(si, g.Ez[si] + pulse(tn))) if tn < t_off else None)
            if n % every == 0 and tn > t_off:
                W.append(s.energy())
                if w0 is None:
                    w0 = W[0]
                if not np.isfinite(W[-1]) or W[-1] > 1e6 * w0:
                    outcome = f"diverged by step {n:,}"
                    break
        W = np.array(W)
        print(f"| {kind} | {'1 mm PEC cube' if with_cube else 'empty'} | {n + 1:,} | {W[-1] / w0:.4g} | "
              f"{W.max() / W.min():.4g} | {outcome} | {time.perf_counter() - a:.0f} |", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=('accuracy', 'stability'))
    ap.add_argument('--size', type=int, default=1, help="accuracy: the cavity's plan, times 24 x 16 mm")
    ap.add_argument('--ns', type=float, default=15, help="accuracy: simulated time, ns")
    ap.add_argument('--steps', type=int, default=60000, help="stability: fine steps")
    a = ap.parse_args()
    if a.what == 'accuracy':
        accuracy(a.ns * 1e-9, a.size)
    else:
        stability(a.steps)


if __name__ == '__main__':
    main()
