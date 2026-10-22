"""
The FDTD operator: a CSX model and the FDTD settings lowered onto the rectilinear Yee grid, as openEMS does it.

The fields are kept as edge voltages and currents (V = E dl on the primary edges, I = H dl' on the dual
edges), so the update needs no geometry at run time:

    I  -= iv * (loop sum of V around the primary face)        iv = dt l' / (mu A)
    V   = vv * V + vi * (loop sum of I around the dual face)  vv, vi from the edge's C = eps A' / l, G = kappa A' / l

Arrays are (3, Nx, Ny, Nz): component, then the node (i, j, k). E_x[i, j, k] is the edge from line i to i+1
at y_j, z_k; H_x[i, j, k] sits at x_i, y_j+1/2, z_k+1/2. Indices past the last edge hold zero coefficients.

    op = build(csx, fdtd)        # fdtd: dict(bc=[6], f0, fc, nr_ts, end_crit, oversampling)

Materials are sampled at cell centres (highest priority wins) and averaged over the four quarter cells around
each edge. Metals (Metal, ConductingSheet: both as PEC) take the edges whose midpoint is in a primitive; flat
primitives snap to the nearest mesh line. Lumped elements spread their R / L / C over the edges in their box
(parallel, or series RLC on each edge), with metal caps; soft E excitations add exc_val x signal to their edges.

Debye materials: each pole is a polarisation current i on the edges, tau di/dt + i = C_p dV/dt (C_p = eps0
eps_delta A' / l), integrated with the trapezoidal rule at the E update's time, which folds part of it into vv
and vi: unconditionally stable, for any tau. PML_n: the n outermost cells of a side are a convolutional PML
(CPML, kappa 1, no alpha) with a cubic conductivity grading, ending in PEC. Dumps: DumpBox regions at the cell
centres (openEMS's dump mode 2), worked out by the engine.
"""
import numpy as np

from . import csx as C

EPS0, MU0, C0 = 8.8541878128e-12, 1.25663706212e-6, 299792458.0
SAFETY = 0.95           # time step: this fraction of the stability limit (C0, the shortest neighbouring cells)
PML_GRADING = 3         # PML conductivity ~ depth^m
PML_R0 = 1e-6           # PML: the reflection of a normal wave through it and back, in theory


class Operator:
    """What the engine needs: the grid (m), dt, the coefficient arrays and the sparse extensions."""
    def __init__(self):
        self.notes = []


def _snap(lines, v):
    return float(lines[int(np.argmin(np.abs(lines - v)))])


def _dual(d):
    """Dual edge lengths at the N nodes from the N-1 primary lengths (half cells at the ends)."""
    out = np.zeros(len(d) + 1)
    out[:-1] += d / 2
    out[1:] += d / 2
    return out


def _flat(shape, a, i, j, k):
    return ((a * shape[0] + i) * shape[1] + j) * shape[2] + k


class Grid:
    """Lines (drawing units and metres), primary / dual lengths, and the points the tests run on."""
    def __init__(self, g):
        self.lines = [np.asarray(g.lines[a], float) for a in range(3)]
        for a in range(3):
            if len(self.lines[a]) < 2:
                raise ValueError(f"gfdtd: the grid needs at least two {C.AXES[a]} lines")
        self.unit = g.delta_unit
        self.m = [l * self.unit for l in self.lines]
        self.d = [np.diff(l) for l in self.m]
        self.dual = [_dual(d) for d in self.d]
        self.shape = tuple(len(l) for l in self.lines)
        self.mid = [0.5 * (l[1:] + l[:-1]) for l in self.lines]        # cell centres, drawing units
        self.tol = [1e-6 * max(np.min(np.diff(l)), 1e-12) for l in self.lines]

    def edge_points(self, a):
        """Coordinates (drawing units) of the midpoints of the a-edges: three 1-D arrays."""
        return [self.mid[b] if b == a else self.lines[b] for b in range(3)]

    def cell_points(self):
        return list(self.mid)


# ---- primitives: which points of a tensor grid are inside ------------------------------------------------------
def _range_mask(coord, lo, hi, tol):
    return (coord >= lo - tol) & (coord <= hi + tol)


def _poly_shape(q):
    from shapely.geometry import Polygon
    p = Polygon(np.asarray(q.points).T)
    return p if p.is_valid else p.buffer(0)


def inside(q, pts, grid, flat_snap=True):
    """Boolean mask (len X, len Y, len Z) of the tensor-grid points pts = [X, Y, Z] inside primitive q (boundary
    included). Flat primitives sit on their nearest mesh line (flat_snap)."""
    import shapely
    X, Y, Z = pts
    shape = (len(X), len(Y), len(Z))
    if isinstance(q, C.Polygon):                                    # Polygon / LinPoly, normal n
        n = q.norm_dir
        u, v = [b for b in range(3) if b != n]
        if isinstance(q, C.LinPoly) and abs(q.length) > 0:
            lo, hi = sorted((q.elevation, q.elevation + q.length))
        else:
            lo = hi = _snap(grid.lines[n], q.elevation) if flat_snap else q.elevation
        mn = _range_mask(pts[n], lo, hi, grid.tol[n])
        poly = _poly_shape(q)
        x0, y0, x1, y1 = poly.bounds
        mu = _range_mask(pts[u], x0, x1, grid.tol[u])
        mv = _range_mask(pts[v], y0, y1, grid.tol[v])
        out = np.zeros(shape, bool)
        if not (mn.any() and mu.any() and mv.any()):
            return out
        iu, iv = np.nonzero(mu)[0], np.nonzero(mv)[0]
        U, V = np.meshgrid(pts[u][iu], pts[v][iv], indexing='ij')
        inp = shapely.intersects_xy(poly, U.ravel(), V.ravel()).reshape(U.shape)
        plane = np.zeros((shape[u], shape[v]), bool)
        plane[np.ix_(iu, iv)] = inp
        return _along(plane, mn, n, shape)
    if isinstance(q, C.Box):
        m = []
        for a in range(3):
            lo, hi = sorted((q.start[a], q.stop[a]))
            if hi - lo < grid.tol[a] and flat_snap:
                lo = hi = _snap(grid.lines[a], lo)
            m.append(_range_mask(pts[a], lo, hi, grid.tol[a]))
        return m[0][:, None, None] & m[1][None, :, None] & m[2][None, None, :]
    if isinstance(q, C.Curve):
        return np.zeros(shape, bool)                                # a wire: on the edges of its path (find_path)
    if isinstance(q, C.Cylinder):
        p0, p1 = np.asarray(q.start), np.asarray(q.stop)
        ax = p1 - p0
        L2 = float(ax @ ax)
        rout = q.radius + (q.shell_width / 2 if isinstance(q, C.CylindricalShell) else 0.0)
        lo, hi = np.minimum(p0, p1) - rout, np.maximum(p0, p1) + rout
        m = [_range_mask(pts[a], lo[a], hi[a], grid.tol[a]) for a in range(3)]
        out = np.zeros(shape, bool)
        idx = [np.nonzero(x)[0] for x in m]
        if not all(len(i) for i in idx) or L2 == 0:
            return out
        P = np.stack(np.meshgrid(*[pts[a][idx[a]] for a in range(3)], indexing='ij'), -1) - p0
        t = (P @ ax) / L2
        r2 = ((P - t[..., None] * ax) ** 2).sum(-1)
        tol = max(grid.tol) * 10
        if isinstance(q, C.CylindricalShell):
            sub = (t >= -1e-9) & (t <= 1 + 1e-9) & (np.abs(np.sqrt(r2) - q.radius) <= q.shell_width / 2 + tol)
        else:
            sub = (t >= -1e-9) & (t <= 1 + 1e-9) & (r2 <= (q.radius + tol) ** 2)
        out[np.ix_(*idx)] = sub
        return out
    raise ValueError(f"gfdtd: primitive {type(q).__name__} not supported")


def _snap_primary(lines, c):
    """openEMS's SnapToMeshLine on the primary mesh: the nearest line (a tie goes to the lower one)."""
    dual = 0.5 * (lines[1:] + lines[:-1])
    return int(min(np.searchsorted(dual, c, 'left'), len(lines) - 1))


def _foot(P, a, b):
    """openEMS's Point_Line_Distance: the foot of P on the line a -> b (0 at a, 1 at b) and its distance."""
    d = b - a
    L2 = float(d @ d)
    t = float((P - a) @ d) / L2 if L2 > 0 else 0.0
    return t, float(np.linalg.norm(P - (a + t * d)))


def find_path(grid, p1, p2):
    """[(axis, (i, j, k))] of the mesh edges along the segment p1 -> p2 (drawing units), as openEMS's FindPath: from
    the line crossing nearest p1, each step to the neighbouring node closest to the segment and further along it."""
    L = grid.lines
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    for a in range(3):
        if (p1[a] < L[a][0] and p2[a] < L[a][0]) or (p1[a] > L[a][-1] and p2[a] > L[a][-1]):
            return []
    s = [_snap_primary(L[a], p1[a]) for a in range(3)]
    e = [_snap_primary(L[a], p2[a]) for a in range(3)]
    node = lambda q: np.array([L[a][q[a]] for a in range(3)])
    cur = list(s)
    cfoot = mfoot = _foot(node(s), p1, p2)[0]
    stop_foot = _foot(node(e), p1, p2)[0]
    path = []
    while mfoot < stop_foot:
        best = None
        for a in range(3):
            for step in (-1, 1):
                q = list(cur)
                q[a] += step
                if not 0 <= q[a] < len(L[a]):
                    continue
                f, dist = _foot(node(q), p1, p2)
                if f > cfoot and (best is None or dist < best[0]):
                    best = (dist, f, a, step)
        if best is None:
            break
        _, mfoot, a, step = best
        lo = list(cur)
        if step < 0:
            lo[a] -= 1
        cur[a] += step
        path.append((a, tuple(lo)))
        cfoot = mfoot
    for a in range(3):                                              # close what is missing to the end node
        if cur[a] > e[a]:
            cur[a] -= 1
            path.append((a, tuple(cur)))
        elif cur[a] < e[a]:
            path.append((a, tuple(cur)))
    return path


def curve_edges(q, grid):
    """find_path over each segment of a Curve primitive."""
    out = []
    for P0, P1 in zip(q.points[:-1], q.points[1:]):
        out += find_path(grid, P0, P1)
    return out


def edge_mask(q, a, grid):
    """inside() for the midpoints of the a-edges, on the (Nx, Ny, Nz) node array (False past the last edge)."""
    m = inside(q, grid.edge_points(a), grid)
    return np.pad(m, [(0, 1) if b == a else (0, 0) for b in range(3)])


def _along(plane, mn, n, shape):
    """plane (the two axes other than n) x mask mn along n -> a full (X, Y, Z) mask."""
    full = np.expand_dims(plane, n)
    line = mn.reshape([-1 if b == n else 1 for b in range(3)])
    return np.broadcast_to(full, shape) & line


# ---- the operator -------------------------------------------------------------------------------------------------
def timestep(grid, factor=SAFETY):
    """The largest stable step for vacuum on this grid: 1 / (c sqrt(sum over axes of the largest 1/(d_i-1 d_i))),
    the neighbouring primary cells around each node (half cells at the ends count once, as 1/d^2)."""
    s = 0.0
    for d in grid.d:
        q = 1.0 / (d * d)
        if len(d) > 1:
            q = np.maximum(q, np.concatenate([1.0 / (d[1:] * d[:-1]), [0]]))
        s += float(q.max())
    return factor / (C0 * np.sqrt(s))


def _edge_average(cell, a, grid):
    """A cell quantity (N-1 each axis) averaged over the four quarter cells around each a-edge (weights: the
    quarter areas); returned on the (Nx, Ny, Nz) node array, zero past the last a-edge."""
    b, c = [x for x in range(3) if x != a]
    P = np.moveaxis(cell, (a, b, c), (0, 1, 2))
    P = np.pad(P, ((0, 0), (1, 1), (1, 1)))
    db, dc = grid.d[b], grid.d[c]
    lo_b, hi_b = np.concatenate([[0], db / 2]), np.concatenate([db / 2, [0]])
    lo_c, hi_c = np.concatenate([[0], dc / 2]), np.concatenate([dc / 2, [0]])
    s = (P[:, :-1, :-1] * lo_b[:, None] * lo_c[None, :] + P[:, 1:, :-1] * hi_b[:, None] * lo_c[None, :] +
         P[:, :-1, 1:] * lo_b[:, None] * hi_c[None, :] + P[:, 1:, 1:] * hi_b[:, None] * hi_c[None, :])
    s /= (grid.dual[b][:, None] * grid.dual[c][None, :])
    s = np.pad(s, ((0, 1), (0, 0), (0, 0)))
    return np.moveaxis(s, (0, 1, 2), (a, b, c))


def _geometry(grid):
    """Per component: edge length l, dual area A' (E) and dual length l', face area A (H), on (Nx, Ny, Nz)."""
    sh = grid.shape
    lE, aE, lH, aH = (np.zeros((3,) + sh) for _ in range(4))
    for a in range(3):
        b, c = [x for x in range(3) if x != a]
        d_a = np.concatenate([grid.d[a], [0]])
        d_b = np.concatenate([grid.d[b], [0]])
        d_c = np.concatenate([grid.d[c], [0]])
        r = lambda v, ax: v.reshape([-1 if x == ax else 1 for x in range(3)])
        lE[a] = np.broadcast_to(r(d_a, a), sh)
        aE[a] = np.broadcast_to(r(grid.dual[b], b) * r(grid.dual[c], c), sh)
        lH[a] = np.broadcast_to(r(grid.dual[a], a), sh)
        aH[a] = np.broadcast_to(r(d_b, b) * r(d_c, c), sh)
    return lE, aE, lH, aH


def _box_ranges(q, grid, a):
    """Edge index range along a and line index ranges across, of the a-edges inside box q (midpoint rule; a
    box that catches no line across takes the nearest one)."""
    out = []
    for b in range(3):
        lo, hi = sorted((q.start[b], q.stop[b]))
        coord = grid.mid[b] if b == a else grid.lines[b]
        if b != a and hi - lo < grid.tol[b]:
            lo = hi = _snap(grid.lines[b], lo)
        idx = np.nonzero(_range_mask(coord, lo, hi, grid.tol[b]))[0]
        if len(idx) == 0:
            if b == a:
                return None
            idx = np.array([int(np.argmin(np.abs(coord - 0.5 * (lo + hi))))])
        out.append((int(idx[0]), int(idx[-1])))
    return out


def _clipped_weights(grid, b, i0, i1, lo, hi):
    """Share of each line i0..i1 in the width of a box [lo, hi] across axis b (its dual cell clipped to the box;
    equal shares for a flat box)."""
    if i0 == i1:
        return np.ones(1)
    L = grid.lines[b]
    w = []
    for i in range(i0, i1 + 1):
        a0 = L[i] - (L[i] - L[i - 1]) / 2 if i > 0 else L[i]
        a1 = L[i] + (L[i + 1] - L[i]) / 2 if i < len(L) - 1 else L[i]
        w.append(max(min(a1, hi) - max(a0, lo), 0.0))
    w = np.asarray(w)
    return w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))


def build(csx, fdtd, log=None):
    """The Operator of a ContinuousStructure (gfdtd.csx, or anything with the same attributes) and the FDTD
    settings dict(bc, f0, fc, nr_ts, end_crit, oversampling)."""
    grid = Grid(csx.GetGrid())
    sh = grid.shape
    op = Operator()
    op.grid, op.shape = grid, sh
    op.dt = timestep(grid)
    ncell = tuple(n - 1 for n in sh)
    bg = csx.background if hasattr(csx, 'background') else dict(Epsilon=1.0, Kappa=0.0)
    eps = np.full(ncell, float(bg.get('Epsilon', 1.0)))
    kap = np.full(ncell, float(bg.get('Kappa', 0.0)))
    prio = np.full(ncell, -np.inf)
    owner = np.full(ncell, -1, np.int32)                            # the Debye material of each cell (index)
    props = [p for p in csx.GetAllProperties() if getattr(p, 'enabled', True)]
    debye = [p for p in props if isinstance(p, C.DebyeMaterial) and p.pole_list()]
    cells = grid.cell_points()
    mats = [(q.priority, p, q) for p in props if isinstance(p, C.Material) for q in p.primitives]
    for pr, p, q in sorted(mats, key=lambda t: t[0]):
        if isinstance(q, C.Polygon) and not isinstance(q, C.LinPoly):
            continue                                                # flat: no cell volume
        m = inside(q, cells, grid) & (pr >= prio)
        eps[m], kap[m], prio[m] = p.epsilon, p.kappa, pr
        owner[m] = debye.index(p) if p in debye else -1
        if getattr(p, 'mue', 1.0) != 1.0 and not getattr(op, '_mue_noted', False):
            op.notes.append(f"material {p.name}: mue {p.mue:g} not modelled (1)")
            op._mue_noted = True
    lE, aE, lH, aH = _geometry(grid)
    with np.errstate(divide='ignore', invalid='ignore'):
        cap = np.zeros((3,) + sh)
        cond = np.zeros((3,) + sh)
        epsE = np.zeros((3,) + sh)
        for a in range(3):
            epsE[a] = _edge_average(eps, a, grid)
            cap[a] = np.where(lE[a] > 0, EPS0 * epsE[a] * aE[a] / lE[a], 0.0)
            cond[a] = np.where(lE[a] > 0, _edge_average(kap, a, grid) * aE[a] / lE[a], 0.0)
    # metals: edges whose midpoint is in a primitive, by priority
    pec = np.full((3,) + sh, -np.inf)
    for p in props:
        if isinstance(p, (C.Metal, C.ConductingSheet)):
            for q in p.primitives:
                if isinstance(q, C.Curve):
                    for a, ijk in curve_edges(q, grid):
                        if ijk[a] < sh[a] - 1:
                            pec[(a,) + ijk] = max(pec[(a,) + ijk], q.priority)
                    continue
                for a in range(3):
                    m = edge_mask(q, a, grid)
                    np.maximum(pec[a], np.where(m, q.priority, -np.inf), out=pec[a])
    if any(isinstance(p, C.ConductingSheet) for p in props):
        op.notes.append("conducting sheets are modelled as PEC (no conductor loss)")
    # lumped elements: parallel R / C into G / C of the edges, parallel L and series RLC as edge branches
    op.par_L, op.series = [], []
    lumped = [(q.priority, p, q) for p in props if isinstance(p, C.LumpedElement) for q in p.primitives]
    for pr, p, q in lumped:
        a = p.ny
        rg = _box_ranges(q, grid, a)
        if rg is None:
            op.notes.append(f"lumped element {p.name}: its box holds no {C.AXES[a]} edge - left out")
            continue
        (e0, e1), others = rg[a], [(b, rg[b]) for b in range(3) if b != a]
        Lbox = grid.d[a][e0:e1 + 1].sum() / grid.unit                # drawing units, like the lines
        (b, (j0, j1)), (c, (k0, k1)) = others
        wb = _clipped_weights(grid, b, j0, j1, *sorted((q.start[b], q.stop[b])))
        wc = _clipped_weights(grid, c, k0, k1, *sorted((q.start[c], q.stop[c])))
        frac = np.multiply.outer(wb, wc)                            # each column's share of the current
        series = p.LEtype == 1 and sum(v is not None for v in (p.R, p.L, p.C)) > 1
        for e in range(e0, e1 + 1):
            share = (grid.d[a][e] / grid.unit) / Lbox               # of the length
            for jj in range(j0, j1 + 1):
                for kk in range(k0, k1 + 1):
                    idx = [0, 0, 0]
                    idx[a], idx[b], idx[c] = e, jj, kk
                    i, j, k = idx
                    if pec[a, i, j, k] > pr:                        # a metal of higher priority keeps the edge
                        continue
                    pec[a, i, j, k] = -np.inf
                    f = frac[jj - j0, kk - k0]
                    Re = None if p.R is None else p.R * share / f
                    Le = None if p.L is None else p.L * share / f
                    Ce = None if p.C is None else p.C * f / share
                    if series:
                        op.series.append((_flat(sh, a, i, j, k), Re or 0.0, Le or 0.0, Ce))
                        continue
                    if Re is not None:
                        if Re > 0:
                            cond[a, i, j, k] += 1.0 / Re
                        else:
                            pec[a, i, j, k] = pr
                    if Ce is not None:
                        cap[a, i, j, k] += Ce
                    if Le is not None and Le > 0:
                        op.par_L.append((_flat(sh, a, i, j, k), Le))
        if p.caps:                                                  # metal faces at both ends
            for line in (e0, e1 + 1):
                for t, (t0, t1), (u, (u0, u1)) in ((b, (j0, j1), (c, (k0, k1))), (c, (k0, k1), (b, (j0, j1)))):
                    for ti in range(t0, t1):
                        for ui in range(u0, u1 + 1):
                            idx = [0, 0, 0]
                            idx[a], idx[t], idx[u] = line, ti, ui
                            pec[t][tuple(idx)] = max(pec[t][tuple(idx)], pr)
    pec_m = pec > -np.inf
    op.n_pec = [int(pec_m[a].sum()) for a in range(3)]
    dt = op.dt
    # Debye poles: i' = ka i + kb (V' - V); the E update sees (i + i') / 2, so kb / 2 joins the edge's C / dt
    kb_sum = np.zeros((3,) + sh)
    op.debye = []                                                   # (flat edge, ka, kb) of every pole on every edge
    for n, p in enumerate(debye):
        mine = owner == n
        for d_eps, tau in p.pole_list():
            for a in range(3):
                with np.errstate(divide='ignore', invalid='ignore'):
                    cp = np.where(lE[a] > 0, EPS0 * _edge_average(np.where(mine, d_eps, 0.0), a, grid) * aE[a] / lE[a],
                                  0.0)
                cp[pec_m[a]] = 0.0
                nz = np.nonzero(cp > 0)
                if not len(nz[0]):
                    continue
                kb = 2 * cp[nz] / (2 * tau + dt)
                kb_sum[a][nz] += kb
                op.debye.append((_flat(sh, a, *nz), np.full(len(kb), (2 * tau - dt) / (2 * tau + dt)), kb))
    if op.debye:
        idx, ka, kb = (np.concatenate(v) for v in zip(*op.debye))
        op.debye = (idx.astype(np.int64), ka, kb)
    else:
        op.debye = None
    with np.errstate(divide='ignore', invalid='ignore'):
        den = cap + cond * dt / 2 + kb_sum * dt / 2
        vv = np.where(den > 0, (cap - cond * dt / 2 + kb_sum * dt / 2) / den, 0.0)
        vi = np.where(den > 0, dt / den, 0.0)
        iv = np.where(aH > 0, dt * lH / (MU0 * aH), 0.0)
    vv[pec_m], vi[pec_m] = 0.0, 0.0
    # boundaries
    bc = [str(b).upper() for b in fdtd['bc']]
    op.bc = bc
    mur0, mur1, murk = [], [], []
    op.pml = []
    for side, b in enumerate(bc):
        a, hi = side // 2, side % 2
        n = sh[a]
        face, inner = (n - 1, n - 2) if hi else (0, 1)
        if b.startswith('PML'):
            op.pml.append(_pml_slab(grid, a, hi, b, dt))
            b = 'PEC'                                               # behind the PML
        for t in range(3):
            if t == a:
                continue
            sl = [slice(None)] * 3
            sl[a] = face
            if b == 'PEC':
                vv[t][tuple(sl)] = 0.0
                vi[t][tuple(sl)] = 0.0
            elif b == 'MUR':
                ok = vi[t][tuple(sl)] > 0
                pos = np.nonzero(ok)
                ijk0 = [None] * 3
                ijk1 = [None] * 3
                rest = [x for x in range(3) if x != a]
                ijk0[a], ijk1[a] = np.full(len(pos[0]), face), np.full(len(pos[0]), inner)
                for r_, pp in zip(rest, pos):
                    ijk0[r_] = ijk1[r_] = pp
                delta = abs(grid.m[a][face] - grid.m[a][inner])
                er = epsE[t][tuple(ijk0)]
                cl = C0 / np.sqrt(np.maximum(er, 1.0))
                mur0.append(_flat(sh, t, *ijk0))
                mur1.append(_flat(sh, t, *ijk1))
                murk.append((cl * dt - delta) / (cl * dt + delta))
            elif b != 'PMC':
                raise ValueError(f"gfdtd: boundary {b} not supported (PEC, PMC, MUR, PML_n)")
    op.mur = (np.concatenate(mur0).astype(np.int64), np.concatenate(mur1).astype(np.int64),
              np.concatenate(murk)) if mur0 else None
    # excitations: E (types 0 soft, 1 hard) on the edges, H (2 soft, 3 hard) on the dual edges, weighted, delayed
    op.exc, op.exc_more = _excitations(props, grid, sh, vv, vi, lE, lH, dt, op.notes)
    # probes
    op.probes = []
    for p in props:
        if isinstance(p, C.ProbeBox):
            for q in p.primitives[:1]:
                pr = _probe(p, q, grid, sh)
                if pr is not None:
                    op.probes.append(pr)
    op.dumps = []
    for p in props:
        if isinstance(p, C.DumpBox) and p.primitives:
            op.dumps += _dump(p, grid, op.notes)
    if op.dumps:                                                    # the geometry the dumps need, kept
        op.lE, op.aE, op.lH = lE, aE, lH
        if any(d['field'] == 'Jc' for d in op.dumps):
            op.kappa_E = np.stack([_edge_average(kap, a, grid) for a in range(3)])
    op.vv, op.vi, op.iv = vv, vi, iv
    op.cells = int(np.prod(ncell))
    # excitation signal: openEMS's Gaussian pulse
    f0, fc = float(fdtd['f0']), float(fdtd['fc'])
    op.f0, op.fc = f0, fc
    n = int(2 * 9.0 / (2 * np.pi * fc) / dt)
    t = np.arange(n + 1) * dt
    gauss = lambda t: np.cos(2 * np.pi * f0 * (t - 9.0 / (2 * np.pi * fc))) * np.exp(-(2 * np.pi * fc * t / 3 - 3) ** 2)
    op.signal = gauss(t)
    op.signal_h = gauss(t + dt / 2)                                # the H excitations' signal: half a step later
    op.delay_max = max([g['delay'] for g in op.exc_more], default=0)
    nyq = max(1, int(1.0 / (2 * (f0 + fc) * dt)))
    op.nyquist = nyq
    op.sub = max(1, nyq // int(fdtd.get('oversampling', 4) or 4))
    if log:
        for line in describe(op):
            log(line)
    return op


def _excitations(props, grid, sh, vv, vi, lE, lH, dt, notes):
    """(idx, amp) of the soft and hard E excitations without delay (one signal for all: the fast path), and
    [dict(field 'V' | 'I', delay steps, hard, idx, amp)] of the others. Edges are taken as openEMS takes them: the
    E edges whose midpoint, the H edges whose dual midpoint (inside the last lines) is in a primitive, curves along
    their path; the amplitude is exc_val times the weight there times the edge's length. Hard E edges lose their
    update (vv = vi = 0), so the signal sets them; hard H edges are set after the H update."""
    main_i, main_a, more = [], [], {}
    for p in props:
        if not isinstance(p, C.Excitation):
            continue
        if p.exc_type not in (0, 1, 2, 3):
            raise ValueError(f"gfdtd: excitation {p.name}: type {p.exc_type} not supported (0-3)")
        hfield = p.exc_type >= 2
        hard = p.exc_type in (1, 3)
        delay = int(p.delay / dt) if p.delay > 0 else 0
        idx, amp = [], []
        for q in p.primitives:
            for a in range(3):
                if p.exc_val[a] == 0:
                    continue
                if isinstance(q, C.Curve):
                    if hfield:
                        continue
                    hits = [ijk for b, ijk in curve_edges(q, grid) if b == a and ijk[a] < sh[a] - 1]
                    if not hits:
                        continue
                    nz = tuple(np.array(v) for v in zip(*hits))
                else:
                    if hfield:                                      # H_a at line a, cell centres across
                        pts = [grid.lines[b] if b == a else grid.mid[b] for b in range(3)]
                        m = inside(q, pts, grid)
                        m = np.pad(m, [(0, 0) if b == a else (0, 1) for b in range(3)])
                        m[tuple(slice(sh[b] - 1, None) if b == a else slice(None) for b in range(3))] = False
                    else:
                        m = edge_mask(q, a, grid) & (vi[a] > 0)
                    nz = np.nonzero(m)
                    if not len(nz[0]):
                        continue
                if hfield:
                    coords = [grid.lines[b][nz[b]] if b == a else grid.mid[b][np.minimum(nz[b], sh[b] - 2)]
                              for b in range(3)]
                    length = lH[a][nz]
                else:
                    coords = [grid.mid[b][nz[b]] if b == a else grid.lines[b][nz[b]] for b in range(3)]
                    length = lE[a][nz]
                w = p.weights(a, coords) * p.exc_val[a] * length
                keep = w != 0
                idx.append(_flat(sh, a, *(v[keep] for v in nz)))
                amp.append(w[keep])
                if hard and not hfield:
                    vv[a][tuple(v[keep] for v in nz)] = 0.0
                    vi[a][tuple(v[keep] for v in nz)] = 0.0
        if not idx:
            continue
        idx, amp = np.concatenate(idx).astype(np.int64), np.concatenate(amp)
        if not hfield and delay == 0:
            main_i.append(idx)
            main_a.append(amp)
        else:
            g = more.setdefault(('I' if hfield else 'V', delay, hard and hfield), ([], []))
            g[0].append(idx)
            g[1].append(amp)
    main = (np.concatenate(main_i) if main_i else np.zeros(0, np.int64),
            np.concatenate(main_a) if main_a else np.zeros(0))
    groups = [dict(field=f, delay=d, hard=h, idx=np.concatenate(i), amp=np.concatenate(a_))
              for (f, d, h), (i, a_) in sorted(more.items())]
    return main, groups


QUANT_BITS = 4         # compressed coefficients: low mantissa bits dropped (2^-20 relative) so that equal ones match


def _q32(x):
    """float32 values with the low QUANT_BITS mantissa bits rounded off, and their bit patterns."""
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    half = np.uint32(1 << (QUANT_BITS - 1))
    bits = (bits + half) & np.uint32(~((1 << QUANT_BITS) - 1) & 0xFFFFFFFF)
    return bits.view(np.float32), bits


class Compressed:
    """The operator without its coefficient arrays, for the compiled kernels (GPU and CPU core). vi = geoE * s and
    iv = geoH come from 1-D line arrays: geoE_a = dt / eps0 * d_a * (1 / dual_b) * (1 / dual_c), geoH_a = dt / mu0 *
    dual_a * (1 / d_b) * (1 / d_c). Each node keeps one code into a table of its three E edges' (vv, s, Debye
    poles); the nodes of one material (and one quarter-cell mix at interfaces) share one.

      code    (Nx, Ny, Nz) uint8, uint16 past 255 codes, uint32 past 65535
      vv, s   (ncode, 3) float32;  pole (ncode, 3) int32: row of kbt, or -1 (no poles)
      kbt     (nrow, P) float32: each pole's kb~ = kb l / (eps0 A'); ka, h (P,): its ka, (1 + ka) / 2
      slot    (Nx, Ny, Nz) int32: the node's row in the pole currents (nslot, 3, P), -1 elsewhere
      G       float32 [d | 1/d | dual | 1/dual], each [x | y | z] (d and 1/d zero past the last line)
      ke, kh  dt / eps0, dt / mu0

    Debye pole currents are kept scaled: i~ = i l / (eps0 A'), so that V' = vv V + vi c - dt s sum(h i~) and
    i~' = ka i~ + kb~ (V' - V) need no geometry."""


def compress(op):
    """op.vv / op.vi / op.iv / op.debye as a Compressed operator (exact to 2^-20 in vv and s)."""
    g, sh, dt = op.grid, op.shape, op.dt
    cp = Compressed()
    pad0 = lambda d: np.concatenate([d, [0.0]])
    d = [pad0(x) for x in g.d]
    idd = [pad0(1.0 / x) for x in g.d]
    cp.G = np.concatenate([np.concatenate(v) for v in (d, idd, g.dual, [1.0 / x for x in g.dual])]).astype(np.float32)
    cp.ke, cp.kh = dt / EPS0, dt / MU0
    r = lambda v, ax: v.reshape([-1 if x == ax else 1 for x in range(3)])
    geo = np.zeros((3,) + sh)
    for a in range(3):
        b, c = [x for x in range(3) if x != a]
        geo[a] = cp.ke * r(d[a], a) * r(1.0 / g.dual[b], b) * r(1.0 / g.dual[c], c)
    with np.errstate(divide='ignore', invalid='ignore'):
        s_ = np.where(geo > 0, op.vi / geo, 0.0)
    vv_q, vv_b = _q32(op.vv.ravel())
    s_q, s_b = _q32(s_.ravel())
    # Debye: the poles by relaxation (ka), each edge's kb~ per pole, and the distinct pole rows
    N = vv_b.size
    pid = np.zeros(N, np.int64)                                    # 0: no poles, else row + 1
    cp.P, cp.ka, cp.h, cp.kbt = 0, np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros((0, 0), np.float32)
    if op.debye is not None:
        idx, ka, kb = op.debye
        kas, col = np.unique(ka, return_inverse=True)
        cp.P = len(kas)
        kbt = np.zeros((N, cp.P))
        with np.errstate(divide='ignore', invalid='ignore'):
            np.add.at(kbt, (idx, col), kb * geo.ravel()[idx] / dt)        # kb~ = kb l / (eps0 A') = kb geo / dt
        rows_e = np.unique(idx)
        kq, kbits = _q32(kbt[rows_e])
        rows, inv = np.unique(kbits, axis=0, return_inverse=True)
        pid[rows_e] = inv.ravel() + 1
        cp.kbt = rows.view(np.float32).reshape(len(rows), cp.P).copy()
        cp.ka = kas.astype(np.float32)
        cp.h = (0.5 * (1 + kas)).astype(np.float32)
    key = (vv_b.astype(np.uint64) << np.uint64(32)) | s_b.astype(np.uint64)
    k1, inv1 = np.unique(key, return_inverse=True)
    both = inv1.astype(np.int64) * (int(pid.max()) + 1) + pid
    uniq, ecode = np.unique(both, return_inverse=True)              # per edge
    first = np.zeros(len(uniq), np.int64)
    first[ecode[::-1]] = np.arange(N)[::-1]                         # one edge of each edge code
    evv, es, epole = vv_q[first].astype(np.float32), s_q[first].astype(np.float32), (pid[first] - 1).astype(np.int32)
    trip = ecode.reshape(3, -1).T                                   # per node: its three edge codes
    nodes, code = np.unique(trip, axis=0, return_inverse=True)
    cp.vv, cp.s, cp.pole = evv[nodes], es[nodes], epole[nodes]      # (ncode, 3)
    cp.ncode = len(nodes)
    dtype = np.uint8 if cp.ncode <= 255 else (np.uint16 if cp.ncode <= 65535 else np.uint32)
    cp.code = code.ravel().astype(dtype).reshape(sh)
    has = (pid.reshape((3,) + sh) > 0).any(0)
    cp.slot = np.full(sh, -1, np.int32)
    cp.nslot = int(has.sum())
    cp.slot[has] = np.arange(cp.nslot, dtype=np.int32)
    return cp


def _pml_slab(grid, a, hi, b, dt):
    """dict(axis, hi, e0, h0, n, bE, cE, bH, cH) of a PML_n side: its n outermost cells along axis a. psi = b psi +
    c (difference across a) at the E edges' lines e0.. (bE, cE) and the H edges' cell centres h0.. (bH, cH)."""
    try:
        n = int(b.split('_')[1])
    except (IndexError, ValueError):
        n = 8
    L = grid.m[a]
    N = len(L)
    n = max(1, min(n, N - 2))
    if hi:
        x0, e0, h0 = L[N - 1 - n], N - n, N - 1 - n                  # E lines N-n .. N-1, H cells N-1-n .. N-2
    else:
        x0, e0, h0 = L[n], 0, 0                                     # E lines 0 .. n-1, H cells 0 .. n-1
    depth = abs(L[-1] - L[N - 1 - n]) if hi else abs(L[n] - L[0])
    smax = -(PML_GRADING + 1) * np.log(PML_R0) * C0 / (2 * depth)   # sigma_max / eps0
    xe = L[e0:e0 + n]
    xh = 0.5 * (L[h0:h0 + n] + L[h0 + 1:h0 + n + 1])
    grade = lambda x: smax * np.clip(np.abs(x - x0) / depth, 0, 1) ** PML_GRADING
    bE, bH = np.exp(-grade(xe) * dt), np.exp(-grade(xh) * dt)
    return dict(axis=a, hi=hi, n=n, e0=int(e0), h0=int(h0), bE=bE, cE=bE - 1, bH=bH, cH=bH - 1, name=b)


DUMP_FIELDS = {0: 'E', 1: 'H', 2: 'Jc', 3: 'J'}                  # dump_type % 10


def _dump(p, grid, notes):
    """[dict(name, field, fd (freqs Hz or None), mode ('cell' | 'node'), lo, hi (index ranges, inclusive), coords (m),
    file_type, sub)] of a dump, one per box (named <name>_<n> when it has several, as openEMS's NF2FF boxes). Dump
    mode 2 picks cells as openEMS does: the box snapped to the nearest lines s0 .. s1, then the cells s0-1 .. s1 (a
    flat box: the cell below its line; the cell past the last line counts, its field is zero). Modes 0 and 1: the
    nodes on the lines s0 .. s1."""
    t = p.dump_type
    if t % 10 not in DUMP_FIELDS or t // 10 not in (0, 1):
        notes.append(f"field dump {p.name}: dump type {t} not supported - skipped")
        return []
    fd = t >= 10
    if fd and not p.frequency:
        notes.append(f"field dump {p.name}: frequency domain with no frequencies - skipped")
        return []
    mode = 'cell' if p.dump_mode == 2 else 'node'
    sub = int(np.max(p.sub_sampling)) if p.sub_sampling is not None else 1
    out = []
    for n, q in enumerate(p.primitives):
        if not isinstance(q, C.Box):
            notes.append(f"field dump {p.name}: a {type(q).__name__} - only boxes are dumped, skipped")
            continue
        lo3, hi3 = np.minimum(q.start, q.stop), np.maximum(q.start, q.stop)
        lo, hi, coords = [], [], []
        for a in range(3):
            L = grid.lines[a]
            s0 = int(np.argmin(np.abs(L - lo3[a])))                  # the box snapped to lines s0 .. s1
            s1 = int(np.argmin(np.abs(L - hi3[a])))
            if mode == 'node':
                i0, i1, pts = s0, s1, L
            else:
                i0 = max(s0 - 1, 0)
                i1 = s1 if s1 > s0 else i0                          # the cells s0-1 .. s1; flat: the one below
                pts = np.concatenate([grid.mid[a], [L[-1] + 0.5 * (L[-1] - L[-2])]])
            lo.append(i0)
            hi.append(i1)
            coords.append(pts[i0:i1 + 1] * grid.unit)
        out.append(dict(name=p.name if len(p.primitives) == 1 else f"{p.name}_{n}", field=DUMP_FIELDS[t % 10],
                        type=t, fd=list(p.frequency) if fd else None, mode=mode, lo=lo, hi=hi, coords=coords,
                        file_type=p.file_type, sub=max(1, sub), unit=grid.unit))
    return out


def _current_nodes(grid, b, lo, hi):
    """The node range (inclusive) a current probe's loop encloses across axis b, as openEMS snaps the box to the dual
    mesh: the nodes whose dual cell overlaps (lo, hi); a flat box: the nearest node (none at a midpoint tie)."""
    L = grid.lines[b]
    dual = np.concatenate([[-np.inf], 0.5 * (L[1:] + L[:-1]), [np.inf]])     # node k's dual cell: dual[k] .. dual[k+1]
    tol = grid.tol[b]
    if hi - lo < tol:
        k = int(np.argmin(np.abs(L - lo)))
        return (k, k) if abs(abs(L[k] - lo) - min(abs(np.delete(L, k) - lo), default=np.inf)) > tol else (k, k - 1)
    ks = np.nonzero((dual[1:] > lo + tol) & (dual[:-1] < hi - tol))[0]
    return (int(ks[0]), int(ks[-1])) if len(ks) else (0, -1)


def _probe(p, q, grid, sh):
    """dict(name, type, field 'V' | 'I', idx, w, start, stop) of a probe box: the terms whose sum is its value."""
    s = [int(np.argmin(np.abs(grid.lines[a] - q.start[a]))) for a in range(3)]
    e = [int(np.argmin(np.abs(grid.lines[a] - q.stop[a]))) for a in range(3)]
    terms = {}
    if p.p_type == 0:                                               # E integral, x then y then z
        pos = list(s)
        for a in range(3):
            step = 1 if e[a] >= s[a] else -1
            while pos[a] != e[a]:
                i = pos[a] if step > 0 else pos[a] - 1
                idx = list(pos)
                idx[a] = i
                f = _flat(sh, a, *idx)
                terms[f] = terms.get(f, 0.0) + step * p.weight
                pos[a] += step
        field = 'V'
    elif p.p_type == 1:                                             # H loop around the box, normal n
        n = p.norm_dir
        if n < 0:
            n = next(a for a in range(3) if s[a] == e[a])
        b, c = (n + 1) % 3, (n + 2) % 3                             # cyclic: the loop's sense is curl's
        cn = 0.5 * (q.start[n] + q.stop[n])
        layer = int(np.clip(np.searchsorted(grid.lines[n], cn - grid.tol[n], 'left') - 1, 0, sh[n] - 2))
        rb = _current_nodes(grid, b, *sorted((q.start[b], q.stop[b])))
        rc = _current_nodes(grid, c, *sorted((q.start[c], q.stop[c])))
        for jb in range(rb[0], rb[1] + 1):
            for kc in range(rc[0], rc[1] + 1):
                idx = [0, 0, 0]
                idx[n], idx[b], idx[c] = layer, jb, kc
                # loop sum of I around the n-edge: +I_c[.] - I_c[.-b] - I_b[.] + I_b[.-c]
                for comp, off, sgn in ((c, b, +1), (b, c, -1)):
                    for shift, s2 in ((0, 1), (1, -1)):
                        jdx = list(idx)
                        jdx[off] -= shift
                        if jdx[off] < 0:
                            continue
                        f = _flat(sh, comp, *jdx)
                        terms[f] = terms.get(f, 0.0) + sgn * s2 * p.weight
        field = 'I'
    elif p.p_type in (10, 11):
        terms = _mode_match(p, grid, sh, s, e)
        if terms is None:
            return None
        field = 'V' if p.p_type == 10 else 'I'
    else:
        return None
    terms = {k: v for k, v in terms.items() if v != 0}
    lo = [grid.lines[a][s[a]] for a in range(3)]
    hi = [grid.lines[a][e[a]] for a in range(3)]
    return dict(name=p.name, type=p.p_type, field=field, idx=np.array(list(terms), np.int64),
                w=np.array(list(terms.values()), float), start=(lo, s), stop=(hi, e), unit=grid.unit)


def _mode_match(p, grid, sh, s, e):
    """The terms of a mode-matching probe, as openEMS's ProcessModeMatch: on the box's plane (normal n) the
    transverse E at the nodes (type 10, interpolated between the edges either side) or H (type 11, the average of
    the four dual edges round the node), times the mode function normalised to unit energy (at the nodes for E, at
    the cell centres for H, as openEMS evaluates them), times the node area. The boundary lines are left out."""
    dual = p.p_type == 11
    lo, hi = [min(a, b) for a, b in zip(s, e)], [max(a, b) for a, b in zip(s, e)]
    for a in range(3):
        if lo[a] == 0:
            lo[a] += 1
        if hi[a] == sh[a] - 1:
            hi[a] -= 1
        elif dual and hi[a] > lo[a] + 1:
            hi[a] -= 1
    flat = [a for a in range(3) if lo[a] == hi[a]]
    if len(flat) != 1:
        return None
    n = flat[0]
    u, v = (n + 1) % 3, (n + 2) % 3
    disc = lambda a, i: (grid.mid[a][i] if i < sh[a] - 1 else grid.lines[a][i] + 0.5 * (grid.lines[a][i] -
                                                                                         grid.lines[a][i - 1])) \
        if dual else grid.lines[a][i]
    width = lambda a, i: (grid.d[a][i] if i < sh[a] - 1 else 0.0) if dual else grid.dual[a][i]
    iu, iv = np.arange(lo[u], hi[u] + 1), np.arange(lo[v], hi[v] + 1)
    cu = np.array([disc(u, i) for i in iu])
    cv = np.array([disc(v, i) for i in iv])
    area = np.outer([width(u, i) for i in iu], [width(v, i) for i in iv])
    coords = [None] * 3
    U, Vv = np.meshgrid(cu, cv, indexing='ij')
    coords[u], coords[v], coords[n] = U, Vv, np.full(U.shape, disc(n, lo[n]))
    fu, fv = p.mode(n, coords)
    norm = np.sqrt(((fu ** 2 + fv ** 2) * area).sum())
    if norm == 0:
        return None
    fu, fv = fu / norm, fv / norm
    terms = {}

    def add(comp, ijk, w):
        f = _flat(sh, comp, *ijk)
        terms[f] = terms.get(f, 0.0) + w * p.weight

    for x, i in enumerate(iu):
        for y, j in enumerate(iv):
            pos = [0, 0, 0]
            pos[n], pos[u], pos[v] = lo[n], i, j
            for comp, f in ((u, fu[x, y]), (v, fv[x, y])):
                wt = f * area[x, y]
                if wt == 0:
                    continue
                if not dual:                                        # E at the node: edges pos and pos-1 along comp
                    d = grid.d[comp]
                    k = pos[comp]
                    if k == sh[comp] - 1:
                        q_ = list(pos)
                        q_[comp] -= 1
                        add(comp, q_, wt / d[k - 1])
                        continue
                    if k == 0:
                        add(comp, pos, wt / d[0])
                        continue
                    rel = d[k] / (d[k] + d[k - 1])
                    add(comp, pos, wt * (1 - rel) / d[k])
                    q_ = list(pos)
                    q_[comp] -= 1
                    add(comp, q_, wt * rel / d[k - 1])
                else:                                               # H at the node: four dual edges round it
                    cP, cPP = (comp + 1) % 3, (comp + 2) % 3
                    if any(pos[b] == sh[b] - 1 for b in range(3)) or pos[cP] == 0 or pos[cPP] == 0:
                        continue
                    for dP, dPP in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        q_ = list(pos)
                        q_[cP] -= dP
                        q_[cPP] -= dPP
                        add(comp, q_, wt / 4 / grid.dual[comp][q_[comp]])
    return terms


def describe(op):
    g = op.grid
    nx, ny, nz = g.shape
    lines = [f"Dimensions\t\t: {nx}x{ny}x{nz} = {op.cells} Cells ({op.cells / 1e6:g} MCells)",
             f"Number of PEC edges\t: {sum(op.n_pec)}",
             f"Timestep (s)\t\t: {op.dt:g}",
             f"Nyquist criteria (TS)\t: {op.nyquist}",
             f"Voltage excitations\t: {len(op.exc[0])}",
             f"Excitation Length (TS)\t: {len(op.signal)}",
             f"Lumped branches\t\t: {len(op.par_L)} parallel L, {len(op.series)} series RLC",
             f"Mur ABC edges\t\t: {0 if op.mur is None else len(op.mur[0])}",
             "PML sides\t\t: " + (', '.join(f"{C.AXES[s_['axis']]}{'max' if s_['hi'] else 'min'} ({s_['n']} cells)"
                                             for s_ in op.pml) or 'none'),
             f"Debye pole edges\t: {0 if op.debye is None else len(op.debye[0])}",
             f"Probes\t\t\t: {len(op.probes)}",
             f"Field dumps\t\t: {len(op.dumps)}",
             f"FDTD timestep is: {op.dt:g} s; Nyquist rate: {op.nyquist} timesteps "
             f"@{1 / (2 * op.nyquist * op.dt):g} Hz"]
    return lines + [f"note: {n}" for n in op.notes]
