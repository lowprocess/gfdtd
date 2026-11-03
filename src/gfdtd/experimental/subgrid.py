"""
Experimental: local mesh refinement for gfdtd (not wired into the engine).

A plain Yee solver in numpy, separate from gfdtd's operator so that the three ways of refining a mesh run on the
same update code and can be compared fairly:

- Grid: one rectilinear grid (any lines per axis: uniform, or graded as gfdtd's meshes are), PEC outer walls.
- Subgridded: a uniform coarse grid with one box refined 3:1 in space and time (odd ratio: every coarse E edge and
  H face on or inside the box lies on a fine one, so only the box's surface needs interpolating).

The coupling, per coarse step n -> n+1:

    1. coarse H, E update everywhere                         (E^n -> E^n+1)
    2. three fine steps; the fine box's tangential E on its surface is the coarse E there, linear in time
       between E^n and E^n+1, constant along each coarse edge and linear across it
    3. the coarse fields strictly inside the box are replaced by the fine ones averaged over each coarse edge
       (E, after the third fine step) or face (H, after the second: fine H^n+1/2 is coarse H's time)

So the coarse grid sees the fine structure only through the fields half a cell inside the box's surface, a step
late. That is Chevalier / Okoniewski-style subgridding; it is not provably stable (see tools/bench_subgrid.py).
"""
import numpy as np

C0 = 299792458.0
MU0 = 4e-7 * np.pi
EPS0 = 1.0 / (MU0 * C0 ** 2)


def dual(d):
    """The dual spacing at the nodes of primary spacings d: half of each neighbour's (half a cell at the ends)."""
    return np.r_[d[0] / 2, (d[:-1] + d[1:]) / 2, d[-1] / 2]


def graded_lines(lo, hi, fine_lo, fine_hi, h_fine, h_coarse, ratio=1.5):
    """Lines over [lo, hi]: h_fine over [fine_lo, fine_hi], growing by `ratio` per cell up to h_coarse away from it."""
    core = np.arange(fine_lo, fine_hi + h_fine / 2, h_fine)

    def side(start, end, sgn):
        out, x, h = [], start, h_fine
        while sgn * (end - x) > 1e-9:
            h = min(h * ratio, h_coarse)
            x = x + sgn * h
            if sgn * (end - x) < 0.5 * h:                     # the last cell: stretch or squeeze onto the wall
                x = end
            out.append(x)
        return out
    left = side(fine_lo, lo, -1)[::-1]
    right = side(fine_hi, hi, +1)
    return np.array(left + list(core) + right)


class Grid:
    """A Yee grid on lines x, y, z (metres). E on the edges, H on the faces; the outer walls are PEC unless
    another solver writes the tangential E there (`driven`)."""

    def __init__(self, x, y, z, dt=None, cfl=0.95):
        self.x, self.y, self.z = (np.asarray(v, float) for v in (x, y, z))
        self.dx, self.dy, self.dz = np.diff(self.x), np.diff(self.y), np.diff(self.z)
        self.dxd, self.dyd, self.dzd = dual(self.dx), dual(self.dy), dual(self.dz)
        nx, ny, nz = self.shape = (len(self.dx), len(self.dy), len(self.dz))
        if dt is None:
            dt = cfl / (C0 * np.sqrt(self.dx.min() ** -2 + self.dy.min() ** -2 + self.dz.min() ** -2))
        self.dt = dt
        self.Ex, self.Ey, self.Ez = (np.zeros(s) for s in ((nx, ny + 1, nz + 1), (nx + 1, ny, nz + 1),
                                                           (nx + 1, ny + 1, nz)))
        self.Hx, self.Hy, self.Hz = (np.zeros(s) for s in ((nx + 1, ny, nz), (nx, ny + 1, nz), (nx, ny, nz + 1)))
        self.epsr = [np.ones_like(F) for F in (self.Ex, self.Ey, self.Ez)]
        self.pec = [np.zeros(F.shape, bool) for F in (self.Ex, self.Ey, self.Ez)]
        self.finalize()

    @property
    def cells(self):
        return int(np.prod(self.shape))

    # ---- geometry --------------------------------------------------------------------------------------------
    def _mids(self, c):
        """The coordinates of component c's E edges (midpoints along c, nodes across)."""
        m = lambda v: (v[:-1] + v[1:]) / 2
        ax = [m(self.x) if c == 0 else self.x, m(self.y) if c == 1 else self.y, m(self.z) if c == 2 else self.z]
        return np.meshgrid(*ax, indexing='ij')

    def add_pec_box(self, lo, hi, tol=1e-12):
        """PEC on every E edge whose midpoint lies in the box [lo, hi] (bounds included), as gfdtd marks metal."""
        for c in range(3):
            X, Y, Z = self._mids(c)
            self.pec[c] |= ((X >= lo[0] - tol) & (X <= hi[0] + tol) & (Y >= lo[1] - tol) & (Y <= hi[1] + tol) &
                            (Z >= lo[2] - tol) & (Z <= hi[2] + tol))
        self.finalize()

    def finalize(self):
        """The update coefficients: dt / (eps A') on the edges that update (not PEC, not on the outer walls)."""
        ce = []
        for c, (F, er, pec) in enumerate(zip((self.Ex, self.Ey, self.Ez), self.epsr, self.pec)):
            k = self.dt / (EPS0 * er)
            k[pec] = 0.0
            F[pec] = 0.0
            for b in range(3):                                      # the outer walls' tangential E never update
                if b != c:
                    sl = [slice(None)] * 3
                    sl[b] = [0, -1]
                    k[tuple(sl)] = 0.0
            ce.append(k)
        self.ce = ce
        self.ch = self.dt / MU0
        self.idx, self.idy, self.idz = 1 / self.dx, 1 / self.dy, 1 / self.dz
        self.idxd, self.idyd, self.idzd = 1 / self.dxd[1:-1], 1 / self.dyd[1:-1], 1 / self.dzd[1:-1]

    def nearest(self, c, p):
        """The index of component c's E edge nearest to point p."""
        X, Y, Z = self._mids(c)
        d = (X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2
        return np.unravel_index(np.argmin(d), d.shape)

    # ---- update ----------------------------------------------------------------------------------------------
    def update_H(self):
        Ex, Ey, Ez, ch = self.Ex, self.Ey, self.Ez, self.ch
        self.Hx -= ch * (np.diff(Ez, axis=1) * self.idy[None, :, None] - np.diff(Ey, axis=2) * self.idz)
        self.Hy -= ch * (np.diff(Ex, axis=2) * self.idz - np.diff(Ez, axis=0) * self.idx[:, None, None])
        self.Hz -= ch * (np.diff(Ey, axis=0) * self.idx[:, None, None] - np.diff(Ex, axis=1) * self.idy[None, :, None])

    def update_E(self):
        Hx, Hy, Hz = self.Hx, self.Hy, self.Hz
        idxd, idyd, idzd = self.idxd[:, None, None], self.idyd[None, :, None], self.idzd
        self.Ex[:, 1:-1, 1:-1] += self.ce[0][:, 1:-1, 1:-1] * (np.diff(Hz, axis=1)[:, :, 1:-1] * idyd -
                                                               np.diff(Hy, axis=2)[:, 1:-1, :] * idzd)
        self.Ey[1:-1, :, 1:-1] += self.ce[1][1:-1, :, 1:-1] * (np.diff(Hx, axis=2)[1:-1, :, :] * idzd -
                                                               np.diff(Hz, axis=0)[:, :, 1:-1] * idxd)
        self.Ez[1:-1, 1:-1, :] += self.ce[2][1:-1, 1:-1, :] * (np.diff(Hy, axis=0)[:, 1:-1, :] * idxd -
                                                               np.diff(Hx, axis=1)[1:-1, :, :] * idyd)

    def energy(self, skip=None):
        """The field energy (J), 1/2 eps E^2 + 1/2 mu H^2 over the edge and face volumes; skip: a slice per
        component (E then H) left out, for the coarse part under a subgrid box."""
        x, y, z, xd, yd, zd = self.dx, self.dy, self.dz, self.dxd, self.dyd, self.dzd
        vol = lambda a, b, c: a[:, None, None] * b[None, :, None] * c[None, None, :]
        vols = [vol(x, yd, zd), vol(xd, y, zd), vol(xd, yd, z), vol(xd, y, z), vol(x, yd, z), vol(x, y, zd)]
        fields = [self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz]
        w = [EPS0 * self.epsr[c] for c in range(3)] + [MU0] * 3
        total = 0.0
        for n, (F, v, k) in enumerate(zip(fields, vols, w)):
            e = k * v * F * F
            if skip is not None:
                e = e.copy()
                e[skip[n]] = 0.0
            total += 0.5 * e.sum()
        return total


def _p_const(n):
    """(3n, n): a coarse edge's value on each of its three fine edges."""
    return np.kron(np.eye(n), np.ones((3, 1)))


def _p_linear(n):
    """(3n + 1, n + 1): coarse node values interpolated linearly onto the fine nodes."""
    P = np.zeros((3 * n + 1, n + 1))
    for f in range(3 * n + 1):
        c, r = divmod(f, 3)
        P[f, c] += 1 - r / 3
        if r:
            P[f, c + 1] += r / 3
    return P


class SubgriddedOverlap:
    """A uniform coarse grid (spacing h) with the box of coarse cells box = ((i0, i1), (j0, j1), (k0, k1)) refined
    3:1 in space and time. Material and metal go on `fine` (and on `coarse` outside the box)."""
    R = 3

    def __init__(self, x, y, z, box, cfl=0.95):
        self.coarse = Grid(x, y, z, cfl=cfl)
        (i0, i1), (j0, j1), (k0, k1) = self.box = box
        R = self.R
        cg = self.coarse
        fx = np.linspace(cg.x[i0], cg.x[i1], R * (i1 - i0) + 1)
        fy = np.linspace(cg.y[j0], cg.y[j1], R * (j1 - j0) + 1)
        fz = np.linspace(cg.z[k0], cg.z[k1], R * (k1 - k0) + 1)
        self.fine = Grid(fx, fy, fz, dt=cg.dt / R)
        n = (i1 - i0, j1 - j0, k1 - k0)
        self.Pc = [_p_const(m) for m in n]
        self.Pn = [_p_linear(m) for m in n]
        self.dt = cg.dt
        # the coarse fields strictly inside the box, which the fine ones replace
        self.inner = [np.s_[i0:i1, j0 + 1:j1, k0 + 1:k1], np.s_[i0 + 1:i1, j0:j1, k0 + 1:k1],
                      np.s_[i0 + 1:i1, j0 + 1:j1, k0:k1], np.s_[i0 + 1:i1, j0:j1, k0:k1],
                      np.s_[i0:i1, j0 + 1:j1, k0:k1], np.s_[i0:i1, j0:j1, k0 + 1:k1]]

    @property
    def cells(self):
        return self.coarse.cells + self.fine.cells

    # ---- coarse -> fine: the box surface's tangential E ------------------------------------------------------
    def _faces(self):
        """The coarse tangential E on the box's six faces: {(field, axis, side): 2D slab}."""
        cg = self.coarse
        (i0, i1), (j0, j1), (k0, k1) = self.box
        out = {}
        for side, (i, j, k) in enumerate(((i0, j0, k0), (i1, j1, k1))):
            out['Ey', 0, side] = cg.Ey[i, j0:j1, k0:k1 + 1]
            out['Ez', 0, side] = cg.Ez[i, j0:j1 + 1, k0:k1]
            out['Ex', 1, side] = cg.Ex[i0:i1, j, k0:k1 + 1]
            out['Ez', 1, side] = cg.Ez[i0:i1 + 1, j, k0:k1]
            out['Ex', 2, side] = cg.Ex[i0:i1, j0:j1 + 1, k]
            out['Ey', 2, side] = cg.Ey[i0:i1 + 1, j0:j1, k]
        return {key: v.copy() for key, v in out.items()}

    def _drive(self, old, new, a):
        """Write the fine box surface's tangential E: the coarse faces at time n + a, prolonged."""
        fg = self.fine
        comp = {'Ex': 0, 'Ey': 1, 'Ez': 2}
        for (name, ax, side), c0 in old.items():
            c = comp[name]
            v = (1 - a) * c0 + a * new[name, ax, side]
            u, w = [b for b in range(3) if b != ax]                 # the two axes of the face, in order
            Pu = self.Pc[u] if u == c else self.Pn[u]
            Pw = self.Pc[w] if w == c else self.Pn[w]
            F = getattr(fg, name)
            sl = [slice(None)] * 3
            sl[ax] = 0 if side == 0 else -1
            F[tuple(sl)] = Pu @ v @ Pw.T

    # ---- fine -> coarse: the fields strictly inside the box --------------------------------------------------
    def _restrict_E(self):
        cg, fg, R = self.coarse, self.fine, self.R
        nx, ny, nz = (b - a for a, b in self.box)
        cg.Ex[self.inner[0]] = fg.Ex[:, R:-1:R, R:-1:R].reshape(nx, R, ny - 1, nz - 1).mean(1)
        cg.Ey[self.inner[1]] = fg.Ey[R:-1:R, :, R:-1:R].reshape(nx - 1, ny, R, nz - 1).mean(2)
        cg.Ez[self.inner[2]] = fg.Ez[R:-1:R, R:-1:R, :].reshape(nx - 1, ny - 1, nz, R).mean(3)

    def _restrict_H(self):
        cg, fg, R = self.coarse, self.fine, self.R
        nx, ny, nz = (b - a for a, b in self.box)
        cg.Hx[self.inner[3]] = fg.Hx[R:-1:R, :, :].reshape(nx - 1, ny, R, nz, R).mean((2, 4))
        cg.Hy[self.inner[4]] = fg.Hy[:, R:-1:R, :].reshape(nx, R, ny - 1, nz, R).mean((1, 4))
        cg.Hz[self.inner[5]] = fg.Hz[:, :, R:-1:R].reshape(nx, R, ny, R, nz - 1).mean((1, 3))

    def step(self, source=None):
        """One coarse step; source(grid) adds the excitation to the coarse grid after its E update."""
        cg, fg = self.coarse, self.fine
        old = self._faces()
        cg.update_H()
        cg.update_E()
        if source:
            source(cg)
        new = self._faces()
        for m in range(1, self.R + 1):
            fg.update_H()
            if 2 * m - 1 == self.R:                                 # fine H at coarse H's time, n + 1/2
                self._restrict_H()
            fg.update_E()
            self._drive(old, new, m / self.R)
        self._restrict_E()

    def energy(self):
        return self.coarse.energy(skip=self.inner) + self.fine.energy()


def cfl_dt(x, y, z, cfl=0.95):
    d = [np.diff(np.asarray(v, float)).min() for v in (x, y, z)]
    return cfl / (C0 * np.sqrt(sum(v ** -2 for v in d)))


class Subgridded:
    """A coarse grid (uniform over the box) with the box of coarse cells box = ((i0, i1), (j0, j1), (k0, k1))
    refined 3:1 in space, coupled through an adjoint pair of operators on the box's surface:

    - coarse -> fine: the fine surface's tangential E is P E_c (constant along each coarse edge, linear across:
      a Dirichlet condition; the fine grid has no unknowns on its surface).
    - fine -> coarse: the coarse surface E's update takes the tangential H inside the box from the fine grid,
      half a fine cell in, as P^T H_f (full weighting), over the distance from there to the coarse H outside.

    The power the fine grid takes through its surface is then exactly (P E_c) . H_f = E_c . (P^T H_f), the power
    the coarse surface gives up, so the coupling neither makes nor loses energy (up to the leapfrog's time
    staggering). The coarse grid has no fields inside the box.

    subcycle=False: one time step, the fine grid's, for both grids. subcycle=True: the coarse grid at its own
    step, three fine steps per coarse one, the fine surface E linear in time between coarse E^n and E^n+1; the
    coarse surface E^n+1 is predicted from the coarse fields alone, corrected after the fine steps with the fine H
    averaged over them, and the fine surface at n+1 rewritten with the corrected value. Material and metal go on
    `fine` inside the box and on `coarse` outside it."""
    R = 3

    def __init__(self, x, y, z, box, subcycle=False, cfl=0.95):
        R = self.R
        self.subcycle = subcycle
        dtc = cfl_dt(x, y, z, cfl)
        if subcycle:
            # the coarse surface E's inside half-derivative spans 2h/3 (h/2 out, h/6 in), along two axes at the box's
            # edges: the coarse step's CFL limit with those two spacings shortened
            q = sorted(np.diff(np.asarray(v, float)).min() ** -2 for v in (x, y, z))
            dtc = cfl / (C0 * np.sqrt(sum(q) + 1.25 * (q[1] + q[2])))
        self.coarse = cg = Grid(x, y, z, dt=dtc if subcycle else dtc / R)
        (i0, i1), (j0, j1), (k0, k1) = self.box = box
        fl = []
        for v, (a, b) in zip((cg.x, cg.y, cg.z), box):
            fl.append(np.r_[np.concatenate([np.linspace(v[n], v[n + 1], R + 1)[:-1] for n in range(a, b)]), v[b]])
        self.fine = fg = Grid(*fl, dt=cg.dt / R if subcycle else cg.dt)
        self.dt = cg.dt
        n = (i1 - i0, j1 - j0, k1 - k0)
        self.Pc = [_p_const(m) for m in n]
        self.Pn = [_p_linear(m) for m in n]
        self.Rc = [P.T / R for P in self.Pc]                              # along an edge: the mean of three
        self.Rn = [P.T / P.sum(0)[:, None] for P in self.Pn]              # across: full weighting
        # the coarse fields inside the box (none: E there is held at 0, H zeroed after each update)
        self.inner = [np.s_[i0:i1, j0 + 1:j1, k0 + 1:k1], np.s_[i0 + 1:i1, j0:j1, k0 + 1:k1],
                      np.s_[i0 + 1:i1, j0 + 1:j1, k0:k1], np.s_[i0 + 1:i1, j0:j1, k0:k1],
                      np.s_[i0:i1, j0 + 1:j1, k0:k1], np.s_[i0:i1, j0:j1, k0 + 1:k1]]
        for c in range(3):
            cg.ce[c][self.inner[c]] = 0.0
        self._faces_setup()

    def _faces_setup(self):
        """Per face and tangential E component: the slices and constants of the surface update."""
        cg, fg = self.coarse, self.fine
        lines = (cg.x, cg.y, cg.z)
        names = ('Ex', 'Ey', 'Ez')
        self.faces = []
        for ax in range(3):
            for side in (0, 1):
                p = self.box[ax][side]
                if p in (0, len(lines[ax]) - 1):                            # on the outer PEC wall: no interface
                    continue
                for c in range(3):
                    if c == ax:
                        continue
                    t = 3 - ax - c                                          # the tangential H in the normal derivative
                    sgn = 1.0 if ax == (c + 1) % 3 else -1.0                # curl_c = dH_c+2/dx_c+1 - dH_c+1/dx_c+2
                    s = 1.0 if side == 0 else -1.0                          # the box lies on the high side of face 0
                    e = [slice(a, b + (0 if k == c else 1)) for k, (a, b) in enumerate(self.box)]
                    e[ax] = p
                    hout, hin = list(e), list(e)
                    hout[ax] = p - 1 if side == 0 else p
                    hin[ax] = p if side == 0 else p - 1
                    hf = [slice(None)] * 3
                    hf[ax] = 0 if side == 0 else -1
                    fe = [slice(None)] * 3
                    fe[ax] = 0 if side == 0 else -1
                    outer = lines[ax][p] - lines[ax][hout[ax]] if side == 0 else lines[ax][p + 1] - lines[ax][p]
                    inner_f = (fg.x, fg.y, fg.z)[ax]
                    finew = inner_f[1] - inner_f[0] if side == 0 else inner_f[-1] - inner_f[-2]
                    dist = abs(outer) / 2 + finew / 2
                    hdual = cg.dxd[p] if ax == 0 else cg.dyd[p] if ax == 1 else cg.dzd[p]
                    u, w = [b for b in range(3) if b != ax]
                    self.faces.append(dict(
                        E=names[c], H='H' + 'xyz'[t], e=tuple(e), hout=tuple(hout), hin=tuple(hin), hf=tuple(hf),
                        fe=tuple(fe), k=sgn * s, dist=dist, hdual=hdual, ce=cg.ce[c][tuple(e)],
                        Ru=self.Rc[u] if u == c else self.Rn[u], Rw=self.Rc[w] if w == c else self.Rn[w],
                        Pu=self.Pc[u] if u == c else self.Pn[u], Pw=self.Pc[w] if w == c else self.Pn[w]))

    @property
    def cells(self):
        return self.coarse.cells + self.fine.cells

    def _zero_inner_H(self):
        cg = self.coarse
        cg.Hx[self.inner[3]] = 0.0
        cg.Hy[self.inner[4]] = 0.0
        cg.Hz[self.inner[5]] = 0.0

    def _correct(self, Hf):
        """Replace, in the coarse surface E just updated, the inside half of its normal derivative: the fine H (Hf: {name: array}) full-weighted, over `dist` instead of the coarse dual."""
        cg = self.coarse
        for f in self.faces:
            heff = f['Ru'] @ Hf[f['H']][f['hf']] @ f['Rw'].T
            H = getattr(cg, f['H'])
            hout, hin = H[f['hout']], H[f['hin']]
            # the coarse update used (hin - hout) / hdual; replace it with (heff - hout) / dist
            getattr(cg, f['E'])[f['e']] += f['ce'] * f['k'] * ((heff - hout) / f['dist'] - (hin - hout) / f['hdual'])

    def _drive(self):
        fg, cg = self.fine, self.coarse
        for f in self.faces:
            getattr(fg, f['E'])[f['fe']] = f['Pu'] @ getattr(cg, f['E'])[f['e']] @ f['Pw'].T

    def _faces_E(self):
        cg = self.coarse
        return [getattr(cg, f['E'])[f['e']].copy() for f in self.faces]

    def _drive_lerp(self, old, new, a):
        fg = self.fine
        for f, e0, e1 in zip(self.faces, old, new):
            getattr(fg, f['E'])[f['fe']] = f['Pu'] @ ((1 - a) * e0 + a * e1) @ f['Pw'].T

    def step(self, source=None):
        """One step (a coarse one when subcycling); source(grid) adds the excitation to the coarse grid."""
        cg, fg = self.coarse, self.fine
        if not self.subcycle:
            cg.update_H()
            self._zero_inner_H()
            fg.update_H()
            cg.update_E()
            self._correct({'Hx': fg.Hx, 'Hy': fg.Hy, 'Hz': fg.Hz})
            if source:
                source(cg)
            fg.update_E()
            self._drive()
            return
        old = self._faces_E()
        cg.update_H()
        self._zero_inner_H()
        cg.update_E()
        if source:
            source(cg)
        pred = self._faces_E()                                              # predictor: no H inside the box
        acc = {'Hx': 0.0, 'Hy': 0.0, 'Hz': 0.0}
        for m in range(1, self.R + 1):
            fg.update_H()
            for k in acc:
                acc[k] = acc[k] + getattr(fg, k)
            fg.update_E()
            self._drive_lerp(old, pred, m / self.R)
        self._correct({k: v / self.R for k, v in acc.items()})            # corrector: the fine H's mean over the step
        self._drive()

    def energy(self):
        """The coarse grid's energy outside the box plus the fine grid's (the surface E counted on the coarse side)."""
        return self.coarse.energy(skip=self.inner) + self.fine.energy()
