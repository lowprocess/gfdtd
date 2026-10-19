"""
Near-field to far-field, as openEMS's nf2ff (openEMS.nf2ff): a box of six faces records E and H in the frequency
domain (node-interpolated HDF5 dumps <name>_E_<n>.h5 / <name>_H_<n>.h5), and CalcNF2FF turns them into the far
field over (theta, phi), written in openEMS's result file and returned as openEMS's nf2ff_results.

    nf = FDTD.CreateNF2FFBox(frequency=[2.4e9])     # before the run: the faces inside the absorbing boundaries
    FDTD.Run(path)
    r = nf.CalcNF2FF(path, [2.4e9], theta, phi, center=[0, 0, 0])
    r.Dmax, r.Prad, r.E_norm[0]                      # (theta, phi) per frequency

The faces carry the equivalent currents J = n x H and M = -n x E (n: outward), summed with trapezoid weights at
the nodes. The far field (e^jwt, r -> inf):

    N = sum J w e^(jk r^.r'),  L = sum M w e^(jk r^.r')
    E_theta = -jk e^-jkr / (4 pi r) (L_phi + eta N_theta),  E_phi = jk e^-jkr / (4 pi r) (L_theta - eta N_phi)

On a planar face e^(jk r^.r') is a product over its two axes, so each sum is two matrix products. Mirrors (a PEC or
PMC side the box rests on) add the image currents. P_rad = |E|^2 / (2 eta). Prad is the power through the
recorded faces, 1/2 Re sum (E x H*).n w (the images not counted), and Dmax = 4 pi r^2 max(P_rad) / Prad, as
openEMS's.
"""
import os

import numpy as np

C0, MU0 = 299792458.0, 1.25663706212e-6
ETA0 = MU0 * C0


class nf2ff:                                                         # noqa: N801  (openEMS's name)
    """openEMS's nf2ff recording box: directions (6 bools: xmin, xmax, ...), mirror (6: 0 off, 1 PEC, 2 PMC),
    frequency (Hz: frequency-domain recording; gfdtd has no time-domain NF2FF)."""
    def __init__(self, CSX, name, start, stop, directions=None, mirror=None, frequency=None, **kw):
        self.CSX, self.name = CSX, name
        self.start, self.stop = np.asarray(start, float), np.asarray(stop, float)
        self.directions = list(directions) if directions is not None else [True] * 6
        self.mirror = list(mirror) if mirror is not None else [0] * 6
        if frequency is None:
            raise NotImplementedError("gfdtd: NF2FF records in the frequency domain only (frequency=...)")
        self.freq = list(np.atleast_1d(np.asarray(frequency, float)))
        self.e_file, self.h_file = f"{name}_E", f"{name}_H"
        kw = {k: v for k, v in kw.items() if k not in ('dump_type', 'dump_mode', 'file_type')}
        self.e_dump = CSX.AddDump(self.e_file, dump_type=10, dump_mode=1, file_type=1, frequency=self.freq, **kw)
        self.h_dump = CSX.AddDump(self.h_file, dump_type=11, dump_mode=1, file_type=1, frequency=self.freq, **kw)
        self.faces = []                                             # (axis, +1 / -1) of each box, in dump order
        for a in range(3):
            for side in (0, 1):
                if self.directions[2 * a + side]:
                    s, e = self.start.copy(), self.stop.copy()
                    if side:
                        s[a] = e[a]
                    else:
                        e[a] = s[a]
                    self.e_dump.AddBox(s, e)
                    self.h_dump.AddBox(s, e)
                    self.faces.append((a, 1 if side else -1))
        self.theta = self.phi = self.center = None

    def _files(self, sim_path, n):
        tag = f"_{n}" if len(self.faces) > 1 else ''
        return (os.path.join(sim_path, f"{self.e_file}{tag}.h5"), os.path.join(sim_path, f"{self.h_file}{tag}.h5"))

    def CalcNF2FF(self, sim_path, freq, theta, phi, radius=1, center=(0, 0, 0), outfile=None, read_cached=True,
                  verbose=0):
        """The far field at radius (m) over theta, phi (degrees), phase centre `center` (m); written to outfile
        (in sim_path; default <name>.h5) and returned as nf2ff_results."""
        freq = list(np.atleast_1d(np.asarray(freq, float)))
        theta, phi = np.atleast_1d(np.asarray(theta, float)), np.atleast_1d(np.asarray(phi, float))
        self.theta, self.phi, self.center = theta, phi, center
        fn = os.path.join(sim_path, outfile if outfile else self.name + '.h5')
        if read_cached and os.path.exists(fn):
            r = nf2ff_results(fn)
            if (len(r.freq) == len(freq) and np.allclose(r.freq, freq, rtol=1e-6) and
                    np.allclose(np.degrees(r.theta), theta) and np.allclose(np.degrees(r.phi), phi)):
                return r
        faces = []
        for n, (a, sgn) in enumerate(self.faces):
            fe, fh = self._files(sim_path, n)
            if os.path.exists(fe) and os.path.exists(fh):
                faces.append(_face(fe, fh, a, sgn, freq))
        if not faces:
            raise RuntimeError(f"CalcNF2FF: no recorded faces of {self.name} in {sim_path}")
        for a in range(3):                                          # image currents of the mirrors
            for side, pos in ((0, self.start[a]), (1, self.stop[a])):
                kind = self.mirror[2 * a + side]
                if kind:
                    faces += [_image(f, a, pos * _unit(self.CSX), kind) for f in faces]
        res = far_field(faces, freq, np.radians(theta), np.radians(phi), radius, np.asarray(center, float))
        write_result(fn, freq, np.radians(theta), np.radians(phi), radius, res)
        return nf2ff_results(fn)


def _unit(CSX):
    try:
        return float(CSX.GetGrid().GetDeltaUnit())
    except AttributeError:
        return 1.0


def _read_fd(path, freq):
    """(coords m [x, y, z], [field (nx, ny, nz, 3) complex per freq]) of one of gfdtd's HDF5 dumps."""
    import h5py
    with h5py.File(path, 'r') as h:
        coords = [np.asarray(h['Mesh'][ax], float) for ax in 'xyz']
        fd = h['FieldData/FD']
        have = np.atleast_1d(fd.attrs['frequency'])
        out = []
        for f in freq:
            i = int(np.argmin(np.abs(have - f)))
            if abs(have[i] - f) > 1e-6 * f:
                raise ValueError(f"CalcNF2FF: {f / 1e9:g} GHz was not recorded ({path})")
            out.append(np.transpose(np.asarray(fd[f'f{i}']), (1, 2, 3, 0)))
    return coords, out


def _weights(c):
    """Trapezoid weights at the nodes c (one node: 1)."""
    if len(c) < 2:
        return np.ones(len(c))
    d = np.diff(c)
    return np.concatenate([[d[0] / 2], (d[1:] + d[:-1]) / 2, [d[-1] / 2]])


def _face(fe, fh, a, sgn, freq):
    """dict(coords, J, M) of one face: J = n x H, M = -n x E (per frequency, (nx, ny, nz, 3)), weights folded in."""
    coords, E = _read_fd(fe, freq)
    _, H = _read_fd(fh, freq)
    n = np.zeros(3)
    n[a] = sgn
    w = np.einsum('i,j,k->ijk', *[_weights(c) if b != a else np.ones(len(c)) for b, c in enumerate(coords)])
    J = [np.cross(n, h) * w[..., None] for h in H]
    M = [-np.cross(n, e) * w[..., None] for e in E]
    P = [0.5 * float(np.real(np.sum(np.cross(e, np.conj(h))[..., a] * w))) * sgn for e, h in zip(E, H)]
    return dict(coords=coords, J=J, M=M, P=P)


def _image(f, a, pos, kind):
    """The image of a face in a mirror plane at pos on axis a: PEC (1) keeps the normal J and the tangential M,
    PMC (2) the other two."""
    coords = [c.copy() for c in f['coords']]
    coords[a] = 2 * pos - coords[a]
    sj = np.full(3, -1.0 if kind == 1 else 1.0)
    sj[a] = -sj[a]
    return dict(coords=coords, J=[j * sj for j in f['J']], M=[m * -sj for m in f['M']], P=[0.0] * len(f['P']))


def far_field(faces, freq, th, ph, r, center):
    """[dict(E_theta, E_phi, P_rad (ntheta, nphi), Prad, Dmax)] per frequency."""
    T, P = np.meshgrid(th, ph, indexing='ij')
    rh = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3)   # (ndir, 3)
    out = []
    for fi, f in enumerate(freq):
        k = 2 * np.pi * f / C0
        N = np.zeros((len(rh), 3), complex)
        L = np.zeros((len(rh), 3), complex)
        for face in faces:
            ph_ax = [np.exp(1j * k * np.outer(rh[:, b], face['coords'][b] - center[b])) for b in range(3)]   # (ndir, n_b)
            for src, acc in ((face['J'][fi], N), (face['M'][fi], L)):
                nx, ny, nz = src.shape[:3]
                for c in range(3):
                    S = src[..., c]
                    if not np.any(S):
                        continue
                    # sum_xyz Px[d, x] Py[d, y] Pz[d, z] S[x, y, z], one axis at a time
                    t = (ph_ax[0] @ S.reshape(nx, ny * nz)).reshape(-1, ny, nz)              # (ndir, ny, nz)
                    t = np.einsum('dyz,dy->dz', t, ph_ax[1])
                    acc[:, c] += np.einsum('dz,dz->d', t, ph_ax[2])
        st, ct, sp, cph = np.sin(T).ravel(), np.cos(T).ravel(), np.sin(P).ravel(), np.cos(P).ravel()
        theta_c = lambda A: (A[:, 0] * cph + A[:, 1] * sp) * ct - A[:, 2] * st
        phi_c = lambda A: -A[:, 0] * sp + A[:, 1] * cph
        g = 1j * k * np.exp(-1j * k * r) / (4 * np.pi * r)
        Et = (-g * (phi_c(L) + ETA0 * theta_c(N))).reshape(T.shape)
        Ep = (g * (theta_c(L) - ETA0 * phi_c(N))).reshape(T.shape)
        Pr = (np.abs(Et) ** 2 + np.abs(Ep) ** 2) / (2 * ETA0)
        prad = sum(face['P'][fi] for face in faces)
        dmax = float(4 * np.pi * r * r * Pr.max() / prad) if prad > 0 else 0.0
        out.append(dict(E_theta=Et, E_phi=Ep, P_rad=Pr, Prad=prad, Dmax=dmax))
    return out


def write_result(fn, freq, th, ph, r, res):
    """openEMS's NF2FF result file, in the layout its Octave / Matlab interface reads (real and imaginary parts,
    (phi, theta)): openEMS's nf2ff_results and kicad2ems's antenna.read_sphere both read it."""
    import h5py
    with h5py.File(fn, 'w') as h:
        m = h.create_group('Mesh')
        m.create_dataset('theta', data=np.asarray(th, float))
        m.create_dataset('phi', data=np.asarray(ph, float))
        m.create_dataset('r', data=np.atleast_1d(float(r)))
        g = h.create_group('nf2ff')
        g.attrs['Frequency'] = np.asarray(freq, float)
        g.attrs['Dmax'] = np.array([x['Dmax'] for x in res])
        g.attrs['Prad'] = np.array([x['Prad'] for x in res])
        for name in ('E_theta', 'E_phi', 'P_rad'):
            fd = g.create_group(name).create_group('FD')
            for n, x in enumerate(res):
                v = np.asarray(x[name]).T                           # (phi, theta)
                if name == 'P_rad':
                    fd.create_dataset(f'f{n}', data=v.astype(float))
                else:
                    fd.create_dataset(f'f{n}_real', data=v.real)
                    fd.create_dataset(f'f{n}_imag', data=v.imag)


class nf2ff_results:                                                 # noqa: N801  (openEMS's name)
    """openEMS's nf2ff results: theta, phi (rad), r, freq, Dmax, Prad, and per frequency E_theta, E_phi, E_norm,
    E_cprh, E_cplh, P_rad (theta, phi)."""
    def __init__(self, fn):
        import h5py
        self.fn = fn
        with h5py.File(fn, 'r') as h:
            self.phi, self.theta = np.array(h['Mesh/phi']), np.array(h['Mesh/theta'])
            self.r = np.array(h['Mesh/r'])
            d = h['nf2ff']
            self.freq, self.Dmax, self.Prad = (np.array(d.attrs[k]) for k in ('Frequency', 'Dmax', 'Prad'))
            TH, PH = np.meshgrid(self.theta, self.phi, indexing='ij')
            cp, sp = np.cos(PH), np.sin(PH)
            self.E_theta, self.E_phi, self.P_rad, self.E_norm, self.E_cprh, self.E_cplh = [], [], [], [], [], []
            for n in range(len(self.freq)):
                rd = lambda name: (np.array(d[f'{name}/FD/f{n}_real']) + 1j * np.array(d[f'{name}/FD/f{n}_imag'])).T
                Et, Ep = rd('E_theta'), rd('E_phi')
                self.E_theta.append(Et)
                self.E_phi.append(Ep)
                self.P_rad.append(np.array(d[f'P_rad/FD/f{n}']).T)
                self.E_norm.append(np.sqrt(np.abs(Et) ** 2 + np.abs(Ep) ** 2))
                self.E_cprh.append((cp + 1j * sp) * (Et + 1j * Ep) / np.sqrt(2.0))
                self.E_cplh.append((cp - 1j * sp) * (Et - 1j * Ep) / np.sqrt(2.0))
