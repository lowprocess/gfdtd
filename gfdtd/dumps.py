"""
Field dumps, in openEMS's files: a time-domain dump is a series of VTK rectilinear grids (<name>_<step>.vtr,
zlib-compressed binary, every sample step), a frequency-domain dump one HDF5 file (<name>.h5: /Mesh, /FieldData/FD
with one complex dataset per frequency, d_order NXYZ, as openEMS 0.37 writes it).

Values are at the cell centres (openEMS's dump mode 2): E and J averaged over the four parallel edges of a cell, H
over its two. J is rot H (the loop sum of the currents around each E edge over its dual area), Jc kappa E.
Frequency-domain values are 2 dt_s sum F(t) exp(-j w t) over the samples, as openEMS's (and its port DFT).

    d = Dumps(op, backend); d.sample(step); ...; d.close(path)
"""
import base64, os, zlib

import numpy as np

VTK_NAME = {'E': 'E-Field', 'H': 'H-Field', 'J': 'J-Field', 'Jc': 'Jc-Field'}
BLOCK = 32768           # VTK's compressed block size


def _take(xp, F, lo, hi):
    """F[lo:hi + 1] on each axis, with zeros where the range runs past the array."""
    out = xp.zeros([h - l + 1 for l, h in zip(lo, hi)], F.dtype)
    src, dst = [], []
    for a in range(3):
        s0, s1 = max(lo[a], 0), min(hi[a], F.shape[a] - 1)
        if s1 < s0:
            return out
        src.append(slice(s0, s1 + 1))
        dst.append(slice(s0 - lo[a], s1 - lo[a] + 1))
    out[tuple(dst)] = F[tuple(src)]
    return out


def _shift(lo, hi, a, d):
    lo, hi = list(lo), list(hi)
    lo[a] += d
    hi[a] += d
    return lo, hi


def _pair(xp, G, a):
    """The sums of neighbours along axis a (the array shrinks by one there)."""
    n = G.shape[a]
    return xp.take(G, xp.arange(n - 1), axis=a) + xp.take(G, xp.arange(1, n), axis=a)


class Region:
    """One dump box's points (cell centres or nodes), with the per-edge scale factors it needs on the backend's
    device. Cell centres: E and J average the four parallel edges of the cell, H its two. Nodes: E and J average
    the two edges along their direction, H the four around the node; edges past the domain don't count. On the GPU
    (E, H, J) one kernel does it all, and adds the DFT term in the same pass (kernels.py: dump)."""
    def __init__(self, d, op, be):
        xp = be.xp
        self.d, self.xp, self.be = d, xp, be
        self.gpu = getattr(be, 'name', '') == 'cupy' and d['field'] in ('E', 'H', 'J')
        if self.gpu:
            self.args = tuple(np.int32(v) for v in ({'E': 0, 'H': 1, 'J': 2}[d['field']], d.get('mode') != 'node',
                                                    *d['lo'], *[h - l + 1 for l, h in zip(d['lo'], d['hi'])]))
            self.shape = tuple(h - l + 1 for l, h in zip(d['lo'], d['hi']))
            self.omega = xp.asarray(2 * np.pi * np.asarray(d['fd'] or [0.0]), np.float32)
            return
        self.lo, self.hi = d['lo'], d['hi']
        self.node = d.get('mode') == 'node'
        self.shape = tuple(h - l + 1 for l, h in zip(self.lo, self.hi))
        f = d['field']
        self.ext, self.scale, self.count = [], [], []               # per component: edge range, scale, edges averaged
        for a in range(3):
            lo, hi = list(self.lo), list(self.hi)
            across = [a] if (f == 'H') != self.node else [b for b in range(3) if b != a]
            for b in across:                                        # the axes it averages along
                if self.node:
                    lo[b] -= 1
                else:
                    hi[b] += 1
            self.ext.append((lo, hi, across))
            if f == 'E':
                s_ = np.where(op.lE[a] > 0, 1.0 / np.where(op.lE[a] > 0, op.lE[a], 1), 0.0)
            elif f == 'H':
                s_ = np.where(op.lH[a] > 0, 1.0 / np.where(op.lH[a] > 0, op.lH[a], 1), 0.0)
            elif f == 'J':
                s_ = np.where(op.aE[a] > 0, 1.0 / np.where(op.aE[a] > 0, op.aE[a], 1), 0.0)
            else:                                                   # Jc: kappa E
                s_ = np.where(op.lE[a] > 0, op.kappa_E[a] / np.where(op.lE[a] > 0, op.lE[a], 1), 0.0)
            self.scale.append(xp.asarray(_take(np, s_, lo, hi), np.float32))
            if self.node:                                           # the edges that exist, per point
                exists = (op.lE[a] > 0) if f != 'H' else (op.iv[a] > 0)
                cnt = _take(np, exists.astype(np.float32), lo, hi)
                for b in across:
                    cnt = _pair(np, cnt, b)
                self.count.append(xp.asarray(1.0 / np.maximum(cnt, 1.0), np.float32))
            else:
                self.count.append(np.float32(0.5 ** len(across)))

    def _kernel(self, out, nf, t):
        be = self.be
        m = int(np.prod(self.shape)) * 3
        be.k['dump'](((m + be.BLOCK - 1) // be.BLOCK,), (be.BLOCK,), (be.V, be.I, be.G, *be.dims, *self.args,
                                                                    np.int32(nf), self.omega, np.float32(t), out))

    def accumulate(self, acc, t):
        """acc (nf, nx, ny, nz, 3) complex += the field x exp(-j w t), on the GPU in one kernel."""
        if self.gpu:
            self._kernel(acc, len(self.d['fd']), t)
        else:
            ph = np.exp(-2j * np.pi * np.asarray(self.d['fd']) * t).astype(np.complex64)
            acc += self.field(self.be.V, self.be.I)[None] * self.xp.asarray(ph).reshape((-1, 1, 1, 1, 1))

    def field(self, V, I):
        """(nx, ny, nz, 3) float32 on the device."""
        if self.gpu:
            out = self.xp.empty(self.shape + (3,), np.float32)
            self._kernel(out, 0, 0.0)
            return out
        xp, f = self.xp, self.d['field']
        out = xp.empty(self.shape + (3,), np.float32)
        for a in range(3):
            lo, hi, across = self.ext[a]
            if f == 'J':
                b, c = (a + 1) % 3, (a + 2) % 3
                G = (_take(xp, I[c], lo, hi) - _take(xp, I[c], *_shift(lo, hi, b, -1))
                     - _take(xp, I[b], lo, hi) + _take(xp, I[b], *_shift(lo, hi, c, -1)))
            else:
                G = _take(xp, (I if f == 'H' else V)[a], lo, hi)
            G = G * self.scale[a]
            for b in across:
                G = _pair(xp, G, b)
            out[..., a] = G * self.count[a]
        return out


def _b64_block(raw):
    """VTK's inline binary, compressed: the header (blocks, block size, last block, compressed sizes) and the data
    encoded separately."""
    blocks = [raw[i:i + BLOCK] for i in range(0, len(raw), BLOCK)] or [b'']
    comp = [zlib.compress(b, 1) for b in blocks]
    head = np.array([len(blocks), BLOCK, len(blocks[-1])] + [len(c) for c in comp], '<u4').tobytes()
    return (base64.b64encode(head) + base64.b64encode(b''.join(comp))).decode()


def write_vtr(path, coords, data, name):
    """A VTK rectilinear grid as openEMS writes its time-domain dumps: coords (m) per axis, data (nx, ny, nz, 3)."""
    nx, ny, nz = data.shape[:3]
    ext = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"
    vals = np.ascontiguousarray(np.transpose(data, (2, 1, 0, 3)), '<f4')          # x fastest
    mag = np.sqrt((vals.astype(np.float64) ** 2).sum(-1))
    out = ['<?xml version="1.0"?>',
           '<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" '
           'compressor="vtkZLibDataCompressor">',
           f'  <RectilinearGrid WholeExtent="{ext}">', f'  <Piece Extent="{ext}">', '    <PointData>',
           f'      <DataArray type="Float32" Name="{name}" NumberOfComponents="3" format="binary" RangeMin="'
           f'{mag.min() if mag.size else 0:g}" RangeMax="{mag.max() if mag.size else 0:g}">',
           '        ' + _b64_block(vals.tobytes()), '      </DataArray>', '    </PointData>', '    <CellData>',
           '    </CellData>', '    <Coordinates>']
    for ax, c in zip('xyz', coords):
        c = np.asarray(c, '<f8')
        out += [f'      <DataArray type="Float64" Name="{ax}" format="binary" RangeMin="{c.min():g}" '
                f'RangeMax="{c.max():g}">', '        ' + _b64_block(c.tobytes()), '      </DataArray>']
    out += ['    </Coordinates>', '  </Piece>', '  </RectilinearGrid>', '</VTKFile>', '']
    tmp = path + '.part'
    with open(tmp, 'w') as fh:
        fh.write('\n'.join(out))
    os.replace(tmp, path)                                           # a live viewer never sees half a frame


def write_h5(path, d, freqs, data):
    """openEMS's frequency-domain HDF5 dump: data (nf, nx, ny, nz, 3) complex."""
    import h5py
    tmp = path + '.part'
    with h5py.File(tmp, 'w') as h:
        h.attrs['openEMS_HDF5_version'] = np.float64(0.3)
        h.attrs['dump_type'] = np.int32(d['type'] % 10)
        g = h.create_group('Mesh')
        g.attrs['mesh_scaling'] = np.float64(d['unit'])
        g.attrs['mesh_type'] = np.int32(0)
        h.attrs['dual_mesh'] = np.uint8(d.get('mode') != 'node')
        for ax, c in zip('xyz', d['coords']):
            g.create_dataset(ax, data=np.asarray(c, np.float64))
        fd = h.create_group('FieldData').create_group('FD')
        fd.attrs['frequency'] = np.asarray(freqs, np.float64)
        for i, (f, v) in enumerate(zip(freqs, data)):
            ds = fd.create_dataset(f'f{i}', data=np.ascontiguousarray(np.transpose(v, (3, 0, 1, 2)), np.complex64))
            ds.attrs['d_order'] = 'NXYZ'
            ds.attrs['frequency'] = np.float64(f)
    os.replace(tmp, path)


class Dumps:
    """The dumps of a run: sample(step) at every sample step (time domain: a frame every `sub` of them; frequency
    domain: one DFT term), close() writes the frequency-domain files."""
    def __init__(self, op, be, path, notes=None):
        self.op, self.be, self.path, self.xp = op, be, path, be.xp
        self.regions = [Region(d, op, be) for d in op.dumps]
        self.acc, self.n = [], 0
        self.h5 = True
        if any(d['fd'] for d in op.dumps):
            try:
                import h5py                                         # noqa: F401
            except ImportError:
                self.h5 = False
                (notes if notes is not None else []).append("h5py is not installed: frequency-domain dumps skipped")
        for r in self.regions:
            fd = r.d['fd']
            self.acc.append(self.xp.zeros((len(fd),) + r.shape + (3,), np.complex64) if fd and self.h5 else None)
        self.dts = op.sub * op.dt

    def __bool__(self):
        return bool(self.regions)

    def sample(self, step):
        """The fields after timestep `step` (E at step dt, H half a step before)."""
        xp, be = self.xp, self.be
        for r, acc in zip(self.regions, self.acc):
            d = r.d
            if d['fd'] is None:
                if (step // self.op.sub) % d['sub'] == 0:
                    F = r.field(be.V, be.I)
                    F = be.cp.asnumpy(F) if xp is not np else F
                    write_vtr(os.path.join(self.path, f"{d['name']}_{step:010d}.vtr"), d['coords'], F,
                              VTK_NAME[d['field']])
            elif acc is not None:
                t = (step - (0.5 if d['field'] == 'H' else 0.0)) * self.op.dt
                r.accumulate(acc, max(t, 0.0))

    def close(self):
        """Write the frequency-domain dumps; returns the files written."""
        out = []
        for r, acc in zip(self.regions, self.acc):
            if acc is None:
                continue
            data = acc * np.float32(2 * self.dts)
            data = self.be.cp.asnumpy(data) if self.xp is not np else data
            fn = os.path.join(self.path, r.d['name'] + '.h5')
            write_h5(fn, r.d, r.d['fd'], data)
            out.append(fn)
        return out
