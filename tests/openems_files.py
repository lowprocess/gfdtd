"""Readers of the openEMS files gfdtd writes, for the tests: the .vtr frames of a time-domain dump, the HDF5 file of
a frequency-domain dump and the step count of a run folder, read as openEMS's own files are read."""
import base64
import glob
import os
import xml.etree.ElementTree as ET
import zlib

import numpy as np

_VTK_TYPES = {'Float32': '<f4', 'Float64': '<f8', 'Int32': '<i4', 'UInt32': '<u4'}


def _decode_binary(text, header_type, compressed, dtype):
    """One inline binary VTK DataArray (base64) -> numpy array."""
    text = ''.join(text.split())
    hdt = np.dtype('<u8' if header_type == 'UInt64' else '<u4')
    hs = hdt.itemsize
    if not compressed:
        raw = base64.b64decode(text)
        n = int(np.frombuffer(raw[:hs], hdt)[0])
        return np.frombuffer(raw[hs:hs + n], dtype)
    first = base64.b64decode(text[:((hs + 2) // 3) * 4])           # the block count leads the header
    nb = int(np.frombuffer(first[:hs], hdt)[0])
    hlen = ((hs * (3 + nb) + 2) // 3) * 4                          # header and data are encoded separately
    head = np.frombuffer(base64.b64decode(text[:hlen])[:hs * (3 + nb)], hdt)
    data = base64.b64decode(text[hlen:])
    out, pos = [], 0
    for c in head[3:]:
        out.append(zlib.decompress(data[pos:pos + int(c)]))
        pos += int(c)
    return np.frombuffer(b''.join(out), dtype)


def read_vtr(path):
    """A time-domain dump frame (.vtr) -> (mesh {x, y, z} mm, field (nx, ny, nz, 3) float32)."""
    root = ET.parse(path).getroot()
    ht = root.get('header_type', 'UInt32')
    comp = root.get('compressor') is not None

    def arr(el):
        dt = np.dtype(_VTK_TYPES[el.get('type')])
        if el.get('format', 'ascii') == 'ascii':
            return np.array(el.text.split(), dt)
        return _decode_binary(el.text or '', ht, comp, dt)

    piece = root.find('RectilinearGrid/Piece')
    coords = [arr(el) for el in piece.find('Coordinates').findall('DataArray')]
    mesh = {ax: (c * 1e3 if np.abs(c).max() < 2.0 else c).astype(float) for ax, c in zip('xyz', coords)}  # m -> mm
    el = piece.find('PointData/DataArray')
    nc = int(el.get('NumberOfComponents', 1))
    v = arr(el).astype(np.float32).reshape(len(coords[2]), len(coords[1]), len(coords[0]), nc)   # x fastest
    return mesh, np.transpose(v, (2, 1, 0, 3))


def read_fd(path):
    """A frequency-domain dump (HDF5) -> (mesh {x, y, z} mm, [(freq Hz, field (nx, ny, nz, 3) complex)])."""
    import h5py
    with h5py.File(path, 'r') as h:
        sc = h['Mesh'].attrs.get('mesh_scaling', None)
        mesh = {}
        for ax in 'xyz':
            v = np.asarray(h['Mesh'][ax], float)
            mesh[ax] = v * 1e3 if np.abs(v).max() < 2.0 else v * (float(sc) * 1e3 if sc is not None else 1.0)
        fd = h['FieldData']['FD']
        out = []
        for i in sorted(int(k[1:]) for k in fd.keys() if k[1:].isdigit()):
            ds = fd[f'f{i}']
            assert ds.attrs['d_order'] in ('NXYZ', b'NXYZ')
            out.append((float(np.atleast_1d(ds.attrs['frequency'])[0]), np.moveaxis(np.asarray(ds), 0, -1)))
    return mesh, out


def _times(fn, n=None):
    """The first column of a text probe file (comment lines start with %); the first n values, or all."""
    out = []
    with open(fn) as fh:
        for line in fh:
            if line.strip() and not line.startswith('%'):
                out.append(float(line.split()[0]))
                if n and len(out) >= n:
                    break
    return out


def steps_taken(path):
    """The timesteps a finished run took: the last sample time of its port voltage probe over the time step of
    the excitation file 'et' (one sample per step). None without the files."""
    t = _times(os.path.join(path, 'et'), 3) if os.path.exists(os.path.join(path, 'et')) else []
    dt = t[2] - t[1] if len(t) >= 3 and t[2] > t[1] else None
    for fn in sorted(glob.glob(os.path.join(path, 'port_ut*'))):
        t = _times(fn)
        if t and dt:
            return int(round(t[-1] / dt))
    return None
