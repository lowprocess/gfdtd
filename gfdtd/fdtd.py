"""
openEMS's Python interface over gfdtd's engine: the FDTD settings, lumped ports and Run, so a script (or
kicad2ems's build_fdtd) written for openEMS runs here by swapping the two imports:

    from gfdtd import ContinuousStructure, openEMS
    FDTD = openEMS(NrTS=100000, EndCriteria=1e-4); FDTD.SetGaussExcite(f0, fc); FDTD.SetBoundaryCond(['MUR'] * 6)
    CSX = ContinuousStructure(); FDTD.SetCSX(CSX); ...
    port = FDTD.AddLumpedPort(1, 50, start, stop, 'z', excite=1)
    FDTD.Run(sim_path); port.CalcPort(sim_path, freq)          # port.uf_inc, uf_ref, ...

FDTD.Write2XML / ReadFromXML use openEMS's XML (an <openEMS> file with <FDTD> and <ContinuousStructure>):
`python -m gfdtd model.xml` runs a file written by openEMS (or by AppCSXCAD's tools) like `openEMS model.xml`.
"""
import glob, os, sys
import xml.etree.ElementTree as ET

import numpy as np

from . import csx as C

BC_NAMES = {0: 'PEC', 1: 'PMC', 2: 'MUR', 3: 'PML_8'}
SIDES = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
OUTPUTS = ('port_ut*', 'port_it*', 'et', 'ht', '*.vtr', '*.h5')     # cleanup=True removes these only


def _bc(b):
    if isinstance(b, (int, np.integer)) or str(b).strip().isdigit():
        return BC_NAMES[int(b)]
    return str(b).strip().upper()


class openEMS:                                                       # noqa: N801  (openEMS's own name)
    writes_log = True           # Run(log=path): its output into a file, without redirecting the process's stdout

    def __init__(self, NrTS=1e9, EndCriteria=1e-5, OverSampling=4, backend='auto', **kw):
        self.nr_ts, self.end_crit, self.oversampling = int(NrTS), float(EndCriteria), int(OverSampling)
        self.excitation = None
        self.bc = ['PEC'] * 6
        self.csx = None
        self.ports = []
        self.backend = backend
        self.op = None
        if kw.get('MaxTime'):
            raise NotImplementedError("gfdtd: MaxTime")

    # ---- settings --------------------------------------------------------------------------------------------
    def SetNumberOfTimeSteps(self, val):
        self.nr_ts = int(val)

    def SetEndCriteria(self, val):
        self.end_crit = float(val)

    def SetOverSampling(self, val):
        self.oversampling = int(val)

    def SetGaussExcite(self, f0, fc):
        self.excitation = dict(type=0, f0=float(f0), fc=float(fc))

    def SetBoundaryCond(self, BC):
        if len(BC) != 6:
            raise ValueError("SetBoundaryCond: six values (xmin, xmax, ymin, ymax, zmin, zmax)")
        self.bc = [_bc(b) for b in BC]

    def SetCSX(self, CSX):
        self.csx = CSX

    def AddLumpedPort(self, port_nr, R, start, stop, p_dir, excite=0, **kw):
        port = LumpedPort(self.csx, port_nr, R, start, stop, p_dir, excite, **kw)
        self.ports.append(port)
        return port

    def settings(self):
        if self.excitation is None:
            raise ValueError("gfdtd: no excitation (SetGaussExcite)")
        return dict(bc=self.bc, f0=self.excitation['f0'], fc=self.excitation['fc'], nr_ts=self.nr_ts,
                    end_crit=self.end_crit, oversampling=self.oversampling)

    def CreateNF2FFBox(self, name='nf2ff', start=None, stop=None, **kw):
        """A near-field to far-field box (gfdtd.nf2ff.nf2ff), placed as openEMS places it: inside the absorbing
        boundaries (a Mur side's 2 lines, a PML_n side's n + 1), no face on a PEC / PMC side, which mirrors."""
        from .nf2ff import nf2ff
        if self.csx is None:
            raise ValueError("CreateNF2FFBox: CSX is not set")
        directions, mirror, size = [True] * 6, [0] * 6, [0] * 6
        for n, b in enumerate(self.bc):
            if b == 'PEC':
                directions[n], mirror[n] = False, 1
            elif b == 'PMC':
                directions[n], mirror[n] = False, 2
            elif b == 'MUR':
                size[n] = 2
            elif b.startswith('PML'):
                size[n] = int(b.split('_')[1]) + 1 if '_' in b else 9
        if start is None or stop is None:
            start, stop = np.zeros(3), np.zeros(3)
            for a in range(3):
                lines = np.asarray(self.csx.GetGrid().GetLines(a), float)
                if len(lines) <= size[2 * a] + size[2 * a + 1]:
                    raise ValueError("CreateNF2FFBox: not enough lines in some direction")
                start[a], stop[a] = lines[size[2 * a]], lines[-size[2 * a + 1] - 1]
        directions = kw.pop('directions', directions)
        mirror = kw.pop('mirror', mirror)
        return nf2ff(self.csx, name, start, stop, directions=directions, mirror=mirror, **kw)

    # ---- XML --------------------------------------------------------------------------------------------------
    def Write2XML(self, file):
        top = ET.Element('openEMS')
        f = ET.SubElement(top, 'FDTD', NumberOfTimesteps=str(self.nr_ts), endCriteria=f"{self.end_crit:g}",
                          OverSampling=str(self.oversampling))
        if self.excitation:
            ET.SubElement(f, 'Excitation', Type='0', f0=f"{self.excitation['f0']:g}", fc=f"{self.excitation['fc']:g}")
        ET.SubElement(f, 'BoundaryCond', **dict(zip(SIDES, self.bc)))
        top.append(self.csx.to_xml())
        C._write(top, file)

    def ReadFromXML(self, file):
        root = ET.parse(file).getroot()
        f = root.find('FDTD')
        if f is None:
            raise ValueError(f"{file}: no <FDTD> settings")
        self.nr_ts = int(float(f.get('NumberOfTimesteps', 1e9)))
        self.end_crit = float(f.get('endCriteria', 1e-5))
        self.oversampling = int(float(f.get('OverSampling', 4)))
        ex = f.find('Excitation')
        if ex is None or int(ex.get('Type', 0)) != 0:
            raise ValueError(f"{file}: gfdtd runs a Gaussian excitation (Type 0) only")
        self.SetGaussExcite(float(ex.get('f0')), float(ex.get('fc')))
        b = f.find('BoundaryCond')
        self.bc = [_bc(b.get(s, 'PEC')) for s in SIDES] if b is not None else ['PEC'] * 6
        self.csx = C.ContinuousStructure()
        self.csx.from_xml(root.find('ContinuousStructure'))
        return True

    # ---- the run ----------------------------------------------------------------------------------------------
    def Run(self, sim_path, cleanup=False, setup_only=False, verbose=None, backend=None, log=None, **kw):
        """Run in sim_path (made if missing; cleanup: remove openEMS's own outputs there first). Returns 0.
        numThreads: the CPU core's threads (0: all); openEMS's debug options are accepted and ignored; backend:
        'auto', 'cupy', 'core', 'numpy'; log: a file for the output (appended) in place of stdout. On the GPU a run
        has a stream of its own, so that runs in several threads overlap."""
        if log is None:
            return self._run(sim_path, cleanup, setup_only, verbose, backend, lambda line: print(line, flush=True), **kw)
        with open(log, 'a') as fh:
            def out(line):
                fh.write(line + '\n')
                fh.flush()
            return self._run(sim_path, cleanup, setup_only, verbose, backend, out, **kw)

    def _run(self, sim_path, cleanup, setup_only, verbose, backend, out, **kw):
        from .operator import build
        from .engine import resolve_backend
        sim_path = os.path.abspath(sim_path)
        if cleanup and os.path.isdir(sim_path):
            for pat in OUTPUTS:
                for fn in glob.glob(os.path.join(sim_path, pat)):
                    os.remove(fn)
        os.makedirs(sim_path, exist_ok=True)
        from . import __version__
        out(f" ---------------------------------------------------------------------- \n"
            f" | gfdtd -- version {__version__} -- GPU FDTD, openEMS / CSXCAD compatible\n"
            f" ---------------------------------------------------------------------- ")
        self.op = op = build(self.csx, self.settings(), log=out)
        if resolve_backend(backend or self.backend) == 'cupy':
            import cupy
            with cupy.cuda.Stream(non_blocking=True) as stream:
                r = self._engine(op, backend, setup_only, verbose, out, sim_path, **kw)
                stream.synchronize()
            return r
        return self._engine(op, backend, setup_only, verbose, out, sim_path, **kw)

    def _engine(self, op, backend, setup_only, verbose, out, sim_path, **kw):
        from .engine import Engine, gpu_info, core_info
        eng = Engine(op, backend or self.backend)
        if eng.be.name == 'core' and int(kw.get('numThreads') or 0) > 0:
            eng.be.core.set_threads(int(kw['numThreads']))
        out(f"Engine: {eng.be.name}" + {'cupy': f" ({gpu_info()[1]})", 'core': f" ({core_info()[1]})"}.get(eng.be.name, ''))
        if setup_only:
            return 0
        self.result = eng.run(sim_path, self.nr_ts, self.end_crit, verbose=1 if verbose is None else verbose, out=out)
        return 0

    def SetAbort(self, val):
        raise NotImplementedError("gfdtd: write a file ABORT into the run folder")


def read_probe(path):
    """(t, values) of an openEMS / gfdtd probe file."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip() and not line.startswith('%'):
                rows.append(line.split()[:2])
    a = np.array(rows, float).reshape(-1, 2)
    return a[:, 0], a[:, 1]


def dft(t, val, freq):
    """openEMS's DFT_time2freq for a pulse: the single-sided spectrum 2 dt sum(val exp(-j w t))."""
    freq = np.atleast_1d(np.asarray(freq, float))
    out = np.empty(len(freq), complex)
    for i in range(0, len(freq), 64):                               # blocks: (64, nt) at a time
        f = freq[i:i + 64]
        out[i:i + 64] = np.exp(-2j * np.pi * np.outer(f, t)) @ val
    return 2 * out * (t[1] - t[0])


class LumpedPort:
    """openEMS's LumpedPort: a resistor R across the box along p_dir, a soft E excitation (when excite != 0), a
    voltage probe through its middle and a current probe across its middle; CalcPort gives uf_tot, if_tot, uf_inc,
    uf_ref, if_inc, if_ref (and the time signals ut_*, it_*) as openEMS's does."""
    def __init__(self, CSX, port_nr, R, start, stop, exc_dir, excite=0, **kw):
        self.CSX, self.number, self.R, self.excite = CSX, port_nr, float(R), excite
        self.start, self.stop = np.array(start, float), np.array(stop, float)
        self.exc_ny = C.ny_dir(exc_dir)
        self.priority = kw.get('priority', 0)
        prefix = kw.get('PortNamePrefix', '')
        lbl = prefix + 'port_{}' + f'_{port_nr}'
        a = self.exc_ny
        self.direction = np.sign(self.stop[a] - self.start[a])
        if self.direction == 0:
            raise ValueError('LumpedPort: start and stop may not be identical in excitation direction')
        if self.R > 0:
            el = CSX.AddLumpedElement(lbl.format('resist'), ny=a, caps=True, R=self.R)
        else:
            el = CSX.AddMetal(lbl.format('resist'))
        el.AddBox(self.start, self.stop, priority=self.priority)
        if excite != 0:
            vec = np.zeros(3)
            vec[a] = -1 * self.direction * excite
            CSX.AddExcitation(lbl.format('excite'), exc_type=0, exc_val=vec).AddBox(self.start, self.stop,
                                                                                    priority=self.priority)
        mid = 0.5 * (self.start + self.stop)
        u0, u1 = mid.copy(), mid.copy()
        u0[a], u1[a] = self.start[a], self.stop[a]
        self.U_filenames = [lbl.format('ut')]
        CSX.AddProbe(self.U_filenames[0], p_type=0, weight=-1).AddBox(u0, u1)
        i0, i1 = self.start.copy(), self.stop.copy()
        i0[a] = i1[a] = mid[a]
        self.I_filenames = [lbl.format('it')]
        CSX.AddProbe(self.I_filenames[0], p_type=1, weight=self.direction, norm_dir=a).AddBox(i0, i1)
        self.Z_ref = None

    def ReadUIData(self, sim_path, freq, signal_type='pulse'):
        self.u_time, self.ut_tot = read_probe(os.path.join(sim_path, self.U_filenames[0]))
        self.i_time, self.it_tot = read_probe(os.path.join(sim_path, self.I_filenames[0]))
        self.uf_tot = dft(self.u_time, self.ut_tot, freq)
        self.if_tot = dft(self.i_time, self.it_tot, freq)

    def CalcPort(self, sim_path, freq, ref_impedance=None, ref_plane_shift=None, signal_type='pulse'):
        self.ReadUIData(sim_path, freq, signal_type)
        self.Z_ref = self.R if ref_impedance is None else ref_impedance
        Z = self.Z_ref
        self.uf_inc = 0.5 * (self.uf_tot + self.if_tot * Z)
        self.if_inc = 0.5 * (self.if_tot + self.uf_tot / Z)
        self.uf_ref = self.uf_tot - self.uf_inc
        self.if_ref = self.if_inc - self.if_tot
        if np.isscalar(Z):
            n = min(len(self.ut_tot), len(self.it_tot))
            self.ut_inc = 0.5 * (self.ut_tot[:n] + self.it_tot[:n] * Z)
            self.it_inc = 0.5 * (self.it_tot[:n] + self.ut_tot[:n] / Z)
            self.ut_ref = self.ut_tot[:n] - self.ut_inc
            self.it_ref = self.it_inc - self.it_tot[:n]
        self.P_inc = 0.5 * np.real(self.uf_inc * np.conj(self.if_inc))
        self.P_ref = 0.5 * np.real(self.uf_ref * np.conj(self.if_ref))
        self.P_acc = 0.5 * np.real(self.uf_tot * np.conj(self.if_tot))


def main(argv=None):
    """python -m gfdtd model.xml [--backend auto|cupy|numpy] [--path DIR]: run an openEMS XML file (outputs in the
    current folder, or --path), as `openEMS model.xml` does."""
    import argparse
    ap = argparse.ArgumentParser(prog='gfdtd', description=main.__doc__)
    ap.add_argument('xml')
    ap.add_argument('--backend', default='auto', help="auto (GPU when CuPy finds one), cupy or numpy")
    ap.add_argument('--path', default='.', help="folder for the probe files (default: the current folder)")
    ap.add_argument('--numThreads', type=int, default=0, help=argparse.SUPPRESS)
    a, _ = ap.parse_known_args(argv)
    F = openEMS()
    F.ReadFromXML(a.xml)
    for s in F.csx.skipped:
        print(f"note: {s}")
    F.Run(a.path, backend=a.backend)
    return 0 if F.result['reason'] != 'unstable' else 1


if __name__ == '__main__':
    sys.exit(main())
