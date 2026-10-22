"""
openEMS's transmission-line and waveguide ports on gfdtd's CSX: MSLPort, StripLinePort, CPWPort, CoaxialPort,
WaveguidePort, RectWGPort, CircWGPort and CurvePort, made as openEMS's ports.py makes them (the same metal, probes,
excitations and post-processing, so a script that uses them runs on gfdtd unchanged and reads the same results).
Ported from openEMS's Python interface (ports.py, (C) 2015-2026 Thorsten Liebig and the openEMS contributors, GPL-3).
The lumped port is gfdtd.fdtd.LumpedPort.

    port = FDTD.AddMSLPort(1, metal, start, stop, 'x', 'z', excite=-1, FeedShift=10, MeasPlaneShift=30)
    FDTD.Run(path); port.CalcPort(path, f, ref_impedance=50)       # uf_inc, uf_ref, beta, Z_ref, ...

The transmission-line ports (MSL, stripline, CPW, coax) take the line's impedance and propagation constant from three
voltage and two current probes along it; the waveguide ports match the field on the port plane to the mode (probe
types 10 and 11) and take the impedance from the mode's cutoff. Mode purity (openEMS's extra probe column) is not
written.
"""
import os

import numpy as np

from . import csx as C

C0 = 299792458.0
MUE0 = 4e-7 * np.pi
EPS0 = 1 / (MUE0 * C0 ** 2)
Z0 = np.sqrt(MUE0 / EPS0)


def _load_ui_file(path):
    """(data (N, cols), column names) of a probe file: the last header line names the columns."""
    comments, rows = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith('%'):
                comments.append(line[1:].strip())
            elif line.strip():
                rows.append(line.split())
    names = comments[-1].split() if comments else None
    return np.array(rows, float).reshape(len(rows), -1), names


def dft_time2freq(t, val, freq, signal_type='pulse'):
    """openEMS's DFT_time2freq: the single-sided spectrum of val(t) at freq."""
    freq = np.atleast_1d(np.asarray(freq, float))
    f_val = np.exp(-2j * np.pi * np.outer(freq, t)) @ val
    if signal_type == 'pulse':
        f_val *= t[1] - t[0]
    elif signal_type == 'periodic':
        f_val /= len(t)
    else:
        raise ValueError(f'unknown signal type "{signal_type}"')
    return 2 * f_val


class UI_data:
    def __init__(self, fns, path, freq, signal_type='pulse', **kw):
        self.path, self.fns = path, [fns] if isinstance(fns, str) else list(fns)
        self.freq = [freq] if np.isscalar(freq) else freq
        self.ui_time, self.ui_val, self.ui_f_val, self.col_names, self.ui_mode_purity = [], [], [], [], []
        for fn in self.fns:
            data, names = _load_ui_file(os.path.join(path, fn))
            self.col_names.append(names)
            self.ui_time.append(data[:, 0])
            self.ui_val.append(data[:, 1])
            self.ui_f_val.append(dft_time2freq(data[:, 0], data[:, 1], self.freq, signal_type))
            purity = names is not None and len(names) > 1 and names[-1] == 'mode_purity'
            self.ui_mode_purity.append(data[:, -1] if purity else None)


class Port:
    """The port base class: CSX, port number, start / stop box, excitation amplitude; priority, PortNamePrefix and
    delay (s) as keywords."""
    def __init__(self, CSX, port_nr, start, stop, excite, **kw):
        self.CSX, self.number, self.excite = CSX, port_nr, excite
        self.start, self.stop = np.array(start, float), np.array(stop, float)
        self.Z_ref = None
        self.U_filenames = kw.get('U_filenames', [])
        self.I_filenames = kw.get('I_filenames', [])
        self.port_props = []
        self.priority = kw.get('priority', 0)
        self.prefix = kw.get('PortNamePrefix', '')
        self.delay = kw.get('delay', 0)
        self.measplane_shift = 0
        self.lbl_temp = self.prefix + 'port_{}' + '_{}'.format(self.number)

    def _AddProbe(self, CSX, name, **kw):
        for prop in CSX.GetPropertiesByName(name):
            if isinstance(prop, C.ProbeBox) and not isinstance(prop, C.DumpBox):
                raise ValueError(f'port {self.number}: a probe named "{name}" already exists, port numbers must be '
                                 'unique (or use a different PortNamePrefix)')
        return CSX.AddProbe(name, **kw)

    def SetEnabled(self, val):
        """Enable or disable this port's excitation (a port made with excite=0 has none to enable)."""
        found = False
        for prop in self.port_props:
            if isinstance(prop, C.Excitation):
                prop.SetEnabled(val)
                found = True
        if not found and val:
            raise Exception('Unable to enable port! No excitation found!')

    def ReadUIData(self, sim_path, freq, signal_type='pulse'):
        self.u_data = UI_data(self.U_filenames, sim_path, freq, signal_type)
        self.uf_tot = sum(self.u_data.ui_f_val)
        self.ut_tot = sum(self.u_data.ui_val)
        self.u_time = self.u_data.ui_time[0]
        self.i_data = UI_data(self.I_filenames, sim_path, freq, signal_type)
        self.if_tot = sum(self.i_data.ui_f_val)
        self.it_tot = sum(self.i_data.ui_val)
        self.i_time = self.i_data.ui_time[0]
        self.u_mode_purity, self.i_mode_purity = self.u_data.ui_mode_purity, self.i_data.ui_mode_purity

    def CalcPort(self, sim_path, freq, ref_impedance=None, ref_plane_shift=None, signal_type='pulse'):
        """Incident and reflected waves (uf_inc, uf_ref, if_*, ut_* / it_* for a scalar reference) and the powers
        P_inc, P_ref, P_acc, as openEMS's Port.CalcPort."""
        self.ReadUIData(sim_path, freq, signal_type)
        if ref_impedance is not None:
            self.Z_ref = ref_impedance
        if self.Z_ref is None:
            raise Exception('Port Z_ref should not be None!')
        if ref_plane_shift is not None:
            if not hasattr(self, 'beta'):
                raise Exception('Port has no beta attribute!')
            shift = ref_plane_shift
            if self.measplane_shift:
                shift -= self.measplane_shift
            shift *= self.CSX.GetGrid().GetDeltaUnit()
            phase = np.real(self.beta) * shift
            uf_tot = self.uf_tot * np.cos(-phase) + 1j * self.if_tot * self.Z_ref * np.sin(-phase)
            if_tot = self.if_tot * np.cos(-phase) + 1j * self.uf_tot / self.Z_ref * np.sin(-phase)
            self.uf_tot, self.if_tot = uf_tot, if_tot
        self.uf_inc = 0.5 * (self.uf_tot + self.if_tot * self.Z_ref)
        self.if_inc = 0.5 * (self.if_tot + self.uf_tot / self.Z_ref)
        self.uf_ref = self.uf_tot - self.uf_inc
        self.if_ref = self.if_inc - self.if_tot
        if type(self.Z_ref) in (int, float):
            self.ut_inc = 0.5 * (self.ut_tot + self.it_tot * self.Z_ref)
            self.it_inc = 0.5 * (self.it_tot + self.ut_tot / self.Z_ref)
            self.ut_ref = self.ut_tot - self.ut_inc
            self.it_ref = self.it_inc - self.it_tot
        self.P_inc = 0.5 * np.real(self.uf_inc * np.conj(self.if_inc))
        self.P_ref = 0.5 * np.real(self.uf_ref * np.conj(self.if_ref))
        self.P_acc = 0.5 * np.real(self.uf_tot * np.conj(self.if_tot))


def _three_lines(mesh, prop_ny, pos, direction):
    """The three propagation lines round the measurement plane at pos (reversed for a port pointing down)."""
    lines = mesh.GetLines(prop_ny)
    idx = int(np.argmin(np.abs(lines - pos)))
    idx = max(1, min(idx, len(lines) - 2))
    ml = lines[idx - 1:idx + 2]
    return ml[::-1] if direction < 0 else ml


class _TLPort(Port):
    """The transmission-line ports' post-processing: Z and beta from the probes at three planes (openEMS's)."""
    def _tl(self, uf_A, uf_B, uf_C, ut_B):
        self.uf_tot, self.ut_tot = uf_B, ut_B
        self.i_data = UI_data(self.I_filenames, self._path, self._freq, self._signal)
        self.if_tot = 0.5 * (self.i_data.ui_f_val[0] + self.i_data.ui_f_val[1])
        self.it_tot = 0.5 * (self.i_data.ui_val[0] + self.i_data.ui_val[1])
        self.i_time = self.i_data.ui_time[0]
        unit = self.CSX.GetGrid().GetDeltaUnit()
        Et = uf_B
        dEt = (uf_C - uf_A) / (np.sum(np.abs(self.U_delta)) * unit)
        Ht = self.if_tot
        dHt = (self.i_data.ui_f_val[1] - self.i_data.ui_f_val[0]) / (np.abs(self.I_delta[0]) * unit)
        beta = np.sqrt(-dEt * dHt / (Ht * Et))
        beta[np.real(beta) < 0] *= -1
        self.beta = beta
        self.Z_ref = np.sqrt(Et * dEt / (Ht * dHt))

    def ReadUIData(self, sim_path, freq, signal_type='pulse'):
        self._path, self._freq, self._signal = sim_path, freq, signal_type
        self.u_data = u = UI_data(self.U_filenames, sim_path, freq, signal_type)
        self.u_time = u.ui_time[0]
        k = len(u.fns) // 3                                         # probes per plane: 1 (MSL, coax) or 2
        f = [sum(u.ui_f_val[n * k:(n + 1) * k]) for n in range(3)]
        self._tl(f[0], f[1], f[2], sum(u.ui_val[k:2 * k]))


class MSLPort(_TLPort):
    """The microstrip port: the strip from start to stop (start's exc_dir coordinate is the strip, stop's the
    ground), propagating along prop_dir. FeedShift, MeasPlaneShift, Feed_R as openEMS's."""
    def __init__(self, CSX, port_nr, metal_prop, start, stop, prop_dir, exc_dir, excite=0, **kw):
        super().__init__(CSX, port_nr=port_nr, start=start, stop=stop, excite=excite, **kw)
        self.exc_ny, self.prop_ny = C.ny_dir(exc_dir), C.ny_dir(prop_dir)
        self.direction = np.sign(self.stop[self.prop_ny] - self.start[self.prop_ny])
        self.upside_down = np.sign(self.stop[self.exc_ny] - self.start[self.exc_ny])
        if not (self.start != self.stop).all():
            raise Exception('Start coordinate must not be equal to stop coordinate')
        if self.exc_ny == self.prop_ny:
            raise Exception('Excitation direction must not be equal to propagation direction')
        self.feed_shift = kw.get('FeedShift', 0)
        self.measplane_shift = kw.get('MeasPlaneShift', 0.5 * abs(self.start[self.prop_ny] - self.stop[self.prop_ny]))
        self.measplane_pos = self.start[self.prop_ny] + self.measplane_shift * self.direction
        self.feed_R = kw.get('Feed_R', np.inf)
        ms0, ms1 = self.start.copy(), self.stop.copy()
        ms1[self.exc_ny] = ms0[self.exc_ny]
        metal_prop.AddBox(ms0, ms1, priority=self.priority)
        lines = CSX.GetGrid().GetLines(self.prop_ny)
        if not len(lines) > 5:
            raise Exception('At least 5 lines in propagation direction required!')
        mi = int(np.argmin(np.abs(lines - self.measplane_pos)))
        mi = min(max(mi, 1), len(lines) - 2)
        self.measplane_shift = abs(self.start[self.prop_ny] - lines[mi])
        pi = np.array([mi - 1, mi, mi + 1])
        if self.direction < 0:
            pi = pi[::-1]
        upos = lines[pi]
        self.U_filenames, self.U_delta = [], np.diff(upos)
        for n, sfx in enumerate('ABC'):
            u0, u1 = 0.5 * (self.start + self.stop), 0.5 * (self.start + self.stop)
            u0[self.prop_ny] = u1[self.prop_ny] = upos[n]
            u0[self.exc_ny], u1[self.exc_ny] = self.start[self.exc_ny], self.stop[self.exc_ny]
            name = self.lbl_temp.format('ut') + sfx
            self.U_filenames.append(name)
            pr = self._AddProbe(CSX, name, p_type=0)
            pr.AddBox(u0, u1)
            self.port_props.append(pr)
        ipos = upos[0:2] + np.diff(upos) / 2.0
        self.I_filenames, self.I_delta = [], np.diff(ipos)
        i0, i1 = self.start.copy(), self.stop.copy()
        i1[self.exc_ny] = self.start[self.exc_ny]
        for n, sfx in enumerate('AB'):
            i0[self.prop_ny] = i1[self.prop_ny] = ipos[n]
            name = self.lbl_temp.format('it') + sfx
            self.I_filenames.append(name)
            pr = self._AddProbe(CSX, name, p_type=1, weight=self.direction, norm_dir=self.prop_ny)
            pr.AddBox(i0, i1)
            self.port_props.append(pr)
        if excite != 0:
            ei = int(np.argmin(np.abs(lines - (self.start[self.prop_ny] + self.feed_shift * self.direction))))
            e0, e1 = self.start.copy(), self.stop.copy()
            e0[self.prop_ny] = e1[self.prop_ny] = lines[ei]
            vec = np.zeros(3)
            vec[self.exc_ny] = -1 * self.upside_down * excite
            exc = CSX.AddExcitation(self.lbl_temp.format('excite'), exc_type=0, exc_val=vec, delay=self.delay)
            exc.AddBox(e0, e1, priority=self.priority)
            self.port_props.append(exc)
        if self.feed_R >= 0 and not np.isinf(self.feed_R):
            r0, r1 = self.start.copy(), self.stop.copy()
            r1[self.prop_ny] = r0[self.prop_ny]
            if self.feed_R == 0:
                metal_prop.AddBox(r0, r1)
            else:
                el = CSX.AddLumpedElement(self.lbl_temp.format('resist'), ny=self.exc_ny, caps=True, R=self.feed_R)
                el.AddBox(r0, r1)
                self.port_props.append(el)

    def ReadUIData(self, sim_path, freq, signal_type='pulse'):
        """The voltage of the middle probe, the current between the planes; Z and beta from the differences."""
        self._path, self._freq, self._signal = sim_path, freq, signal_type
        self.u_data = u = UI_data(self.U_filenames, sim_path, freq, signal_type)
        self.u_time = u.ui_time[0]
        self._tl(u.ui_f_val[0], u.ui_f_val[1], u.ui_f_val[2], u.ui_val[1])


class WaveguidePort(Port):
    """Any waveguide mode: E_WG_func / H_WG_func (three expressions each) or E_WG_file / H_WG_file (mode files),
    kc the mode's cutoff wavenumber (1/m), exc_dir the propagation direction; excite_type 0 (E) or 2 (H);
    local_origin 'corner', 'center', a point or None."""
    def __init__(self, CSX, port_nr, start, stop, exc_dir, E_WG_func, H_WG_func, kc, excite=0, excite_type=0,
                 E_WG_file=None, H_WG_file=None, local_origin=None, **kw):
        super().__init__(CSX, port_nr=port_nr, start=start, stop=stop, excite=excite, **kw)
        self.exc_ny = C.ny_dir(exc_dir)
        self.ny_P, self.ny_PP = (self.exc_ny + 1) % 3, (self.exc_ny + 2) % 3
        self.direction = np.sign(stop[self.exc_ny] - start[self.exc_ny])
        self.ref_index = 1
        if excite != 0 and stop[self.exc_ny] == start[self.exc_ny]:
            raise Exception('Port length in excitation direction may not be zero if port is excited!')
        self.kc, self.E_func, self.H_func, self.E_file, self.H_file = kc, E_WG_func, H_WG_func, E_WG_file, H_WG_file
        if isinstance(local_origin, str):
            if local_origin == 'corner':
                local_origin = np.minimum(start, stop)
            elif local_origin == 'center':
                local_origin = 0.5 * (np.array(start) + np.array(stop))
            else:
                raise Exception(f"unknown local_origin value '{local_origin}'; use 'corner', 'center' or [x,y,z]")
        elif local_origin is not None:
            local_origin = np.array(local_origin, float)
        use_func = self.E_func is not None and self.H_func is not None
        if excite != 0:
            e0, e1 = np.array(start, float), np.array(stop, float)
            e1[self.exc_ny] = e0[self.exc_ny]
            vec = excite * np.ones(3)
            vec[self.exc_ny] = 0
            exc = CSX.AddExcitation(self.lbl_temp.format('excite'), exc_type=excite_type, exc_val=vec,
                                    delay=self.delay)
            if excite_type not in (0, 1, 2, 3):
                raise Exception('Unsupported excitation type. Only 0 or 2 for WaveguidePort')
            if not use_func:
                if not (isinstance(self.E_file, str) and isinstance(self.H_file, str)):
                    raise Exception('Both E_WG_file and H_WG_file must be strings (HDF5 file paths)')
                exc.SetWeightFile(self.E_file if excite_type in (0, 1) else self.H_file)
                d = [0, 0, 0]
                d[self.exc_ny] = 1
                exc.SetPropagationDir(d)
            else:
                if not isinstance(self.E_func, list):
                    raise Exception('Unsupported input type for "E_WG_func" or "H_WG_func". Expected a list of strings')
                exc.SetWeightFunction([str(x) for x in (self.E_func if excite_type in (0, 1) else self.H_func)])
            if local_origin is not None:
                exc.SetWeightOrigin(local_origin)
            exc.AddBox(e0, e1, priority=self.priority)
            self.port_props.append(exc)
        m0, m1 = np.array(start, float), np.array(stop, float)
        m0[self.exc_ny] = m1[self.exc_ny]
        self.measplane_shift = abs(stop[self.exc_ny] - start[self.exc_ny])
        for kind, ptype, func, fn, w in (('ut', 10, self.E_func, self.E_file, 1), ('it', 11, self.H_func, self.H_file,
                                                                                 self.direction)):
            name = self.lbl_temp.format(kind)
            kwp = {'mode_function': [str(x) for x in func]} if use_func else {}
            pr = self._AddProbe(CSX, name, p_type=ptype, weight=w, **kwp)
            if not use_func:
                pr.SetModeFile(fn)
            if local_origin is not None:
                pr.SetModeOrigin(local_origin)
            pr.AddBox(m0, m1)
            self.port_props.append(pr)
            (self.U_filenames if kind == 'ut' else self.I_filenames).append(name)

    def CalcPort(self, sim_path, freq, ref_impedance=None, ref_plane_shift=None, signal_type='pulse', ZL=-1):
        """beta and the wave impedance ZL from the cutoff (or ZL given), the S-parameters on ZL unless
        ref_impedance is given."""
        k = 2.0 * np.pi * np.asarray(freq) / C0 * self.ref_index
        self.beta = np.sqrt(k ** 2 - self.kc ** 2 + 0j)
        self.ZL = k * Z0 / self.beta if ZL <= 0 else ZL
        if ref_impedance is None:
            self.Z_ref = self.ZL
        super().CalcPort(sim_path, freq, ref_impedance, ref_plane_shift, signal_type)


class RectWGPort(WaveguidePort):
    """A rectangular waveguide's TE_mn mode (Pozar): a, b in metres along the first and second transverse axes,
    mode_name like 'TE10'."""
    def __init__(self, CSX, port_nr, start, stop, exc_dir, a, b, mode_name, excite=0, local_origin='corner', **kw):
        exc_ny = C.ny_dir(exc_dir)
        ny_P, ny_PP = (exc_ny + 1) % 3, (exc_ny + 2) % 3
        self.WG_size, self.WG_mode = [a, b], mode_name
        if len(mode_name) != 4:
            raise Exception('Invalid mode definition')
        if not mode_name.startswith('TE'):
            raise Exception(f'Currently only TE-modes are supported! Mode found: {mode_name}')
        self.TE, self.TM = True, False
        unit = CSX.GetGrid().GetDeltaUnit()
        M, N = float(mode_name[2]), float(mode_name[3])
        self.M, self.N = M, N
        kc = np.sqrt((M * np.pi / a) ** 2 + (N * np.pi / b) ** 2)
        a, b = a / unit, b / unit
        nP, nPP = 'xyz'[ny_P], 'xyz'[ny_PP]
        E, H = [0, 0, 0], [0, 0, 0]
        if N > 0:
            E[ny_P] = f'{N / b}*cos({M * np.pi / a}*{nP})*sin({N * np.pi / b}*{nPP})'
            H[ny_PP] = f'{N / b}*cos({M * np.pi / a}*{nP})*sin({N * np.pi / b}*{nPP})'
        if M > 0:
            E[ny_PP] = f'{-1 * M / a}*sin({M * np.pi / a}*{nP})*cos({N * np.pi / b}*{nPP})'
            H[ny_P] = f'{M / a}*sin({M * np.pi / a}*{nP})*cos({N * np.pi / b}*{nPP})'
        super().__init__(CSX, port_nr=port_nr, start=start, stop=stop, exc_dir=exc_dir, E_WG_func=E, H_WG_func=H,
                         kc=kc, excite=excite, local_origin=local_origin, **kw)


class CircWGPort(WaveguidePort):
    """A circular waveguide's TE_nm mode (Pozar), radius in metres, pol_ang the polarisation (rad); Cartesian
    meshes only (the profile masked to rho < radius)."""
    _pnm = {(0, 1): 3.832, (1, 1): 1.841, (2, 1): 3.054, (0, 2): 7.016, (1, 2): 5.331, (2, 2): 6.706,
            (0, 3): 10.174, (1, 3): 8.536, (2, 3): 9.970}

    def __init__(self, CSX, port_nr, start, stop, exc_dir, radius, mode_name, pol_ang=0, excite=0,
                 local_origin='center', **kw):
        if not mode_name.upper().startswith('TE'):
            raise Exception('CircWGPort: only TE modes are supported')
        n, m = int(mode_name[2]), int(mode_name[3])
        pnm = self._pnm.get((n, m))
        if pnm is None:
            raise Exception(f'CircWGPort: unsupported TE_nm mode: {mode_name}')
        unit = CSX.GetGrid().GetDeltaUnit()
        kc = pnm / radius
        kd = kc * unit
        exc_ny = C.ny_dir(exc_dir)
        ny_P, ny_PP = (exc_ny + 1) % 3, (exc_ny + 2) % 3
        u, v = 'xyz'[ny_P], 'xyz'[ny_PP]
        rho, ang = f'sqrt({u}*{u}+{v}*{v})', f'(atan2({v},{u}))-{pol_ang:.15g}'
        c_r, c_a = -1.0 / kd ** 2, 1.0 / kd
        Er = f'{c_r:.15g}/({rho})*cos({ang})*j1({kd:.15g}*({rho}))'
        Ea = f'{c_a:.15g}*sin({ang})*0.5*(j0({kd:.15g}*({rho}))-jn(2,{kd:.15g}*({rho})))'
        Hr = f'{-c_a:.15g}*sin({ang})*0.5*(j0({kd:.15g}*({rho}))-jn(2,{kd:.15g}*({rho})))'
        Ha = f'{c_r:.15g}/({rho})*cos({ang})*j1({kd:.15g}*({rho}))'
        mask = f'(({rho})<{radius / unit:.15g})'
        ca, sa = f'{u}/({rho})', f'{v}/({rho})'
        E, H = ['0', '0', '0'], ['0', '0', '0']
        E[ny_P] = f'(({Er})*({ca})-({Ea})*({sa}))*{mask}'
        E[ny_PP] = f'(({Er})*({sa})+({Ea})*({ca}))*{mask}'
        H[ny_P] = f'(({Hr})*({ca})-({Ha})*({sa}))*{mask}'
        H[ny_PP] = f'(({Hr})*({sa})+({Ha})*({ca}))*{mask}'
        super().__init__(CSX, port_nr=port_nr, start=start, stop=stop, exc_dir=exc_dir, E_WG_func=E, H_WG_func=H,
                         kc=kc, excite=excite, local_origin=local_origin, **kw)


class CoaxialPort(_TLPort):
    """A coaxial line along prop_dir from start to stop (its axis): inner radius r_i, outer conductor r_o .. r_os
    (drawing units), pec_prop for the conductors, mat_prop (or None) for the filling. FeedShift, MeasPlaneShift,
    Feed_R (0 or inf)."""
    def __init__(self, CSX, port_nr, pec_prop, mat_prop, start, stop, prop_dir, r_i, r_o, r_os, excite=0, **kw):
        super().__init__(CSX, port_nr=port_nr, start=start, stop=stop, excite=excite, **kw)
        start, stop = self.start, self.stop
        self.prop_ny = C.ny_dir(prop_dir)
        self.ny_P, self.ny_PP = (self.prop_ny + 1) % 3, (self.prop_ny + 2) % 3
        self.direction = np.sign(stop[self.prop_ny] - start[self.prop_ny])
        if self.direction == 0:
            raise Exception('CoaxialPort: start and stop must differ in propagation direction')
        self.r_i, self.r_o = r_i, r_o
        feed_shift, feed_R = kw.get('FeedShift', 0), kw.get('Feed_R', np.inf)
        mpos = 0.5 * (start[self.prop_ny] + stop[self.prop_ny])
        if 'MeasPlaneShift' in kw:
            mpos = start[self.prop_ny] + self.direction * kw['MeasPlaneShift']
        pec_prop.AddCylinder(start, stop, r_i, priority=self.priority)
        pec_prop.AddCylindricalShell(start, stop, 0.5 * (r_o + r_os), r_os - r_o, priority=self.priority)
        if mat_prop is not None:
            mat_prop.AddCylindricalShell(start, stop, 0.5 * (r_i + r_o), r_o - r_i, priority=self.priority - 1)
        self.port_props.append(pec_prop)
        mesh = CSX.GetGrid()
        ml = _three_lines(mesh, self.prop_ny, mpos, self.direction)
        self.measplane_shift = abs(ml[1] - start[self.prop_ny])
        self.U_delta = np.diff(ml)
        ipos = ml[:2] + np.diff(ml) / 2.0
        self.I_delta = np.diff(ipos)
        self.U_filenames = []
        for n, sfx in enumerate('ABC'):
            v0 = start.copy()
            v0[self.prop_ny] = ml[n]
            v0[self.ny_P] = start[self.ny_P] + r_i
            v1 = v0.copy()
            v1[self.ny_P] = start[self.ny_P] + r_o
            name = self.lbl_temp.format('ut') + sfx
            self.U_filenames.append(name)
            pr = self._AddProbe(CSX, name, p_type=0, weight=1)
            pr.AddBox(v0, v1)
            self.port_props.append(pr)
        margin = 0.1 * (r_o - r_i)
        self.I_filenames = []
        for n, sfx in enumerate('AB'):
            i0 = np.zeros(3)
            i0[self.prop_ny] = 0.5 * (ml[n] + ml[n + 1])
            i0[self.ny_P] = start[self.ny_P] - r_i - margin
            i0[self.ny_PP] = start[self.ny_PP] - r_i - margin
            i1 = i0.copy()
            i1[self.ny_P] = start[self.ny_P] + r_i + margin
            i1[self.ny_PP] = start[self.ny_PP] + r_i + margin
            name = self.lbl_temp.format('it') + sfx
            self.I_filenames.append(name)
            pr = self._AddProbe(CSX, name, p_type=1, weight=self.direction, norm_dir=self.prop_ny)
            pr.AddBox(i0, i1)
            self.port_props.append(pr)
        if excite != 0:
            lines = mesh.GetLines(self.prop_ny)
            fi = int(np.argmin(np.abs(lines - (start[self.prop_ny] + feed_shift * self.direction))))
            mc = np.min(np.diff(lines))
            x0 = start.copy()
            x0[self.prop_ny] = lines[fi] - 0.01 * mc
            x1 = x0.copy()
            x1[self.prop_ny] = lines[fi] + 0.01 * mc
            cx, cy = start[self.ny_P], start[self.ny_PP]
            dX, dY = f"({'xyz'[self.ny_P]}-{cx:.15g})", f"({'xyz'[self.ny_PP]}-{cy:.15g})"
            r2 = f'({dX}*{dX}+{dY}*{dY})'
            mask = f'*(sqrt({r2})<{r_o:.15g})*(sqrt({r2})>{r_i:.15g})'
            fE = ['0', '0', '0']
            fE[self.ny_P], fE[self.ny_PP] = f'{dX}/{r2}{mask}', f'{dY}/{r2}{mask}'
            vec = excite * np.ones(3)
            vec[self.prop_ny] = 0
            exc = CSX.AddExcitation(self.lbl_temp.format('excite'), exc_type=0, exc_val=vec, delay=self.delay)
            exc.SetWeightFunction(fE)
            exc.AddCylindricalShell(x0, x1, 0.5 * (r_i + r_o), r_o - r_i, priority=0)
            self.port_props.append(exc)
        t0, t1 = start.copy(), stop.copy()
        t1[self.prop_ny] = t0[self.prop_ny]
        if feed_R == 0:
            pec_prop.AddBox(t0, t1, priority=self.priority)
            pec_prop.AddCylindricalShell(t0, t1, 0.5 * (r_i + r_o), r_o - r_i, priority=self.priority)
        elif np.isinf(feed_R):
            pass
        elif feed_R > 0:
            raise NotImplementedError('CoaxialPort: finite Feed_R > 0 is not yet supported; use Feed_R=0 or Feed_R=inf')
        else:
            raise Exception('CoaxialPort: Feed_R <= 0 is not allowed')


class StripLinePort(_TLPort):
    """A stripline: the strip from start to stop (flat in the exc_dir axis), height to each ground plane, voltage
    probes up and down from the strip. FeedShift, MeasPlaneShift, Feed_R."""
    def __init__(self, CSX, port_nr, metal_prop, start, stop, prop_dir, exc_dir, height, excite=0, **kw):
        super().__init__(CSX, port_nr=port_nr, start=start, stop=stop, excite=excite, **kw)
        start, stop = self.start, self.stop
        self.prop_ny = C.ny_dir(prop_dir)
        self.height_ny = C.ny_dir(exc_dir)
        self.width_ny = 3 - self.prop_ny - self.height_ny
        if start[self.height_ny] != stop[self.height_ny]:
            raise Exception('StripLinePort: start/stop in height direction must be equal')
        self.direction = np.sign(stop[self.prop_ny] - start[self.prop_ny])
        if self.direction == 0:
            raise Exception('StripLinePort: start/stop in prop direction must differ')
        feed_shift, feed_R = kw.get('FeedShift', 0), kw.get('Feed_R', np.inf)
        ns, ne = np.minimum(start, stop), np.maximum(start, stop)
        metal_prop.AddBox(start, stop, priority=self.priority)
        self.port_props.append(metal_prop)
        mpos = 0.5 * (ns[self.prop_ny] + ne[self.prop_ny])
        if 'MeasPlaneShift' in kw:
            mpos = start[self.prop_ny] + self.direction * kw['MeasPlaneShift']
        mesh = CSX.GetGrid()
        ml = _three_lines(mesh, self.prop_ny, mpos, self.direction)
        self.measplane_shift = abs(ml[1] - start[self.prop_ny])
        self.U_delta = np.diff(ml)
        ipos = ml[:2] + np.diff(ml) / 2.0
        self.I_delta = np.diff(ipos)
        wl = mesh.GetLines(self.width_ny)
        wc = wl[int(np.argmin(np.abs(wl - 0.5 * (ns[self.width_ny] + ne[self.width_ny]))))]
        hv = np.zeros(3)
        hv[self.height_ny] = height
        self.U_filenames = []
        for n, pair in enumerate((('A1', 'A2'), ('B1', 'B2'), ('C1', 'C2'))):
            pt = np.zeros(3)
            pt[self.prop_ny], pt[self.width_ny], pt[self.height_ny] = ml[n], wc, start[self.height_ny]
            for sfx, sgn in zip(pair, (1, -1)):
                name = self.lbl_temp.format('ut') + sfx
                self.U_filenames.append(name)
                pr = self._AddProbe(CSX, name, p_type=0, weight=0.5)
                pr.AddBox(pt, pt + sgn * hv, priority=self.priority)
                self.port_props.append(pr)
        self._current_probes(mesh, start, ns, ne, ml)
        if excite != 0:
            lines = mesh.GetLines(self.prop_ny)
            fi = int(np.argmin(np.abs(lines - (start[self.prop_ny] + feed_shift * self.direction))))
            x0 = np.zeros(3)
            x0[self.prop_ny], x0[self.width_ny], x0[self.height_ny] = lines[fi], ns[self.width_ny], ns[self.height_ny]
            x1 = x0.copy()
            x1[self.width_ny], x1[self.height_ny] = ne[self.width_ny], ne[self.height_ny]
            ev = np.zeros(3)
            ev[self.height_ny] = excite
            for lbl, sgn in (('excite_1', 1), ('excite_2', -1)):
                exc = CSX.AddExcitation(self.lbl_temp.format(lbl), exc_type=0, exc_val=sgn * ev, delay=self.delay)
                exc.AddBox(x0, x1 + sgn * hv, priority=self.priority)
                self.port_props.append(exc)
        r0, r1 = start.copy(), stop.copy()
        r1[self.prop_ny] = r0[self.prop_ny]
        if feed_R > 0 and not np.isinf(feed_R):
            el = CSX.AddLumpedElement(self.lbl_temp.format('resist'), ny=self.height_ny, R=2 * feed_R)
            el.AddBox(r0, r1 + hv, priority=self.priority)
            el.AddBox(r0, r1 - hv, priority=self.priority)
            self.port_props.append(el)
        elif feed_R == 0:
            metal_prop.AddBox(r0, r1 + hv, priority=self.priority)
            metal_prop.AddBox(r0, r1 - hv, priority=self.priority)
        elif feed_R < 0:
            raise Exception('StripLinePort: Feed_R must be >= 0')

    def _current_probes(self, mesh, start, ns, ne, ml):
        """Two current loops round the strip (its width plus half a cell, two cells either side in height)."""
        hl, wl = mesh.GetLines(self.height_ny), mesh.GetLines(self.width_ny)
        hi = int(np.argmin(np.abs(hl - start[self.height_ny])))
        hi = max(2, min(hi, len(hl) - 3))
        w0 = int(np.argmin(np.abs(wl - ns[self.width_ny])))
        w0 = max(1, min(w0, len(wl) - 1))
        w1 = int(np.argmin(np.abs(wl - ne[self.width_ny])))
        w1 = max(0, min(w1, len(wl) - 2))
        b0, b1 = np.zeros(3), np.zeros(3)
        b0[self.width_ny], b0[self.height_ny] = 0.5 * (wl[w0 - 1] + wl[w0]), 0.5 * (hl[hi - 2] + hl[hi - 1])
        b1[self.width_ny], b1[self.height_ny] = 0.5 * (wl[w1] + wl[w1 + 1]), 0.5 * (hl[hi + 1] + hl[hi + 2])
        self.I_filenames = []
        for n, sfx in enumerate('AB'):
            i0, i1 = b0.copy(), b1.copy()
            i0[self.prop_ny] = i1[self.prop_ny] = 0.5 * (ml[n] + ml[n + 1])
            name = self.lbl_temp.format('it') + sfx
            self.I_filenames.append(name)
            pr = self._AddProbe(self.CSX, name, p_type=1, weight=self.direction, norm_dir=self.prop_ny)
            pr.AddBox(i0, i1)
            self.port_props.append(pr)


class CPWPort(StripLinePort):
    """A coplanar waveguide: the centre strip from start to stop (flat in the plane's normal), exc_dir across the
    gaps, gap_width each; voltage probes across both gaps. FeedShift, MeasPlaneShift, Feed_R."""
    def __init__(self, CSX, port_nr, metal_prop, start, stop, prop_dir, exc_dir, gap_width, excite=0, **kw):
        Port.__init__(self, CSX, port_nr=port_nr, start=start, stop=stop, excite=excite, **kw)
        start, stop = self.start, self.stop
        self.prop_ny = C.ny_dir(prop_dir)
        self.width_ny = C.ny_dir(exc_dir)
        if self.width_ny == self.prop_ny:
            raise Exception('CPWPort: exc_dir must differ from prop_dir')
        self.height_ny = 3 - self.prop_ny - self.width_ny
        if start[self.height_ny] != stop[self.height_ny]:
            raise Exception('CPWPort: start/stop in height direction must be equal')
        self.direction = np.sign(stop[self.prop_ny] - start[self.prop_ny])
        if self.direction == 0:
            raise Exception('CPWPort: start/stop in prop direction must differ')
        feed_shift, feed_R = kw.get('FeedShift', 0), kw.get('Feed_R', np.inf)
        ns, ne = np.minimum(start, stop), np.maximum(start, stop)
        metal_prop.AddBox(start, stop, priority=self.priority)
        self.port_props.append(metal_prop)
        mpos = 0.5 * (ns[self.prop_ny] + ne[self.prop_ny])
        if 'MeasPlaneShift' in kw:
            mpos = start[self.prop_ny] + self.direction * kw['MeasPlaneShift']
        mesh = CSX.GetGrid()
        ml = _three_lines(mesh, self.prop_ny, mpos, self.direction)
        self.measplane_shift = abs(ml[1] - start[self.prop_ny])
        self.U_delta = np.diff(ml)
        ipos = ml[:2] + np.diff(ml) / 2.0
        self.I_delta = np.diff(ipos)
        wc = 0.5 * (ns[self.width_ny] + ne[self.width_ny])
        hw = 0.5 * (ne[self.width_ny] - ns[self.width_ny])
        g0, g1 = np.zeros(3), np.zeros(3)
        g0[self.width_ny], g1[self.width_ny] = hw, hw + gap_width
        self.U_filenames = []
        for n, pair in enumerate((('A1', 'A2'), ('B1', 'B2'), ('C1', 'C2'))):
            pt = np.zeros(3)
            pt[self.prop_ny], pt[self.width_ny], pt[self.height_ny] = ml[n], wc, start[self.height_ny]
            for sfx, sgn in zip(pair, (-1, 1)):
                name = self.lbl_temp.format('ut') + sfx
                self.U_filenames.append(name)
                pr = self._AddProbe(CSX, name, p_type=0, weight=0.5)
                pr.AddBox(pt + sgn * g0, pt + sgn * g1, priority=self.priority)
                self.port_props.append(pr)
        self._current_probes(mesh, start, ns, ne, ml)
        if excite != 0:
            lines = mesh.GetLines(self.prop_ny)
            fi = int(np.argmin(np.abs(lines - (start[self.prop_ny] + feed_shift * self.direction))))
            pt = np.zeros(3)
            pt[self.prop_ny], pt[self.width_ny], pt[self.height_ny] = lines[fi], wc, start[self.height_ny]
            for lbl, sgn in (('excite_1', -1), ('excite_2', 1)):
                ev = np.zeros(3)
                ev[self.width_ny] = sgn * excite
                exc = CSX.AddExcitation(self.lbl_temp.format(lbl), exc_type=0, exc_val=ev, delay=self.delay)
                exc.AddBox(pt + sgn * g0, pt + sgn * g1, priority=self.priority)
                self.port_props.append(exc)
        rp = np.zeros(3)
        rp[self.prop_ny], rp[self.width_ny], rp[self.height_ny] = start[self.prop_ny], wc, start[self.height_ny]
        if feed_R > 0 and not np.isinf(feed_R):
            el = CSX.AddLumpedElement(self.lbl_temp.format('resist'), ny=self.width_ny, R=2 * feed_R)
            for sgn in (-1, 1):
                el.AddBox(rp + sgn * g0, rp + sgn * g1, priority=self.priority)
            self.port_props.append(el)
        elif feed_R == 0:
            for sgn in (-1, 1):
                metal_prop.AddBox(rp + sgn * g0, rp + sgn * g1, priority=self.priority)
        elif feed_R < 0:
            raise Exception('CPWPort: Feed_R must be >= 0')


class CurvePort(Port):
    """A lumped port on one mesh edge: R across the edge nearest the middle of start -> stop, joined to start and
    stop by PEC curves when they are further apart than one cell."""
    def __init__(self, CSX, port_nr, R, start, stop, excite=0, **kw):
        super().__init__(CSX, port_nr=port_nr, start=start, stop=stop, excite=excite, **kw)
        self.R = self.Z_ref = R
        mesh = CSX.GetGrid()
        lines = [mesh.GetLines(n) for n in range(3)]
        self.port_dir = d0 = int(np.argmax(np.abs(self.stop - self.start)))
        d1, d2 = (d0 + 1) % 3, (d0 + 2) % 3
        ns, ne = (self.start, self.stop) if self.start[d0] <= self.stop[d0] else (self.stop, self.start)
        snap = lambda v, ln: int(np.argmin(np.abs(ln - v)))
        si = [snap(ns[d], lines[d]) for d in range(3)]
        ei = [snap(ne[d], lines[d]) for d in range(3)]
        if abs(si[d0] - ei[d0]) != 1:
            ps = [0, 0, 0]
            ps[d0] = snap(0.5 * (ns[d0] + ne[d0]), lines[d0])
            ps[d1] = snap(0.5 * (ns[d1] + ne[d1]), lines[d1])
            ps[d2] = snap(0.5 * (ns[d2] + ne[d2]), lines[d2])
            pe = list(ps)
            pe[d0] = ps[d0] + 1
            es = np.array([lines[d][ps[d]] for d in range(3)])
            ee = np.array([lines[d][pe[d]] for d in range(3)])
            metal = CSX.AddMetal(self.lbl_temp.format('PEC'))
            self.port_props.append(metal)
            for up in (ns, ne):
                near = es if np.linalg.norm(up - es) <= np.linalg.norm(up - ee) else ee
                metal.AddCurve([[up[0], near[0]], [up[1], near[1]], [up[2], near[2]]], priority=self.priority)
        else:
            ps, pe = si, ei
            es = np.array([lines[d][ps[d]] for d in range(3)])
            ee = np.array([lines[d][pe[d]] for d in range(3)])
        dl = lambda d, k: lines[d][k] - lines[d][k - 1]
        i0, i1 = es.copy(), ee.copy()
        i0[d1] -= dl(d1, ps[d1]) / 2
        i1[d1] += dl(d1, ps[d1] + 1) / 2
        i0[d2] -= dl(d2, ps[d2]) / 2
        i1[d2] += dl(d2, ps[d2] + 1) / 2
        i0[d0] = i1[d0] = 0.5 * (es[d0] + ee[d0])
        self.U_filenames = [self.lbl_temp.format('ut')]
        pr = self._AddProbe(CSX, self.U_filenames[0], p_type=0, weight=-1)
        pr.AddBox(es, ee)
        self.port_props.append(pr)
        self.I_filenames = [self.lbl_temp.format('it')]
        pr = self._AddProbe(CSX, self.I_filenames[0], p_type=1, weight=1)
        pr.AddBox(i0, i1)
        self.port_props.append(pr)
        if R > 0 and not np.isinf(R):
            el = CSX.AddLumpedElement(self.lbl_temp.format('resist'), ny=d0, R=R)
            el.AddBox(es, ee, priority=self.priority)
            self.port_props.append(el)
        elif R == 0:
            sh = CSX.AddMetal(self.lbl_temp.format('short'))
            sh.AddBox(es, ee, priority=self.priority)
            self.port_props.append(sh)
        if excite:
            vec = excite * (np.array(pe) != np.array(ps)).astype(float)
            exc = CSX.AddExcitation(self.lbl_temp.format('excite'), exc_type=0, exc_val=vec, delay=self.delay)
            exc.AddBox(es, ee, priority=self.priority)
            self.port_props.append(exc)

    def CalcPort(self, sim_path, freq, ref_impedance=None, ref_plane_shift=None, signal_type='pulse'):
        if ref_impedance is None:
            self.Z_ref = self.R
        super().CalcPort(sim_path, freq, ref_impedance, None, signal_type)
