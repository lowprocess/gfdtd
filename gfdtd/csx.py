"""
CSXCAD-compatible geometry: the part of CSXCAD's Python API that openEMS scripts (and kicad2ems's build_fdtd) use,
and CSXCAD's XML format both ways, so a model built for openEMS can be built, written, read and run here.

    CSX = ContinuousStructure()
    grid = CSX.GetGrid(); grid.SetDeltaUnit(1e-3); grid.SetLines('x', [...])
    CSX.AddMaterial('FR4', epsilon=4.3, kappa=0.01).AddLinPoly([xs, ys], 'z', 0.0, 1.6, priority=1)
    CSX.AddMetal('cu').AddPolygon([xs, ys], 'z', 1.6, priority=10)
    CSX.Write2XML('model.xml');  CSX2 = ContinuousStructure(); CSX2.ReadFromXML('model.xml')

Properties: Material, DebyeMaterial, Metal, ConductingSheet, LumpedElement, Excitation (with weight functions or a
mode file), ProbeBox (with mode functions or a mode file), DumpBox (and LorentzMaterial, read with its static
epsilon). Primitives: Box, Polygon, LinPoly, Cylinder, CylindricalShell, Curve. Anything else in a file read here is
kept out of the model and listed in `CSX.skipped`.

Weight and mode functions are CSXCAD's function-parser expressions in x, y, z, rho, a, r, t (drawing units, from
the weight origin): evaluate(expr, coords) runs them on numpy arrays.
"""
import re
import xml.etree.ElementTree as ET

import numpy as np

AXES = 'xyz'


def ny_dir(d):
    """0 / 1 / 2 of 'x' / 'y' / 'z' (or an int, or '-z' and the like: the sign is dropped)."""
    if isinstance(d, str):
        return AXES.index(d.strip().lstrip('+-').lower())
    return int(d)


def _num(v):
    return float(np.asarray(v, float))


def _vec(v):
    return [float(x) for x in np.asarray(v, float).ravel()]


def _fmt(v):
    return f"{float(v):.6e}"


def _floats(text, n=None):
    """'4.3,1,1' -> [4.3, 1.0, 1.0]; a single value repeated to n."""
    vals = [float(x) for x in str(text).replace(';', ',').split(',') if x.strip()]
    if n and len(vals) == 1:
        vals *= n
    return vals


def _special(name):
    import scipy.special as sp
    return getattr(sp, name)


_FUNCS = dict(sin=np.sin, cos=np.cos, tan=np.tan, asin=np.arcsin, acos=np.arccos, atan=np.arctan, atan2=np.arctan2,
              sinh=np.sinh, cosh=np.cosh, tanh=np.tanh, asinh=np.arcsinh, acosh=np.arccosh, atanh=np.arctanh,
              exp=np.exp, log=np.log, log10=np.log10, log2=np.log2, sqrt=np.sqrt, cbrt=np.cbrt, abs=np.abs,
              floor=np.floor, ceil=np.ceil, trunc=np.trunc, int=np.rint, min=np.minimum, max=np.maximum,
              pow=np.power, hypot=np.hypot, sec=lambda v: 1 / np.cos(v), csc=lambda v: 1 / np.sin(v),
              cot=lambda v: 1 / np.tan(v), pi=np.pi,
              j0=lambda v: _special('j0')(v), j1=lambda v: _special('j1')(v), jn=lambda n, v: _special('jv')(n, v),
              y0=lambda v: _special('y0')(v), y1=lambda v: _special('y1')(v), yn=lambda n, v: _special('yv')(n, v),
              **{'if': None})


def _fparser_to_python(expr):
    """A CSXCAD (fparser) expression as a numpy one: ^ is the power, = equality, & | logical and / or, if(c, a, b)."""
    e = str(expr).strip()
    e = e.replace('^', '**')
    e = re.sub(r'(?<![<>!=])=(?!=)', '==', e)
    e = e.replace('&', ' * ').replace('|', ' + ')                    # on 0 / 1 values; | then clipped below
    e = re.sub(r'\bif\s*\(', '_if(', e)
    return e


def evaluate(expr, coords):
    """A weight / mode function on arrays: coords = (x, y, z) relative to its origin (drawing units). Returns an array
    broadcast to the coordinates; nan and inf become 0, as openEMS's mode matching does."""
    x, y, z = (np.asarray(c, float) for c in coords)
    x, y, z = np.broadcast_arrays(x, y, z)
    if expr is None or str(expr).strip() in ('', '0'):
        return np.zeros(x.shape)
    rho = np.sqrt(x * x + y * y)
    with np.errstate(all='ignore'):
        ns = dict(_FUNCS, x=x, y=y, z=z, rho=rho, a=np.arctan2(y, x), r=np.sqrt(x * x + y * y + z * z),
                  t=np.arcsin(1) - np.arctan(z / rho),
                  _if=lambda c, a, b: np.where(np.asarray(c) != 0, a, b))
        ns.pop('if')
        v = eval(compile(_fparser_to_python(expr), '<weight function>', 'eval'), {'__builtins__': {}}, ns)
        v = np.broadcast_to(np.asarray(v, float), x.shape).copy()
    v[~np.isfinite(v)] = 0.0
    return v


def read_mode_file(path):
    """openEMS's mode file (HDF5): 1-D 'x', 'y' (the two transverse axes) and 2-D 'Vx', 'Vy' on them."""
    import h5py
    with h5py.File(path, 'r') as f:
        if float(f.attrs.get('Version', 0)) < 1.0:
            raise ValueError(f"{path}: not a mode file (no Version >= 1)")
        return tuple(np.asarray(f[k], float) for k in ('x', 'y', 'Vx', 'Vy'))


def mode_file_values(data, u, v):
    """(Vx, Vy) of a mode file bilinearly interpolated at (u, v), clamped to its grid (CSModeData::LinInterp2)."""
    X, Y, A, B = data
    u = np.clip(np.asarray(u, float), X[0], X[-1])
    v = np.clip(np.asarray(v, float), Y[0], Y[-1])
    i = np.clip(np.searchsorted(X, u, 'left') - 1, 0, len(X) - 2)
    j = np.clip(np.searchsorted(Y, v, 'left') - 1, 0, len(Y) - 2)
    x1, x2, y1, y2 = X[i], X[i + 1], Y[j], Y[j + 1]
    D = 1.0 / ((x2 - x1) * (y2 - y1))
    f = lambda M: D * (M[i, j] * (x2 - u) * (y2 - v) + M[i + 1, j] * (u - x1) * (y2 - v) +
                       M[i, j + 1] * (x2 - u) * (v - y1) + M[i + 1, j + 1] * (u - x1) * (v - y1))
    return f(A), f(B)


class CSRectGrid:
    """Rectilinear grid: mesh lines per axis in drawing units, DeltaUnit metres per drawing unit."""
    def __init__(self):
        self.lines = [np.zeros(0), np.zeros(0), np.zeros(0)]
        self.delta_unit = 1.0

    def SetDeltaUnit(self, unit):
        self.delta_unit = _num(unit)

    def GetDeltaUnit(self):
        return self.delta_unit

    def SetLines(self, d, lines):
        self.lines[ny_dir(d)] = np.unique(np.asarray(lines, float))

    def AddLine(self, d, lines):
        a = ny_dir(d)
        self.lines[a] = np.unique(np.concatenate([self.lines[a], np.atleast_1d(np.asarray(lines, float))]))

    def GetLines(self, d, do_sort=False):
        return np.array(self.lines[ny_dir(d)])

    def GetQtyLines(self, d):
        return len(self.lines[ny_dir(d)])

    def ClearLines(self, d):
        self.lines[ny_dir(d)] = np.zeros(0)

    def Snap2LineNumber(self, d, value):
        lines = self.lines[ny_dir(d)]
        return int(np.argmin(np.abs(lines - value)))


class Primitive:
    kind = ''

    def __init__(self, prop, priority=0):
        self.prop, self.priority = prop, int(priority)

    def GetPriority(self):
        return self.priority

    def SetPriority(self, val):
        self.priority = int(val)

    def GetProperty(self):
        return self.prop


class Box(Primitive):
    kind = 'Box'

    def __init__(self, prop, start, stop, priority=0):
        super().__init__(prop, priority)
        self.start, self.stop = _vec(start), _vec(stop)

    def GetStart(self):
        return np.array(self.start)

    def GetStop(self):
        return np.array(self.stop)


class Polygon(Primitive):
    """A flat polygon at `elevation` normal to `norm_dir`; points = [coords of the first in-plane axis, of the
    second] (in-plane axes in xyz order without norm_dir)."""
    kind = 'Polygon'

    def __init__(self, prop, points, norm_dir, elevation, priority=0):
        super().__init__(prop, priority)
        self.points = np.asarray([np.asarray(points[0], float), np.asarray(points[1], float)])
        self.norm_dir, self.elevation = ny_dir(norm_dir), _num(elevation)


class LinPoly(Polygon):
    """A polygon extruded from `elevation` by `length` along norm_dir."""
    kind = 'LinPoly'

    def __init__(self, prop, points, norm_dir, elevation, length, priority=0):
        super().__init__(prop, points, norm_dir, elevation, priority)
        self.length = _num(length)


class Cylinder(Primitive):
    kind = 'Cylinder'

    def __init__(self, prop, start, stop, radius, priority=0):
        super().__init__(prop, priority)
        self.start, self.stop, self.radius = _vec(start), _vec(stop), _num(radius)


class CylindricalShell(Cylinder):
    """The points whose distance from the axis start -> stop is within shell_width / 2 of radius."""
    kind = 'CylindricalShell'

    def __init__(self, prop, start, stop, radius, shell_width, priority=0):
        super().__init__(prop, start, stop, radius, priority)
        self.shell_width = _num(shell_width)


class Curve(Primitive):
    """A polyline (points: [xs, ys, zs], as CSXCAD's): a wire along the mesh edges nearest to it in metals and
    excitations."""
    kind = 'Curve'

    def __init__(self, prop, points, priority=0):
        super().__init__(prop, priority)
        self.points = np.asarray(points, float).reshape(3, -1).T    # CSXCAD's layout: [xs, ys, zs]; kept N x 3

    def GetNumberOfPoints(self):
        return len(self.points)


class Property:
    """A CSXCAD property: its type (the XML tag), name, attributes and primitives."""
    kind = ''

    def __init__(self, name, **attrs):
        self.name, self.attrs, self.primitives, self.enabled = name, attrs, [], True

    def GetName(self):
        return self.name

    def GetType(self):
        return self.kind

    def GetQtyPrimitives(self):
        return len(self.primitives)

    def GetAllPrimitives(self):
        return list(self.primitives)

    def SetEnabled(self, val):
        self.enabled = bool(val)

    def _add(self, prim):
        self.primitives.append(prim)
        return prim

    def AddBox(self, start, stop, priority=0, **kw):
        return self._add(Box(self, start, stop, priority))

    def AddPolygon(self, points, norm_dir, elevation, priority=0, **kw):
        return self._add(Polygon(self, points, norm_dir, elevation, priority))

    def AddLinPoly(self, points, norm_dir, elevation, length, priority=0, **kw):
        return self._add(LinPoly(self, points, norm_dir, elevation, length, priority))

    def AddCylinder(self, start, stop, radius, priority=0, **kw):
        return self._add(Cylinder(self, start, stop, radius, priority))

    def AddCylindricalShell(self, start, stop, radius, shell_width, priority=0, **kw):
        return self._add(CylindricalShell(self, start, stop, radius, shell_width, priority))

    def AddCurve(self, points, priority=0, **kw):
        return self._add(Curve(self, points, priority))

    def __repr__(self):
        return f"{self.kind}({self.name!r}, {len(self.primitives)} primitive(s))"


class Material(Property):
    kind = 'Material'

    def __init__(self, name, epsilon=1.0, mue=1.0, kappa=0.0, sigma=0.0, **kw):
        super().__init__(name)
        self.epsilon, self.mue, self.kappa, self.sigma = _num(epsilon), _num(mue), _num(kappa), _num(sigma)

    def SetMaterialProperty(self, **kw):
        for k, v in kw.items():
            setattr(self, k.lower(), _num(v))

    def GetMaterialProperty(self, prop_name):
        return getattr(self, prop_name.lower())


class DebyeMaterial(Material):
    """epsilon(w) = epsilon + sum over the poles of eps_delta / (1 + j w eps_relax), and kappa: openEMS's
    CSPropDebyeMaterial (epsilon is the value at infinite frequency)."""
    kind = 'DebyeMaterial'

    def __init__(self, name, order=0, **kw):
        super().__init__(name, **kw)
        self.poles = {}

    def SetDispersiveMaterialProperty(self, order=0, **kw):
        self.poles.setdefault(int(order), {}).update({k.lower(): _num(v) for k, v in kw.items()})

    def GetDispersiveMaterialProperty(self, prop_name, order=0):
        return self.poles.get(int(order), {}).get(prop_name.lower(), 0.0)

    def GetDispersionOrder(self):
        return len(self.poles)

    def pole_list(self):
        """[(eps_delta, eps_relax s)] of the poles that act (both above zero)."""
        return [(d.get('eps_delta', 0.0), d.get('eps_relax', 0.0)) for _, d in sorted(self.poles.items())
                if d.get('eps_delta', 0.0) > 0 and d.get('eps_relax', 0.0) > 0]


class Metal(Property):
    kind = 'Metal'


class ConductingSheet(Property):
    kind = 'ConductingSheet'

    def __init__(self, name, conductivity=0.0, thickness=0.0, **kw):
        super().__init__(name)
        self.conductivity, self.thickness = _num(conductivity), _num(thickness)


class LumpedElement(Property):
    """R / L / C between the faces of its box along `ny`; LEtype 0 parallel, 1 series. caps: metal end caps."""
    kind = 'LumpedElement'

    def __init__(self, name, ny, caps=True, R=None, C=None, L=None, LEtype=0, **kw):
        super().__init__(name)
        self.ny, self.caps, self.LEtype = ny_dir(ny), bool(caps), int(LEtype)
        val = lambda v: None if v is None or not np.isfinite(float(v)) else float(v)
        self.R, self.C, self.L = val(R), val(C), val(L)


class Excitation(Property):
    """exc_type 0: soft E field (added to the edges in its primitives), 1: hard E field (the edges set to it), 2: soft
    H field, 3: hard H field; exc_val: the E (V/m) or H (A/m) vector, times the excitation signal, delay seconds late.
    A weight function per component (SetWeightFunction, in x, y, z, rho, a, r, t from SetWeightOrigin) or a mode
    file (SetWeightFile, with SetPropagationDir) scales it across the primitives."""
    kind = 'Excitation'

    def __init__(self, name, exc_type=0, exc_val=(0, 0, 0), delay=0.0, number=0, **kw):
        super().__init__(name)
        self.exc_type, self.exc_val, self.delay, self.number = int(exc_type), _vec(exc_val), _num(delay), int(number)
        self.weight = ['1', '1', '1']
        self.weight_file, self.weight_origin, self.prop_dir = '', [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def SetExcitation(self, val):
        self.exc_val = _vec(val)

    def GetExcitation(self):
        return list(self.exc_val)

    def SetDelay(self, val):
        self.delay = _num(val)

    def GetDelay(self):
        return self.delay

    def SetExcitType(self, val):
        self.exc_type = int(val)

    def GetExcitType(self):
        return self.exc_type

    def SetEnabled(self, val):
        self.enabled = bool(val)

    def GetEnabled(self):
        return self.enabled

    def SetWeightFunction(self, func):
        self.weight = [str(f) for f in func]
        self.weight_file = ''

    def GetWeightFunction(self):
        return list(self.weight)

    def SetWeightFile(self, path):
        self.weight_file = str(path)

    def SetWeightOrigin(self, origin):
        self.weight_origin = _vec(origin)

    def SetPropagationDir(self, val):
        self.prop_dir = _vec(val)

    def weights(self, a, coords):
        """The weight of component a at coords (x, y, z arrays, drawing units)."""
        x, y, z = (np.asarray(c, float) - o for c, o in zip(coords, self.weight_origin))
        if self.weight_file:
            n = int(np.argmax(np.abs(self.prop_dir)))
            if not np.count_nonzero(self.prop_dir) == 1:
                raise ValueError(f"excitation {self.name}: a mode file needs one propagation direction")
            if a == n:
                return np.zeros(np.broadcast(x, y, z).shape)
            loc = (x, y, z)
            vx, vy = mode_file_values(read_mode_file(self.weight_file), loc[(n + 1) % 3], loc[(n + 2) % 3])
            return vy if a == (n + 2) % 3 else vx
        return evaluate(self.weight[a], (x, y, z))


class ProbeBox(Property):
    """p_type 0: voltage (the E integral from start to stop), 1: current (the H loop around the box, normal
    norm_dir), 10 / 11: voltage / current mode matching (the transverse E / H on the box's plane against a mode
    function or mode file, as openEMS's waveguide ports); the probe's value is multiplied by weight. Written to a
    file of its name."""
    kind = 'ProbeBox'

    def __init__(self, name, p_type=0, weight=1.0, norm_dir=-1, mode_function=None, frequency=None, **kw):
        super().__init__(name)
        self.p_type, self.weight = int(p_type), _num(weight)
        self.norm_dir = -1 if norm_dir is None or (not isinstance(norm_dir, str) and int(norm_dir) < 0) else ny_dir(norm_dir)
        self.mode_function = ['', '', ''] if mode_function is None else [str(f) for f in mode_function]
        self.mode_file, self.mode_origin = '', [0.0, 0.0, 0.0]
        self.frequency = [] if frequency is None else _vec(frequency)
        self.extra = {k: v for k, v in kw.items()}

    def GetProbeType(self):
        return self.p_type

    def SetModeFunction(self, func):
        self.mode_function = [str(f) for f in func]

    def GetModeFunction(self, ny):
        return self.mode_function[ny]

    def SetModeFile(self, path):
        self.mode_file = str(path)

    def GetModeFile(self):
        return self.mode_file

    def SetModeOrigin(self, origin):
        self.mode_origin = _vec(origin)

    def GetModeOrigin(self, ny):
        return self.mode_origin[ny]

    def mode(self, n, coords):
        """The mode's two transverse components on a plane normal to axis n, at coords (drawing units): axes (n+1)%3
        and (n+2)%3, from the mode file (its Vx, Vy at those axes' local coordinates) or the expressions."""
        loc = [np.asarray(c, float) - o for c, o in zip(coords, self.mode_origin)]
        u, v = (n + 1) % 3, (n + 2) % 3
        if self.mode_file:
            return mode_file_values(read_mode_file(self.mode_file), loc[u], loc[v])
        return evaluate(self.mode_function[u], loc), evaluate(self.mode_function[v], loc)


class DumpBox(Property):
    """A field dump over its box, as openEMS's: dump_type 0 E, 1 H, 2 conduction current density, 3 total current
    density (rot H); 10-13 the same in the frequency domain at `frequency` (Hz). file_type 0 VTK, 1 HDF5;
    dump_mode 0 at the nodes, 2 at the cell centres (interpolated). sub_sampling: every n-th time sample."""
    kind = 'DumpBox'

    def __init__(self, name, dump_type=0, dump_mode=1, file_type=0, frequency=None, sub_sampling=None, **kw):
        super().__init__(name)
        self.dump_type, self.dump_mode, self.file_type = int(dump_type), int(dump_mode), int(file_type)
        self.frequency = [] if frequency is None else _vec(frequency)
        self.sub_sampling = sub_sampling
        self.extra = {k: v for k, v in kw.items() if k not in ('opt_resolution', 'multi_grid_level')}

    def GetDumpType(self):
        return self.dump_type

    def SetDumpType(self, val):
        self.dump_type = int(val)

    def GetDumpMode(self):
        return self.dump_mode

    def SetDumpMode(self, val):
        self.dump_mode = int(val)

    def GetFileType(self):
        return self.file_type

    def SetFileType(self, val):
        self.file_type = int(val)

    def SetFrequency(self, freq):
        self.frequency = _vec(freq)

    def AddFrequency(self, freq):
        self.frequency += _vec(np.atleast_1d(freq))

    def GetFrequency(self):
        return list(self.frequency)

    def ClearFrequency(self):
        self.frequency = []


class ParameterSet:
    pass


class ContinuousStructure:
    def __init__(self, **kw):
        self.grid = CSRectGrid()
        self.properties = []
        self.background = dict(Epsilon=1.0, Mue=1.0, Kappa=0.0, Sigma=0.0)
        self.params = ParameterSet()
        self.skipped = []                                           # what a file held that gfdtd does not model

    def GetGrid(self):
        return self.grid

    def GetParameterSet(self):
        return self.params

    def AddProperty(self, prop):
        self.properties.append(prop)
        return prop

    def AddMaterial(self, name, **kw):
        return self.AddProperty(Material(name, **kw))

    def AddMetal(self, name):
        return self.AddProperty(Metal(name))

    def AddConductingSheet(self, name, conductivity=0.0, thickness=0.0, **kw):
        return self.AddProperty(ConductingSheet(name, conductivity, thickness))

    def AddLumpedElement(self, name, ny, caps=True, **kw):
        return self.AddProperty(LumpedElement(name, ny, caps, **kw))

    def AddExcitation(self, name, exc_type, exc_val, delay=0.0, **kw):
        return self.AddProperty(Excitation(name, exc_type, exc_val, delay, **kw))

    def AddProbe(self, name, p_type, weight=1.0, norm_dir=-1, **kw):
        return self.AddProperty(ProbeBox(name, p_type, weight, norm_dir, **kw))

    def AddDebyeMaterial(self, name, order=0, **kw):
        """openEMS builds these as CSPropDebyeMaterial(CSX.GetParameterSet(), order, name, ...)."""
        return self.AddProperty(DebyeMaterial(name, order=order, **kw))

    def AddDump(self, name, **kw):
        return self.AddProperty(DumpBox(name, **kw))

    def GetPropertiesByName(self, name):
        return [p for p in self.properties if p.name == name]

    def GetAllProperties(self):
        return list(self.properties)

    def GetQtyProperties(self):
        return len(self.properties)

    # ---- XML ---------------------------------------------------------------------------------------------
    def to_xml(self):
        """The <ContinuousStructure> element, as CSXCAD writes it."""
        root = ET.Element('ContinuousStructure', CoordSystem='0')
        g = ET.SubElement(root, 'RectilinearGrid', DeltaUnit=f"{self.grid.delta_unit:g}", CoordSystem='0')
        for a, tag in enumerate(('XLines', 'YLines', 'ZLines')):
            el = ET.SubElement(g, tag, Qty=str(len(self.grid.lines[a])))
            el.text = ",".join(f"{v:.15g}" for v in self.grid.lines[a])
        ET.SubElement(root, 'BackgroundMaterial', **{k: f"{v:g}" for k, v in self.background.items()})
        ET.SubElement(root, 'ParameterSet')
        props = ET.SubElement(root, 'Properties')
        for i, p in enumerate(self.properties):
            _write_property(props, p, i)
        return root

    def Write2XML(self, path):
        top = ET.Element('openEMS')
        top.append(self.to_xml())
        _write(top, path)
        return True

    def ReadFromXML(self, path):
        root = ET.parse(path).getroot()
        el = root if root.tag == 'ContinuousStructure' else root.find('ContinuousStructure')
        if el is None:
            raise ValueError(f"{path}: no <ContinuousStructure>")
        self.from_xml(el)
        return True

    def from_xml(self, el):
        g = el.find('RectilinearGrid')
        if g is None:
            raise ValueError("CSX: no <RectilinearGrid> (gfdtd needs a Cartesian rectilinear grid)")
        if g.get('CoordSystem', '0') not in ('0', 'cartesian'):
            raise ValueError("CSX: cylindrical grids are not supported by gfdtd")
        self.grid.SetDeltaUnit(float(g.get('DeltaUnit', '1')))
        for a, tag in enumerate(('XLines', 'YLines', 'ZLines')):
            t = g.find(tag)
            self.grid.SetLines(a, _floats(t.text or '') if t is not None else [])
        bg = el.find('BackgroundMaterial')
        if bg is not None:
            for k in self.background:
                if bg.get(k) is not None:
                    self.background[k] = float(bg.get(k))
        props = el.find('Properties')
        for pe in (props if props is not None else []):
            p = _read_property(pe, self.skipped)
            if p is not None:
                self.properties.append(p)


def _write(top, path):
    ET.indent(top, '    ')
    with open(path, 'wb') as fh:
        fh.write(b'<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n')
        fh.write(ET.tostring(top, encoding='utf-8'))
        fh.write(b'\n')


def _three(v):
    return ",".join(_fmt(x) for x in (v, 1.0, 1.0)) if np.isscalar(v) else ",".join(_fmt(x) for x in v)


def _write_property(parent, p, pid):
    a = dict(ID=str(pid), Name=p.name)
    if isinstance(p, Material):
        a['Isotropy'] = '1'
    elif isinstance(p, ConductingSheet):
        a.update(Conductivity=_fmt(p.conductivity), Thickness=_fmt(p.thickness))
    elif isinstance(p, LumpedElement):
        a.update(Direction=str(p.ny), Caps='1' if p.caps else '0')
        for k in ('R', 'C', 'L'):
            if getattr(p, k) is not None:
                a[k] = _fmt(getattr(p, k))
        a['LEtype'] = _fmt(p.LEtype)
    elif isinstance(p, Excitation):
        a.update(Number=str(p.number), Enabled='1' if p.enabled else '0', Frequency=_fmt(0), Delay=_fmt(p.delay),
                 Type=str(p.exc_type), Excite=",".join(_fmt(v) for v in p.exc_val),
                 PropDir=",".join(_fmt(v) for v in p.prop_dir))
        if p.weight_file:
            a['WeightFile'] = p.weight_file
    elif isinstance(p, ProbeBox):
        a.update(Visible='0', Number='0', Type=str(p.p_type), Weight=f"{p.weight:g}", NormDir=str(p.norm_dir),
                 StartTime='0', StopTime='0')
        if p.mode_file:
            a['ModeFile'] = p.mode_file
    elif isinstance(p, DumpBox):
        a.update(Visible='0', Number='0', Type='0', Weight='1', NormDir='-1', StartTime='0', StopTime='0',
                 DumpType=str(p.dump_type), DumpMode=str(p.dump_mode), FileType=str(p.file_type), MultiGridLevel='0')
        if p.sub_sampling is not None:
            a['SubSampling'] = ",".join(str(int(v)) for v in np.broadcast_to(p.sub_sampling, 3))
        a.update({k: str(v) for k, v in p.extra.items()})
    el = ET.SubElement(parent, p.kind, **a)
    prims = ET.SubElement(el, 'Primitives')
    for q in p.primitives:
        _write_primitive(prims, q)
    if isinstance(p, Material):
        pa = dict(Epsilon=_three(p.epsilon), Mue=_three(p.mue), Kappa=_three(p.kappa), Sigma=_three(p.sigma),
                  Density=_fmt(0))
        if isinstance(p, DebyeMaterial):
            for o, d in sorted(p.poles.items()):
                pa[f"EpsilonDelta_{o + 1}"] = _three(d.get('eps_delta', 0.0))
                pa[f"EpsilonRelaxTime_{o + 1}"] = _three(d.get('eps_relax', 0.0))
        ET.SubElement(el, 'Property', **pa)
    if isinstance(p, Excitation):
        if any(p.weight_origin):
            ET.SubElement(el, 'WeightOrigin', **dict(zip('XYZ', (_fmt(v) for v in p.weight_origin))))
        ET.SubElement(el, 'Weight', **dict(zip('XYZ', p.weight)))
    if isinstance(p, ProbeBox) and not isinstance(p, DumpBox):
        if p.frequency:
            ET.SubElement(el, 'FD_Samples').text = ",".join(f"{f:.15g}" for f in p.frequency)
        if any(p.mode_function):
            ET.SubElement(el, 'ModeFunction', **dict(zip('XYZ', p.mode_function)))
        if any(p.mode_origin):
            ET.SubElement(el, 'ModeOrigin', **dict(zip('XYZ', (_fmt(v) for v in p.mode_origin))))
    if isinstance(p, DumpBox) and p.frequency:
        ET.SubElement(el, 'FD_Samples').text = ",".join(f"{f:.15g}" for f in p.frequency)


def _write_primitive(parent, q):
    pr = dict(Priority=str(q.priority))
    if isinstance(q, Box):
        el = ET.SubElement(parent, 'Box', **pr)
        ET.SubElement(el, 'P1', X=_fmt(q.start[0]), Y=_fmt(q.start[1]), Z=_fmt(q.start[2]))
        ET.SubElement(el, 'P2', X=_fmt(q.stop[0]), Y=_fmt(q.stop[1]), Z=_fmt(q.stop[2]))
    elif isinstance(q, Cylinder):
        extra = dict(ShellWidth=_fmt(q.shell_width)) if isinstance(q, CylindricalShell) else {}
        el = ET.SubElement(parent, q.kind, **pr, Radius=_fmt(q.radius), **extra)
        ET.SubElement(el, 'P1', X=_fmt(q.start[0]), Y=_fmt(q.start[1]), Z=_fmt(q.start[2]))
        ET.SubElement(el, 'P2', X=_fmt(q.stop[0]), Y=_fmt(q.stop[1]), Z=_fmt(q.stop[2]))
    elif isinstance(q, Curve):
        el = ET.SubElement(parent, 'Curve', **pr)
        for P in q.points:
            ET.SubElement(el, 'Vertex', X=_fmt(P[0]), Y=_fmt(P[1]), Z=_fmt(P[2]))
    else:                                                           # Polygon, LinPoly
        a = dict(pr, Elevation=_fmt(q.elevation), NormDir=str(q.norm_dir), QtyVertices=str(q.points.shape[1]))
        if isinstance(q, LinPoly):
            a['Length'] = _fmt(q.length)
        el = ET.SubElement(parent, q.kind, **a)
        for u, v in q.points.T:
            ET.SubElement(el, 'Vertex', X1=_fmt(u), X2=_fmt(v))


def _point(el):
    return [float(el.get(k, 0)) for k in ('X', 'Y', 'Z')]


def _first(el, key, default=0.0):
    v = el.get(key)
    return _floats(v)[0] if v is not None else default


def _read_property(pe, skipped):
    tag, name = pe.tag, pe.get('Name', '')
    if tag in ('Material', 'DebyeMaterial', 'LorentzMaterial'):
        pr = pe.find('Property')
        g = (lambda k, d: _first(pr, k, d)) if pr is not None else (lambda k, d: d)
        p = (DebyeMaterial if tag != 'Material' else Material)(name, epsilon=g('Epsilon', 1.0), mue=g('Mue', 1.0),
                                                               kappa=g('Kappa', 0.0), sigma=g('Sigma', 0.0))
        if tag == 'DebyeMaterial' and pr is not None:
            n = 1
            while pr.get(f'EpsilonDelta_{n}') is not None:
                p.SetDispersiveMaterialProperty(order=n - 1, eps_delta=g(f'EpsilonDelta_{n}', 0.0),
                                                eps_relax=g(f'EpsilonRelaxTime_{n}', 0.0))
                n += 1
        elif tag != 'Material':
            p.kind = tag
            skipped.append(f"{tag} {name}: dispersion not modelled, its static epsilon used")
        if pr is not None and pe.get('Isotropy', '1') == '0':
            skipped.append(f"Material {name}: anisotropic, its x values used")
    elif tag == 'Metal':
        p = Metal(name)
    elif tag == 'ConductingSheet':
        p = ConductingSheet(name, float(pe.get('Conductivity', 0)), float(pe.get('Thickness', 0)))
    elif tag == 'LumpedElement':
        val = lambda k: float(pe.get(k)) if pe.get(k) not in (None, 'nan') else None
        p = LumpedElement(name, int(pe.get('Direction', 0)), pe.get('Caps', '1') == '1', R=val('R'), C=val('C'),
                          L=val('L'), LEtype=int(float(pe.get('LEtype', 0))))
    elif tag == 'Excitation':
        p = Excitation(name, int(pe.get('Type', 0)), _floats(pe.get('Excite', '0,0,0'), 3), float(pe.get('Delay', 0)),
                       number=int(pe.get('Number', 0)))
        p.enabled = pe.get('Enabled', '1') != '0'
        p.prop_dir = _floats(pe.get('PropDir', '0,0,0'), 3)
        p.weight_file = pe.get('WeightFile', '')
        w, wo = pe.find('Weight'), pe.find('WeightOrigin')
        if w is not None and not p.weight_file:
            p.weight = [w.get(k, '1') for k in 'XYZ']
        if wo is not None:
            p.weight_origin = [float(wo.get(k, 0)) for k in 'XYZ']
    elif tag == 'ProbeBox':
        p = ProbeBox(name, int(pe.get('Type', 0)), float(pe.get('Weight', 1)), int(pe.get('NormDir', -1)))
        p.mode_file = pe.get('ModeFile', '')
        mf, mo, fd = pe.find('ModeFunction'), pe.find('ModeOrigin'), pe.find('FD_Samples')
        if mf is not None:
            p.mode_function = [mf.get(k, '') for k in 'XYZ']
        if mo is not None:
            p.mode_origin = [float(mo.get(k, 0)) for k in 'XYZ']
        if fd is not None and (fd.text or '').strip():
            p.frequency = _floats(fd.text)
    elif tag == 'DumpBox':
        fd = pe.find('FD_Samples')
        sub_s = pe.get('SubSampling')
        p = DumpBox(name, dump_type=int(pe.get('DumpType', 0)), dump_mode=int(pe.get('DumpMode', 1)),
                    file_type=int(pe.get('FileType', 0)),
                    frequency=_floats(fd.text) if fd is not None and (fd.text or '').strip() else None,
                    sub_sampling=[int(float(v)) for v in sub_s.split(',')] if sub_s else None)
    else:
        skipped.append(f"{tag} {name}: property type not supported")
        return None
    prims = pe.find('Primitives')
    for q in (prims if prims is not None else []):
        pri = int(float(q.get('Priority', 0)))
        if q.tag == 'Box':
            p.AddBox(_point(q.find('P1')), _point(q.find('P2')), priority=pri)
        elif q.tag == 'Cylinder':
            p.AddCylinder(_point(q.find('P1')), _point(q.find('P2')), float(q.get('Radius')), priority=pri)
        elif q.tag == 'CylindricalShell':
            p.AddCylindricalShell(_point(q.find('P1')), _point(q.find('P2')), float(q.get('Radius')),
                                  float(q.get('ShellWidth')), priority=pri)
        elif q.tag == 'Curve':
            p.AddCurve(np.array([_point(v) for v in q.findall('Vertex')]).T, priority=pri)
        elif q.tag in ('Polygon', 'LinPoly'):
            vs = [(float(v.get('X1')), float(v.get('X2'))) for v in q.findall('Vertex')]
            pts = [np.array([u for u, _ in vs]), np.array([w for _, w in vs])]
            nd, el = int(q.get('NormDir', 2)), float(q.get('Elevation', 0))
            if q.tag == 'LinPoly':
                p.AddLinPoly(pts, nd, el, float(q.get('Length', 0)), priority=pri)
            else:
                p.AddPolygon(pts, nd, el, priority=pri)
        else:
            skipped.append(f"{p.kind} {name}: primitive {q.tag} not supported")
    return p
