"""gfdtd's openEMS ports beyond the lumped one (gfdtd.ports: MSL, stripline, CPW, coax, waveguides, curve) and what
they need from the CSX and the operator: weight / mode functions, cylindrical shells, curves, mode-matching probes,
hard, H and delayed excitations, current probes snapped to the dual mesh. Small models with known answers; the
comparison with openEMS itself is in docs/gfdtd.md (Ports)."""
import os

import numpy as np
import pytest

from gfdtd import ContinuousStructure, openEMS
from gfdtd import csx as C
from gfdtd.operator import build, find_path, Grid, _current_nodes


def smooth(lines, res):
    lines = np.unique(np.asarray(lines, float))
    out = [lines[0]]
    for a, b in zip(lines[:-1], lines[1:]):
        n = max(1, int(np.ceil((b - a) / res - 1e-9)))
        out += list(np.linspace(a, b, n + 1)[1:])
    return np.array(out)


def _fdtd(f0, fc, bc, nrts=20000):
    F = openEMS(NrTS=nrts, EndCriteria=1e-4)
    F.SetGaussExcite(f0, fc)
    F.SetBoundaryCond(bc)
    csx = ContinuousStructure()
    F.SetCSX(csx)
    csx.GetGrid().SetDeltaUnit(1e-3)
    return F, csx


def test_function_parser_expressions():
    v = C.evaluate('2*cos(pi*x/4)*sin(y)^2', ([0, 2], [np.pi / 2, np.pi / 2], [0, 0]))
    np.testing.assert_allclose(v, [2, 0], atol=1e-12)
    np.testing.assert_allclose(C.evaluate('(rho<1.5)*j1(rho)', ([1, 3], [0, 0], [0, 0])),
                               [0.4400505857, 0], rtol=1e-8)
    np.testing.assert_allclose(C.evaluate('if(x>=0, x, -1)', ([-2, 3], [0, 0], [0, 0])), [-1, 3])
    np.testing.assert_allclose(C.evaluate('1/x', ([0, 2], [0, 0], [0, 0])), [0, 0.5])    # inf -> 0


def test_find_path_walks_the_edges_nearest_a_diagonal():
    g = C.CSRectGrid()
    for a in 'xyz':
        g.SetLines(a, np.arange(0, 6.0))
    path = find_path(Grid(g), [0, 0, 0], [4, 2, 0])
    assert len(path) == 6 and sum(1 for a, _ in path if a == 0) == 4 and all(ijk[2] == 0 for _, ijk in path)
    pos = np.zeros(3, int)
    for a, ijk in path:                                             # a connected walk from start to end
        assert tuple(pos) in (ijk, tuple(np.array(ijk) + np.eye(3, dtype=int)[a]))
        pos = np.array(ijk) + np.eye(3, dtype=int)[a] if tuple(pos) == ijk else np.array(ijk)
    assert tuple(pos) == (4, 2, 0)


def test_current_probe_snaps_like_openems():
    """openEMS snaps a current box to the dual mesh: on lines, the nodes from start to stop; on half-cell
    coordinates (the stripline / CPW ports' boxes), the nodes between them; flat, the nearest node."""
    g = C.CSRectGrid()
    g.SetLines('x', [0, 1, 2, 3, 4])
    for a in 'yz':
        g.SetLines(a, [0, 1])
    grid = Grid(g)
    assert _current_nodes(grid, 0, 1.0, 3.0) == (1, 3)
    assert _current_nodes(grid, 0, 0.5, 2.5) == (1, 2)
    assert _current_nodes(grid, 0, 1.2, 1.2) == (1, 1)


def test_new_csx_features_xml_both_ways(tmp_path):
    csx = ContinuousStructure()
    g = csx.GetGrid()
    g.SetDeltaUnit(1e-3)
    for a in 'xyz':
        g.SetLines(a, [0, 1, 2])
    m = csx.AddMetal('m')
    m.AddCylindricalShell([0, 0, 0], [0, 0, 2], 1.0, 0.2, priority=3)
    m.AddCurve([[0, 1, 2], [0, 0, 1], [0, 1, 1]], priority=4)
    e = csx.AddExcitation('e', exc_type=2, exc_val=[1, 1, 0], delay=1e-10)
    e.SetWeightFunction(['cos(x)', 'y^2', '0'])
    e.SetWeightOrigin([1, 2, 3])
    e.SetPropagationDir([0, 0, 1])
    e.AddBox([0, 0, 1], [2, 2, 1])
    p = csx.AddProbe('p', p_type=11, weight=-1, mode_function=['sin(x)', '0', '0'])
    p.SetModeOrigin([0.5, 0, 0])
    p.AddBox([0, 0, 1], [2, 2, 1])
    path = str(tmp_path / 'm.xml')
    csx.Write2XML(path)
    c2 = ContinuousStructure()
    c2.ReadFromXML(path)
    assert not c2.skipped
    m2, e2, p2 = c2.GetAllProperties()
    sh, cv = m2.primitives
    assert isinstance(sh, C.CylindricalShell) and sh.shell_width == pytest.approx(0.2) and sh.priority == 3
    np.testing.assert_allclose(cv.points, [[0, 0, 0], [1, 0, 1], [2, 1, 1]])
    assert (e2.exc_type, e2.weight, e2.weight_origin, e2.prop_dir) == (2, ['cos(x)', 'y^2', '0'], [1, 2, 3], [0, 0, 1])
    assert e2.delay == pytest.approx(1e-10)
    assert (p2.p_type, p2.weight, p2.mode_function[0], p2.mode_origin) == (11, -1, 'sin(x)', [0.5, 0, 0])


def test_excitations_hard_weighted_delayed_and_h():
    """A hard E excitation takes its edges out of the update; a weighted one scales each edge by the function at
    its midpoint; a delayed one and an H one go to their own groups, on the H signal half a step later."""
    F, csx = _fdtd(1e9, 1e9, ['PEC'] * 6)
    g = csx.GetGrid()
    for a in 'xyz':
        g.SetLines(a, np.arange(0, 5.0))
    csx.AddExcitation('hard', exc_type=1, exc_val=[0, 0, 1]).AddBox([1, 1, 1], [1, 1, 2])
    w = csx.AddExcitation('w', exc_type=0, exc_val=[2, 0, 0])
    w.SetWeightFunction(['x', '1', '1'])
    w.AddBox([0, 3, 3], [2, 3, 3])
    csx.AddExcitation('late', exc_type=0, exc_val=[0, 1, 0], delay=1e-11).AddBox([3, 0, 1], [3, 1, 1])
    csx.AddExcitation('h', exc_type=2, exc_val=[0, 0, 1]).AddBox([2, 2, 2], [3, 3, 3])
    op = build(csx, F.settings())
    flat = lambda a, i, j, k: ((a * 5 + i) * 5 + j) * 5 + k
    assert op.vv[2, 1, 1, 1] == 0 and op.vi[2, 1, 1, 1] == 0
    amp = dict(zip(op.exc[0].tolist(), op.exc[1].tolist()))
    assert amp[flat(2, 1, 1, 1)] == pytest.approx(1e-3)
    assert amp[flat(0, 0, 3, 3)] == pytest.approx(2 * 0.5 * 1e-3) and amp[flat(0, 1, 3, 3)] == pytest.approx(2 * 1.5e-3)
    late, h = sorted(op.exc_more, key=lambda x: x['field'])[::-1]
    assert (late['field'], late['delay'], h['field'], h['hard']) == ('V', int(1e-11 / op.dt), 'I', False)
    assert flat(2, 2, 2, 2) in h['idx'].tolist()
    assert op.signal_h[3] == pytest.approx(np.interp(3.5, np.arange(len(op.signal)), op.signal), rel=1e-2)


def test_delayed_port_starts_late(tmp_path):
    """A lumped port's delay (openEMS's `delay`): its voltage is the undelayed one, steps later."""
    out = []
    for delay in (0.0, 2e-10):
        F, csx = _fdtd(2e9, 1e9, ['MUR'] * 6, nrts=2500)
        g = csx.GetGrid()
        for a in 'xyz':
            g.SetLines(a, np.arange(-10, 11.0))
        port = F.AddLumpedPort(1, 50, [0, 0, -1], [0, 0, 1], 'z', excite=1, delay=delay)
        path = str(tmp_path / f'd{delay:g}')
        F.Run(path, backend='numpy', verbose=0)
        port.ReadUIData(path, [1e9])
        out.append((port.u_time, port.ut_tot, F.op.dt))
    (t0, u0, dt), (t1, u1, _) = out
    late = int(2e-10 / dt) * dt                                     # openEMS rounds the delay down to whole steps
    np.testing.assert_allclose(np.interp(t1, t0 + late, u0, left=0), u1, atol=1e-2 * np.abs(u0).max())   # sampled: interp


def test_rect_waveguide_port_through_an_empty_guide(tmp_path):
    """TE10 ports at both ends of an empty WR-90 guide: the mode goes through whole, nothing reflects, and the
    port's impedance is the TE10 wave impedance."""
    a, b, L = 22.86, 10.16, 80.0
    F, csx = _fdtd(10e9, 3e9, ['PEC', 'PEC', 'PEC', 'PEC', 'PML_8', 'PML_8'])
    g = csx.GetGrid()
    g.SetLines('x', smooth([0, a], 1.2))
    g.SetLines('y', smooth([0, b], 1.2))
    g.SetLines('z', smooth([0, L], 1.0))
    p1 = F.AddRectWaveGuidePort(0, [0, 0, 10], [a, b, 15], 'z', a * 1e-3, b * 1e-3, 'TE10', excite=1)
    p2 = F.AddRectWaveGuidePort(1, [0, 0, L - 10], [a, b, L - 15], 'z', a * 1e-3, b * 1e-3, 'TE10')
    path = str(tmp_path / 'wg')
    F.Run(path, verbose=0)
    f = np.linspace(8.5e9, 11.5e9, 31)
    for p in (p1, p2):
        p.CalcPort(path, f)
    s11, s21 = p1.uf_ref / p1.uf_inc, p2.uf_ref / p1.uf_inc
    assert np.abs(s11).max() < 0.06 and np.abs(np.abs(s21) - 1).max() < 0.04
    k = 2 * np.pi * f / 299792458.0
    np.testing.assert_allclose(np.real(p1.ZL), 376.73 * k / np.sqrt(k ** 2 - (np.pi / (a * 1e-3)) ** 2), rtol=1e-3)


def test_coax_port_measures_the_line_impedance(tmp_path):
    """A PTFE-filled coaxial line between two coax ports: Z from the probes as 60 / sqrt(er) ln(ro / ri)."""
    ri, ro, ros, L, er = 1.0, 2.3, 2.8, 40.0, 2.1
    F, csx = _fdtd(5e9, 4e9, ['PEC', 'PEC', 'PEC', 'PEC', 'MUR', 'MUR'])
    g = csx.GetGrid()
    xs = smooth([-3.2, -ros, -ro, -ri, ri, ro, ros, 3.2], 0.15)
    g.SetLines('x', xs)
    g.SetLines('y', xs)
    g.SetLines('z', smooth([0, L], 0.5))
    pec, fill = csx.AddMetal('pec'), csx.AddMaterial('fill', epsilon=er)
    p1 = F.AddCoaxialPort(1, pec, fill, [0, 0, 0], [0, 0, 20], 'z', ri, ro, ros, excite=1, FeedShift=3,
                          MeasPlaneShift=12)
    p2 = F.AddCoaxialPort(2, pec, fill, [0, 0, L], [0, 0, 20], 'z', ri, ro, ros, MeasPlaneShift=12)
    path = str(tmp_path / 'coax')
    F.Run(path, verbose=0)
    f = np.linspace(2e9, 8e9, 13)
    for p in (p1, p2):
        p.CalcPort(path, f)
    z = 60 / np.sqrt(er) * np.log(ro / ri)
    np.testing.assert_allclose(np.real(p1.Z_ref), z, rtol=0.04)
    assert np.abs(p2.uf_ref / p1.uf_inc).min() > 0.97


def test_msl_port_on_a_through_line(tmp_path):
    """Two MSL ports on a 1.5 mm strip over 0.8 mm (er 4.4): the line's impedance from the probes is the
    closed form's (Hammerstad-Jensen; within 8 %: zero-thickness copper on a 0.25 mm mesh), and on it the line
    matches and goes through."""
    from kicad2ems.solvers.lines.tlines import microstrip
    h, w, L = 0.8, 1.5, 40.0
    F, csx = _fdtd(5e9, 4e9, ['MUR', 'MUR', 'MUR', 'MUR', 'PEC', 'MUR'])
    g = csx.GetGrid()
    g.SetLines('x', smooth([-L / 2, L / 2], 0.4))
    g.SetLines('y', smooth([-8, -w / 2, w / 2, 8], 0.25))
    g.SetLines('z', smooth([0, h, h + 6], 0.2))
    csx.AddMaterial('sub', epsilon=4.4).AddBox([-L / 2, -8, 0], [L / 2, 8, h])
    cu = csx.AddMetal('cu')
    cu.AddBox([-L / 2 + 12, -w / 2, h], [L / 2 - 12, w / 2, h], priority=10)
    p1 = F.AddMSLPort(1, cu, [-L / 2, -w / 2, h], [-L / 2 + 12, w / 2, 0], 'x', 'z', excite=-1, FeedShift=2,
                      MeasPlaneShift=8, priority=10)
    p2 = F.AddMSLPort(2, cu, [L / 2, -w / 2, h], [L / 2 - 12, w / 2, 0], 'x', 'z', MeasPlaneShift=8, priority=10)
    path = str(tmp_path / 'msl')
    F.Run(path, verbose=0)
    f = np.linspace(1e9, 6e9, 11)
    for p in (p1, p2):
        p.CalcPort(path, f)
    z_line = microstrip(w, h, 4.4)[0]
    np.testing.assert_allclose(np.real(p1.Z_ref), z_line, rtol=0.08)
    p1.CalcPort(path, f, ref_impedance=p1.Z_ref)
    p2.CalcPort(path, f, ref_impedance=p1.Z_ref)
    assert np.abs(p1.uf_ref / p1.uf_inc).max() < 0.08 and np.abs(p2.uf_ref / p1.uf_inc).min() > 0.95


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_lumped_port_matched_along_every_axis(tmp_path, axis):
    """A 50 ohm port driving a 50 ohm resistor straight across from it reflects little along x, y and z alike. (The
    current loop of a probe normal to y once ran the wrong way round: |S11| came out near 95 along y.)"""
    a = 'xyz'.index(axis)
    F, csx = _fdtd(2e9, 1e9, ['MUR'] * 6, nrts=4000)
    g = csx.GetGrid()
    for ax in 'xyz':
        g.SetLines(ax, np.arange(-10, 11.0))
    p0, p1, q0, q1 = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    p1[a] = q1[a] = 1.0
    q0[(a + 1) % 3] = q1[(a + 1) % 3] = 1.0                          # the load one cell beside the port
    port = F.AddLumpedPort(1, 50, p0, p1, axis, excite=1)
    csx.AddLumpedElement('load', ny=a, caps=True, R=50).AddBox(q0, q1)
    m = csx.AddMetal('m')
    for z0 in (0.0, 1.0):                                           # straps from the port's ends to the load's
        s, e = np.zeros(3), np.zeros(3)
        s[a] = e[a] = z0
        e[(a + 1) % 3] = 1.0
        m.AddBox(s, e)
    path = str(tmp_path / axis)
    F.Run(path, backend='numpy', verbose=0)
    port.CalcPort(path, [0.5e9, 1e9])
    assert np.abs(port.uf_ref / port.uf_inc).max() < 0.2
