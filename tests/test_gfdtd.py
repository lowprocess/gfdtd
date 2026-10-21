"""gfdtd: CSXCAD / openEMS compatibility (the API, XML both ways), the operator, the physics on small models with
known answers (numpy backend; the GPU backend against it when CuPy finds a GPU), and the kicad2ems plugin."""
import os

import numpy as np
import pytest

from gfdtd import ContinuousStructure, openEMS
from gfdtd import csx as C
from gfdtd.engine import gpu_info, core_info
from gfdtd.operator import build, timestep, Grid

# as CSXCAD 0.7 / openEMS 0.37 write it (FDTD.Write2XML), trimmed
OPENEMS_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<openEMS>
    <FDTD NumberOfTimesteps="1000" endCriteria="0.0001" OverSampling="4" TimeStepMethod="3">
        <Excitation Type="0" f0="3e+09" fc="2.5e+09" />
        <BoundaryCond xmin="MUR" xmax="MUR" ymin="PML_8" ymax="PEC" zmin="PMC" zmax="2" />
    </FDTD>
    <ContinuousStructure CoordSystem="0">
        <RectilinearGrid DeltaUnit="0.001" CoordSystem="0">
            <XLines Qty="11">0,1,2,3,4,5,6,7,8,9,10</XLines>
            <YLines Qty="6">0,1,2,3,4,5</YLines>
            <ZLines Qty="4">0,0.5,1,2</ZLines>
        </RectilinearGrid>
        <BackgroundMaterial Epsilon="1" Mue="1" Kappa="0" Sigma="0" />
        <ParameterSet />
        <Properties>
            <Material ID="0" Name="FR4" Isotropy="1">
                <FillColor R="103" G="198" B="105" a="123" />
                <Primitives>
                    <LinPoly Priority="1" Elevation="0.000000e+00" NormDir="2" QtyVertices="4" Length="1.000000e+00">
                        <Vertex X1="0.000000e+00" X2="0.000000e+00" />
                        <Vertex X1="1.000000e+01" X2="0.000000e+00" />
                        <Vertex X1="1.000000e+01" X2="5.000000e+00" />
                        <Vertex X1="0.000000e+00" X2="5.000000e+00" />
                    </LinPoly>
                </Primitives>
                <Property Epsilon="4.300000e+00,1.000000e+00,1.000000e+00" Mue="1.000000e+00,1.000000e+00,1.000000e+00" Kappa="1.000000e-02,0.000000e+00,0.000000e+00" Sigma="0.000000e+00,0.000000e+00,0.000000e+00" Density="0.000000e+00" />
            </Material>
            <Metal ID="1" Name="cu">
                <Primitives>
                    <Polygon Priority="10" Elevation="1.000000e+00" NormDir="2" QtyVertices="4">
                        <Vertex X1="1.000000e+00" X2="2.000000e+00" />
                        <Vertex X1="9.000000e+00" X2="2.000000e+00" />
                        <Vertex X1="9.000000e+00" X2="3.000000e+00" />
                        <Vertex X1="1.000000e+00" X2="3.000000e+00" />
                    </Polygon>
                </Primitives>
            </Metal>
            <Metal ID="3" Name="vias">
                <Primitives>
                    <Cylinder Priority="11" Radius="2.000000e-01">
                        <P1 X="5.000000e+00" Y="2.500000e+00" Z="0.000000e+00" />
                        <P2 X="5.000000e+00" Y="2.500000e+00" Z="1.000000e+00" />
                    </Cylinder>
                </Primitives>
            </Metal>
            <LumpedElement ID="5" Name="port_resist_1" Direction="2" Caps="1" R="5.000000e+01" LEtype="0.000000e+00">
                <Primitives>
                    <Box Priority="12">
                        <P1 X="1.000000e+00" Y="2.000000e+00" Z="0.000000e+00" />
                        <P2 X="1.000000e+00" Y="3.000000e+00" Z="1.000000e+00" />
                    </Box>
                </Primitives>
            </LumpedElement>
            <Excitation ID="6" Name="port_excite_1" Number="0" Enabled="1" Frequency="0.000000e+00" Delay="0.000000e+00" Type="0" Excite="0.000000e+00,0.000000e+00,-1.000000e+00" PropDir="0.000000e+00,0.000000e+00,0.000000e+00">
                <Primitives>
                    <Box Priority="12">
                        <P1 X="1.000000e+00" Y="2.000000e+00" Z="0.000000e+00" />
                        <P2 X="1.000000e+00" Y="3.000000e+00" Z="1.000000e+00" />
                    </Box>
                </Primitives>
                <Weight X="1.000000e+00" Y="1.000000e+00" Z="1.000000e+00" />
            </Excitation>
            <ProbeBox ID="7" Name="port_ut_1" Visible="0" Number="0" Type="0" Weight="-1" NormDir="-1" StartTime="0" StopTime="0">
                <Primitives>
                    <Box Priority="0">
                        <P1 X="1.000000e+00" Y="2.500000e+00" Z="0.000000e+00" />
                        <P2 X="1.000000e+00" Y="2.500000e+00" Z="1.000000e+00" />
                    </Box>
                </Primitives>
            </ProbeBox>
            <ProbeBox ID="8" Name="port_it_1" Visible="0" Number="0" Type="1" Weight="1" NormDir="2" StartTime="0" StopTime="0">
                <Primitives>
                    <Box Priority="0">
                        <P1 X="1.000000e+00" Y="2.000000e+00" Z="5.000000e-01" />
                        <P2 X="1.000000e+00" Y="3.000000e+00" Z="5.000000e-01" />
                    </Box>
                </Primitives>
            </ProbeBox>
            <RotPoly ID="9" Name="odd"><Primitives /></RotPoly>
        </Properties>
    </ContinuousStructure>
</openEMS>
'''


def test_reads_openems_xml(tmp_path):
    fn = tmp_path / 'model.xml'
    fn.write_text(OPENEMS_XML)
    F = openEMS()
    F.ReadFromXML(str(fn))
    assert F.bc == ['MUR', 'MUR', 'PML_8', 'PEC', 'PMC', 'MUR']          # names, and openEMS's numbers (2: MUR)
    assert F.excitation == dict(type=0, f0=3e9, fc=2.5e9) and F.nr_ts == 1000 and F.end_crit == 1e-4
    csx = F.csx
    assert csx.GetGrid().GetDeltaUnit() == 1e-3 and list(csx.GetGrid().GetLines('z')) == [0, 0.5, 1, 2]
    kinds = [(p.kind, p.name) for p in csx.GetAllProperties()]
    assert kinds == [('Material', 'FR4'), ('Metal', 'cu'), ('Metal', 'vias'), ('LumpedElement', 'port_resist_1'),
                     ('Excitation', 'port_excite_1'), ('ProbeBox', 'port_ut_1'), ('ProbeBox', 'port_it_1')]
    assert any('RotPoly' in s for s in csx.skipped)                   # unknown: listed, not modelled
    fr4 = csx.GetPropertiesByName('FR4')[0]
    assert (fr4.epsilon, fr4.kappa) == (4.3, 0.01)
    lp = fr4.primitives[0]
    assert isinstance(lp, C.LinPoly) and lp.length == 1.0 and lp.points.shape == (2, 4)
    le = csx.GetPropertiesByName('port_resist_1')[0]
    assert (le.ny, le.R, le.C, le.caps) == (2, 50.0, None, True)
    ex = csx.GetPropertiesByName('port_excite_1')[0]
    assert ex.exc_val == [0, 0, -1.0]
    it = csx.GetPropertiesByName('port_it_1')[0]
    assert (it.p_type, it.norm_dir, it.weight) == (1, 2, 1.0)
    via = csx.GetPropertiesByName('vias')[0].primitives[0]
    assert via.radius == 0.2 and via.start == [5, 2.5, 0]


def test_xml_round_trip(tmp_path):
    F = openEMS(NrTS=500, EndCriteria=1e-3)
    F.SetGaussExcite(2e9, 1e9)
    F.SetBoundaryCond(['MUR'] * 4 + ['PEC', 'PML_8'])
    csx = ContinuousStructure()
    F.SetCSX(csx)
    g = csx.GetGrid()
    g.SetDeltaUnit(1e-3)
    for a in 'xyz':
        g.SetLines(a, np.linspace(0, 4, 9))
    csx.AddMaterial('sub', epsilon=3.5, kappa=1e-3).AddBox([0, 0, 0], [4, 4, 1], priority=1)
    csx.AddMetal('cu').AddPolygon([np.array([1., 3, 3]), np.array([1., 1, 3])], 'z', 1.0, priority=10)
    csx.AddConductingSheet('gnd', conductivity=5.8e7, thickness=35e-6).AddBox([0, 0, 0], [4, 4, 0], priority=10)
    csx.AddLumpedElement('part', ny='x', caps=False, R=10.0, L=1e-9, C=1e-12, LEtype=1).AddBox([1, 1, 1], [2, 2, 1])
    F.AddLumpedPort(1, 50, [0.5, 2, 0], [0.5, 2, 1], 'z', excite=1, priority=12)
    fn = str(tmp_path / 'rt.xml')
    F.Write2XML(fn)
    G = openEMS()
    G.ReadFromXML(fn)
    assert G.bc == F.bc and G.excitation == F.excitation and (G.nr_ts, G.end_crit) == (500, 1e-3)
    a, b = csx.GetAllProperties(), G.csx.GetAllProperties()
    assert [(p.kind, p.name, len(p.primitives)) for p in a] == [(p.kind, p.name, len(p.primitives)) for p in b]
    part = G.csx.GetPropertiesByName('part')[0]
    assert (part.ny, part.R, part.L, part.C, part.LEtype, part.caps) == (0, 10.0, 1e-9, 1e-12, 1, False)
    poly = G.csx.GetPropertiesByName('cu')[0].primitives[0]
    np.testing.assert_allclose(poly.points, [[1, 3, 3], [1, 1, 3]])
    sheet = G.csx.GetPropertiesByName('gnd')[0]
    assert sheet.conductivity == 5.8e7 and sheet.thickness == 35e-6


def test_timestep_is_the_courant_limit_on_a_uniform_grid():
    csx = ContinuousStructure()
    for a in 'xyz':
        csx.GetGrid().SetLines(a, np.arange(0, 5.01, 0.5))
    csx.GetGrid().SetDeltaUnit(1e-3)
    d = 0.5e-3
    assert timestep(Grid(csx.GetGrid()), 1.0) == pytest.approx(d / (299792458.0 * np.sqrt(3)), rel=1e-9)


def test_operator_metal_edges_and_dielectric_average(tmp_path):
    """A sheet on a mesh line takes the edges inside it; an edge on a dielectric's top face averages the two media;
    a via takes the z edges through it."""
    fn = tmp_path / 'model.xml'
    fn.write_text(OPENEMS_XML)
    F = openEMS()
    F.ReadFromXML(str(fn))
    op = build(F.csx, F.settings())
    nx, ny, nz = op.shape
    flat = lambda a, i, j, k: ((a * nx + i) * ny + j) * nz + k
    # the cu sheet (x 1..9, y 2..3 at z = 1): the x edge at y = 2 from x = 4 to 5 is PEC, the one at y = 1 is not
    assert op.vi[0, 4, 2, 2] == 0 and op.vi[0, 4, 1, 2] > 0
    # the via at (5, 2.5): no line there, so no z edge inside it; the port's z edges carry the 50 ohm
    assert op.vi[2, 1, 2, 0] > 0
    # x edges at z = 1 off the sheet: half FR4, half air -> C between the two
    from gfdtd.operator import EPS0
    g = op.grid
    cap = op.dt / op.vi[0, 4, 4, 2]                                   # no loss in air: vi = dt / C
    A, l = g.dual[1][4] * g.dual[2][2], g.d[0][4]
    er = cap * l / (EPS0 * A)
    w_lo, w_hi = g.d[2][1] / 2, g.d[2][2] / 2                         # quarter cells below (FR4) and above (air)
    assert er == pytest.approx((4.3 * w_lo + 1 * w_hi) / (w_lo + w_hi), rel=0.05)   # the FR4 half is lossy
    assert len(op.exc[0]) == 4                                        # 2 columns (y = 2, 3) x 2 z edges
    assert sorted(p['name'] for p in op.probes) == ['port_it_1', 'port_ut_1']
    assert [(sl['axis'], sl['hi'], sl['n']) for sl in op.pml] == [(1, 0, 4)]   # ymin PML_8: 4 of the 5 cells


def _load_s11(tmp_path, backend='numpy', fill=None, **el):
    """S11 of a port between two small plates, loaded at their far edge by a lumped element (none: open); PEC box.
    fill(csx): adds a material between the plates."""
    F = openEMS(NrTS=20000, EndCriteria=1e-5)
    F.SetGaussExcite(1e9, 1e9)
    F.SetBoundaryCond(['PEC'] * 6)
    csx = ContinuousStructure()
    F.SetCSX(csx)
    g = csx.GetGrid()
    g.SetDeltaUnit(1e-3)
    g.SetLines('x', np.arange(-3, 3.01, 0.5))
    g.SetLines('y', np.arange(-3, 3.01, 0.5))
    g.SetLines('z', np.arange(0, 3.01, 0.5))
    m = csx.AddMetal('plate')
    m.AddBox([-1, -1, 1], [1, 1, 1], priority=10)
    m.AddBox([-1, -1, 2], [1, 1, 2], priority=10)
    p = F.AddLumpedPort(1, 50, [-1, 0, 1], [-1, 0, 2], 'z', excite=1, priority=12)
    if el:
        csx.AddLumpedElement('load', ny='z', caps=True, **el).AddBox([1, 0, 1], [1, 0, 2], priority=12)
    if fill:
        fill(csx).AddBox([-1, -1, 1], [1, 1, 2], priority=1)
    path = str(tmp_path / f"run_{backend}")
    F.Run(path, cleanup=True, verbose=0, backend=backend)
    f = np.array([0.1e9, 0.3e9])
    p.CalcPort(path, f)
    return f, p.uf_ref / p.uf_inc, F.result, path


def _z(s11):
    return 50 * (1 + s11) / (1 - s11)


@pytest.mark.parametrize('el', [dict(R=150.0), dict(R=50.0), dict(R=20.0)])
def test_resistive_load(tmp_path, el):
    f, s11, res, _ = _load_s11(tmp_path, **el)
    assert res['reason'] == 'end'
    assert _z(s11[0]).real == pytest.approx(el['R'], rel=0.02)


def test_reactive_loads(tmp_path):
    """L, C and series RLC loads as the ideal elements plus the fixture's own ~1.4 nH loop and ~0.1 pF plates."""
    w = 2 * np.pi * np.array([0.1e9, 0.3e9])
    f, s, _, _ = _load_s11(tmp_path / 'c', C=1e-12)
    assert np.all((-1 / (w * _z(s).imag) > 1.0e-12) & (-1 / (w * _z(s).imag) < 1.2e-12))
    f, s, _, _ = _load_s11(tmp_path / 'l', L=10e-9)
    assert np.all((_z(s).imag / w > 10e-9) & (_z(s).imag / w < 12e-9))
    f, s, _, _ = _load_s11(tmp_path / 'rlc', R=20.0, L=5e-9, C=2e-12, LEtype=1)
    z = _z(s[1])
    assert z.real == pytest.approx(20, rel=0.15)
    assert z.imag == pytest.approx(w[1] * 5e-9 - 1 / (w[1] * 2e-12), rel=0.1)
    assert np.all(np.abs(s) < 1.005)                                  # passive


@pytest.mark.parametrize('el', [dict(L=10e-9), dict(R=0.0, L=5e-9, C=2e-12, LEtype=1)])
def test_end_criterion_counts_the_lumped_energy(el):
    """The energy the end criterion watches includes the lumped branches' (L I^2 / 2, q^2 / 2C): in a lossless PEC
    box with the plates charged and an L or series LC across them, the field's own energy swings by half as it
    moves into the branch and back, the total is steady, and steadiest with the branch's energy as it is (not
    scaled): the L and C are the element's. A lumped LC filter had ended its run early on the field alone."""
    from gfdtd.engine import make_backend
    F = openEMS(NrTS=100, EndCriteria=1e-5)
    F.SetGaussExcite(1e9, 1e9)
    F.SetBoundaryCond(['PEC'] * 6)
    csx = ContinuousStructure()
    F.SetCSX(csx)
    g = csx.GetGrid()
    g.SetDeltaUnit(1e-3)
    g.SetLines('x', np.arange(-3, 3.01, 0.5))
    g.SetLines('y', np.arange(-3, 3.01, 0.5))
    g.SetLines('z', np.arange(0, 3.01, 0.5))
    m = csx.AddMetal('plate')
    m.AddBox([-1, -1, 1], [1, 1, 1], priority=10)
    m.AddBox([-1, -1, 2], [1, 1, 2], priority=10)
    csx.AddLumpedElement('load', ny='z', caps=True, **el).AddBox([1, 0, 1], [1, 0, 2], priority=12)
    be = make_backend(build(csx, F.settings(), log=lambda *a: None), 'core' if core_info()[0] else 'numpy')
    be.V[2, 4:9, 4:9, 2:4] = 1.0                                      # Ez between the plates
    total, lumped = [], []
    for s in range(4000):
        be.step(None)
        if s % 10 == 0:
            total.append(be.energy())
            lumped.append(be.lumped_energy())
    total, lumped = np.array(total[5:]), np.array(lumped[5:])
    field = total - lumped
    flat = lambda k: (field + k * lumped).min() / (field + k * lumped).max()
    assert field.min() / field.max() < 0.6 and lumped.max() > 0.3 * total.max()
    assert flat(1) > 0.8 and flat(1) > flat(0.75) and flat(1) > flat(1.25)


def test_debye_material_xml_both_ways(tmp_path):
    csx = ContinuousStructure()
    m = csx.AddDebyeMaterial('sub', order=2, epsilon=3.0, kappa=1e-3)
    m.SetDispersiveMaterialProperty(order=0, eps_delta=1.5, eps_relax=30e-12)
    m.SetDispersiveMaterialProperty(order=1, eps_delta=0.5, eps_relax=3e-12)
    m.AddBox([0, 0, 0], [1, 1, 1])
    fn = str(tmp_path / 'm.xml')
    csx.Write2XML(fn)
    text = open(fn).read()
    assert 'EpsilonDelta_2="5.000000e-01' in text and 'EpsilonRelaxTime_1="3.000000e-11' in text   # CSXCAD's names
    back = ContinuousStructure()
    back.ReadFromXML(fn)
    p = back.GetPropertiesByName('sub')[0]
    assert isinstance(p, C.DebyeMaterial) and not back.skipped
    assert p.pole_list() == [(1.5, 30e-12), (0.5, 3e-12)] and p.epsilon == 3.0


def test_debye_material_matches_its_value_at_one_frequency(tmp_path):
    """A capacitor filled with a Debye material and one filled with a plain material of the same complex
    permittivity at 0.3 GHz: the same impedance there, and not at 0.1 GHz."""
    e_inf, d_eps, tau, kap = 2.0, 3.0, 0.5e-9, 1e-3
    f, w = 0.3e9, 2 * np.pi * 0.3e9
    er = e_inf + d_eps / (1 + 1j * w * tau)

    def debye(csx):
        m = csx.AddDebyeMaterial('sub', order=1, epsilon=e_inf, kappa=kap)
        m.SetDispersiveMaterialProperty(order=0, eps_delta=d_eps, eps_relax=tau)
        return m
    plain = lambda csx: csx.AddMaterial('sub', epsilon=er.real, kappa=kap - w * 8.8541878128e-12 * er.imag)
    _, a, ra, _ = _load_s11(tmp_path / 'd', fill=debye)
    _, b, rb, _ = _load_s11(tmp_path / 'p', fill=plain)
    za, zb = _z(a), _z(b)
    assert ra['reason'] == rb['reason'] == 'end'
    assert abs(za[1] - zb[1]) < 0.01 * abs(zb[1])
    assert abs(za[0] - zb[0]) > 0.1 * abs(zb[0])                      # dispersive: other values elsewhere


def _plate_wave(tmp_path, n, bc, backend='numpy', dumps=(), debye=False):
    """A radial wave between two PEC plates (z: 2 cells, 1 mm cells), from a source column at the centre of an n x n
    cell domain; the voltage at 15 mm along x. Returns (t, u, run folder, the openEMS object)."""
    F = openEMS(NrTS=330, EndCriteria=1e-9)
    F.SetGaussExcite(10e9, 10e9)
    F.SetBoundaryCond([bc] * 4 + ['PEC', 'PEC'])
    csx = ContinuousStructure()
    F.SetCSX(csx)
    g = csx.GetGrid()
    g.SetDeltaUnit(1e-3)
    h = n // 2
    g.SetLines('x', np.arange(-h, h + 0.01, 1.0))
    g.SetLines('y', np.arange(-h, h + 0.01, 1.0))
    g.SetLines('z', [0, 1, 2])
    csx.AddExcitation('src', exc_type=0, exc_val=[0, 0, 1]).AddBox([0, 0, 0], [0, 0, 2])
    csx.AddProbe('ut', p_type=0).AddBox([15, 0, 0], [15, 0, 2])
    if debye:
        m = csx.AddDebyeMaterial('slab', order=1, epsilon=2.0)
        m.SetDispersiveMaterialProperty(order=0, eps_delta=1.0, eps_relax=20e-12)
        m.AddBox([-8, -8, 0], [8, 8, 2], priority=1)
    for name, kw, box in dumps:
        csx.AddDump(name, **kw).AddBox(*box)
    path = str(tmp_path / f"{bc}_{n}_{backend}")
    F.Run(path, cleanup=True, verbose=0, backend=backend)
    from gfdtd.fdtd import read_probe
    t, u = read_probe(os.path.join(path, 'ut'))
    return t, u, path, F


def test_pml_absorbs_far_better_than_mur(tmp_path):
    """Against a domain large enough for no reflection to come back in the time: the 8-cell PML's error is a small
    fraction of first-order Mur's."""
    t, ref, _, _ = _plate_wave(tmp_path, 160, 'PEC')
    _, pml, _, F = _plate_wave(tmp_path, 50, 'PML_8')
    _, mur, _, _ = _plate_wave(tmp_path, 50, 'MUR')
    clean = t < 0.14 / 299792458.0                                    # the reference's own wall echo: 80 + 65 mm
    ref, pml, mur = ref[clean], pml[clean], mur[clean]
    peak = np.abs(ref).max()
    e_pml, e_mur = np.abs(pml - ref).max() / peak, np.abs(mur - ref).max() / peak
    assert len(F.op.pml) == 4 and F.op.mur is None
    assert e_pml < 0.01 and e_pml < 0.25 * e_mur, (e_pml, e_mur)


DUMPS = [('Et', dict(dump_type=0, file_type=0, dump_mode=2), ([-20, -20, 1], [20, 20, 1])),
         ('Ef', dict(dump_type=10, file_type=1, dump_mode=2, frequency=[8e9, 12e9]), ([-20, -20, 1], [20, 20, 1])),
         ('Hf', dict(dump_type=11, file_type=1, dump_mode=2, frequency=[8e9]), ([-5, -3, 0], [5, 3, 2])),
         ('Jt', dict(dump_type=3, file_type=0, dump_mode=2), ([0, -10, 0], [0, 10, 2]))]


def test_dumps_in_openems_files(tmp_path):
    """Time-domain dumps as .vtr frames, frequency-domain ones as openEMS's HDF5, both read by kicad2ems's readers,
    on openEMS's cells (box snapped to lines s0..s1: cells s0-1..s1; flat: the cell below); the frequency-domain
    values are the DFT of the time-domain samples."""
    from kicad2ems.solvers.openems.fields import read_vtr
    from kicad2ems.solvers.openems.dumps import read_fd
    import glob
    _, _, path, F = _plate_wave(tmp_path, 50, 'PML_8', dumps=DUMPS)
    frames = sorted(glob.glob(os.path.join(path, 'Et_*.vtr')))
    assert len(frames) == 330 // F.op.sub + 1
    mesh, et = read_vtr(frames[10])
    assert et.shape == (42, 42, 1, 3) and mesh['z'][0] == pytest.approx(0.5) and mesh['x'][[0, -1]] == pytest.approx([-20.5, 20.5])
    mesh, hf = read_fd(os.path.join(path, 'Hf.h5'))
    assert hf[0][1].shape == (12, 8, 3, 3) and mesh['x'][[0, -1]] == pytest.approx([-5.5, 5.5])
    assert len(glob.glob(os.path.join(path, 'Jt_*.vtr'))) == len(frames)
    _, ef = read_fd(os.path.join(path, 'Ef.h5'))
    assert [f for f, _ in ef] == [8e9, 12e9]
    # Ez at one point: the HDF5 value against 2 dt_s sum E(t) exp(-j w t) of the frames
    steps = [int(os.path.basename(f)[3:-4]) for f in frames]
    ez = np.array([read_vtr(f)[1][30, 20, 0, 2] for f in frames])
    t = np.array(steps) * F.op.dt
    want = 2 * F.op.sub * F.op.dt * np.sum(ez * np.exp(-2j * np.pi * 8e9 * t))
    assert ef[0][1][30, 20, 0, 2] == pytest.approx(want, rel=1e-3)
    assert np.abs(want) > 0


def test_port_power_goes_into_the_load(tmp_path):
    """Power balance: what the port measures going in is what the 50 ohm load takes. (The current files' first
    sample once sat at t = 0 instead of -dt/2: DFTs that take dt from t1 - t0 scaled every current by 1 - 0.5 / sub.)"""
    from gfdtd.fdtd import read_probe, dft

    def load_probe(csx):
        csx.AddProbe('uload', p_type=0, weight=-1).AddBox([1, 0, 1], [1, 0, 2])
        return csx.AddMaterial('air', epsilon=1.0)
    f, s11, res, path = _load_s11(tmp_path, R=50.0, fill=load_probe)
    t, i = read_probe(os.path.join(path, 'port_it_1'))
    assert t[0] == pytest.approx(-0.5 * (t[2] - t[1]) / F_SUB(path), rel=1e-6)
    t, u = read_probe(os.path.join(path, 'uload'))
    p_load = 0.5 * np.abs(dft(t, u, f)) ** 2 / 50
    tu, uu = read_probe(os.path.join(path, 'port_ut_1'))
    ti, ii = read_probe(os.path.join(path, 'port_it_1'))
    p_in = 0.5 * np.real(dft(tu, uu, f) * np.conj(dft(ti, ii, f)))
    np.testing.assert_allclose(p_in, p_load, rtol=2e-3)


def F_SUB(path):
    """The probe sampling interval in timesteps of a run folder (et: one sample per step)."""
    from gfdtd.fdtd import read_probe
    t_et, _ = read_probe(os.path.join(path, 'et'))
    t_u, _ = read_probe(os.path.join(path, 'port_ut_1'))
    return round((t_u[1] - t_u[0]) / (t_et[1] - t_et[0]))


def test_compressed_operator_gives_the_dense_coefficients(tmp_path):
    """compress(): vv by code, vi = geoE s, iv = geoH from the line arrays, the Debye poles by row."""
    from gfdtd.operator import compress
    _, _, _, F = _plate_wave(tmp_path, 30, 'PML_8', debye=True)
    op = F.op
    c = compress(op)
    assert c.code.dtype == np.uint8 and c.code.shape == op.shape and c.ncode < 100 and c.P == 1 and c.nslot > 0
    np.testing.assert_allclose(np.moveaxis(c.vv[c.code], -1, 0), op.vv, atol=2e-6)
    g = op.grid
    G = [c.G[i * sum(op.shape):(i + 1) * sum(op.shape)] for i in range(4)]
    ax = lambda arr, a: np.split(arr, np.cumsum(op.shape)[:2])[a]
    r = lambda v, a: v.reshape([-1 if x == a else 1 for x in range(3)])
    for a in range(3):
        b, d = (a + 1) % 3, (a + 2) % 3
        geoE = c.ke * r(ax(G[0], a), a) * r(ax(G[3], b), b) * r(ax(G[3], d), d)
        geoH = c.kh * r(ax(G[2], a), a) * r(ax(G[1], b), b) * r(ax(G[1], d), d)
        np.testing.assert_allclose(geoE * c.s[c.code, a], op.vi[a], rtol=2e-6, atol=1e-30)
        np.testing.assert_allclose(geoH, op.iv[a], rtol=2e-6, atol=1e-30)


@pytest.mark.skipif(not core_info()[0], reason='no compiled CPU core (python -m gfdtd.core.build)')
def test_core_matches_numpy_with_pml_debye_and_dumps(tmp_path):
    from kicad2ems.solvers.openems.dumps import read_fd
    from gfdtd.engine import resolve_backend, core_threads
    assert resolve_backend('cpu') == 'core' and 1 <= core_threads() <= (os.cpu_count() or 1)
    for bc in ('PML_8', 'MUR'):
        _, a, pa, _ = _plate_wave(tmp_path / 'n', 50, bc, 'numpy', dumps=DUMPS[1:2], debye=True)
        _, b, pb, _ = _plate_wave(tmp_path / 'c', 50, bc, 'core', dumps=DUMPS[1:2], debye=True)
        np.testing.assert_allclose(a, b, atol=1e-5 * np.abs(a).max())
        fa, fb = read_fd(os.path.join(pa, 'Ef.h5'))[1][0][1], read_fd(os.path.join(pb, 'Ef.h5'))[1][0][1]
        np.testing.assert_allclose(fa, fb, atol=1e-4 * np.abs(fa).max())


def _dipole(tmp_path, mirror=False):
    """A z dipole (or, mirror: a monopole on a PEC ground at z = 0) for 3 GHz, gap fed, PML; the NF2FF box placed
    by CreateNF2FFBox. Returns (nf2ff results, P_acc at the recorded frequency, the NF2FF box)."""
    F = openEMS(NrTS=20000, EndCriteria=1e-5)
    F.SetGaussExcite(3e9, 2e9)
    F.SetBoundaryCond(['PML_8'] * 4 + (['PEC', 'PML_8'] if mirror else ['PML_8'] * 2))
    csx = ContinuousStructure()
    F.SetCSX(csx)
    g = csx.GetGrid()
    g.SetDeltaUnit(1e-3)
    g.SetLines('x', np.arange(-40, 40.01, 2.0))
    g.SetLines('y', np.arange(-40, 40.01, 2.0))
    g.SetLines('z', np.arange(0 if mirror else -50, 50.01, 1.0))
    m = csx.AddMetal('arm')
    m.AddBox([0, 0, 1], [0, 0, 23], priority=10)
    if not mirror:
        m.AddBox([0, 0, -23], [0, 0, -1], priority=10)
    port = F.AddLumpedPort(1, 36.5 if mirror else 73, [0, 0, 0 if mirror else -1], [0, 0, 1], 'z', excite=1,
                           priority=5)
    nf = F.CreateNF2FFBox(frequency=[3e9])
    path = str(tmp_path / ('mono' if mirror else 'dipole'))
    F.Run(path, cleanup=True, verbose=0, backend='cpu')
    port.CalcPort(path, [3e9])
    r = nf.CalcNF2FF(path, [3e9], np.arange(0, 181, 5.0), np.arange(0, 361, 5.0))
    return r, float(np.atleast_1d(port.P_acc)[0]), nf


def test_nf2ff_box_as_openems_places_it(tmp_path):
    F = openEMS()
    F.SetBoundaryCond(['PML_8', 'MUR', 'PEC', 'PMC', 'PML_4', 'PML_4'])
    csx = ContinuousStructure()
    F.SetCSX(csx)
    g = csx.GetGrid()
    for a in 'xyz':
        g.SetLines(a, np.arange(0, 21.0))
    nf = F.CreateNF2FFBox(frequency=[1e9])
    assert list(nf.start) == [9, 0, 5] and list(nf.stop) == [18, 20, 15]     # PML_n: n + 1 lines in, Mur 2
    assert nf.directions == [True, True, False, False, True, True] and nf.mirror == [0, 0, 1, 2, 0, 0]
    assert len(nf.e_dump.primitives) == 4 and nf.e_dump.dump_type == 10 and nf.h_dump.dump_mode == 1


def test_dipole_far_field(tmp_path):
    """A half-wave dipole: 2.15 dBi, all of the accepted power radiated (Prad: the flux through the box), the
    pattern a doughnut about z; as a monopole on a PEC mirror: 3 dB more, the same efficiency."""
    r, p_acc, _ = _dipole(tmp_path)
    assert 10 * np.log10(r.Dmax[0]) == pytest.approx(2.15, abs=0.15)
    assert r.Prad[0] / p_acc == pytest.approx(1.0, abs=0.02)
    e = r.E_norm[0]                                                   # (theta, phi)
    assert e[0].max() < 0.05 * e.max() and np.ptp(e[18]) < 0.02 * e.max()   # no field along z; round at theta 90
    rm, pm, _ = _dipole(tmp_path, mirror=True)
    assert 10 * np.log10(rm.Dmax[0]) == pytest.approx(5.15, abs=0.25)
    assert rm.Prad[0] / pm == pytest.approx(1.0, abs=0.03)


def test_probe_files_are_openems_format(tmp_path):
    from kicad2ems.solvers.openems.simulation import steps_taken
    _, _, res, path = _load_s11(tmp_path, R=50.0)
    names = sorted(os.listdir(path))
    assert {'et', 'port_ut_1', 'port_it_1'} <= set(names)
    head = open(os.path.join(path, 'port_it_1')).read().splitlines()[:4]
    assert head[3].split() == ['%', 't/s', 'current']
    n = steps_taken(path)                                             # kicad2ems's reader of openEMS's files
    assert n is not None and abs(n - res['steps']) <= 200


def test_abort_file_ends_a_run(tmp_path):
    path = tmp_path / 'run_numpy'
    path.mkdir()
    (path / 'ABORT').write_text('')
    _, _, res, _ = _load_s11(tmp_path, R=50.0)
    assert res['reason'] == 'abort' and res['steps'] == 100


@pytest.mark.skipif(not gpu_info()[0], reason='no GPU for CuPy')
def test_gpu_matches_numpy(tmp_path):
    _, a, _, pa = _load_s11(tmp_path / 'n', backend='numpy', R=20.0, L=5e-9, C=2e-12, LEtype=1)
    _, b, _, pb = _load_s11(tmp_path / 'g', backend='cupy', R=20.0, L=5e-9, C=2e-12, LEtype=1)
    np.testing.assert_allclose(a, b, atol=1e-4)
    from gfdtd.fdtd import read_probe
    np.testing.assert_allclose(read_probe(os.path.join(pa, 'port_ut_1'))[1], read_probe(os.path.join(pb, 'port_ut_1'))[1],
                               atol=1e-6 * np.abs(read_probe(os.path.join(pa, 'port_ut_1'))[1]).max() + 1e-12)


NODE_DUMPS = [('Hn', dict(dump_type=11, file_type=1, dump_mode=1, frequency=[8e9]), ([-6, -5, 0], [6, 5, 2])),
              ('En', dict(dump_type=10, file_type=1, dump_mode=0, frequency=[8e9]), ([-25, -25, 0], [25, 25, 2])),
              ('Jn', dict(dump_type=13, file_type=1, dump_mode=1, frequency=[8e9]), ([-6, -6, 1], [6, 6, 1]))]


@pytest.mark.skipif(not gpu_info()[0], reason='no GPU for CuPy')
@pytest.mark.parametrize('mode', ['fused', 'split', 'separate'])
def test_gpu_matches_numpy_with_pml_debye_and_dumps(tmp_path, monkeypatch, mode):
    """The GPU's three ways of running a step (PML in the H / E kernels, as its own launches, every extension
    apart), and its dump kernel (every field and dump mode), against numpy."""
    import glob
    from kicad2ems.solvers.openems.dumps import read_fd
    from kicad2ems.solvers.openems.fields import read_vtr
    monkeypatch.setenv('GFDTD_STEP', mode)
    _, a, pa, _ = _plate_wave(tmp_path, 50, 'PML_8', 'numpy', dumps=DUMPS + NODE_DUMPS, debye=True)
    _, b, pb, F = _plate_wave(tmp_path, 50, 'PML_8', 'cupy', dumps=DUMPS + NODE_DUMPS, debye=True)
    np.testing.assert_allclose(a, b, atol=1e-5 * np.abs(a).max())
    for f in glob.glob(os.path.join(pa, '*.h5')):
        fa, fb = read_fd(f)[1][0][1], read_fd(f.replace(pa, pb))[1][0][1]
        np.testing.assert_allclose(fa, fb, atol=1e-5 * np.abs(fa).max())
    for name in ('Et', 'Jt'):
        fs = sorted(glob.glob(os.path.join(pa, f'{name}_*.vtr')))
        A = np.array([read_vtr(x)[1] for x in fs])
        B = np.array([read_vtr(x.replace(pa, pb))[1] for x in fs])
        np.testing.assert_allclose(A, B, atol=1e-5 * np.abs(A).max())


@pytest.mark.skipif(not gpu_info()[0], reason='no GPU for CuPy')
def test_gpu_runs_in_threads(tmp_path):
    """Runs in threads (a GPU stream each, the output in a log file each) give what they give one at a time."""
    import threading
    from gfdtd.engine import concurrent_runs
    assert concurrent_runs(0.29e6, 4) == 4 and concurrent_runs(0.59e6, 4) == 2 and concurrent_runs(2e6, 4) == 1
    _, ref, _, _ = _plate_wave(tmp_path / 'one', 50, 'PML_8', 'cupy', debye=True)
    got = {}

    def run(n):
        got[n] = _plate_wave(tmp_path / f't{n}', 50, 'PML_8', 'cupy', debye=True)[1]
    th = [threading.Thread(target=run, args=(n,)) for n in range(3)]
    [t.start() for t in th]
    [t.join() for t in th]
    for n in range(3):
        np.testing.assert_allclose(got[n], ref, atol=1e-6 * np.abs(ref).max())


def test_run_writes_its_log(tmp_path, capsys):
    """Run(log=path): the output into the file (for runs in threads), none on stdout."""
    fn = tmp_path / 'model.xml'
    fn.write_text(OPENEMS_XML)
    F = openEMS()
    F.ReadFromXML(str(fn))
    log = tmp_path / 'run.log'
    capsys.readouterr()
    F.Run(str(tmp_path / 'logged'), log=str(log), backend='numpy')
    assert capsys.readouterr().out == ''
    text = log.read_text()
    assert 'FDTD timestep is' in text and 'Time for ' in text and 'iterations' in text


def test_command_line_runs_an_xml(tmp_path, capsys):
    from gfdtd.fdtd import main
    fn = tmp_path / 'model.xml'
    fn.write_text(OPENEMS_XML)
    assert main([str(fn), '--backend', 'numpy', '--path', str(tmp_path / 'run')]) == 0
    out = capsys.readouterr().out
    assert 'note: RotPoly odd' in out and 'Timestep' in out
    assert os.path.exists(tmp_path / 'run' / 'port_ut_1')


# ---- the kicad2ems plugin ----------------------------------------------------------------------------------------
def test_plugin_registered():
    from kicad2ems.solvers import registry, get
    sv = registry()['gfdtd']
    assert get('gpu-fdtd') is sv and not sv.experimental and not sv.in_auto and not sv.in_all
    assert sv.check()[0]                                              # numpy at worst
    assert sv.capabilities['antenna'].startswith('yes')
    assert [o.key for o in sv.options] == ['gfdtd_backend']


def test_engine_api_and_run_dir():
    from kicad2ems.solvers.openems.simulation import engine_api, run_dir
    import gfdtd
    assert engine_api(None) is None and engine_api('openems') is None
    cs, fd = engine_api('gfdtd:numpy')
    assert cs is gfdtd.ContinuousStructure and fd().backend == 'numpy'

    class P:
        name = 'P1+'

    class M:
        ports = [P()]
    assert run_dir(M, 0, '/o').endswith(os.path.join('openems', 'exc_P1p'))
    assert run_dir(M, 0, '/o', 'gfdtd:cupy').endswith(os.path.join('gfdtd', 'exc_P1p'))


def test_build_fdtd_builds_the_board_on_gfdtd(demo_files):
    """kicad2ems's openEMS model, built through gfdtd's API, lowers to an operator: ports, copper, vias."""
    from kicad2ems.importers.kicad import KiCadBoard
    from kicad2ems.board import Board
    from kicad2ems.solvers.openems.boundaries import resolve_bc
    from kicad2ems.solvers.openems.simulation import build_fdtd, engine_api
    from kicad2ems.solvers.openems.mesh import auto_mesh
    base, _ = demo_files
    m = Board(KiCadBoard(base + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=base + '-top.pos', ground='B.Cu')
    bc = resolve_bc('auto', m)
    mesh = auto_mesh(m, 1e9, 8e9, bc=bc, res=10, log=lambda *a: None)
    FDTD, CSX, ports = build_fdtd(m, mesh, 1e9, 8e9, 0, bc=bc, api=engine_api('gfdtd:numpy'))
    op = build(CSX, FDTD.settings())
    assert len(ports) == len(m.ports) == 2 and len(op.exc[0]) > 0
    assert op.n_pec[2] > 0                                            # the vias' z edges
    assert len(op.probes) == 4 and op.bc == bc
