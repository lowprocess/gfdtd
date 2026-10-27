"""The kicad2ems plugin (gfdtd.kicad2ems): registered, its engine API, and kicad2ems's FDTD model built through
gfdtd's API. Needs a kicad2ems checkout (conftest.py)."""
import os

import pytest

import gfdtd
from gfdtd.operator import build


def test_plugin_registered():
    from kicad2ems.solvers import registry, get
    sv = registry()['gfdtd']
    assert get('gpu-fdtd') is sv and sv.in_auto and not sv.in_all
    assert sv.check()[0]                                              # numpy at worst
    assert sv.capabilities['antenna'].startswith('yes')
    assert sv.options[-1].key == 'gfdtd_backend'                    # its own, after the shared FDTD settings
    shared = {o.key: o for o in registry()['openems'].options}
    assert all(shared[o.key] is o for o in sv.options[:-1])


def test_engine_api_and_run_dir():
    from kicad2ems.fdtd.engines import engine_api, engine_name
    from kicad2ems.fdtd.run import run_dir
    assert engine_name(None) == ('openems', 'openEMS') and engine_name('gfdtd:cupy') == ('gfdtd', 'gfdtd')
    with pytest.raises(ValueError, match='unknown FDTD engine'):
        engine_api('hfss')
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
    from kicad2ems.fdtd.boundaries import resolve_bc
    from kicad2ems.fdtd.model import build_fdtd
    from kicad2ems.fdtd.engines import engine_api
    from kicad2ems.fdtd.mesh import auto_mesh
    base, _ = demo_files
    m = Board(KiCadBoard(base + '.kicad_pcb'), nets=['RF', 'GND'], pos_file=base + '-top.pos', ground='B.Cu')
    bc = resolve_bc('auto', m)
    mesh = auto_mesh(m, 1e9, 8e9, bc=bc, res=10, log=lambda *a: None)
    FDTD, CSX, ports = build_fdtd(m, mesh, 1e9, 8e9, 0, bc=bc, api=engine_api('gfdtd:numpy'))
    op = build(CSX, FDTD.settings())
    assert len(ports) == len(m.ports) == 2 and len(op.exc[0]) > 0
    assert op.n_pec[2] > 0                                            # the vias' z edges
    assert len(op.probes) == 4 and op.bc == bc
