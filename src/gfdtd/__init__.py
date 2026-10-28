"""
gfdtd: an FDTD solver in the manner of openEMS, built to run on a GPU (CuPy: NVIDIA CUDA or AMD ROCm), with
numpy on the CPU as the fallback and the reference. Models come in through CSXCAD's Python API or its XML
files, so a model made for openEMS runs here unchanged:

    from gfdtd import ContinuousStructure, openEMS          # instead of CSXCAD's / openEMS's
    python -m gfdtd model.xml                                # an openEMS XML file, as `openEMS model.xml`

What it models (the MVP): rectilinear grids; isotropic dielectrics with conductivity (kappa); metals and
conducting sheets as PEC; lumped elements (R, L, C, parallel or series, with caps); openEMS's ports (lumped, MSL, stripline, CPW, coaxial,
rectangular / circular / any waveguide, curve); a Gaussian excitation, soft or hard, E or H, weighted and delayed;
PEC, PMC, Mur and PML boundaries; Debye materials; voltage, current and mode-matching probes; field dumps; NF2FF; the
energy end criterion; ABORT files.

    csx.py       CSXCAD's structure: grid, properties, primitives, XML
    fdtd.py      openEMS's interface: settings, LumpedPort, Run, XML; the command line
    ports.py     openEMS's other ports: MSL, stripline, CPW, coax, waveguides, curve
    operator.py  the model on the grid: coefficients, lumped elements, excitations, boundaries, probes
    engine.py    the time loop, the numpy and CuPy backends, status lines and probe files
    kernels.py   the GPU kernels

API_VERSION is (major, minor), the version of what another package may rely on (kicad2ems-fdtd, which runs
kicad2ems's models here): a minor release adds to it, a major one removes or changes something in it.

    ContinuousStructure, openEMS(..., backend=)   openEMS's / CSXCAD's Python interface, as openEMS scripts use it
    engine.gpu_info(), engine.core_info()        (usable, one line) of the GPU and the compiled CPU core
    engine.resolve_backend(name)                 'cupy', 'core' or 'numpy' for 'auto', 'gpu', 'cpu', ...
    engine.concurrent_runs(cells, runs)          how many runs to have on the GPU at once
    __version__
"""
__version__ = '261028'
API_VERSION = (1, 0)

from .csx import ContinuousStructure                                # noqa: E402,F401
from .fdtd import openEMS, LumpedPort                               # noqa: E402,F401
from .ports import (MSLPort, StripLinePort, CPWPort, CoaxialPort, WaveguidePort, RectWGPort, CircWGPort,  # noqa: E402,F401
                    CurvePort)
