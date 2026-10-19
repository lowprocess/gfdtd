"""
gfdtd: an FDTD solver in the manner of openEMS, built to run on a GPU (CuPy: NVIDIA CUDA or AMD ROCm), with
numpy on the CPU as the fallback and the reference. Models come in through CSXCAD's Python API or its XML
files, so a model made for openEMS runs here unchanged:

    from gfdtd import ContinuousStructure, openEMS          # instead of CSXCAD's / openEMS's
    python -m gfdtd model.xml                                # an openEMS XML file, as `openEMS model.xml`

What it models (the MVP): rectilinear grids; isotropic dielectrics with conductivity (kappa); metals and
conducting sheets as PEC; lumped elements (R, L, C, parallel or series, with caps); lumped ports; a Gaussian
excitation; PEC, PMC and first-order Mur boundaries (PML_n runs as Mur); voltage and current probes; the
energy end criterion; ABORT files. Not yet: PML, dispersive materials, field dumps, NF2FF, other ports.

    csx.py       CSXCAD's structure: grid, properties, primitives, XML
    fdtd.py      openEMS's interface: settings, LumpedPort, Run, XML; the command line
    operator.py  the model on the grid: coefficients, lumped elements, excitations, boundaries, probes
    engine.py    the time loop, the numpy and CuPy backends, status lines and probe files
    kernels.py   the GPU kernels
"""
__version__ = '261019-r2'

from .csx import ContinuousStructure                                # noqa: E402,F401
from .fdtd import openEMS, LumpedPort                               # noqa: E402,F401
