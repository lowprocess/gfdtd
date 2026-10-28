"""Build the optional compiled CPU core gfdtd._core:  python -m gfdtd.core.build   (needs g++ with OpenMP and
pybind11, as mom25d's core). Without it the CPU runs on numpy (some 15x slower); the GPU does not need it."""
from emdev.build import build_core

build_core(__file__, 'gfdtd_core.cpp', arch_env='GFDTD_ARCH', includes_here=True)
