"""Build the optional compiled CPU core gfdtd._core:  python -m gfdtd.core.build   (needs pybind11 and g++ with
OpenMP, or on Windows MSVC or MinGW-w64: emsdev.build, as mom25d's core). Without it the CPU runs on numpy (some 15x slower); the GPU does not need it."""
from emsdev.build import build_core

build_core(__file__, 'gfdtd_core.cpp', arch_env='GFDTD_ARCH', includes_here=True)
