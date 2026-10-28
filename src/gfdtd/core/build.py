"""Build the optional compiled CPU core gfdtd._core:  python -m gfdtd.core.build   (needs g++ with OpenMP and
pybind11, as mom25d's core). Without it the CPU runs on numpy (some 15x slower); the GPU does not need it."""
import os, subprocess, sysconfig
import pybind11
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, '..', '_core' + sysconfig.get_config_var('EXT_SUFFIX'))   # next to the package modules
arch = os.environ.get('GFDTD_ARCH', '-march=native')
cmd = ['g++', '-O3', arch, '-fopenmp', '-shared', '-fPIC', '-std=c++17', '-Wno-attributes',
       '-I' + pybind11.get_include(), '-I' + sysconfig.get_paths()['include'], '-I' + here,
       os.path.join(here, 'gfdtd_core.cpp'), '-o', out]
print(' '.join(cmd)); subprocess.check_call(cmd); print('built', os.path.abspath(out))
