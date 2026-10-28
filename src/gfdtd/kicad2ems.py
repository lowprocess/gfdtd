"""
gfdtd as a kicad2ems solver plugin and FDTD engine (the entry points of gfdtd's pyproject): kicad2ems's FDTD model
(kicad2ems.fdtd.plugin) solved by gfdtd's engine (GPU via CuPy, its compiled
CPU core, or numpy) through its CSXCAD / openEMS interface. The board, the mesh (--res, --fine, --nz, ...), the
ports, the parts and the end criteria are the FDTD settings shared with openEMS; only the engine differs. Runs go
to <out>/gfdtd/exc_<port>, the CSX file to <name>_gfdtd.xml (also runnable with `python -m gfdtd`), the antenna
results to <name>_gfdtd_antenna.json.

Not yet: the --converge levels, conductor loss (conducting sheets are PEC), phased drives (--drive).
"""
import os

from kicad2ems.api.solver import Option, AUTO
from kicad2ems.api.solver import FDTD, MESH, RUN

G = 'gfdtd'                                     # its section in kicad2ems's GUI
# gfdtd's speed over openEMS's estimate (24 threads), per backend: the examples' runs on an RX 7800 XT and an
# i7-13700K, 8-70x on the GPU and 2-2.3x on the core (docs/gfdtd.md#speed); numpy ~10x slower than openEMS.
# START: seconds of setup a run (CuPy's kernels, the core's threads)
SPEEDUP, START = dict(cupy=8.0, core=2.0, numpy=0.1), 2.0


class GFDTD(FDTD):
    api = (1, 0)                    # kicad2ems.api
    name, aliases, label = 'gfdtd', ('gpu-fdtd',), 'gfdtd (GPU FDTD)'
    in_auto = True                  # auto: picked over openEMS where it can (no --sigma, --converge, --drive)
    in_all = False                  # --solvers all: openEMS's model again, named on its own
    previewed = False
    order = 40                      # after openEMS and mom25d in kicad2ems's lists
    tag, conductor_loss = '_gfdtd', False
    capabilities = dict(FDTD.capabilities,
                        conductor_loss='ignored: copper is PEC (conducting sheets as PEC)',
                        dispersion='yes: fitted at mid-band, or wideband Debye',
                        antenna='yes: one driven port (no phased drive)')
    options = MESH + RUN + [
        Option('gfdtd_backend', 'auto', "gfdtd: where it runs - auto (the GPU when CuPy finds one, else the CPU), gpu "
               "(CuPy: CUDA or ROCm) or cpu (the compiled core: python -m gfdtd.core.build; else numpy, slow); its "
               "mesh and run settings are the FDTD ones shared with openEMS (--res, --end-db, ...)", choices=('auto', 'gpu', 'cpu'), name='backend', gui='Runs on', group=G),
    ]

    def check(self, s=None):
        from gfdtd.engine import gpu_info, core_info
        gpu, what = gpu_info()
        want = getattr(s, 'gfdtd_backend', 'auto') if s is not None else 'auto'
        if want == 'gpu' and not gpu:
            return False, f"gfdtd_backend = gpu: {what}"
        if gpu and want != 'cpu':
            return True, f"GPU: {what}"
        core, cwhat = core_info()
        return True, f"CPU: {cwhat if core else 'numpy (slow; build the core: python -m gfdtd.core.build)'}" + (
            '' if gpu else f"; {what}")

    def resources(self):
        """The GPU gfdtd's engine would run on (CuPy's first device), or why there is none."""
        from emdev import gpu
        try:
            return dict(gpus=[gpu.cupy_device(0).as_dict()], why='')
        except gpu.Unavailable as e:
            return dict(gpus=[], why=str(e))

    def details(self):
        ok, what = self.check()
        if not ok:
            return ok, [what]
        import gfdtd
        return ok, [f"gfdtd {gfdtd.__version__} ({os.path.dirname(gfdtd.__file__)})", what]

    def validate(self, s):
        return None

    def assess(self, run):
        """Named, gfdtd runs without what it lacks (with a note, prepare); for auto, those runs are openEMS's."""
        if AUTO not in run.plan:
            return []
        s = run.s
        return [f"{what}: openEMS's" for what, on in (("--sigma (gfdtd's copper is PEC)", s.sigma),
                                                    ("--converge", getattr(s, "converge", None)), ("--drive", s.drive)) if on]

    def estimate(self, run):
        """openEMS's estimate (the same mesh and timesteps), at gfdtd's speed on the backend it would use."""
        from gfdtd.engine import resolve_backend
        e = super().estimate(run)
        where = resolve_backend(self.engine(run.s).partition(':')[2])
        return dict(e, seconds=e['seconds'] / SPEEDUP[where] + START,
                    what=e['what'] + {'cupy': ' on the GPU', 'core': ' on the CPU core'}.get(where, ' in numpy'))

    def jobs(self, run, p):
        """Runs at once: --jobs when given; on the GPU as many as its cache holds (engine.concurrent_runs), on the
        CPU one (the core's threads already share the cores)."""
        from gfdtd.engine import concurrent_runs, resolve_backend
        s = run.s
        if s.jobs > 1:
            return s.jobs
        if resolve_backend(self.engine(s).partition(':')[2]) != 'cupy':
            return 1
        return concurrent_runs(p.grid['info']['cells'], len(p.excite))

    def engine(self, s):
        return 'gfdtd:' + {'gpu': 'cupy'}.get(s.gfdtd_backend, s.gfdtd_backend)       # cpu: the core, else numpy

    def about(self, s):
        return f"gfdtd, 3D FDTD ({s.gfdtd_backend})"

    def prepare(self, run):
        p = super().prepare(run)
        s = run.s
        for what, on in (("--converge", getattr(s, 'converge', None)), ("--sigma (copper is PEC)", s.sigma)):
            if on:
                run.log(f"gfdtd: {what} not supported yet - ignored")
        self.summary(run)['backend'] = s.gfdtd_backend
        run.log(f"gfdtd: {p.grid['info']['cells'] / 1e6:.2f} M cells, boundaries {','.join(p.bc)}, {self.check(s)[1]}")
        return p


LABEL = 'gfdtd'                     # the engine's name in kicad2ems's log
IN_THREADS = True                   # one process: the runs share the GPU (a stream each)


def api(backend=''):
    """(ContinuousStructure, openEMS) of gfdtd's engine (kicad2ems.fdtd.engines) on a backend: cupy, core, numpy or
    auto."""
    import functools
    import gfdtd
    return gfdtd.ContinuousStructure, functools.partial(gfdtd.openEMS, backend=backend or 'auto')
