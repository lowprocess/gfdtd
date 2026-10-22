# gfdtd (GPU FDTD)

[← Index](index.md)

gfdtd is a 3D FDTD solver written for kicad2ems, in the manner of openEMS, that runs on a GPU, or on the CPU
through a compiled core. It is bundled in this repository (the `gfdtd` package, beside `mom25d`). Name it
(`--solvers gfdtd`) or tick it under Solvers in the GUI, where its section (Runs on: auto, gpu, cpu) follows
openEMS's: it uses openEMS's mesh and run settings. `--solvers auto` picks it over openEMS whenever it can run:
not with `--sigma`, `--converge` or `--drive`, which it would ignore (see [Which solver](usage.md#which)). In
the GUI those settings are in the *openEMS + gfdtd (FDTD)* section. A preview with gfdtd and without openEMS
draws the mesh (`<name>_mesh.png`, and the grid in the Viewer's 3D view).

It gets **openEMS's model**: kicad2ems builds the CSXCAD geometry with the same code (`build_fdtd`), through
gfdtd's copy of CSXCAD's and openEMS's Python interface. The mesh, boundaries, ports, parts, excitation and
end criteria are openEMS's and take openEMS's settings (`--res`, `--fine`, `--nz`, `--bc`, `--end-db`,
`--end-on`, `--max-steps`, `--excite`, `--diel`, `--dump`, `--dump-port`, antenna mode's `--ff-freq`,
`--ff-step`). Only the engine differs. On the examples below the two agree to within 0.004–0.05 in |S|. On one
mid-range GPU gfdtd runs 8–20 times faster than openEMS on a 24-thread CPU; its CPU core, on that same CPU, about
twice as fast, and five times with PML and wideband dielectrics (see [Speed](#speed)).

```bash
kicad2ems examples/rf_coupler.toml --solvers gfdtd              # the GPU when CuPy finds one, else the CPU
kicad2ems examples/rf_coupler.toml --solvers openems,gfdtd      # both, compared in the report
kicad2ems board.toml --solvers gfdtd --gfdtd-backend cpu        # the CPU core (numpy without it: slow)
kicad2ems board.toml --solvers gfdtd --bc PML_8 --diel wideband --dump "E@F.Cu-0.05@2,4"
kicad2ems examples/antenna_dipole.toml --solvers gfdtd          # antenna mode: gain, efficiency, patterns
```

| Setting | Meaning |
|---|---|
| `--gfdtd-backend auto\|gpu\|cpu` (`[solver.gfdtd] backend`) | `auto` (default): the GPU when CuPy finds one, else the CPU. `gpu`: CuPy (CUDA or ROCm), or the run stops. `cpu`: the compiled core, else numpy |

The GPU needs CuPy for your stack: one of the `gpu-*` extras (`pip install '.[gpu-rocm7]'`), as for mom25d
([GPU](mom.md#gpu)). The CPU core is built once, as mom25d's (g++ with OpenMP, and pybind11: the `core`
extra):

```bash
python -m gfdtd.core.build          # gfdtd/_core*.so; GFDTD_ARCH=-march=x86-64-v3 for a portable build
```

Without it the CPU runs on numpy, some 15 times slower. `kicad2ems --check` shows which one gfdtd will use.
`GFDTD_BACKEND=cupy|core|numpy` overrides the choice for every run.

## CSXCAD and openEMS compatibility

gfdtd reads and writes openEMS's XML files and has the part of the CSXCAD / openEMS Python interface that
openEMS scripts use, so a model built for openEMS runs on gfdtd too:

```python
from gfdtd import ContinuousStructure, openEMS        # in place of CSXCAD's and openEMS's
FDTD = openEMS(NrTS=100000, EndCriteria=1e-4)
FDTD.SetGaussExcite(f0, fc); FDTD.SetBoundaryCond(['MUR'] * 6)
CSX = ContinuousStructure(); FDTD.SetCSX(CSX)
...                                                   # AddMaterial, AddMetal, AddLumpedElement, AddPolygon, ...
port = FDTD.AddLumpedPort(1, 50, start, stop, 'z', excite=1)
nf = FDTD.CreateNF2FFBox(frequency=[2.4e9])          # optional: the far field (antennas)
FDTD.Run(sim_path)                                    # backend='auto' | 'cupy' | 'core' | 'numpy'
port.CalcPort(sim_path, freq)                         # uf_inc, uf_ref, if_tot, ... as openEMS's
ff = nf.CalcNF2FF(sim_path, [2.4e9], theta, phi)      # ff.Dmax, ff.Prad, ff.E_norm, ... as openEMS's
```

```bash
gfdtd model.xml --path run/        # an openEMS XML file (FDTD.Write2XML), as `openEMS model.xml`
```

A run writes what openEMS writes, in openEMS's formats: its status lines (`[@ 4s] Timestep: ... || Speed: ...
|| Energy: ...`), the probe files `port_ut_N` / `port_it_N`, the excitation `et` and the field dumps
(`<name>_<step>.vtr` frames, `<name>.h5`). A file `ABORT` in the
run folder ends a run. So kicad2ems's progress lines, port watch (`--end-on ports`), Stop and S-parameter
code work on gfdtd unchanged. Each kicad2ems run also writes `<name>_gfdtd.xml`, the model as an openEMS XML
file that both `gfdtd` and `openEMS` can run.

## What it models

| | gfdtd |
|---|---|
| Grid | rectilinear, Cartesian |
| Materials | isotropic epsilon and conductivity (kappa), averaged over the four cells around each edge |
| Dispersion | Debye materials (`DebyeMaterial`, any number of poles): `--diel wideband` |
| Metal | `Metal` and `ConductingSheet`, both as PEC: no conductor loss (`--sigma` is ignored) |
| Primitives | Box, Polygon, LinPoly, Cylinder, CylindricalShell, Curve (a wire along the mesh edges nearest to it, as openEMS's) |
| Lumped elements | R, L, C: parallel, or series RLC (`LEtype=1`), with metal caps |
| Ports | all of openEMS's Python ports: lumped, curve, microstrip (MSL), stripline, CPW, coaxial, rectangular and circular waveguide, any waveguide mode by functions or mode file ([Ports](#ports)) |
| Excitations | soft and hard, E and H field (types 0-3), weighted by functions of x, y, z, rho, a, r, t or a mode file, delayed |
| Probes | voltage, current, voltage and current mode matching (types 10, 11) |
| Excitation | Gaussian pulse (openEMS's) |
| Boundaries | PEC, PMC, first-order Mur, PML_n (convolutional PML over the n outermost cells, PEC behind it) |
| Field dumps | `DumpBox` E, H, J (rot H), Jc (kappa E); time domain (.vtr) and frequency domain (HDF5); at the cell centres (dump mode 2) or the nodes (modes 0, 1); one file per box |
| Far field | openEMS's NF2FF box (`CreateNF2FFBox`, `CalcNF2FF`), frequency domain, PEC / PMC mirrors: antenna mode |
| End | the energy in the field and in the lumped parts (`EndCriteria`), timestep limit, `ABORT` |

Not yet: conductor loss (`--sigma`: conducting sheets run as PEC), Lorentz materials, phased drives in antenna
mode (`--drive`: the first port alone), time-domain NF2FF, `--converge`, the mode-purity column of
mode-matching probes, field probes (types 2, 3), and cylindrical grids. A file that holds these is still read:
what gfdtd doesn't model is listed as a note.

<a id="ports"></a>
## Ports

`gfdtd.ports` has openEMS's Python port classes, made as openEMS makes them (the same metal, probes and
excitations, the same post-processing), and `openEMS` has their `Add...Port` methods:

```python
port = FDTD.AddMSLPort(1, metal, start, stop, 'x', 'z', excite=-1, FeedShift=10, MeasPlaneShift=30)
port = FDTD.AddStripLinePort(1, metal, start, stop, 'x', 'z', height, excite=1)
port = FDTD.AddCPWPort(1, metal, start, stop, 'x', 'y', gap, excite=1)
port = FDTD.AddCoaxialPort(1, pec, fill, start, stop, 'z', r_i, r_o, r_os, excite=1)
port = FDTD.AddRectWaveGuidePort(0, start, stop, 'z', a, b, 'TE10', excite=1)
port = FDTD.AddCircWaveGuidePort(0, start, stop, 'z', radius, 'TE11', excite=1)
port = FDTD.AddWaveGuidePort(0, start, stop, 'z', E_func, H_func, kc, excite=1)   # or E_file / H_file
port = FDTD.AddCurvePort(1, 50, start, stop, excite=1)
```

The transmission-line ports take the line's impedance and propagation constant from three voltage and two current
probes along it; the waveguide ports match the field on the port plane to the mode (mode-matching probes, at the
nodes for E and the cells for H, normalised as openEMS's) and take the impedance from the mode's cutoff. Small
models on both engines (largest |S| difference over the band; the same model built through either API):

| Port | Model | ΔS11 | ΔS21 |
|---|---|---|---|
| rectangular waveguide, TE10 | WR-90 with a dielectric block, 8-12 GHz | 0.007 | 0.003 |
| coaxial | PTFE line with an air section, 1-9 GHz | 0.007 | 0.018 |
| stripline | a wide section, 1-9 GHz | 0.006 | 0.003 |
| microstrip | an open stub, 1-9 GHz | 0.029 | 0.021 |
| curve | a microstrip line between two curve ports | 0.008 | 0.010 |
| CPW (lumped ports across one gap) | no ground plane, 6 mm of air past the ends | 0.004 | 0.004 |

With the strips running into the Mur walls instead (the board's ends at the boundary), the two engines part by
up to 0.1 in S11 on the open CPW: they treat metal at an absorbing wall differently. kicad2ems keeps air
between the board and the boundary, so it doesn't meet this.

Before these ports, a current probe whose normal was y ran its loop the wrong way round (the current came out
negated, |S11| near 95 for a lumped port along y); x and z were right, which is why no example showed it.

## How it works

The operator (`gfdtd/operator.py`) uses openEMS's formulation. Fields are kept as edge voltages (V = E dl)
and currents (I = H dl′), so the update needs only three coefficient arrays per field component and no
geometry at run time:

```
I  -= iv * (loop sum of V around the primary face)        iv = dt l' / (mu A)
V   = vv * V + vi * (loop sum of I around the dual face)  from C = eps A' / l, G = kappa A' / l of the edge
```

- **Materials** are sampled at cell centres; the highest priority wins. Each E edge averages the four
  quarter cells around it.
- **Metals** take the edges whose midpoint lies in a primitive. Flat primitives snap to the nearest mesh line.
- **Lumped elements** spread R, L and C over the edges in their box, weighted by each column's share of
  the width. Parallel L and series RLC are branch currents on their edges.
- **Debye poles:** each pole is a polarisation current on its edges, tau di/dt + i = C_p dV/dt
  (C_p = eps0 eps_delta A′ / l, averaged like epsilon). It is integrated with the trapezoidal rule at the E
  update's time, which folds part of it into vv and vi: stable for any tau.
- **PML:** CPML (kappa 1, no alpha) with a cubic grading and R0 = 1e-6 over the PML_n cells. Each slab keeps
  two auxiliary terms per field along its axis; corners take both slabs' terms. The grading is the stretched
  coordinate's, the same for E and H, so the PML matches any dielectric that runs into it.
- **Dumps** are at the cell centres, on the cells openEMS's dump mode 2 picks (the box snapped to lines s0..s1,
  then the cells s0-1..s1), or at the nodes (modes 0, 1), sampled with the probes. The frequency domain is a
  running DFT on the device: 2 dt_s sum F(t) exp(-j w t), as openEMS's.
- **Far field** (`gfdtd/nf2ff.py`): the NF2FF box records E and H at the nodes of its six faces (as openEMS's,
  inside a Mur side's 2 lines or a PML_n side's n + 1; no face on a PEC / PMC side, which mirrors instead). The
  faces carry J = n x H and M = -n x E. On a planar face the phase e^(jk r.r') is a product over its two axes,
  so the radiation integrals are matrix products. Prad is the flux through the faces, 1/2 Re sum (E x H*).n,
  and Dmax = 4 pi r^2 max(S) / Prad, as openEMS's. The result file is openEMS's, which its `nf2ff_results` and
  kicad2ems's GUI read.
- **Time step:** the stability limit of the rectilinear grid (the shortest neighbouring cells around each
  node), times 0.95. That is about 10 % shorter than openEMS's Rennings step.

The engine (`gfdtd/engine.py`) is a Python time loop over one backend. The GPU and the CPU core run the same
update code (`gfdtd/core/fdtd_core.h`) on the **compressed operator**:

- There are no coefficient arrays. iv is pure geometry, dt dual_a / (mu d_b d_c), from three line arrays. vi is
  that geometry times a per-edge factor s. vv and s come from a table through one code per node, covering its
  three E edges: 8 bits, or 16 / 32 past 255 / 65535 codes. All nodes of one material (or one quarter-cell mix
  at an interface) share a code; the example boards need 40–60. This cuts each cell update from 108 to 73 bytes
  of memory traffic.
- The **Debye poles** are folded into the E update: one pass, no extra kernels. The pole currents are stored
  only for the nodes that have them, scaled so that their coefficients don't depend on the geometry either.
- The table is exact to 2^-20 (vv and s are rounded so that equal values match).

The three backends:

- **CuPy** (`gfdtd/kernels.py`): one thread per node, in blocks of 128. The same source compiles for ROCm
  through hiprtc. Fields are float32. A timestep is three launches:
  - `step_h`: H, the node's PML terms and the Mur copies.
  - `step_e`: E with the Debye poles and the node's PML terms.
  - `step_post`: the excitation, lumped branches and Mur update, together. Their edges must be disjoint, which
    is checked at setup; otherwise they run as separate kernels.

  With PML, the backend first times the PML inside the two kernels against one launch of its own per field, and
  keeps the faster. The trial steps don't change the zero fields. Inside wins on small and large models; at
  ~1 M cells (cache-resident) the bigger kernels' occupancy matters more. Nodes are decoded in 32-bit
  arithmetic, since the GPU emulates 64-bit division (up to 715 M nodes). A dump box is one kernel per sample
  that interpolates and adds the DFT term. Probe samples stay on the device. Excitations run in threads, each on
  its own stream, as many at once as fit in the GPU's cache together (`concurrent_runs`: up to 1.2 M cells in
  all; `--jobs` or `GFDTD_CONCURRENT_CELLS` set it).
- **core** (`gfdtd/core/gfdtd_core.cpp`, pybind11 and OpenMP): rows along z with the boundary tests outside the
  loops, so that the compiler vectorises them; Mur and PML too. Denormals are flushed to zero while it runs
  (the GPU does this anyway). Before that, the tiny fields ahead of the first wave ran the first few thousand
  steps 5–10 times slower. It runs one thread per physical core of the fastest kind: the performance cores of
  a hybrid CPU, no hyper-threads. On the i7-13700K that is 8 threads. On an idle machine 8 and all 24 run the
  example boards within 10 % of each other. With the desktop and a browser running, 24 threads were 2–4 times
  slower on the hairpin: every barrier waits for the slowest thread, and any busy core holds it up. 8 leave
  room. `--threads` or `OMP_NUM_THREADS` set it.
- **numpy**: the dense coefficient arrays as array slices. It is slow, and it is the reference the other two
  are tested against (1e-6).

Probe samples stay on the device and are written out in chunks. The energy is checked every 100 timesteps,
one host synchronisation each: the field's, and the lumped parts' (L I²/2 in an inductor, q²/2C in a series
capacitor). A filter of lumped parts rings in them, and on the field's energy alone its runs had ended early.

## Accuracy and speed

**PML, Debye, dumps.** A microstrip on a strongly dispersive substrate (eps_inf 3, one pole of 1.5 at 30 ps), PML
on five sides. Debye is checked against a plain material of the same complex epsilon at 5 GHz, where the two
must agree: gfdtd's Debye run gives S21 −4.240 dB / 40.9°, its plain run −4.235 dB / 41.0°, and openEMS's plain
run −4.234 dB / 41.2°. openEMS's own Debye run gives −4.744 dB / 36.8° there, 0.5 dB more loss than its plain
material. The PML agrees with openEMS's PML to 0.02 in S11 and 0.005 in S21. In the tests, a radial wave
through an 8-cell PML is within 5e-5 of the peak of a reflection-free reference, against 6 % for Mur. Dump
grids are openEMS's exactly. Values agree within ~3 %, of which 1.8 % is the incident wave itself (the maps
are scaled to 1 V incident).

rf_coupler with `--bc PML_8 --diel wideband --dump ...`: max |ΔS| 0.004 against openEMS. openEMS took 235 s;
gfdtd took 11 s on the GPU and 49 s on the CPU core (1.27 M cells).

**Antenna mode.** examples/antenna_dipole.toml, `--solvers openems,gfdtd`. Both find resonance at 2.435 GHz:
S11 −20.7 dB (openEMS) and −21.0 dB (gfdtd); Zin 62.8 + j7.3 and 63.0 + j6.8 ohm. From 1.9 to 3.1 GHz
directivity agrees within 0.02 dB and efficiency within 1 %; the pattern cuts within 0.05 dB. openEMS took 13 s,
gfdtd 1 s. At the band edges gfdtd needs a deeper end criterion. Its fields decay faster, so the default ends
the run while the far field's low end is still settling: with `--end-db -60 --end-on energy` it gives 73.3 %
efficiency at 1.5 GHz against openEMS's 74.1 %. On a half-wave dipole in free space (the tests) gfdtd gives
2.16 dBi and 100.1 % efficiency; openEMS 2.17 dBi and 100.0 %.

<a id="speed"></a>
**Default settings.** `--solvers openems,gfdtd` on the examples. openEMS 0.37 used 24 threads on an i7-13700K;
gfdtd ran on an AMD RX 7800 XT (ROCm 7, float32), and its CPU core on the i7. Both engines used the same mesh.

| Example | Cells | max \|ΔS\| | openEMS | gfdtd GPU | gfdtd CPU core |
|---|---|---|---|---|---|
| rf_coupler (P1 driven) | 0.56 M | 0.006 | 25 s | 3 s | 11 s |
| rf_diffvia (2 runs, at once on the GPU) | 0.41 M | 0.011 (mode conversion below −45 dB) | 148 s | 11 s | 66 s |
| demo_lowpass (L, C parts) | 0.18 M | 0.03 | 70 s | 1 s | 4 s |
| hairpin (high-Q filter) | 0.29 M | 0.02 | 49 s | 6 s | 25 s |

On demo_lowpass, openEMS's field energy stalled above `--end-db`, and the port watch ended its run after
205,000 timesteps. gfdtd's energy, the lumped parts' included, fell to −40 dB after 26,900 timesteps. The CPU
core's S-parameters are the GPU's within 1e-6.

**The engine alone** (`python tools/bench_gfdtd.py`): a microstrip on a Debye substrate, PML (or Mur) on five
sides, per timestep, in M cell updates per second. "First version" is the dense-coefficient GPU engine with PML:

| Cells | GPU, first version | GPU, PML | GPU, Mur | CPU core, PML / Mur | numpy |
|---|---|---|---|---|---|
| 0.1 M | 1,095 | 2,830 | 4,630 | 310 / 490 | 49 |
| 1 M | 4,040 | 6,480 | 9,120 | 700 / 830 | 52 |
| 4 M | 2,500 | 5,190 | 5,360 | 485 / 780 | – |
| 16 M | 2,670 | 5,690 | 5,510 | 485 / 620 | – |

Both compiled backends are bound by memory bandwidth on large models: the GPU moves ~420 GB/s of its 624 GB/s
over a whole step (73 bytes per cell update), the core ~45 GB/s of DRAM. Up to ~1 M cells the fields fit in
the GPU's 64 MB Infinity Cache and the CPU's 30 MB L3, hence the faster middle rows.

What each change gave on the GPU, measured on the benchmark (fused against the same step in separate kernels):

| Change | Gain |
|---|---|
| Coefficients compressed, Debye in the E update | 1.26× at 1 M, 1.5× at 4–16 M cells (PML) |
| One code per node, 8-bit (73 instead of 78 bytes) | 4–8 % on large models, none on small |
| The step in three launches (Mur, excitation, lumped parts) | 1.45× at 0.1 M, 1.2× at 0.3 M cells, ~1 % past 4 M (Mur) |
| The PML inside the H / E kernels | 1.25× at 0.1 M and at 4–16 M cells; at 0.3–1 M the PML stays its own launch (the setup trial picks) |
| 32-bit node decoding, blocks of 128 | 4–8 % |
| Excitations side by side (threads, a stream each) | 1.2–1.46× in all for 2–4 runs of 0.1–0.6 M cells; rf_diffvia's 2 runs 1.25× |
| A dump box in one kernel per sample | the far-field box's cost per step from ~38 to ~2 µs (antenna_dipole: 67 µs per step without it) |

Tried and left out:

- **Marching along x** (a thread per (y, z) column): +12 % at 1 M cells, −16 % at 4–16 M.
- **A graph of the step's kernels, or a compiled host loop:** 2–8 %. Python issues a step's launches faster
  than the GPU runs them.
- **More runs at once than the cache holds:** 2 runs of 1.5 M cells ran at 0.64× of one.
- **Temporal blocking** (several steps per tile in cache): not tried. It would be the next step for large
  models, but every extension (probes, Mur, lumped parts, excitations, dumps) would have to work tile by tile.

## Files

```
gfdtd/
  csx.py        CSXCAD's structure: grid, properties, primitives; its XML both ways
  fdtd.py       openEMS's interface: settings, LumpedPort, the Add...Port methods, Run, the XML; the command line
  ports.py      openEMS's other ports: MSL, stripline, CPW, coaxial, waveguides, curve
  operator.py   the model on the grid: coefficients, lumped elements, excitations, boundaries, probes
  engine.py     the time loop, the numpy, core and CuPy backends, status lines and probe files
  dumps.py      the field dumps: cell-centre and node values, .vtr frames, running DFT, HDF5
  nf2ff.py      the far field: openEMS's NF2FF box, CalcNF2FF and its result file
  kernels.py    the GPU kernels around core/fdtd_core.h
  core/         fdtd_core.h (the update code, GPU and CPU), gfdtd_core.cpp (the CPU core), build.py
kicad2ems/solvers/gfdtd.py   the solver plugin (openEMS's plugin, run on gfdtd)
tests/test_gfdtd.py          compatibility, operator, physics on small models, the far field, GPU and core
                             against numpy
tests/test_gfdtd_ports.py    the ports, weighted / hard / H / delayed excitations, mode matching, shells,
                             curves, current-probe snapping
tools/bench_gfdtd.py         the engine's per-operation timings by backend and model size
```
