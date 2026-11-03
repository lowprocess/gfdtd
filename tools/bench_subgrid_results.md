# Subgridding experiment: results (2026-11-03)

`python tools/bench_subgrid.py accuracy|stability` on `gfdtd.experimental.subgrid` (numpy, float64, one thread,
a 2-core cloud VM, no GPU). Wall times compare the meshes with each other only; they say nothing about gfdtd's
GPU or core backends. Cell-updates (cells x steps) is the hardware-independent cost.

## Accuracy: re-entrant post cavity, 15 ns

A 1 x 1 mm PEC post from the floor to 1/3 mm below the ceiling of an 8 mm high PEC cavity. Errors are against
the uniform 1/3 mm mesh.

24 x 16 mm:

| mesh | cells | steps | cell-updates | wall (s) | f1 = 5.635 GHz | f2 = 14.454 GHz | f3 = 15.823 GHz |
|---|---|---|---|---|---|---|---|
| fine, uniform 1/3 mm | 82,944 | 24,597 | 2,040 M | 111 | ref | ref | ref |
| coarse, uniform 1 mm (1 mm gap) | 3,072 | 8,199 | 25 M | 2 | +14.05 % | +1.68 % | +0.16 % |
| graded, 1/3 to 1 mm by 1.5 | 8,448 | 24,597 | 208 M | 12 | -0.17 % | -0.05 % | -0.10 % |
| subgrid, 7 x 7 x 8 mm box at 3:1 | 13,656 | 24,597 | 336 M | 28 | -0.00 % | +0.08 % | +0.08 % |

48 x 32 mm (the same post, four times the plan area):

| mesh | cells | steps | cell-updates | wall (s) | f1 = 4.373 GHz | f2 = 7.885 GHz | f3 = 9.904 GHz |
|---|---|---|---|---|---|---|---|
| fine, uniform 1/3 mm | 331,776 | 24,597 | 8,161 M | 544 | ref | ref | ref |
| coarse, uniform 1 mm (1 mm gap) | 12,288 | 8,199 | 101 M | 6 | +7.66 % | -1.01 % | -0.03 % |
| graded, 1/3 to 1 mm by 1.5 | 24,640 | 24,597 | 606 M | 34 | -0.16 % | -0.04 % | -0.05 % |
| subgrid, 7 x 7 x 8 mm box at 3:1 | 22,872 | 24,597 | 563 M | 39 | -0.04 % | +0.00 % | -0.07 % |

- Both refinements get the gap-set mode to within 0.2 % at 6-14x fewer cell-updates than the uniform fine mesh.
  The subgrid is closer on f1 (0.00-0.04 % vs 0.16-0.17 %); the graded mesh pays for its stretched cells around
  the post.
- Cost: on the small cavity the graded mesh is cheaper (its fine lines cross little else). Quadrupling the plan
  area grows the graded mesh 2.9x but the subgrid only 1.7x, and the subgrid becomes the cheaper of the two: the
  graded mesh's fine bands run across the whole board, the box doesn't. With several features at different
  x and y, as on a real board, the bands multiply and cross; the boxes don't.
- The subgrid's wall time is higher per cell than its cell count suggests: the interface (a few dozen small
  matrix products per step) costs as much as ~6,000 cells here. On a GPU this would be a few extra small kernels.

## Stability: lossless closed cavity, energy after the pulse

20 x 16 x 12 mm, a 6 x 6 x 4 mm box fully inside (all six faces are interfaces).

| coupling | box contents | steps run | W(end) / W(after pulse) | outcome |
|---|---|---|---|---|
| none (coarse grid alone) | - | 60,000 | 0.97 | bounded |
| overlap (overwrite coarse fields inside the box) | empty | 2,000 | 4e6 | diverged by step 2,000 |
| adjoint, subcycled 3:1 in time | empty | 500 | 5e33 | diverged by step 500 |
| adjoint, one time step | empty | 60,000 | 1.02 | bounded |
| adjoint, one time step | 1 mm PEC cube | 40,250 | 1e6 | diverged by step 40,250 |

The accuracy model's subgrid (the post in the box), run on: bounded (0.98-1.03) to 110,000 steps, then 1.6x at
120,000: the same late-time growth, later.

## Conclusions

1. 3:1 boxed refinement gives the accuracy of a uniform fine mesh at a fraction of the cost, and scales better
   than graded tensor lines as the board grows.
2. **None of the couplings here is fit for production.** The adjoint one is stable empty but grows once metal
   sits in the box (onset 40k-110k steps, well within a long board run). Its energy pairing is not exact: the
   coarse surface E's normal derivative spans 2h/3 while its energy weight uses the coarse dual h, the full
   weighting is truncated at the box's edges, and the fine grid's normal H on the surface isn't paired with
   anything on the coarse side.
3. Subcycling (the real payoff: the coarse grid at 3x the step) needs an interface that is energy-conserving in
   time as well, which the explicit predictor-corrector here is not.
4. Next, if this is pursued: a provably stable construction (an SBP-SAT or dissipative Huygens-surface
   subgridding from the literature) in this harness, rerun the stability table to 10^6 steps, and only then port
   it to `fdtd_core.h`. Until then graded meshing (what gfdtd does now) is the safe choice.
