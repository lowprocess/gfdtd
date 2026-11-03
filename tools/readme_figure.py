#!/usr/bin/env python3
"""The README's speed figure: M cell updates per second of each backend against the model's size, from
tools/bench.py (a microstrip on a Debye substrate, PML or Mur on five sides) on an RX 7800 XT (ROCm 7 / RADV) and
an i7-13700K (8 threads).

    python tools/readme_figure.py [out.png]          # default docs/speed.png
"""
import os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VK_CELLS = [0.29, 0.97, 3.96, 15.78]                           # M cells; Vulkan was timed in ms per step
# (label, colour, style, M cells, M cell updates / s)
RUNS = [('CuPy, PML', '#d62728', '-', [0.1, 1, 4, 16], [2830, 6480, 5190, 5690]),
        ('CuPy, Mur', '#d62728', '--', [0.1, 1, 4, 16], [4630, 9120, 5360, 5510]),
        ('Vulkan, PML', '#1f77b4', '-', VK_CELLS, [c / ms * 1e3 for c, ms in zip(VK_CELLS, [0.066, 0.153, 0.802, 2.735])]),
        ('Vulkan, Mur', '#1f77b4', '--', VK_CELLS, [c / ms * 1e3 for c, ms in zip(VK_CELLS, [0.049, 0.106, 0.750, 2.869])]),
        ('CPU core, PML', '#2ca02c', '-', [0.1, 1, 4, 16], [310, 700, 485, 485]),
        ('CPU core, Mur', '#2ca02c', '--', [0.1, 1, 4, 16], [490, 830, 780, 620]),
        ('numpy, PML', '#7f7f7f', '-', [0.1, 1], [49, 52])]


def main(out):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for label, colour, style, cells, speed in RUNS:
        ax.loglog(cells, speed, style, color=colour, marker='o', ms=4, label=label)
    ax.axvspan(0.08, 1.1, color='#f2f2f2', zorder=0)
    ax.text(0.3, 25, "the fields fit in the GPU's 64 MB cache", ha='center', fontsize=8, color='#555')
    ax.set(xlabel='model size (M cells)', ylabel='M cell updates / s', ylim=(20, 20000))
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8, ncol=2, loc='lower right')
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'docs', 'speed.png'))
