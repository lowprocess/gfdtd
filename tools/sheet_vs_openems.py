"""
Conductor loss against openEMS: a 40 mm microstrip (0.4 mm on 0.2 mm, eps_r 3.66, lumped 50 ohm ports), strip and
ground as 35 um copper conducting sheets, run on gfdtd and, from the same XML, on openEMS; and on gfdtd with PEC
for the loss the sheets add.

    python tools/sheet_vs_openems.py RUN_FOLDER [--openems PATH] [--backend core]
"""
import argparse, os, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import numpy as np

from gfdtd import ContinuousStructure, openEMS
SIG, T = 5.8e7, 35e-6


def model(kind, L=40.0, w=0.4, h=0.2, er=3.66):
    F = openEMS(NrTS=200000, EndCriteria=1e-4)
    F.SetGaussExcite(10e9, 10e9)
    F.SetBoundaryCond(['MUR'] * 6)
    C = ContinuousStructure(); F.SetCSX(C)
    g = C.GetGrid(); g.SetDeltaUnit(1e-3)
    x = np.r_[np.linspace(-L / 2 - 3, -L / 2, 7)[:-1], np.arange(-L / 2, L / 2 + 1e-9, 0.25), np.linspace(L / 2, L / 2 + 3, 7)[1:]]
    yi = np.r_[-w / 2 - 0.1, np.linspace(-w / 2, w / 2, 5), w / 2 + 0.1]
    y = np.unique(np.r_[np.linspace(-4, -w / 2 - 0.1, 12), yi, np.linspace(w / 2 + 0.1, 4, 12)])
    z = np.unique(np.r_[-1.0, -0.5, np.linspace(0, h, 5), np.linspace(h, h + 3, 16)])
    g.SetLines('x', x); g.SetLines('y', y); g.SetLines('z', z)
    sub = C.AddMaterial('sub', epsilon=er, kappa=0)
    sub.AddBox([x[0], y[0], 0], [x[-1], y[-1], h], priority=1)
    met = C.AddMetal('cu') if kind == 'pec' else C.AddConductingSheet('cu', conductivity=SIG, thickness=T)
    met.AddBox([-L / 2, -w / 2, h], [L / 2, w / 2, h], priority=10)
    met.AddBox([x[0], y[0], 0], [x[-1], y[-1], 0], priority=10)
    p1 = F.AddLumpedPort(1, 50, [-L / 2, -w / 2, 0], [-L / 2, w / 2, h], 'z', excite=1, priority=5)
    p2 = F.AddLumpedPort(2, 50, [L / 2, -w / 2, 0], [L / 2, w / 2, h], 'z', excite=0, priority=5)
    return F, C, (p1, p2)


def s21(F, ports, path):
    f = np.linspace(0.5e9, 20e9, 79)
    for p in ports:
        p.CalcPort(path, f, ref_impedance=50)
    return f, ports[1].uf_ref / ports[0].uf_inc, ports[0].uf_ref / ports[0].uf_inc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root')
    ap.add_argument('--openems', default='openEMS', help="the openEMS executable")
    ap.add_argument('--backend', default='auto', help="gfdtd's backend")
    a = ap.parse_args()
    root, OEMS = a.root, a.openems
    res = {}
    for kind, eng in (('sheet', 'gfdtd'), ('sheet', 'openems'), ('pec', 'gfdtd')):
        F, C, ports = model(kind)
        path = os.path.join(root, f'{kind}_{eng}')
        os.makedirs(path, exist_ok=True)
        a = time.time()
        if eng == 'gfdtd':
            F.Run(path, backend=a.backend, log=os.path.join(path, 'log.txt'))
        else:
            F.Write2XML(os.path.join(path, 'model.xml'))
            subprocess.run([OEMS, 'model.xml'], cwd=path, stdout=open(os.path.join(path, 'log.txt'), 'w'),
                           stderr=subprocess.STDOUT, check=True)
        f, S21, S11 = s21(F, ports, path)
        res[kind, eng] = (S21, S11)
        print(f"{kind:5s} {eng:7s} {time.time() - a:6.0f} s", flush=True)
    il = lambda k: -20 * np.log10(np.abs(res[k][0]))
    print("\n  f GHz | IL pec gfdtd | IL sheet gfdtd | IL sheet openEMS | sheet loss gfdtd | openEMS | |dS21| gfdtd-openEMS | |dS11|")
    for i in range(0, len(f), 6):
        d = abs(res['sheet', 'gfdtd'][0][i] - res['sheet', 'openems'][0][i])
        d11 = abs(res['sheet', 'gfdtd'][1][i] - res['sheet', 'openems'][1][i])
        print(f"  {f[i] / 1e9:5.1f} | {il(('pec', 'gfdtd'))[i]:8.3f} | {il(('sheet', 'gfdtd'))[i]:8.3f} | {il(('sheet', 'openems'))[i]:8.3f} |"
              f" {il(('sheet', 'gfdtd'))[i] - il(('pec', 'gfdtd'))[i]:8.3f} | {il(('sheet', 'openems'))[i] - il(('pec', 'gfdtd'))[i]:8.3f} | {d:.4f} | {d11:.4f}")


if __name__ == '__main__':
    main()
