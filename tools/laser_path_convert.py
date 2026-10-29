#!/usr/bin/env python3
"""Convert scan paths to the `laser_path file` waypoint format.

Output rows are `x y z p [v]`: a move from the previous point to (x, y, z)
at power fraction p (0 = laser off), optionally at speed v. Coordinates stay
in input units; scale/shift them in the deck (`laser_path file F speed V
scale S shift DX DY DZ`). A change of z starts a new layer.

  laser_path_convert.py fd PATH > out.path
      am/finitediff 5-column path (x, y, z, distance, flag). A row flagged
      1 (recoat) or 2 (jump) ends a scan; the move to the next row is off.

  laser_path_convert.py gcode FILE [--travel-feed F] [--feed-scale K] > out.path
      G0/G1 X Y Z [E] [F] [S], G90/G91. A G1 move is on when it extrudes
      (E increases) or sets S > 0, and its feed is below --travel-feed;
      G0 moves are off. --feed-scale K writes v = F * K (e.g. mm/min to m/s:
      1.6667e-5).

  laser_path_convert.py insitu FILE.h5 LAYER --power Q [--every N]
                         [--window X0 X1 Y0 Y1 --margin M] > layer.path
      One layer of in-situ monitoring data (HDF5 layer_K groups of
      t_spectral + spectral [x y P ...], or layer_K datasets of [t x y P];
      mm, s, W). Rows are in METRES (deck: no scale) with p = measured
      power / Q and v = measured speed; a sample gap > 1.5 x the sample
      period is a laser-off jump whose row carries its measured duration.
      Each move is split every N samples (default 10). --window (mm) keeps
      only samples within M mm of the box; dropped time goes into the jumps.
      A '# t0' comment gives the time of the first row: pass it as
      `laser_path file ... t0 T`.

  laser_path_convert.py selftest
"""
import re
import sys


def from_fd(lines):
    out, off = [], False
    for ln in lines:
        c = [float(v) for v in ln.replace(',', ' ').split()]
        if len(c) < 5:
            continue
        out.append((c[0], c[1], c[2], 0.0 if (off or not out) else 1.0, None))
        off = c[4] != 0.0
    return out


def from_gcode(lines, travel_feed=None, feed_scale=None):
    pos = {'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'E': 0.0}
    feed, power, absolute, out = 0.0, 0.0, True, []
    for ln in lines:
        ln = ln.split(';')[0].strip().upper()
        w = dict((m[0], float(m[1:])) for m in re.findall(r'[A-Z][-+.0-9]+', ln))
        g = w.get('G')
        if g == 90: absolute = True
        if g == 91: absolute = False
        if g not in (0, 1) or not any(k in w for k in 'XYZ'):
            if 'E' in w and g == 92: pos['E'] = w['E']
            if 'F' in w: feed = w['F']
            continue
        feed = w.get('F', feed)
        power = w.get('S', power)
        new = dict(pos)
        for k in 'XYZE':
            if k in w:
                new[k] = w[k] if absolute else pos[k] + w[k]
        on = g == 1 and (new['E'] > pos['E'] or power > 0.0) and \
            not (travel_feed and feed >= travel_feed)
        if not out:
            out.append((pos['X'], pos['Y'], pos['Z'], 0.0, None))
        v = feed * feed_scale if (feed_scale and on) else None
        out.append((new['X'], new['Y'], new['Z'], 1.0 if on else 0.0, v))
        pos = new
    return out


def read_insitu(path, layer):
    import h5py   # only this mode needs it
    with h5py.File(path, 'r') as f:
        o = f[f'layer_{layer}']
        z = float(o.attrs['z height (mm)'][0])
        if isinstance(o, h5py.Dataset):
            a = o[:]
            return a[:, 0], a[:, 1], a[:, 2], a[:, 3], z
        s = o['spectral'][:]
        return o['t_spectral'][:, 0], s[:, 0], s[:, 1], s[:, 2], z


def from_insitu(t, x, y, P, z, Q, every=10, window=None, margin=0.0):
    """Returns (t0, rows); rows in m, v in m/s (on) or s (off)."""
    keep = [True] * len(t)
    if window:
        x0, x1, y0, y1 = window
        keep = [x0 - margin <= xi <= x1 + margin and y0 - margin <= yi <= y1 + margin
                for xi, yi in zip(x, y)]
    gap = 1.5 * sorted(b - a for a, b in zip(t, t[1:]))[(len(t) - 1) // 2]
    runs, cur = [], []                    # runs of kept samples within one move
    for i in range(len(t)):
        if cur and (not keep[i] or t[i] - t[i - 1] > gap):
            runs.append(cur); cur = []
        if keep[i]:
            cur.append(i)
    if cur:
        runs.append(cur)
    out, t0, t_end = [], None, None
    for r in runs:
        b = r[::every] + ([r[-1]] if (len(r) - 1) % every else [])
        pts = [b[0]]                      # drop zero-length pieces (5 um position quantum)
        for j in b[1:]:
            if (x[j], y[j]) != (x[pts[-1]], y[pts[-1]]):
                pts.append(j)
            elif j == b[-1] and len(pts) > 1:
                pts[-1] = j
        if len(pts) < 2:
            continue   # ponytail: a run that never moves can't emit in a path file; its time goes into the next jump
        i = pts[0]
        if t0 is None:
            t0 = t[i]
            out.append((x[i] * 1e-3, y[i] * 1e-3, z * 1e-3, 0.0, None))
        else:
            out.append((x[i] * 1e-3, y[i] * 1e-3, z * 1e-3, 0.0, t[i] - t_end))
        for a, j in zip(pts, pts[1:]):
            d = ((x[j] - x[a]) ** 2 + (y[j] - y[a]) ** 2) ** 0.5 * 1e-3
            p = sum(P[a:j + 1]) / (j + 1 - a) / Q
            out.append((x[j] * 1e-3, y[j] * 1e-3, z * 1e-3, p, d / (t[j] - t[a])))
        t_end = t[pts[-1]]
    return t0, out


def write(rows, f=sys.stdout):
    for x, y, z, p, v in rows:
        f.write(f'{x:.10g} {y:.10g} {z:.10g} {p:.6g}' + (f' {v:.10g}' if v else '') + '\n')


def selftest():
    fd = from_fd(['30, 30, 21, 0, 0', '330, 30, 21, 300, 2', '330, 40, 21, 300, 0',
                  '30, 40, 21, 600, 1', '30, 40, 27, 600, 0', '30, 60, 27, 620, 0'])
    assert [r[3] for r in fd] == [0, 1, 0, 1, 0, 1], fd
    gc = from_gcode(['G21', 'G90', 'G1 Z0.35 F7800', 'G1 X1 Y1 F7800',
                     'G1 X2 Y1 E0.1 F1800', 'G1 X2 Y2 E0.2', 'G0 X5 Y5',
                     'G1 X6 Y5 S200 F600'], travel_feed=7800, feed_scale=1 / 60000)
    assert [r[3] for r in gc] == [0, 0, 0, 1, 1, 0, 1], gc
    assert gc[2][2] == 0.35 and abs(gc[3][4] - 0.03) < 1e-12, gc
    # in situ: two 1 mm hatches at 1 m/s (20 us samples), a 1.1 ms jump between
    n, dt = 51, 2e-5
    t = [i * dt for i in range(n)] + [(n - 1) * dt + 1.1e-3 + i * dt for i in range(n)]
    x = [i * 0.02 for i in range(n)] * 2
    y = [0.0] * n + [0.12] * n
    P = [150.0] + [300.0] * (n - 1) + [150.0] + [300.0] * (n - 1)
    t0, rows = from_insitu(t, x, y, P, 0.03, Q=300.0, every=10)
    assert t0 == 0.0 and len(rows) == 1 + 5 + 1 + 5, rows
    # replay the written file's timing as laser_path does: on rows d / v, off rows v
    import io
    buf = io.StringIO(); write(rows, buf)
    text = [[float(c) for c in ln.split()] for ln in buf.getvalue().splitlines()]
    tt, prev = 0.0, text[0]
    for r in text[1:]:
        d = ((r[0] - prev[0]) ** 2 + (r[1] - prev[1]) ** 2) ** 0.5
        tt += d / r[4] if r[3] > 0 else r[4]
        prev = r
    assert abs(tt - t[-1]) < 1e-12, tt
    assert abs(rows[1][4] - 1.0) < 1e-9 and abs(rows[6][4] - 1.1e-3) < 1e-15, rows
    assert abs(rows[1][3] - (150 + 300 * 10) / 11 / 300) < 1e-12 and rows[2][3] == 1.0, rows
    # window keeps x <= 0.5 mm: each hatch is cut, the dropped time is in the jump
    t0, rows = from_insitu(t, x, y, P, 0.03, Q=300.0, every=10, window=(-1, 0.5, -1, 1))
    assert len(rows) == 1 + 3 + 1 + 3 and abs(rows[4][4] - (t[n] - t[25])) < 1e-15, rows
    print('selftest ok')


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a or a[0] not in ('fd', 'gcode', 'insitu', 'selftest'):
        sys.exit(__doc__)
    opt = lambda k: float(a[a.index(k) + 1]) if k in a else None
    if a[0] == 'selftest':
        selftest()
    elif a[0] == 'fd':
        write(from_fd(open(a[1])))
    elif a[0] == 'insitu':
        if opt('--power') is None:
            sys.exit('insitu needs --power Q (the deck\'s laser power, W)')
        w = [float(v) for v in a[a.index('--window') + 1:a.index('--window') + 5]] \
            if '--window' in a else None
        t0, rows = from_insitu(*read_insitu(a[1], int(a[2])), Q=opt('--power'),
                               every=int(opt('--every') or 10), window=w,
                               margin=opt('--margin') or 0.0)
        if t0 is None:
            sys.exit(f'layer {a[2]}: no emitting move inside the window')
        print(f'# in situ {a[1]} layer {a[2]}: x y z (m) p v (on: m/s, off: s)')
        print(f'# t0 {t0:.9f}')
        write(rows)
    else:
        write(from_gcode(open(a[1]), opt('--travel-feed'), opt('--feed-scale')))
