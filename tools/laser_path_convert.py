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


def write(rows, f=sys.stdout):
    for x, y, z, p, v in rows:
        f.write(f'{x:.6g} {y:.6g} {z:.6g} {p:g}' + (f' {v:.6g}' if v else '') + '\n')


def selftest():
    fd = from_fd(['30, 30, 21, 0, 0', '330, 30, 21, 300, 2', '330, 40, 21, 300, 0',
                  '30, 40, 21, 600, 1', '30, 40, 27, 600, 0', '30, 60, 27, 620, 0'])
    assert [r[3] for r in fd] == [0, 1, 0, 1, 0, 1], fd
    gc = from_gcode(['G21', 'G90', 'G1 Z0.35 F7800', 'G1 X1 Y1 F7800',
                     'G1 X2 Y1 E0.1 F1800', 'G1 X2 Y2 E0.2', 'G0 X5 Y5',
                     'G1 X6 Y5 S200 F600'], travel_feed=7800, feed_scale=1 / 60000)
    assert [r[3] for r in gc] == [0, 0, 0, 1, 1, 0, 1], gc
    assert gc[2][2] == 0.35 and abs(gc[3][4] - 0.03) < 1e-12, gc
    print('selftest ok')


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a or a[0] not in ('fd', 'gcode', 'selftest'):
        sys.exit(__doc__)
    if a[0] == 'selftest':
        selftest()
    elif a[0] == 'fd':
        write(from_fd(open(a[1])))
    else:
        opt = lambda k: float(a[a.index(k) + 1]) if k in a else None
        write(from_gcode(open(a[1]), opt('--travel-feed'), opt('--feed-scale')))
