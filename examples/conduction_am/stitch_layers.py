#!/usr/bin/env python3
"""Box-per-layer stitch build of large_rectangle.path on the conduction source.

usage: stitch_layers.py selftest
       stitch_layers.py <spparks-binary> <outdir> [--layers N] [--window W]
                        [--min-window W] [--factor F] [--floor D]

Box n holds layer n: the full 360 x 250-site part plane, from its layer top
down W sites, continuing the microstructure through <outdir>/stitch.st. Each
box is a fresh thermal history (T0 at start, as the tall-box layer reset).

Box depth adapts. Every deck has `floor_check D`: the app stops with an error,
before the box's stitch dump, if liquid comes within D sites of the box
bottom, and prints the deepest liquid as it goes. The driver then reruns that
box with a 1.5x deeper window. After a box completes, the next window is
max(min-window, ceil(F * deepest melt below the top) + D). Melt pools change
with layer (heat builds up in the part), so a fixed depth is a guess.

Completed boxes (completion.marker) are skipped; touch <outdir>/STOP to end
after the current box. Never kill a running box (HIP kernels mid-flight wedge
the GPU). Per-box windows and melt depths go to <outdir>/boxes.txt.
"""
import argparse, math, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DX, SPEED, PAUSE = 5e-6, 1.4, 5e-3

DECK = '''app_style        additive/texture/kk 1000000 5e-6 5e-6 2
seed             {seed}
dimension        3
lattice          sc/26n 1.0
region           box block 0 360 0 250 {zlo} {zhi}
boundary         n n n
create_box       box
create_sites     box
{init}
region           plate block -200 560 -200 450 -200 18.5
region           footprint block 19 344 20 235 -1000 1000
region           part union 2 plate footprint
sector           yes
sweep            color
diag_style       array d6 sum d6 max
temperature      0.0
misorientation_target random_rotation
liquidus         1708
solidus          1675
solid_front_vel  4 1.091e-5 -2.034e-4 2.74e-3 1.151e-4
nucleation_density      1e14
critical_undercooling   5
undercooling_deviation  3
mean_nuclei_volume      8.4375e-16
nuclei_volume_deviation 1.5625e-17
floor_check      {floor}
reset_time       {t0}
setup_temperature_source conduction 150.0 1.0 13.8 3.6166e-6 300.0 483.0 59.4e-6 59.4e-6 99.0e-6 &
    k_powder 0.3 k_table 6 300 13.8 600 20.0 900 24.8 1200 28.4 1500 30.6 1675 31.3 &
    cp_table 7 300 483 473 528 873 595 1073 611 1253 624 1675 714 1708 847 &
    latent 2.9e5 convection 25 300 emissivity 0.4 &
    margin 0.5e-3 0.5e-3 0.5e-3 0.5e-3 plate {plate} plate_top 92.5e-6 part part far_bc fixed sts 128
laser_path file {path} speed 1.4 scale 5e-6 shift 0 0 2.5e-6 jump_time {pause}
stats            1e-3
dump             st stitch {t1} {st} i1 i2 d1 d2 d3 d4 d8 d9 d10
dump_modify      st delay {t1}
run              {t1} upto
'''
# Box 0: spin-42 substrate below z 18.5; powder keeps the random spin from
# `set site range` and a zero quaternion, which init turns into a random
# orientation per site.
INIT0 = '''set site range 1 1000000
region           substrate block 0 360 0 250 0 18.5
set i2 value 3 region substrate
set i1 value 42 region substrate
set d1 value 1.0 region substrate
set d2 value 0.0 region substrate
set d3 value 0.0 region substrate
set d4 value 0.0 region substrate
set d6 value 300'''
# Later boxes: sites absent from the stitch file (the new layer, i2 < 1) get a
# random spin and a zero (= random) orientation; powder becomes i2 1.
FIELDS = ('i1', 'i2', 'd1', 'd2', 'd3', 'd4', 'd8', 'd9', 'd10')
INITN = '''{loads}set d1 value 0.0 if i2 < 1
set d2 value 0.0 if i2 < 1
set d3 value 0.0 if i2 < 1
set d4 value 0.0 if i2 < 1
set i1 range 1 1000000 if i2 < 1
set i2 value 1 if i2 < 2
set d6 value 300
set d8 value 0.0 if i2 != 3
set d9 value 0.0 if i2 != 3
set d10 value 0.0 if i2 != 3'''


def split_layers(path):
    """Rows `x y z p` in lattice units; a z change starts a layer."""
    layers = []
    for l in open(path):
        r = l.split()
        if not r or r[0].startswith('#'): continue
        if not layers or r[2] != layers[-1][0][2]: layers.append([])
        layers[-1].append(r)
    return layers


def duration(L):
    """Scan time of one layer: emitting moves at SPEED, each jump waits PAUSE."""
    t = 0.0
    for a, b in zip(L, L[1:]):
        if float(b[3]) > 0:
            t += math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])) * DX / SPEED
        else:
            t += PAUSE
    return t


def deepest(out):
    """Last `floor_check: deepest liquid z = Z` in a box's screen output."""
    z = re.findall(r'floor_check: deepest liquid z = (\S+)', open(out).read())
    return float(z[-1]) if z else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('binary'); ap.add_argument('outdir')
    ap.add_argument('--layers', type=int, default=20)
    ap.add_argument('--window', type=int, default=36, help='first box depth below the layer top, sites')
    ap.add_argument('--min-window', type=int, default=12, help='never shallower than this, sites')
    ap.add_argument('--factor', type=float, default=1.5, help='next window = factor x deepest melt + floor')
    ap.add_argument('--floor', type=int, default=3, help='floor_check distance, sites')
    a = ap.parse_args()
    binary, out = os.path.abspath(a.binary), os.path.abspath(a.outdir)
    os.makedirs(out, exist_ok=True)
    st = f'{out}/stitch.st'
    layers = split_layers(f'{HERE}/large_rectangle.path')[:a.layers]
    win = a.window
    for n, L in enumerate(layers):
        d = f'{out}/box_{n:02d}'; os.makedirs(d, exist_ok=True)
        ztop = int(float(L[0][2]))               # top site of layer n
        if os.path.exists(f'{d}/completion.marker'):
            z = deepest(f'{d}/out.txt')
            if z is not None: win = max(a.min_window, math.ceil(a.factor * (ztop - z)) + a.floor)
            continue
        if os.path.exists(f'{out}/STOP'): sys.exit(f'STOP file present, stopping before box {n}')
        path = f'{d}/layer.path'
        open(path, 'w').write(''.join(' '.join(r) + '\n' for r in L))
        t0 = 10.0 * n
        t1 = round(t0 + duration(L) + 0.005, 6)
        init = INIT0 if n == 0 else INITN.format(loads=''.join(f'set {k} stitch {st} last\n' for k in FIELDS))
        while True:
            zlo = max(0, ztop - win)
            open(f'{d}/in.deck', 'w').write(DECK.format(
                seed=56789 + n, zlo=zlo, zhi=ztop + 1, init=init, floor=a.floor * DX, t0=t0,
                plate=round(zlo * DX + 0.5e-3, 9), path=path, pause=PAUSE, t1=t1, st=st))
            print(f'box {n}: z [{zlo}, {ztop}] window {ztop - zlo}', flush=True)
            with open(f'{d}/in.deck') as fi, open(f'{d}/out.txt', 'w') as fo:
                rc = subprocess.call(['mpirun', '-np', '1', '--bind-to', 'none', binary, '-log', 'log.spparks'],
                                     stdin=fi, stdout=fo, stderr=subprocess.STDOUT, cwd=d)
            text = open(f'{d}/out.txt').read()
            if rc == 0 and 'Loop time' in text:
                break
            if 'floor_check: liquid' not in text or zlo == 0:
                sys.exit(f'box {n} FAILED (see {d}/out.txt)')
            win = math.ceil(1.5 * win)            # melt hit the floor: deepen, rerun
            print(f'box {n}: melt reached the floor, retrying with window {win}', flush=True)
        open(f'{d}/completion.marker', 'w').close()
        z = deepest(f'{d}/out.txt')
        depth = ztop - z if z is not None else 0
        with open(f'{out}/boxes.txt', 'a') as f:
            f.write(f'{n} zlo {zlo} ztop {ztop} window {ztop - zlo} melt_depth_sites {depth:g}\n')
        win = max(a.min_window, math.ceil(a.factor * depth) + a.floor)
    print('all boxes done')


def selftest():
    L = split_layers(f'{HERE}/large_rectangle.path')
    assert len(L) == 21 and [int(float(l[0][2])) for l in L[:3]] == [21, 27, 33], [l[0][2] for l in L[:3]]
    assert [round(duration(l) * 1e3, 1) for l in L[:2]] == [116.4, 171.0], [duration(l) for l in L[:2]]
    print('selftest ok')


if __name__ == '__main__':
    selftest() if sys.argv[1:] == ['selftest'] else main()
