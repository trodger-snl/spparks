#!/usr/bin/env python3
"""Unattended Sobol calibration of the GP700 build against Lamb et al. 2023.
Parameters (per point): misorientation D [deg^2/um], nucleation_density [/m^3],
critical_undercooling [K]. Each point is a box-per-layer build on a proxy
geometry (default 600 um square, 7 hatches, 8 layers, dt 4 us), analysed with
gp700_analysis.py (floor 100 um -> all grains originating, central crop), and
scored: RMS of (simulated - Lamb Fig. 5a) mean SROD over 25..240 um height.
Resumable; never edits the shared template/driver (substitutes on a copy).

usage: sobol_study.py --out DIR --binary BIN [--n 8] [--deadline HH:MM] [--dry]
"""
import argparse, json, os, re, subprocess, sys, time, datetime
import numpy as np
from scipy.stats import qmc
HERE = os.path.dirname(os.path.abspath(__file__))
PY = '/home/tron/miniconda3/envs/WORK/bin/python'
ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--binary', required=True)
ap.add_argument('--n', type=int, default=8); ap.add_argument('--seed', type=int, default=12345)
ap.add_argument('--nx', type=int, default=200); ap.add_argument('--nh', type=int, default=7)
ap.add_argument('--layers', type=int, default=8); ap.add_argument('--dt', default='4e-6')
ap.add_argument('--deadline', default=None, help='HH:MM local; do not START new points after this')
ap.add_argument('--dry', action='store_true')
ap.add_argument('--range', nargs=3, action='append', default=[], metavar=('NAME','LO','HI'), help='override/add a design range (log for D/nucleation, lin otherwise); LO==HI fixes it')
ap.add_argument('--design-seed', type=int, default=0, help='Sobol scramble seed')
ap.add_argument('--extra', default=None, help='extra deck line(s) inserted after smooth_greedy_multiproposal (use ; to separate)')
ap.add_argument('--score-only', default=None, help='score this summary.json and exit')
ap.add_argument('--fixed', default=None, help='JSON dict of parameters: run this single point instead of a Sobol design')
ap.add_argument('--greedy', choices=['on','off'], default=None, help='override smooth_greedy_multiproposal')
ap.add_argument('--nrefine', type=int, default=None, help='override the app_style nrefine argument')
ap.add_argument('--substrate', default=None, help='stitch file from make_substrate.py; box 0 starts from it')
ap.add_argument('--D-range', type=float, nargs=2, default=[0.2, 0.8], help='D range, deg^2/um (log-uniform)')
ap.add_argument('--texture', action='store_true', help='add texture_parameters c2 (1..4, mobility 1 + c2 cos(2.5 theta)) as a 4th parameter')
a = ap.parse_args()
if not a.score_only: os.makedirs(a.out, exist_ok=True)
DX = 3e-6; PAUSE = 8.8e-3; TGAP = 10.0
RANGES = {'D_deg2_um': (a.D_range[0], a.D_range[1], 'log'), 'nucleation_density': (1e13, 1e14, 'log'), 'critical_undercooling': (3.0, 25.0, 'lin')}
if a.texture: RANGES['texture_c2'] = (1.0, 4.0, 'lin')
for nm, lo, hi in a.range: RANGES[nm] = (float(lo), float(hi), 'log' if nm in ('D_deg2_um','nucleation_density') else 'lin')
lamb = json.load(open(os.path.join(HERE, '..', '..', 'docs', 'lamb2023_fig5a_srod.json')))
def score(summary):
    rows = np.array(summary['srod'].get('height_originating', []))
    if len(rows) == 0: return None
    d, mean, p5, p95 = rows[:,0], rows[:,1], rows[:,2], rows[:,3]
    m = (d >= 25) & (d <= 240)
    if m.sum() < 5: return None
    lm = np.interp(d[m], lamb['x'], lamb['mean'])
    return dict(rms_srod_dev=float(np.sqrt(np.mean((mean[m]-lm)**2))), mean_bias=float(np.mean(mean[m]-lm)),
                band_ratio=float(np.mean((p95[m]-p5[m])/mean[m])), n_filtered=summary['n_filtered'],
                frac_001_15deg=summary['frac_001_15deg'], n_grains=summary['n_grains'], hmax=summary['max_height_um'])
if a.score_only:
    print(json.dumps(score(json.load(open(a.score_only))))); sys.exit(0)
# ---- design
sob = qmc.Sobol(d=len(RANGES), scramble=True, seed=a.design_seed).random(a.n)
if a.fixed: sob = []
pts = []
for u in sob:
    p = {}
    for (k, (lo, hi, sc)), x in zip(RANGES.items(), u):
        p[k] = float(np.exp(np.log(lo) + x*(np.log(hi)-np.log(lo)))) if sc == 'log' else float(lo + x*(hi-lo))
    pts.append(p)
if a.fixed: pts = [json.loads(a.fixed)]
json.dump(pts, open(f'{a.out}/design.json', 'w'), indent=1)
tpl = open(os.path.join(HERE, 'in.gp700_layer')).read()
def deck(p, n, out, st, seed):
    ztop = 50 + 10*(n+1); zlo = max(0, ztop-60); z0 = ztop*DX
    lend = a.nx*DX + 100e-6; ylast = 40e-6 + 80e-6*(a.nh-1)
    laser = (f"laser_path start -100e-6 40e-6 {z0} end {lend} 40e-6 speed 1.137 hatch 80e-6 {a.nh} serpentine pause {PAUSE}" if n % 2 == 0 else
             f"laser_path start {ylast} -100e-6 {z0} end {ylast} {lend} speed 1.137 hatch 80e-6 {a.nh} serpentine pause {PAUSE}")
    tbox = round(a.nh*(a.nx*DX+200e-6)/1.137 + (a.nh-1)*PAUSE + 2e-3, 6)
    t0 = n*TGAP; t1 = n*TGAP + tbox
    init = ("set site range 1 1000000\nset i2 value 1\nset d6 value 473" if (n == 0 and not a.substrate) else
            "\n".join([f"set {f} stitch {st} last" for f in ('i1','i2','d1','d2','d3','d4','d8','d9')] +
                      ["set i1 range 1 1000000 if i2 < 1", "set i2 value 1 if i2 < 2", "set d6 value 473",
                       "set d8 value 0.0 if i2 != 3", "set d9 value 0.0 if i2 != 3"]))
    s = tpl
    for k, v in {'__SEED__': str(seed+n), '__ZLO__': str(zlo), '__ZHI__': str(ztop), '__INIT_OR_STITCH__': init,
                 '__LASER__': laser, '__T0__': str(t0), '__T1__': str(t1), '__STITCH__': st, '__NX__': str(a.nx), '__DT__': a.dt}.items():
        s = s.replace(k, v)
    s = re.sub(r'^nucleation_density\s+.*$', f"nucleation_density      {p['nucleation_density']:.4g}", s, flags=re.M)
    s = re.sub(r'^critical_undercooling\s+.*$', f"critical_undercooling   {p['critical_undercooling']:.4g}", s, flags=re.M)
    s = re.sub(r'^misorientation_diffusion\s+\S+', f"misorientation_diffusion {p['D_deg2_um']*1e6:.4g}", s, flags=re.M)
    if 'texture_c2' in p: s = re.sub(r'^texture_parameters\s+.*$', f"texture_parameters {p['texture_c2']:.4g} 2.5", s, flags=re.M)
    if a.greedy: s = re.sub(r'^smooth_greedy_multiproposal\s+.*$', f'smooth_greedy_multiproposal {a.greedy}', s, flags=re.M)
    if a.extra: s = re.sub(r'^(smooth_greedy_multiproposal\s+.*)$', lambda m: m.group(1) + '\n' + a.extra.replace(';', '\n'), s, flags=re.M)
    if a.nrefine is not None: s = re.sub(r'^(app_style additive/texture/kk \S+ \S+ \S+) \d+', rf'\g<1> {a.nrefine}', s, flags=re.M)
    open(out, 'w').write(s); return t1
def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)
res_path = f'{a.out}/results.jsonl'
done = {json.loads(l)['point'] for l in open(res_path)} if os.path.exists(res_path) else set()
for k, p in enumerate(pts):
    if k in done: log(f'point {k} done, skipping'); continue
    if a.deadline and datetime.datetime.now().strftime('%H:%M') >= a.deadline and datetime.datetime.now().hour < 12:
        log(f'deadline {a.deadline} passed; not starting point {k}'); break
    pd = f'{a.out}/pt_{k:02d}'; os.makedirs(pd, exist_ok=True); st = f'{pd}/gp700.st'
    if a.substrate and not os.path.exists(st) and not a.dry:
        import shutil; shutil.copy(a.substrate, st)
    log(f'point {k}: ' + ', '.join(f'{kk}={vv:.4g}' for kk, vv in p.items()))
    ok = True
    for n in range(a.layers):
        bd = f'{pd}/box_{n:03d}'; os.makedirs(bd, exist_ok=True)
        if os.path.exists(f'{bd}/completion.marker'): continue
        deck(p, n, f'{bd}/in.box', st, a.seed)
        if a.dry: continue
        cmd = f'. /home/tron/spack/share/spack/setup-env.sh && spack env activate spparks-dev && cd {bd} && mpirun -np 1 --bind-to none {a.binary} < in.box > log.spparks 2>&1'
        r = subprocess.run(['bash', '-c', cmd])
        if r.returncode != 0 or 'Loop time' not in open(f'{bd}/log.spparks').read():
            log(f'point {k} box {n} FAILED (rc {r.returncode})'); ok = False; break
        open(f'{bd}/completion.marker', 'w').close()
    if a.dry: log(f'point {k}: decks written (dry)'); continue
    if not ok: continue
    crop = [a.nx*DX*1e6*0.25, a.nx*DX*1e6*0.75]*2
    an = f'{pd}/analysis'
    r = subprocess.run([PY, os.path.join(HERE, 'gp700_analysis.py'), st, '--out', an, '--floor-um', '100',
                        '--crop', *[f'{c:.0f}' for c in (crop[0], crop[1], crop[0], crop[1])], '--D', f"{p['D_deg2_um']*1e6:g}"],
                       stdout=open(f'{pd}/analysis.log', 'w'), stderr=subprocess.STDOUT)
    try:
        sc = score(json.load(open(f'{an}/summary.json')))
    except Exception as e:
        log(f'point {k}: analysis failed: {e}'); sc = None
    rec = dict(point=k, params=p, score=sc, finished=datetime.datetime.now().isoformat())
    open(res_path, 'a').write(json.dumps(rec) + '\n')
    log(f'point {k}: score {sc}')
log('sobol study finished')
