#!/usr/bin/env python3
"""Material parameter cards (docs/MATERIAL_CARDS.md): lint and front-velocity fits.

usage: material_card.py lint <card.toml> [--doi]
       material_card.py fit-front <card.toml> [--degree N]
       material_card.py selftest

lint       schema, canonical units, increasing tables, every value has a ref that
           resolves, changelog agrees with [card] version/modified, solid_front_vel
           reproduces front_velocity_source; --doi also checks each DOI resolves.
fit-front  least-squares polynomial (SPPARKS order, highest power first, in
           undercooling dT = liquidus - T) to [microstructure].front_velocity_source
           over [0, liquidus - solidus]; prints a solid_front_vel line and the error.
Python 3.11+ standard library only.
"""
import sys, tomllib, urllib.request

SCHEMA = 1
REF_LITERALS = {'estimate', 'calibration'}
# Canonical unit of every record SPPARKS reads, by path; process.<name>.* by suffix.
UNITS = {
    'thermal.density': 'kg/m^3', 'thermal.solidus': 'K', 'thermal.liquidus': 'K',
    'thermal.latent_fusion': 'J/kg', 'thermal.emissivity': '1',
    'thermal.conductivity': 'W/(m K)', 'thermal.heat_capacity': 'J/(kg K)',
    'thermal.linear.conductivity': 'W/(m K)', 'thermal.linear.heat_capacity': 'J/(kg K)',
    'thermal.linear.T_ref': 'K', 'powder.conductivity': 'W/(m K)', 'powder.packing': '1',
    'optical.absorptivity': '1', 'microstructure.solid_front_vel': 'm/s',
    'microstructure.front_velocity_source': 'm/s',
    'microstructure.nucleation_density': '1/m^3', 'microstructure.critical_undercooling': 'K',
    'microstructure.undercooling_deviation': 'K', 'microstructure.mean_nuclei_volume': 'm^3',
    'microstructure.nuclei_volume_deviation': 'm^3',
}
PROCESS_UNITS = {'power': 'W', 'speed': 'm/s', 'hatch': 'm', 'layer_thickness': 'm', 'spot': 'm',
                 'source_depth': 'm', 'preheat': 'K', 'absorptivity': '1', 'convection': 'W/(m^2 K)',
                 'validation': 'm'}
SPOT_CONVENTIONS = {'1/e radius', '1/e2 radius', '1/e2 diameter', 'D4sigma', 'FWHM'}
UNCERTAINTY_TYPES = {'1sigma', '2sigma', 'bounds', 'relative_1sigma'}


def polyval(c, x):
    return sum(a * x ** (len(c) - 1 - i) for i, a in enumerate(c))


def front_source(rec, dT):
    """V(dT) of a literature front-velocity record: power_law a*dT^n, or points."""
    if rec['form'] == 'power_law':
        return rec['a'] * dT ** rec['n']
    if rec['form'] == 'points':   # piecewise linear through (dT, V)
        xs, vs = rec['dT'], rec['V']
        if dT <= xs[0]: return vs[0]
        for i in range(1, len(xs)):
            if dT <= xs[i]:
                return vs[i-1] + (dT - xs[i-1]) / (xs[i] - xs[i-1]) * (vs[i] - vs[i-1])
        return vs[-1]
    raise ValueError(f"front_velocity_source form {rec['form']!r} is power_law or points")


def fit_poly(f, lo, hi, degree, n=200):
    """Least-squares polynomial through f on [lo, hi]; normal equations, highest power first."""
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    m = degree + 1
    A = [[sum(x ** (2 * degree - i - j) for x in xs) for j in range(m)] for i in range(m)]
    b = [sum(f(x) * x ** (degree - i) for x in xs) for i in range(m)]
    for c in range(m):                          # Gauss-Jordan with partial pivoting
        p = max(range(c, m), key=lambda r: abs(A[r][c]))
        A[c], A[p], b[c], b[p] = A[p], A[c], b[p], b[c]
        for r in range(m):
            if r != c:
                k = A[r][c] / A[c][c]
                A[r] = [a - k * q for a, q in zip(A[r], A[c])]
                b[r] -= k * b[c]
    return [b[i] / A[i][i] for i in range(m)]


def max_rel_err(c, f, lo, hi, n=200):
    vmax = max(abs(f(lo + (hi - lo) * i / (n - 1))) for i in range(n))
    return max(abs(polyval(c, x) - f(x)) / vmax for x in (lo + (hi - lo) * i / (n - 1) for i in range(n)))


def lint(card, check_doi=False):
    """List of problems (empty = clean)."""
    errs = []
    head = card.get('card', {})
    if head.get('schema_version') != SCHEMA: errs.append(f'card.schema_version must be {SCHEMA}')
    for k in ('material', 'version', 'created', 'modified', 'status'):
        if k not in head: errs.append(f'card.{k} missing')
    if head.get('status') not in (None, 'draft', 'reviewed'): errs.append('card.status is draft or reviewed')
    log = card.get('changelog', [])
    if not log: errs.append('changelog is empty')
    elif (log[-1].get('version'), log[-1].get('date')) != (head.get('version'), head.get('modified')):
        errs.append('last changelog entry must match card.version and card.modified')
    refs = card.get('references', {})
    for key, r in refs.items():
        if 'citation' not in r: errs.append(f'references.{key}: citation missing')

    def walk(tbl, path):
        for k, v in tbl.items():
            p = f'{path}.{k}' if path else k
            if isinstance(v, dict) and ('unit' in v or 'ref' in v or 'value' in v or 'coeffs' in v):
                yield p, v
            elif isinstance(v, dict):
                yield from walk(v, p)

    for sec in ('thermal', 'powder', 'optical', 'microstructure', 'process'):
        for p, rec in walk(card.get(sec, {}), sec):
            parts = p.split('.')
            unit = PROCESS_UNITS.get(parts[-1]) if sec == 'process' else UNITS.get(p)
            if unit is None and sec != 'process': errs.append(f'{p}: unknown record (not in the schema)')
            elif unit is not None and rec.get('unit') != unit: errs.append(f'{p}: unit must be {unit!r}')
            r = rec.get('ref')
            keys = r if isinstance(r, list) else [r] if r else []
            if not keys: errs.append(f'{p}: ref missing')
            for key in keys:
                if key not in refs and key not in REF_LITERALS: errs.append(f'{p}: ref {key!r} not in [references]')
            if 'uncertainty_type' in rec and rec['uncertainty_type'] not in UNCERTAINTY_TYPES:
                errs.append(f'{p}: uncertainty_type is one of {sorted(UNCERTAINTY_TYPES)}')
            if 'T' in rec:
                T = rec['T']
                if len(T) != len(rec.get('value', [])): errs.append(f'{p}: T and value differ in length')
                if any(b <= a for a, b in zip(T, T[1:])): errs.append(f'{p}: T must increase strictly')
            if parts[-1] == 'spot' and rec.get('convention') not in SPOT_CONVENTIONS:
                errs.append(f'{p}: convention is one of {sorted(SPOT_CONVENTIONS)}')

    th, mi = card.get('thermal', {}), card.get('microstructure', {})
    if 'solidus' in th and 'liquidus' in th and th['solidus']['value'] >= th['liquidus']['value']:
        errs.append('thermal: solidus must be below liquidus')
    if 'solid_front_vel' in mi and 'front_velocity_source' in mi:
        sf, src = mi['solid_front_vel'], mi['front_velocity_source']
        lo, hi = sf.get('fit_range_K', [0, th['liquidus']['value'] - th['solidus']['value']])
        e = max_rel_err(sf['coeffs'], lambda x: front_source(src, x), lo, hi)
        if e > sf.get('max_rel_err', 0.02):
            errs.append(f'microstructure.solid_front_vel: {e:.3g} max relative error vs front_velocity_source')
    if check_doi:
        for key, r in refs.items():
            if 'doi' not in r: continue
            try:
                urllib.request.urlopen(urllib.request.Request(f"https://doi.org/{r['doi']}", method='HEAD'), timeout=20)
            except Exception as ex:
                errs.append(f'references.{key}: DOI {r["doi"]} does not resolve ({ex})')
    return errs


def load(path):
    with open(path, 'rb') as f:
        return tomllib.load(f)


def selftest():
    import os
    card = load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'materials', '316L.toml'))
    assert lint(card) == [], lint(card)
    bad = dict(card, thermal=dict(card['thermal'], emissivity={'value': 0.4, 'unit': '1', 'ref': 'nobody2099'}))
    assert any('nobody2099' in e for e in lint(bad)), lint(bad)
    bad = dict(card, thermal=dict(card['thermal'], density={'value': 7.9, 'unit': 'g/cm^3', 'ref': 'estimate'}))
    assert any('unit must be' in e for e in lint(bad))
    src = {'form': 'power_law', 'a': 1e-4, 'n': 2.0}       # a quadratic is fitted exactly
    c = fit_poly(lambda x: front_source(src, x), 0, 33, 3)
    assert max_rel_err(c, lambda x: front_source(src, x), 0, 33) < 1e-9, c
    pts = {'form': 'points', 'dT': [0, 10, 20], 'V': [0, 1, 3]}
    assert front_source(pts, 15) == 2.0 and front_source(pts, 30) == 3
    print('selftest ok')


def main(argv):
    if argv[:1] == ['selftest']: return selftest()
    if len(argv) >= 2 and argv[0] == 'lint':
        errs = lint(load(argv[1]), '--doi' in argv)
        for e in errs: print(e)
        print(f'{argv[1]}: ' + ('ok' if not errs else f'{len(errs)} problem(s)'))
        sys.exit(1 if errs else 0)
    if len(argv) >= 2 and argv[0] == 'fit-front':
        card = load(argv[1]); deg = int(argv[argv.index('--degree') + 1]) if '--degree' in argv else 3
        th, src = card['thermal'], card['microstructure']['front_velocity_source']
        hi = th['liquidus']['value'] - th['solidus']['value']
        f = lambda x: front_source(src, x)
        c = fit_poly(f, 0.0, hi, deg)
        print(f'solid_front_vel = {{ coeffs = [{", ".join(f"{a:.6g}" for a in c)}], unit = "m/s", dT_unit = "K", '
              f'ref = {src["ref"]!r}, fit_of = "front_velocity_source", fit_range_K = [0, {hi:g}], '
              f'max_rel_err = {max_rel_err(c, f, 0.0, hi):.2g} }}'.replace("'", '"'))
        return
    sys.exit(__doc__)


if __name__ == '__main__':
    main(sys.argv[1:])
