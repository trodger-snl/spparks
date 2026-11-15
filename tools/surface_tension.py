#!/usr/bin/env python3
"""Preliminary surface tension sigma(T, x) of a liquid alloy graded between two end members.

usage: surface_tension.py table A B [--S ppm[,ppm]] [--O ppm[,ppm]] [--s-model sahoo|mills]
                                    [--T lo hi step] [--nx n] [--beta b] [--ideal] [-o file]
       surface_tension.py point A [--S ppm] [--O ppm] [--T K] ...
       surface_tension.py --self-check

A, B   a material card (materials/316L.toml: [card].composition_wt_pct) or element mass
       fractions "Ti=0.90,Al=0.06,V=0.04" (normalised). x is the MASS fraction of B.
table  columns T [K], x, sigma [N/m], dsigma/dT [N/(m K)], dsigma/dx [N/m]; when an end member
       is a card with [flow], the header compares the model with the card's measured values.

Model (docs/MATERIAL_CARDS.md, "Surface tension model"):
  metallic part   Butler equation, Tanaka et al. form (ref tanaka1996), solved for sigma and
                  the surface composition; bulk excess Gibbs energy from Redlich-Kister
                  binaries (Muggianu sum), surface excess = beta * bulk form at x^surf.
  surfactants     Sahoo-DebRoy-McNallan Langmuir term (ref sahoo1988) for dissolved S and O,
                  subtracted from the metallic value; Fe, Ni (and Cr-O) solvents only.
Every constant below carries a ref key into REFS; REFS says what was opened and what is
second-hand. Python 3.11+ standard library only.
"""
import argparse, math, sys, tomllib

R = 8.314462618        # J/(mol K)
NA = 6.02214076e23     # 1/mol
L_PACK = 1.091         # A_i = L N_A^(1/3) V_i^(2/3), close-packed monolayer (tanaka1996 Eq. 2)
BETA = 0.83            # G^E,surf / G^E,bulk for liquid alloys (tanaka1996 Eq. 8)

# "opened": the number was read in the full text for this tool; "second-hand": read in the
# named secondary source; "abstract": read in the publisher's abstract only.
REFS = {
    'sahoo1988': 'P. Sahoo, T. DebRoy, M.J. McNallan, Metall. Trans. B 19 (1988) 483-491, doi 10.1007/BF02657748. '
                 'Opened (author copy, modeling.matse.psu.edu): Eq. 6-11, Tables II and III.',
    'mcnallan1991': 'M.J. McNallan, T. DebRoy, Metall. Trans. B 22 (1991) 557-560, doi 10.1007/BF02654294. '
                    'Opened (scanned author copy): Eq. 1, 8-11, Tables I and II.',
    'zacharia1989': 'T. Zacharia, S.A. David, J.M. Vitek, T. DebRoy, Weld. J. 68 (1989) 499s-509s. Opened: Eq. 1-3, Table 1 '
                    '(heat of adsorption -1.88e5 kJ/kg mole, not the -1.66e5 of sahoo1988; Gamma_s 1.33e-8).',
    'mills2004': 'K.C. Mills, Y. Su, Z. Li, R.F. Brooks, ISIJ Int. 44 (2004) 1661-1668, doi 10.2355/isijinternational.44.1661. '
                 'Opened: Eq. 23-27, Table 4.',
    'tanaka1996': 'T. Tanaka, K. Hack, T. Iida, S. Hara, Z. Metallkd. 87 (1996) 380-389. Opened (Osaka Univ. repository, '
                  'hdl 11094/26520): Eq. 1-3, 5, 8; Table 2 (T_m, V_m, sigma_m of Iida & Guthrie 1988).',
    'butler1932': 'J.A.V. Butler, Proc. R. Soc. A 135 (1932) 348-375. Not opened; the equation is used as written in tanaka1996.',
    'leitner2021': 'T. Leitner, doctoral thesis, Graz University of Technology (2021) (the 316L and IN625 cards cite it). '
                   'Opened: Tables 3.2 (Fe, Fe-Ni), 3.5 (Ni), 3.8, 3.9, 3.15.',
    'brillo2016': 'J. Brillo, Thermophysical Properties of Multicomponent Liquid Alloys, De Gruyter (2016). Opened (a copy '
                  'on a third-party server): Tables A.2, A.3, A.40, B.12, B.14, Eq. 4.31-4.33.',
    'nakamoto2020': 'M. Nakamoto, T. Tanaka, ISIJ Int. 60 (2020) 2141-2146, doi 10.2355/isijinternational.ISIJINT-2019-796. '
                    'Opened: Table 1 (V_m and expansion from Iida & Guthrie 1988, sigma(T) from Mills & Su, Int. Mater. '
                    'Rev. 51 (2006) 329; its Ni row repeats the Fe row, a misprint, and is not used).',
    'assael2006': 'M.J. Assael et al., J. Phys. Chem. Ref. Data 35 (2006) 285-300, doi 10.1063/1.2149380. Opened: density '
                  'correlations of liquid Al and Fe.',
    'ozawa2017': 'S. Ozawa, Y. Kudo, K. Kuribayashi, Y. Watanabe, T. Ishikawa, Mater. Trans. 58 (2017) 1664-1669. Opened: Eq. 5.',
    'reiplinger2022': 'B. Reiplinger, Y. Plevachuk, J. Brillo, J. Mater. Sci. 57 (2022) 21828-21840, '
                      'doi 10.1007/s10853-022-07995-y. Opened: Tables 4, 5 (Ti, V, Ti-V).',
    'reiplinger2024': 'B. Reiplinger, Seimiya, J. Brillo, J. Mater. Sci. (2024), doi 10.1007/s10853-024-09951-4. Opened as the '
                      'submitted manuscript (DLR elib 210252), not the published version: Table 1 (V density).',
    'wessing2017': 'J.J. Wessing, J. Brillo, Metall. Mater. Trans. A 48 (2017) 868-882, doi 10.1007/s11661-016-3886-8. Opened as '
                   'the accepted manuscript (DLR elib 110065): Table 5 (Al-Ti), Table 7.',
    'paradis2002mo': 'P.-F. Paradis, T. Ishikawa, S. Yoda, Int. J. Thermophys. 23 (2002) 555, doi 10.1023/A:1015169721771. '
                     'Abstract only (density of liquid Mo).',
    'paradis2002nb': 'P.-F. Paradis, T. Ishikawa, S. Yoda, Int. J. Thermophys. 23 (2002) 825, doi 10.1023/A:1015459222027 '
                     '(surface tension of Ti, Zr, Nb); J. Mater. Sci. 36 (2001) 5125, doi 10.1023/A:1012477308332 (Nb '
                     'density). Abstracts only.',
    'paradis1999zr': 'P.-F. Paradis, W.-K. Rhim, J. Mater. Res. 14 (1999) 3713. Search-result snippet only (Zr density).',
    'matcalc_ni': 'MatCalc open database mc_ni_v2.036 (TU Wien, Open Database License), '
                  'matcalc.at/images/stories/Download/Database/mc_ni_v2036.tdb. Opened: the L(LIQUID,A,B;k) lines. The '
                  'original assessments it attributes them to (named per pair below) were NOT opened. Cr-Fe, Cr-Ni and '
                  'Fe-Ni agree with brillo2016 Table B.14, Al-Ti with wessing2017 Table 7, Ti-V with reiplinger2022 Table 1.',
    'memory': 'Recalled, NOT verified against any source for this tool. Replace before relying on it.',
}

# Pure liquid elements: sigma = sigma_m + dsdT (T - Tm) [N/m], rho = rho_m + drho (T - Tm) [kg/m^3], M [g/mol].
# Every fit is extrapolated outside its range (deeply undercooled Mo, Nb, Cr in a steel or Ni melt).
ELEMENTS = {
    'Fe': dict(M=55.845, Tm=1811, sigma_m=1.867, dsdT=-3.3e-4, rho_m=7034.96, drho=-0.926, ref='leitner2021 assael2006',
               note='sigma: Leitner Table 3.2, levitation, 99.995 % Fe, 1811-2085 K. Same table: Brooks 1870 - 0.43, '
                    'Brillo & Egry 1920 - 0.397 mN/m (also brillo2016 Table A.3); sahoo1988 uses 1943 - 0.43 (T - 1809)'),
    'Ni': dict(M=58.693, Tm=1728, sigma_m=1.743, dsdT=-3.1e-4, rho_m=7930, drho=-1.01, ref='leitner2021 brillo2016',
               note='sigma: Leitner Table 3.5, 1600-1870 K; Brillo & Egry 1.77 - 3.3e-4 (brillo2016 Table A.3). '
                    'rho: brillo2016 Table A.2 (at 1727 K)'),
    'Cr': dict(M=51.996, Tm=2176, sigma_m=1.710, dsdT=-5.4e-4, rho_m=6287, drho=-0.69, ref='nakamoto2020',
               note='second-hand: sigma is Mills & Su 2006, rho from V_m 8.27e-6 m^3/mol and expansion 1.1e-4 /K of '
                    'Iida & Guthrie, both as tabulated by nakamoto2020. sahoo1988 quotes 2.032 - 5.4e-4 (T - 2133)'),
    'Mn': dict(M=54.938, Tm=1518, sigma_m=1.152, dsdT=-3.5e-4, rho_m=5759, drho=-0.92, ref='nakamoto2020',
               note='second-hand, as Cr (V_m 9.54e-6, expansion 1.6e-4); Iida & Guthrie sigma_m 1.090 (tanaka1996 Table 2)'),
    'Mo': dict(M=95.95, Tm=2896, sigma_m=2.250, dsdT=-2.6e-4, rho_m=9100, drho=-0.60, ref='tanaka1996 memory paradis2002mo',
               note='sigma_m: Iida & Guthrie via tanaka1996 Table 2 (second-hand). dsdT is FROM MEMORY (an electrostatic '
                    'levitation value of about -0.26 mN/(m K)), unverified. rho: abstract only'),
    'Nb': dict(M=92.906, Tm=2742, sigma_m=1.937, dsdT=-1.99e-4, rho_m=7950, drho=-0.23, ref='paradis2002nb',
               note='abstracts only; electrostatic levitation, 2320-2915 K'),
    'Ti': dict(M=47.867, Tm=1941, sigma_m=1.52, dsdT=-1.77e-4, rho_m=4193, drho=-0.238, ref='reiplinger2022 ozawa2017',
               note='sigma: Reiplinger Table 4, levitation (O rose from 0.08 to 0.23 at.% in 90 min). Same table: 1.56 - 1.65e-4 '
                    '(wessing2017), 1.557 - 1.56e-4 (electrostatic levitation); Mills & Su 1.670 - 1.6e-4 (nakamoto2020). '
                    'rho: Ozawa Eq. 5, electrostatic levitation, 1640-2090 K'),
    'Al': dict(M=26.982, Tm=933.47, sigma_m=0.866, dsdT=-1.46e-4, rho_m=2377.23, drho=-0.311, ref='brillo2016 assael2006',
               note='sigma: brillo2016 Table A.3 (levitation). Literature spans 0.87 to 0.98 N/m and -1.5e-4 to -3.5e-4; '
                    'nakamoto2020 uses 0.979 - 2.71e-4. Extrapolated 1000 K above the fits in a Ti melt'),
    'V': dict(M=50.942, Tm=2183, sigma_m=1.94, dsdT=-3.29e-4, rho_m=5510, drho=-0.356, ref='reiplinger2022 reiplinger2024',
              note='sigma: Reiplinger 2022 Table 4/5. rho: 2024 manuscript Table 1 (the 2022 paper has 5550, -0.601)'),
    'Zr': dict(M=91.224, Tm=2128, sigma_m=1.500, dsdT=-1.11e-4, rho_m=6240, drho=-0.29, ref='paradis2002nb paradis1999zr',
               note='sigma: abstract only (1800-2400 K); rho: search snippet only. An older fit of the same group gives '
                    '1.459 - 2.44e-4'),
}

# Liquid Redlich-Kister parameters, key (A, B): G^E = x_A x_B sum_k L_k (x_A - x_B)^k, L_k = a + b T [J/mol],
# listed as (a, b). All read in matcalc_ni; 'from' is the assessment that database attributes them to.
RK = {
    ('Cr', 'Fe'): dict(L=[(-17737, 7.996546), (-1331, 0)], ref='matcalc_ni', src='Frisk & Qiu 1994'),
    ('Fe', 'Ni'): dict(L=[(-16911, 5.1622), (10180, -4.146656)], ref='matcalc_ni', src='Mathon 2009'),
    ('Cr', 'Ni'): dict(L=[(318, -7.3318), (16941, -6.3696)], ref='matcalc_ni', src='B.-J. Lee 1992'),
    ('Mo', 'Ni'): dict(L=[(-39597, 15.935), (-7373, 4.102), (-12123, 5.551)], ref='matcalc_ni', src='Zhou 2005'),
    ('Fe', 'Mo'): dict(L=[(-6973, -0.37), (-9424, 4.502)], ref='matcalc_ni', src='Miettinen 1999 report'),
    ('Cr', 'Mo'): dict(L=[(15810, -6.714), (-6220, 0)], ref='matcalc_ni', src='Miettinen 1999 report'),
    ('Nb', 'Ni'): dict(L=[(-80037.3, -6.31498), (97884.9, -19.01069), (13500, 0)], ref='matcalc_ni',
                       src='Mathon 2009; L2 an unpublished adjustment of the database author'),
    ('Fe', 'Nb'): dict(L=[(-48231.518, 11.2225), (-9786.6216, 5.1445), (29181.806, -14.6036)], ref='matcalc_ni', src='Mathon 2009'),
    ('Cr', 'Nb'): dict(L=[(-10304.91, 0.60618), (-8513.45, 0.50080)], ref='matcalc_ni', src='Costa Neto 1993'),
    ('Fe', 'Mn'): dict(L=[(-3950, 0.489), (1145, 0)], ref='matcalc_ni', src='Huang 1991'),
    ('Cr', 'Mn'): dict(L=[(-15009, 13.6587), (504, 0.9479)], ref='matcalc_ni', src='not looked up'),
    ('Mn', 'Ni'): dict(L=[(-44423.1238, -2.353), (11941.36, -6.8522)], ref='matcalc_ni', src='not looked up'),
    ('Al', 'Ti'): dict(L=[(-118048, 41.972), (-23613, 19.704), (34757, -13.844)], ref='matcalc_ni', src='Witusiewicz 2008/2009'),
    ('Ti', 'V'): dict(L=[(1400, 0), (4100, 0)], ref='matcalc_ni', src='Miettinen 1999 report; COST 507 per reiplinger2022'),
    ('Al', 'V'): dict(L=[(-50725, 9), (-15000, 8)], ref='matcalc_ni',
                      src='Miettinen 1999 report; reiplinger2024 uses -57725 + 9 T, -18000 + 8 T instead'),
    ('Mo', 'Ti'): dict(L=[(-17494.7, -3.57111), (-463.9, 9.0476)], ref='matcalc_ni', src='Chung, Shim, Lee 1999'),
    ('Nb', 'Ti'): dict(L=[(3000, 0)], ref='matcalc_ni', src='Witusiewicz 2009'),
    ('Ti', 'Zr'): dict(L=[(-968, 0)], ref='matcalc_ni', src='Tokunaga 2007'),
    ('Fe', 'Ti'): dict(L=[(-76247, 17.845), (7990, -6.059), (4345, -2.844)], ref='matcalc_ni', src='De Keyzer 2009'),
    ('Cr', 'Ti'): dict(L=[(5250, 0), (1500, 0)], ref='matcalc_ni', src='Schuster & Du 1999'),
    ('Al', 'Mo'): dict(L=[(-100000, 35), (-15000, 6.3)], ref='matcalc_ni', src='Miettinen 1999 report; Saunders COST 507'),
    ('Al', 'Nb'): dict(L=[(-106943, 28.263), (8552, 0), (15000, 0)], ref='matcalc_ni', src='Witusiewicz 2009; Miettinen'),
    ('Mo', 'V'): dict(L=[(17784, 0)], ref='matcalc_ni', src='Frisk 2002'),
    ('Nb', 'Zr'): dict(L=[(10311, 0), (6709, 0)], ref='matcalc_ni', src='not looked up'),
}

# Surfactant sets of sahoo1988 Eq. 6: sigma = sigma_m - A (T - Tm) - R T gamma ln(1 + k1 a exp(-dH/(R T))),
# a in wt%. The paper's kg mole units are converted: gamma [mol/m^2] = 1e3 x [kg mole/m^2], dH [J/mol] =
# [kJ/kg mole]. 'fit': fitted to data in the paper (Tables II, III); 'estimate': the paper's own estimate.
SURFACTANT = {
    ('Fe', 'S'): dict(sigma_m=1.943, A=4.3e-4, Tm=1809, k1=3.18e-3, dH=-1.66e5, gamma=1.30e-5, ref='sahoo1988', kind='fit',
                      note='Eq. 7; Table II prints -166.2e3. zacharia1989 and later weld models use dH = -1.88e5'),
    ('Fe', 'O'): dict(sigma_m=1.943, A=4.3e-4, Tm=1809, k1=1.38e-2, dH=-1.463e5, gamma=2.03e-5, ref='sahoo1988', kind='fit'),
    ('Ni', 'S'): dict(sigma_m=1.845, A=4.3e-4, Tm=1726, k1=3.18e-3, dH=-1.47e5, gamma=1.5e-5, ref='sahoo1988', kind='estimate',
                      note='Eq. 10: k1 copied from Fe-S, dH from an electronegativity plot; the text gives Gamma_s 1.6e-8, the equation 1.5e-8'),
    ('Cr', 'O'): dict(sigma_m=2.032, A=5.4e-4, Tm=2133, k1=1.38e-2, dH=-1.67e5, gamma=5.1e-6, ref='sahoo1988', kind='estimate',
                      note='Eq. 8: k1 copied from Fe-O'),
    ('Ni', 'O'): dict(sigma_m=1.845, A=4.3e-4, Tm=1726, k1=1.38e-2, dH=-1.463e5, gamma=2.03e-5, ref='estimate', kind='estimate',
                      note='NOT in sahoo1988: the Fe-O constants borrowed for Ni so that Ni alloys have an oxygen term'),
}


# ---------------------------------------------------------------- pure elements

def sigma_pure(el, T, data=None):
    e = (data or ELEMENTS)[el]
    return e['sigma_m'] + e['dsdT'] * (T - e['Tm'])


def molar_volume(el, T, data=None):
    """m^3/mol from the linear liquid density fit (extrapolated when undercooled)."""
    e = (data or ELEMENTS)[el]
    return e['M'] * 1e-3 / (e['rho_m'] + e['drho'] * (T - e['Tm']))


def molar_area(el, T, data=None):
    return L_PACK * NA ** (1 / 3) * molar_volume(el, T, data) ** (2 / 3)


# ---------------------------------------------------------------- excess Gibbs energy

def partial_excess(els, x, T, rk=None):
    """Partial excess Gibbs energies [J/mol] of a Redlich-Kister/Muggianu liquid,
    G^E = sum_{i<j} x_i x_j sum_k L_k (x_i - x_j)^k, L_k = a_k + b_k T.
    A pair without parameters is ideal."""
    rk = RK if rk is None else rk
    n = len(els)
    G, dG = 0.0, [0.0] * n
    for i in range(n):
        for j in range(i + 1, n):
            p, sign = rk.get((els[i], els[j])), 1.0
            if p is None:
                p, sign = rk.get((els[j], els[i])), -1.0   # stored as (j, i): odd terms flip
            if p is None:
                continue
            d = x[i] - x[j]
            s = ds = 0.0
            for k, (a, b) in enumerate(p['L']):
                Lk = (a + b * T) * sign ** k
                s += Lk * d ** k
                if k:
                    ds += k * Lk * d ** (k - 1)
            G += x[i] * x[j] * s
            dG[i] += x[j] * s + x[i] * x[j] * ds
            dG[j] += x[i] * s - x[i] * x[j] * ds
    xdG = sum(xi * g for xi, g in zip(x, dG))
    return [G + g - xdG for g in dG]


# ---------------------------------------------------------------- Butler equation

def butler(els, xb, T, beta=BETA, rk=None, data=None):
    """Surface tension [N/m] and surface mole fractions of an n-component liquid:
    sigma = sigma_i + (R T / A_i) ln(xs_i / xb_i) + (beta G_i^E(T, xs) - G_i^E(T, xb)) / A_i
    for every component i, with sum(xs) = 1. rk = {} gives the ideal solution."""
    keep = [i for i, v in enumerate(xb) if v > 0]
    e = [els[i] for i in keep]
    b = [xb[i] / sum(xb[i] for i in keep) for i in keep]
    s0 = [sigma_pure(el, T, data) for el in e]
    A = [molar_area(el, T, data) for el in e]
    gb = partial_excess(e, b, T, rk)
    xs = b[:]
    for _ in range(2000):
        gs = partial_excess(e, xs, T, rk)
        # for fixed excess terms f(sigma) = sum_i xs_i(sigma) - 1 is monotone: bisect
        c = [s0[i] + (beta * gs[i] - gb[i]) / A[i] for i in range(len(e))]
        lo, hi = min(c), max(c)
        for _ in range(60):
            sig = 0.5 * (lo + hi)
            if sum(b[i] * math.exp(A[i] * (sig - c[i]) / (R * T)) for i in range(len(e))) > 1:
                hi = sig
            else:
                lo = sig
        new = [b[i] * math.exp(A[i] * (sig - c[i]) / (R * T)) for i in range(len(e))]
        tot = sum(new)
        new = [v / tot for v in new]
        err = max(abs(u - v) for u, v in zip(new, xs))
        xs = [0.5 * (u + v) for u, v in zip(new, xs)]   # damped fixed point
        if err < 1e-13:
            break
    else:
        # ponytail: damped fixed point; a strongly demixing G^E (several surface states)
        # needs a Newton solve with continuation in x
        raise RuntimeError(f'Butler iteration did not converge for {dict(zip(e, b))} at {T} K')
    full = [0.0] * len(els)
    for i, v in zip(keep, xs):
        full[i] = v
    return sig, full


# ---------------------------------------------------------------- surfactant (Sahoo) term

def langmuir_drop(T, a, k1, dH, gamma):
    """R T Gamma_s ln(1 + k1 a exp(-dH/(R T))) [N/m]: the last term of sahoo1988 Eq. 6.
    a: activity (wt% in the paper's fits), dH [J/mol] (negative), gamma [mol/m^2]."""
    return R * T * gamma * math.log1p(k1 * a * math.exp(-dH / (R * T)))


def sahoo(T, a, pair):
    """sahoo1988 Eq. 6 for a binary metal-surfactant pair of SURFACTANT, e.g. ('Fe', 'S')."""
    p = SURFACTANT[pair]
    return p['sigma_m'] - p['A'] * (T - p['Tm']) - langmuir_drop(T, a, p['k1'], p['dH'], p['gamma'])


def sahoo_dsdT(T, a, pair):
    """sahoo1988 Eq. 11 with the enthalpy of mixing of the solute taken as zero."""
    p = SURFACTANT[pair]
    Ka = p['k1'] * a * math.exp(-p['dH'] / (R * T))
    return -p['A'] - R * p['gamma'] * math.log1p(Ka) - Ka / (1 + Ka) * p['gamma'] * p['dH'] / T


def mills_stainless(T, wtS):
    """mills2004 Eq. 26, stainless steel with about 5 ppm soluble O [N/m]."""
    return 1e-3 * (1840 - 0.4 * (T - 1823)
                   - 0.056 * T * math.log1p(math.exp(28798 / T - 8.5647) * 0.68 * wtS))


def surfactant_drop(w, x, T, S_ppm, O_ppm, s_model='sahoo'):
    """Surface tension drop [N/m] by dissolved S and O in an alloy of mass fractions w
    (mole fractions x). Estimate: the solvent constants are the mole-fraction average
    (dH, ln k1, Gamma) over the elements that have a set; S and O terms simply add."""
    drop = 0.0
    wCr = 100 * w.get('Cr', 0.0)
    for sol, ppm in (('S', S_ppm), ('O', O_ppm)):
        if ppm <= 0:
            continue
        if sol == 'S' and s_model == 'mills':
            drop += 1e-3 * 0.056 * T * math.log1p(math.exp(28798 / T - 8.5647) * 0.68 * ppm * 1e-4)
            continue
        sets = {el: SURFACTANT[(el, sol)] for el in x if (el, sol) in SURFACTANT}
        cover = sum(x[el] for el in sets)
        if cover < 0.5:
            raise ValueError(f'no {sol} surfactant constants for this solvent '
                             f'(only {cover:.0%} of the atoms are {"/".join(sorted(k[0] for k in SURFACTANT if k[1] == sol))}); run with --{sol} 0')
        dH = sum(x[el] * p['dH'] for el, p in sets.items()) / cover
        lnk = sum(x[el] * math.log(p['k1']) for el, p in sets.items()) / cover
        gam = sum(x[el] * p['gamma'] for el, p in sets.items()) / cover
        a = ppm * 1e-4                                           # wt%
        if sol == 'S':
            a *= 10 ** ((-94.2 / T + 0.0396) * wCr)              # mcnallan1991 Eq. 10, Table II
        drop += langmuir_drop(T, a, math.exp(lnk), dH, gam)
    return drop


# ---------------------------------------------------------------- alloys and the table

def mole_fractions(w):
    n = {el: v / ELEMENTS[el]['M'] for el, v in w.items() if v > 0}
    tot = sum(n.values())
    return {el: v / tot for el, v in n.items()}


def sigma_alloy(w, T, S_ppm=0.0, O_ppm=0.0, beta=BETA, ideal=False, s_model='sahoo'):
    x = mole_fractions(w)
    els = sorted(x)
    sig, _ = butler(els, [x[el] for el in els], T, beta, {} if ideal else None)
    return sig - surfactant_drop(w, x, T, S_ppm, O_ppm, s_model)


def mix(wA, wB, xB):
    return {el: (1 - xB) * wA.get(el, 0.0) + xB * wB.get(el, 0.0) for el in set(wA) | set(wB)}


def point(wA, wB, T, xB, S=(0, 0), O=(0, 0), **kw):
    """(sigma, dsigma/dT, dsigma/dx) at temperature T and mass fraction xB of alloy B."""
    def f(T, xB):
        return sigma_alloy(mix(wA, wB, xB), T, S[0] + xB * (S[1] - S[0]), O[0] + xB * (O[1] - O[0]), **kw)
    h = 1e-4
    x0, x1 = max(xB - h, 0.0), min(xB + h, 1.0)     # one-sided at the end members
    return f(T, xB), (f(T + 0.5, xB) - f(T - 0.5, xB)), (f(T, x1) - f(T, x0)) / (x1 - x0)


def parse_alloy(arg):
    """-> (name, mass fractions, card or None). Card ranges 'a-b' count as their midpoint,
    'bal' and '>=' entries share the balance, '<' limits count as zero."""
    if '=' in arg:
        w = {k.strip(): float(v) for k, v in (kv.split('=') for kv in arg.split(','))}
        card, name = None, arg
    else:
        with open(arg, 'rb') as fh:
            card = tomllib.load(fh)
        name = card['card']['material']
        w, bal = {}, []
        for k, v in card['card']['composition_wt_pct'].items():
            el = k.split('_')[0]                    # Nb_plus_Ta -> Nb
            if k == 'note' or el not in ELEMENTS:
                continue
            v = str(v).strip()
            if v.startswith('<'):
                continue
            if v == 'bal' or v.startswith('>'):
                bal.append(el)
            else:
                lim = [float(s) for s in v.split('-')]
                w[el] = sum(lim) / len(lim)
        rest = 100 - sum(w.values())
        for el in bal:
            w[el] = rest / len(bal)
    unknown = sorted(set(w) - set(ELEMENTS))
    if unknown:
        sys.exit(f'{arg}: no pure-element data for {", ".join(unknown)} (have {", ".join(sorted(ELEMENTS))})')
    tot = sum(w.values())
    return name, {el: v / tot for el, v in w.items() if v > 0}, card


def pair(s):
    v = [float(t) for t in s.split(',')]
    return (v[0], v[-1])


def cmd_table(args, out):
    nameA, wA, cardA = parse_alloy(args.A)
    nameB, wB, cardB = parse_alloy(args.B) if args.B else (nameA, wA, None)
    kw = dict(S=pair(args.S), O=pair(args.O), beta=args.beta, ideal=args.ideal, s_model=args.s_model)
    fmt = lambda w: ' '.join(f'{el}={100 * v:.2f}' for el, v in sorted(w.items(), key=lambda t: -t[1]))
    print(f'# surface_tension.py: Butler ({"ideal" if args.ideal else "Redlich-Kister"}, beta {args.beta}) '
          f'+ Langmuir S/O term ({args.s_model})', file=out)
    print(f'# x = 0: {nameA}  wt% {fmt(wA)}  S {kw["S"][0]:g} ppm  O {kw["O"][0]:g} ppm', file=out)
    if args.B:
        print(f'# x = 1: {nameB}  wt% {fmt(wB)}  S {kw["S"][1]:g} ppm  O {kw["O"][1]:g} ppm', file=out)
    missing = sorted({f'{a}-{b}' for w in (wA, wB) for a in w for b in w
                      if a < b and (a, b) not in RK and (b, a) not in RK})
    if missing and not args.ideal:
        print(f'# binaries treated as ideal (no Redlich-Kister set): {" ".join(missing)}', file=out)
    for xB, card in ((0.0, cardA), (1.0, cardB)):
        flow = (card or {}).get('flow', {})
        if 'surface_tension' in flow:
            T = flow['surface_tension']['T_range'][0]
            s, dT, _ = point(wA, wB, T, xB, **kw)
            print(f'# {card["card"]["material"]} at {T:g} K: model sigma {s:.3f} N/m, dsigma/dT {dT:+.2e} N/(m K); '
                  f'card {flow["surface_tension"]["value"]:.3f}, {flow["dsigma_dT"]["value"]:+.2e} '
                  f'(sigma {100 * (s / flow["surface_tension"]["value"] - 1):+.1f} %)', file=out)
    if args.cmd == 'point':
        s, dT, _ = point(wA, wB, args.T[0], 0.0, **kw)
        print(f'{args.T[0]:g} K: sigma {s:.4f} N/m, dsigma/dT {dT:+.3e} N/(m K)', file=out)
        return
    lo, hi, step = args.T
    print('# T[K] x[mass fraction of x=1 alloy] sigma[N/m] dsigma_dT[N/(m K)] dsigma_dx[N/m]', file=out)
    for iT in range(int(round((hi - lo) / step)) + 1):
        for ix in range(args.nx):
            T, xB = lo + iT * step, ix / (args.nx - 1)
            s, dT, dx = point(wA, wB, T, xB, **kw)
            print(f'{T:g} {xB:.6g} {s:.6f} {dT:.6e} {dx:.6e}', file=out)


# ---------------------------------------------------------------- self-check
def self_check():
    close = lambda a, b, tol: abs(a - b) <= tol
    for d in (ELEMENTS, RK, SURFACTANT):                         # every constant has a ref that resolves
        for k, v in d.items():
            assert all(r in REFS or r == 'estimate' for r in v['ref'].split()), (k, v['ref'])

    # --- Redlich-Kister partials: sum x_i G_i = G^E, regular-solution closed form, pair order
    T = 2000.0
    els, x = ['Al', 'Ti', 'V'], [0.2, 0.7, 0.1]
    g = partial_excess(els, x, T)
    G = 0.0
    for (a, b), p in RK.items():
        if a in els and b in els:
            xa, xb = x[els.index(a)], x[els.index(b)]
            G += xa * xb * sum((c0 + c1 * T) * (xa - xb) ** k for k, (c0, c1) in enumerate(p['L']))
    assert close(sum(xi * gi for xi, gi in zip(x, g)), G, 1e-6)
    reg = {('A', 'B'): dict(L=[(-20000.0, 0.0)])}
    assert close(partial_excess(['A', 'B'], [0.3, 0.7], T, reg)[0], -20000 * 0.7 ** 2, 1e-9)
    g1, g2 = partial_excess(['Fe', 'Ni'], [0.3, 0.7], T), partial_excess(['Ni', 'Fe'], [0.7, 0.3], T)
    assert close(g1[0], g2[1], 1e-9) and close(g1[1], g2[0], 1e-9)

    # --- Butler: pure-element limits, ideal closed form (equal areas), residual of every equation
    for xb, el in (([1.0, 0.0], 'Fe'), ([0.0, 1.0], 'Ni')):
        assert butler(['Fe', 'Ni'], xb, 1900.0)[0] == sigma_pure(el, 1900.0)
    assert close(butler(['Fe', 'Ni'], [1 - 1e-9, 1e-9], 1900.0)[0], sigma_pure('Fe', 1900.0), 1e-6)
    fake = {'A': dict(M=50.0, Tm=1800, sigma_m=1.9, dsdT=-3e-4, rho_m=7000.0, drho=0.0),
            'B': dict(M=50.0, Tm=1800, sigma_m=1.2, dsdT=-2e-4, rho_m=7000.0, drho=0.0)}
    A = molar_area('A', T, fake)
    for xB in (0.01, 0.3, 0.9):
        exact = -R * T / A * math.log(sum(xi * math.exp(-A * sigma_pure(e, T, fake) / (R * T))
                                          for xi, e in ((1 - xB, 'A'), (xB, 'B'))))
        assert close(butler(['A', 'B'], [1 - xB, xB], T, rk={}, data=fake)[0], exact, 1e-10)
    sig, xs = butler(els, x, T)
    gb, gs = partial_excess(els, x, T), partial_excess(els, xs, T)
    assert close(sum(xs), 1, 1e-12)
    for i, el in enumerate(els):
        Ai = molar_area(el, T)
        assert close(sigma_pure(el, T) + R * T / Ai * math.log(xs[i] / x[i]) + (BETA * gs[i] - gb[i]) / Ai, sig, 1e-9), el
    assert xs[0] > x[0]                                          # Al (lowest sigma) segregates to the surface

    # --- sahoo1988
    assert close(sahoo(1809, 0.0, ('Fe', 'S')), 1.943, 1e-12) and close(sahoo_dsdT(2500, 0.0, ('Fe', 'S')), -4.3e-4, 1e-12)
    assert close(sahoo(1873, 0.0, ('Fe', 'S')), 1.915, 1e-3)     # mcnallan1991 Table I: sigma_Fe(1873 K) = 1.915 N/m
    # Eq. 7 at 1923 K, a_S = 0.1 by hand: 1.943 - 0.049 - 8.3145*1923*1.3e-5 ln(1 + 0.00318*0.1*e^10.383) = 1.391
    assert close(sahoo(1923, 0.1, ('Fe', 'S')), 1.391, 2e-3)
    for pr, a in ((('Fe', 'S'), 0.01), (('Fe', 'O'), 0.02), (('Ni', 'S'), 0.01)):          # Eq. 11 = d(Eq. 6)/dT
        fd = (sahoo(2000.5, a, pr) - sahoo(1999.5, a, pr))
        assert close(sahoo_dsdT(2000, a, pr), fd, 1e-8), pr
    assert sahoo_dsdT(1900, 0.02, ('Fe', 'S')) > 0 > sahoo_dsdT(2600, 0.02, ('Fe', 'S'))    # Fig. 8(b): sign change
    # Sec. III.C: below 0.002 wt% O dsigma/dT is "practically always negative"
    assert all(sahoo_dsdT(T, 0.001, ('Fe', 'O')) < 0 for T in range(1809, 3000, 50))

    def t_sign_change(a, dH):
        p = dict(SURFACTANT[('Fe', 'S')], dH=dH)
        lo, hi = 1809.0, 4000.0
        for _ in range(50):
            m = 0.5 * (lo + hi)
            Ka = p['k1'] * a * math.exp(-dH / (R * m))
            if -p['A'] - R * p['gamma'] * math.log1p(Ka) - Ka / (1 + Ka) * p['gamma'] * dH / m > 0:
                lo = m
            else:
                hi = m
        return lo
    # zacharia1989: 90 ppm S positive "until about 2200 K", 240 ppm until "approximately 2450 K" (their Fig. 1),
    # with dH = -1.88e5. The sahoo1988 value -1.66e5 puts the change 250 K lower.
    tz = t_sign_change(0.009, -1.88e5)
    assert close(tz, 2200, 25), tz
    print(f'Fe-S sign change of dsigma/dT, 90 ppm S: {tz:.0f} K with dH -1.88e5 (zacharia1989: about 2200 K), '
          f'{t_sign_change(0.009, -1.66e5):.0f} K with dH -1.66e5 (sahoo1988)')
    print(f'                                240 ppm S: {t_sign_change(0.024, -1.88e5):.0f} K with dH -1.88e5 '
          f'(zacharia1989 Fig. 1: approximately 2450 K), {t_sign_change(0.024, -1.66e5):.0f} K with -1.66e5')

    # --- mills2004 Eq. 26 reproduces its Table 4 (mN/m)
    for T, ppm, tab in ((1800, 10, 1772), (1800, 20, 1729), (1900, 20, 1736), (1950, 30, 1713)):
        assert close(1e3 * mills_stainless(T, ppm * 1e-4), tab, 1.0), (T, ppm)

    # --- card parsing
    import os
    cards = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'materials')
    name, w, card = parse_alloy(os.path.join(cards, 'IN625.toml'))
    assert name == 'IN625' and close(sum(w.values()), 1, 1e-12) and close(w['Nb'], 0.0365, 1e-9) and 'Fe' not in w
    w64 = parse_alloy(os.path.join(cards, 'Ti6Al4V.toml'))[1]
    assert close(w64['Ti'], 0.89875, 1e-9)

    # --- published measurements (printed, loosely asserted: a mismatch here is a finding, not a bug)
    print('\nmodel vs published measurements (metallic Butler part only, no S/O term):')
    print(f'{"alloy":<34}{"T [K]":>7}{"meas.":>8}{"model":>8}{"ideal":>8}{"diff":>8}  source')
    at = lambda **x: {el: v * ELEMENTS[el]['M'] for el, v in x.items()}                    # at. fractions -> mass
    rows = [('Ti87.5 V12.5 (at.%)', at(Ti=.875, V=.125), 2100, 1.473, 'reiplinger2022 Table 5'),
            ('Ti75 V25', at(Ti=.75, V=.25), 2100, 1.506, 'reiplinger2022 Table 5'),
            ('Ti50 V50', at(Ti=.5, V=.5), 2100, 1.580, 'reiplinger2022 Table 5'),
            ('Al10 Ti90', at(Al=.1, Ti=.9), 1950, 1.50, 'wessing2017 Table 5 (1.53, 1.47)'),
            ('Al20 Ti80', at(Al=.2, Ti=.8), 1950, 1.315, 'wessing2017 Table 5 (1.35, 1.28)'),
            ('Al50 Ti50', at(Al=.5, Ti=.5), 1757, 1.07, 'wessing2017 Table 5'),
            ('Ti-6Al-4V (wt%)', w64, 1933, 1.493, 'Ti6Al4V card (Mohr 2020)'),
            ('Fe50 Ni50 (at.%)', at(Fe=.5, Ni=.5), 1713, 1.91, 'brillo2016 Table A.40'),
            ('Fe-40Ni (wt%)', dict(Fe=.6, Ni=.4), 1724, 1.741, 'leitner2021 Table 3.2'),
            ('Fe50 Ni50 (Yeum model, at.%)', at(Fe=.5, Ni=.5), 1873, 1.866, 'mcnallan1991 Table I (calculated)'),
            ('Fe50 Cr50 (Yeum model, at.%)', at(Fe=.5, Cr=.5), 1873, 2.027, 'mcnallan1991 Table I (calculated)')]
    for name, w, T, meas, src in rows:
        tot = sum(w.values())
        w = {el: v / tot for el, v in w.items()}
        s, si = sigma_alloy(w, T), sigma_alloy(w, T, ideal=True)
        print(f'{name:<34}{T:>7}{meas:>8.3f}{s:>8.3f}{si:>8.3f}{100 * (s / meas - 1):>+7.1f}%  {src}')
        assert abs(s / meas - 1) < 0.25, name
    print('\nself-check passed')


def main():
    if '--self-check' in sys.argv:
        return self_check()
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=['table', 'point'])
    ap.add_argument('A')
    ap.add_argument('B', nargs='?')
    ap.add_argument('--S', default='0', help='dissolved sulfur, ppm by mass: one value, or "A,B"')
    ap.add_argument('--O', default='0', help='dissolved (soluble, not total) oxygen, ppm by mass')
    ap.add_argument('--s-model', choices=['sahoo', 'mills'], default='sahoo',
                    help='sulfur term: sahoo1988 Fe-S/Ni-S + Cr activity, or the mills2004 stainless fit')
    ap.add_argument('--T', type=float, nargs='+', default=None, help='table: lo hi step [K]; point: T')
    ap.add_argument('--nx', type=int, default=11)
    ap.add_argument('--beta', type=float, default=BETA)
    ap.add_argument('--ideal', action='store_true', help='ideal solution (ignore the Redlich-Kister sets)')
    ap.add_argument('-o', '--output')
    args = ap.parse_args()
    if args.cmd == 'table' and not args.B:
        ap.error('table needs two end members')
    args.T = args.T or ([1600.0, 3000.0, 50.0] if args.cmd == 'table' else [1900.0])
    if args.cmd == 'table' and len(args.T) != 3:
        ap.error('--T lo hi step')
    try:
        if args.output:
            with open(args.output, 'w') as out:
                cmd_table(args, out)
        else:
            cmd_table(args, sys.stdout)
    except ValueError as e:
        sys.exit(f'error: {e}')


if __name__ == '__main__':
    main()
