"""Schematics for docs/FREE_SURFACE_LBM.md (2-D x-z slices of the 3-D rules).
python3 docs/figures/make_fs_figures.py  ->  docs/figures/fs_*.svg (matplotlib; text stays text)"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 11, "svg.fonttype": "none", "font.family": "DejaVu Sans"})
OUT = os.path.dirname(os.path.abspath(__file__))
FLUID, WALL, GAS, IFC = 0, 1, 2, 3
C_LIQ, C_IFC, C_WALL, C_GAS = "#3d7cc9", "#cfe2f7", "#9a9a9a", "#ffffff"
C_REC, C_BB, C_PIN, C_EX = "#e07b00", "#555555", "#8e44ad", "#7d3c98"
NAME = {FLUID: "FLUID", WALL: "WALL (solid)", GAS: "GAS", IFC: "INTERFACE"}


def cell(ax, i, k, t, eps=None, text=None, lw=0.8, tcol="k", fs=9.0):
    """One cell at column i, layer k; an interface cell is shaded to its fill from the bottom."""
    ax.add_patch(Rectangle((i, k), 1, 1, fc={FLUID: C_LIQ, WALL: C_WALL, GAS: C_GAS, IFC: C_IFC}[t],
                           ec="#444", lw=lw))
    if t == IFC and eps is not None:
        e = min(max(eps, 0.0), 1.0)
        ax.add_patch(Rectangle((i, k), 1, e, fc=C_LIQ, ec="none", alpha=0.55))
        ax.add_patch(Rectangle((i, k), 1, 1, fc="none", ec="#444", lw=lw))
    if text is not None:
        ax.text(i + 0.5, k + 0.5, text, ha="center", va="center", fontsize=fs, color=tcol)


def legend(ax, y, x0=0.0, items=(FLUID, IFC, GAS, WALL), extra=()):
    """A row of type swatches under a grid, in data units."""
    x = x0
    for t in items:
        cell(ax, x, y, t, eps=0.5 if t == IFC else None)
        ax.text(x + 1.25, y + 0.5, NAME[t] + (" (shaded to its fill)" if t == IFC else ""), va="center", fontsize=10)
        x += 2.0 + 0.155 * len(NAME[t]) + (2.6 if t == IFC else 0)
    for lab, col, ls in extra:
        ax.plot([x, x + 0.9], [y + 0.5, y + 0.5], color=col, ls=ls, lw=2.2)
        ax.text(x + 1.15, y + 0.5, lab, va="center", fontsize=10)
        x += 1.6 + 0.155 * len(lab)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 1. cells and columns
def fig_cells():
    nc, nr = 15, 9
    h = [4.0, 4.0, 4.0, 5.45, 6.1, 6.5, 6.7, 6.6, 6.3, 5.8, 5.25, 3.0, 3.0, 3.0, 3.0]   # metal height per column
    wall_top = [4, 4, 4, 2, 1, 1, 0, 0, 1, 1, 2, 3, 3, 3, 3]                            # solid below this layer
    T = np.full((nc, nr), GAS); E = np.zeros((nc, nr))
    for i in range(nc):
        for k in range(nr):
            if k < wall_top[i]: T[i, k] = WALL
            elif k < int(h[i]): T[i, k] = FLUID; E[i, k] = 1
            elif k < h[i]: T[i, k] = IFC; E[i, k] = h[i] - k
    for i in range(nc):                          # liquid touching gas (8 neighbours in the slice) is interface
        for k in range(nr):
            if T[i, k] == FLUID and any(0 <= i + a < nc and 0 <= k + b < nr and T[i + a, k + b] == GAS
                                        for a in (-1, 0, 1) for b in (-1, 0, 1)):
                T[i, k] = IFC
    fig, ax = plt.subplots(figsize=(12, 8.2))
    for i in range(nc):
        for k in range(nr):
            cell(ax, i, k, T[i, k], E[i, k] if T[i, k] == IFC else None)
    for i in range(nc):                          # the height function
        ax.plot([i, i + 1], [h[i], h[i]], color="k", lw=2.4, solid_capstyle="butt")
    # column 6: h, its force cells, the top full cell
    i = 6; kt = int(h[i]); kb = kt - 1
    ax.add_patch(Rectangle((i + 0.04, kb + 0.04), 0.92, kt - kb + 0.92, fc="none", ec="#c0392b", lw=2.2, ls="--"))
    ax.annotate("column (i, j): h = k_f + 1 + Σ fill of the interface cells above k_f\n"
                "force cells skb..skt (red dashed): the surface force f_s / M acts on each\n"
                "(k_f = skb here: the top full cell)",
                xy=(i + 0.5, h[i] + 0.1), xytext=(6.3, nr + 0.45), fontsize=10.5, ha="left", color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b"))
    # contact angle at the dry, lower rim (column 2 beside column 3)
    hc, hn, tq = h[3], h[2], np.tan(np.radians(20))
    ax.plot([2, 3], [hc - tq] * 2, color=C_REC, lw=2.4, ls="--")
    ax.plot([2, 3], [hn + 0.03] * 2, color=C_PIN, lw=2.4, ls=":")
    ax.annotate("hot dry solid: ghost height h_c - tan(fs_angle)\n(liquid pushed out at the angle; 20° by default)",
                xy=(2.5, hc - tq), xytext=(-5.0, 9.6), fontsize=10, color=C_REC,
                arrowprops=dict(arrowstyle="->", color=C_REC))
    ax.annotate("cold dry solid (below fs_pin):\nghost at its own height = 90°, held back",
                xy=(2.5, hn), xytext=(-5.0, 5.4), fontsize=10, color=C_PIN,
                arrowprops=dict(arrowstyle="->", color=C_PIN))
    ax.text(12.0, 1.4, "bare plate ahead:\nthe same rule\n(solid, lower, dry)", fontsize=9.5, ha="center")
    ax.text(1.5, 2.0, "frozen rim\n(solid on top:\nskt = -1)", fontsize=9.5, ha="center", color="w")
    ax.text(nc + 0.2, h[nc - 1] + 2.2, "gas: a pressure,\nnot a fluid", fontsize=10)
    ax.text(6.5, 0.45, "pool bottom: liquid against solid (bounce-back)", fontsize=9.5, ha="center", color="w")
    legend(ax, -1.6, extra=(("height function h", "k", "-"),))
    ax.set_xlim(-5.2, nc + 2.6); ax.set_ylim(-2, nr + 2.4); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Step 1: cell types and the column height function (x-z slice; z up)", loc="left")
    save(fig, "fs_cells.svg")


# ---------------------------------------------------------------- 2. link treatment
def fig_links():
    nb = {(-1, -1): (FLUID, None), (0, -1): (FLUID, None), (1, -1): (WALL, None), (-1, 0): (IFC, 0.4),
          (1, 0): (GAS, None), (-1, 1): (GAS, None), (0, 1): (GAS, None), (1, 1): (IFC, 0.2)}
    fig, ax = plt.subplots(figsize=(15, 7.2))
    s = 3.2                                       # cell size (drawn)
    o = np.array([0.3, 0.3])
    def box(a, b, t, e, lab):
        x, y = o + s * np.array([a + 1, b + 1])
        ax.add_patch(Rectangle((x, y), s, s, fc={FLUID: C_LIQ, WALL: C_WALL, GAS: C_GAS, IFC: C_IFC}[t], ec="#444"))
        if t == IFC: ax.add_patch(Rectangle((x, y), s, s * e, fc=C_LIQ, ec="none", alpha=0.55))
        ax.text(x + 0.12, y + s - 0.12, lab, fontsize=9, va="top")
    box(0, 0, IFC, 0.6, "x: INTERFACE\nε 0.6")
    for (a, b), (t, e) in nb.items():
        box(a, b, t, e, NAME[t].split()[0] + (f"\nε {e}" if e else ""))
    ctr = o + s * np.array([1.5, 1.5])
    def arrow(a, b, col, ls="-", lab=None):
        src = o + s * np.array([a + 1.5, b + 1.5])
        d = ctr - src; u = d / np.hypot(*d)
        ax.add_patch(FancyArrowPatch(tuple(src + 0.42 * s * u), tuple(ctr - 0.3 * s * u), arrowstyle="-|>",
                                     mutation_scale=18, color=col, lw=2.4, ls=ls))
        if lab: ax.text(*(src - 0.08 * s * u), lab, fontsize=9, color=col, ha="center", va="center",
                        bbox=dict(fc="w", ec="none", alpha=0.85, pad=1.5))
    arrow(0, -1, C_LIQ, lab="streamed\nΔm = f_in - f_out")
    arrow(-1, -1, C_LIQ, lab="streamed, diagonal\n(liquid between)")
    arrow(-1, 0, C_LIQ, lab="streamed\nΔm × ½(ε_x + ε_n)")
    arrow(1, -1, C_BB, lab="bounce-back\n(no mass)")
    arrow(1, 0, C_REC, lab="rebuilt from gas\n(no mass)")
    arrow(0, 1, C_REC, lab="rebuilt")
    arrow(-1, 1, C_REC, lab="rebuilt")
    arrow(1, 1, C_REC, ls="--", lab="liquid, but only\nacross gas: rebuilt")
    txt = ("Pull form: one thread per cell gathers what arrives on each of the 18 links\n"
           "(a 2-D slice shows 8). A link is treated by where it comes from:\n\n"
           "liquid (FLUID, INTERFACE)  streamed; the cell gains\n"
           "      Δm_q = φ (f_q,in - f_q̄,out),  φ = ½(ε_x + ε_n) between two\n"
           "      interface cells, else 1. The same φ-weighted mass goes to\n"
           "      the six face fluxes that melt_advect uses.\n"
           "GAS                        rebuilt (Körner 2005), no mass:\n"
           "      f_q = f^eq_q(ρ_G, u) + f^eq_q̄(ρ_G, u) - f_q̄\n"
           "      ρ_G - 1 = 3 (p_gas - σ κ)        from step 1\n"
           "      under a column's surface (a void): ρ_G - 6σ more\n"
           "WALL, block edge          half-way bounce-back, no mass\n\n"
           "A diagonal from liquid needs a liquid cell between to stream\n"
           "(else gas between: rebuilt; walls between: bounce-back).\n"
           "In an interface cell every force is scaled by clamp(ε, 0, 1),\n"
           "and τ is raised to fs_tau (1.0).")
    txt = re.sub(r" {2,}", lambda m: "\u00a0" * len(m.group()), txt)   # SVG collapses runs of spaces
    ax.text(o[0] + 3 * s + 0.6, o[1] + 3 * s, txt, fontsize=10.5, va="top", family="DejaVu Sans Mono")
    ax.set_xlim(0, 3 * s + 15.5); ax.set_ylim(0, o[1] + 3 * s + 0.3); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Step 2: what arrives at an interface cell x along each link", loc="left")
    save(fig, "fs_links.svg")


# ---------------------------------------------------------------- 3. conversion
def fig_convert():
    nc, nr, kap = 9, 4, 1e-3
    T = np.full((nc, nr), GAS); M = np.zeros((nc, nr))
    T[:, 0] = FLUID; M[:, 0] = 1.0
    T[:, 1] = IFC; M[:, 1] = [0.5, 0.7, 0.9, 1.06, 0.8, 0.6, 0.4, -0.03, 0.5]
    nbs = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    def nb(i, k):
        return [(i + a, k + b) for a, b in nbs if 0 <= i + a < nc and 0 <= k + b < nr]
    # 1. mark (as FlowFsMark, rho = 1)
    mk = np.zeros((nc, nr), int)
    for i in range(nc):
        for k in range(nr):
            if T[i, k] != IFC: continue
            nf = any(T[p] == IFC and M[p] > 1 + kap for p in nb(i, k))
            held = len(nb(i, k)) < 8 or any(T[p] in (FLUID, WALL) for p in nb(i, k))
            if M[i, k] > 1 + kap: mk[i, k] = 1
            elif not nf and (not held or M[i, k] < -kap): mk[i, k] = 2
    # 2. retype
    T2 = T.copy(); M2 = M.copy()
    for i in range(nc):
        for k in range(nr):
            if T[i, k] == IFC: T2[i, k] = FLUID if mk[i, k] == 1 else GAS if mk[i, k] == 2 else IFC
            elif T[i, k] in (GAS, FLUID):
                want = 1 if T[i, k] == GAS else 2
                if any(mk[p] == want for p in nb(i, k)):
                    T2[i, k] = IFC; M2[i, k] = 0.0 if T[i, k] == GAS else 1.0
    # 3. excess: equal shares to the neighbours that are interface in type2
    M3 = M2.copy(); share = {}
    for i in range(nc):
        for k in range(nr):
            if not mk[i, k]: continue
            ex = M[i, k] - (1.0 if mk[i, k] == 1 else 0.0)
            rec = [p for p in nb(i, k) if T2[p] == IFC]
            share[(i, k)] = (ex / len(rec), rec)
            for p in rec: M3[p] += ex / len(rec)
    for (i, k), t in np.ndenumerate(T2):
        if t != IFC: M3[i, k] = 1.0 if t == FLUID else 0.0
    fig, axs = plt.subplots(4, 1, figsize=(11, 15.5))
    lab = {1: "FILL", 2: "EMPTY"}
    rows = [("Step 3, conversion (x-z slice). Before: interface masses m (ρ = 1); one cell overfilled, one overdrawn", T, M, None),
            ("1. FlowFsMark: m > (1 + κ)ρ → FILL;  m < -κρ (or a lone droplet) → EMPTY, unless a neighbour fills",
             T, M, mk),
            ("2. FlowFsRetype (into type2): FILL → FLUID, its GAS neighbours → INTERFACE at m = 0 (f = f^eq of the\n"
             "    liquid neighbours' mean ρ, u); EMPTY → GAS, its FLUID neighbours → INTERFACE at full mass", T2, M2, None),
            ("3. FlowFsExcess: each converted cell's excess (m - ρ, or m) in equal shares to the neighbours that are\n"
             "    INTERFACE in type2 (gathered by the receivers: no atomics); fill ε = m/ρ", T2, M3, None)]
    for ax, (title, Tt, Mt, mm) in zip(axs, rows):
        for i in range(nc):
            for k in range(nr):
                t = Tt[i, k]
                txt = f"{Mt[i, k]:.3g}" if t == IFC else ""
                if mm is not None and mm[i, k]: txt = f"{Mt[i, k]:.3g}\n{lab[mm[i, k]]}"
                cell(ax, i, k, t, Mt[i, k] if t == IFC else None, txt, fs=9.5,
                     tcol="#b03a2e" if (mm is not None and mm[i, k]) else "k")
                if mm is not None and mm[i, k]:
                    ax.add_patch(Rectangle((i + 0.05, k + 0.05), 0.9, 0.9, fc="none", ec="#b03a2e", lw=2.2))
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlim(-0.2, nc + 5.2); ax.set_ylim(-0.2, nr + 0.2); ax.set_aspect("equal"); ax.axis("off")
    ax = axs[3]
    for (i, k), (sh, rec) in share.items():
        for p in rec:
            d = np.array([p[0] - i, p[1] - k], float); d /= np.hypot(*d)
            ax.add_patch(FancyArrowPatch(tuple(np.array([i + 0.5, k + 0.5]) + 0.25 * d),
                                         tuple(np.array([p[0] + 0.5, p[1] + 0.5]) - 0.3 * d), arrowstyle="-|>",
                                         mutation_scale=11, color=C_EX, lw=1.4))
        ax.text(nc + 0.3, k + (2.7 if sh > 0 else 0.9), f"cell {i}: {'+' if sh > 0 else ''}{sh:.3g} to each of {len(rec)}",
                fontsize=10, color=C_EX)
    axs[1].text(nc + 0.3, 1.5, "κ = fs_kappa = 1e-3\n(hysteresis)", fontsize=10)
    axs[2].text(nc + 0.3, 1.2, "interface layer stays\nclosed: no FLUID\ntouches GAS", fontsize=10)
    axs[3].text(nc + 0.3, 0.1, "In 3-D: 18 neighbours.\nNo receiver: tallied as lost.", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fs_convert.svg")


# ---------------------------------------------------------------- 4. coupling timeline
def fig_coupling():
    steps = [
        ("FlowBox, FlowWin::move", "hold the liquid's bounding box + 3 cells (same box on every rank)", None),
        ("FlowFsTypes", "new types from this step's T: melts → FLUID, freezes → WALL;\n"
                        "frozen interface stacks capped / wait / settle / round → reservoir", None),
        ("FlowFill", "T, liquid fraction, composition (σ, ρ, ν per cell with properties cards)", None),
        ("fs_repair", "FLUID touching GAS → full INTERFACE", "type, ε, mass, ρ, u, f"),
        ("FlowFsPowder", "powder per column: Gaussian about the beam, only where the surface cell\n"
                         "is fully liquid; the reservoir is repaid through it (±50 %)", None),
        ("LOOP", None, None),
        ("tallies", "caught and lost mass: one reduction per thermal step → reservoir, spatter log", "sum"),
        ("FlowFsOut", "write-back: interface cells are gas to the stencil, their stack shares the\n"
                      "host's enthalpy (PART, Cap); a filled cell becomes DENSE; an emptied one GAS", None),
        ("fs_surface_z", "the laser hangs from each column's top full cell; z range of the stage", None),
        ("conduction stage", "enthalpy step on the graded grid (ConductionStep)", "grid halo"),
        ("phase_out", "sites follow the metal: i2 5 → 2 (melted in), 2 → 5 (emptied)", None)]
    loop = [("surface()", "FlowFsColumns, FlowFsSurface: h, κ, ρ_G, Marangoni force", None),
            ("step()", "FlowStep: collide, stream / rebuild / bounce, mass exchange, face fluxes", "flux, ρ, u"),
            ("melt_advect", "FlowAdvect × 3: enthalpy q and composition x with the face fluxes", "q, x per sweep"),
            ("inject", "FlowFsInject: powder into each column's lowest interface cell", "mass, q, x"),
            ("convert()", "FlowFsMark, FlowFsRetype, FlowFsExcess", "mark; type2; then\ntype, ε, m, ρ, q, x, f")]
    nrow = len(steps) - 1 + len(loop) + 1.2
    fig, ax = plt.subplots(figsize=(12.5, 0.95 * nrow + 1.2))
    y = nrow
    def box(x, y, w, name, desc, exch, col):
        ax.add_patch(FancyBboxPatch((x, y - 0.78), w, 0.7, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=col, ec="#333", lw=1))
        ax.text(x + 0.15, y - 0.43, name, fontsize=10.5, weight="bold", va="center", family="DejaVu Sans Mono")
        if desc: ax.text(x + 3.0, y - 0.43, desc, fontsize=9.5, va="center")
        if exch:
            ax.text(11.15, y - 0.43, exch, fontsize=9, va="center", color=C_PIN)
            ax.plot([x + w, 11.05], [y - 0.43] * 2, color=C_PIN, lw=0.8, ls=":")
    for name, desc, exch in steps:
        if name != "LOOP":
            box(0.0, y, 10.8, name, desc, exch, "#eef3fa"); y -= 0.95; continue
        top = y
        y -= 0.45
        for n2, d2, e2 in loop:
            box(0.6, y, 10.2, n2, d2, e2, "#fdf2e3"); y -= 0.95
        ax.add_patch(FancyBboxPatch((0.25, y + 0.1), 10.7, top - y - 0.2, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="none", ec=C_REC, lw=2))
        ax.text(0.4, top - 0.12, "F.run(h): n sub-steps (speed or capillary bound, even); post() = advect, inject, convert",
                fontsize=10, color=C_REC, va="top")
        ax.annotate("", xy=(0.3, top - 0.55), xytext=(0.3, y + 0.35),
                    arrowprops=dict(arrowstyle="-|>", color=C_REC, lw=1.6, connectionstyle="arc3,rad=-0.5"))
        y -= 0.3
    ax.text(11.15, nrow + 0.25, "several ranks:\nexchanged after", fontsize=9.5, color=C_PIN, va="bottom")
    ax.set_xlim(-0.3, 13.6); ax.set_ylim(y + 0.2, nrow + 1.0); ax.axis("off")
    ax.set_title("One thermal step with free surface (flow_step, then the conduction stage)", loc="left")
    save(fig, "fs_coupling.svg")


# ---------------------------------------------------------------- 5. freezing an interface stack
def fig_freeze():
    cases = [
        ("capped", "a full cell on top of the stack:\nevery cell becomes solid", [0.3, 0.6], True, False,
         [WALL, WALL], "-(0.7 + 0.4) = -1.1"),
        ("waiting", "liquid beside the stack is still\nabove the lowest solidus: stays,\nheld by the mushy drag (f_l = 1e-3)",
         [0.5, 0.3], False, True, [IFC, IFC], "0 (decided later)"),
        ("settled (2+ cells)", "the stack's mass refills it from\nthe bottom; then each cell whole\nif ≥ 0.5, else gas",
         [0.4, 0.7, 0.2], False, False, [WALL, GAS, GAS], "-0.6 + 0.7 + 0.2 = +0.3"),
        ("single", "one cell: whole if ≥ 0.5,\nelse gas", [0.6], False, False, [WALL], "-0.4")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 6.6))
    for ax, (name, desc, fills, cap, hot, after, res) in zip(axs, cases):
        for x0, lab in ((0.0, "before"), (2.6, "after")):
            k = 0
            cell(ax, x0 + 0.6, k, WALL, text="host\n< solidus", fs=8.5, tcol="w"); k += 1
            for n, e in enumerate(fills):
                if lab == "before":
                    cell(ax, x0 + 0.6, k, IFC, e, f"{e}", fs=9.5)
                else:
                    cell(ax, x0 + 0.6, k, after[n], e if after[n] == IFC else None,
                         f"{e}" if after[n] == IFC else "", fs=9.5)
                k += 1
            if cap: cell(ax, x0 + 0.6, k, WALL, text="cap", fs=9, tcol="w"); k += 1
            cell(ax, x0 + 0.6, k, GAS)
            ax.text(x0 + 1.1, -0.45, lab, ha="center", fontsize=10)
            if hot and lab == "before":
                cell(ax, x0 - 0.4, 1, FLUID, text="hot", fs=8.5, tcol="w")
        ax.annotate("", xy=(3.1, 1.6), xytext=(1.75, 1.6), arrowprops=dict(arrowstyle="-|>", lw=1.6))
        ax.set_title(name, fontsize=12, loc="left")
        ax.text(0.0, -1.0, desc, fontsize=10, va="top")
        ax.text(0.0, -2.6, f"reservoir: {res}", fontsize=10, va="top", color=C_EX)
        ax.set_xlim(-0.5, 4.4); ax.set_ylim(-3.2, 5.4); ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("Freezing an interface stack when its host (the full cell below) drops below its solidus "
                 "(FlowFsTypes);\nfills shown in the cells; metal made is drawn from the reservoir (-), "
                 "metal removed goes to it (+) and is repaid through the powder feed", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fs_freeze.svg")


if __name__ == "__main__":
    fig_cells(); fig_links(); fig_convert(); fig_coupling(); fig_freeze()
    print("wrote", ", ".join(sorted(f for f in os.listdir(OUT) if f.startswith("fs_") and f.endswith(".svg"))))
