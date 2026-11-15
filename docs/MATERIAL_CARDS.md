# Material parameter cards

A material card is a TOML file (`materials/<name>.toml`) holding the
material parameters the additive apps need. That covers the conduction
source's tables, the constants Rosenthal and Moser use, solidus/liquidus and
the microstructure calibrations, the liquid properties the melt-pool flow
model needs, plus example LPBF and laser DED process sets. Every value
carries a citation, so a run's log records which data it used. Example:
`materials/316L.toml`.

## Using a card in a deck

```
material   ../../materials/316L.toml process single_track_340W
setup_temperature_source conduction card margin 1e-3 1e-3 1e-3 1e-3 plate 1e-3 part part
laser_path start ... end ... speed card hatch card 2 serpentine
```

**`material <file> [process <name>]`** loads the card and selects one process
set.
- It sets `liquidus`, `solidus`, `solid_front_vel` and the nucleation
  scalars (`nucleation_density`, `critical_undercooling`,
  `undercooling_deviation`, `mean_nuclei_volume`,
  `nuclei_volume_deviation`) from the card.
- It prints the card's name, version and modified date, then every value it
  read with its unit and ref.
- **Put it first.** A deck command after `material` overrides the card value,
  and one before it is overwritten.

**`setup_temperature_source <type> card [...]`** builds the source's
arguments from the card; `<type>` is `rosenthal`, `moser` or `conduction`
(standard mode only). The expanded command is printed, so you can paste it
into a deck that doesn't use the card.

| Argument | From the card |
|---|---|
| Q, lambda, T0 | process `power`, `absorptivity` (or `[optical]`), `preheat` |
| sx, sy, sz | process `spot` converted to the 1/e radius; `source_depth` |
| k, cp (conduction) | `[thermal]` tables at T0 |
| k, cp (rosenthal, moser) | `[thermal.linear]` conductivity/heat_capacity, or the tables at `T_ref` |
| alpha | k / (density · cp), so the conduction solver's ρ is the card density |
| conduction keywords | `k_table`, `cp_table`, `latent`, `k_powder`, `emissivity`, `layer_thickness`, `convection` |

Overrides:
- `Q v`, `lambda v`, `T0 v` and `spot sx sy sz` after `card` replace the
  card's values.
- Any other keyword (grid, geometry, or a material keyword such as
  `k_powder 0.5`) is appended after the card's, and the last occurrence wins.

**No built-in defaults with a card.** Once `material` is loaded, every
material value the model uses must come from the card or from an explicit
deck command; otherwise the run stops with an error naming what is missing.
- `setup_temperature_source ... card` needs Q, lambda, T0, the spot,
  `source_depth`, density, and k and cp.
- For conduction it also needs `latent`, `k_powder`, `emissivity`,
  `layer_thickness` and `convection`.
- `init_app` checks `liquidus`, `solidus`, `solid_front_vel` and the five
  nucleation values.
- To turn a term off, say so in the deck, e.g. `convection 0 300` or
  `emissivity 0`.
- Decks without a card keep the old defaults.

`powder.packing` is recorded but never passed: `packing` < 1 is for
single-layer tests only (docs/CONDUCTION_SOURCE.md, Material).

**`speed card` and `hatch card`** in `laser_path` take the selected process
set's `speed` and `hatch`.

## Format (schema 1)

```toml
[card]
schema_version = 1          # the loader rejects anything else
material = "316L"
version = "1.0.0"           # major: values that change results; minor: data added; patch: notes, refs
created = 2026-10-17
modified = 2026-10-17       # TOML dates
status = "draft"            # draft | reviewed
reviewed_by = ""
designation = { ... }       # free form: UNS, spec
composition_wt_pct = { ... }

[[changelog]]               # one entry per version; the last matches version/modified
version = "1.0.0"
date = 2026-10-17
author = "..."
changes = "..."

[references.key]            # citation required; doi, url, isbn optional
citation = "..."
doi = "..."
```

**Scalar record.** For example:

```toml
density = { value = 7900, unit = "kg/m^3", ref = "key",
            uncertainty = 0.01, uncertainty_type = "relative_1sigma",
            T_range = [300, 300], note = "..." }
```

**Table record.** For example:

```toml
conductivity = { T = [...], value = [...], unit = "W/(m K)", ref = "key",
                 extrapolation = "clamp" }
```

`T` must strictly increase, and outside it the value is held at the end
value (`clamp`).

**Fields shared by both records:**
- **`ref`** is a references key, a list of keys, `"estimate"` or
  `"calibration"`; a SPPARKS fit with no literature source is a
  calibration. It is never empty.
- **`uncertainty_type`** is one of `1sigma`, `2sigma`, `bounds` (then
  `uncertainty = [lo, hi]`) or `relative_1sigma`.

**Units.** All values are SI. The loader requires the `unit` string to match
the table below exactly; it never converts units.

| Record | Unit |
|---|---|
| `thermal.density` | `kg/m^3` |
| `thermal.solidus`, `thermal.liquidus`, `thermal.linear.T_ref` | `K` |
| `thermal.latent_fusion` | `J/kg` |
| `thermal.conductivity` (table), `thermal.linear.conductivity`, `powder.conductivity` | `W/(m K)` |
| `thermal.heat_capacity` (table), `thermal.linear.heat_capacity` | `J/(kg K)` |
| `thermal.emissivity`, `powder.packing`, `optical.absorptivity` | `1` |
| `thermal.emissivity_table` (table against T, recorded only) | `1` |
| `powder.psd` (any of `D10`, `D50`, `D90`, increasing; recorded only) | `m` |
| `flow.surface_tension` | `N/m` |
| `flow.dsigma_dT` | `N/(m K)` |
| `flow.viscosity` (dynamic; scalar or table) | `Pa s` |
| `flow.density_liquid` | `kg/m^3`, with `T_ref` (K) and `slope` (`slope_unit = "kg/(m^3 K)"`) |
| `flow.thermal_expansion` (volumetric) | `1/K` |
| `microstructure.solid_front_vel` | `m/s`, with `dT_unit = "K"` |
| `microstructure.nucleation_density` | `1/m^3` |
| `microstructure.critical_undercooling`, `undercooling_deviation` | `K` |
| `microstructure.mean_nuclei_volume`, `nuclei_volume_deviation` | `m^3` |
| `process.<set>.power` | `W` |
| `process.<set>.speed` | `m/s` |
| `process.<set>.hatch`, `layer_thickness`, `spot`, `source_depth` | `m` |
| `process.<set>.preheat` | `K` |
| `process.<set>.absorptivity` | `1` |
| `process.<set>.convection` | `W/(m^2 K)`, as `{ h, T_inf }` |
| `process.<set>.powder_feed` (DED, recorded only) | `kg/s` |
| `process.<set>.capture_efficiency` (DED, recorded only) | `1` |
| `process.<set>.beam_diameter` (recorded only) | `m`, with `profile` |
| `process.<set>.outcome` (recorded only) | none: `{ relative_density, cracking, ref }` |

**Front velocity.**
- `solid_front_vel` is what SPPARKS evaluates, in undercooling
  ΔT = liquidus − T, in one of two forms:
  - `{ coeffs = [...] }`: a polynomial, highest power first, any number of
    coefficients;
  - `{ a, n }`: the power law V = a ΔT^n (a, n > 0).
- Literature data go in `front_velocity_source`, with their ref, in one of
  three forms:
  - `{ form = "power_law", a, n }` (V = a ΔT^n; copy a and n exactly into
    `solid_front_vel`);
  - `{ form = "polynomial", coeffs = [...] }` (a KGT polynomial; copy it
    exactly into `solid_front_vel`);
  - `{ form = "points", dT = [...], V = [...] }`.
- For `points`, `tools/material_card.py fit-front [--degree N]` turns the
  source into `coeffs`. It is a relative least-squares fit with V(0) = 0, so slow fronts
  at small undercooling count as much as fast ones. A cubic is off by 30 to
  140 % for ΔT^3.1 to ΔT^3.5 laws; degree 5 or 6 gets within about 5 %.
- `lint` checks that `solid_front_vel` stays within `max_rel_err` (default
  0.1) of the source at every point of (0, `fit_range_K` max]. It also checks
  that it is non-negative over (0, liquidus − solidus].

**Process sets.**
- `[process.<name>]` holds the records in the table above, plus free-form
  `note`/`machine` text and an optional measured block,
  `validation = { melt_pool_width, melt_pool_depth, unit = "m", ref }`, for
  calibration checks.
- `spot.convention` is `1/e radius` (Moser's sx), `1/e2 radius`,
  `1/e2 diameter`, `D4sigma` or `FWHM`.
- `source_depth` is Moser's sz, a calibration rather than a beam property.
- `outcome = { relative_density, cracking, ref }` records what the build
  came out as: `relative_density` is a fraction in (0, 1] (the note says how
  it was measured), `cracking` is `none`, `cracked` or `not reported`. Either
  may be left out, not both.
- A single track melted on bare plate is named
  `<source>_single_track_<power>W`, carries `method = "... single track on
  bare plate (no powder)"` and has no `layer_thickness`.

**Absorptivity.** `optical.absorptivity` and a set's `absorptivity` may carry
`surface`: `flat solid` (the default), `powder bed`, `liquid` or
`in process` (an effective value measured or calibrated during melting).
When no calorimetric measurement exists the value may be 1 - R at normal
incidence from tabulated n and k at the laser wavelength: mark the note
`DERIVED`, name the data, and say that it is a flat, clean, room-temperature
value and a lower limit for a powder bed or a keyhole.

**Emissivity.** SPPARKS reads the scalar `thermal.emissivity`. A measured
temperature dependence goes in `thermal.emissivity_table` next to it.

**Pure metals and congruent melters.** A pure element has one melting point,
but the apps need `solidus < liquidus`. Record the melting point as
`liquidus` with its source and set `solidus` a few K below with
`ref = "estimate"` and a note starting `MODELLING CHOICE` that gives the gap
and why (W, Ta, Nb, Mo use 10 K; Bi2Te3 uses 5 K).

**Tables that stop short of melting.** If a conductivity or heat-capacity
table ends below the liquidus, or has no liquid value, its note says the
table is HELD at its last value and that this is not a measurement.

**How a source was read.** Each `citation` ends with how it was read: "read
in full", "abstract only", "data table read from <archive>" or "text
searched". A value taken from an abstract says so in its note.

**DED process sets.** Laser directed energy deposition (blown powder, LENS)
sets are named `ded_<source>_<power>` and carry `method = "laser DED, ..."`.
They use the same records, with `hatch` for the stepover and
`layer_thickness` for the layer height (z step), plus three records that
SPPARKS does not read:
- `powder_feed`: powder mass flow at the nozzle.
- `capture_efficiency`: the fraction of the fed powder that ends up in the
  track or part (catchment efficiency); the note says which.
- `beam_diameter`, with `profile = "top-hat"`, `"gaussian"` or `"unstated"`:
  the spot as the paper gives it when it is not a Gaussian measure that
  `spot.convention` covers. Most DED papers print a bare "spot size". A set
  with `beam_diameter` and no `spot` gives `setup_temperature_source ... card`
  no source size, so the run stops and asks for `spot sx sy sz` in the deck;
  choosing that radius is then the user's modelling decision.

`validation` may hold `track_width` and `track_height` (the bead on the
surface) next to `melt_pool_depth` (penetration below the substrate
surface). A DED set without its own `absorptivity` falls back to
`[optical]`, which is a bare-plate or powder-bed value; the fraction of the
laser power absorbed in DED also depends on the powder stream (the Ti6Al4V
and IN625 cards quote a calorimetric measurement in a comment).

**Flow.** `[flow]` holds the liquid-state properties of the conduction
source's melt-pool flow (`flow lbm`, doc/temperature_source_conduction.txt).
SPPARKS does not read the section yet, so pass the values in the deck:

| `flow lbm` keyword | From the card |
|---|---|
| `dsigma_dT` | `flow.dsigma_dT` |
| `beta` | `flow.thermal_expansion` |
| `viscosity` (kinematic, m^2/s) | `flow.viscosity` / liquid density at the same temperature; each card's viscosity note gives the number |

- `surface_tension` is the value at `T_range` (the liquidus); the flow model
  uses only its temperature coefficient.
- `dsigma_dT` carries a `composition` string. For steels and Ni alloys it
  depends on dissolved sulfur and oxygen and changes sign: a clean melt has
  about -4e-4 N/(m K), 20 to 30 ppm S is enough for a positive value (316L
  card). Check the composition against the feedstock before using the number,
  and expect it to hold only near the measured range, a few hundred K above
  the liquidus at most.
- `viscosity` is the dynamic viscosity, as a scalar at `T_range` or as a
  table.
- `density_liquid = { value, T_ref, slope, ... }` is the linear fit
  rho(T) = value + slope (T - T_ref).
- `thermal_expansion` is the volumetric coefficient, -slope/value of
  `density_liquid`, derived for the card.
- The IN718 values are secondary or unreviewed, and several others rest on a
  thesis or a single levitation campaign; the notes say which.

## Tooling

`tools/material_card.py` uses the Python 3.11+ standard library only:

```
material_card.py lint materials/316L.toml [--doi]   # schema, units, refs, changelog, front fit; --doi resolves DOIs
material_card.py fit-front materials/316L.toml      # front_velocity_source -> solid_front_vel line
material_card.py selftest
```

## Adding a material (research agents)

Start from `materials/316L.toml`, keep `status = "draft"`, and run
`lint --doi` until it is clean. A person reviews the card before it becomes
`reviewed`. The DOI check is there because agent-written citations can be
fabricated; a DOI that resolves still has to be checked against the value.

When choosing data:
- Prefer primary measurements (NIST, Mills 2002, Valencia & Quested,
  peer-reviewed datasets) over vendor sheets.
- Record uncertainty where the source gives it.
- Mark values with no source `estimate`, and SPPARKS fits `calibration`.
- Put measured LPBF and DED process sets, with melt-pool or bead sizes if
  reported, in their own `[process.<name>]`.
- For `[flow]`, record the sulfur and oxygen content of the measured sample
  in `dsigma_dT.composition`, or say that the source does not give it.

## Surface tension model (preliminary): `tools/surface_tension.py`

`flow lbm` takes one constant `dsigma_dT`. For a graded build the surface
tension depends on temperature and on the composition x (d11), and its
gradient has a solutal part, dσ/dx. `tools/surface_tension.py` (Python 3.11+
standard library) computes a table σ(T, x) between two end-member alloys from
literature constants. SPPARKS does not read the table yet. Nothing in it is
fitted to the cards: where it misses a card's measured value, that is the
state of the model.

```
surface_tension.py table materials/316L.toml materials/IN625.toml --S 20 --O 5 -o sigma.txt
surface_tension.py table materials/Ti6Al4V.toml "Ti=0.677,V=0.323" --T 1900 2900 50 --nx 21
surface_tension.py point materials/316L.toml --S 20 --T 1708
surface_tension.py --self-check
```

- An end member is a card (`composition_wt_pct`: a range counts as its
  midpoint, `bal` and `>=` entries take the balance, `<` limits count as
  zero) or element mass fractions on the command line. x is the **mass
  fraction of the second alloy**; `--S` and `--O` are dissolved sulfur and
  oxygen in ppm by mass, one value or `A,B` (linear in x).
- Output: `T x sigma dsigma_dT dsigma_dx`, SI (K, N/m, N/(m K), N/m per unit
  x), T outer and x inner, `#` header lines. When an end member is a card
  with `[flow]`, the header compares the model with the card at the card's
  temperature.

**Equations as implemented.**

1. Metallic part: the Butler equation in the form of Tanaka, Hack, Iida and
   Hara (Z. Metallkd. 87 (1996) 380, Eq. 1-3 and 8). For every component i

   σ = σ_i + (R T / A_i) ln(x_i^s / x_i^b) + (β G_i^E(T, x^s) − G_i^E(T, x^b)) / A_i,

   with Σ x_i^s = 1, A_i = 1.091 N_A^(1/3) V_i^(2/3), β = 0.83, and G_i^E
   the **partial** excess Gibbs energy of i in the bulk liquid, evaluated at
   the surface composition x^s in the first term. σ_i(T) and V_i(T) = M_i /
   ρ_i(T) are linear fits for the pure liquid elements. G^E is a
   Redlich-Kister/Muggianu sum over the binaries, G^E = Σ x_i x_j Σ_k L_k
   (x_i − x_j)^k with L_k = a + b T; a binary without parameters is ideal
   (listed in the table header). `--ideal` and `--beta` change the two
   choices. Solver: bisection on σ inside a damped fixed point on x^s.
2. Sulfur and oxygen: the Langmuir term of Sahoo, DebRoy and McNallan
   (Metall. Trans. B 19 (1988) 483, Eq. 6) is subtracted,

   σ = σ_metal − R T Γ_s ln(1 + k1 a exp(−ΔH0 / (R T))),

   with a in wt% and the paper's sets for Fe-S, Fe-O (fitted), Ni-S and Cr-O
   (the paper's estimates). Following McNallan and DebRoy (Metall. Trans. B
   22 (1991) 557), who use the Fe-S set unchanged for Fe-Ni-Cr, the sulfur
   activity is a_S = [wt% S] 10^(e [wt% Cr]) with e = −94.2/T + 0.0396.
   `--s-model mills` uses instead the stainless-steel fit of Mills 2004
   (Eq. 26: Γ_s R = 0.056 mN/(m K), ln K = 28798/T − 8.5647, a = 0.68 wt% S).

**Choices made for this tool, not taken from a source** (all in
`surfactant_drop`): the pure-metal term of Sahoo's equation is replaced by
the Butler value of the alloy; the constants of a mixed solvent are the
mole-fraction average of ΔH0, ln k1 and Γ_s over the elements that have a
set; the S and O terms add (no competition for sites); Ni-O borrows the Fe-O
constants; the oxygen activity is the wt% given. `--O` therefore means
*soluble* oxygen: Mills 2004 takes about 5 ppm for Al-killed stainless steel,
whatever the total oxygen. The Mills fit already contains that oxygen, so use
it with `--O 0`.

**Constants and sources.** Every number sits in the script's `REFS`,
`ELEMENTS`, `RK` and `SURFACTANT` blocks with a ref and a statement of what
was opened. In short:
- Opened and read: Sahoo 1988, McNallan and DebRoy 1991, Zacharia et al.
  1989, Mills 2004, Tanaka 1996, the Leitner thesis (Fe, Ni), Brillo's 2016
  book (Al, Ni density), Assael 2006 (Fe, Al density), Reiplinger 2022 (Ti,
  V), Ozawa 2017 (Ti density), and the MatCalc open Ni database for all
  Redlich-Kister sets.
- Second-hand: Cr and Mn (Mills and Su 2006 and Iida and Guthrie, as
  tabulated by Nakamoto and Tanaka 2020); the Mo surface tension at the
  melting point. The assessments behind the Redlich-Kister sets were not
  opened; only Cr-Fe, Cr-Ni, Fe-Ni, Al-Ti and Ti-V were cross-checked
  against a second listing.
- Abstract or snippet only: Nb, Zr, the Mo density. **From memory,
  unverified:** the Mo temperature coefficient.
- Keene 1993, Mills and Su 2006, Iida and Guthrie, Mills 2002 and Butler
  1932 were not opened.

**What the self-check shows** (`--self-check` prints the table):
- Sahoo's Fe-S constants put the sign change of dσ/dT for 90 ppm S at
  1949 K. Zacharia et al. (and the weld models after them) use ΔH0 =
  −1.88e5 instead of Sahoo's −1.66e5 kJ/kg mole, which gives 2207 K, their
  "about 2200 K". The tool uses Sahoo's printed value.
- Butler against levitation data, no S/O term: Ti-V within +3 to +4 %,
  Al-Ti −2 to −8 % (the regular solution matters there: the ideal solution
  is 13 to 16 % low), Ti-6Al-4V −7 %, Fe-Ni ±5 % (two measurements that
  themselves differ by about 10 %).

**Against the cards (2026-11-14).**

| Case | σ model / card [N/m] | dσ/dT model / card [N/(m K)] |
|---|---|---|
| 316L, 1708 K, no S, O | 1.903 / 1.586 | −3.6e-4 / +3.2e-4 |
| 316L, S 20, O 0 (sahoo) | 1.848 / 1.586 | −1.3e-4 / +3.2e-4 |
| 316L, S 20, O 0 (mills) | 1.725 / 1.586 | +3.3e-4 / +3.2e-4 |
| 316L, S 20, O 5 (sahoo) | 1.788 / 1.586 | +1.6e-4 / +3.2e-4 |
| 316L, S 20, O 30 (sahoo) | 1.609 / 1.586 | +6.8e-4 / +3.2e-4 |
| IN625, 1623 K, no S, O | 1.885 / 1.779 | −3.5e-4 / −1.8e-4 |
| IN625, S 20, O 5 (sahoo) | 1.765 / 1.779 | +2.2e-4 / −1.8e-4 |
| Ti-6Al-4V, 1933 K | 1.392 / 1.493 | −1.7e-4 / −1.9e-4 |

No single setting reproduces both σ and dσ/dT of the 316L card. With the
card's 20 ppm S the Sahoo term is about a third of the Mills term (0.06
against 0.18 N/m), the Mills fit gets the slope and misses σ by 9 %, and
treating all 30 ppm O as dissolved gets σ and doubles the slope. The IN625
sample's S and O are unknown, so its row tests nothing beyond the clean
value (6 % high).

**Limits.**
- Preliminary. Expect ±5 to 10 % in σ for the metallic part and no better
  than the sign and order of magnitude of dσ/dT once S or O is present.
- Pure-element fits are extrapolated far outside their range: Al 1000 K
  above its data in a Ti melt, Mo, Nb and Cr 500 to 1300 K undercooled in a
  steel or Ni melt.
- No surfactant term for Ti alloys (the tool stops if asked): no Langmuir
  constants for O, N or S in liquid Ti were found. Oxygen dissolves in liquid
  Ti to several at.% and is reported to matter above about 1 at.% (Reiplinger 2022).
- No ternary interaction parameters; elements other than Al, Cr, Fe, Mn, Mo,
  Nb, Ni, Ti, V, Zr are refused (Si, C, Sn, Ta are missing).
- A strongly demixing G^E can have more than one surface state; the solver
  then fails to converge rather than choosing.
- Evaporation, and the change of S and O during the process, are outside the
  model; levitation samples lose S and Mn within minutes (316L card note).
