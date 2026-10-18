# Material parameter cards

A material card is a TOML file (`materials/<name>.toml`) holding the
material parameters the additive apps need. That covers the conduction
source's tables, the constants Rosenthal and Moser use, solidus/liquidus and
the microstructure calibrations, plus example LPBF process sets. Every value
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
| `powder.psd` (`{ D10, D50, D90 }`, recorded only) | `m` |
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
- Put measured LPBF process sets, with melt-pool sizes if reported, in
  their own `[process.<name>]`.
