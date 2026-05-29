# Helper scripts

This directory contains two standalone Python helpers used while
setting up Moser-source simulations.

## `crystal_to_quat.py` — crystal-direction → quaternion

Convert a substrate orientation given in crystallographic notation
(`[hkl]` for the build-direction normal and `[uvw]` for the in-plane
scan direction) into the SPPARKS scalar-first quaternion `(q0, qx, qy,
qz)` that goes into the d1..d4 site fields.

```bash
# Identity (cube-on-face): [001] || +Z, [100] || +X
python crystal_to_quat.py --normal 001 --inplane 100

# Goss orientation, emitted as a SPPARKS `set` command for spin 7
python crystal_to_quat.py --normal 110 --inplane 001 --as-set 7

# Bunge Euler angles (degrees, ZXZ)
python crystal_to_quat.py --euler-bunge 30 45 0

# Sanity-check an existing quaternion: report which [hkl] it places
# along the build (+Z) and scan (+X) directions
python crystal_to_quat.py --check 0.270598 0.653281 0.270598 0.653281
```

For glued multi-digit indices like `[12 0 5]`, separate components
with spaces or commas: `--normal "12 0 5"`. Negative indices may be
written with a leading `-` (e.g. `1-10` is `[1, -1, 0]`).

A `--selftest` flag runs a few sanity round-trips:
```bash
python crystal_to_quat.py --selftest
```

Requires `numpy` and `scipy`.

## `generate_fluctuations.py` — stochastic ΔW/W, ΔD/D tracks (reference)

`generate_fluctuations.py` produces ΔW/W, ΔD/D vs. arc-length tracks
in an ASCII three-column file:

    s[m]   dW_over_W   dD_over_D

with header lines (`#`) capturing the config used to produce it.

**Note:** SPPARKS no longer has a built-in command that loads these
files. Stochastic fluctuations are now produced internally by the
Moser/Green's-function source via the `laser_fluctuations psd ...`
input command (see `examples/moser_am/in.additive.moser_multiscan_fluct`).
This Python tool is kept as a reference implementation and for
external workflows that want to generate stochastic tracks for
non-SPPARKS consumers (analysis, plotting, comparison studies).

## Install

```bash
conda activate WORK   # or any env with numpy + pyyaml
pip install numpy pyyaml          # matplotlib only needed for --plot
```

## Run

```bash
cd examples/moser_am/scripts
python generate_fluctuations.py configs/lorentzian_5pct.yaml \
    -o ../lorentzian_5pct.dat

# Optional: visual sanity check (time series + periodogram)
python generate_fluctuations.py configs/lorentzian_5pct.yaml \
    -o ../lorentzian_5pct.dat --plot
```

The script prints empirical mean, σ, range, and Pearson correlation for
both channels and warns if any sample exceeds |Δ|>0.5 (linearization in
the SPPARKS source breaks down).

## Config schema

```yaml
length_m: 0.005          # total scan track length to generate
dx_m: 5e-6               # sample spacing
seed: 12345              # PRNG seed for reproducibility
correlation: 0.7         # Pearson correlation between dW and dD, in [-1, 1]

dW:
  type: lorentzian       # white | pink | red | power_law | lorentzian
                         # | narrow_band | multi_peak
  sigma: 0.05            # target RMS of dW/W
  tau_m: 200e-6          # correlation length (lorentzian only)
dD:
  type: lorentzian
  sigma: 0.08
  tau_m: 200e-6
```

Other PSD shapes:

| `type`        | extra keys              | meaning                                |
|---------------|-------------------------|----------------------------------------|
| `white`       | —                       | flat spectrum                          |
| `pink`        | —                       | 1/f                                    |
| `red`         | —                       | 1/f²                                   |
| `power_law`   | `beta`                  | 1/f^β                                  |
| `lorentzian`  | `tau_m`                 | exponentially-correlated noise         |
| `narrow_band` | `f0`, `df`              | Gaussian bump at spatial freq f₀ (1/m) |
| `multi_peak`  | `peaks: [{f0,df,weight}]` | sum of Gaussian bumps                |

`f0`, `df` are spatial frequencies in 1/m; spatial period = 1/f₀.

## In-source equivalent

The same Lorentzian, white, pink and narrow_band shapes are available
as a built-in command on the Moser source — no Python preprocessing
or file I/O required:

```spparks
setup_temperature_source moser 285.0 0.155 11.2 2.0959e-6 573.0 650.0 \
                               50e-6 50e-6 25e-6
laser_fluctuations psd lorentzian sigma_W 0.05 sigma_D 0.06 tau 200e-6 \
                                  rho 0.5 seed 12345 dt 5e-6
laser_path start 0.1e-3 0.4e-3 0.4e-3 end 1.1e-3 0.4e-3 speed 1.2 repeats 1
```

The Python tool is only useful if you need a track for an external
consumer or want to inspect the time series outside SPPARKS.
