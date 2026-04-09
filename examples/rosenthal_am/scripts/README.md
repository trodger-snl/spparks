# Stochastic fluctuation generator

`generate_fluctuations.py` produces ΔW/W, ΔD/D vs. arc-length tracks
suitable for the SPPARKS `rosenthal_fluctuations` command. The generator
emits an ASCII three-column file:

    s[m]   dW_over_W   dD_over_D

with header lines (`#`) capturing the config used to produce it. The
SPPARKS C++ side parses comments transparently.

## Install

```bash
conda activate WORK   # or any env with numpy + pyyaml
pip install numpy pyyaml          # matplotlib only needed for --plot
```

## Run

```bash
cd examples/rosenthal_am/scripts
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

## Mapping into SPPARKS

After generating `lorentzian_5pct.dat`, point an input script at it:

```spparks
setup_temperature_source rosenthal standard 285.0 0.0775 11.2 2.0959e-6 573.0
rosenthal_path           start 0.1e-3 0.4e-3 0.4e-3 end 1.1e-3 0.4e-3 \
                         speed 1.2 repeats 1
rosenthal_fluctuations   lorentzian_5pct.dat mode continuous
```

`mode periodic` (default) wraps the noise on each repeat — good for
"what does this exact roughness profile do" comparisons. `mode continuous`
uses the cumulative laser arc length without modulo and clamps to the
last sample once exhausted, so generate a track at least as long as
`speed × total_run_time` for that mode.

If `setup_temperature_source rosenthal standard …` is used, the
fluctuation command auto-promotes the source to `anisotropic` with
η_y=η_z=1, so no other input changes are needed. `keyhole` mode is not
supported.
