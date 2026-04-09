#!/usr/bin/env python3
"""Generate stochastic ΔW/W, ΔD/D sequences for SPPARKS rosenthal_fluctuations.

Reads a YAML config that specifies a target spectral shape (white, 1/f^β,
Lorentzian, narrow-band, multi-peak), target RMS, correlation length, and a
Pearson correlation between the W and D channels. Emits a 3-column ASCII
file consumed by the SPPARKS `rosenthal_fluctuations` command:

    s[m]   dW_over_W   dD_over_D

Usage:
    python generate_fluctuations.py config.yaml -o fluct.dat
    python generate_fluctuations.py config.yaml -o fluct.dat --plot

Required: numpy, pyyaml. matplotlib only for --plot.
"""

import argparse
import sys

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# PSD shapes. All return S(f) on the input freq grid, normalized so that
# the analytical variance ∫S df ≈ sigma^2 (Parseval). Final exact-σ
# normalization is applied to the time series.
# ---------------------------------------------------------------------------

def psd_white(f, sigma):
    s = np.ones_like(f)
    return _normalize_psd(f, s, sigma)


def psd_power_law(f, sigma, beta):
    """1/f^β. β=1 -> pink, β=2 -> red. f[0]=0 handled with a gentle floor."""
    s = np.zeros_like(f)
    pos = f > 0
    s[pos] = 1.0 / (f[pos] ** beta)
    return _normalize_psd(f, s, sigma)


def psd_lorentzian(f, sigma, tau_m):
    """S(f) ∝ 1 / (1 + (2πf τ)^2). tau_m is correlation length in meters."""
    omega_tau = 2.0 * np.pi * f * tau_m
    s = 1.0 / (1.0 + omega_tau ** 2)
    return _normalize_psd(f, s, sigma)


def psd_narrow_band(f, sigma, f0, df):
    """Gaussian bump centered at spatial frequency f0 (1/m), width df (1/m)."""
    s = np.exp(-0.5 * ((f - f0) / max(df, 1e-30)) ** 2)
    return _normalize_psd(f, s, sigma)


def psd_multi_peak(f, sigma, peaks):
    """Sum of Gaussian bumps. peaks: list of dicts {f0, df, weight}."""
    s = np.zeros_like(f)
    for p in peaks:
        f0 = float(p["f0"])
        df = float(p["df"])
        w = float(p.get("weight", 1.0))
        s += w * np.exp(-0.5 * ((f - f0) / max(df, 1e-30)) ** 2)
    return _normalize_psd(f, s, sigma)


def _normalize_psd(f, s, sigma):
    if len(f) < 2:
        return s
    df = f[1] - f[0]
    integral = np.sum(s) * df
    if integral <= 0.0:
        return s
    # Two-sided variance from a one-sided rfft PSD: 2 * ∫₀^fNyq S df.
    # We want that to equal sigma^2.
    return s * (sigma ** 2) / (2.0 * integral)


def build_psd(spec, f):
    t = spec.get("type", "white").lower()
    sigma = float(spec["sigma"])
    if t == "white":
        return psd_white(f, sigma)
    if t in ("pink", "1/f"):
        return psd_power_law(f, sigma, 1.0)
    if t in ("red", "brown", "1/f2"):
        return psd_power_law(f, sigma, 2.0)
    if t == "power_law":
        return psd_power_law(f, sigma, float(spec.get("beta", 1.0)))
    if t == "lorentzian":
        return psd_lorentzian(f, sigma, float(spec["tau_m"]))
    if t in ("narrow_band", "narrowband"):
        return psd_narrow_band(f, sigma, float(spec["f0"]), float(spec["df"]))
    if t in ("multi_peak", "multipeak"):
        return psd_multi_peak(f, sigma, spec["peaks"])
    raise ValueError(f"unknown PSD type '{t}'")


# ---------------------------------------------------------------------------
# Correlated pair generator
# ---------------------------------------------------------------------------

def generate_correlated_pair(S_W, S_D, rho, n_samples, dx, rng):
    """Return (dW, dD) real time series with the requested PSDs and Pearson
    correlation rho. Uses the standard FFT-domain method:
        Z_D' = rho Z_W + sqrt(1-rho^2) Z_D
    which preserves the marginal spectra and yields the requested rho.
    """
    n_freq = len(S_W)
    # Independent complex Gaussian Fourier coefficients.
    def _draw():
        re = rng.standard_normal(n_freq)
        im = rng.standard_normal(n_freq)
        z = (re + 1j * im) / np.sqrt(2.0)
        # Real signal: DC and Nyquist must be real.
        z[0] = z[0].real
        if n_samples % 2 == 0:
            z[-1] = z[-1].real
        return z

    Z_W = _draw()
    Z_D_indep = _draw()
    Z_D = rho * Z_W + np.sqrt(max(0.0, 1.0 - rho ** 2)) * Z_D_indep

    # Multiply by sqrt(S * n_samples / (2*dx)) — the FFT scale that takes a
    # unit-variance complex Gaussian into a time series with the target PSD
    # under numpy's rfft normalization. We rescale exactly to target σ
    # afterwards anyway, so the absolute prefactor only needs to be order-1.
    scale_W = np.sqrt(np.maximum(S_W, 0.0)) * np.sqrt(n_samples / (2.0 * dx))
    scale_D = np.sqrt(np.maximum(S_D, 0.0)) * np.sqrt(n_samples / (2.0 * dx))
    dW = np.fft.irfft(Z_W * scale_W, n=n_samples)
    dD = np.fft.irfft(Z_D * scale_D, n=n_samples)
    return dW, dD


def rescale_to_sigma(x, sigma):
    s = float(np.std(x))
    if s <= 0.0:
        return x
    return x * (sigma / s)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_dat(path, s, dW, dD, header_lines):
    with open(path, "w") as fh:
        for h in header_lines:
            fh.write("# " + h.rstrip() + "\n")
        fh.write("# columns: s[m]   dW_over_W   dD_over_D\n")
        for si, w, d in zip(s, dW, dD):
            fh.write(f"{si:.9e}  {w:.6e}  {d:.6e}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="YAML config file")
    ap.add_argument("-o", "--output", required=True, help="output .dat file")
    ap.add_argument("--plot", action="store_true",
                    help="render time series and periodogram (matplotlib)")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    length_m = float(cfg["length_m"])
    dx = float(cfg["dx_m"])
    seed = int(cfg.get("seed", 12345))
    rho = float(cfg.get("correlation", 0.0))

    n_samples = int(round(length_m / dx))
    if n_samples < 4:
        sys.exit("config: length_m / dx_m must give >= 4 samples")
    s = np.arange(n_samples) * dx
    f = np.fft.rfftfreq(n_samples, d=dx)

    rng = np.random.default_rng(seed)

    S_W = build_psd(cfg["dW"], f)
    S_D = build_psd(cfg["dD"], f)

    dW, dD = generate_correlated_pair(S_W, S_D, rho, n_samples, dx, rng)

    sigma_W_target = float(cfg["dW"]["sigma"])
    sigma_D_target = float(cfg["dD"]["sigma"])
    dW = rescale_to_sigma(dW, sigma_W_target)
    dD = rescale_to_sigma(dD, sigma_D_target)

    # Stats
    emp_corr = float(np.corrcoef(dW, dD)[0, 1]) if n_samples > 1 else 0.0
    print(f"  n_samples       = {n_samples}")
    print(f"  length_m        = {length_m:.4e}")
    print(f"  dx_m            = {dx:.4e}")
    print(f"  dW: mean={dW.mean(): .3e}  sigma={dW.std():.3e}  "
          f"min={dW.min(): .3e}  max={dW.max(): .3e}")
    print(f"  dD: mean={dD.mean(): .3e}  sigma={dD.std():.3e}  "
          f"min={dD.min(): .3e}  max={dD.max(): .3e}")
    print(f"  Pearson rho    = {emp_corr:+.3f}  (target {rho:+.3f})")

    if max(np.abs(dW).max(), np.abs(dD).max()) > 0.5:
        print("WARNING: |dW| or |dD| > 0.5; first-order linearization in "
              "the SPPARKS source is no longer accurate.", file=sys.stderr)

    header = [
        f"generated by generate_fluctuations.py",
        f"config: {args.config}",
        f"seed={seed} length_m={length_m} dx_m={dx} rho={rho}",
        f"dW: {cfg['dW']}",
        f"dD: {cfg['dD']}",
        f"empirical: sigma_W={dW.std():.4e} sigma_D={dD.std():.4e} "
        f"corr={emp_corr:+.4f}",
    ]
    write_dat(args.output, s, dW, dD, header)
    print(f"  wrote {args.output}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            sys.exit("--plot requires matplotlib")
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        axes[0, 0].plot(s * 1e3, dW, lw=0.7)
        axes[0, 0].set_xlabel("s [mm]")
        axes[0, 0].set_ylabel("dW/W")
        axes[0, 1].plot(s * 1e3, dD, lw=0.7, color="C1")
        axes[0, 1].set_xlabel("s [mm]")
        axes[0, 1].set_ylabel("dD/D")
        axes[1, 0].loglog(f[1:], np.abs(np.fft.rfft(dW)[1:]) ** 2, lw=0.7)
        axes[1, 0].set_xlabel("spatial freq [1/m]")
        axes[1, 0].set_ylabel("|FFT(dW)|^2")
        axes[1, 1].loglog(f[1:], np.abs(np.fft.rfft(dD)[1:]) ** 2,
                          lw=0.7, color="C1")
        axes[1, 1].set_xlabel("spatial freq [1/m]")
        axes[1, 1].set_ylabel("|FFT(dD)|^2")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
