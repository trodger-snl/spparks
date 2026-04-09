/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   http://www.cs.sandia.gov/~sjplimp/spparks.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPPARKS directory.
------------------------------------------------------------------------- */

#include "temperature_source_moser.h"
#include "error.h"
#include "domain.h"
#include "math_const.h"

#include "GREENAM/GreenAM_Constants.h"
#include "GREENAM/GreenAM_Properties.h"
#include "GREENAM/GreenAM_Util.h"
#include "GREENAM/GreenAM_LaserScan.h"
#include "GREENAM/GreenAM_ScanIntegration.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SPPARKS_NS::MathConst;

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

MoserGreenTemperatureSource::MoserGreenTemperatureSource(SPPARKS *spk)
  : TemperatureSource(spk),
    Q(0.0), lambda(0.0), k_th(0.0), alpha(0.0), T0_default(300.0),
    cp(0.0), rho(0.0),
    sx(0.0), sy(0.0), sz(0.0),
    char_length(0.5),
    scan_t_origin(0.0),
    scan_x0(0.0), scan_y0(0.0), scan_x1(0.0), scan_y1(0.0), scan_zl(0.0),
    scan_speed(0.0), scan_repeats(0),
    scan_built(false),
    psd_spec_set(false),
    psd_rng(12345),
    psd_norm(0.0, 1.0),
    ar_state_W(0.0), ar_state_D(0.0), ar_state_P(0.0),
    voss_step(0),
    osc_x_W(0.0), osc_v_W(0.0),
    osc_x_D(0.0), osc_v_D(0.0),
    osc_x_P(0.0), osc_v_P(0.0)
{
  ambient_temperature = T0_default;
  voss_W.fill(0.0);
  voss_D.fill(0.0);
  voss_P.fill(0.0);
  psd_spec = PsdSpec{PsdShape::WHITE, 0.0, 0.0, 0.0, 0.0,
                     12345UL, 5.0e-6, 0.0, 0.0, 0.0};
}

/* ---------------------------------------------------------------------- */

MoserGreenTemperatureSource::~MoserGreenTemperatureSource()
{
  cleanup();
}

/* ----------------------------------------------------------------------
   Setup. Expected form (SI units throughout):

     setup_temperature_source moser <Q> <lambda> <k> <alpha> <T0> <cp>
                                    <sx> <sy> <sz>

   Q [W]            : total laser power
   lambda [-]       : absorption efficiency (absorbed power = lambda * Q)
   k [W/(m K)]      : thermal conductivity
   alpha [m^2/s]    : thermal diffusivity = k/(rho cp)
   T0 [K]           : ambient / preheat temperature
   cp [J/(kg K)]    : specific heat capacity (used to recover rho)
   sx [m]           : ellipsoid Gaussian width along scan direction
   sy [m]           : ellipsoid Gaussian width perpendicular to scan
   sz [m]           : ellipsoid Gaussian width in depth direction
------------------------------------------------------------------------- */

void MoserGreenTemperatureSource::setup_temperature_source(const std::vector<std::string> &args)
{
  if (args.size() < 9) {
    error->all(FLERR,
      "setup_temperature_source moser: expected <Q> <lambda> <k> <alpha> <T0> <cp> <sx> <sy> <sz>");
  }

  auto parse = [&](size_t i) -> double {
    try { return std::stod(args[i]); }
    catch (const std::exception &) {
      error->all(FLERR,"setup_temperature_source moser: invalid numeric argument");
      return 0.0;
    }
  };

  Q          = parse(0);
  lambda     = parse(1);
  k_th       = parse(2);
  alpha      = parse(3);
  T0_default = parse(4);
  cp         = parse(5);
  sx         = parse(6);
  sy         = parse(7);
  sz         = parse(8);

  if (Q     <= 0.0) error->all(FLERR,"moser: Q must be > 0");
  if (lambda <= 0.0) error->all(FLERR,"moser: lambda must be > 0");
  if (k_th  <= 0.0) error->all(FLERR,"moser: k must be > 0");
  if (alpha <= 0.0) error->all(FLERR,"moser: alpha must be > 0");
  if (cp    <= 0.0) error->all(FLERR,"moser: cp must be > 0");
  if (sx    <= 0.0) error->all(FLERR,"moser: sx must be > 0");
  if (sy    <= 0.0) error->all(FLERR,"moser: sy must be > 0");
  if (sz    <= 0.0) error->all(FLERR,"moser: sz must be > 0");

  // rho derived from alpha = k/(rho cp); used only for diagnostics and
  // to populate the GREENAM ThermalProperties struct.
  rho = k_th / (alpha * cp);

  ambient_temperature = T0_default;
  source_initialized = true;

  if (domain->me == 0) print_source_info();
}

/* ----------------------------------------------------------------------
   Build a piecewise-linear scan from a single straight-line repeated
   path. The waypoint layout for N repeats is:

     wp 0      = (start,           x0, y0)   power lambda*Q  -- scan #0
     wp 1      = (start + L/v,     x1, y1)   power 0         -- transit
     wp 2      = (start + L/v + e, x0, y0)   power lambda*Q  -- scan #1
     wp 3      = (start + 2L/v+e,  x1, y1)   power 0         -- transit
     ...
     wp 2N-1   = (..., x1, y1)               power 0         -- final stop
     wp 2N     = (large_time,      x1, y1)   (unused power)  -- velocity-calc sentinel

   The integrator skips power=0 segments. The trailing large-time
   sentinel is needed because GREENAM derives each segment's velocity
   from waypoint differences, so the array must contain nlines+1
   waypoints even though only nlines segments are stored.
------------------------------------------------------------------------- */

void MoserGreenTemperatureSource::build_scan(double start_time,
                                             double x0, double y0,
                                             double x1, double y1,
                                             double laser_plane_z,
                                             double speed,
                                             int repeats)
{
  if (!source_initialized)
    error->all(FLERR,"moser build_scan: source must be set up first");
  if (repeats < 1) repeats = 1;
  if (speed <= 0.0)
    error->all(FLERR,"moser build_scan: speed must be > 0");

  scan_t_origin = start_time;
  scan_x0 = x0;  scan_y0 = y0;
  scan_x1 = x1;  scan_y1 = y1;
  scan_zl = laser_plane_z;
  scan_speed = speed;
  scan_repeats = repeats;

  const double L = std::sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
  if (L <= 0.0)
    error->all(FLERR,"moser build_scan: zero-length path");
  const double dt_scan = L / speed;
  // Tiny offset between back-to-back repeats so the (xs[i+1]-xs[i])/(ts[i+1]-ts[i])
  // velocity calculation in the GREENAM constructor stays well-defined.
  // The intervening segment has power=0 and is culled by the integrator.
  const double eps_t = 1.0e-12;

  const int n_active = repeats;
  const int n_transit = repeats;          // includes a final 0-power "stop" segment
  const int nlines    = n_active + n_transit;
  const int nwp       = nlines + 1;

  sierra::greenam::VectorWrapper<double> ts("Time",  nwp);
  sierra::greenam::VectorWrapper<double> xs("XLaser", nwp);
  sierra::greenam::VectorWrapper<double> ys("YLaser", nwp);
  sierra::greenam::VectorWrapper<double> ps("Power",  nwp);

  const double abs_power = lambda * Q;
  double t = start_time;
  int wp = 0;

  for (int r = 0; r < repeats; ++r) {
    // active scan segment
    ts(wp) = t;
    xs(wp) = x0;
    ys(wp) = y0;
    ps(wp) = abs_power;
    ++wp;

    t += dt_scan;
    // transit segment (power 0). For the last repeat this is the
    // "stop" sentinel that pegs the laser at (x1, y1) for the
    // remainder of the simulation; the next waypoint extends it to a
    // distant future time so the velocity calc stays finite.
    ts(wp) = t;
    xs(wp) = x1;
    ys(wp) = y1;
    ps(wp) = 0.0;
    ++wp;

    if (r < repeats - 1) {
      // Bring the laser back to (x0, y0) for the next repeat. The
      // tiny eps_t keeps the velocity finite for the wraparound
      // segment we just appended.
      t += eps_t;
    }
  }

  // The time at which the last active emission ends. The fluctuation
  // table needs to cover [start_time, t_end_active]; everything past
  // this is in the 0-power "stop" segment that the integrator culls.
  const double t_end_active = t;

  // Final velocity-calc sentinel waypoint, far in the future. Power
  // unused (only the first nlines power values are stored).
  ts(wp) = t + 1.0e6;
  xs(wp) = x1;
  ys(wp) = y1;
  ps(wp) = 0.0;
  ++wp;

  if (wp != nwp) {
    error->all(FLERR,"moser build_scan: internal waypoint count mismatch");
  }

  ts.commit();
  xs.commit();
  ys.commit();
  ps.commit();

  sierra::greenam::ThermalProperties<double> th{k_th, rho, cp};
  sierra::greenam::EllipsoidProperties<double> ep{sx, sy, sz};

  try {
    scan_ = std::make_shared<GREENAM_Scan>(th, ep, ts, xs, ys, ps, laser_plane_z);
  } catch (const std::runtime_error &e) {
    std::string msg = "moser build_scan: GREENAM LaserScan construction failed: ";
    msg += e.what();
    error->all(FLERR, msg.c_str());
  }
  integrator_ = std::make_shared<GREENAM_Integ>(*scan_);
  scan_built = true;

  if (domain->me == 0) {
    std::cout << "moser: built scan with " << repeats << " repeats, "
              << "L=" << L*1e3 << " mm, dt_scan=" << dt_scan*1e3 << " ms, "
              << "absorbed P=" << abs_power << " W" << std::endl;
  }

  // If a PSD spec was registered before laser_path, materialize the
  // fluctuation table now and attach it to the scan. The table covers
  // the active emission window [start_time, t_end_active]; the 0-power
  // stop segments past t_end_active are culled by the integrator and
  // never see the table.
  if (psd_spec_set) {
    populate_fluctuation_table(start_time, t_end_active);
    scan_->set_fluctuation_table(&fluct_table_);
    if (domain->me == 0) {
      std::cout << "moser: attached fluctuation table with "
                << fluct_table_.size() << " samples, "
                << "t in [" << start_time << ", " << t_end_active
                << "] s" << std::endl;
    }
  }
}

/* ----------------------------------------------------------------------
   Per-site temperature evaluation. Returns absolute temperature [K] at
   the world-space site location and absolute simulation time. Sites
   above the laser plane (z > zl) are vacuum and stay at T0.
------------------------------------------------------------------------- */

double MoserGreenTemperatureSource::get_temperature_at_xyz_and_time(double x, double y,
                                                                    double z, double time)
{
  check_initialization();
  if (!scan_built || !integrator_) return T0_default;

  // Half-space cutoff (matches the existing Rosenthal source).
  if (z > scan_zl) return T0_default;

  const double t_local = time - scan_t_origin;
  if (t_local <= 0.0) return T0_default;

  const double rise = integrator_->integrate_point_adaptive(x, y, z, t_local, char_length);
  return T0_default + rise;
}

/* ---------------------------------------------------------------------- */

void MoserGreenTemperatureSource::update_temperatures(double /*dt*/, double /*simulation_time*/)
{
  // Analytic Green's function: nothing to refresh per timestep.
}

/* ---------------------------------------------------------------------- */

bool MoserGreenTemperatureSource::needs_data_refresh(double /*simulation_time*/)
{
  return false;
}

/* ---------------------------------------------------------------------- */

void MoserGreenTemperatureSource::cleanup()
{
  integrator_.reset();
  scan_.reset();
  scan_built = false;
  fluct_table_.clear();
  psd_reset_filter_state();
  source_initialized = false;
}

/* ---------------------------------------------------------------------- */

void MoserGreenTemperatureSource::print_source_info() const
{
  std::cout << "Moser/Green temperature source (unsteady ellipsoid integral)\n"
            << "  Q          = " << Q          << " W\n"
            << "  lambda     = " << lambda     << "\n"
            << "  P_absorbed = " << lambda * Q << " W\n"
            << "  k          = " << k_th       << " W/(m K)\n"
            << "  alpha      = " << alpha      << " m^2/s\n"
            << "  cp         = " << cp         << " J/(kg K)\n"
            << "  rho        = " << rho        << " kg/m^3 (derived)\n"
            << "  T0         = " << T0_default << " K\n"
            << "  sx, sy, sz = " << sx << ", " << sy << ", " << sz << " m\n"
            << "  char_len   = " << char_length << "\n";
  if (scan_built) {
    std::cout << "  scan: (" << scan_x0 << "," << scan_y0 << ") -> ("
              << scan_x1 << "," << scan_y1 << ") at z=" << scan_zl
              << ", v=" << scan_speed << " m/s, repeats=" << scan_repeats
              << ", t_origin=" << scan_t_origin << " s\n";
  }
  if (psd_spec_set) {
    const char *sname = "white";
    if (psd_spec.shape == PsdShape::LORENTZIAN)       sname = "lorentzian";
    else if (psd_spec.shape == PsdShape::PINK)        sname = "pink";
    else if (psd_spec.shape == PsdShape::NARROW_BAND) sname = "narrow_band";
    std::cout << "  fluctuations: psd " << sname
              << " sigma_W=" << psd_spec.sigma_W
              << " sigma_D=" << psd_spec.sigma_D
              << " sigma_P=" << psd_spec.sigma_P
              << " rho="     << psd_spec.rho
              << " seed="    << psd_spec.seed
              << " dt_psd="  << psd_spec.dt_psd << " s";
    if (psd_spec.shape == PsdShape::LORENTZIAN)
      std::cout << " tau=" << psd_spec.tau;
    if (psd_spec.shape == PsdShape::NARROW_BAND)
      std::cout << " f0=" << psd_spec.f0 << " df=" << psd_spec.df;
    std::cout << "\n";
  }
  std::cout.flush();
}

/* ----------------------------------------------------------------------
   PSD spec stash + filter helpers + table population.

   Time-domain recursive filters with trivariate Gaussian innovations.
   Identical chain on every MPI rank for the same seed (no MPI_Bcast
   needed). The W↔D channels share a Pearson correlation rho; the P
   channel is independent.
------------------------------------------------------------------------- */

void MoserGreenTemperatureSource::set_psd_spec(const PsdSpec &spec)
{
  if (spec.sigma_W < 0.0 || spec.sigma_D < 0.0 || spec.sigma_P < 0.0)
    error->all(FLERR,"laser_fluctuations psd: sigma_W, sigma_D, sigma_P must be >= 0");
  if (spec.rho < -1.0 || spec.rho > 1.0)
    error->all(FLERR,"laser_fluctuations psd: rho must be in [-1, 1]");
  if (spec.dt_psd <= 0.0)
    error->all(FLERR,"laser_fluctuations psd: dt must be > 0");
  switch (spec.shape) {
  case PsdShape::WHITE:       break;
  case PsdShape::LORENTZIAN:
    if (spec.tau <= 0.0)
      error->all(FLERR,"laser_fluctuations psd lorentzian: tau must be > 0");
    break;
  case PsdShape::PINK:        break;
  case PsdShape::NARROW_BAND:
    if (spec.f0 <= 0.0)
      error->all(FLERR,"laser_fluctuations psd narrow_band: f0 must be > 0");
    if (spec.df <= 0.0)
      error->all(FLERR,"laser_fluctuations psd narrow_band: df must be > 0");
    break;
  }

  psd_spec = spec;
  psd_spec_set = true;
}

void MoserGreenTemperatureSource::psd_reset_filter_state()
{
  ar_state_W = 0.0;
  ar_state_D = 0.0;
  ar_state_P = 0.0;
  voss_W.fill(0.0);
  voss_D.fill(0.0);
  voss_P.fill(0.0);
  voss_step = 0;
  osc_x_W = 0.0; osc_v_W = 0.0;
  osc_x_D = 0.0; osc_v_D = 0.0;
  osc_x_P = 0.0; osc_v_P = 0.0;
}

void MoserGreenTemperatureSource::psd_warmup(int n_steps)
{
  double dummy_W = 0.0, dummy_D = 0.0, dummy_P = 0.0;
  for (int i = 0; i < n_steps; ++i) {
    psd_generate_next_sample(dummy_W, dummy_D, dummy_P);
  }
}

void MoserGreenTemperatureSource::psd_draw_trivariate(double &eps_W,
                                                      double &eps_D,
                                                      double &eps_P)
{
  // Three independent unit normals.
  const double z1 = psd_norm(psd_rng);
  const double z2 = psd_norm(psd_rng);
  const double z3 = psd_norm(psd_rng);
  const double rho = psd_spec.rho;
  // W and D share Pearson correlation rho; P is independent.
  eps_W = z1;
  eps_D = rho * z1 + std::sqrt(std::max(0.0, 1.0 - rho * rho)) * z2;
  eps_P = z3;
}

void MoserGreenTemperatureSource::psd_generate_next_sample(double &dW, double &dD, double &dP)
{
  double eps_W, eps_D, eps_P;
  psd_draw_trivariate(eps_W, eps_D, eps_P);

  switch (psd_spec.shape) {

  case PsdShape::WHITE: {
    dW = psd_spec.sigma_W * eps_W;
    dD = psd_spec.sigma_D * eps_D;
    dP = psd_spec.sigma_P * eps_P;
    return;
  }

  case PsdShape::LORENTZIAN: {
    // AR(1) with alpha = exp(-dt/tau): produces exact Lorentzian
    // (Ornstein-Uhlenbeck) autocorrelation in steady state. dt is the
    // emission-time sample spacing (seconds), tau the correlation time.
    const double alpha_ar = std::exp(-psd_spec.dt_psd / psd_spec.tau);
    const double drive_scale = std::sqrt(1.0 - alpha_ar * alpha_ar);
    const double driveW = psd_spec.sigma_W * drive_scale;
    const double driveD = psd_spec.sigma_D * drive_scale;
    const double driveP = psd_spec.sigma_P * drive_scale;
    ar_state_W = alpha_ar * ar_state_W + driveW * eps_W;
    ar_state_D = alpha_ar * ar_state_D + driveD * eps_D;
    ar_state_P = alpha_ar * ar_state_P + driveP * eps_P;
    dW = ar_state_W;
    dD = ar_state_D;
    dP = ar_state_P;
    return;
  }

  case PsdShape::PINK: {
    // Voss-McCartney 1/f: at step n, update only the level k =
    // trailing_zeros(n). Sum of K independent levels approximates a
    // 1/f spectrum across ~K decades.
    ++voss_step;
    int k = 0;
    std::uint64_t n = voss_step;
    while ((n & 1ULL) == 0ULL && k < VOSS_K - 1) { n >>= 1; ++k; }
    const double scale = 1.0 / std::sqrt(static_cast<double>(VOSS_K));
    voss_W[k] = (psd_spec.sigma_W * scale) * eps_W;
    voss_D[k] = (psd_spec.sigma_D * scale) * eps_D;
    voss_P[k] = (psd_spec.sigma_P * scale) * eps_P;
    double sW = 0.0, sD = 0.0, sP = 0.0;
    for (int i = 0; i < VOSS_K; ++i) { sW += voss_W[i]; sD += voss_D[i]; sP += voss_P[i]; }
    dW = sW;
    dD = sD;
    dP = sP;
    return;
  }

  case PsdShape::NARROW_BAND: {
    // Damped harmonic oscillator driven by white noise:
    //   x'' + 2 zeta omega0 x' + omega0^2 x = drive * xi(s)
    //   omega0 = 2 pi f0,  zeta ~ pi df / omega0
    // Stationary variance var_x = drive^2 / (4 zeta omega0^3); pick
    // drive so that var_x = sigma^2. Sub-step the integration so
    // omega0 * dt_sub stays small.
    const double omega0 = 2.0 * MY_PI * psd_spec.f0;
    double zeta = MY_PI * psd_spec.df / omega0;
    if (zeta <= 0.0)  zeta = 1.0e-6;
    if (zeta >= 1.0)  zeta = 0.999;
    int n_sub = static_cast<int>(std::ceil(omega0 * psd_spec.dt_psd / 0.1));
    if (n_sub < 1) n_sub = 1;
    const double dts = psd_spec.dt_psd / n_sub;
    const double var_pref = 4.0 * zeta * omega0 * omega0 * omega0 * dts;
    const double driveW_imp = std::sqrt(var_pref * psd_spec.sigma_W * psd_spec.sigma_W);
    const double driveD_imp = std::sqrt(var_pref * psd_spec.sigma_D * psd_spec.sigma_D);
    const double driveP_imp = std::sqrt(var_pref * psd_spec.sigma_P * psd_spec.sigma_P);
    for (int kstep = 0; kstep < n_sub; ++kstep) {
      double e_W, e_D, e_P;
      if (kstep == 0) { e_W = eps_W; e_D = eps_D; e_P = eps_P; }
      else            { psd_draw_trivariate(e_W, e_D, e_P); }
      // Semi-implicit Euler:
      osc_v_W += dts * (-2.0 * zeta * omega0 * osc_v_W
                        - omega0 * omega0 * osc_x_W) + driveW_imp * e_W;
      osc_x_W += dts * osc_v_W;
      osc_v_D += dts * (-2.0 * zeta * omega0 * osc_v_D
                        - omega0 * omega0 * osc_x_D) + driveD_imp * e_D;
      osc_x_D += dts * osc_v_D;
      osc_v_P += dts * (-2.0 * zeta * omega0 * osc_v_P
                        - omega0 * omega0 * osc_x_P) + driveP_imp * e_P;
      osc_x_P += dts * osc_v_P;
    }
    dW = osc_x_W;
    dD = osc_x_D;
    dP = osc_x_P;
    return;
  }
  }
  // Unreachable
  dW = 0.0; dD = 0.0; dP = 0.0;
}

void MoserGreenTemperatureSource::populate_fluctuation_table(double t_start,
                                                              double t_end)
{
  if (!psd_spec_set)
    error->all(FLERR,"moser populate_fluctuation_table: psd_spec not set");
  if (t_end <= t_start)
    error->all(FLERR,"moser populate_fluctuation_table: empty time interval");

  // Reset the chain for reproducibility (multiple build_scan() calls
  // with the same seed produce the same table).
  psd_rng.seed(psd_spec.seed);
  psd_norm = std::normal_distribution<double>(0.0, 1.0);
  psd_reset_filter_state();
  // Warmup so the filter starts in steady state. Pink needs ~K * 2^K =
  // 384 samples; lorentzian needs ~5*tau/dt; narrow_band needs
  // ~1/(2 zeta omega0)/dt. 1000 covers all of these for typical configs.
  psd_warmup(1000);

  const double dt = psd_spec.dt_psd;
  const std::size_t n_samples =
    static_cast<std::size_t>(std::ceil((t_end - t_start) / dt)) + 1;

  fluct_table_.t_grid.clear();
  fluct_table_.factor_W.clear();
  fluct_table_.factor_D.clear();
  fluct_table_.factor_P.clear();
  fluct_table_.t_grid.reserve(n_samples);
  fluct_table_.factor_W.reserve(n_samples);
  fluct_table_.factor_D.reserve(n_samples);
  fluct_table_.factor_P.reserve(n_samples);

  bool warned_linearization = false;
  for (std::size_t i = 0; i < n_samples; ++i) {
    const double t_i = t_start + i * dt;
    double dW = 0.0, dD = 0.0, dP = 0.0;
    psd_generate_next_sample(dW, dD, dP);

    if (!warned_linearization &&
        (std::fabs(dW) > 0.5 || std::fabs(dD) > 0.5 || std::fabs(dP) > 0.5) &&
        domain->me == 0) {
      std::cout << "WARNING: laser_fluctuations |dW|, |dD|, or |dP| > 0.5; "
                << "first-order linearization on (sx, sy, sz, P) is no longer "
                << "accurate." << std::endl;
      warned_linearization = true;
    }

    double fW = 1.0 + dW;
    double fD = 1.0 + dD;
    double fP = 1.0 + dP;
    // Defensive clamps: keep all factors strictly positive so the
    // integrand can't divide by zero or take a sqrt of a negative.
    if (fW < 1.0e-3) fW = 1.0e-3;
    if (fD < 1.0e-3) fD = 1.0e-3;
    if (fP < 0.0)    fP = 0.0;

    fluct_table_.t_grid.push_back(t_i);
    fluct_table_.factor_W.push_back(fW);
    fluct_table_.factor_D.push_back(fD);
    fluct_table_.factor_P.push_back(fP);
  }
}
