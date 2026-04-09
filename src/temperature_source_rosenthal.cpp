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

#include "temperature_source_rosenthal.h"
#include "error.h"
#include "domain.h"
#include "math_const.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

using namespace SPPARKS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

RosenthalTemperatureSource::RosenthalTemperatureSource(SPPARKS *spk)
  : TemperatureSource(spk),
    mode(RosenthalMode::STANDARD),
    Q(0.0), lambda(0.0), lambda_p(0.0), lambda_l(0.0),
    k_cond(0.0), alpha(0.0), T0_default(300.0),
    eta_y(1.0), eta_z(1.0),
    d_keyhole(0.0), n_quad(8),
    r_min(0.0),
    fluct_periodic(true),
    fluct_total_length(0.0),
    fluct_warned_clamp(false),
    fluct_warned_linearization(false),
    eta_y_eff(1.0), eta_z_eff(1.0),
    psd_active(false),
    psd_rng(12345),
    psd_norm(0.0, 1.0),
    psd_s_prev(0.0), psd_dW_prev(0.0), psd_dD_prev(0.0),
    psd_s_next(0.0), psd_dW_next(0.0), psd_dD_next(0.0),
    ar_state_W(0.0), ar_state_D(0.0),
    voss_step(0),
    osc_x_W(0.0), osc_v_W(0.0), osc_x_D(0.0), osc_v_D(0.0)
{
  ambient_temperature = T0_default;
  voss_W.fill(0.0);
  voss_D.fill(0.0);
  psd_spec = PsdSpec{PsdShape::WHITE, 0.0, 0.0, 0.0, 12345, 0.0, 0.0, 0.0, 0.0};
}

/* ---------------------------------------------------------------------- */

RosenthalTemperatureSource::~RosenthalTemperatureSource()
{
  cleanup();
}

/* ----------------------------------------------------------------------
   Setup. Expected forms (SI units throughout):

   standard <Q> <lambda>   <k> <alpha> <T0>
   aniso    <Q> <lambda>   <k> <alpha> <T0> <eta_y> <eta_z>
   keyhole  <Q> <lambda_p> <lambda_l> <k> <alpha> <T0> <d> [<n_quad>]
------------------------------------------------------------------------- */

void RosenthalTemperatureSource::setup_temperature_source(const std::vector<std::string> &args)
{
  if (args.empty()) {
    error->all(FLERR,"setup_temperature_source rosenthal: missing mode (standard|aniso|keyhole)");
  }

  const std::string &mode_str = args[0];

  auto parse = [&](size_t i, const char *what) -> double {
    if (i >= args.size())
      error->all(FLERR,"setup_temperature_source rosenthal: missing argument");
    try {
      return std::stod(args[i]);
    } catch (const std::exception &) {
      (void)what;
      error->all(FLERR,"setup_temperature_source rosenthal: invalid numeric argument");
      return 0.0;
    }
  };

  if (mode_str == "standard") {
    if (args.size() < 6)
      error->all(FLERR,"rosenthal standard: expected <Q> <lambda> <k> <alpha> <T0>");
    mode       = RosenthalMode::STANDARD;
    Q          = parse(1,"Q");
    lambda     = parse(2,"lambda");
    k_cond     = parse(3,"k");
    alpha      = parse(4,"alpha");
    T0_default = parse(5,"T0");
  }
  else if (mode_str == "aniso" || mode_str == "anisotropic") {
    if (args.size() < 8)
      error->all(FLERR,"rosenthal aniso: expected <Q> <lambda> <k> <alpha> <T0> <eta_y> <eta_z>");
    mode       = RosenthalMode::ANISOTROPIC;
    Q          = parse(1,"Q");
    lambda     = parse(2,"lambda");
    k_cond     = parse(3,"k");
    alpha      = parse(4,"alpha");
    T0_default = parse(5,"T0");
    eta_y      = parse(6,"eta_y");
    eta_z      = parse(7,"eta_z");
  }
  else if (mode_str == "keyhole") {
    if (args.size() < 8)
      error->all(FLERR,"rosenthal keyhole: expected <Q> <lambda_p> <lambda_l> <k> <alpha> <T0> <d> [<n_quad>]");
    mode       = RosenthalMode::KEYHOLE;
    Q          = parse(1,"Q");
    lambda_p   = parse(2,"lambda_p");
    lambda_l   = parse(3,"lambda_l");
    k_cond     = parse(4,"k");
    alpha      = parse(5,"alpha");
    T0_default = parse(6,"T0");
    d_keyhole  = parse(7,"d");
    if (args.size() >= 9) {
      n_quad = static_cast<int>(parse(8,"n_quad"));
      if (n_quad < 2) n_quad = 2;
      if (n_quad > 64) n_quad = 64;
    }
    build_gauss_legendre_quadrature();
  }
  else {
    error->all(FLERR,"setup_temperature_source rosenthal: unknown mode (use standard|aniso|keyhole)");
  }

  ambient_temperature = T0_default;
  // Initialize per-step eta cache to nominal values; set_arc_length()
  // overwrites these when fluctuations are loaded.
  eta_y_eff = eta_y;
  eta_z_eff = eta_z;
  validate_after_setup();
  source_initialized = true;

  if (domain->me == 0) print_source_info();
}

/* ----------------------------------------------------------------------
   Required base-class entry point. The driving app uses
   rosenthal_pointwise() directly with pool-local coordinates, but this
   xyz/time API is honored as a fallback that returns the ambient T
   (no path means no laser position).
------------------------------------------------------------------------- */

double RosenthalTemperatureSource::get_temperature_at_xyz_and_time(double /*x*/, double /*y*/,
                                                                   double /*z*/, double /*time*/)
{
  check_initialization();
  return ambient_temperature;
}

/* ---------------------------------------------------------------------- */

void RosenthalTemperatureSource::update_temperatures(double /*dt*/, double /*simulation_time*/)
{
  // Analytical solution: nothing to load.
}

/* ---------------------------------------------------------------------- */

bool RosenthalTemperatureSource::needs_data_refresh(double /*simulation_time*/)
{
  return false;
}

/* ----------------------------------------------------------------------
   Fast-forward query: the source itself has no path geometry, so it
   declares "yes I support time queries" but returns +inf here. The
   driving app intercepts and computes the real next-active time using
   its scan_layer state.
------------------------------------------------------------------------- */

double RosenthalTemperatureSource::get_next_time_with_temperature(double /*current_time*/,
                                                                  double /*threshold_temp*/)
{
  return std::numeric_limits<double>::max();
}

/* ----------------------------------------------------------------------
   Core kernel (Eq. B4 form):
   T_rise = (Q_eff / (2 pi k R)) * exp(-v (xi + R) / (2 alpha))
------------------------------------------------------------------------- */

double RosenthalTemperatureSource::rosenthal_kernel(double Q_eff, double v,
                                                    double xi, double R) const
{
  if (R < r_min) R = r_min;
  if (R <= 0.0)  return 0.0;
  const double prefactor = Q_eff / (2.0 * MY_PI * k_cond * R);
  const double exponent  = -v * (xi + R) / (2.0 * alpha);
  return prefactor * std::exp(exponent);
}

/* ----------------------------------------------------------------------
   Mode dispatch evaluated in the moving (pool-local) frame on a free
   surface. Returns absolute temperature in K.

   Half-space convention:
     - z_rel > 0  (above the laser plane): no material, return T0
     - z_rel <= 0 (below the laser plane): apply the image-source method,
       which doubles the unbounded-medium kernel result. The factor of 2
       in front of every kernel call below is the explicit image source.
------------------------------------------------------------------------- */

double RosenthalTemperatureSource::rosenthal_pointwise(double xi, double y_rel, double z_rel,
                                                       double v, double T0) const
{
  // Free-surface cutoff: anything above the laser plane is vacuum.
  if (z_rel > 0.0) return T0;

  switch (mode) {

  case RosenthalMode::STANDARD: {
    const double R = std::sqrt(xi*xi + y_rel*y_rel + z_rel*z_rel);
    return T0 + 2.0 * rosenthal_kernel(lambda * Q, v, xi, R);
  }

  case RosenthalMode::ANISOTROPIC: {
    // Use the cached effective etas (set per-step by set_arc_length when
    // fluctuations are loaded; equal to the nominal etas otherwise).
    const double yy = eta_y_eff * y_rel;
    const double zz = eta_z_eff * z_rel;
    const double R_eta = std::sqrt(xi*xi + yy*yy + zz*zz);
    return T0 + 2.0 * rosenthal_kernel(lambda * Q, v, xi, R_eta);
  }

  case RosenthalMode::KEYHOLE: {
    // Surface point source (image-doubled)
    const double R = std::sqrt(xi*xi + y_rel*y_rel + z_rel*z_rel);
    double T_rise = 2.0 * rosenthal_kernel(lambda_p * Q, v, xi, R);

    // Distributed line source (image-doubled at every quadrature node):
    // R'(D) = sqrt(xi^2 + y^2 + (D + z)^2). Each quadrature node sits
    // on a buried line element whose image lies above the surface, so
    // the same factor of 2 applies elementwise.
    for (int i = 0; i < n_quad; ++i) {
      const double D    = quad_nodes[i];
      const double w    = quad_weights[i];
      const double zarg = D + z_rel;
      const double Rp   = std::sqrt(xi*xi + y_rel*y_rel + zarg*zarg);
      T_rise += 2.0 * w * rosenthal_kernel(lambda_l * Q, v, xi, Rp);
    }
    return T0 + T_rise;
  }
  }
  return T0;  // unreachable
}

/* ----------------------------------------------------------------------
   Conservative upper bound on Rosenthal temperature at distance >= R.
   Always >= the true rosenthal_pointwise() value for any site at that
   distance, so the fast-forward predictor cannot skip a real heating
   event by relying on this bound.

   Note: includes the half-space image-source factor of 2 to match
   rosenthal_pointwise().
------------------------------------------------------------------------- */

double RosenthalTemperatureSource::rosenthal_peak_at_distance(double R, double T0) const
{
  if (R < r_min) R = r_min;
  if (R <= 0.0) return T0;

  // ANISOTROPIC: R_eta <= R / min(eta_y, eta_z, 1) for any (xi,y,z)
  // satisfying xi^2+y^2+z^2 = R^2. Use the smallest effective R.
  // STANDARD/KEYHOLE: R unchanged.
  double R_eff = R;
  if (mode == RosenthalMode::ANISOTROPIC) {
    double s = 1.0;
    if (eta_y < s) s = eta_y;
    if (eta_z < s) s = eta_z;
    if (s > 0.0) R_eff = R * s;
    if (R_eff < r_min) R_eff = r_min;
  }

  // Sum the maximum kernel contributions for the active mode. The
  // exponential factor exp(-v*(xi+R)/(2 alpha)) <= 1 always (it equals
  // 1 when xi = -R, i.e. straight downstream), so the absolute peak is
  //   2 * Q_eff_total / (2 pi k R_eff)        (factor 2 = image source)
  const double prefactor = 2.0 / (2.0 * MY_PI * k_cond * R_eff);

  double T_rise = 0.0;
  switch (mode) {
  case RosenthalMode::STANDARD:
  case RosenthalMode::ANISOTROPIC:
    T_rise = lambda * Q * prefactor;
    break;
  case RosenthalMode::KEYHOLE:
    // Point source at R_eff (= R since not ANISOTROPIC) plus a coarse
    // upper bound on the line integral: each quadrature node has the
    // same prefactor cap, so sum the weights and multiply.
    {
      double w_total = 0.0;
      for (int i = 0; i < n_quad; ++i) w_total += quad_weights[i];
      T_rise = (lambda_p * Q + lambda_l * Q * w_total) * prefactor;
    }
    break;
  }
  return T0 + T_rise;
}

/* ----------------------------------------------------------------------
   Build Gauss-Legendre nodes/weights on [-d, d] for the keyhole line
   integral. Uses Newton iteration on Legendre polynomials.
------------------------------------------------------------------------- */

void RosenthalTemperatureSource::build_gauss_legendre_quadrature()
{
  quad_nodes.assign(n_quad, 0.0);
  quad_weights.assign(n_quad, 0.0);

  // Standard Newton-Raphson on Legendre P_n to find roots in [-1,1]
  const int n = n_quad;
  const double tol = 1.0e-14;
  for (int i = 0; i < (n + 1) / 2; ++i) {
    // Initial guess (Tricomi/asymptotic)
    double x = std::cos(MY_PI * (i + 0.75) / (n + 0.5));
    double dp = 0.0;
    for (int it = 0; it < 100; ++it) {
      // Evaluate P_n(x) and P_n'(x) by recurrence
      double p1 = 1.0, p2 = 0.0;
      for (int j = 1; j <= n; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0*j - 1.0) * x * p2 - (j - 1.0) * p3) / j;
      }
      dp = n * (x * p1 - p2) / (x * x - 1.0);
      const double dx = p1 / dp;
      x -= dx;
      if (std::fabs(dx) < tol) break;
    }
    // Map from [-1, 1] to [-d, d]
    quad_nodes[i]         = -d_keyhole * x;
    quad_nodes[n - 1 - i] =  d_keyhole * x;
    const double w = 2.0 / ((1.0 - x*x) * dp * dp);
    // Jacobian for [-1,1] -> [-d,d] is d
    quad_weights[i]         = d_keyhole * w;
    quad_weights[n - 1 - i] = d_keyhole * w;
  }
}

/* ---------------------------------------------------------------------- */

void RosenthalTemperatureSource::validate_after_setup() const
{
  if (Q <= 0.0)      error->all(FLERR,"rosenthal: Q (laser power) must be > 0");
  if (k_cond <= 0.0) error->all(FLERR,"rosenthal: k (thermal conductivity) must be > 0");
  if (alpha <= 0.0)  error->all(FLERR,"rosenthal: alpha (thermal diffusivity) must be > 0");

  switch (mode) {
  case RosenthalMode::STANDARD:
    if (lambda <= 0.0) error->all(FLERR,"rosenthal standard: lambda must be > 0");
    break;
  case RosenthalMode::ANISOTROPIC:
    if (lambda <= 0.0) error->all(FLERR,"rosenthal aniso: lambda must be > 0");
    if (eta_y <= 0.0 || eta_z <= 0.0)
      error->all(FLERR,"rosenthal aniso: eta_y and eta_z must be > 0");
    break;
  case RosenthalMode::KEYHOLE:
    if (lambda_p <= 0.0 || lambda_l <= 0.0)
      error->all(FLERR,"rosenthal keyhole: lambda_p and lambda_l must be > 0");
    if (d_keyhole <= 0.0)
      error->all(FLERR,"rosenthal keyhole: d (line-source half-depth) must be > 0");
    if (static_cast<int>(quad_nodes.size()) != n_quad)
      error->all(FLERR,"rosenthal keyhole: quadrature was not built");
    break;
  }
}

/* ---------------------------------------------------------------------- */

void RosenthalTemperatureSource::print_source_info() const
{
  const char *mname = "STANDARD";
  if (mode == RosenthalMode::ANISOTROPIC) mname = "ANISOTROPIC";
  else if (mode == RosenthalMode::KEYHOLE) mname = "KEYHOLE";

  std::cout << "Rosenthal temperature source [" << mname << "]\n"
            << "  Q          = " << Q          << " W\n"
            << "  k          = " << k_cond     << " W/(m K)\n"
            << "  alpha      = " << alpha      << " m^2/s\n"
            << "  T0         = " << T0_default << " K\n"
            << "  r_min      = " << r_min      << " m\n";
  if (mode == RosenthalMode::STANDARD) {
    std::cout << "  lambda     = " << lambda << "\n";
  } else if (mode == RosenthalMode::ANISOTROPIC) {
    std::cout << "  lambda     = " << lambda << "\n"
              << "  eta_y      = " << eta_y  << "\n"
              << "  eta_z      = " << eta_z  << "\n";
  } else {
    std::cout << "  lambda_p   = " << lambda_p  << "\n"
              << "  lambda_l   = " << lambda_l  << "\n"
              << "  d_keyhole  = " << d_keyhole << " m\n"
              << "  n_quad     = " << n_quad    << "\n";
  }
  if (!fluct_s.empty()) {
    std::cout << "  fluctuations: " << fluct_s.size() << " samples, "
              << "s in [" << fluct_s.front() << ", " << fluct_s.back() << "] m, "
              << (fluct_periodic ? "periodic" : "continuous") << "\n";
  }
  if (psd_active) {
    const char *sname = "white";
    if (psd_spec.shape == PsdShape::LORENTZIAN)  sname = "lorentzian";
    else if (psd_spec.shape == PsdShape::PINK)   sname = "pink";
    else if (psd_spec.shape == PsdShape::NARROW_BAND) sname = "narrow_band";
    std::cout << "  fluctuations: psd " << sname
              << " sigma_W=" << psd_spec.sigma_W
              << " sigma_D=" << psd_spec.sigma_D
              << " rho="     << psd_spec.rho
              << " dx="      << psd_spec.dx << "\n";
  }
  std::cout.flush();
}

/* ----------------------------------------------------------------------
   Stochastic ΔW/W, ΔD/D modulation of eta_y, eta_z (anisotropic mode).
------------------------------------------------------------------------- */

void RosenthalTemperatureSource::load_fluctuations(const std::vector<double> &s,
                                                   const std::vector<double> &dW_over_W,
                                                   const std::vector<double> &dD_over_D,
                                                   bool periodic)
{
  if (s.empty() || s.size() != dW_over_W.size() || s.size() != dD_over_D.size()) {
    error->all(FLERR,"rosenthal_fluctuations: arrays must be non-empty and equal length");
  }
  for (size_t i = 1; i < s.size(); ++i) {
    if (s[i] <= s[i-1])
      error->all(FLERR,"rosenthal_fluctuations: s column must be strictly increasing");
  }
  for (size_t i = 0; i < s.size(); ++i) {
    if (std::fabs(dW_over_W[i]) >= 1.0 || std::fabs(dD_over_D[i]) >= 1.0)
      error->all(FLERR,"rosenthal_fluctuations: |dW/W| and |dD/D| must be < 1");
  }
  fluct_s        = s;
  fluct_dW       = dW_over_W;
  fluct_dD       = dD_over_D;
  fluct_periodic = periodic;
  fluct_total_length = fluct_s.back() - fluct_s.front();
  fluct_warned_clamp = false;
  fluct_warned_linearization = false;
  // Reset cache to nominal until set_arc_length is called.
  eta_y_eff = eta_y;
  eta_z_eff = eta_z;
}

void RosenthalTemperatureSource::set_arc_length(double s)
{
  // -------- PSD streaming generator path ----------
  if (psd_active) {
    // Advance the streaming chain until s lies in [psd_s_prev, psd_s_next].
    while (s > psd_s_next) {
      psd_s_prev   = psd_s_next;
      psd_dW_prev  = psd_dW_next;
      psd_dD_prev  = psd_dD_next;
      psd_s_next  += psd_spec.dx;
      psd_generate_next_sample(psd_dW_next, psd_dD_next);
    }
    double t = (psd_s_next > psd_s_prev)
                 ? (s - psd_s_prev) / (psd_s_next - psd_s_prev)
                 : 0.0;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    double dW = (1.0 - t) * psd_dW_prev + t * psd_dW_next;
    double dD = (1.0 - t) * psd_dD_prev + t * psd_dD_next;

    if (!fluct_warned_linearization &&
        (std::fabs(dW) > 0.5 || std::fabs(dD) > 0.5) && domain->me == 0) {
      std::cout << "WARNING: rosenthal_fluctuations |dW/W| or |dD/D| > 0.5; "
                << "first-order linearization eta_eff = eta*(1 - dW/W) is no "
                << "longer accurate." << std::endl;
      fluct_warned_linearization = true;
    }

    eta_y_eff = eta_y * (1.0 - dW);
    eta_z_eff = eta_z * (1.0 - dD);
    if (eta_y_eff < 1.0e-3) eta_y_eff = 1.0e-3;
    if (eta_z_eff < 1.0e-3) eta_z_eff = 1.0e-3;
    return;
  }

  // -------- File-loaded fluctuation path ----------
  if (fluct_s.empty()) return;

  double s_lookup = s;
  if (fluct_periodic) {
    if (fluct_total_length > 0.0) {
      s_lookup = fluct_s.front()
               + std::fmod(s - fluct_s.front(), fluct_total_length);
      if (s_lookup < fluct_s.front()) s_lookup += fluct_total_length;
    } else {
      s_lookup = fluct_s.front();
    }
  } else {
    if (s_lookup > fluct_s.back()) {
      if (!fluct_warned_clamp && domain->me == 0) {
        std::cout << "WARNING: rosenthal_fluctuations continuous track exhausted "
                  << "(s=" << s << " > " << fluct_s.back()
                  << "); clamping to last value." << std::endl;
        fluct_warned_clamp = true;
      }
      s_lookup = fluct_s.back();
    }
    if (s_lookup < fluct_s.front()) s_lookup = fluct_s.front();
  }

  // Linear interp via lower_bound on fluct_s.
  auto it = std::lower_bound(fluct_s.begin(), fluct_s.end(), s_lookup);
  std::ptrdiff_t idx = it - fluct_s.begin();
  if (idx <= 0) idx = 1;
  if (static_cast<size_t>(idx) >= fluct_s.size()) idx = fluct_s.size() - 1;
  const size_t i = static_cast<size_t>(idx);
  const double s0 = fluct_s[i-1];
  const double s1 = fluct_s[i];
  const double t  = (s1 > s0) ? (s_lookup - s0) / (s1 - s0) : 0.0;
  const double dW = (1.0 - t) * fluct_dW[i-1] + t * fluct_dW[i];
  const double dD = (1.0 - t) * fluct_dD[i-1] + t * fluct_dD[i];

  if (!fluct_warned_linearization &&
      (std::fabs(dW) > 0.5 || std::fabs(dD) > 0.5) && domain->me == 0) {
    std::cout << "WARNING: rosenthal_fluctuations |dW/W| or |dD/D| > 0.5; "
              << "first-order linearization eta_eff = eta*(1 - dW/W) is no longer "
              << "accurate." << std::endl;
    fluct_warned_linearization = true;
  }

  eta_y_eff = eta_y * (1.0 - dW);
  eta_z_eff = eta_z * (1.0 - dD);
  // Defensive clamp: keep effective etas strictly positive.
  if (eta_y_eff < 1.0e-3) eta_y_eff = 1.0e-3;
  if (eta_z_eff < 1.0e-3) eta_z_eff = 1.0e-3;
}

void RosenthalTemperatureSource::promote_to_anisotropic()
{
  if (mode == RosenthalMode::ANISOTROPIC) return;
  if (mode == RosenthalMode::KEYHOLE) {
    error->all(FLERR,"rosenthal_fluctuations: not supported with keyhole mode");
  }
  // STANDARD -> ANISOTROPIC with unit etas (kernel result is identical).
  mode = RosenthalMode::ANISOTROPIC;
  eta_y = 1.0;
  eta_z = 1.0;
  eta_y_eff = 1.0;
  eta_z_eff = 1.0;
  if (domain->me == 0) {
    std::cout << "rosenthal_fluctuations: auto-promoted STANDARD source to "
              << "ANISOTROPIC (eta_y=eta_z=1)" << std::endl;
  }
}

/* ----------------------------------------------------------------------
   In-source PSD streaming generator. Time-domain recursive filters with
   bivariate Gaussian innovations to enforce W↔D Pearson correlation.
   Identical chain on every MPI rank (same seed) so no MPI_Bcast needed.
------------------------------------------------------------------------- */

void RosenthalTemperatureSource::init_psd_generator(const PsdSpec &spec)
{
  if (spec.sigma_W < 0.0 || spec.sigma_D < 0.0)
    error->all(FLERR,"rosenthal_fluctuations psd: sigma_W and sigma_D must be >= 0");
  if (spec.rho < -1.0 || spec.rho > 1.0)
    error->all(FLERR,"rosenthal_fluctuations psd: rho must be in [-1, 1]");
  if (spec.dx <= 0.0)
    error->all(FLERR,"rosenthal_fluctuations psd: dx must be > 0");
  switch (spec.shape) {
  case PsdShape::WHITE:
    break;
  case PsdShape::LORENTZIAN:
    if (spec.tau <= 0.0)
      error->all(FLERR,"rosenthal_fluctuations psd lorentzian: tau must be > 0");
    break;
  case PsdShape::PINK:
    break;
  case PsdShape::NARROW_BAND:
    if (spec.f0 <= 0.0)
      error->all(FLERR,"rosenthal_fluctuations psd narrow_band: f0 must be > 0");
    if (spec.df <= 0.0)
      error->all(FLERR,"rosenthal_fluctuations psd narrow_band: df must be > 0");
    break;
  }

  psd_spec = spec;
  psd_active = true;
  psd_rng.seed(spec.seed);
  psd_norm = std::normal_distribution<double>(0.0, 1.0);
  fluct_warned_linearization = false;

  psd_reset_filter_state();
  // Warm up the filter so it starts in (or near) steady state. Pink needs
  // ~K * 2^K = 384 samples; lorentzian needs ~5*tau/dx; narrow_band needs
  // ~1/(2*zeta*omega0)/dx. 1000 covers all of these for typical configs.
  psd_warmup(1000);

  // Seed the two-sample interpolation window.
  psd_s_prev  = 0.0;
  psd_dW_prev = 0.0;
  psd_dD_prev = 0.0;
  psd_generate_next_sample(psd_dW_prev, psd_dD_prev);
  psd_s_next  = spec.dx;
  psd_generate_next_sample(psd_dW_next, psd_dD_next);

  if (domain->me == 0) {
    const char *sname = "white";
    if (spec.shape == PsdShape::LORENTZIAN)  sname = "lorentzian";
    else if (spec.shape == PsdShape::PINK)    sname = "pink";
    else if (spec.shape == PsdShape::NARROW_BAND) sname = "narrow_band";
    std::cout << "rosenthal_fluctuations psd: shape=" << sname
              << " sigma_W=" << spec.sigma_W
              << " sigma_D=" << spec.sigma_D
              << " rho="     << spec.rho
              << " seed="    << spec.seed
              << " dx="      << spec.dx;
    if (spec.shape == PsdShape::LORENTZIAN)  std::cout << " tau=" << spec.tau;
    if (spec.shape == PsdShape::NARROW_BAND) std::cout << " f0="  << spec.f0
                                                       << " df="  << spec.df;
    std::cout << std::endl;
  }
}

void RosenthalTemperatureSource::psd_reset_filter_state()
{
  ar_state_W = 0.0;
  ar_state_D = 0.0;
  voss_W.fill(0.0);
  voss_D.fill(0.0);
  voss_step = 0;
  osc_x_W = 0.0; osc_v_W = 0.0;
  osc_x_D = 0.0; osc_v_D = 0.0;
}

void RosenthalTemperatureSource::psd_warmup(int n_steps)
{
  double dummy_W = 0.0, dummy_D = 0.0;
  for (int i = 0; i < n_steps; ++i) {
    psd_generate_next_sample(dummy_W, dummy_D);
  }
}

void RosenthalTemperatureSource::psd_draw_bivariate(double &eps_W, double &eps_D)
{
  const double z1 = psd_norm(psd_rng);
  const double z2 = psd_norm(psd_rng);
  const double rho = psd_spec.rho;
  eps_W = z1;
  eps_D = rho * z1 + std::sqrt(std::max(0.0, 1.0 - rho * rho)) * z2;
}

void RosenthalTemperatureSource::psd_generate_next_sample(double &dW, double &dD)
{
  double eps_W, eps_D;
  psd_draw_bivariate(eps_W, eps_D);

  switch (psd_spec.shape) {

  case PsdShape::WHITE: {
    dW = psd_spec.sigma_W * eps_W;
    dD = psd_spec.sigma_D * eps_D;
    return;
  }

  case PsdShape::LORENTZIAN: {
    // AR(1) with alpha = exp(-dx/tau): produces exact Lorentzian (Ornstein-
    // Uhlenbeck) autocorrelation in steady state.
    const double alpha   = std::exp(-psd_spec.dx / psd_spec.tau);
    const double driveW  = psd_spec.sigma_W * std::sqrt(1.0 - alpha * alpha);
    const double driveD  = psd_spec.sigma_D * std::sqrt(1.0 - alpha * alpha);
    ar_state_W = alpha * ar_state_W + driveW * eps_W;
    ar_state_D = alpha * ar_state_D + driveD * eps_D;
    dW = ar_state_W;
    dD = ar_state_D;
    return;
  }

  case PsdShape::PINK: {
    // Voss-McCartney 1/f: at step n, update only the level k =
    // trailing_zeros(n). Sum of K independent levels approximates a 1/f
    // spectrum across ~K decades.
    ++voss_step;
    int k = 0;
    std::uint64_t n = voss_step;
    while ((n & 1ULL) == 0ULL && k < VOSS_K - 1) { n >>= 1; ++k; }
    const double scale = 1.0 / std::sqrt(static_cast<double>(VOSS_K));
    voss_W[k] = (psd_spec.sigma_W * scale) * eps_W;
    voss_D[k] = (psd_spec.sigma_D * scale) * eps_D;
    double sW = 0.0, sD = 0.0;
    for (int i = 0; i < VOSS_K; ++i) { sW += voss_W[i]; sD += voss_D[i]; }
    dW = sW;
    dD = sD;
    return;
  }

  case PsdShape::NARROW_BAND: {
    // Damped harmonic oscillator driven by white noise:
    //   x'' + 2ζω₀ x' + ω₀² x = drive · ξ(s)
    //   ω₀ = 2π f₀, ζ ≈ π df / ω₀
    // Stationary variance var_x = drive² / (4 ζ ω₀³); pick drive so that
    // var_x = sigma². Sub-step the integration so that ω₀ * dt_sub << 1.
    const double omega0 = 2.0 * MY_PI * psd_spec.f0;
    double zeta = MY_PI * psd_spec.df / omega0;
    if (zeta <= 0.0)  zeta = 1.0e-6;
    if (zeta >= 1.0)  zeta = 0.999;  // keep underdamped
    // sub-step until omega0*dt_sub <= 0.1 (~3% phase error per substep)
    int n_sub = static_cast<int>(std::ceil(omega0 * psd_spec.dx / 0.1));
    if (n_sub < 1) n_sub = 1;
    const double dts = psd_spec.dx / n_sub;
    // Variance prefactor for this sub-step length:
    //   drive_var = 4 ζ ω₀³ σ²    (continuous-time intensity)
    //   per sub-step impulse std = sqrt(drive_var * dts)
    const double driveW_imp = std::sqrt(4.0 * zeta * omega0 * omega0 * omega0
                                        * psd_spec.sigma_W * psd_spec.sigma_W
                                        * dts);
    const double driveD_imp = std::sqrt(4.0 * zeta * omega0 * omega0 * omega0
                                        * psd_spec.sigma_D * psd_spec.sigma_D
                                        * dts);
    for (int k = 0; k < n_sub; ++k) {
      // Re-draw innovations for each sub-step. The first sub-step uses the
      // already-drawn (eps_W, eps_D); subsequent sub-steps draw their own
      // correlated pair so the rho structure is preserved.
      double e_W, e_D;
      if (k == 0) { e_W = eps_W; e_D = eps_D; }
      else        { psd_draw_bivariate(e_W, e_D); }
      // Semi-implicit Euler:
      osc_v_W += dts * (-2.0 * zeta * omega0 * osc_v_W
                        - omega0 * omega0 * osc_x_W) + driveW_imp * e_W;
      osc_x_W += dts * osc_v_W;
      osc_v_D += dts * (-2.0 * zeta * omega0 * osc_v_D
                        - omega0 * omega0 * osc_x_D) + driveD_imp * e_D;
      osc_x_D += dts * osc_v_D;
    }
    dW = osc_x_W;
    dD = osc_x_D;
    return;
  }
  }
  // Unreachable
  dW = 0.0;
  dD = 0.0;
}

/* ---------------------------------------------------------------------- */

void RosenthalTemperatureSource::cleanup()
{
  quad_nodes.clear();
  quad_weights.clear();
  fluct_s.clear();
  fluct_dW.clear();
  fluct_dD.clear();
  fluct_total_length = 0.0;
  fluct_warned_clamp = false;
  fluct_warned_linearization = false;
  psd_active = false;
  psd_reset_filter_state();
  source_initialized = false;
}
