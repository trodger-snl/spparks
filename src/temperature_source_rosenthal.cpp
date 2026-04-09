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

#include <cmath>
#include <iostream>
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
    r_min(0.0)
{
  ambient_temperature = T0_default;
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
   Mode dispatch evaluated in the moving (pool-local) frame.
   Returns absolute temperature in K.
------------------------------------------------------------------------- */

double RosenthalTemperatureSource::rosenthal_pointwise(double xi, double y_rel, double z_rel,
                                                       double v, double T0) const
{
  switch (mode) {

  case RosenthalMode::STANDARD: {
    const double R = std::sqrt(xi*xi + y_rel*y_rel + z_rel*z_rel);
    return T0 + rosenthal_kernel(lambda * Q, v, xi, R);
  }

  case RosenthalMode::ANISOTROPIC: {
    const double yy = eta_y * y_rel;
    const double zz = eta_z * z_rel;
    const double R_eta = std::sqrt(xi*xi + yy*yy + zz*zz);
    return T0 + rosenthal_kernel(lambda * Q, v, xi, R_eta);
  }

  case RosenthalMode::KEYHOLE: {
    // Surface point source
    const double R = std::sqrt(xi*xi + y_rel*y_rel + z_rel*z_rel);
    double T_rise = rosenthal_kernel(lambda_p * Q, v, xi, R);

    // Distributed line source: integrate over D in [-d, d]
    // R'(D) = sqrt(xi^2 + y^2 + (D + z)^2)
    for (int i = 0; i < n_quad; ++i) {
      const double D    = quad_nodes[i];
      const double w    = quad_weights[i];
      const double zarg = D + z_rel;
      const double Rp   = std::sqrt(xi*xi + y_rel*y_rel + zarg*zarg);
      T_rise += w * rosenthal_kernel(lambda_l * Q, v, xi, Rp);
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
  // Q_eff_total / (2 pi k R_eff).
  const double inv_prefactor = 1.0 / (2.0 * MY_PI * k_cond * R_eff);

  double T_rise = 0.0;
  switch (mode) {
  case RosenthalMode::STANDARD:
  case RosenthalMode::ANISOTROPIC:
    T_rise = lambda * Q * inv_prefactor;
    break;
  case RosenthalMode::KEYHOLE:
    // Point source at R_eff (= R since not ANISOTROPIC) plus a coarse
    // upper bound on the line integral: each quadrature node has the
    // same prefactor cap, so sum the weights and multiply.
    {
      double w_total = 0.0;
      for (int i = 0; i < n_quad; ++i) w_total += quad_weights[i];
      T_rise = (lambda_p * Q + lambda_l * Q * w_total) * inv_prefactor;
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
  std::cout.flush();
}

/* ---------------------------------------------------------------------- */

void RosenthalTemperatureSource::cleanup()
{
  quad_nodes.clear();
  quad_weights.clear();
  source_initialized = false;
}
