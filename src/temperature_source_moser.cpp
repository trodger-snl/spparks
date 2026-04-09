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

#include "GREENAM/GreenAM_Constants.h"
#include "GREENAM/GreenAM_Properties.h"
#include "GREENAM/GreenAM_Util.h"
#include "GREENAM/GreenAM_LaserScan.h"
#include "GREENAM/GreenAM_ScanIntegration.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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
    scan_built(false)
{
  ambient_temperature = T0_default;
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
  std::cout.flush();
}
