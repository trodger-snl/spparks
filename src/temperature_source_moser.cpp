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
    mode(Mode::STANDARD),
    char_length(0.5),
    scan_t_origin(0.0),
    scan_x0(0.0), scan_y0(0.0), scan_x1(0.0), scan_y1(0.0),
    scan_laser_plane_z(0.0),
    scan_speed(0.0), scan_repeats(0),
    scan_built(false),
    n_lobes_(1)
{
  // Lobe members are default-initialized via the Lobe struct's in-class
  // member-initializers (sx=sy=sz=0, power_fraction=1, rng seeded,
  // filter state zeroed, fluct_table empty).
  ambient_temperature = T0_default;
}

/* ---------------------------------------------------------------------- */

MoserGreenTemperatureSource::~MoserGreenTemperatureSource()
{
  cleanup();
}

/* ----------------------------------------------------------------------
   Setup. Two forms are supported (SI units throughout):

     (1) STANDARD (single ellipsoid; legacy form, no mode token):

           setup_temperature_source moser <Q> <lambda> <k> <alpha> <T0>
                                          <cp> <sx> <sy> <sz>

         Equivalently, the explicit form:

           setup_temperature_source moser standard <Q> <lambda> <k>
                                          <alpha> <T0> <cp> <sx> <sy> <sz>

     (2) KEYHOLE (overlapping double-ellipsoid, Goldak-style):

           setup_temperature_source moser keyhole <Q> <lambda> <k>
                                          <alpha> <T0> <cp>
                                          <f_top> <sx_top> <sy_top>
                                                  <sz_top> <z_top>
                                          <f_bot> <sx_bot> <sy_bot>
                                                  <sz_bot> <z_bot>

         f_top and f_bot must sum to 1 (within 1e-6); each is the
         fraction of the absorbed power lambda*Q assigned to that lobe.
         z_top and z_bot are POSITIVE depths below the laser plane in
         meters (matches the rosenthal-keyhole `d` convention). Typical
         values: z_top = 0 (cap sits at the laser plane), z_bot ~
         0.5–1.5 * sz_bot.

   Common parameters (both forms):
     Q [W]            : total laser power
     lambda [-]       : absorption efficiency (absorbed power = lambda * Q)
     k [W/(m K)]      : thermal conductivity
     alpha [m^2/s]    : thermal diffusivity = k/(rho cp)
     T0 [K]           : ambient / preheat temperature
     cp [J/(kg K)]    : specific heat capacity (used to recover rho)
------------------------------------------------------------------------- */

void MoserGreenTemperatureSource::setup_temperature_source(const std::vector<std::string> &args)
{
  auto parse = [&](size_t i) -> double {
    try { return std::stod(args[i]); }
    catch (const std::exception &) {
      error->all(FLERR,"setup_temperature_source moser: invalid numeric argument");
      return 0.0;
    }
  };

  // Decide mode by looking at the first token.
  size_t first_numeric = 0;
  if (!args.empty() && args[0] == "standard") {
    mode = Mode::STANDARD;
    first_numeric = 1;
  }
  else if (!args.empty() && args[0] == "keyhole") {
    mode = Mode::KEYHOLE;
    first_numeric = 1;
  }
  else {
    mode = Mode::STANDARD;
    first_numeric = 0;
  }

  if (mode == Mode::STANDARD) {
    if (args.size() - first_numeric < 9) {
      error->all(FLERR,
        "setup_temperature_source moser [standard]: expected <Q> <lambda> <k> <alpha> <T0> <cp> <sx> <sy> <sz>");
    }
    n_lobes_ = 1;

    Q          = parse(first_numeric + 0);
    lambda     = parse(first_numeric + 1);
    k_th       = parse(first_numeric + 2);
    alpha      = parse(first_numeric + 3);
    T0_default = parse(first_numeric + 4);
    cp         = parse(first_numeric + 5);
    lobes_[0].sx = parse(first_numeric + 6);
    lobes_[0].sy = parse(first_numeric + 7);
    lobes_[0].sz = parse(first_numeric + 8);
    lobes_[0].z_offset = 0.0;
    lobes_[0].power_fraction = 1.0;
  }
  else {
    // KEYHOLE: 6 common floats + 5 per lobe * 2 lobes = 16 floats
    if (args.size() - first_numeric < 16) {
      error->all(FLERR,
        "setup_temperature_source moser keyhole: expected <Q> <lambda> <k> <alpha> <T0> <cp> "
        "<f_top> <sx_top> <sy_top> <sz_top> <z_top> "
        "<f_bot> <sx_bot> <sy_bot> <sz_bot> <z_bot>");
    }
    n_lobes_ = 2;

    Q          = parse(first_numeric + 0);
    lambda     = parse(first_numeric + 1);
    k_th       = parse(first_numeric + 2);
    alpha      = parse(first_numeric + 3);
    T0_default = parse(first_numeric + 4);
    cp         = parse(first_numeric + 5);

    lobes_[0].power_fraction = parse(first_numeric + 6);
    lobes_[0].sx             = parse(first_numeric + 7);
    lobes_[0].sy             = parse(first_numeric + 8);
    lobes_[0].sz             = parse(first_numeric + 9);
    lobes_[0].z_offset       = parse(first_numeric + 10);

    lobes_[1].power_fraction = parse(first_numeric + 11);
    lobes_[1].sx             = parse(first_numeric + 12);
    lobes_[1].sy             = parse(first_numeric + 13);
    lobes_[1].sz             = parse(first_numeric + 14);
    lobes_[1].z_offset       = parse(first_numeric + 15);
  }

  // Common-parameter validation
  if (Q     <= 0.0) error->all(FLERR,"moser: Q must be > 0");
  if (lambda <= 0.0) error->all(FLERR,"moser: lambda must be > 0");
  if (k_th  <= 0.0) error->all(FLERR,"moser: k must be > 0");
  if (alpha <= 0.0) error->all(FLERR,"moser: alpha must be > 0");
  if (cp    <= 0.0) error->all(FLERR,"moser: cp must be > 0");

  // Per-lobe validation
  for (int i = 0; i < n_lobes_; ++i) {
    if (lobes_[i].sx <= 0.0) error->all(FLERR,"moser: sx must be > 0");
    if (lobes_[i].sy <= 0.0) error->all(FLERR,"moser: sy must be > 0");
    if (lobes_[i].sz <= 0.0) error->all(FLERR,"moser: sz must be > 0");
    if (lobes_[i].z_offset < 0.0)
      error->all(FLERR,"moser keyhole: z_top and z_bot must be >= 0 (positive depths below laser plane)");
    if (lobes_[i].power_fraction < 0.0)
      error->all(FLERR,"moser keyhole: f_top and f_bot must be >= 0");
  }

  if (mode == Mode::KEYHOLE) {
    const double fsum = lobes_[0].power_fraction + lobes_[1].power_fraction;
    if (std::fabs(fsum - 1.0) > 1.0e-6)
      error->all(FLERR,"moser keyhole: f_top + f_bot must equal 1 (within 1e-6)");
  }

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

   In KEYHOLE mode the waypoint geometry is identical for both lobes;
   only the per-lobe absorbed power (lambda*Q*f_lobe) and the LaserScan
   zl (= laser_plane_z - z_offset) differ. Each lobe gets its own
   GREENAM_Scan + GREENAM_Integ + (optional) FluctuationTable.
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
  scan_laser_plane_z = laser_plane_z;
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

  // Build the path waypoints once; only the per-lobe `ps` array varies.
  sierra::greenam::VectorWrapper<double> ts("Time",  nwp);
  sierra::greenam::VectorWrapper<double> xs("XLaser", nwp);
  sierra::greenam::VectorWrapper<double> ys("YLaser", nwp);

  const double abs_power = lambda * Q;
  double t = start_time;
  int wp = 0;

  for (int r = 0; r < repeats; ++r) {
    ts(wp) = t;
    xs(wp) = x0;
    ys(wp) = y0;
    ++wp;

    t += dt_scan;
    ts(wp) = t;
    xs(wp) = x1;
    ys(wp) = y1;
    ++wp;

    if (r < repeats - 1) {
      t += eps_t;
    }
  }

  // The time at which the last active emission ends. The fluctuation
  // table needs to cover [start_time, t_end_active]; everything past
  // this is in the 0-power "stop" segment that the integrator culls.
  const double t_end_active = t;

  // Final velocity-calc sentinel waypoint, far in the future.
  ts(wp) = t + 1.0e6;
  xs(wp) = x1;
  ys(wp) = y1;
  ++wp;

  if (wp != nwp) {
    error->all(FLERR,"moser build_scan: internal waypoint count mismatch");
  }

  ts.commit();
  xs.commit();
  ys.commit();

  sierra::greenam::ThermalProperties<double> th{k_th, rho, cp};

  // Build a per-lobe scan + integrator + (optional) fluctuation table.
  for (int li = 0; li < n_lobes_; ++li) {
    Lobe &L_ref = lobes_[li];

    // Per-lobe waypoint power array (path geometry is shared, but power
    // is scaled by f_lobe so each LaserScan is self-describing for
    // total-absorbed-power audits).
    sierra::greenam::VectorWrapper<double> ps("Power", nwp);
    int wp2 = 0;
    for (int r = 0; r < repeats; ++r) {
      ps(wp2++) = abs_power * L_ref.power_fraction;   // active scan
      ps(wp2++) = 0.0;                                 // transit/stop
    }
    ps(wp2++) = 0.0;                                   // sentinel
    ps.commit();

    sierra::greenam::EllipsoidProperties<double> ep{L_ref.sx, L_ref.sy, L_ref.sz};
    const double zl_eff = laser_plane_z - L_ref.z_offset;

    try {
      L_ref.scan = std::make_shared<GREENAM_Scan>(th, ep, ts, xs, ys, ps, zl_eff);
    } catch (const std::runtime_error &e) {
      std::string msg = "moser build_scan: GREENAM LaserScan construction failed: ";
      msg += e.what();
      error->all(FLERR, msg.c_str());
    }
    L_ref.integrator = std::make_shared<GREENAM_Integ>(*L_ref.scan);

    if (L_ref.psd_spec_set) {
      populate_fluctuation_table(L_ref, start_time, t_end_active);
      L_ref.scan->set_fluctuation_table(&L_ref.fluct_table);
    }
  }

  scan_built = true;

  if (domain->me == 0) {
    if (mode == Mode::KEYHOLE) {
      std::cout << "moser keyhole: built " << n_lobes_ << " lobes, "
                << repeats << " repeats, "
                << "L=" << L*1e3 << " mm, dt_scan=" << dt_scan*1e3 << " ms, "
                << "absorbed P_total=" << abs_power << " W ("
                << "P_top=" << abs_power * lobes_[0].power_fraction
                << ", P_bot=" << abs_power * lobes_[1].power_fraction
                << ")" << std::endl;
    } else {
      std::cout << "moser: built scan with " << repeats << " repeats, "
                << "L=" << L*1e3 << " mm, dt_scan=" << dt_scan*1e3 << " ms, "
                << "absorbed P=" << abs_power << " W" << std::endl;
    }
    for (int li = 0; li < n_lobes_; ++li) {
      if (lobes_[li].psd_spec_set) {
        const char *tag = (mode == Mode::KEYHOLE)
                          ? (li == LOBE_TOP ? "top" : "bot")
                          : "";
        std::cout << "moser: attached fluctuation table for lobe " << tag
                  << " with " << lobes_[li].fluct_table.size() << " samples, "
                  << "t in [" << start_time << ", " << t_end_active
                  << "] s" << std::endl;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   Per-site temperature evaluation. Returns absolute temperature [K] at
   the world-space site location and absolute simulation time. Sites
   above the laser plane (z > scan_laser_plane_z) are vacuum and stay
   at T0. The cutoff is the actual free surface, NOT any individual
   lobe's effective zl: the bottom keyhole lobe is below the surface but
   the surface itself is unchanged.
------------------------------------------------------------------------- */

double MoserGreenTemperatureSource::get_temperature_at_xyz_and_time(double x, double y,
                                                                    double z, double time)
{
  check_initialization();
  if (!scan_built) return T0_default;

  if (z > scan_laser_plane_z) return T0_default;

  const double t_local = time - scan_t_origin;
  if (t_local <= 0.0) return T0_default;

  double rise = 0.0;
  for (int li = 0; li < n_lobes_; ++li) {
    if (!lobes_[li].integrator) continue;
    rise += lobes_[li].integrator->integrate_point_adaptive(x, y, z, t_local, char_length);
  }
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
  for (int li = 0; li < 2; ++li) {
    lobes_[li].integrator.reset();
    lobes_[li].scan.reset();
    lobes_[li].fluct_table.clear();
    psd_reset_filter_state(lobes_[li]);
  }
  scan_built = false;
  source_initialized = false;
}

/* ---------------------------------------------------------------------- */

bool MoserGreenTemperatureSource::has_fluctuations() const
{
  for (int li = 0; li < n_lobes_; ++li) {
    if (lobes_[li].psd_spec_set) return true;
  }
  return false;
}

/* ---------------------------------------------------------------------- */

void MoserGreenTemperatureSource::print_source_info() const
{
  std::cout << "Moser/Green temperature source (unsteady ellipsoid integral)\n"
            << "  mode       = " << (mode == Mode::KEYHOLE ? "keyhole (double-ellipsoid)" : "standard")
            << "\n"
            << "  Q          = " << Q          << " W\n"
            << "  lambda     = " << lambda     << "\n"
            << "  P_absorbed = " << lambda * Q << " W (total)\n"
            << "  k          = " << k_th       << " W/(m K)\n"
            << "  alpha      = " << alpha      << " m^2/s\n"
            << "  cp         = " << cp         << " J/(kg K)\n"
            << "  rho        = " << rho        << " kg/m^3 (derived)\n"
            << "  T0         = " << T0_default << " K\n"
            << "  char_len   = " << char_length << "\n";
  for (int li = 0; li < n_lobes_; ++li) {
    print_lobe_info(li);
  }
  if (scan_built) {
    std::cout << "  scan: (" << scan_x0 << "," << scan_y0 << ") -> ("
              << scan_x1 << "," << scan_y1 << ") at z=" << scan_laser_plane_z
              << ", v=" << scan_speed << " m/s, repeats=" << scan_repeats
              << ", t_origin=" << scan_t_origin << " s\n";
  }
  std::cout.flush();
}

void MoserGreenTemperatureSource::print_lobe_info(int li) const
{
  const Lobe &L_ref = lobes_[li];
  const char *tag;
  if (mode == Mode::KEYHOLE) tag = (li == LOBE_TOP) ? "top" : "bot";
  else                       tag = "single";
  std::cout << "  lobe[" << tag << "]: f=" << L_ref.power_fraction
            << " sx,sy,sz=" << L_ref.sx << "," << L_ref.sy << "," << L_ref.sz << " m"
            << " z_offset=" << L_ref.z_offset << " m"
            << " (P_abs=" << lambda * Q * L_ref.power_fraction << " W)\n";
  if (L_ref.psd_spec_set) {
    const char *sname = "white";
    if (L_ref.psd_spec.shape == PsdShape::LORENTZIAN)       sname = "lorentzian";
    else if (L_ref.psd_spec.shape == PsdShape::PINK)        sname = "pink";
    else if (L_ref.psd_spec.shape == PsdShape::NARROW_BAND) sname = "narrow_band";
    std::cout << "    fluctuations: psd " << sname
              << " sigma_W=" << L_ref.psd_spec.sigma_W
              << " sigma_D=" << L_ref.psd_spec.sigma_D
              << " sigma_P=" << L_ref.psd_spec.sigma_P
              << " rho="     << L_ref.psd_spec.rho
              << " seed="    << L_ref.psd_spec.seed
              << " dt_psd="  << L_ref.psd_spec.dt_psd << " s";
    if (L_ref.psd_spec.shape == PsdShape::LORENTZIAN)
      std::cout << " tau=" << L_ref.psd_spec.tau;
    if (L_ref.psd_spec.shape == PsdShape::NARROW_BAND)
      std::cout << " f0=" << L_ref.psd_spec.f0 << " df=" << L_ref.psd_spec.df;
    std::cout << "\n";
  }
}

/* ----------------------------------------------------------------------
   PSD spec stash + filter helpers + table population.

   Time-domain recursive filters with trivariate Gaussian innovations.
   Identical chain on every MPI rank for the same seed (no MPI_Bcast
   needed). The W↔D channels share a Pearson correlation rho; the P
   channel is independent. In KEYHOLE mode each lobe carries its own
   independent stream so cap and depth lobes can pulse out of phase.
------------------------------------------------------------------------- */

void MoserGreenTemperatureSource::set_psd_spec(int idx, const PsdSpec &spec)
{
  if (idx < 0 || idx >= n_lobes_)
    error->all(FLERR,
      "laser_fluctuations psd: lobe index out of range "
      "(STANDARD mode supports only top; KEYHOLE supports top and bot)");
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

  lobes_[idx].psd_spec = spec;
  lobes_[idx].psd_spec_set = true;
}

void MoserGreenTemperatureSource::psd_reset_filter_state(Lobe &L)
{
  L.ar_state_W = 0.0;
  L.ar_state_D = 0.0;
  L.ar_state_P = 0.0;
  L.voss_W.fill(0.0);
  L.voss_D.fill(0.0);
  L.voss_P.fill(0.0);
  L.voss_step = 0;
  L.osc_x_W = 0.0; L.osc_v_W = 0.0;
  L.osc_x_D = 0.0; L.osc_v_D = 0.0;
  L.osc_x_P = 0.0; L.osc_v_P = 0.0;
}

void MoserGreenTemperatureSource::psd_warmup(Lobe &L, int n_steps)
{
  double dummy_W = 0.0, dummy_D = 0.0, dummy_P = 0.0;
  for (int i = 0; i < n_steps; ++i) {
    psd_generate_next_sample(L, dummy_W, dummy_D, dummy_P);
  }
}

void MoserGreenTemperatureSource::psd_draw_trivariate(Lobe &L,
                                                      double &eps_W,
                                                      double &eps_D,
                                                      double &eps_P)
{
  // Three independent unit normals.
  const double z1 = L.norm(L.rng);
  const double z2 = L.norm(L.rng);
  const double z3 = L.norm(L.rng);
  const double rho = L.psd_spec.rho;
  // W and D share Pearson correlation rho; P is independent.
  eps_W = z1;
  eps_D = rho * z1 + std::sqrt(std::max(0.0, 1.0 - rho * rho)) * z2;
  eps_P = z3;
}

void MoserGreenTemperatureSource::psd_generate_next_sample(Lobe &L,
                                                           double &dW,
                                                           double &dD,
                                                           double &dP)
{
  double eps_W, eps_D, eps_P;
  psd_draw_trivariate(L, eps_W, eps_D, eps_P);

  switch (L.psd_spec.shape) {

  case PsdShape::WHITE: {
    dW = L.psd_spec.sigma_W * eps_W;
    dD = L.psd_spec.sigma_D * eps_D;
    dP = L.psd_spec.sigma_P * eps_P;
    return;
  }

  case PsdShape::LORENTZIAN: {
    // AR(1) with alpha = exp(-dt/tau): produces exact Lorentzian
    // (Ornstein-Uhlenbeck) autocorrelation in steady state. dt is the
    // emission-time sample spacing (seconds), tau the correlation time.
    const double alpha_ar = std::exp(-L.psd_spec.dt_psd / L.psd_spec.tau);
    const double drive_scale = std::sqrt(1.0 - alpha_ar * alpha_ar);
    const double driveW = L.psd_spec.sigma_W * drive_scale;
    const double driveD = L.psd_spec.sigma_D * drive_scale;
    const double driveP = L.psd_spec.sigma_P * drive_scale;
    L.ar_state_W = alpha_ar * L.ar_state_W + driveW * eps_W;
    L.ar_state_D = alpha_ar * L.ar_state_D + driveD * eps_D;
    L.ar_state_P = alpha_ar * L.ar_state_P + driveP * eps_P;
    dW = L.ar_state_W;
    dD = L.ar_state_D;
    dP = L.ar_state_P;
    return;
  }

  case PsdShape::PINK: {
    // Voss-McCartney 1/f: at step n, update only the level k =
    // trailing_zeros(n). Sum of K independent levels approximates a
    // 1/f spectrum across ~K decades.
    ++L.voss_step;
    int k = 0;
    std::uint64_t n = L.voss_step;
    while ((n & 1ULL) == 0ULL && k < VOSS_K - 1) { n >>= 1; ++k; }
    const double scale = 1.0 / std::sqrt(static_cast<double>(VOSS_K));
    L.voss_W[k] = (L.psd_spec.sigma_W * scale) * eps_W;
    L.voss_D[k] = (L.psd_spec.sigma_D * scale) * eps_D;
    L.voss_P[k] = (L.psd_spec.sigma_P * scale) * eps_P;
    double sW = 0.0, sD = 0.0, sP = 0.0;
    for (int i = 0; i < VOSS_K; ++i) { sW += L.voss_W[i]; sD += L.voss_D[i]; sP += L.voss_P[i]; }
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
    const double omega0 = 2.0 * MY_PI * L.psd_spec.f0;
    double zeta = MY_PI * L.psd_spec.df / omega0;
    if (zeta <= 0.0)  zeta = 1.0e-6;
    if (zeta >= 1.0)  zeta = 0.999;
    int n_sub = static_cast<int>(std::ceil(omega0 * L.psd_spec.dt_psd / 0.1));
    if (n_sub < 1) n_sub = 1;
    const double dts = L.psd_spec.dt_psd / n_sub;
    const double var_pref = 4.0 * zeta * omega0 * omega0 * omega0 * dts;
    const double driveW_imp = std::sqrt(var_pref * L.psd_spec.sigma_W * L.psd_spec.sigma_W);
    const double driveD_imp = std::sqrt(var_pref * L.psd_spec.sigma_D * L.psd_spec.sigma_D);
    const double driveP_imp = std::sqrt(var_pref * L.psd_spec.sigma_P * L.psd_spec.sigma_P);
    for (int kstep = 0; kstep < n_sub; ++kstep) {
      double e_W, e_D, e_P;
      if (kstep == 0) { e_W = eps_W; e_D = eps_D; e_P = eps_P; }
      else            { psd_draw_trivariate(L, e_W, e_D, e_P); }
      // Semi-implicit Euler:
      L.osc_v_W += dts * (-2.0 * zeta * omega0 * L.osc_v_W
                          - omega0 * omega0 * L.osc_x_W) + driveW_imp * e_W;
      L.osc_x_W += dts * L.osc_v_W;
      L.osc_v_D += dts * (-2.0 * zeta * omega0 * L.osc_v_D
                          - omega0 * omega0 * L.osc_x_D) + driveD_imp * e_D;
      L.osc_x_D += dts * L.osc_v_D;
      L.osc_v_P += dts * (-2.0 * zeta * omega0 * L.osc_v_P
                          - omega0 * omega0 * L.osc_x_P) + driveP_imp * e_P;
      L.osc_x_P += dts * L.osc_v_P;
    }
    dW = L.osc_x_W;
    dD = L.osc_x_D;
    dP = L.osc_x_P;
    return;
  }
  }
  // Unreachable
  dW = 0.0; dD = 0.0; dP = 0.0;
}

void MoserGreenTemperatureSource::populate_fluctuation_table(Lobe &L,
                                                              double t_start,
                                                              double t_end)
{
  if (!L.psd_spec_set)
    error->all(FLERR,"moser populate_fluctuation_table: psd_spec not set");
  if (t_end <= t_start)
    error->all(FLERR,"moser populate_fluctuation_table: empty time interval");

  // Reset the chain for reproducibility (multiple build_scan() calls
  // with the same seed produce the same table).
  L.rng.seed(L.psd_spec.seed);
  L.norm = std::normal_distribution<double>(0.0, 1.0);
  psd_reset_filter_state(L);
  // Warmup so the filter starts in steady state. Pink needs ~K * 2^K =
  // 384 samples; lorentzian needs ~5*tau/dt; narrow_band needs
  // ~1/(2 zeta omega0)/dt. 1000 covers all of these for typical configs.
  psd_warmup(L, 1000);

  const double dt = L.psd_spec.dt_psd;
  const std::size_t n_samples =
    static_cast<std::size_t>(std::ceil((t_end - t_start) / dt)) + 1;

  L.fluct_table.t_grid.clear();
  L.fluct_table.factor_W.clear();
  L.fluct_table.factor_D.clear();
  L.fluct_table.factor_P.clear();
  L.fluct_table.t_grid.reserve(n_samples);
  L.fluct_table.factor_W.reserve(n_samples);
  L.fluct_table.factor_D.reserve(n_samples);
  L.fluct_table.factor_P.reserve(n_samples);

  bool warned_linearization = false;
  for (std::size_t i = 0; i < n_samples; ++i) {
    const double t_i = t_start + i * dt;
    double dW = 0.0, dD = 0.0, dP = 0.0;
    psd_generate_next_sample(L, dW, dD, dP);

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

    L.fluct_table.t_grid.push_back(t_i);
    L.fluct_table.factor_W.push_back(fW);
    L.fluct_table.factor_D.push_back(fD);
    L.fluct_table.factor_P.push_back(fP);
  }
}
