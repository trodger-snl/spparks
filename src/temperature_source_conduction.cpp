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

#include "temperature_source_conduction.h"
#include "domain.h"
#include "error.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

using namespace SPPARKS_NS;

ConductionTemperatureSource::ConductionTemperatureSource(SPPARKS *spk) :
  MoserTemperatureSource(spk) {}

/* ----------------------------------------------------------------------
   Keywords are stripped; the remaining tokens are Moser's STANDARD form.
------------------------------------------------------------------------- */

void ConductionTemperatureSource::setup_temperature_source(
    const std::vector<std::string> &args)
{
  auto num = [&](size_t i) -> double {
    if (i >= args.size())
      error->all(FLERR,"setup_temperature_source conduction: missing keyword value");
    try { return std::stod(args[i]); }
    catch (const std::exception &) {
      error->all(FLERR,"setup_temperature_source conduction: invalid numeric argument");
      return 0.0;
    }
  };
  auto table = [&](size_t &i, std::vector<double> &T, std::vector<double> &v) {
    const int n = (int) num(++i);
    if (n < 1) error->all(FLERR,"setup_temperature_source conduction: table needs n >= 1");
    T.resize(n); v.resize(n);
    for (int j = 0; j < n; j++) { T[j] = num(++i); v[j] = num(++i); }
    for (int j = 1; j < n; j++)
      if (T[j] <= T[j-1])
        error->all(FLERR,"setup_temperature_source conduction: table temperatures must increase");
  };

  std::vector<std::string> moser_args;
  double k_powder = -1.0;
  for (size_t i = 0; i < args.size(); i++) {
    const std::string &a = args[i];
    if (a == "keyhole")
      error->all(FLERR,"setup_temperature_source conduction: keyhole is not supported");
    else if (a == "standard") continue;
    else if (a == "k_table") table(i, cfg.k_T, cfg.k_v);
    else if (a == "cp_table") table(i, cfg.cp_T, cfg.cp_v);
    else if (a == "latent") cfg.latent = num(++i);
    else if (a == "k_powder") k_powder = num(++i);
    else if (a == "convection") { cfg.h = num(++i); cfg.T_inf = num(++i); }
    else if (a == "emissivity") cfg.emissivity = num(++i);
    else if (a == "margin") for (int j = 0; j < 4; j++) cfg.margin[j] = num(++i);
    else if (a == "plate") cfg.plate = num(++i);
    else if (a == "plate_top") { cfg.plate_top = num(++i); cfg.plate_top_set = true; }
    else if (a == "stretch") cfg.stretch = num(++i);
    else if (a == "max_cell") cfg.max_cell = num(++i);
    else if (a == "part") {
      if (++i >= args.size()) error->all(FLERR,"setup_temperature_source conduction: part needs a region ID");
      cfg.part_region = args[i];
    }
    else if (a == "layer_thickness") cfg.layer_thickness = num(++i);
    else if (a == "sts") cfg.sts_smax = (int) num(++i);
    else if (a == "far_bc") {
      if (++i >= args.size()) error->all(FLERR,"setup_temperature_source conduction: far_bc needs a value");
      if (args[i] == "fixed") cfg.far_bc = FAR_FIXED;
      else if (args[i] == "adiabatic") cfg.far_bc = FAR_ADIABATIC;
      else error->all(FLERR,"setup_temperature_source conduction: far_bc is fixed or adiabatic");
    }
    else moser_args.push_back(a);
  }

  MoserTemperatureSource::setup_temperature_source(moser_args);

  if (cfg.k_T.empty()) { cfg.k_T = {0.0}; cfg.k_v = {k_th}; }
  if (cfg.cp_T.empty()) { cfg.cp_T = {0.0}; cfg.cp_v = {cp}; }
  cfg.k_powder = (k_powder >= 0.0) ? k_powder : 0.01 * k_th;
  if (cfg.stretch < 1.0)
    error->all(FLERR,"setup_temperature_source conduction: stretch must be >= 1");
  for (double m : cfg.margin)
    if (m < 0.0) error->all(FLERR,"setup_temperature_source conduction: margin must be >= 0");
  if (cfg.sts_smax < 0 || cfg.sts_smax == 1)
    error->all(FLERR,"setup_temperature_source conduction: sts is 0 (off) or >= 2 stages");
  if (cfg.plate < 0.0 || cfg.layer_thickness < 0.0 || cfg.latent < 0.0)
    error->all(FLERR,"setup_temperature_source conduction: plate, layer_thickness, latent must be >= 0");
  if (!cfg.part_region.empty() &&
      domain->find_region((char *) cfg.part_region.c_str()) < 0)
    error->all(FLERR,"setup_temperature_source conduction: part region ID does not exist");
  configured = true;
  print_source_info();
}

double ConductionTemperatureSource::get_temperature_at_xyz_and_time(
    double, double, double, double)
{
  error->all(FLERR,"conduction temperature source has no point query: it runs "
             "only in additive/texture/kk, without temperature_smooth");
  return 0.0;
}

void ConductionTemperatureSource::print_source_info() const
{
  // Moser's setup prints through this before the defaults below are set.
  if (domain->me != 0 || !configured) return;
  std::cout << "conduction: absorbed P=" << lambda * Q << " W, rho=" << rho
            << ", k_powder=" << cfg.k_powder << ", " << cfg.k_T.size()
            << "-point k table, " << cfg.cp_T.size() << "-point cp table, latent="
            << cfg.latent << " J/kg, h=" << cfg.h << ", emissivity="
            << cfg.emissivity << ", stretch=" << cfg.stretch << std::endl;
}

void ConductionTemperatureSource::reset_history_and_start(double t_start,
                                                          double new_ambient)
{
  MoserTemperatureSource::reset_history_and_start(t_start, new_ambient);
  request_reset(new_ambient);
}

void ConductionTemperatureSource::request_reset(double T_reset)
{
  reset_pending = true;
  reset_T = T_reset;
}

bool ConductionTemperatureSource::take_reset(double &T_reset)
{
  if (!reset_pending) return false;
  reset_pending = false;
  T_reset = reset_T;
  return true;
}

/* ----------------------------------------------------------------------
   Segments are disjoint in time. The returned fraction is the power-weighted
   on-time of [t0, t1]; the position is the midpoint of the overlap with the
   segment that overlaps longest (one position per sub-step).
   Segments are sorted, so both queries start at the first segment ending
   after the query time (binary search): a G-code layer can hold many
   thousands, and laser_window runs every sub-step.
------------------------------------------------------------------------- */

static std::vector<ScanSeg>::const_iterator first_ending_after(
  const std::vector<ScanSeg> &segs, double t)
{
  return std::upper_bound(segs.begin(), segs.end(), t,
                          [](double v, const ScanSeg &g) { return v < g.t1; });
}

double ConductionTemperatureSource::laser_window(double t0, double t1,
                                                 double &x, double &y) const
{
  x = y = 0.0;
  if (!scan_built || t1 <= t0) return 0.0;
  double on = 0.0, best = 0.0;
  for (auto it = first_ending_after(segs_, t0); it != segs_.end() && it->t0 < t1; ++it) {
    const Seg &g = *it;
    const double a = std::max(t0, g.t0), b = std::min(t1, g.t1);
    if (b <= a || g.p <= 0.0) continue;
    on += (b - a) * g.p;
    if (b - a > best) {
      best = b - a;
      const double u = (0.5 * (a + b) - g.t0) / (g.t1 - g.t0);
      x = g.x0 + u * (g.x1 - g.x0);
      y = g.y0 + u * (g.y1 - g.y0);
    }
  }
  return on / (t1 - t0);
}

double ConductionTemperatureSource::next_laser_on(double t) const
{
  double next = std::numeric_limits<double>::max();
  if (!scan_built) return next;
  for (auto it = first_ending_after(segs_, t); it != segs_.end(); ++it) {
    if (it->p <= 0.0) continue;
    return t >= it->t0 ? t : it->t0;   // first emitting segment not yet over
  }
  return next;
}

/* ----------------------------------------------------------------------
   H(T) = int_0^T rho cp dT' + rho L f(T), f linear from 0 at ts to 1 at tl.
   1 K spacing to 6000 K; the solver extrapolates beyond with the end slopes.
------------------------------------------------------------------------- */

void ConductionTemperatureSource::enthalpy_table(double ts, double tl,
                                                 double &T_lo, double &dT,
                                                 std::vector<double> &H) const
{
  auto cp_at = [&](double T) {
    const auto &Tt = cfg.cp_T, &v = cfg.cp_v;
    if (T <= Tt.front()) return v.front();
    if (T >= Tt.back()) return v.back();
    const size_t j = std::upper_bound(Tt.begin(), Tt.end(), T) - Tt.begin();
    const double w = (T - Tt[j-1]) / (Tt[j] - Tt[j-1]);
    return v[j-1] + w * (v[j] - v[j-1]);
  };
  auto melt = [&](double T) {
    if (cfg.latent <= 0.0) return 0.0;
    if (tl <= ts) return T >= tl ? 1.0 : 0.0;
    return std::min(1.0, std::max(0.0, (T - ts) / (tl - ts)));
  };

  T_lo = 0.0;
  dT = 1.0;
  const int n = 6001;
  H.assign(n, 0.0);
  double sens = 0.0;
  for (int i = 1; i < n; i++) {
    const double Ta = T_lo + (i-1) * dT, Tb = Ta + dT;
    sens += 0.5 * (cp_at(Ta) + cp_at(Tb)) * dT * rho;
    H[i] = sens;
  }
  for (int i = 0; i < n; i++) H[i] += rho * cfg.latent * melt(T_lo + i * dT);
}
