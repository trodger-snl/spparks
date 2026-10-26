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
#include "material_card.h"
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
    size_t end = 0;
    double v = 0.0;
    try { v = std::stod(args[i], &end); } catch (const std::exception &) {}
    if (end == 0 || end != args[i].size()) {
      std::string msg = "setup_temperature_source conduction: invalid numeric argument " + args[i];
      error->all(FLERR,msg.c_str());
    }
    return v;
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
  std::string plate_card;
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
    else if (a == "plate_material") {
      if (++i >= args.size()) error->all(FLERR,"setup_temperature_source conduction: plate_material needs a card file");
      plate_card = args[i];
    }
    else if (a == "plate_growth") cfg.plate_growth = num(++i);
    else if (a == "layer_thickness") cfg.layer_thickness = num(++i);
    else if (a == "packing") cfg.packing = num(++i);
    else if (a == "sts") cfg.sts_smax = (int) num(++i);
    else if (a == "ded") {
      cfg.ded = true;
      while (i + 1 < args.size()) {
        const std::string &b = args[i+1];
        if (b == "radius") { i++; cfg.ded_radius = num(++i); }
        else if (b == "T_birth") { i++; cfg.T_birth = num(++i); }
        else if (b == "depth") { i++; cfg.ded_depth = num(++i); }
        else break;
      }
    }
    else if (a == "far_bc") {
      if (++i >= args.size()) error->all(FLERR,"setup_temperature_source conduction: far_bc needs a value");
      if (args[i] == "fixed") cfg.far_bc = FAR_FIXED;
      else if (args[i] == "adiabatic") cfg.far_bc = FAR_ADIABATIC;
      else error->all(FLERR,"setup_temperature_source conduction: far_bc is fixed or adiabatic");
    }
    else {
      // positional (Moser) values are numbers; any other word is a typo
      size_t end = 0;
      try { std::stod(a, &end); } catch (const std::exception &) {}
      if (end == 0 || end != a.size()) {
        std::string msg = "setup_temperature_source conduction: unknown keyword " + a;
        error->all(FLERR,msg.c_str());
      }
      moser_args.push_back(a);
    }
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
  if (cfg.packing <= 0.0 || cfg.packing > 1.0)
    error->all(FLERR,"setup_temperature_source conduction: packing must be in (0, 1]");
  if (cfg.ded) {
    if (cfg.layer_thickness > 0.0 || cfg.packing < 1.0)
      error->all(FLERR,"setup_temperature_source conduction: ded has no powder bed "
                 "(layer_thickness must be 0 and packing 1)");
    if (cfg.ded_radius < 0.0) cfg.ded_radius = 2.0 * sigma_x();
    if (cfg.T_birth < 0.0) cfg.T_birth = T0_default;
    if (cfg.ded_radius == 0.0 || cfg.ded_depth == 0.0)
      error->all(FLERR,"setup_temperature_source conduction: ded radius and depth must be > 0");
  }
  if (!cfg.part_region.empty() &&
      domain->find_region((char *) cfg.part_region.c_str()) < 0)
    error->all(FLERR,"setup_temperature_source conduction: part region ID does not exist");
  if (!plate_card.empty()) {
    if (!cfg.plate_top_set)
      error->all(FLERR,"setup_temperature_source conduction: plate_material needs plate_top (the plate surface)");
    if (cfg.plate_growth < 0.0)
      error->all(FLERR,"setup_temperature_source conduction: plate_growth must be >= 0");
    auto &pm = cfg.plate_mat;
    try {
      const MaterialCard c = MaterialCard::load(plate_card, "", world);
      auto need = [&](std::optional<double> v, const char *key) {
        if (!v) throw std::runtime_error("plate_material " + plate_card + ": " + key + " is not in the card");
        return *v;
      };
      auto tab = [&](const MaterialCard::Table &t, std::optional<double> lin, const char *key,
                     std::vector<double> &T, std::vector<double> &v) {
        if (!t.T.empty()) { T = t.T; v = t.v; }
        else { T = {0.0}; v = {need(lin, key)}; }
      };
      tab(c.k, c.k_lin, "thermal.conductivity", pm.k_T, pm.k_v);
      tab(c.cp, c.cp_lin, "thermal.heat_capacity", pm.cp_T, pm.cp_v);
      pm.rho = need(c.density, "thermal.density");
      pm.latent = need(c.latent, "thermal.latent_fusion");
      pm.ts = need(c.solidus, "thermal.solidus");
      pm.tl = need(c.liquidus, "thermal.liquidus");
      pm.name = c.material;
    } catch (const std::exception &e) {
      error->all(FLERR, e.what());
    }
    if (pm.tl < pm.ts)
      error->all(FLERR,"setup_temperature_source conduction: plate_material liquidus is below its solidus");
    pm.on = true;
  }
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
  const auto &pm = cfg.plate_mat;
  if (pm.on)
    std::cout << "conduction: plate " << pm.name << " below z=" << cfg.plate_top
              << " m (i2 = 6 up to " << plate_surface() << " m): rho=" << pm.rho << ", "
              << pm.k_T.size() << "-point k table, " << pm.cp_T.size() << "-point cp table, latent="
              << pm.latent << " J/kg, solidus " << pm.ts << " K, liquidus " << pm.tl << " K"
              << std::endl;
  if (cfg.ded)
    std::cout << "conduction: ded, birth radius " << cfg.ded_radius << " m, T_birth "
              << cfg.T_birth << " K, depth "
              << (cfg.ded_depth > 0.0 ? std::to_string(cfg.ded_depth) + " m" : "layer step")
              << std::endl;
}

double ConductionTemperatureSource::plate_surface() const
{
  return cfg.plate_top + cfg.plate_growth;
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
                                                 std::vector<double> &H, bool plate) const
{
  const auto &pm = cfg.plate_mat;
  const double rho_m = plate ? pm.rho : rho;
  const double latent = plate ? pm.latent : cfg.latent;
  auto cp_at = [&](double T) {
    const auto &Tt = plate ? pm.cp_T : cfg.cp_T, &v = plate ? pm.cp_v : cfg.cp_v;
    if (T <= Tt.front()) return v.front();
    if (T >= Tt.back()) return v.back();
    const size_t j = std::upper_bound(Tt.begin(), Tt.end(), T) - Tt.begin();
    const double w = (T - Tt[j-1]) / (Tt[j] - Tt[j-1]);
    return v[j-1] + w * (v[j] - v[j-1]);
  };
  auto melt = [&](double T) {
    if (latent <= 0.0) return 0.0;
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
    sens += 0.5 * (cp_at(Ta) + cp_at(Tb)) * dT * rho_m;
    H[i] = sens;
  }
  for (int i = 0; i < n; i++) H[i] += rho_m * latent * melt(T_lo + i * dT);
}
