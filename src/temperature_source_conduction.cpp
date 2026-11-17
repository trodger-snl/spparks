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
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace SPPARKS_NS;

ConductionTemperatureSource::ConductionTemperatureSource(SPPARKS *spk) :
  MoserTemperatureSource(spk) {}

namespace {

using Material = ConductionTemperatureSource::Config::Material;

// A card's dense properties; what the card lacks is an error (no defaults).
Material card_material(const std::string &path, MPI_Comm world)
{
  const MaterialCard c = MaterialCard::load(path, "", world);
  auto need = [&](std::optional<double> v, const char *key) {
    if (!v) throw std::runtime_error(path + ": " + key + " is not in the card");
    return *v;
  };
  auto tab = [&](const MaterialCard::Table &t, std::optional<double> lin, const char *key,
                 std::vector<double> &T, std::vector<double> &v) {
    if (!t.T.empty()) { T = t.T; v = t.v; }
    else { T = {0.0}; v = {need(lin, key)}; }
  };
  Material m;
  tab(c.k, c.k_lin, "thermal.conductivity", m.k_T, m.k_v);
  tab(c.cp, c.cp_lin, "thermal.heat_capacity", m.cp_T, m.cp_v);
  m.rho = need(c.density, "thermal.density");
  m.latent = need(c.latent, "thermal.latent_fusion");
  m.ts = need(c.solidus, "thermal.solidus");
  m.tl = need(c.liquidus, "thermal.liquidus");
  m.name = c.material;
  if (m.tl < m.ts) throw std::runtime_error(path + ": liquidus is below the solidus");
  return m;
}

/* ----------------------------------------------------------------------
   flow properties cards: the liquid's surface tension, density and
   viscosity over the composition, from the [flow] sections of the cards at
   the nodes x. Between two nodes, w the mass fraction of the second:
     sigma  (1-w) a + w b, each linear in T (a mix of the pure values: no
            surface segregation; a sigma_table replaces it)
     rho    1 / ((1-w) / rho_a + w / rho_b): volumes add; kept as a line in
            T through Tr and Tr + 1000 K
     eta    (1-w) a + w b, each card's value at its own liquidus
   ponytail: eta has no T dependence here; add a T axis if it matters.
   sigma_table: rows "T x sigma ..." (tools/surface_tension.py table), T
   outer, x inner, both uniform; x is d11. Every rank reads the file.
------------------------------------------------------------------------- */

void flow_props(const std::vector<double> &xs, const std::vector<std::string> &cards,
                const std::string &sigma_file, MPI_Comm world,
                ConductionTemperatureSource::Config::Flow::Props &p)
{
  p = {};
  std::vector<MaterialCard> c;
  for (const std::string &f : cards) {
    c.push_back(MaterialCard::load(f, "", world));
    const MaterialCard &m = c.back();
    if (!m.sigma || !m.dsigma_dT || !m.rho_liq || m.visc.T.empty() || !m.liquidus)
      throw std::runtime_error(f + ": [flow] needs surface_tension, dsigma_dT, density_liquid and viscosity");
  }
  // node pair and weight at x
  auto at = [&](double x, size_t &j) {
    j = 1;
    while (j + 1 < xs.size() && xs[j] < x) j++;
    return std::min(1.0, std::max(0.0, (x - xs[j-1]) / (xs[j] - xs[j-1])));
  };
  auto rho = [&](const MaterialCard &m, double T) { return *m.rho_liq + *m.rho_liq_slope * (T - *m.rho_liq_T); };
  p.n = 65;
  p.Tr = *c[0].rho_liq_T;
  p.r0.resize(p.n); p.r1.resize(p.n); p.eta.resize(p.n);
  for (int i = 0; i < p.n; i++) {
    size_t j;
    const double w = at(xs.front() + (xs.back() - xs.front()) * i / (p.n - 1), j);
    const MaterialCard &a = c[j-1], &b = c[j];
    auto mix = [&](double T) { return 1.0 / ((1.0 - w) / rho(a, T) + w / rho(b, T)); };
    p.r0[i] = mix(p.Tr);
    p.r1[i] = (mix(p.Tr + 1000.0) - p.r0[i]) / 1000.0;
    p.eta[i] = (1.0 - w) * MaterialCard::table_at(a.visc, *a.liquidus) + w * MaterialCard::table_at(b.visc, *b.liquidus);
  }
  if (sigma_file.empty()) {
    p.nT = 2; p.nx = p.n; p.T0 = p.Tr; p.dT = 1000.0;
    p.x0 = xs.front(); p.dx = (xs.back() - xs.front()) / (p.n - 1);
    p.sig.resize(2 * p.n);
    for (int t = 0; t < 2; t++)
      for (int i = 0; i < p.n; i++) {
        size_t j;
        const double w = at(p.x0 + p.dx * i, j), T = p.T0 + p.dT * t;
        auto s = [&](const MaterialCard &m) { return *m.sigma + *m.dsigma_dT * (T - *m.sigma_T); };
        p.sig[t * p.n + i] = (1.0 - w) * s(c[j-1]) + w * s(c[j]);
      }
    return;
  }
  FILE *fp = fopen(sigma_file.c_str(), "r");
  if (!fp) throw std::runtime_error("cannot open sigma_table " + sigma_file);
  std::vector<double> T, x;
  char line[1024];
  while (fgets(line, sizeof(line), fp)) {
    double t, xv, s;
    if (line[0] == '#' || sscanf(line, "%lf %lf %lf", &t, &xv, &s) != 3) continue;
    T.push_back(t); x.push_back(xv); p.sig.push_back(s);
  }
  fclose(fp);
  const size_t nrow = T.size();
  size_t nx = 1;
  while (nx < nrow && T[nx] == T[0]) nx++;
  if (nx < 2 || nrow % nx || nrow / nx < 2)
    throw std::runtime_error(sigma_file + ": needs rows T x sigma, T outer and x inner, 2 or more of each");
  p.nx = (int) nx; p.nT = (int) (nrow / nx);
  p.T0 = T[0]; p.dT = T[nx] - T[0]; p.x0 = x[0]; p.dx = x[1] - x[0];
  for (size_t r = 0; r < nrow; r++)
    if (std::fabs(T[r] - (p.T0 + p.dT * (r / nx))) > 1e-6 * std::fabs(p.dT) ||
        std::fabs(x[r] - (p.x0 + p.dx * (r % nx))) > 1e-6 * std::fabs(p.dx) || p.dT <= 0.0 || p.dx <= 0.0)
      throw std::runtime_error(sigma_file + ": T and x must be uniform grids, T outer and x inner");
}

// (1-w) a + w b: scalars linearly, tables on the union of both T grids.
Material blend(const Material &a, const Material &b, double w)
{
  auto at = [](const std::vector<double> &T, const std::vector<double> &v, double t) {
    return MaterialCard::table_at({T, v}, t);
  };
  auto table = [&](const std::vector<double> &Ta, const std::vector<double> &va,
                   const std::vector<double> &Tb, const std::vector<double> &vb,
                   std::vector<double> &T, std::vector<double> &v) {
    T = Ta; T.insert(T.end(), Tb.begin(), Tb.end());
    std::sort(T.begin(), T.end());
    T.erase(std::unique(T.begin(), T.end()), T.end());
    v.resize(T.size());
    for (size_t i = 0; i < T.size(); i++)
      v[i] = (1.0 - w) * at(Ta, va, T[i]) + w * at(Tb, vb, T[i]);
  };
  Material m;
  table(a.k_T, a.k_v, b.k_T, b.k_v, m.k_T, m.k_v);
  table(a.cp_T, a.cp_v, b.cp_T, b.cp_v, m.cp_T, m.cp_v);
  m.rho = (1.0 - w) * a.rho + w * b.rho;
  m.latent = (1.0 - w) * a.latent + w * b.latent;
  m.ts = (1.0 - w) * a.ts + w * b.ts;
  m.tl = (1.0 - w) * a.tl + w * b.tl;
  return m;
}

}

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
  std::vector<double> comp_x;
  std::vector<std::string> comp_cards;
  int comp_bins = 21;
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
    else if (a == "evaporation") {
      const int n = (int) num(++i);
      if (n < 1 || n > 2) error->all(FLERR,"setup_temperature_source conduction: evaporation takes 1 or 2 species (Tb Lv M each)");
      cfg.evap.clear();
      for (int j = 0; j < 3 * n; j++) {
        cfg.evap.push_back(num(++i));
        if (cfg.evap.back() <= 0.0) error->all(FLERR,"setup_temperature_source conduction: evaporation Tb, Lv and M must be positive");
      }
    }
    else if (a == "evaporation_coef") cfg.evap_coef = num(++i);
    else if (a == "evaporation_ambient") {
      cfg.evap_ambient = num(++i);
      if (cfg.evap_ambient < 0.0) error->all(FLERR,"setup_temperature_source conduction: evaporation_ambient must be >= 0");
    }
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
    else if (a == "composition") {
      const int n = (int) num(++i);
      if (n < 2) error->all(FLERR,"setup_temperature_source conduction: composition needs n >= 2 nodes");
      comp_x.clear(); comp_cards.clear();
      for (int j = 0; j < n; j++) {
        comp_x.push_back(num(++i));
        if (++i >= args.size()) error->all(FLERR,"setup_temperature_source conduction: composition needs x card pairs");
        comp_cards.push_back(args[i]);
      }
      while (i + 2 < args.size() && (args[i+1] == "bins" || args[i+1] == "mix" || args[i+1] == "feed")) {
        const std::string k = args[++i], v = args[++i];
        if (k == "bins") comp_bins = (int) num(i);
        else if (k == "feed") {
          if (v != "surface" && v != "slab")
            error->all(FLERR,"setup_temperature_source conduction: composition feed is surface or slab");
          cfg.comp_feed_surface = v == "surface";
        }
        else if (v == "pool" || v == "none" || v == "flow") {
          cfg.comp_mix = v != "none"; cfg.comp_flow = v == "flow";
        }
        else error->all(FLERR,"setup_temperature_source conduction: composition mix is pool, flow or none");
      }
    }
    else if (a == "plate_growth") cfg.plate_growth = num(++i);
    else if (a == "layer_thickness") cfg.layer_thickness = num(++i);
    else if (a == "packing") cfg.packing = num(++i);
    else if (a == "sts") cfg.sts_smax = (int) num(++i);
    else if (a == "flow") {
      if (args.at(++i) != "lbm") error->all(FLERR,"setup_temperature_source conduction: flow takes lbm");
      cfg.flow.on = true;
      while (i + 1 < args.size()) {
        const std::string &k = args[i+1];
        if (k != "free_surface" && i + 2 >= args.size()) break;
        if (k == "properties") {
          const std::string &v = args[i+2];
          if (v != "cards" && v != "constant")
            error->all(FLERR,"setup_temperature_source conduction: flow properties is cards or constant");
          cfg.flow.cards = v == "cards";
          i += 2;
          continue;
        }
        if (k == "sigma_table") { cfg.flow.sigma_file = args[i+2]; i += 2; continue; }
        if (k == "free_surface") { cfg.flow.fs = true; i += 1; continue; }
        if (k == "powder") {
          if (i + 3 >= args.size()) error->all(FLERR,"setup_temperature_source conduction: flow powder needs a rate (kg/s) and a radius (m)");
          cfg.flow.powder = num(i + 2); cfg.flow.powder_radius = num(i + 3);
          i += 3;
          if (i + 2 < args.size() && args[i+1] == "T") { cfg.flow.powder_T = num(i + 2); i += 2; }
          continue;
        }
        double *v = k == "dsigma_dT" ? &cfg.flow.dsigma_dT : k == "beta" ? &cfg.flow.beta
                  : k == "viscosity" ? &cfg.flow.viscosity : k == "mushy" ? &cfg.flow.mushy
                  : k == "u_ref" ? &cfg.flow.u_ref : k == "smagorinsky" ? &cfg.flow.smag
                  : k == "sigma" ? &cfg.flow.sigma : k == "fs_tau" ? &cfg.flow.fs_tau
                  : k == "fs_raw" ? &cfg.flow.fs_raw : k == "fs_cap" ? &cfg.flow.fs_cap : k == "fs_edge" ? &cfg.flow.fs_edge : nullptr;
        if (!v) break;
        *v = num(i + 2);
        i += 2;
      }
      if (cfg.flow.viscosity <= 0.0 || cfg.flow.u_ref <= 0.0 || cfg.flow.mushy < 0.0 || cfg.flow.smag < 0.0)
        error->all(FLERR,"setup_temperature_source conduction: flow needs viscosity, u_ref > 0 and mushy, smagorinsky >= 0");
    }
    else if (a == "carry") {
      const std::string v = args.at(++i);
      if (v != "yes" && v != "no") error->all(FLERR,"setup_temperature_source conduction: carry is yes or no");
      cfg.carry = v == "yes";
    }
    else if (a == "far") cfg.far = num(++i);
    else if (a == "init") {
      if (i + 1 < args.size() && args[i+1] == "d6") { i++; cfg.init_d6 = true; }
      else if (i + 2 < args.size() && args[i+1] == "grid") { cfg.init_grid = args[i+2]; i += 2; }
      else error->all(FLERR,"setup_temperature_source conduction: init takes d6 or grid FILE");
    }
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
    else if (a == "beam") {
      if (++i >= args.size() || (args[i] != "gaussian" && args[i] != "square"))
        error->all(FLERR,"setup_temperature_source conduction: beam is gaussian or square");
      cfg.beam_square = args[i] == "square";
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
  if (cfg.far < 0.0)
    error->all(FLERR,"setup_temperature_source conduction: far must be >= 0");
  if (cfg.far > 0.0 && (cfg.ded || cfg.sts_smax == 0))
    error->all(FLERR,"setup_temperature_source conduction: far needs RKL2 (sts >= 2) and no ded");
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
    try {
      static_cast<Config::Material &>(cfg.plate_mat) = card_material(plate_card, world);
    } catch (const std::exception &e) {
      error->all(FLERR, ("setup_temperature_source conduction: plate_material " + std::string(e.what())).c_str());
    }
    cfg.plate_mat.on = true;
  }
  cfg.comp.clear();
  if (!comp_x.empty()) {
    for (size_t j = 1; j < comp_x.size(); j++)
      if (comp_x[j] <= comp_x[j-1])
        error->all(FLERR,"setup_temperature_source conduction: composition x must increase");
    // a byte per cell holds the material: the bins and the plate
    if (comp_bins < 2 || comp_bins > 255 - 1)
      error->all(FLERR,"setup_temperature_source conduction: composition bins must be 2..254");
    std::vector<Material> nodes;
    try {
      for (const std::string &f : comp_cards) nodes.push_back(card_material(f, world));
    } catch (const std::exception &e) {
      error->all(FLERR, ("setup_temperature_source conduction: composition " + std::string(e.what())).c_str());
    }
    if (cfg.comp_feed_surface && !(cfg.comp_flow && cfg.ded))
      error->all(FLERR,"setup_temperature_source conduction: composition feed surface needs mix flow and ded");
    if (cfg.comp_flow && !cfg.flow.on)
      error->all(FLERR,"setup_temperature_source conduction: composition mix flow needs flow lbm");
    cfg.comp_x0 = comp_x.front(); cfg.comp_x1 = comp_x.back();
    if (cfg.flow.cards) {
      if (!cfg.comp_flow)
        error->all(FLERR,"setup_temperature_source conduction: flow properties cards needs composition ... mix flow");
      try {
        flow_props(comp_x, comp_cards, cfg.flow.sigma_file, world, cfg.flow.props);
      } catch (const std::exception &e) {
        error->all(FLERR, ("setup_temperature_source conduction: flow properties cards: " + std::string(e.what())).c_str());
      }
    }
    for (int b = 0; b < comp_bins; b++) {
      const double x = cfg.comp_x0 + (cfg.comp_x1 - cfg.comp_x0) * b / (comp_bins - 1);
      size_t j = 1;
      while (j + 1 < comp_x.size() && comp_x[j] < x) j++;
      const double w = std::min(1.0, std::max(0.0, (x - comp_x[j-1]) / (comp_x[j] - comp_x[j-1])));
      cfg.comp.push_back(blend(nodes[j-1], nodes[j], w));
      cfg.comp.back().name = nodes[0].name + "-" + nodes.back().name + " x=" + MaterialCard::num(x);
    }
  }
  if (cfg.flow.fs) {
    int np;
    MPI_Comm_size(world, &np);
    if (np > 1) error->all(FLERR,"setup_temperature_source conduction: flow free_surface runs on one rank");
    if (!cfg.ded) error->all(FLERR,"setup_temperature_source conduction: flow free_surface needs ded");
    if (cfg.comp_feed_surface)
      error->all(FLERR,"setup_temperature_source conduction: flow free_surface feeds by powder, not feed surface");
    if (!cfg.comp.empty() && !cfg.comp_flow)
      error->all(FLERR,"setup_temperature_source conduction: flow free_surface with composition needs mix flow");
    if (cfg.far > 0.0) error->all(FLERR,"setup_temperature_source conduction: flow free_surface does not take far");
    if (cfg.flow.powder < 0.0 || (cfg.flow.powder > 0.0 && cfg.flow.powder_radius <= 0.0) || cfg.flow.sigma <= 0.0)
      error->all(FLERR,"setup_temperature_source conduction: flow powder needs a rate >= 0 and a radius > 0, sigma > 0");
    if (cfg.flow.powder_T < 0.0) cfg.flow.powder_T = T0_default;
  }
  if (cfg.flow.on && (cfg.flow.cards || !cfg.flow.sigma_file.empty()) && cfg.flow.props.n == 0)
    error->all(FLERR,"setup_temperature_source conduction: flow properties cards and sigma_table "
               "need composition ... mix flow (and sigma_table needs properties cards)");
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
  for (size_t s = 0; s + 2 < cfg.evap.size(); s += 3)
    std::cout << "conduction: evaporation, species " << s / 3 << ": Tb=" << cfg.evap[s] << " K, Lv="
              << cfg.evap[s+1] << " J/kg, M=" << cfg.evap[s+2] << " kg/mol, coefficient " << cfg.evap_coef
              << ", ambient " << cfg.evap_ambient << " Pa" << std::endl;
  if (cfg.far > 0.0)
    std::cout << "conduction: far " << cfg.far << " m: passes farther from the box "
              << "ride RKL2 super-steps; the run ends once the box is done" << std::endl;
  const auto &pm = cfg.plate_mat;
  if (pm.on)
    std::cout << "conduction: plate " << pm.name << " below z=" << cfg.plate_top
              << " m (i2 = 6 up to " << plate_surface() << " m): rho=" << pm.rho << ", "
              << pm.k_T.size() << "-point k table, " << pm.cp_T.size() << "-point cp table, latent="
              << pm.latent << " J/kg, solidus " << pm.ts << " K, liquidus " << pm.tl << " K"
              << std::endl;
  if (!cfg.comp.empty()) {
    const auto &a = cfg.comp.front(), &b = cfg.comp.back();
    std::cout << "conduction: composition " << cfg.comp.size() << " bins, x " << cfg.comp_x0
              << ".." << cfg.comp_x1 << " (d11): solidus " << a.ts << ".." << b.ts << " K, liquidus "
              << a.tl << ".." << b.tl << " K, rho " << a.rho << ".." << b.rho
              << (cfg.comp_flow ? ", mix flow" : cfg.comp_mix ? ", mix pool" : ", mix none")
              << (cfg.comp_feed_surface ? ", feed surface" : "") << std::endl;
  }
  if (cfg.flow.on && cfg.flow.props.n) {
    const auto &p = cfg.flow.props;
    auto sg = [&](int t, int i) { return p.sig[(size_t) t * p.nx + i]; };
    std::cout << "conduction: flow lbm, properties cards: at " << p.Tr << " K, x "
              << cfg.comp_x0 << " / " << cfg.comp_x1 << ": liquid density " << p.r0.front() << " / "
              << p.r0.back() << " kg/m^3, viscosity " << p.eta.front() << " / " << p.eta.back()
              << " Pa s; surface tension "
              << (cfg.flow.sigma_file.empty() ? "from the cards" : "from " + cfg.flow.sigma_file)
              << ", at " << p.T0 << " K " << sg(0, 0) << " / " << sg(0, p.nx - 1)
              << " N/m, dsigma/dT " << (sg(1, 0) - sg(0, 0)) / p.dT << " / "
              << (sg(1, p.nx - 1) - sg(0, p.nx - 1)) / p.dT << " N/m/K; mushy " << cfg.flow.mushy
              << " 1/s, u_ref " << cfg.flow.u_ref << " m/s, smagorinsky " << cfg.flow.smag << std::endl;
  } else if (cfg.flow.on)
    std::cout << "conduction: flow lbm, dsigma/dT " << cfg.flow.dsigma_dT << " N/m/K, beta "
              << cfg.flow.beta << " 1/K, viscosity " << cfg.flow.viscosity << " m^2/s, mushy "
              << cfg.flow.mushy << " 1/s, u_ref " << cfg.flow.u_ref << " m/s, smagorinsky "
              << cfg.flow.smag << std::endl;
  if (cfg.flow.fs)
    std::cout << "conduction: flow lbm free surface"
              << (cfg.flow.props.n ? "" : ", surface tension " + std::to_string(cfg.flow.sigma) + " N/m")
              << "; powder " << cfg.flow.powder << " kg/s within " << cfg.flow.powder_radius
              << " m of the axis at " << cfg.flow.powder_T << " K; no cell is born" << std::endl;
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

std::vector<ConductionTemperatureSource::Config::Material>
ConductionTemperatureSource::materials(double ts, double tl) const
{
  std::vector<Config::Material> m = cfg.comp;
  if (m.empty()) {
    Config::Material p;
    p.k_T = cfg.k_T; p.k_v = cfg.k_v; p.cp_T = cfg.cp_T; p.cp_v = cfg.cp_v;
    p.rho = rho; p.latent = cfg.latent; p.ts = ts; p.tl = tl;
    m.push_back(p);
  }
  if (cfg.plate_mat.on) m.push_back(cfg.plate_mat);
  return m;
}

int ConductionTemperatureSource::comp_bin(double x) const
{
  if (cfg.comp.empty()) return 0;
  const int n = (int) cfg.comp.size();
  const double u = (x - cfg.comp_x0) / (cfg.comp_x1 - cfg.comp_x0) * (n - 1);
  return std::max(0, std::min(n - 1, (int) std::lround(u)));
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

void ConductionTemperatureSource::set_box(double xlo, double xhi, double ylo, double yhi)
{
  box_[0] = xlo; box_[1] = xhi; box_[2] = ylo; box_[3] = yhi;
  box_set = true;
}

/* ----------------------------------------------------------------------
   `far`: the earliest time >= t at which the laser is on inside the lattice
   box grown by far on each side (xy), or max() if it never is. Each emitting
   segment is clipped to that rectangle (Liang-Barsky); a segment that left it
   before t does not count.
------------------------------------------------------------------------- */

double ConductionTemperatureSource::next_near(double t) const
{
  if (!scan_built || !box_set) return t;
  const double f = cfg.far;
  const double xlo = box_[0] - f, xhi = box_[1] + f, ylo = box_[2] - f, yhi = box_[3] + f;
  for (auto it = first_ending_after(segs_, t); it != segs_.end(); ++it) {
    const ScanSeg &g = *it;
    if (g.p <= 0.0) continue;
    double u0 = 0.0, u1 = 1.0;
    auto clip = [&](double p, double q) {   // keep p u <= q
      if (p == 0.0) return q >= 0.0;
      const double r = q / p;
      if (p < 0.0) { if (r > u1) return false; u0 = std::max(u0, r); }
      else         { if (r < u0) return false; u1 = std::min(u1, r); }
      return true;
    };
    const double dxs = g.x1 - g.x0, dys = g.y1 - g.y0;
    if (!(clip(-dxs, g.x0 - xlo) && clip(dxs, xhi - g.x0) &&
          clip(-dys, g.y0 - ylo) && clip(dys, yhi - g.y0))) continue;
    if (g.t0 + u1 * (g.t1 - g.t0) < t) continue;
    return std::max(t, g.t0 + u0 * (g.t1 - g.t0));
  }
  return std::numeric_limits<double>::max();
}

/* ----------------------------------------------------------------------
   The laser's on-time over [t0, t1] as points along the path, at most ds
   apart: (x, y, w) triples, w the power-weighted share of [t0, t1], so the
   weights sum to laser_window's fraction.
------------------------------------------------------------------------- */

void ConductionTemperatureSource::laser_samples(double t0, double t1, double ds,
                                                std::vector<double> &xyw) const
{
  xyw.clear();
  if (!scan_built || t1 <= t0) return;
  for (auto it = first_ending_after(segs_, t0); it != segs_.end() && it->t0 < t1; ++it) {
    const ScanSeg &g = *it;
    const double a = std::max(t0, g.t0), b = std::min(t1, g.t1);
    if (b <= a || g.p <= 0.0) continue;
    const double vx = (g.x1 - g.x0) / (g.t1 - g.t0), vy = (g.y1 - g.y0) / (g.t1 - g.t0);
    const int n = std::max(1, (int) std::ceil(std::hypot(vx, vy) * (b - a) / ds));
    const double h = (b - a) / n, w = h * g.p / (t1 - t0);
    for (int k = 0; k < n; k++) {
      const double tm = a + (k + 0.5) * h - g.t0;
      xyw.push_back(g.x0 + vx * tm);
      xyw.push_back(g.y0 + vy * tm);
      xyw.push_back(w);
    }
  }
}

/* ----------------------------------------------------------------------
   H(T) = int_0^T rho cp dT' + rho L f(T), f linear from 0 at ts to 1 at tl.
   1 K spacing to 6000 K; the solver extrapolates beyond with the end slopes.
------------------------------------------------------------------------- */

void ConductionTemperatureSource::enthalpy_table(const Config::Material &m,
                                                 double &T_lo, double &dT,
                                                 std::vector<double> &H)
{
  const double rho_m = m.rho, latent = m.latent, ts = m.ts, tl = m.tl;
  auto cp_at = [&](double T) {
    const auto &Tt = m.cp_T, &v = m.cp_v;
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
