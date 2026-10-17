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

#include "material_card.h"
// toml++ trips clang's -ffast-math inf warning and a C++23 deprecation
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnan-infinity-disabled"
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#include "../lib/tomlplusplus/toml.hpp"
#pragma clang diagnostic pop
#include <charconv>
#include <cmath>
#include <sstream>
#include <stdexcept>

using namespace SPPARKS_NS;

namespace {

[[noreturn]] void fail(const std::string &where, const std::string &msg)
{
  throw std::runtime_error("material card: " + where + ": " + msg);
}

std::string ref_text(const toml::table &rec, const std::string &where)
{
  const toml::node *r = rec.get("ref");
  if (auto s = r ? r->value<std::string>() : std::nullopt) {
    if (!s->empty()) return *s;
  } else if (const toml::array *a = r ? r->as_array() : nullptr) {
    std::string out;
    for (const toml::node &e : *a)
      if (auto s = e.value<std::string>()) out += (out.empty() ? "" : ", ") + *s;
    if (!out.empty()) return out;
  }
  fail(where, "needs a ref (citation key, list of keys, \"estimate\" or \"calibration\")");
}

const toml::table *record(const toml::table &root, const std::string &where, const char *unit)
{
  const toml::table *rec = root.at_path(where).as_table();
  if (!rec) {
    if (root.at_path(where)) fail(where, "must be a table such as { value = ..., unit = ..., ref = ... }");
    return nullptr;
  }
  auto u = (*rec)["unit"].value<std::string>();
  if (!u || *u != unit) fail(where, std::string("unit must be \"") + unit + "\"");
  return rec;
}

double number(const toml::table &rec, const char *key, const std::string &where)
{
  auto v = rec[key].value<double>();
  if (!v) fail(where, std::string("missing numeric '") + key + "'");
  return *v;
}

std::vector<double> numbers(const toml::table &rec, const char *key, const std::string &where)
{
  const toml::array *a = rec[key].as_array();
  if (!a || a->empty()) fail(where, std::string("'") + key + "' must be a non-empty array");
  std::vector<double> out;
  for (const toml::node &e : *a) {
    auto v = e.value<double>();
    if (!v) fail(where, std::string("'") + key + "' must hold numbers");
    out.push_back(*v);
  }
  return out;
}

struct Reader {
  const toml::table &root;
  std::vector<std::string> &summary;

  void note(const std::string &where, const std::string &val, const char *unit,
            const toml::table &rec) {
    summary.push_back(where + " = " + val + " " + unit + " [" + ref_text(rec, where) + "]");
  }
  MaterialCard::opt scalar(const std::string &where, const char *unit) {
    const toml::table *rec = record(root, where, unit);
    if (!rec) return std::nullopt;
    const double v = number(*rec, "value", where);
    note(where, MaterialCard::num(v), unit, *rec);
    return v;
  }
  MaterialCard::Table table(const std::string &where, const char *unit) {
    MaterialCard::Table t;
    const toml::table *rec = record(root, where, unit);
    if (!rec) return t;
    t.T = numbers(*rec, "T", where);
    t.v = numbers(*rec, "value", where);
    if (t.T.size() != t.v.size()) fail(where, "T and value differ in length");
    for (size_t i = 1; i < t.T.size(); i++)
      if (t.T[i] <= t.T[i-1]) fail(where, "T must increase strictly");
    if (auto e = (*rec)["extrapolation"].value<std::string>(); e && *e != "clamp")
      fail(where, "extrapolation must be \"clamp\"");
    note(where, std::to_string(t.T.size()) + " points " + MaterialCard::num(t.T.front()) +
         ".." + MaterialCard::num(t.T.back()) + " K", unit, *rec);
    return t;
  }
};

}

/* ---------------------------------------------------------------------- */

std::string MaterialCard::num(double v)
{
  char buf[32];
  auto r = std::to_chars(buf, buf + sizeof(buf), v);
  return std::string(buf, r.ptr);
}

double MaterialCard::table_at(const Table &t, double T)
{
  if (T <= t.T.front()) return t.v.front();
  if (T >= t.T.back()) return t.v.back();
  size_t i = 1;
  while (t.T[i] < T) i++;
  const double f = (T - t.T[i-1]) / (t.T[i] - t.T[i-1]);
  return t.v[i-1] + f * (t.v[i] - t.v[i-1]);
}

/* ---------------------------------------------------------------------- */

MaterialCard MaterialCard::parse(const std::string &text, const std::string &path,
                                 const std::string &process)
{
  toml::table root;
  try { root = toml::parse(text, path); }
  catch (const toml::parse_error &e) {
    std::ostringstream os; os << e;
    fail(path, os.str());
  }

  MaterialCard c;
  const toml::table *card = root["card"].as_table();
  if (!card) fail(path, "missing [card] table");
  auto schema = (*card)["schema_version"].value<int64_t>();
  if (!schema) fail("card.schema_version", "missing");
  if (*schema != 1) fail("card.schema_version", std::to_string(*schema) + " is not supported (this build reads 1)");
  auto mat = (*card)["material"].value<std::string>();
  auto ver = (*card)["version"].value<std::string>();
  if (!mat || !ver) fail("card", "needs material and version");
  const toml::node *mod = card->get("modified");
  if (!mod || !mod->is_date()) fail("card.modified", "must be a date (YYYY-MM-DD)");
  std::ostringstream md; md << *mod->as_date();
  c.material = *mat; c.version = *ver; c.modified = md.str();

  Reader rd{root, c.summary};
  c.density  = rd.scalar("thermal.density", "kg/m^3");
  c.solidus  = rd.scalar("thermal.solidus", "K");
  c.liquidus = rd.scalar("thermal.liquidus", "K");
  c.latent   = rd.scalar("thermal.latent_fusion", "J/kg");
  c.emissivity = rd.scalar("thermal.emissivity", "1");
  c.k  = rd.table("thermal.conductivity", "W/(m K)");
  c.cp = rd.table("thermal.heat_capacity", "J/(kg K)");
  c.k_lin  = rd.scalar("thermal.linear.conductivity", "W/(m K)");
  c.cp_lin = rd.scalar("thermal.linear.heat_capacity", "J/(kg K)");
  c.T_ref  = rd.scalar("thermal.linear.T_ref", "K");
  c.k_powder = rd.scalar("powder.conductivity", "W/(m K)");
  c.packing  = rd.scalar("powder.packing", "1");
  c.absorptivity = rd.scalar("optical.absorptivity", "1");

  if (const toml::table *rec = record(root, "microstructure.solid_front_vel", "m/s")) {
    if ((*rec)["dT_unit"].value<std::string>() != std::optional<std::string>("K"))
      fail("microstructure.solid_front_vel", "dT_unit must be \"K\"");
    c.front = numbers(*rec, "coeffs", "microstructure.solid_front_vel");
    std::string s;
    for (double v : c.front) s += (s.empty() ? "" : " ") + num(v);
    rd.note("microstructure.solid_front_vel", "[" + s + "] (dT in K)", "m/s", *rec);
  }
  c.nucleation_density      = rd.scalar("microstructure.nucleation_density", "1/m^3");
  c.critical_undercooling   = rd.scalar("microstructure.critical_undercooling", "K");
  c.undercooling_deviation  = rd.scalar("microstructure.undercooling_deviation", "K");
  c.mean_nuclei_volume      = rd.scalar("microstructure.mean_nuclei_volume", "m^3");
  c.nuclei_volume_deviation = rd.scalar("microstructure.nuclei_volume_deviation", "m^3");

  if (!process.empty()) {
    const toml::table *sets = root["process"].as_table();
    if (!sets || !sets->get(process)) {
      std::string have;
      if (sets) for (auto &&[name, node] : *sets) have += " " + std::string(name.str());
      fail("process." + process, "no such process set (card has:" + (have.empty() ? " none" : have) + ")");
    }
    c.process = process;
    const std::string p = "process." + process + ".";
    c.power = rd.scalar(p + "power", "W");
    c.speed = rd.scalar(p + "speed", "m/s");
    c.hatch = rd.scalar(p + "hatch", "m");
    c.layer_thickness = rd.scalar(p + "layer_thickness", "m");
    c.source_depth = rd.scalar(p + "source_depth", "m");
    c.preheat = rd.scalar(p + "preheat", "K");
    if (auto a = rd.scalar(p + "absorptivity", "1")) c.absorptivity = a;
    if (const toml::table *rec = record(root, p + "spot", "m")) {
      // Moser's source is exp(-r^2/sx^2): sx is the 1/e radius
      const double v = number(*rec, "value", p + "spot");
      auto conv = (*rec)["convention"].value<std::string>().value_or("");
      if (conv == "1/e radius") c.spot_sigma = v;
      else if (conv == "1/e2 radius") c.spot_sigma = v / std::sqrt(2.0);
      else if (conv == "1/e2 diameter" || conv == "D4sigma") c.spot_sigma = v / (2.0 * std::sqrt(2.0));
      else if (conv == "FWHM") c.spot_sigma = v / (2.0 * std::sqrt(std::log(2.0)));
      else fail(p + "spot", "convention must be \"1/e radius\", \"1/e2 radius\", \"1/e2 diameter\", \"D4sigma\" or \"FWHM\"");
      rd.note(p + "spot", num(v) + " (" + conv + ") -> sx " + num(*c.spot_sigma), "m", *rec);
    }
    if (const toml::table *rec = record(root, p + "convection", "W/(m^2 K)")) {
      c.conv_h = number(*rec, "h", p + "convection");
      c.conv_Tinf = number(*rec, "T_inf", p + "convection");
      rd.note(p + "convection", "h " + num(*c.conv_h) + ", T_inf " + num(*c.conv_Tinf) + " K", "W/(m^2 K)", *rec);
    }
  }
  return c;
}

/* ----------------------------------------------------------------------
   setup_temperature_source <type> card [Q v] [lambda v] [T0 v] [spot sx sy sz] [...]
   -> the type's explicit token list. Deck keywords the translation does not
   know are appended after the card's, and the parsers keep the last value.
------------------------------------------------------------------------- */

std::vector<std::string> MaterialCard::source_tokens(const std::string &type,
                                                     const std::vector<std::string> &args) const
{
  opt Q = power, lam = absorptivity, T0 = preheat;
  opt sx = spot_sigma, sy = spot_sigma, sz = source_depth;
  std::vector<std::string> rest;
  auto arg = [&](size_t &i) {
    if (++i >= args.size()) fail("setup_temperature_source card", "missing value after " + args[i-1]);
    try { return std::stod(args[i]); }
    catch (const std::exception &) { fail("setup_temperature_source card", "bad number " + args[i]); }
  };
  for (size_t i = 0; i < args.size(); i++) {
    const std::string &a = args[i];
    if (a == "Q") Q = arg(i);
    else if (a == "lambda") lam = arg(i);
    else if (a == "T0") T0 = arg(i);
    else if (a == "spot") { sx = arg(i); sy = arg(i); sz = arg(i); }
    else if (a == "keyhole" || a == "aniso" || a == "anisotropic" || a == "standard")
      fail("setup_temperature_source card", "only the standard mode reads cards; give '" + a + "' explicitly instead");
    else rest.push_back(a);
  }
  auto need = [&](opt v, const char *what) {
    if (!v)
      fail("setup_temperature_source " + type + " card",
           std::string(what) + " is not in the card" + (process.empty() ? " (no process set selected)" : "") +
           "; give it in the deck");
    return *v;
  };
  const double q = need(Q, "Q (process power)"), l = need(lam, "lambda (absorptivity)"),
               t0 = need(T0, "T0 (process preheat)");

  double kk, cc;
  if (type == "conduction") {      // positional k and cp at T0; the tables carry T dependence
    kk = k.T.empty() ? need(k_lin, "thermal.conductivity") : table_at(k, t0);
    cc = cp.T.empty() ? need(cp_lin, "thermal.heat_capacity") : table_at(cp, t0);
  } else {                         // linear models: explicit constants, or the tables at T_ref
    kk = k_lin ? *k_lin : k.T.empty() ? need(k_lin, "thermal.linear.conductivity")
                        : table_at(k, need(T_ref, "thermal.linear.T_ref"));
    cc = cp_lin ? *cp_lin : cp.T.empty() ? need(cp_lin, "thermal.linear.heat_capacity")
                          : table_at(cp, need(T_ref, "thermal.linear.T_ref"));
  }
  const double alpha = kk / (need(density, "thermal.density") * cc);

  std::vector<std::string> out;
  if (type == "rosenthal") {
    out = {"standard", num(q), num(l), num(kk), num(alpha), num(t0)};
  } else if (type == "moser" || type == "conduction") {
    out = {"standard", num(q), num(l), num(kk), num(alpha), num(t0), num(cc),
           num(need(sx, "spot")), num(need(sy, "spot")), num(need(sz, "source_depth"))};
  } else {
    fail("setup_temperature_source " + type, "cards serve rosenthal, moser and conduction");
  }
  if (type == "conduction") {
    auto table = [&](const char *kw, const Table &t) {
      if (t.T.empty()) return;
      out.push_back(kw); out.push_back(std::to_string(t.T.size()));
      for (size_t i = 0; i < t.T.size(); i++) { out.push_back(num(t.T[i])); out.push_back(num(t.v[i])); }
    };
    auto scalar = [&](const char *kw, opt v) {
      if (v) { out.push_back(kw); out.push_back(num(*v)); }
    };
    table("k_table", k); table("cp_table", cp);
    scalar("latent", latent); scalar("k_powder", k_powder); scalar("emissivity", emissivity);
    scalar("layer_thickness", layer_thickness);
    if (conv_h) { out.push_back("convection"); out.push_back(num(*conv_h)); out.push_back(num(*conv_Tinf)); }
    // ponytail: powder.packing is recorded, not passed: packing < 1 assumes no
    // shrinkage, valid for single-layer tests only (docs/CONDUCTION_SOURCE.md)
  }
  out.insert(out.end(), rest.begin(), rest.end());
  return out;
}
