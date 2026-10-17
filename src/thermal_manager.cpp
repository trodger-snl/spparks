/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   http://www.cs.sandia.gov/~sjplimp/spparks.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPPARKS directory.

   ThermalManager: temperature-I/O orchestration extracted from
   AppAdditiveTexture. Owns the modular temperature source, laser scan-path
   state, multi-pass pause/reset scheduling, the fast-forward predictors, and
   the optional solidification-band temperature smoothing + diagnostics. The
   microstructure-evolution physics stays in AppAdditiveTexture.
------------------------------------------------------------------------- */

#include "thermal_manager.h"
#include "thermal_host.h"
#include "temperature_source_rosenthal.h"
#include "temperature_source_moser.h"
#include "temperature_source_conduction.h"
#include "temperature_source_hdf5_unstructured.h"
#include "comm_lattice.h"
#include "domain.h"
#include "timer.h"
#include "error.h"
#include "math_const.h"
#include "spktype.h"

#include <iostream>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>

using namespace SPPARKS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

ThermalManager::ThermalManager(SPPARKS *spk, ThermalHost *host) :
  Pointers(spk), host_(host)
{
  // Initialize modular temperature source system
  temperature_source = nullptr;
  fast_forward_search_window = 0.1;  // Default 100ms search window

  // Rosenthal scan-path state (only used when temperature_source is Rosenthal)
  scan_layer = RASTER::Layer();
  scan_layer_z = 0.0;
  scan_layer_time = 0.0;
  scan_layer_active = false;
  laser_path_set = false;

  // Multi-pass pause control (disabled until laser_path specifies pause/pause_below)
  laser_pause_constant     = 0.0;
  laser_pause_below        = 0.0;
  laser_reset_temperature  = 0.0;
  pending_moser_passes     = 0;


  // Solidification-band smoothing defaults (off until `temperature_smooth`)
  temperature_smooth_enabled = false;
  smooth_tmin  = 0.0;
  smooth_tmax  = 0.0;
  smooth_guard = 20.0;
  smooth_sigma_xy = 1.2;
  smooth_sigma_z  = 1.2;
  smooth_alpha = 0.4;
  smooth_passes = 1;
  smooth_diag_interval = 0;
}

/* ----------------------------------------------------------------------
   setup_temperature_source command: instantiate a temperature source via
   the factory and configure it with the remaining arguments.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   material <file> [process <name>]: load a material card (docs/MATERIAL_CARDS.md).
   Rank 0 reads the file; every rank parses the broadcast text.
------------------------------------------------------------------------- */

void ThermalManager::material_cmd(int narg, char **arg)
{
  if (narg != 1 && !(narg == 3 && strcmp(arg[1],"process") == 0))
    error->all(FLERR,"Illegal material command: material <file> [process <name>]");
  std::string text;
  int n = 0;
  if (domain->me == 0) {
    std::ifstream f(arg[0], std::ios::binary);
    if (f) text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    n = f ? (int) text.size() : -1;
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, world);
  if (n < 0) error->all(FLERR,(std::string("material: cannot open ") + arg[0]).c_str());
  text.resize(n);
  MPI_Bcast(text.data(), n, MPI_CHAR, 0, world);
  try {
    card_ = std::make_unique<MaterialCard>(
      MaterialCard::parse(text, arg[0], narg == 3 ? arg[2] : ""));
  } catch (const std::exception &e) {
    error->all(FLERR, e.what());
  }
  if (domain->me == 0) {
    std::cout << "material card " << arg[0] << ": " << card_->material << " version "
              << card_->version << " modified " << card_->modified;
    if (!card_->process.empty()) std::cout << ", process " << card_->process;
    std::cout << std::endl;
    for (const std::string &l : card_->summary) std::cout << "  " << l << std::endl;
  }
}

void ThermalManager::setup_cmd(int narg, char **arg)
{
  if (narg < 1) {
    error->all(FLERR,"Illegal setup_temperature_source command: must specify type");
  }

  std::string source_type = arg[0];
  std::vector<std::string> source_args;

  // Convert remaining arguments to string vector
  for (int i = 1; i < narg; i++) {
    source_args.push_back(std::string(arg[i]));
  }

  // "card": the material card supplies the positional and material keywords
  if (!source_args.empty() && source_args[0] == "card") {
    if (!card_)
      error->all(FLERR,"setup_temperature_source ... card: no material card (use the material command first)");
    try {
      source_args = card_->source_tokens(source_type,
        std::vector<std::string>(source_args.begin() + 1, source_args.end()));
    } catch (const std::exception &e) {
      error->all(FLERR, e.what());
    }
    if (domain->me == 0) {
      std::cout << "setup_temperature_source " << source_type << " from card " << card_->material
                << " " << card_->version << ":";
      for (const std::string &t : source_args) std::cout << " " << t;
      std::cout << std::endl;
    }
  }

  // Create temperature source using factory
  temperature_source = SPPARKS_NS::create_temperature_source(source_type, spk);
  if (!temperature_source) {
    error->all(FLERR,"Failed to create temperature source");
  }

  // Setup the temperature source with provided arguments
  temperature_source->setup_temperature_source(source_args);

  if (domain->me == 0) {
    std::cout << "Modular temperature source (" << source_type << ") enabled" << std::endl;
  }
}

/* ----------------------------------------------------------------------
   laser_path command: define a single linear scan, optionally repeated N
   times. Coordinates and velocity are SI (meters, m/s).
------------------------------------------------------------------------- */

void ThermalManager::laser_path_cmd(int narg, char **arg)
{
  // setup_temperature_source must precede laser_path: the Moser source
  // consumes the path geometry inside build_scan, and the Rosenthal
  // source needs r_min sized below.
  if (!temperature_source)
    error->all(FLERR,
      "laser_path: setup_temperature_source must be called first "
      "(no active temperature source). Add a 'setup_temperature_source "
      "<type> ...' line before 'laser_path'.");

  // Expected token layout (0-based, command name already stripped):
  //   start X0 Y0 Z0 end X1 Y1 speed V [repeats N | hatch S N] [serpentine]
  //     [pause <T> | pause_below <Tk>]
  // hatch S N: N passes, pass k offset by k*S along the line's in-plane
  // normal; serpentine reverses the direction on odd passes.
  // speed card / hatch card: the selected process set's value
  std::vector<std::string> subst(arg, arg + narg);
  std::vector<char *> argv_sub;
  for (int i = 1; i < narg; i++) {
    if (subst[i] != "card" || (subst[i-1] != "speed" && subst[i-1] != "hatch")) continue;
    const MaterialCard::opt v = !card_ ? std::nullopt : subst[i-1] == "speed" ? card_->speed : card_->hatch;
    if (!v)
      error->all(FLERR,("laser_path: " + subst[i-1] + " card needs a material card with a process set "
                        "giving " + subst[i-1]).c_str());
    subst[i] = MaterialCard::num(*v);
  }
  for (std::string &t : subst) argv_sub.push_back(t.data());
  arg = argv_sub.data();

  if (narg >= 1 && strcmp(arg[0],"file") == 0) {
    laser_path_file(narg - 1, arg + 1);
    return;
  }
  path_layers.clear();

  if (narg < 9)
    error->all(FLERR,"Illegal laser_path command: expected start X0 Y0 Z0 end X1 Y1 speed V [repeats N | hatch S N] [serpentine] [pause T | pause_below Tk]");
  if (strcmp(arg[0],"start") != 0)
    error->all(FLERR,"laser_path: expected keyword 'start'");
  const double x0 = atof(arg[1]);
  const double y0 = atof(arg[2]);
  const double z0 = atof(arg[3]);
  if (strcmp(arg[4],"end") != 0)
    error->all(FLERR,"laser_path: expected keyword 'end' after start coordinates");
  const double x1 = atof(arg[5]);
  const double y1 = atof(arg[6]);
  if (strcmp(arg[7],"speed") != 0)
    error->all(FLERR,"laser_path: expected keyword 'speed' after end coordinates");
  const double v = atof(arg[8]);
  if (v <= 0.0)
    error->all(FLERR,"laser_path: speed must be > 0");

  int repeats = 1;
  double hatch = 0.0;
  bool serpentine = false;
  double pause_constant   = 0.0;
  double pause_below_T    = 0.0;
  double reset_temperature_T = 0.0;
  int i = 9;
  while (i < narg) {
    if (strcmp(arg[i],"repeats") == 0) {
      if (i + 1 >= narg)
        error->all(FLERR,"laser_path: missing value after 'repeats'");
      repeats = atoi(arg[i+1]);
      if (repeats < 1)
        error->all(FLERR,"laser_path: repeats must be >= 1");
      i += 2;
    }
    else if (strcmp(arg[i],"hatch") == 0) {
      if (i + 2 >= narg)
        error->all(FLERR,"laser_path: expected 'hatch <spacing_m> <n_passes>'");
      hatch = atof(arg[i+1]);
      repeats = atoi(arg[i+2]);
      if (hatch <= 0.0 || repeats < 1)
        error->all(FLERR,"laser_path: hatch spacing must be > 0 and n_passes >= 1");
      i += 3;
    }
    else if (strcmp(arg[i],"serpentine") == 0) {
      serpentine = true;
      i += 1;
    }
    else if (strcmp(arg[i],"pause") == 0) {
      if (i + 1 >= narg)
        error->all(FLERR,"laser_path: missing value after 'pause'");
      pause_constant = atof(arg[i+1]);
      if (pause_constant <= 0.0)
        error->all(FLERR,"laser_path: pause must be > 0 seconds");
      i += 2;
    }
    else if (strcmp(arg[i],"pause_below") == 0) {
      if (i + 1 >= narg)
        error->all(FLERR,"laser_path: missing value after 'pause_below'");
      pause_below_T = atof(arg[i+1]);
      if (pause_below_T <= 0.0)
        error->all(FLERR,"laser_path: pause_below threshold must be > 0 K");
      i += 2;
    }
    else if (strcmp(arg[i],"reset_temperature") == 0) {
      if (i + 1 >= narg)
        error->all(FLERR,"laser_path: missing value after 'reset_temperature'");
      reset_temperature_T = atof(arg[i+1]);
      if (reset_temperature_T <= 0.0)
        error->all(FLERR,"laser_path: reset_temperature must be > 0 K");
      i += 2;
    }
    else {
      error->all(FLERR,"laser_path: unknown keyword (expected 'repeats', 'hatch', 'serpentine', 'pause', 'pause_below', or 'reset_temperature')");
    }
  }

  if (pause_constant > 0.0 && pause_below_T > 0.0)
    error->all(FLERR,
      "laser_path: 'pause' and 'pause_below' are mutually exclusive. "
      "Use 'pause <T>' for a fixed inter-pass delay, or "
      "'pause_below <Tk>' for a temperature-threshold-gated delay.");
  if (reset_temperature_T > 0.0 && pause_constant <= 0.0 && pause_below_T <= 0.0)
    error->all(FLERR,
      "laser_path: 'reset_temperature' requires either 'pause <T>' or "
      "'pause_below <Tk>' to define when the reset is applied.");
  if (hatch > 0.0 && reset_temperature_T > 0.0)
    error->all(FLERR,
      "laser_path: 'hatch' and 'reset_temperature' cannot be combined "
      "(a reset restarts the pass index at the base line).");

  // Both pause modes require a time-resolved source. Rosenthal is the
  // steady-state moving-point solution, so each pass is already independent
  // of every preceding pass; reject pause at parse time.
  const bool src_is_rosenthal =
    (rosenthal_source() != nullptr);
  if ((pause_constant > 0.0 || pause_below_T > 0.0) && src_is_rosenthal) {
    error->all(FLERR,
      "laser_path pause/pause_below: not supported for the Rosenthal "
      "source. Rosenthal is a steady-state moving-point solution; "
      "multi-pass repeats are already physically independent (no "
      "thermal history carries between passes). Use a time-resolved "
      "source (moser) if you need cooling "
      "between passes.");
  }
  if ((pause_constant > 0.0 || pause_below_T > 0.0) && repeats < 2) {
    if (domain->me == 0) {
      std::cout << "laser_path: warning: pause keyword specified but "
                   "repeats=1; no inter-pass interval will occur."
                << std::endl;
    }
  }

  laser_pause_constant    = pause_constant;
  laser_pause_below       = pause_below_T;
  laser_reset_temperature = reset_temperature_T;

  // Build N paths back-to-back in the Rosenthal scan_layer (Moser does not
  // consume scan_layer): the base line offset r*hatch along its in-plane
  // normal, reversed on odd passes when serpentine.
  std::vector<RASTER::Path> paths;
  paths.reserve(repeats);
  {
    const double ddx = x1 - x0, ddy = y1 - y0, L = std::sqrt(ddx*ddx + ddy*ddy);
    const double nx = (L > 0.0) ? -ddy / L : 0.0, ny = (L > 0.0) ? ddx / L : 0.0;
    for (int r = 0; r < repeats; ++r) {
      const double off = r * hatch;
      RASTER::Point a(x0 + off*nx, y0 + off*ny, 0.0), b(x1 + off*nx, y1 + off*ny, 0.0);
      if (serpentine && (r % 2 == 1)) paths.emplace_back(b, a, v);
      else                            paths.emplace_back(a, b, v);
    }
  }

  scan_layer = RASTER::Layer(paths, /*thickness*/0);
  scan_layer_z = z0;
  scan_layer_time = host_->time;  // anchor the layer pose to the current sim time
  scan_layer_active = true;
  laser_path_set = true;

  // Size the singularity cutoff to the lattice so the 1/R prefactor at the
  // laser site stays finite. Only meaningful for a Rosenthal source.
  if (auto* ros = rosenthal_source()) {
    ros->set_r_min(0.5 * host_->dx);
  }

  // Hand the path geometry to the Moser source so it can build its own
  // GREENAM LaserScan/ScanIntegration. Dynamic scheduling (only pass 1 built
  // upfront, remaining appended one at a time) is used when the app must
  // interpose between passes: pause_below alone, or pause + reset_temperature.
  const bool dynamic_pass_scheduling =
    (pause_below_T > 0.0) ||
    (pause_constant > 0.0 && reset_temperature_T > 0.0);
  if (auto* moser = moser_source()) {
    if (dynamic_pass_scheduling) {
      moser->build_scan(host_->time, x0, y0, x1, y1, z0, v, /*repeats*/1, /*pause*/0.0,
                        hatch, serpentine);
      pending_moser_passes = repeats - 1;
    } else {
      moser->build_scan(host_->time, x0, y0, x1, y1, z0, v, repeats, pause_constant,
                        hatch, serpentine);
      pending_moser_passes = 0;
    }
  }

  if (domain->me == 0) {
    std::cout << "laser_path: (" << x0 << "," << y0 << "," << z0
              << ") -> (" << x1 << "," << y1 << "," << z0 << ")"
              << " speed=" << v << " m/s, repeats=" << repeats;
    if (hatch > 0.0)
      std::cout << ", hatch=" << hatch << " m" << (serpentine ? " serpentine" : "");
    if (pause_constant > 0.0)
      std::cout << ", pause=" << pause_constant << " s";
    if (pause_below_T > 0.0)
      std::cout << ", pause_below=" << pause_below_T << " K";
    if (reset_temperature_T > 0.0)
      std::cout << ", reset_temperature=" << reset_temperature_T << " K";
    std::cout << std::endl;
  }
}

/* ----------------------------------------------------------------------
   laser_path file F speed V [scale S] [shift DX DY DZ] [jump_speed V]
     [jump_time T] [recoat T] [layers N]
   Waypoint file, one row per point: x y z p [v] (whitespace or commas,
   '#' comments). Each row is a move from the previous point at power
   fraction p (0 = laser off) and speed v (default V). Off moves take
   jump_time if given, else distance / jump_speed (default V). A change of
   z starts a new layer at that point; layers run one after another, each
   once the previous one has cooled below the solidus with no liquid left,
   plus recoat seconds. Coordinates are file units * scale + shift (m).
------------------------------------------------------------------------- */

void ThermalManager::laser_path_file(int narg, char **arg)
{
  if (narg < 3 || strcmp(arg[1],"speed") != 0)
    error->all(FLERR,"Illegal laser_path file command: expected file F speed V ...");
  if (!moser_source() || rosenthal_source())
    error->all(FLERR,"laser_path file: requires the moser or conduction source");
  const double v = atof(arg[2]);
  double scale = 1.0, shift[3] = {0.0, 0.0, 0.0}, jump_speed = v, jump_time = 0.0;
  double recoat = 0.0;
  int nlayers = 0;
  for (int i = 3; i < narg; ) {
    auto need = [&](int n) {
      if (i + n >= narg) error->all(FLERR,"laser_path file: missing keyword value");
    };
    if (strcmp(arg[i],"scale") == 0) { need(1); scale = atof(arg[i+1]); i += 2; }
    else if (strcmp(arg[i],"shift") == 0) {
      need(3);
      for (int d = 0; d < 3; d++) shift[d] = atof(arg[i+1+d]);
      i += 4;
    }
    else if (strcmp(arg[i],"jump_speed") == 0) { need(1); jump_speed = atof(arg[i+1]); i += 2; }
    else if (strcmp(arg[i],"jump_time") == 0) { need(1); jump_time = atof(arg[i+1]); i += 2; }
    else if (strcmp(arg[i],"recoat") == 0) { need(1); recoat = atof(arg[i+1]); i += 2; }
    else if (strcmp(arg[i],"layers") == 0) { need(1); nlayers = atoi(arg[i+1]); i += 2; }
    else error->all(FLERR,"laser_path file: unknown keyword (expected scale, shift, "
                    "jump_speed, jump_time, recoat or layers)");
  }
  if (v <= 0.0 || scale <= 0.0 || jump_speed <= 0.0 || jump_time < 0.0 ||
      recoat < 0.0 || nlayers < 0)
    error->all(FLERR,"laser_path file: speeds and scale must be > 0, times >= 0");

  // Rank 0 reads rows of 5 values (v = 0 when absent) and broadcasts them.
  std::vector<double> rows;
  if (domain->me == 0) {
    FILE *fp = fopen(arg[0], "r");
    if (!fp) error->one(FLERR,"laser_path file: cannot open path file");
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
      if (char *c = strchr(line, '#')) *c = '\0';
      for (char *c = line; *c; c++) if (*c == ',') *c = ' ';
      double r[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
      const int n = sscanf(line, "%lf %lf %lf %lf %lf", r, r+1, r+2, r+3, r+4);
      if (n <= 0) continue;
      if (n < 4) error->one(FLERR,"laser_path file: a row needs x y z p [v]");
      rows.insert(rows.end(), r, r + 5);
    }
    fclose(fp);
  }
  int nr = (int) rows.size();
  MPI_Bcast(&nr, 1, MPI_INT, 0, world);
  rows.resize(nr);
  MPI_Bcast(rows.data(), nr, MPI_DOUBLE, 0, world);
  if (nr < 10) error->all(FLERR,"laser_path file: needs at least two points");

  path_layers.clear();
  double t = 0.0;
  auto P = [&](int r, int d) { return rows[5*r+d] * scale + shift[d]; };
  for (int r = 1; r < nr / 5; r++) {
    const double x0 = P(r-1,0), y0 = P(r-1,1), z0 = P(r-1,2);
    const double x1 = P(r,0), y1 = P(r,1), z1 = P(r,2);
    const double p = rows[5*r+3], vr = rows[5*r+4] > 0.0 ? rows[5*r+4] : v;
    if (path_layers.empty() || z1 != z0) {          // first point of a layer
      if (!path_layers.empty() && path_layers.back().segs.empty()) path_layers.pop_back();
      path_layers.push_back({z1, {}});
      t = 0.0;
      if (z1 != z0) continue;
    }
    const double d = std::sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
    if (p > 0.0) {
      if (d <= 0.0) continue;
      path_layers.back().segs.push_back({t, t + d / vr, x0, y0, x1, y1, p});
      t += d / vr;
    } else {
      // floor keeps a zero-time jump from putting two waypoints at one time
      t += std::max(jump_time > 0.0 ? jump_time : d / jump_speed, 1.0e-12);
    }
  }
  if (!path_layers.empty() && path_layers.back().segs.empty()) path_layers.pop_back();
  if (path_layers.empty()) error->all(FLERR,"laser_path file: no emitting move");
  if (nlayers > 0 && (size_t) nlayers < path_layers.size()) path_layers.resize(nlayers);

  laser_pause_constant = laser_pause_below = laser_reset_temperature = 0.0;
  pending_moser_passes = 0;
  path_recoat = recoat;
  laser_path_set = true;
  scan_layer_active = false;   // Rosenthal state; file paths are Moser-only
  if (domain->me == 0) {
    size_t nseg = 0;
    for (const auto &L : path_layers) nseg += L.segs.size();
    std::cout << "laser_path file " << arg[0] << ": " << path_layers.size()
              << " layers, " << nseg << " emitting segments, z "
              << path_layers.front().z << " .. " << path_layers.back().z
              << " m, recoat " << recoat << " s" << std::endl;
  }
  start_path_layer(0, host_->time);
}

void ThermalManager::start_path_layer(size_t k, double t_start)
{
  path_layer = k;
  const PathLayer &L = path_layers[k];
  scan_layer_z = L.z;
  moser_source()->build_scan_path(t_start, L.segs, L.z);
  if (domain->me == 0)
    std::cout << "laser_path file: layer " << k + 1 << " of " << path_layers.size()
              << " at z=" << L.z << " m starts at t=" << t_start << " s ("
              << L.segs.back().t1 << " s of scan)" << std::endl;
}

/* ----------------------------------------------------------------------
   laser_fluctuations command: in-source PSD form (Moser source only).
------------------------------------------------------------------------- */

void ThermalManager::laser_fluctuations_cmd(int narg, char **arg)
{
  if (narg < 1)
    error->all(FLERR,"Illegal laser_fluctuations command: expected psd <shape> ...");

  if (!temperature_source) {
    error->all(FLERR,"laser_fluctuations: setup_temperature_source must be called first");
  }
  MoserTemperatureSource *moser =
    moser_source();
  if (!moser) {
    error->all(FLERR,"laser_fluctuations: only the Moser unsteady Green's-function source supports fluctuations");
  }

  // The fluctuation table is materialized inside build_scan (called from
  // laser_path_cmd). Reject the reversed order explicitly.
  if (moser->has_scan()) {
    error->all(FLERR,
      "laser_fluctuations must be issued BEFORE laser_path "
      "(the fluctuation table is materialized inside laser_path once the "
      "scan duration is known). Correct sequence: "
      "setup_temperature_source moser ... ; laser_fluctuations psd ... ; "
      "laser_path start ... . If you intend to change fluctuations after "
      "the scan was built, re-issue laser_path so build_scan reruns.");
  }

  if (strcmp(arg[0],"psd") != 0)
    error->all(FLERR,"laser_fluctuations: expected 'psd' subcommand");
  if (narg < 2)
    error->all(FLERR,"laser_fluctuations psd: missing <shape>");

  // Optional top|bot|both selector between `psd` and the shape token.
  enum class Selector { NONE, TOP, BOT, BOTH };
  Selector which = Selector::NONE;
  int shape_idx = 1;
  if      (strcmp(arg[1],"top")  == 0) { which = Selector::TOP;  shape_idx = 2; }
  else if (strcmp(arg[1],"bot")  == 0) { which = Selector::BOT;  shape_idx = 2; }
  else if (strcmp(arg[1],"both") == 0) { which = Selector::BOTH; shape_idx = 2; }

  const bool is_keyhole = (moser->get_mode() == MoserTemperatureSource::MoserMode::KEYHOLE);
  if (is_keyhole && which == Selector::NONE)
    error->all(FLERR,
      "laser_fluctuations psd: keyhole mode requires top|bot|both selector");
  if (!is_keyhole && which != Selector::NONE)
    error->all(FLERR,
      "laser_fluctuations psd: top|bot|both selector is only valid in keyhole mode");

  if (shape_idx >= narg)
    error->all(FLERR,"laser_fluctuations psd: missing <shape>");

  MoserTemperatureSource::PsdSpec spec;
  spec.sigma_W = 0.0;
  spec.sigma_D = 0.0;
  spec.sigma_P = 0.0;
  spec.rho     = 0.0;
  spec.seed    = 12345UL;
  spec.dt_psd  = 5.0e-6;
  spec.tau     = 0.0;
  spec.f0      = 0.0;
  spec.df      = 0.0;

  if      (strcmp(arg[shape_idx],"white")       == 0) spec.shape = MoserTemperatureSource::PsdShape::WHITE;
  else if (strcmp(arg[shape_idx],"lorentzian")  == 0) spec.shape = MoserTemperatureSource::PsdShape::LORENTZIAN;
  else if (strcmp(arg[shape_idx],"pink")        == 0) spec.shape = MoserTemperatureSource::PsdShape::PINK;
  else if (strcmp(arg[shape_idx],"narrow_band") == 0) spec.shape = MoserTemperatureSource::PsdShape::NARROW_BAND;
  else error->all(FLERR,"laser_fluctuations psd: unknown shape (use white|lorentzian|pink|narrow_band)");

  int i = shape_idx + 1;
  while (i < narg) {
    const char *k = arg[i];
    if (i + 1 >= narg)
      error->all(FLERR,"laser_fluctuations psd: missing value for keyword");
    const double val = atof(arg[i+1]);
    if      (strcmp(k,"sigma_W") == 0) spec.sigma_W = val;
    else if (strcmp(k,"sigma_D") == 0) spec.sigma_D = val;
    else if (strcmp(k,"sigma_P") == 0) spec.sigma_P = val;
    else if (strcmp(k,"rho")     == 0) spec.rho     = val;
    else if (strcmp(k,"seed")    == 0) spec.seed    = static_cast<unsigned long>(atol(arg[i+1]));
    else if (strcmp(k,"dt")      == 0) spec.dt_psd  = val;
    else if (strcmp(k,"tau")     == 0) spec.tau     = val;
    else if (strcmp(k,"f0")      == 0) spec.f0      = val;
    else if (strcmp(k,"df")      == 0) spec.df      = val;
    else error->all(FLERR,"laser_fluctuations psd: unknown keyword");
    i += 2;
  }

  // Dispatch to the appropriate lobe(s).
  if (which == Selector::NONE || which == Selector::TOP) {
    moser->set_psd_spec(MoserTemperatureSource::LOBE_TOP, spec);
  }
  if (which == Selector::BOT) {
    moser->set_psd_spec(MoserTemperatureSource::LOBE_BOT, spec);
  }
  if (which == Selector::BOTH) {
    moser->set_psd_spec(MoserTemperatureSource::LOBE_TOP, spec);
    // Derive bot's seed from top's so the two streams are independent yet
    // identically reproducible across MPI ranks.
    MoserTemperatureSource::PsdSpec bot_spec = spec;
    bot_spec.seed = spec.seed ^ 0x9E3779B97F4A7C15ULL;
    moser->set_psd_spec(MoserTemperatureSource::LOBE_BOT, bot_spec);
  }

  if (domain->me == 0) {
    std::cout << "laser_fluctuations psd:\n";
    if (which != Selector::BOT) moser->print_lobe_info(MoserTemperatureSource::LOBE_TOP);
    if (which == Selector::BOT || which == Selector::BOTH)
      moser->print_lobe_info(MoserTemperatureSource::LOBE_BOT);
    std::cout.flush();
  }
}

/* ----------------------------------------------------------------------
   fast_forward_search_window command setter.
------------------------------------------------------------------------- */

void ThermalManager::set_fast_forward_search_window(double w)
{
  fast_forward_search_window = w;
  if (fast_forward_search_window <= 0.0)
    error->all(FLERR,"Illegal fast_forward_search_window value: must be > 0");
}

/* ----------------------------------------------------------------------
   temperature_smooth command: configure solidification-band smoothing.
------------------------------------------------------------------------- */

void ThermalManager::smooth_cmd(int narg, char **arg)
{
  // Usage: temperature_smooth <tmin> <tmax> <sigma_xy> <sigma_z>
  //                           [alpha] [guard] [passes] [diag_interval]
  if (narg < 4 || narg > 8)
    error->all(FLERR,"Illegal temperature_smooth command");
  smooth_tmin     = atof(arg[0]);
  smooth_tmax     = atof(arg[1]);
  smooth_sigma_xy = atof(arg[2]);
  smooth_sigma_z  = atof(arg[3]);
  if (narg >= 5) smooth_alpha         = atof(arg[4]);
  if (narg >= 6) smooth_guard         = atof(arg[5]);
  if (narg >= 7) smooth_passes        = atoi(arg[6]);
  if (narg >= 8) smooth_diag_interval = atoi(arg[7]);
  if (smooth_tmax <= smooth_tmin)
    error->all(FLERR,"temperature_smooth: tmax must be > tmin");
  if (smooth_sigma_xy <= 0.0)
    error->all(FLERR,"temperature_smooth: sigma_xy must be > 0");
  if (smooth_sigma_z <= 0.0)
    error->all(FLERR,"temperature_smooth: sigma_z must be > 0");
  if (smooth_alpha < 0.0 || smooth_alpha > 1.0)
    error->all(FLERR,"temperature_smooth: alpha must be in [0,1]");
  if (smooth_guard < 0.0)
    error->all(FLERR,"temperature_smooth: guard must be >= 0");
  if (smooth_passes < 1)
    error->all(FLERR,"temperature_smooth: passes must be >= 1");
  if (smooth_diag_interval < 0)
    error->all(FLERR,"temperature_smooth: diag_interval must be >= 0");
  temperature_smooth_enabled = true;
  if (domain->me == 0)
    fprintf(screen,
      "Temperature smoothing enabled: window=[%.1f,%.1f] K, "
      "guard=%.1f K, sigma_xy=%.2f sigma_z=%.2f sites, "
      "alpha=%.2f, passes=%d, diag_interval=%d\n",
      smooth_tmin, smooth_tmax, smooth_guard,
      smooth_sigma_xy, smooth_sigma_z,
      smooth_alpha, smooth_passes, smooth_diag_interval);
}

/* ----------------------------------------------------------------------
   nodal_smooth command: forward mesh-Laplacian smoothing to the source.
------------------------------------------------------------------------- */

void ThermalManager::nodal_smooth_cmd(int narg, char **arg)
{
  // Usage: nodal_smooth <sigma_xy> <sigma_z> <passes> <alpha>
  if (narg != 4)
    error->all(FLERR,"Illegal nodal_smooth command");
  if (!temperature_source)
    error->all(FLERR,"nodal_smooth requires setup_temperature_source first");
  const double sxy = atof(arg[0]);
  const double sz  = atof(arg[1]);
  const int    K   = atoi(arg[2]);
  const double al  = atof(arg[3]);
  if (sxy <= 0.0) error->all(FLERR,"nodal_smooth: sigma_xy must be > 0");
  if (sz  <= 0.0) error->all(FLERR,"nodal_smooth: sigma_z must be > 0");
  if (K    < 1)   error->all(FLERR,"nodal_smooth: passes must be >= 1");
  if (al < 0.0 || al > 1.0)
    error->all(FLERR,"nodal_smooth: alpha must be in [0,1]");
  temperature_source->enable_nodal_smoothing(sxy, sz, K, al);
  if (domain->me == 0)
    fprintf(screen,
      "Nodal mesh-Laplacian smoothing enabled: "
      "sigma_xy=%.3f sigma_z=%.3f passes=%d alpha=%.3f\n",
      sxy, sz, K, al);
}

/* ----------------------------------------------------------------------
   Fill app->T from the modular temperature source at simulation_time.
------------------------------------------------------------------------- */

void ThermalManager::update_temperatures(double simulation_time)
{
  timer->stamp();

  if (!temperature_source) {
    timer->stamp(TIME_APP);
    return; // No temperature source configured
  }

  double *T = host_->T;
  double **xyz = host_->xyz;
  const int nlocal = host_->nlocal;
  const double dx = host_->dx;

  // Cache the dynamic_cast results outside the loop.
  HDF5UnstructuredTemperatureSource* hdf5_source = this->hdf5_source();
  RosenthalTemperatureSource* ros_source = hdf5_source ? nullptr : rosenthal_source();

  // -------------------------------------------------------------------
  // HDF5 path: per-site element cache + nodal precompute.
  // -------------------------------------------------------------------
  if (hdf5_source) {
    hdf5_source->prepare_for_timestep(simulation_time);
    for (int i = 0; i < nlocal; i++) {
      T[i] = hdf5_source->get_temperature_at_site_fast(i);
    }

  // -------------------------------------------------------------------
  // Rosenthal path: analytical evaluation in pool-local frame.
  // -------------------------------------------------------------------
  } else if (ros_source) {
    if (!laser_path_set) {
      error->all(FLERR,"Rosenthal source selected but no laser_path defined");
    }

    // Sync the laser pose to simulation_time, advancing in small dt-sized
    // chunks (move(dt) does not carry remainder across path boundaries).
    if (scan_layer_active) {
      double advance = simulation_time - scan_layer_time;
      const double step = (host_->dt > 0.0) ? host_->dt : 1.0e-6;
      while (advance > step) {
        if (!scan_layer.move(step)) { scan_layer_active = false; break; }
        advance -= step;
      }
      if (scan_layer_active && advance > 0.0) {
        if (!scan_layer.move(advance)) scan_layer_active = false;
      }
    }
    scan_layer_time = simulation_time;

    const double v_laser = scan_layer.get_speed();
    // Use the source's own preheat/ambient (Rosenthal T0), not the app's
    // t_room -- they may differ.
    const double T0 = ros_source->get_ambient_temperature();
    for (int i = 0; i < nlocal; i++) {
      double phys[3] = { xyz[i][0]*dx, xyz[i][1]*dx, xyz[i][2]*dx };
      RASTER::Point loc =
        scan_layer.compute_position_relative_to_pool(phys, scan_layer_z);
      T[i] = scan_layer_active
               ? ros_source->rosenthal_pointwise(loc[0], loc[1], loc[2], v_laser, T0)
               : T0;
    }

  // -------------------------------------------------------------------
  // Generic xyz fallback for any other future source type.
  // -------------------------------------------------------------------
  } else {
    for (int i = 0; i < nlocal; i++) {
      double x_physical = xyz[i][0] * dx;
      double y_physical = xyz[i][1] * dx;
      double z_physical = xyz[i][2] * dx;
      T[i] = temperature_source->get_temperature_at_xyz_and_time(x_physical, y_physical, z_physical, simulation_time);
    }
  }

  // Optional Gaussian smoothing of the solidification band.
  if (temperature_smooth_enabled) apply_smoothing();

  timer->stamp(TIME_APP);
}

/* ----------------------------------------------------------------------
   One step of thermal orchestration: update T, run Moser pause/reset and
   pause_below dynamic-fire logic, then the fast-forward predictor. Returns
   the decision the app applies (advance time / activate powder / end run).
------------------------------------------------------------------------- */

ThermalManager::ThermalStep ThermalManager::advance(double current_time, double stoptime)
{
  // Moser: the per-pass pause/reset control flow is shared with the device
  // path; the host refill fills T[] and reduces its peak.
  if (moser_source()) {
    return moser_advance(current_time, stoptime, [this](double t) {
      update_temperatures(t);
      double local_max = 0.0;
      for (int i = 0; i < host_->nlocal; i++)
        local_max = std::max(local_max, host_->T[i]);
      double gmax;
      MPI_Allreduce(&local_max, &gmax, 1, MPI_DOUBLE, MPI_MAX, world);
      return gmax;
    });
  }

  ThermalStep step{current_time, false, false};

  // Update temperature first to get current temperatures
  update_temperatures(current_time);

  // Active-site threshold: HDF5 sources expose their own; Rosenthal uses the
  // liquidus tl.
  HDF5UnstructuredTemperatureSource* hdf5_source = this->hdf5_source();
  RosenthalTemperatureSource* ros_source = hdf5_source ? nullptr : rosenthal_source();
  double fast_forward_threshold = host_->t_room;
  if (hdf5_source) fast_forward_threshold = hdf5_source->get_fast_forward_threshold();
  else if (ros_source) fast_forward_threshold = host_->tl;

  int t_active = 0;
  for (int i = 0; i < host_->nlocal; i++) {
    if (host_->T[i] > fast_forward_threshold) t_active = 1;
  }

  // Use MPI_Allreduce to check if t_active is 0 on all processors
  int global_t_active;
  MPI_Allreduce(&t_active, &global_t_active, 1, MPI_INT, MPI_SUM, world);

  // If no active sites, fast forward.
  if (global_t_active == 0 && current_time > 1e-6) {
    const double local_next_time = ros_source
      ? rosenthal_next_active_time(current_time, fast_forward_threshold)
      : temperature_source->get_next_time_with_temperature(current_time, fast_forward_threshold);
    if (predict_skip(local_next_time, current_time, stoptime, fast_forward_threshold,
                     "host", step))
      update_temperatures(step.new_time);
  }

  return step;
}

bool ThermalManager::predict_skip(double local_next, double current_time,
                                  double stoptime, double threshold,
                                  const char *tag, ThermalStep &step)
{
  // Earliest wake-up across the entire MPI communicator is the single time we
  // may safely advance to.
  double next_thermal_time;
  MPI_Allreduce(&local_next, &next_thermal_time, 1, MPI_DOUBLE, MPI_MIN, world);
  if (!(next_thermal_time > current_time &&
        next_thermal_time < std::numeric_limits<double>::max()))
    return false;

  // Skip to whichever comes first: next thermal activity or run end.
  const double target_time = std::min(next_thermal_time, stoptime);
  const double time_skip = target_time - current_time;
  if (time_skip <= 1e-6) return false;   // only a meaningful skip

  if (domain->me == 0) {
    std::cout << "Fast-forward (" << tag << "): skipping " << time_skip
              << " s from t=" << current_time << " to " << target_time
              << " (all T < " << threshold << " K; next thermal activity at "
              << next_thermal_time << ")" << std::endl;
  }
  step.new_time = target_time;
  // Activate powder after a significant skip (new layer deposition).
  if (time_skip > 1.0) step.activate_powder = true;
  return true;
}

/* ----------------------------------------------------------------------
   Build a RosenthalDevicePlan for the current step: advance the laser scan
   path to simulation_time on the host and snapshot the laser pose + material
   parameters so a device kernel can evaluate the temperature per site. See
   rosenthal_device.h. The scan-path advance mirrors update_temperatures()'s
   Rosenthal branch exactly, so the device fill matches the host fill.
------------------------------------------------------------------------- */

RosenthalDevicePlan ThermalManager::rosenthal_device_plan(double simulation_time)
{
  RosenthalDevicePlan plan;

  RosenthalTemperatureSource* ros =
    rosenthal_source();
  if (!ros || !laser_path_set) return plan;  // plan.valid stays false

  // Solidification-band smoothing is applied in update_temperatures(), which the
  // device fill bypasses; keep smoothed runs on the host path.
  if (temperature_smooth_enabled) return plan;

  // Material parameters; false (e.g. KEYHOLE) -> host path.
  if (!ros->fill_device_plan(plan)) { plan.valid = false; return plan; }

  // Sync the laser pose to simulation_time in dt-sized chunks (move(dt) does
  // not carry remainder across path boundaries) -- identical to the host loop.
  if (scan_layer_active) {
    double advance = simulation_time - scan_layer_time;
    const double step = (host_->dt > 0.0) ? host_->dt : 1.0e-6;
    while (advance > step) {
      if (!scan_layer.move(step)) { scan_layer_active = false; break; }
      advance -= step;
    }
    if (scan_layer_active && advance > 0.0) {
      if (!scan_layer.move(advance)) scan_layer_active = false;
    }
  }
  scan_layer_time = simulation_time;

  const RASTER::Point pos = scan_layer.get_position();
  const RASTER::Point dir = scan_layer.get_unit_dir();
  plan.px = pos[0];
  plan.py = pos[1];
  plan.layer_z = scan_layer_z;
  plan.ux = dir[0];
  plan.uy = dir[1];
  plan.v = scan_layer.get_speed();
  plan.scan_active = scan_layer_active;
  plan.T0 = ros->get_ambient_temperature();
  plan.valid = true;
  return plan;
}

/* ----------------------------------------------------------------------
   Fast-forward decision for the Rosenthal device path. Same logic as the
   Rosenthal branch of advance(), but the active-site test uses the device-
   computed global max temperature instead of a host T scan, and temperatures
   are NOT refilled here (the caller refills its device view at new_time).
------------------------------------------------------------------------- */

ThermalManager::ThermalStep
ThermalManager::rosenthal_fast_forward(double current_time, double stoptime,
                                       double global_max_T)
{
  ThermalStep step;
  step.new_time = current_time;
  step.activate_powder = false;
  step.end_run = false;

  RosenthalTemperatureSource* ros =
    rosenthal_source();
  if (!ros) return step;

  // `laser_path ... pause_below <Tk>` overrides the default liquidus threshold.
  const double fast_forward_threshold =
    (laser_pause_below > 0.0) ? laser_pause_below : host_->tl;

  // global_max_T is already reduced across ranks, so this test is consistent
  // on every rank -> all ranks take the same branch (keeps the MIN reduce
  // below collective-safe).
  const bool any_active = (global_max_T > fast_forward_threshold);

  if (!any_active && current_time > 1e-6) {
    predict_skip(rosenthal_next_active_time(current_time, fast_forward_threshold),
                 current_time, stoptime, fast_forward_threshold, "device", step);
  }
  return step;
}

/* ----------------------------------------------------------------------
   Device (Kokkos) Moser path: build a MoserDevicePlan for the current step.
   Unlike Rosenthal there is no scan-path pose to advance (the Moser source
   integrates over its own absolute-time path history); we just let the source
   do any per-step bookkeeping and snapshot its scalar parameters. Returns an
   invalid plan (host path) for non-Moser sources, KEYHOLE / fluctuations, or
   when a dynamic inter-pass pause/reset is configured.
------------------------------------------------------------------------- */

MoserDevicePlan ThermalManager::moser_device_plan(double simulation_time)
{
  MoserDevicePlan plan;

  MoserTemperatureSource* moser =
    moser_source();
  if (!moser) return plan;  // plan.valid stays false

  // Solidification-band smoothing is applied in update_temperatures(), which the
  // device fill bypasses; keep smoothed runs on the host path. (Dynamic
  // inter-pass pause/reset is handled on the device via moser_advance_device.)
  if (temperature_smooth_enabled) return plan;

  if (!moser->fill_device_plan(plan)) { plan.valid = false; return plan; }
  plan.time = simulation_time;
  return plan;
}

/* ----------------------------------------------------------------------
   Moser driver: time-gated pause+reset firing, pause_below dynamic firing /
   clean exit, and the fast-forward predictor, shared by the host path
   (advance(), refill = update_temperatures + max scan) and the device path
   (moser_advance_device(), refill = the registered device filler). The
   pause/reset firings (reset_history_and_start / append_pass) rebuild the
   scan from temperature feedback on the host; each is followed by a refill.
   The static (no-pause) case falls straight through to the predictor. The
   global max is MPI-reduced, so the active-site test is consistent on every
   rank and all ranks make the same collective calls.
------------------------------------------------------------------------- */

ThermalManager::ThermalStep
ThermalManager::moser_advance_device(double current_time, double stoptime)
{
  if (!moser_source() || !moser_device_filler_)
    return ThermalStep{current_time, false, false};
  if (temperature_smooth_enabled && conduction_source())
    error->all(FLERR,"temperature_smooth is not supported with the conduction source");
  return moser_advance(current_time, stoptime, moser_device_filler_);
}

ThermalManager::ThermalStep
ThermalManager::moser_advance(double current_time, double stoptime,
                              const std::function<double(double)> &refill)
{
  ThermalStep step{current_time, false, false};
  MoserTemperatureSource* moser = moser_source();

  // Fill the field at current_time and get the global peak temperature.
  double gmax = refill(current_time);

  // Moser pause + reset_temperature: time-gated dynamic fire (fires on elapsed
  // time since the last pass end, independent of temperature).
  if (laser_pause_constant > 0.0 && laser_reset_temperature > 0.0 &&
      pending_moser_passes > 0) {
    const double next_fire =
      moser->get_last_pass_end_time() + laser_pause_constant;
    if (current_time >= next_fire) {
      moser->reset_history_and_start(current_time, laser_reset_temperature);
      --pending_moser_passes;
      if (domain->me == 0) {
        std::cout << "pause+reset: gap of " << laser_pause_constant
                  << " s elapsed at t=" << current_time
                  << " s; firing next Moser pass (reset to "
                  << laser_reset_temperature << " K, "
                  << pending_moser_passes << " more queued)" << std::endl;
      }
      gmax = refill(current_time);   // scan changed -> refill
    }
  }

  const double fast_forward_threshold =
    (laser_pause_below > 0.0) ? laser_pause_below : host_->tl;

  if (gmax <= fast_forward_threshold && current_time > 1e-6) {
    bool dynamic_fire = false;

    // pause_below: fire a pending pass once the peak temperature drops below
    // threshold AFTER the current pass has finished emitting.
    if (laser_pause_below > 0.0 && pending_moser_passes > 0) {
      if (current_time >= moser->get_last_pass_end_time()) {
        if (laser_reset_temperature > 0.0)
          moser->reset_history_and_start(current_time, laser_reset_temperature);
        else
          moser->append_pass(current_time);
        --pending_moser_passes;
        dynamic_fire = true;
        if (domain->me == 0) {
          std::cout << "pause_below: max(T) < " << fast_forward_threshold
                    << " K reached at t=" << current_time
                    << " s; firing next Moser pass (";
          if (laser_reset_temperature > 0.0)
            std::cout << "reset to " << laser_reset_temperature << " K, ";
          std::cout << pending_moser_passes << " more queued)" << std::endl;
        }
        gmax = refill(current_time);   // scan changed -> refill
      }
    }

    // Layer finished: the last segment has emitted, the field is below the
    // solidus and no site is liquid (any_liquid is collective; every
    // condition before it is rank-uniform). A path file starts its next
    // layer after the recoat time, from a fresh field (Moser: new history;
    // conduction: the new plane rebuilds the grid at T0), as am/finitediff.
    // Conduction with no layer left resets to ambient and skips to stoptime.
    ConductionTemperatureSource *cond = conduction_source();
    const bool more_layers = path_layer + 1 < path_layers.size();
    if ((cond || more_layers) && !dynamic_fire && pending_moser_passes == 0 &&
        current_time >= moser->get_last_pass_end_time() &&
        (!cond || cond->next_laser_on(current_time) == std::numeric_limits<double>::max()) &&
        gmax < host_->ts && current_time < stoptime &&
        !(host_->any_liquid && host_->any_liquid())) {
      if (more_layers) {
        const double t_next = current_time + path_recoat;
        start_path_layer(path_layer + 1, t_next);
        step.new_time = std::min(t_next, stoptime);
      } else {
        if (domain->me == 0)
          std::cout << "Fast-forward (conduction): layer done, max(T) < "
                    << host_->ts << " K and no liquid at t=" << current_time
                    << " s; field reset to " << cond->ambient()
                    << " K, skipping to stoptime=" << stoptime << " s" << std::endl;
        cond->request_reset(cond->ambient());
        step.new_time = stoptime;
      }
      refill(step.new_time);
      return step;
    }

    // pause_below: once the FINAL pass has finished and the peak has dropped
    // below threshold, advance to stoptime for a clean early exit.
    if (!dynamic_fire && laser_pause_below > 0.0 &&
        pending_moser_passes == 0 &&
        current_time >= moser->get_last_pass_end_time() &&
        current_time < stoptime) {
      if (domain->me == 0) {
        std::cout << "pause_below: final pass complete and max(T) < "
                  << fast_forward_threshold << " K at t=" << current_time
                  << " s; advancing to stoptime=" << stoptime
                  << " s for clean exit." << std::endl;
      }
      step.new_time = stoptime;
      step.end_run = true;
      return step;
    }

    // Fast-forward predict-skip (also the sole behavior in the static case).
    // The conduction source skips to the next laser-on time; its refill
    // integrates the gap (RKL2 super-steps), so only the app steps are saved.
    const double next = cond ? cond->next_laser_on(current_time)
                             : moser_next_active_time(current_time, fast_forward_threshold);
    if (!dynamic_fire &&
        predict_skip(next, current_time, stoptime, fast_forward_threshold,
                     cond ? "conduction" : "Moser", step))
      refill(step.new_time);   // refill at the advanced time
  }
  return step;
}

/* ----------------------------------------------------------------------
   Device (Kokkos) HDF5 unstructured-mesh path. The expensive host work --
   HDF5 reads, layer load, per-site element search -- stays on the host; only
   the per-site barycentric interpolation moves to the device. See hdf5_device.h.
------------------------------------------------------------------------- */

bool ThermalManager::hdf5_device_eligible() const
{
  if (!temperature_source) return false;
  // Solidification-band smoothing is applied in update_temperatures(), which the
  // device fill bypasses; keep smoothed runs on the host path.
  if (temperature_smooth_enabled) return false;
  return dynamic_cast<HDF5UnstructuredTemperatureSource*>(
             temperature_source.get()) != nullptr;
}

HDF5DevicePlan ThermalManager::hdf5_device_plan(double simulation_time)
{
  HDF5DevicePlan plan;

  HDF5UnstructuredTemperatureSource* hdf5 =
    hdf5_source();
  if (!hdf5 || temperature_smooth_enabled) return plan;  // plan.valid stays false

  // Per-step host prep: advance source state, load the layer if needed, rebuild
  // the per-site element cache at a layer boundary (bumps cache_generation), and
  // precompute the nodal-temperature slice (time interpolation + optional nodal
  // mesh smoothing). This is exactly the host fast path (update_temperatures
  // Path 3) minus the per-site T loop -- that loop is what moves to the device.
  hdf5->prepare_for_timestep(simulation_time);

  plan.version = hdf5->cache_generation();
  plan.nsites = hdf5->num_cache_sites();   // nall (owned + ghost); fills positionally
  plan.n_nodes = hdf5->num_loaded_nodes();
  plan.default_temp = hdf5->get_default_temp();
  plan.sim_time = simulation_time;          // device-side time interpolation
  plan.valid = true;
  return plan;
}

RosenthalTemperatureSource *ThermalManager::rosenthal_source() const
{
  return dynamic_cast<RosenthalTemperatureSource*>(temperature_source.get());
}

MoserTemperatureSource *ThermalManager::moser_source() const
{
  return dynamic_cast<MoserTemperatureSource*>(temperature_source.get());
}

HDF5UnstructuredTemperatureSource *ThermalManager::hdf5_source() const
{
  return dynamic_cast<HDF5UnstructuredTemperatureSource*>(temperature_source.get());
}

ConductionTemperatureSource *ThermalManager::conduction_source() const
{
  return dynamic_cast<ConductionTemperatureSource*>(temperature_source.get());
}

/* ----------------------------------------------------------------------
   Device HDF5 driver. Mirrors the HDF5 branch of advance() -- the global
   active-site reduction (here via the device-computed global max) and the
   generic fast-forward predictor (get_next_time_with_temperature) -- but
   replaces the host per-site fill + host-temperature scan with the registered
   device filler. HDF5 has no pause/reset, so this is fill + predict-skip. The
   global max is already MPI-reduced, so the active-site test is consistent on
   every rank and all ranks make the same collective calls.
------------------------------------------------------------------------- */

ThermalManager::ThermalStep
ThermalManager::hdf5_advance_device(double current_time, double stoptime)
{
  ThermalStep step;
  step.new_time = current_time;
  step.activate_powder = false;
  step.end_run = false;

  HDF5UnstructuredTemperatureSource* hdf5 =
    hdf5_source();
  if (!hdf5 || !hdf5_device_filler_) return step;

  // Fill the device field at current_time and get the global peak temperature.
  double gmax = hdf5_device_filler_(current_time);

  const double fast_forward_threshold = hdf5->get_fast_forward_threshold();

  if (gmax <= fast_forward_threshold && current_time > 1e-6 &&
      predict_skip(hdf5->get_next_time_with_temperature(current_time, fast_forward_threshold),
                   current_time, stoptime, fast_forward_threshold, "device HDF5", step))
    hdf5_device_filler_(step.new_time);   // refill at the advanced time
  return step;
}

/* ----------------------------------------------------------------------
   Run the isotherm-compactness diagnostic on the configured cadence.
------------------------------------------------------------------------- */

void ThermalManager::maybe_run_diagnostics(int step_count)
{
  if (temperature_smooth_enabled && smooth_diag_interval > 0 &&
      step_count % smooth_diag_interval == 0) {
    compute_diagnostics();
  }
}

/* ----------------------------------------------------------------------
   Fast-forward predictor for the Rosenthal source. Walks a copy of
   scan_layer forward and returns the earliest time the conservative upper
   bound on the Rosenthal temperature over the local domain reaches
   threshold_temp.
------------------------------------------------------------------------- */

double ThermalManager::rosenthal_next_active_time(double current_time, double threshold_temp)
{
  if (!laser_path_set || !scan_layer_active)
    return std::numeric_limits<double>::max();

  auto* ros = rosenthal_source();
  if (!ros) return std::numeric_limits<double>::max();

  double lo[3], hi[3];
  if (!local_bbox(lo, hi)) return std::numeric_limits<double>::max();

  const double T0 = ros->get_ambient_temperature();
  // Cheap early-out: if even the unmodulated peak at r_min can't reach the
  // threshold, no laser position ever will.
  {
    const double T_max_possible = ros->rosenthal_peak_at_distance(0.0, T0);
    if (T_max_possible < threshold_temp) {
      return std::numeric_limits<double>::max();
    }
  }

  // Walk a copy of the layer forward in time using the search window.
  RASTER::Layer probe = scan_layer;
  const double ddt = (fast_forward_search_window > 0.0)
                       ? fast_forward_search_window : 0.01;
  // Cap the look-ahead so we don't spin forever on a never-active path.
  const double max_lookahead = 3600.0; // 1 hour of sim time
  double t = current_time;
  const double t_stop = current_time + max_lookahead;

  while (t < t_stop) {
    // Closest point on the local box to the current laser position.
    const RASTER::Point laser = probe.get_position();
    const double dxr = std::clamp(laser[0], lo[0], hi[0]) - laser[0];
    const double dyr = std::clamp(laser[1], lo[1], hi[1]) - laser[1];
    const double dzr = std::clamp(scan_layer_z, lo[2], hi[2]) - scan_layer_z;
    const double R_min_box = std::sqrt(dxr*dxr + dyr*dyr + dzr*dzr);

    const double T_upper = ros->rosenthal_peak_at_distance(R_min_box, T0);
    if (T_upper >= threshold_temp) return t;

    // Advance the probe by ddt; stop if path is exhausted.
    if (!probe.move(ddt)) return std::numeric_limits<double>::max();
    t += ddt;
  }
  return std::numeric_limits<double>::max();
}

/* ----------------------------------------------------------------------
   Fast-forward predictor for the Moser source. Walks each remaining active
   scan interval forward, evaluating the Moser integrator at the closest
   bounding-box point, and returns the first time it reaches threshold_temp.
------------------------------------------------------------------------- */

double ThermalManager::moser_next_active_time(double current_time, double threshold_temp)
{
  if (!laser_path_set) return std::numeric_limits<double>::max();

  auto* moser = moser_source();
  if (!moser || !moser->has_scan()) return std::numeric_limits<double>::max();

  double lo[3], hi[3];
  if (!local_bbox(lo, hi)) return std::numeric_limits<double>::max();

  const double laser_z = moser->get_scan_laser_plane_z();
  const double ddt = (fast_forward_search_window > 0.0)
                       ? fast_forward_search_window : 0.01;
  for (const auto &g : moser->segments()) {
    if (g.t1 <= current_time || g.p <= 0.0) continue;  // done or dark
    double t = std::max(current_time, g.t0);
    while (t <= g.t1) {
      // Lerp laser (x,y) along this segment.
      const double s = (t - g.t0) / (g.t1 - g.t0);
      // Closest point on local box to laser.
      const double cx = std::clamp(g.x0 + s * (g.x1 - g.x0), lo[0], hi[0]);
      const double cy = std::clamp(g.y0 + s * (g.y1 - g.y0), lo[1], hi[1]);
      const double cz = std::clamp(laser_z, lo[2], hi[2]);
      const double T_probe = moser->get_temperature_at_xyz_and_time(cx, cy, cz, t);
      if (T_probe >= threshold_temp) return t;
      t += ddt;
    }
  }
  return std::numeric_limits<double>::max();
}

bool ThermalManager::local_bbox(double lo[3], double hi[3]) const
{
  const int nlocal = host_->nlocal;
  if (nlocal == 0) return false;
  double **xyz = host_->xyz;
  const double dx = host_->dx;
  for (int d = 0; d < 3; ++d) lo[d] = hi[d] = xyz[0][d] * dx;
  for (int i = 1; i < nlocal; ++i)
    for (int d = 0; d < 3; ++d) {
      const double v = xyz[i][d] * dx;
      lo[d] = std::min(lo[d], v);
      hi[d] = std::max(hi[d], v);
    }
  return true;
}

/* ----------------------------------------------------------------------
   Gaussian smoothing of T[] inside the user-specified solidification
   window. Called from update_temperatures() when enabled.
------------------------------------------------------------------------- */

void ThermalManager::apply_smoothing()
{
  double *T = host_->T;
  double **xyz = host_->xyz;
  int *numneigh = host_->numneigh;
  int **neighbor = host_->neighbor;
  const int nlocal = host_->nlocal;

  const double two_sxy2 = 2.0 * smooth_sigma_xy * smooth_sigma_xy;
  const double two_sz2  = 2.0 * smooth_sigma_z  * smooth_sigma_z;
  const double lo_outer = smooth_tmin - smooth_guard;
  const double hi_outer = smooth_tmax + smooth_guard;
  const double ramp_lo = (smooth_guard > 0.0) ? smooth_guard : 1.0;
  const double ramp_hi = (smooth_guard > 0.0) ? smooth_guard : 1.0;

  if ((int)smooth_buffer.size() < nlocal) smooth_buffer.resize(nlocal);

  // Precompute Gaussian weights per 26-neighbor slot (uniform SC_26N lattice).
  constexpr int MAX_NBR = 26;
  double w_slot[MAX_NBR];
  bool have_weight_lut = false;
  for (int ref = 0; ref < nlocal && !have_weight_lut; ref++) {
    if (numneigh[ref] < MAX_NBR) continue;
    bool all_valid = true;
    for (int j = 0; j < MAX_NBR; j++) {
      if (neighbor[ref][j] < 0) { all_valid = false; break; }
    }
    if (!all_valid) continue;
    const double xr = xyz[ref][0];
    const double yr = xyz[ref][1];
    const double zr = xyz[ref][2];
    for (int j = 0; j < MAX_NBR; j++) {
      const int nj = neighbor[ref][j];
      const double dxn = xyz[nj][0] - xr;
      const double dyn = xyz[nj][1] - yr;
      const double dzn = xyz[nj][2] - zr;
      const double expo = (dxn*dxn + dyn*dyn) / two_sxy2 + dzn*dzn / two_sz2;
      w_slot[j] = exp(-expo);
    }
    have_weight_lut = true;
  }

  for (int pass = 0; pass < smooth_passes; pass++) {
    // Ghosts need current T before each pass.
    host_->comm->all_selective(host_->ghost_iindices, host_->nghost_iarray,
                              host_->ghost_dindices, host_->nghost_darray);

    for (int i = 0; i < nlocal; i++) {
      const double Ti = T[i];
      if (Ti < lo_outer || Ti > hi_outer) {
        smooth_buffer[i] = Ti;
        continue;
      }

      double wsum = 1.0;   // self
      double Tsum = Ti;    // self-contribution

      const int nn = numneigh[i];
      if (have_weight_lut) {
        for (int j = 0; j < nn; j++) {
          const int nj = neighbor[i][j];
          if (nj < 0) continue;
          const double Tj = T[nj];
          // Skip neighbors whose T is outside the smoothing window.
          if (Tj < lo_outer || Tj > hi_outer) continue;
          const double w = w_slot[j];
          wsum += w;
          Tsum += w * Tj;
        }
      } else {
        const double xi = xyz[i][0];
        const double yi = xyz[i][1];
        const double zi = xyz[i][2];
        for (int j = 0; j < nn; j++) {
          const int nj = neighbor[i][j];
          if (nj < 0) continue;
          const double Tj = T[nj];
          if (Tj < lo_outer || Tj > hi_outer) continue;
          const double dxn = xyz[nj][0] - xi;
          const double dyn = xyz[nj][1] - yi;
          const double dzn = xyz[nj][2] - zi;
          const double expo = (dxn*dxn + dyn*dyn) / two_sxy2 + dzn*dzn / two_sz2;
          const double w = exp(-expo);
          wsum += w;
          Tsum += w * Tj;
        }
      }
      const double Tavg = Tsum / wsum;

      // Guard-band taper so the blend factor ramps to zero at the outer edges.
      double aeff = smooth_alpha;
      if (Ti < smooth_tmin) {
        aeff *= (Ti - lo_outer) / ramp_lo;
      } else if (Ti > smooth_tmax) {
        aeff *= (hi_outer - Ti) / ramp_hi;
      }
      if (aeff < 0.0) aeff = 0.0;

      smooth_buffer[i] = (1.0 - aeff) * Ti + aeff * Tavg;
    }

    // Commit this pass before the next one.
    for (int i = 0; i < nlocal; i++) T[i] = smooth_buffer[i];
  }

  // Leave ghosts consistent with the final committed T.
  host_->comm->all_selective(host_->ghost_iindices, host_->nghost_iarray,
                            host_->ghost_dindices, host_->nghost_darray);
}

/* ----------------------------------------------------------------------
   In-situ smoothing diagnostic: isotherm compactness of the T >= liquidus
   region. Assumes T ghosts are current (caller must ensure this).
------------------------------------------------------------------------- */

void ThermalManager::compute_diagnostics()
{
  double *T = host_->T;
  int *numneigh = host_->numneigh;
  int **neighbor = host_->neighbor;
  const int nlocal = host_->nlocal;
  const double tl = host_->tl;

  bigint V_local = 0, S_local = 0;
  double sumT_local = 0.0;
  double maxT_local = -1.0e30;

  for (int i = 0; i < nlocal; i++) {
    const double Ti = T[i];
    if (Ti < tl) continue;

    V_local++;
    sumT_local += Ti;
    if (Ti > maxT_local) maxT_local = Ti;

    const int nn = numneigh[i];
    for (int j = 0; j < nn; j++) {
      const int nj = neighbor[i][j];
      if (nj < 0) continue;
      if (T[nj] < tl) { S_local++; break; }
    }
  }

  bigint V_global = 0, S_global = 0;
  double sumT_global = 0.0, maxT_global = 0.0;
  MPI_Allreduce(&V_local,    &V_global,    1, MPI_SPK_BIGINT, MPI_SUM, world);
  MPI_Allreduce(&S_local,    &S_global,    1, MPI_SPK_BIGINT, MPI_SUM, world);
  MPI_Allreduce(&sumT_local, &sumT_global, 1, MPI_DOUBLE,     MPI_SUM, world);
  MPI_Allreduce(&maxT_local, &maxT_global, 1, MPI_DOUBLE,     MPI_MAX, world);

  if (domain->me == 0) {
    const double ideal = cbrt(36.0 * MY_PI);  // ~= 4.8360
    fprintf(screen, "\n=== Smoothing diagnostic (T >= liquidus=%.1f K) ===\n", tl);
    if (V_global > 0) {
      const double V = static_cast<double>(V_global);
      const double ratio = static_cast<double>(S_global) / cbrt(V * V);  // V^(2/3)
      const double meanT = sumT_global / V;
      fprintf(screen,
        "  Pool volume:       " BIGINT_FORMAT " sites\n"
        "  Pool surface:      " BIGINT_FORMAT " sites\n"
        "  S/V^(2/3):         %.3f  (ideal 3D sphere = %.3f)\n"
        "  Pool mean T:       %.2f K   max T: %.2f K\n",
        V_global, S_global, ratio, ideal, meanT, maxT_global);
    } else {
      fprintf(screen, "  No sites above liquidus this step.\n");
    }
  }
}
