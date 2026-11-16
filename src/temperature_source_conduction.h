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

#ifndef SPK_TEMPERATURE_SOURCE_CONDUCTION_H
#define SPK_TEMPERATURE_SOURCE_CONDUCTION_H

#include "temperature_source_moser.h"
#include <string>
#include <vector>

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   Nonlinear heat conduction source (finite volume, enthalpy form).

     setup_temperature_source conduction <Q> <lambda> <k> <alpha> <T0> <cp>
                                         <sx> <sy> <sz> [keyword value ...]

   The nine positional values are Moser's STANDARD form: the same absorbed
   Gaussian (exp(-x^2/sx^2) per axis, into the half space below the laser
   plane) and the same laser_path pass scheduling, inherited unchanged. With
   constant properties and no losses the solution converges to Moser's.

   This class holds the configuration, the property tables and the laser
   pose; the state (enthalpy field on a graded grid) lives in the Kokkos
   solver (KOKKOS/conduction_solver_kokkos), so the source only runs under
   additive/texture/kk. It has no point query: get_temperature_at_xyz_and_time
   errors.
------------------------------------------------------------------------- */

class ConductionTemperatureSource : public MoserTemperatureSource {
 public:
  ConductionTemperatureSource(class SPPARKS *);

  void setup_temperature_source(const std::vector<std::string> &args) override;
  double get_temperature_at_xyz_and_time(double, double, double, double) override;
  void print_source_info() const override;
  void reset_history_and_start(double t_start, double new_ambient) override;

  enum FarBC { FAR_FIXED = 0, FAR_ADIABATIC = 1 };

  // Configuration (SI). Read by the solver.
  struct Config {
    std::vector<double> k_T, k_v;      // dense k(T) table (1 entry = constant)
    std::vector<double> cp_T, cp_v;    // cp(T) table (1 entry = constant)
    double latent = 0.0;               // J/kg, released over [ts, tl]
    double k_powder = 0.0;
    double h = 0.0, T_inf = 300.0;     // convection
    double emissivity = 0.0;
    // evaporation: Tb (K, at 1 atm), Lv (J/kg), M (kg/mol) of one species, or
    // of two: the composition's ends (x = comp_x0 and comp_x1). Empty: off.
    std::vector<double> evap;
    double evap_coef = 0.82;           // net fraction of the Hertz-Knudsen flux into vacuum
    // ambient pressure (Pa): the loss is zero until the liquid's vapour
    // pressure passes it (the surface boils); 0 = vacuum, the full rate
    double evap_ambient = 101325.0;
    double margin[4] = {0, 0, 0, 0};   // xlo xhi ylo yhi beyond the lattice box
    double plate = 0.0;                // grid depth below the lattice box
    double plate_top = -1e300;         // all cells below this z are dense
    bool   plate_top_set = false;
    double stretch = 1.25;
    double max_cell = 0.0;             // 0 = uncapped
    std::string part_region;           // empty = whole domain is part
    double layer_thickness = 0.0;      // powder depth outside the box
    double packing = 1.0;              // powder mass fraction of dense (1 = dense)
    int    far_bc = FAR_FIXED;
    // flow lbm: melt-pool flow (KOKKOS/melt_flow_kokkos.h) on the lattice
    // box's liquid cells; it carries the enthalpy (and, mix flow, d11).
    struct Flow {
      bool   on = false;
      double dsigma_dT = -2.6e-4;       // N/m/K (thermal Marangoni)
      double beta = 1.7e-4;             // 1/K, volumetric expansion of the liquid
      double viscosity = 8.5e-7;        // m^2/s, kinematic
      double mushy = 1.0e4;             // 1/s, Carman-Kozeny drag constant
      double u_ref = 1.0;               // m/s, expected speed (sets the first flow step)
      double smag = 0.17;               // Smagorinsky constant, 0 = off
      // properties cards (needs composition ... mix flow): surface tension,
      // liquid density and viscosity follow the composition x and T, from
      // the [flow] sections of the composition's cards; dsigma_dT, beta and
      // viscosity above are then not used. sigma_file: sigma(T, x) from a
      // table of tools/surface_tension.py instead of the cards' linear mix.
      bool cards = false;
      std::string sigma_file;
      // free_surface (needs ded, one rank): the pool's surface moves
      // (MeltFlow's fs). The grid reaches the top of the lattice, the laser
      // lands on each column's surface, and metal is wherever the liquid
      // went: sites follow it (i2 5 <-> 2). No cell is born; see powder.
      bool fs = false;
      double sigma = 1.5;               // N/m at the liquidus, without properties cards
      // powder (free_surface): mass rate (kg/s) blown onto the surface in a
      // Gaussian of this 1/e radius about the laser axis, at temperature T
      // (default T0); only what lands on liquid is caught.
      double powder = 0.0, powder_radius = 0.0, powder_T = -1.0;
      // Tables made at setup. sig[iT * nx + ix] (N/m) on uniform grids from
      // (T0, x0), steps (dT, dx); linear beyond the T range, held in x.
      // rho = r0 + r1 (T - Tr) and the dynamic viscosity eta at n points
      // uniform in x over [comp_x0, comp_x1].
      struct Props {
        int nT = 0, nx = 0, n = 0;
        double T0 = 0.0, dT = 1.0, x0 = 0.0, dx = 1.0, Tr = 0.0;
        std::vector<double> sig, r0, r1, eta;
      } props;
    } flow;
    bool   carry = false;               // a layer's grid rebuild keeps the field (else T0)
    int    sts_smax = 32;               // RKL2 stages per laser-off super-step; 0 = explicit
    bool   init_d6 = false;             // init d6: the run's first grid takes in-box T from d6
    std::string init_grid;              // init grid FILE: ... and every cell's T from a dump conduction file
    double far = 0.0;                   // m; laser farther than this from the box is far (0 = off)
    // One material's dense properties: k(T) and cp(T) tables, density,
    // latent heat released over [ts, tl].
    struct Material {
      std::vector<double> k_T, k_v, cp_T, cp_v;
      double rho = 0.0, latent = 0.0, ts = 0.0, tl = 0.0;
      std::string name;
    };
    // plate_material: the dense material of i2 = 6 sites and of the cells
    // below plate_top outside the box.
    struct Plate : Material { bool on = false; } plate_mat;
    // composition: the part's material varies with the site's composition
    // x (d11). comp holds bins at x uniform over [comp_x0, comp_x1], each
    // interpolated linearly between the cards at their nodes; a site takes
    // the nearest bin. Empty: one part material (the keywords above).
    // ponytail: cards at nodes stand in for a composition database file.
    std::vector<Material> comp;
    double comp_x0 = 0.0, comp_x1 = 1.0;
    // mix pool: every liquid site takes the mean x of all liquid sites, each
    // step (a well-mixed pool: dilution). none: x stays as loaded.
    bool comp_mix = true;
    // mix flow (needs flow lbm): x is carried by the melt-pool flow instead
    // (comp_mix stays true: x changes during the run)
    bool comp_flow = false;
    // feed surface (mix flow + ded): new material enters at the pool's top
    // instead of as a slab of fed composition (see flow_step)
    bool comp_feed_surface = false;
    double plate_growth = 0.0;          // i2 = 6 reaches plate_top + this (m)
    // ded: no powder bed. Non-part cells are gas; i2 = 0/1 cells are unborn
    // (gas) until the laser axis passes within ded_radius, then powder at T_birth.
    bool   ded = false;
    double ded_radius = -1.0;           // m; default 2 sx
    double T_birth = -1.0;              // K; default T0
    double ded_depth = -1.0;            // m below the source plane; default the layer step
  };
  const Config &config() const { return cfg; }

  double density() const { return rho; }
  double ambient() const { return T0_default; }
  double diffusivity() const { return alpha; }   // at T0 [m^2/s]
  double absorbed_power() const { return lambda * Q; }
  double sigma_x() const { return lobes_[0].sx; }
  double sigma_y() const { return lobes_[0].sy; }
  double sigma_z() const { return lobes_[0].sz; }
  bool fluctuating() const { return lobes_[0].psd_spec_set; }

  // Laser over [t0, t1]: returns the on-fraction of the interval and the
  // beam position at the midpoint of its on-time (x, y in m).
  double laser_window(double t0, double t1, double &x, double &y) const;

  // Earliest time >= t at which the laser is on (t itself while on), or
  // DBL_MAX when no scheduled pass remains. Drives the fast-forward.
  double next_laser_on(double t) const;

  // `far`: the lattice box (xy faces, m), set by the solver at each grid build
  void set_box(double xlo, double xhi, double ylo, double yhi);
  double next_near(double t) const;
  void laser_samples(double t0, double t1, double ds, std::vector<double> &xyw) const;
  void request_freeze() { freeze_pending = true; }
  bool take_freeze() { const bool f = freeze_pending; freeze_pending = false; return f; }

  // Pending field reset (reset_temperature, end of layer): returns true
  // once per request.
  void request_reset(double T_reset);
  bool take_reset(double &T_reset);

  // The solver's materials, index = a cell's material byte: the part
  // (ts, tl: the app's), or the composition bins, then the plate if on.
  std::vector<Config::Material> materials(double ts, double tl) const;
  int plate_index() const { return cfg.plate_mat.on ? ncomp() : -1; }
  int ncomp() const { return cfg.comp.empty() ? 1 : (int) cfg.comp.size(); }
  // composition x -> its bin (0 without composition)
  int comp_bin(double x) const;

  // Enthalpy table H(T) on a uniform T grid (J/m^3), built from m's cp(T),
  // density and latent heat released linearly over [ts, tl].
  static void enthalpy_table(const Config::Material &m, double &T_lo, double &dT,
                             std::vector<double> &H);

  // Sites solidifying below this z (m) return to i2 = 6 (plate_mat.on only).
  double plate_surface() const;

 private:
  Config cfg;
  bool configured = false;   // set once the keyword defaults are filled in
  bool reset_pending = false;
  double reset_T = 0.0;
  bool freeze_pending = false;
  bool box_set = false;
  double box_[4] = {0, 0, 0, 0};
};

}

#endif
