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
    int    sts_smax = 32;               // RKL2 stages per laser-off super-step; 0 = explicit
    // plate_material: the dense material of i2 = 6 sites and of the cells
    // below plate_top outside the box. Same keys as the part above.
    struct Plate {
      bool on = false;
      std::vector<double> k_T, k_v, cp_T, cp_v;
      double rho = 0.0, latent = 0.0, ts = 0.0, tl = 0.0;
      std::string name;
    } plate_mat;
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

  // Pending field reset (reset_temperature, end of layer): returns true
  // once per request.
  void request_reset(double T_reset);
  bool take_reset(double &T_reset);

  // Enthalpy table H(T) on a uniform T grid (J/m^3), built from cp(T),
  // density and latent heat released linearly over [ts, tl]; the plate
  // material's when plate is true.
  void enthalpy_table(double ts, double tl, double &T_lo, double &dT,
                      std::vector<double> &H, bool plate = false) const;

  // Sites solidifying below this z (m) return to i2 = 6 (plate_mat.on only).
  double plate_surface() const;

 private:
  Config cfg;
  bool configured = false;   // set once the keyword defaults are filled in
  bool reset_pending = false;
  double reset_T = 0.0;
};

}

#endif
