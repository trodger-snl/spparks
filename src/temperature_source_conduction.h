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
    double stretch = 1.1;
    double max_cell = 0.0;             // 0 = uncapped
    std::string part_region;           // empty = whole domain is part
    double layer_thickness = 0.0;      // powder depth outside the box
    int    far_bc = FAR_FIXED;
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

  // Pending reset_temperature request: returns true once per reset.
  bool take_reset(double &T_reset);

  // Enthalpy table H(T) on a uniform T grid (J/m^3), built from cp(T),
  // density and latent heat released linearly over [ts, tl].
  void enthalpy_table(double ts, double tl, double &T_lo, double &dT,
                      std::vector<double> &H) const;

 private:
  Config cfg;
  bool configured = false;   // set once the keyword defaults are filled in
  bool reset_pending = false;
  double reset_T = 0.0;
};

}

#endif
