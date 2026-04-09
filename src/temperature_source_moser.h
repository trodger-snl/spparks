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

#ifndef SPK_TEMPERATURE_SOURCE_MOSER_H
#define SPK_TEMPERATURE_SOURCE_MOSER_H

#include "temperature_source.h"
#include <memory>
#include <string>
#include <vector>

// Forward-declare the GREENAM templates so the cpp file is the only
// translation unit that has to instantiate them. The instantiation
// types are defined in the cpp via aliases.
namespace sierra { namespace greenam {
  template <typename RealType> struct VectorWrapper;
  template <typename RealType, typename ArrayType> struct LaserScan;
  template <typename RealType, typename ArrayType, unsigned MaxOrder> struct ScanIntegration;
}}

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   Time-domain Green's function temperature source after Moser et al.
   "Generation of an Engineering Scale Thermal History via the Use of an
    Adaptive Space-Time Grid" (SAND2017, Section 2.1).

   Eq. 4 of that report:

       T(x,y,z,t) = T0
                  + (2 / (pi^{3/2} rho cp)) *
                    integral_{0}^{t} P(s) *
                      exp( -(x-xl(s))^2/(sx^2+4 alpha (t-s))
                           -(y-yl(s))^2/(sy^2+4 alpha (t-s))
                           -(z-zl(s))^2/(sz^2+4 alpha (t-s)) )
                      / sqrt( (sx^2+4 alpha (t-s)) (sy^2+4 alpha (t-s))
                              (sz^2+4 alpha (t-s)) )  ds

   The factor of 2 is the half-space image source (insulating free
   surface, source mirrored across z = zl).  The integrand is the
   convolution of an ellipsoidal Gaussian source with the unsteady
   point heat-source Green's function.  In the limit (sx, sy, sz) -> 0
   it reduces to the standard moving-point Rosenthal kernel.

   This source is path-aware: it owns its own piecewise-linear scan
   path (laser segments with constant power and velocity per segment),
   built by the driving app from the laser_path command.  Sites are
   queried in the WORLD frame (meters), and absolute simulation time is
   used as the integration upper limit.

   Implementation notes:
     * The actual integrator lives in the vendored GREENAM/ headers
       (sierra::greenam::ScanIntegration).  This class is a thin
       SPPARKS-side wrapper.
     * No MPI communication is needed; the integrator is purely local
       and produces identical results on every rank for the same
       (x,y,z,t).
     * Material parameters reuse the existing Rosenthal-style inputs
       (Q, lambda, k, alpha, T0) plus a single specific heat cp.
       The volumetric heat capacity rho*cp = k/alpha is computed
       internally; cp is only needed to recover rho for the
       diagnostic print.
------------------------------------------------------------------------- */

class MoserGreenTemperatureSource : public TemperatureSource {
 public:
  MoserGreenTemperatureSource(class SPPARKS *);
  virtual ~MoserGreenTemperatureSource();

  // TemperatureSource interface
  virtual void setup_temperature_source(const std::vector<std::string> &args) override;
  virtual double get_temperature_at_xyz_and_time(double x, double y, double z, double time) override;
  virtual void update_temperatures(double dt, double simulation_time) override;
  virtual bool needs_data_refresh(double simulation_time) override;
  virtual void cleanup() override;
  virtual std::string get_source_type() const override { return "moser"; }
  virtual void print_source_info() const override;

  // Build a scan from a RASTER-style straight-line repeated path. Called
  // by AppAdditiveExtTempTexture::laser_path_cmd when the active
  // temperature source is Moser. Coordinates and times are SI units;
  // start_time is the absolute simulation time at which the laser begins
  // its first repeat (typically 0 at run start). All segments share the
  // single (Q*lambda) absorbed power configured at setup time; transits
  // between repeats are 0-power placeholder segments that the integrator
  // skips.
  void build_scan(double start_time,
                  double x0, double y0,
                  double x1, double y1,
                  double laser_plane_z,
                  double speed,
                  int repeats);

  bool has_scan() const { return scan_ != nullptr; }

  // Quadrature character length (Moser's `char_length`); 0.5 gives
  // ~1e-4 fractional integration error per the GREENAM comments.
  void set_char_length(double cl) { char_length = cl; }

 private:
  // Material / source parameters
  double Q;          // total laser power [W]
  double lambda;     // absorption efficiency
  double k_th;       // thermal conductivity [W/(m K)]
  double alpha;      // thermal diffusivity [m^2/s]
  double T0_default; // ambient/preheat temperature [K]
  double cp;         // specific heat [J/(kg K)]
  double rho;        // density derived from k/(alpha*cp) [kg/m^3]
  double sx, sy, sz; // ellipsoid Gaussian widths [m]

  // Quadrature tuning
  double char_length;

  // Stored scan geometry (for diagnostic print only; the authoritative
  // copy lives inside scan_)
  double scan_t_origin;
  double scan_x0, scan_y0, scan_x1, scan_y1, scan_zl;
  double scan_speed;
  int    scan_repeats;
  bool   scan_built;

  // GREENAM integrator pointers (instantiated in the cpp file via
  // forward-declared templates so the GREENAM headers are pulled into
  // exactly one translation unit).
  using GREENAM_Array  = sierra::greenam::VectorWrapper<double>;
  using GREENAM_Scan   = sierra::greenam::LaserScan<double, GREENAM_Array>;
  using GREENAM_Integ  = sierra::greenam::ScanIntegration<double, GREENAM_Array, 30>;

  std::shared_ptr<GREENAM_Scan>  scan_;
  std::shared_ptr<GREENAM_Integ> integrator_;
};

}

#endif
