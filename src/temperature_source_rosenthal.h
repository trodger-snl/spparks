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

#ifndef SPK_TEMPERATURE_SOURCE_ROSENTHAL_H
#define SPK_TEMPERATURE_SOURCE_ROSENTHAL_H

#include "temperature_source.h"
#include <string>
#include <vector>

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   Analytical Rosenthal moving point heat source on a free surface
   (half-space, image-source method).

   The laser plane (scan_layer_z in the driving app) is treated as a
   free surface. For sites ABOVE the laser plane (z_rel > 0) there is
   no material and rosenthal_pointwise() returns T0. For sites BELOW
   the laser plane (z_rel <= 0) the textbook image-source method
   doubles the unbounded-medium kernel:

       T = T0 + 2 * (lambda*Q) / (2*pi*k*R) * exp(-v*(xi+R)/(2*alpha))

   This is the physically correct form for a moving point heat source
   on the surface of a thermally insulating half-space (no flux through
   the free surface). It produces a hemispherical pool below the laser
   plane and zero heat above, regardless of where the laser plane is
   positioned within the simulation domain.

   Path-agnostic: this class does not own a scan path. The driving app
   provides pool-local coordinates (xi, y_rel, z_rel) and the current
   scan velocity each timestep, and queries rosenthal_pointwise().

   Three modes are supported, all derived from the analytical formulas
   in J G Pauza et al 2021 Modelling Simul. Mater. Sci. Eng. 29 055019,
   Appendix B (with the explicit factor of 2 from the image source):

   STANDARD    (Eq. B4):  R  = sqrt(xi^2 + y^2 + z^2)

   ANISOTROPIC (Eq. B6):  R replaced by
                          R_eta = sqrt(xi^2 + (eta_y*y)^2 + (eta_z*z)^2)

   KEYHOLE     (Eq. B8):  point + line source combination
                          T = point(lambda_p) + integral_{-d}^{d} line(lambda_l, D) dD
                          R' = sqrt(xi^2 + y^2 + (D + z)^2)
                          Line integral evaluated by Gauss-Legendre quadrature;
                          the factor of 2 image source applies to BOTH the
                          surface point source AND every line-source quadrature
                          node.

   IMPORTANT lambda convention: the Pauza paper writes the kernel with
   prefactor 1/(2*pi) and then evaluates it in the lower half-space, so
   their published lambda values implicitly absorb the factor-of-2 image
   source. To reproduce paper Table B1 pool sizes here, divide the
   paper's lambda values by 2 (e.g. paper case-3 lambda=0.155 becomes
   lambda=0.0775 in this implementation).
------------------------------------------------------------------------- */

class RosenthalTemperatureSource : public TemperatureSource {
 public:
  enum class RosenthalMode { STANDARD, ANISOTROPIC, KEYHOLE };

  RosenthalTemperatureSource(class SPPARKS *);
  virtual ~RosenthalTemperatureSource();

  // TemperatureSource interface
  virtual void setup_temperature_source(const std::vector<std::string> &args) override;
  virtual double get_temperature_at_xyz_and_time(double x, double y, double z, double time) override;
  virtual void update_temperatures(double dt, double simulation_time) override;
  virtual bool needs_data_refresh(double simulation_time) override;
  virtual void cleanup() override;
  virtual std::string get_source_type() const override { return "rosenthal"; }
  virtual void print_source_info() const override;

  // Fast-forward support: the source declares it; the driving app
  // implements the actual time prediction using path geometry.
  virtual bool supports_time_queries() const override { return true; }
  virtual double get_next_time_with_temperature(double current_time, double threshold_temp) override;

  // Pool-local-frame query used by AppAdditiveExtTempTexture every step.
  // (xi, y_rel, z_rel) are coordinates of a lattice site relative to the
  // current laser position, with xi along the scan direction. v is the
  // current scan speed (m/s). T0 is the preheat/ambient temperature (K).
  double rosenthal_pointwise(double xi, double y_rel, double z_rel,
                             double v, double T0) const;

  // Conservative upper bound on the Rosenthal temperature for ANY site
  // whose distance to the current laser position is >= R. Used by the
  // driving app's fast-forward predictor to guarantee that real heating
  // events are never skipped. Bounds details:
  //   - exponent term -v(xi+R)/(2 alpha) is maximized at zero (xi=-R),
  //     so the kernel reduces to Q_eff_total / (2 pi k R)
  //   - ANISOTROPIC: R_eta <= R / min(eta_y, eta_z, 1) shrinks R
  //   - KEYHOLE: sum point + line integrals' max contributions
  double rosenthal_peak_at_distance(double R, double T0) const;

  // Cutoff radius for the 1/R singularity at the laser site.
  // Set by the app from its lattice spacing dx.
  void set_r_min(double r) { r_min = r; }

  RosenthalMode get_mode() const { return mode; }

 private:
  RosenthalMode mode;

  // Material / laser parameters (SI units)
  double Q;            // total laser power [W]
  double lambda;       // absorption efficiency (STANDARD, ANISOTROPIC)
  double lambda_p;     // point-source absorption (KEYHOLE)
  double lambda_l;     // line-source absorption (KEYHOLE)
  double k_cond;       // thermal conductivity [W/(m K)]
  double alpha;        // thermal diffusivity [m^2/s]
  double T0_default;   // preheat / ambient [K]; also stored in base ambient_temperature

  // Anisotropic-mode shape factors
  double eta_y;
  double eta_z;

  // Keyhole-mode line source
  double d_keyhole;    // half-depth of line integral [m]
  int    n_quad;       // number of Gauss-Legendre nodes
  std::vector<double> quad_nodes;    // mapped to [-d, d]
  std::vector<double> quad_weights;  // mapped weights (include Jacobian)

  // Singularity cutoff (meters); 0 means no cutoff (set externally)
  double r_min;

  // Helpers
  double rosenthal_kernel(double Q_eff, double v, double xi, double R) const;
  void   build_gauss_legendre_quadrature();
  void   validate_after_setup() const;
};

}

#endif
