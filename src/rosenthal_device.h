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

#ifndef SPK_ROSENTHAL_DEVICE_H
#define SPK_ROSENTHAL_DEVICE_H

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   RosenthalDevicePlan is a plain (Kokkos-free) snapshot of everything a
   device kernel needs to evaluate the analytical Rosenthal temperature for
   one timestep: the laser pose in the SPPARKS coordinate system (advanced on
   the host by ThermalManager, which owns the scan path) plus the material /
   source parameters (exported by RosenthalTemperatureSource).

   ThermalManager fills this once per step (cheap scalar work on the host); the
   Kokkos app then launches a parallel_for over all sites that evaluates the
   per-site temperature directly into the device temperature view -- replacing
   the host per-site exp/sqrt loop and the host->device sync.

   The pool-local transform (am_raster.h compute_position_relative_to_pool) and
   the Rosenthal kernel (temperature_source_rosenthal.cpp rosenthal_pointwise /
   rosenthal_kernel) are reproduced device-side from these scalars; keep this
   struct in sync with those two functions.

   KEYHOLE mode (Gauss-Legendre line integral) is not represented here -- it
   needs the quadrature node/weight arrays on the device; for keyhole the app
   falls back to the host path (valid is left false). STANDARD and ANISOTROPIC
   are the closed-form modes handled on device.
------------------------------------------------------------------------- */

struct RosenthalDevicePlan {
  bool   valid = false;       // false => not a device-eligible Rosenthal step;
                              //          caller must use the host path
  bool   scan_active = false; // false => laser path exhausted; T = T0 everywhere
  int    mode = 0;            // 0 = STANDARD, 1 = ANISOTROPIC, 2 = KEYHOLE

  // Laser pose in SPPARKS coordinates (physical meters), current timestep.
  double px = 0.0, py = 0.0;  // laser position (x,y)
  double layer_z = 0.0;       // scan-plane z (free surface)
  double ux = 1.0, uy = 0.0;  // scan unit direction in the x-y plane
  double v  = 0.0;            // scan speed (m/s)
  double T0 = 0.0;            // preheat / ambient temperature (K)

  // Material / source parameters (SI). lambdaQ is the image-source-ready
  // effective power lambda*Q for the point source (STANDARD / ANISOTROPIC).
  double lambdaQ = 0.0;
  double k_cond  = 0.0;       // thermal conductivity [W/(m K)]
  double alpha   = 0.0;       // thermal diffusivity [m^2/s]
  double r_min   = 0.0;       // 1/R singularity cutoff (meters)
  double eta_y   = 1.0;       // ANISOTROPIC transverse shape factor
  double eta_z   = 1.0;       // ANISOTROPIC depth shape factor

  // KEYHOLE (mode 2): a surface point source (lambda_pQ) plus a buried line
  // source (lambda_lQ) integrated over n_quad Gauss-Legendre nodes. The node
  // positions/weights (already mapped to [-d,d] with the Jacobian baked in) are
  // a fixed array uploaded separately to a device View.
  double lambda_pQ = 0.0;
  double lambda_lQ = 0.0;
  int    n_quad = 0;
};

}

#endif
