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

#ifndef SPK_MOSER_DEVICE_H
#define SPK_MOSER_DEVICE_H

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   MoserDevicePlan is a plain (Kokkos-free) snapshot of the scalar state a
   device kernel needs to evaluate the time-domain Green's-function (Moser)
   temperature for one timestep. The variable-length scan-segment arrays
   (bounds/xl/yl/powers/normals/speeds) are NOT in here -- they are uploaded
   separately to device Views and refreshed when `version` changes (the scan is
   rebuilt by append_pass / reset_history_and_start).

   The per-site evaluation reproduces, device-side, the segment loop in
   GREENAM ScanIntegration::integrate_point_adaptive() and the Green's-ellipsoid
   integrand GaussGreenEllips::operator(); keep them in sync.

   valid is false (host path) for KEYHOLE mode, configured stochastic
   fluctuations, dynamic inter-pass pause/reset, or an unbuilt scan -- only the
   single-lobe STANDARD, no-fluctuation, statically-scheduled case runs on the
   device, mirroring the Rosenthal STANDARD/ANISOTROPIC device scope.
------------------------------------------------------------------------- */

struct MoserDevicePlan {
  bool   valid = false;
  double time = 0.0;          // integration upper limit (absolute sim time)
  double T0 = 0.0;            // ambient / preheat (K)
  double alpha = 0.0;         // thermal diffusivity [m^2/s]
  double c = 0.0;             // volumetric heat capacity rho*cp [J/(m^3 K)]
  double sx = 0.0, sy = 0.0, sz = 0.0;  // ellipsoid sigmas (single lobe)
  double zl = 0.0;            // laser-plane z (free surface)
  double char_length = 0.0;   // quadrature resolution control
  int    nseg = 0;            // number of scan segments
  long   version = -1;        // scan generation; bump -> re-upload segment arrays
};

}

#endif
