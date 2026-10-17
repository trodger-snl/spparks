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
   temperature for one timestep. The variable-length data -- the per-segment
   scan arrays and the per-lobe fluctuation tables -- are NOT in here; they are
   uploaded separately to device Views and refreshed when `version` changes (the
   scan is rebuilt by append_pass / reset_history_and_start).

   The per-site evaluation reproduces, device-side, the segment loop in GREENAM
   ScanIntegration::integrate_point_adaptive(), the GaussGreenEllips integrand
   (with the optional FluctuationTable modulation), and the keyhole sum over
   lobes (MoserTemperatureSource::get_temperature_at_xyz_and_time). Keep them in
   sync.

   Supports STANDARD (n_lobes = 1) and KEYHOLE (n_lobes = 2), with or without
   stochastic fluctuations. The path geometry (bounds/xl/yl/normals/speeds) is
   shared across lobes; only the per-segment power, the ellipsoid widths, the
   laser-plane z, and the fluctuation table differ per lobe.

   valid is false (host path) only for an unbuilt scan or more lobes than this
   POD can hold.
------------------------------------------------------------------------- */

struct MoserDevicePlan {
  bool   valid = false;
  double time = 0.0;          // integration upper limit (absolute sim time)
  double T0 = 0.0;            // ambient / preheat (K)
  double alpha = 0.0;         // thermal diffusivity [m^2/s]
  double c = 0.0;             // volumetric heat capacity rho*cp [J/(m^3 K)]
  double char_length = 0.0;   // quadrature resolution control
  int    nseg = 0;            // number of scan segments (shared across lobes)

  // Per-lobe data (STANDARD uses lobe 0 only; KEYHOLE uses 0=top, 1=bot).
  static constexpr int MAX_LOBES = 2;
  int    n_lobes = 1;
  double sx[MAX_LOBES] = {0.0, 0.0};
  double sy[MAX_LOBES] = {0.0, 0.0};
  double sz[MAX_LOBES] = {0.0, 0.0};
  double zl[MAX_LOBES] = {0.0, 0.0};   // effective laser-plane z (plane - z_offset)
  int    fluct_n[MAX_LOBES] = {0, 0};  // fluctuation-table samples per lobe (0 = none)

  long   version = -1;        // scan generation; bump -> re-upload device arrays
};

}

#endif
