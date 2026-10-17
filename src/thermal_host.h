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

#ifndef SPK_THERMAL_HOST_H
#define SPK_THERMAL_HOST_H

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   ThermalHost is a plain snapshot of the lattice state that ThermalManager
   needs from its owning app. It decouples ThermalManager from a concrete App
   subclass: instead of a typed back-pointer + friend access into
   AppAdditiveTexture, the manager reads/writes through this struct, which the
   owning app populates (via its refresh_thermal_host()) at the top of
   app_update() and before dispatching any thermal command. This lets BOTH the
   CPU app (AppAdditiveTexture) and the Kokkos app (AppAdditiveTextureKokkos),
   which do not share an additive-app base class, reuse the same ThermalManager.

   All fields are stable within a single app_update() step:
     - T is the per-site temperature array; ThermalManager WRITES into it
       (the pointer itself is refreshed by the app after grow_app()).
     - the remaining pointers/scalars are read-only to ThermalManager.
------------------------------------------------------------------------- */

struct ThermalHost {
  double *T = nullptr;              // per-site temperature (manager writes this)
  double **xyz = nullptr;          // site coordinates
  int *numneigh = nullptr;         // neighbor counts per site
  int **neighbor = nullptr;        // neighbor lists per site
  int nlocal = 0;                  // # owned sites
  double dx = 0.0;                 // source lattice spacing (physical m)
  double tl = 0.0;                 // liquidus temperature (K)
  double ts = 0.0;                 // solidus temperature (K)
  double t_room = 0.0;             // ambient/room temperature (K)
  double time = 0.0;               // current simulation time (s)
  double dt = 0.0;                 // simulation timestep (s)
  class CommLattice *comm = nullptr;  // ghost-communication driver
  const int *ghost_iindices = nullptr; // iarray indices to ghost-comm
  int nghost_iarray = 0;               // count
  const int *ghost_dindices = nullptr; // darray indices to ghost-comm
  int nghost_darray = 0;               // count
};

}

#endif
