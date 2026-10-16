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

#ifndef SPK_CONDUCTION_SOLVER_KOKKOS_H
#define SPK_CONDUCTION_SOLVER_KOKKOS_H

#include "pointers.h"
#include <Kokkos_Core.hpp>
#include <vector>

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   Device state of the conduction temperature source: enthalpy on a
   cell-centred tensor-product grid. Inside the lattice box the cells ARE
   the lattice sites (spacing dx); outside, cells grow geometrically out to
   the margins and the plate depth. Explicit, sub-cycled, finite volume.

   MPI: x/y follow the lattice procgrid (z is not split); each rank owns the
   cells over its own sites, and edge ranks also own the margin cells. A
   1-cell halo of T and phase is exchanged in x then y every sub-step.

   ponytail: edge ranks carry all margin cells; rebalance by cell count if
   margin-heavy decks show the imbalance.
   ponytail: explicit step is bound by the finest cell, so pauses cost
   pause/dt_fd sub-steps; RKL2 super-time-stepping keeps the same halo'd
   stencil if that dominates.
------------------------------------------------------------------------- */

class ConductionSolverKokkos : protected Pointers {
 public:
  ConductionSolverKokkos(class SPPARKS *, class ConductionTemperatureSource *);

  // Advance the field to t, then write T for all nall sites into d_T.
  // Returns the max T over owned sites (for the fused-max convention).
  double fill(double t, Kokkos::View<double*> d_T,
              Kokkos::View<double*[3]> d_xyz, Kokkos::View<int*> d_af,
              int nlocal, int nall, double dx, double ts, double tl);

  enum Phase { GAS = 0, POWDER = 1, DENSE = 3 };

 private:
  class ConductionTemperatureSource *src;

  bool built = false;
  int built_nall = -1;
  double built_zl = 0.0;
  double t_fd = 0.0;
  double dt_fd = 0.0;
  double dx = 0.0, tl = 0.0;

  // local grid: owned x cells 1..nxo, y cells 1..nyo (0 and n+1 are halo),
  // z cells 0..nz-1
  int nxo = 0, nyo = 0, nz = 0;
  bool edge[4] = {false, false, false, false};   // xlo xhi ylo yhi
  int nbr[4] = {-1, -1, -1, -1};
  double zl = 0.0;

  Kokkos::View<double***> H, T, Tn;
  Kokkos::View<unsigned char***> P, Pn;
  // per local cell: width and lower/upper face per axis (halo included)
  Kokkos::View<double*> wx, wy, wz, fxlo, fxhi, fylo, fyhi, fzlo, fzhi;
  Kokkos::View<double*> Ex, Ey, Ez;   // per-axis Gaussian integrals, per sub-step
  double zsrc = 0.0;                  // source plane: laser plane, capped at the top face
  Kokkos::View<double*> Htab, kT, kV;
  double Htab_T0 = 0.0, Htab_dT = 1.0;
  Kokkos::View<int*[3]> site_cell;   // local (i,j,k) per site, i<0 = above grid

  Kokkos::View<double*> sbuf, rbuf;

  void build(Kokkos::View<double*[3]> d_xyz, Kokkos::View<int*> d_af,
             int nlocal, int nall, double ts);
  void reset_field(double Tr);
  void step(double h, double t0);
  void exchange();
  double sample(Kokkos::View<double*> d_T, int nlocal, int nall);
};

}

#endif
