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
#include "spktype.h"
#include "gpu_aware_mpi.h"
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
   Laser-off intervals use RKL2 super-steps (sts stages); laser-on
   intervals stay explicit at dt_fd.
------------------------------------------------------------------------- */

class ConductionSolverKokkos : protected Pointers {
 public:
  ConductionSolverKokkos(class SPPARKS *, class ConductionTemperatureSource *);

  // Advance the field to t; then sample() writes T for all nall sites into
  // d_T and returns the max T over owned sites (the fused-max convention).
  void advance(double t, Kokkos::View<double*[3], Kokkos::LayoutRight> d_xyz,
               Kokkos::View<int*> d_af, int nlocal, int nall, double dx, double ts, double tl);

  // LOOSE: flag bit, see packing. PLATE: flag bit, the plate material.
  // UNBORN: flag bit on POWDER under ded, gas until the laser deposits it
  enum Phase { GAS = 0, POWDER = 1, DENSE = 3, LOOSE = 4, PLATE = 8, UNBORN = 16 };

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

  // State in KK_FLOAT. H is stored relative to Href = H(T0) when
  // KK_FLOAT is float (it resolves small increments near T0 that way; 0 in
  // FP64, so the FP64 arithmetic is the original).
  Kokkos::View<KK_FLOAT***> H, T, Tn;
  Kokkos::View<KK_FLOAT***> Ya, Yb, L0;   // RKL2 stage buffers (allocated on demand)
  Kokkos::View<unsigned char***> P, Pn;
  // per local cell: width and lower/upper face per axis (halo included)
  Kokkos::View<double*> wx, wy, wz, fxlo, fxhi, fylo, fyhi, fzlo, fzhi;
  Kokkos::View<KK_FLOAT*> Ex, Ey, Ez;   // per-axis Gaussian integrals, per sub-step
  double zsrc = 0.0;                  // source plane: laser plane, capped at the top face
  double birth_depth = 1e300;         // ded: cells born within this of zsrc
  double Hbirth = 0.0;                // ded: H(T_birth) - Href
  Kokkos::View<KK_FLOAT*> Htab, Htab2;   // H(T) - Href, part and plate
  Kokkos::View<double*> kT, kV, kT2, kV2;
  // KK_FLOAT copies for the stencil (aliases under FP64): FP32 then reads
  // floats instead of converting doubles at FP64 rate
  Kokkos::View<KK_FLOAT*> kTr, kVr, kT2r, kV2r, wxr, wyr, wzr;
  // so the stencil multiplies: k-table slopes, 1/width, 1/(H(i+1) - H(i))
  Kokkos::View<KK_FLOAT*> kSr, kS2r, iwx, iwy, iwz, iHtab, iHtab2;
  double Htab_T0 = 0.0, Htab_dT = 1.0;
  Kokkos::View<int*[3]> site_cell;   // local (i,j,k) per site, i<0 = above grid

  Kokkos::View<double*> sbuf, rbuf;
  // host staging of the packed halo (MPIStagingSpace); unused (empty) where MPI
  // can read sbuf/rbuf itself: host-accessible device memory, or GPU-aware MPI
  Kokkos::View<double*, MPIStagingSpace> hsbuf, hrbuf;
  bool stage_host = false;

  void build(Kokkos::View<double*[3], Kokkos::LayoutRight> d_xyz, Kokkos::View<int*> d_af,
             int nlocal, int nall, double ts);
  double H_at(Kokkos::View<KK_FLOAT*> tab, double Tr) const;
  void step(double h, double t0);
  void sts_step(double tau, int s);

  // These define device lambdas (KOKKOS_LAMBDA), which nvcc rejects in a
  // private or protected member function, and Stage is a kernel argument,
  // which nvcc also needs public; not part of the class interface.
 public:
  // One stage update. Explicit Euler (sts = false): H += h L(H). RKL2 stage
  // (sts = true): Yout = mu Y1 + nu Y2 + (1-mu-nu) H + mut L(Y1) + gat L0,
  // with T holding T(Y1); store_L0 saves L(Y1) (stage 1, Y1 = H).
  struct Stage {
    bool sts = false, store_L0 = false, phase = true;
    double h = 0.0, mu = 1.0, nu = 0.0, mut = 0.0, gat = 0.0;
    double Pq = 0.0, xl = 0.0, yl = 0.0;
  };
  void stage(const Stage &c, Kokkos::View<KK_FLOAT***> Y1,
             Kokkos::View<KK_FLOAT***> Y2, Kokkos::View<KK_FLOAT***> Yout);
  void exchange();
  void reset_field(double Tr);
  // Write T for all nall sites into d_T; return the max over owned sites.
  // Each owned site's T also goes to on_owned(s, T), so the caller's own
  // per-site pass (the app's active-front list) rides along instead of
  // reading the whole lattice again.
  template <class F>
  double sample(Kokkos::View<double*> d_T, int nlocal, int nall, const F on_owned) {
    auto T_ = T; auto sc = site_cell;
    const double Ta = ambient();
    double m = 0.0;
    Kokkos::parallel_reduce("ConductionSample", nall, KOKKOS_LAMBDA(const int s, double &mx) {
      const int i = sc(s, 0);
      const double v = i < 0 ? Ta : T_(i, sc(s, 1), sc(s, 2));
      d_T(s) = v;
      if (s < nlocal) {
        if (v > mx) mx = v;
        on_owned(s, v);
      }
    }, Kokkos::Max<double>(m));
    return m;
  }

 private:
  double ambient() const;
};

}

#endif
