/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator

   Website
   https://spparks.github.io/

   See authors
   https://spparks.github.io/authors.html

   Copyright(C) 1999-2025 National Technology & Engineering Solutions
   of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with
   NTESS, the U.S. Government retains certain rights in this software.

   This software is distributed under the GNU General Public License.  See
   LICENSE in top-level SPPARKS directory.
------------------------------------------------------------------------- */

#ifdef APP_CLASS
AppStyle(phasefield/kk,AppPhaseFieldKokkos)

#else

#ifndef SPK_APP_PHASEFIELD_KOKKOS_H
#define SPK_APP_PHASEFIELD_KOKKOS_H

#include "app_lattice_kokkos.h"
#include "app_phasefield.h"
#include "error.h"

namespace SPPARKS_NS {

// app_style phasefield on the device: the model's aux() and update() run as
// kernels; the fields stay on the device, copied back only when output is due
class AppPhaseFieldKokkos : public LatticeKokkos<AppPhaseField> {
 public:
  AppPhaseFieldKokkos(class SPPARKS *, int, char **);

  void grow_app() override;
  void init_app() override;
  void setup_app() override { need_upload = true; }
  void app_update(double) override;

  // define device lambdas, so public (nvcc)
  template <class M> void step_kernels(const M &);

  void sync_app_data_to_device() override;
  void sync_app_data_to_host() override;
  void sync_ghosts_to_device() override;
  void allocate_app_views() override;
  void sweep_color_kokkos(int, int, int) override {
    error->all(FLERR, "phasefield/kk has no sweeps");
  }

  using DevCtx = PFCtx<Kokkos::View<int**>, Kokkos::View<double*>, Kokkos::View<int*>>;
  Kokkos::View<double*> d_f[PF_MAXF], d_fnew[PF_MAXF], d_aux[PF_MAXA];
  Kokkos::View<int*> d_phase;
  Kokkos::View<double*> d_tab;
  Kokkos::View<PFPhaseTable*> d_ptab;
  Kokkos::View<int> d_flags;
  bool need_upload = true;
};

}

#endif
#endif
