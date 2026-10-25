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
AppStyle(phasefield/potts/table/kk,AppPhaseFieldPottsTableKokkos)

#else

#ifndef SPK_APP_PHASEFIELD_POTTS_TABLE_KOKKOS_H
#define SPK_APP_PHASEFIELD_POTTS_TABLE_KOKKOS_H

#include "app_lattice_kokkos.h"
#include "app_phasefield_potts_table.h"

namespace SPPARKS_NS {

// phasefield/potts/table on the device: the colour sweep and the coupled
// c/T finite-difference steps run as kernels over the same pftable::
// functions as the CPU app, which owns parsing, tables and the host state.
class AppPhaseFieldPottsTableKokkos : public LatticeKokkos<AppPhaseFieldPottsTable> {
 public:
  AppPhaseFieldPottsTableKokkos(class SPPARKS *, int, char **);

  void grow_app() override;
  void init_app() override;

  // define device lambdas, so public (nvcc)
  void app_update(double) override;
  void sweep_color_kokkos(int, int, int) override;

  void sync_app_data_to_device() override;
  void sync_app_data_to_host() override;
  void sync_ghosts_to_device() override;
  void allocate_app_views() override;

  Kokkos::View<int*> d_spin, d_phase;
  Kokkos::View<double*> d_c, d_T, d_mu, d_cnew, d_Tnew;
  Kokkos::View<double*> d_tab;
  Kokkos::View<PFPhaseTable*> d_ptab;
  Kokkos::View<int*> d_reset;
  Kokkos::View<double*> d_resetval;
  Kokkos::View<int> d_flags;
  int reset_uploaded = -1;      // pf_resetversion on the device

 private:
  void upload_reset_list();
};

}

#endif
#endif
