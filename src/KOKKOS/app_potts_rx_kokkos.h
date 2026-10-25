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
AppStyle(potts/rx/kk,AppPottsRXKokkos)

#else

#ifndef SPK_APP_POTTS_RX_KOKKOS_H
#define SPK_APP_POTTS_RX_KOKKOS_H

#include "app_lattice_kokkos.h"
#include "app_potts_rx.h"

namespace SPPARKS_NS {

// potts/rx with the colour sweep and the nucleation on the device; the CPU
// app keeps the parsing, checks and tables
class AppPottsRXKokkos : public LatticeKokkos<AppPottsRX> {
 public:
  AppPottsRXKokkos(class SPPARKS *, int, char **);

  void grow_app() override;
  void init_app() override;

  void sync_app_data_to_device() override;
  void sync_app_data_to_host() override;
  void sync_ghosts_to_device() override;
  void allocate_app_views() override;

  // define device lambdas, so public (nvcc)
  void sweep_color_kokkos(int, int, int) override;
  void app_update(double) override;

  Kokkos::View<int*> d_spin;
  Kokkos::View<double*> d_h, d_w;
  Kokkos::View<tagint*> d_id;
  Kokkos::View<int*> d_flag;   // a seed's new spin, or 0
};

}

#endif
#endif
