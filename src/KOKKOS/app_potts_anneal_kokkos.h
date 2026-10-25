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
AppStyle(potts/anneal/kk,AppPottsAnnealKokkos)

#else

#ifndef SPK_APP_POTTS_ANNEAL_KOKKOS_H
#define SPK_APP_POTTS_ANNEAL_KOKKOS_H

#include "app_lattice_kokkos.h"
#include "app_potts_anneal.h"

namespace SPPARKS_NS {

// potts/anneal with the colour sweep, recovery and nucleation on the device;
// the CPU app keeps the parsing, checks, stored_energy and tables
class AppPottsAnnealKokkos : public LatticeKokkos<AppPottsAnneal> {
 public:
  AppPottsAnnealKokkos(class SPPARKS *, int, char **);

  void grow_app() override;
  void init_app() override;

  void sync_app_data_to_device() override;
  void sync_app_data_to_host() override;
  void sync_ghosts_to_device() override;
  void allocate_app_views() override;

  // define device lambdas, so public (nvcc)
  void sweep_color_kokkos(int, int, int) override;
  void app_update(double) override;

  using IView = Kokkos::View<int*>;
  using DView = Kokkos::View<double*>;
  IView d_spin, d_phase, d_seed, d_next;
  DView d_q0, d_qx, d_qy, d_qz, d_h;
  DView d_sym, d_orient;
  Kokkos::View<tagint*> d_id;

  anneal::Sites<IView, DView> device_sites() const
  {
    return {d_spin, d_phase, d_q0, d_qx, d_qy, d_qz, d_h};
  }
};

}

#endif
#endif
