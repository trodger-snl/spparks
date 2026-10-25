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
AppStyle(potts/pin/kk,AppPottsPinKokkos)

#else

#ifndef SPK_APP_POTTS_PIN_KOKKOS_H
#define SPK_APP_POTTS_PIN_KOKKOS_H

#include "app_lattice_kokkos.h"
#include "app_potts_pin.h"

namespace SPPARKS_NS {

// potts/pin with the colour sweep on the device; the CPU app keeps the
// parsing, the pin command and the KMC events
class AppPottsPinKokkos : public LatticeKokkos<AppPottsPin> {
 public:
  AppPottsPinKokkos(class SPPARKS *, int, char **);

  void grow_app() override;
  void init_app() override;
  void setup_end_app() override;

  void sync_app_data_to_device() override;
  void sync_app_data_to_host() override;
  void sync_ghosts_to_device() override;
  void allocate_app_views() override;

  // defines a device lambda, so public (nvcc)
  void sweep_color_kokkos(int, int, int) override;

  struct Boltz { double b[33]; };  // exp(-dE/T), dE = 0..maxneigh

  Kokkos::View<int*> d_spin;
  Boltz boltz;
};

}

#endif
#endif
