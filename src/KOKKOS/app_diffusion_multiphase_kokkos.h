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
AppStyle(diffusion/multiphase/kk,AppDiffusionMultiphaseKokkos)

#else

#ifndef SPK_APP_DIFFUSION_MULTIPHASE_KOKKOS_H
#define SPK_APP_DIFFUSION_MULTIPHASE_KOKKOS_H

#include "app_lattice_kokkos.h"
#include "app_diffusion_multiphase.h"

namespace SPPARKS_NS {

// diffusion/multiphase with the colour sweep on the device: the phases,
// pins and pair weights become tables indexed by a dense phase number;
// the CPU app keeps the parsing and the KMC events
class AppDiffusionMultiphaseKokkos : public LatticeKokkos<AppDiffusionMultiphase> {
 public:
  AppDiffusionMultiphaseKokkos(class SPPARKS *, int, char **);

  void grow_app() override;
  void init_app() override;

  void sync_app_data_to_device() override;
  void sync_app_data_to_host() override;
  void sync_ghosts_to_device() override;
  void allocate_app_views() override;

  // defines a device lambda, so public (nvcc)
  void sweep_color_kokkos(int, int, int) override;

  Kokkos::View<int*> d_lattice;
  Kokkos::View<int*> d_pidx;      // phase label -> 0..nphase-1
  Kokkos::View<int*> d_pinned;    // per phase
  Kokkos::View<double*> d_w;      // nphase x nphase pair weights
  int nphase;
};

}

#endif
#endif
