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

#ifndef SPK_APP_LATTICE_KOKKOS_H
#define SPK_APP_LATTICE_KOKKOS_H

#include "app_lattice.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <vector>

namespace SPPARKS_NS {

class AppLatticeKokkos : public AppLattice {
public:
  AppLatticeKokkos(class SPPARKS *, int, char **);
  virtual ~AppLatticeKokkos();

  // === Kokkos type aliases ===
  using pool_type = Kokkos::Random_XorShift64_Pool<>;

  // === Shared Kokkos data (all derived classes use these) ===
  // Device-only: filled from AppLattice's numneigh/neighbor, which stay the
  // host copy. (A DualView here kept a second 26*nall-int host array.)
  // Exactly one neighbor view is allocated, chosen by neighbor_row_major.
  // ponytail: two layouts. The potts colour sweeps coalesce on LayoutLeft
  // (row-major measured 23% slower for potts/kk, 18% for neighonly, 256^3
  // gfx1151); the additive front kernels gather whole rows and ran 10% faster
  // row-major, which is also the host block's layout. Look for one layout that
  // suits both (e.g. colour-packed rows) if this split spreads.
  Kokkos::View<int*> d_numneigh;
  Kokkos::View<int**> d_neighbor;                          // LayoutLeft on GPUs
  Kokkos::View<int**, Kokkos::LayoutRight> d_neighbor_rm;  // host layout
  bool neighbor_row_major = false;
  pool_type *rand_pool;

  // === Color-based site lists for GPU ===
  // When bothflag=1 (sectors + colors), we have per-sector color lists
  // When bothflag=0, we have a single set of color lists (sector 0 only)
  int kokkos_ncolors;                    // Number of colors (8 for 26-neighbor)
  int kokkos_nsectors;                   // Number of sectors to track (1 if !bothflag)
  // Per-sector color lists stored in vectors
  std::vector<std::vector<int>> color_offsets;           // CSR offsets [sector][ncolors+1], host
  std::vector<Kokkos::View<int*>> d_color_sites_vec;    // Site indices [sector][sites_in_sector]
  // Current sector's color sites view (set in iterate_rejection before calling sweep_color_kokkos)
  // Derived classes use this in their sweep functors
  Kokkos::View<int*> d_color_sites;

  // === Per-site RNG streams for `sweep color/strict` (see random_park_kokkos.h)
  // Set iff sweepflag == COLOR_STRICT. Holds each site's Park/Miller seed,
  // carried across sweeps exactly as AppLattice::sweep_nomask_strict carries
  // siteseeds[] on the host, so the stream a site draws from depends on nothing
  // but the site itself -- no thread ids, no pool lock order. Indexed by site
  // index, the same `i` the sweep uses for spin. Empty when not strict.
  Kokkos::View<int*> d_siteseeds;
  int strict_rng;                        // 1 iff sweepflag == COLOR_STRICT
  void setup_siteseeds_kokkos();         // upload siteseeds[] after color lists

  // === Shared methods ===
  virtual void init_app() override;
  virtual void grow_app() override;
  void setup_color_lists_kokkos();
  virtual void iterate_rejection(double stoptime) override;

  // === Virtual methods derived classes MUST implement ===
  // Sync app-specific data (spin, quaternions, etc.)
  virtual void sync_app_data_to_device() = 0;
  virtual void sync_app_data_to_host() = 0;
  // Upload only the ghost range [nlocal,nall) after a forward exchange.
  // Default is the full upload; apps whose sync is a pure copy override it.
  virtual void sync_ghosts_to_device() { sync_app_data_to_device(); }
  void ensure_host_data_current();
  // Copy d[0,n) into host[0,n): a straight download.
  template <class T>
  static void download_range(T* host, const Kokkos::View<T*>& d, int n) {
    if (n <= 0) return;
    Kokkos::deep_copy(
        Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(host, n),
        Kokkos::subview(d, Kokkos::make_pair(0, n)));
  }
  // Copy host[first,last) into d[first,last): a straight range upload.
  template <class T>
  static void upload_range(const Kokkos::View<T*>& d, const T* host, int first, int last) {
    if (last <= first) return;
    const auto r = Kokkos::make_pair(first, last);
    Kokkos::deep_copy(Kokkos::subview(d, r),
        Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            host + first, last - first));
  }
  bool host_data_stale = false;

  // Allocate app-specific Kokkos views
  virtual void allocate_app_views() = 0;
  void allocate_device_views();
  int views_maxneigh = -1;

  // The actual sweep kernel for one color
  virtual void sweep_color_kokkos(int color, int start, int count) = 0;

protected:
  // Shared neighbor sync
  void sync_neighbors_to_device();
};

}

#endif
