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

#ifndef SPK_MEMORY_KOKKOS_H
#define SPK_MEMORY_KOKKOS_H

#include "memory.h"
#include <Kokkos_Core.hpp>

namespace SPPARKS_NS {

// Type aliases for common Kokkos types
template<typename T>
using View1D = Kokkos::View<T*>;

template<typename T>
using View2D = Kokkos::View<T**>;

template<typename T>
using DualView1D = Kokkos::DualView<T*>;

template<typename T>
using DualView2D = Kokkos::DualView<T**>;

class MemoryKokkos : public Memory {
public:
  MemoryKokkos(class SPPARKS *);
  ~MemoryKokkos();

  // Kokkos-aware allocation methods
  template<typename T>
  void create_kokkos(DualView1D<T> &view, int n, const char *name) {
    view = DualView1D<T>(name, n);
  }

  template<typename T>
  void create_kokkos(DualView2D<T> &view, int n1, int n2, const char *name) {
    view = DualView2D<T>(name, n1, n2);
  }

  template<typename T>
  void destroy_kokkos(DualView1D<T> &view) {
    view = DualView1D<T>();
  }

  template<typename T>
  void destroy_kokkos(DualView2D<T> &view) {
    view = DualView2D<T>();
  }

  // Sync helpers
  template<typename ViewType>
  void sync_device(ViewType &view) {
    view.sync_device();
  }

  template<typename ViewType>
  void sync_host(ViewType &view) {
    view.sync_host();
  }

  template<typename ViewType>
  void modify_device(ViewType &view) {
    view.modify_device();
  }

  template<typename ViewType>
  void modify_host(ViewType &view) {
    view.modify_host();
  }
};

}

#endif
