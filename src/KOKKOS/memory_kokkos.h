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
#include <Kokkos_DualView.hpp>

namespace SPPARKS_NS {

class MemoryKokkos : public Memory {
public:
  MemoryKokkos(class SPPARKS *);
  ~MemoryKokkos();

  // Kokkos-aware allocation methods
  template<typename T>
  void create_kokkos(Kokkos::DualView<T*> &view, int n, const char *name) {
    view = Kokkos::DualView<T*>(name, n);
  }

  template<typename T>
  void create_kokkos(Kokkos::DualView<T**> &view, int n1, int n2, const char *name) {
    view = Kokkos::DualView<T**>(name, n1, n2);
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
