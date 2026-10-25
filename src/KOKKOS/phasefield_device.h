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

#ifndef SPK_PHASEFIELD_DEVICE_H
#define SPK_PHASEFIELD_DEVICE_H

#include <Kokkos_Core.hpp>
#include "phasefield_models.h"
#include <utility>
#include <vector>

namespace SPPARKS_NS {

// The fields of an AppPhaseField on the device and one model step over them
// (AppPhaseField::step_fields), for phasefield/kk and phasefield/potts(/table)/kk.
struct PFDevice {
  using Ctx = PFCtx<Kokkos::View<int**>, Kokkos::View<double*>, Kokkos::View<int*>>;
  Kokkos::View<double*> f[PF_MAXF], fnew[PF_MAXF], aux[PF_MAXA];
  Kokkos::View<int*> phase;
  Kokkos::View<double*> tab;
  Kokkos::View<PFPhaseTable*> ptab;
  Kokkos::View<int> flags{"pf_flags"};

  void allocate(int nall, int nfield, int naux) {
    for (int k = 0; k < nfield; k++) {
      f[k] = Kokkos::View<double*>("pf_f", nall);
      fnew[k] = Kokkos::View<double*>("pf_fnew", nall);   // swapped with f
    }
    for (int k = 0; k < naux; k++) aux[k] = Kokkos::View<double*>("pf_aux", nall);
    phase = Kokkos::View<int*>("pf_phase", nall);
  }

  void upload_tables(const std::vector<PFPhaseTable> &pt, const std::vector<double> &t) {
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    tab = Kokkos::View<double*>("pf_tab", t.size());
    Kokkos::deep_copy(tab, Kokkos::View<const double*, Kokkos::HostSpace, Unmanaged>(
                             t.data(), t.size()));
    ptab = Kokkos::View<PFPhaseTable*>("pf_ptab", pt.size());
    Kokkos::deep_copy(ptab, Kokkos::View<const PFPhaseTable*, Kokkos::HostSpace, Unmanaged>(
                              pt.data(), pt.size()));
  }

  // flag bits since the last call (fences)
  int take_flags() {
    int v;
    Kokkos::deep_copy(v, flags);
    Kokkos::deep_copy(flags, 0);
    return v;
  }

  // what a model sees, over the current fields
  Ctx ctx(const FDStencil &st, const Kokkos::View<int**> &nbr) const {
    Ctx x;
    x.st = st;
    x.nbr = nbr;
    for (int k = 0; k < PF_MAXF; k++) x.f[k] = f[k];
    for (int k = 0; k < PF_MAXA; k++) x.a[k] = aux[k];
    x.phase = phase;
    x.pt = ptab.data();
    x.tab = tab.data();
    return x;
  }

  // aux on owned sites + full-list ghosts, then the update of the owned
  // sites; f and fnew swap, so f holds the new fields (their ghosts are
  // refreshed by the caller's next exchange)
  template <class M>
  void step(const M &m, const FDStencil &st, const Kokkos::View<int**> &nbr,
            const Kokkos::View<int*> &nn, int nloc, int nall, int mn, int nf) {
    const Ctx x = ctx(st, nbr);

    Kokkos::Array<Kokkos::View<double*>, PF_MAXF> fn;
    for (int k = 0; k < PF_MAXF; k++) fn[k] = fnew[k];
    auto fl = flags;
    const int na = M::naux;

    if (na)
      Kokkos::parallel_for("PFAux", nall, KOKKOS_LAMBDA(const int i) {
        if (i < nloc || nn(i) == mn) {
          double a[PF_MAXA];
          m.aux(x, i, a);
          for (int k = 0; k < na; k++) x.a[k](i) = a[k];
        }
      });
    Kokkos::parallel_for("PFUpdate", nloc, KOKKOS_LAMBDA(const int i) {
      double out[PF_MAXF];
      const int b = m.update(x, i, out);
      for (int k = 0; k < nf; k++) fn[k](i) = out[k];
      if (b) Kokkos::atomic_or(&fl(), b);
    });
    for (int k = 0; k < nf; k++) std::swap(f[k], fnew[k]);
  }
};

}

#endif
