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

// Second-order central differences for fields that live on the sites of a
// lattice app (one value per site, neighbours from the app's neighbor lists).
// The axis neighbours of a site are picked out of its neighbor list by cmap,
// so any cubic lattice whose list holds the +-x,y,z sites works (line/2n,
// sq/4n, sq/8n, sc/6n, sc/26n). Header-only and callable on host and device,
// so a CPU app and its Kokkos twin evaluate the same arithmetic:
//   nbr(i,k) -> index of neighbour k of site i  (int** via FDHostNbr, or a
//               2d Kokkos::View, indexed with size_t i)
//   f[s]     -> field value at site s           (double*, or a 1d View)
// A site's stencil reaches its axis neighbours only, so it is valid wherever
// those neighbours exist: owned sites and ghosts with a full neighbor list.

#ifndef SPK_FD_STENCIL_H
#define SPK_FD_STENCIL_H

#include <stddef.h>

#if defined(SPK_KOKKOS) && __has_include(<Kokkos_Macros.hpp>)
#include <Kokkos_Macros.hpp>
#endif
#ifdef KOKKOS_INLINE_FUNCTION
#define FD_FN KOKKOS_INLINE_FUNCTION
#else
#define FD_FN inline
#endif

namespace SPPARKS_NS {

struct FDHostNbr {
  int **neighbor;
  FD_FN int operator()(size_t i, int k) const { return neighbor[i][k]; }
};

struct FDStencil {
  int dim = 0;
  int cmap[6] = {0, 0, 0, 0, 0, 0};   // neighbour slot of -x,+x,-y,+y,-z,+z
  double h = 0.0, inv_h2 = 0.0, inv_2h = 0.0;

  // build cmap for the app's lattice and check it against site coords
  void setup(class SPPARKS *, class AppLattice *, int print);

  template <class N>
  FD_FN int nb(const N &nbr, int i, int k) const { return nbr((size_t) i, cmap[k]); }

  // Laplacian of f at site i
  template <class N, class F>
  FD_FN double lap(const N &nbr, const F &f, int i) const {
    double s = 0.0;
    for (int k = 0; k < 2 * dim; k++) s += f[nb(nbr, i, k)];
    return (s - 2.0 * dim * f[i]) * inv_h2;
  }

  // div(k grad f) at site i, k at each face the mean of its two sites:
  // conservative (each face's flux leaves one site and enters the other)
  template <class N, class F, class K>
  FD_FN double div_k_grad(const N &nbr, const F &f, const K &k, int i) const {
    double s = 0.0;
    for (int m = 0; m < 2 * dim; m++) {
      const int j = nb(nbr, i, m);
      s += 0.5 * (k[i] + k[j]) * (f[j] - f[i]);
    }
    return s * inv_h2;
  }

  // grad f . grad g at site i
  template <class N, class F, class G>
  FD_FN double grad_dot(const N &nbr, const F &f, const G &g, int i) const {
    double s = 0.0;
    for (int d = 0; d < dim; d++) {
      const int m = nb(nbr, i, 2 * d), p = nb(nbr, i, 2 * d + 1);
      s += (f[p] - f[m]) * (g[p] - g[m]);
    }
    return s * inv_2h * inv_2h;
  }
};

}

#endif
