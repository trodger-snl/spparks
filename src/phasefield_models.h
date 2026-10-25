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

// Phase-field models for app_style phasefield (and phasefield/kk). A model
// is one struct; the app is a generic explicit driver that, every step,
//   1. evaluates aux(): naux helper fields (e.g. the chemical potential mu)
//      on owned sites and ghosts with full neighbour lists, so a model may
//      take their Laplacian (naux > 0 asks the app for 2 ghost layers);
//   2. evaluates update(): the new value of every field at each owned site
//      (forward Euler is the model's own choice of formula).
// Both are header-only and callable on host and device: x.f[k] / x.a[k] are
// double* on the host and 1d Kokkos Views on the device, x.nbr is the
// neighbour accessor of fd_stencil.h, so the CPU and Kokkos apps run the same
// arithmetic. To add a model: write its struct here, give it a parse() line
// in AppPhaseField's constructor and a case in both apps' dispatch switches.

#ifndef SPK_PHASEFIELD_MODELS_H
#define SPK_PHASEFIELD_MODELS_H

#include "fd_stencil.h"
#include "app_phasefield_potts_table.h"   // pftable:: lookup, mu, c/T step
#include <math.h>

namespace SPPARKS_NS {

// ponytail: fixed field cap; sparse order parameters if grain growth needs more
#define PF_MAXF 32
#define PF_MAXA 1

// what a model sees at a site
template <class N, class D, class I>
struct PFCtx {
  FDStencil st;
  N nbr;
  D f[PF_MAXF];                 // fields d1..dN
  D a[PF_MAXA];                 // aux fields
  I phase;                      // i1 (tabulated models)
  const PFPhaseTable *pt;       // tables (tabulated models)
  const double *tab;
};

/* ----------------------------------------------------------------------
   Allen-Cahn, non-conserved order parameter phi (d1):
   f = W phi^2 (1-phi)^2,  dphi/dt = -L (f'(phi) - kappa lap phi)
------------------------------------------------------------------------- */

struct PFAllenCahn {
  double L, kappa, W, dt;
  static const int naux = 0;
  int nfield() const { return 1; }

  template <class C> FD_FN void aux(const C &, int, double *) const {}

  template <class C> FD_FN int update(const C &x, int i, double *out) const {
    const double p = x.f[0][i];
    const double fp = 2.0 * W * p * (1.0 - p) * (1.0 - 2.0 * p);
    out[0] = p - dt * L * (fp - kappa * x.st.lap(x.nbr, x.f[0], i));
    return 0;
  }

  template <class C> FD_FN double energy(const C &x, int i) const {
    const double p = x.f[0][i];
    return W * p * p * (1.0 - p) * (1.0 - p) +
           0.5 * kappa * x.st.grad_dot(x.nbr, x.f[0], x.f[0], i);
  }

  // forward-Euler bound: dt L (max|f''| + kappa lambda_max) <= 2
  double dt_max(const FDStencil &st) const {
    return 2.0 / (L * (2.0 * W + kappa * 4.0 * st.dim * st.inv_h2));
  }
};

/* ----------------------------------------------------------------------
   Cahn-Hilliard, conserved composition c (d1):
   f = W c^2 (1-c)^2,  mu = f'(c) - kappa lap c,  dc/dt = M lap mu
------------------------------------------------------------------------- */

struct PFCahnHilliard {
  double M, kappa, W, dt;
  static const int naux = 1;
  int nfield() const { return 1; }

  template <class C> FD_FN void aux(const C &x, int i, double *a) const {
    const double c = x.f[0][i];
    a[0] = 2.0 * W * c * (1.0 - c) * (1.0 - 2.0 * c) -
           kappa * x.st.lap(x.nbr, x.f[0], i);
  }

  template <class C> FD_FN int update(const C &x, int i, double *out) const {
    out[0] = x.f[0][i] + dt * M * x.st.lap(x.nbr, x.a[0], i);
    return 0;
  }

  template <class C> FD_FN double energy(const C &x, int i) const {
    const double c = x.f[0][i];
    return W * c * c * (1.0 - c) * (1.0 - c) +
           0.5 * kappa * x.st.grad_dot(x.nbr, x.f[0], x.f[0], i);
  }

  double dt_max(const FDStencil &st) const {
    const double lam = 4.0 * st.dim * st.inv_h2;
    return 2.0 / (M * lam * (2.0 * W + kappa * lam));
  }
};

/* ----------------------------------------------------------------------
   Fan-Chen grain growth, N non-conserved order parameters eta_k (d1..dN):
   f = sum_k (-alpha/2 eta_k^2 + beta/4 eta_k^4) + gamma sum_{k<l} eta_k^2 eta_l^2
   deta_k/dt = -L (-alpha eta_k + beta eta_k^3 + 2 gamma eta_k sum_{l!=k}
                   eta_l^2 - kappa lap eta_k)
------------------------------------------------------------------------- */

struct PFGrainGrowth {
  int n;
  double L, kappa, alpha, beta, gamma, dt;
  static const int naux = 0;
  int nfield() const { return n; }

  template <class C> FD_FN void aux(const C &, int, double *) const {}

  template <class C> FD_FN int update(const C &x, int i, double *out) const {
    double sum2 = 0.0;
    for (int k = 0; k < n; k++) sum2 += x.f[k][i] * x.f[k][i];
    for (int k = 0; k < n; k++) {
      const double e = x.f[k][i];
      const double fp = -alpha * e + beta * e * e * e + 2.0 * gamma * e * (sum2 - e * e);
      out[k] = e - dt * L * (fp - kappa * x.st.lap(x.nbr, x.f[k], i));
    }
    return 0;
  }

  template <class C> FD_FN double energy(const C &x, int i) const {
    double s2 = 0.0, s4 = 0.0, g = 0.0;
    for (int k = 0; k < n; k++) {
      const double e2 = x.f[k][i] * x.f[k][i];
      s2 += e2;
      s4 += e2 * e2;
      g += x.st.grad_dot(x.nbr, x.f[k], x.f[k], i);
    }
    return -0.5 * alpha * s2 + 0.25 * beta * s4 + 0.5 * gamma * (s2 * s2 - s4) +
           0.5 * kappa * g;
  }

  // bound on |f''| over eta in [0,1]: alpha + 3 beta + 2 gamma (n-1)
  double dt_max(const FDStencil &st) const {
    return 2.0 / (L * (alpha + 3.0 * beta + 2.0 * gamma * (n - 1) +
                       kappa * 4.0 * st.dim * st.inv_h2));
  }
};

/* ----------------------------------------------------------------------
   Cahn-Hilliard + coupled heat with tabulated G(c,T) per phase (i1):
   c (d1), T (d2); the equations of phasefield/potts/table (pftable::)
------------------------------------------------------------------------- */

struct PFCHHeat {
  PFParams q;                   // q.dt is the step
  static const int naux = 1;
  int nfield() const { return 2; }

  template <class C> FD_FN void aux(const C &x, int i, double *a) const {
    a[0] = pftable::chem_potential(x.st, x.nbr, x.phase, x.f[0], x.f[1], i,
                                   x.pt, x.tab, q);
  }

  template <class C> FD_FN int update(const C &x, int i, double *out) const {
    return pftable::step(x.st, x.nbr, x.f[0], x.f[1], x.a[0], i, q, out[0], out[1]);
  }

  template <class C> FD_FN double energy(const C &x, int i) const {
    return pftable::lookup(x.pt[x.phase[i]], x.tab, x.f[0][i], x.f[1][i], 0, q.penalty) +
           0.5 * q.ch_energy * x.st.grad_dot(x.nbr, x.f[0], x.f[0], i);
  }

  // composition (curvature of G unknown: kappa term only) and heat bounds
  double dt_max(const FDStencil &st) const {
    const double lam = 4.0 * st.dim * st.inv_h2;
    const double dc = 2.0 / (2.0 * q.mobility * q.ch_energy * lam * lam);
    const double dT = 2.0 * q.specific_heat / (q.conductivity * lam);
    return dc < dT ? dc : dT;
  }
};

}

#endif
