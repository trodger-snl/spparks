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
// arithmetic. To add a model: write its struct here, parse it in
// AppPhaseField's constructor and add a case to AppPhaseField::with_model.

#ifndef SPK_PHASEFIELD_MODELS_H
#define SPK_PHASEFIELD_MODELS_H

#include "fd_stencil.h"
#include <math.h>
#include <vector>

namespace SPPARKS_NS {

class Error;
class Table;

/* ----------------------------------------------------------------------
   Tabulated free energy G(c,T) per phase and the coupled c/T equations
   (ch_heat, phasefield/potts/table), host and device. Arrays are anything
   indexable with [] (host pointers or 1d Kokkos Views); nbr as in
   fd_stencil.h.
------------------------------------------------------------------------- */

// one phase's free energy table, stored in a flat array: G at tab[g],
// dG/dc at tab[dg], row-major nrow (T) x ncol (c from 0 to 1), even spacing
struct PFPhaseTable {
  int g, dg, nrow, ncol;
  int spin_lo, spin_hi;
  double t0, inv_dt, inv_dc;
};

struct PFParams {
  double dt, mobility, ch_energy, conductivity, heat_of_transport, specific_heat;
  double penalty;          // out-of-range c: G(edge) + penalty*excess^2
  double scale_potts;
  int enforce;             // clamp c to [0,1]
};

namespace pftable {

// bilinear interpolation in (T,c); dflag = 1 -> dG/dc. T outside the table
// extrapolates linearly from the edge cells.
FD_FN double lookup(const PFPhaseTable &p, const double *tab,
                    double c, double T, int dflag, double penalty)
{
  double excess = c < 0.0 ? c : (c > 1.0 ? c - 1.0 : 0.0);
  if (excess != 0.0 && dflag) return 2.0 * penalty * excess;

  double x = (T - p.t0) * p.inv_dt;
  int iT = (int) x;
  if (iT < 0) iT = 0;
  if (iT > p.nrow - 2) iT = p.nrow - 2;
  double fT = x - iT;

  const double *m = tab + (dflag ? p.dg : p.g) + iT * p.ncol;
  if (excess != 0.0) {
    int iC = c < 0.0 ? 0 : p.ncol - 1;
    return m[iC] + fT * (m[iC + p.ncol] - m[iC]) + penalty * excess * excess;
  }

  double y = c * p.inv_dc;
  int iC = (int) y;
  if (iC > p.ncol - 2) iC = p.ncol - 2;
  double fC = y - iC;
  const double *n = m + p.ncol;
  return m[iC] + fT * (n[iC] - m[iC]) + fC * (m[iC + 1] - m[iC]) +
         fC * fT * (m[iC] - n[iC] - m[iC + 1] + n[iC + 1]);
}

// chemical potential mu = dG/dc - kappa lap(c); valid where the axis
// neighbours exist (owned sites, ghosts with full neighbour lists)
template <class N, class P, class D>
FD_FN double chem_potential(const FDStencil &st, const N &nbr, const P &phase,
                            const D &c, const D &T, int i,
                            const PFPhaseTable *pt, const double *tab,
                            const PFParams &q)
{
  return lookup(pt[phase[i]], tab, c[i], T[i], 1, q.penalty) -
         q.ch_energy * st.lap(nbr, c, i);
}

// one explicit step of the coupled composition / heat equations at site i:
//   dc/dt    = M (2 lap(mu) + Q* div(grad T / T))
//   Cp dT/dt = 2 M Q* lap(mu) + k lap(T)
// div(grad T / T) in flux form (1/T averaged onto faces), so composition is
// conserved exactly. Returns flag bits: 1 = c left [0,1], 2 = T reached 0.
template <class D> struct Inverse {
  D f;
  FD_FN double operator[](int s) const { return 1.0 / f[s]; }
};

template <class N, class D>
FD_FN int step(const FDStencil &st, const N &nbr, const D &c, const D &T,
               const D &mu, int i, const PFParams &q, double &cnew, double &Tnew)
{
  double lapmu = st.lap(nbr, mu, i);
  double lapT = st.lap(nbr, T, i);
  double Ti = T[i];
  double soret = st.div_k_grad(nbr, T, Inverse<D>{T}, i);

  cnew = c[i] + q.dt * q.mobility * (2.0 * lapmu + q.heat_of_transport * soret);
  Tnew = Ti + (q.dt / q.specific_heat) *
              (2.0 * q.mobility * q.heat_of_transport * lapmu + q.conductivity * lapT);

  int flag = 0;
  if (q.enforce) cnew = cnew > 1.0 ? 1.0 : (cnew < 0.0 ? 0.0 : cnew);
  else if (cnew > 1.0 || cnew < 0.0) flag |= 1;
  if (!(Tnew > 0.0)) flag |= 2;
  return flag;
}

// host: check and pack tables (app_phasefield.cpp)
void pack_tables(SPPARKS_NS::Error *, SPPARKS_NS::Table **, int n, double scale_free,
                 std::vector<PFPhaseTable> &ptab, std::vector<double> &tab);

}

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
