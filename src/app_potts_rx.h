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
AppStyle(potts/rx,AppPottsRX)

#else

#ifndef SPK_APP_POTTS_RX_H
#define SPK_APP_POTTS_RX_H

#include "app_potts.h"
#include "fd_stencil.h"   // FD_FN, FDHostNbr
#include "grain_walk.h"   // gw_mix
#include <climits>

namespace SPPARKS_NS {

// Recrystallization events, shared by potts/rx (host arrays) and potts/rx/kk
// (Kokkos views). Site state is (spin, H, w): spin > nspins is a
// recrystallized grain, H the stored energy, w = exp(-H/T).
namespace rx {

enum {MAXNEIGH = 32};

struct Params {
  int nspins;
  double temp, tinv;              // tinv = 0 at T = 0
  double boltz[2*MAXNEIGH+1];   // exp(-d/T), d = -MAXNEIGH..MAXNEIGH
};

// Flip site i to the spin and stored energy of a random neighbour, by one
// Metropolis test on dE = (change in # unlike neighbours) + (H_j - H_i),
// J = 1. Uphill: accept with exp(-dE/T) = boltz[d] w_j / w_i, compared as
// products so host and device (and fast-math) decide alike.
template <class G, class NN, class N, class S, class D>
FD_FN int event(int i, G &g, const NN &nn, const N &nbr, const S &spin,
                const D &h, const D &w, const Params &p)
{
  const int n = nn[i];
  int k = (int) (n * g.uniform());
  if (k >= n) k = n - 1;
  const int j = nbr((size_t) i, k);
  const int old = spin[i], s = spin[j];
  if (s == old) return 0;

  int ei = 0, ef = 0;
  for (int m = 0; m < n; m++) {
    const int t = spin[nbr((size_t) i, m)];
    ei += t != old;
    ef += t != s;
  }
  const double de = (double) (ef - ei) + (h[j] - h[i]);
  if (de > 0.0) {
    if (p.temp == 0.0) return 0;
    if (g.uniform() * w[i] > p.boltz[ef - ei + MAXNEIGH] * w[j]) return 0;
  }
  spin[i] = s;
  h[i] = h[j];
  w[i] = w[j];
  return 1;
}

// stored-energy hardening over one step, dH/dt = rate - recovery*H, as
// H = (H + d)*f with d, f made on the host (exact, and no a*b+c for a
// compiler to fuse differently on host and device); w = exp(-H tinv),
// tinv = 0 at T = 0
template <class D>
FD_FN void harden(int i, const D &h, const D &w, double d, double f,
                  double tinv)
{
  h[i] = (h[i] + d) * f;
  w[i] = exp(-h[i] * tinv);
}

struct Nucleation {
  int nspins;
  double hmin, prob;
  int any;              // recrystallized grains may renucleate (hardening)
  int boundary;         // only sites with an unlike neighbour seed
  tagint idbase;        // nspins + nglobal: renucleated ids lie above it
};

// the spin of the nucleus site c seeds this step, or 0: c has H >= hmin
// (and is deformed, unless any), and a keyed draw < prob. A deformed seed
// takes nspins + its id; a renucleating one a hashed id above idbase.
template <class NN, class N, class S, class D>
FD_FN int seed(int c, tagint gid, const NN &nn, const N &nbr, const S &spin,
               const D &h, const Nucleation &q, unsigned long long salt)
{
  const int s = spin[c];
  if ((!q.any && s > q.nspins) || h[c] < q.hmin) return 0;
  if (q.boundary) {
    int m = 0;
    while (m < nn[c] && spin[nbr((size_t) c, m)] == s) m++;
    if (m == nn[c]) return 0;
  }
  const unsigned long long r = gw_mix(salt ^ (unsigned long long) gid);
  if ((double) (r >> 11) * (1.0 / 9007199254740992.0) >= q.prob) return 0;
  if (s <= q.nspins) return q.nspins + (int) gid;
  return (int) (q.idbase + 1 + gw_mix(r) % (unsigned long long)
                (INT_MAX - q.idbase));
}

// the new spin of site i: that of the lowest-id seed among i and its
// neighbours, or 0. Only deformed sites join, unless any.
template <class NN, class N, class S, class F, class I>
FD_FN int nucleus(int i, const NN &nn, const N &nbr, const S &spin,
                  const F &flag, const I &id, const Nucleation &q)
{
  if (!q.any && spin[i] > q.nspins) return 0;
  int c = flag[i] ? i : -1;
  for (int m = 0; m < nn[i]; m++) {
    const int j = nbr((size_t) i, m);
    if (flag[j] && (c < 0 || id[j] < id[c])) c = j;
  }
  return c < 0 ? 0 : flag[c];
}

}

class AppPottsRX : public AppPotts {
 public:
  AppPottsRX(class SPPARKS *, int, char **);
  void grow_app() override;
  void init_app() override;
  void setup_end_app() override;
  void app_update(double) override;
  void stats(char *) override;
  void stats_header(char *) override;

  double site_energy(int) override;
  void site_event_rejection(int, class RandomPark *) override;

 protected:
  int *seedspin;                 // i2: a nucleus seed's new spin, or 0
  double *h, *w;                 // stored energy (d1), exp(-H/T) (d2)
  double nrate;                  // nucleation rate per site per MCS
  double hrate, hrecover;        // hardening: dH/dt = hrate - hrecover*H
  rx::Nucleation nuc;
  unsigned long long key;        // deck seed, for the nucleation draws
  rx::Params par;

  unsigned long long salt() const;   // gw_mix key of this sweep
  void hardening_step(double, double &, double &) const;
};

}

#endif
#endif
