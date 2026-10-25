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

namespace SPPARKS_NS {

// Recrystallization events, shared by potts/rx (host arrays) and potts/rx/kk
// (Kokkos views). Site state is (spin, H, w): spin > nspins is a
// recrystallized grain, H the stored energy, w = exp(-H/T).
namespace rx {

enum {MAXNEIGH = 32};

struct Params {
  int nspins;
  double temp;
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

// site c seeds a nucleus this step: deformed, H >= hmin, keyed draw < prob
FD_FN bool seeds(tagint gid, int s, double hc, int nspins, double hmin,
                 double prob, unsigned long long salt)
{
  if (s > nspins || hc < hmin) return false;
  const unsigned long long r = gw_mix(salt ^ (unsigned long long) gid);
  return (double) (r >> 11) * (1.0 / 9007199254740992.0) < prob;
}

// the nucleus a deformed site i joins: the lowest id among i and its
// neighbours that seed (flag), or 0
template <class NN, class N, class S, class F, class I>
FD_FN tagint nucleus(int i, const NN &nn, const N &nbr, const S &spin,
                     const F &flag, const I &id, int nspins)
{
  if (spin[i] > nspins) return 0;
  tagint best = flag[i] ? id[i] : 0;
  for (int m = 0; m < nn[i]; m++) {
    const int j = nbr((size_t) i, m);
    if (flag[j] && (best == 0 || id[j] < best)) best = id[j];
  }
  return best;
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
  double *h, *w;                 // stored energy (d1), exp(-H/T) (d2)
  double nrate, hmin;            // nucleation rate per site per MCS, H floor
  unsigned long long key;        // deck seed, for the nucleation draws
  rx::Params par;

  unsigned long long salt() const;   // gw_mix key of this sweep
};

}

#endif
#endif
