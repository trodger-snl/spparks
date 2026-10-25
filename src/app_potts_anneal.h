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
AppStyle(potts/anneal,AppPottsAnneal)

#else

#ifndef SPK_APP_POTTS_ANNEAL_H
#define SPK_APP_POTTS_ANNEAL_H

#include "app_potts_quaternion.h"
#include "app_potts_rx.h"   // rx::seed, rx::nucleus, FD_FN, gw_mix
#include "crystalline_orientations/quaternion_ops.h"

namespace SPPARKS_NS {

// Heat treatment of a grain structure with an orientation per site (e.g. the
// additive/texture output), shared by potts/anneal (host arrays) and
// potts/anneal/kk (Kokkos views). Grains are labels (spin): sites of one
// label feel no boundary, so orientation gradients inside a grain survive.
namespace anneal {

enum {VOID = 5};                 // i2 of a void site (as additive/texture)
enum {MAXNEIGH = 32};

struct Params {
  const double *sym;             // symmetry quaternions, 4 per operator
  int nsym;
  double theta_cut;              // Read-Shockley cutoff (degrees)
  double temp, tinv;             // tinv = 0 at T = 0
  double theta_m;                // mobility M = 1 - exp(-5 (theta/theta_m)^4); 0 = off
};

// the per-site arrays: spin, phase (i2), q0..qz, stored energy
template <class I, class D>
struct Sites {
  I spin, phase;
  D q0, qx, qy, qz, h;
};

template <class I, class D>
FD_FN void get_q(const Sites<I, D> &s, int i, double *q)
{
  q[0] = s.q0[i];
  q[1] = s.qx[i];
  q[2] = s.qy[i];
  q[3] = s.qz[i];
}

// boundary energy of site i if it had label lab and orientation q: Read-
// Shockley over the non-void neighbours of another label
template <class NN, class N, class I, class D>
FD_FN double boundary(int i, int lab, const double *q, const NN &nn, const N &nbr,
                      const Sites<I, D> &s, const Params &p)
{
  double e = 0.0;
  for (int m = 0; m < nn[i]; m++) {
    const int k = nbr((size_t) i, m);
    if (s.phase[k] == VOID || s.spin[k] == lab) continue;
    double qk[4];
    get_q(s, k, qk);
    e += quaternion::read_shockley(
      quaternion::disorientation_deg(p.sym, p.nsym, q, qk), p.theta_cut);
  }
  return e;
}

// Site i takes the label, orientation and stored energy of a neighbour of a
// random distinct other label, by one Metropolis test on dE = d(boundary) +
// H_j - H_i, scaled by the mobility of the boundary it crosses. Voids never
// change and lend nothing. Returns 1 if the site changed.
template <class G, class NN, class N, class I, class D>
FD_FN int event(int i, G &g, const NN &nn, const N &nbr, const Sites<I, D> &s,
                const Params &p)
{
  if (s.phase[i] == VOID) return 0;
  const int lab = s.spin[i];
  int cand[MAXNEIGH];
  int nc = 0;
  for (int m = 0; m < nn[i]; m++) {
    const int k = nbr((size_t) i, m);
    if (s.phase[k] == VOID || s.spin[k] == lab) continue;
    int c = 0;
    while (c < nc && s.spin[cand[c]] != s.spin[k]) c++;
    if (c == nc) cand[nc++] = k;
  }
  if (nc == 0) return 0;

  int c = (int) (nc * g.uniform());
  if (c >= nc) c = nc - 1;
  const int j = cand[c];
  double qi[4], qj[4];
  get_q(s, i, qi);
  get_q(s, j, qj);

  const double de = boundary(i, s.spin[j], qj, nn, nbr, s, p) -
                    boundary(i, lab, qi, nn, nbr, s, p) + (s.h[j] - s.h[i]);
  double prob = 1.0;
  if (p.theta_m > 0.0) {
    const double r = quaternion::disorientation_deg(p.sym, p.nsym, qi, qj) / p.theta_m;
    prob = 1.0 - exp(-5.0 * r * r * r * r);
  }
  if (de > 0.0) {
    if (p.temp == 0.0) return 0;
    prob *= exp(-de * p.tinv);
  }
  if (prob < 1.0 && g.uniform() >= prob) return 0;

  s.spin[i] = s.spin[j];
  s.q0[i] = qj[0];
  s.qx[i] = qj[1];
  s.qy[i] = qj[2];
  s.qz[i] = qj[3];
  s.h[i] = s.h[j];
  return 1;
}

// one dilation pass of the nuclei seeded this step: a joinable site (not a
// void, deformed) takes the lowest new label (nspins + seed id, so the
// lowest seed id) among itself and its neighbours, or 0
template <class NN, class N, class I>
FD_FN int grow_nucleus(int i, const NN &nn, const N &nbr, const I &spin,
                       const I &phase, const I &seed, int nspins)
{
  if (phase[i] == VOID || spin[i] > nspins) return 0;
  int best = seed[i];
  for (int m = 0; m < nn[i]; m++) {
    const int v = seed[nbr((size_t) i, m)];
    if (v && (best == 0 || v < best)) best = v;
  }
  return best;
}

// recovery over one step: H *= f, f = exp(-r dt) made on the host
template <class D>
FD_FN void recover(int i, const D &h, double f) { h[i] *= f; }

}

class AppPottsAnneal : public AppPottsQuaternion {
 public:
  AppPottsAnneal(class SPPARKS *, int, char **);
  void grow_app() override;
  void init_app() override;
  void setup_end_app() override;
  void input_app(char *, int, char **) override;
  void app_update(double) override;
  void stats(char *) override;
  void stats_header(char *) override;

  double site_energy(int) override;
  void site_event_rejection(int, class RandomPark *) override;

 protected:
  int *phase, *seedspin;         // i2 (as additive/texture), i3 nucleus seeds
  double *h;                     // d5 stored energy
  double nrate;                  // nucleation rate per site per MCS
  int nradius;                   // nucleus = seed dilated nradius times
  double rrecover;               // recovery: dH/dt = -rrecover*H
  double theta_m;                // mobility cutoff (degrees), 0 = M = 1
  rx::Nucleation nuc;
  unsigned long long key;        // deck seed, for nucleation draws
  anneal::Params par;

  enum {NORIENT = 65536};
  std::vector<double> orient;    // nucleus orientations, 4 per entry

  unsigned long long salt() const;
  bigint nseeds() const;
  anneal::Sites<int *, double *> host_sites() const
  {
    return {spin, phase, q0, qx, qy, qz, h};
  }
  void stored_energy(int, char **);
};

}

#endif
#endif
