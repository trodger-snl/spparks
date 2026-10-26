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
// potts/anneal/kk (Kokkos views). Grains are labels (spin). By default sites
// of one label feel no boundary, so orientation gradients inside a grain
// survive; with energy disorientation every bond costs Read-Shockley energy,
// so they evolve (subgrain coarsening) and labels are only bookkeeping.
namespace anneal {

enum {VOID = 5};                 // i2 of a void site (as additive/texture)
enum {MAXNEIGH = 32};

// R: precision of the disorientation + Read-Shockley math, A: of the energy
// sums (KK_FLOAT / KK_ACC_FLOAT in potts/anneal/kk, double on the CPU).
// Stored orientations and H stay double.
template <class R, class A = R>
struct ParamsT {
  const R *sym;                  // symmetry quaternions, 4 per operator
  int nsym;
  R theta_cut;                   // Read-Shockley cutoff (degrees)
  double temp, tinv;             // tinv = 0 at T = 0
  R theta_m;                     // mobility M = 1 - exp(-5 (theta/theta_m)^4); 0 = off
  int disorient;                 // 1 = every bond costs energy, not only unlike labels
};
using Params = ParamsT<double>;

// the per-site arrays: spin, phase (i2), q0..qz, stored energy
template <class I, class D>
struct Sites {
  I spin, phase;
  D q0, qx, qy, qz, h;
};

template <class R, class I, class D>
FD_FN void get_q(const Sites<I, D> &s, int i, R *q)
{
  q[0] = R(s.q0[i]);
  q[1] = R(s.qx[i]);
  q[2] = R(s.qy[i]);
  q[3] = R(s.qz[i]);
}

// boundary energy of site i if it had label lab and orientation q: Read-
// Shockley over the non-void neighbours of another label (of any label
// with disorient)
template <class R, class A, class NN, class N, class I, class D>
FD_FN A boundary(int i, int lab, const R *q, const NN &nn, const N &nbr,
                 const Sites<I, D> &s, const ParamsT<R, A> &p)
{
  A e = A(0);
  for (int m = 0; m < nn[i]; m++) {
    const int k = nbr((size_t) i, m);
    if (s.phase[k] == VOID || (!p.disorient && s.spin[k] == lab)) continue;
    R qk[4];
    get_q(s, k, qk);
    e += quaternion::read_shockley(
      quaternion::disorientation_deg(p.sym, p.nsym, q, qk), p.theta_cut);
  }
  return e;
}

// Site i takes the label, orientation and stored energy of a neighbour of a
// random distinct other label, by one Metropolis test on dE = d(boundary) +
// H_j - H_i, scaled by the mobility of the boundary it crosses. With
// disorient the candidates are all neighbours of another label or
// orientation, and one of the same label lends only its orientation (H
// stays: the stored energy below the voxel does not move with a subgrain
// boundary). Voids never change and lend nothing. Returns 1 if the site
// changed.
template <class R, class A, class G, class NN, class N, class I, class D>
FD_FN int event(int i, G &g, const NN &nn, const N &nbr, const Sites<I, D> &s,
                const ParamsT<R, A> &p)
{
  if (s.phase[i] == VOID) return 0;
  const int lab = s.spin[i];
  int cand[MAXNEIGH];
  int nc = 0;
  for (int m = 0; m < nn[i]; m++) {
    const int k = nbr((size_t) i, m);
    if (s.phase[k] == VOID) continue;
    if (p.disorient) {
      if (s.spin[k] != lab || s.q0[k] != s.q0[i] || s.qx[k] != s.qx[i] ||
          s.qy[k] != s.qy[i] || s.qz[k] != s.qz[i]) cand[nc++] = k;
      continue;
    }
    if (s.spin[k] == lab) continue;
    int c = 0;
    while (c < nc && s.spin[cand[c]] != s.spin[k]) c++;
    if (c == nc) cand[nc++] = k;
  }
  if (nc == 0) return 0;

  int c = (int) (nc * g.uniform());
  if (c >= nc) c = nc - 1;
  const int j = cand[c];
  R qi[4], qj[4];
  get_q(s, i, qi);
  get_q(s, j, qj);

  const int same = s.spin[j] == lab;
  const A de = boundary(i, s.spin[j], qj, nn, nbr, s, p) -
               boundary(i, lab, qi, nn, nbr, s, p) +
               A(same ? 0.0 : s.h[j] - s.h[i]);
  double prob = 1.0;
  if (p.theta_m > R(0)) {
    const double r = quaternion::disorientation_deg(p.sym, p.nsym, qi, qj) / p.theta_m;
    prob = 1.0 - exp(-5.0 * r * r * r * r);
  }
  if (de > 0.0) {
    if (p.temp == 0.0) return 0;
    prob *= exp(-de * p.tinv);
  }
  if (prob < 1.0 && g.uniform() >= prob) return 0;

  s.spin[i] = s.spin[j];
  s.q0[i] = s.q0[j];
  s.qx[i] = s.qx[j];
  s.qy[i] = s.qy[j];
  s.qz[i] = s.qz[j];
  if (!same) s.h[i] = s.h[j];
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

// merge: is the bond i-k counted (both non-void, different labels, both
// deformed or both recrystallized)
template <class I>
FD_FN bool merge_counted(int i, int k, const I &spin, const I &phase, int nspins)
{
  return phase[i] != VOID && phase[k] != VOID && spin[k] != spin[i] &&
         (spin[i] <= nspins) == (spin[k] <= nspins);
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
  int disorient;                 // energy disorientation (1) or labels (0)
  double merge_theta;            // merge labels below this disorientation, 0 = off
  int merge_every;               // ... at each run's start and every N MCS (0 = start only)
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
  void keywords(int, char **, const char *);
 public:
  void merge_labels(int);
  std::vector<int> merge_resolve(const std::vector<int> &, int);   // and kk's
};

}

#endif
#endif
