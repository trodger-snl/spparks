/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   http://www.cs.sandia.gov/~sjplimp/spparks.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPPARKS directory.
------------------------------------------------------------------------- */

#ifdef APP_CLASS
AppStyle(phasefield/potts,AppPhaseFieldPottsTable)
AppStyle(phasefield/potts/table,AppPhaseFieldPottsTable)

#else

#ifndef SPK_APP_PHASEFIELD_POTTS_TABLE_H
#define SPK_APP_PHASEFIELD_POTTS_TABLE_H

#include "app_phasefield.h"
#include <math.h>
#include <vector>

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   Potts per-site physics, shared by the CPU app and its kk twin
   (the field physics is in phasefield_models.h)
------------------------------------------------------------------------- */

namespace pftable {

// phase whose spin range holds spin, else old
FD_FN int phase_of(const PFPhaseTable *pt, int nphases, int spin, int old)
{
  for (int p = 0; p < nphases; p++)
    if (spin >= pt[p].spin_lo && spin <= pt[p].spin_hi) return p;
  return old;
}

// bulk free energy (bulk(i,p), a PFBulk) + Potts interface energy of site
// i with spin s, phase p
template <class NN, class N, class S, class B>
FD_FN double site_energy(const NN &numneigh, const N &nbr, const S &spin,
                         const B &bulk, int i, int s, int p, double scale_potts)
{
  int eng = 0;
  for (int j = 0; j < numneigh[i]; j++)
    if (spin[nbr((size_t) i, j)] != s) eng++;
  return bulk(i, p) + eng * scale_potts;
}

// rKMC flip to a random unlike neighbour spin, Metropolis accept/reject.
// Same draws as AppPottsNeighOnly: one for the pick, one more only if uphill
// at T > 0. Returns 1 if the spin changed.
template <class Gen, class NN, class N, class S, class B>
FD_FN int rejection_event(int i, Gen &gen, const NN &numneigh, const N &nbr,
                          const S &spin, const S &phase, const B &bulk,
                          const PFPhaseTable *pt, int nphases, double scale_potts,
                          double temperature, double t_inverse)
{
  const int oldstate = spin[i];
  int unique[32];
  int nevent = 0;
  for (int j = 0; j < numneigh[i]; j++) {
    int value = spin[nbr((size_t) i, j)];
    if (value == oldstate) continue;
    int m;
    for (m = 0; m < nevent; m++)
      if (value == unique[m]) break;
    if (m < nevent) continue;
    unique[nevent++] = value;
  }
  if (nevent == 0) return 0;

  const int oldphase = phase[i];
  double einitial = site_energy(numneigh, nbr, spin, bulk, i, oldstate, oldphase,
                                scale_potts);
  int iran = (int) (nevent * gen.uniform());
  if (iran >= nevent) iran = nevent - 1;
  int s = unique[iran];
  int p = phase_of(pt, nphases, s, oldphase);
  double efinal = site_energy(numneigh, nbr, spin, bulk, i, s, p, scale_potts);

  if (efinal > einitial) {
    if (temperature == 0.0) return 0;
    if (gen.uniform() > exp((einitial - efinal) * t_inverse)) return 0;
  }
  spin[i] = s;
  phase[i] = p;
  return 1;
}

}

// Potts grains (spin = i1, phase = i2) over an AppPhaseField model: rKMC
// sweeps flip spins (and phases), each Potts step runs pf_nsteps field
// steps. phasefield/potts/table: ch_heat (c = d1, T = d2, tabulated G per
// phase); phasefield/potts: two_phase (c = d1, analytic f, alpha/beta).
class AppPhaseFieldPottsTable : public AppPhaseField {
 public:

  AppPhaseFieldPottsTable(class SPPARKS *, int, char **);

  void input_app(char *, int, char **);
  void init_app();
  void setup_end_app();
  void grow_app();

  void site_event_rejection(int,RandomPark *);
  void app_update(double);
  double site_energy(int);

 protected:

  int nspins;
  int *spin;
  double *c;      //concentration
  double *T;      //temperature (ch_heat)

  bool init_site_phase;
  std::vector<int> spin_lo, spin_hi;   // each phase's spin range
  int pf_nsteps;                  // PF steps per rkmc step (dt_rkmc)

  //variables for reseting the concentration (constant sink and source)
  std::vector<int> pf_resetlist;
  std::vector<double> pf_resetlistvals;
  int pf_resetversion = 0;          // bumped on each change of the list

  // run fn on the model the Potts events use
  template <class Fn> void with_potts_model(Fn fn) {
    if (model == TWO_PHASE) fn(tp);
    else fn(chh);
  }

  // phasefield/potts keywords, applied at the first init
  bool init_values = false, reset_xbounds = false;

  void parse_two_phase(int, char **);
  void set_site_phase(int);
  void set_pf_resetlist(int, char **);
};

}

#endif
#endif
