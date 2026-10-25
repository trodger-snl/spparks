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
AppStyle(phasefield/potts/table,AppPhaseFieldPottsTable)

#else

#ifndef SPK_APP_PHASEFIELD_POTTS_TABLE_H
#define SPK_APP_PHASEFIELD_POTTS_TABLE_H

#include "app_potts_neighonly.h"
#include "fd_stencil.h"
#include <math.h>
#include <vector>

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   Per-site physics, shared by the CPU app and phasefield/potts/table/kk so
   both evaluate the same arithmetic. Arrays are anything indexable with []
   (host pointers or 1d Kokkos Views); nbr as in fd_stencil.h.
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

// phase whose spin range holds spin, else old
FD_FN int phase_of(const PFPhaseTable *pt, int nphases, int spin, int old)
{
  for (int p = 0; p < nphases; p++)
    if (spin >= pt[p].spin_lo && spin <= pt[p].spin_hi) return p;
  return old;
}

// bulk free energy + Potts interface energy of site i with spin s, phase p
template <class NN, class N, class S, class D>
FD_FN double site_energy(const NN &numneigh, const N &nbr, const S &spin,
                         const D &c, const D &T, int i, int s, int p,
                         const PFPhaseTable *pt, const double *tab,
                         const PFParams &q)
{
  int eng = 0;
  for (int j = 0; j < numneigh[i]; j++)
    if (spin[nbr((size_t) i, j)] != s) eng++;
  return lookup(pt[p], tab, c[i], T[i], 0, q.penalty) + eng * q.scale_potts;
}

// rKMC flip to a random unlike neighbour spin, Metropolis accept/reject.
// Same draws as AppPottsNeighOnly: one for the pick, one more only if uphill
// at T > 0. Returns 1 if the spin changed.
template <class Gen, class NN, class N, class S, class D>
FD_FN int rejection_event(int i, Gen &gen, const NN &numneigh, const N &nbr,
                          const S &spin, const S &phase, const D &c, const D &T,
                          const PFPhaseTable *pt, int nphases, const double *tab,
                          const PFParams &q, double temperature, double t_inverse)
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
  double einitial = site_energy(numneigh, nbr, spin, c, T, i, oldstate, oldphase,
                                pt, tab, q);
  int iran = (int) (nevent * gen.uniform());
  if (iran >= nevent) iran = nevent - 1;
  int s = unique[iran];
  int p = phase_of(pt, nphases, s, oldphase);
  double efinal = site_energy(numneigh, nbr, spin, c, T, i, s, p, pt, tab, q);

  if (efinal > einitial) {
    if (temperature == 0.0) return 0;
    if (gen.uniform() > exp((einitial - efinal) * t_inverse)) return 0;
  }
  spin[i] = s;
  phase[i] = p;
  return 1;
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
// div(grad T / T) = lap(T)/T - |grad T|^2/T^2. Returns flag bits:
// 1 = c left [0,1], 2 = T reached 0.
template <class N, class D>
FD_FN int step(const FDStencil &st, const N &nbr, const D &c, const D &T,
               const D &mu, int i, const PFParams &q, double &cnew, double &Tnew)
{
  double lapmu = st.lap(nbr, mu, i);
  double lapT = st.lap(nbr, T, i);
  double Ti = T[i];
  double soret = lapT / Ti - st.grad_dot(nbr, T, T, i) / (Ti * Ti);

  cnew = c[i] + q.dt * q.mobility * (2.0 * lapmu + q.heat_of_transport * soret);
  Tnew = Ti + (q.dt / q.specific_heat) *
              (2.0 * q.mobility * q.heat_of_transport * lapmu + q.conductivity * lapT);

  int flag = 0;
  if (q.enforce) cnew = cnew > 1.0 ? 1.0 : (cnew < 0.0 ? 0.0 : cnew);
  else if (cnew > 1.0 || cnew < 0.0) flag |= 1;
  if (!(Tnew > 0.0)) flag |= 2;
  return flag;
}

}

class AppPhaseFieldPottsTable : public AppPottsNeighOnly {
 public:

  AppPhaseFieldPottsTable(class SPPARKS *, int, char **);
  ~AppPhaseFieldPottsTable();

  void input_app(char *, int, char **);
  void init_app();
  void setup_end_app();
  void grow_app();
  void *extract_app(char *);

  void site_event_rejection(int,RandomPark *);
  void app_update(double);
  double site_energy(int);

 protected:

  double *c;      //concentration
  double *T;      //temperature
  double *mu;     //chemical potential, all sites - kept locally
  double *cnew;   //concentration following evolution - kept locally
  double *Tnew;   //temperature following evolution - kept locally
  int nmax_app;   //size of mu, cnew, Tnew

  int *phase;         //pointer to phase variable

  FDStencil stencil;              // axis neighbours, lattice spacing
  bool print_cmap;                // flag to print the connectivity map
  bool init_site_phase;
  double scale_free_energy;       // value by which to scale the free energy
  double scale_comp_energy_penalty; // value by which to scale the composition energy penalty when it is out of bounds

  PFParams params;                // mobility, CH factor, heat transport, ...

  struct Phase {
    char *id;
    int spin_start,spin_end;
    class Table *table;
  };

  int nphases;
  Phase **phases;           // list of phases
  std::vector<PFPhaseTable> ptab;   // per-phase table layout
  std::vector<double> tab;          // all phases' G and dG/dc, scaled

  //variables for evolving the phase field
  double dt_phasefield;           //time step for phasefield solver
  double dt_phasefield_mult;      // number of PF iterations per rkmc event

  //variables for reseting the concentration (constant sink and source)
  std::vector<int> pf_resetlist;
  std::vector<double> pf_resetlistvals;
  int pf_resetversion = 0;          // bumped on each change of the list

  //warn once when concentration has deviated outside [0,1]
  int warn_concentration_deviation_all;

  // methods unique to this class
  int find_phase(char *);
  void set_site_phase(int);
  void process_phases();
  void set_pf_resetlist(int, char **);
  void check_step_flags(int);
};

}

#endif
#endif
