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
AppStyle(sinter,AppSinter)

#else

#ifndef APP_SINTER_H
#define APP_SINTER_H

#include <vector>

#include "app_lattice.h"
#include "fd_stencil.h"   // FD_FN

namespace SPPARKS_NS {

// The rKMC events of the sintering model, shared by sinter (host arrays) and
// sinter/kk (Kokkos views): S, NN index as spin[i], numneigh[i]; N as
// nbr((size_t) i, k); G draws with uniform(). The draws are those of the
// original app, so a site's stream gives the same events in both.
namespace sinter {

enum {FRAME = -1, VACANT = 0};   // spin values; grains are > 0
enum {MAXNEIGH = 32, MAXDE = 2*MAXNEIGH};
enum {GG, PM, VAC};              // the events, for T and boltz

struct Params {
  double gg, pm, anni;           // event frequencies, summing to 1
  double gb_factor;              // scales anni with the grain-boundary area
  double T[3];                   // temperatures
  double boltz[3][MAXDE+1];      // exp(-dE/T), dE = 0..MAXDE, if T > 0
  int sintering;                 // time > time_sinter_start: make vacancies
};

// # of neighbours with another spin
template <class NN, class N, class S>
FD_FN int energy(int i, const NN &nn, const N &nbr, const S &spin)
{
  const int s = spin[i];
  int e = 0;
  for (int k = 0; k < nn[i]; k++)
    if (spin[nbr((size_t) i, k)] != s) e++;
  return e;
}

// Metropolis test for event k: a draw only if uphill at T > 0. The table
// (made on the host) keeps exp() out of the device code, whose fast-math
// exp differs from the host's.
template <class G>
FD_FN bool accept(int ei, int ef, int k, const Params &p, G &g)
{
  if (ef <= ei) return true;
  if (p.T[k] == 0.0) return false;
  return !(g.uniform() > p.boltz[k][ef - ei]);
}

// the neighbouring grain spin (not old) that minimises site i's energy,
// starting from a random neighbour; e is i's energy with spin old on entry,
// the minimum on exit. Leaves spin[i] at the last spin tried.
template <class G, class NN, class N, class S>
FD_FN int min_energy_spin(int i, int old, G &g, const NN &nn, const N &nbr,
                          const S &spin, int &e)
{
  const int n = nn[i];
  int k0 = (int) (n * g.uniform());
  if (k0 >= n) k0 = n - 1;
  int unique[MAXNEIGH];
  int nu = 0, best = old;
  for (int k = 0; k < n; k++) {
    if (e <= n / 2) break;             // no spin can do better
    const int v = spin[nbr((size_t) i, (k + k0) % n)];
    if (v == FRAME || v == VACANT || v == old) continue;
    int m;
    for (m = 0; m < nu; m++)
      if (unique[m] == v) break;
    if (m < nu) continue;
    unique[nu++] = v;
    spin[i] = v;
    const int ev = energy(i, nn, nbr, spin);
    if (ev < e) {
      e = ev;
      best = v;
    }
  }
  return best;
}

// One rKMC attempt at site i: grain growth on a grain site; on a pore site
// either vacancy creation (once sintering) or pore migration, both a swap
// with a random grain neighbour j after which i takes its lowest-energy
// neighbouring grain spin. A vacancy is a pore with only grain neighbours;
// vac = j when one was made, else -1. Returns 1 if accepted.
template <class G, class NN, class N, class S>
FD_FN int event(int i, G &g, const NN &nn, const N &nbr, const S &spin,
                const Params &p, int &vac)
{
  vac = -1;
  const int old = spin[i];
  if (old == FRAME) return 0;

  if (old == VACANT) {
    const bool make = p.sintering && g.uniform() < p.anni * p.gb_factor;
    if (!make && !(g.uniform() < p.pm)) return 0;
    const int n = nn[i];
    int k = (int) (n * g.uniform());
    if (k >= n) k = n - 1;
    const int j = nbr((size_t) i, k);
    const int ns = spin[j];
    if (ns <= VACANT) return 0;

    const int ei = energy(i, nn, nbr, spin) + energy(j, nn, nbr, spin);
    spin[i] = ns;
    spin[j] = old;
    const int ej = energy(j, nn, nbr, spin);   // the same for any grain at i
    if (make && ej != nn[j]) {                 // j would not be a vacancy
      spin[i] = old;
      spin[j] = ns;
      return 0;
    }
    int ef = energy(i, nn, nbr, spin);
    spin[i] = min_energy_spin(i, ns, g, nn, nbr, spin, ef);
    if (!accept(ei, ef + ej, make ? VAC : PM, p, g)) {
      spin[i] = old;
      spin[j] = ns;
      return 0;
    }
    if (make) vac = j;
    return 1;
  }

  if (!(g.uniform() < p.gg)) return 0;
  const int ei = energy(i, nn, nbr, spin);
  int unique[MAXNEIGH];
  int nu = 0;
  for (int k = 0; k < nn[i]; k++) {
    const int v = spin[nbr((size_t) i, k)];
    if (v == FRAME || v == VACANT || v == old) continue;
    int m;
    for (m = 0; m < nu; m++)
      if (unique[m] == v) break;
    if (m == nu) unique[nu++] = v;
  }
  const int r = (int) (nu * g.uniform());
  if (r >= nu) return 0;
  spin[i] = unique[r];
  if (!accept(ei, energy(i, nn, nbr, spin), GG, p, g)) {
    spin[i] = old;
    return 0;
  }
  return 1;
}

// If j is a vacancy, the spin of most of its neighbours (ties to the first
// counted from neighbour k0), else 0
template <class NN, class N, class S>
FD_FN int adjacent_grain(int j, unsigned k0, const NN &nn, const N &nbr,
                         const S &spin)
{
  if (spin[j] != VACANT) return 0;
  const int n = nn[j];
  int unique[MAXNEIGH], count[MAXNEIGH];
  int nu = 0;
  for (int k = 0; k < n; k++) {
    const int v = spin[nbr((size_t) j, (k + k0) % n)];
    if (v == VACANT) return 0;
    if (v == FRAME) continue;
    int m;
    for (m = 0; m < nu; m++)
      if (unique[m] == v) break;
    if (m == nu) {
      unique[nu++] = v;
      count[m] = 0;
    }
    count[m]++;
  }
  int best = 0, most = 0;
  for (int m = 0; m < nu; m++)
    if (count[m] > most) {
      most = count[m];
      best = unique[m];
    }
  return best;
}

// the other grains beside grain site i, distinct, into u
template <class NN, class N, class S>
FD_FN int other_grains(int i, const NN &nn, const N &nbr, const S &spin,
                       int *u)
{
  const int s = spin[i];
  int nu = 0;
  for (int k = 0; k < nn[i]; k++) {
    const int v = spin[nbr((size_t) i, k)];
    if (v <= VACANT || v == s) continue;
    int m;
    for (m = 0; m < nu; m++)
      if (u[m] == v) break;
    if (m == nu) u[nu++] = v;
  }
  return nu;
}

// splitmix64: a draw keyed to (site, sweep), the same on any rank or device
FD_FN unsigned mix(unsigned long long x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return (unsigned) (x ^ (x >> 31));
}

// lattice cell of a site id (create_sites box numbering)
FD_FN void cell(tagint gid, int nbasis, int nx, int ny, int c[3])
{
  const tagint q = (gid - 1) / nbasis;
  c[0] = q % nx;
  c[1] = q / nx % ny;
  c[2] = q / ((tagint) nx * ny);
}

}

class AppSinter : public AppLattice {
  friend class DiagSinterDensity;
  friend class DiagSinterFreeEnergy;
  friend class DiagSinterFreeEnergyPore;

 public:
  AppSinter(class SPPARKS *, int, char **);

  void grow_app();
  void input_app(char *, int, char **);
  void init_app();
  void setup_app();
  void app_update(double);

  double site_energy(int);
  double site_propensity(int) {return 0.0;}
  void site_event(int, class RandomPark *) {}
  void site_event_rejection(int, class RandomPark *);

  void stats_header(char *);
  void stats(char *);

  int in_density_box(int) const;
  double calculate_density();

  struct Vac {                   // a vacancy to annihilate
    tagint gid;
    int grain;                   // its adjacent grain
    bool operator<(const Vac &o) const
      { return gid < o.gid || (gid == o.gid && grain < o.grain); }
  };

 protected:
  int *spin;
  sinter::Params par;
  double time_sinter_start;
  int nx, ny, nz, nbasis, dimension;
  unsigned long long key;        // deck seed, for sinter::mix

  std::vector<int> made;         // vacancies this sweep made (local indices)
  int vac_made;                  // annihilations since the last stats

  // grain-boundary area -> gb_factor (Garcia-Cardona 2011)
  double gb_ini, delta_t, density, next_gb;
  double gb_average();
  double gb_update(double);
  virtual void gb_faces(long long &, std::vector<unsigned long long> &);

  // the end-of-sweep annihilation, the same on any # of ranks; the hooks
  // are where sinter/kk works on its device copy
  void annihilate();
  void path(tagint, const long long *, std::vector<tagint> &);
  virtual int nmade() {return made.size();}
  virtual void update_ghosts();
  virtual void candidates(std::vector<Vac> &);
  virtual void grain_sums(const std::vector<int> &, std::vector<long long> &);
  virtual void get_sites(const std::vector<int> &, std::vector<int> &);
  virtual void set_sites(const std::vector<int> &, const std::vector<int> &);

  // owned site of a site id, or -1
  std::vector<int> cell2local;
  int clo[3], cn[3];
  int local_of(tagint) const;
  unsigned long long salt() const;   // sinter::mix key of this sweep

  void set_params(double);

  void overimpose_frame();
};

}

#endif
#endif
