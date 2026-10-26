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
AppStyle(phasefield,AppPhaseField)

#else

#ifndef SPK_APP_PHASEFIELD_H
#define SPK_APP_PHASEFIELD_H

#include "app_lattice.h"
#include "phasefield_models.h"
#include <vector>

namespace SPPARKS_NS {

// Phase-field-only lattice app: no spins, no sweeps; every step advances
// the fields of one model (phasefield_models.h) by one explicit step.
// phasefield/potts(/table) derives from it (Potts spins over the
// two_phase or ch_heat model).
class AppPhaseField : public AppLattice {
 public:
  AppPhaseField(class SPPARKS *, int, char **);
  ~AppPhaseField();

  void input_app(char *, int, char **);
  void init_app();
  void setup_end_app();
  void grow_app();
  void *extract_app(char *);
  void app_update(double);
  double site_energy(int);

  // no events: the update-only loop never asks for them
  void site_event_rejection(int, class RandomPark *) {}
  double site_propensity(int) { return 0.0; }
  void site_event(int, class RandomPark *) {}

 protected:
  // for a child that parses its own app_style arguments and sets the model
  AppPhaseField(class SPPARKS *, int, char **, int parse);
  bool yes_no(const char *word, const char *command);   // else error

  enum Model { ALLEN_CAHN, CAHN_HILLIARD, GRAIN_GROWTH, CH_HEAT, TWO_PHASE } model;
  PFAllenCahn ac;
  PFCahnHilliard ch;
  PFGrainGrowth gg;
  PFCHHeat chh;
  PFTwoPhase tp;                    // phasefield/potts only
  int nfield, naux;
  double dt;

  FDStencil stencil;
  int print_stencil;
  int *phase;                       // i1 (i2 in phasefield/potts(/table))
  double *aux[PF_MAXA];             // all sites
  double *fnew[PF_MAXF];            // owned sites
  int nmax_app;
  int warned;

  // ch_heat tables, one per phase name
  std::vector<char *> phase_names;
  std::vector<class Table *> tables;
  std::vector<PFPhaseTable> ptab;
  std::vector<double> tab;
  double scale_free, scale_penalty;

  // run fn on the selected model struct
  template <class Fn> void with_model(Fn fn) {
    switch (model) {
    case ALLEN_CAHN: fn(ac); break;
    case CAHN_HILLIARD: fn(ch); break;
    case GRAIN_GROWTH: fn(gg); break;
    case CH_HEAT: fn(chh); break;
    case TWO_PHASE: fn(tp); break;
    }
  }
  PFCtx<FDHostNbr, double *, int *> host_ctx();
  int step_fields();
  void set_dt(double);
  void check_flags(int);
  void add_phase(const char *);
  void check_ch_heat();
};

}

#endif
#endif
