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

#ifdef DUMP_CLASS
#ifdef SPPARKS_HIGHFIVE

DumpStyle(conduction, DumpConductionKokkos)

#endif
#else

#ifndef SPK_DUMP_CONDUCTION_KOKKOS_H
#define SPK_DUMP_CONDUCTION_KOKKOS_H

#include "dump.h"

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   dump ID conduction DELTA FILE: the conduction source's whole grid (cells
   past the lattice box included) to one HDF5 file, overwritten at each
   dump; ConductionSolverKokkos::write_grid has the layout. `init grid FILE`
   on the next run's conduction source reads it back. additive/texture/kk.
------------------------------------------------------------------------- */

class DumpConductionKokkos : public Dump {
 public:
  DumpConductionKokkos(class SPPARKS *, int, char **);
  void write(double) override;

 private:
  void init_style() override;
  void write_header(bigint, double) override {}
  int count() override { return 0; }
  void pack(tagint *) override {}
  void write_data(int, double *) override {}
};

}

#endif
#endif
