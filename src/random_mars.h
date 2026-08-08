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

#ifndef SPK_RANDOM_MARS_H
#define SPK_RANDOM_MARS_H

#include "pointers.h"

namespace SPPARKS_NS {

class RanMars : protected Pointers {
 public:
  RanMars(class SPPARKS *);
  ~RanMars();
  void init(int);
  double uniform();

  // The value given to the `seed` command. Exposed so a caller can derive its
  // own deterministic stream WITHOUT drawing from the master generator --
  // consuming a uniform here would shift every stream seeded after it.
  int initial_seed() const { return seed_value; }

 private:
  int initflag;
  int seed_value;
  int i97,j97;
  double c,cd,cm;
  double *u;
};

}

#endif

/* ERROR/WARNING messages:

E: Seed command has not been used

The seed command must be used if another command requires random
numbers.

*/
