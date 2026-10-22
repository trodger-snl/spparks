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

#ifdef COMMAND_CLASS
CommandStyle(set,Set)

#else

#ifndef SPK_SET_H
#define SPK_SET_H

#include "pointers.h"

#include <unordered_map>

namespace SPPARKS_NS {

class Set : protected Pointers {
 public:
  Set(class SPPARKS *);
  void command(int, char **);

 private:
  int siteindex;
  int count;
  int ivalue,ivaluelo,ivaluehi;
  double dvalue,dvaluelo,dvaluehi;
  int loopflag,regionflag,iregion;
  int skip_novalue;
  double fraction;
  char *filename,*tstamp;

  struct Condition {                     // list of if-test conditions
    int lhs,type,index,stride;
    int op;
    int irhs;
    tagint trhs;                         // for id: 64-bit with SPPARKS_BIGBIG
    double drhs;
  };
  Condition *cond;
  int ncondition;

  int latticeflag;
  class AppLattice *applattice;
  class AppOffLattice *appoff;

  typedef std::unordered_map<tagint,int> MyHash;
  typedef std::unordered_map<tagint,int>::iterator MyIterator;

  // local methods
  
  void set_single(int, int);
  void set_range(int, int);
  void set_displace(int, int);
  void set_stitch(int, int);
  void set_hdf5(int, int);
  void set_binary_file(int, int);

  // shared by set_stitch and set_hdf5: copy a per-site temp buffer into the
  // live array honoring loop/region/fraction/if/skip_novalue
  void apply_temp_buffer(int lhs, int32_t *itmp, double *dtmp,
                         int32_t novalue_i32, double novalue_f64);

  int condition(int);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Set command before sites exist

Self-explanatory.

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPPARKS to see the offending
line.

E: Setting a quantity application does not support

The application defines what variables it supports.  You cannot set a
variable with the set command on a variable that isn't supported.

E: Set command region ID does not exist

Self-explanatory.

E: Set if test on quantity application does not support

The application defines what variables it supports.  You cannot do an
if test with the set command on a variable that isn't supported.

*/
