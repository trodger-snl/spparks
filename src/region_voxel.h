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

#ifdef REGION_CLASS
RegionStyle(voxel,RegVoxel)

#else

#ifndef SPK_REGION_VOXEL_H
#define SPK_REGION_VOXEL_H

#include "region.h"
#include <vector>

namespace SPPARKS_NS {

class RegVoxel : public Region {
 public:
  RegVoxel(class SPPARKS *, int, char **);
  int match(double, double, double);

 private:
  int nx,ny,nz;
  double lo[3],cell[3];
  std::vector<unsigned char> v;     // x fastest, then y, then z
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Cannot open region voxel file

Self-explanatory.

E: Invalid region voxel file

The first line must be "voxel nx ny nz xlo ylo zlo dx dy dz" with
positive sizes, followed by nx*ny*nz bytes.

*/
