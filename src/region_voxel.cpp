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

#include "mpi.h"
#include "math.h"
#include "stdio.h"
#include "string.h"
#include "region_voxel.h"
#include "error.h"

using namespace SPPARKS_NS;

/* ----------------------------------------------------------------------
   region ID voxel FILE: a 0/1 voxel mask, e.g. a part's footprint stacked
   layer by layer (tools/insitu_footprint.py). The file is one text line
   "voxel nx ny nz xlo ylo zlo dx dy dz" (distance units) and nx*ny*nz
   bytes, x fastest. Outside the grid in x/y nothing matches; below or above
   it the bottom or top slice extends (the part continues down to the plate).
------------------------------------------------------------------------- */

RegVoxel::RegVoxel(SPPARKS *spk, int narg, char **arg) :
  Region(spk, narg, arg)
{
  if (narg < 3) error->all(FLERR,"Illegal region voxel command");
  options(narg-3,&arg[3]);

  int me;
  MPI_Comm_rank(world,&me);
  double h[9];
  long n = 0;
  int ok = 1;
  FILE *fp = nullptr;
  if (me == 0) {
    fp = fopen(arg[2],"rb");
    if (!fp) ok = -1;
    else if (fscanf(fp,"voxel %d %d %d %lf %lf %lf %lf %lf %lf",&nx,&ny,&nz,
                    &h[3],&h[4],&h[5],&h[6],&h[7],&h[8]) != 9 ||
             nx < 1 || ny < 1 || nz < 1 ||
             h[6] <= 0.0 || h[7] <= 0.0 || h[8] <= 0.0 || fgetc(fp) != '\n') ok = 0;
    h[0] = nx; h[1] = ny; h[2] = nz;
  }
  MPI_Bcast(&ok,1,MPI_INT,0,world);
  if (ok < 0) error->all(FLERR,"Cannot open region voxel file");
  if (ok == 1) {
    MPI_Bcast(h,9,MPI_DOUBLE,0,world);
    nx = static_cast<int>(h[0]); ny = static_cast<int>(h[1]); nz = static_cast<int>(h[2]);
    n = (long) nx*ny*nz;
    v.resize(n);
    if (me == 0 && fread(v.data(),1,n,fp) != (size_t) n) ok = 0;
    MPI_Bcast(&ok,1,MPI_INT,0,world);
  }
  if (fp) fclose(fp);
  if (!ok) error->all(FLERR,"Invalid region voxel file");
  MPI_Bcast(v.data(),n,MPI_UNSIGNED_CHAR,0,world);

  const double scale[3] = {xscale,yscale,zscale};
  for (int d = 0; d < 3; d++) {
    lo[d] = scale[d]*h[3+d];
    cell[d] = scale[d]*h[6+d];
  }
  extent_xlo = lo[0]; extent_xhi = lo[0] + nx*cell[0];
  extent_ylo = lo[1]; extent_yhi = lo[1] + ny*cell[1];
  extent_zlo = -1.0e20; extent_zhi = 1.0e20;
}

/* ---------------------------------------------------------------------- */

int RegVoxel::match(double x, double y, double z)
{
  const int i = static_cast<int>(floor((x - lo[0])/cell[0]));
  const int j = static_cast<int>(floor((y - lo[1])/cell[1]));
  int k = static_cast<int>(floor((z - lo[2])/cell[2]));
  k = k < 0 ? 0 : (k >= nz ? nz-1 : k);
  int inside = 0;
  if (i >= 0 && i < nx && j >= 0 && j < ny) inside = v[((long) k*ny + j)*nx + i] != 0;
  return !(inside ^ interior);
}
