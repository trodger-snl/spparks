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

#include "spktype.h"
#include "mpi.h"
#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "lattice.h"
#include "domain.h"
#include "memory.h"
#include "error.h"

using namespace SPPARKS_NS;

// same as in other files

enum{NONE,LINE_2N,SQ_4N,SQ_8N,TRI,SC_6N,SC_26N,FCC,BCC,DIAMOND,
       FCC_OCTA_TETRA,RANDOM_1D,RANDOM_2D,RANDOM_3D};

/* ---------------------------------------------------------------------- */

Lattice::Lattice(SPPARKS *spk, int narg, char **arg) : Pointers(spk)
{
  // parse style arg

  if (narg < 1) error->all(FLERR,"Illegal lattice command");

  if (strcmp(arg[0],"none") == 0) style = NONE;
  else if (strcmp(arg[0],"line/2n") == 0) style = LINE_2N;
  else if (strcmp(arg[0],"sq/4n") == 0) style = SQ_4N;
  else if (strcmp(arg[0],"sq/8n") == 0) style = SQ_8N;
  else if (strcmp(arg[0],"tri") == 0) style = TRI;
  else if (strcmp(arg[0],"sc/6n") == 0) style = SC_6N;
  else if (strcmp(arg[0],"sc/26n") == 0) style = SC_26N;
  else if (strcmp(arg[0],"fcc") == 0) style = FCC;
  else if (strcmp(arg[0],"bcc") == 0) style = BCC;
  else if (strcmp(arg[0],"diamond") == 0) style = DIAMOND;
  else if (strcmp(arg[0],"fcc/octa/tetra") == 0) style = FCC_OCTA_TETRA;
  else if (strcmp(arg[0],"random/1d") == 0) style = RANDOM_1D;
  else if (strcmp(arg[0],"random/2d") == 0) style = RANDOM_2D;
  else if (strcmp(arg[0],"random/3d") == 0) style = RANDOM_3D;
  else error->all(FLERR,"Illegal lattice command");

  if (style == NONE) {
    if (narg > 1) error->all(FLERR,"Illegal lattice command");
    return;
  }

  if (style == LINE_2N || style == SQ_4N || style == SQ_8N ||
      style == TRI || style == SC_6N || style == SC_26N ||
      style == FCC || style == BCC || style == DIAMOND || 
      style == FCC_OCTA_TETRA) {
    if (narg != 2) error->all(FLERR,"Illegal lattice command");
    latconst = atof(arg[1]);
  }

  if (style == RANDOM_1D || style == RANDOM_2D || style == RANDOM_3D) {
    if (narg != 3) error->all(FLERR,"Illegal lattice command");
    latconst = 1.0;
    nrandom = ATOTAGINT(arg[1]);
    cutoff = atof(arg[2]);
  }

  // check dimensionality

  if ((style == LINE_2N || style == RANDOM_1D) && 
      domain->dimension != 1)
    error->all(FLERR,"Lattice style does not match dimension");
  if ((style == SQ_4N || style == SQ_8N || style == TRI || 
       style == RANDOM_2D) && 
      domain->dimension != 2)
    error->all(FLERR,"Lattice style does not match dimension");
  if ((style == SC_6N || style == SC_26N || style == FCC || 
       style == BCC || style == DIAMOND || style == FCC_OCTA_TETRA ||
       style == RANDOM_3D) && 
      domain->dimension != 3)
    error->all(FLERR,"Lattice style does not match dimension");

  // set basis atoms for each style

  nbasis = 0;
  basis = NULL;

  if (style == LINE_2N || style == SQ_4N || style == SQ_8N ||
      style == SC_6N || style == SC_26N) {
    add_basis(0.0,0.0,0.0);
  } else if (style == TRI) {
    add_basis(0.0,0.0,0.0);
    add_basis(0.5,0.5,0.0);
  } else if (style == BCC) {
    add_basis(0.0,0.0,0.0);
    add_basis(0.5,0.5,0.5);
  } else if (style == FCC) {
    add_basis(0.0,0.0,0.0);
    add_basis(0.0,0.5,0.5);
    add_basis(0.5,0.0,0.5);
    add_basis(0.5,0.5,0.0);
  } else if (style == DIAMOND) {
    add_basis(0.0,0.0,0.0);
    add_basis(0.0,0.5,0.5);
    add_basis(0.5,0.0,0.5);
    add_basis(0.5,0.5,0.0);
    add_basis(0.25,0.25,0.25);
    add_basis(0.25,0.75,0.75);
    add_basis(0.75,0.25,0.75);
    add_basis(0.75,0.75,0.25);
  } else if (style == FCC_OCTA_TETRA) {
    add_basis(0.0,0.0,0.0);
    add_basis(0.0,0.5,0.5);
    add_basis(0.5,0.0,0.5);
    add_basis(0.5,0.5,0.0);
    add_basis(0.5,0.0,0.0);
    add_basis(0.0,0.5,0.0);
    add_basis(0.0,0.0,0.5);
    add_basis(0.5,0.5,0.5);
    add_basis(0.25,0.25,0.25);
    add_basis(0.75,0.25,0.25);
    add_basis(0.25,0.75,0.25);
    add_basis(0.75,0.75,0.25);
    add_basis(0.25,0.25,0.75);
    add_basis(0.75,0.25,0.75);
    add_basis(0.25,0.75,0.75);
    add_basis(0.75,0.75,0.75);
  }

  // set defaults for optional args

  origin[0] = origin[1] = origin[2] = 0.0;

  orientx[0] = 1;  orientx[1] = 0;  orientx[2] = 0;
  orienty[0] = 0;  orienty[1] = 1;  orienty[2] = 0;
  orientz[0] = 0;  orientz[1] = 0;  orientz[2] = 1;

  a1[0] = 1.0;  a1[1] = 0.0;  a1[2] = 0.0;
  a2[0] = 0.0;  a2[1] = 1.0;  a2[2] = 0.0;
  a3[0] = 0.0;  a3[1] = 0.0;  a3[2] = 1.0;

  if (style == TRI) a2[1] = sqrt(3.0);

  // lattice spacings

  xlattice = a1[0]*latconst;
  ylattice = a2[1]*latconst;
  zlattice = a3[2]*latconst;
}

/* ---------------------------------------------------------------------- */

Lattice::~Lattice()
{
  memory->destroy(basis);
}

/* ----------------------------------------------------------------------
   add a basis atom to list
   x,y,z = fractional coords within unit cell
------------------------------------------------------------------------- */

void Lattice::add_basis(double x, double y, double z)
{
  memory->grow(basis,nbasis+1,3,"lattice:basis");
  basis[nbasis][0] = x;
  basis[nbasis][1] = y;
  basis[nbasis][2] = z;
  nbasis++;
}

/* ----------------------------------------------------------------------
   # of colors to partition lattice into
------------------------------------------------------------------------- */

int Lattice::ncolors(int delcolor)
{
  int n = 0;

  int nx = domain->nx;
  int ny = domain->ny;
  int nz = domain->nz;
  if (nx == 0 || ny == 0 || nz == 0)
    error->all(FLERR,"Cannot use coloring without domain nx,ny,nz defined");

  if (style == LINE_2N) {
    if (delcolor == 1) n = 2;
    if (nx % 2)
      error->all(FLERR,"Color stencil is incommensurate with lattice size");
  } else if (style == SQ_4N) {
    if (delcolor == 1) n = 2;
    if (nx % 2 || ny % 2)
      error->all(FLERR,"Color stencil is incommensurate with lattice size");
  } else if (style == SQ_8N) {
    n = (delcolor+1)*(delcolor+1);
    if (nx % (delcolor+1) || ny % (delcolor+1))
      error->all(FLERR,"Color stencil is incommensurate with lattice size");
  } else if (style == TRI) {
    if (delcolor == 1) n = 4;
    if (nx % 2)
      error->all(FLERR,"Color stencil is incommensurate with lattice size");
  } else if (style == SC_6N) {
    if (delcolor == 1) n = 2;
    if (nx % 2 || ny % 2 || nz % 2)
      error->all(FLERR,"Color stencil is incommensurate with lattice size");
  } else if (style == SC_26N) {
    n = (delcolor+1)*(delcolor+1)*(delcolor+1);
    if (nx % (delcolor+1) || ny % (delcolor+1) || nz % (delcolor+1))
      error->all(FLERR,"Color stencil is incommensurate with lattice size");
  } else if (style == FCC) {
    if (delcolor == 1) n = 4;
  } else if (style == BCC) {
    if (delcolor == 1) n = 2;
  }

  return n;
}

/* ----------------------------------------------------------------------
   # of colors to partition lattice into (any box size)
   Unlike ncolors(), this does not require dimensions divisible by color block
   Used for non-periodic boundaries where site IDs may not be contiguous
------------------------------------------------------------------------- */

int Lattice::ncolors_any(int delcolor)
{
  int n = 0;

  if (style == LINE_2N) {
    if (delcolor == 1) n = 2;
  } else if (style == SQ_4N) {
    if (delcolor == 1) n = 2;
  } else if (style == SQ_8N) {
    n = (delcolor+1)*(delcolor+1);
  } else if (style == TRI) {
    if (delcolor == 1) n = 4;
  } else if (style == SC_6N) {
    if (delcolor == 1) n = 2;
  } else if (style == SC_26N) {
    n = (delcolor+1)*(delcolor+1)*(delcolor+1);
  } else if (style == FCC) {
    if (delcolor == 1) n = 4;
  } else if (style == BCC) {
    if (delcolor == 1) n = 2;
  }

  return n;
}

/* ----------------------------------------------------------------------
   convert xyz coordinates to a color (1 to Ncolor)
   Works for any box size, does not require contiguous site IDs
   Uses lattice spacing and box origin to compute grid position
------------------------------------------------------------------------- */

int Lattice::xyz2color(double x, double y, double z, int delcolor)
{
  int ncolor1d,icolor;

  // compute grid indices from coordinates using lattice spacing
  // use box lower bounds as origin reference
  int i = static_cast<int>(floor((x - domain->boxxlo) / xlattice + 0.5));
  int j = static_cast<int>(floor((y - domain->boxylo) / ylattice + 0.5));
  int k = static_cast<int>(floor((z - domain->boxzlo) / zlattice + 0.5));

  if (style == LINE_2N) {
    icolor = i % 2;

  } else if (style == SQ_4N) {
    icolor = (i+j) % 2;

  } else if (style == SQ_8N) {
    ncolor1d = delcolor+1;
    // ensure positive modulo for negative indices
    int ii = ((i % ncolor1d) + ncolor1d) % ncolor1d;
    int jj = ((j % ncolor1d) + ncolor1d) % ncolor1d;
    icolor = ncolor1d*jj + ii;

  } else if (style == TRI) {
    // TRI lattice has 2 basis atoms per unit cell, use position within cell
    // color based on both grid position and sub-lattice
    int ii = ((i % 2) + 2) % 2;
    int jj = ((j % 2) + 2) % 2;
    icolor = 2*jj + ii;

  } else if (style == SC_6N) {
    icolor = (((i % 2) + 2) % 2 + ((j % 2) + 2) % 2 + ((k % 2) + 2) % 2) % 2;

  } else if (style == SC_26N) {
    ncolor1d = delcolor+1;
    // ensure positive modulo for negative indices
    int ii = ((i % ncolor1d) + ncolor1d) % ncolor1d;
    int jj = ((j % ncolor1d) + ncolor1d) % ncolor1d;
    int kk = ((k % ncolor1d) + ncolor1d) % ncolor1d;
    icolor = ncolor1d*ncolor1d*kk + ncolor1d*jj + ii;

  } else if (style == FCC) {
    // FCC has 4 basis atoms - determine which based on fractional position
    double fx = (x - domain->boxxlo) / xlattice - i;
    double fy = (y - domain->boxylo) / ylattice - j;
    double fz = (z - domain->boxzlo) / zlattice - k;
    // basis atoms at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    if (fabs(fx) < 0.25 && fabs(fy) < 0.25 && fabs(fz) < 0.25) icolor = 0;
    else if (fabs(fx) < 0.25 && fabs(fy-0.5) < 0.25 && fabs(fz-0.5) < 0.25) icolor = 1;
    else if (fabs(fx-0.5) < 0.25 && fabs(fy) < 0.25 && fabs(fz-0.5) < 0.25) icolor = 2;
    else icolor = 3;

  } else if (style == BCC) {
    // BCC has 2 basis atoms - determine which based on fractional position
    double fx = (x - domain->boxxlo) / xlattice - i;
    double fy = (y - domain->boxylo) / ylattice - j;
    double fz = (z - domain->boxzlo) / zlattice - k;
    // basis atoms at (0,0,0) and (0.5,0.5,0.5)
    if (fabs(fx) < 0.25 && fabs(fy) < 0.25 && fabs(fz) < 0.25) icolor = 0;
    else icolor = 1;
  }

  return icolor+1;
}

/* ----------------------------------------------------------------------
   convert a lattice ID (1 to Nsites) to a color (1 to Ncolor)
------------------------------------------------------------------------- */

int Lattice::id2color(tagint idsite, int delcolor)
{
  tagint i,j,k;
  int ncolor1d,icolor;

  int nx = domain->nx;
  int ny = domain->ny;
  int nz = domain->nz;
  if (nx == 0 || ny == 0 || nz == 0)
    error->all(FLERR,"Cannot use coloring without domain nx,ny,nz defined");

  idsite--;

  if (style == SQ_4N) {
    i = idsite % nx;
    j = idsite / nx;
    icolor = (i+j) % 2;

  } else if (style == SQ_8N) {
    ncolor1d = delcolor+1;
    i = idsite % nx;
    j = idsite / nx;
    icolor = ncolor1d*(j%ncolor1d) + i%ncolor1d;

  } else if (style == TRI) {
    icolor = idsite % 4;

  } else if (style == SC_6N) {
    i = idsite % nx;
    j = (idsite%(nx*ny)) / nx;
    k = idsite / (nx*ny);
    icolor = (i+j+k) % 2;

  } else if (style == SC_26N) {
    ncolor1d = delcolor+1;
    i = idsite % nx;
    j = (idsite%(nx*ny)) / nx;
    k = idsite / (nx*ny);
    icolor = ncolor1d*ncolor1d*(k%ncolor1d) + 
      ncolor1d*(j%ncolor1d) + i%ncolor1d;

  } else if (style == FCC) {
    icolor = idsite % 4;

  } else if (style == BCC) {
    icolor = idsite % 2;
  }

  return icolor+1;
}
