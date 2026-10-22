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
   # of colors to partition lattice into, for xyz2color()
   Unlike ncolors(), this does not need domain nx,ny,nz (site IDs may not
   be contiguous), but a periodic dim must still hold a whole number of
   color periods or the wrap puts two neighbors on one color
------------------------------------------------------------------------- */

int Lattice::ncolors_any(int delcolor)
{
  int n = 0;
  int period = 1;   // color period in unit cells along each colored dim

  if (style == LINE_2N) {
    if (delcolor == 1) n = period = 2;
  } else if (style == SQ_4N) {
    if (delcolor == 1) n = period = 2;
  } else if (style == SQ_8N) {
    period = delcolor+1;
    n = period*period;
  } else if (style == TRI) {
    if (delcolor == 1) n = 4;
    period = 2;
  } else if (style == SC_6N) {
    if (delcolor == 1) n = period = 2;
  } else if (style == SC_26N) {
    period = delcolor+1;
    n = period*period*period;
  } else if (style == FCC) {
    if (delcolor == 1) n = 4;
  } else if (style == BCC) {
    if (delcolor == 1) n = 2;
  }

  // TRI colors along x only; the others along every dim they span

  int dimension = domain->dimension;
  int check[3] = {1, dimension > 1 && style != TRI, dimension > 2};
  double prd[3] = {domain->xprd, domain->yprd, domain->zprd};
  double spacing[3] = {xlattice, ylattice, zlattice};
  for (int d = 0; d < 3; d++)
    if (n && check[d] && domain->periodicity[d] &&
        lround(prd[d]/spacing[d]) % period)
      error->all(FLERR,"Color stencil is incommensurate with lattice size");

  return n;
}

/* ----------------------------------------------------------------------
   convert xyz coordinates to a color (1 to Ncolor)
   Works for any box size, does not require contiguous site IDs
   Same colors as id2color() for a full periodic box
------------------------------------------------------------------------- */

int Lattice::xyz2color(double x, double y, double z, int delcolor)
{
  int ncolor1d,icolor;

  // half-cell index h = 2*cell + 2*basis offset, from the origin create_sites
  // uses (box lo in periodic dims, 0 otherwise); basis offsets are 0 or 0.5
  // for every colorable style, so h's parity is the offset

  double xyz[3] = {x, y, z};
  double lo[3] = {domain->boxxlo, domain->boxylo, domain->boxzlo};
  double spacing[3] = {xlattice, ylattice, zlattice};
  int cell[3],half[3];
  for (int d = 0; d < 3; d++) {
    double orig = domain->periodicity[d] ? lo[d] : 0.0;
    long h = lround(2.0*(xyz[d]-orig)/spacing[d]);
    half[d] = static_cast<int>(h & 1);
    cell[d] = static_cast<int>((h - half[d]) / 2);
  }

  // non-negative cell index mod m
  auto mod = [&](int d, int m) { return ((cell[d] % m) + m) % m; };

  if (style == LINE_2N) {
    icolor = mod(0,2);

  } else if (style == SQ_4N) {
    icolor = (mod(0,2) + mod(1,2)) % 2;

  } else if (style == SQ_8N) {
    ncolor1d = delcolor+1;
    icolor = ncolor1d*mod(1,ncolor1d) + mod(0,ncolor1d);

  } else if (style == TRI) {
    // basis (0,0) and (0.5,0.5); id2color's idsite % 4
    icolor = 2*mod(0,2) + half[0];

  } else if (style == SC_6N) {
    icolor = (mod(0,2) + mod(1,2) + mod(2,2)) % 2;

  } else if (style == SC_26N) {
    ncolor1d = delcolor+1;
    icolor = ncolor1d*ncolor1d*mod(2,ncolor1d) +
      ncolor1d*mod(1,ncolor1d) + mod(0,ncolor1d);

  } else if (style == FCC) {
    // basis index: (0,0,0) (0,.5,.5) (.5,0,.5) (.5,.5,0); id2color's idsite % 4
    icolor = 2*half[0] + half[1];

  } else if (style == BCC) {
    // basis index: (0,0,0) (.5,.5,.5)
    icolor = half[0];
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
  // tagint: the int product nx*ny overflows on a plane past 2^31 sites
  tagint nxy = (tagint) nx * ny;

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
    j = (idsite%nxy) / nx;
    k = idsite / nxy;
    icolor = (i+j+k) % 2;

  } else if (style == SC_26N) {
    ncolor1d = delcolor+1;
    i = idsite % nx;
    j = (idsite%nxy) / nx;
    k = idsite / nxy;
    icolor = ncolor1d*ncolor1d*(k%ncolor1d) + 
      ncolor1d*(j%ncolor1d) + i%ncolor1d;

  } else if (style == FCC) {
    icolor = idsite % 4;

  } else if (style == BCC) {
    icolor = idsite % 2;
  }

  return icolor+1;
}
