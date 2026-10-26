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

// define integer data types used by SPPARKS and associated size limits

// smallint = variables for on-procesor system (nlocal, nghost, etc)
// tagint = variables for site IDs (nglobal, id[], etc)
// bigint = variables for total system tallies (naccept, nreject, etc)

// smallint must be an int, as defined by C compiler
// tagint can be 32-bit or 64-bit int, must be >= smallint
// bigint can be 32-bit or 64-bit int, must be >= tagint

// MPI_SPK_TAGINT = MPI data type corresponding to a tagint
// MPI_SPK_BIGINT = MPI data type corresponding to a bigint

// NOTE: if your machine/MPI does not support "long long" ints,
//       but only "long" ints, then you will need to change
//       MPI_LONG_LONG to MPI_LONG, and atoll to atol

#ifndef SPK_SPKTYPE_H
#define SPK_SPKTYPE_H

#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

#include "limits.h"
#include "stdint.h"
#include "inttypes.h"

namespace SPPARKS_NS {

// default to 32-bit smallint and tagint, 64-bit bigint

#if !defined(SPPARKS_SMALLSMALL) && !defined(SPPARKS_BIGBIG) && !defined(SPPARKS_SMALLBIG)
#define SPPARKS_SMALLBIG
#endif

// default settings, models restricted to less than 2B sites
// 32-bit smallint and tagint, 64-bit bigint

#ifdef SPPARKS_SMALLBIG

typedef int smallint;
typedef int tagint;
typedef int64_t bigint;

#define MAXSMALLINT INT_MAX
#define MAXTAGINT INT_MAX
#define MAXBIGINT INT64_MAX

#define MPI_SPK_TAGINT MPI_INT
#define MPI_SPK_BIGINT MPI_LONG_LONG

#define TAGINT_FORMAT "%d"
#define BIGINT_FORMAT "%" PRId64

#define ATOTAGINT atoi
#define ATOBIGINT atoll

#endif

// for models with over 2B sites
// 32-bit smallint, 64-bit tagint and bigint

#ifdef SPPARKS_BIGBIG

typedef int smallint;
typedef int64_t tagint;
typedef int64_t bigint;

#define MAXSMALLINT INT_MAX
#define MAXTAGINT INT64_MAX
#define MAXBIGINT INT64_MAX

#define MPI_SPK_TAGINT MPI_LONG_LONG
#define MPI_SPK_BIGINT MPI_LONG_LONG

#define TAGINT_FORMAT "%" PRId64
#define BIGINT_FORMAT "%" PRId64

#define ATOTAGINT atoll
#define ATOBIGINT atoll

#endif

// for machines that do not support 64-bit ints
// 32-bit smallint and tagint and bigint

#ifdef SPPARKS_SMALLSMALL

typedef int smallint;
typedef int tagint;
typedef int bigint;

#define MAXSMALLINT INT_MAX
#define MAXTAGINT INT_MAX
#define MAXBIGINT INT_MAX

#define MPI_SPK_TAGINT MPI_INT
#define MPI_SPK_BIGINT MPI_INT

#define TAGINT_FORMAT "%d"
#define BIGINT_FORMAT "%d"

#define ATOTAGINT atoi
#define ATOBIGINT atoi

#endif

// Kokkos floating-point precision, named as in LAMMPS (CMake -DKOKKOS_PREC=
//   double|mixed|single -> SPPARKS_KOKKOS_DOUBLE_DOUBLE / SINGLE_DOUBLE /
//   SINGLE_SINGLE). KK_FLOAT is the precision of the hot-kernel math,
//   KK_ACC_FLOAT of the per-site sums it feeds:
//     double  double/double  production, and the bit-for-bit CPU reference
//     mixed   float/double   dev builds for desktop GPUs with a reduced FP64
//                            rate (RDNA/consumer); bits differ from FP64
//     single  float/float    also narrows the accumulators
//   Code using them must reproduce its previous arithmetic exactly in double.
//   Current users:
//   - additive/texture/kk device temperature math: Rosenthal/Moser kernels
//     and the HDF5 gather-and-blend (weights, nodal slice, raw samples) in
//     KK_FLOAT; the Moser quadrature sums in KK_ACC_FLOAT. The d_temp field,
//     plan PODs, site-minus-laser differences and absolute simulation times
//     stay double in every mode (~4e4 s on a 3e-6 s step: float cannot
//     bracket it).
//   - conduction temperature source (ConductionSolverKokkos): fluxes, the
//     H update, T(H) and the stored state (H, T, RKL2 buffers), all KK_FLOAT.
//     With float H is stored relative to H(T0) so it resolves small
//     increments near ambient; the offset is 0 in double.
//   - potts/quaternion/kk disorientation + Read-Shockley energy
//     (QuaternionOps::compute_disorientation / read_shockley), device and
//     host, in KK_FLOAT; the per-site energy sums in KK_ACC_FLOAT. Stored
//     quaternions, symmetry table and global energy totals stay double.
//   - potts/anneal/kk sweep: disorientation + Read-Shockley in KK_FLOAT,
//     the energy change in KK_ACC_FLOAT (anneal::ParamsT); stored
//     orientations, H and the label-merge disorientations stay double.
//   - phasefield/kk, phasefield/potts(/table)/kk field step: the models'
//     aux() and update() (phasefield_models.h, fd_stencil.h) in KK_FLOAT,
//     stencil and order-parameter sums in KK_ACC_FLOAT. Fields, aux fields
//     and the step's increments to them stay double; below double the
//     stencil takes neighbour differences from the doubles first (sum f_j -
//     2d f_i cancels in float). The Potts events and energies stay double.

#if !defined(SPPARKS_KOKKOS_SINGLE_SINGLE) && !defined(SPPARKS_KOKKOS_DOUBLE_DOUBLE) \
  && !defined(SPPARKS_KOKKOS_SINGLE_DOUBLE)
#define SPPARKS_KOKKOS_DOUBLE_DOUBLE
#endif

#if defined(SPPARKS_KOKKOS_SINGLE_SINGLE)       // single
typedef float KK_FLOAT;
typedef float KK_ACC_FLOAT;
#elif defined(SPPARKS_KOKKOS_SINGLE_DOUBLE)     // mixed
typedef float KK_FLOAT;
typedef double KK_ACC_FLOAT;
#else                                           // double
typedef double KK_FLOAT;
typedef double KK_ACC_FLOAT;
#endif

}

#endif
