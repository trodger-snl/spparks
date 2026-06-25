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
#include "spparks.h"
#include "input.h"

#ifdef SPK_KOKKOS
#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#endif

using namespace SPPARKS_NS;

#ifdef SPK_KOKKOS
/* ----------------------------------------------------------------------
   Multi-GPU rank->device binding.

   Number of GPUs visible to the active Kokkos backend (respects
   ROCR_/HIP_/CUDA_VISIBLE_DEVICES). Returns -1 for host-only backends
   (Serial/OpenMP) -- nothing to bind.
------------------------------------------------------------------------- */
static int spk_visible_gpu_count()
{
#if defined(KOKKOS_ENABLE_HIP)
  int n = 0; return (hipGetDeviceCount(&n) == hipSuccess) ? n : -1;
#elif defined(KOKKOS_ENABLE_CUDA)
  int n = 0; return (cudaGetDeviceCount(&n) == cudaSuccess) ? n : -1;
#else
  return -1;
#endif
}

/* ----------------------------------------------------------------------
   Bind each MPI rank to a GPU by NODE-LOCAL rank (round-robin), so ranks
   spread across a node's GPUs instead of all piling onto device 0 (which
   oversubscribes one GPU and serializes its kernels). Sets KOKKOS_DEVICE_ID,
   which Kokkos::initialize() reads; a command-line --kokkos-device-id still
   overrides it (command line beats environment).

   Only acts when (a) the user/scheduler has NOT already chosen a device and
   (b) more than one GPU is visible -- so a scheduler that already pins one GPU
   per rank (count==1) and single-GPU/host builds are left untouched. Node-local
   rank comes from a shared-memory communicator split, which works on ANY
   launcher; Kokkos' own round-robin only triggers for a fixed set of launcher
   env vars (OMPI_/SLURM_/MPICH/...), so this closes that gap.
------------------------------------------------------------------------- */
static void spk_bind_gpu_by_local_rank(int argc, char **argv)
{
  if (std::getenv("KOKKOS_DEVICE_ID") || std::getenv("KOKKOS_MAP_DEVICE_ID_BY"))
    return;  // explicit user/scheduler device choice -- respect it
  for (int i = 1; i < argc; ++i)
    if (std::strncmp(argv[i], "--kokkos-device-id", 18) == 0) return;

  const int ndev = spk_visible_gpu_count();
  if (ndev <= 1) return;  // host backend or single visible GPU: nothing to bind

  MPI_Comm node;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
  int local_rank = 0, local_size = 0;
  MPI_Comm_rank(node, &local_rank);
  MPI_Comm_size(node, &local_size);
  MPI_Comm_free(&node);

  const int dev = local_rank % ndev;
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%d", dev);
  setenv("KOKKOS_DEVICE_ID", buf, 0);  // overwrite=0 (already checked unset above)

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank == 0) {
    std::printf("SPPARKS Kokkos: %d visible GPU(s); binding MPI ranks to GPUs by "
                "node-local rank (this node: %d rank(s) over %d GPU(s)). "
                "Override with KOKKOS_DEVICE_ID or --kokkos-device-id.\n",
                ndev, local_size, ndev);
    std::fflush(stdout);
  }
}
#endif

/* ----------------------------------------------------------------------
   main program to drive SPPARKS
------------------------------------------------------------------------- */

int main(int argc, char **argv)
{
  MPI_Init(&argc,&argv);

#ifdef SPK_KOKKOS
  spk_bind_gpu_by_local_rank(argc, argv);
  Kokkos::initialize(argc, argv);
#endif

  SPPARKS *spk = new SPPARKS(argc,argv,MPI_COMM_WORLD);
  spk->input->file();
  delete spk;

#ifdef SPK_KOKKOS
  Kokkos::finalize();
#endif

  MPI_Finalize();
}
