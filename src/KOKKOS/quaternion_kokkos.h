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

#ifndef SPK_QUATERNION_KOKKOS_H
#define SPK_QUATERNION_KOKKOS_H

#include "app_lattice_kokkos.h"
#include "crystalline_orientations/quaternion.h"
#include "error.h"
#include "random_mars.h"
#include <cmath>
#include <cstdint>
#include <vector>

namespace SPPARKS_NS {

// Device spin + quaternion layer over a CPU quaternion app Base
// (AppPottsQuaternion or a subclass). Base owns spin and q0..qz (iarray[0],
// darray[0..3]), nspins and the symmetries; this layer adds their device
// copies, the host<->device syncs and the run-start orientation fill.
template <class Base>
class QuaternionKokkos : public LatticeKokkos<Base> {
  using LK = LatticeKokkos<Base>;

public:
  using LK::LK;
  using LK::nlocal; using LK::nghost; using LK::maxneigh;

  int spin_lo = 1;   // lowest valid spin (highest is nspins)

  Kokkos::View<int*> d_spin;
  Kokkos::View<double*> d_q0, d_qx, d_qy, d_qz;

  // Base's symmetries on the device (uploaded each init_app; read-only)
  Kokkos::View<double*> d_symmetries;
  int num_symmetries = 0;

  void grow_app() override {
    Base::grow_app();   // spin, q0..qz (and a subclass's own aliases)
    LK::grow_app();
  }

  void allocate_app_views() override {
    const int nall = nlocal + nghost;
    d_spin = this->site_view("spin", spin, nall);
    d_q0 = this->site_view("q0", q0, nall);
    d_qx = this->site_view("qx", qx, nall);
    d_qy = this->site_view("qy", qy, nall);
    d_qz = this->site_view("qz", qz, nall);
  }

  void init_app() override {
    this->dt_sweep = 1.0 / maxneigh;

    int flag = 0;
    for (int i = 0; i < nlocal; i++)
      if (spin[i] < spin_lo || spin[i] > nspins) flag = 1;
    int flagall;
    MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_SUM, world);
    if (flagall) error->all(FLERR, "One or more sites have invalid values");

    // Random orientation for every site without one, keyed to the site's
    // GLOBAL id (as the color/strict seeds in AppLattice::init()): repeatable
    // run to run and identical at any rank count. From ranmaster's stored
    // seed rather than ranmaster->uniform() so that no other stream shifts.
    const uint64_t qseed = (uint64_t) ranmaster->initial_seed();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nlocal; i++) {
      double norm = sqrt(q0[i]*q0[i] + qx[i]*qx[i] + qy[i]*qy[i] + qz[i]*qz[i]);
      if (norm < 1.0e-6) {
        quaternion::SiteOrientationRNG rng(qseed, (uint64_t) this->id[i]);
        std::vector<double> quat = quaternion::generate_unit_quaternion(rng);
        q0[i] = quat[0];
        qx[i] = quat[1];
        qy[i] = quat[2];
        qz[i] = quat[3];
      }
    }

    LK::init_app();

    num_symmetries = symmetries.size() / 4;
    d_symmetries = Kokkos::View<double*>("symmetries", symmetries.size());
    Kokkos::deep_copy(d_symmetries, Kokkos::View<const double*, Kokkos::HostSpace,
                      Kokkos::MemoryUnmanaged>(symmetries.data(), symmetries.size()));

    // Normalise once, in place on the host, then upload an exact copy: the
    // host and device hold the same bits from here on.
    normalize_host(nlocal + nghost, false);
    sync_app_data_to_device();
  }

  // An exact copy, apart from normalising clearly non-unit host input
  void sync_app_data_to_device() override {
    const int nall = nlocal + nghost;
    normalize_host(nall, true);
    LK::upload_range(d_spin, spin, 0, nall);
    LK::upload_range(d_q0, q0, 0, nall);
    LK::upload_range(d_qx, qx, 0, nall);
    LK::upload_range(d_qy, qy, 0, nall);
    LK::upload_range(d_qz, qz, 0, nall);
  }

  // Owned sites only: host ghosts come from the ghost exchange
  void sync_app_data_to_host() override {
    LK::download_range(spin, d_spin, nlocal);
    LK::download_range(q0, d_q0, nlocal);
    LK::download_range(qx, d_qx, nlocal);
    LK::download_range(qy, d_qy, nlocal);
    LK::download_range(qz, d_qz, nlocal);
  }

  // The ghost range after a forward exchange: a pure copy, since ghosts are
  // another rank's owned values, already unit (the run-start upload
  // normalises; sweeps only copy whole quaternions between sites).
  void sync_ghosts_to_device() override {
    const int first = nlocal, last = nlocal + nghost;
    LK::upload_range(d_spin, spin, first, last);
    LK::upload_range(d_q0, q0, first, last);
    LK::upload_range(d_qx, qx, first, last);
    LK::upload_range(d_qy, qy, first, last);
    LK::upload_range(d_qz, qz, first, last);
  }

protected:
  using Base::spin; using Base::q0; using Base::qx; using Base::qy; using Base::qz;
  using Base::nspins; using Base::symmetries;
  using LK::error; using LK::world; using LK::ranmaster;

  // Normalise host quaternions in place. off_unit_only: leave any quaternion
  // within 1e-10 of unit length untouched. Renormalising is not idempotent in
  // the last bit -- a value can flip between two neighbours on alternate
  // passes -- so values that came from the device must never be renormalised,
  // or the trajectory depends on how often the data happened to cross the bus
  // (rank count, number of runs). init_app does the one full pass; later
  // uploads only fix clearly non-unit input (a `set` between runs).
  // Per-element independent, so threading is bit-identical.
  void normalize_host(int n, bool off_unit_only) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
      double norm = sqrt(q0[i]*q0[i] + qx[i]*qx[i] + qy[i]*qy[i] + qz[i]*qz[i]);
      if (norm > 1.0e-15 && (!off_unit_only || fabs(norm - 1.0) > 1.0e-10)) {
        double inv_norm = 1.0 / norm;
        q0[i] *= inv_norm;
        qx[i] *= inv_norm;
        qy[i] *= inv_norm;
        qz[i] *= inv_norm;
      }
    }
  }
};

}

#endif
