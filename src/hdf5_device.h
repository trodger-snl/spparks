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

#ifndef SPK_HDF5_DEVICE_H
#define SPK_HDF5_DEVICE_H

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   HDF5DevicePlan is a plain (Kokkos-free) snapshot of the scalar state a
   device kernel needs to evaluate the per-site temperature for an HDF5
   unstructured-mesh source at one timestep. The variable-length data is NOT
   in here; it is uploaded separately to device Views:

     - The per-site element cache (node_idx[4] + barycentric weights[3] + a
       validity flag) is STATIC for the lifetime of a layer. It is re-uploaded
       only when `version` changes (the host rebuilds the cache at a layer
       boundary -- HDF5UnstructuredTemperatureSource::build_site_element_cache
       bumps its generation counter).

     - The per-node temperature slice (cached_nodal_temps, length n_nodes) is
       host-computed each step (time interpolation + optional nodal mesh
       smoothing) and re-uploaded every step.

   The per-site evaluation reproduces, device-side, the host fast-path
   barycentric interpolation in
   HDF5UnstructuredTemperatureSource::get_temperature_at_site_fast() -- same
   operation order, so the device field matches the host field bit-for-bit:

     T = nt[i0]
         + w0 * (nt[i1] - nt[i0])
         + w1 * (nt[i2] - nt[i0])
         + w2 * (nt[i3] - nt[i0])      (invalid site -> default_temp)

   The cache is built over OWNED + GHOST sites (nall): the chunk-load pads the
   subdomain by grid_cell_size_multiplier cells (>> the one-cell ghost layer),
   so every ghost site lies inside this rank's loaded mesh and gets a valid
   cache entry. The device kernel fills all nall sites POSITIONALLY (like the
   Rosenthal/Moser device paths) -- ghost T is correct on-device with no host
   round-trip and no T ghost comm, because every rank evaluates the same global
   thermal field at the same physical point (a ghost computed from the halo
   equals the owning rank's value by construction).

   valid is false (host path) when the source is not HDF5 or when
   solidification-band temperature_smooth is enabled (that smoothing is applied
   in update_temperatures(), which the device fill bypasses).
------------------------------------------------------------------------- */

struct HDF5DevicePlan {
  bool   valid = false;
  long   version = -1;       // cache generation; bump -> re-upload cache Views
  int    nsites = 0;         // device kernel range = cache entries = nall
                             // (owned + ghost; ghosts have valid entries via the
                             // chunk-load pad, so they fill positionally)
  int    n_nodes = 0;        // loaded nodes (length of the nodal-temp slice)
  double default_temp = 0.0; // ambient T for sites with no containing element
  double sim_time = 0.0;     // current simulation time (device-side time interp)
};

}

#endif
