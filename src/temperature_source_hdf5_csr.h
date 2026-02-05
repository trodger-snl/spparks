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

#ifndef SPK_TEMPERATURE_SOURCE_HDF5_CSR_H
#define SPK_TEMPERATURE_SOURCE_HDF5_CSR_H

#include "temperature_source_hdf5_unstructured.h"

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   HDF5 CSR (Compressed Sparse Row) temperature source

   Reads CSR-format HDF5 files where times and temperatures are stored
   as flat 1D arrays with a nodeOffsets index, eliminating the padded
   rectangular allocation that causes OOM for large datasets.

   File format differences from hdf5_unstructured:
     temperatures: (total_count,) float64   (was (N_nodes, maxData) float64)
     times:        (total_count,) float64   (was (N_nodes, maxData) float64)
     nodeOffsets:  (N_nodes+1,)   int64     (new dataset)
   All other datasets (nodeCoords, elementToNode, dataCounts, etc.) are identical.
------------------------------------------------------------------------- */

class HDF5CSRTemperatureSource : public HDF5UnstructuredTemperatureSource {
public:
  HDF5CSRTemperatureSource(class SPPARKS *spk);
  virtual ~HDF5CSRTemperatureSource() = default;
  virtual std::string get_source_type() const override { return "hdf5_csr"; }

protected:
  virtual void read_time_temperature_data(
      HighFive::Group& grp,
      const std::vector<std::array<size_t, 2>>& nodeSlices) override;
};

}

#endif
