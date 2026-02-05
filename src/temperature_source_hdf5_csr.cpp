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

#include "temperature_source_hdf5_csr.h"
#include "universe.h"
#include "error.h"
#include "highfive/highfive.hpp"
#include <numeric>

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

HDF5CSRTemperatureSource::HDF5CSRTemperatureSource(SPPARKS *spk) :
  HDF5UnstructuredTemperatureSource(spk)
{
}

/* ---------------------------------------------------------------------- */

void HDF5CSRTemperatureSource::read_time_temperature_data(
  HighFive::Group& grp,
  const std::vector<std::array<size_t, 2>>& nodeSlices)
{
  // ------------------------------------------------------------------
  // 1. Build offset slices from nodeSlices.  Each node slice [a,b)
  //    needs nodeOffsets[a..b] (inclusive), i.e. range [a, b+1).
  //    Adjacent selected chunks share a boundary index, so these
  //    ranges can overlap.  Merge them to avoid HDF5 deduplication
  //    that would shorten the read buffer.
  // ------------------------------------------------------------------
  std::vector<std::array<size_t, 2>> mergedOffsetSlices;
  for (const auto& ns : nodeSlices) {
    size_t start = ns[0];
    size_t end   = ns[1] + 1;
    if (!mergedOffsetSlices.empty() && start <= mergedOffsetSlices.back()[1]) {
      mergedOffsetSlices.back()[1] = std::max(mergedOffsetSlices.back()[1], end);
    } else {
      mergedOffsetSlices.push_back({start, end});
    }
  }

  // Precompute cumulative buffer positions for the merged slices so we
  // can map any global nodeOffset index to its position in the buffer.
  std::vector<size_t> sliceBufStart(mergedOffsetSlices.size());
  sliceBufStart[0] = 0;
  for (size_t i = 1; i < mergedOffsetSlices.size(); i++) {
    sliceBufStart[i] = sliceBufStart[i - 1] +
                       (mergedOffsetSlices[i - 1][1] - mergedOffsetSlices[i - 1][0]);
  }

  // ------------------------------------------------------------------
  // 2. Read merged nodeOffsets in a single HDF5 read.
  // ------------------------------------------------------------------
  std::vector<int64_t> rawOffsets;
  grp.getDataSet("nodeOffsets").select(
    build_hyperslab_1d(mergedOffsetSlices)).read(rawOffsets);

  // ------------------------------------------------------------------
  // 3. Map global nodeOffset indices back to buffer positions, then
  //    convert to data ranges in the flat times/temperatures arrays.
  // ------------------------------------------------------------------
  // Lambda: given a global index known to fall within one of the merged
  // slices, return its position in rawOffsets.
  size_t searchHint = 0;  // monotonically advancing hint
  auto toBufPos = [&](size_t globalIdx) -> size_t {
    for (size_t i = searchHint; i < mergedOffsetSlices.size(); i++) {
      if (globalIdx >= mergedOffsetSlices[i][0] &&
          globalIdx <  mergedOffsetSlices[i][1]) {
        searchHint = i;
        return sliceBufStart[i] + (globalIdx - mergedOffsetSlices[i][0]);
      }
    }
    throw std::runtime_error("nodeOffset index not in any merged slice");
  };

  std::vector<std::array<size_t, 2>> dataSlices;
  dataSlices.reserve(nodeSlices.size());
  size_t totalDataElements = 0;

  for (const auto& ns : nodeSlices) {
    int64_t dataStart = rawOffsets[toBufPos(ns[0])];
    int64_t dataEnd   = rawOffsets[toBufPos(ns[1])];
    dataSlices.push_back({static_cast<size_t>(dataStart),
                          static_cast<size_t>(dataEnd)});
    totalDataElements += static_cast<size_t>(dataEnd - dataStart);
  }

  // ------------------------------------------------------------------
  // 4. Read flat times and temperatures via 1D hyperslab — only actual
  //    data, no padding.
  // ------------------------------------------------------------------
  auto dataHyperSlab = build_hyperslab_1d(dataSlices);

  std::vector<double> timesFlat;
  grp.getDataSet("times").select(dataHyperSlab).read(timesFlat);

  std::vector<double> tempsFlat;
  grp.getDataSet("temperatures").select(dataHyperSlab).read(tempsFlat);

  // ------------------------------------------------------------------
  // 5. Build local row offsets from dataCounts (already loaded by
  //    load_layer before this method is called).
  // ------------------------------------------------------------------
  size_t nrows = dataCounts.size();
  std::vector<size_t> localOffsets(nrows + 1);
  localOffsets[0] = 0;
  for (size_t i = 0; i < nrows; i++) {
    localOffsets[i + 1] = localOffsets[i] + dataCounts[i];
  }

  // ------------------------------------------------------------------
  // 6. Construct CompressedArray2D directly — zero expansion.
  // ------------------------------------------------------------------
  times = CompressedArray2D<double>(std::move(timesFlat), std::move(localOffsets));

  // Rebuild localOffsets for temperatures (moved away above)
  std::vector<size_t> localOffsets2(nrows + 1);
  localOffsets2[0] = 0;
  for (size_t i = 0; i < nrows; i++) {
    localOffsets2[i + 1] = localOffsets2[i] + dataCounts[i];
  }
  temperatures = CompressedArray2D<double>(std::move(tempsFlat), std::move(localOffsets2));

  if (universe->me == 0) {
    fprintf(screen, "  CSR read: %zu data elements (no rectangular expansion)\n",
            totalDataElements);
  }
}
