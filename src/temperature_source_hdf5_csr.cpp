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
#ifdef SPPARKS_HIGHFIVE
#include "highfive/highfive.hpp"
#endif
#include <algorithm>
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
#ifndef SPPARKS_HIGHFIVE
  (void)grp;
  (void)nodeSlices;
  error->all(FLERR, "HDF5 temperature sources need a build with HighFive (-DSPPARKS_ENABLE_HIGHFIVE=ON)");
#else
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

  // ------------------------------------------------------------------
  // 4. Row offsets from dataCounts, which load_layer has zeroed for nodes
  //    whose series is never read; those rows stay empty.
  // ------------------------------------------------------------------
  std::vector<size_t> localOffsets(dataCounts.size() + 1, 0);
  std::inclusive_scan(dataCounts.begin(), dataCounts.end(), localOffsets.begin() + 1,
                      std::plus<size_t>(), size_t{0});

  // ------------------------------------------------------------------
  // 5. Per chunk: read its contiguous data range into a reused buffer and
  //    copy out the kept rows. Peak temporary = the largest chunk's data.
  //    Chunks with no kept row are not read. Raw-pointer reads: HighFive's
  //    vector reader knows only std::allocator.
  // ------------------------------------------------------------------
  SeriesVector<double> timesFlat(localOffsets.back());
  SeriesVector<double> tempsFlat(localOffsets.back());
  auto timesDset = grp.getDataSet("times");
  auto tempsDset = grp.getDataSet("temperatures");
  std::vector<double> buf;
  size_t row = 0;  // first local row of this chunk

  for (const auto& ns : nodeSlices) {
    const size_t n = ns[1] - ns[0];
    if (localOffsets[row + n] == localOffsets[row]) { row += n; continue; }

    // A chunk's nodeOffsets [ns0, ns1] lie contiguously in one merged slice.
    const size_t base = toBufPos(ns[0]);
    const size_t dataStart = static_cast<size_t>(rawOffsets[base]);
    buf.resize(static_cast<size_t>(rawOffsets[base + n]) - dataStart);

    auto strip = [&](HighFive::DataSet& dset, SeriesVector<double>& flat) {
      dset.select({dataStart}, {buf.size()}).read(buf.data());
      for (size_t r = 0; r < n; r++) {
        const unsigned count = dataCounts[row + r];
        if (count)
          std::copy_n(buf.data() + (static_cast<size_t>(rawOffsets[base + r]) - dataStart),
                      count, flat.data() + localOffsets[row + r]);
      }
    };
    strip(timesDset, timesFlat);
    strip(tempsDset, tempsFlat);
    row += n;
  }

  std::vector<size_t> localOffsets2 = localOffsets;
  times = CompressedArray2D<double>(std::move(timesFlat), std::move(localOffsets));
  temperatures = CompressedArray2D<double>(std::move(tempsFlat), std::move(localOffsets2));
#endif
}
