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

#ifdef DUMP_CLASS
#ifdef SPPARKS_HIGHFIVE

DumpStyle(hdf5, DumpHDF5)

#endif
#else

#ifndef SPK_DUMP_HDF5_H
#define SPK_DUMP_HDF5_H

#ifdef SPPARKS_HIGHFIVE

#include "dump.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// HighFive is forward declared so that style_dump.h (and therefore output.cpp)
// does not pull the whole HDF5 C++ header set in.  The destructor is declared
// here and defined in the .cpp, where HighFive::File is complete.
namespace HighFive {
class File;
}

namespace SPPARKS_NS {

// Read box offset/count ([z][y][x]) of a chunked 3-D int32 or float64
// dataset (an hid_t) into out, inflating the chunks on all threads.  Returns
// false if the layout is not handled or a read fails; the caller then falls
// back to an ordinary read.  Used by dump hdf5's read-back and "set hdf5".
bool hdf5_read_box_parallel(int64_t dataset, const size_t *offset, const size_t *count,
                            void *out);

// One chunk compressed ahead of an H5Dwrite_chunk: chunk-grid origin
// ([z][y][x] element offset), filter mask, filtered bytes.
struct HDF5Chunk {
  uint64_t off[3];
  uint32_t mask;
  std::vector<unsigned char> bytes;
};

// Ranks writing in turn write each chunk once, from its owner: one of the
// ranks whose slab touches it (chunk_owner). The other ranks' parts of the owner's chunks
// ([z][y][x] element offset and count, start in bytes) reach it beforehand.
struct HDF5Piece {
  uint64_t off[3], cnt[3], at;
};
struct HDF5Shared {
  int me;
  std::vector<uint64_t> slabs; // 6 per rank: z,y,x offset then z,y,x count
  std::vector<HDF5Piece> pieces;
  std::vector<unsigned char> bytes;
};

/* ----------------------------------------------------------------------
   dump hdf5: latest-state-only lattice output.

   One chunked 3-D dataset per field, indexed [z][y][x] so x is fastest and a
   rank's owned sites are a single contiguous hyperslab off app->iarray /
   app->darray with no repacking.  Each dump overwrites in place, so the file
   holds one value per site rather than a per-timestep history -- that is the
   whole point of the style, and it is why it needs neither the versioned block
   store nor the retention machinery that 'dump stitch' carries.

   Sites never written keep the dataset fill value (-1 for int fields, 0.0 for
   double), matching the "no value present" sentinels 'dump stitch' uses, so
   'set ... hdf5 ... skip_novalue' means the same thing as its stitch analogue.
------------------------------------------------------------------------- */

class DumpHDF5 : public Dump {
public:
  DumpHDF5(class SPPARKS *, int, char **);
  virtual ~DumpHDF5();
  void write(double);

private:
  int *fields;  // descriptor for each field (SITE, IARRAY, DARRAY)
  int *vtype;   // type of each field (INT, DOUBLE)
  int *vindex;  // index into app->iarray / app->darray
  std::vector<std::string> labels; // dataset name per field: site, iN, dN

  class AppLattice *applattice;
  std::shared_ptr<HighFive::File> file;

  bool opened;             // file created/reopened yet?
  int64_t origin[3];       // absolute lattice index of dataset element [0,0,0]
  int64_t dims[3];         // current global extent, x,y,z order
  int64_t chunk[3];        // chunk shape, x,y,z order
  bool user_origin;        // origin set via dump_modify
  bool user_extent;        // extent set via dump_modify
  bool user_chunk;         // chunk set via dump_modify
  int64_t req_origin[3];
  int64_t req_extent[3];
  int64_t req_chunk[3];
  int verbose;             // report collective-IO mode after each write
  int atomic;              // write a copy, then rename it over the file (else in place)
  int compress;            // deflate level of new datasets, 0 = off
  int turns;               // np>1: ranks write in turn (1), collectively (0), or auto (-1)

  // turns: each field's chunks this rank owns, compressed before the turns
  // start; packed_ok[i] = 0 if field i could not be packed
  std::vector<std::vector<HDF5Chunk>> packed;
  std::vector<char> packed_ok;

  // private methods

  void init_style();
  int count() { return 0; } // NOT USED by 'dump_hdf5'
  void pack(tagint *) {}
  virtual void write_header(bigint, double) {}
  virtual void write_data(int, double *) {}
  int parse_fields(int, char **);
  int modify_param(int, char **);

  void open_file(const char *fname);
  void write_fields(double, bool pack);
  void *field_data(int i);
  bool share(int i, int64_t dataset, const std::vector<size_t> &offset,
             const std::vector<size_t> &count, HDF5Shared &sh);
  bool use_turns();
  void take_turns(const char *fname);
  void my_slab(std::vector<size_t> &offset, std::vector<size_t> &count);
  void verify(const char *fname, double time);
  void stage_copy(const char *tmp);
  void rebuild_if_needed();
  void commit(const char *tmp);
  void sync_file(const char *fname);
  void choose_chunk(const int64_t *extent, int64_t *out);
  void ensure_dataset(const std::string &label, int type);
  void extend_to(const int64_t *needed);
};

} // namespace SPPARKS_NS

#endif // SPPARKS_HIGHFIVE
#endif // SPK_DUMP_HDF5_H
#endif
