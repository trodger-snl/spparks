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

#include "dump_hdf5.h"

#ifdef SPPARKS_HIGHFIVE

#include "app.h"
#include "app_lattice.h"
#include "domain.h"
#include "error.h"
#include "lattice.h"
#include "memory.h"
#include "spktype.h"

#include "highfive/highfive.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <map>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/fs.h> // FICLONE
#include <sys/ioctl.h>
#endif
#ifdef SPPARKS_ZLIB
#include <zlib.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace SPPARKS_NS;

// customize by adding keyword to 1st enum
// (kept identical to dump_stitch/dump_text so field parsing reads the same)

enum { ID, SITE, X, Y, Z, ENERGY, PROPENSITY, IARRAY, DARRAY };
enum { INT, DOUBLE, TAGINT };

static const int SPPARKS_HDF5_DUMP_VERSION = 1;

/* ---------------------------------------------------------------------- */

DumpHDF5::DumpHDF5(SPPARKS *spk, int narg, char **arg) : Dump(spk, narg, arg) {
  if (multifile || multiproc)
    error->all(FLERR, "Dump hdf5 cannot use '*' or '%' wildcards");

  if (narg < 5)
    error->all(FLERR, "Illegal dump hdf5 command");

  size_one = narg - 4;
  fields = new int[size_one];
  vtype = new int[size_one];
  vindex = new int[size_one];
  labels.resize(size_one);

  applattice = NULL;
  opened = false;
  user_origin = user_extent = user_chunk = false;
  verbose = 0;
  atomic = 0;      // in place: no whole-file copy per dump (see dump_modify atomic)
  compress = 1;    // deflate level 1: cheap now that chunks deflate on all threads
  turns = -1;
  for (int d = 0; d < 3; d++) {
    origin[d] = 0;
    dims[d] = 0;
    chunk[d] = 0;
    req_origin[d] = 0;
    req_extent[d] = 0;
    req_chunk[d] = 0;
  }

  parse_fields(narg, arg);

  for (int i = 0; i < size_one; i++)
    if (INT != vtype[i] && DOUBLE != vtype[i])
      error->all(FLERR, "Dump hdf5 only supports 'integer' or 'double' values.");
}

/* ---------------------------------------------------------------------- */

DumpHDF5::~DumpHDF5() {
  delete[] fields;
  delete[] vtype;
  delete[] vindex;
  // shared_ptr<HighFive::File> is destroyed here, where the type is complete;
  // that closes the file and flushes metadata.
  file.reset();
}

/* ---------------------------------------------------------------------- */

void DumpHDF5::init_style() {
  if (app->appclass != App::LATTICE)
    error->all(FLERR, "Dump hdf5 only allowed for on-lattice apps");

  applattice = (AppLattice *)app;

  if (!applattice->simple)
    error->all(FLERR, "Dump hdf5 only allowed for simple lattices");
}

/* ----------------------------------------------------------------------
   the chunk shape, x,y,z: dump_modify chunk, else a fixed compact block.
   Compact so that a box written into a bigger file rewrites little beyond
   itself; ~0.5 MB of doubles, big enough for deflate and a small chunk
   index.  Chunks tiled to the processor grid only lined up with the run
   that created the file -- a box chain re-decomposes every box, and on one
   processor that rule made each chunk a full row of the whole file.
   Measured on 99M-site boxes of a chain: 128x128x4 and 256x256x2 were the
   fastest of the shapes tried; 64x64x8 slows reads, full rows slow both.
------------------------------------------------------------------------- */

void DumpHDF5::choose_chunk(const int64_t *extent, int64_t *out) {
  const int64_t compact[3] = {128, 128, 4};
  for (int d = 0; d < 3; d++)
    out[d] = std::max((int64_t)1, std::min(user_chunk ? req_chunk[d] : compact[d], extent[d]));
}

/* ----------------------------------------------------------------------
   create the dataset for one field if it is not already in the file
------------------------------------------------------------------------- */

void DumpHDF5::ensure_dataset(const std::string &label, int type) {
  if (file->exist(label))
    return;

  int64_t c[3];
  choose_chunk(dims, c);
  if (!user_chunk)
    for (int d = 0; d < 3; d++)
      chunk[d] = c[d];

  // datasets are [z][y][x]; chunk[] and dims[] are stored x,y,z
  std::vector<size_t> shape = {(size_t)dims[2], (size_t)dims[1],
                               (size_t)dims[0]};
  std::vector<size_t> maxshape = {HighFive::DataSpace::UNLIMITED,
                                  HighFive::DataSpace::UNLIMITED,
                                  HighFive::DataSpace::UNLIMITED};
  HighFive::DataSpace space(shape, maxshape);

  HighFive::DataSetCreateProps props;
  // Chunking must be added first: it is what materializes the underlying
  // property list, and getId() below needs a real plist to set fill on.
  props.add(HighFive::Chunking(
      std::vector<hsize_t>{(hsize_t)c[2], (hsize_t)c[1], (hsize_t)c[0]}));
  if (compress > 0) {
    // shuffle groups the bytes of each value, which is what lets deflate
    // find the redundancy in doubles; parallel HDF5 then requires every
    // write to be collective (write_fields and copy_planes both are)
    if (H5Zfilter_avail(H5Z_FILTER_DEFLATE) <= 0)
      error->all(FLERR, "Dump hdf5 compress: this HDF5 has no deflate filter");
    props.add(HighFive::Shuffle());
    props.add(HighFive::Deflate(compress));
  }

  hid_t plist = props.getId();
  if (plist <= 0)
    error->all(FLERR, "Dump hdf5 could not create dataset property list");

  if (type == INT) {
    int32_t fillv = -1;
    if (H5Pset_fill_value(plist, H5T_NATIVE_INT32, &fillv) < 0)
      error->all(FLERR, "Dump hdf5 could not set integer fill value");
  } else {
    double fillv = 0.0;
    if (H5Pset_fill_value(plist, H5T_NATIVE_DOUBLE, &fillv) < 0)
      error->all(FLERR, "Dump hdf5 could not set double fill value");
  }
  // sites no box has written read as the fill value, which is the no_value
  // sentinel set hdf5 skip_novalue compares against. It is a value test:
  // a double field's real 0.0 is skipped too, as with stitch.
  H5Pset_fill_time(plist, H5D_FILL_TIME_ALLOC);

  HighFive::DataSet dset =
      (type == INT)
          ? file->createDataSet(label, space, HighFive::AtomicType<int32_t>(),
                                props)
          : file->createDataSet(label, space, HighFive::AtomicType<double>(),
                                props);

  dset.createAttribute<std::string>("field", label);
  dset.createAttribute<std::string>("type",
                                    type == INT ? "int32" : "float64");
  dset.createAttribute<int32_t>("per_site_length", 1);
  if (type == INT)
    dset.createAttribute<int32_t>("no_value", -1);
  else
    dset.createAttribute<double>("no_value", 0.0);
}

/* ----------------------------------------------------------------------
   HDF5 runs the filter pipeline one chunk at a time on one thread: ~36 s to
   read back a 99M-site box's eight compressed fields, ~45 s to write them.
   Here the chunks a box touches go through a batch at a time: the file I/O
   (H5Dread_chunk, H5Dwrite_chunk) is serial, the inflate/deflate and the
   copy in or out of the box run on all threads.  Writing, a chunk the box
   only partly covers is read and inflated first so the rest of it survives.
   Handles the layouts dump hdf5 writes: chunked 3-D little-endian
   int32/float64 with deflate, optionally after shuffle; returns false for
   anything else, which is left to H5Dread/H5Dwrite.  Writes need the
   serial driver: under MPI-IO a direct chunk write allocates file space
   independently, which parallel HDF5 requires to be collective.

   For ranks writing in turn, a write splits in two.  pack_out: build and
   compress the chunks into pack_out instead of writing them (all ranks at
   once, before the turns; reading the old contents of partly covered
   chunks is safe then, as each chunk has one writer).  pack_in: only
   write those (in this rank's turn).  With sh, only the chunks this rank
   owns, filled in with the other ranks' parts.
------------------------------------------------------------------------- */

// the rank that writes chunk c (chunk-grid coordinates): one of the ranks
// whose slab touches it, picked by the chunk's position so that neighbouring
// chunks alternate between them -- the lowest would own (and compress) every
// chunk split across ranks, e.g. all of them when chunks span the box in x
// ponytail: O(ranks) per chunk; index the slabs if thousands of ranks make it show
static int chunk_owner(const std::array<hsize_t, 3> &c, const hsize_t *cd,
                       const std::vector<uint64_t> &slabs) {
  int touching[64], n = 0; // a chunk touches few slabs; more just narrows the choice
  for (size_t r = 0; r < slabs.size() / 6 && n < 64; r++) {
    const uint64_t *b = &slabs[6 * r];
    bool hit = true;
    for (int k = 0; k < 3; k++)
      if (b[3 + k] == 0 || b[k] >= (c[k] + 1) * cd[k] || b[k] + b[3 + k] <= c[k] * cd[k])
        hit = false;
    if (hit) touching[n++] = (int)r;
  }
  return n ? touching[(c[0] + c[1] + c[2]) % n] : -1;
}

static bool box_chunks(hid_t ds, const size_t *offset, const size_t *count, void *data,
                       bool write, std::vector<HDF5Chunk> *pack_out = nullptr,
                       const std::vector<HDF5Chunk> *pack_in = nullptr,
                       const HDF5Shared *sh = nullptr) {
#ifndef SPPARKS_ZLIB
  return false;
#else
  hid_t dcpl = H5Dget_create_plist(ds), type = H5Dget_type(ds), space = H5Dget_space(ds);
  hsize_t dims[3] = {0, 0, 0}, cd[3] = {0, 0, 0};
  size_t es = 0;
  if (H5Tequal(type, H5T_STD_I32LE) > 0 && H5Tequal(H5T_NATIVE_INT32, H5T_STD_I32LE) > 0) es = 4;
  if (H5Tequal(type, H5T_IEEE_F64LE) > 0 && H5Tequal(H5T_NATIVE_DOUBLE, H5T_IEEE_F64LE) > 0) es = 8;
  int nf = H5Pget_nfilters(dcpl);
  bool ok = es && H5Sget_simple_extent_ndims(space) == 3 &&
            H5Sget_simple_extent_dims(space, dims, nullptr) == 3 &&
            H5Pget_layout(dcpl) == H5D_CHUNKED && H5Pget_chunk(dcpl, 3, cd) == 3 && nf >= 1 &&
            nf <= 2;
  H5Z_filter_t filt[2] = {-1, -1};
  bool deflate = false;
  int level = 6;
  for (int i = 0; ok && i < nf; i++) {
    unsigned flags, cdv[8];
    size_t nelmts = 8;
    filt[i] = H5Pget_filter2(dcpl, i, &flags, &nelmts, cdv, 0, nullptr, nullptr);
    if (filt[i] == H5Z_FILTER_DEFLATE) {
      deflate = true;
      if (nelmts > 0) level = cdv[0];
    } else if (filt[i] != H5Z_FILTER_SHUFFLE) ok = false;
  }
  // writing applies the filters forward; only the order dump hdf5 creates
  if (ok && write && filt[nf - 1] != H5Z_FILTER_DEFLATE) ok = false;
  unsigned char fill[8] = {0};
  if (ok && H5Pget_fill_value(dcpl, es == 4 ? H5T_NATIVE_INT32 : H5T_NATIVE_DOUBLE, fill) < 0)
    ok = false;
  H5Tclose(type);
  H5Pclose(dcpl);
  H5Sclose(space);
  for (int k = 0; ok && k < 3; k++)
    if (count[k] == 0 || offset[k] + count[k] > dims[k]) ok = false;
  if (!ok || !deflate) return false;
  if (pack_in) {
    for (const HDF5Chunk &c : *pack_in) {
      hsize_t off[3] = {c.off[0], c.off[1], c.off[2]};
      if (H5Dwrite_chunk(ds, H5P_DEFAULT, c.mask, off, c.bytes.size(), c.bytes.data()) < 0)
        return false;
    }
    return true;
  }

  const size_t nelem = cd[0] * cd[1] * cd[2], chunk_bytes = nelem * es;

  // other ranks' parts, by the chunk each lies in
  std::map<std::array<hsize_t, 3>, std::vector<const HDF5Piece *>> parts;
  if (sh)
    for (const HDF5Piece &p : sh->pieces)
      parts[{p.off[0] / cd[0], p.off[1] / cd[1], p.off[2] / cd[2]}].push_back(&p);
  const std::vector<const HDF5Piece *> none;

  // every chunk the box touches (that this rank owns), as chunk-grid
  // coordinates, whether it is wholly covered, and the other ranks' parts in it
  std::vector<std::array<hsize_t, 3>> chunks;
  std::vector<char> whole;
  std::vector<const std::vector<const HDF5Piece *> *> pcs;
  for (hsize_t z = offset[0] / cd[0]; z <= (offset[0] + count[0] - 1) / cd[0]; z++)
    for (hsize_t y = offset[1] / cd[1]; y <= (offset[1] + count[1] - 1) / cd[1]; y++)
      for (hsize_t x = offset[2] / cd[2]; x <= (offset[2] + count[2] - 1) / cd[2]; x++) {
        std::array<hsize_t, 3> c = {z, y, x};
        if (sh && chunk_owner(c, cd, sh->slabs) != sh->me) continue;
        size_t covered = 1;
        for (int k = 0; k < 3; k++)
          covered *= std::min<hsize_t>((c[k] + 1) * cd[k], offset[k] + count[k]) -
                     std::max<hsize_t>(c[k] * cd[k], offset[k]);
        auto it = parts.find(c);
        const std::vector<const HDF5Piece *> *in = it == parts.end() ? &none : &it->second;
        for (const HDF5Piece *p : *in) covered += p->cnt[0] * p->cnt[1] * p->cnt[2];
        char w = covered == nelem; // slabs are disjoint, so volumes add
        chunks.push_back(c);
        whole.push_back(w);
        pcs.push_back(in);
      }


  const size_t batch = 64; // ~64 compressed chunks in flight
  std::vector<std::vector<unsigned char>> comp(batch);
  std::vector<uint32_t> mask(batch);
  std::vector<char> filled(batch);
  unsigned char *box = (unsigned char *)data;
  for (size_t b0 = 0; b0 < chunks.size(); b0 += batch) {
    const size_t nb = std::min(batch, chunks.size() - b0);
    for (size_t b = 0; b < nb; b++) {
      hsize_t off[3] = {chunks[b0 + b][0] * cd[0], chunks[b0 + b][1] * cd[1],
                        chunks[b0 + b][2] * cd[2]};
      if (write && whole[b0 + b]) continue; // overwritten entirely, nothing to keep
      haddr_t addr;
      hsize_t nbytes = 0;
      unsigned fmask = 0;
      if (H5Dget_chunk_info_by_coord(ds, off, &fmask, &addr, &nbytes) < 0) return false;
      filled[b] = (addr == HADDR_UNDEF); // never allocated: all fill value
      if (filled[b]) continue;
      comp[b].resize(nbytes);
      if (H5Dread_chunk(ds, H5P_DEFAULT, off, &mask[b], comp[b].data()) < 0) return false;
    }
    int bad = 0;
#pragma omp parallel reduction(| : bad)
    {
      std::vector<unsigned char> buf1(chunk_bytes), buf2(chunk_bytes);
#pragma omp for schedule(dynamic)
      for (size_t b = 0; b < nb; b++) {
        unsigned char *raw = buf1.data(); // the chunk's values, unfiltered
        if (write && whole[b0 + b]) {
        } else if (filled[b]) {
          for (size_t i = 0; i < nelem; i++) memcpy(&buf1[i * es], fill, es);
        } else {
          const unsigned char *src = comp[b].data();
          size_t len = comp[b].size();
          // undo the pipeline last filter first; mask bit i = filter i skipped
          for (int i = nf - 1; i >= 0; i--) {
            if (mask[b] & (1u << i)) continue;
            unsigned char *to = (src == buf1.data()) ? buf2.data() : buf1.data();
            if (filt[i] == H5Z_FILTER_DEFLATE) {
              uLongf n = chunk_bytes;
              if (uncompress(to, &n, src, len) != Z_OK || n != chunk_bytes) {
                bad = 1;
                break;
              }
            } else { // shuffle: byte j of element e was stored at j*nelem + e
              if (len != chunk_bytes) {
                bad = 1;
                break;
              }
              for (size_t j = 0; j < es; j++)
                for (size_t e = 0; e < nelem; e++) to[e * es + j] = src[j * nelem + e];
            }
            src = to;
            len = chunk_bytes;
          }
          if (bad || len != chunk_bytes) {
            bad = 1;
            continue;
          }
          if (src == comp[b].data()) memcpy(buf1.data(), src, chunk_bytes); // all skipped
          else raw = (unsigned char *)src;
        }
        // the part of this chunk inside the box, one x-run at a time
        hsize_t lo[3], hi[3];
        for (int k = 0; k < 3; k++) {
          lo[k] = std::max<hsize_t>(chunks[b0 + b][k] * cd[k], offset[k]);
          hi[k] = std::min<hsize_t>((chunks[b0 + b][k] + 1) * cd[k], offset[k] + count[k]);
        }
        for (hsize_t z = lo[0]; z < hi[0]; z++)
          for (hsize_t y = lo[1]; y < hi[1]; y++) {
            size_t in_chunk = (((z - chunks[b0 + b][0] * cd[0]) * cd[1] +
                                (y - chunks[b0 + b][1] * cd[1])) * cd[2] +
                               (lo[2] - chunks[b0 + b][2] * cd[2])) * es;
            size_t in_box = (((z - offset[0]) * count[1] + (y - offset[1])) * count[2] +
                             (lo[2] - offset[2])) * es;
            if (write) memcpy(raw + in_chunk, box + in_box, (hi[2] - lo[2]) * es);
            else memcpy(box + in_box, raw + in_chunk, (hi[2] - lo[2]) * es);
          }
        if (!write) continue;
        for (const HDF5Piece *p : *pcs[b0 + b])
          for (hsize_t z = 0; z < p->cnt[0]; z++)
            for (hsize_t y = 0; y < p->cnt[1]; y++) {
              size_t in_chunk = (((p->off[0] + z - chunks[b0 + b][0] * cd[0]) * cd[1] +
                                  (p->off[1] + y - chunks[b0 + b][1] * cd[1])) * cd[2] +
                                 (p->off[2] - chunks[b0 + b][2] * cd[2])) * es;
              memcpy(raw + in_chunk, sh->bytes.data() + p->at + (z * p->cnt[1] + y) * p->cnt[2] * es,
                     p->cnt[2] * es);
            }

        // apply the pipeline: [shuffle,] deflate.  Like HDF5's optional deflate,
        // a chunk that does not shrink is stored as is with its mask bit set.
        unsigned char *src = raw;
        mask[b] = 0;
        if (nf == 2) { // shuffle
          unsigned char *to = (src == buf1.data()) ? buf2.data() : buf1.data();
          for (size_t j = 0; j < es; j++)
            for (size_t e = 0; e < nelem; e++) to[j * nelem + e] = src[e * es + j];
          src = to;
        }
        comp[b].resize(compressBound(chunk_bytes));
        uLongf n = comp[b].size();
        if (compress2(comp[b].data(), &n, src, chunk_bytes, level) == Z_OK && n < chunk_bytes) {
          comp[b].resize(n);
        } else {
          mask[b] |= 1u << (nf - 1);
          comp[b].assign(src, src + chunk_bytes);
        }
      }
    }
    if (bad) return false;
    if (write)
      for (size_t b = 0; b < nb; b++) {
        hsize_t off[3] = {chunks[b0 + b][0] * cd[0], chunks[b0 + b][1] * cd[1],
                          chunks[b0 + b][2] * cd[2]};
        if (pack_out)
          pack_out->push_back({{off[0], off[1], off[2]}, mask[b], std::move(comp[b])});
        else if (H5Dwrite_chunk(ds, H5P_DEFAULT, mask[b], off, comp[b].size(), comp[b].data()) < 0)
          return false;
      }
  }
  return true;
#endif
}

bool SPPARKS_NS::hdf5_read_box_parallel(int64_t dataset, const size_t *offset, const size_t *count,
                                        void *out) {
  return box_chunks(dataset, offset, count, out, false);
}

/* ----------------------------------------------------------------------
   MPI-IO only when there is more than one rank: a single rank gets the
   plain POSIX driver, which keeps the MPI-IO layer and HDF5's parallel
   compressed-write path out of one-rank runs (a box chain lost a dump to
   failed writes there that HDF5 reported as success)
------------------------------------------------------------------------- */

static HighFive::FileAccessProps access_props(MPI_Comm world, int nprocs) {
  HighFive::FileAccessProps fapl;
#ifdef H5_HAVE_PARALLEL
  // Every rank takes the same path through every dump, so collective
  // metadata would be safe here.  It is still left off: it converts a
  // subtle divergence into a hang rather than a slowdown, and it is not
  // needed for correctness.
  if (nprocs > 1) fapl.add(HighFive::MPIOFileAccess(world, MPI_INFO_NULL));
#endif
  return fapl;
}

/* ----------------------------------------------------------------------
   create the file, or reopen one an earlier invocation left behind
------------------------------------------------------------------------- */

void DumpHDF5::open_file(const char *fname) {
  // global lattice extent of THIS run; a staged build's window is only part
  // of the whole part being built
  int64_t lo[3] = {applattice->xlo_simple, applattice->ylo_simple,
                   applattice->zlo_simple};
  int64_t hi[3] = {applattice->xhi_simple, applattice->yhi_simple,
                   applattice->zhi_simple};

  int exists = 0;
  if (me == 0) {
    struct stat sb;
    exists = (stat(fname, &sb) == 0) ? 1 : 0;
  }
  MPI_Bcast(&exists, 1, MPI_INT, 0, world);

  try {
    HighFive::FileAccessProps fapl = access_props(world, nprocs);
    // A compressed chunk rewritten larger moves, and the space it left is
    // reused only while the file stays open: free space is not persisted
    // across dumps, so a compressed file leaks some space per dump.  Both
    // persistent strategies (FSM_AGGR and PAGE) deadlock or fail in the
    // flush of a parallel compressed write in HDF5 1.12.3 and 1.14.6.
    // ponytail: leaks ~15% over a chain, "h5repack" the file if that matters
    HighFive::FileCreateProps fcpl;
    fcpl.add(HighFive::FileSpaceStrategy(H5F_FSPACE_STRATEGY_FSM_AGGR, 0, 1));
    file = exists ? std::make_shared<HighFive::File>(fname, HighFive::File::ReadWrite, fapl)
                  : std::make_shared<HighFive::File>(
                        fname,
                        HighFive::File::ReadWrite | HighFive::File::Create |
                            HighFive::File::Truncate,
                        fcpl, fapl);
  } catch (const std::exception &e) {
    char msg[512];
    snprintf(msg, sizeof(msg), "Dump hdf5 could not open %s: %s", fname,
             e.what());
    error->all(FLERR, msg);
  }

  try {
    if (exists && file->hasAttribute("origin")) {
      std::vector<int64_t> o, d;
      file->getAttribute("origin").read(o);
      file->getAttribute("dims").read(d);
      for (int k = 0; k < 3; k++) {
        origin[k] = o[k];
        dims[k] = d[k];
      }
      if (file->hasAttribute("chunk")) {
        std::vector<int64_t> ch;
        file->getAttribute("chunk").read(ch);
        for (int k = 0; k < 3; k++)
          chunk[k] = ch[k];
      }
      // An existing file's origin is fixed: HDF5 indices start at 0 and
      // cannot be prepended to, so a domain extending below it cannot be
      // represented without rewriting every dataset.  Say so rather than
      // silently clipping.
      for (int k = 0; k < 3; k++)
        if (lo[k] < origin[k]) {
          char msg[512];
          snprintf(msg, sizeof(msg),
                   "Dump hdf5 domain starts below the origin stored in %s "
                   "(dim %d: domain %lld < origin %lld); recreate the file or "
                   "pre-declare it with 'dump_modify <id> origin'",
                   filename, k, (long long)lo[k], (long long)origin[k]);
          error->all(FLERR, msg);
        }
    } else {
      for (int k = 0; k < 3; k++) {
        origin[k] = user_origin ? req_origin[k] : lo[k];
        if (user_origin && lo[k] < origin[k])
          error->all(FLERR,
                     "Dump hdf5 'dump_modify origin' is above the domain");
        dims[k] = user_extent ? req_extent[k] : (hi[k] - origin[k] + 1);
        if (dims[k] < 1)
          dims[k] = 1;
      }
      file->createAttribute<int32_t>("spparks_hdf5_dump_version",
                                     SPPARKS_HDF5_DUMP_VERSION);
      file->createAttribute<std::string>("dim_order", "zyx");
      file->createAttribute<std::vector<int64_t>>(
          "origin", std::vector<int64_t>(origin, origin + 3));
      file->createAttribute<std::vector<int64_t>>(
          "dims", std::vector<int64_t>(dims, dims + 3));
      file->createAttribute<double>(
          "lattice_constant", domain->lattice ? domain->lattice->xlattice : 1.0);
      file->createAttribute<int32_t>("dimension", domain->dimension);
      // NOTE: pass real vectors, not braced lists -- a braced list here binds
      // to the createAttribute(name, DataSpace) overload instead.
      std::vector<double> blo = {domain->boxxlo, domain->boxylo, domain->boxzlo};
      std::vector<double> bhi = {domain->boxxhi, domain->boxyhi, domain->boxzhi};
      file->createAttribute<std::vector<double>>("boxlo", blo);
      file->createAttribute<std::vector<double>>("boxhi", bhi);
      file->createAttribute<int32_t>("nprocs", nprocs);
      file->createAttribute<double>("time", 0.0);
    }
  } catch (const std::exception &e) {
    char msg[512];
    snprintf(msg, sizeof(msg), "Dump hdf5 metadata error on %s: %s", fname,
             e.what());
    error->all(FLERR, msg);
  }

  opened = true;
}

/* ----------------------------------------------------------------------
   grow every dataset in the file so that 'needed' fits.

   Every dataset, not just this dump's fields: 'dims' is the whole file's
   extent, and a field another box dumped but this one does not would
   otherwise stay short of it (set hdf5 then read past its end, and the
   next dump of that field failed). Files a short field was left in by
   earlier versions are repaired here too.

   H5Dset_extent is collective; the file's metadata, and so this loop, is
   identical on every rank.
------------------------------------------------------------------------- */

void DumpHDF5::extend_to(const int64_t *needed) {
  std::vector<HighFive::DataSet> sets;
  for (const std::string &name : file->listObjectNames())
    if (file->getObjectType(name) == HighFive::ObjectType::Dataset)
      sets.push_back(file->getDataSet(name));

  // never shrink: the extent is the largest of dims, needed and every dataset
  bool grew = false;
  for (int k = 0; k < 3; k++) {
    int64_t want = needed[k];
    for (HighFive::DataSet &d : sets) want = std::max(want, (int64_t)d.getDimensions()[2 - k]);
    if (want > dims[k]) {
      dims[k] = want;
      grew = true;
    }
  }

  std::vector<size_t> shape = {(size_t)dims[2], (size_t)dims[1],
                               (size_t)dims[0]};
  for (HighFive::DataSet &d : sets)
    if (d.getDimensions() != shape) d.resize(shape);

  if (grew)
    file->getAttribute("dims").write(std::vector<int64_t>(dims, dims + 3));
}

/* ----------------------------------------------------------------------
   write a snapshot: overwrite each field in place
------------------------------------------------------------------------- */

void DumpHDF5::write(double time) {
  if (!opened)
    rebuild_if_needed();

  // atomic: write-then-rename, so the file under its own name is always a
  // complete earlier dump, whatever happens to this process mid-write.
  // Otherwise write the file in place: no copy of the whole file per dump,
  // but a crash mid-write can damage it. Either way the file is open only
  // while this dump writes, and is closed and synced before the run goes on.
  std::string tmp = std::string(filename) + ".tmp";
  const char *target = atomic ? tmp.c_str() : filename;
  if (atomic) stage_copy(target);
  double t[7] = {MPI_Wtime(), 0, 0, 0, 0, 0, 0}, w = t[0];
  open_file(target);
  bool in_turns = use_turns();
  t[1] = MPI_Wtime() - w;
  // with turns, fields go in batches that bound the compressed chunks held
  // in memory; otherwise all at once
  for (int first = 0; first < size_one;) {
    w = MPI_Wtime();
    if (first > 0) open_file(target);
    int next = write_fields(time, in_turns, first);
    t[2] += MPI_Wtime() - w;
    w = MPI_Wtime();
    try {
      file->flush(); // H5Fflush; under MPI-IO it ends in MPI_File_sync
    } catch (const std::exception &e) {
      // one rank may fail while the others go on into the collective close
      error->one(FLERR, (std::string("Dump hdf5 flush failed: ") + e.what()).c_str());
    }
    file.reset(); // collective close
    opened = false;
    t[3] += MPI_Wtime() - w;
    w = MPI_Wtime();
    if (in_turns) take_turns(target, first, next);
    t[4] += MPI_Wtime() - w;
    first = next;
  }
  w = MPI_Wtime();
  verify(target, time);
  t[5] = MPI_Wtime() - w;
  w = MPI_Wtime();
  if (atomic) commit(target);
  else sync_file(target);
  t[6] = MPI_Wtime() - w;
  if (verbose && me == 0 && screen)
    fprintf(screen,
            "dump hdf5: open %.2f, %s %.2f, close %.2f, turns %.2f, read-back %.2f, "
            "sync %.2f s\n",
            t[1], in_turns ? "share+compress" : "write", t[2], t[3], t[4], t[5], t[6]);
}

/* ----------------------------------------------------------------------
   reopen the closed file and check that every field of this rank's box
   reads back as written. A dump whose writes failed without HDF5 noticing
   then stops the run instead of handing damaged state to the next box.
   The box is decoded by box_chunks, the code that encoded it; one z-plane
   of it is also read through HDF5's own filter pipeline, so a mistake made
   the same way in both directions does not pass. Every dataset in the file
   must span 'dims'.
------------------------------------------------------------------------- */

void DumpHDF5::verify(const char *fname, double time) {
  std::vector<size_t> offset, count;
  my_slab(offset, count);
  size_t nsites = count[0] * count[1] * count[2];

  int bad = -1; // index of the first field that differs, -1 = all match
  std::string why;
  try {
    // independent reads: a rank whose parallel read falls back must not
    // wait in a collective the others skipped
    HighFive::File f(fname, HighFive::File::ReadOnly, access_props(world, nprocs));
    std::vector<size_t> shape = {(size_t)dims[2], (size_t)dims[1], (size_t)dims[0]};
    for (const std::string &name : f.listObjectNames())
      if (f.getObjectType(name) == HighFive::ObjectType::Dataset &&
          f.getDataSet(name).getDimensions() != shape && why.empty())
        why = name + " does not span the file's dims";
    std::vector<int32_t> ibuf;
    std::vector<double> dbuf;
    // the plane read through HDF5's pipeline
    const size_t zp = count[0] / 2, plane = count[1] * count[2];
    HighFive::HyperSlab pslab{HighFive::RegularHyperSlab({offset[0] + zp, offset[1], offset[2]},
                                                         {1, count[1], count[2]})};
    for (int i = 0; i < size_one; i++) {
      HighFive::DataSet dset = f.getDataSet(labels[i]);
      HighFive::HyperSlab slab{HighFive::RegularHyperSlab(offset, count)};
      int same;
      if (vtype[i] == INT) {
        const int *src = (fields[i] == SITE) ? app->iarray[0] : app->iarray[vindex[i]];
        ibuf.resize(nsites);
        if (!hdf5_read_box_parallel(dset.getId(), offset.data(), count.data(), ibuf.data()))
          dset.select(slab).read_raw(ibuf.data(), HighFive::AtomicType<int32_t>());
        same = memcmp(ibuf.data(), src, nsites * sizeof(int32_t)) == 0;
        dset.select(pslab).read_raw(ibuf.data(), HighFive::AtomicType<int32_t>());
        same = same && memcmp(ibuf.data(), src + zp * plane, plane * sizeof(int32_t)) == 0;
      } else {
        const double *src = app->darray[vindex[i]];
        dbuf.resize(nsites);
        if (!hdf5_read_box_parallel(dset.getId(), offset.data(), count.data(), dbuf.data()))
          dset.select(slab).read_raw(dbuf.data(), HighFive::AtomicType<double>());
        same = memcmp(dbuf.data(), src, nsites * sizeof(double)) == 0;
        dset.select(pslab).read_raw(dbuf.data(), HighFive::AtomicType<double>());
        same = same && memcmp(dbuf.data(), src + zp * plane, plane * sizeof(double)) == 0;
      }
      if (!same && bad < 0) bad = i;
    }
    double t;
    f.getAttribute("time").read(t);
    if (t != time && bad < 0) why = "time attribute";
  } catch (const std::exception &e) {
    why = e.what();
  }

  // lowest failing field on any rank; size_one = a read error, INT_MAX = all good
  int code = bad >= 0 ? bad : (why.empty() ? INT_MAX : size_one), first;
  MPI_Allreduce(&code, &first, 1, MPI_INT, MPI_MIN, world);
  if (first == INT_MAX) return;
  if (first < size_one) why = labels[first];
  else if (why.empty()) why = "read error on another rank";
  char msg[512];
  snprintf(msg, sizeof(msg),
           "Dump hdf5 read-back of %s does not match what was written (%s)%s", fname,
           why.c_str(),
           atomic ? "; the previous dump is intact" : "; the file is damaged, restore a backup");
  error->all(FLERR, msg);
}

/* ----------------------------------------------------------------------
   fsync a closed file (the serial-HDF5 flush does not), so the next run
   reads what this dump wrote even after an OS crash
------------------------------------------------------------------------- */

void DumpHDF5::sync_file(const char *fname) {
  int ok = 1;
  if (me == 0) {
    int fd = open(fname, O_RDONLY);
    if (fd < 0 || fsync(fd) != 0) ok = 0;
    if (fd >= 0) close(fd);
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, world);
  if (!ok) {
    char msg[512];
    snprintf(msg, sizeof(msg), "Dump hdf5 could not sync %s", fname);
    error->all(FLERR, msg);
  }
}

/* ----------------------------------------------------------------------
   copy planes [z0,z1) of src into dst, shifted by off (z,y,x), a bounded
   slab at a time. The writes are collective, which parallel HDF5 requires
   for a compressed dataset: every rank makes niter calls, with an empty
   selection once its own planes are done.
------------------------------------------------------------------------- */

template <typename T>
static void copy_planes(HighFive::DataSet &src, HighFive::DataSet &dst,
                        size_t z0, size_t z1, const std::vector<size_t> &d,
                        const size_t *off, size_t step, int niter, bool collective) {
  size_t plane = d[1] * d[2];
  // a predefined type: an id taken from a temporary HighFive::AtomicType is
  // closed when the temporary dies
  hid_t memtype = sizeof(T) == sizeof(int32_t) ? H5T_NATIVE_INT32 : H5T_NATIVE_DOUBLE;
  hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
#ifdef H5_HAVE_PARALLEL
  if (collective) H5Pset_dxpl_mpio(xfer, H5FD_MPIO_COLLECTIVE);
#endif
  hid_t fspace = H5Dget_space(dst.getId());
  std::vector<T> buf(1);
  herr_t rc = 0;
  for (int it = 0; it < niter && rc >= 0; it++) {
    size_t z = z0 + (size_t)it * step;
    hsize_t nz = z < z1 ? std::min(step, z1 - z) : 0;
    hsize_t count[3] = {nz, d[1], d[2]}, one[3] = {1, 1, 1};
    hid_t mspace = H5Screate_simple(3, nz ? count : one, NULL);
    if (nz) {
      buf.resize(nz * plane);
      src.select({z, 0, 0}, {(size_t)nz, d[1], d[2]}).read_raw(buf.data());
      hsize_t start[3] = {z + off[0], off[1], off[2]};
      H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
    } else {
      H5Sselect_none(fspace);
      H5Sselect_none(mspace);
    }
    rc = H5Dwrite(dst.getId(), memtype, mspace, fspace, xfer, buf.data());
    H5Sclose(mspace);
  }
  H5Sclose(fspace);
  H5Pclose(xfer);
  if (rc < 0)
    throw std::runtime_error("H5Dwrite failed while copying " + dst.getPath());
}

/* ----------------------------------------------------------------------
   rewrite the file once if this run's domain starts below its origin in
   any dimension (HDF5 indices cannot be prepended to, so every dataset is
   copied into a wider frame) or if its datasets' compression differs from
   'dump_modify compress' (filters are fixed at dataset creation) or if it
   persists its free space (fixed at file creation). Each rank copies a share of the z planes. The rewrite goes through tmp and a
   rename like any atomic dump, so the old file survives a crash.
------------------------------------------------------------------------- */

void DumpHDF5::rebuild_if_needed() {
  int64_t lo[3] = {applattice->xlo_simple, applattice->ylo_simple,
                   applattice->zlo_simple};
  int64_t hi[3] = {applattice->xhi_simple, applattice->yhi_simple,
                   applattice->zhi_simple};

  int64_t od[6] = {0, 0, 0, 0, 0, 0}; // stored origin, then dims (x,y,z)
  double t = 0.0;
  int below = 0, relayout = 0;
  if (me == 0) {
    struct stat sb;
    if (stat(filename, &sb) == 0) {
      try {
        HighFive::File f(filename, HighFive::File::ReadOnly);
        if (f.hasAttribute("origin") && f.hasAttribute("dims")) {
          std::vector<int64_t> o, d;
          f.getAttribute("origin").read(o);
          f.getAttribute("dims").read(d);
          for (int k = 0; k < 3; k++) {
            od[k] = o[k];
            od[3 + k] = d[k];
            if (lo[k] < o[k]) below = 1;
          }
          if (f.hasAttribute("time")) f.getAttribute("time").read(t);
          // datasets whose compression differs from the requested setting
          // are rebuilt (filters are fixed when a dataset is created)
          for (const std::string &name : f.listObjectNames()) {
            hid_t dcpl = H5Dget_create_plist(f.getDataSet(name).getId());
            if ((H5Pget_nfilters(dcpl) > 0) != (compress > 0)) relayout = 1;
            H5Pclose(dcpl);
          }
          // files from before free-space persistence was dropped: a parallel
          // write to one deadlocks, and the setting is fixed at creation
          hid_t fcpl = H5Fget_create_plist(f.getId());
          H5F_fspace_strategy_t strategy;
          hbool_t persist = 0;
          hsize_t threshold;
          H5Pget_file_space_strategy(fcpl, &strategy, &persist, &threshold);
          H5Pclose(fcpl);
          if (persist) relayout = 1;
        }
      } catch (const std::exception &e) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Dump hdf5 could not read %s: %s", filename, e.what());
        error->one(FLERR, msg);
      }
    }
  }
  int why[2] = {below, relayout};
  MPI_Bcast(why, 2, MPI_INT, 0, world);
  below = why[0];
  relayout = why[1];
  if (!below && !relayout)
    return;
  MPI_Bcast(od, 6, MPI_LONG_LONG, 0, world);
  MPI_Bcast(&t, 1, MPI_DOUBLE, 0, world);

  // the new frame holds the old datasets and this run's domain
  bool save_origin = user_origin, save_extent = user_extent;
  int64_t save_req[6];
  for (int k = 0; k < 3; k++) {
    save_req[k] = req_origin[k];
    save_req[3 + k] = req_extent[k];
    req_origin[k] = std::min(od[k], lo[k]);
    req_extent[k] = std::max(od[k] + od[3 + k], hi[k] + 1) - req_origin[k];
  }
  user_origin = user_extent = true;
  std::string tmp = std::string(filename) + ".tmp";
  if (me == 0) unlink(tmp.c_str());
  MPI_Barrier(world);
  open_file(tmp.c_str());
  user_origin = save_origin;
  user_extent = save_extent;
  for (int k = 0; k < 3; k++) {
    req_origin[k] = save_req[k];
    req_extent[k] = save_req[3 + k];
  }

  try {
    HighFive::File old(filename, HighFive::File::ReadOnly, access_props(world, nprocs));
    // dataset index order is z,y,x; od[] and origin[] are x,y,z
    size_t off[3] = {(size_t)(od[2] - origin[2]), (size_t)(od[1] - origin[1]),
                     (size_t)(od[0] - origin[0])};
    for (const std::string &name : old.listObjectNames()) {
      HighFive::DataSet src = old.getDataSet(name);
      bool isint = src.getDataType() == HighFive::AtomicType<int32_t>();
      ensure_dataset(name, isint ? INT : DOUBLE); // collective, same order everywhere
      HighFive::DataSet dst = file->getDataSet(name);
      std::vector<size_t> d = src.getDimensions();
      size_t per = (d[0] + nprocs - 1) / nprocs;
      size_t z0 = std::min(d[0], (size_t)me * per), z1 = std::min(d[0], z0 + per);
      size_t elem = isint ? sizeof(int32_t) : sizeof(double);
      size_t step = std::max<size_t>(1, ((size_t)256 << 20) / std::max<size_t>(1, d[1] * d[2] * elem));
      int mine = (int)((z1 - z0 + step - 1) / step), niter;
      MPI_Allreduce(&mine, &niter, 1, MPI_INT, MPI_MAX, world);
      if (isint) copy_planes<int32_t>(src, dst, z0, z1, d, off, step, niter, nprocs > 1);
      else copy_planes<double>(src, dst, z0, z1, d, off, step, niter, nprocs > 1);
    }
    file->getAttribute("time").write(t);
    file->flush();
  } catch (const std::exception &e) {
    // a failing rank leaves copy_planes while the others are still in its
    // collective writes, where no agreement can reach them: stop the job
    // (the old file is untouched, the rewrite went to tmp)
    char msg[512];
    snprintf(msg, sizeof(msg), "Dump hdf5 could not move the origin of %s: %s",
             filename, e.what());
    error->one(FLERR, msg);
  }
  file.reset();
  opened = false;
  commit(tmp.c_str());

  if (me == 0) {
    char msg[512];
    if (below)
      snprintf(msg, sizeof(msg),
               "dump hdf5: domain starts below the origin of %s; rewrote it with "
               "origin (%lld %lld %lld), was (%lld %lld %lld)\n",
               filename, (long long)origin[0], (long long)origin[1], (long long)origin[2],
               (long long)od[0], (long long)od[1], (long long)od[2]);
    else
      snprintf(msg, sizeof(msg), "dump hdf5: rewrote %s %s, without persistent free space\n",
               filename, compress > 0 ? "compressed" : "uncompressed");
    if (screen) fputs(msg, screen);
    if (logfile) fputs(msg, logfile);
  }
}

/* ----------------------------------------------------------------------
   copy the current file to tmp, so the dump overwrites only the copy.
   A reflink (btrfs, XFS) is instant; elsewhere (Lustre, GPFS, ext4) every
   rank copies one slice through MPI-IO, so the copy runs at the parallel
   file system's aggregate bandwidth rather than one client's.
------------------------------------------------------------------------- */

void DumpHDF5::stage_copy(const char *tmp) {
  int exists = 0, cloned = 0;
  if (me == 0) {
    unlink(tmp); // a crash mid-dump leaves a stale copy behind
    struct stat sb;
    exists = (stat(filename, &sb) == 0) ? 1 : 0;
#ifdef FICLONE
    if (exists) {
      int in = open(filename, O_RDONLY);
      int out = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, sb.st_mode & 07777);
      cloned = (in >= 0 && out >= 0 && ioctl(out, FICLONE, in) == 0) ? 1 : 0;
      if (in >= 0) close(in);
      if (out >= 0) close(out);
      if (!cloned) unlink(tmp);
    }
#endif
  }
  int flags[2] = {exists, cloned};
  MPI_Bcast(flags, 2, MPI_INT, 0, world);
  if (!flags[0] || flags[1])
    return; // first dump creates tmp from scratch, or the reflink did it

  int err = 0;
  const size_t piece = (size_t)64 << 20;
  std::vector<char> buf;
#ifdef H5_HAVE_PARALLEL
  MPI_File in, out;
  err |= MPI_File_open(world, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &in);
  err |= MPI_File_open(world, tmp, MPI_MODE_WRONLY | MPI_MODE_CREATE,
                       MPI_INFO_NULL, &out);
  if (err == MPI_SUCCESS) {
    MPI_Offset size;
    MPI_File_get_size(in, &size);
    MPI_Offset per = (size + nprocs - 1) / nprocs;
    MPI_Offset lo = std::min(size, (MPI_Offset)me * per);
    MPI_Offset hi = std::min(size, lo + per);
    if (hi > lo) buf.resize(std::min((MPI_Offset)piece, hi - lo));
    for (MPI_Offset off = lo; off < hi && !err; off += piece) {
      int n = (int)std::min((MPI_Offset)piece, hi - off);
      MPI_Status st;
      err |= MPI_File_read_at(in, off, buf.data(), n, MPI_BYTE, &st);
      err |= MPI_File_write_at(out, off, buf.data(), n, MPI_BYTE, &st);
    }
    err |= MPI_File_sync(out); // collective
    MPI_File_close(&in);
    MPI_File_close(&out);
  }
#else
  if (me == 0) {
    FILE *in = fopen(filename, "rb"), *out = fopen(tmp, "wb");
    if (!in || !out) err = 1;
    buf.resize(piece);
    size_t n;
    while (!err && (n = fread(buf.data(), 1, piece, in)) > 0)
      if (fwrite(buf.data(), 1, n, out) != n) err = 1;
    if (in && ferror(in)) err = 1;
    if (in) fclose(in);
    if (out && fclose(out) != 0) err = 1;
  }
#endif
  int errall;
  MPI_Allreduce(&err, &errall, 1, MPI_INT, MPI_MAX, world);
  if (errall) {
    char msg[512];
    snprintf(msg, sizeof(msg), "Dump hdf5 could not copy %s to %s", filename, tmp);
    error->all(FLERR, msg);
  }
}

/* ----------------------------------------------------------------------
   make the finished copy durable, then atomically replace the file with it
------------------------------------------------------------------------- */

void DumpHDF5::commit(const char *tmp) {
  int ok = 1;
  if (me == 0) {
    int fd = open(tmp, O_RDONLY); // the serial-HDF5 flush does not fsync
    if (fd < 0 || fsync(fd) != 0) ok = 0;
    if (fd >= 0) close(fd);
    if (ok && rename(tmp, filename) != 0) ok = 0;
    if (ok) {
      // the rename itself lives in the directory
      std::string f(filename);
      size_t slash = f.rfind('/');
      std::string dir = slash == std::string::npos ? "." : f.substr(0, slash ? slash : 1);
      int dfd = open(dir.c_str(), O_RDONLY | O_DIRECTORY);
      if (dfd >= 0) {
        fsync(dfd);
        close(dfd);
      }
    }
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, world);
  if (!ok) {
    char msg[512];
    snprintf(msg, sizeof(msg),
             "Dump hdf5 could not replace %s with %s (the previous dump is intact)",
             filename, tmp);
    error->all(FLERR, msg);
  }
}

/* ----------------------------------------------------------------------
   write every field into the open file
------------------------------------------------------------------------- */

void DumpHDF5::my_slab(std::vector<size_t> &offset, std::vector<size_t> &count) {
  // this rank's owned box in absolute lattice indices, half-open
  int64_t lo[3] = {applattice->xlo_me_simple, applattice->ylo_me_simple,
                   applattice->zlo_me_simple};
  int64_t n[3] = {applattice->xhi_me_simple - applattice->xlo_me_simple + 1,
                  applattice->yhi_me_simple - applattice->ylo_me_simple + 1,
                  applattice->zhi_me_simple - applattice->zlo_me_simple + 1};

  for (int k = 0; k < 3; k++)
    if (lo[k] < origin[k])
      error->all(FLERR, "Dump hdf5 subdomain starts below the file origin");

  // dataset index order is z,y,x
  offset = {(size_t)(lo[2] - origin[2]), (size_t)(lo[1] - origin[1]),
            (size_t)(lo[0] - origin[0])};
  count = {(size_t)n[2], (size_t)n[1], (size_t)n[0]};
}

// site ordering for a simple lattice is x fastest, then y, then z -- the
// same order as the [z][y][x] hyperslab, so a field is written with no repacking
void *DumpHDF5::field_data(int i) {
  if (vtype[i] == INT) return (fields[i] == SITE) ? app->iarray[0] : app->iarray[vindex[i]];
  return app->darray[vindex[i]];
}

static void write_slab(HighFive::DataSet &dset, const std::vector<size_t> &offset,
                       const std::vector<size_t> &count, bool isint, void *src,
                       const HighFive::DataTransferProps &xfer) {
  HighFive::HyperSlab slab{HighFive::RegularHyperSlab(offset, count)};
  if (isint) dset.select(slab).write_raw((int *)src, HighFive::AtomicType<int32_t>(), xfer);
  else dset.select(slab).write_raw((double *)src, HighFive::AtomicType<double>(), xfer);
}

/* ----------------------------------------------------------------------
   whether this dump's ranks write in turn: always with turns yes, never
   with turns no or on one rank; by default when compressing and every
   rank has more than one thread to compress with
------------------------------------------------------------------------- */

bool DumpHDF5::use_turns() {
  if (nprocs == 1 || turns == 0) return false;
  if (turns == 1) return true;
  int nthreads = 1, fewest;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#endif
  MPI_Allreduce(&nthreads, &fewest, 1, MPI_INT, MPI_MIN, world);
  return compress > 0 && fewest > 1;
}

/* ----------------------------------------------------------------------
   turns, field i: send this rank's parts of the chunks other ranks own to
   their owners, and receive the parts of the chunks this rank owns, so each
   chunk is written once. A chunk written in two turns would move twice,
   and the space it left in the first turn is lost to the file.
   false (on every rank) if the dataset is not chunked.
------------------------------------------------------------------------- */

bool DumpHDF5::share(int i, int64_t dataset, const std::vector<size_t> &offset,
                     const std::vector<size_t> &count, HDF5Shared &sh) {
  sh.me = me;
  sh.slabs.resize(6 * nprocs);
  uint64_t mine[6] = {offset[0], offset[1], offset[2], count[0], count[1], count[2]};
  MPI_Allgather(mine, 6, MPI_UINT64_T, sh.slabs.data(), 6, MPI_UINT64_T, world);

  hsize_t cd[3];
  hid_t dcpl = H5Dget_create_plist(dataset);
  bool chunked = H5Pget_layout(dcpl) == H5D_CHUNKED && H5Pget_chunk(dcpl, 3, cd) == 3;
  H5Pclose(dcpl);
  if (!chunked) return false;

  // this rank's part of each chunk another rank owns: a header (offset,
  // count, start) and the values, x-run by x-run, per destination
  const size_t es = vtype[i] == INT ? 4 : 8;
  const unsigned char *src = (const unsigned char *)field_data(i);
  std::vector<std::vector<uint64_t>> hdr(nprocs);
  std::vector<std::vector<unsigned char>> val(nprocs);
  for (hsize_t z = offset[0] / cd[0]; z <= (offset[0] + count[0] - 1) / cd[0]; z++)
    for (hsize_t y = offset[1] / cd[1]; y <= (offset[1] + count[1] - 1) / cd[1]; y++)
      for (hsize_t x = offset[2] / cd[2]; x <= (offset[2] + count[2] - 1) / cd[2]; x++) {
        std::array<hsize_t, 3> c = {z, y, x};
        int r = chunk_owner(c, cd, sh.slabs);
        if (r == me) continue;
        uint64_t lo[3], n[3];
        for (int k = 0; k < 3; k++) {
          lo[k] = std::max<uint64_t>(c[k] * cd[k], offset[k]);
          n[k] = std::min<uint64_t>((c[k] + 1) * cd[k], offset[k] + count[k]) - lo[k];
        }
        hdr[r].insert(hdr[r].end(), {lo[0], lo[1], lo[2], n[0], n[1], n[2], val[r].size()});
        for (uint64_t zz = lo[0]; zz < lo[0] + n[0]; zz++)
          for (uint64_t yy = lo[1]; yy < lo[1] + n[1]; yy++) {
            const unsigned char *run =
                src + (((zz - offset[0]) * count[1] + (yy - offset[1])) * count[2] +
                       (lo[2] - offset[2])) * es;
            val[r].insert(val[r].end(), run, run + n[2] * es);
          }
      }

  // MPI counts are int: flatten per destination and check the totals fit
  std::vector<int> hs(nprocs), hr(nprocs), vs(nprocs), vr(nprocs), hsd(nprocs), hrd(nprocs),
      vsd(nprocs), vrd(nprocs);
  std::vector<uint64_t> hflat;
  std::vector<unsigned char> vflat;
  for (int r = 0; r < nprocs; r++) {
    hs[r] = hdr[r].size();
    vs[r] = val[r].size();
    hflat.insert(hflat.end(), hdr[r].begin(), hdr[r].end());
    vflat.insert(vflat.end(), val[r].begin(), val[r].end());
    std::vector<unsigned char>().swap(val[r]);
  }
  MPI_Alltoall(hs.data(), 1, MPI_INT, hr.data(), 1, MPI_INT, world);
  MPI_Alltoall(vs.data(), 1, MPI_INT, vr.data(), 1, MPI_INT, world);
  int64_t got = 0, hgot = 0;
  for (int r = 0; r < nprocs; r++) got += vr[r], hgot += hr[r];
  int64_t biggest = std::max<int64_t>(vflat.size(), got), most;
  MPI_Allreduce(&biggest, &most, 1, MPI_INT64_T, MPI_MAX, world);
  if (most > INT_MAX)
    error->all(FLERR, "Dump hdf5 turns: over 2 GB of one field shared between ranks' "
                      "chunks; use smaller chunks or dump_modify turns no");
  for (int r = 1; r < nprocs; r++) {
    hsd[r] = hsd[r - 1] + hs[r - 1];
    hrd[r] = hrd[r - 1] + hr[r - 1];
    vsd[r] = vsd[r - 1] + vs[r - 1];
    vrd[r] = vrd[r - 1] + vr[r - 1];
  }
  std::vector<uint64_t> hin(hgot);
  sh.bytes.resize(got);
  MPI_Alltoallv(hflat.data(), hs.data(), hsd.data(), MPI_UINT64_T, hin.data(), hr.data(),
                hrd.data(), MPI_UINT64_T, world);
  MPI_Alltoallv(vflat.data(), vs.data(), vsd.data(), MPI_BYTE, sh.bytes.data(), vr.data(),
                vrd.data(), MPI_BYTE, world);
  for (int r = 0; r < nprocs; r++)
    for (int h = hrd[r]; h < hrd[r] + hr[r]; h += 7) {
      const uint64_t *v = &hin[h];
      sh.pieces.push_back({{v[0], v[1], v[2]}, {v[3], v[4], v[5]}, v[6] + vrd[r]});
    }
  return true;
}

/* ----------------------------------------------------------------------
   ranks write in turn. Every rank has already built and compressed the
   chunks it owns (write_fields, all ranks at once, all threads); now, in
   rank order, each rank opens the closed file alone with the POSIX driver,
   writes them and closes it, so the next rank's allocations see its writes.
   Only the file I/O is serial. Relies on close-to-open consistency, which
   Lustre and GPFS give and NFS does not.
------------------------------------------------------------------------- */

namespace {
// The turns already give one rank the file at a time; an HDF5 lock would
// add nothing, and fails outright on a file system mounted without flock.
struct NoFileLocking {
  void apply(hid_t fapl) const { H5Pset_file_locking(fapl, false, true); }
};
} // namespace

void DumpHDF5::take_turns(const char *fname, int i0, int i1) {
  std::vector<size_t> offset, count;
  my_slab(offset, count);
  int token = 0;
  std::string why;
  if (me > 0) MPI_Recv(&token, 1, MPI_INT, me - 1, 0, world, MPI_STATUS_IGNORE);
  try {
    {
      HighFive::FileAccessProps fapl;
      fapl.add(NoFileLocking{});
      HighFive::File f(fname, HighFive::File::ReadWrite, fapl);
      for (int i = i0; i < i1; i++) {
        HighFive::DataSet dset = f.getDataSet(labels[i]);
        void *src = field_data(i);
        if (!packed_ok[i])
          write_slab(dset, offset, count, vtype[i] == INT, src, {});
        else if (!box_chunks(dset.getId(), offset.data(), count.data(), src, true, nullptr,
                             &packed[i]))
          throw std::runtime_error("writing the chunks of " + labels[i] + " failed");
        std::vector<HDF5Chunk>().swap(packed[i]);
      }
      f.flush();
    } // closed
    // the POSIX close does not fsync, and only this node holds what it wrote:
    // sync before the next rank goes, or a crash of this node after the dump
    // loses chunks the index already points at
    int fd = open(fname, O_RDONLY);
    bool synced = fd >= 0 && fsync(fd) == 0;
    if (fd >= 0) close(fd);
    if (!synced) throw std::runtime_error("fsync failed");
  } catch (const std::exception &e) {
    why = e.what();
  }
  // pass the turn on even after a failure, or every later rank waits forever
  if (me < nprocs - 1) MPI_Send(&token, 1, MPI_INT, me + 1, 0, world);

  int failed = !why.empty(), first;
  int rank = failed ? me : nprocs;
  MPI_Allreduce(&rank, &first, 1, MPI_INT, MPI_MIN, world);
  if (first == nprocs) return;
  if (failed) fprintf(stderr, "Dump hdf5 rank %d: %s\n", me, why.c_str());
  char msg[512];
  snprintf(msg, sizeof(msg), "Dump hdf5 write failed on %s in rank %d's turn (see stderr)%s",
           fname, first,
           atomic ? "; the previous dump is intact" : "; the file is damaged, restore a backup");
  error->all(FLERR, msg);
}

// compressed chunks a rank may hold between turns: an 8-field AM dump is
// ~30 B/site, 3 GB on a 99M-site rank, too much beside the app on a
// shared-memory APU. ponytail: fixed; make it a dump_modify knob if needed
static const int64_t TURN_BYTES = (int64_t)1 << 30;

int DumpHDF5::write_fields(double time, bool pack, int first) {
  std::vector<size_t> offset, count;
  my_slab(offset, count);

  int64_t mine[3], needed[3];
  for (int k = 0; k < 3; k++)
    mine[k] = (int64_t)(offset[2 - k] + count[2 - k]);
  MPI_Allreduce(mine, needed, 3, MPI_LONG_LONG, MPI_MAX, world);

  try {
    for (int i = 0; i < size_one; i++)
      ensure_dataset(labels[i], vtype[i]);

    extend_to(needed);

    HighFive::DataTransferProps xfer;
#ifdef H5_HAVE_PARALLEL
    if (nprocs > 1) xfer.add(HighFive::UseCollectiveIO{});
#endif

    if (pack) {
      packed.assign(size_one, {});
      packed_ok.assign(size_one, 0);
    }
    int64_t held = 0;
    for (int i = first; i < size_one; i++) {
      HighFive::DataSet dset = file->getDataSet(labels[i]);
      void *src = field_data(i);
      if (pack) { // compressed here on all ranks at once, written in take_turns
        HDF5Shared sh;
        int ok = share(i, dset.getId(), offset, count, sh) &&
                 box_chunks(dset.getId(), offset.data(), count.data(), src, true, &packed[i],
                            nullptr, &sh);
        int all;
        MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, world);
        packed_ok[i] = all; // else every rank writes its whole slab in its turn
        for (const HDF5Chunk &c : packed[i]) held += c.bytes.size();
        int64_t most;
        MPI_Allreduce(&held, &most, 1, MPI_INT64_T, MPI_MAX, world);
        if (most > TURN_BYTES && i + 1 < size_one) {
          file->getAttribute("time").write(time);
          return i + 1; // write this batch in turns before packing more
        }
        continue;
      }
      if (nprocs > 1 || !box_chunks(dset.getId(), offset.data(), count.data(), src, true))
        write_slab(dset, offset, count, vtype[i] == INT, src, xfer);

#ifdef H5_HAVE_PARALLEL
      if (verbose && nprocs > 1) {
        // The only way to tell whether HDF5 actually did collective I/O or
        // silently fell back; without it any scaling number is uninterpretable.
        uint32_t cause_local = 0, cause_global = 0;
        H5Pget_mpio_no_collective_cause(xfer.getId(), &cause_local,
                                        &cause_global);
        if (me == 0 && (cause_local || cause_global) && screen)
          fprintf(screen,
                  "dump hdf5: %s not collective (local cause 0x%x, "
                  "global cause 0x%x)\n",
                  labels[i].c_str(), cause_local, cause_global);
      }
#endif
    }

    file->getAttribute("time").write(time);
  } catch (const std::exception &e) {
    // the writes are collective: a rank that fails here leaves the others in
    // the next collective, so stop the job rather than hang it
    char msg[512];
    snprintf(msg, sizeof(msg), "Dump hdf5 write failed on %s: %s%s", filename,
             e.what(), atomic ? "; the previous dump is intact" : "");
    error->one(FLERR, msg);
  }
  return size_one;
}

/* ---------------------------------------------------------------------- */

int DumpHDF5::parse_fields(int narg, char **arg) {
  int i;
  for (int iarg = 4; iarg < narg; iarg++) {
    i = iarg - 4;
    char label[16];

    if (strcmp(arg[iarg], "site") == 0) {
      fields[i] = SITE;
      vtype[i] = INT;
      vindex[i] = 0;
      if (app->iarray == NULL)
        error->all(FLERR, "Dumping a quantity application does not support");
      labels[i] = "site";
    } else if (arg[iarg][0] == 'i' && isdigit(arg[iarg][1])) {
      fields[i] = IARRAY;
      vtype[i] = INT;
      vindex[i] = atoi(&arg[iarg][1]);
      if (latticeflag && (vindex[i] < 1 || vindex[i] > app->ninteger))
        error->all(FLERR, "Invalid keyword in dump command");
      snprintf(label, sizeof(label), "i%d", vindex[i]);
      labels[i] = label;
      vindex[i]--;
    } else if (arg[iarg][0] == 'd' && isdigit(arg[iarg][1])) {
      fields[i] = DARRAY;
      vtype[i] = DOUBLE;
      vindex[i] = atoi(&arg[iarg][1]);
      if (latticeflag && (vindex[i] < 1 || vindex[i] > app->ndouble))
        error->all(FLERR, "Invalid keyword in dump command");
      snprintf(label, sizeof(label), "d%d", vindex[i]);
      labels[i] = label;
      vindex[i]--;
    } else {
      // id/x/y/z/energy/propensity have no per-site array to snapshot; say so
      // instead of accepting them and silently writing nothing
      error->all(FLERR, "Dump hdf5 supports only 'site', iN and dN fields");
    }
  }

  return narg;
}

/* ---------------------------------------------------------------------- */

int DumpHDF5::modify_param(int narg, char **arg) {
  if (strcmp(arg[0], "chunk") == 0) {
    if (narg < 4)
      error->all(FLERR, "Illegal dump_modify command");
    for (int k = 0; k < 3; k++) {
      req_chunk[k] = atoi(arg[1 + k]);
      if (req_chunk[k] < 1)
        error->all(FLERR, "Dump_modify chunk values must be >= 1");
    }
    user_chunk = true;
    return 4;
  }
  if (strcmp(arg[0], "origin") == 0) {
    if (narg < 4)
      error->all(FLERR, "Illegal dump_modify command");
    for (int k = 0; k < 3; k++)
      req_origin[k] = atoi(arg[1 + k]);
    user_origin = true;
    return 4;
  }
  if (strcmp(arg[0], "extent") == 0) {
    if (narg < 4)
      error->all(FLERR, "Illegal dump_modify command");
    for (int k = 0; k < 3; k++) {
      req_extent[k] = atoi(arg[1 + k]);
      if (req_extent[k] < 1)
        error->all(FLERR, "Dump_modify extent values must be >= 1");
    }
    user_extent = true;
    return 4;
  }
  if (strcmp(arg[0], "compress") == 0) {
    if (narg < 2)
      error->all(FLERR, "Illegal dump_modify command");
    compress = atoi(arg[1]);
    if (compress < 0 || compress > 9)
      error->all(FLERR, "Dump_modify compress level must be 0 (off) to 9");
    return 2;
  }
  if (strcmp(arg[0], "atomic") == 0) {
    if (narg < 2)
      error->all(FLERR, "Illegal dump_modify command");
    if (strcmp(arg[1], "yes") == 0) atomic = 1;
    else if (strcmp(arg[1], "no") == 0) atomic = 0;
    else error->all(FLERR, "Illegal dump_modify command");
    return 2;
  }
  if (strcmp(arg[0], "turns") == 0) {
    if (narg < 2)
      error->all(FLERR, "Illegal dump_modify command");
    if (strcmp(arg[1], "yes") == 0) turns = 1;
    else if (strcmp(arg[1], "no") == 0) turns = 0;
    else error->all(FLERR, "Illegal dump_modify command");
    return 2;
  }
  if (strcmp(arg[0], "verbose") == 0) {
    if (narg < 2)
      error->all(FLERR, "Illegal dump_modify command");
    verbose = atoi(arg[1]);
    return 2;
  }
  return 0;
}

#endif // SPPARKS_HIGHFIVE
