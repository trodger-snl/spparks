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

using namespace SPPARKS_NS;

// customize by adding keyword to 1st enum
// (kept identical to dump_stitch/dump_text so field parsing reads the same)

enum { ID, SITE, X, Y, Z, ENERGY, PROPENSITY, IARRAY, DARRAY };
enum { INT, DOUBLE, TAGINT };

// Chunks below this are dominated by chunk-index overhead; above this a
// partially touched chunk wastes too much I/O.
static const int64_t CHUNK_MIN_BYTES = 256 * 1024;
static const int64_t CHUNK_MAX_BYTES = 4 * 1024 * 1024;

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
  atomic = 1;
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
   choose a chunk shape that tiles the per-rank subdomains.

   ceil(extent/procgrid) is exactly rank 0's subdomain extent in each
   dimension and every other rank is within one site of it, so chunks land
   approximately one per rank.  That keeps a chunk from being shared between
   ranks, which is what would otherwise turn a collective write into a
   read-modify-write.  The byte clamp keeps the chunk index from exploding on
   one end and from wasting I/O on a partial touch at the other.
------------------------------------------------------------------------- */

void DumpHDF5::choose_chunk(const int64_t *extent, size_t elemsize,
                            int64_t *out) {
  if (user_chunk) {
    for (int d = 0; d < 3; d++)
      out[d] = std::max((int64_t)1, std::min(req_chunk[d], extent[d]));
    return;
  }

  for (int d = 0; d < 3; d++) {
    int64_t p = domain->procgrid[d] > 0 ? domain->procgrid[d] : 1;
    out[d] = (extent[d] + p - 1) / p;
    if (out[d] < 1)
      out[d] = 1;
    if (out[d] > extent[d])
      out[d] = extent[d];
  }

  // shrink the slowest-varying dimension first (z, then y) so that x stays
  // contiguous for as long as possible
  int order[3] = {2, 1, 0};
  for (int k = 0; k < 3; k++) {
    int d = order[k];
    while (out[0] * out[1] * out[2] * (int64_t)elemsize > CHUNK_MAX_BYTES &&
           out[d] > 1)
      out[d] = (out[d] + 1) / 2;
  }

  // grow x, then y, then z back toward the minimum useful chunk, never past
  // the current extent
  int grow[3] = {0, 1, 2};
  for (int k = 0; k < 3; k++) {
    int d = grow[k];
    while (out[0] * out[1] * out[2] * (int64_t)elemsize < CHUNK_MIN_BYTES &&
           out[d] < extent[d]) {
      int64_t next = std::min(extent[d], out[d] * 2);
      if (next == out[d])
        break;
      out[d] = next;
    }
  }
}

/* ----------------------------------------------------------------------
   create the dataset for one field if it is not already in the file
------------------------------------------------------------------------- */

void DumpHDF5::ensure_dataset(const std::string &label, int type) {
  if (file->exist(label))
    return;

  size_t elemsize = (type == INT) ? sizeof(int32_t) : sizeof(double);
  int64_t c[3];
  choose_chunk(dims, elemsize, c);
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
  // materialize the fill immediately, so a reader can tell "never written"
  // from "written as zero"
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
#ifdef H5_HAVE_PARALLEL
    HighFive::FileAccessProps fapl;
    fapl.add(HighFive::MPIOFileAccess(world, MPI_INFO_NULL));
    // Unlike the HDF5 temperature-source reader, every rank takes the same
    // path through every dump, so collective metadata would be safe here.  It
    // is still left off in v1: it converts a subtle divergence into a hang
    // rather than a slowdown, and it is not needed for correctness.
    file = std::make_shared<HighFive::File>(
        fname,
        exists ? HighFive::File::ReadWrite
               : (HighFive::File::ReadWrite | HighFive::File::Create |
                  HighFive::File::Truncate),
        fapl);
#else
    file = std::make_shared<HighFive::File>(
        fname,
        exists ? HighFive::File::ReadWrite
               : (HighFive::File::ReadWrite | HighFive::File::Create |
                  HighFive::File::Truncate));
#endif
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
                   fname, k, (long long)lo[k], (long long)origin[k]);
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
   grow every dataset so that 'needed' fits.

   H5Dset_extent is collective, so every rank calls it unconditionally even
   when nothing changed for that rank -- branching here would hang the ranks
   that do need the resize.
------------------------------------------------------------------------- */

void DumpHDF5::extend_to(const int64_t *needed) {
  int64_t want[3];
  for (int k = 0; k < 3; k++)
    want[k] = std::max(dims[k], needed[k]);

  bool grew = false;
  for (int k = 0; k < 3; k++)
    if (want[k] != dims[k])
      grew = true;
  if (!grew)
    return;

  for (int k = 0; k < 3; k++)
    dims[k] = want[k];

  std::vector<size_t> shape = {(size_t)dims[2], (size_t)dims[1],
                               (size_t)dims[0]};
  for (int i = 0; i < size_one; i++)
    file->getDataSet(labels[i]).resize(shape);

  file->getAttribute("dims").write(std::vector<int64_t>(dims, dims + 3));
}

/* ----------------------------------------------------------------------
   write a snapshot: overwrite each field in place
------------------------------------------------------------------------- */

void DumpHDF5::write(double time) {
  if (!atomic) {
    if (!opened)
      open_file(filename);
    write_fields(time);
    return;
  }

  // write-then-rename: the file under its own name is always a complete
  // earlier dump, whatever happens to this process mid-write
  std::string tmp = std::string(filename) + ".tmp";
  stage_copy(tmp.c_str());
  open_file(tmp.c_str());
  write_fields(time);
  try {
    file->flush(); // H5Fflush; under MPI-IO it ends in MPI_File_sync
  } catch (const std::exception &e) {
    error->all(FLERR, (std::string("Dump hdf5 flush failed: ") + e.what()).c_str());
  }
  file.reset(); // collective close
  opened = false;
  commit(tmp.c_str());
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

void DumpHDF5::write_fields(double time) {

  // this rank's owned box in absolute lattice indices, half-open
  int64_t lo[3] = {applattice->xlo_me_simple, applattice->ylo_me_simple,
                   applattice->zlo_me_simple};
  int64_t n[3] = {applattice->xhi_me_simple - applattice->xlo_me_simple + 1,
                  applattice->yhi_me_simple - applattice->ylo_me_simple + 1,
                  applattice->zhi_me_simple - applattice->zlo_me_simple + 1};

  for (int k = 0; k < 3; k++)
    if (lo[k] < origin[k])
      error->all(FLERR, "Dump hdf5 subdomain starts below the file origin");

  int64_t mine[3], needed[3];
  for (int k = 0; k < 3; k++)
    mine[k] = lo[k] - origin[k] + n[k];
  MPI_Allreduce(mine, needed, 3, MPI_LONG_LONG, MPI_MAX, world);

  try {
    for (int i = 0; i < size_one; i++)
      ensure_dataset(labels[i], vtype[i]);

    extend_to(needed);

    std::vector<size_t> offset = {(size_t)(lo[2] - origin[2]),
                                  (size_t)(lo[1] - origin[1]),
                                  (size_t)(lo[0] - origin[0])};
    std::vector<size_t> count = {(size_t)n[2], (size_t)n[1], (size_t)n[0]};

    HighFive::DataTransferProps xfer;
#ifdef H5_HAVE_PARALLEL
    xfer.add(HighFive::UseCollectiveIO{});
#endif

    for (int i = 0; i < size_one; i++) {
      HighFive::DataSet dset = file->getDataSet(labels[i]);
      HighFive::HyperSlab slab{HighFive::RegularHyperSlab(offset, count)};
      if (vtype[i] == INT) {
        // site ordering for a simple lattice is x fastest, then y, then z --
        // the same order as the [z][y][x] hyperslab, so no repacking
        const int *src = (fields[i] == SITE) ? app->iarray[0]
                                             : app->iarray[vindex[i]];
        dset.select(slab).write_raw(src, HighFive::AtomicType<int32_t>(), xfer);
      } else {
        const double *src = app->darray[vindex[i]];
        dset.select(slab).write_raw(src, HighFive::AtomicType<double>(), xfer);
      }

#ifdef H5_HAVE_PARALLEL
      if (verbose) {
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
    char msg[512];
    snprintf(msg, sizeof(msg), "Dump hdf5 write failed on %s: %s", filename,
             e.what());
    error->all(FLERR, msg);
  }
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
  if (strcmp(arg[0], "atomic") == 0) {
    if (narg < 2)
      error->all(FLERR, "Illegal dump_modify command");
    if (strcmp(arg[1], "yes") == 0) atomic = 1;
    else if (strcmp(arg[1], "no") == 0) atomic = 0;
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
