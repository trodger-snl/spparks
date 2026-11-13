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

#ifndef SPK_HDF5_UTIL_H
#define SPK_HDF5_UTIL_H

// The few HDF5 C API helpers SPPARKS needs: an id that closes itself, and
// typed attribute / dataset / slab I/O.  Every failure throws
// std::runtime_error with HDF5's innermost error description, so a caller
// wraps a block of calls in one try/catch(std::exception).
//
// Files written match what HighFive 2.x wrote: native numeric types, scalar
// attributes in a scalar dataspace, vectors 1-D, strings variable-length
// UTF-8 (h5py reads them as str).  tools/hdf5_util_check.cpp exercises it.

#include "hdf5.h"
#include "mpi.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace SPPARKS_NS {
namespace h5 {

[[noreturn]] inline void fail(const std::string &what) {
  std::string detail;
  hid_t st = H5Eget_current_stack();
  if (st >= 0) {
    H5Ewalk2(st, H5E_WALK_UPWARD,
             [](unsigned n, const H5E_error2_t *e, void *d) -> herr_t {
               if (n == 0 && e->desc) *static_cast<std::string *>(d) = e->desc;
               return 0;
             },
             &detail);
    H5Eclose_stack(st);
  }
  throw std::runtime_error(detail.empty() ? what : what + ": " + detail);
}

inline hid_t check(hid_t rc, const std::string &what) {
  if (rc < 0) fail(what);
  return rc;
}

// an hid_t of any kind (file, group, dataset, space, type, plist) that is
// released when it goes out of scope
class Handle {
 public:
  explicit Handle(hid_t id = -1) : id_(id) {}
  Handle(Handle &&o) noexcept : id_(o.id_) { o.id_ = -1; }
  Handle &operator=(Handle &&o) noexcept {
    std::swap(id_, o.id_);
    return *this;
  }
  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  ~Handle() {
    if (id_ >= 0) H5Idec_ref(id_);
  }
  operator hid_t() const { return id_; }
  // hand the id to a caller that closes it itself (H5Idec_ref)
  hid_t release() { return std::exchange(id_, -1); }

 private:
  hid_t id_;
};

template <class T> hid_t native() {
  if constexpr (std::is_same_v<T, double>) return H5T_NATIVE_DOUBLE;
  else if constexpr (std::is_same_v<T, float>) return H5T_NATIVE_FLOAT;
  else if constexpr (std::is_same_v<T, int32_t>) return H5T_NATIVE_INT32;
  else if constexpr (std::is_same_v<T, int64_t>) return H5T_NATIVE_INT64;
  else if constexpr (std::is_same_v<T, unsigned>) return H5T_NATIVE_UINT;
  else if constexpr (std::is_same_v<T, unsigned char>) return H5T_NATIVE_UCHAR;
  else static_assert(sizeof(T) == 0, "h5::native: no HDF5 type for T");
}

/* ---------------------------------------------------------------------- */
// files and objects

// comm = MPI-IO over that communicator (serial HDF5 ignores it); lock =
// false drops HDF5's flock, which fails on file systems mounted without it
inline Handle file_access(const MPI_Comm *comm = nullptr, bool lock = true) {
  Handle p(check(H5Pcreate(H5P_FILE_ACCESS), "H5Pcreate"));
#ifdef H5_HAVE_PARALLEL
  if (comm) check(H5Pset_fapl_mpio(p, *comm, MPI_INFO_NULL), "H5Pset_fapl_mpio");
#else
  (void)comm;
#endif
  if (!lock) check(H5Pset_file_locking(p, false, true), "H5Pset_file_locking");
  return p;
}

// flags H5F_ACC_RDONLY or H5F_ACC_RDWR
inline Handle open_file(const std::string &name, unsigned flags, hid_t fapl = H5P_DEFAULT) {
  return Handle(check(H5Fopen(name.c_str(), flags, fapl), "unable to open file " + name));
}

// truncates an existing file
inline Handle create_file(const std::string &name, hid_t fapl = H5P_DEFAULT,
                          hid_t fcpl = H5P_DEFAULT) {
  return Handle(check(H5Fcreate(name.c_str(), H5F_ACC_TRUNC, fcpl, fapl),
                      "unable to create file " + name));
}

// a group or a dataset
inline Handle open(hid_t loc, const std::string &name) {
  return Handle(check(H5Oopen(loc, name.c_str(), H5P_DEFAULT), "unable to open " + name));
}

inline bool exists(hid_t loc, const std::string &name) {
  return H5Lexists(loc, name.c_str(), H5P_DEFAULT) > 0;
}

// names of the datasets directly under loc, in name order
inline std::vector<std::string> datasets(hid_t loc) {
  H5G_info_t gi;
  check(H5Gget_info(loc, &gi), "unable to list objects");
  std::vector<std::string> out;
  for (hsize_t i = 0; i < gi.nlinks; i++) {
    ssize_t n = H5Lget_name_by_idx(loc, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0,
                                   H5P_DEFAULT);
    std::string s(check(n, "unable to list objects"), '\0');
    check(H5Lget_name_by_idx(loc, ".", H5_INDEX_NAME, H5_ITER_INC, i, s.data(), n + 1,
                             H5P_DEFAULT),
          "unable to list objects");
    if (H5Iget_type(open(loc, s)) == H5I_DATASET) out.push_back(s);
  }
  return out;
}

inline std::vector<size_t> dims(hid_t dset) {
  Handle sp(check(H5Dget_space(dset), "H5Dget_space"));
  std::vector<hsize_t> d(check(H5Sget_simple_extent_ndims(sp), "H5Sget_simple_extent_ndims"));
  H5Sget_simple_extent_dims(sp, d.data(), nullptr);
  return std::vector<size_t>(d.begin(), d.end());
}

// dcpl carries chunking/filters/fill; unlimited lets every dimension grow
inline Handle create_dataset(hid_t loc, const std::string &name, hid_t type,
                             const std::vector<size_t> &shape, hid_t dcpl = H5P_DEFAULT,
                             bool unlimited = false) {
  std::vector<hsize_t> d(shape.begin(), shape.end()), mx(d.size(), H5S_UNLIMITED);
  Handle sp(check(H5Screate_simple((int)d.size(), d.data(), unlimited ? mx.data() : nullptr),
                  "H5Screate_simple"));
  return Handle(check(H5Dcreate2(loc, name.c_str(), type, sp, H5P_DEFAULT, dcpl, H5P_DEFAULT),
                      "unable to create dataset " + name));
}

/* ---------------------------------------------------------------------- */
// attributes on a file, group or dataset

template <class T> void read_attr(hid_t obj, const char *name, std::vector<T> &v) {
  Handle a(check(H5Aopen(obj, name, H5P_DEFAULT), std::string("unable to open attribute ") + name));
  Handle sp(check(H5Aget_space(a), "H5Aget_space"));
  v.resize(check(H5Sget_simple_extent_npoints(sp), "H5Sget_simple_extent_npoints"));
  check(H5Aread(a, native<T>(), v.data()), std::string("unable to read attribute ") + name);
}

template <class T> void read_attr(hid_t obj, const char *name, T &v) {
  std::vector<T> tmp;
  read_attr(obj, name, tmp);
  if (tmp.size() != 1) throw std::runtime_error(std::string("attribute ") + name + " is not a scalar");
  v = tmp[0];
}

inline bool has_attr(hid_t obj, const char *name) { return H5Aexists(obj, name) > 0; }

// create the attribute, or overwrite the one already there
inline void put_attr(hid_t obj, const char *name, hid_t type, hid_t space, const void *buf) {
  const std::string what = std::string("unable to write attribute ") + name;
  Handle a(check(has_attr(obj, name) ? H5Aopen(obj, name, H5P_DEFAULT)
                                     : H5Acreate2(obj, name, type, space, H5P_DEFAULT, H5P_DEFAULT),
                 what));
  check(H5Awrite(a, type, buf), what);
}

template <class T> void write_attr(hid_t obj, const char *name, const std::vector<T> &v) {
  hsize_t n = v.size();
  Handle sp(check(H5Screate_simple(1, &n, nullptr), "H5Screate_simple"));
  put_attr(obj, name, native<T>(), sp, v.data());
}

template <class T> void write_attr(hid_t obj, const char *name, const T &v) {
  Handle sp(check(H5Screate(H5S_SCALAR), "H5Screate"));
  put_attr(obj, name, native<T>(), sp, &v);
}

inline void write_attr(hid_t obj, const char *name, const std::string &v) {
  Handle t(check(H5Tcopy(H5T_C_S1), "H5Tcopy"));
  check(H5Tset_size(t, H5T_VARIABLE), "H5Tset_size");
  check(H5Tset_cset(t, H5T_CSET_UTF8), "H5Tset_cset");
  Handle sp(check(H5Screate(H5S_SCALAR), "H5Screate"));
  const char *p = v.c_str();
  put_attr(obj, name, t, sp, &p);
}

/* ---------------------------------------------------------------------- */
// dataset I/O; HDF5 converts between the stored and the native type

template <class T> void read(hid_t dset, std::vector<T> &v) {
  Handle sp(check(H5Dget_space(dset), "H5Dget_space"));
  v.resize(check(H5Sget_simple_extent_npoints(sp), "H5Sget_simple_extent_npoints"));
  check(H5Dread(dset, native<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data()), "unable to read dataset");
}

template <class T> void write(hid_t dset, const std::vector<T> &v) {
  check(H5Dwrite(dset, native<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data()),
        "unable to write dataset");
}

// the box at offset/count; the memory space has the box's shape (a flat one
// sends a chunked read down HDF5's per-element mapping path, ~4x slower)
template <class T>
void slab(hid_t dset, const std::vector<size_t> &offset, const std::vector<size_t> &count,
          T *buf, bool write, hid_t xfer = H5P_DEFAULT) {
  std::vector<hsize_t> o(offset.begin(), offset.end()), c(count.begin(), count.end());
  Handle fs(check(H5Dget_space(dset), "H5Dget_space"));
  check(H5Sselect_hyperslab(fs, H5S_SELECT_SET, o.data(), nullptr, c.data(), nullptr),
        "unable to select hyperslab");
  Handle ms(check(H5Screate_simple((int)c.size(), c.data(), nullptr), "H5Screate_simple"));
  if (write)
    check(H5Dwrite(dset, native<T>(), ms, fs, xfer, buf), "unable to write hyperslab");
  else
    check(H5Dread(dset, native<T>(), ms, fs, xfer, buf), "unable to read hyperslab");
}

template <class T>
void read_slab(hid_t dset, const std::vector<size_t> &offset, const std::vector<size_t> &count,
               T *buf, hid_t xfer = H5P_DEFAULT) {
  slab(dset, offset, count, buf, false, xfer);
}

template <class T>
void write_slab(hid_t dset, const std::vector<size_t> &offset, const std::vector<size_t> &count,
                const T *buf, hid_t xfer = H5P_DEFAULT) {
  slab(dset, offset, count, const_cast<T *>(buf), true, xfer);
}

// rows [a,b) of a 1-D dataset (ncols = 0), or columns [0,ncols) of those
// rows of a 2-D one, packed in file order into a flat v
template <class T>
void read_rows(hid_t dset, const std::vector<std::array<size_t, 2>> &rows, size_t ncols,
               std::vector<T> &v) {
  if (rows.empty()) throw std::runtime_error("no rows to read");
  Handle fs(check(H5Dget_space(dset), "H5Dget_space"));
  hsize_t n = 0;
  for (size_t r = 0; r < rows.size(); r++) {
    hsize_t start[2] = {rows[r][0], 0}, cnt[2] = {rows[r][1] - rows[r][0], ncols};
    check(H5Sselect_hyperslab(fs, r ? H5S_SELECT_OR : H5S_SELECT_SET, start, nullptr, cnt, nullptr),
          "unable to select rows");
    n += cnt[0] * (ncols ? ncols : 1);
  }
  Handle ms(check(H5Screate_simple(1, &n, nullptr), "H5Screate_simple"));
  v.resize(n);
  check(H5Dread(dset, native<T>(), ms, fs, H5P_DEFAULT, v.data()), "unable to read rows");
}

} // namespace h5
} // namespace SPPARKS_NS

#endif
