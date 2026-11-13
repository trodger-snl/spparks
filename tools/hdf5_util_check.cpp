/* Self-check for src/hdf5_util.h.  Build and run (spparks-dev env active):

     H=$(dirname $(dirname $(which h5cc)))
     mpicxx -std=c++17 -I src -I $H/include tools/hdf5_util_check.cpp \
       -L $H/lib -Wl,-rpath,$H/lib -lhdf5 -o h5chk
     mpirun -np 1 ./h5chk h5chk.h5 && mpirun -np 2 ./h5chk h5chk.h5

   Prints "hdf5_util ok" on success; any failed check aborts. */

#undef NDEBUG
#include "hdf5_util.h"

#include <cassert>
#include <cstdio>
#include <cstring>

using namespace SPPARKS_NS;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int me, np;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  const std::string fn = argc > 1 ? argv[1] : "h5chk.h5";
  MPI_Comm world = MPI_COMM_WORLD;

  {
    h5::Handle fapl = h5::file_access(np > 1 ? &world : nullptr);
    h5::Handle f = h5::create_file(fn, fapl);

    // attributes: scalar, vector, string, overwrite
    h5::write_attr(f, "version", (int32_t)3);
    h5::write_attr(f, "dim_order", std::string("zyx"));
    h5::write_attr(f, "origin", std::vector<int64_t>{1, -2, 3});
    h5::write_attr(f, "time", 0.0);
    h5::write_attr(f, "time", 2.5);

    // chunked, growable 3-D int32 with a fill value; each rank writes its z plane
    h5::Handle dcpl(H5Pcreate(H5P_DATASET_CREATE));
    hsize_t ch[3] = {1, 2, 2};
    H5Pset_chunk(dcpl, 3, ch);
    int32_t fill = -1;
    H5Pset_fill_value(dcpl, H5T_NATIVE_INT32, &fill);
    h5::Handle d = h5::create_dataset(f, "site", h5::native<int32_t>(),
                                      {(size_t)np, 3, 4}, dcpl, true);
    h5::write_attr(d, "no_value", (int32_t)-1);
    std::vector<int32_t> plane(12);
    for (int i = 0; i < 12; i++) plane[i] = 100 * me + i;
    h5::write_slab(d, {(size_t)me, 0, 0}, {1, 3, 4}, plane.data());
    h5::check(H5Dset_extent(d, std::vector<hsize_t>{(hsize_t)np, 4, 4}.data()), "extent");
    assert((h5::dims(d) == std::vector<size_t>{(size_t)np, 4, 4}));

    // 2-D float64 table and a 1-D vector, collectively created, filled by rank 0
    h5::Handle t = h5::create_dataset(f, "table", h5::native<double>(), {5, 3});
    h5::Handle v = h5::create_dataset(f, "vec", h5::native<int64_t>(), {6});
    h5::Handle g(H5Gcreate2(f, "0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    if (me == 0) { // independent writes: under MPI-IO only one rank writes
      std::vector<double> tb(15);
      for (int i = 0; i < 15; i++) tb[i] = i;
      h5::write_slab(t, {0, 0}, {5, 3}, tb.data());
      if (np == 1) h5::write(v, std::vector<int64_t>{0, 10, 20, 30, 40, 50});
      else h5::write_slab(v, {0}, {6}, std::vector<int64_t>{0, 10, 20, 30, 40, 50}.data());
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  {
    h5::Handle f = h5::open_file(fn, H5F_ACC_RDONLY);
    int32_t ver;
    double t;
    std::vector<int64_t> o;
    h5::read_attr(f, "version", ver);
    h5::read_attr(f, "time", t);
    h5::read_attr(f, "origin", o);
    assert(ver == 3 && t == 2.5 && (o == std::vector<int64_t>{1, -2, 3}));
    assert(h5::has_attr(f, "dim_order") && !h5::has_attr(f, "nope"));
    assert(h5::exists(f, "site") && !h5::exists(f, "nope"));
    assert((h5::datasets(f) == std::vector<std::string>{"site", "table", "vec"}));

    // string attribute is variable-length UTF-8, as HighFive wrote it
    h5::Handle a(H5Aopen(f, "dim_order", H5P_DEFAULT)), ty(H5Aget_type(a));
    assert(H5Tis_variable_str(ty) > 0 && H5Tget_cset(ty) == H5T_CSET_UTF8);
    h5::Handle sp(H5Aget_space(a));
    assert(H5Sget_simple_extent_type(sp) == H5S_SCALAR);

    // slab read back, with the grown margin at the fill value
    h5::Handle d = h5::open(f, "site");
    std::vector<int32_t> box(16);
    h5::read_slab(d, {(size_t)(np - 1), 0, 0}, {1, 4, 4}, box.data());
    for (int i = 0; i < 12; i++) assert(box[i] == 100 * (np - 1) + i);
    for (int i = 12; i < 16; i++) assert(box[i] == -1);
    int32_t nv;
    h5::read_attr(d, "no_value", nv);
    assert(nv == -1);

    // whole reads and row unions, with conversion int64 -> unsigned
    std::vector<double> tb;
    h5::read(h5::open(f, "table"), tb);
    assert(tb.size() == 15 && tb[14] == 14);
    std::vector<double> rows;
    h5::read_rows(h5::open(f, "table"), {{0, 1}, {3, 5}}, 2, rows);
    assert((rows == std::vector<double>{0, 1, 9, 10, 12, 13}));
    std::vector<unsigned> r1;
    h5::read_rows(h5::open(f, "vec"), {{1, 2}, {4, 6}}, 0, r1);
    assert((r1 == std::vector<unsigned>{10, 40, 50}));

    // failures throw with HDF5's description (its own stack print is off)
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    bool threw = false;
    try {
      h5::open(f, "missing");
    } catch (const std::exception &e) {
      threw = std::strcmp(e.what(), "unable to open missing: object 'missing' doesn't exist") == 0;
      if (!threw) printf("unexpected message: %s\n", e.what());
    }
    assert(threw);
    try {
      double x;
      h5::read_attr(f, "origin", x);
      threw = false;
    } catch (const std::exception &) {
      threw = true;
    }
    assert(threw);
  }

  if (me == 0) printf("hdf5_util ok (np=%d)\n", np);
  MPI_Finalize();
}
