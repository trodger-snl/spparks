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

#ifndef SPK_KOKKOS_DOUBLE_BUFFER_H
#define SPK_KOKKOS_DOUBLE_BUFFER_H

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

namespace SPPARKS_NS {

/* ----------------------------------------------------------------------
   KokkosDoubleBuffer - Template class for double-buffered arrays

   PURPOSE:
   Provides race-free parallel updates in rejection Monte Carlo sweeps by
   maintaining separate read and write buffers on device. All threads read
   from read_buffer (immutable during sweep) and write to write_buffer
   (independent writes). After sweep, buffers are swapped via pointer exchange.

   USAGE:
   - Allocate once with size = nlocal + nghost
   - Call sync_to_device_full() during initialization (sets both buffers)
   - Call sync_to_device() after MPI communication (updates both buffers)
   - Call sync_to_host() before MPI communication (reads from read buffer)
   - Call swap() after each sweep (O(1) pointer swap)

   MEMORY ARCHITECTURE:
   Host Array (T*)  ↔  DualView  ↔  Device View
                                        ↓
                              read_buffer  (device only)
                              write_buffer (device only)

   REQUIREMENTS:
   - Apps must handle any data transformations (e.g., normalization)
     before/after calling sync methods
   - Framework is Kokkos-only, not compatible with non-Kokkos apps
   - Supports only int and double types (instantiated at end of header)
------------------------------------------------------------------------- */

template<typename T>
class KokkosDoubleBuffer {
public:
  using view_type = Kokkos::View<T*>;
  using dualview_type = Kokkos::DualView<T*>;

  // DualView for host-device staging (used for MPI sync)
  dualview_type staging;

  // Device-only read/write buffers (for double-buffering)
  view_type read_buffer;
  view_type write_buffer;

  KokkosDoubleBuffer() : _size(0), _enable_profiling(false),
    _count_to_device_full(0), _count_to_device(0),
    _count_to_device_selective(0), _count_to_host(0),
    _time_to_device_full(0.0), _time_to_device(0.0),
    _time_to_device_selective(0.0), _time_to_host(0.0) {}

  /* ----------------------------------------------------------------------
     Allocate DualView and both device buffers
     Call whenever nlocal+nghost changes
  ---------------------------------------------------------------------- */
  void allocate(const char* label, size_t n) {
    _size = n;
    staging = dualview_type(label, n);

    // Allocate separate read/write buffers on device
    char read_label[256], write_label[256];
    snprintf(read_label, 256, "%s_read", label);
    snprintf(write_label, 256, "%s_write", label);

    read_buffer = view_type(read_label, n);
    write_buffer = view_type(write_label, n);
  }

  /* ----------------------------------------------------------------------
     Swap read/write buffer pointers (O(1) operation)
     Call after each sweep to make writes visible for next sweep
  ---------------------------------------------------------------------- */
  void swap() {
    auto temp = read_buffer;
    read_buffer = write_buffer;
    write_buffer = temp;
  }

  /* ----------------------------------------------------------------------
     Full sync: host → DualView → both device buffers
     Use ONLY during initialization to set both buffers to same starting state

     IMPORTANT: Caller is responsible for any data transformations
     (e.g., quaternion normalization) before calling this method
  ---------------------------------------------------------------------- */
  void sync_to_device_full(const T* host_data) {
    Kokkos::Timer timer;
    if (_enable_profiling) timer.reset();

    auto h_view = staging.h_view;
    for (size_t i = 0; i < _size; i++) {
      h_view(i) = host_data[i];
    }
    staging.modify_host();
    staging.sync_device();

    // Copy to BOTH buffers (initialization)
    Kokkos::deep_copy(read_buffer, staging.d_view);
    Kokkos::deep_copy(write_buffer, staging.d_view);

    if (_enable_profiling) {
      _time_to_device_full += timer.seconds();
      _count_to_device_full++;
    }
  }

  /* ----------------------------------------------------------------------
     MPI sync: host → DualView → both device buffers
     Use after MPI communication to update ghost sites and resync all sites

     WHY BOTH BUFFERS: After MPI, host has authoritative data (updated ghosts).
     Both buffers must be synchronized because:
     - read_buffer needs updated ghosts for energy calculations
     - write_buffer needs same starting point to maintain consistency

     IMPORTANT: Caller is responsible for any data transformations
     (e.g., quaternion normalization) before calling this method
  ---------------------------------------------------------------------- */
  void sync_to_device(const T* host_data) {
    Kokkos::Timer timer;
    if (_enable_profiling) timer.reset();

    auto h_view = staging.h_view;
    for (size_t i = 0; i < _size; i++) {
      h_view(i) = host_data[i];
    }
    staging.modify_host();
    staging.sync_device();

    // Copy to BOTH buffers (critical for MPI + sectors correctness)
    Kokkos::deep_copy(read_buffer, staging.d_view);
    Kokkos::deep_copy(write_buffer, staging.d_view);

    if (_enable_profiling) {
      _time_to_device += timer.seconds();
      _count_to_device++;
    }
  }

  /* ----------------------------------------------------------------------
     Selective MPI sync: host → DualView → selected device buffer(s)
     Use for optimized sync after MPI communication

     OPTIMIZATION: After MPI forward communication, only read_buffer needs
     updating (write_buffer will be overwritten in next sweep anyway).
     This reduces device memory copies by 33%.

     sync_read:  If true, sync read_buffer (needed for next sweep)
     sync_write: If true, sync write_buffer (only needed for initialization)

     IMPORTANT: Caller is responsible for any data transformations
     (e.g., quaternion normalization) before calling this method
  ---------------------------------------------------------------------- */
  void sync_to_device_selective(const T* host_data, bool sync_read = true,
                                  bool sync_write = false) {
    Kokkos::Timer timer;
    if (_enable_profiling) timer.reset();

    auto h_view = staging.h_view;
    for (size_t i = 0; i < _size; i++) {
      h_view(i) = host_data[i];
    }
    staging.modify_host();
    staging.sync_device();

    // Selectively copy to buffers based on flags
    if (sync_read) {
      Kokkos::deep_copy(read_buffer, staging.d_view);
    }
    if (sync_write) {
      Kokkos::deep_copy(write_buffer, staging.d_view);
    }

    if (_enable_profiling) {
      _time_to_device_selective += timer.seconds();
      _count_to_device_selective++;
    }
  }

  /* ----------------------------------------------------------------------
     Device → host sync: read_buffer → DualView → host
     Use before MPI communication to get current device state

     Copies from read_buffer (the "final state" after buffer swaps)

     IMPORTANT: Caller is responsible for any data transformations
     (e.g., quaternion normalization) after calling this method
  ---------------------------------------------------------------------- */
  void sync_to_host(T* host_data) {
    Kokkos::Timer timer;
    if (_enable_profiling) timer.reset();

    // Copy from read buffer (final state after swaps) to DualView
    Kokkos::deep_copy(staging.d_view, read_buffer);

    staging.modify_device();
    staging.sync_host();

    // Copy to host array
    auto h_view = staging.h_view;
    for (size_t i = 0; i < _size; i++) {
      host_data[i] = h_view(i);
    }

    if (_enable_profiling) {
      _time_to_host += timer.seconds();
      _count_to_host++;
    }
  }

  /* ----------------------------------------------------------------------
     Get current size
  ---------------------------------------------------------------------- */
  size_t size() const { return _size; }

  /* ----------------------------------------------------------------------
     Profiling control - enable/disable timing and counting
  ---------------------------------------------------------------------- */
  void enable_profiling(bool enable = true) { _enable_profiling = enable; }
  bool is_profiling_enabled() const { return _enable_profiling; }

  /* ----------------------------------------------------------------------
     Reset profiling counters and timers
  ---------------------------------------------------------------------- */
  void reset_profiling() {
    _count_to_device_full = 0;
    _count_to_device = 0;
    _count_to_device_selective = 0;
    _count_to_host = 0;
    _time_to_device_full = 0.0;
    _time_to_device = 0.0;
    _time_to_device_selective = 0.0;
    _time_to_host = 0.0;
  }

  /* ----------------------------------------------------------------------
     Get profiling statistics
  ---------------------------------------------------------------------- */
  void get_profiling_stats(int& count_to_device_full, int& count_to_device,
                           int& count_to_device_selective, int& count_to_host,
                           double& time_to_device_full, double& time_to_device,
                           double& time_to_device_selective, double& time_to_host) const {
    count_to_device_full = _count_to_device_full;
    count_to_device = _count_to_device;
    count_to_device_selective = _count_to_device_selective;
    count_to_host = _count_to_host;
    time_to_device_full = _time_to_device_full;
    time_to_device = _time_to_device;
    time_to_device_selective = _time_to_device_selective;
    time_to_host = _time_to_host;
  }

  /* ----------------------------------------------------------------------
     Get total sync count
  ---------------------------------------------------------------------- */
  int get_total_sync_count() const {
    return _count_to_device_full + _count_to_device +
           _count_to_device_selective + _count_to_host;
  }

  /* ----------------------------------------------------------------------
     Get total sync time
  ---------------------------------------------------------------------- */
  double get_total_sync_time() const {
    return _time_to_device_full + _time_to_device +
           _time_to_device_selective + _time_to_host;
  }

  /* ----------------------------------------------------------------------
     Get bytes transferred (estimate)
  ---------------------------------------------------------------------- */
  size_t get_bytes_per_sync() const {
    return _size * sizeof(T);
  }

  /* ----------------------------------------------------------------------
     Print profiling statistics to file pointer (e.g., screen or logfile)
  ---------------------------------------------------------------------- */
  void print_profiling_stats(FILE* fp, const char* label) const {
    if (!_enable_profiling) return;

    int total_count = get_total_sync_count();
    double total_time = get_total_sync_time();
    size_t bytes_per = get_bytes_per_sync();

    fprintf(fp, "Sync profiling for %s:\n", label);
    fprintf(fp, "  Total syncs:           %d\n", total_count);
    fprintf(fp, "  Total time:            %.6f s\n", total_time);
    fprintf(fp, "  Bytes per sync:        %zu (%.2f KB, %.2f MB)\n",
            bytes_per, bytes_per/1024.0, bytes_per/(1024.0*1024.0));
    fprintf(fp, "  sync_to_device_full:   count=%d time=%.6f s (%.3f%%)\n",
            _count_to_device_full, _time_to_device_full,
            total_time > 0 ? 100.0*_time_to_device_full/total_time : 0.0);
    fprintf(fp, "  sync_to_device:        count=%d time=%.6f s (%.3f%%)\n",
            _count_to_device, _time_to_device,
            total_time > 0 ? 100.0*_time_to_device/total_time : 0.0);
    fprintf(fp, "  sync_to_device_select: count=%d time=%.6f s (%.3f%%)\n",
            _count_to_device_selective, _time_to_device_selective,
            total_time > 0 ? 100.0*_time_to_device_selective/total_time : 0.0);
    fprintf(fp, "  sync_to_host:          count=%d time=%.6f s (%.3f%%)\n",
            _count_to_host, _time_to_host,
            total_time > 0 ? 100.0*_time_to_host/total_time : 0.0);

    if (total_count > 0) {
      fprintf(fp, "  Avg time per sync:     %.6f s\n", total_time/total_count);
      fprintf(fp, "  Est. total MB xfer:    %.2f MB\n",
              (bytes_per * total_count)/(1024.0*1024.0));
    }
  }

private:
  size_t _size;

  // Profiling enable flag
  bool _enable_profiling;

  // Profiling counters
  int _count_to_device_full;
  int _count_to_device;
  int _count_to_device_selective;
  int _count_to_host;

  // Profiling timers (accumulated time in seconds)
  double _time_to_device_full;
  double _time_to_device;
  double _time_to_device_selective;
  double _time_to_host;
};

/* ----------------------------------------------------------------------
   KokkosDoubleBufferManager - Manages multiple double-buffered arrays

   PURPOSE:
   Simplifies management of multiple arrays with coordinated operations.
   Reduces boilerplate code in apps that use multiple buffered arrays.

   USAGE EXAMPLE:

   KokkosDoubleBufferManager manager;
   manager.add_int_buffer("spin");
   manager.add_double_buffer("q0");
   manager.add_double_buffer("qx");
   manager.add_double_buffer("qy");
   manager.add_double_buffer("qz");

   manager.allocate_all(nlocal + nghost);

   // Access individual buffers
   auto& spin_buf = manager.get_int_buffer("spin");
   auto& q0_buf = manager.get_double_buffer("q0");

   // Coordinated operations
   manager.swap_all();

   // Individual sync (for custom transformations)
   spin_buf.sync_to_device(spin);
   q0_buf.sync_to_device(q0_normalized);  // After app-specific normalization
------------------------------------------------------------------------- */

class KokkosDoubleBufferManager {
public:
  /* ----------------------------------------------------------------------
     Add buffers before allocation
  ---------------------------------------------------------------------- */
  void add_int_buffer(const char* name) {
    _int_buffers[name] = KokkosDoubleBuffer<int>();
    _int_buffer_order.push_back(name);
  }

  void add_double_buffer(const char* name) {
    _double_buffers[name] = KokkosDoubleBuffer<double>();
    _double_buffer_order.push_back(name);
  }

  /* ----------------------------------------------------------------------
     Allocate all buffers with same size
  ---------------------------------------------------------------------- */
  void allocate_all(size_t n) {
    for (const auto& name : _int_buffer_order) {
      _int_buffers[name].allocate(name.c_str(), n);
    }
    for (const auto& name : _double_buffer_order) {
      _double_buffers[name].allocate(name.c_str(), n);
    }
  }

  /* ----------------------------------------------------------------------
     Swap all buffers
  ---------------------------------------------------------------------- */
  void swap_all() {
    for (auto& pair : _int_buffers) {
      pair.second.swap();
    }
    for (auto& pair : _double_buffers) {
      pair.second.swap();
    }
  }

  /* ----------------------------------------------------------------------
     Access individual buffers
  ---------------------------------------------------------------------- */
  KokkosDoubleBuffer<int>& get_int_buffer(const char* name) {
    return _int_buffers[name];
  }

  KokkosDoubleBuffer<double>& get_double_buffer(const char* name) {
    return _double_buffers[name];
  }

  /* ----------------------------------------------------------------------
     Check if buffer exists
  ---------------------------------------------------------------------- */
  bool has_int_buffer(const char* name) const {
    return _int_buffers.find(name) != _int_buffers.end();
  }

  bool has_double_buffer(const char* name) const {
    return _double_buffers.find(name) != _double_buffers.end();
  }

private:
  std::map<std::string, KokkosDoubleBuffer<int>> _int_buffers;
  std::map<std::string, KokkosDoubleBuffer<double>> _double_buffers;
  std::vector<std::string> _int_buffer_order;
  std::vector<std::string> _double_buffer_order;
};

}  // namespace SPPARKS_NS

#endif

/* ERROR/WARNING messages:

E: KokkosDoubleBuffer buffer not found

The requested buffer name was not registered with add_int_buffer or
add_double_buffer before allocation.

*/
