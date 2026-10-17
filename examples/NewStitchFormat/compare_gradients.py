#!/usr/bin/env python
"""
Compare gradient calculations between CPU and Kokkos versions of SPPARKS AM app.

Loads stitch files and compares d8 (gradient magnitude) values at matching timesteps.
"""

import numpy as np
from stitch import libstitch
import sys

def load_stitch_data(filename):
    """Load stitch file and return data arrays."""
    print(f"\nLoading {filename}...")

    # Open file - returns (status, file_handle)
    status, fh = libstitch.open(filename)
    if status != 0:
        raise RuntimeError(f"Failed to open {filename}, status={status}")

    # Get timesteps - returns (status, times_array)
    status, times = libstitch.get_times(fh)
    if status != 0:
        raise RuntimeError(f"Failed to get times, status={status}")

    # Get fields - returns 6-tuple: (status, field_ids, field_names, field_sizes, field_offsets, field_defaults)
    result = libstitch.get_fields(fh)
    status, field_ids, field_names, field_sizes, field_offsets, field_defaults = result
    if status != 0:
        raise RuntimeError(f"Failed to get fields, status={status}")

    print(f"  Number of timesteps: {len(times)}")
    if len(times) > 0:
        print(f"  Time range: {times[0]:.6f} to {times[-1]:.6f} s")
    print(f"  Available fields: {field_names}")

    return fh, times, field_names

def compare_gradients(cpu_file, kokkos_file):
    """Compare gradient magnitudes between CPU and Kokkos versions."""

    # Load data
    cpu_fh, cpu_times, cpu_fields = load_stitch_data(cpu_file)
    kokkos_fh, kokkos_times, kokkos_fields = load_stitch_data(kokkos_file)

    print("\n" + "="*80)
    print("GRADIENT COMPARISON")
    print("="*80)

    # Check if we have matching timesteps
    tolerance = 1e-6
    matching_pairs = []

    for i, kt in enumerate(kokkos_times):
        matches = np.abs(np.array(cpu_times) - kt) < tolerance
        if np.any(matches):
            cpu_idx = np.where(matches)[0][0]
            matching_pairs.append((i, cpu_idx, kt))

    if not matching_pairs:
        print("\nWARNING: No matching timesteps found between CPU and Kokkos runs!")
        print("This is likely due to a timing issue in one of the runs.")
        print("\nComparing last available timestep from each run...")
        print("(This will validate the gradient calculation algorithm even if times differ)\n")

        if len(kokkos_times) == 0 or len(cpu_times) == 0:
            print("ERROR: One or both files have no timesteps!")
            return

        # Use last timestep from each, which should have gradient data
        matching_pairs = [(len(kokkos_times)-1, len(cpu_times)-1, kokkos_times[-1])]

    # Process each timestep pair
    for i, cpu_idx, kt in matching_pairs:
        cpu_time = cpu_times[cpu_idx]
        if abs(kt - cpu_time) > tolerance:
            print(f"\n--- Kokkos t = {kt:.6f} s vs CPU t = {cpu_time:.6f} s (DIFFERENT TIMES) ---")
        else:
            print(f"\n--- Timestep {i}: t = {kt:.6f} s ---")

        # Get d8 field ID from kokkos fields
        d8_idx_kk = kokkos_fields.index('d8')
        # Get fields metadata to find field ID
        result = libstitch.get_fields(kokkos_fh)
        _, kk_field_ids, _, _, _, _ = result
        d8_id_kk = kk_field_ids[d8_idx_kk]

        # Get d8 field ID from CPU fields
        d8_idx_cpu = cpu_fields.index('d8')
        result = libstitch.get_fields(cpu_fh)
        _, cpu_field_ids, _, _, _, _ = result
        d8_id_cpu = cpu_field_ids[d8_idx_cpu]

        # Get global bounds for reading entire domain
        status, kk_bounds = libstitch.get_global_bounds(kokkos_fh, d8_id_kk)
        if status != 0:
            print(f"Warning: Failed to get Kokkos bounds, status={status}")
            continue

        status, cpu_bounds = libstitch.get_global_bounds(cpu_fh, d8_id_cpu)
        if status != 0:
            print(f"Warning: Failed to get CPU bounds, status={status}")
            continue

        # Get gradient data (d8)
        # read_block(fh, field_id, time_value, bounds) -> (rc, data, read_flag)
        status, kokkos_grad, _ = libstitch.read_block(kokkos_fh, d8_id_kk, kt, kk_bounds)
        if status != 0:
            print(f"Warning: Failed to read Kokkos d8 at time {kt}, status={status}")
            continue

        cpu_time = cpu_times[cpu_idx]
        status, cpu_grad, _ = libstitch.read_block(cpu_fh, d8_id_cpu, cpu_time, cpu_bounds)
        if status != 0:
            print(f"Warning: Failed to read CPU d8 at time {cpu_time}, status={status}")
            continue

        # Basic statistics
        print(f"\nKokkos gradients:")
        print(f"  Min:  {np.min(kokkos_grad):.6e}")
        print(f"  Max:  {np.max(kokkos_grad):.6e}")
        print(f"  Mean: {np.mean(kokkos_grad):.6e}")
        print(f"  Sum:  {np.sum(kokkos_grad):.6e}")

        print(f"\nCPU gradients:")
        print(f"  Min:  {np.min(cpu_grad):.6e}")
        print(f"  Max:  {np.max(cpu_grad):.6e}")
        print(f"  Mean: {np.mean(cpu_grad):.6e}")
        print(f"  Sum:  {np.sum(cpu_grad):.6e}")

        # Compute differences
        abs_diff = np.abs(kokkos_grad - cpu_grad)
        rel_diff = np.where(cpu_grad != 0, abs_diff / np.abs(cpu_grad), 0)

        print(f"\nAbsolute differences:")
        print(f"  Max:  {np.max(abs_diff):.6e}")
        print(f"  Mean: {np.mean(abs_diff):.6e}")
        print(f"  RMS:  {np.sqrt(np.mean(abs_diff**2)):.6e}")

        print(f"\nRelative differences (where CPU != 0):")
        nonzero_mask = cpu_grad != 0
        if np.any(nonzero_mask):
            print(f"  Max:  {np.max(rel_diff[nonzero_mask])*100:.6f}%")
            print(f"  Mean: {np.mean(rel_diff[nonzero_mask])*100:.6f}%")

        # Count exact matches
        exact_matches = np.sum(kokkos_grad == cpu_grad)
        print(f"\nExact matches: {exact_matches}/{len(kokkos_grad)} ({exact_matches/len(kokkos_grad)*100:.2f}%)")

        # Histogram of differences
        print(f"\nDifference histogram:")
        bins = [1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-0]
        for i in range(len(bins)-1):
            count = np.sum((abs_diff >= bins[i]) & (abs_diff < bins[i+1]))
            print(f"  [{bins[i]:.0e}, {bins[i+1]:.0e}): {count} sites")
        count = np.sum(abs_diff >= bins[-1])
        print(f"  >= {bins[-1]:.0e}: {count} sites")

    # Close files
    libstitch.close(cpu_fh)
    libstitch.close(kokkos_fh)

if __name__ == "__main__":
    cpu_file = "DumpFiles_gradient_test/am_data_cpu.st"
    kokkos_file = "DumpFiles_gradient_test_kk/am_data_kokkos.st"

    if len(sys.argv) > 1:
        cpu_file = sys.argv[1]
    if len(sys.argv) > 2:
        kokkos_file = sys.argv[2]

    compare_gradients(cpu_file, kokkos_file)
