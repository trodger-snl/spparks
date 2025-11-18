# Kokkos AM App Timing Issue

**Status**: Open
**Priority**: High
**Phase**: Phase 2 (Temperature/Gradients)
**Date Identified**: November 2025

## Summary

The Kokkos version of the AM application (`app_additive_ext_temp_texture_kokkos`) has a timing issue where `app_update()` is not called at regular intervals during simulation runs. This prevents proper validation of gradient calculations against the CPU version.

## Observed Behavior

### CPU Version (Expected)
- Input: `run 0.005 upto`, `stats 2e-4`, `dump ... 0.001`
- Output: ~50 stats calls, 5 dumps at t = 0.001, 0.002, 0.003, 0.004, 0.005s
- Time steps regularly from t=0.0001s to t=0.005s

### Kokkos Version (Actual)
- Input: Same as CPU
- Output: Only 2 stats calls at t=0.0001s and t=0.0099s
- Only 1 dump at t=0.0099s
- Appears to skip intermediate timesteps

## Investigation Findings

### What It's NOT
- ❌ **Not fast-forward logic**: Disabling fast-forward in CPU version didn't help
- ❌ **Not `app_update_only` flag**: Both versions use the same flag
- ❌ **Not HDF5 temperature loading**: Temperature loads correctly in both versions

### Likely Cause
The issue appears to be in how the Kokkos version integrates with the **solver's time stepping loop**. The solver isn't calling `app_update()` at the expected intervals.

**Key differences:**
1. CPU version inherits from `AppPottsQuaternion` (non-Kokkos)
2. Kokkos version inherits from `AppPottsQuaternionKokkos`
3. Parent class time management may differ between versions

## Impact

**Blocking:**
- Gradient validation against CPU version
- Can't verify numerical accuracy of thermal gradient calculations

**Workarounds:**
- Manual inspection of gradient computation code shows correct algorithm
- Code review confirms proper implementation of weighted least-squares gradient
- Single-timestep validation shows gradient values in expected range

## Files Affected

- `src/KOKKOS/app_additive_ext_temp_texture_kokkos.cpp` - Kokkos app implementation
- `src/KOKKOS/app_potts_quaternion_kokkos.cpp` - Parent class (may need investigation)
- Test files: `examples/NewStitchFormat/in.gradient_test_*`

## Reproduction

```bash
cd examples/NewStitchFormat

# CPU version (works correctly)
mpirun -np 6 ../../src/build/spparks -in in.gradient_test_cpu

# Kokkos version (shows timing issue)
mpirun -np 6 ../../src/build/spparks -in in.gradient_test_kokkos

# Check output
python3 << 'EOF'
from stitch import libstitch
for file in ["DumpFiles_gradient_test/am_data_cpu.st",
             "DumpFiles_gradient_test_kk/am_data_kokkos.st"]:
    status, fh = libstitch.open(file)
    status, times = libstitch.get_times(fh)
    print(f"{file}: {len(times)} timesteps - {times}")
    libstitch.close(fh)
EOF
```

Expected: Both files have 5 timesteps
Actual: CPU has 5, Kokkos has 1

## Next Steps (Phase 3)

1. **Investigate solver interaction**
   - Examine how `AppPottsQuaternionKokkos` manages time stepping
   - Compare with `AppPottsQuaternion` time management
   - Check if parent class `iterate_rejection()` needs to be called differently

2. **Add time management**
   - May need to add explicit time increment logic to Kokkos version
   - Consider adding fast-forward logic from CPU version
   - Ensure `time` variable updates correctly

3. **Validate fix**
   - Run both versions with matching parameters
   - Verify identical timesteps in output
   - Complete gradient validation testing

## Related Files

- `docs/AM_KOKKOS_PLAN.md` - Overall Kokkos implementation plan
- `docs/KOKKOS_IMPLEMENTATION_ROADMAP.md` - Roadmap document
- `examples/NewStitchFormat/compare_gradients.py` - Comparison script (working)

## Notes

- Gradient computation kernel itself appears correct
- Temperature loading from HDF5 works correctly
- Device/host synchronization works as expected
- Issue is specifically in time stepping, not physics calculations
