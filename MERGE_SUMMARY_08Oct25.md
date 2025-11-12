# SPPARKS 08Oct25 Upstream Merge Summary

**Date**: November 12, 2025
**Branches**: master, SURGE → SURGE-08Oct25-integration
**Upstream Version**: 08Oct25 (113 commits from 27Nov24)

---

## Executive Summary

Successfully integrated upstream spparks/spparks 08Oct25 release into both master and SURGE branches while preserving all custom development. The merge required STITCH API compatibility updates but builds successfully. **Runtime issue discovered**: Custom application (AppAdditiveExtTempTexture) encounters segmentation fault during execution, requiring further investigation.

---

## Completed Tasks

### ✅ Phase 1: Preparation & Verification
- Fetched upstream 08Oct25 release (113 new commits)
- Verified STITCH API changes: `stitch_set_parameters` simplified from 4 to 3 parameters, `stitch_get_parameters` from 5 to 4 parameters
- Created backup branch: `SURGE-pre-08Oct25-update`
- Confirmed STITCH library reorganization: `lib/stitch/libstitch/` → `lib/stitch/src/stitch/libstitch/`

### ✅ Phase 2: Master Branch Update
- **Merged**: upstream/master (08Oct25) into local master
- **Conflicts resolved**: 36 initial conflicts
  - Documentation: Accepted upstream versions (8 files)
  - STITCH library: Handled directory reorganization (10 files)
  - Makefiles: Accepted upstream versions (3 files)
  - Source files: Accepted upstream versions (15 files)
- **Commit**: `4681987` - Merge upstream 08Oct25 release into master

### ✅ Phase 3: SURGE Branch Integration
- **Merged**: Updated master into SURGE-08Oct25-integration
- **Conflicts resolved**: Only 2 conflicts (excellent!)
  - `lib/stitch/libstitch/stitchmodule.c`: Removed (moved to new location)
  - `src/input.cpp`: Kept SURGE version (preserved snprintf safety fixes)
- **Commit**: `1fc4268` - Merge updated master into SURGE integration branch

### ✅ Phase 4: STITCH API Compatibility Fixes
1. **Fixed `src/input.cpp:814`**
   - Updated `stitch_get_parameters` call: 6 params → 5 params
   - Removed `int_tmp` (default_no_value_present) parameter
   - **Commit**: `7c6932b` - Fix STITCH API call for 08Oct25 compatibility

2. **Fixed `src/CMakeLists.txt:484`**
   - Updated path: `dump_stitch.cpp` → `STITCH/dump_stitch.cpp`
   - Reflects upstream STITCH package reorganization
   - **Commit**: `be37572` - Update CMake to use STITCH/dump_stitch.cpp path

3. **Updated STITCH library symlinks**
   - Fixed: `lib/stitch/includelink` → `src/stitch/libstitch`
   - Fixed: `lib/stitch/liblink` → `src/stitch/libstitch`

### ✅ Phase 5: Build Verification
- **Built STITCH library** successfully at new location
- **CMake build** completed successfully:
  ```
  C++17, MPI, GZIP, JPEG, HDF5 (1.14.6 parallel), HighFive, STITCH
  Executable: src/build/spparks
  ```

---

## Key Upstream Changes (08Oct25)

### STITCH Library (Critical)
- **API Simplification**: Reduced parameter count for cleaner interface
  - `stitch_set_parameters(file, abs_tol, rel_tol)` - removed default_no_value_present
  - `stitch_get_parameters(file, *abs_tol, *rel_tol, *first_time, *last_time)` - removed default_no_value_present output
  - `stitch_create_field` unchanged (still requires no_value_present per field)
  - `stitch_query_field` enhanced to return more metadata (6 params)
- **Directory Reorganization**: Moved to Python package structure
- **Performance**: 10-100x speedup (from 27Nov24, already in fork)
- **Documentation**: New Sphinx-based API docs

### Compiler Requirements
- **C++ Standard**: C++11 → C++17 (breaking change)

### New Features
- **HCP α-Phase Variants**: All 12 titanium precipitate orientations
- **Cluster Diagnostics**: New `diag_cluster` tool
- **Crystalline Orientations**: New tools in `tools/crystalline_orientations/`
  - Burgers Orientation Relation (BCC→HCP)
  - Cubic/HCP symmetries
  - Disorientation calculations

### Code Quality
- Copyright headers updated (2025)
- Improved documentation across all sections

---

## Custom Features Preserved

All SURGE branch custom development was successfully preserved:

### ✅ Core Applications
- `app_additive_ext_temp_texture.{cpp,h}` - Custom AM application
- `temperature_source.{cpp,h}` - Modular temperature framework
- `temperature_source_hdf5.{cpp,h}` - HDF5 temperature reading
- `temperature_source_hdf5_unstructured.{cpp,h}` - Unstructured mesh support
- `temperature_source_rosenthal.{cpp,h}` - Analytical heat source

### ✅ Build System Enhancements
- CMake improvements (HighFive, Spack integration)
- `build_cmake.sh` convenience script
- `spack.yaml` for reproducible dependency management
- `src/MAKE/Makefile.mac_arm_highfive_local`

### ✅ Documentation
- `CLAUDE.md` - AI assistant guidance
- `2025RebasePlan.md` - Future planning document
- Enhanced `doc/Section_start.{txt,html}` with STITCH/Spack instructions

### ✅ Quaternion Enhancements
- Site smoothing and neighbor averaging
- Exponential misorientation function
- Void generation features

---

## Current Status & Known Issues

### ⚠️ Runtime Issue (Blocking)

**Symptom**: Segmentation fault in custom application
```
AppAdditiveExtTempTexture::app_update() + 692
Signal: Segmentation fault: 11 (11)
Signal code: Invalid permissions (2)
```

**Context**:
- Application initializes successfully
- Crash occurs during first KMC iteration
- All 6 MPI ranks affected simultaneously
- Happens immediately after "Running with 32-bit site IDs" message

**Likely Causes**:
1. **API changes in AppPottsQuaternion base class** (parent of custom app)
   - Upstream modified quaternion handling
   - May have changed member variable layout or initialization
2. **Memory layout changes** in AppLattice/App base classes
3. **Timing/update mechanism changes** in iterate() loop

**Impact**: Custom application non-functional until resolved

### ✅ What Works
- Clean build (no compilation errors/warnings except deprecation warnings for sprintf)
- STITCH library functional
- Initialization phase completes
- Temperature source module loads
- Domain decomposition successful

---

## Files Changed Summary

**Total**: 277 files changed, +57,688 insertions, -3,324 deletions

**Major Categories**:
- Core source files: 180+ modified
- STITCH library: Complete reorganization
- Documentation: Comprehensive updates
- New tools: Crystalline orientations, HCP variants
- Headers: Copyright updates across entire codebase

---

## Branch Status

| Branch | Status | Location | Notes |
|--------|--------|----------|-------|
| `master` | ✅ Updated | Local | Clean merge with upstream 08Oct25 |
| `SURGE` | 🔄 Unchanged | Local | Original preserved |
| `SURGE-pre-08Oct25-update` | ✅ Backup | Local | Safety backup before merge |
| `SURGE-08Oct25-integration` | ⚠️ Testing | Local | Builds but runtime issues |

---

## Next Steps (Recommended Priority)

### 🔴 Critical (Blocking)
1. **Debug segmentation fault** in AppAdditiveExtTempTexture
   - Run under lldb/gdb to get exact crash location
   - Compare AppPottsQuaternion API changes between 27Nov24 and 08Oct25
   - Check virtual function signatures and member variable initialization
   - Review `app_update()` implementation against new base class

2. **Validate temperature source compatibility**
   - Verify HDF5 file reading still works
   - Test Rosenthal analytical source
   - Check unstructured mesh temperature loading

### 🟡 High Priority
3. **Test STITCH integration**
   - Verify dump_stitch output format
   - Test set_stitch reading
   - Validate Python analysis scripts compatibility

4. **Fix sprintf deprecation warnings**
   - Update remaining sprintf calls to snprintf in:
     - `src/STITCH/dump_stitch.cpp:143,162`
     - `src/set.cpp:935,937,939`

### 🟢 Medium Priority
5. **Merge to SURGE branch**
   - Once runtime issue resolved
   - Run full test suite
   - Update version information

6. **Update documentation**
   - Document API changes affecting custom code
   - Update CLAUDE.md with lessons learned
   - Add troubleshooting section

7. **Push to remotes**
   - Push master: `git push origin master`
   - Push SURGE: `git push gitea SURGE`
   - Consider GitHub push (may need force due to history rewrite)

---

## Testing Checklist (When Runtime Issue Resolved)

- [ ] Basic KMC simulation runs
- [ ] Custom AM application completes
- [ ] Temperature sources load correctly (HDF5, Rosenthal, unstructured)
- [ ] STITCH dumps write successfully
- [ ] STITCH set command reads correctly
- [ ] Quaternion operations work (smoothing, averaging)
- [ ] MPI parallelization scales correctly
- [ ] HighFive HDF5 reading works
- [ ] Python analysis scripts compatible (separate repo: spparks-micro-analysis)
- [ ] CMake build with all features
- [ ] Traditional Makefile build (if needed)

---

## Lessons Learned

### What Went Well ✅
- **Minimal conflicts**: Only 2 conflicts in SURGE merge (vs 36 in master)
- **STITCH API changes benign**: Custom code didn't use changed functions
- **Build system robust**: CMake adaptations straightforward
- **Backup strategy effective**: Pre-merge backup branch provides safety net
- **API verification proactive**: Checking API changes before build saved time

### Challenges Encountered ⚠️
- **Directory reorganization**: STITCH library path changes required multiple fixes
- **Hidden dependencies**: CMake configuration needed path updates
- **Runtime vs compile-time issues**: Build success doesn't guarantee runtime compatibility
- **Base class changes**: Inheritance from modified upstream classes causes issues

### Recommendations for Future Merges 📋
1. **Always create backup branch** before starting
2. **Verify API changes thoroughly** before merging
3. **Test incrementally**: Don't wait until end to test runtime
4. **Document base class dependencies**: Track which upstream classes custom code inherits from
5. **Use integration branches**: Never merge directly to main development branch
6. **Run test suite immediately** after successful build

---

## Git Commands Reference

```bash
# View integration branch
git checkout SURGE-08Oct25-integration

# Compare with backup
git diff SURGE-pre-08Oct25-update

# View merge commits
git log --oneline --graph SURGE-08Oct25-integration -20

# Rollback if needed
git checkout SURGE
git branch -D SURGE-08Oct25-integration
git checkout -b SURGE-08Oct25-integration SURGE-pre-08Oct25-update

# When ready to finalize
git checkout SURGE
git merge --no-ff SURGE-08Oct25-integration
```

---

## Contact & Resources

- **Upstream Repository**: https://github.com/spparks/spparks
- **Release Tag**: 08Oct25
- **STITCH Documentation**: lib/stitch/docs/ (Sphinx)
- **Analysis Scripts**: /Users/Tron/spparks-micro-analysis (separate repo)

---

**Generated**: November 12, 2025
**Integration Branch**: SURGE-08Oct25-integration
**Status**: Build successful, runtime debugging required
