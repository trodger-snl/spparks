# Rebase Assessment: SPPARKS 2024 → 2025

I've completed a thorough analysis comparing your current 2024 SPPARKS repository with the October 2025 release. Here's what I found:

---

## **Critical Stitch API Changes**

The October 2025 release contains **breaking changes** to the Stitch API that will impact SPPARKS source code and the microstructure analysis scripts (now in a separate repository at `/Users/Tron/spparks-micro-analysis`).

### 1. **`stitch_set_parameters` signature changed**

**2024 (current):**
```c
int stitch_set_parameters(const StitchFile * file,
                         double absolute_tolerance,
                         double relative_tolerance,
                         int64_t default_no_value_present);
```

**2025 (new):**
```c
int stitch_set_parameters(const StitchFile * file,
                         double absolute_tolerance,
                         double relative_tolerance);
```

**Impact:** The `default_no_value_present` parameter has been **removed**. Now each field must specify its own no-value-present (nvp) value when created.

### 2. **`stitch_get_parameters` signature changed**

**2024 (current):**
```c
int stitch_get_parameters(const StitchFile * file,
                         double * absolute_tolerance,
                         double * relative_tolerance,
                         int64_t * default_no_value_present,
                         double * first_time,
                         double * last_time);
```

**2025 (new):**
```c
int stitch_get_parameters(const StitchFile * file,
                         double * absolute_tolerance,
                         double * relative_tolerance,
                         double * first_time,
                         double * last_time);
```

**Impact:** No longer returns `default_no_value_present`.

### 3. **`stitch_query_field` signature expanded**

**2024 (current):**
```c
int stitch_query_field(const StitchFile * file,
                      const char * label,
                      int64_t * field_id);
```

**2025 (new):**
```c
int stitch_query_field(const StitchFile * file,
                      const char * label,
                      int64_t * field_id,
                      enum STITCH_TYPES * type,
                      int32_t * per_site_length,
                      union StitchTypesUnion * no_value_present_value);
```

**Impact:** Now returns field type, length, and nvp value - provides more metadata.

### 4. **`stitch_create_field` signature unchanged**

The signature is the same, but the semantic meaning has changed - the `no_value_present` parameter is now **required** (not optional with a default).

### 5. **Additional API Addition**

**2025 adds:**
```c
int stitch_query_field_by_id(const StitchFile * file,
                            int64_t field_id,
                            enum STITCH_TYPES * type,
                            int32_t * per_site_length,
                            union StitchTypesUnion * no_value_present_value);
```

This is a new function for querying field metadata when you have the ID but not the name.

### 6. **Deprecated Function Removed**

**2024 had (now removed):**
```c
int stitch_set_field_no_value_present(const StitchFile * file,
                                     int64_t field_id,
                                     union StitchTypesUnion no_value_present);
```

This function is **deprecated** in 2025 and marked for removal. The nvp value must now be set during `stitch_create_field()`.

---

## **Impact on Your Code**

### **SPPARKS Source Code** (src/)
- ✅ **Already updated** in the 2025 release
- Files affected: `src/set.cpp:719,783,791,799`, `src/input.cpp:815`
- These will automatically be fixed when you rebase

### **Microstructure Analysis Scripts** ⚠️ **WILL BREAK**

**Note:** As of November 2025, the `micro_analysis_scripts/` directory has been extracted to a separate repository at `/Users/Tron/spparks-micro-analysis` with full git history preserved. Updates will need to be made in that repository.

#### Files requiring updates (in spparks-micro-analysis repo):

1. **`stitch_reader.py`** (lines 738, 266, 633, 595)
   - `set_parameters()` method: Remove `no_value_present` parameter
   - `create_field()` calls: Ensure nvp is always provided
   - `query_field()` wrapper: May need to handle extra return values

2. **`stitch_simple_distribution/stitch_reader.py`** (duplicate, same issues)

3. **`ovito-stitch-readers/src/StitchReaders/stitch_base.py`**
   - Likely uses similar API patterns

4. **`Testing/test_stitch_writer.py`**
   - Test code that uses `set_parameters()` and `create_field()`

5. **`unified_analysis.py`**
   - May call Stitch API functions

6. **`check_stitch_bounds.py`** and **`bounds_usage_examples.py`**
   - Use `libstitch` but may not call affected functions

---

## **Rebase Difficulty Assessment**

### **Overall Difficulty: MODERATE** 🟡

**Why moderate and not difficult:**
- API changes are **well-documented** in release notes
- Changes are **breaking but straightforward** - mostly parameter removal
- SPPARKS core code is **already updated** in the 2025 release
- Your analysis scripts follow consistent patterns (easier to batch-fix)

**Why not easy:**
- Requires careful testing of all Python scripts
- The `set_parameters()` function is used in multiple places
- Need to verify no-value-present handling is correct for each field type

---

## **Migration Strategy**

### **Phase 1: SPPARKS Core (Already Done in 2025 Release)**
The SPPARKS source code changes are minimal and already handled in the 2025 release.

### **Phase 2: Python Scripts (Requires Your Attention)**

**Step 1: Update `set_parameters()` calls**
```python
# OLD (2024)
writer.set_parameters(absolute_tolerance=1.0e-9,
                     relative_tolerance=1.0e-15,
                     no_value_present=-1)

# NEW (2025)
writer.set_parameters(absolute_tolerance=1.0e-9,
                     relative_tolerance=1.0e-15)
```

**Step 2: Ensure `create_field()` always specifies nvp**
```python
# OLD (2024) - nvp could be defaulted
writer.create_field(field_name, field_type='float64')

# NEW (2025) - nvp must be explicit
writer.create_field(field_name, field_type='float64', no_value=-1.0)
```

**Step 3: Update `query_field()` wrapper**
```python
# OLD (2024)
rc, field_id = libstitch.query_field(self.fid, field_name)

# NEW (2025) - if using raw API
rc, field_id, field_type, field_len, nvp = libstitch.query_field(
    self.fid, field_name)
```

Note: Your current wrapper in `stitch_reader.py:266` may already handle this if the Python binding provides a backward-compatible interface.

---

## **Testing Requirements**

After rebase, you must test:

1. ✅ **SPPARKS compilation** - Should work immediately
2. ⚠️ **All stitch_reader.py functionality**
   - Reading existing .st files
   - Writing new .st files
   - Query operations
3. ⚠️ **Ovito readers** - Ensure they still work with updated API
4. ⚠️ **Analysis scripts** - Run through typical workflows

---

## **Recommended Approach**

1. **Create a test branch**
   ```bash
   git checkout -b test-2025-rebase
   ```

2. **Rebase onto 2025 release**
   ```bash
   # First, add the 2025 release as a reference
   git remote add spparks2025 /Users/Tron/Downloads/spparks-08Oct25
   git fetch spparks2025

   # Or if you want to use a local directory without adding a remote:
   # Copy the .git directory or use git format-patch/apply
   ```

3. **Fix analysis scripts** in separate repository at `/Users/Tron/spparks-micro-analysis` (see migration strategy above)

4. **Run comprehensive tests** with existing .st files

5. **Document any additional changes needed**

---

## **Additional Notes**

- The 2025 Stitch library has **better documentation** (new `docs/` directory, Sphinx API docs)
- Import syntax already changed in November 2024: `from stitch import libstitch` (your scripts already use this ✅)
- The structural reorganization (src/stitch/libstitch/ instead of libstitch/) shouldn't affect users
- Your `spack.yaml` and CMake builds should continue working

**Verdict:** The rebase is **feasible and recommended**. The API changes are breaking but well-documented, and most of the work will be updating your Python analysis scripts to remove the deprecated `no_value_present` parameter from `set_parameters()` calls.

---

## **Release Notes Summary**

From the 2025 Stitch release notes:

### October 2025
- API updated with breaking changes
- `set_parameters()` now only takes tolerances (nvp removed)
- `create_field()` must specify nvp value for each field
- Array-valued fields supported (experimental, not tested - for quaternions)
- Sphinx-based Python API documentation added
- Date-based version numbering (YYYYMMDD) added

### November 2024 (Already in your repo)
- Performance improvements (10-100x speedup for 'set stitch' command)
- Deprecated numpy.distutils, switched to setuptools
- Import syntax changed to `from stitch import libstitch`

---

## **File-by-File Change Summary**

### Core SPPARKS Files (already updated in 2025)

#### `src/set.cpp`
- Line 719: `stitch_get_parameters()` - removed `int64_t` nvp parameter
- Line 783: `stitch_query_field()` - now returns type, length, nvp
- Line 791: `stitch_create_field()` - always specify nvp
- Line 799: `stitch_create_field()` - always specify nvp

#### `src/input.cpp`
- Line 815: `stitch_get_parameters()` - removed `int64_t` nvp parameter

### Analysis Scripts (require manual updates in separate repository)

**Repository:** `/Users/Tron/spparks-micro-analysis`

#### `stitch_reader.py`
- Line 266: `query_field()` wrapper
- Line 595: `create_field()` method definition
- Line 633: `create_field()` call to libstitch
- Line 680: `create_field()` call
- Line 718: `create_field()` call
- Line 725: `set_parameters()` method definition
- Line 738: `set_parameters()` call to libstitch

#### `stitch_simple_distribution/stitch_reader.py`
- Same pattern as main stitch_reader.py

#### Additional files to check:
- `ovito-stitch-readers/src/StitchReaders/stitch_base.py`
- `Testing/test_stitch_writer.py`
- `unified_analysis.py`
