# Adding HighFive to SPPARKS Makefile Build

This guide shows how to add HighFive C++ HDF5 wrapper support to SPPARKS using the traditional Makefile system.

## Quick Setup

### Option 1: Local Installation (Recommended)

1. **Install HighFive locally:**
   ```bash
   cd /Users/Tron/spparks
   ./install_highfive_local.sh
   ```

2. **Build with HighFive support:**
   ```bash
   cd src
   make mac_arm_highfive_local
   ```

This approach:
- Installs HighFive to `external/highfive/` within your project
- Doesn't interfere with Homebrew or system directories
- Works with your existing hdf5-mpi installation

### Option 2: Modify Existing Makefile.mac_arm

1. **Install HighFive** (same as above)

2. **Edit your existing Makefile:**
   ```bash
   cd src/MAKE
   cp Makefile.mac_arm Makefile.mac_arm_backup  # Backup original
   ```

3. **Add HighFive support to Makefile.mac_arm:**

   **Change this line:**
   ```makefile
   SPK_INC = -DSPPARKS_GZIP -DSPPARKS_UNORDERED_MAP -DSPPARKS_JPEG -DSPPARKS_HDF
   ```

   **To this:**
   ```makefile
   SPK_INC = -DSPPARKS_GZIP -DSPPARKS_UNORDERED_MAP -DSPPARKS_JPEG -DSPPARKS_HDF -DSPPARKS_HIGHFIVE
   ```

   **Add HighFive include path after the HDF5 section:**
   ```makefile
   # HighFive C++ wrapper for HDF5
   HIGHFIVE_INC = -I/opt/homebrew/include
   HIGHFIVE_PATH = 
   HIGHFIVE_LIB = 
   ```

   **Update the EXTRA_INC line:**
   ```makefile
   EXTRA_INC = $(SPK_INC) $(PKG_INC) $(MPI_INC) $(JPG_INC) $(PKG_SYSINC) $(HDF_INC) $(HIGHFIVE_INC)
   ```

4. **Build SPPARKS:**
   ```bash
   make mac_arm
   ```

## Manual HighFive Installation (Alternative)

If you prefer to install HighFive manually:

```bash
# Install dependencies
brew install hdf5 cmake

# Clone and build HighFive
git clone --depth 1 --branch v2.9.0 https://github.com/BlueBrain/HighFive.git
cd HighFive
mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/homebrew \
    -DHIGHFIVE_EXAMPLES=OFF \
    -DHIGHFIVE_UNIT_TESTS=OFF

make -j$(sysctl -n hw.ncpu)
sudo make install
```

## Verification

After installation, verify HighFive is available:

```bash
# Check headers
ls /opt/homebrew/include/highfive/

# Test compilation
echo '#include <highfive/highfive.hpp>' | c++ -x c++ -I/opt/homebrew/include -c - -o /dev/null
```

## Using HighFive in Code

In your C++ files, you can now use:

```cpp
#ifdef SPPARKS_HIGHFIVE
#include <highfive/highfive.hpp>

void example_usage() {
    HighFive::File file("data.h5", HighFive::File::ReadOnly);
    auto dataset = file.getDataSet("temperature");
    
    std::vector<double> data;
    dataset.read(data);
}
#endif
```

## Benefits

- **Simplified chunk handling**: Union of hyperslabs instead of individual reads
- **Type safety**: Automatic type conversion and validation  
- **Modern C++**: RAII, exceptions, templates
- **Better performance**: Fewer HDF5 API calls
- **Easier debugging**: Clear error messages

## Troubleshooting

**If compilation fails with HighFive not found:**
- Verify HighFive is installed: `ls /opt/homebrew/include/highfive/`
- Check your include path matches the installation location
- For Intel Macs, use `/usr/local/include` instead of `/opt/homebrew/include`

**If you get HDF5 linking errors:**
- Make sure HDF5 is installed: `brew install hdf5`
- Verify HDF5 libraries: `ls /opt/homebrew/lib/libhdf5*`