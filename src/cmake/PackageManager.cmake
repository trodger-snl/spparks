# Package management system for SPPARKS CMake build
# Equivalent to the Makefile package system (yes-package/no-package)
#
# IMPORTANT: Package files live in two locations:
#   - src/PACKAGE/  (reference version, tracked in git)
#   - src/          (installed/compiled version, ignored by git)
#
# GOLDEN RULE: Always edit src/PACKAGE/ files, never src/ installed copies!
# Changes to installed files will be LOST on package reinstall.

# Development mode option
option(SPPARKS_DEV_MODE "Enable development mode with symlinks instead of copies (Unix only)" OFF)

if(SPPARKS_DEV_MODE AND WIN32)
    message(WARNING "SPPARKS_DEV_MODE is not supported on Windows, falling back to file copies")
    set(SPPARKS_DEV_MODE OFF CACHE BOOL "Development mode disabled on Windows" FORCE)
endif()

# Available packages
set(SPPARKS_AVAILABLE_PACKAGES
    "stitch"
    CACHE STRING "List of available SPPARKS packages"
)

# Package status tracking
set(SPPARKS_PACKAGE_STATUS_FILE "${CMAKE_CURRENT_BINARY_DIR}/package_status.cmake")

# Function to load package status
function(load_package_status)
    if(EXISTS "${SPPARKS_PACKAGE_STATUS_FILE}")
        include("${SPPARKS_PACKAGE_STATUS_FILE}")
    endif()
endfunction()

# Function to save package status
function(save_package_status)
    file(WRITE "${SPPARKS_PACKAGE_STATUS_FILE}"
         "# SPPARKS Package Status - Auto-generated\n")
    foreach(PKG ${SPPARKS_AVAILABLE_PACKAGES})
        string(TOUPPER "${PKG}" PKG_UPPER)
        if(SPPARKS_PACKAGE_${PKG_UPPER})
            file(APPEND "${SPPARKS_PACKAGE_STATUS_FILE}"
                 "set(SPPARKS_PACKAGE_${PKG_UPPER} ON)\n")
        endif()
    endforeach()
endfunction()

# Function to check if package files are in sync
# Compares files in src/PACKAGE/ with their installed copies in src/
function(check_package_sync PACKAGE_NAME)
    string(TOUPPER "${PACKAGE_NAME}" PKG_UPPER)
    set(PKG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${PKG_UPPER}")

    if(NOT EXISTS "${PKG_DIR}")
        return()
    endif()

    # Find all .cpp and .h files in package directory
    file(GLOB PKG_CPP_FILES "${PKG_DIR}/*.cpp")
    file(GLOB PKG_H_FILES "${PKG_DIR}/*.h")
    set(PKG_FILES ${PKG_CPP_FILES} ${PKG_H_FILES})

    set(FILES_DIFFER FALSE)
    set(DIFFERING_FILES "")

    foreach(PKG_FILE ${PKG_FILES})
        get_filename_component(FILENAME "${PKG_FILE}" NAME)
        set(SRC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")

        if(EXISTS "${SRC_FILE}")
            # Compare files
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E compare_files "${PKG_FILE}" "${SRC_FILE}"
                RESULT_VARIABLE FILES_ARE_DIFFERENT
                OUTPUT_QUIET
                ERROR_QUIET
            )

            if(FILES_ARE_DIFFERENT)
                set(FILES_DIFFER TRUE)
                list(APPEND DIFFERING_FILES "${FILENAME}")
            endif()
        endif()
    endforeach()

    if(FILES_DIFFER)
        message(WARNING "")
        message(WARNING "╔═══════════════════════════════════════════════════════════════")
        message(WARNING "║ PACKAGE SYNC WARNING: ${PACKAGE_NAME} package files differ!")
        message(WARNING "╠═══════════════════════════════════════════════════════════════")
        message(WARNING "║ The following installed files differ from package source:")
        foreach(FILE ${DIFFERING_FILES})
            message(WARNING "║   • ${FILE}")
        endforeach()
        message(WARNING "║")
        message(WARNING "║ GOLDEN RULE: Always edit src/${PKG_UPPER}/ files, not src/ copies!")
        message(WARNING "║")
        message(WARNING "║ If you edited src/ files by mistake:")
        message(WARNING "║   1. Copy your changes to src/${PKG_UPPER}/")
        message(WARNING "║   2. Reconfigure to reinstall: cmake ..")
        message(WARNING "║")
        message(WARNING "║ If src/${PKG_UPPER}/ has the correct version:")
        message(WARNING "║   • Reconfigure to reinstall: cmake ..")
        message(WARNING "╚═══════════════════════════════════════════════════════════════")
        message(WARNING "")
    endif()
endfunction()

# Function to install a package
function(install_package PACKAGE_NAME)
    string(TOUPPER "${PACKAGE_NAME}" PKG_UPPER)

    if(NOT "${PACKAGE_NAME}" IN_LIST SPPARKS_AVAILABLE_PACKAGES)
        message(FATAL_ERROR "Package '${PACKAGE_NAME}' does not exist")
    endif()

    # Package-specific installation logic (always run to ensure libraries are built)
    if(PACKAGE_NAME STREQUAL "stitch")
        install_stitch_package()
        # Variables are set as CACHE variables in install_stitch_package(), so no need to propagate
    endif()

    if(SPPARKS_PACKAGE_${PKG_UPPER})
        message(STATUS "Package '${PACKAGE_NAME}' is already installed")
        return()
    endif()

    message(STATUS "Installing package '${PACKAGE_NAME}'...")

    # Mark package as installed
    set(SPPARKS_PACKAGE_${PKG_UPPER} ON PARENT_SCOPE)
    set(SPPARKS_PACKAGE_${PKG_UPPER} ON CACHE BOOL "Package ${PACKAGE_NAME} is installed")
    save_package_status()

    # Check sync status after installation
    check_package_sync("${PACKAGE_NAME}")

    message(STATUS "Package '${PACKAGE_NAME}' installed successfully")
endfunction()

# Function to uninstall a package
function(uninstall_package PACKAGE_NAME)
    string(TOUPPER "${PACKAGE_NAME}" PKG_UPPER)
    
    if(NOT "${PACKAGE_NAME}" IN_LIST SPPARKS_AVAILABLE_PACKAGES)
        message(FATAL_ERROR "Package '${PACKAGE_NAME}' does not exist")
    endif()
    
    if(NOT SPPARKS_PACKAGE_${PKG_UPPER})
        message(STATUS "Package '${PACKAGE_NAME}' is not installed")
        return()
    endif()
    
    message(STATUS "Uninstalling package '${PACKAGE_NAME}'...")
    
    # Package-specific uninstallation logic
    if(PACKAGE_NAME STREQUAL "stitch")
        uninstall_stitch_package()
    endif()
    
    # Mark package as uninstalled
    set(SPPARKS_PACKAGE_${PKG_UPPER} OFF PARENT_SCOPE)
    set(SPPARKS_PACKAGE_${PKG_UPPER} OFF CACHE BOOL "Package ${PACKAGE_NAME} is installed")
    save_package_status()
    
    message(STATUS "Package '${PACKAGE_NAME}' uninstalled successfully")
endfunction()

# STITCH package installation
function(install_stitch_package)
    set(STITCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/STITCH")
    set(STITCH_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib/stitch")

    # Check for new path structure first (src/stitch/libstitch), fall back to old (libstitch)
    if(EXISTS "${STITCH_LIB_DIR}/src/stitch/libstitch")
        set(STITCH_BUILD_DIR "${STITCH_LIB_DIR}/src/stitch/libstitch")
    elseif(EXISTS "${STITCH_LIB_DIR}/libstitch")
        set(STITCH_BUILD_DIR "${STITCH_LIB_DIR}/libstitch")
    else()
        message(FATAL_ERROR "STITCH library directory not found. Tried:\n"
                "  ${STITCH_LIB_DIR}/src/stitch/libstitch\n"
                "  ${STITCH_LIB_DIR}/libstitch")
    endif()

    if(NOT EXISTS "${STITCH_DIR}")
        message(FATAL_ERROR "STITCH package directory not found")
    endif()

    # Step 1: Build the stitch library if needed
    message(STATUS "Checking for STITCH library...")
    message(STATUS "STITCH build directory: ${STITCH_BUILD_DIR}")

    # Check if library exists
    set(STITCH_LIB_PATH "${STITCH_BUILD_DIR}/libstitch.a")

    if(NOT EXISTS "${STITCH_LIB_PATH}")
        message(STATUS "STITCH library not found, building it now...")
        message(STATUS "Building STITCH library in ${STITCH_BUILD_DIR}")

        # Build the library using the existing Makefile
        # Use MPI compiler if available, otherwise use system compiler
        if(SPPARKS_ENABLE_MPI AND MPI_C_COMPILER)
            set(STITCH_CC "${MPI_C_COMPILER}")
            set(STITCH_CXX "${MPI_CXX_COMPILER}")
        else()
            set(STITCH_CC "${CMAKE_C_COMPILER}")
            set(STITCH_CXX "${CMAKE_CXX_COMPILER}")
        endif()

        # Use CMAKE_MAKE_PROGRAM for portability (instead of hardcoded make)
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E env CC=${STITCH_CC} CXX=${STITCH_CXX} ${CMAKE_MAKE_PROGRAM} stitch.lib
            WORKING_DIRECTORY "${STITCH_BUILD_DIR}"
            RESULT_VARIABLE BUILD_RESULT
            OUTPUT_VARIABLE BUILD_OUTPUT
            ERROR_VARIABLE BUILD_ERROR
        )

        if(NOT BUILD_RESULT EQUAL 0)
            message(FATAL_ERROR
                "Failed to build STITCH library.\n"
                "Working directory: ${STITCH_BUILD_DIR}\n"
                "Error output:\n${BUILD_ERROR}\n${BUILD_OUTPUT}\n"
                "Try building manually:\n"
                "  cd ${STITCH_BUILD_DIR}\n"
                "  make stitch.lib"
            )
        endif()

        # Verify library was created
        if(NOT EXISTS "${STITCH_LIB_PATH}")
            message(FATAL_ERROR
                "STITCH library build appeared to succeed but ${STITCH_LIB_PATH} not found"
            )
        endif()

        message(STATUS "STITCH library built successfully")
    else()
        message(STATUS "STITCH library found at ${STITCH_LIB_PATH}")
    endif()

    # Step 2: Create symlinks if they don't exist
    set(LIBLINK_PATH "${STITCH_LIB_DIR}/liblink")
    set(INCLUDELINK_PATH "${STITCH_LIB_DIR}/includelink")

    # Determine symlink target based on which directory structure exists
    if(EXISTS "${STITCH_LIB_DIR}/src/stitch/libstitch")
        set(SYMLINK_TARGET "src/stitch/libstitch")
    else()
        set(SYMLINK_TARGET "libstitch")
    endif()

    if(NOT EXISTS "${LIBLINK_PATH}")
        message(STATUS "Creating liblink symlink to ${SYMLINK_TARGET}...")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E create_symlink ${SYMLINK_TARGET} liblink
            WORKING_DIRECTORY "${STITCH_LIB_DIR}"
            RESULT_VARIABLE SYMLINK_RESULT
        )
        if(NOT SYMLINK_RESULT EQUAL 0)
            message(WARNING "Failed to create liblink symlink, trying alternative method...")
            execute_process(
                COMMAND ln -s ${SYMLINK_TARGET} liblink
                WORKING_DIRECTORY "${STITCH_LIB_DIR}"
                RESULT_VARIABLE SYMLINK_RESULT2
            )
            if(NOT SYMLINK_RESULT2 EQUAL 0)
                message(FATAL_ERROR "Failed to create liblink symlink")
            endif()
        endif()
    endif()

    if(NOT EXISTS "${INCLUDELINK_PATH}")
        message(STATUS "Creating includelink symlink to ${SYMLINK_TARGET}...")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E create_symlink ${SYMLINK_TARGET} includelink
            WORKING_DIRECTORY "${STITCH_LIB_DIR}"
            RESULT_VARIABLE SYMLINK_RESULT
        )
        if(NOT SYMLINK_RESULT EQUAL 0)
            message(WARNING "Failed to create includelink symlink, trying alternative method...")
            execute_process(
                COMMAND ln -s ${SYMLINK_TARGET} includelink
                WORKING_DIRECTORY "${STITCH_LIB_DIR}"
                RESULT_VARIABLE SYMLINK_RESULT2
            )
            if(NOT SYMLINK_RESULT2 EQUAL 0)
                message(FATAL_ERROR "Failed to create includelink symlink")
            endif()
        endif()
    endif()

    # Step 3: Verify we can find the library
    find_library(STITCH_LIBRARY
        NAMES stitch
        PATHS "${STITCH_LIB_DIR}/liblink"
        NO_DEFAULT_PATH
    )

    if(NOT STITCH_LIBRARY)
        message(FATAL_ERROR
            "STITCH library built but could not be found.\n"
            "Expected at: ${STITCH_LIB_DIR}/liblink/libstitch.a\n"
            "Please check the build output above for errors."
        )
    endif()

    message(STATUS "STITCH library found: ${STITCH_LIBRARY}")

    # Step 4: Set include and library paths as cache variables (global scope)
    set(STITCH_INCLUDE_DIR "${STITCH_LIB_DIR}/includelink" CACHE PATH "STITCH include directory" FORCE)
    set(STITCH_LIBRARY_DIR "${STITCH_LIB_DIR}/liblink" CACHE PATH "STITCH library directory" FORCE)
    set(STITCH_LIBRARY "${STITCH_LIBRARY}" CACHE FILEPATH "STITCH library path" FORCE)

    # Step 5: Install STITCH files to source directory
    file(GLOB STITCH_SOURCES "${STITCH_DIR}/*.cpp")
    file(GLOB STITCH_HEADERS "${STITCH_DIR}/*.h")

    if(SPPARKS_DEV_MODE)
        message(STATUS "Development mode: Creating symlinks for STITCH package files")
        foreach(SRC_FILE ${STITCH_SOURCES} ${STITCH_HEADERS})
            get_filename_component(FILENAME "${SRC_FILE}" NAME)
            set(DEST_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")

            # Remove existing file/symlink if present
            if(EXISTS "${DEST_FILE}" OR IS_SYMLINK "${DEST_FILE}")
                file(REMOVE "${DEST_FILE}")
            endif()

            # Create symlink
            file(CREATE_LINK "${SRC_FILE}" "${DEST_FILE}" SYMBOLIC)
            message(STATUS "  Linked: ${FILENAME}")
        endforeach()
        message(STATUS "STITCH package files linked (development mode)")
        message(STATUS "  Edits to src/STITCH/ will be reflected immediately!")
    else()
        # Production mode: Copy files
        foreach(SRC_FILE ${STITCH_SOURCES})
            get_filename_component(FILENAME "${SRC_FILE}" NAME)
            configure_file("${SRC_FILE}" "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}" COPYONLY)
        endforeach()

        foreach(HDR_FILE ${STITCH_HEADERS})
            get_filename_component(FILENAME "${HDR_FILE}" NAME)
            configure_file("${HDR_FILE}" "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}" COPYONLY)
        endforeach()
        message(STATUS "STITCH package files copied to source directory")
    endif()

    # Add to source lists
    list(APPEND SPPARKS_SOURCES dump_stitch.cpp)
    set(SPPARKS_SOURCES "${SPPARKS_SOURCES}" PARENT_SCOPE)

    message(STATUS "STITCH package files copied to source directory")
    message(STATUS "STITCH package installation complete")
endfunction()

# STITCH package uninstallation
function(uninstall_stitch_package)
    # Remove STITCH files from source directory
    set(STITCH_FILES
        dump_stitch.cpp
        dump_stitch.h
    )
    
    foreach(FILE ${STITCH_FILES})
        set(FULL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${FILE}")
        if(EXISTS "${FULL_PATH}")
            file(REMOVE "${FULL_PATH}")
            message(STATUS "Removed ${FILE}")
        endif()
    endforeach()

    # Clear STITCH-related cache variables to avoid stale references
    unset(STITCH_INCLUDE_DIR CACHE)
    unset(STITCH_LIBRARY_DIR CACHE)
    unset(STITCH_LIBRARY CACHE)
    message(STATUS "Cleared STITCH cache variables")

    message(STATUS "STITCH package files removed from source directory")
endfunction()

# Function to show package status
function(show_package_status)
    message(STATUS "=== SPPARKS Package Status ===")
    foreach(PKG ${SPPARKS_AVAILABLE_PACKAGES})
        string(TOUPPER "${PKG}" PKG_UPPER)
        if(SPPARKS_PACKAGE_${PKG_UPPER})
            message(STATUS "  ${PKG}: INSTALLED")
        else()
            message(STATUS "  ${PKG}: NOT INSTALLED")
        endif()
    endforeach()
    message(STATUS "==============================")
endfunction()

# Function to install all packages
function(install_all_packages)
    message(STATUS "Installing all packages...")
    foreach(PKG ${SPPARKS_AVAILABLE_PACKAGES})
        install_package("${PKG}")
    endforeach()
endfunction()

# Function to uninstall all packages
function(uninstall_all_packages)
    message(STATUS "Uninstalling all packages...")
    foreach(PKG ${SPPARKS_AVAILABLE_PACKAGES})
        uninstall_package("${PKG}")
    endforeach()
endfunction()

# Create package options
foreach(PKG ${SPPARKS_AVAILABLE_PACKAGES})
    string(TOUPPER "${PKG}" PKG_UPPER)
    option(SPPARKS_PACKAGE_${PKG_UPPER} "Enable ${PKG} package" OFF)
endforeach()

# Auto-install packages based on options
function(process_package_options)
    load_package_status()

    foreach(PKG ${SPPARKS_AVAILABLE_PACKAGES})
        string(TOUPPER "${PKG}" PKG_UPPER)

        if(SPPARKS_PACKAGE_${PKG_UPPER})
            # Check if package is not currently installed
            if(NOT DEFINED SPPARKS_PACKAGE_${PKG_UPPER}_CURRENT OR
               NOT SPPARKS_PACKAGE_${PKG_UPPER}_CURRENT)
                install_package("${PKG}")
                set(SPPARKS_PACKAGE_${PKG_UPPER}_CURRENT ON CACHE INTERNAL "")
            else()
                # Package already installed - check sync status
                check_package_sync("${PKG}")
            endif()
        else()
            # Check if package is currently installed
            if(DEFINED SPPARKS_PACKAGE_${PKG_UPPER}_CURRENT AND
               SPPARKS_PACKAGE_${PKG_UPPER}_CURRENT)
                uninstall_package("${PKG}")
                set(SPPARKS_PACKAGE_${PKG_UPPER}_CURRENT OFF CACHE INTERNAL "")
            endif()
        endif()
    endforeach()
endfunction()

# Custom targets for package management
add_custom_target(package-status
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package_status.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Show package installation and sync status"
)

add_custom_target(package-diff
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package_diff.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Show detailed differences between package source and installed files"
)

add_custom_target(install-all-packages
    COMMAND ${CMAKE_COMMAND} -DSPPARKS_INSTALL_ALL_PACKAGES=ON 
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package_installer.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Install all packages"
)

add_custom_target(uninstall-all-packages
    COMMAND ${CMAKE_COMMAND} -DSPPARKS_UNINSTALL_ALL_PACKAGES=ON 
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package_installer.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Uninstall all packages"
)

# Create package installer script
file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package_installer.cmake"
"# Package installer script
include(\"${CMAKE_CURRENT_SOURCE_DIR}/cmake/PackageManager.cmake\")

if(SPPARKS_INSTALL_ALL_PACKAGES)
    install_all_packages()
elseif(SPPARKS_UNINSTALL_ALL_PACKAGES)
    uninstall_all_packages()
endif()
")

# Process package options during configuration
process_package_options()

# Add global include directories for packages
if(SPPARKS_PACKAGE_STITCH)
    if(STITCH_INCLUDE_DIR)
        include_directories(${STITCH_INCLUDE_DIR})
        message(STATUS "Added STITCH include directory: ${STITCH_INCLUDE_DIR}")
    else()
        message(WARNING "STITCH package enabled but STITCH_INCLUDE_DIR not set")
    endif()
endif()