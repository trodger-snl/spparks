# Package management system for SPPARKS CMake build
# Equivalent to the Makefile package system (yes-package/no-package)

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

# Function to install a package
function(install_package PACKAGE_NAME)
    string(TOUPPER "${PACKAGE_NAME}" PKG_UPPER)
    
    if(NOT "${PACKAGE_NAME}" IN_LIST SPPARKS_AVAILABLE_PACKAGES)
        message(FATAL_ERROR "Package '${PACKAGE_NAME}' does not exist")
    endif()
    
    if(SPPARKS_PACKAGE_${PKG_UPPER})
        message(STATUS "Package '${PACKAGE_NAME}' is already installed")
        return()
    endif()
    
    message(STATUS "Installing package '${PACKAGE_NAME}'...")
    
    # Package-specific installation logic
    if(PACKAGE_NAME STREQUAL "stitch")
        install_stitch_package()
    endif()
    
    # Mark package as installed
    set(SPPARKS_PACKAGE_${PKG_UPPER} ON PARENT_SCOPE)
    set(SPPARKS_PACKAGE_${PKG_UPPER} ON CACHE BOOL "Package ${PACKAGE_NAME} is installed")
    save_package_status()
    
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
    
    if(NOT EXISTS "${STITCH_DIR}")
        message(FATAL_ERROR "STITCH package directory not found")
    endif()
    
    # Copy STITCH files to source directory
    file(GLOB STITCH_SOURCES "${STITCH_DIR}/*.cpp")
    file(GLOB STITCH_HEADERS "${STITCH_DIR}/*.h")
    
    foreach(SRC_FILE ${STITCH_SOURCES})
        get_filename_component(FILENAME "${SRC_FILE}" NAME)
        configure_file("${SRC_FILE}" "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}" COPYONLY)
    endforeach()
    
    foreach(HDR_FILE ${STITCH_HEADERS})
        get_filename_component(FILENAME "${HDR_FILE}" NAME)
        configure_file("${HDR_FILE}" "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}" COPYONLY)
    endforeach()
    
    # Add to source lists
    list(APPEND SPPARKS_SOURCES dump_stitch.cpp)
    set(SPPARKS_SOURCES "${SPPARKS_SOURCES}" PARENT_SCOPE)
    
    message(STATUS "STITCH package files copied to source directory")
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
    COMMAND ${CMAKE_COMMAND} -E echo "=== SPPARKS Package Status ==="
    COMMENT "Show package installation status"
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