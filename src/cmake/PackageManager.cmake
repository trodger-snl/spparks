# STITCH: build lib/stitch's libstitch.a with its own Makefile at configure time
# and expose STITCH_INCLUDE_DIR / STITCH_LIBRARY. The dump_stitch sources compile
# straight from src/STITCH/ (see ConditionalSources.cmake).

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
        # Pass CC/CXX as make COMMAND-LINE variables, not environment variables.
        # lib/stitch's Makefile assigns CC=`which mpicc` internally, and a
        # makefile assignment beats the environment -- so the previous
        # `cmake -E env CC=... make` form was silently ignored and stitch was
        # always built with whatever `which mpicc` found, which need not be the
        # toolchain the rest of the build uses (and need not exist at all).
        # Command-line assignments do override makefile assignments.
        #
        # -fPIC is required: SPPARKS links as a PIE, and a non-PIC libstitch.a
        # fails at link with
        #   ld.lld: error: relocation R_X86_64_32 cannot be used against local
        #   symbol; recompile with -fPIC   (sqlite3.o, stitch.o)
        # The Makefile has no CFLAGS hook, so fold the flag into CC/CXX.
        execute_process(
            COMMAND ${CMAKE_MAKE_PROGRAM} "CC=${STITCH_CC} -fPIC" "CXX=${STITCH_CXX} -fPIC" stitch.lib
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
    # Clear any cached NOTFOUND result from previous configurations
    unset(STITCH_LIBRARY CACHE)
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

endfunction()

if(SPPARKS_PACKAGE_STITCH)
    install_stitch_package()
    include_directories(${STITCH_INCLUDE_DIR})
endif()
