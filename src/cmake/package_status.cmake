# Package status checker script
# Usage: cmake -P package_status.cmake

# Available packages (must match PackageManager.cmake)
set(SPPARKS_AVAILABLE_PACKAGES "stitch" "kokkos")

message("╔═══════════════════════════════════════════════════════════════")
message("║ SPPARKS Package Status")
message("╠═══════════════════════════════════════════════════════════════")

foreach(PKG ${SPPARKS_AVAILABLE_PACKAGES})
    string(TOUPPER "${PKG}" PKG_UPPER)
    set(PKG_DIR "${CMAKE_CURRENT_LIST_DIR}/../${PKG_UPPER}")

    if(NOT EXISTS "${PKG_DIR}")
        message("║ ${PKG}: NOT AVAILABLE (directory not found)")
        continue()
    endif()

    # Check if any files from this package are installed
    file(GLOB PKG_CPP "${PKG_DIR}/*.cpp")
    file(GLOB PKG_H "${PKG_DIR}/*.h")
    set(PKG_FILES ${PKG_CPP} ${PKG_H})

    set(FILES_INSTALLED FALSE)
    set(FILES_DIFFER FALSE)
    set(DIFFER_COUNT 0)

    foreach(PKG_FILE ${PKG_FILES})
        get_filename_component(FILENAME "${PKG_FILE}" NAME)
        set(SRC_FILE "${CMAKE_CURRENT_LIST_DIR}/../${FILENAME}")

        if(EXISTS "${SRC_FILE}")
            set(FILES_INSTALLED TRUE)

            # Compare files
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E compare_files "${PKG_FILE}" "${SRC_FILE}"
                RESULT_VARIABLE FILES_ARE_DIFFERENT
                OUTPUT_QUIET
                ERROR_QUIET
            )

            if(FILES_ARE_DIFFERENT)
                set(FILES_DIFFER TRUE)
                math(EXPR DIFFER_COUNT "${DIFFER_COUNT} + 1")
            endif()
        endif()
    endforeach()

    if(FILES_INSTALLED)
        if(FILES_DIFFER)
            message("║ ${PKG}: INSTALLED ⚠️  WARNING - ${DIFFER_COUNT} file(s) differ from source!")
        else()
            message("║ ${PKG}: INSTALLED ✓ (in sync)")
        endif()
    else()
        message("║ ${PKG}: NOT INSTALLED")
    endif()
endforeach()

message("╠═══════════════════════════════════════════════════════════════")
message("║ GOLDEN RULE: Always edit src/PACKAGE/ files, not src/ copies!")
message("║")
message("║ Commands:")
message("║   cmake --build . --target package-status  # Show this status")
message("║   cmake --build . --target package-diff    # Show file differences")
message("╚═══════════════════════════════════════════════════════════════")
