# Package diff script - shows detailed differences
# Usage: cmake -P package_diff.cmake

# Available packages (must match PackageManager.cmake)
set(SPPARKS_AVAILABLE_PACKAGES "stitch" "kokkos")

set(ANY_DIFFS FALSE)

foreach(PKG ${SPPARKS_AVAILABLE_PACKAGES})
    string(TOUPPER "${PKG}" PKG_UPPER)
    set(PKG_DIR "${CMAKE_CURRENT_LIST_DIR}/../${PKG_UPPER}")

    if(NOT EXISTS "${PKG_DIR}")
        continue()
    endif()

    # Find all package files
    file(GLOB PKG_CPP "${PKG_DIR}/*.cpp")
    file(GLOB PKG_H "${PKG_DIR}/*.h")
    set(PKG_FILES ${PKG_CPP} ${PKG_H})

    set(PKG_HAS_DIFFS FALSE)
    set(DIFF_LIST "")

    foreach(PKG_FILE ${PKG_FILES})
        get_filename_component(FILENAME "${PKG_FILE}" NAME)
        set(SRC_FILE "${CMAKE_CURRENT_LIST_DIR}/../${FILENAME}")

        if(EXISTS "${SRC_FILE}")
            # Compare files
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E compare_files "${PKG_FILE}" "${SRC_FILE}"
                RESULT_VARIABLE FILES_ARE_DIFFERENT
                OUTPUT_QUIET
                ERROR_QUIET
            )

            if(FILES_ARE_DIFFERENT)
                set(PKG_HAS_DIFFS TRUE)
                set(ANY_DIFFS TRUE)
                list(APPEND DIFF_LIST "${FILENAME}")
            endif()
        endif()
    endforeach()

    if(PKG_HAS_DIFFS)
        message("╔═══════════════════════════════════════════════════════════════")
        message("║ Package: ${PKG}")
        message("╠═══════════════════════════════════════════════════════════════")
        message("║ Files that differ between src/${PKG_UPPER}/ and src/:")
        message("║")
        foreach(FILE ${DIFF_LIST})
            message("║   • ${FILE}")
            message("║     Source: src/${PKG_UPPER}/${FILE}")
            message("║     Installed: src/${FILE}")
            message("║")

            # Try to show diff if diff command is available
            find_program(DIFF_COMMAND diff)
            if(DIFF_COMMAND)
                message("║     Differences:")
                execute_process(
                    COMMAND ${DIFF_COMMAND} -u "${PKG_DIR}/${FILE}" "${CMAKE_CURRENT_LIST_DIR}/../${FILE}"
                    OUTPUT_VARIABLE DIFF_OUTPUT
                    ERROR_QUIET
                )
                # Limit output to first 20 lines
                string(REPLACE "\n" ";" DIFF_LINES "${DIFF_OUTPUT}")
                list(LENGTH DIFF_LINES NUM_LINES)
                if(NUM_LINES GREATER 20)
                    list(SUBLIST DIFF_LINES 0 20 DIFF_LINES)
                    list(APPEND DIFF_LINES "     ... (output truncated, use 'diff' command for full output)")
                endif()
                foreach(LINE ${DIFF_LINES})
                    message("║     ${LINE}")
                endforeach()
                message("║")
            endif()
        endforeach()
        message("║ To sync files:")
        message("║   1. If src/${PKG_UPPER}/ is correct: cmake ..")
        message("║   2. If src/ has your changes: copy them to src/${PKG_UPPER}/ first!")
        message("╚═══════════════════════════════════════════════════════════════")
        message("")
    endif()
endforeach()

if(NOT ANY_DIFFS)
    message("╔═══════════════════════════════════════════════════════════════")
    message("║ No differences found between package source and installed files")
    message("║ All installed packages are in sync! ✓")
    message("╚═══════════════════════════════════════════════════════════════")
endif()
