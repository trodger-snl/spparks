# Generates src/style_<type>.h: the #include lists through which App / Command /
# Diag / Dump / Pair / Region / Solve classes register themselves (the *Style(...)
# macros in their headers). Two entry points, one file:
#   include(cmake/generate_styles.cmake)   at configure time -> all styles, plus
#                                          a style_headers target that regenerates
#                                          one style when its headers change
#   cmake -DSTYLE_TYPE=<type> -DSPK_SOURCE_DIR=.. -DSPK_BINARY_DIR=.. -P <this>
#                                          from that target (script mode)
# Feature gating lives in ConditionalSources.cmake (should_include_header); in
# script mode the feature flags come from <build>/feature_status.cmake.

if(NOT DEFINED SPK_SOURCE_DIR)
    set(SPK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
if(NOT DEFINED SPK_BINARY_DIR)
    set(SPK_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()
if(CMAKE_SCRIPT_MODE_FILE)
    include("${SPK_BINARY_DIR}/feature_status.cmake")
    include("${SPK_SOURCE_DIR}/cmake/ConditionalSources.cmake")
endif()

# style | class macro | header prefix ("" = every header, filtered by macro)
set(SPK_STYLE_RULES
    "app|APP_CLASS|app_"
    "command|COMMAND_CLASS|"
    "diag|DIAG_CLASS|diag_"
    "dump|DUMP_CLASS|dump_"
    "pair|PAIR_CLASS|pair_"
    "region|REGION_CLASS|region_"
    "solve|SOLVE_CLASS|solve_"
)

# Directories whose headers can register styles.
set(SPK_STYLE_DIRS "${SPK_SOURCE_DIR}")

# Sets SPK_STYLE_${STYLE}_HEADERS (candidate headers, for build dependencies)
# in the caller's scope and rewrites style_${STYLE}.h if its content changed.
function(spk_generate_style STYLE CLASS_PATTERN PREFIX)
    set(candidates "")
    foreach(dir IN LISTS SPK_STYLE_DIRS)
        file(GLOB found "${dir}/${PREFIX}*.h")
        list(APPEND candidates ${found})
    endforeach()
    set(SPK_STYLE_${STYLE}_HEADERS "${candidates}" PARENT_SCOPE)

    set(content "")
    foreach(header IN LISTS candidates)
        file(READ "${header}" body)
        if(NOT body MATCHES "${CLASS_PATTERN}")
            continue()
        endif()
        get_filename_component(name "${header}" NAME)
        should_include_header("${name}" ok why)
        if(ok)
            string(APPEND content "#include \"${name}\"\n")
        else()
            message(STATUS "style_${STYLE}.h: skipping ${name} -- ${why}")
        endif()
    endforeach()

    set(out "${SPK_SOURCE_DIR}/style_${STYLE}.h")
    set(old "")
    if(EXISTS "${out}")
        file(READ "${out}" old)
    endif()
    if(NOT EXISTS "${out}" OR NOT old STREQUAL content)
        file(WRITE "${out}" "${content}")
        message(STATUS "Generated ${out}")
    endif()
endfunction()

foreach(rule IN LISTS SPK_STYLE_RULES)
    string(REPLACE "|" ";" parts "${rule}")
    list(GET parts 0 style)
    list(GET parts 1 pattern)
    list(GET parts 2 prefix)
    if(STYLE_TYPE AND NOT style STREQUAL STYLE_TYPE)
        continue()
    endif()
    spk_generate_style("${style}" "${pattern}" "${prefix}")

    if(NOT CMAKE_SCRIPT_MODE_FILE)
        add_custom_command(
            OUTPUT "${SPK_SOURCE_DIR}/style_${style}.h"
            DEPENDS ${SPK_STYLE_${style}_HEADERS} "${SPK_BINARY_DIR}/feature_status.cmake"
            COMMAND ${CMAKE_COMMAND}
                -DSTYLE_TYPE=${style}
                -DSPK_SOURCE_DIR=${SPK_SOURCE_DIR}
                -DSPK_BINARY_DIR=${SPK_BINARY_DIR}
                -P "${CMAKE_CURRENT_LIST_FILE}"
            COMMENT "Regenerating style_${style}.h"
        )
        list(APPEND SPK_STYLE_OUTPUTS "${SPK_SOURCE_DIR}/style_${style}.h")
    endif()
endforeach()

if(NOT CMAKE_SCRIPT_MODE_FILE)
    add_custom_target(style_headers DEPENDS ${SPK_STYLE_OUTPUTS})
endif()
