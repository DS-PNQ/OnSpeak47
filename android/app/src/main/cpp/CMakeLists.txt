cmake_minimum_required(VERSION 3.22.1)
project("hymt_gguf")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Optimization — root cause #1 of the 16s translations.
#
# AGP ALWAYS passes an explicit -DCMAKE_BUILD_TYPE to CMake (Debug for
# assembleDebug), so a "if(NOT CMAKE_BUILD_TYPE)" guard NEVER fires: the
# previous RelWithDebInfo force was silently ignored and the entire
# llama.cpp/ggml stack compiled as default -O0 code (verified in
# .cxx/Debug/*/CMakeCache.txt = Debug, with no -O flag in the ninja rules).
# ggml matmuls at -O0 are 5-10x slower than -O2 — RTranslator's speed came
# from optimized native code, not from anything else.
#
# Fix: append -O2 to the per-config flag variables that the Debug build
# actually uses (CMake's default Debug flags are just "-g"). This MUST be
# set before FetchContent_MakeAvailable below: the ggml/llama targets are
# created there and read these variables at create time. Clang last-wins,
# so "-g -O2" keeps debug symbols while optimizing.
#
# Optional further win on dotprod-capable SoCs (Snapdragon 845+, 2018+):
#   -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod  (NEON dot-product kernels)
#   -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+i8mm  (SD8Elite / X Elite)
# bumps ggml-cpu to the optimized kernels. Verify on-device before
# enabling: Cortex-A53/A55-only SoCs lack the feature and would crash.
# Pass via gradle: externalNativeBuild { cmake { arguments "-DGGML_CPU_ARM_ARCH=..." } }
# ---------------------------------------------------------------------------
string(APPEND CMAKE_C_FLAGS_DEBUG " -O2")
string(APPEND CMAKE_CXX_FLAGS_DEBUG " -O2")
# Release (assembleRelease) already defaults to -O2 -DNDEBUG; promote to -O3
# for ggml matmuls. RelWithDebInfo (profiling builds) same treatment.
string(APPEND CMAKE_C_FLAGS_RELEASE " -O3")
string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
string(APPEND CMAKE_C_FLAGS_RELWITHDEBINFO " -O2")
string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -O2")


# ---------------------------------------------------------------------------
# llama.cpp (STQ kernel — PR #22836, STQ1_0 ternary quantization)
#
# The Hy-MT 1.25-bit GGUF only decodes with llama.cpp built with the STQ
# kernel (PR #22836, still OPEN — not merged into master as of 2026-09-09,
# head = sjl623:STQ_0 @ 1e411d8f5a1e23525fa3265dfb4bd76265465397).
#
# To keep Android builds offline-friendly (the original FetchContent git
# clone of https://github.com/ggerganov/llama.cpp.git failed on this machine
# with "Could not resolve host: github.com"), the source is vendored once
# into third_party/llama.cpp (see third_party/README.md). FetchContent is
# only used as a fallback when the vendored checkout is missing.
# ---------------------------------------------------------------------------

set(LLAMA_CPP_LOCAL "${CMAKE_CURRENT_LIST_DIR}/third_party/llama.cpp")
set(LLAMA_CPP_PR_SHA "1e411d8f5a1e23525fa3265dfb4bd76265465397")  # sjl623:STQ_0 (PR #22836)

include(FetchContent)

# FetchContent_Declare must always be called for the FETCHCONTENT_SOURCE_DIR_<name>
# override to be honoured by FetchContent_MakeAvailable (CMake 3.22).
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/sjl623/llama.cpp.git
    GIT_TAG        ${LLAMA_CPP_PR_SHA}
)

if(EXISTS "${LLAMA_CPP_LOCAL}/CMakeLists.txt" AND EXISTS "${LLAMA_CPP_LOCAL}/ggml")
    # Prefer the vendored snapshot: no network access needed at configure time.
    set(FETCHCONTENT_SOURCE_DIR_LLAMA_CPP "${LLAMA_CPP_LOCAL}"
        CACHE PATH "Local llama.cpp source (overrides the git fetch)" FORCE)
    message(STATUS "llama.cpp: using vendored copy at ${LLAMA_CPP_LOCAL}")
else()
    message(WARNING
        "llama.cpp: vendored copy not found under third_party/. "
        "Falling back to FetchContent git clone of PR #22836 "
        "(https://github.com/sjl623/llama.cpp.git @ ${LLAMA_CPP_PR_SHA}) — "
        "this requires network access to github.com. "
        "Run 'download_llama_cpp.ps1' in third_party/ to vendor it.")
endif()

set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp)

add_library(hymt_gguf SHARED
    hymt_gguf_jni.cpp
)

# ---------------------------------------------------------------------------
# llama.cpp's own CMakeLists.txt forces its libraries into
# ${CMAKE_BINARY_DIR}/bin, which AGP does NOT package into the APK (it only
# packages CMAKE_LIBRARY_OUTPUT_DIRECTORY). Copy each llama.cpp library
# next to libhymt_gguf.so so the APK bundles the full closure.
# ---------------------------------------------------------------------------
foreach(_llama_lib llama ggml ggml-base ggml-cpu)
    if(TARGET ${_llama_lib})
        add_custom_command(TARGET hymt_gguf POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:${_llama_lib}>
                    $<TARGET_FILE_DIR:hymt_gguf>
            COMMENT "Packaging ${_llama_lib} for the APK")
    endif()
endforeach()

target_include_directories(hymt_gguf PRIVATE
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
)

target_link_libraries(hymt_gguf
    llama
    log
)

# ---------------------------------------------------------------------------
# Belt-and-suspenders for the -O2 fix at the top of this file: attach the
# option to the targets themselves. Target options travel with the target
# and are immune to any flag-variable overwrite inside the fetched llama.cpp
# projects. $<CONFIG:Debug> matches the build type AGP passes explicitly;
# clang is last-wins, and target COMPILE_OPTIONS come after the per-config
# flags on the command line, so this -O2 overrides the toolchain default.
# Release gets -O3 for ggml matmuls.
# ---------------------------------------------------------------------------
foreach(_opt_target llama ggml ggml-base ggml-cpu hymt_gguf)
    if(TARGET ${_opt_target})
        target_compile_options(${_opt_target} PRIVATE $<$<CONFIG:Debug>:-O2> $<$<CONFIG:Release>:-O3> $<$<CONFIG:RelWithDebInfo>:-O2>)
    endif()
endforeach()
