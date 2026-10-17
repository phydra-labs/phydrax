# Pinned upstream sources, compiled as an internal static PIC dependency only.
include(FetchContent)
find_program(PHX_PATCH_EXECUTABLE patch REQUIRED)
set(PHX_AMGX_ARCHIVE_URL "https://codeload.github.com/NVIDIA/AMGX/tar.gz/91a8413")
# A shared development download cache avoids storing the archive per build tree.
if(EXISTS "${PHX_CUDA_DOWNLOAD_DIR}/amgx-91a8413.tar.gz")
  set(PHX_AMGX_ARCHIVE_URL "${PHX_CUDA_DOWNLOAD_DIR}/amgx-91a8413.tar.gz")
endif()
FetchContent_Declare(phx_amgx_source
  URL "${PHX_AMGX_ARCHIVE_URL}"
  URL_HASH SHA256=a5a9034b7dbfd7ae486436074609faafad0730c3c9acb07e90caecff6f216c0d
  DOWNLOAD_DIR "${PHX_CUDA_DOWNLOAD_DIR}"
  DOWNLOAD_NAME amgx-91a8413.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR phydrax-no-upstream-build
  PATCH_COMMAND "${PHX_PATCH_EXECUTABLE}" -p1 -i "${CMAKE_CURRENT_LIST_DIR}/../patches/amgx-classical-amg-cuda13.patch"
        COMMAND "${PHX_PATCH_EXECUTABLE}" -p1 -i "${CMAKE_CURRENT_LIST_DIR}/../patches/amgx-dddi-only.patch"
        COMMAND "${PHX_PATCH_EXECUTABLE}" --binary -p1 -i "${CMAKE_CURRENT_LIST_DIR}/../patches/amgx-dddi-v-cycle.patch"
        COMMAND "${PHX_PATCH_EXECUTABLE}" -p1 -i "${CMAKE_CURRENT_LIST_DIR}/../patches/amgx-device-library-handles.patch"
)
# Populate without upstream's shared library, examples, tests, MPI discovery,
# install rules, or global compiler flags. These are all outside our C API build.
FetchContent_MakeAvailable(phx_amgx_source)
file(GLOB_RECURSE PHX_AMGX_SOURCES CONFIGURE_DEPENDS "${phx_amgx_source_SOURCE_DIR}/src/*.cu")
list(FILTER PHX_AMGX_SOURCES EXCLUDE REGEX "/src/tests/")
list(FILTER PHX_AMGX_SOURCES EXCLUDE REGEX "/(utest|testframework|test_utils|unit_test)\\.cu$")
add_library(phx_amgx_static STATIC ${PHX_AMGX_SOURCES})
set_target_properties(phx_amgx_static PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  CUDA_RUNTIME_LIBRARY Shared
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
)
target_include_directories(phx_amgx_static PUBLIC "${phx_amgx_source_SOURCE_DIR}/include"
  PRIVATE "${phx_amgx_source_SOURCE_DIR}/external/rapidjson/include")
# No AMGX_WITH_MPI: the companion is deliberately a single-device, non-MPI build.
# Host dDDI and integer auxiliary types remain internal; no float/complex modes.
target_compile_definitions(phx_amgx_static PRIVATE
  PHX_AMGX_DDDI_ONLY RAPIDJSON_DEFINED THRUST_CUB_WRAPPED_NAMESPACE=amgx)
target_compile_options(phx_amgx_static PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-Wno-deprecated-gpu-targets;-Xcompiler=-Wno-terminate>
)
target_link_libraries(phx_amgx_static PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::cusolver m pthread)
