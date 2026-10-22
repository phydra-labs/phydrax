cmake_minimum_required(VERSION 3.28)
# Packaging a redistributable must not discover a compiler or a system CUDA SDK.
project(phydrax_cuest LANGUAGES NONE)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _phx_version_line
  REGEX "^version = \"[0-9]+\\.[0-9]+\\.[0-9]+\"$" LIMIT_COUNT 1)
string(REGEX REPLACE "^version = \"([0-9.]+)\"$" "\\1"
  PHX_CUEST_PACKAGE_VERSION "${_phx_version_line}")
if(NOT PHX_CUEST_PACKAGE_VERSION)
  message(FATAL_ERROR "cuEST runtime pyproject.toml must declare a release version")
endif()
set(PHX_CUEST_SOURCE_VERSION "0.2.2.2")

if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "cuEST runtime wheels require Linux")
endif()
if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
  message(FATAL_ERROR "cuEST runtime wheels do not support cross-compilation")
endif()
# These are the CUDA wrapper's immutable C redistributable pins, not the
# independently versioned Python NVIDIA wheel or the Phydrax release number.
if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(_phx_cuest_platform "linux-x86_64")
  set(_phx_cuest_sha "71641ba4aaa068dad037b9837f6fd0920edb7283f8297eccb9bedba6f405d228")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(_phx_cuest_platform "linux-sbsa")
  set(_phx_cuest_sha "413e9050d2c1e998a0a6eb803e1c468217fbeebe25e5e80068c22b5ce8e66c31")
else()
  message(FATAL_ERROR "Unsupported cuEST runtime architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif()
if(NOT DEFINED SKBUILD_METADATA_DIR)
  message(FATAL_ERROR "Build phydrax-cuest through its wheel backend")
endif()
if(NOT DEFINED PHX_CUDA_DOWNLOAD_DIR)
  set(PHX_CUDA_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloads")
endif()

function(_phx_cuest_archive platform sha)
  set(stem "libcuest-${platform}-${PHX_CUEST_SOURCE_VERSION}_cuda13-archive")
  set(archive "${PHX_CUDA_DOWNLOAD_DIR}/${stem}.tar.xz")
  file(MAKE_DIRECTORY "${PHX_CUDA_DOWNLOAD_DIR}")
  file(LOCK "${PHX_CUDA_DOWNLOAD_DIR}/${stem}.lock" GUARD FUNCTION TIMEOUT 600)
  set(url
    "https://developer.download.nvidia.com/compute/cuest/redist/libcuest/${platform}/${stem}.tar.xz")
  if(NOT EXISTS "${archive}")
    file(DOWNLOAD "${url}" "${archive}" EXPECTED_HASH "SHA256=${sha}" TLS_VERIFY ON)
  endif()
  file(SHA256 "${archive}" actual_sha)
  if(NOT actual_sha STREQUAL sha)
    message(FATAL_ERROR "cuEST C redistributable archive SHA256 mismatch")
  endif()
  # Re-extract verified bytes on each configure; an existing extracted tree is
  # not evidence that its runtime or license still matches the archive.
  set(extraction "${CMAKE_BINARY_DIR}/cuest-support")
  file(REMOVE_RECURSE "${extraction}")
  file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${extraction}")
  set(PHX_CUEST_ROOT "${extraction}/${stem}" PARENT_SCOPE)
  set(PHX_CUEST_ARCHIVE_URL "${url}" PARENT_SCOPE)
  set(PHX_CUEST_ARCHIVE_SHA256 "${sha}" PARENT_SCOPE)
endfunction()
_phx_cuest_archive("${_phx_cuest_platform}" "${_phx_cuest_sha}")
file(SHA256 "${PHX_CUEST_ROOT}/lib/libcuest.so.0.2.2" _phx_library_sha)
file(SHA256 "${PHX_CUEST_ROOT}/LICENSE" _phx_license_sha)
set(_phx_manifest "${CMAKE_BINARY_DIR}/phydrax-cuest-build.json")
file(WRITE "${_phx_manifest}"
  "{\n"
  "  \"architecture\": \"${CMAKE_HOST_SYSTEM_PROCESSOR}\",\n"
  "  \"archive_sha256\": \"${PHX_CUEST_ARCHIVE_SHA256}\",\n"
  "  \"archive_url\": \"${PHX_CUEST_ARCHIVE_URL}\",\n"
  "  \"cuda_major\": 13,\n"
  "  \"library\": \"lib/libcuest.so.0\",\n"
  "  \"library_sha256\": \"${_phx_library_sha}\",\n"
  "  \"license\": \"phydrax_cuest-${PHX_CUEST_PACKAGE_VERSION}.dist-info/licenses/cuEST-LICENSE.txt\",\n"
  "  \"license_sha256\": \"${_phx_license_sha}\",\n"
  "  \"package_version\": \"${PHX_CUEST_PACKAGE_VERSION}\",\n"
  "  \"redistribution_manifest_url\": \"https://developer.download.nvidia.com/compute/cuest/redist/redistrib_0.2.2.json\",\n"
  "  \"soname\": \"libcuest.so.0\",\n"
  "  \"source_version\": \"${PHX_CUEST_SOURCE_VERSION}\",\n"
  "  \"vendor_platform\": \"${_phx_cuest_platform}\"\n"
  "}\n")

# Install the real ELF under its SONAME: wheels do not preserve symlinks, and
# the wrapper's relocatable RPATH must work without loader preloading.
install(FILES "${PHX_CUEST_ROOT}/lib/libcuest.so.0.2.2"
  DESTINATION "phydrax_cuest/lib" RENAME "libcuest.so.0")
install(FILES "${_phx_manifest}" DESTINATION "phydrax_cuest" RENAME "build.json")
# Preserve the complete vendor document, including all third-party terms.
install(FILES "${PHX_CUEST_ROOT}/LICENSE"
  DESTINATION "${SKBUILD_METADATA_DIR}/licenses" RENAME "cuEST-LICENSE.txt")
get_filename_component(_phx_repo "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
install(FILES "${_phx_repo}/LICENSE" "${_phx_repo}/NOTICE"
  DESTINATION "${SKBUILD_METADATA_DIR}/licenses")
