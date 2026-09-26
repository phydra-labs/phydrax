cmake_minimum_required(VERSION 3.16)

project(phydrax_tioga_worker LANGUAGES CXX)

include(GNUInstallDirs)

find_package(TIOGA CONFIG REQUIRED)

if(TARGET TIOGA::tioga)
  set(_phydrax_tioga_target TIOGA::tioga)
elseif(TARGET tioga)
  set(_phydrax_tioga_target tioga)
else()
  message(FATAL_ERROR "TIOGA did not provide a tioga library target.")
endif()

if(NOT DEFINED TIOGA_HAS_NODEGID OR NOT TIOGA_HAS_NODEGID)
  message(
    FATAL_ERROR
    "The Phydrax worker requires an explicit true TIOGA_HAS_NODEGID capability."
  )
endif()

# The worker must use the MPI implementation TIOGA was compiled with; the
# package records its MPI compiler wrapper, which selects that implementation.
if(NOT DEFINED TIOGA_CXX_COMPILER OR NOT EXISTS "${TIOGA_CXX_COMPILER}")
  message(
    FATAL_ERROR
    "The TIOGA package must record the MPI C++ compiler wrapper it was built with."
  )
endif()
if(NOT DEFINED MPI_CXX_COMPILER)
  set(MPI_CXX_COMPILER "${TIOGA_CXX_COMPILER}")
endif()
find_package(MPI REQUIRED COMPONENTS CXX)
get_filename_component(_phydrax_tioga_wrapper "${TIOGA_CXX_COMPILER}" REALPATH)
get_filename_component(_phydrax_mpi_wrapper "${MPI_CXX_COMPILER}" REALPATH)
if(NOT _phydrax_tioga_wrapper STREQUAL _phydrax_mpi_wrapper)
  message(
    FATAL_ERROR
    "MPI_CXX_COMPILER (${MPI_CXX_COMPILER}) differs from the TIOGA build wrapper (${TIOGA_CXX_COMPILER}); mixing MPI implementations is unsupported."
  )
endif()

set(
  PHYDRAX_TIOGA_REVISION
  ""
  CACHE STRING
  "Exact TIOGA release or Git revision reported by the worker"
)
if(NOT "${PHYDRAX_TIOGA_REVISION}" MATCHES "^[A-Za-z0-9][-A-Za-z0-9._+:]*$")
  message(
    FATAL_ERROR
    "Set PHYDRAX_TIOGA_REVISION to the exact non-whitespace TIOGA release or Git revision."
  )
endif()

# TIOGA does not export this option, so the builder declares it explicitly.
set(
  PHYDRAX_TIOGA_ENABLE_UNIQUEID
  ""
  CACHE STRING
  "TIOGA_ENABLE_UNIQUEID (ON or OFF) of the linked TIOGA build"
)
if(PHYDRAX_TIOGA_ENABLE_UNIQUEID STREQUAL "ON")
  set(_phydrax_tioga_uniqueid 1)
elseif(PHYDRAX_TIOGA_ENABLE_UNIQUEID STREQUAL "OFF")
  set(_phydrax_tioga_uniqueid 0)
else()
  message(
    FATAL_ERROR
    "Set PHYDRAX_TIOGA_ENABLE_UNIQUEID to ON or OFF, matching the TIOGA_ENABLE_UNIQUEID of the linked TIOGA build."
  )
endif()

add_executable(phydrax-tioga-worker worker.cpp)
target_compile_features(phydrax-tioga-worker PRIVATE cxx_std_17)
set_target_properties(
  phydrax-tioga-worker
  PROPERTIES CXX_EXTENSIONS OFF INSTALL_RPATH_USE_LINK_PATH ON
)
target_compile_definitions(
  phydrax-tioga-worker
  PRIVATE
    PHYDRAX_WORKER_WITH_MPI
    "PHYDRAX_TIOGA_REVISION=\"${PHYDRAX_TIOGA_REVISION}\""
    "PHYDRAX_TIOGA_ENABLE_UNIQUEID=${_phydrax_tioga_uniqueid}"
)
target_include_directories(
  phydrax-tioga-worker
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common" ${TIOGA_INCLUDE_DIRS}
)
target_link_libraries(
  phydrax-tioga-worker
  PRIVATE "${_phydrax_tioga_target}" MPI::MPI_CXX
)

install(TARGETS phydrax-tioga-worker RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
