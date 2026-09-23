cmake_minimum_required(VERSION 3.16)

project(phydrax_tioga_bridge LANGUAGES CXX)

include(GNUInstallDirs)

find_package(MPI REQUIRED COMPONENTS CXX)
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
    "The Phydrax bridge requires an explicit true TIOGA_HAS_NODEGID capability."
  )
endif()

get_target_property(
  _phydrax_tioga_link_interface
  "${_phydrax_tioga_target}"
  INTERFACE_LINK_LIBRARIES
)
string(
  FIND
  "${_phydrax_tioga_link_interface}"
  "MPI::MPI_CXX"
  _phydrax_tioga_mpi_target_index
)
if(_phydrax_tioga_mpi_target_index EQUAL -1)
  message(
    FATAL_ERROR
    "The TIOGA package target must export MPI::MPI_CXX so the bridge cannot mix MPI implementations."
  )
endif()

set(
  PHYDRAX_TIOGA_REVISION
  ""
  CACHE STRING
  "Exact TIOGA release or Git revision reported by the bridge"
)
if(NOT "${PHYDRAX_TIOGA_REVISION}" MATCHES "^[A-Za-z0-9][-A-Za-z0-9._+:]*$")
  message(
    FATAL_ERROR
    "Set PHYDRAX_TIOGA_REVISION to the exact non-whitespace TIOGA release or Git revision."
  )
endif()

add_executable(phydrax-tioga bridge.cpp)
target_compile_features(phydrax-tioga PRIVATE cxx_std_14)
set_target_properties(
  phydrax-tioga
  PROPERTIES CXX_EXTENSIONS OFF INSTALL_RPATH_USE_LINK_PATH ON
)
target_compile_definitions(
  phydrax-tioga
  PRIVATE "PHYDRAX_TIOGA_REVISION=\"${PHYDRAX_TIOGA_REVISION}\""
)
target_link_libraries(phydrax-tioga PRIVATE "${_phydrax_tioga_target}" MPI::MPI_CXX)

install(TARGETS phydrax-tioga RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
