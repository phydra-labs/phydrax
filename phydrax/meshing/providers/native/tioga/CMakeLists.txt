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

if(DEFINED TIOGA_HAS_NODEGID AND NOT TIOGA_HAS_NODEGID)
  message(FATAL_ERROR "The Phydrax bridge requires TIOGA global node-ID support.")
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
