cmake_minimum_required(VERSION 3.16)

project(phydrax_omega_h_worker LANGUAGES CXX)

include(GNUInstallDirs)

find_package(Omega_h CONFIG REQUIRED)
if(NOT TARGET Omega_h::omega_h)
  message(FATAL_ERROR "Omega_h did not provide the required Omega_h::omega_h target.")
endif()

add_executable(phydrax-omega-h-worker main.cpp)
target_compile_features(phydrax-omega-h-worker PRIVATE cxx_std_17)
target_include_directories(
  phydrax-omega-h-worker PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common"
)
# An MPI-enabled Omega_h runs the collective worker loop; the source refuses
# to compile if this definition and Omega_h's own configuration disagree.
if(Omega_h_USE_MPI)
  target_compile_definitions(phydrax-omega-h-worker PRIVATE PHYDRAX_WORKER_WITH_MPI)
endif()
set_target_properties(
  phydrax-omega-h-worker
  PROPERTIES CXX_EXTENSIONS OFF INSTALL_RPATH_USE_LINK_PATH ON
)
target_link_libraries(phydrax-omega-h-worker PRIVATE Omega_h::omega_h)

install(TARGETS phydrax-omega-h-worker RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
