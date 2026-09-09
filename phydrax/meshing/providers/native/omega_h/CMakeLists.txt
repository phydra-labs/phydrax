cmake_minimum_required(VERSION 3.16)

project(phydrax_omega_h_bridge LANGUAGES CXX)

include(GNUInstallDirs)

find_package(Omega_h CONFIG REQUIRED)
if(NOT TARGET Omega_h::omega_h)
  message(FATAL_ERROR "Omega_h did not provide the required Omega_h::omega_h target.")
endif()

add_executable(phydrax_omega_h main.cpp)
target_compile_features(phydrax_omega_h PRIVATE cxx_std_14)
set_target_properties(
  phydrax_omega_h
  PROPERTIES CXX_EXTENSIONS OFF INSTALL_RPATH_USE_LINK_PATH ON
)
target_link_libraries(phydrax_omega_h PRIVATE Omega_h::omega_h)

install(TARGETS phydrax_omega_h RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
