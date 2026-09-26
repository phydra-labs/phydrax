cmake_minimum_required(VERSION 3.16)

project(phydrax_mmg_worker LANGUAGES C CXX)

include(GNUInstallDirs)

# OFF: the serial worker phydrax-mmg-worker (mmg2d, mmgs, mmg3d).
# ON: the collective worker phydrax-parmmg-worker (ParMmg tetrahedral
# remeshing across MPI ranks); ParMmg links its own Mmg build, so point
# CMAKE_PREFIX_PATH at the ParMmg install and the Mmg it was built against.
option(PHYDRAX_MMG_WITH_PARMMG "Build the collective ParMmg worker" OFF)

find_package(mmg CONFIG REQUIRED)

function(phydrax_mmg_library output name)
  if(TARGET Mmg::lib${name}_so)
    set(${output} Mmg::lib${name}_so PARENT_SCOPE)
  elseif(TARGET Mmg::lib${name}_a)
    set(${output} Mmg::lib${name}_a PARENT_SCOPE)
  else()
    message(FATAL_ERROR "The Mmg installation does not provide lib${name}.")
  endif()
endfunction()

phydrax_mmg_library(PHYDRAX_MMG3D mmg3d)

if(PHYDRAX_MMG_WITH_PARMMG)
  find_package(MPI REQUIRED COMPONENTS CXX)
  find_package(ParMmg CONFIG REQUIRED)
  set(PHYDRAX_WORKER phydrax-parmmg-worker)
  add_executable(${PHYDRAX_WORKER} worker.cpp)
  target_compile_definitions(
    ${PHYDRAX_WORKER} PRIVATE PHYDRAX_MMG_WITH_PARMMG PHYDRAX_WORKER_WITH_MPI
  )
  target_link_libraries(
    ${PHYDRAX_WORKER} PRIVATE ParMmg::ParMmg ${PHYDRAX_MMG3D} MPI::MPI_CXX
  )
else()
  phydrax_mmg_library(PHYDRAX_MMG2D mmg2d)
  phydrax_mmg_library(PHYDRAX_MMGS mmgs)
  set(PHYDRAX_WORKER phydrax-mmg-worker)
  add_executable(${PHYDRAX_WORKER} worker.cpp)
  target_link_libraries(
    ${PHYDRAX_WORKER} PRIVATE ${PHYDRAX_MMG2D} ${PHYDRAX_MMGS} ${PHYDRAX_MMG3D}
  )
endif()

target_include_directories(
  ${PHYDRAX_WORKER} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common"
)
target_compile_features(${PHYDRAX_WORKER} PRIVATE cxx_std_17)
set_target_properties(
  ${PHYDRAX_WORKER} PROPERTIES CXX_EXTENSIONS OFF INSTALL_RPATH_USE_LINK_PATH ON
)

install(TARGETS ${PHYDRAX_WORKER} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
