function(phx_cuda_add_component name)
  cmake_parse_arguments(PHX "" "" "SOURCES;LINK_LIBRARIES;HANDLERS" ${ARGN})
  if(PHX_UNPARSED_ARGUMENTS OR NOT PHX_SOURCES)
    message(FATAL_ERROR "phx_cuda_add_component(${name}) requires SOURCES and accepts LINK_LIBRARIES/HANDLERS")
  endif()
  set(target "phydrax_cuda_${name}")
  add_library(${target} SHARED ${PHX_SOURCES} "${PHX_CUDA_SOURCE_DIR}/src/common/identity.cpp")
  target_link_libraries(${target} PRIVATE phx_cuda_common phx_cuda_ffi CUDA::cudart ${PHX_LINK_LIBRARIES})
  target_compile_features(${target} PRIVATE cxx_std_17 cuda_std_17)
  target_compile_definitions(${target} PRIVATE
    PHX_CUDA_COMPONENT="${name}" PHX_CUDA_VERSION="${PHX_CUDA_VERSION}"
    PHX_CUDA_BUILD_HASH="${PHX_CUDA_BUILD_HASH}" PHX_CUDA_MAJOR=13)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_RUNTIME_LIBRARY Shared)
  if(WIN32)
    target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/EHsc;/bigobj;/utf-8>)
    target_link_options(${target} PRIVATE "LINKER:/MAP:${CMAKE_BINARY_DIR}/${target}.map")
    install(TARGETS ${target} RUNTIME DESTINATION "phydrax_cuda13/lib")
    install(FILES "${CMAKE_BINARY_DIR}/${target}.map" DESTINATION "phydrax_cuda13/lib")
  else()
    target_link_options(${target} PRIVATE
      "LINKER:--version-script=${PHX_CUDA_SOURCE_DIR}/src/common/exports.map"
      "LINKER:--exclude-libs,ALL" "LINKER:-z,defs" "LINKER:--enable-new-dtags")
    set_target_properties(${target} PROPERTIES
      CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
      BUILD_WITH_INSTALL_RPATH ON BUILD_RPATH "${PHX_CUDA_RUNPATH}" INSTALL_RPATH "${PHX_CUDA_RUNPATH}"
      INSTALL_RPATH_USE_LINK_PATH OFF
      LINK_DEPENDS "${PHX_CUDA_SOURCE_DIR}/src/common/exports.map")
    install(TARGETS ${target} LIBRARY DESTINATION "phydrax_cuda13/lib")
  endif()
  set_property(GLOBAL APPEND PROPERTY PHX_COMPONENT_MANIFEST_ENTRIES
    "\"${name}\":\"lib/$<TARGET_FILE_NAME:${target}>\"")
endfunction()
