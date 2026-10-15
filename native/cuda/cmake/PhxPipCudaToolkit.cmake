# Assemble an isolated, writable toolkit view; never mutate NVIDIA installations.
set(PHX_CUDA_TOOLKIT "${CMAKE_BINARY_DIR}/pip-cuda" CACHE PATH "NVIDIA wheel toolkit view")
set(PHX_CUDA_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloads" CACHE PATH "Verified source download cache")
file(MAKE_DIRECTORY "${PHX_CUDA_DOWNLOAD_DIR}")
set(PHX_CUDA_TOOLCHAIN_PINS "{}")
if(NOT Python_EXECUTABLE)
  message(FATAL_ERROR "Python_EXECUTABLE must name the CUDA build environment")
endif()
set(_phx_discover [=[
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import sys

major, root, build = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
root.mkdir(parents=True, exist_ok=True)
libdirs = set()
wheels = {}
site = None

def link(source, relative):
    # CUDA 12 compiler files take precedence over overlapping wheel records.
    # Keep that choice stable during discovery, not just the later overlay.
    if major == 12:
        compiler_root = build / 'cuda-nvcc12' / 'cuda_nvcc-linux-x86_64-12.9.86-archive'
        compiler_file = compiler_root / relative
        if not compiler_file.is_file() and relative.parts[0] == 'lib':
            compiler_file = compiler_root / 'lib64' / Path(*relative.parts[1:])
        if compiler_file.is_file():
            source = compiler_file
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    if destination.exists():
        raise RuntimeError(f"Conflicting toolkit file: {destination}")
    destination.symlink_to(source.resolve())

for distribution in sorted(metadata.distributions(), key=lambda item: item.metadata['Name']):
    name = distribution.metadata['Name'].lower().replace('_', '-')
    if not name.startswith('nvidia-'):
        continue
    files = distribution.files or ()
    selected = []
    for record in files:
        parts = record.parts
        if not parts or parts[0] != 'nvidia':
            continue
        if ('cu12' in parts or name.endswith('-cu12')) and major != 12:
            continue
        if ('cu13' in parts or name.endswith('-cu13')) and major != 13:
            continue
        selected.append(record)
    if not selected:
        continue
    wheels[name] = distribution.version
    for record in selected:
        source = Path(distribution.locate_file(record))
        if not source.is_file():
            continue
        parts = record.parts
        site = Path(distribution.locate_file('')).resolve()
        for kind in ('nvvm', 'include', 'bin', 'lib', 'lib64'):
            if kind not in parts:
                continue
            index = parts.index(kind)
            relative = Path('lib' if kind == 'lib64' else kind, *parts[index + 1:])
            if relative.name.endswith('.py') or '__pycache__' in relative.parts:
                break
            link(source, relative)
            if kind in ('lib', 'lib64') and '.so' in source.name:
                libdirs.add(source.parent.resolve())
                match = re.match(r'^(lib.*\.so)\..+$', source.name)
                if match:
                    development = root / 'lib' / match[1]
                    if not development.exists():
                        development.symlink_to(source.resolve())
            break
if site is None:
    raise RuntimeError('No matching NVIDIA toolkit wheels found')
origin = site / f'phydrax_cuda{major}' / 'lib'
runpath = ['$ORIGIN/' + os.path.relpath(directory, origin) for directory in sorted(libdirs)]
(root / 'discovery.json').write_text(json.dumps({'wheels': wheels, 'library_dirs': [str(p) for p in sorted(libdirs)], 'runpath': runpath}, sort_keys=True, indent=2) + '\n')
print(';'.join(runpath))
]=])
execute_process(COMMAND "${Python_EXECUTABLE}" -c "${_phx_discover}" "${PHX_CUDA_MAJOR}" "${PHX_CUDA_TOOLKIT}" "${CMAKE_BINARY_DIR}"
  RESULT_VARIABLE _phx_result OUTPUT_VARIABLE PHX_CUDA_RUNPATH OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _phx_result EQUAL 0)
  message(FATAL_ERROR "Cannot assemble CUDA ${PHX_CUDA_MAJOR} toolkit from ${Python_EXECUTABLE}")
endif()
if(NOT EXISTS "${PHX_CUDA_TOOLKIT}/lib64")
  file(CREATE_LINK lib "${PHX_CUDA_TOOLKIT}/lib64" SYMBOLIC)
endif()
if(PHX_CUDA_MAJOR STREQUAL "12")
  set(_phx_nvcc_url "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz")
  set(_phx_nvcc_sha "7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35")
  set(_phx_archive "${PHX_CUDA_DOWNLOAD_DIR}/cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz")
  file(LOCK "${PHX_CUDA_DOWNLOAD_DIR}/cuda-nvcc12.lock" GUARD FILE TIMEOUT 600)
  if(NOT EXISTS "${_phx_archive}")
    file(DOWNLOAD "${_phx_nvcc_url}" "${_phx_archive}" EXPECTED_HASH "SHA256=${_phx_nvcc_sha}" TLS_VERIFY ON SHOW_PROGRESS)
  endif()
  file(SHA256 "${_phx_archive}" _phx_actual_sha)
  if(NOT _phx_actual_sha STREQUAL _phx_nvcc_sha)
    message(FATAL_ERROR "CUDA 12 compiler archive SHA256 mismatch")
  endif()
  set(_phx_redist "${CMAKE_BINARY_DIR}/cuda-nvcc12/cuda_nvcc-linux-x86_64-12.9.86-archive")
  if(NOT EXISTS "${_phx_redist}/bin/nvcc")
    file(ARCHIVE_EXTRACT INPUT "${_phx_archive}" DESTINATION "${CMAKE_BINARY_DIR}/cuda-nvcc12")
  endif()
  # Overlay the compiler's own executable-relative files, retaining wheel headers/libraries.
  set(_phx_compiler_overlay [=[
from pathlib import Path
import sys

source, target = map(Path, sys.argv[1:])
for path in source.rglob('*'):
    if not path.is_file():
        continue
    destination = target / path.relative_to(source)
    if destination.is_symlink():
        if destination.resolve() == path.resolve():
            continue
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f'Conflicting compiler toolkit file: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(path.resolve())
]=])
  execute_process(COMMAND "${Python_EXECUTABLE}" -c "${_phx_compiler_overlay}"
    "${_phx_redist}" "${PHX_CUDA_TOOLKIT}" COMMAND_ERROR_IS_FATAL ANY)
  string(JSON PHX_CUDA_TOOLCHAIN_PINS SET "${PHX_CUDA_TOOLCHAIN_PINS}" cuda_nvcc
    "{\"url\":\"${_phx_nvcc_url}\",\"sha256\":\"${_phx_nvcc_sha}\",\"version\":\"12.9.86\"}")
endif()
# NVIDIA wheels omit driver stubs and cuobjdump. Fetch pinned redistributables
# for link-time stubs and inspection only, never their runtime libraries.
# CPU-only manylinux builders need no installed GPU driver or system toolkit.
if(PHX_CUDA_MAJOR STREQUAL "13")
  set(_phx_support_specs
    "cuda_cudart|13.0.96|25b8071951baba827be1580b841d363464f6ee6c39f48d33a81646f90cc95ed1"
    "cuda_nvml_dev|13.0.87|92d441fa7ae41bba92e06c928cacdb32f4703763512f68d10c9c230f09ffe6d8"
    "cuda_cuobjdump|13.0.85|bd624dcb2089842add8f293efab1d21aa076c98b36d8dcb64347fe08fb03315d")
else()
  set(_phx_support_specs
    "cuda_cudart|12.9.79|1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99"
    "cuda_nvml_dev|12.9.79|1ad0866dbfff6a9e2661f5348e4ca4c2a4e40882b90014ab127f2734856ecccb"
    "cuda_cuobjdump|12.9.82|ee0de40e8c18068bfcc53e73510e7e7a1a80555205347940df67fa525d24452f")
endif()
function(_phx_support_link source destination)
  if(IS_SYMLINK "${destination}")
    file(REAL_PATH "${destination}" existing)
    file(REAL_PATH "${source}" desired)
    if(existing STREQUAL desired)
      return()
    endif()
  endif()
  file(CREATE_LINK "${source}" "${destination}" SYMBOLIC)
endfunction()
foreach(spec IN LISTS _phx_support_specs)
  string(REPLACE "|" ";" fields "${spec}")
  list(GET fields 0 family)
  list(GET fields 1 version)
  list(GET fields 2 sha)
  set(stem "${family}-linux-x86_64-${version}-archive")
  set(url "https://developer.download.nvidia.com/compute/cuda/redist/${family}/linux-x86_64/${stem}.tar.xz")
  set(archive "${PHX_CUDA_DOWNLOAD_DIR}/${stem}.tar.xz")
  file(LOCK "${PHX_CUDA_DOWNLOAD_DIR}/${stem}.lock" GUARD FILE TIMEOUT 600)
  if(NOT EXISTS "${archive}")
    file(DOWNLOAD "${url}" "${archive}" EXPECTED_HASH "SHA256=${sha}" TLS_VERIFY ON)
  endif()
  file(SHA256 "${archive}" actual_sha)
  if(NOT actual_sha STREQUAL sha)
    message(FATAL_ERROR "${family} support archive SHA256 mismatch")
  endif()
  set(extracted "${CMAKE_BINARY_DIR}/cuda-support/${stem}")
  if(NOT EXISTS "${extracted}")
    file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${CMAKE_BINARY_DIR}/cuda-support")
  endif()
  file(MAKE_DIRECTORY "${PHX_CUDA_TOOLKIT}/lib/stubs")
  file(GLOB stubs "${extracted}/lib/stubs/*.so")
  foreach(stub IN LISTS stubs)
    get_filename_component(name "${stub}" NAME)
    _phx_support_link("${stub}" "${PHX_CUDA_TOOLKIT}/lib/stubs/${name}")
  endforeach()
  if(family STREQUAL "cuda_cuobjdump")
    file(MAKE_DIRECTORY "${PHX_CUDA_TOOLKIT}/bin")
    _phx_support_link("${extracted}/bin/cuobjdump" "${PHX_CUDA_TOOLKIT}/bin/cuobjdump")
    set(pin_name cuda_cuobjdump)
  else()
    set(pin_name "${family}_driver_stub")
  endif()
  string(JSON PHX_CUDA_TOOLCHAIN_PINS SET "${PHX_CUDA_TOOLCHAIN_PINS}" "${pin_name}"
    "{\"url\":\"${url}\",\"sha256\":\"${sha}\",\"version\":\"${version}\"}")
endforeach()
set(CUDAToolkit_ROOT "${PHX_CUDA_TOOLKIT}" CACHE PATH "Isolated NVIDIA wheel toolkit" FORCE)
set(CMAKE_CUDA_COMPILER "${PHX_CUDA_TOOLKIT}/bin/nvcc" CACHE FILEPATH "NVIDIA wheel compiler" FORCE)
execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" --version OUTPUT_VARIABLE _phx_nvcc_version COMMAND_ERROR_IS_FATAL ANY)
if(NOT _phx_nvcc_version MATCHES "release ${PHX_CUDA_MAJOR}\\.")
  message(FATAL_ERROR "Wrong nvcc major: ${_phx_nvcc_version}")
endif()
