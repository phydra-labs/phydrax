# Assemble an isolated, writable toolkit view; never mutate NVIDIA installations.
set(PHX_CUDA_TOOLKIT "${CMAKE_BINARY_DIR}/pip-cuda" CACHE PATH "NVIDIA wheel toolkit view")
set(PHX_CUDA_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloads" CACHE PATH "Verified source download cache")
file(MAKE_DIRECTORY "${PHX_CUDA_DOWNLOAD_DIR}")
set(PHX_CUDA_TOOLCHAIN_PINS "{}")
if(NOT Python_EXECUTABLE)
  message(FATAL_ERROR "Python_EXECUTABLE must name the CUDA build environment")
endif()
if(CMAKE_HOST_WIN32)
  include(PhxWindowsCudaToolkit)
  return()
endif()
set(_phx_discover [=[
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import sys

root = Path(sys.argv[1])
architecture = platform.machine()
if sys.platform != 'linux' or architecture not in ('x86_64', 'aarch64'):
    raise RuntimeError(f'CUDA wheel builds require Linux x86_64 or aarch64: {sys.platform}/{architecture}')
root.mkdir(parents=True, exist_ok=True)
libdirs = set()
wheels = {}
site = None

def link(source, relative):
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
    if name.startswith('nvidia-'):
        package_roots = ('nvidia',)
    elif re.fullmatch(r'cutensor-cu\d+', name):
        package_roots = ('cutensor',)
    elif re.fullmatch(r'cutensornet-cu\d+', name):
        package_roots = ('cuquantum',)
    else:
        continue
    suffix = re.search(r'-cu(\d+)$', name)
    if suffix is not None and suffix.group(1) != '13':
        raise RuntimeError(f'Foreign NVIDIA distribution in build environment: {name}')
    files = distribution.files or ()
    selected = []
    for record in files:
        parts = record.parts
        if not parts or parts[0] not in package_roots:
            continue
        if any(re.fullmatch(r'cu\d+', part) and part != 'cu13' for part in parts):
            raise RuntimeError(f'Foreign CUDA toolkit record: {name}: {record}')
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
origin = site / 'phydrax_cuda13' / 'lib'
runpath = ['$ORIGIN/' + os.path.relpath(directory, origin) for directory in sorted(libdirs)]
(root / 'discovery.json').write_text(json.dumps({'host_architecture': architecture, 'wheels': wheels, 'library_dirs': [str(p) for p in sorted(libdirs)], 'runpath': runpath}, sort_keys=True, indent=2) + '\n')
print(';'.join(runpath))
]=])
execute_process(COMMAND "${Python_EXECUTABLE}" -c "${_phx_discover}" "${PHX_CUDA_TOOLKIT}"
  RESULT_VARIABLE _phx_result OUTPUT_VARIABLE PHX_CUDA_RUNPATH OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _phx_result EQUAL 0)
  message(FATAL_ERROR "Cannot assemble CUDA 13 toolkit from ${Python_EXECUTABLE}")
endif()
if(NOT EXISTS "${PHX_CUDA_TOOLKIT}/lib64")
  file(CREATE_LINK lib "${PHX_CUDA_TOOLKIT}/lib64" SYMBOLIC)
endif()
# NVIDIA wheels omit driver stubs and cuobjdump. Fetch pinned redistributables
# for link-time stubs and inspection only, never their runtime libraries.
# CPU-only manylinux builders need no installed GPU driver or system toolkit.
# Pins from NVIDIA's redistrib_13.0.2.json; aarch64 uses the SBSA archives,
# not Jetson/tegra artifacts. Each archive is verified before extraction.
file(READ "${PHX_CUDA_TOOLKIT}/discovery.json" _phx_discovery)
string(JSON _phx_host_architecture GET "${_phx_discovery}" host_architecture)
if(_phx_host_architecture STREQUAL "x86_64")
  set(_phx_redist_platform linux-x86_64)
  set(_phx_support_specs
    "cuda_cudart|13.0.96|25b8071951baba827be1580b841d363464f6ee6c39f48d33a81646f90cc95ed1"
    "cuda_nvml_dev|13.0.87|92d441fa7ae41bba92e06c928cacdb32f4703763512f68d10c9c230f09ffe6d8"
    "cuda_cuobjdump|13.0.85|bd624dcb2089842add8f293efab1d21aa076c98b36d8dcb64347fe08fb03315d")
elseif(_phx_host_architecture STREQUAL "aarch64")
  set(_phx_redist_platform linux-sbsa)
  set(_phx_support_specs
    "cuda_cudart|13.0.96|e2c78564330a09d890f98a066a2082a32c78256ffcc6bccca1696bd2abbaf445"
    "cuda_nvml_dev|13.0.87|71740823e837c2a1a6e314f0fe801f8fa6aef1055220e9a504f6f4c67e21d72b"
    "cuda_cuobjdump|13.0.85|cd0304f70ff905421da2ca0fe53cfc384baa48d6ff6b34f92a80cf723064e827")
else()
  message(FATAL_ERROR "Unsupported CUDA build host architecture: ${_phx_host_architecture}")
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
  set(stem "${family}-${_phx_redist_platform}-${version}-archive")
  set(url "https://developer.download.nvidia.com/compute/cuda/redist/${family}/${_phx_redist_platform}/${stem}.tar.xz")
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
if(NOT _phx_nvcc_version MATCHES "release 13\\.")
  message(FATAL_ERROR "Expected CUDA 13 nvcc: ${_phx_nvcc_version}")
endif()
