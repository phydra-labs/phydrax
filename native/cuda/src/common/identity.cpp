#include "phydrax_cuda.h"
extern "C" {
const char* phx_cuda_version() { return PHX_CUDA_VERSION; }
const char* phx_cuda_build_hash() { return PHX_CUDA_BUILD_HASH; }
int32_t phx_cuda_major() { return PHX_CUDA_MAJOR; }
const char* phx_cuda_component() { return PHX_CUDA_COMPONENT; }
}
