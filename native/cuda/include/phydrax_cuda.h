#ifndef PHYDRAX_CUDA_H
#define PHYDRAX_CUDA_H
#include <stdint.h>
#if defined(__GNUC__)
#define PHX_CUDA_EXPORT __attribute__((visibility("default")))
#else
#define PHX_CUDA_EXPORT
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
  PHX_CUDA_SUCCESS = 0, PHX_CUDA_INVALID_ARGUMENT = 1, PHX_CUDA_CUDA_ERROR = 2,
  PHX_CUDA_PROVIDER_ERROR = 3, PHX_CUDA_UNSUPPORTED = 4
} phx_cuda_status;
PHX_CUDA_EXPORT const char* phx_cuda_version(void);
PHX_CUDA_EXPORT const char* phx_cuda_build_hash(void);
PHX_CUDA_EXPORT int32_t phx_cuda_major(void);
PHX_CUDA_EXPORT const char* phx_cuda_component(void);
PHX_CUDA_EXPORT const char* phx_cuda_last_error(void);
PHX_CUDA_EXPORT int32_t phx_cuda_handler_count(void);
PHX_CUDA_EXPORT const char* phx_cuda_handler_name(int32_t index);
PHX_CUDA_EXPORT void* phx_cuda_handler(int32_t index);
PHX_CUDA_EXPORT int32_t phx_cuda_registry_partitions(void);
PHX_CUDA_EXPORT int32_t phx_cuda_registry_partition(int32_t index, int32_t* device,
  int64_t* partition, int64_t* live, int64_t* capacity, int64_t* rebuilds);
PHX_CUDA_EXPORT int32_t phx_cuda_release(int64_t partition, const int32_t* ids,
  int64_t count, int64_t* released);
PHX_CUDA_EXPORT int32_t phx_cuda_release_all(int64_t* released);
PHX_CUDA_EXPORT int32_t phx_cuda_shutdown(void);
#ifdef __cplusplus
}
#endif
#endif
