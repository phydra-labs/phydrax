#pragma once
#include "phydrax_cuda.h"
#include "xla/ffi/api/c_api.h"

// These handlers belong to the site-linked cudss_mgmn DSO, not the core cuDSS DSO.
extern "C" PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_mgmn_exchange_f32(XLA_FFI_CallFrame*);
extern "C" PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_mgmn_exchange_f64(XLA_FFI_CallFrame*);
extern "C" PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_mgmn_reduce_f32(XLA_FFI_CallFrame*);
extern "C" PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_mgmn_reduce_f64(XLA_FFI_CallFrame*);
extern "C" PHX_CUDA_EXPORT XLA_FFI_Error* phydrax_cudss_mgmn_reduce_i32(XLA_FFI_CallFrame*);
