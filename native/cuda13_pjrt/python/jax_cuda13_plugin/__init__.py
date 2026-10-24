# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Upstream-required import package containing the actual compiled CUDA kernels.

JAX imports this package while its own library module is only partially loaded.
Only the standalone runtime owner is imported here; no JAX/core import cycle.
"""

try:
    from phydrax_cuda13_pjrt._jax_contract import validate_installed_sources
    from phydrax_cuda13_pjrt.runtime import prepare_kernels
except ImportError as error:
    # JAX treats ImportError as an absent optional kernel package. An installed
    # but damaged runtime is not absence and must not enable CPU fallback.
    raise RuntimeError("The installed Windows CUDA runtime is incomplete") from error

validate_installed_sources()
prepare_kernels()
