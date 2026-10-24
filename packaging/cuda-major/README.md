# CUDA installed-intent marker

The metadata-only `phydrax-cuda-major` distribution has version `13.0`, built
from `13/`. It contains no Python module or CUDA code. Phydrax's `cuda13` extra
requires `==13.*`, recording explicit installed CUDA intent for diagnostics
even before a CUDA backend is initialized. CUDA 13 is the only supported major;
the marker is not a CUDA runtime or driver.

On native Windows, installed CUDA intent requires the matching numerical and
PJRT/kernel companion wheels. The marker is not a successful CPU-only substitute;
unsupported Windows architectures must fail binary resolution or admission.

Publish the marker wheel to the package index **before** publishing a Phydrax
release that references it. Build it with `uv build --wheel` from `13/`.
Never re-release, replace, or repurpose this marker version.

The marker includes the repository's Phydra Non-Production License and the
required license notice. The release gate `tools/check_cuda_extras.py` builds
the wheel locally and supplies it through `--find-links` before publication.
