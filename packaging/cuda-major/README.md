# CUDA-major exclusivity markers

The metadata-only `phydrax-cuda-major` distribution has versions `12.0` and
`13.0`, built from `12/` and `13/`. It contains no Python module or CUDA code.
Phydrax's `cuda12` and `cuda13` extras require `==12.*` and `==13.*`, respectively.
These incompatible requirements enforce exclusivity in published wheel metadata
for pip and uv on every platform, independently of development-only uv settings.

Publish both marker wheels to the package index **before** publishing any Phydrax
release that references them. Build each with `uv build --wheel` from its marker
directory. Never re-release, replace, or repurpose either marker version; their
meaning is permanent. Future CUDA majors require a new marker version.

Each marker includes the repository's Phydra Non-Production License and the
required license notice. The release gate `tools/check_cuda_extras.py` builds
these wheels locally and supplies them through `--find-links` before publication.
