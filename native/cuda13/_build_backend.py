"""Build CUDA companion wheels with scikit-build-core; never publish an sdist."""

from __future__ import annotations

from typing import NoReturn

from scikit_build_core.build import (
    build_editable,
    build_wheel,
    get_requires_for_build_editable,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_editable,
    prepare_metadata_for_build_wheel,
)


__all__ = [
    "build_editable",
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_editable",
    "prepare_metadata_for_build_wheel",
]


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, str | list[str]] | None = None,
) -> NoReturn:
    """Refuse source distributions: CUDA companions are distributed as wheels only."""
    raise RuntimeError("CUDA companions do not support sdists; build with --wheel.")


def get_requires_for_build_sdist(
    config_settings: dict[str, str | list[str]] | None = None,
) -> NoReturn:
    """Refuse sdist requests before a frontend installs further build dependencies."""
    raise RuntimeError("CUDA companions do not support sdists; build with --wheel.")
