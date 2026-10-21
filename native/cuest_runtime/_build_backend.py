"""Package the pinned cuEST runtime as Linux wheels, without compiling an SDK."""

from __future__ import annotations

import platform
import sys
import sysconfig
from typing import NoReturn

from scikit_build_core import build as _build


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

type ConfigSettings = dict[str, str | list[str]] | None


def _require_supported_host() -> None:
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("phydrax-cuest supports only Linux x86_64 and aarch64.")
    if sys.version_info[:2] != (3, 13) or sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise RuntimeError("phydrax-cuest requires standard, GIL-enabled Python 3.13.")


def build_wheel(
    wheel_directory: str,
    config_settings: ConfigSettings = None,
    metadata_directory: str | None = None,
) -> str:
    """Build a platform wheel containing only the verified binary redistributable."""
    _require_supported_host()
    return _build.build_wheel(wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_wheel(config_settings: ConfigSettings = None) -> list[str]:
    """Refuse unsupported hosts before requesting additional build dependencies."""
    _require_supported_host()
    return _build.get_requires_for_build_wheel(config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: ConfigSettings = None,
) -> str:
    """Prepare inert dependency metadata without downloading or loading cuEST."""
    _require_supported_host()
    return _build.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_sdist(
    sdist_directory: str,
    config_settings: ConfigSettings = None,
) -> NoReturn:
    """Refuse source distribution of the private application runtime."""
    raise RuntimeError("phydrax-cuest supports wheels only; build with --wheel.")


def get_requires_for_build_sdist(config_settings: ConfigSettings = None) -> NoReturn:
    """Refuse sdist requests before requesting additional dependencies."""
    raise RuntimeError("phydrax-cuest supports wheels only; build with --wheel.")


def build_editable(
    wheel_directory: str,
    config_settings: ConfigSettings = None,
    metadata_directory: str | None = None,
) -> NoReturn:
    """Refuse editable installs, which do not own a complete runtime RECORD."""
    raise RuntimeError(
        "phydrax-cuest does not support editable installs; install a wheel."
    )


def get_requires_for_build_editable(config_settings: ConfigSettings = None) -> NoReturn:
    """Refuse editable requests before requesting additional dependencies."""
    raise RuntimeError(
        "phydrax-cuest does not support editable installs; install a wheel."
    )


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: ConfigSettings = None,
) -> NoReturn:
    """Refuse metadata for unsupported editable installations."""
    raise RuntimeError(
        "phydrax-cuest does not support editable installs; install a wheel."
    )
