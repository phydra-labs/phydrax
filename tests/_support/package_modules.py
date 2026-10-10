"""Import every Phydrax module that the installed optional providers can load."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Mapping

import phydrax as phx
from phydrax.atomistic.interchange import is_ase_available


# Provider modules that import an optional dependency at load time. Their
# public facades resolve them lazily behind the provider's availability gate, so
# a missing provider leaves exactly these modules unimported.
_OPTIONAL_PROVIDER_MODULES: Mapping[str, Callable[[], bool]] = {
    "phydrax.atomistic.interchange._ase_calculator": is_ase_available,
}


def import_phydrax_modules() -> None:
    """Import every `phydrax` module whose optional provider is installed."""
    for info in pkgutil.walk_packages(phx.__path__, "phydrax."):
        available = _OPTIONAL_PROVIDER_MODULES.get(info.name)
        if available is None or available():
            importlib.import_module(info.name)
