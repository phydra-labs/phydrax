from importlib import import_module

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from ._acoustics import __all__ as _acoustics_all
from ._actuator_complete import __all__ as _actuator_complete_all
from ._assimilation import __all__ as _assimilation_all
from ._control_complete import __all__ as _control_complete_all
from ._fsi import __all__ as _fsi_all
from ._learning_complete import __all__ as _learning_complete_all
from ._random_complete import __all__ as _random_complete_all
from ._workflows import (
    actuator_line_sources,
    actuator_surface_sources,
    PassiveVortexProbes,
    PrescribedVortexRigidMotion,
    VortexRigidMotionState,
)


_FACADE_EXPORT_MODULES = (
    "._acoustics",
    "._actuator_complete",
    "._assimilation",
    "._control_complete",
    "._fsi",
    "._learning_complete",
    "._random_complete",
)


def __getattr__(name: str):
    for module_name in reversed(_FACADE_EXPORT_MODULES):
        module = import_module(module_name, __package__)
        if name in module.__all__:
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "PassiveVortexProbes",
    "PrescribedVortexRigidMotion",
    "VortexRigidMotionState",
    "actuator_line_sources",
    "actuator_surface_sources",
]

__all__ += [
    name
    for name in (
        *_acoustics_all,
        *_actuator_complete_all,
        *_assimilation_all,
        *_control_complete_all,
        *_fsi_all,
        *_learning_complete_all,
        *_random_complete_all,
    )
    if name not in __all__
]
