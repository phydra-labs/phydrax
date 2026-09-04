#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable geometric, wave, transport, and guided-coupling optics."""

from . import (
    beamlets as beamlets,
    geometric as geometric,
    materials as materials,
    sbs as sbs,
    transport as transport,
    wave as wave,
)
from .beamlets import *  # noqa: F403
from .beamlets import __all__ as _beamlets_all
from .geometric import *  # noqa: F403
from .geometric import __all__ as _geometric_all
from .materials import *  # noqa: F403
from .materials import __all__ as _materials_all
from .sbs import *  # noqa: F403
from .sbs import __all__ as _sbs_all
from .transport import *  # noqa: F403
from .transport import __all__ as _transport_all
from .wave import *  # noqa: F403
from .wave import __all__ as _wave_all


__all__ = [
    "beamlets",
    "geometric",
    "materials",
    "sbs",
    "transport",
    "wave",
]
__all__ += [
    name
    for name in (
        *_beamlets_all,
        *_geometric_all,
        *_materials_all,
        *_sbs_all,
        *_transport_all,
        *_wave_all,
    )
    if name not in __all__
]
