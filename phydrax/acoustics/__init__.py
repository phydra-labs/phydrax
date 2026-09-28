#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from importlib import import_module
from typing import Any, TYPE_CHECKING

from ._core import (
    AcousticMedium,
    acoustics_candidate_profiles,
    monopole_pressure,
    normal_incidence_transmission_loss,
    vibroacoustic_power,
)
from ._coupled import VibroacousticResult, VibroacousticSystem
from ._helmholtz import solve_helmholtz


if TYPE_CHECKING:
    from ._bubbly import (
        bubbly_medium_candidate_profiles,
        BubblyMediumDispersionEvidence,
        BubblyMediumDispersionPlan,
        BubblyMediumDispersionResult,
        BubblyMediumDispersionStatus,
        solve_bubbly_medium_dispersion,
        wood_sound_speed,
    )


# `_bubbly` composes `phydrax.bubble_dynamics`, whose solver import chain reaches
# packages that are still initializing when the root package imports this
# facade; its exports resolve on first access.
_BUBBLY_EXPORTS = frozenset(
    {
        "BubblyMediumDispersionEvidence",
        "BubblyMediumDispersionPlan",
        "BubblyMediumDispersionResult",
        "BubblyMediumDispersionStatus",
        "bubbly_medium_candidate_profiles",
        "solve_bubbly_medium_dispersion",
        "wood_sound_speed",
    }
)


def __getattr__(name: str) -> Any:
    if name in _BUBBLY_EXPORTS:
        return import_module(f"{__name__}._bubbly").__dict__[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VibroacousticResult",
    "VibroacousticSystem",
    "AcousticMedium",
    "acoustics_candidate_profiles",
    "monopole_pressure",
    "normal_incidence_transmission_loss",
    "vibroacoustic_power",
    "solve_helmholtz",
    "BubblyMediumDispersionEvidence",
    "BubblyMediumDispersionPlan",
    "BubblyMediumDispersionResult",
    "BubblyMediumDispersionStatus",
    "bubbly_medium_candidate_profiles",
    "solve_bubbly_medium_dispersion",
    "wood_sound_speed",
]
