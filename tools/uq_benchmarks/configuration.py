#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


ProfileName = Literal["smoke", "standard"]


@dataclass(frozen=True)
class BenchmarkConfiguration:
    """Numerical fidelity and workload controls shared by all matrix scenarios."""

    profile: ProfileName
    num_chains: int
    num_warmup: int
    num_draws: int
    pathfinder_samples: int
    smc_particles: int
    posterior_prediction_samples: int
    calibration_cases: int
    gp_repeats: int
    jit_warm_repetitions: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


PROFILES: dict[ProfileName, BenchmarkConfiguration] = {
    "smoke": BenchmarkConfiguration(
        profile="smoke",
        num_chains=4,
        num_warmup=100,
        num_draws=150,
        pathfinder_samples=768,
        smc_particles=800,
        posterior_prediction_samples=1_024,
        calibration_cases=256,
        gp_repeats=5,
        jit_warm_repetitions=3,
    ),
    "standard": BenchmarkConfiguration(
        profile="standard",
        num_chains=4,
        num_warmup=500,
        num_draws=1_000,
        pathfinder_samples=4_096,
        smc_particles=4_000,
        posterior_prediction_samples=8_192,
        calibration_cases=2_048,
        gp_repeats=12,
        jit_warm_repetitions=10,
    ),
}


def get_configuration(profile: str, /) -> BenchmarkConfiguration:
    if profile == "smoke":
        return PROFILES["smoke"]
    if profile == "standard":
        return PROFILES["standard"]
    choices = ", ".join(sorted(PROFILES))
    raise ValueError(f"Unknown benchmark profile {profile!r}; choose {choices}.")


__all__ = ["BenchmarkConfiguration", "PROFILES", "ProfileName", "get_configuration"]
