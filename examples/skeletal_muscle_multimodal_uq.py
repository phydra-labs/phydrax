#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.personalization import (
    SkeletalMultimodalLikelihoodPlan,
    SkeletalObservationChannel,
)
from phydrax.uq import ParameterSpace


def main() -> None:
    force = SkeletalObservationChannel(
        "force", "observed_force", "example-load-cell",
        jnp.asarray((100.0, 120.0, 140.0)), 2.0, jnp.asarray((True, True, True))
    )
    emg = SkeletalObservationChannel(
        "surface-emg", "surface_electric_potential", "example-electrodes",
        jnp.asarray((1.0e-4, -2.0e-5, 5.0e-5)), 1.0e-5,
        jnp.asarray((True, False, True))
    )
    likelihood = SkeletalMultimodalLikelihoodPlan((force, emg))
    base_force = jnp.asarray((100.0, 120.0, 140.0))
    base_emg = jnp.asarray((1.0e-4, -2.0e-5, 5.0e-5))
    space = ParameterSpace(
        {"force_scale": jnp.asarray(1.0)},
        log_prior=lambda value: -0.5 * ((value["force_scale"] - 1.0) / 0.2) ** 2,
    )
    posterior = likelihood.posterior(
        space,
        {
            "force": lambda value: value["force_scale"] * base_force,
            "surface-emg": lambda value: base_emg,
        },
    )
    log_density, gradient = posterior.validate()
    payload = {
        "plan_id": likelihood.plan_id,
        "channel_ids": list(likelihood.channel_ids),
        "log_density": float(log_density),
        "force_scale_gradient": float(gradient["force_scale"]),
        "claim_scope": "fixed observed modalities assembled as core UQ terms",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
