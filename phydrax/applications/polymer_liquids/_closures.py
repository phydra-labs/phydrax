#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class PRISMClosureKind(StrEnum):
    HNC = "hnc"
    PERCUS_YEVICK = "percus-yevick"
    MEAN_SPHERICAL = "mean-spherical"
    MARTYNOV_SARKISOV = "martynov-sarkisov"


class PRISMClosurePlan(StrictModule, NonTrainableState):
    kind: PRISMClosureKind = eqx.field(static=True)
    hard_core_diameters: Array | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: PRISMClosureKind,
        /,
        *,
        hard_core_diameters: ArrayLike | None = None,
    ):
        if not isinstance(kind, PRISMClosureKind):
            raise TypeError("kind must be PRISMClosureKind.")
        diameters = None
        if hard_core_diameters is not None:
            host = np.asarray(hard_core_diameters, dtype=np.float64)
            if (
                host.ndim != 2
                or host.shape[0] != host.shape[1]
                or np.any(~np.isfinite(host))
                or np.any(host <= 0.0)
                or not np.allclose(host, host.T)
            ):
                raise ValueError(
                    "Hard-core diameters must be positive symmetric matrices."
                )
            diameters = jnp.asarray(host)
        if kind is PRISMClosureKind.MEAN_SPHERICAL and diameters is None:
            raise ValueError("The mean-spherical closure requires hard-core diameters.")
        self.kind = kind
        self.hard_core_diameters = diameters
        self.plan_id = canonical_fingerprint(
            {
                "kind": "prism-closure-plan",
                "closure": kind.value,
                "hard_core_diameters": (
                    None
                    if diameters is None
                    else array_tree_fingerprint(np.asarray(diameters))
                ),
            }
        )


class PRISMClosureEvaluation(StrictModule):
    direct_correlation: Array
    radial_distribution: Array
    core_mask: Array
    domain_margin: Array
    successful: Array
    closure_id: str = eqx.field(static=True)


def evaluate_prism_closure(
    plan: PRISMClosurePlan,
    radii: ArrayLike,
    gamma: ArrayLike,
    beta_potential: ArrayLike,
    /,
) -> PRISMClosureEvaluation:
    if not isinstance(plan, PRISMClosurePlan):
        raise TypeError("plan must be PRISMClosurePlan.")
    radial = jnp.asarray(radii)
    indirect = jnp.asarray(gamma)
    potential = jnp.asarray(beta_potential, dtype=indirect.dtype)
    if (
        radial.ndim != 1
        or indirect.ndim != 3
        or indirect.shape[0] != indirect.shape[1]
        or indirect.shape[-1] != radial.size
        or potential.shape != indirect.shape
    ):
        raise ValueError("Closure radii, gamma, and potential shapes are incompatible.")
    if plan.hard_core_diameters is None:
        core = jnp.zeros_like(indirect, dtype=jnp.bool_)
    else:
        if plan.hard_core_diameters.shape != indirect.shape[:2]:
            raise ValueError("Hard-core diameter matrix does not match the site count.")
        core = radial[None, None, :] < plan.hard_core_diameters[:, :, None]
    if plan.kind is PRISMClosureKind.HNC:
        exponent = -potential + indirect
        outside_correlation = jnp.exp(exponent) - 1.0 - indirect
        margin = jnp.asarray(jnp.inf, dtype=indirect.dtype)
        domain = jnp.all(jnp.isfinite(exponent))
    elif plan.kind is PRISMClosureKind.PERCUS_YEVICK:
        boltzmann_minus_one = jnp.exp(-potential) - 1.0
        outside_correlation = boltzmann_minus_one * (1.0 + indirect)
        margin = jnp.asarray(jnp.inf, dtype=indirect.dtype)
        domain = jnp.all(jnp.isfinite(boltzmann_minus_one))
    elif plan.kind is PRISMClosureKind.MEAN_SPHERICAL:
        outside_correlation = -potential
        margin = jnp.asarray(jnp.inf, dtype=indirect.dtype)
        domain = jnp.asarray(True)
    else:
        radicand = 1.0 + 2.0 * indirect
        margin = jnp.min(radicand)
        safe_radicand = jnp.where(radicand > 0.0, radicand, 1.0)
        bridge = jnp.sqrt(safe_radicand) - 1.0
        outside_correlation = jnp.exp(-potential + bridge) - 1.0 - indirect
        domain = jnp.all(radicand > 0.0)
    direct = jnp.where(core, -1.0 - indirect, outside_correlation)
    total = direct + indirect
    radial_distribution = 1.0 + total
    successful = (
        domain
        & jnp.all(jnp.isfinite(indirect))
        & jnp.all(jnp.isfinite(potential))
        & jnp.all(jnp.isfinite(direct))
        & jnp.all(jnp.where(core, radial_distribution == 0.0, True))
    )
    return PRISMClosureEvaluation(
        direct,
        radial_distribution,
        core,
        margin,
        successful,
        plan.plan_id,
    )


__all__ = [
    "PRISMClosureEvaluation",
    "PRISMClosureKind",
    "PRISMClosurePlan",
    "evaluate_prism_closure",
]
