#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._evolution import AbstractDifferentiableEvolution
from ._linearization import EvolutionJacobianAction


class EvolutionSensitivityStatus(IntEnum):
    SUCCESS = 0
    PRIMAL_INVALID = 1
    TANGENT_INVALID = 2
    PERTURBATION_INVALID = 3
    REFERENCE_INVALID = 4
    NONFINITE = 5
    JVP_MISMATCH = 6
    DUALITY_MISMATCH = 7
    REFERENCE_MISMATCH = 8


class EvolutionSensitivityPolicy(StrictModule, NonTrainableState):
    """Perturbation and acceptance policy for one evolution linearization audit."""

    perturbations: tuple[float, float] = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        perturbations: tuple[float, float] = (1.0e-4, 5.0e-5),
        /,
        *,
        relative_tolerance: float = 2.0e-4,
        absolute_tolerance: float = 1.0e-8,
    ):
        steps = tuple(float(value) for value in perturbations)
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance)
        if (
            len(steps) != 2
            or any(not isfinite(value) or value <= 0.0 for value in steps)
            or not steps[0] > steps[1]
        ):
            raise ValueError(
                "perturbations must contain two finite positive decreasing values."
            )
        if not isfinite(relative) or relative < 0.0:
            raise ValueError("relative_tolerance must be finite and nonnegative.")
        if not isfinite(absolute) or absolute < 0.0:
            raise ValueError("absolute_tolerance must be finite and nonnegative.")
        if relative == 0.0 and absolute == 0.0:
            raise ValueError("At least one sensitivity tolerance must be positive.")
        self.perturbations = steps
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.policy_id = canonical_fingerprint(
            {
                "kind": "evolution-sensitivity-policy",
                "perturbations": list(steps),
                "relative_tolerance": relative,
                "absolute_tolerance": absolute,
            }
        )


class EvolutionSensitivityEvidence(StrictModule, NonTrainableState):
    """Numerical consistency evidence for one local evolution derivative."""

    jvp_defects: Array
    taylor_remainders: Array
    taylor_refinement_ratio: Array
    duality_defect: Array
    reference_tangent_defect: Array
    perturbed_valid: Array
    primal_valid: Array
    tangent_valid: Array
    finite: Array
    valid: Array
    status: Array
    policy_id: str = eqx.field(static=True)
    evolution_id: str = eqx.field(static=True)
    tangent_method_id: str = eqx.field(static=True)
    reference_evolution_id: str | None = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def _maximum_absolute(value: Array, /) -> Array:
    return jnp.max(jnp.abs(value), initial=0.0)


def certify_evolution_sensitivity(
    evolution: AbstractDifferentiableEvolution,
    state: ArrayLike,
    direction: ArrayLike,
    cotangent: ArrayLike,
    source_coordinate: ArrayLike,
    target_coordinate: ArrayLike,
    /,
    *,
    args: Any = None,
    policy: EvolutionSensitivityPolicy | None = None,
    reference_evolution: AbstractDifferentiableEvolution | None = None,
) -> EvolutionSensitivityEvidence:
    """Audit JVP, transpose duality, Taylor behavior, and optional refinement."""
    if not isinstance(evolution, AbstractDifferentiableEvolution):
        raise TypeError("evolution must be an AbstractDifferentiableEvolution.")
    if not evolution.state_layout.geometry.trivial:
        raise ValueError(
            "Evolution sensitivity certification initially requires Euclidean geometry."
        )
    selected = EvolutionSensitivityPolicy() if policy is None else policy
    if not isinstance(selected, EvolutionSensitivityPolicy):
        raise TypeError("policy must be EvolutionSensitivityPolicy or None.")
    value = jnp.asarray(state)
    tangent_direction = jnp.asarray(direction)
    dual = jnp.asarray(cotangent)
    expected = evolution.state_layout.shape
    if (
        value.shape != expected
        or tangent_direction.shape != expected
        or dual.shape != expected
    ):
        raise ValueError("state, direction, and cotangent must match the state layout.")
    source = jnp.asarray(source_coordinate)
    target = jnp.asarray(target_coordinate)
    if source.shape != () or target.shape != ():
        raise ValueError("Evolution segment coordinates must be scalar.")

    primal = evolution.advance(value, source, target, args)
    tangent = evolution.tangent_action(
        value,
        tangent_direction,
        source,
        target,
        args,
    )
    action = EvolutionJacobianAction(
        evolution,
        value,
        source,
        target,
        args=args,
    )
    transposed = action.transpose_mv(dual)

    defects = []
    remainders = []
    perturbation_valid = []
    for epsilon in selected.perturbations:
        plus = evolution.advance(
            value + epsilon * tangent_direction,
            source,
            target,
            args,
        )
        minus = evolution.advance(
            value - epsilon * tangent_direction,
            source,
            target,
            args,
        )
        finite_difference = (plus.final_state - minus.final_state) / (2.0 * epsilon)
        defects.append(_maximum_absolute(tangent.tangent - finite_difference))
        remainders.append(
            _maximum_absolute(
                plus.final_state - primal.final_state - epsilon * tangent.tangent
            )
        )
        perturbation_valid.append(plus.valid & minus.valid)
    jvp_defects = jnp.stack(tuple(defects))
    taylor_remainders = jnp.stack(tuple(remainders))
    tiny = jnp.finfo(taylor_remainders.dtype).tiny
    taylor_ratio = taylor_remainders[1] / jnp.maximum(taylor_remainders[0], tiny)
    duality_forward = jnp.vdot(dual, tangent.tangent)
    duality_reverse = jnp.vdot(transposed, tangent_direction)
    duality = jnp.abs(duality_forward - duality_reverse)
    perturbed_valid = jnp.stack(tuple(perturbation_valid))

    if reference_evolution is None:
        reference_defect = jnp.asarray(0.0, dtype=value.real.dtype)
        reference_valid = jnp.asarray(True)
        reference_id = None
    else:
        if not isinstance(reference_evolution, AbstractDifferentiableEvolution):
            raise TypeError(
                "reference_evolution must be AbstractDifferentiableEvolution or None."
            )
        if (
            reference_evolution.system.system_id != evolution.system.system_id
            or reference_evolution.state_layout.layout_id
            != evolution.state_layout.layout_id
        ):
            raise ValueError(
                "Reference evolution must preserve the system and state layout."
            )
        reference = reference_evolution.tangent_action(
            value,
            tangent_direction,
            source,
            target,
            args,
        )
        reference_defect = _maximum_absolute(reference.tangent - tangent.tangent)
        reference_valid = reference.valid
        reference_id = reference_evolution.evolution_id

    jvp_scale = jnp.maximum(1.0, _maximum_absolute(tangent.tangent))
    jvp_threshold = selected.absolute_tolerance + selected.relative_tolerance * jvp_scale
    duality_scale = jnp.maximum(
        1.0,
        jnp.maximum(jnp.abs(duality_forward), jnp.abs(duality_reverse)),
    )
    duality_threshold = (
        selected.absolute_tolerance + selected.relative_tolerance * duality_scale
    )
    reference_scale = jnp.maximum(
        1.0,
        _maximum_absolute(tangent.tangent),
    )
    reference_threshold = (
        selected.absolute_tolerance + selected.relative_tolerance * reference_scale
    )
    perturbations_valid = jnp.all(perturbed_valid)
    finite = (
        jnp.all(jnp.isfinite(primal.final_state))
        & jnp.all(jnp.isfinite(tangent.tangent))
        & jnp.all(jnp.isfinite(transposed))
        & jnp.all(jnp.isfinite(jvp_defects))
        & jnp.all(jnp.isfinite(taylor_remainders))
        & jnp.isfinite(duality)
        & jnp.isfinite(reference_defect)
    )
    jvp_valid = jnp.all(jvp_defects <= jvp_threshold)
    duality_valid = duality <= duality_threshold
    reference_matches = reference_defect <= reference_threshold
    valid = (
        primal.valid
        & tangent.valid
        & action.primal.valid
        & perturbations_valid
        & reference_valid
        & finite
        & jvp_valid
        & duality_valid
        & reference_matches
    )
    status = jnp.where(
        ~primal.valid,
        int(EvolutionSensitivityStatus.PRIMAL_INVALID),
        jnp.where(
            ~(tangent.valid & action.primal.valid),
            int(EvolutionSensitivityStatus.TANGENT_INVALID),
            jnp.where(
                ~perturbations_valid,
                int(EvolutionSensitivityStatus.PERTURBATION_INVALID),
                jnp.where(
                    ~reference_valid,
                    int(EvolutionSensitivityStatus.REFERENCE_INVALID),
                    jnp.where(
                        ~finite,
                        int(EvolutionSensitivityStatus.NONFINITE),
                        jnp.where(
                            ~jvp_valid,
                            int(EvolutionSensitivityStatus.JVP_MISMATCH),
                            jnp.where(
                                ~duality_valid,
                                int(EvolutionSensitivityStatus.DUALITY_MISMATCH),
                                jnp.where(
                                    ~reference_matches,
                                    int(EvolutionSensitivityStatus.REFERENCE_MISMATCH),
                                    int(EvolutionSensitivityStatus.SUCCESS),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    evidence_id = canonical_fingerprint(
        {
            "kind": "evolution-sensitivity-evidence",
            "policy": selected.policy_id,
            "evolution": evolution.evolution_id,
            "tangent_method": evolution.tangent_method_id,
            "reference_evolution": reference_id,
        }
    )
    return EvolutionSensitivityEvidence(
        jvp_defects=jvp_defects,
        taylor_remainders=taylor_remainders,
        taylor_refinement_ratio=taylor_ratio,
        duality_defect=duality,
        reference_tangent_defect=reference_defect,
        perturbed_valid=perturbed_valid,
        primal_valid=primal.valid,
        tangent_valid=tangent.valid,
        finite=finite,
        valid=valid,
        status=status,
        policy_id=selected.policy_id,
        evolution_id=evolution.evolution_id,
        tangent_method_id=evolution.tangent_method_id,
        reference_evolution_id=reference_id,
        evidence_id=evidence_id,
    )


__all__ = [
    "EvolutionSensitivityEvidence",
    "EvolutionSensitivityPolicy",
    "EvolutionSensitivityStatus",
    "certify_evolution_sensitivity",
]
