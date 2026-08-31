#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from .._force_density import ForceDensityResult
from .._force_density_stability import analyze_force_density_mechanisms
from ._blocks import AbstractAxialLaw, LinearAxialLaw, MemberNetworkAssembly
from ._equilibrium import MemberNetworkInputs
from ._properties import MemberPropertyMap
from ._reference import (
    MemberDOFLayout,
    MemberKinematics,
    MemberNetworkDefinition,
    MemberReferenceState,
)


class StructuralEvidenceVerdict(IntEnum):
    CERTIFIED = 0
    FAILED = 1
    INCOMPLETE = 2


class PrestressTarget(StrictModule):
    """Force-density geometry, target forces, and applied-load equilibrium."""

    positions: Array
    axial_forces: Array
    applied_loads: Array
    support_reactions: Array
    source_problem_id: str = eqx.field(static=True)

    @classmethod
    def from_force_density(cls, result: ForceDensityResult, /) -> PrestressTarget:
        if not isinstance(result, ForceDensityResult):
            raise TypeError("result must be a ForceDensityResult.")
        return cls(
            result.state.positions,
            result.state.axial_forces,
            result.state.applied_nodal_loads,
            result.state.support_reactions,
            result.provenance.problem_id,
        )


class PrestressFabricationPolicy(StrictModule):
    """Rest-length, actuator-stroke, and group compatibility requirements."""

    minimum_rest_length: Array
    maximum_rest_length: Array
    minimum_stroke: Array
    maximum_stroke: Array
    group_tolerance: float = eqx.field(static=True)
    force_tolerance: float = eqx.field(static=True)
    require_stability: bool = eqx.field(static=True)
    require_sequence: bool = eqx.field(static=True)

    def __init__(
        self,
        minimum_rest_length: ArrayLike,
        maximum_rest_length: ArrayLike,
        minimum_stroke: ArrayLike,
        maximum_stroke: ArrayLike,
        /,
        *,
        group_tolerance: float = 1.0e-6,
        force_tolerance: float = 1.0e-6,
        require_stability: bool = True,
        require_sequence: bool = True,
    ):
        minimum = jnp.asarray(minimum_rest_length)
        maximum = jnp.asarray(maximum_rest_length, dtype=minimum.dtype)
        stroke_minimum = jnp.asarray(minimum_stroke, dtype=minimum.dtype)
        stroke_maximum = jnp.asarray(maximum_stroke, dtype=minimum.dtype)
        if not (
            minimum.shape == maximum.shape == stroke_minimum.shape == stroke_maximum.shape
        ):
            raise ValueError("Prestress fabrication arrays must share one shape.")
        if bool(
            jnp.any(~jnp.isfinite(minimum))
            | jnp.any(~jnp.isfinite(maximum))
            | jnp.any(~jnp.isfinite(stroke_minimum))
            | jnp.any(~jnp.isfinite(stroke_maximum))
            | jnp.any(minimum <= 0.0)
            | jnp.any(maximum < minimum)
            | jnp.any(stroke_maximum < stroke_minimum)
        ):
            raise ValueError("Prestress fabrication bounds are inadmissible.")
        tolerances = (float(group_tolerance), float(force_tolerance))
        if any(not isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError("Prestress tolerances must be finite and positive.")
        self.minimum_rest_length = minimum
        self.maximum_rest_length = maximum
        self.minimum_stroke = stroke_minimum
        self.maximum_stroke = stroke_maximum
        self.group_tolerance, self.force_tolerance = tolerances
        self.require_stability = bool(require_stability)
        self.require_sequence = bool(require_sequence)


class PrestressRealizabilityResult(StrictModule):
    """Constitutive, fabrication, actuation, equilibrium, and stability evidence."""

    verdict: Array
    rest_lengths: Array
    reconstructed_forces: Array
    force_error: Array
    self_stress_component: Array
    load_carrying_component: Array
    actuator_stroke: Array
    rest_length_margin: Array
    stroke_margin: Array
    group_deviation: Array
    constitutive_valid: Array
    fabrication_valid: Array
    actuator_valid: Array
    equilibrium_valid: Array
    sign_valid: Array
    stability_valid: Array
    sequence_valid: Array
    complete: Array

    @property
    def successful(self) -> Array:
        return self.verdict == int(StructuralEvidenceVerdict.CERTIFIED)


def _member_lengths(target: PrestressTarget, structure, /) -> Array:
    vectors = target.positions[structure.receivers] - target.positions[structure.senders]
    return jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))


def assess_prestress_realizability(
    target: PrestressTarget,
    definition: MemberNetworkDefinition,
    axial_law: AbstractAxialLaw,
    fabrication: PrestressFabricationPolicy,
    /,
    *,
    stability_valid: ArrayLike | None = None,
    sequence_valid: ArrayLike | None = None,
    member_roles: Literal["bilateral", "tension-only", "compression-only"] = "bilateral",
) -> PrestressRealizabilityResult:
    """Invert stress-free lengths and aggregate all declared realizability evidence."""
    structure = definition.structure
    count = structure.member_count
    for name, value in (
        ("minimum_rest_length", fabrication.minimum_rest_length),
        ("maximum_rest_length", fabrication.maximum_rest_length),
        ("minimum_stroke", fabrication.minimum_stroke),
        ("maximum_stroke", fabrication.maximum_stroke),
    ):
        if value.shape != (count,):
            raise ValueError(f"{name} must match the member count.")
    properties = definition.properties.structural_arrays()
    lengths = _member_lengths(target, structure)
    rigidity = properties["young"] * properties["area"]
    denominator = rigidity + target.axial_forces
    rest_lengths = axial_law.inverse_rest_length(lengths, target.axial_forces, rigidity)
    reconstructed = rigidity * (lengths - rest_lengths) / rest_lengths
    force_error = reconstructed - target.axial_forces
    constitutive_valid = jnp.all(
        jnp.isfinite(rest_lengths)
        & (rest_lengths > 0.0)
        & (denominator > 0.0)
        & (jnp.abs(force_error) <= fabrication.force_tolerance)
    )
    sign_valid = jnp.asarray(True)
    if member_roles == "tension-only":
        sign_valid = jnp.all(target.axial_forces >= 0.0)
    elif member_roles == "compression-only":
        sign_valid = jnp.all(target.axial_forces <= 0.0)
    elif member_roles != "bilateral":
        raise ValueError("Unknown prestress member role.")
    lower_margin = rest_lengths - fabrication.minimum_rest_length
    upper_margin = fabrication.maximum_rest_length - rest_lengths
    rest_margin = jnp.minimum(lower_margin, upper_margin)
    fabrication_valid = jnp.all(rest_margin >= 0.0)
    stroke = lengths - rest_lengths
    stroke_margin = jnp.minimum(
        stroke - fabrication.minimum_stroke,
        fabrication.maximum_stroke - stroke,
    )
    actuator_valid = jnp.all(stroke_margin >= 0.0)
    groups = np.asarray(definition.properties.fabrication_group)
    group_deviation = jnp.zeros_like(rest_lengths)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        mean = jnp.mean(rest_lengths[indices])
        group_deviation = group_deviation.at[indices].set(
            jnp.abs(rest_lengths[indices] - mean)
        )
    fabrication_valid = fabrication_valid & jnp.all(
        group_deviation <= fabrication.group_tolerance
    )
    mechanisms = analyze_force_density_mechanisms(structure, target.positions)
    modes = mechanisms.self_stress_modes
    mask = mechanisms.self_stress_mask.astype(modes.dtype)
    masked_modes = modes * mask[None, :]
    self_stress = masked_modes @ (
        jnp.swapaxes(masked_modes, -1, -2) @ target.axial_forces
    )
    load_carrying = target.axial_forces - self_stress
    equilibrium_residual = (
        mechanisms.rigidity_matrix.T @ target.axial_forces
        - structure.reduce(target.applied_loads)
    )
    equilibrium_valid = (
        jnp.sqrt(jnp.sum(equilibrium_residual**2)) <= fabrication.force_tolerance
    )
    stability = (
        jnp.asarray(False)
        if stability_valid is None
        else jnp.asarray(stability_valid, dtype=bool)
    )
    sequence = (
        jnp.asarray(False)
        if sequence_valid is None
        else jnp.asarray(sequence_valid, dtype=bool)
    )
    complete = jnp.asarray(
        (not fabrication.require_stability or stability_valid is not None)
        and (not fabrication.require_sequence or sequence_valid is not None)
    )
    required_stability = jnp.asarray(not fabrication.require_stability) | stability
    required_sequence = jnp.asarray(not fabrication.require_sequence) | sequence
    certified = (
        constitutive_valid
        & fabrication_valid
        & actuator_valid
        & equilibrium_valid
        & sign_valid
        & required_stability
        & required_sequence
        & complete
    )
    failed = (
        ~constitutive_valid
        | ~fabrication_valid
        | ~actuator_valid
        | ~equilibrium_valid
        | ~sign_valid
        | (fabrication.require_stability & (stability_valid is not None) & ~stability)
        | (fabrication.require_sequence & (sequence_valid is not None) & ~sequence)
    )
    verdict = jnp.where(
        certified,
        int(StructuralEvidenceVerdict.CERTIFIED),
        jnp.where(
            failed,
            int(StructuralEvidenceVerdict.FAILED),
            int(StructuralEvidenceVerdict.INCOMPLETE),
        ),
    ).astype(jnp.int32)
    return PrestressRealizabilityResult(
        verdict,
        rest_lengths,
        reconstructed,
        force_error,
        self_stress,
        load_carrying,
        stroke,
        rest_margin,
        stroke_margin,
        group_deviation,
        constitutive_valid,
        fabrication_valid,
        actuator_valid,
        equilibrium_valid,
        sign_valid,
        stability,
        sequence,
        complete,
    )


def member_network_from_force_density(
    result: ForceDensityResult,
    structure,
    properties: MemberPropertyMap,
    assembly: MemberNetworkAssembly,
    /,
    *,
    rest_length_mode: Literal[
        "stress-free-current", "force-compatible"
    ] = "force-compatible",
    axial_law: AbstractAxialLaw | None = None,
    constrain_rotations: bool = True,
) -> tuple[
    PrestressTarget,
    MemberNetworkDefinition,
    MemberNetworkInputs,
    MemberKinematics,
]:
    """Build one constitutive member-network seed from force-density evidence."""
    target = PrestressTarget.from_force_density(result)
    law = LinearAxialLaw() if axial_law is None else axial_law
    lengths = _member_lengths(target, structure)
    if rest_length_mode == "stress-free-current":
        rest_lengths = lengths
    elif rest_length_mode == "force-compatible":
        arrays = properties.structural_arrays()
        rest_lengths = law.inverse_rest_length(
            lengths,
            target.axial_forces,
            arrays["young"] * arrays["area"],
        )
    else:
        raise ValueError("Unknown rest_length_mode.")
    reference = MemberReferenceState(
        structure,
        target.positions,
        rest_lengths=rest_lengths,
    )
    rotation_dimension = 1 if structure.dimension == 2 else 3
    rotation_constraints = jnp.full(
        (structure.node_count, rotation_dimension), constrain_rotations
    )
    dofs = MemberDOFLayout(structure, rotation_constrained=rotation_constraints)
    definition = MemberNetworkDefinition(structure, reference, properties, dofs)
    kinematics = MemberKinematics(target.positions, reference.rotation_vectors)
    inputs = MemberNetworkInputs(
        structure.prescribed_values(target.positions),
        dofs.prescribed_rotations(reference.rotation_vectors),
        target.applied_loads,
        jnp.zeros(
            (structure.node_count, rotation_dimension), dtype=target.positions.dtype
        ),
        rest_lengths,
    )
    return target, definition, inputs, kinematics


__all__ = [
    "PrestressFabricationPolicy",
    "PrestressRealizabilityResult",
    "PrestressTarget",
    "StructuralEvidenceVerdict",
    "assess_prestress_realizability",
    "member_network_from_force_density",
]
