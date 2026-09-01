#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import ArraySpace
from ....nonlinear import (
    implicit_root_result,
    NewtonKrylov,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._blocks import AbstractMemberBlock, MemberBlockEvaluation


class CatenaryRegime(IntEnum):
    SLACK = 0
    CATENARY = 1
    NEAR_STRAIGHT = 2
    NEAR_VERTICAL = 3
    ZERO_DISTRIBUTED_LOAD = 4
    INADMISSIBLE = 5


class ElasticCatenaryReference(StrictModule, NonTrainableState):
    unstretched_length: Array
    axial_rigidity: Array
    distributed_load: Array
    thermal_strain: Array
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        unstretched_length: ArrayLike,
        axial_rigidity: ArrayLike,
        distributed_load: ArrayLike,
        /,
        *,
        thermal_strain: ArrayLike = 0.0,
        reference_id: str | None = None,
    ):
        length = jnp.asarray(unstretched_length)
        rigidity = jnp.asarray(axial_rigidity, dtype=length.dtype)
        load = jnp.asarray(distributed_load, dtype=length.dtype)
        thermal = jnp.asarray(thermal_strain, dtype=length.dtype)
        if length.shape != () or rigidity.shape != () or thermal.shape != ():
            raise ValueError("Catenary scalar reference values must be scalar.")
        if load.shape != (3,):
            raise ValueError(
                "distributed_load must be a three-vector per reference length."
            )
        if bool(
            ~jnp.isfinite(length)
            | ~jnp.isfinite(rigidity)
            | (length <= 0.0)
            | (rigidity <= 0.0)
            | jnp.any(~jnp.isfinite(load))
            | ~jnp.isfinite(thermal)
        ):
            raise ValueError("Catenary reference values are inadmissible.")
        self.unstretched_length = length
        self.axial_rigidity = rigidity
        self.distributed_load = load
        self.thermal_strain = thermal
        self.reference_id = str(
            reference_id
            or canonical_fingerprint(
                {
                    "kind": "elastic-catenary-reference",
                    "length": float(length),
                    "rigidity": float(rigidity),
                    "load": load.tolist(),
                    "thermal": float(thermal),
                }
            )
        )


class CatenarySolvePolicy(StrictModule, NonTrainableState):
    quadrature_order: int = eqx.field(static=True)
    absolute_residual: float = eqx.field(static=True)
    relative_residual: float = eqx.field(static=True)
    minimum_tension: float = eqx.field(static=True)
    straight_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        quadrature_order: int = 16,
        absolute_residual: float = 1.0e-10,
        relative_residual: float = 1.0e-10,
        minimum_tension: float = 1.0e-10,
        straight_tolerance: float = 1.0e-6,
    ):
        if int(quadrature_order) < 4:
            raise ValueError("quadrature_order must be at least four.")
        values = (
            float(absolute_residual),
            float(relative_residual),
            float(minimum_tension),
            float(straight_tolerance),
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Catenary tolerances must be finite and positive.")
        self.quadrature_order = int(quadrature_order)
        (
            self.absolute_residual,
            self.relative_residual,
            self.minimum_tension,
            self.straight_tolerance,
        ) = values


class ElasticCatenaryState(StrictModule):
    start_force: Array
    end_force: Array
    chord: Array
    centerline: Array
    quadrature_tension: Array
    minimum_tension: Array
    strained_length: Array
    sag: Array
    strain_energy: Array
    load_potential: Array
    regime: Array
    nonlinear_result: NonlinearResult | None
    valid: Array


class ElasticCatenaryBlock(AbstractMemberBlock):
    references: tuple[ElasticCatenaryReference, ...]
    policy: CatenarySolvePolicy

    def __init__(
        self,
        member_indices: ArrayLike,
        references: tuple[ElasticCatenaryReference, ...],
        /,
        *,
        policy: CatenarySolvePolicy | None = None,
        block_id: str | None = None,
    ):
        indices = jnp.asarray(member_indices, dtype=jnp.int32)
        references_ = tuple(references)
        if indices.ndim != 1 or len(references_) != indices.size:
            raise ValueError("Catenary references must match member indices.")
        self.member_indices = indices
        self.references = references_
        self.policy = CatenarySolvePolicy() if policy is None else policy
        self.block_id = str(block_id or "elastic-catenary-block")

    def evaluate(self, definition, kinematics, /):
        states = []
        for member, reference in zip(
            np.asarray(self.member_indices), self.references, strict=True
        ):
            sender = int(np.asarray(definition.structure.senders[member]))
            receiver = int(np.asarray(definition.structure.receivers[member]))
            states.append(
                solve_elastic_catenary(
                    kinematics.positions[sender],
                    kinematics.positions[receiver],
                    reference,
                    policy=self.policy,
                )
            )
        energy = jnp.sum(
            jnp.stack(
                tuple(state.strain_energy + state.load_potential for state in states)
            )
        )
        start = jnp.stack(tuple(state.start_force for state in states))
        end = jnp.stack(tuple(state.end_force for state in states))
        axial = 0.5 * (
            jnp.sqrt(jnp.sum(start * start, axis=-1))
            + jnp.sqrt(jnp.sum(end * end, axis=-1))
        )
        valid = jnp.all(jnp.stack(tuple(state.valid for state in states)))
        count = self.member_indices.size
        return MemberBlockEvaluation(
            energy,
            self.member_indices,
            axial,
            jnp.zeros((count, 2), dtype=energy.dtype),
            jnp.zeros((count, 2), dtype=energy.dtype),
            jnp.zeros((count,), dtype=energy.dtype),
            jnp.ones((count,), dtype=bool),
            jnp.ones((count,), dtype=bool),
            jnp.stack(tuple(state.minimum_tension for state in states)),
            valid,
        )


def _quadrature(policy: CatenarySolvePolicy, dtype) -> tuple[Array, Array]:
    points, weights = np.polynomial.legendre.leggauss(policy.quadrature_order)
    return jnp.asarray(points, dtype=dtype), jnp.asarray(weights, dtype=dtype)


def _integrated_geometry(
    start_force: Array,
    reference: ElasticCatenaryReference,
    policy: CatenarySolvePolicy,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    points, weights = _quadrature(policy, start_force.dtype)
    length = reference.unstretched_length * (1.0 + reference.thermal_strain)
    material_coordinate = 0.5 * length * (points + 1.0)
    force = (
        start_force[None, :] - material_coordinate[:, None] * reference.distributed_load
    )
    tension = jnp.sqrt(jnp.sum(force * force, axis=-1))
    unit = force / jnp.maximum(tension[:, None], policy.minimum_tension)
    tangent = unit + force / reference.axial_rigidity
    weighted = 0.5 * length * weights
    displacement = jnp.sum(weighted[:, None] * tangent, axis=0)
    coordinates = jnp.cumsum(weighted[:, None] * tangent, axis=0)
    strained_length = jnp.sum(weighted * (1.0 + tension / reference.axial_rigidity))
    strain_energy = jnp.sum(weighted * tension**2 / (2.0 * reference.axial_rigidity))
    return displacement, coordinates, tension, strained_length, strain_energy


def solve_elastic_catenary(
    start: ArrayLike,
    end: ArrayLike,
    reference: ElasticCatenaryReference,
    /,
    *,
    policy: CatenarySolvePolicy | None = None,
) -> ElasticCatenaryState:
    """Solve extensible catenary endpoint equilibrium with implicit derivatives."""
    policy_ = CatenarySolvePolicy() if policy is None else policy
    start_ = jnp.asarray(start)
    end_ = jnp.asarray(end, dtype=start_.dtype)
    if start_.shape != (3,) or end_.shape != (3,):
        raise ValueError("Catenary endpoints must be three-vectors.")
    chord = end_ - start_
    chord_length = jnp.sqrt(jnp.dot(chord, chord))
    load_norm = jnp.sqrt(jnp.dot(reference.distributed_load, reference.distributed_load))
    effective_length = reference.unstretched_length * (1.0 + reference.thermal_strain)
    if bool(
        (load_norm <= policy_.straight_tolerance) & (chord_length <= effective_length)
    ):
        centerline = jnp.stack((start_, end_))
        zero = jnp.zeros((3,), dtype=start_.dtype)
        return ElasticCatenaryState(
            zero,
            zero,
            chord,
            centerline,
            jnp.zeros((policy_.quadrature_order,), dtype=start_.dtype),
            jnp.asarray(0.0, dtype=start_.dtype),
            chord_length,
            jnp.asarray(0.0, dtype=start_.dtype),
            jnp.asarray(0.0, dtype=start_.dtype),
            jnp.asarray(0.0, dtype=start_.dtype),
            jnp.asarray(int(CatenaryRegime.SLACK), dtype=jnp.int32),
            None,
            jnp.asarray(True),
        )
    unit_chord = chord / chord_length
    elastic_force = reference.axial_rigidity * jnp.maximum(
        chord_length / effective_length - 1.0, policy_.minimum_tension
    )
    initial = (
        elastic_force * unit_chord + 0.5 * effective_length * reference.distributed_load
    )

    def residual(force, args):
        del args
        displacement, _, _, _, _ = _integrated_geometry(force, reference, policy_)
        return displacement - chord

    problem = NonlinearSystemProblem(
        residual,
        state_space=ArraySpace((3,), dtype=start_.dtype),
        problem_id=f"{reference.reference_id}:catenary-root",
    )
    solved = implicit_root_result(
        problem,
        initial,
        method=NewtonKrylov(),
        termination=NonlinearTermination(
            absolute_residual=policy_.absolute_residual,
            relative_residual=policy_.relative_residual,
            maximum_steps=100,
        ),
    )
    displacement, coordinates, tension, strained_length, strain_energy = (
        _integrated_geometry(solved.state, reference, policy_)
    )
    del displacement
    centerline = jnp.concatenate((start_[None, :], start_[None, :] + coordinates), axis=0)
    end_force = -(solved.state - effective_length * reference.distributed_load)
    chord_unit = chord / chord_length
    transverse = (
        centerline
        - start_[None, :]
        - (
            jnp.sum((centerline - start_[None, :]) * chord_unit, axis=-1)[:, None]
            * chord_unit
        )
    )
    sag = jnp.max(jnp.sqrt(jnp.sum(transverse * transverse, axis=-1)))
    load_potential = (
        -jnp.sum(reference.distributed_load[None, :] * centerline[1:])
        * effective_length
        / policy_.quadrature_order
    )
    minimum = jnp.min(tension)
    near_vertical = (
        jnp.abs(jnp.dot(unit_chord, reference.distributed_load))
        >= (1.0 - policy_.straight_tolerance) * load_norm
    )
    near_straight = sag <= policy_.straight_tolerance * chord_length
    regime = jnp.where(
        load_norm <= policy_.straight_tolerance,
        int(CatenaryRegime.ZERO_DISTRIBUTED_LOAD),
        jnp.where(
            near_vertical,
            int(CatenaryRegime.NEAR_VERTICAL),
            jnp.where(
                near_straight,
                int(CatenaryRegime.NEAR_STRAIGHT),
                int(CatenaryRegime.CATENARY),
            ),
        ),
    ).astype(jnp.int32)
    valid = solved.successful & (minimum > policy_.minimum_tension)
    return ElasticCatenaryState(
        solved.state,
        end_force,
        chord,
        centerline,
        tension,
        minimum,
        strained_length,
        sag,
        strain_energy,
        load_potential,
        regime,
        solved,
        valid,
    )


__all__ = [
    "CatenaryRegime",
    "CatenarySolvePolicy",
    "ElasticCatenaryBlock",
    "ElasticCatenaryReference",
    "ElasticCatenaryState",
    "solve_elastic_catenary",
]
