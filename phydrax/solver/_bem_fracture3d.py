#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import AbstractLinearOperator


class BEMFractureProblem3D(StrictModule, NonTrainableState):
    """Fixed-crack conforming displacement-jump/contact operator problem."""

    traction_operator: AbstractLinearOperator
    normals: Array
    initial_gap: Array
    friction_coefficient: float = eqx.field(static=True)
    cohesive_strength: float = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        traction_operator: AbstractLinearOperator,
        normals: ArrayLike,
        initial_gap: ArrayLike,
        /,
        *,
        friction_coefficient: float = 0.0,
        cohesive_strength: float = 0.0,
        relaxation: float = 0.1,
        maximum_iterations: int = 100,
    ):
        if not isinstance(traction_operator, AbstractLinearOperator):
            raise TypeError(
                "traction_operator must be a prepared conforming BEM operator."
            )
        normal = np.asarray(normals, dtype=float)
        gap = np.asarray(initial_gap, dtype=float)
        if (
            normal.ndim != 2
            or normal.shape[1] != 3
            or gap.shape != (normal.shape[0],)
            or traction_operator.source.size != 3 * normal.shape[0]
            or traction_operator.target.size != 3 * normal.shape[0]
        ):
            raise ValueError("Fracture operator/normals/gap shapes are incompatible.")
        norms = np.linalg.norm(normal, axis=1)
        if (
            np.any(~np.isfinite(normal))
            or np.any(np.abs(norms - 1.0) > 1e-8)
            or np.any(~np.isfinite(gap))
        ):
            raise ValueError("Fracture normals must be finite unit rows and gaps finite.")
        friction, cohesion, omega = map(
            float, (friction_coefficient, cohesive_strength, relaxation)
        )
        iterations = int(maximum_iterations)
        if friction < 0 or cohesion < 0 or not 0 < omega <= 1 or iterations <= 0:
            raise ValueError(
                "Contact/fracture policy values violate their bounded envelope."
            )
        self.traction_operator = traction_operator
        self.normals = jnp.asarray(normal)
        self.initial_gap = jnp.asarray(gap)
        self.friction_coefficient = friction
        self.cohesive_strength = cohesion
        self.relaxation = omega
        self.maximum_iterations = iterations
        self.problem_id = canonical_fingerprint(
            {
                "kind": "bem-fracture-problem-3d",
                "operator": traction_operator.operator_id,
                "normals": array_tree_fingerprint(normal),
                "gap": array_tree_fingerprint(gap),
                "friction": friction,
                "cohesion": cohesion,
                "iterations": iterations,
            }
        )


class BEMFractureResult3D(StrictModule):
    displacement_jump: Array
    traction: Array
    normal_gap: Array
    normal_complementarity_defect: Array
    friction_cone_defect: Array
    residual_norm: Array
    active_contact: Array
    sticking: Array
    successful: Array
    problem_id: str = eqx.field(static=True)


class PreparedBEMFracture3D(StrictModule, NonTrainableState):
    problem: BEMFractureProblem3D
    prepared_id: str = eqx.field(static=True)

    def solve(
        self,
        applied_traction: ArrayLike,
        /,
        *,
        initial_jump: ArrayLike | None = None,
        tolerance: float = 1e-6,
    ) -> BEMFractureResult3D:
        load = jnp.asarray(applied_traction, dtype=self.problem.normals.dtype)
        count = self.problem.normals.shape[0]
        if load.shape == (count, 3):
            load = load.reshape((-1,))
        if load.shape != (3 * count,):
            raise ValueError(
                "applied_traction must have shape (point_count, 3) or flat equivalent."
            )
        initial = (
            jnp.zeros_like(load)
            if initial_jump is None
            else jnp.asarray(initial_jump, dtype=load.dtype).reshape((-1,))
        )
        if initial.shape != load.shape:
            raise ValueError("initial_jump has incompatible shape.")
        normals = self.problem.normals
        gap0 = self.problem.initial_gap
        omega = self.problem.relaxation
        friction = self.problem.friction_coefficient
        cohesion = self.problem.cohesive_strength

        def iteration(_, jump_flat):
            residual = self.problem.traction_operator.mv(jump_flat) - load
            candidate = jump_flat - omega * residual
            vectors = candidate.reshape((count, 3))
            normal_opening = jnp.sum(vectors * normals, axis=1)
            normal_opening = jnp.maximum(normal_opening, gap0)
            tangential = vectors - jnp.sum(vectors * normals, axis=1)[:, None] * normals
            tangential_norm = jnp.linalg.norm(tangential, axis=1)
            bound = friction * jnp.maximum(normal_opening - gap0, 0.0) + cohesion
            scale = jnp.minimum(
                1.0,
                bound
                / jnp.maximum(tangential_norm, jnp.finfo(tangential_norm.dtype).tiny),
            )
            projected = normal_opening[:, None] * normals + scale[:, None] * tangential
            return projected.reshape((-1,))

        jump = jax.lax.fori_loop(0, self.problem.maximum_iterations, iteration, initial)
        traction = (self.problem.traction_operator.mv(jump) - load).reshape((count, 3))
        opening = jump.reshape((count, 3))
        normal_gap = jnp.sum(opening * normals, axis=1) - gap0
        normal_traction = jnp.sum(traction * normals, axis=1)
        tangential_traction = traction - normal_traction[:, None] * normals
        complementarity = jnp.max(jnp.abs(jnp.minimum(normal_gap, normal_traction)))
        cone = jnp.max(
            jnp.maximum(
                jnp.linalg.norm(tangential_traction, axis=1)
                - friction * jnp.maximum(normal_traction, 0.0)
                - cohesion,
                0.0,
            )
        )
        residual = self.problem.traction_operator.mv(jump) - load
        residual_norm = jnp.linalg.norm(residual)
        tol = jnp.asarray(tolerance, dtype=residual_norm.dtype)
        successful = (
            jnp.all(jnp.isfinite(jump)) & (complementarity <= tol) & (cone <= tol)
        )
        active = normal_gap <= tol
        sticking = (
            jnp.linalg.norm(tangential_traction, axis=1)
            < friction * jnp.maximum(normal_traction, 0.0) + cohesion - tol
        )
        return BEMFractureResult3D(
            jump.reshape((count, 3)),
            traction,
            normal_gap,
            complementarity,
            cone,
            residual_norm,
            active,
            sticking,
            successful,
            self.problem.problem_id,
        )


class BEMFractureEpochTransition3D(StrictModule, NonTrainableState):
    source: PreparedBEMFracture3D
    target: PreparedBEMFracture3D
    source_to_target: Array
    differentiable: bool = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    def transfer_jump(self, source_jump: ArrayLike, /) -> Array:
        values = jnp.asarray(source_jump).reshape((-1, 3))
        return values[self.source_to_target]


def prepare_bem_fracture_3d(problem: BEMFractureProblem3D, /) -> PreparedBEMFracture3D:
    if not isinstance(problem, BEMFractureProblem3D):
        raise TypeError("problem must be BEMFractureProblem3D.")
    return PreparedBEMFracture3D(
        problem,
        canonical_fingerprint(
            {"kind": "prepared-bem-fracture-3d", "problem": problem.problem_id}
        ),
    )


def advance_bem_fracture_3d(
    source: PreparedBEMFracture3D,
    target: PreparedBEMFracture3D,
    source_to_target: ArrayLike,
    /,
) -> BEMFractureEpochTransition3D:
    routes = np.asarray(source_to_target, dtype=np.int32)
    target_count = target.problem.normals.shape[0]
    source_count = source.problem.normals.shape[0]
    if (
        routes.shape != (target_count,)
        or np.any(routes < 0)
        or np.any(routes >= source_count)
    ):
        raise ValueError("Fracture growth requires explicit complete parent routes.")
    transition_id = canonical_fingerprint(
        {
            "kind": "bem-fracture-epoch-transition-3d",
            "source": source.prepared_id,
            "target": target.prepared_id,
            "routes": array_tree_fingerprint(routes),
            "differentiable": False,
        }
    )
    return BEMFractureEpochTransition3D(
        source, target, jnp.asarray(routes), False, transition_id
    )


__all__ = [
    "BEMFractureEpochTransition3D",
    "BEMFractureProblem3D",
    "BEMFractureResult3D",
    "PreparedBEMFracture3D",
    "advance_bem_fracture_3d",
    "prepare_bem_fracture_3d",
]
