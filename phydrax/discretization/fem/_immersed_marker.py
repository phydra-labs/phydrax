#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import AbstractVectorSpace, DenseLinearOperator
from .._lagrangian_marker import (
    LagrangianMarkerDiscretization,
    LagrangianMarkerKinematics,
)


class FiniteElementImmersedMarkerMapPlan(StrictModule, NonTrainableState):
    """Fixed FE-coordinate interpolation into active Lagrangian marker values."""

    markers: LagrangianMarkerDiscretization
    configuration_space: AbstractVectorSpace
    interpolation_matrix: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        markers: LagrangianMarkerDiscretization,
        configuration_space: AbstractVectorSpace,
        interpolation_matrix: ArrayLike,
        /,
    ):
        if not isinstance(markers, LagrangianMarkerDiscretization):
            raise TypeError("markers must be LagrangianMarkerDiscretization.")
        if not isinstance(configuration_space, AbstractVectorSpace):
            raise TypeError("configuration_space must be AbstractVectorSpace.")
        matrix = np.asarray(interpolation_matrix)
        expected = (
            markers.active_velocity_space.size,
            configuration_space.size,
        )
        if matrix.shape != expected:
            raise ValueError(f"interpolation_matrix must have shape {expected}.")
        if not np.issubdtype(matrix.dtype, np.inexact):
            matrix = matrix.astype(float)
        if np.any(~np.isfinite(matrix)):
            raise ValueError("interpolation_matrix must be finite.")
        self.markers = markers
        self.configuration_space = configuration_space
        self.interpolation_matrix = jnp.asarray(matrix)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-immersed-marker-map-plan",
                "markers": markers.prepared_id,
                "configuration_space": configuration_space.space_id,
                "interpolation": array_tree_fingerprint(matrix),
            }
        )

    def prepare(self, /) -> PreparedFiniteElementImmersedMarkerMap:
        return PreparedFiniteElementImmersedMarkerMap(self)


class PreparedFiniteElementImmersedMarkerMap(StrictModule, NonTrainableState):
    """Prepared H/H* map between FE configuration and marker velocity spaces."""

    markers: LagrangianMarkerDiscretization
    configuration_space: AbstractVectorSpace
    interpolation: DenseLinearOperator
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: FiniteElementImmersedMarkerMapPlan, /):
        if not isinstance(plan, FiniteElementImmersedMarkerMapPlan):
            raise TypeError("plan must be FiniteElementImmersedMarkerMapPlan.")
        interpolation = DenseLinearOperator(
            plan.interpolation_matrix,
            source=plan.configuration_space,
            target=plan.markers.active_velocity_space,
            operator_id=f"finite-element-marker-H/{plan.plan_id}",
        )
        self.markers = plan.markers
        self.configuration_space = plan.configuration_space
        self.interpolation = interpolation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-element-immersed-marker-map",
                "plan": plan.plan_id,
                "interpolation": interpolation.operator_id,
            }
        )

    def active_position(self, configuration: PyTree, /):
        return self.interpolation.mv(configuration)

    def active_velocity(self, velocity: PyTree, /):
        return self.interpolation.mv(velocity)

    def kinematics(
        self, configuration: PyTree, velocity: PyTree, /
    ) -> LagrangianMarkerKinematics:
        active_position = self.active_position(configuration)
        active_velocity = self.active_velocity(velocity)
        return self.markers.kinematics(
            self.markers.expand_active(active_position),
            self.markers.expand_active(active_velocity),
        )

    def structural_load(self, marker_force_density: ArrayLike, /):
        values = self.markers.active_velocity_space.validate(
            jnp.asarray(marker_force_density)
        )
        return self.interpolation.adjoint_mv(values)


__all__ = [
    "FiniteElementImmersedMarkerMapPlan",
    "PreparedFiniteElementImmersedMarkerMap",
]
