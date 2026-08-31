#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CochainDiscretization


class UnstructuredMagneticState(StrictModule):
    face_flux: Array
    time: Array


class UnstructuredFaradayDiagnostics(StrictModule):
    constraint_before: Array
    constraint_after: Array
    constraint_change: Array
    successful: Array


class UnstructuredConstrainedTransportPlan(StrictModule, NonTrainableState):
    """Topology-exact Faraday update on an arbitrary prepared cochain complex."""

    cochain: CochainDiscretization
    spatial_dimension: int = eqx.field(static=True)
    magnetic_degree: int = eqx.field(static=True)
    electromotive_degree: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cochain: CochainDiscretization,
        spatial_dimension: int,
        /,
    ):
        dimension = int(spatial_dimension)
        if (
            not isinstance(cochain, CochainDiscretization)
            or dimension not in (2, 3)
            or cochain.max_degree < dimension
        ):
            raise ValueError("Unstructured constrained transport topology is invalid.")
        self.cochain = cochain
        self.spatial_dimension = dimension
        self.magnetic_degree = dimension - 1
        self.electromotive_degree = dimension - 2
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-constrained-transport",
                "cochain": cochain.prepared_id,
                "spatial_dimension": dimension,
            }
        )

    def initialize(
        self,
        face_flux: ArrayLike,
        time: ArrayLike = 0.0,
        /,
    ) -> UnstructuredMagneticState:
        magnetic = jnp.asarray(face_flux)
        expected = self.cochain.cell_counts[self.magnetic_degree]
        if magnetic.shape != (expected,):
            raise ValueError("Unstructured magnetic flux shape is invalid.")
        constraint = self.cochain.exterior_derivative(self.magnetic_degree, magnetic)
        magnetic = eqx.error_if(
            magnetic,
            jnp.max(jnp.abs(constraint), initial=0.0) > 1e-10,
            "Initial unstructured magnetic flux violates the cochain constraint.",
        )
        return UnstructuredMagneticState(
            magnetic,
            jnp.asarray(time, dtype=magnetic.dtype).reshape(()),
        )

    def advance(
        self,
        state: UnstructuredMagneticState,
        edge_electromotive: ArrayLike,
        end_time: ArrayLike,
        /,
    ) -> tuple[UnstructuredMagneticState, UnstructuredFaradayDiagnostics]:
        electromotive = jnp.asarray(edge_electromotive)
        expected = self.cochain.cell_counts[self.electromotive_degree]
        if electromotive.shape != (expected,):
            raise ValueError("Unstructured electromotive cochain shape is invalid.")
        end = jnp.asarray(end_time, dtype=state.time.dtype).reshape(())
        step = end - state.time
        rate = -self.cochain.exterior_derivative(self.electromotive_degree, electromotive)
        candidate = state.face_flux + step * rate
        before = self.cochain.exterior_derivative(self.magnetic_degree, state.face_flux)
        after = self.cochain.exterior_derivative(self.magnetic_degree, candidate)
        change = jnp.max(jnp.abs(after - before), initial=0.0)
        successful = (
            jnp.isfinite(step)
            & (step > 0.0)
            & jnp.all(jnp.isfinite(candidate))
            & (change <= 1e-10)
        )
        accepted = UnstructuredMagneticState(
            jnp.where(successful, candidate, state.face_flux),
            jnp.where(successful, end, state.time),
        )
        diagnostics = UnstructuredFaradayDiagnostics(
            constraint_before=before,
            constraint_after=after,
            constraint_change=change,
            successful=successful,
        )
        return accepted, diagnostics


__all__ = [
    "UnstructuredConstrainedTransportPlan",
    "UnstructuredFaradayDiagnostics",
    "UnstructuredMagneticState",
]
