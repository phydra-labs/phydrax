#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..finite_difference._mapped_grid import PreparedMappedTensorGrid
from ._discretization import LatticeBoltzmannDiscretization
from ._geometry import LatticeBoltzmannGeometryKind


MappedKineticSource = Callable[[Array, Array, PreparedMappedTensorGrid, Any], Array]


class MappedLatticeBoltzmannEvidence(StrictModule):
    minimum_jacobian: Array
    maximum_jacobian: Array
    metric_identity_residual: Array
    free_stream_residual: Array
    source_mass_residual: Array
    source_momentum_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MappedLatticeBoltzmannStepResult(StrictModule):
    populations: Array
    evidence: MappedLatticeBoltzmannEvidence
    successful: Array


class MappedLatticeBoltzmannPlan(StrictModule, NonTrainableState):
    """Reference-lattice streaming plus an explicit curvilinear kinetic source."""

    reference: LatticeBoltzmannDiscretization
    mapped_grid: PreparedMappedTensorGrid
    metric_source: MappedKineticSource
    source_id: str = eqx.field(static=True)
    metric_tolerance: float = eqx.field(static=True)
    geometry_kind: LatticeBoltzmannGeometryKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference: LatticeBoltzmannDiscretization,
        mapped_grid: PreparedMappedTensorGrid,
        metric_source: MappedKineticSource,
        /,
        *,
        source_id: str,
        metric_tolerance: float = 1.0e-8,
    ):
        if not isinstance(reference, LatticeBoltzmannDiscretization):
            raise TypeError("reference must be LatticeBoltzmannDiscretization.")
        if not isinstance(mapped_grid, PreparedMappedTensorGrid):
            raise TypeError("mapped_grid must be PreparedMappedTensorGrid.")
        if mapped_grid.reference_grid.shape != reference.grid.shape:
            raise ValueError("Mapped and LBM reference grids must have equal shapes.")
        mapped_points = np.asarray(mapped_grid.reference_grid.points)
        lattice_points = np.asarray(reference.grid.points)
        tolerance = 128.0 * np.finfo(lattice_points.dtype).eps
        if not np.allclose(mapped_points, lattice_points, rtol=tolerance, atol=tolerance):
            raise ValueError(
                "Mapped metric nodes must coincide with the LBM cell centres."
            )
        if not callable(metric_source):
            raise TypeError("metric_source must be callable.")
        identifier = str(source_id)
        tolerance = float(metric_tolerance)
        if not identifier or not jnp.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Mapped source identity and tolerance are invalid.")
        self.reference = reference
        self.mapped_grid = mapped_grid
        self.metric_source = metric_source
        self.source_id = identifier
        self.metric_tolerance = tolerance
        self.geometry_kind = LatticeBoltzmannGeometryKind.MAPPED
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-lattice-boltzmann-plan",
                "reference": reference.prepared_id,
                "mapped": mapped_grid.prepared_id,
                "source": identifier,
                "metric_tolerance": tolerance,
            }
        )

    def advance(
        self,
        time: Array,
        populations: Array,
        reference_step: Callable[[Array, Any], Array],
        /,
        *,
        args: Any = None,
    ) -> MappedLatticeBoltzmannStepResult:
        values = self.reference.validate_populations(populations)
        streamed_collided = reference_step(values, args)
        source = jnp.asarray(
            self.metric_source(time, values, self.mapped_grid, args),
            dtype=values.dtype,
        )
        if source.shape != values.shape:
            raise ValueError("Mapped kinetic source must match population shape.")
        candidate = streamed_collided + source
        velocities = jnp.asarray(
            self.reference.velocity_set.velocities, dtype=values.dtype
        )
        source_mass = jnp.max(jnp.abs(jnp.sum(source, axis=-1)))
        source_momentum = jnp.max(
            jnp.abs(oe.contract("...q,qd->...d", source, velocities))
        )
        metric = self.mapped_grid.metric_report
        minimum_jacobian = jnp.min(self.mapped_grid.jacobian)
        maximum_jacobian = jnp.max(self.mapped_grid.jacobian)
        successful = (
            jnp.all(jnp.isfinite(candidate))
            & (minimum_jacobian > 0.0)
            & (metric.metric_identity_residual <= self.metric_tolerance)
            & (metric.free_stream_residual <= self.metric_tolerance)
            & (source_mass <= self.metric_tolerance)
            & (source_momentum <= self.metric_tolerance)
        )
        accepted = jnp.where(successful, candidate, values)
        evidence = MappedLatticeBoltzmannEvidence(
            minimum_jacobian,
            maximum_jacobian,
            metric.metric_identity_residual,
            metric.free_stream_residual,
            source_mass,
            source_momentum,
            successful,
            self.plan_id,
        )
        return MappedLatticeBoltzmannStepResult(accepted, evidence, successful)


__all__ = [
    "MappedKineticSource",
    "MappedLatticeBoltzmannEvidence",
    "MappedLatticeBoltzmannPlan",
    "MappedLatticeBoltzmannStepResult",
]
