#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._lifecycle import (
    AbstractDiscretizationPlan,
    AbstractPreparedDiscretization,
    validate_prepared_metadata,
)
from .._measure import DiscreteMeasure
from .._spaces import DiscreteFieldSpace
from .._support import DiscreteSupport
from .._tensor_support import PreparedTensorGrid
from ._lattice import LatticeBoltzmannVelocitySet
from ._precision import LatticeBoltzmannPrecisionPolicy


class LatticeBoltzmannPlan(AbstractDiscretizationPlan):
    """Uniform cell-centred support and velocity quadrature for LBM."""

    grid: PreparedTensorGrid
    velocity_set: LatticeBoltzmannVelocitySet
    precision: LatticeBoltzmannPrecisionPolicy
    field_name: str = eqx.field(static=True)
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        velocity_set: LatticeBoltzmannVelocitySet,
        /,
        *,
        precision: LatticeBoltzmannPrecisionPolicy | None = None,
        field_name: str = "populations",
        key: DiscretizationKey | None = None,
        plan_id: str | None = None,
    ):
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("LBM plan requires a PreparedTensorGrid.")
        if not isinstance(velocity_set, LatticeBoltzmannVelocitySet):
            raise TypeError("velocity_set must be a LatticeBoltzmannVelocitySet.")
        velocity_set.require("athermal-hydrodynamics")
        velocity_set.require("nearest-neighbor-streaming")
        precision_ = LatticeBoltzmannPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, LatticeBoltzmannPrecisionPolicy):
            raise TypeError("precision must be a LatticeBoltzmannPrecisionPolicy.")
        field = str(field_name)
        if not field:
            raise ValueError("field_name must be non-empty.")
        key_ = (
            DiscretizationKey(
                "lattice_boltzmann",
                DiscretizationRole.PHYSICAL,
                domain_labels=grid.axis_names,
            )
            if key is None
            else key
        )
        if not isinstance(key_, DiscretizationKey):
            raise TypeError("key must be a DiscretizationKey.")
        capabilities = (
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.MATRIX_FREE,
        )
        generated = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-plan",
                "grid": grid.prepared_id,
                "velocity_set": velocity_set.lattice_id,
                "precision": precision_.policy_id,
                "field_name": field,
                "key": key_.key_id,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.grid = grid
        self.velocity_set = velocity_set
        self.precision = precision_
        self.field_name = field
        self.key = key_
        self.capabilities = capabilities
        self.plan_id = identifier

    def prepare(
        self, /, *, numeric_version: str = "0"
    ) -> "LatticeBoltzmannDiscretization":
        return LatticeBoltzmannDiscretization(self, numeric_version=numeric_version)


class LatticeBoltzmannDiscretization(AbstractPreparedDiscretization):
    """Prepared isotropic cell lattice with trailing population components."""

    plan: LatticeBoltzmannPlan
    grid: PreparedTensorGrid
    velocity_set: LatticeBoltzmannVelocitySet
    precision: LatticeBoltzmannPrecisionPolicy
    population_space: DiscreteFieldSpace
    density_space: DiscreteFieldSpace
    velocity_space: DiscreteFieldSpace
    pressure_space: DiscreteFieldSpace
    cell_size: Array
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    population_shape: tuple[int, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport

    def __init__(self, plan: LatticeBoltzmannPlan, /, *, numeric_version: str = "0"):
        if not isinstance(plan, LatticeBoltzmannPlan):
            raise TypeError("plan must be a LatticeBoltzmannPlan.")
        grid = plan.grid
        dimension = len(grid.shape)
        if dimension not in (2, 3):
            raise ValueError("LBM requires a two- or three-dimensional tensor grid.")
        if plan.velocity_set.dimension != dimension:
            raise ValueError("Velocity-set dimension does not match the tensor grid.")
        if any(axis.primary_entity != "interval" for axis in grid.structured_axes):
            raise ValueError("LBM requires cell-centred interval-primary axes.")
        if any(axis.basis != "uniform" for axis in grid.axes):
            raise ValueError("LBM requires uniform tensor-grid axes.")
        widths = tuple(np.asarray(axis.interval_widths) for axis in grid.structured_axes)
        if any(
            values.size == 0
            or np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
            or not np.allclose(values, values[0], rtol=1e-12, atol=1e-14)
            for values in widths
        ):
            raise ValueError("Every LBM axis must have one finite uniform cell width.")
        spacing = tuple(float(values[0]) for values in widths)
        if not np.allclose(spacing, spacing[0], rtol=1e-12, atol=1e-14):
            raise ValueError("LBM requires equal cell size on every spatial axis.")

        q = plan.velocity_set.population_count
        dtype = jnp.dtype(plan.precision.population_dtype)
        population_space = grid.field_space(
            plan.field_name,
            component_shape=(q,),
            dtype=dtype,
            representation="custom",
            conformity="unrestricted",
        )
        density_space = grid.field_space(
            "density", dtype=dtype, representation="point_value"
        )
        velocity_space = grid.field_space(
            "velocity",
            component_shape=(dimension,),
            dtype=dtype,
            representation="point_value",
        )
        pressure_space = grid.field_space(
            "pressure", dtype=dtype, representation="point_value"
        )
        field_spaces = (
            population_space,
            density_space,
            velocity_space,
            pressure_space,
        )
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=(
                "uniform isotropic cell-centred lattice",
                "trailing population component axis",
                "fixed nearest-neighbour velocity set",
            ),
            resource_counts={
                "cells": grid.size,
                "populations_per_cell": q,
                "population_scalars": grid.size * q,
                "population_bytes": grid.size
                * q
                * plan.precision.resource_assumptions.itemsize("storage"),
            },
        )
        spaces, measures, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=grid.support,
            field_spaces=field_spaces,
            measures=(grid.measure,),
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-lattice-boltzmann-discretization",
                "plan": plan.plan_id,
                "grid": grid.prepared_id,
                "velocity_set": plan.velocity_set.lattice_id,
                "precision": plan.precision.policy_id,
                "cell_size": spacing[0],
                "population_shape": [*grid.shape, q],
                "numeric_version": version,
            }
        )
        self.plan = plan
        self.grid = grid
        self.velocity_set = plan.velocity_set
        self.precision = plan.precision
        self.population_space = population_space
        self.density_space = density_space
        self.velocity_space = velocity_space
        self.pressure_space = pressure_space
        self.cell_size = jnp.asarray(spacing[0], dtype=jnp.float64)
        self.key = plan.key
        self.support = grid.support
        self.field_spaces = spaces
        self.measures = measures
        self.capabilities = capabilities
        self.population_shape = (*grid.shape, q)
        self.periodic = tuple(axis.periodic for axis in grid.structured_axes)
        self.plan_id = plan.plan_id
        self.prepared_id = prepared_id
        self.numeric_version = version
        self.preparation = preparation

    @property
    def precision_evidence(self):
        return self.precision.evidence()

    @property
    def resource_evidence_id(self) -> str:
        return self.precision.resource_assumptions.assumptions_id

    def validate_populations(self, populations: ArrayLike, /) -> Array:
        values = jnp.asarray(populations)
        if values.shape != self.population_shape:
            raise ValueError(
                f"LBM populations must have shape {self.population_shape}; got {values.shape}."
            )
        if values.dtype != jnp.dtype(self.precision.population_dtype):
            raise ValueError(
                "LBM population dtype does not match the prepared precision policy."
            )
        return values


__all__ = ["LatticeBoltzmannDiscretization", "LatticeBoltzmannPlan"]
