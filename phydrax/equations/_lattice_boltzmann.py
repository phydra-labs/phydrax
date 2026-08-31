#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from operator import index
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..discretization.lattice_boltzmann import (
    LatticeAcceleration,
    LatticeBoltzmannBoundaryPlan,
    LatticeBoltzmannDiscretization,
    LatticeBoltzmannGeometrySnapshot,
    LatticeBoltzmannMacroscopicState,
    LatticeBoltzmannMethodPlan,
    LatticeBoltzmannRuntimeParameters,
    LatticeBoltzmannScaling,
    PreparedLatticeBoltzmannDynamics,
    StagedLatticeBoltzmannBoundaryPlan,
    VelocityDependentAccelerationPlan,
)
from ..geometry import CompiledGeometry, GeometryCapability


class LatticeBoltzmannProblem(StrictModule, NonTrainableState):
    """Athermal weakly compressible Newtonian flow on a lattice."""

    reference_density: Array
    acceleration: LatticeAcceleration | None
    implicit_acceleration: VelocityDependentAccelerationPlan | None
    name: str = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    acceleration_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        spatial_dimension: int,
        /,
        *,
        reference_density: float = 1.0,
        acceleration: LatticeAcceleration | None = None,
        acceleration_id: str | None = None,
        implicit_acceleration: VelocityDependentAccelerationPlan | None = None,
        problem_id: str | None = None,
    ):
        name_ = str(name)
        if not name_:
            raise ValueError("Lattice-Boltzmann problem name must be non-empty.")
        if isinstance(spatial_dimension, bool):
            raise TypeError("spatial_dimension must be an integer.")
        dimension = index(spatial_dimension)
        if dimension not in (2, 3):
            raise ValueError("Lattice-Boltzmann flow requires dimension two or three.")
        density = float(reference_density)
        if not np.isfinite(density) or density <= 0.0:
            raise ValueError("reference_density must be finite and positive.")
        if acceleration is not None and not callable(acceleration):
            raise TypeError("acceleration must be callable or None.")
        if acceleration is None:
            if acceleration_id is not None:
                raise ValueError("acceleration_id requires an acceleration callable.")
            source_id = None
        else:
            source_id = "" if acceleration_id is None else str(acceleration_id)
            if not source_id:
                raise ValueError("Acceleration requires a non-empty acceleration_id.")
        if implicit_acceleration is not None and not isinstance(
            implicit_acceleration, VelocityDependentAccelerationPlan
        ):
            raise TypeError(
                "implicit_acceleration must be VelocityDependentAccelerationPlan or None."
            )
        if acceleration is not None and implicit_acceleration is not None:
            raise ValueError(
                "Explicit and velocity-dependent acceleration are mutually exclusive."
            )
        generated = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-problem",
                "name": name_,
                "dimension": dimension,
                "reference_density": density,
                "acceleration": source_id,
                "implicit_acceleration": (
                    None
                    if implicit_acceleration is None
                    else implicit_acceleration.plan_id
                ),
            }
        )
        identifier = generated if problem_id is None else str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.reference_density = jnp.asarray(density, dtype=jnp.float64)
        self.acceleration = acceleration
        self.implicit_acceleration = implicit_acceleration
        self.name = name_
        self.spatial_dimension = dimension
        self.acceleration_id = source_id
        self.problem_id = identifier


class CompiledLatticeBoltzmannProblem(StrictModule, NonTrainableState):
    problem: LatticeBoltzmannProblem
    discretization: LatticeBoltzmannDiscretization
    method: LatticeBoltzmannMethodPlan
    boundary: Any
    scaling: LatticeBoltzmannScaling
    dynamics: PreparedLatticeBoltzmannDynamics
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def initialize_state(
        self,
        density: ArrayLike,
        velocity: ArrayLike,
        parameters: LatticeBoltzmannRuntimeParameters,
        /,
        *,
        time: ArrayLike = 0.0,
    ) -> Array:
        return self.dynamics.initialize_state(density, velocity, parameters, time=time)

    def macroscopic_state(
        self,
        time: ArrayLike,
        populations: ArrayLike,
        parameters: LatticeBoltzmannRuntimeParameters,
        /,
    ) -> LatticeBoltzmannMacroscopicState:
        return self.dynamics.macroscopic_state(time, populations, parameters)


def snapshot_lattice_boltzmann_geometry(
    discretization: LatticeBoltzmannDiscretization,
    geometry: CompiledGeometry,
    /,
    *,
    fluid_inside: bool = True,
) -> LatticeBoltzmannGeometrySnapshot:
    """Freeze one live region query into an immutable LBM fluid mask."""

    if not isinstance(discretization, LatticeBoltzmannDiscretization):
        raise TypeError("discretization must be an LBM discretization.")
    if not isinstance(geometry, CompiledGeometry):
        raise TypeError("geometry must be CompiledGeometry.")
    geometry.require(GeometryCapability.REGION_QUERY)
    if geometry.ambient_dimension != discretization.velocity_set.dimension:
        raise ValueError("Geometry and lattice dimensions do not match.")
    inside = np.asarray(geometry.contains(discretization.grid.points), dtype=bool)
    if inside.shape != (discretization.grid.size,):
        raise ValueError("Geometry region query returned an incompatible point mask.")
    mask = inside if fluid_inside else ~inside
    source_id = canonical_fingerprint(
        {
            "kind": "lattice-boltzmann-geometry-source",
            "kernel": type(geometry.kernel).__name__,
            "schema": repr(geometry.schema),
            "state": array_tree_fingerprint(
                tuple(np.asarray(value) for value in geometry.state.values)
            ),
            "fluid_inside": bool(fluid_inside),
        }
    )
    return LatticeBoltzmannGeometrySnapshot(
        discretization,
        mask.reshape(discretization.grid.shape),
        source_id=source_id,
    )


def compile_lattice_boltzmann_problem(
    problem: LatticeBoltzmannProblem,
    discretization: LatticeBoltzmannDiscretization,
    method: LatticeBoltzmannMethodPlan,
    boundary: LatticeBoltzmannBoundaryPlan | StagedLatticeBoltzmannBoundaryPlan,
    /,
    *,
    time_step: float,
) -> CompiledLatticeBoltzmannProblem:
    """Compile one typed athermal collide-and-route flow problem."""

    if not isinstance(problem, LatticeBoltzmannProblem):
        raise TypeError("problem must be a LatticeBoltzmannProblem.")
    if not isinstance(discretization, LatticeBoltzmannDiscretization):
        raise TypeError("discretization must be an LBM discretization.")
    if not isinstance(method, LatticeBoltzmannMethodPlan):
        raise TypeError("method must be a LatticeBoltzmannMethodPlan.")
    if not isinstance(
        boundary, (LatticeBoltzmannBoundaryPlan, StagedLatticeBoltzmannBoundaryPlan)
    ):
        raise TypeError("boundary must be a lattice-Boltzmann boundary plan.")
    if problem.spatial_dimension != discretization.velocity_set.dimension:
        raise ValueError("Problem and lattice dimensions do not match.")
    dt = float(time_step)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("time_step must be finite and positive.")
    scaling = LatticeBoltzmannScaling(
        float(discretization.cell_size),
        dt,
        float(problem.reference_density),
        sound_speed_squared=float(discretization.velocity_set.sound_speed_squared),
    )
    prepared_boundary = boundary.prepare(discretization)
    dynamics = PreparedLatticeBoltzmannDynamics(
        discretization,
        scaling,
        method,
        prepared_boundary,
        acceleration=problem.acceleration,
        acceleration_id=problem.acceleration_id,
        implicit_acceleration=problem.implicit_acceleration,
    )
    method_key = DiscretizationKey(
        "lattice_boltzmann_method",
        DiscretizationRole.TEMPORAL,
        domain_labels=discretization.grid.axis_names,
    )
    boundary_key = DiscretizationKey(
        "lattice_boltzmann_boundary",
        DiscretizationRole.AUXILIARY,
        domain_labels=discretization.grid.axis_names,
    )
    records = (
        DiscretizationRecord(
            discretization.key,
            "lattice-boltzmann-discretization",
            discretization.prepared_id,
            numeric_version=discretization.numeric_version,
            precision_evidence_id=discretization.precision_evidence_id,
            resource_evidence_id=discretization.resource_evidence_id,
        ),
        DiscretizationRecord(
            method_key,
            "lattice-boltzmann-method",
            method.method_id,
            dependency_key_ids=(discretization.key.key_id,),
        ),
        DiscretizationRecord(
            boundary_key,
            "lattice-boltzmann-boundary",
            prepared_boundary.boundary_id,
            dependency_key_ids=(discretization.key.key_id,),
        ),
    )
    bundle = DiscretizationBundle(records)
    compilation_id = canonical_fingerprint(
        {
            "kind": "compiled-lattice-boltzmann-problem",
            "problem": problem.problem_id,
            "dynamics": dynamics.prepared_id,
            "bundle": bundle.bundle_id,
        }
    )
    return CompiledLatticeBoltzmannProblem(
        problem,
        discretization,
        method,
        prepared_boundary,
        scaling,
        dynamics,
        bundle,
        compilation_id,
    )


__all__ = [
    "CompiledLatticeBoltzmannProblem",
    "LatticeBoltzmannProblem",
    "compile_lattice_boltzmann_problem",
    "snapshot_lattice_boltzmann_geometry",
]
