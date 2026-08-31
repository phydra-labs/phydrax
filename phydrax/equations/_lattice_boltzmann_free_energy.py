#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from operator import index

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..discretization.lattice_boltzmann._boundary import (
    LatticeBoltzmannBoundaryPlan,
    PreparedLatticeBoltzmannBoundary,
)
from ..discretization.lattice_boltzmann._discretization import (
    LatticeBoltzmannDiscretization,
)
from ..discretization.lattice_boltzmann._free_energy import (
    FreeEnergyLBMMethod,
    FreeEnergyLBMRuntimeParameters,
    FreeEnergyLBMState,
    FreeEnergyMacroscopicState,
    PreparedFreeEnergyLBMDynamics,
)
from ..discretization.lattice_boltzmann._scaling import LatticeBoltzmannScaling


class FreeEnergyLatticeBoltzmannProblem(StrictModule, NonTrainableState):
    """Binary diffuse-interface flow with coupled phase populations."""

    reference_density: Array
    name: str = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        spatial_dimension: int,
        /,
        *,
        reference_density: float = 1.0,
        problem_id: str | None = None,
    ):
        name_ = str(name)
        if not name_:
            raise ValueError("Free-energy problem name must be non-empty.")
        if isinstance(spatial_dimension, bool):
            raise TypeError("spatial_dimension must be an integer.")
        dimension = index(spatial_dimension)
        if dimension not in (2, 3):
            raise ValueError("Free-energy LBM requires dimension two or three.")
        density = float(reference_density)
        if not np.isfinite(density) or density <= 0.0:
            raise ValueError("reference_density must be finite and positive.")
        generated = canonical_fingerprint(
            {
                "kind": "free-energy-lattice-boltzmann-problem",
                "name": name_,
                "dimension": dimension,
                "reference_density": density,
            }
        )
        identifier = generated if problem_id is None else str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.reference_density = jnp.asarray(density, dtype=jnp.float64)
        self.name = name_
        self.spatial_dimension = dimension
        self.problem_id = identifier


class CompiledFreeEnergyLatticeBoltzmannProblem(StrictModule, NonTrainableState):
    problem: FreeEnergyLatticeBoltzmannProblem
    discretization: LatticeBoltzmannDiscretization
    method: FreeEnergyLBMMethod
    boundary: PreparedLatticeBoltzmannBoundary
    scaling: LatticeBoltzmannScaling
    dynamics: PreparedFreeEnergyLBMDynamics
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def initialize_state(
        self,
        density: ArrayLike,
        phase: ArrayLike,
        velocity: ArrayLike,
        parameters: FreeEnergyLBMRuntimeParameters,
        /,
    ) -> FreeEnergyLBMState:
        return self.dynamics.initialize_state(density, phase, velocity, parameters)

    def macroscopic_state(
        self,
        state: FreeEnergyLBMState,
        parameters: FreeEnergyLBMRuntimeParameters,
        /,
    ) -> FreeEnergyMacroscopicState:
        return self.dynamics.macroscopic_state(state, parameters)


def compile_free_energy_lattice_boltzmann_problem(
    problem: FreeEnergyLatticeBoltzmannProblem,
    discretization: LatticeBoltzmannDiscretization,
    method: FreeEnergyLBMMethod,
    boundary: LatticeBoltzmannBoundaryPlan,
    /,
    *,
    time_step: float,
) -> CompiledFreeEnergyLatticeBoltzmannProblem:
    """Compile one typed hydrodynamic/free-energy population problem."""

    if not isinstance(problem, FreeEnergyLatticeBoltzmannProblem):
        raise TypeError("problem must be FreeEnergyLatticeBoltzmannProblem.")
    if not isinstance(discretization, LatticeBoltzmannDiscretization):
        raise TypeError("discretization must be an LBM discretization.")
    if not isinstance(method, FreeEnergyLBMMethod):
        raise TypeError("method must be FreeEnergyLBMMethod.")
    if not isinstance(boundary, LatticeBoltzmannBoundaryPlan):
        raise TypeError("boundary must be LatticeBoltzmannBoundaryPlan.")
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
    dynamics = PreparedFreeEnergyLBMDynamics(
        discretization, scaling, method, prepared_boundary
    )
    method_key = DiscretizationKey(
        "free_energy_lattice_boltzmann_method",
        DiscretizationRole.TEMPORAL,
        domain_labels=discretization.grid.axis_names,
    )
    boundary_key = DiscretizationKey(
        "free_energy_lattice_boltzmann_boundary",
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
            "free-energy-lattice-boltzmann-method",
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
            "kind": "compiled-free-energy-lattice-boltzmann-problem",
            "problem": problem.problem_id,
            "dynamics": dynamics.prepared_id,
            "bundle": bundle.bundle_id,
        }
    )
    return CompiledFreeEnergyLatticeBoltzmannProblem(
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
    "CompiledFreeEnergyLatticeBoltzmannProblem",
    "FreeEnergyLatticeBoltzmannProblem",
    "compile_free_energy_lattice_boltzmann_problem",
]
