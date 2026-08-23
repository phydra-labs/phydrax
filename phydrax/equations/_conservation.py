#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..discretization.finite_volume import (
    AbstractNumericalFluxPlan,
    AbstractWavePropagationPlan,
    ConvexStateLimiterPlan,
    EntropyConservativeEulerFluxPlan,
    EntropyStableEulerFluxPlan,
    FiniteVolumeBoundarySet,
    FiniteVolumeDiscretization,
    FiniteVolumeMethodPlan,
    FWaveShallowWaterPlan,
    HLLCFluxPlan,
    MappedFiniteVolumeDiscretization,
    PreparedFiniteVolumeDynamics,
    RoeFluxPlan,
)
from ._hyperbolic_systems import (
    AbstractAdmissibleSystem,
    AbstractCharacteristicSystem,
    AbstractConservationSystem,
    AbstractEntropySystem,
    CompressibleNavierStokesSystem,
    EulerSystem,
    ShallowWaterSystem,
)


class ConservationProblemIR(StrictModule):
    """Conservation or balance law with explicit system, boundaries, and source."""

    system: AbstractConservationSystem
    boundaries: FiniteVolumeBoundarySet
    source: Any = eqx.field(static=True)
    name: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        field_name: str,
        system: AbstractConservationSystem,
        boundaries: FiniteVolumeBoundarySet,
        /,
        *,
        source=None,
        problem_id: str | None = None,
    ):
        name_ = str(name)
        field = str(field_name)
        if not name_ or not field:
            raise ValueError("Conservation problem and field names must be non-empty.")
        if not isinstance(system, AbstractConservationSystem):
            raise TypeError("system must be an AbstractConservationSystem.")
        if not isinstance(boundaries, FiniteVolumeBoundarySet):
            raise TypeError("boundaries must be a FiniteVolumeBoundarySet.")
        if source is not None and not callable(source):
            raise TypeError("source must be callable or None.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "conservation-problem",
                    "name": name_,
                    "field": field,
                    "system": system.system_id,
                    "boundaries": boundaries.boundary_set_id,
                    "source": None if source is None else repr(source),
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.system = system
        self.boundaries = boundaries
        self.source = source
        self.name = name_
        self.field_name = field
        self.problem_id = identifier


class CompiledConservationProblem(StrictModule):
    """Executable structured finite-volume residual with complete provenance."""

    problem: ConservationProblemIR
    discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization
    method: FiniteVolumeMethodPlan
    dynamics: PreparedFiniteVolumeDynamics
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: ConservationProblemIR,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        method: FiniteVolumeMethodPlan,
        dynamics: PreparedFiniteVolumeDynamics,
        /,
    ):
        if problem.field_name != discretization.cell_space.name:
            raise ValueError("Conserved field name must match the finite-volume space.")
        compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-conservation-problem",
                "problem": problem.problem_id,
                "discretization": discretization.prepared_id,
                "method": method.method_id,
                "dynamics": dynamics.dynamics_id,
            }
        )
        form_key = DiscretizationKey(
            "conservation_form",
            DiscretizationRole.RESIDUAL,
            domain_labels=discretization.key.domain_labels,
        )
        self.problem = problem
        self.discretization = discretization
        self.method = method
        self.dynamics = dynamics
        self.discretization_bundle = DiscretizationBundle(
            (
                DiscretizationRecord(
                    discretization.key,
                    type(discretization).__name__,
                    discretization.prepared_id,
                    numeric_version=discretization.numeric_version,
                ),
                DiscretizationRecord(
                    form_key,
                    "compiled-conservation-form",
                    compilation_id,
                    dependency_key_ids=(discretization.key.key_id,),
                ),
            )
        )
        self.compilation_id = compilation_id

    def face_fluxes(self, time: Array, state: Array, args: Any = None, /):
        return self.dynamics.face_fluxes(time, state, args)

    def residual_with_diagnostics(
        self,
        time: Array,
        state: Array,
        args: Any = None,
        /,
    ):
        return self.dynamics.residual_with_diagnostics(time, state, args)

    def stable_step(
        self,
        state: Array,
        args: Any = None,
        /,
        *,
        cfl: float = 0.45,
    ) -> Array:
        return self.dynamics.stable_step(state, args, cfl=cfl)

    def linearize(self, time: Array, state: Array, args: Any = None, /):
        return self.dynamics.linearize(time, state, args)

    def __call__(self, time: Array, state: Array, args: Any = None) -> Array:
        return self.dynamics(time, state, args)


def _validate_method(
    problem: ConservationProblemIR,
    method: FiniteVolumeMethodPlan,
    /,
) -> None:
    system = problem.system
    solver = method.interface_solver
    if isinstance(solver, RoeFluxPlan) and not isinstance(
        system, AbstractCharacteristicSystem
    ):
        raise ValueError("Roe flux requires a characteristic conservation system.")
    euler_systems = (EulerSystem, CompressibleNavierStokesSystem)
    if isinstance(solver, HLLCFluxPlan) and not isinstance(system, euler_systems):
        raise ValueError("HLLC flux requires an Euler-compatible system.")
    if isinstance(
        solver, (EntropyConservativeEulerFluxPlan, EntropyStableEulerFluxPlan)
    ) and not isinstance(system, euler_systems):
        raise ValueError("Euler entropy fluxes require an Euler-compatible system.")
    if method.positivity is not None and not isinstance(
        system, AbstractAdmissibleSystem
    ):
        raise ValueError("Positivity limiting requires an admissible system.")
    if method.entropy_diagnostics and not isinstance(system, AbstractEntropySystem):
        raise ValueError("Entropy diagnostics require an entropy system.")
    if isinstance(solver, FWaveShallowWaterPlan) and not isinstance(
        system, ShallowWaterSystem
    ):
        raise ValueError("Shallow-water f-wave flux requires ShallowWaterSystem.")
    if isinstance(solver, AbstractWavePropagationPlan) and method.positivity is not None:
        raise ValueError("Wave-propagation positivity limiting is not yet a face-state policy.")
    if not isinstance(solver, (AbstractNumericalFluxPlan, AbstractWavePropagationPlan)):
        raise TypeError("Finite-volume method has an invalid interface solver.")
    if method.positivity is not None and not isinstance(
        method.positivity, ConvexStateLimiterPlan
    ):
        raise TypeError("Finite-volume positivity policy is invalid.")


def compile_conservation_problem(
    problem: ConservationProblemIR,
    discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
    method: FiniteVolumeMethodPlan,
    /,
    *,
    capacity=None,
    bathymetry=None,
) -> CompiledConservationProblem:
    """Lower one conservation system onto prepared structured finite volumes."""
    if not isinstance(problem, ConservationProblemIR):
        raise TypeError("problem must be a ConservationProblemIR.")
    if not isinstance(
        discretization,
        (FiniteVolumeDiscretization, MappedFiniteVolumeDiscretization),
    ):
        raise TypeError("discretization must be prepared finite-volume geometry.")
    if not isinstance(method, FiniteVolumeMethodPlan):
        raise TypeError("method must be a FiniteVolumeMethodPlan.")
    _validate_method(problem, method)
    dynamics = PreparedFiniteVolumeDynamics(
        problem.system,
        discretization,
        method,
        problem.boundaries,
        capacity=capacity,
        bathymetry=bathymetry,
        source=problem.source,
    )
    return CompiledConservationProblem(problem, discretization, method, dynamics)


__all__ = [
    "CompiledConservationProblem",
    "ConservationProblemIR",
    "compile_conservation_problem",
]
