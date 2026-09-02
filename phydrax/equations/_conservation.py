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
from ..discretization.fem._boundary import FiniteElementBoundarySet
from ..discretization.fem._generic import FiniteElementDiscretization
from ..discretization.finite_difference import (
    PreparedSBPConservationDynamics,
    SBPFluxDifferencingMethodPlan,
    TensorSBPDiscretization,
)
from ..discretization.finite_volume import (
    AbstractNumericalFluxPlan,
    AbstractWavePropagationPlan,
    ConvexStateLimiterPlan,
    EinfeldtHLLFluxPlan,
    EntropyConservativeEulerFluxPlan,
    EntropyStableEulerFluxPlan,
    FiniteVolumeBoundarySet,
    FiniteVolumeDiscretization,
    FiniteVolumeMethodPlan,
    FiniteVolumePrecisionPolicy,
    HLLCFluxPlan,
    MappedFiniteVolumeDiscretization,
    NoSlipAdiabaticWallBoundary,
    NoSlipIsothermalWallBoundary,
    PiecewiseConstantReconstruction,
    PreparedFiniteVolumeDynamics,
    PreparedTriangleFiniteVolumeDynamics,
    PreparedUnstructuredFiniteVolumeDynamics,
    PrescribedHeatFluxWallBoundary,
    RoeFluxPlan,
    ShallowWaterHydrostaticHLLPlan,
    TriangleFiniteVolumeBoundarySet,
    TriangleFiniteVolumeDiscretization,
    TriangleFiniteVolumeMethodPlan,
    TriangleKExactReconstructionPlan,
    TriangleMUSCLReconstructionPlan,
    UnstructuredFiniteVolumeBoundarySet,
    UnstructuredFiniteVolumeCouplingPlan,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumeMethodPlan,
)
from ..discretization.spectral import (
    PreparedSpectralConservationDynamics,
    SpectralConservationMethodPlan,
    TensorSpectralDiscretization,
)
from ._entropy_pair import ConvexEntropyPair
from ._hyperbolic_systems import (
    AbstractAdmissibleSystem,
    AbstractCharacteristicSystem,
    AbstractConservationSystem,
    CompressibleNavierStokesSystem,
    EulerSystem,
    ShallowWaterSystem,
)
from ._multiphase import TwoMaterialVOFSystem
from ._shallow_water_sources import ShallowWaterCoriolisSource
from .fem._conservation import (
    DGSEMConservationMethodPlan,
    PreparedDGSEMConservationDynamics,
)
from .fem._nodal_conservation import (
    NodalDGConservationMethodPlan,
    PreparedNodalDGConservationDynamics,
)


class ConservationProblemIR(StrictModule):
    """Conservation or balance law with explicit system, boundaries, and source."""

    system: AbstractConservationSystem
    boundaries: (
        FiniteVolumeBoundarySet
        | TriangleFiniteVolumeBoundarySet
        | UnstructuredFiniteVolumeBoundarySet
        | FiniteElementBoundarySet
        | None
    )
    source: Any = eqx.field(static=True)
    source_id: str | None = eqx.field(static=True)
    name: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        field_name: str,
        system: AbstractConservationSystem,
        boundaries: (
            FiniteVolumeBoundarySet
            | TriangleFiniteVolumeBoundarySet
            | UnstructuredFiniteVolumeBoundarySet
            | FiniteElementBoundarySet
            | None
        ),
        /,
        *,
        source=None,
        source_id: str | None = None,
        problem_id: str | None = None,
    ):
        name_ = str(name)
        field = str(field_name)
        if not name_ or not field:
            raise ValueError("Conservation problem and field names must be non-empty.")
        if not isinstance(system, AbstractConservationSystem):
            raise TypeError("system must be an AbstractConservationSystem.")
        if boundaries is not None and not isinstance(
            boundaries,
            (
                FiniteVolumeBoundarySet,
                TriangleFiniteVolumeBoundarySet,
                UnstructuredFiniteVolumeBoundarySet,
                FiniteElementBoundarySet,
            ),
        ):
            raise TypeError(
                "boundaries must be a prepared conservation boundary set or None."
            )
        if source is not None and not callable(source):
            raise TypeError("source must be callable or None.")
        source_identifier = None if source_id is None else str(source_id)
        if (source is None) != (source_identifier is None) or source_identifier == "":
            raise ValueError(
                "A source callable requires exactly one non-empty source_id."
            )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "conservation-problem",
                    "name": name_,
                    "field": field,
                    "system": system.system_id,
                    "boundaries": (
                        "periodic" if boundaries is None else boundaries.boundary_set_id
                    ),
                    "source": source_identifier,
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
        self.source_id = source_identifier
        self.name = name_
        self.field_name = field
        self.problem_id = identifier


class CompiledConservationProblem(StrictModule):
    """Executable conservation semidiscretization with complete provenance."""

    problem: ConservationProblemIR
    discretization: (
        FiniteVolumeDiscretization
        | MappedFiniteVolumeDiscretization
        | TriangleFiniteVolumeDiscretization
        | UnstructuredFiniteVolumeDiscretization
        | TensorSpectralDiscretization
        | TensorSBPDiscretization
        | FiniteElementDiscretization
    )
    method: (
        FiniteVolumeMethodPlan
        | TriangleFiniteVolumeMethodPlan
        | UnstructuredFiniteVolumeMethodPlan
        | SpectralConservationMethodPlan
        | SBPFluxDifferencingMethodPlan
        | DGSEMConservationMethodPlan
        | NodalDGConservationMethodPlan
    )
    dynamics: (
        PreparedFiniteVolumeDynamics
        | PreparedTriangleFiniteVolumeDynamics
        | PreparedUnstructuredFiniteVolumeDynamics
        | PreparedSpectralConservationDynamics
        | PreparedSBPConservationDynamics
        | PreparedDGSEMConservationDynamics
        | PreparedNodalDGConservationDynamics
    )
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: ConservationProblemIR,
        discretization: (
            FiniteVolumeDiscretization
            | MappedFiniteVolumeDiscretization
            | TriangleFiniteVolumeDiscretization
            | UnstructuredFiniteVolumeDiscretization
            | TensorSpectralDiscretization
            | TensorSBPDiscretization
            | FiniteElementDiscretization
        ),
        method: (
            FiniteVolumeMethodPlan
            | TriangleFiniteVolumeMethodPlan
            | UnstructuredFiniteVolumeMethodPlan
            | SpectralConservationMethodPlan
            | SBPFluxDifferencingMethodPlan
            | DGSEMConservationMethodPlan
            | NodalDGConservationMethodPlan
        ),
        dynamics: (
            PreparedFiniteVolumeDynamics
            | PreparedTriangleFiniteVolumeDynamics
            | PreparedUnstructuredFiniteVolumeDynamics
            | PreparedSpectralConservationDynamics
            | PreparedSBPConservationDynamics
            | PreparedDGSEMConservationDynamics
            | PreparedNodalDGConservationDynamics
        ),
        /,
    ):
        if isinstance(discretization, TensorSpectralDiscretization):
            state_space_name = discretization.modal_space.name
        elif isinstance(discretization, TensorSBPDiscretization):
            state_space_name = discretization.state_space.name
        elif isinstance(discretization, FiniteElementDiscretization):
            if len(discretization.field_spaces) != 1:
                raise ValueError(
                    "Finite-element conservation requires exactly one conserved field."
                )
            state_space_name = discretization.field_spaces[0].name
        else:
            state_space_name = discretization.cell_space.name
        if problem.field_name != state_space_name:
            raise ValueError(
                "Conserved field name must match the prepared spatial state space."
            )
        identity_payload = {
            "kind": "compiled-conservation-problem",
            "problem": problem.problem_id,
            "discretization": discretization.prepared_id,
            "method": method.method_id,
            "dynamics": dynamics.dynamics_id,
        }
        if isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            coupling = dynamics.coupling
            identity_payload["coupling"] = coupling.prepared_id
            identity_payload["embedded_metrics"] = (
                None
                if coupling.embedded_metrics is None
                else coupling.embedded_metrics.metrics_id
            )
            identity_payload["embedded_stabilization_policy"] = (
                None
                if coupling.embedded_stabilization_policy is None
                else coupling.embedded_stabilization_policy.policy_id
            )
            identity_payload["cut_boundaries"] = coupling.cut_boundary_id
        compilation_id = canonical_fingerprint(identity_payload)
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
                    precision_evidence_id=discretization.precision_evidence_id,
                    resource_evidence_id=discretization.resource_evidence_id,
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
        if isinstance(self.dynamics, PreparedSpectralConservationDynamics):
            raise NotImplementedError(
                "Global spectral conservation has no interface face-flux array."
            )
        return self.dynamics.face_fluxes(time, state, args)

    def weak_residual(self, time: Array, state: Array, args: Any = None, /) -> Array:
        if not isinstance(
            self.dynamics,
            (PreparedDGSEMConservationDynamics, PreparedNodalDGConservationDynamics),
        ):
            raise TypeError(
                "A dual-valued weak residual is specific to finite-element DG methods."
            )
        return self.dynamics.weak_residual(time, state, args)

    def mass_inverted_rate(self, time: Array, state: Array, args: Any = None, /) -> Array:
        if not isinstance(
            self.dynamics,
            (PreparedDGSEMConservationDynamics, PreparedNodalDGConservationDynamics),
        ):
            raise TypeError(
                "An explicit FE mass solve is specific to finite-element DG methods."
            )
        return self.dynamics.mass_inverted_rate(time, state, args)

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
        if isinstance(self.dynamics, PreparedSpectralConservationDynamics):
            raise NotImplementedError(
                "Spectral stable-step selection requires a separate modal CFL policy."
            )
        return self.dynamics.stable_step(state, args, cfl=cfl)

    def linearize(self, time: Array, state: Array, args: Any = None, /):
        return self.dynamics.linearize(time, state, args)

    def __call__(self, time: Array, state: Array, args: Any = None) -> Array:
        return self.dynamics(time, state, args)


def _validate_entropy_pair(
    problem: ConservationProblemIR,
    entropy_pair: ConvexEntropyPair | None,
    /,
) -> None:
    if entropy_pair is None:
        return
    if not isinstance(entropy_pair, ConvexEntropyPair):
        raise TypeError("entropy_pair must be a ConvexEntropyPair or None.")
    system = problem.system
    if entropy_pair.system.system_id != system.system_id:
        raise ValueError("entropy_pair must target the conservation problem system.")
    if entropy_pair.component_count != system.component_count:
        raise ValueError("entropy_pair component count must match the system.")
    if entropy_pair.dimension != system.dimension:
        raise ValueError("entropy_pair dimension must match the system.")


def _validate_method(
    problem: ConservationProblemIR,
    method: FiniteVolumeMethodPlan,
    entropy_pair: ConvexEntropyPair | None,
    /,
) -> None:
    system = problem.system
    if isinstance(problem.source, ShallowWaterCoriolisSource):
        if not isinstance(system, ShallowWaterSystem) or system.dimension != 2:
            raise ValueError(
                "Shallow-water Coriolis forcing requires a two-dimensional "
                "ShallowWaterSystem."
            )
    solver = method.interface_solver
    if isinstance(solver, RoeFluxPlan) and not isinstance(
        system, AbstractCharacteristicSystem
    ):
        raise ValueError("Roe flux requires a characteristic conservation system.")
    euler_systems = (EulerSystem, CompressibleNavierStokesSystem)
    if isinstance(solver, (HLLCFluxPlan, EinfeldtHLLFluxPlan)) and not isinstance(
        system, euler_systems
    ):
        raise ValueError(
            "HLLC and Einfeldt HLL fluxes require an Euler-compatible system."
        )
    if isinstance(
        solver, (EntropyConservativeEulerFluxPlan, EntropyStableEulerFluxPlan)
    ) and not isinstance(system, euler_systems):
        raise ValueError("Euler entropy fluxes require an Euler-compatible system.")
    if method.positivity is not None and not isinstance(system, AbstractAdmissibleSystem):
        raise ValueError("Positivity limiting requires an admissible system.")
    if isinstance(solver, ShallowWaterHydrostaticHLLPlan):
        if not isinstance(system, ShallowWaterSystem):
            raise ValueError("Hydrostatic shallow-water HLL requires ShallowWaterSystem.")
        if system.dimension not in (1, 2):
            raise ValueError("Hydrostatic shallow water supports one or two dimensions.")
    if isinstance(solver, AbstractWavePropagationPlan) and method.positivity is not None:
        raise ValueError(
            "Wave-propagation positivity limiting is not yet a face-state policy."
        )
    if not isinstance(
        solver,
        (
            AbstractNumericalFluxPlan,
            AbstractWavePropagationPlan,
            ShallowWaterHydrostaticHLLPlan,
        ),
    ):
        raise TypeError("Finite-volume method has an invalid interface solver.")
    if method.positivity is not None and not isinstance(
        method.positivity, ConvexStateLimiterPlan
    ):
        raise TypeError("Finite-volume positivity policy is invalid.")
    if entropy_pair is not None:
        if method.viscous is not None:
            raise ValueError(
                "Entropy pair diagnostics are unsupported with viscous "
                "finite-volume fluxes until viscous entropy production is "
                "represented separately."
            )
        if not isinstance(solver, AbstractNumericalFluxPlan):
            raise ValueError(
                "Entropy pair diagnostics require a numerical flux interface solver."
            )


def compile_conservation_problem(
    problem: ConservationProblemIR,
    discretization: (
        FiniteVolumeDiscretization
        | MappedFiniteVolumeDiscretization
        | TriangleFiniteVolumeDiscretization
        | UnstructuredFiniteVolumeDiscretization
        | TensorSpectralDiscretization
        | TensorSBPDiscretization
        | FiniteElementDiscretization
    ),
    method: (
        FiniteVolumeMethodPlan
        | TriangleFiniteVolumeMethodPlan
        | UnstructuredFiniteVolumeMethodPlan
        | SpectralConservationMethodPlan
        | SBPFluxDifferencingMethodPlan
        | DGSEMConservationMethodPlan
        | NodalDGConservationMethodPlan
    ),
    /,
    *,
    capacity=None,
    bathymetry=None,
    precision: FiniteVolumePrecisionPolicy | None = None,
    coupling: UnstructuredFiniteVolumeCouplingPlan | None = None,
    entropy_pair: ConvexEntropyPair | None = None,
) -> CompiledConservationProblem:
    """Lower one conservation system onto a prepared spatial method."""
    if not isinstance(problem, ConservationProblemIR):
        raise TypeError("problem must be a ConservationProblemIR.")
    _validate_entropy_pair(problem, entropy_pair)
    if isinstance(discretization, FiniteElementDiscretization):
        if isinstance(method, NodalDGConservationMethodPlan):
            if not isinstance(problem.boundaries, FiniteElementBoundarySet):
                raise TypeError("Nodal DG requires exhaustive FiniteElementBoundarySet.")
            if any(
                value is not None for value in (capacity, bathymetry, precision, coupling)
            ):
                raise ValueError(
                    "Nodal DG does not accept finite-volume capacity, bathymetry, "
                    "precision, or coupling options."
                )
            dynamics = PreparedNodalDGConservationDynamics(
                problem.system,
                discretization,
                method,
                problem.boundaries,
                source=problem.source,
                entropy_pair=entropy_pair,
            )
            return CompiledConservationProblem(problem, discretization, method, dynamics)
        if not isinstance(method, DGSEMConservationMethodPlan):
            raise TypeError(
                "Finite-element DGSEM geometry requires DGSEMConservationMethodPlan."
            )
        if isinstance(problem.system, EulerSystem):
            if method.viscous is not None:
                raise ValueError("Euler DGSEM cannot compile a viscous DG plan.")
        elif isinstance(problem.system, CompressibleNavierStokesSystem):
            if method.viscous is None:
                raise ValueError(
                    "Compressible Navier–Stokes DGSEM requires a viscous DG plan."
                )
        else:
            raise TypeError(
                "Finite-element DGSEM supports EulerSystem or "
                "CompressibleNavierStokesSystem."
            )
        if problem.boundaries is not None and not isinstance(
            problem.boundaries, FiniteElementBoundarySet
        ):
            raise TypeError(
                "Finite-element DGSEM requires FiniteElementBoundarySet or None."
            )
        if any(
            value is not None for value in (capacity, bathymetry, precision, coupling)
        ):
            raise ValueError(
                "Finite-element DGSEM does not accept finite-volume capacity, "
                "bathymetry, precision, or coupling options."
            )
        certificate = method.compatibility
        if (certificate is None) != (entropy_pair is None):
            raise ValueError(
                "DGSEM entropy_pair and flux compatibility certificate must be "
                "supplied together."
            )
        dynamics = PreparedDGSEMConservationDynamics(
            problem.system,
            discretization,
            method,
            boundaries=problem.boundaries,
            source=problem.source,
            entropy_pair=entropy_pair,
        )
        return CompiledConservationProblem(problem, discretization, method, dynamics)
    if isinstance(discretization, TensorSBPDiscretization):
        if not isinstance(method, SBPFluxDifferencingMethodPlan):
            raise TypeError("Tensor SBP geometry requires SBPFluxDifferencingMethodPlan.")
        if not isinstance(problem.system, EulerSystem):
            raise TypeError(
                "Initial SBP flux differencing supports inviscid EulerSystem only."
            )
        if problem.boundaries is not None:
            raise ValueError("Periodic SBP conservation requires boundaries=None.")
        if any(
            value is not None for value in (capacity, bathymetry, precision, coupling)
        ):
            raise ValueError(
                "SBP conservation does not accept finite-volume capacity, "
                "bathymetry, precision, or coupling options."
            )
        if method.entropy_diagnostics != (entropy_pair is not None):
            raise ValueError(
                "SBP entropy diagnostics and entropy_pair must be enabled together."
            )
        dynamics = PreparedSBPConservationDynamics(
            problem.system,
            discretization,
            method,
            source=problem.source,
            entropy_pair=entropy_pair,
        )
        return CompiledConservationProblem(problem, discretization, method, dynamics)
    if isinstance(discretization, TensorSpectralDiscretization):
        if not isinstance(method, SpectralConservationMethodPlan):
            raise TypeError(
                "Tensor spectral geometry requires SpectralConservationMethodPlan."
            )
        if problem.boundaries is not None:
            raise ValueError("Periodic spectral conservation requires boundaries=None.")
        if any(
            value is not None for value in (capacity, bathymetry, precision, coupling)
        ):
            raise ValueError(
                "Spectral conservation does not accept finite-volume capacity, "
                "bathymetry, precision, or coupling options."
            )
        if method.entropy_diagnostics != (entropy_pair is not None):
            raise ValueError(
                "Spectral entropy diagnostics and entropy_pair must be enabled together."
            )
        dynamics = PreparedSpectralConservationDynamics(
            problem.system,
            discretization,
            method,
            source=problem.source,
            entropy_pair=entropy_pair,
        )
        return CompiledConservationProblem(problem, discretization, method, dynamics)
    if problem.boundaries is None:
        raise ValueError(
            "Finite-volume conservation requires an explicit prepared boundary set."
        )
    if isinstance(problem.system, TwoMaterialVOFSystem) and not isinstance(
        discretization, UnstructuredFiniteVolumeDiscretization
    ):
        raise ValueError(
            "TwoMaterialVOFSystem requires prepared unstructured VOF coupling "
            "and the piecewise-constant per-stage PLIC path."
        )
    if isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
        if not isinstance(method, UnstructuredFiniteVolumeMethodPlan):
            raise TypeError(
                "Unstructured geometry requires UnstructuredFiniteVolumeMethodPlan."
            )
        if not isinstance(problem.boundaries, UnstructuredFiniteVolumeBoundarySet):
            raise TypeError("Unstructured geometry requires patch boundary ownership.")
        if entropy_pair is not None:
            raise ValueError(
                "entropy_pair diagnostics currently support structured and mapped "
                "finite-volume geometry only."
            )
        if isinstance(
            method.interface_solver, (HLLCFluxPlan, EinfeldtHLLFluxPlan)
        ) and not isinstance(
            problem.system, (EulerSystem, CompressibleNavierStokesSystem)
        ):
            raise ValueError(
                "Unstructured HLLC and Einfeldt HLL fluxes require an "
                "Euler-compatible system."
            )
        if capacity is not None:
            raise ValueError(
                "Unstructured finite volume does not support capacity fields."
            )
        if bathymetry is not None:
            raise ValueError(
                "Unstructured finite volume does not support bathymetry fields."
            )
        coupling_plan = (
            UnstructuredFiniteVolumeCouplingPlan() if coupling is None else coupling
        )
        if not isinstance(coupling_plan, UnstructuredFiniteVolumeCouplingPlan):
            raise TypeError(
                "coupling must be UnstructuredFiniteVolumeCouplingPlan or None."
            )
        prepared_coupling = coupling_plan.prepare(discretization)
        coupling_plan.validate_execution_support()
        if (
            prepared_coupling.motion is not None
            and type(method.reconstruction) is not PiecewiseConstantReconstruction
        ):
            raise ValueError(
                "Moving unstructured finite-volume coupling requires exact "
                "PiecewiseConstantReconstruction because prepared high-order "
                "operators bind static geometry "
                f"(coupling={prepared_coupling.prepared_id}, "
                f"method={method.method_id}, "
                f"reconstruction={type(method.reconstruction).__name__})."
            )
        if (
            isinstance(problem.system, TwoMaterialVOFSystem)
            and prepared_coupling.vof is None
        ):
            raise ValueError(
                "TwoMaterialVOFSystem requires prepared unstructured VOF coupling "
                "and the piecewise-constant per-stage PLIC path."
            )
        if prepared_coupling.vof is not None:
            if not isinstance(problem.system, TwoMaterialVOFSystem):
                raise TypeError(
                    "Unstructured VOF coupling requires TwoMaterialVOFSystem."
                )
            if type(method.reconstruction) is not PiecewiseConstantReconstruction:
                raise ValueError(
                    "Per-stage PLIC coupling currently requires exact "
                    "PiecewiseConstantReconstruction."
                )
            if any(
                component is not None
                for component in (
                    prepared_coupling.motion,
                    prepared_coupling.amr,
                    prepared_coupling.overset,
                    prepared_coupling.sliding,
                )
            ):
                raise ValueError(
                    "Two-material VOF execution does not yet support "
                    "motion/AMR/overset/sliding combinations."
                )
            if (
                prepared_coupling.embedded_boundary is not None
                and prepared_coupling.contact_angles is None
            ):
                raise ValueError(
                    "VOF with embedded geometry requires explicit contact-angle policies."
                )
        if prepared_coupling.embedded_metrics is not None:
            if type(method.reconstruction) is not PiecewiseConstantReconstruction:
                raise ValueError(
                    "Stationary embedded-boundary finite-volume coupling requires "
                    "exact PiecewiseConstantReconstruction; high-order cut "
                    "reconstruction is not certified "
                    f"(coupling={prepared_coupling.prepared_id}, "
                    f"method={method.method_id}, "
                    f"reconstruction={type(method.reconstruction).__name__})."
                )
            if getattr(method, "viscous", None) is not None or isinstance(
                problem.system, CompressibleNavierStokesSystem
            ):
                raise ValueError(
                    "Viscous embedded-boundary methods are not supported; cut-wall "
                    "viscous closure must fail closed."
                )
        dynamics = PreparedUnstructuredFiniteVolumeDynamics(
            problem.system,
            discretization,
            method,
            problem.boundaries,
            source=problem.source,
            source_id=problem.source_id,
            precision=precision,
            coupling=prepared_coupling,
        )
        return CompiledConservationProblem(problem, discretization, method, dynamics)
    if coupling is not None:
        raise ValueError(
            "Unstructured finite-volume coupling requires unstructured geometry."
        )
    if isinstance(discretization, TriangleFiniteVolumeDiscretization):
        if not isinstance(method, TriangleFiniteVolumeMethodPlan):
            raise TypeError("Triangle geometry requires TriangleFiniteVolumeMethodPlan.")
        if not isinstance(problem.boundaries, TriangleFiniteVolumeBoundarySet):
            raise TypeError("Triangle geometry requires patch boundary ownership.")
        if entropy_pair is not None:
            raise ValueError(
                "entropy_pair diagnostics currently support structured and mapped "
                "finite-volume geometry only."
            )
        if isinstance(
            method.interface_solver, (HLLCFluxPlan, EinfeldtHLLFluxPlan)
        ) and not isinstance(
            problem.system, (EulerSystem, CompressibleNavierStokesSystem)
        ):
            raise ValueError(
                "Triangle HLLC and Einfeldt HLL fluxes require an "
                "Euler-compatible system."
            )
        triangle_reconstruction_geometry = None
        if isinstance(method.reconstruction, TriangleMUSCLReconstructionPlan):
            triangle_reconstruction_geometry = (
                method.reconstruction.gradient.discretization.prepared_id
            )
        elif isinstance(method.reconstruction, TriangleKExactReconstructionPlan):
            triangle_reconstruction_geometry = (
                method.reconstruction.prepared.discretization.prepared_id
            )
        if (
            triangle_reconstruction_geometry is not None
            and triangle_reconstruction_geometry != discretization.prepared_id
        ):
            raise ValueError("Triangle reconstruction belongs to a different geometry.")
        thermal_walls = (
            NoSlipAdiabaticWallBoundary,
            NoSlipIsothermalWallBoundary,
            PrescribedHeatFluxWallBoundary,
        )
        if (
            any(
                isinstance(boundary, thermal_walls)
                for boundary in problem.boundaries.boundaries
            )
            and method.viscous is None
        ):
            raise ValueError(
                "Triangle no-slip and thermal walls require viscous closure."
            )
        if method.viscous is not None:
            if not isinstance(problem.system, CompressibleNavierStokesSystem):
                raise ValueError("Triangle viscous flux requires Navier-Stokes physics.")
            if (
                method.viscous.gradient.discretization.prepared_id
                != discretization.prepared_id
            ):
                raise ValueError(
                    "Triangle viscous gradient belongs to a different geometry."
                )
        if capacity is not None:
            raise ValueError(
                "Triangle finite volume does not yet support capacity fields."
            )
        if bathymetry is not None:
            raise ValueError(
                "Triangle finite volume does not yet support bathymetry fields."
            )
        dynamics = PreparedTriangleFiniteVolumeDynamics(
            problem.system,
            discretization,
            method,
            problem.boundaries,
            source=problem.source,
            source_id=problem.source_id,
            precision=precision,
        )
        return CompiledConservationProblem(problem, discretization, method, dynamics)
    if not isinstance(
        discretization,
        (FiniteVolumeDiscretization, MappedFiniteVolumeDiscretization),
    ):
        raise TypeError("discretization must be prepared finite-volume geometry.")
    if not isinstance(method, FiniteVolumeMethodPlan):
        raise TypeError("Structured geometry requires FiniteVolumeMethodPlan.")
    _validate_method(problem, method, entropy_pair)
    solver = method.interface_solver
    bed_aware = isinstance(solver, ShallowWaterHydrostaticHLLPlan)
    if bathymetry is None and bed_aware:
        raise ValueError("The selected shallow-water interface requires bathymetry.")
    if bathymetry is not None and not bed_aware:
        raise ValueError(
            "Bathymetry requires a bed-aware shallow-water interface method."
        )
    if isinstance(solver, ShallowWaterHydrostaticHLLPlan) and isinstance(
        discretization, MappedFiniteVolumeDiscretization
    ):
        raise ValueError(
            "Initial hydrostatic shallow water requires Cartesian structured geometry."
        )
    dynamics = PreparedFiniteVolumeDynamics(
        problem.system,
        discretization,
        method,
        problem.boundaries,
        capacity=capacity,
        bathymetry=bathymetry,
        precision=precision,
        source=problem.source,
        source_id=problem.source_id,
        entropy_pair=entropy_pair,
    )
    return CompiledConservationProblem(problem, discretization, method, dynamics)


__all__ = [
    "CompiledConservationProblem",
    "ConservationProblemIR",
    "compile_conservation_problem",
]
