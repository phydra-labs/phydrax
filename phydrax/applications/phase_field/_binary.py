#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._phase_field import DoubleWellFreeEnergy
from ..._strict import StrictModule
from ..._thermodynamics import (
    BinaryPhaseThermodynamicClosure,
    BinaryThermodynamicParameters,
)
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite, tree_where
from ...discretization import FiniteElementDiscretization, IntegrationDomain
from ...equations import (
    compile_finite_element_functional,
    CompiledFiniteElementProblem,
    FiniteElementExecutionPolicy,
)
from ...integration import (
    GaussLegendreRule,
    ReferenceHexahedronRule,
    ReferencePrismRule,
    ReferencePyramidRule,
    ReferenceQuadrilateralRule,
    ReferenceTetrahedronRule,
    ReferenceTriangleRule,
)
from ...linalg import ArraySpace
from ...nonlinear import (
    AbstractNonlinearMethod,
    NewtonKrylov,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ...solver import (
    AbstractFixedStepMethod,
    FixedStepResult,
    ProductionCaseManifest,
    ProductionRunPlan,
    RobustRetryPolicy,
)
from ...variational import FieldJetSpec, Functional, LocalIntegralTerm
from ._boundary import (
    AbstractPhaseFieldSurfaceEnergy,
    PhaseFieldBoundaryPlan,
    PrescribedPhaseFieldFlux,
)
from ._energy_laws import (
    AbstractBulkEvolutionLaw,
    ConvexSplitDoubleWellLaw,
    DiscreteGradientBulkLaw,
    PhaseFieldEnergyLedger,
)
from ._mobility import (
    AbstractPhaseFieldMobility,
    as_phase_field_mobility,
    ScalarPhaseFieldMobility,
)
from ._stochastic import PhaseFieldNoisePlan


class BinaryPhaseFieldModel(StrictModule, NonTrainableState):
    """One binary free energy shared by phase-field dynamics and diagnostics."""

    thermodynamics: BinaryThermodynamicParameters
    closure: BinaryPhaseThermodynamicClosure
    evolution_law: AbstractBulkEvolutionLaw
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics: BinaryThermodynamicParameters,
        /,
        *,
        closure: BinaryPhaseThermodynamicClosure | None = None,
        evolution_law: AbstractBulkEvolutionLaw | None = None,
    ):
        if not isinstance(thermodynamics, BinaryThermodynamicParameters):
            raise TypeError("thermodynamics must be BinaryThermodynamicParameters.")
        selected = BinaryPhaseThermodynamicClosure() if closure is None else closure
        if not isinstance(selected, BinaryPhaseThermodynamicClosure):
            raise TypeError("closure must be BinaryPhaseThermodynamicClosure.")
        law = (
            ConvexSplitDoubleWellLaw()
            if evolution_law is None
            and isinstance(selected.free_energy, DoubleWellFreeEnergy)
            else DiscreteGradientBulkLaw()
            if evolution_law is None
            else evolution_law
        )
        if not isinstance(law, AbstractBulkEvolutionLaw):
            raise TypeError("evolution_law must implement AbstractBulkEvolutionLaw.")
        self.thermodynamics = thermodynamics
        self.closure = selected
        self.evolution_law = law
        self.model_id = canonical_fingerprint(
            {
                "kind": "binary-phase-field-model",
                "closure": selected.closure_id,
                "evolution_law": law.law_id,
                "thermodynamics": {
                    "bulk_scale": thermodynamics.bulk_scale,
                    "gradient_coefficient": thermodynamics.gradient_coefficient,
                },
            }
        )

    @property
    def effective_bulk_scale(self) -> Array:
        if not isinstance(self.closure.free_energy, DoubleWellFreeEnergy):
            raise TypeError("Effective quartic scale requires DoubleWellFreeEnergy.")
        dtype = self.thermodynamics.bulk_scale.dtype
        return self.thermodynamics.bulk_scale * self.closure.free_energy.scale.astype(
            dtype
        )

    def energy_components(self, phase: Array, gradient: Array, /) -> tuple[Array, Array]:
        value = jnp.asarray(phase)
        grad = jnp.asarray(gradient, dtype=value.dtype)
        if grad.shape[:-1] != value.shape:
            raise ValueError("Phase-field gradient shape must extend the phase shape.")
        bulk = self.thermodynamics.bulk_scale.astype(value.dtype) * (
            self.closure.free_energy.density(value)
        )
        gradient_energy = (
            0.5
            * self.thermodynamics.gradient_coefficient.astype(value.dtype)
            * ein.contract("...d,...d->...", grad, grad)
        )
        return bulk, gradient_energy

    def energy_density(self, phase: Array, gradient: Array, /) -> Array:
        bulk, gradient_energy = self.energy_components(phase, gradient)
        return bulk + gradient_energy

    def bulk_incremental_density(
        self,
        current: Array,
        previous: Array,
        /,
    ) -> Array:
        return self.thermodynamics.bulk_scale.astype(current.dtype) * (
            self.evolution_law.incremental_density(
                self.closure.free_energy, current, previous
            )
        )

    def bulk_discrete_derivative(
        self,
        current: Array,
        previous: Array,
        /,
    ) -> Array:
        return self.thermodynamics.bulk_scale.astype(current.dtype) * (
            self.evolution_law.derivative(self.closure.free_energy, current, previous)
        )


class PhaseFieldAcceptancePolicy(StrictModule, NonTrainableState):
    """Physical acceptance tolerances for one closed phase-field step."""

    absolute_energy_tolerance: float = eqx.field(static=True)
    relative_energy_tolerance: float = eqx.field(static=True)
    absolute_mass_tolerance: float = eqx.field(static=True)
    relative_mass_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        absolute_energy_tolerance: float = 1.0e-9,
        relative_energy_tolerance: float = 1.0e-8,
        absolute_mass_tolerance: float = 1.0e-8,
        relative_mass_tolerance: float = 1.0e-8,
    ):
        values = tuple(
            float(value)
            for value in (
                absolute_energy_tolerance,
                relative_energy_tolerance,
                absolute_mass_tolerance,
                relative_mass_tolerance,
            )
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Phase-field acceptance tolerances must be nonnegative.")
        (
            self.absolute_energy_tolerance,
            self.relative_energy_tolerance,
            self.absolute_mass_tolerance,
            self.relative_mass_tolerance,
        ) = values
        self.policy_id = canonical_fingerprint(
            {
                "kind": "phase-field-acceptance-policy",
                "absolute_energy_tolerance": values[0],
                "relative_energy_tolerance": values[1],
                "absolute_mass_tolerance": values[2],
                "relative_mass_tolerance": values[3],
            }
        )

    def energy_tolerance(self, before: Array, after: Array, /) -> Array:
        scale = jnp.maximum(jnp.abs(before), jnp.abs(after))
        return self.absolute_energy_tolerance + self.relative_energy_tolerance * scale

    def mass_tolerance(
        self,
        reference: Array,
        domain_measure: Array,
        /,
    ) -> Array:
        scale = jnp.maximum(jnp.abs(reference), jnp.abs(domain_measure))
        return self.absolute_mass_tolerance + self.relative_mass_tolerance * scale


class PhaseFieldResolutionEvidence(StrictModule, NonTrainableState):
    """Fixed-mesh diffuse-interface resolution evidence."""

    characteristic_width: Array
    transition_width: Array
    maximum_cell_diameter: Array
    cells_across_transition: Array
    minimum_cells: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        characteristic_width: ArrayLike,
        maximum_cell_diameter: ArrayLike,
        minimum_cells: float,
        /,
    ):
        width = jnp.asarray(characteristic_width)
        diameter = jnp.asarray(maximum_cell_diameter, dtype=width.dtype)
        minimum = float(minimum_cells)
        if width.shape != () or diameter.shape != ():
            raise ValueError("Phase-field resolution quantities must be scalar.")
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_cells must be positive and finite.")
        width_host = float(np.asarray(width))
        diameter_host = float(np.asarray(diameter))
        if (
            not math.isfinite(width_host)
            or width_host <= 0.0
            or not math.isfinite(diameter_host)
            or diameter_host <= 0.0
        ):
            raise ValueError("Phase-field resolution requires positive finite scales.")
        transition = 4.0 * width
        cells = transition / diameter
        passed = float(np.asarray(cells)) >= minimum
        self.characteristic_width = width
        self.transition_width = transition
        self.maximum_cell_diameter = diameter
        self.cells_across_transition = cells
        self.minimum_cells = minimum
        self.passed = passed
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "phase-field-resolution-evidence",
                "characteristic_width": width,
                "maximum_cell_diameter": diameter,
                "minimum_cells": minimum,
                "passed": passed,
            }
        )

    def require_supported(self, /) -> None:
        if not self.passed:
            raise ValueError(
                "Diffuse interface is underresolved for the production phase-field profile."
            )


class AllenCahnAcceptedState(StrictModule):
    phase: Array
    mass: Array
    energy: Array


class CahnHilliardAcceptedState(StrictModule):
    concentration: Array
    chemical_potential: Array
    reference_mass: Array
    cumulative_mass_source: Array
    mass: Array
    energy: Array


class PhaseFieldStepEvidence(StrictModule):
    nonlinear_successful: Array
    finite: Array
    energy_stable: Array
    mass_conserved: Array
    accepted: Array
    mass_before: Array
    mass_after: Array
    reference_mass: Array
    mass_defect: Array
    mass_tolerance: Array
    energy_before: Array
    energy_after: Array
    dissipation: Array
    energy_balance_defect: Array
    energy_tolerance: Array
    minimum_phase: Array
    maximum_phase: Array
    nonlinear_residual: Array
    nonlinear_iterations: Array
    nonlinear_work: Array
    ledger: PhaseFieldEnergyLedger
    boundary_work: Array
    stochastic_work: Array
    mass_source: Array
    mobility_successful: Array
    noise_successful: Array
    mass_conservation_required: bool = eqx.field(static=True)


class PhaseFieldStepResult(StrictModule):
    candidate_state: object
    accepted_state: object
    successful: Array
    evidence: PhaseFieldStepEvidence
    nonlinear_result: object


class PhaseFieldProductionCase(StrictModule, NonTrainableState):
    case_name: str = eqx.field(static=True)
    initial_state: object
    manifest: ProductionCaseManifest
    case_id: str = eqx.field(static=True)

    def __init__(
        self,
        case_name: str,
        initial_state: object,
        /,
        *,
        method_id: str,
        precision_id: str,
        topology_id: str,
        geometry_layout_id: str,
        dtype: str,
    ):
        name = str(case_name)
        if not name:
            raise ValueError("Phase-field production case name must be nonempty.")
        identifier = canonical_fingerprint(
            {
                "kind": "phase-field-production-case",
                "name": name,
                "method": method_id,
                "precision": precision_id,
                "topology": topology_id,
                "geometry_layout": geometry_layout_id,
                "dtype": dtype,
                "initial_state": array_tree_fingerprint(initial_state),
            }
        )
        self.case_name = name
        self.initial_state = initial_state
        self.manifest = ProductionCaseManifest(
            problem_id=identifier,
            method_id=method_id,
            precision_id=precision_id,
            topology_id=topology_id,
            geometry_layout_id=geometry_layout_id,
            dtype=dtype,
        )
        self.case_id = identifier


class _AllenCahnStepArguments(StrictModule):
    previous_value: tuple[Array, ...]
    previous_gradient: tuple[Array, ...]
    step_size: Array
    time: Array
    model: BinaryPhaseFieldModel
    mobility: AbstractPhaseFieldMobility
    block_names: tuple[str, ...] = eqx.field(static=True)
    user_args: object


class _CahnHilliardStepArguments(StrictModule):
    previous_value: tuple[Array, ...]
    previous_gradient: tuple[Array, ...]
    step_size: Array
    time: Array
    model: BinaryPhaseFieldModel
    mobility: AbstractPhaseFieldMobility
    block_names: tuple[str, ...] = eqx.field(static=True)
    user_args: object


def _previous_block(arguments, geometry):
    if geometry.block_name is None:
        if len(arguments.block_names) != 1:
            raise ValueError("Multiblock phase-field density requires block metadata.")
        index = 0
    else:
        index = arguments.block_names.index(geometry.block_name)
    return (
        arguments.previous_value[index],
        arguments.previous_gradient[index],
    )


def _gradient_incremental_density(model, current, previous):
    kappa = model.thermodynamics.gradient_coefficient.astype(current.dtype)
    if model.evolution_law.exact_identity:
        return (
            0.25
            * kappa
            * ein.contract("...d,...d->...", current + previous, current + previous)
        )
    return 0.5 * kappa * ein.contract("...d,...d->...", current, current)


def _allen_cahn_stationarity_density(jets, geometry, context):
    arguments = context.user_args
    if not isinstance(arguments, _AllenCahnStepArguments):
        raise TypeError("Allen-Cahn functional requires its dynamic step arguments.")
    phase = jets["phase"]
    if phase.value is None or phase.gradient is None:
        raise ValueError("Allen-Cahn functional requires value and gradient jets.")
    previous_value, previous_gradient = _previous_block(arguments, geometry)
    delta = phase.value - previous_value
    mobility = arguments.mobility.evaluate(
        previous_value,
        geometry.points,
        arguments.time,
        arguments.user_args,
    )
    if not arguments.mobility.scalar_kinetics:
        raise ValueError("Allen-Cahn kinetics require a scalar mobility law.")
    coefficient = mobility.tensor[..., 0, 0]
    energy = arguments.model.bulk_incremental_density(
        phase.value, previous_value
    ) + _gradient_incremental_density(arguments.model, phase.gradient, previous_gradient)
    return 0.5 * delta**2 / arguments.step_size + coefficient * energy


def _cahn_hilliard_stationarity_density(jets, geometry, context):
    arguments = context.user_args
    if not isinstance(arguments, _CahnHilliardStepArguments):
        raise TypeError("Cahn-Hilliard functional requires its dynamic step arguments.")
    concentration = jets["concentration"]
    chemical = jets["chemical_potential"]
    if (
        concentration.value is None
        or concentration.gradient is None
        or chemical.value is None
        or chemical.gradient is None
    ):
        raise ValueError("Cahn-Hilliard functional requires value and gradient jets.")
    previous_value, previous_gradient = _previous_block(arguments, geometry)
    mobility_quadratic, _ = arguments.mobility.quadratic(
        chemical.gradient,
        previous_value,
        geometry.points,
        arguments.time,
        arguments.user_args,
    )
    energy = arguments.model.bulk_incremental_density(
        concentration.value, previous_value
    ) + _gradient_incremental_density(
        arguments.model, concentration.gradient, previous_gradient
    )
    return (
        (concentration.value - previous_value) * chemical.value
        + 0.5 * arguments.step_size * mobility_quadratic
        - energy
    )


class _EnergyArguments(StrictModule):
    time: Array
    user_args: object


class _PhysicalCellEnergy(StrictModule):
    model: BinaryPhaseFieldModel

    def __call__(self, jets, geometry, context):
        del geometry, context
        phase = jets["phase"]
        if phase.value is None or phase.gradient is None:
            raise ValueError("Physical energy requires phase value and gradient.")
        return self.model.energy_density(phase.value, phase.gradient)


class _SurfaceStepDensity(StrictModule):
    surface: AbstractPhaseFieldSurfaceEnergy
    factor: float = eqx.field(static=True)

    def __call__(self, jets, geometry, context):
        arguments = context.user_args
        if not isinstance(
            arguments, (_AllenCahnStepArguments, _CahnHilliardStepArguments)
        ):
            raise TypeError("Surface stationarity requires phase-field step arguments.")
        phase = jets["phase"] if "phase" in jets else jets["concentration"]
        if phase.value is None:
            raise ValueError("Surface stationarity requires a phase trace.")
        return self.factor * self.surface.density(
            phase.value,
            geometry.points,
            arguments.time + arguments.step_size,
            arguments.user_args,
        )


class _PhysicalSurfaceDensity(StrictModule):
    surface: AbstractPhaseFieldSurfaceEnergy

    def __call__(self, jets, geometry, context):
        arguments = context.user_args
        if not isinstance(arguments, _EnergyArguments):
            raise TypeError("Surface energy requires energy-evaluation arguments.")
        phase = jets["phase"]
        if phase.value is None:
            raise ValueError("Surface energy requires a phase trace.")
        return self.surface.density(
            phase.value,
            geometry.points,
            arguments.time,
            arguments.user_args,
        )


class _FluxStepDensity(StrictModule):
    flux: PrescribedPhaseFieldFlux

    def __call__(self, jets, geometry, context):
        arguments = context.user_args
        if not isinstance(arguments, _CahnHilliardStepArguments):
            raise TypeError("Boundary flux requires Cahn-Hilliard step arguments.")
        chemical = jets["chemical_potential"]
        if chemical.value is None:
            raise ValueError("Boundary flux requires a chemical trace.")
        value = self.flux.evaluate(
            geometry.points,
            arguments.time + arguments.step_size,
            arguments.user_args,
        )
        return -arguments.step_size * value * chemical.value


class _FluxWorkDensity(StrictModule):
    flux: PrescribedPhaseFieldFlux

    def __call__(self, jets, geometry, context):
        arguments = context.user_args
        if not isinstance(arguments, _CahnHilliardStepArguments):
            raise TypeError("Boundary flux work requires step arguments.")
        chemical = jets["chemical_potential"]
        if chemical.value is None:
            raise ValueError("Boundary flux work requires a chemical trace.")
        value = self.flux.evaluate(
            geometry.points,
            arguments.time + arguments.step_size,
            arguments.user_args,
        )
        return arguments.step_size * value * chemical.value


class _FluxAmountDensity(StrictModule):
    flux: PrescribedPhaseFieldFlux

    def __call__(self, jets, geometry, context):
        arguments = context.user_args
        if not isinstance(arguments, _CahnHilliardStepArguments):
            raise TypeError("Boundary flux amount requires step arguments.")
        chemical = jets["chemical_potential"]
        if chemical.value is None:
            raise ValueError("Boundary flux amount requires a chemical trace.")
        value = self.flux.evaluate(
            geometry.points,
            arguments.time + arguments.step_size,
            arguments.user_args,
        )
        return arguments.step_size * value + 0.0 * chemical.value


def _allen_cahn_functional(
    boundary: PhaseFieldBoundaryPlan | None,
    mobility: AbstractPhaseFieldMobility,
) -> tuple[Functional, dict[str, IntegrationDomain | None]]:
    terms = [
        LocalIntegralTerm(
            "stationarity",
            region="cells",
            fields=(FieldJetSpec("phase", value=True, gradient=True),),
            density=_allen_cahn_stationarity_density,
            density_id="allen-cahn-energy-compatible-density",
        )
    ]
    regions: dict[str, IntegrationDomain | None] = {"cells": None}
    if boundary is not None:
        if not isinstance(mobility, ScalarPhaseFieldMobility):
            raise ValueError("Allen-Cahn surface laws require scalar mobility.")
        factor = float(np.asarray(mobility.value))
        for patch in boundary.surface_patches:
            region = f"surface/{patch.name}"
            terms.append(
                LocalIntegralTerm(
                    region,
                    region=region,
                    fields=(FieldJetSpec("phase", value=True),),
                    density=_SurfaceStepDensity(patch.surface_energy, factor),
                    density_id=f"allen-cahn-surface/{patch.patch_id}",
                )
            )
            regions[region] = patch.domain
    return (
        Functional(
            "allen-cahn-energy-compatible-step",
            tuple(terms),
            variable_fields=("phase",),
        ),
        regions,
    )


def _nonlinear_configuration_id(method: AbstractNonlinearMethod, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "nonlinear-method-configuration",
            "method": method.method_id,
            "structure": str(jax.tree.structure(method)),
            "arrays": array_tree_fingerprint(method),
        }
    )


def _cahn_hilliard_functional(
    boundary: PhaseFieldBoundaryPlan | None,
) -> tuple[Functional, dict[str, IntegrationDomain | None]]:
    terms = [
        LocalIntegralTerm(
            "stationarity",
            region="cells",
            fields=(
                FieldJetSpec("concentration", value=True, gradient=True),
                FieldJetSpec("chemical_potential", value=True, gradient=True),
            ),
            density=_cahn_hilliard_stationarity_density,
            density_id="cahn-hilliard-energy-compatible-density",
        )
    ]
    regions: dict[str, IntegrationDomain | None] = {"cells": None}
    if boundary is not None:
        for patch in boundary.surface_patches:
            region = f"surface/{patch.name}"
            terms.append(
                LocalIntegralTerm(
                    region,
                    region=region,
                    fields=(FieldJetSpec("concentration", value=True),),
                    density=_SurfaceStepDensity(patch.surface_energy, -1.0),
                    density_id=f"cahn-hilliard-surface/{patch.patch_id}",
                )
            )
            regions[region] = patch.domain
        for patch in boundary.flux_patches:
            region = f"flux/{patch.name}"
            terms.append(
                LocalIntegralTerm(
                    region,
                    region=region,
                    fields=(FieldJetSpec("chemical_potential", value=True),),
                    density=_FluxStepDensity(patch.mass_flux),
                    density_id=f"cahn-hilliard-flux/{patch.patch_id}",
                )
            )
            regions[region] = patch.domain
    return (
        Functional(
            "cahn-hilliard-energy-compatible-step",
            tuple(terms),
            variable_fields=("concentration", "chemical_potential"),
        ),
        regions,
    )


def _physical_energy_functional(
    model: BinaryPhaseFieldModel,
    boundary: PhaseFieldBoundaryPlan | None,
) -> tuple[Functional, dict[str, IntegrationDomain | None]]:
    terms = [
        LocalIntegralTerm(
            "bulk-gradient",
            region="cells",
            fields=(FieldJetSpec("phase", value=True, gradient=True),),
            density=_PhysicalCellEnergy(model),
            density_id=f"binary-physical-energy/{model.model_id}",
        )
    ]
    regions: dict[str, IntegrationDomain | None] = {"cells": None}
    if boundary is not None:
        for patch in boundary.surface_patches:
            region = f"surface/{patch.name}"
            terms.append(
                LocalIntegralTerm(
                    region,
                    region=region,
                    fields=(FieldJetSpec("phase", value=True),),
                    density=_PhysicalSurfaceDensity(patch.surface_energy),
                    density_id=f"physical-surface/{patch.patch_id}",
                )
            )
            regions[region] = patch.domain
    return Functional(
        "binary-phase-field-physical-energy",
        tuple(terms),
        variable_fields=("phase",),
    ), regions


def _flux_diagnostic_functional(
    boundary: PhaseFieldBoundaryPlan,
    *,
    work: bool,
) -> tuple[Functional, dict[str, IntegrationDomain | None]]:
    terms = []
    regions: dict[str, IntegrationDomain | None] = {}
    for patch in boundary.flux_patches:
        region = f"flux/{patch.name}"
        density = (
            _FluxWorkDensity(patch.mass_flux)
            if work
            else _FluxAmountDensity(patch.mass_flux)
        )
        terms.append(
            LocalIntegralTerm(
                region,
                region=region,
                fields=(FieldJetSpec("chemical_potential", value=True),),
                density=density,
                density_id=(
                    f"flux-work/{patch.patch_id}"
                    if work
                    else f"flux-amount/{patch.patch_id}"
                ),
            )
        )
        regions[region] = patch.domain
    return Functional(
        "phase-field-boundary-flux-work" if work else "phase-field-boundary-flux-amount",
        tuple(terms),
        variable_fields=("chemical_potential",),
    ), regions


def _termination_payload(termination: NonlinearTermination, /) -> dict[str, object]:
    return {
        "absolute_residual": termination.absolute_residual,
        "relative_residual": termination.relative_residual,
        "maximum_residual": termination.maximum_residual,
        "absolute_step": termination.absolute_step,
        "relative_step": termination.relative_step,
        "maximum_steps": termination.maximum_steps,
        "maximum_evaluations": termination.maximum_evaluations,
        "maximum_linear_iterations": termination.maximum_linear_iterations,
        "divergence_factor": termination.divergence_factor,
    }


def _validated_mobility(
    value: AbstractPhaseFieldMobility | ArrayLike,
    dtype,
    /,
) -> AbstractPhaseFieldMobility:
    mobility = as_phase_field_mobility(value)
    if isinstance(mobility, ScalarPhaseFieldMobility):
        mobility = ScalarPhaseFieldMobility(mobility.value.astype(dtype))
    return mobility


def _execution_policy(
    value: FiniteElementExecutionPolicy | None,
    /,
) -> FiniteElementExecutionPolicy:
    policy = (
        FiniteElementExecutionPolicy(
            realization="matrix_free",
            local_kernel="auto",
            accumulation="deterministic",
        )
        if value is None
        else value
    )
    if not isinstance(policy, FiniteElementExecutionPolicy):
        raise TypeError("execution_policy must be FiniteElementExecutionPolicy.")
    if policy.accumulation != "deterministic":
        raise ValueError(
            "Production phase-field execution requires deterministic accumulation."
        )
    return policy


def _nonlinear_method(
    value: AbstractNonlinearMethod | None, /
) -> AbstractNonlinearMethod:
    selected = NewtonKrylov() if value is None else value
    if not isinstance(selected, AbstractNonlinearMethod):
        raise TypeError("nonlinear must implement AbstractNonlinearMethod.")
    if not selected.capabilities.jit:
        raise ValueError("Production phase-field nonlinear solve must be JIT-compatible.")
    return selected


def _nonlinear_termination(value: NonlinearTermination | None, /) -> NonlinearTermination:
    selected = NonlinearTermination() if value is None else value
    if not isinstance(selected, NonlinearTermination):
        raise TypeError("termination must be NonlinearTermination.")
    return selected


def _acceptance_policy(
    value: PhaseFieldAcceptancePolicy | None,
    /,
) -> PhaseFieldAcceptancePolicy:
    selected = PhaseFieldAcceptancePolicy() if value is None else value
    if not isinstance(selected, PhaseFieldAcceptancePolicy):
        raise TypeError("acceptance must be PhaseFieldAcceptancePolicy.")
    return selected


def _validate_discretization(
    discretization: FiniteElementDiscretization,
    field_names: tuple[str, ...],
    /,
) -> tuple[int, ...]:
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    discretization.precision_policy.validate_backend()
    precision = discretization.precision_policy
    precision_values = (
        precision.storage_dtype,
        precision.geometry_dtype,
        precision.evaluation_dtype,
        precision.accumulation_dtype,
        precision.output_dtype,
    )
    if any(value != "float64" for value in precision_values):
        raise ValueError("Production phase-field execution requires float64 precision.")
    if not discretization.mesh.blocks:
        raise ValueError("Production phase-field execution requires cell blocks.")
    indices = tuple(discretization._field_index(name) for name in field_names)
    supported_cells = {
        "triangle",
        "quadrilateral",
        "tetrahedron",
        "hexahedron",
        "prism",
        "pyramid",
    }
    for index in indices:
        elements = discretization.elements[index]
        if len(elements) != len(discretization.mesh.blocks):
            raise ValueError("Phase-field elements must resolve every cell block.")
        for block, element in zip(discretization.mesh.blocks, elements, strict=True):
            if (
                element.family not in ("Lagrange", "TensorProductLagrange")
                or element.cell_kind != block.cell_kind
                or block.cell_kind not in supported_cells
                or element.degree < 1
                or element.conformity != "H1"
                or element.value_shape
            ):
                raise ValueError(
                    "Production phase-field fields require scalar conforming "
                    "Lagrange elements on supported cell blocks."
                )
        space = discretization.field_spaces[index].vector_space
        if not isinstance(space, ArraySpace) or len(space.shape) != 1:
            raise ValueError("Production phase-field fields must be scalar arrays.")
        if space.dtype != np.dtype("float64"):
            raise ValueError("Production phase-field fields require float64 storage.")
    if len(indices) == 2:
        first, second = indices
        if any(
            left.element_id != right.element_id
            for left, right in zip(
                discretization.elements[first],
                discretization.elements[second],
                strict=True,
            )
        ) or any(
            not np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(
                discretization.dof_maps[first].cell_dofs,
                discretization.dof_maps[second].cell_dofs,
                strict=True,
            )
        ):
            raise ValueError(
                "Cahn-Hilliard concentration and chemical fields must share one layout."
            )
    return indices


def _maximum_cell_diameter(discretization: FiniteElementDiscretization, /) -> float:
    coordinates = np.asarray(discretization.mesh.coordinates, dtype=np.float64)
    diameters = []
    for block in discretization.mesh.blocks:
        vertices = coordinates[np.asarray(block.vertices, dtype=np.int32)]
        differences = vertices[:, :, None, :] - vertices[:, None, :, :]
        squared = np.sum(differences * differences, axis=-1)
        diameters.append(float(np.sqrt(np.max(squared))))
    diameter = max(diameters)
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("Phase-field mesh has no positive finite cell diameter.")
    return diameter


def _resolution_evidence(
    model: BinaryPhaseFieldModel,
    discretization: FiniteElementDiscretization,
    minimum_cells: float,
    /,
) -> PhaseFieldResolutionEvidence:
    evidence = PhaseFieldResolutionEvidence(
        model.closure.characteristic_interface_width(model.thermodynamics),
        _maximum_cell_diameter(discretization),
        minimum_cells,
    )
    evidence.require_supported()
    return evidence


def _local_data(
    discretization: FiniteElementDiscretization,
    field_index: int,
    block_index: int,
    state: Array,
    /,
) -> Array:
    dofs = discretization.dof_maps[field_index].cell_dofs[block_index]
    orientation = discretization.dof_maps[field_index].orientations[block_index]
    return state[dofs] * orientation


def _quadrature_fields(
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array,
    /,
) -> tuple[tuple[Array, Array, Array, Array], ...]:
    blocks = []
    for block_index, geometry in enumerate(discretization.block_geometries[field_index]):
        local = _local_data(discretization, field_index, block_index, state)
        value = ein.contract("qi,ci->cq", geometry.basis_values, local)
        gradient = ein.contract("cqid,ci->cqd", geometry.physical_gradients, local)
        blocks.append(
            (
                value,
                gradient,
                geometry.physical_points,
                geometry.physical_weights,
            )
        )
    return tuple(blocks)


def _integral(
    discretization: FiniteElementDiscretization,
    values: tuple[Array, ...],
    weights: tuple[Array, ...],
    /,
) -> Array:
    contributions = tuple(
        discretization.precision_policy.sum(value * weight)
        for value, weight in zip(values, weights, strict=True)
    )
    return sum(contributions[1:], start=contributions[0])


def _mass(
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array,
    /,
) -> Array:
    blocks = _quadrature_fields(discretization, field_index, state)
    return _integral(
        discretization,
        tuple(block[0] for block in blocks),
        tuple(block[3] for block in blocks),
    )


def _energy_components(
    model: BinaryPhaseFieldModel,
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array,
    /,
) -> tuple[Array, Array]:
    blocks = _quadrature_fields(discretization, field_index, state)
    components = tuple(model.energy_components(block[0], block[1]) for block in blocks)
    weights = tuple(block[3] for block in blocks)
    bulk = _integral(
        discretization, tuple(component[0] for component in components), weights
    )
    gradient = _integral(
        discretization, tuple(component[1] for component in components), weights
    )
    return bulk, gradient


def _energy(
    model: BinaryPhaseFieldModel,
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array,
    /,
) -> Array:
    bulk, gradient = _energy_components(model, discretization, field_index, state)
    return bulk + gradient


def _domain_measure(
    discretization: FiniteElementDiscretization,
    field_index: int,
    /,
) -> Array:
    weights = tuple(
        geometry.physical_weights
        for geometry in discretization.block_geometries[field_index]
    )
    return sum(
        (discretization.precision_policy.sum(weight) for weight in weights[1:]),
        start=discretization.precision_policy.sum(weights[0]),
    )


def _reference_rules(
    discretization: FiniteElementDiscretization,
    field_indices: tuple[int, ...],
    /,
):
    factories = {
        "triangle": ReferenceTriangleRule,
        "quadrilateral": ReferenceQuadrilateralRule,
        "tetrahedron": ReferenceTetrahedronRule,
        "hexahedron": ReferenceHexahedronRule,
        "prism": ReferencePrismRule,
        "pyramid": ReferencePyramidRule,
    }
    rules = {}
    for block_index, block in enumerate(discretization.mesh.blocks):
        degree = max(
            discretization.elements[index][block_index].degree for index in field_indices
        )
        rules[block.name] = factories[block.cell_kind](
            GaussLegendreRule(max(degree + 1, 2))
        )
    return rules


def _production_run_plan(
    method: AbstractFixedStepMethod,
    /,
    *,
    step_size: float,
    end_time: float,
    maximum_steps: int,
    checkpoint_interval: int,
    segment_steps: int,
    retry_policy: RobustRetryPolicy | None,
) -> ProductionRunPlan:
    retry = RobustRetryPolicy(maximum_retries=2) if retry_policy is None else retry_policy
    return ProductionRunPlan(
        method,
        retry,
        step_size=step_size,
        end_time=end_time,
        maximum_steps=maximum_steps,
        checkpoint_interval=checkpoint_interval,
        segment_steps=segment_steps,
    )


class AllenCahnFEMPlan(StrictModule, NonTrainableState):
    """Prepare one closed, convex-split binary Allen-Cahn FE route."""

    model: BinaryPhaseFieldModel
    mobility: AbstractPhaseFieldMobility
    nonlinear: AbstractNonlinearMethod
    termination: NonlinearTermination
    acceptance: PhaseFieldAcceptancePolicy
    execution_policy: FiniteElementExecutionPolicy
    minimum_transition_cells: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: BinaryPhaseFieldModel,
        mobility: AbstractPhaseFieldMobility | ArrayLike,
        /,
        *,
        nonlinear: AbstractNonlinearMethod | None = None,
        termination: NonlinearTermination | None = None,
        acceptance: PhaseFieldAcceptancePolicy | None = None,
        execution_policy: FiniteElementExecutionPolicy | None = None,
        minimum_transition_cells: float = 4.0,
    ):
        if not isinstance(model, BinaryPhaseFieldModel):
            raise TypeError("model must be BinaryPhaseFieldModel.")
        mobility_ = _validated_mobility(
            mobility,
            model.thermodynamics.bulk_scale.dtype,
        )
        nonlinear_ = _nonlinear_method(nonlinear)
        termination_ = _nonlinear_termination(termination)
        acceptance_ = _acceptance_policy(acceptance)
        execution_ = _execution_policy(execution_policy)
        minimum = float(minimum_transition_cells)
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_transition_cells must be positive and finite.")
        self.model = model
        self.mobility = mobility_
        self.nonlinear = nonlinear_
        self.termination = termination_
        self.acceptance = acceptance_
        self.execution_policy = execution_
        self.minimum_transition_cells = minimum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "allen-cahn-fem-plan",
                "model": model.model_id,
                "mobility": mobility_.mobility_id,
                "nonlinear": _nonlinear_configuration_id(nonlinear_),
                "termination": _termination_payload(termination_),
                "acceptance": acceptance_.policy_id,
                "execution": execution_.policy_id,
                "minimum_transition_cells": minimum,
            }
        )

    def prepare(
        self,
        discretization: FiniteElementDiscretization,
        field_name: str,
        /,
        *,
        boundary: PhaseFieldBoundaryPlan | None = None,
        noise: PhaseFieldNoisePlan | None = None,
        constraints: Any = None,
    ) -> PreparedAllenCahnFEM:
        return PreparedAllenCahnFEM(
            self,
            discretization,
            field_name,
            boundary=boundary,
            noise=noise,
            constraints=constraints,
        )


class PreparedAllenCahnFEM(AbstractFixedStepMethod, NonTrainableState):
    """Compiled Allen-Cahn dynamics with physical accepted-step evidence."""

    plan: AllenCahnFEMPlan
    discretization: FiniteElementDiscretization
    compiled: CompiledFiniteElementProblem
    energy_compiled: CompiledFiniteElementProblem
    problem: NonlinearSystemProblem
    resolution: PhaseFieldResolutionEvidence
    domain_measure: Array
    boundary: PhaseFieldBoundaryPlan | None
    noise: PhaseFieldNoisePlan | None
    field_name: str = eqx.field(static=True)
    field_index: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: AllenCahnFEMPlan,
        discretization: FiniteElementDiscretization,
        field_name: str,
        /,
        *,
        boundary: PhaseFieldBoundaryPlan | None = None,
        noise: PhaseFieldNoisePlan | None = None,
        constraints: Any = None,
    ):
        if not isinstance(plan, AllenCahnFEMPlan):
            raise TypeError("plan must be AllenCahnFEMPlan.")
        name = str(field_name)
        if not name:
            raise ValueError("Allen-Cahn field name must be nonempty.")
        if boundary is not None and not isinstance(boundary, PhaseFieldBoundaryPlan):
            raise TypeError("boundary must be PhaseFieldBoundaryPlan or None.")
        if noise is not None and (
            not isinstance(noise, PhaseFieldNoisePlan) or noise.kind != "allen-cahn"
        ):
            raise TypeError("Allen-Cahn noise must be an Allen-Cahn noise plan.")
        (field_index,) = _validate_discretization(discretization, (name,))
        field_space = discretization.field_spaces[field_index].vector_space
        if not isinstance(field_space, ArraySpace):
            raise TypeError("Allen-Cahn field space must be ArraySpace.")
        if noise is not None and noise.basis.shape[0] != field_space.shape[0]:
            raise ValueError("Allen-Cahn stochastic basis does not match field DOFs.")
        resolution = _resolution_evidence(
            plan.model,
            discretization,
            plan.minimum_transition_cells,
        )
        rules = _reference_rules(discretization, (field_index,))
        functional, regions = _allen_cahn_functional(boundary, plan.mobility)
        regions["cells"] = discretization.cell_domain
        compiled = compile_finite_element_functional(
            functional,
            discretization,
            fields={"phase": name},
            regions=regions,
            rules={"cells": tuple(rules.items())},
            constraints=constraints,
            execution_policy=plan.execution_policy,
        )
        energy_functional, energy_regions = _physical_energy_functional(
            plan.model, boundary
        )
        energy_regions["cells"] = discretization.cell_domain
        energy_compiled = compile_finite_element_functional(
            energy_functional,
            discretization,
            fields={"phase": name},
            regions=energy_regions,
            rules={"cells": tuple(rules.items())},
            constraints=constraints,
            execution_policy=plan.execution_policy,
        )
        self.plan = plan
        self.discretization = discretization
        self.compiled = compiled
        self.energy_compiled = energy_compiled
        self.problem = compiled.as_nonlinear_problem()
        self.resolution = resolution
        self.domain_measure = _domain_measure(discretization, field_index)
        self.boundary = boundary
        self.noise = noise
        self.field_name = name
        self.field_index = field_index
        self.method_id = canonical_fingerprint(
            {
                "kind": "prepared-allen-cahn-fem",
                "plan": plan.plan_id,
                "discretization": discretization.prepared_id,
                "field": name,
                "compilation": compiled.compilation_id,
                "energy_compilation": energy_compiled.compilation_id,
                "boundary": None if boundary is None else boundary.boundary_plan_id,
                "noise": None if noise is None else noise.noise_plan_id,
                "resolution": resolution.evidence_id,
            }
        )

    @property
    def model(self) -> BinaryPhaseFieldModel:
        return self.plan.model

    @property
    def mobility(self) -> AbstractPhaseFieldMobility:
        return self.plan.mobility

    def mass(self, phase: ArrayLike, /) -> Array:
        value = self.discretization.field_spaces[self.field_index].vector_space.validate(
            phase
        )
        return _mass(self.discretization, self.field_index, value)

    def energy(
        self,
        phase: ArrayLike,
        /,
        *,
        time: ArrayLike = 0.0,
        args: object = None,
    ) -> Array:
        value = self.discretization.field_spaces[self.field_index].vector_space.validate(
            phase
        )
        return self.energy_compiled.potential(
            value,
            _EnergyArguments(jnp.asarray(time, dtype=value.dtype), args),
        )

    def initialize(self, phase: ArrayLike, /) -> AllenCahnAcceptedState:
        value = self.discretization.field_spaces[self.field_index].vector_space.validate(
            phase
        )
        if not bool(np.asarray(tree_allfinite(value))):
            raise ValueError("Allen-Cahn initial state must be finite.")
        mass = self.mass(value)
        energy = self.energy(value)
        if not bool(np.asarray(jnp.isfinite(mass) & jnp.isfinite(energy))):
            raise ValueError("Allen-Cahn initial diagnostics must be finite.")
        return AllenCahnAcceptedState(value, mass, energy)

    def step_detailed(
        self,
        step_index: Array,
        time: Array,
        state: AllenCahnAcceptedState,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> PhaseFieldStepResult:
        del step_index
        if not isinstance(state, AllenCahnAcceptedState):
            raise TypeError("Allen-Cahn step state must be AllenCahnAcceptedState.")
        step = jnp.asarray(step_size, dtype=state.phase.dtype)
        time_ = jnp.asarray(time, dtype=state.phase.dtype)
        if step.shape != () or time_.shape != ():
            raise ValueError("Allen-Cahn time and step size must be scalar.")
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Allen-Cahn step size must be positive and finite.",
        )
        if self.noise is None:
            effective_phase = state.phase
            stochastic_work = jnp.asarray(0.0, dtype=state.phase.dtype)
            noise_successful = jnp.asarray(True)
        else:
            noise = self.noise.increment(time_, time_ + step, dtype=state.phase.dtype)
            effective_phase = state.phase + noise.increment
            stochastic_work = self.energy(
                effective_phase, time=time_, args=args
            ) - self.energy(state.phase, time=time_, args=args)
            noise_successful = noise.successful
        previous_blocks = _quadrature_fields(
            self.discretization,
            self.field_index,
            effective_phase,
        )
        previous_value = tuple(block[0] for block in previous_blocks)
        previous_gradient = tuple(block[1] for block in previous_blocks)
        arguments = _AllenCahnStepArguments(
            previous_value,
            previous_gradient,
            step,
            time_,
            self.model,
            self.mobility,
            tuple(block.name for block in self.discretization.mesh.blocks),
            args,
        )
        nonlinear = self.plan.nonlinear.solve(
            self.problem,
            effective_phase,
            termination=self.plan.termination,
            args=arguments,
        )
        candidate_phase = nonlinear.state
        mass_after = self.mass(candidate_phase)
        energy_before = self.energy(state.phase, time=time_, args=args)
        energy_after = self.energy(candidate_phase, time=time_ + step, args=args)
        candidate_blocks = _quadrature_fields(
            self.discretization,
            self.field_index,
            candidate_phase,
        )
        dissipation_values = []
        mobility_successful = jnp.asarray(True)
        for previous, candidate_block in zip(
            previous_blocks, candidate_blocks, strict=True
        ):
            mobility = self.mobility.evaluate(previous[0], previous[2], time_, args)
            coefficient = mobility.tensor[..., 0, 0]
            dissipation_values.append(
                (candidate_block[0] - previous[0]) ** 2 / (coefficient * step)
            )
            mobility_successful = mobility_successful & mobility.successful
        dissipation = _integral(
            self.discretization,
            tuple(dissipation_values),
            tuple(block[3] for block in candidate_blocks),
        )
        bulk_before, gradient_before = _energy_components(
            self.model, self.discretization, self.field_index, state.phase
        )
        bulk_after, gradient_after = _energy_components(
            self.model, self.discretization, self.field_index, candidate_phase
        )
        surface_before = energy_before - bulk_before - gradient_before
        surface_after = energy_after - bulk_after - gradient_after
        boundary_work = self.energy(
            candidate_phase, time=time_ + step, args=args
        ) - self.energy(candidate_phase, time=time_, args=args)
        energy_tolerance = self.plan.acceptance.energy_tolerance(
            energy_before,
            energy_after,
        )
        ledger = PhaseFieldEnergyLedger(
            bulk_before=bulk_before,
            bulk_after=bulk_after,
            gradient_before=gradient_before,
            gradient_after=gradient_after,
            surface_before=surface_before,
            surface_after=surface_after,
            kinetic_dissipation=dissipation,
            boundary_work=boundary_work,
            stochastic_work=stochastic_work,
            tolerance=energy_tolerance,
            ledger_id=f"allen-cahn/{self.method_id}",
        )
        mass_defect = jnp.abs(mass_after - state.mass)
        mass_tolerance = self.plan.acceptance.mass_tolerance(
            state.mass,
            self.domain_measure,
        )
        candidate = AllenCahnAcceptedState(
            candidate_phase,
            mass_after,
            energy_after,
        )
        finite = (
            tree_allfinite(candidate)
            & ledger.finite
            & jnp.isfinite(nonlinear.diagnostics.final_residual_norm)
        )
        energy_stable = ledger.closed
        mass_conserved = jnp.asarray(True)
        successful = (
            nonlinear.successful
            & finite
            & energy_stable
            & mobility_successful
            & noise_successful
        )
        accepted = tree_where(successful, candidate, state)
        work = (
            nonlinear.diagnostics.residual_evaluations
            + nonlinear.diagnostics.jvp_evaluations
            + nonlinear.diagnostics.linear_iterations
        )
        evidence = PhaseFieldStepEvidence(
            nonlinear.successful,
            finite,
            energy_stable,
            mass_conserved,
            successful,
            state.mass,
            mass_after,
            state.mass,
            mass_defect,
            mass_tolerance,
            energy_before,
            energy_after,
            dissipation,
            ledger.total_residual,
            energy_tolerance,
            jnp.min(candidate_phase),
            jnp.max(candidate_phase),
            nonlinear.diagnostics.final_residual_norm,
            nonlinear.diagnostics.iterations,
            work,
            ledger,
            boundary_work,
            stochastic_work,
            jnp.asarray(0.0, dtype=state.phase.dtype),
            mobility_successful,
            noise_successful,
            False,
        )
        return PhaseFieldStepResult(
            candidate,
            accepted,
            successful,
            evidence,
            nonlinear,
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: AllenCahnAcceptedState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        result = self.step_detailed(step_index, time, state, step_size, args)
        evidence = result.evidence
        residual = jnp.maximum(
            evidence.nonlinear_residual,
            jnp.maximum(
                evidence.energy_balance_defect - evidence.energy_tolerance,
                0.0,
            ),
        )
        return FixedStepResult(
            result.candidate_state,
            result.accepted_state,
            result.successful,
            residual,
            evidence.nonlinear_iterations,
            evidence.nonlinear_work,
            jnp.asarray(False),
            jnp.zeros((), dtype=state.phase.dtype),
        )

    def production_case(
        self,
        case_name: str,
        initial_state: AllenCahnAcceptedState,
        /,
    ) -> PhaseFieldProductionCase:
        if not isinstance(initial_state, AllenCahnAcceptedState):
            raise TypeError("Allen-Cahn production case requires initialized state.")
        if not bool(np.asarray(tree_allfinite(initial_state))):
            raise ValueError("Allen-Cahn production case state must be finite.")
        return PhaseFieldProductionCase(
            case_name,
            initial_state,
            method_id=self.method_id,
            precision_id=self.discretization.precision_policy.policy_id,
            topology_id=self.discretization.default_runtime.topology_id,
            geometry_layout_id=(self.discretization.default_runtime.geometry_layout_id),
            dtype=jnp.dtype(initial_state.phase.dtype).name,
        )

    def production_run_plan(
        self,
        /,
        *,
        step_size: float,
        end_time: float,
        maximum_steps: int,
        checkpoint_interval: int,
        segment_steps: int = 32,
        retry_policy: RobustRetryPolicy | None = None,
    ) -> ProductionRunPlan:
        return _production_run_plan(
            self,
            step_size=step_size,
            end_time=end_time,
            maximum_steps=maximum_steps,
            checkpoint_interval=checkpoint_interval,
            segment_steps=segment_steps,
            retry_policy=retry_policy,
        )


class CahnHilliardFEMPlan(StrictModule, NonTrainableState):
    """Prepare one closed, convex-split binary Cahn-Hilliard FE route."""

    model: BinaryPhaseFieldModel
    mobility: AbstractPhaseFieldMobility
    nonlinear: AbstractNonlinearMethod
    termination: NonlinearTermination
    acceptance: PhaseFieldAcceptancePolicy
    execution_policy: FiniteElementExecutionPolicy
    minimum_transition_cells: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: BinaryPhaseFieldModel,
        mobility: AbstractPhaseFieldMobility | ArrayLike,
        /,
        *,
        nonlinear: AbstractNonlinearMethod | None = None,
        termination: NonlinearTermination | None = None,
        acceptance: PhaseFieldAcceptancePolicy | None = None,
        execution_policy: FiniteElementExecutionPolicy | None = None,
        minimum_transition_cells: float = 4.0,
    ):
        if not isinstance(model, BinaryPhaseFieldModel):
            raise TypeError("model must be BinaryPhaseFieldModel.")
        mobility_ = _validated_mobility(
            mobility,
            model.thermodynamics.bulk_scale.dtype,
        )
        nonlinear_ = _nonlinear_method(nonlinear)
        termination_ = _nonlinear_termination(termination)
        acceptance_ = _acceptance_policy(acceptance)
        execution_ = _execution_policy(execution_policy)
        minimum = float(minimum_transition_cells)
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_transition_cells must be positive and finite.")
        self.model = model
        self.mobility = mobility_
        self.nonlinear = nonlinear_
        self.termination = termination_
        self.acceptance = acceptance_
        self.execution_policy = execution_
        self.minimum_transition_cells = minimum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cahn-hilliard-fem-plan",
                "model": model.model_id,
                "mobility": mobility_.mobility_id,
                "nonlinear": _nonlinear_configuration_id(nonlinear_),
                "termination": _termination_payload(termination_),
                "acceptance": acceptance_.policy_id,
                "execution": execution_.policy_id,
                "minimum_transition_cells": minimum,
            }
        )

    def prepare(
        self,
        discretization: FiniteElementDiscretization,
        concentration_field: str,
        chemical_field: str,
        /,
        *,
        boundary: PhaseFieldBoundaryPlan | None = None,
        noise: PhaseFieldNoisePlan | None = None,
        constraints: Any = None,
    ) -> PreparedCahnHilliardFEM:
        return PreparedCahnHilliardFEM(
            self,
            discretization,
            concentration_field,
            chemical_field,
            boundary=boundary,
            noise=noise,
            constraints=constraints,
        )


class PreparedCahnHilliardFEM(AbstractFixedStepMethod, NonTrainableState):
    """Compiled Cahn-Hilliard dynamics with conservative accepted-step evidence."""

    plan: CahnHilliardFEMPlan
    discretization: FiniteElementDiscretization
    compiled: CompiledFiniteElementProblem
    energy_compiled: CompiledFiniteElementProblem
    flux_work_compiled: CompiledFiniteElementProblem | None
    flux_amount_compiled: CompiledFiniteElementProblem | None
    problem: NonlinearSystemProblem
    resolution: PhaseFieldResolutionEvidence
    domain_measure: Array
    boundary: PhaseFieldBoundaryPlan | None
    noise: PhaseFieldNoisePlan | None
    concentration_field: str = eqx.field(static=True)
    chemical_field: str = eqx.field(static=True)
    concentration_index: int = eqx.field(static=True)
    chemical_index: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: CahnHilliardFEMPlan,
        discretization: FiniteElementDiscretization,
        concentration_field: str,
        chemical_field: str,
        /,
        *,
        boundary: PhaseFieldBoundaryPlan | None = None,
        noise: PhaseFieldNoisePlan | None = None,
        constraints: Any = None,
    ):
        if not isinstance(plan, CahnHilliardFEMPlan):
            raise TypeError("plan must be CahnHilliardFEMPlan.")
        concentration = str(concentration_field)
        chemical = str(chemical_field)
        if not concentration or not chemical or concentration == chemical:
            raise ValueError("Cahn-Hilliard field names must be distinct and nonempty.")
        if boundary is not None and not isinstance(boundary, PhaseFieldBoundaryPlan):
            raise TypeError("boundary must be PhaseFieldBoundaryPlan or None.")
        if noise is not None and (
            not isinstance(noise, PhaseFieldNoisePlan) or noise.kind != "cahn-hilliard"
        ):
            raise TypeError("Cahn-Hilliard noise must be a Cahn-Hilliard noise plan.")
        concentration_index, chemical_index = _validate_discretization(
            discretization,
            (concentration, chemical),
        )
        concentration_space = discretization.field_spaces[
            concentration_index
        ].vector_space
        if not isinstance(concentration_space, ArraySpace):
            raise TypeError("Cahn-Hilliard concentration space must be ArraySpace.")
        if noise is not None and noise.basis.shape[0] != concentration_space.shape[0]:
            raise ValueError("Cahn-Hilliard stochastic basis does not match field DOFs.")
        resolution = _resolution_evidence(
            plan.model,
            discretization,
            plan.minimum_transition_cells,
        )
        rules = _reference_rules(discretization, (concentration_index, chemical_index))
        functional, regions = _cahn_hilliard_functional(boundary)
        regions["cells"] = discretization.cell_domain
        compiled = compile_finite_element_functional(
            functional,
            discretization,
            fields={
                "concentration": concentration,
                "chemical_potential": chemical,
            },
            regions=regions,
            rules={"cells": tuple(rules.items())},
            constraints=constraints,
            execution_policy=plan.execution_policy,
        )
        energy_functional, energy_regions = _physical_energy_functional(
            plan.model, boundary
        )
        energy_regions["cells"] = discretization.cell_domain
        energy_compiled = compile_finite_element_functional(
            energy_functional,
            discretization,
            fields={"phase": concentration},
            regions=energy_regions,
            rules={"cells": tuple(rules.items())},
            constraints=(
                None
                if constraints is None or concentration not in constraints
                else {concentration: constraints[concentration]}
            ),
            execution_policy=plan.execution_policy,
        )
        flux_work_compiled = None
        flux_amount_compiled = None
        if boundary is not None and boundary.flux_patches:
            chemical_constraints = (
                None
                if constraints is None or chemical not in constraints
                else {chemical: constraints[chemical]}
            )
            work_functional, work_regions = _flux_diagnostic_functional(
                boundary, work=True
            )
            flux_work_compiled = compile_finite_element_functional(
                work_functional,
                discretization,
                fields={"chemical_potential": chemical},
                regions=work_regions,
                constraints=chemical_constraints,
                execution_policy=plan.execution_policy,
            )
            amount_functional, amount_regions = _flux_diagnostic_functional(
                boundary, work=False
            )
            flux_amount_compiled = compile_finite_element_functional(
                amount_functional,
                discretization,
                fields={"chemical_potential": chemical},
                regions=amount_regions,
                constraints=chemical_constraints,
                execution_policy=plan.execution_policy,
            )
        self.plan = plan
        self.discretization = discretization
        self.compiled = compiled
        self.energy_compiled = energy_compiled
        self.flux_work_compiled = flux_work_compiled
        self.flux_amount_compiled = flux_amount_compiled
        self.problem = compiled.as_nonlinear_problem()
        self.resolution = resolution
        self.domain_measure = _domain_measure(discretization, concentration_index)
        self.boundary = boundary
        self.noise = noise
        self.concentration_field = concentration
        self.chemical_field = chemical
        self.concentration_index = concentration_index
        self.chemical_index = chemical_index
        self.method_id = canonical_fingerprint(
            {
                "kind": "prepared-cahn-hilliard-fem",
                "plan": plan.plan_id,
                "discretization": discretization.prepared_id,
                "concentration_field": concentration,
                "chemical_field": chemical,
                "compilation": compiled.compilation_id,
                "energy_compilation": energy_compiled.compilation_id,
                "boundary": None if boundary is None else boundary.boundary_plan_id,
                "noise": None if noise is None else noise.noise_plan_id,
                "resolution": resolution.evidence_id,
            }
        )

    @property
    def model(self) -> BinaryPhaseFieldModel:
        return self.plan.model

    @property
    def mobility(self) -> AbstractPhaseFieldMobility:
        return self.plan.mobility

    def mass(self, concentration: ArrayLike, /) -> Array:
        value = self.discretization.field_spaces[
            self.concentration_index
        ].vector_space.validate(concentration)
        return _mass(self.discretization, self.concentration_index, value)

    def energy(
        self,
        concentration: ArrayLike,
        /,
        *,
        time: ArrayLike = 0.0,
        args: object = None,
    ) -> Array:
        value = self.discretization.field_spaces[
            self.concentration_index
        ].vector_space.validate(concentration)
        return self.energy_compiled.potential(
            value,
            _EnergyArguments(jnp.asarray(time, dtype=value.dtype), args),
        )

    def initialize(
        self,
        concentration: ArrayLike,
        /,
        *,
        chemical_potential: ArrayLike | None = None,
    ) -> CahnHilliardAcceptedState:
        concentration_space = self.discretization.field_spaces[
            self.concentration_index
        ].vector_space
        chemical_space = self.discretization.field_spaces[
            self.chemical_index
        ].vector_space
        value = concentration_space.validate(concentration)
        chemical = (
            chemical_space.zeros()
            if chemical_potential is None
            else chemical_space.validate(chemical_potential)
        )
        if not bool(np.asarray(tree_allfinite((value, chemical)))):
            raise ValueError("Cahn-Hilliard initial fields must be finite.")
        mass = self.mass(value)
        energy = self.energy(value)
        if not bool(np.asarray(jnp.isfinite(mass) & jnp.isfinite(energy))):
            raise ValueError("Cahn-Hilliard initial diagnostics must be finite.")
        return CahnHilliardAcceptedState(
            value,
            chemical,
            mass,
            jnp.asarray(0.0, dtype=mass.dtype),
            mass,
            energy,
        )

    def step_detailed(
        self,
        step_index: Array,
        time: Array,
        state: CahnHilliardAcceptedState,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> PhaseFieldStepResult:
        del step_index
        if not isinstance(state, CahnHilliardAcceptedState):
            raise TypeError("Cahn-Hilliard step state must be CahnHilliardAcceptedState.")
        step = jnp.asarray(step_size, dtype=state.concentration.dtype)
        time_ = jnp.asarray(time, dtype=state.concentration.dtype)
        if step.shape != () or time_.shape != ():
            raise ValueError("Cahn-Hilliard time and step size must be scalar.")
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Cahn-Hilliard step size must be positive and finite.",
        )
        if self.noise is None:
            effective_concentration = state.concentration
            stochastic_work = jnp.asarray(0.0, dtype=state.concentration.dtype)
            noise_successful = jnp.asarray(True)
        else:
            noise = self.noise.increment(
                time_, time_ + step, dtype=state.concentration.dtype
            )
            effective_concentration = state.concentration + noise.increment
            stochastic_work = self.energy(
                effective_concentration, time=time_, args=args
            ) - self.energy(state.concentration, time=time_, args=args)
            noise_successful = noise.successful
        previous_blocks = _quadrature_fields(
            self.discretization,
            self.concentration_index,
            effective_concentration,
        )
        previous_value = tuple(block[0] for block in previous_blocks)
        previous_gradient = tuple(block[1] for block in previous_blocks)
        arguments = _CahnHilliardStepArguments(
            previous_value,
            previous_gradient,
            step,
            time_,
            self.model,
            self.mobility,
            tuple(block.name for block in self.discretization.mesh.blocks),
            args,
        )
        nonlinear = self.plan.nonlinear.solve(
            self.problem,
            (effective_concentration, state.chemical_potential),
            termination=self.plan.termination,
            args=arguments,
        )
        candidate_concentration, candidate_chemical = nonlinear.state
        mass_after = self.mass(candidate_concentration)
        energy_before = self.energy(state.concentration, time=time_, args=args)
        energy_after = self.energy(candidate_concentration, time=time_ + step, args=args)
        chemical_blocks = _quadrature_fields(
            self.discretization,
            self.chemical_index,
            candidate_chemical,
        )
        dissipation_values = []
        mobility_successful = jnp.asarray(True)
        for previous, chemical_block in zip(
            previous_blocks, chemical_blocks, strict=True
        ):
            quadratic, mobility = self.mobility.quadratic(
                chemical_block[1],
                previous[0],
                chemical_block[2],
                time_,
                args,
            )
            dissipation_values.append(step * quadratic)
            mobility_successful = mobility_successful & mobility.successful
        dissipation = _integral(
            self.discretization,
            tuple(dissipation_values),
            tuple(block[3] for block in chemical_blocks),
        )
        flux_work = (
            jnp.asarray(0.0, dtype=energy_after.dtype)
            if self.flux_work_compiled is None
            else self.flux_work_compiled.potential(candidate_chemical, arguments)
        )
        mass_source = (
            jnp.asarray(0.0, dtype=mass_after.dtype)
            if self.flux_amount_compiled is None
            else self.flux_amount_compiled.potential(candidate_chemical, arguments)
        )
        cumulative_source = state.cumulative_mass_source + mass_source
        mass_target = state.reference_mass + cumulative_source
        mass_defect = jnp.abs(mass_after - mass_target)
        mass_tolerance = self.plan.acceptance.mass_tolerance(
            mass_target,
            self.domain_measure,
        )
        bulk_before, gradient_before = _energy_components(
            self.model,
            self.discretization,
            self.concentration_index,
            state.concentration,
        )
        bulk_after, gradient_after = _energy_components(
            self.model,
            self.discretization,
            self.concentration_index,
            candidate_concentration,
        )
        surface_before = energy_before - bulk_before - gradient_before
        surface_after = energy_after - bulk_after - gradient_after
        explicit_surface_work = self.energy(
            candidate_concentration, time=time_ + step, args=args
        ) - self.energy(candidate_concentration, time=time_, args=args)
        boundary_work = explicit_surface_work + flux_work
        energy_tolerance = self.plan.acceptance.energy_tolerance(
            energy_before,
            energy_after,
        )
        ledger = PhaseFieldEnergyLedger(
            bulk_before=bulk_before,
            bulk_after=bulk_after,
            gradient_before=gradient_before,
            gradient_after=gradient_after,
            surface_before=surface_before,
            surface_after=surface_after,
            diffusion_dissipation=dissipation,
            boundary_work=boundary_work,
            stochastic_work=stochastic_work,
            tolerance=energy_tolerance,
            ledger_id=f"cahn-hilliard/{self.method_id}",
        )
        candidate = CahnHilliardAcceptedState(
            candidate_concentration,
            candidate_chemical,
            state.reference_mass,
            cumulative_source,
            mass_after,
            energy_after,
        )
        finite = (
            tree_allfinite(candidate)
            & ledger.finite
            & jnp.isfinite(mass_source)
            & jnp.isfinite(nonlinear.diagnostics.final_residual_norm)
        )
        energy_stable = ledger.closed
        mass_conserved = mass_defect <= mass_tolerance
        successful = (
            nonlinear.successful
            & finite
            & energy_stable
            & mass_conserved
            & mobility_successful
            & noise_successful
        )
        accepted = tree_where(successful, candidate, state)
        work = (
            nonlinear.diagnostics.residual_evaluations
            + nonlinear.diagnostics.jvp_evaluations
            + nonlinear.diagnostics.linear_iterations
        )
        evidence = PhaseFieldStepEvidence(
            nonlinear.successful,
            finite,
            energy_stable,
            mass_conserved,
            successful,
            state.mass,
            mass_after,
            mass_target,
            mass_defect,
            mass_tolerance,
            energy_before,
            energy_after,
            dissipation,
            ledger.total_residual,
            energy_tolerance,
            jnp.min(candidate_concentration),
            jnp.max(candidate_concentration),
            nonlinear.diagnostics.final_residual_norm,
            nonlinear.diagnostics.iterations,
            work,
            ledger,
            boundary_work,
            stochastic_work,
            mass_source,
            mobility_successful,
            noise_successful,
            True,
        )
        return PhaseFieldStepResult(
            candidate,
            accepted,
            successful,
            evidence,
            nonlinear,
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: CahnHilliardAcceptedState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        result = self.step_detailed(step_index, time, state, step_size, args)
        evidence = result.evidence
        residual = jnp.maximum(
            evidence.nonlinear_residual,
            jnp.maximum(
                jnp.maximum(
                    evidence.energy_balance_defect - evidence.energy_tolerance,
                    0.0,
                ),
                jnp.maximum(
                    evidence.mass_defect - evidence.mass_tolerance,
                    0.0,
                ),
            ),
        )
        return FixedStepResult(
            result.candidate_state,
            result.accepted_state,
            result.successful,
            residual,
            evidence.nonlinear_iterations,
            evidence.nonlinear_work,
            jnp.asarray(False),
            jnp.zeros((), dtype=state.concentration.dtype),
        )

    def production_case(
        self,
        case_name: str,
        initial_state: CahnHilliardAcceptedState,
        /,
    ) -> PhaseFieldProductionCase:
        if not isinstance(initial_state, CahnHilliardAcceptedState):
            raise TypeError("Cahn-Hilliard production case requires initialized state.")
        if not bool(np.asarray(tree_allfinite(initial_state))):
            raise ValueError("Cahn-Hilliard production case state must be finite.")
        return PhaseFieldProductionCase(
            case_name,
            initial_state,
            method_id=self.method_id,
            precision_id=self.discretization.precision_policy.policy_id,
            topology_id=self.discretization.default_runtime.topology_id,
            geometry_layout_id=(self.discretization.default_runtime.geometry_layout_id),
            dtype=jnp.dtype(initial_state.concentration.dtype).name,
        )

    def production_run_plan(
        self,
        /,
        *,
        step_size: float,
        end_time: float,
        maximum_steps: int,
        checkpoint_interval: int,
        segment_steps: int = 32,
        retry_policy: RobustRetryPolicy | None = None,
    ) -> ProductionRunPlan:
        return _production_run_plan(
            self,
            step_size=step_size,
            end_time=end_time,
            maximum_steps=maximum_steps,
            checkpoint_interval=checkpoint_interval,
            segment_steps=segment_steps,
            retry_policy=retry_policy,
        )


__all__ = [
    "AllenCahnAcceptedState",
    "AllenCahnFEMPlan",
    "BinaryPhaseFieldModel",
    "CahnHilliardAcceptedState",
    "CahnHilliardFEMPlan",
    "PhaseFieldAcceptancePolicy",
    "PhaseFieldProductionCase",
    "PhaseFieldResolutionEvidence",
    "PhaseFieldStepEvidence",
    "PhaseFieldStepResult",
    "PreparedAllenCahnFEM",
    "PreparedCahnHilliardFEM",
]
