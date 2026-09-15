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
from ...discretization import FiniteElementDiscretization
from ...equations import (
    compile_finite_element_functional,
    CompiledFiniteElementProblem,
    FiniteElementExecutionPolicy,
)
from ...integration import GaussLegendreRule, ReferenceTriangleRule
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


class BinaryPhaseFieldModel(StrictModule, NonTrainableState):
    """One binary free energy shared by phase-field dynamics and diagnostics."""

    thermodynamics: BinaryThermodynamicParameters
    closure: BinaryPhaseThermodynamicClosure
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics: BinaryThermodynamicParameters,
        /,
        *,
        closure: BinaryPhaseThermodynamicClosure | None = None,
    ):
        if not isinstance(thermodynamics, BinaryThermodynamicParameters):
            raise TypeError("thermodynamics must be BinaryThermodynamicParameters.")
        selected = BinaryPhaseThermodynamicClosure() if closure is None else closure
        if not isinstance(selected, BinaryPhaseThermodynamicClosure):
            raise TypeError("closure must be BinaryPhaseThermodynamicClosure.")
        self.thermodynamics = thermodynamics
        self.closure = selected
        self.model_id = canonical_fingerprint(
            {
                "kind": "binary-phase-field-model",
                "closure": selected.closure_id,
                "thermodynamics": {
                    "bulk_scale": thermodynamics.bulk_scale,
                    "gradient_coefficient": thermodynamics.gradient_coefficient,
                    "wetting_strength": thermodynamics.wetting_strength,
                },
            }
        )

    @property
    def effective_bulk_scale(self) -> Array:
        """Return the complete quartic prefactor, including free-energy scale."""

        if not isinstance(self.closure.free_energy, DoubleWellFreeEnergy):
            raise TypeError("Effective quartic scale requires DoubleWellFreeEnergy.")
        dtype = self.thermodynamics.bulk_scale.dtype
        return self.thermodynamics.bulk_scale * self.closure.free_energy.scale.astype(
            dtype
        )

    def energy_density(self, phase: Array, gradient: Array, /) -> Array:
        """Return canonical bulk-plus-gradient physical energy density."""

        value = jnp.asarray(phase)
        grad = jnp.asarray(gradient, dtype=value.dtype)
        if grad.shape[:-1] != value.shape:
            raise ValueError("Phase-field gradient shape must extend the phase shape.")
        bulk = self.thermodynamics.bulk_scale.astype(value.dtype)
        kappa = self.thermodynamics.gradient_coefficient.astype(value.dtype)
        return bulk * self.closure.free_energy.density(
            value
        ) + 0.5 * kappa * ein.contract("...d,...d->...", grad, grad)

    def convex_split_bulk_derivative(
        self,
        current: Array,
        previous: Array,
        /,
    ) -> Array:
        """Return the quartic convex-current/concave-previous bulk derivative."""

        if not isinstance(self.closure.free_energy, DoubleWellFreeEnergy):
            raise TypeError(
                "Convex-split phase-field dynamics require DoubleWellFreeEnergy."
            )
        value = jnp.asarray(current)
        old = jnp.asarray(previous, dtype=value.dtype)
        return self.effective_bulk_scale.astype(value.dtype) * (value**3 - old)

    def require_production_supported(self, /) -> None:
        """Reject constitutive features absent from the qualified closed system."""

        if not isinstance(self.closure.free_energy, DoubleWellFreeEnergy):
            raise TypeError(
                "Production phase-field dynamics require DoubleWellFreeEnergy."
            )
        wetting = float(np.asarray(self.thermodynamics.wetting_strength))
        if wetting != 0.0:
            raise ValueError(
                "Production phase-field dynamics do not yet include wetting energy."
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
                "Diffuse interface is underresolved for the production phase-field "
                "profile."
            )


class AllenCahnAcceptedState(StrictModule):
    phase: Array
    mass: Array
    energy: Array


class CahnHilliardAcceptedState(StrictModule):
    concentration: Array
    chemical_potential: Array
    reference_mass: Array
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
    previous_value: Array
    step_size: Array
    model: BinaryPhaseFieldModel
    mobility: Array


class _CahnHilliardStepArguments(StrictModule):
    previous_value: Array
    step_size: Array
    model: BinaryPhaseFieldModel
    mobility: Array


def _allen_cahn_stationarity_density(jets, geometry, context):
    del geometry
    arguments = context.user_args
    if not isinstance(arguments, _AllenCahnStepArguments):
        raise TypeError("Allen-Cahn functional requires its dynamic step arguments.")
    phase = jets["phase"]
    if phase.value is None or phase.gradient is None:
        raise ValueError("Allen-Cahn functional requires value and gradient jets.")
    delta = phase.value - arguments.previous_value
    kappa = arguments.model.thermodynamics.gradient_coefficient.astype(phase.value.dtype)
    gradient_squared = ein.contract("...d,...d->...", phase.gradient, phase.gradient)
    bulk = arguments.model.effective_bulk_scale.astype(phase.value.dtype) * (
        0.25 * phase.value**4 - arguments.previous_value * phase.value
    )
    return (
        0.5 * delta**2 / arguments.step_size
        + arguments.mobility * bulk
        + 0.5 * arguments.mobility * kappa * gradient_squared
    )


def _cahn_hilliard_stationarity_density(jets, geometry, context):
    del geometry
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
    kappa = arguments.model.thermodynamics.gradient_coefficient.astype(
        concentration.value.dtype
    )
    bulk = arguments.model.effective_bulk_scale.astype(concentration.value.dtype) * (
        0.25 * concentration.value**4 - arguments.previous_value * concentration.value
    )
    chemical_gradient_squared = ein.contract(
        "...d,...d->...", chemical.gradient, chemical.gradient
    )
    concentration_gradient_squared = ein.contract(
        "...d,...d->...", concentration.gradient, concentration.gradient
    )
    return (
        (concentration.value - arguments.previous_value) * chemical.value
        + 0.5 * arguments.step_size * arguments.mobility * chemical_gradient_squared
        - bulk
        - 0.5 * kappa * concentration_gradient_squared
    )


def _allen_cahn_functional() -> Functional:
    return Functional(
        "allen-cahn-convex-split-step",
        (
            LocalIntegralTerm(
                "stationarity",
                region="cells",
                fields=(FieldJetSpec("phase", value=True, gradient=True),),
                density=_allen_cahn_stationarity_density,
                density_id="allen-cahn-convex-split-density",
            ),
        ),
        variable_fields=("phase",),
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


def _cahn_hilliard_functional() -> Functional:
    return Functional(
        "cahn-hilliard-convex-split-step",
        (
            LocalIntegralTerm(
                "stationarity",
                region="cells",
                fields=(
                    FieldJetSpec("concentration", value=True, gradient=True),
                    FieldJetSpec("chemical_potential", value=True, gradient=True),
                ),
                density=_cahn_hilliard_stationarity_density,
                density_id="cahn-hilliard-convex-split-density",
            ),
        ),
        variable_fields=("concentration", "chemical_potential"),
    )


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


def _validated_mobility(value: ArrayLike, dtype, /) -> Array:
    mobility = jnp.asarray(value, dtype=dtype)
    if mobility.shape != ():
        raise ValueError("Phase-field mobility must be scalar.")
    host = float(np.asarray(mobility))
    if not math.isfinite(host) or host <= 0.0:
        raise ValueError("Phase-field mobility must be positive and finite.")
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
    if len(discretization.mesh.blocks) != 1:
        raise ValueError(
            "Production phase-field execution requires one homogeneous cell block."
        )
    block = discretization.mesh.blocks[0]
    if block.cell_kind != "triangle":
        raise ValueError("Production phase-field execution requires triangle cells.")
    indices = tuple(discretization._field_index(name) for name in field_names)
    for index in indices:
        elements = discretization.elements[index]
        if len(elements) != 1:
            raise ValueError("Phase-field fields require one resolved element.")
        element = elements[0]
        if (
            element.family != "Lagrange"
            or element.cell_kind != "triangle"
            or element.degree != 1
            or element.conformity != "H1"
            or element.value_shape
        ):
            raise ValueError(
                "Production phase-field fields require scalar conforming P1 triangles."
            )
        space = discretization.field_spaces[index].vector_space
        if not isinstance(space, ArraySpace) or len(space.shape) != 1:
            raise ValueError("Production phase-field fields must be scalar arrays.")
        if space.dtype != np.dtype("float64"):
            raise ValueError("Production phase-field fields require float64 storage.")
    if len(indices) == 2:
        first, second = indices
        if discretization.elements[first][0].element_id != discretization.elements[
            second
        ][0].element_id or not np.array_equal(
            np.asarray(discretization.dof_maps[first].cell_dofs[0]),
            np.asarray(discretization.dof_maps[second].cell_dofs[0]),
        ):
            raise ValueError(
                "Cahn-Hilliard concentration and chemical fields must share one P1 layout."
            )
    return indices


def _maximum_cell_diameter(discretization: FiniteElementDiscretization, /) -> float:
    block = discretization.mesh.blocks[0]
    coordinates = np.asarray(discretization.mesh.coordinates, dtype=float)
    vertices = coordinates[np.asarray(block.vertices, dtype=np.int32)]
    differences = vertices[:, :, None, :] - vertices[:, None, :, :]
    squared = np.sum(differences * differences, axis=-1)
    diameter = float(np.sqrt(np.max(squared)))
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
    state: Array,
    /,
) -> Array:
    dofs = discretization.dof_maps[field_index].cell_dofs[0]
    orientation = discretization.dof_maps[field_index].orientations[0]
    return state[dofs] * orientation


def _quadrature_fields(
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array,
    /,
) -> tuple[Array, Array]:
    local = _local_data(discretization, field_index, state)
    geometry = discretization.block_geometries[field_index][0]
    value = ein.contract("qi,ci->cq", geometry.basis_values, local)
    gradient = ein.contract("cqid,ci->cqd", geometry.physical_gradients, local)
    return value, gradient


def _integral(
    discretization: FiniteElementDiscretization,
    field_index: int,
    values: Array,
    /,
) -> Array:
    weights = discretization.block_geometries[field_index][0].physical_weights
    return discretization.precision_policy.sum(values * weights)


def _mass(
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array,
    /,
) -> Array:
    value, _ = _quadrature_fields(discretization, field_index, state)
    return _integral(discretization, field_index, value)


def _energy(
    model: BinaryPhaseFieldModel,
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array,
    /,
) -> Array:
    value, gradient = _quadrature_fields(discretization, field_index, state)
    return _integral(
        discretization,
        field_index,
        model.energy_density(value, gradient),
    )


def _domain_measure(
    discretization: FiniteElementDiscretization,
    field_index: int,
    /,
) -> Array:
    weights = discretization.block_geometries[field_index][0].physical_weights
    return discretization.precision_policy.sum(weights)


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
    mobility: Array
    nonlinear: AbstractNonlinearMethod
    termination: NonlinearTermination
    acceptance: PhaseFieldAcceptancePolicy
    execution_policy: FiniteElementExecutionPolicy
    minimum_transition_cells: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: BinaryPhaseFieldModel,
        mobility: ArrayLike,
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
        model.require_production_supported()
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
                "mobility": mobility_,
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
    ) -> PreparedAllenCahnFEM:
        return PreparedAllenCahnFEM(self, discretization, field_name)


class PreparedAllenCahnFEM(AbstractFixedStepMethod):
    """Compiled Allen-Cahn dynamics with physical accepted-step evidence."""

    plan: AllenCahnFEMPlan
    discretization: FiniteElementDiscretization
    compiled: CompiledFiniteElementProblem
    problem: NonlinearSystemProblem
    resolution: PhaseFieldResolutionEvidence
    domain_measure: Array
    field_name: str = eqx.field(static=True)
    field_index: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: AllenCahnFEMPlan,
        discretization: FiniteElementDiscretization,
        field_name: str,
        /,
    ):
        if not isinstance(plan, AllenCahnFEMPlan):
            raise TypeError("plan must be AllenCahnFEMPlan.")
        name = str(field_name)
        if not name:
            raise ValueError("Allen-Cahn field name must be nonempty.")
        (field_index,) = _validate_discretization(discretization, (name,))
        resolution = _resolution_evidence(
            plan.model,
            discretization,
            plan.minimum_transition_cells,
        )
        compiled = compile_finite_element_functional(
            _allen_cahn_functional(),
            discretization,
            fields={"phase": name},
            regions={"cells": discretization.cell_domain},
            rules={
                "cells": (
                    (
                        discretization.mesh.blocks[0].name,
                        ReferenceTriangleRule(GaussLegendreRule(2)),
                    ),
                )
            },
            execution_policy=plan.execution_policy,
        )
        self.plan = plan
        self.discretization = discretization
        self.compiled = compiled
        self.problem = compiled.as_nonlinear_problem()
        self.resolution = resolution
        self.domain_measure = _domain_measure(discretization, field_index)
        self.field_name = name
        self.field_index = field_index
        self.method_id = canonical_fingerprint(
            {
                "kind": "prepared-allen-cahn-fem",
                "plan": plan.plan_id,
                "discretization": discretization.prepared_id,
                "field": name,
                "compilation": compiled.compilation_id,
                "resolution": resolution.evidence_id,
            }
        )

    @property
    def model(self) -> BinaryPhaseFieldModel:
        return self.plan.model

    @property
    def mobility(self) -> Array:
        return self.plan.mobility

    def mass(self, phase: ArrayLike, /) -> Array:
        value = self.discretization.field_spaces[self.field_index].vector_space.validate(
            phase
        )
        return _mass(self.discretization, self.field_index, value)

    def energy(self, phase: ArrayLike, /) -> Array:
        value = self.discretization.field_spaces[self.field_index].vector_space.validate(
            phase
        )
        return _energy(self.model, self.discretization, self.field_index, value)

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
        del step_index, time, args
        if not isinstance(state, AllenCahnAcceptedState):
            raise TypeError("Allen-Cahn step state must be AllenCahnAcceptedState.")
        step = jnp.asarray(step_size, dtype=state.phase.dtype)
        if step.shape != ():
            raise ValueError("Allen-Cahn step size must be scalar.")
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Allen-Cahn step size must be positive and finite.",
        )
        previous_value, _ = _quadrature_fields(
            self.discretization,
            self.field_index,
            state.phase,
        )
        arguments = _AllenCahnStepArguments(
            previous_value,
            step,
            self.model,
            self.mobility,
        )
        nonlinear = self.plan.nonlinear.solve(
            self.problem,
            state.phase,
            termination=self.plan.termination,
            args=arguments,
        )
        candidate_phase = nonlinear.state
        mass_after = self.mass(candidate_phase)
        energy_after = self.energy(candidate_phase)
        candidate_value, _ = _quadrature_fields(
            self.discretization,
            self.field_index,
            candidate_phase,
        )
        delta = candidate_value - previous_value
        dissipation = _integral(
            self.discretization,
            self.field_index,
            delta**2 / (self.mobility * step),
        )
        balance = energy_after - state.energy + dissipation
        energy_tolerance = self.plan.acceptance.energy_tolerance(
            state.energy,
            energy_after,
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
            & jnp.isfinite(dissipation)
            & jnp.isfinite(balance)
            & jnp.isfinite(nonlinear.diagnostics.final_residual_norm)
        )
        energy_stable = balance <= energy_tolerance
        mass_conserved = jnp.asarray(True)
        successful = nonlinear.successful & finite & energy_stable
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
            state.energy,
            energy_after,
            dissipation,
            balance,
            energy_tolerance,
            jnp.min(candidate_phase),
            jnp.max(candidate_phase),
            nonlinear.diagnostics.final_residual_norm,
            nonlinear.diagnostics.iterations,
            work,
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
    mobility: Array
    nonlinear: AbstractNonlinearMethod
    termination: NonlinearTermination
    acceptance: PhaseFieldAcceptancePolicy
    execution_policy: FiniteElementExecutionPolicy
    minimum_transition_cells: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: BinaryPhaseFieldModel,
        mobility: ArrayLike,
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
        model.require_production_supported()
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
                "mobility": mobility_,
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
    ) -> PreparedCahnHilliardFEM:
        return PreparedCahnHilliardFEM(
            self,
            discretization,
            concentration_field,
            chemical_field,
        )


class PreparedCahnHilliardFEM(AbstractFixedStepMethod):
    """Compiled Cahn-Hilliard dynamics with conservative accepted-step evidence."""

    plan: CahnHilliardFEMPlan
    discretization: FiniteElementDiscretization
    compiled: CompiledFiniteElementProblem
    problem: NonlinearSystemProblem
    resolution: PhaseFieldResolutionEvidence
    domain_measure: Array
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
    ):
        if not isinstance(plan, CahnHilliardFEMPlan):
            raise TypeError("plan must be CahnHilliardFEMPlan.")
        concentration = str(concentration_field)
        chemical = str(chemical_field)
        if not concentration or not chemical or concentration == chemical:
            raise ValueError("Cahn-Hilliard field names must be distinct and nonempty.")
        concentration_index, chemical_index = _validate_discretization(
            discretization,
            (concentration, chemical),
        )
        resolution = _resolution_evidence(
            plan.model,
            discretization,
            plan.minimum_transition_cells,
        )
        compiled = compile_finite_element_functional(
            _cahn_hilliard_functional(),
            discretization,
            fields={
                "concentration": concentration,
                "chemical_potential": chemical,
            },
            regions={"cells": discretization.cell_domain},
            rules={
                "cells": (
                    (
                        discretization.mesh.blocks[0].name,
                        ReferenceTriangleRule(GaussLegendreRule(2)),
                    ),
                )
            },
            execution_policy=plan.execution_policy,
        )
        self.plan = plan
        self.discretization = discretization
        self.compiled = compiled
        self.problem = compiled.as_nonlinear_problem()
        self.resolution = resolution
        self.domain_measure = _domain_measure(discretization, concentration_index)
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
                "resolution": resolution.evidence_id,
            }
        )

    @property
    def model(self) -> BinaryPhaseFieldModel:
        return self.plan.model

    @property
    def mobility(self) -> Array:
        return self.plan.mobility

    def mass(self, concentration: ArrayLike, /) -> Array:
        value = self.discretization.field_spaces[
            self.concentration_index
        ].vector_space.validate(concentration)
        return _mass(self.discretization, self.concentration_index, value)

    def energy(self, concentration: ArrayLike, /) -> Array:
        value = self.discretization.field_spaces[
            self.concentration_index
        ].vector_space.validate(concentration)
        return _energy(
            self.model,
            self.discretization,
            self.concentration_index,
            value,
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
        return CahnHilliardAcceptedState(value, chemical, mass, mass, energy)

    def step_detailed(
        self,
        step_index: Array,
        time: Array,
        state: CahnHilliardAcceptedState,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> PhaseFieldStepResult:
        del step_index, time, args
        if not isinstance(state, CahnHilliardAcceptedState):
            raise TypeError("Cahn-Hilliard step state must be CahnHilliardAcceptedState.")
        step = jnp.asarray(step_size, dtype=state.concentration.dtype)
        if step.shape != ():
            raise ValueError("Cahn-Hilliard step size must be scalar.")
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Cahn-Hilliard step size must be positive and finite.",
        )
        previous_value, _ = _quadrature_fields(
            self.discretization,
            self.concentration_index,
            state.concentration,
        )
        arguments = _CahnHilliardStepArguments(
            previous_value,
            step,
            self.model,
            self.mobility,
        )
        nonlinear = self.plan.nonlinear.solve(
            self.problem,
            (state.concentration, state.chemical_potential),
            termination=self.plan.termination,
            args=arguments,
        )
        candidate_concentration, candidate_chemical = nonlinear.state
        mass_after = self.mass(candidate_concentration)
        energy_after = self.energy(candidate_concentration)
        _, chemical_gradient = _quadrature_fields(
            self.discretization,
            self.chemical_index,
            candidate_chemical,
        )
        dissipation = (
            step
            * self.mobility
            * _integral(
                self.discretization,
                self.chemical_index,
                ein.contract("...d,...d->...", chemical_gradient, chemical_gradient),
            )
        )
        balance = energy_after - state.energy + dissipation
        energy_tolerance = self.plan.acceptance.energy_tolerance(
            state.energy,
            energy_after,
        )
        mass_defect = jnp.abs(mass_after - state.reference_mass)
        mass_tolerance = self.plan.acceptance.mass_tolerance(
            state.reference_mass,
            self.domain_measure,
        )
        candidate = CahnHilliardAcceptedState(
            candidate_concentration,
            candidate_chemical,
            state.reference_mass,
            mass_after,
            energy_after,
        )
        finite = (
            tree_allfinite(candidate)
            & jnp.isfinite(dissipation)
            & jnp.isfinite(balance)
            & jnp.isfinite(nonlinear.diagnostics.final_residual_norm)
        )
        energy_stable = balance <= energy_tolerance
        mass_conserved = mass_defect <= mass_tolerance
        successful = nonlinear.successful & finite & energy_stable & mass_conserved
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
            state.reference_mass,
            mass_defect,
            mass_tolerance,
            state.energy,
            energy_after,
            dissipation,
            balance,
            energy_tolerance,
            jnp.min(candidate_concentration),
            jnp.max(candidate_concentration),
            nonlinear.diagnostics.final_residual_norm,
            nonlinear.diagnostics.iterations,
            work,
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
