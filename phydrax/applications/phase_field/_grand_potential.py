#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite, tree_where
from ...discretization import FiniteElementDiscretization
from ...equations import (
    CellResidualAction,
    compile_finite_element_problem,
    CompiledFiniteElementProblem,
    FiniteElementExecutionPolicy,
    FiniteElementForm,
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
from ...nonlinear import NewtonKrylov, NonlinearResult, NonlinearTermination
from ...solver import (
    AbstractFixedStepMethod,
    FixedStepResult,
    ProductionRunPlan,
    RobustRetryPolicy,
)
from ._binary import PhaseFieldProductionCase
from ._mobility import AbstractPhaseFieldMobility, as_phase_field_mobility


class GrandPotentialPhaseEvaluation(StrictModule):
    grand_potential: Array
    composition: Array
    susceptibility: Array
    helmholtz: Array
    finite: Array
    stable: Array
    phase_id: str = eqx.field(static=True)


class QuadraticGrandPotentialPhase(StrictModule, NonTrainableState):
    """Concave quadratic grand potential with an exact linear composition law."""

    reference: Array
    reference_composition: Array
    susceptibility: Array
    component_count: int = eqx.field(static=True)
    phase_id: str = eqx.field(static=True)

    def __init__(
        self,
        phase_id: str,
        reference: ArrayLike,
        reference_composition: ArrayLike,
        susceptibility: ArrayLike,
        /,
    ):
        identifier = str(phase_id)
        reference_ = np.asarray(reference)
        composition = np.asarray(reference_composition)
        response = np.asarray(susceptibility)
        if (
            not identifier
            or reference_.shape != ()
            or not np.isfinite(reference_)
            or composition.ndim != 1
            or composition.size == 0
            or np.any(~np.isfinite(composition))
            or response.shape != (composition.size, composition.size)
            or np.any(~np.isfinite(response))
        ):
            raise ValueError("Quadratic grand-potential phase data are invalid.")
        symmetric = 0.5 * (response + response.T)
        scale = max(float(np.max(np.abs(symmetric))), 1.0)
        tolerance = 128.0 * np.finfo(symmetric.dtype).eps * scale
        if (
            np.max(np.abs(response - response.T)) > tolerance
            or np.min(np.linalg.eigvalsh(symmetric)) < -tolerance
        ):
            raise ValueError("Grand-potential susceptibility must be symmetric PSD.")
        self.reference = jnp.asarray(reference_)
        self.reference_composition = jnp.asarray(composition)
        self.susceptibility = jnp.asarray(symmetric)
        self.component_count = composition.size
        self.phase_id = canonical_fingerprint(
            {
                "kind": "quadratic-grand-potential-phase",
                "declared_id": identifier,
                "reference": float(reference_),
                "composition": array_tree_fingerprint(composition),
                "susceptibility": array_tree_fingerprint(symmetric),
            }
        )

    def evaluate(self, chemical_potential: Array, /) -> GrandPotentialPhaseEvaluation:
        potential = jnp.asarray(chemical_potential)
        if potential.shape[-1] != self.component_count:
            raise ValueError("Chemical-potential component count is incompatible.")
        composition = self.reference_composition.astype(potential.dtype) + ein.contract(
            "ij,...j->...i", self.susceptibility.astype(potential.dtype), potential
        )
        grand = (
            self.reference.astype(potential.dtype)
            - ein.contract("i,...i->...", self.reference_composition, potential)
            - 0.5
            * ein.contract(
                "...i,ij,...j->...",
                potential,
                self.susceptibility.astype(potential.dtype),
                potential,
            )
        )
        helmholtz = grand + ein.contract("...i,...i->...", potential, composition)
        eigenvalues = jnp.linalg.eigvalsh(self.susceptibility)
        finite = (
            jnp.all(jnp.isfinite(grand))
            & jnp.all(jnp.isfinite(composition))
            & jnp.all(jnp.isfinite(helmholtz))
        )
        stable = jnp.min(eigenvalues) >= -64.0 * jnp.finfo(eigenvalues.dtype).eps
        return GrandPotentialPhaseEvaluation(
            grand,
            composition,
            self.susceptibility,
            helmholtz,
            finite,
            stable,
            self.phase_id,
        )


class GrandPotentialMaterialCatalog(StrictModule, NonTrainableState):
    phases: tuple[QuadraticGrandPotentialPhase, ...]
    phase_ids: tuple[str, ...] = eqx.field(static=True)
    phase_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    catalog_id: str = eqx.field(static=True)

    def __init__(self, phases: Sequence[QuadraticGrandPotentialPhase], /):
        values = tuple(phases)
        if (
            len(values) < 2
            or any(
                not isinstance(value, QuadraticGrandPotentialPhase) for value in values
            )
            or len({value.phase_id for value in values}) != len(values)
            or len({value.component_count for value in values}) != 1
        ):
            raise ValueError(
                "Grand-potential catalog requires distinct compatible phase laws."
            )
        self.phases = values
        self.phase_ids = tuple(value.phase_id for value in values)
        self.phase_count = len(values)
        self.component_count = values[0].component_count
        self.catalog_id = canonical_fingerprint(
            {"kind": "grand-potential-material-catalog", "phases": self.phase_ids}
        )


class GrandPotentialMixtureEvaluation(StrictModule):
    weights: Array
    grand_potential: Array
    composition: Array
    helmholtz: Array
    barrier: Array
    finite: Array
    stable: Array


class GrandPotentialMixtureModel(StrictModule, NonTrainableState):
    catalog: GrandPotentialMaterialCatalog
    barrier_scale: Array
    gradient_coefficient: Array
    kinetic_coefficient: Array
    mobility: AbstractPhaseFieldMobility
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        catalog: GrandPotentialMaterialCatalog,
        /,
        *,
        barrier_scale: ArrayLike,
        gradient_coefficient: ArrayLike,
        kinetic_coefficient: ArrayLike,
        mobility: AbstractPhaseFieldMobility | ArrayLike,
    ):
        if not isinstance(catalog, GrandPotentialMaterialCatalog):
            raise TypeError("catalog must be GrandPotentialMaterialCatalog.")
        scalars = tuple(
            np.asarray(value)
            for value in (barrier_scale, gradient_coefficient, kinetic_coefficient)
        )
        if any(
            value.shape != () or not np.isfinite(value) or value <= 0.0
            for value in scalars
        ):
            raise ValueError("Grand-potential mixture scales must be positive.")
        mobility_ = as_phase_field_mobility(mobility)
        self.catalog = catalog
        self.barrier_scale = jnp.asarray(scalars[0])
        self.gradient_coefficient = jnp.asarray(scalars[1])
        self.kinetic_coefficient = jnp.asarray(scalars[2])
        self.mobility = mobility_
        self.model_id = canonical_fingerprint(
            {
                "kind": "grand-potential-mixture-model",
                "catalog": catalog.catalog_id,
                "barrier_scale": float(scalars[0]),
                "gradient_coefficient": float(scalars[1]),
                "kinetic_coefficient": float(scalars[2]),
                "mobility": mobility_.mobility_id,
            }
        )

    def evaluate(
        self,
        phase_logits: Array,
        chemical_potential: Array,
        /,
    ) -> GrandPotentialMixtureEvaluation:
        logits = jnp.asarray(phase_logits)
        potential = jnp.asarray(chemical_potential, dtype=logits.dtype)
        if logits.shape[:-1] != potential.shape[:-1]:
            raise ValueError("Phase and chemical-potential batch shapes must match.")
        if logits.shape[-1] != self.catalog.phase_count or potential.shape[-1] != (
            self.catalog.component_count
        ):
            raise ValueError("Grand-potential field component counts are incompatible.")
        weights = jax.nn.softmax(logits, axis=-1)
        evaluations = tuple(phase.evaluate(potential) for phase in self.catalog.phases)
        grand_by_phase = jnp.stack(
            tuple(evaluation.grand_potential for evaluation in evaluations), axis=-1
        )
        composition_by_phase = jnp.stack(
            tuple(evaluation.composition for evaluation in evaluations), axis=-2
        )
        helmholtz_by_phase = jnp.stack(
            tuple(evaluation.helmholtz for evaluation in evaluations), axis=-1
        )
        grand = ein.contract("...p,...p->...", weights, grand_by_phase)
        composition = ein.contract("...p,...pc->...c", weights, composition_by_phase)
        helmholtz = ein.contract("...p,...p->...", weights, helmholtz_by_phase)
        barrier = self.barrier_scale.astype(logits.dtype) * jnp.sum(
            weights**2 * (1.0 - weights) ** 2, axis=-1
        )
        finite = (
            jnp.all(jnp.isfinite(weights))
            & jnp.all(jnp.isfinite(grand))
            & jnp.all(jnp.isfinite(composition))
            & jnp.all(jnp.isfinite(helmholtz))
        )
        stable = jnp.all(jnp.stack(tuple(value.stable for value in evaluations)))
        return GrandPotentialMixtureEvaluation(
            weights,
            grand + barrier,
            composition,
            helmholtz + barrier,
            barrier,
            finite,
            stable,
        )


class DenseGrandPotentialAcceptedState(StrictModule):
    phase_logits: Array
    chemical_potential: Array
    reference_components: Array
    components: Array
    energy: Array


class GrandPotentialStepEvidence(StrictModule):
    nonlinear_successful: Array
    finite: Array
    energy_stable: Array
    components_conserved: Array
    accepted: Array
    energy_before: Array
    energy_after: Array
    dissipation: Array
    energy_defect: Array
    energy_tolerance: Array
    component_defect: Array
    component_tolerance: Array
    nonlinear_residual: Array
    nonlinear_iterations: Array


class GrandPotentialStepResult(StrictModule):
    candidate_state: DenseGrandPotentialAcceptedState
    accepted_state: DenseGrandPotentialAcceptedState
    successful: Array
    evidence: GrandPotentialStepEvidence
    nonlinear_result: NonlinearResult


class _GrandPotentialStepArguments(StrictModule):
    previous_logits: tuple[Array, ...]
    previous_composition: tuple[Array, ...]
    step_size: Array
    time: Array
    user_args: object


class _PhaseResidualKernel(StrictModule):
    model: GrandPotentialMixtureModel
    block_index: int = eqx.field(static=True)

    def __call__(
        self,
        values,
        gradients,
        points,
        weights,
        test_basis,
        test_gradients,
        context,
    ):
        logits, chemical = values
        logits_gradient, _ = gradients
        logits_gradient = jnp.moveaxis(logits_gradient, 2, -1)
        arguments = context.user_args
        if not isinstance(arguments, _GrandPotentialStepArguments):
            raise TypeError("Grand-potential phase residual needs step arguments.")
        previous = arguments.previous_logits[self.block_index]

        def local_density(local_logits, local_chemical):
            return self.model.evaluate(local_logits, local_chemical).grand_potential

        flat_logits = logits.reshape((-1, logits.shape[-1]))
        flat_chemical = chemical.reshape((-1, chemical.shape[-1]))
        force = jax.vmap(jax.grad(local_density, argnums=0))(
            flat_logits, flat_chemical
        ).reshape(logits.shape)
        transient = (logits - previous) / arguments.step_size
        local = transient + self.model.kinetic_coefficient.astype(logits.dtype) * force
        return ein.contract("cq,cqp,qi->cip", weights, local, test_basis) + (
            self.model.kinetic_coefficient.astype(logits.dtype)
            * self.model.gradient_coefficient.astype(logits.dtype)
            * ein.contract(
                "cq,cqid,cqpd->cip",
                weights,
                test_gradients,
                logits_gradient,
            )
        )


class _ChemicalResidualKernel(StrictModule):
    model: GrandPotentialMixtureModel
    block_index: int = eqx.field(static=True)

    def __call__(
        self,
        values,
        gradients,
        points,
        weights,
        test_basis,
        test_gradients,
        context,
    ):
        logits, chemical = values
        _, chemical_gradient = gradients
        chemical_gradient = jnp.moveaxis(chemical_gradient, 2, -1)
        arguments = context.user_args
        if not isinstance(arguments, _GrandPotentialStepArguments):
            raise TypeError("Grand-potential chemical residual needs step arguments.")
        previous = arguments.previous_composition[self.block_index]
        current = self.model.evaluate(logits, chemical).composition
        transient = (current - previous) / arguments.step_size
        mobility = self.model.mobility.evaluate(
            arguments.previous_composition[self.block_index],
            points,
            arguments.time,
            arguments.user_args,
        )
        if mobility.tensor.shape[-2:] == (1, 1):
            flux = mobility.tensor[..., 0, 0, None, None] * chemical_gradient
        else:
            flux = ein.contract("cqde,cqke->cqkd", mobility.tensor, chemical_gradient)
        return ein.contract(
            "cq,cqk,qi->cik", weights, transient, test_basis
        ) + ein.contract("cq,cqid,cqkd->cik", weights, test_gradients, flux)


def _reference_rule(cell_kind: str, degree: int):
    order = max(int(degree) + 1, 2)
    rule = GaussLegendreRule(order)
    factories = {
        "triangle": ReferenceTriangleRule,
        "quadrilateral": ReferenceQuadrilateralRule,
        "tetrahedron": ReferenceTetrahedronRule,
        "hexahedron": ReferenceHexahedronRule,
        "prism": ReferencePrismRule,
        "pyramid": ReferencePyramidRule,
    }
    if cell_kind not in factories:
        raise ValueError(f"Unsupported grand-potential cell kind {cell_kind!r}.")
    return factories[cell_kind](rule)


def _block_quadrature(
    discretization: FiniteElementDiscretization,
    field_index: int,
    state: Array,
    rules,
):
    results = []
    for block_index, _rule in enumerate(rules):
        geometry = discretization.block_geometries[field_index][block_index]
        dofs = discretization.dof_maps[field_index].cell_dofs[block_index]
        orientation = discretization.dof_maps[field_index].orientations[block_index]
        local = state[dofs] * orientation.reshape(
            orientation.shape + (1,) * (state.ndim - 1)
        )
        value = ein.contract("qi,ci...->cq...", geometry.basis_values, local)
        gradient = jnp.moveaxis(
            ein.contract("cqid,ci...->cqd...", geometry.physical_gradients, local),
            2,
            -1,
        )
        results.append(
            (value, gradient, geometry.physical_points, geometry.physical_weights)
        )
    return tuple(results)


class GrandPotentialFEMPlan(StrictModule, NonTrainableState):
    model: GrandPotentialMixtureModel
    nonlinear: NewtonKrylov
    termination: NonlinearTermination
    execution_policy: FiniteElementExecutionPolicy
    absolute_energy_tolerance: float = eqx.field(static=True)
    relative_energy_tolerance: float = eqx.field(static=True)
    component_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: GrandPotentialMixtureModel,
        /,
        *,
        nonlinear: NewtonKrylov | None = None,
        termination: NonlinearTermination | None = None,
        execution_policy: FiniteElementExecutionPolicy | None = None,
        absolute_energy_tolerance: float = 1.0e-8,
        relative_energy_tolerance: float = 1.0e-8,
        component_tolerance: float = 1.0e-8,
    ):
        if not isinstance(model, GrandPotentialMixtureModel):
            raise TypeError("model must be GrandPotentialMixtureModel.")
        values = tuple(
            float(value)
            for value in (
                absolute_energy_tolerance,
                relative_energy_tolerance,
                component_tolerance,
            )
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Grand-potential tolerances must be nonnegative.")
        self.model = model
        self.nonlinear = NewtonKrylov() if nonlinear is None else nonlinear
        self.termination = NonlinearTermination() if termination is None else termination
        self.execution_policy = (
            FiniteElementExecutionPolicy(
                realization="matrix_free",
                local_kernel="auto",
                accumulation="deterministic",
            )
            if execution_policy is None
            else execution_policy
        )
        self.absolute_energy_tolerance = values[0]
        self.relative_energy_tolerance = values[1]
        self.component_tolerance = values[2]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "grand-potential-fem-plan",
                "model": model.model_id,
                "nonlinear": self.nonlinear.method_id,
                "execution": self.execution_policy.policy_id,
                "tolerances": values,
            }
        )

    def prepare(
        self,
        discretization: FiniteElementDiscretization,
        phase_field: str,
        chemical_field: str,
        /,
    ) -> PreparedGrandPotentialFEM:
        return PreparedGrandPotentialFEM(
            self, discretization, phase_field, chemical_field
        )


class PreparedGrandPotentialFEM(AbstractFixedStepMethod):
    plan: GrandPotentialFEMPlan
    discretization: FiniteElementDiscretization
    compiled: CompiledFiniteElementProblem
    rules: tuple[object, ...]
    phase_field: str = eqx.field(static=True)
    chemical_field: str = eqx.field(static=True)
    phase_index: int = eqx.field(static=True)
    chemical_index: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: GrandPotentialFEMPlan,
        discretization: FiniteElementDiscretization,
        phase_field: str,
        chemical_field: str,
        /,
    ):
        if not isinstance(plan, GrandPotentialFEMPlan):
            raise TypeError("plan must be GrandPotentialFEMPlan.")
        phase_index = discretization._field_index(phase_field)
        chemical_index = discretization._field_index(chemical_field)
        phase_space = discretization.field_spaces[phase_index].vector_space
        chemical_space = discretization.field_spaces[chemical_index].vector_space
        if not isinstance(phase_space, ArraySpace) or not isinstance(
            chemical_space, ArraySpace
        ):
            raise TypeError("Grand-potential fields require ArraySpace coordinates.")
        phase_shape = phase_space.shape
        chemical_shape = chemical_space.shape
        if phase_shape[1:] != (plan.model.catalog.phase_count,) or chemical_shape[1:] != (
            plan.model.catalog.component_count,
        ):
            raise ValueError(
                "Grand-potential FE component shapes do not match the material catalog."
            )
        rules = tuple(
            _reference_rule(
                block.cell_kind,
                max(
                    discretization.elements[phase_index][block_index].degree,
                    discretization.elements[chemical_index][block_index].degree,
                ),
            )
            for block_index, block in enumerate(discretization.mesh.blocks)
        )
        rule_map = {
            block.name: rule
            for block, rule in zip(discretization.mesh.blocks, rules, strict=True)
        }
        actions = []
        for block_index, block in enumerate(discretization.mesh.blocks):
            domain = discretization.cell_block_domain(block.name)
            local_rule = {block.name: rule_map[block.name]}
            actions.extend(
                (
                    CellResidualAction(
                        phase_field,
                        (phase_field, chemical_field),
                        _PhaseResidualKernel(plan.model, block_index),
                        domain=domain,
                        rules=local_rule,
                        action_id=f"grand-potential-phase/{block.name}",
                    ),
                    CellResidualAction(
                        chemical_field,
                        (phase_field, chemical_field),
                        _ChemicalResidualKernel(plan.model, block_index),
                        domain=domain,
                        rules=local_rule,
                        action_id=f"grand-potential-components/{block.name}",
                    ),
                )
            )
        form = FiniteElementForm(
            "grand-potential-multiphase-step",
            (phase_field, chemical_field),
            tuple(actions),
        )
        compiled = compile_finite_element_problem(
            form,
            discretization,
            execution_policy=plan.execution_policy,
        )
        self.plan = plan
        self.discretization = discretization
        self.compiled = compiled
        self.rules = rules
        self.phase_field = phase_field
        self.chemical_field = chemical_field
        self.phase_index = phase_index
        self.chemical_index = chemical_index
        self.method_id = canonical_fingerprint(
            {
                "kind": "prepared-grand-potential-fem",
                "plan": plan.plan_id,
                "discretization": discretization.prepared_id,
                "compilation": compiled.compilation_id,
            }
        )

    def _quadrature(self, phase_logits: Array, chemical: Array):
        phase_blocks = _block_quadrature(
            self.discretization, self.phase_index, phase_logits, self.rules
        )
        chemical_blocks = _block_quadrature(
            self.discretization, self.chemical_index, chemical, self.rules
        )
        return phase_blocks, chemical_blocks

    def _integrals(self, phase_logits: Array, chemical: Array):
        phase_blocks, chemical_blocks = self._quadrature(phase_logits, chemical)
        component = jnp.zeros(
            (self.plan.model.catalog.component_count,), dtype=phase_logits.dtype
        )
        energy = jnp.asarray(0.0, dtype=phase_logits.dtype)
        for (logits, logits_gradient, _, weights), (mu, _, _, _) in zip(
            phase_blocks, chemical_blocks, strict=True
        ):
            evaluation = self.plan.model.evaluate(logits, mu)
            gradient_energy = (
                0.5
                * self.plan.model.gradient_coefficient
                * ein.contract("...pd,...pd->...", logits_gradient, logits_gradient)
            )
            energy = energy + jnp.sum((evaluation.helmholtz + gradient_energy) * weights)
            component = component + jnp.sum(
                evaluation.composition * weights[..., None], axis=(0, 1)
            )
        return component, energy

    def initialize(
        self,
        phase_logits: ArrayLike,
        chemical_potential: ArrayLike,
        /,
    ) -> DenseGrandPotentialAcceptedState:
        phase = self.discretization.field_spaces[self.phase_index].vector_space.validate(
            phase_logits
        )
        chemical = self.discretization.field_spaces[
            self.chemical_index
        ].vector_space.validate(chemical_potential)
        if not bool(np.asarray(tree_allfinite((phase, chemical)))):
            raise ValueError("Grand-potential initial state must be finite.")
        components, energy = self._integrals(phase, chemical)
        return DenseGrandPotentialAcceptedState(
            phase, chemical, components, components, energy
        )

    def step_detailed(
        self,
        step_index: Array,
        time: Array,
        state: DenseGrandPotentialAcceptedState,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> GrandPotentialStepResult:
        del step_index
        if not isinstance(state, DenseGrandPotentialAcceptedState):
            raise TypeError("state must be DenseGrandPotentialAcceptedState.")
        step = jnp.asarray(step_size, dtype=state.phase_logits.dtype)
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Grand-potential step size must be positive and finite.",
        )
        phase_blocks, chemical_blocks = self._quadrature(
            state.phase_logits, state.chemical_potential
        )
        previous_logits = tuple(block[0] for block in phase_blocks)
        previous_composition = tuple(
            self.plan.model.evaluate(phase[0], chemical[0]).composition
            for phase, chemical in zip(phase_blocks, chemical_blocks, strict=True)
        )
        arguments = _GrandPotentialStepArguments(
            previous_logits,
            previous_composition,
            step,
            jnp.asarray(time),
            args,
        )
        nonlinear = self.plan.nonlinear.solve(
            self.compiled.as_nonlinear_problem(),
            (state.phase_logits, state.chemical_potential),
            termination=self.plan.termination,
            args=arguments,
        )
        candidate_phase, candidate_chemical = nonlinear.state
        components, energy = self._integrals(candidate_phase, candidate_chemical)
        candidate = DenseGrandPotentialAcceptedState(
            candidate_phase,
            candidate_chemical,
            state.reference_components,
            components,
            energy,
        )
        component_defect = jnp.max(jnp.abs(components - state.reference_components))
        component_tolerance = jnp.asarray(
            self.plan.component_tolerance, dtype=energy.dtype
        )
        phase_after, chemical_after = self._quadrature(
            candidate_phase, candidate_chemical
        )
        dissipation = jnp.asarray(0.0, dtype=energy.dtype)
        for block_index, (old, new, chemical_block) in enumerate(
            zip(phase_blocks, phase_after, chemical_after, strict=True)
        ):
            delta = new[0] - old[0]
            phase_dissipation = jnp.sum(
                delta**2
                * new[3][..., None]
                / (self.plan.model.kinetic_coefficient * step)
            )
            mobility_quadratic, _ = self.plan.model.mobility.quadratic(
                chemical_block[1],
                previous_composition[block_index],
                chemical_block[2],
                jnp.asarray(time),
                args,
            )
            if mobility_quadratic.ndim == chemical_block[3].ndim + 1:
                mobility_quadratic = jnp.sum(mobility_quadratic, axis=-1)
            diffusion = step * jnp.sum(mobility_quadratic * chemical_block[3])
            dissipation = dissipation + phase_dissipation + diffusion
        tolerance = self.plan.absolute_energy_tolerance + (
            self.plan.relative_energy_tolerance
            * jnp.maximum(jnp.abs(state.energy), jnp.abs(energy))
        )
        defect = energy - state.energy + dissipation
        finite = tree_allfinite(candidate) & jnp.isfinite(dissipation)
        energy_stable = defect <= tolerance
        conserved = component_defect <= component_tolerance
        successful = nonlinear.successful & finite & energy_stable & conserved
        accepted = tree_where(successful, candidate, state)
        evidence = GrandPotentialStepEvidence(
            nonlinear.successful,
            finite,
            energy_stable,
            conserved,
            successful,
            state.energy,
            energy,
            dissipation,
            defect,
            tolerance,
            component_defect,
            component_tolerance,
            nonlinear.diagnostics.final_residual_norm,
            nonlinear.diagnostics.iterations,
        )
        return GrandPotentialStepResult(
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
        state: DenseGrandPotentialAcceptedState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        result = self.step_detailed(step_index, time, state, step_size, args)
        evidence = result.evidence
        residual = jnp.maximum(
            evidence.nonlinear_residual,
            jnp.maximum(
                evidence.energy_defect - evidence.energy_tolerance,
                evidence.component_defect - evidence.component_tolerance,
            ),
        )
        work = (
            result.nonlinear_result.diagnostics.residual_evaluations
            + result.nonlinear_result.diagnostics.jvp_evaluations
            + result.nonlinear_result.diagnostics.linear_iterations
        )
        return FixedStepResult(
            result.candidate_state,
            result.accepted_state,
            result.successful,
            residual,
            evidence.nonlinear_iterations,
            work,
            jnp.asarray(False),
            jnp.zeros((), dtype=state.phase_logits.dtype),
        )

    def production_case(
        self,
        case_name: str,
        initial_state: DenseGrandPotentialAcceptedState,
        /,
    ) -> PhaseFieldProductionCase:
        if not isinstance(initial_state, DenseGrandPotentialAcceptedState):
            raise TypeError("Grand-potential production case requires initialized state.")
        if not bool(np.asarray(tree_allfinite(initial_state))):
            raise ValueError("Grand-potential production state must be finite.")
        return PhaseFieldProductionCase(
            case_name,
            initial_state,
            method_id=self.method_id,
            precision_id=self.discretization.precision_policy.policy_id,
            topology_id=self.discretization.default_runtime.topology_id,
            geometry_layout_id=(self.discretization.default_runtime.geometry_layout_id),
            dtype=jnp.dtype(initial_state.phase_logits.dtype).name,
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
        retry = (
            RobustRetryPolicy(maximum_retries=2) if retry_policy is None else retry_policy
        )
        return ProductionRunPlan(
            self,
            retry,
            step_size=step_size,
            end_time=end_time,
            maximum_steps=maximum_steps,
            checkpoint_interval=checkpoint_interval,
            segment_steps=segment_steps,
        )


__all__ = [
    "DenseGrandPotentialAcceptedState",
    "GrandPotentialFEMPlan",
    "GrandPotentialMaterialCatalog",
    "GrandPotentialMixtureEvaluation",
    "GrandPotentialMixtureModel",
    "GrandPotentialPhaseEvaluation",
    "GrandPotentialStepEvidence",
    "GrandPotentialStepResult",
    "PreparedGrandPotentialFEM",
    "QuadraticGrandPotentialPhase",
]
