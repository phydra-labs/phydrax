#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import isfinite, prod
from numbers import Integral
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DenseLinearOperator,
    EmbeddedTensorProductLinearOperator,
    MaterializationPolicy,
    materialize,
    TensorProductSpace,
)
from ..operators.quantum import (
    apply_local_unitary_to_state,
    HilbertRegisterLayout,
    unitarity_residual,
)


ProductFormulaOrder = Literal[1, 2]
LocalHamiltonianDifferentiationMode = Literal["autodiff", "reversible-product-formula"]


class LocalHamiltonianTerm(StrictModule):
    """One Hermitian generator on an ordered subset of Hilbert-register wires."""

    generator: Array
    product_factors: tuple[Array, ...] | None
    hermiticity_residual: Array
    factorization_residual: Array
    finite: Array
    valid: Array
    target_wire_ids: tuple[str, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        generator: ArrayLike,
        target_wire_ids: Sequence[str],
        /,
        *,
        product_factors: Sequence[ArrayLike] | None = None,
        tolerance: float = 1e-8,
        term_id: str | None = None,
    ):
        value = jnp.asarray(generator)
        if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] == 0:
            raise ValueError("generator must be one nonempty square matrix.")
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(float)
        if not jnp.issubdtype(value.dtype, jnp.complexfloating):
            value = value.astype(jnp.result_type(value.dtype, 1j))
        targets = tuple(str(target) for target in target_wire_ids)
        if not targets or any(not target for target in targets):
            raise ValueError("target_wire_ids must be nonempty strings.")
        if len(set(targets)) != len(targets):
            raise ValueError("target_wire_ids must be unique.")
        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        if product_factors is None:
            factors = None
            factorization_residual = jnp.asarray(0.0, dtype=jnp.real(value).dtype)
            factors_finite = jnp.asarray(True)
        else:
            factors = tuple(
                jnp.asarray(factor, dtype=value.dtype) for factor in product_factors
            )
            if len(factors) != len(targets):
                raise ValueError("product_factors must align with target_wire_ids.")
            if any(
                factor.ndim != 2
                or factor.shape[0] != factor.shape[1]
                or factor.shape[0] == 0
                for factor in factors
            ):
                raise ValueError(
                    "Every product factor must be one nonempty square matrix."
                )
            if prod(factor.shape[0] for factor in factors) != value.shape[0]:
                raise ValueError("product_factors do not match the generator dimension.")
            reconstructed = jnp.asarray([[1.0]], dtype=value.dtype)
            for factor in factors:
                reconstructed = jnp.kron(reconstructed, factor)
            factorization_residual = jnp.max(jnp.abs(reconstructed - value))
            factors_finite = jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(factor)) for factor in factors))
            )
        residual = jnp.max(jnp.abs(value - jnp.conj(value.T)))
        finite = jnp.all(jnp.isfinite(value)) & factors_finite
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "local-hamiltonian-term",
                    "targets": list(targets),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "factored": factors is not None,
                    "factor_shapes": (
                        []
                        if factors is None
                        else [list(factor.shape) for factor in factors]
                    ),
                    "tolerance": tolerance_,
                }
            )
            if term_id is None
            else str(term_id)
        )
        if not identifier:
            raise ValueError("term_id must be nonempty.")
        self.generator = value
        self.product_factors = factors
        self.hermiticity_residual = residual
        self.factorization_residual = factorization_residual
        self.finite = finite
        self.valid = (
            finite
            & jnp.isfinite(residual)
            & (residual <= tolerance_)
            & jnp.isfinite(factorization_residual)
            & (factorization_residual <= tolerance_)
        )
        self.target_wire_ids = targets
        self.tolerance = tolerance_
        self.term_id = identifier

    @classmethod
    def from_product(
        cls,
        factors: Sequence[ArrayLike],
        target_wire_ids: Sequence[str],
        /,
        *,
        tolerance: float = 1e-8,
        term_id: str | None = None,
    ) -> LocalHamiltonianTerm:
        """Construct a term while retaining an exact single-site factorization."""

        values = tuple(jnp.asarray(factor) for factor in factors)
        if not values:
            raise ValueError("factors must be nonempty.")
        dtype = jnp.result_type(*values, 1j)
        values = tuple(value.astype(dtype) for value in values)
        generator = jnp.asarray([[1.0]], dtype=dtype)
        for value in values:
            generator = jnp.kron(generator, value)
        return cls(
            generator,
            target_wire_ids,
            product_factors=values,
            tolerance=tolerance,
            term_id=term_id,
        )


class LocalHamiltonian(StrictModule):
    """Fixed ordered sum of local Hermitian terms on one register layout."""

    layout: HilbertRegisterLayout
    terms: tuple[LocalHamiltonianTerm, ...]
    finite: Array
    valid: Array
    dtype: str = eqx.field(static=True)
    hamiltonian_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: HilbertRegisterLayout,
        terms: Sequence[LocalHamiltonianTerm],
        /,
        *,
        hamiltonian_id: str | None = None,
    ):
        if not isinstance(layout, HilbertRegisterLayout):
            raise TypeError("layout must be a HilbertRegisterLayout.")
        selected = tuple(terms)
        if not selected or not all(
            isinstance(term, LocalHamiltonianTerm) for term in selected
        ):
            raise ValueError("terms must contain at least one LocalHamiltonianTerm.")
        dtypes = {str(term.generator.dtype) for term in selected}
        if len(dtypes) != 1:
            raise TypeError("All local Hamiltonian terms must share one dtype.")
        for term in selected:
            expected = layout.target_dimension(term.target_wire_ids)
            if term.generator.shape != (expected, expected):
                raise ValueError("A term generator does not match its target dimensions.")
        finite = jnp.all(jnp.stack(tuple(term.finite for term in selected)))
        valid = jnp.all(jnp.stack(tuple(term.valid for term in selected)))
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "local-hamiltonian",
                    "layout": layout.layout_id,
                    "terms": [term.term_id for term in selected],
                    "dtype": next(iter(dtypes)),
                }
            )
            if hamiltonian_id is None
            else str(hamiltonian_id)
        )
        if not identifier:
            raise ValueError("hamiltonian_id must be nonempty.")
        self.layout = layout
        self.terms = selected
        self.finite = finite
        self.valid = valid
        self.dtype = next(iter(dtypes))
        self.hamiltonian_id = identifier


class FixedGridLocalHamiltonian(StrictModule):
    """Piecewise-constant coefficients for a fixed local Hamiltonian term sum."""

    hamiltonian: LocalHamiltonian
    time_grid: Array
    coefficients: Array
    hbar: Array
    source_valid: Array
    positive_intervals: Array
    finite: Array
    valid: Array
    interval_count: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: LocalHamiltonian,
        time_grid: ArrayLike,
        coefficients: ArrayLike,
        /,
        *,
        hbar: ArrayLike = 1.0,
        source_valid: ArrayLike = True,
        schedule_id: str | None = None,
    ):
        if not isinstance(hamiltonian, LocalHamiltonian):
            raise TypeError("hamiltonian must be a LocalHamiltonian.")
        times = jnp.asarray(time_grid)
        values = jnp.asarray(coefficients)
        if times.ndim != 1 or times.shape[0] < 2:
            raise ValueError("time_grid must have shape (interval_count + 1,).")
        expected = (times.shape[0] - 1, len(hamiltonian.terms))
        if values.shape != expected:
            raise ValueError(f"coefficients must have shape {expected}.")
        if jnp.issubdtype(times.dtype, jnp.complexfloating):
            raise TypeError("time_grid must be real.")
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("coefficients must be real.")
        dtype = jnp.result_type(times, values, float)
        times = times.astype(dtype)
        values = values.astype(dtype)
        hbar_ = jnp.asarray(hbar, dtype=dtype)
        if hbar_.shape != ():
            raise ValueError("hbar must be scalar.")
        source_valid_ = jnp.asarray(source_valid, dtype=bool)
        if source_valid_.shape != ():
            raise ValueError("source_valid must be scalar.")
        intervals = jnp.diff(times)
        positive = jnp.all(intervals > 0.0)
        finite = (
            jnp.all(jnp.isfinite(times))
            & jnp.all(jnp.isfinite(values))
            & jnp.isfinite(hbar_)
        )
        valid = finite & positive & (hbar_ > 0.0) & hamiltonian.valid & source_valid_
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "fixed-grid-local-hamiltonian",
                    "hamiltonian": hamiltonian.hamiltonian_id,
                    "grid_shape": list(times.shape),
                    "coefficient_shape": list(values.shape),
                    "dtype": str(dtype),
                }
            )
            if schedule_id is None
            else str(schedule_id)
        )
        if not identifier:
            raise ValueError("schedule_id must be nonempty.")
        self.hamiltonian = hamiltonian
        self.time_grid = times
        self.coefficients = values
        self.hbar = hbar_
        self.source_valid = source_valid_
        self.positive_intervals = positive
        self.finite = finite
        self.valid = valid
        self.interval_count = int(times.shape[0] - 1)
        self.schedule_id = identifier


class LocalHamiltonianEvolutionStatus(IntEnum):
    """Outcome of one fixed-grid local Hamiltonian evolution."""

    SUCCESS = 0
    INVALID_INPUT = 1
    NUMERICAL_FAILURE = 2
    INVARIANT_FAILURE = 3


class LocalHamiltonianEvolutionPolicy(StrictModule):
    """Product-formula, resource, save, and differentiation contract."""

    order: ProductFormulaOrder = eqx.field(static=True)
    differentiation: LocalHamiltonianDifferentiationMode = eqx.field(static=True)
    maximum_intervals: int = eqx.field(static=True)
    maximum_state_elements: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    norm_tolerance: float = eqx.field(static=True)
    unitarity_tolerance: float = eqx.field(static=True)
    save_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        order: ProductFormulaOrder = 2,
        differentiation: LocalHamiltonianDifferentiationMode = "autodiff",
        maximum_intervals: int = 1_000_000,
        maximum_state_elements: int = 1 << 28,
        maximum_workspace_bytes: int = 1 << 30,
        norm_tolerance: float = 1e-8,
        unitarity_tolerance: float = 1e-8,
        save_indices: Sequence[int] = (),
    ):
        if order not in (1, 2):
            raise ValueError("order must be one or two.")
        if differentiation not in ("autodiff", "reversible-product-formula"):
            raise ValueError("Unknown local-Hamiltonian differentiation mode.")
        for name, value in (
            ("maximum_intervals", maximum_intervals),
            ("maximum_state_elements", maximum_state_elements),
            ("maximum_workspace_bytes", maximum_workspace_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be a positive integer.")
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive.")
        if any(
            not isfinite(float(value)) or float(value) < 0.0
            for value in (norm_tolerance, unitarity_tolerance)
        ):
            raise ValueError("Evolution tolerances must be finite and non-negative.")
        saves = tuple(int(index) for index in save_indices)
        if any(index < 0 for index in saves) or len(set(saves)) != len(saves):
            raise ValueError("save_indices must be unique and non-negative.")
        if differentiation == "reversible-product-formula" and saves:
            raise ValueError(
                "reversible-product-formula does not retain intermediate saved states."
            )
        self.order = order
        self.differentiation = differentiation
        self.maximum_intervals = int(maximum_intervals)
        self.maximum_state_elements = int(maximum_state_elements)
        self.maximum_workspace_bytes = int(maximum_workspace_bytes)
        self.norm_tolerance = float(norm_tolerance)
        self.unitarity_tolerance = float(unitarity_tolerance)
        self.save_indices = saves


class LocalHamiltonianEvolutionCostEstimate(StrictModule):
    """Logical storage and local-exponential estimate."""

    hilbert_dimension: int = eqx.field(static=True)
    interval_count: int = eqx.field(static=True)
    term_count: int = eqx.field(static=True)
    local_exponential_count: int = eqx.field(static=True)
    maximum_local_dimension: int = eqx.field(static=True)
    state_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)


class LocalHamiltonianEvolutionPlan(StrictModule):
    """Content-addressed execution structure for fixed-grid local evolution."""

    policy: LocalHamiltonianEvolutionPolicy
    cost: LocalHamiltonianEvolutionCostEstimate
    schedule_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedLocalHamiltonianEvolution(StrictModule):
    """Numerical schedule bound to one local-evolution plan."""

    schedule: FixedGridLocalHamiltonian
    plan: LocalHamiltonianEvolutionPlan
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class LocalHamiltonianEvolutionDiagnostics(StrictModule):
    """Norm, local-unitarity, and finite execution evidence."""

    initial_norm_residual: Array
    final_norm_residual: Array
    maximum_step_norm_residual: Array
    maximum_local_unitarity_residual: Array
    schedule_valid: Array
    finite: Array
    valid: Array


class LocalHamiltonianEvolutionResult(StrictModule):
    """Final and explicitly saved exact-state local evolution outputs."""

    final_state: Array
    saved_states: Array
    step_norm_residuals: Array
    local_unitarity_residuals: Array
    status: Array
    diagnostics: LocalHamiltonianEvolutionDiagnostics
    numeric_version: Array
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(LocalHamiltonianEvolutionStatus.SUCCESS)


def plan_local_hamiltonian_evolution(
    schedule: FixedGridLocalHamiltonian,
    policy: LocalHamiltonianEvolutionPolicy | None = None,
    /,
) -> LocalHamiltonianEvolutionPlan:
    """Validate fixed structure and estimate one product-formula execution."""

    if not isinstance(schedule, FixedGridLocalHamiltonian):
        raise TypeError("schedule must be a FixedGridLocalHamiltonian.")
    selected = LocalHamiltonianEvolutionPolicy() if policy is None else policy
    if not isinstance(selected, LocalHamiltonianEvolutionPolicy):
        raise TypeError("policy must be a LocalHamiltonianEvolutionPolicy or None.")
    if schedule.interval_count > selected.maximum_intervals:
        raise ValueError("Schedule interval count exceeds maximum_intervals.")
    if any(index > schedule.interval_count for index in selected.save_indices):
        raise ValueError("save_indices must lie within the fixed time grid.")
    itemsize = np.dtype(schedule.hamiltonian.terms[0].generator.dtype).itemsize
    dimension = schedule.hamiltonian.layout.dimension
    state_bytes = dimension * itemsize
    maximum_local = max(term.generator.shape[0] for term in schedule.hamiltonian.terms)
    workspace = itemsize * (2 * dimension + maximum_local * maximum_local)
    if dimension > selected.maximum_state_elements:
        raise ValueError("Hilbert dimension exceeds maximum_state_elements.")
    if workspace > selected.maximum_workspace_bytes:
        raise ValueError("Local evolution workspace exceeds maximum_workspace_bytes.")
    sweeps = 1 if selected.order == 1 else 2
    exponential_count = schedule.interval_count * len(schedule.hamiltonian.terms) * sweeps
    cost = LocalHamiltonianEvolutionCostEstimate(
        dimension,
        schedule.interval_count,
        len(schedule.hamiltonian.terms),
        exponential_count,
        maximum_local,
        state_bytes,
        workspace,
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "local-hamiltonian-evolution-plan",
            "schedule": schedule.schedule_id,
            "order": selected.order,
            "differentiation": selected.differentiation,
            "save_indices": list(selected.save_indices),
            "limits": {
                "maximum_intervals": selected.maximum_intervals,
                "maximum_state_elements": selected.maximum_state_elements,
                "maximum_workspace_bytes": selected.maximum_workspace_bytes,
            },
        }
    )
    return LocalHamiltonianEvolutionPlan(selected, cost, schedule.schedule_id, plan_id)


def _validate_schedule_plan(
    schedule: FixedGridLocalHamiltonian,
    plan: LocalHamiltonianEvolutionPlan,
    /,
) -> None:
    candidate = plan_local_hamiltonian_evolution(schedule, plan.policy)
    if candidate.plan_id != plan.plan_id:
        raise ValueError("FixedGridLocalHamiltonian does not match the plan structure.")


def prepare_local_hamiltonian_evolution(
    schedule: FixedGridLocalHamiltonian,
    plan: LocalHamiltonianEvolutionPlan | None = None,
    /,
    *,
    policy: LocalHamiltonianEvolutionPolicy | None = None,
) -> PreparedLocalHamiltonianEvolution:
    """Bind a numerical fixed-grid Hamiltonian to one execution plan."""

    if plan is None:
        selected = plan_local_hamiltonian_evolution(schedule, policy)
    else:
        if policy is not None:
            raise ValueError("Specify plan or policy, not both.")
        selected = plan
        _validate_schedule_plan(schedule, selected)
    return PreparedLocalHamiltonianEvolution(
        schedule,
        selected,
        jnp.asarray(0, dtype=jnp.int32),
        canonical_fingerprint(
            {"kind": "prepared-local-hamiltonian-evolution", "plan": selected.plan_id}
        ),
    )


def refresh_local_hamiltonian_evolution(
    prepared: PreparedLocalHamiltonianEvolution,
    schedule: FixedGridLocalHamiltonian,
    /,
) -> PreparedLocalHamiltonianEvolution:
    """Refresh generators and coefficients without changing execution structure."""

    if not isinstance(prepared, PreparedLocalHamiltonianEvolution):
        raise TypeError("prepared must be a PreparedLocalHamiltonianEvolution.")
    _validate_schedule_plan(schedule, prepared.plan)
    return PreparedLocalHamiltonianEvolution(
        schedule,
        prepared.plan,
        prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        prepared.prepared_id,
    )


def _state_norm_residual(state: Array, /) -> Array:
    norms = jnp.sum(jnp.real(state * jnp.conj(state)), axis=-1)
    return jnp.max(jnp.abs(norms - 1.0))


def _apply_term(
    state: Array,
    term: LocalHamiltonianTerm,
    coefficient: Array,
    interval: Array,
    hbar: Array,
    layout: HilbertRegisterLayout,
    /,
) -> tuple[Array, Array]:
    unitary = jsp.linalg.expm(-1j * interval * coefficient * term.generator / hbar)
    evolved = apply_local_unitary_to_state(
        layout,
        unitary,
        term.target_wire_ids,
        state,
    )
    return evolved, unitarity_residual(unitary)


def _product_formula_step(
    state: Array,
    coefficients: Array,
    interval: Array,
    schedule: FixedGridLocalHamiltonian,
    order: ProductFormulaOrder,
    /,
) -> tuple[Array, Array]:
    residuals: list[Array] = []
    if order == 1:
        for index, term in enumerate(schedule.hamiltonian.terms):
            state, residual = _apply_term(
                state,
                term,
                coefficients[index],
                interval,
                schedule.hbar,
                schedule.hamiltonian.layout,
            )
            residuals.append(residual)
    else:
        half = 0.5 * interval
        for index, term in enumerate(schedule.hamiltonian.terms):
            state, residual = _apply_term(
                state,
                term,
                coefficients[index],
                half,
                schedule.hbar,
                schedule.hamiltonian.layout,
            )
            residuals.append(residual)
        for index in range(len(schedule.hamiltonian.terms) - 1, -1, -1):
            term = schedule.hamiltonian.terms[index]
            state, residual = _apply_term(
                state,
                term,
                coefficients[index],
                half,
                schedule.hbar,
                schedule.hamiltonian.layout,
            )
            residuals.append(residual)
    return state, jnp.stack(residuals)


def _apply_generator(
    state: Array,
    generator: Array,
    target_wire_ids: tuple[str, ...],
    coefficient: Array,
    interval: Array,
    hbar: Array,
    layout: HilbertRegisterLayout,
    /,
) -> tuple[Array, Array]:
    unitary = jsp.linalg.expm(-1j * interval * coefficient * generator / hbar)
    evolved = apply_local_unitary_to_state(
        layout,
        unitary,
        target_wire_ids,
        state,
    )
    return evolved, unitarity_residual(unitary)


def _product_formula_arrays(
    state: Array,
    generators: tuple[Array, ...],
    coefficients: Array,
    interval: Array,
    hbar: Array,
    layout: HilbertRegisterLayout,
    targets: tuple[tuple[str, ...], ...],
    order: ProductFormulaOrder,
    /,
) -> tuple[Array, Array]:
    residuals: list[Array] = []
    scale = interval if order == 1 else 0.5 * interval
    for index, (generator, target) in enumerate(zip(generators, targets, strict=True)):
        state, residual = _apply_generator(
            state,
            generator,
            target,
            coefficients[index],
            scale,
            hbar,
            layout,
        )
        residuals.append(residual)
    if order == 2:
        for index in range(len(generators) - 1, -1, -1):
            state, residual = _apply_generator(
                state,
                generators[index],
                targets[index],
                coefficients[index],
                scale,
                hbar,
                layout,
            )
            residuals.append(residual)
    return state, jnp.stack(residuals)


def _inverse_product_formula_arrays(
    state: Array,
    generators: tuple[Array, ...],
    coefficients: Array,
    interval: Array,
    hbar: Array,
    layout: HilbertRegisterLayout,
    targets: tuple[tuple[str, ...], ...],
    order: ProductFormulaOrder,
    /,
) -> Array:
    if order == 2:
        return _product_formula_arrays(
            state,
            generators,
            coefficients,
            -interval,
            hbar,
            layout,
            targets,
            order,
        )[0]
    for index in range(len(generators) - 1, -1, -1):
        state, _ = _apply_generator(
            state,
            generators[index],
            targets[index],
            coefficients[index],
            -interval,
            hbar,
            layout,
        )
    return state


def _reversible_primal(
    inputs: tuple[tuple[Array, ...], Array, Array, Array, Array],
    layout: HilbertRegisterLayout,
    targets: tuple[tuple[str, ...], ...],
    order: ProductFormulaOrder,
    /,
) -> tuple[Array, Array, Array]:
    generators, coefficients, time_grid, hbar, initial_state = inputs
    intervals = jnp.diff(time_grid)

    def step(state, values):
        interval, coefficient_row = values
        evolved, residuals = _product_formula_arrays(
            state,
            generators,
            coefficient_row,
            interval,
            hbar,
            layout,
            targets,
            order,
        )
        return evolved, (_state_norm_residual(evolved), residuals)

    final, (norm_residuals, unitary_residuals) = jax.lax.scan(
        step,
        initial_state,
        (intervals, coefficients),
    )
    return (
        final,
        jax.lax.stop_gradient(norm_residuals),
        jax.lax.stop_gradient(unitary_residuals),
    )


@eqx.filter_custom_vjp
def _reversible_evolution(
    inputs: tuple[tuple[Array, ...], Array, Array, Array, Array],
    layout: HilbertRegisterLayout,
    targets: tuple[tuple[str, ...], ...],
    order: ProductFormulaOrder,
    reconstruction_tolerance: float,
    /,
) -> tuple[Array, Array, Array]:
    del reconstruction_tolerance
    return _reversible_primal(inputs, layout, targets, order)


@_reversible_evolution.def_fwd
def _reversible_evolution_forward(
    perturbed,
    inputs: tuple[tuple[Array, ...], Array, Array, Array, Array],
    layout: HilbertRegisterLayout,
    targets: tuple[tuple[str, ...], ...],
    order: ProductFormulaOrder,
    reconstruction_tolerance: float,
    /,
):
    del perturbed, reconstruction_tolerance
    output = _reversible_primal(inputs, layout, targets, order)
    return output, output[0]


@_reversible_evolution.def_bwd
def _reversible_evolution_backward(
    final_state: Array,
    grad_output,
    perturbed,
    inputs: tuple[tuple[Array, ...], Array, Array, Array, Array],
    layout: HilbertRegisterLayout,
    targets: tuple[tuple[str, ...], ...],
    order: ProductFormulaOrder,
    reconstruction_tolerance: float,
    /,
):
    del perturbed
    generators, coefficients, time_grid, hbar, initial_state = inputs
    final_bar = grad_output[0]
    if final_bar is None:
        final_bar = jnp.zeros_like(final_state)
    generator_gradients = tuple(jnp.zeros_like(generator) for generator in generators)
    coefficient_gradients = jnp.zeros_like(coefficients)
    time_gradients = jnp.zeros_like(time_grid)
    hbar_gradient = jnp.zeros_like(hbar)
    reverse_indices = jnp.arange(coefficients.shape[0] - 1, -1, -1)

    def reverse_step(carry, index):
        (
            current_state,
            current_bar,
            generator_gradient,
            coefficient_gradient,
            time_gradient,
            accumulated_hbar_gradient,
            maximum_reconstruction_residual,
        ) = carry
        coefficient_row = coefficients[index]
        interval = time_grid[index + 1] - time_grid[index]
        previous_state = _inverse_product_formula_arrays(
            current_state,
            generators,
            coefficient_row,
            interval,
            hbar,
            layout,
            targets,
            order,
        )

        def interval_evolution(
            generator_values,
            coefficient_values,
            duration,
            hbar_value,
            state_value,
        ):
            return _product_formula_arrays(
                state_value,
                generator_values,
                coefficient_values,
                duration,
                hbar_value,
                layout,
                targets,
                order,
            )[0]

        reconstructed, pullback = jax.vjp(
            interval_evolution,
            generators,
            coefficient_row,
            interval,
            hbar,
            previous_state,
        )
        (
            generator_increment,
            coefficient_increment,
            interval_increment,
            hbar_increment,
            previous_bar,
        ) = pullback(current_bar)
        generator_gradient = jax.tree_util.tree_map(
            lambda accumulated, increment: accumulated + increment,
            generator_gradient,
            generator_increment,
        )
        coefficient_gradient = coefficient_gradient.at[index].add(coefficient_increment)
        time_gradient = time_gradient.at[index].add(-interval_increment)
        time_gradient = time_gradient.at[index + 1].add(interval_increment)
        accumulated_hbar_gradient = accumulated_hbar_gradient + hbar_increment
        reconstruction = jnp.max(jnp.abs(reconstructed - current_state))
        return (
            previous_state,
            previous_bar,
            generator_gradient,
            coefficient_gradient,
            time_gradient,
            accumulated_hbar_gradient,
            jnp.maximum(maximum_reconstruction_residual, reconstruction),
        ), None

    initial_carry = (
        final_state,
        final_bar,
        generator_gradients,
        coefficient_gradients,
        time_gradients,
        hbar_gradient,
        jnp.asarray(0.0, dtype=jnp.real(final_state).dtype),
    )
    (
        (
            reconstructed_initial,
            initial_bar,
            generator_gradients,
            coefficient_gradients,
            time_gradients,
            hbar_gradient,
            reconstruction_residual,
        ),
        _,
    ) = jax.lax.scan(reverse_step, initial_carry, reverse_indices)
    initial_match = jnp.max(jnp.abs(reconstructed_initial - initial_state))
    reconstruction_residual = jnp.maximum(
        reconstruction_residual,
        initial_match,
    )
    first_gradient = eqx.error_if(
        generator_gradients[0],
        reconstruction_residual > reconstruction_tolerance,
        "Reversible product-formula reconstruction exceeded its tolerance.",
    )
    generator_gradients = (first_gradient,) + generator_gradients[1:]
    return (
        generator_gradients,
        coefficient_gradients,
        time_gradients,
        hbar_gradient,
        initial_bar,
    )


def _solve_reversible(
    prepared: PreparedLocalHamiltonianEvolution,
    initial_state: Array,
    /,
) -> tuple[Array, Array, Array]:
    schedule = prepared.schedule
    generators = tuple(term.generator for term in schedule.hamiltonian.terms)
    targets = tuple(term.target_wire_ids for term in schedule.hamiltonian.terms)
    return _reversible_evolution(
        (
            generators,
            schedule.coefficients,
            schedule.time_grid,
            schedule.hbar,
            initial_state,
        ),
        schedule.hamiltonian.layout,
        targets,
        prepared.plan.policy.order,
        prepared.plan.policy.norm_tolerance,
    )


def _solve_autodiff(
    prepared: PreparedLocalHamiltonianEvolution,
    initial_state: Array,
    /,
) -> tuple[Array, Array, tuple[Array, Array]]:
    schedule = prepared.schedule
    intervals = jnp.diff(schedule.time_grid)
    save_indices = jnp.asarray(prepared.plan.policy.save_indices, dtype=jnp.int32)
    saved = jnp.zeros(
        (len(prepared.plan.policy.save_indices),) + initial_state.shape,
        dtype=initial_state.dtype,
    )
    if prepared.plan.policy.save_indices:
        initial_mask = save_indices == 0
        saved = jnp.where(
            initial_mask.reshape(
                (len(prepared.plan.policy.save_indices),) + (1,) * initial_state.ndim
            ),
            initial_state[None, ...],
            saved,
        )

    def step(carry, inputs):
        state, saved_states = carry
        index, interval, coefficients = inputs
        evolved, residuals = _product_formula_step(
            state,
            coefficients,
            interval,
            schedule,
            prepared.plan.policy.order,
        )
        if prepared.plan.policy.save_indices:
            save_mask = save_indices == index + 1
            saved_states = jnp.where(
                save_mask.reshape(
                    (len(prepared.plan.policy.save_indices),) + (1,) * evolved.ndim
                ),
                evolved[None, ...],
                saved_states,
            )
        return (evolved, saved_states), (
            _state_norm_residual(evolved),
            residuals,
        )

    (final, saved), (norm_residuals, unitary_residuals) = jax.lax.scan(
        step,
        (initial_state, saved),
        (
            jnp.arange(schedule.interval_count, dtype=jnp.int32),
            intervals,
            schedule.coefficients,
        ),
    )
    return final, saved, (norm_residuals, unitary_residuals)


def solve_local_hamiltonian_evolution(
    prepared: PreparedLocalHamiltonianEvolution,
    initial_state: ArrayLike,
    /,
) -> LocalHamiltonianEvolutionResult:
    """Execute exact-state local product-formula evolution on a fixed grid."""

    if not isinstance(prepared, PreparedLocalHamiltonianEvolution):
        raise TypeError("prepared must be a PreparedLocalHamiltonianEvolution.")
    state = jnp.asarray(initial_state)
    dimension = prepared.plan.cost.hilbert_dimension
    if state.ndim == 0 or state.shape[-1] != dimension:
        raise ValueError(f"initial_state must end with dimension {dimension}.")
    if state.size > prepared.plan.policy.maximum_state_elements:
        raise ValueError("initial_state exceeds maximum_state_elements.")
    dtype = prepared.schedule.hamiltonian.terms[0].generator.dtype
    if not jnp.issubdtype(state.dtype, jnp.complexfloating):
        state = state.astype(dtype)
    else:
        state = state.astype(jnp.result_type(state.dtype, dtype))

    if prepared.plan.policy.differentiation == "reversible-product-formula":
        final, norm_residuals, unitary_residuals = _solve_reversible(
            prepared,
            state,
        )
        saved = jnp.empty((0,) + state.shape, dtype=state.dtype)
    else:
        final, saved, residuals = _solve_autodiff(prepared, state)
        norm_residuals, unitary_residuals = residuals
    initial_norm = _state_norm_residual(state)
    final_norm = _state_norm_residual(final)
    maximum_step_norm = jnp.max(norm_residuals)
    maximum_unitarity = jnp.max(unitary_residuals)
    finite = (
        jnp.all(jnp.isfinite(final))
        & jnp.all(jnp.isfinite(norm_residuals))
        & jnp.all(jnp.isfinite(unitary_residuals))
    )
    valid = (
        prepared.schedule.valid
        & finite
        & (initial_norm <= prepared.plan.policy.norm_tolerance)
        & (final_norm <= prepared.plan.policy.norm_tolerance)
        & (maximum_step_norm <= prepared.plan.policy.norm_tolerance)
        & (maximum_unitarity <= prepared.plan.policy.unitarity_tolerance)
    )
    status = jnp.where(
        ~prepared.schedule.valid,
        int(LocalHamiltonianEvolutionStatus.INVALID_INPUT),
        jnp.where(
            ~finite,
            int(LocalHamiltonianEvolutionStatus.NUMERICAL_FAILURE),
            jnp.where(
                ~valid,
                int(LocalHamiltonianEvolutionStatus.INVARIANT_FAILURE),
                int(LocalHamiltonianEvolutionStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    diagnostics = LocalHamiltonianEvolutionDiagnostics(
        initial_norm,
        final_norm,
        maximum_step_norm,
        maximum_unitarity,
        prepared.schedule.valid,
        finite,
        valid,
    )
    return LocalHamiltonianEvolutionResult(
        final,
        saved,
        norm_residuals,
        unitary_residuals,
        status,
        diagnostics,
        prepared.numeric_version,
        prepared.plan.plan_id,
    )


def local_hamiltonian_linear_operator(
    hamiltonian: LocalHamiltonian,
    coefficients: ArrayLike | None = None,
    /,
) -> AbstractLinearOperator:
    """Expose one local Hamiltonian sum as a structured linear operator."""

    if not isinstance(hamiltonian, LocalHamiltonian):
        raise TypeError("hamiltonian must be a LocalHamiltonian.")
    values = (
        jnp.ones((len(hamiltonian.terms),), dtype=float)
        if coefficients is None
        else jnp.asarray(coefficients)
    )
    if values.shape != (len(hamiltonian.terms),):
        raise ValueError("coefficients must have one entry per Hamiltonian term.")
    dtype = hamiltonian.terms[0].generator.dtype
    ambient = TensorProductSpace(
        tuple(
            ArraySpace((dimension,), dtype=dtype)
            for dimension in hamiltonian.layout.local_dimensions
        )
    )
    result: AbstractLinearOperator | None = None
    for coefficient, term in zip(values, hamiltonian.terms, strict=True):
        local_space = ArraySpace((term.generator.shape[0],), dtype=dtype)
        local = DenseLinearOperator(
            term.generator,
            source=local_space,
            target=local_space,
            operator_id=term.term_id,
        )
        embedded = EmbeddedTensorProductLinearOperator(
            local,
            ambient,
            hamiltonian.layout.target_indices(term.target_wire_ids),
        )
        contribution = coefficient * embedded
        result = contribution if result is None else result + contribution
    assert result is not None
    return result


def materialize_local_hamiltonian(
    hamiltonian: LocalHamiltonian,
    coefficients: ArrayLike | None = None,
    /,
    *,
    policy: MaterializationPolicy | None = None,
) -> Array:
    """Materialize a local Hamiltonian only under an explicit resource policy."""

    selected = MaterializationPolicy() if policy is None else policy
    return materialize(
        local_hamiltonian_linear_operator(hamiltonian, coefficients),
        selected,
    )


__all__ = [
    "LocalHamiltonianDifferentiationMode",
    "FixedGridLocalHamiltonian",
    "LocalHamiltonian",
    "LocalHamiltonianEvolutionCostEstimate",
    "LocalHamiltonianEvolutionDiagnostics",
    "LocalHamiltonianEvolutionPlan",
    "LocalHamiltonianEvolutionPolicy",
    "LocalHamiltonianEvolutionResult",
    "LocalHamiltonianEvolutionStatus",
    "LocalHamiltonianTerm",
    "PreparedLocalHamiltonianEvolution",
    "ProductFormulaOrder",
    "local_hamiltonian_linear_operator",
    "materialize_local_hamiltonian",
    "plan_local_hamiltonian_evolution",
    "prepare_local_hamiltonian_evolution",
    "refresh_local_hamiltonian_evolution",
    "solve_local_hamiltonian_evolution",
]
