#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...dynamics import DAEStructure, DifferentialAlgebraicSystem, TimeGrid
from ...linalg import (
    ArraySpace,
    BiCGStab,
    DenseLU,
    FailurePolicy,
    FGMRES,
    FunctionLinearOperator,
    GMRES,
    ILUPreconditionerBuilder,
    LinearSolvePolicy,
    LinearSystem,
    MaterializationPolicy,
    PreconditioningPolicy,
    prepare as prepare_linear,
    solve as solve_linear,
    TolerancePolicy,
)
from ...nonlinear import JacobianPolicy, NewtonKrylov, NonlinearTermination
from ...solver import (
    DAEAdaptivePolicy,
    DAEInitializationSpec,
    DAESolvePolicy,
    DifferentialAlgebraicProblem,
    solve_dae,
)
from ...solver._dae_initialization import _scaled_space
from ...solver._implicit_stage import ImplicitStageArguments, ImplicitStageResidual
from ...sparse import (
    compile_sparse_jacobian,
    SparseCoordinateOperator,
    SparseDerivativePlan,
)
from ._continuum import PreparedSemiconductorDevice, SemiconductorOperatingPoint
from ._materials import SemiconductorMaterial


class SemiconductorLinearEvidence(StrictModule):
    """Per-RHS native status, equilibrated defects, and unscaled defects."""

    successful: Array
    status: Array
    residual_norm: Array
    residual_threshold: Array
    operating_point_valid: Array
    unscaled_residual_norm: Array


class SemiconductorSmallSignalResult(StrictModule):
    angular_frequencies: Array
    admittance: Array
    evidence: SemiconductorLinearEvidence
    terminal_kcl_defect: Array
    voltage_gauge_defect: Array


class SemiconductorSensitivityResult(StrictModule):
    observable: Array
    derivatives: Array
    evidence: SemiconductorLinearEvidence


class SemiconductorTransientResult(StrictModule):
    times: Array
    coordinates: Array
    coordinate_rates: Array
    voltages: Array
    terminal_currents: Array
    conduction_currents: Array
    displacement_currents: Array
    terminal_charges: Array
    successful: Array
    solution: Any
    rate_evidence: SemiconductorLinearEvidence


def _linear_policy(
    policy: LinearSolvePolicy | None,
    size: int,
    dense_reference_limit: int | None = None,
    *,
    setup_operator: SparseCoordinateOperator | None = None,
) -> LinearSolvePolicy:
    if policy is None:
        return LinearSolvePolicy(
            GMRES(restart=min(size, 40)),
            tolerance=TolerancePolicy(relative=1e-12, absolute=1e-14, max_steps=1000),
            preconditioning=None
            if setup_operator is None
            else PreconditioningPolicy(
                ILUPreconditionerBuilder(),
                setup_operator=setup_operator,
                side="right",
                refresh="frozen",
            ),
            materialization=MaterializationPolicy(max_entries=1, max_bytes=16),
            failure=FailurePolicy("status"),
        )
    if isinstance(policy.method, DenseLU):
        if dense_reference_limit is None or not 0 < size <= dense_reference_limit:
            raise ValueError(
                "Dense reference solves require an explicit dense_reference_limit "
                "at least as large as the state dimension."
            )
    elif not isinstance(policy.method, (GMRES, FGMRES, BiCGStab)):
        raise TypeError("Select native GMRES/FGMRES/BiCGStab or bounded DenseLU.")
    if policy.failure.mode != "status":
        raise ValueError("Analysis requires native linear failure='status' evidence.")
    return policy


def _stiffness_setup(prepared, flat):
    """Reuse native graph coloring; never probe an all-state basis."""
    operator = prepared._derivative(flat).operator(flat)
    space = ArraySpace(flat.shape, dtype=flat.dtype)
    return SparseCoordinateOperator(
        operator.relation, operator.coefficients, source=space, target=space
    )


def _equilibrate_setup(operator):
    """Bound each equation's coefficient sum before iterative error control."""
    rows = operator.relation.target_indices
    row_norm = (
        jnp.zeros((operator.target.size,), dtype=jnp.real(operator.coefficients).dtype)
        .at[rows]
        .add(jnp.abs(operator.coefficients))
    )
    inverse = 1 / jnp.maximum(row_norm, 1)
    return eqx.tree_at(
        lambda value: value.coefficients,
        operator,
        operator.coefficients * inverse[rows],
    ), inverse


def _operator(action: Callable[[Array], Array], template: Array):
    space = ArraySpace(template.shape, dtype=template.dtype)
    return FunctionLinearOperator(action, source=space, target=space)


def _solve_action(action, rhs, policy, prepared=None, *, row_scale=1):
    scaled_rhs = row_scale * rhs
    if prepared is None:
        operator = _operator(lambda x: row_scale * action(x), rhs)
        result = solve_linear(LinearSystem(operator), scaled_rhs, policy=policy)
    else:
        result = solve_linear(prepared, scaled_rhs)
    image = action(result.value)
    defect = jnp.linalg.norm(row_scale * image - scaled_rhs)
    unscaled_defect = jnp.linalg.norm(image - rhs)
    threshold = policy.tolerance.absolute + policy.tolerance.relative * jnp.linalg.norm(
        scaled_rhs
    )
    valid = (
        jnp.all(result.successful)
        & jnp.all(jnp.isfinite(result.value))
        & jnp.isfinite(defect)
        & (defect <= threshold)
    )
    return result.value, valid, result.status, defect, threshold, unscaled_defect


def _complex_action(real_action, value):
    # JAX linearizations at real primals require real tangents. Extend linearly,
    # not through a complex perturbation of the semiconductor constitutive laws.
    return real_action(jnp.real(value)) + 1j * real_action(jnp.imag(value))


def _point_valid(prepared, point, tolerance):
    residual = prepared.time_scale * prepared.residual(point.coordinates, point.voltages)
    return (
        jnp.all(point.successful)
        & jnp.all(jnp.isfinite(point.coordinates))
        & jnp.all(jnp.isfinite(residual))
        & (jnp.max(jnp.abs(residual)) <= tolerance)
    )


def _evidence(records, point_valid, shape):
    valid, status, defects, thresholds, unscaled = (
        jnp.stack(items).reshape(shape) for items in zip(*records, strict=True)
    )
    return SemiconductorLinearEvidence(
        valid & point_valid,
        status,
        defects,
        thresholds,
        jnp.asarray(point_valid),
        unscaled,
    )


def _require_dynamic_charge_storage(prepared):
    if any(
        isinstance(model, SemiconductorMaterial)
        and model.incomplete_ionization is not None
        for model in prepared.plan.material_models
    ):
        raise ValueError(
            "AC/transient incomplete ionization requires explicit dynamic impurity "
            "populations; the local-equilibrium ionization closure is steady-only."
        )


def semiconductor_small_signal(
    prepared: PreparedSemiconductorDevice,
    operating_point: SemiconductorOperatingPoint,
    angular_frequencies: ArrayLike,
    /,
    *,
    linear_policy: LinearSolvePolicy | None = None,
    dense_reference_limit: int | None = None,
    residual_tolerance: float = 1e-7,
    conservation_tolerance: float = 1e-4,
) -> SemiconductorSmallSignalResult:
    """Terminal admittance in siemens, exp(+i omega t), current INTO the device.

    Solve (R_u + i omega S_u) du = -R_V dV with native matrix-free
    linalg. Only one state-sized RHS/response is live at a time; the retained
    result has shape (frequency, output terminal, driven terminal). Dense LU
    is available solely as an explicitly bounded reference, never a fallback.
    Default solves use sparse ILU and row equilibration. Acceptance additionally
    checks terminal KCL and voltage-gauge invariance relative to peak admittance;
    both defects are retained in siemens, independently of linear solver status.
    """
    _require_dynamic_charge_storage(prepared)
    if not np.isfinite(conservation_tolerance) or conservation_tolerance < 0:
        raise ValueError("conservation_tolerance must be finite and nonnegative.")
    frequencies = jnp.asarray(angular_frequencies, dtype=float)
    host = np.asarray(frequencies)
    if host.ndim != 1 or not host.size or np.any(~np.isfinite(host)) or np.any(host < 0):
        raise ValueError(
            "angular_frequencies must be a nonempty finite nonnegative vector."
        )
    point = operating_point
    u, volts = point.coordinates, point.voltages
    if linear_policy is not None:
        policy = _linear_policy(linear_policy, u.size, dense_reference_limit)
    point_valid = _point_valid(prepared, point, residual_tolerance)
    scale = prepared.time_scale
    flat = u.reshape(-1)
    _, stiffness = jax.linearize(
        lambda x: (scale * prepared.residual(x.reshape(u.shape), volts)).reshape(-1), flat
    )
    _, storage = jax.linearize(
        lambda x: (scale * prepared.storage(x.reshape(u.shape))).reshape(-1), flat
    )
    _, bias = jax.linearize(
        lambda v: (scale * prepared.residual(u, v)).reshape(-1), volts
    )
    _, conduction = jax.linearize(
        lambda x: prepared.terminal_current(x.reshape(u.shape)), flat
    )
    _, charge = jax.linearize(
        lambda x: prepared.terminal_charge(x.reshape(u.shape)), flat
    )
    if linear_policy is None:
        setup = _stiffness_setup(prepared, flat)
        mass = compile_sparse_jacobian(
            lambda x, _: (scale * prepared.storage(x.reshape(u.shape))).reshape(-1),
            flat,
            source=setup.source,
            target=setup.target,
            structure=prepared.coloring,
            compiler="native",
            mode="fwd",
        ).coefficients(flat)
        complex_space = ArraySpace(flat.shape, dtype=jnp.result_type(flat, 1j))
    admittances, records = [], []
    for omega in frequencies:
        columns = []

        def action(x, omega=omega):
            return _complex_action(stiffness, x) + 1j * omega * _complex_action(
                storage, x
            )

        if linear_policy is None:
            shifted_setup = SparseCoordinateOperator(
                setup.relation,
                setup.coefficients + 1j * omega * mass,
                source=complex_space,
                target=complex_space,
            )
            shifted_setup, row_scale = _equilibrate_setup(shifted_setup)
            policy = _linear_policy(None, u.size, setup_operator=shifted_setup)
        else:
            row_scale = 1
        template = flat.astype(jnp.result_type(flat, 1j))
        operator_action = lambda x, row_scale=row_scale, action=action: (
            row_scale * action(x)
        )
        native = prepare_linear(
            LinearSystem(_operator(operator_action, template)), policy
        )
        for index in range(volts.size):
            drive = jax.nn.one_hot(index, volts.size, dtype=volts.dtype)
            rhs = -bias(drive).astype(jnp.result_type(flat, 1j))
            response, valid, status, defect, threshold, unscaled = _solve_action(
                action, rhs, policy, native, row_scale=row_scale
            )
            current = _complex_action(
                conduction, response
            ) + 1j * omega * _complex_action(charge, response)
            valid = valid & jnp.all(jnp.isfinite(current))
            columns.append(
                jnp.where(valid & point_valid, current, jnp.nan + 1j * jnp.nan)
            )
            records.append((valid, status, defect, threshold, unscaled))
        admittances.append(jnp.stack(columns, axis=-1))
    evidence = _evidence(records, point_valid, (frequencies.size, volts.size))
    admittance = jnp.stack(admittances)
    kcl = jnp.sum(admittance, axis=1)
    gauge = jnp.sum(admittance, axis=2)
    limit = conservation_tolerance * jnp.max(jnp.abs(admittance), axis=(1, 2))
    conservative = (jnp.max(jnp.abs(kcl), axis=1) <= limit) & (
        jnp.max(jnp.abs(gauge), axis=1) <= limit
    )
    evidence = eqx.tree_at(
        lambda value: value.successful,
        evidence,
        evidence.successful & conservative[:, None],
    )
    admittance = jnp.where(
        evidence.successful[:, None, :], admittance, jnp.nan + 1j * jnp.nan
    )
    return SemiconductorSmallSignalResult(frequencies, admittance, evidence, kcl, gauge)


def semiconductor_sensitivity(
    prepared: PreparedSemiconductorDevice,
    point: SemiconductorOperatingPoint,
    /,
    *,
    parameters: ArrayLike | None = None,
    parameterize: Callable[[Array], tuple[PreparedSemiconductorDevice, Array]]
    | None = None,
    directions: ArrayLike | None = None,
    observable: Callable[[PreparedSemiconductorDevice, Array, Array], Array]
    | None = None,
    linear_policy: LinearSolvePolicy | None = None,
    dense_reference_limit: int | None = None,
    residual_tolerance: float = 1e-7,
) -> SemiconductorSensitivityResult:
    """Implicit observable derivatives, without differentiating Newton iterations.

    Default parameters are SI terminal biases. For material or fixed-topology
    geometry derivatives, parameterize(theta) returns (prepared_device, volts)
    using differentiable resolved-array replacement (e.g. eqx.tree_at). It must
    retain support topology, contact masks, and terminal order. Geometry changes
    must update positions, physical volumes AND transmissibility; changing
    positions alone is not a physical geometry derivative. Doping changes should
    use plan.with_doping to reclose mobilities and contact neutrality. Prepared
    numerical coefficients are derived afresh from these differentiable arrays.
    The observable defaults to terminal current in amperes. Directions have shape (K, P);
    omitted directions mean the P coordinate directions, generated one at a
    time. Derivatives have observable.shape + (K,). Failed columns are NaN.
    """
    if (parameters is None) != (parameterize is None):
        raise ValueError("parameters and parameterize must be supplied together.")
    theta = point.voltages if parameters is None else jnp.asarray(parameters, dtype=float)
    if theta.ndim != 1 or theta.size == 0 or bool(jnp.any(~jnp.isfinite(theta))):
        raise ValueError("parameters must be a nonempty finite vector.")
    builder = (lambda value: (prepared, value)) if parameterize is None else parameterize
    base, volts = builder(theta)
    _same_topology(prepared, base)
    if volts.shape != point.voltages.shape:
        raise ValueError("Parameterization changed the terminal layout.")
    observe = (
        (lambda device, u, v: device.terminal_current(u))
        if observable is None
        else observable
    )
    directions_ = (
        None if directions is None else jnp.asarray(directions, dtype=theta.dtype)
    )
    if directions_ is not None and (
        directions_.ndim != 2
        or directions_.shape[1] != theta.size
        or directions_.shape[0] == 0
        or bool(jnp.any(~jnp.isfinite(directions_)))
    ):
        raise ValueError("directions must be a finite nonempty (K, P) array.")
    u = point.coordinates
    flat = u.reshape(-1)
    scale = base.time_scale
    setup = _stiffness_setup(base, flat) if linear_policy is None else None
    setup, row_scale = _equilibrate_setup(setup) if setup is not None else (None, 1)
    policy = _linear_policy(
        linear_policy, flat.size, dense_reference_limit, setup_operator=setup
    )
    residual = lambda x, device, v: (
        scale * device.residual(x.reshape(u.shape), v)
    ).reshape(-1)
    _, stiffness = jax.linearize(lambda x: residual(x, base, volts), flat)
    native = prepare_linear(
        LinearSystem(_operator(lambda x: row_scale * stiffness(x), flat)), policy
    )

    def parameter_residual(value):
        device, voltage = builder(value)
        return residual(flat, device, voltage)

    _, parameter_action = jax.linearize(parameter_residual, theta)

    def observable_parameters(value):
        device, voltage = builder(value)
        return jnp.asarray(observe(device, u, voltage))

    value, observable_parameter_action = jax.linearize(observable_parameters, theta)
    _, observable_state_action = jax.linearize(
        lambda x: jnp.asarray(observe(base, x.reshape(u.shape), volts)), flat
    )
    base_residual = residual(flat, base, volts)
    point_valid = (
        jnp.all(point.successful)
        & jnp.all(jnp.isfinite(base_residual))
        & (jnp.max(jnp.abs(base_residual)) <= residual_tolerance)
        & jnp.all(jnp.isfinite(value))
    )
    derivatives, records = [], []
    count = theta.size if directions_ is None else directions_.shape[0]
    for index in range(count):
        direction = (
            jax.nn.one_hot(index, theta.size, dtype=theta.dtype)
            if directions_ is None
            else directions_[index]
        )
        tangent, valid, status, defect, threshold, unscaled = _solve_action(
            stiffness, -parameter_action(direction), policy, native, row_scale=row_scale
        )
        derivative = observable_state_action(tangent) + observable_parameter_action(
            direction
        )
        valid = valid & jnp.all(jnp.isfinite(derivative))
        derivatives.append(jnp.where(valid & point_valid, derivative, jnp.nan))
        records.append((valid, status, defect, threshold, unscaled))
    return SemiconductorSensitivityResult(
        value, jnp.stack(derivatives, axis=-1), _evidence(records, point_valid, (count,))
    )


def _same_topology(left, right):
    a, b = left.plan, right.plan
    if (
        a.support.source_topology_id != b.support.source_topology_id
        or a.terminal_names != b.terminal_names
        or a.support.positions.shape != b.support.positions.shape
    ):
        raise ValueError("Sensitivity requires fixed topology and terminal layout.")
    if a.layout.native.layout_id != b.layout.native.layout_id:
        raise ValueError("Sensitivity requires the same named state layout.")
    if left.coloring.pattern.pattern_id != right.coloring.pattern.pattern_id:
        raise ValueError("Sensitivity requires fixed state-coupling topology.")
    for x, y in (
        (a.support.tail, b.support.tail),
        (a.support.head, b.support.head),
        (a.support.node_ids, b.support.node_ids),
        (a.terminal_index, b.terminal_index),
        (a.semiconductor_mask, b.semiconductor_mask),
        (a.ohmic_mask, b.ohmic_mask),
        (a.potential_mask, b.potential_mask),
        (a.material_index, b.material_index),
        (jnp.asarray([a.layout.size]), jnp.asarray([b.layout.size])),
    ):
        if not np.array_equal(np.asarray(x), np.asarray(y)):
            raise ValueError(
                "Sensitivity parameterization changed discrete topology or contacts."
            )


def _storage_coordinates(prepared, u):
    return prepared.storage_coordinates(u)


def _coordinates_from_storage(prepared, z):
    return prepared.coordinates_from_storage(z)


def _differential_mask(prepared):
    return prepared.differential_mask.reshape(-1)


def _dae_policy(policy, size):
    if policy is not None:
        return policy
    method = NewtonKrylov(linear_policy=_linear_policy(None, size))
    # Physical-second stages can have large rate Jacobians: a small state
    # correction does not imply a small residual.
    termination = NonlinearTermination(
        absolute_residual=1e-8,
        relative_residual=0,
        absolute_step=0,
        relative_step=0,
        maximum_steps=32,
    )
    return DAESolvePolicy(
        nonlinear_method=method,
        initialization_method=method,
        nonlinear_termination=termination,
        adaptive=DAEAdaptivePolicy(),
    )


class _StageJacobian(StrictModule):
    derivative: SparseDerivativePlan

    def __call__(self, state, args):
        return self.derivative.operator(state, args)


def _dae_stage_method(problem, structure, time):
    system, initial = problem.system, problem.initial_state
    arguments = ImplicitStageArguments(
        time=time,
        shift=jnp.max(system.state_rate_scale),
        rate_offset=-jnp.max(system.state_rate_scale) * initial,
        explicit_value=jnp.zeros_like(initial),
        fallback_state=initial,
        active=True,
        model_args=problem.args,
    )
    source = _scaled_space(
        initial.shape,
        initial.dtype,
        system.state_scale,
        space_id=f"{system.system_id}:implicit-state",
    )
    target = _scaled_space(
        initial.shape,
        initial.dtype,
        jnp.ones_like(system.residual_scale),
        space_id=f"{system.system_id}:implicit-residual",
    )
    derivative = compile_sparse_jacobian(
        ImplicitStageResidual(system, problem.input_policy),
        initial,
        source=source,
        target=target,
        sample_args=arguments,
        structure=structure,
        compiler="native",
        mode="fwd",
    )
    operator = derivative.operator(initial, arguments)
    coordinates = ArraySpace(initial.shape, dtype=initial.dtype)
    setup = SparseCoordinateOperator(
        operator.relation, operator.coefficients, source=coordinates, target=coordinates
    )
    # The exact stage Jacobian refreshes at every Newton state. Its native
    # coordinate solve uses a frozen sparse transit-scale approximation.
    linear = _linear_policy(None, initial.size, setup_operator=setup)
    return NewtonKrylov(
        jacobian_policy=JacobianPolicy("explicit", operator=_StageJacobian(derivative)),
        linear_policy=linear,
    )


def _dae_problem(prepared, coordinates, voltage_function):
    shape = coordinates.shape
    mask = _differential_mask(prepared)
    roles = tuple("differential" if value else "algebraic" for value in np.asarray(mask))
    initial = _storage_coordinates(prepared, coordinates).reshape(-1)
    state_scale = jnp.maximum(jnp.abs(initial), 1)
    # Carrier residuals are relative to their initial stored population. Without
    # this scaling, BDF subtraction of a large neutral background amplifies
    # roundoff beyond an absolute ni-normalized residual tolerance.
    residual_scale = jnp.where(mask, state_scale, 1)

    def residual(time, state, state_rate, args):
        del args
        u = _coordinates_from_storage(prepared, state.reshape(shape))
        stored_rate = jnp.where(mask, state_rate, 0).reshape(shape)
        return (
            prepared.time_scale
            * (stored_rate + prepared.residual(u, voltage_function(time)))
        ).reshape(-1)

    system = DifferentialAlgebraicSystem(
        residual,
        state_shape=(coordinates.size,),
        structure=DAEStructure(roles),
        state_scale=state_scale,
        residual_scale=residual_scale,
        state_rate_scale=jnp.ones((coordinates.size,)) / prepared.time_scale,
        system_id=f"semiconductor:{prepared.plan.support.source_id}:density-dae",
    )
    return DifferentialAlgebraicProblem(
        system,
        initial,
        initialization=DAEInitializationSpec.index_one(),
    )


def _consistent_rate(prepared, u, volts, voltage_rate, policy):
    """Recover algebraic rates by differentiating constraints, not setting zero."""
    shape = u.shape
    state = _storage_coordinates(prepared, u).reshape(-1)
    mask = _differential_mask(prepared)
    scale = prepared.time_scale
    residual = lambda z, v: (
        scale
        * prepared.residual(_coordinates_from_storage(prepared, z.reshape(shape)), v)
    ).reshape(-1)
    value, stiffness = jax.linearize(lambda z: residual(z, volts), state)
    bias_rate = jax.jvp(lambda v: residual(state, v), (volts,), (voltage_rate,))[1]
    # Differential inventory rates follow balance laws. Contact/dielectric
    # algebraic rates are known physical-coordinate derivatives; solve only
    # unconstrained potential and material-interface trace rates.
    rate = -jnp.where(mask, value, 0)
    plan = prepared.plan
    fixed = prepared.layout.pack(
        potential=plan.potential_mask,
        electron=~(plan.semiconductor_mask & ~plan.ohmic_mask),
        hole=~(plan.semiconductor_mask & ~plan.ohmic_mask),
    ).astype(bool)
    if plan.electrothermal:
        fixed = prepared.layout.set(fixed, "lattice_energy", plan.ohmic_mask)
    if plan.carrier_energy:
        carrier_fixed = plan.ohmic_mask | ~plan.semiconductor_mask
        fixed = prepared.layout.set(fixed, "electron_energy", carrier_fixed)
        fixed = prepared.layout.set(fixed, "hole_energy", carrier_fixed)
    terminal_rate = voltage_rate[jnp.maximum(plan.terminal_index, 0)]
    coordinate_rate = prepared.layout.pack(
        potential=jnp.where(plan.potential_mask, terminal_rate / plan.thermal_voltage, 0),
        electron=jnp.where(plan.ohmic_mask, -terminal_rate / plan.thermal_voltage, 0),
        hole=jnp.where(plan.ohmic_mask, -terminal_rate / plan.thermal_voltage, 0),
    )
    fixed_storage_rate = jax.jvp(
        lambda coordinates: _storage_coordinates(prepared, coordinates),
        (u,),
        (coordinate_rate,),
    )[1].reshape(-1)
    fixed_flat = fixed.reshape(-1)
    rate = jnp.where(fixed_flat, scale * fixed_storage_rate, rate)
    free_indices = jnp.asarray(
        np.flatnonzero(np.asarray(~mask & ~fixed_flat)), dtype=jnp.int32
    )
    algebraic_indices = jnp.asarray(np.flatnonzero(np.asarray(~mask)), dtype=jnp.int32)
    constrained = stiffness(rate) + scale * bias_rate
    if free_indices.size:
        rhs = -constrained[free_indices]

        def free_direction(direction):
            return jnp.zeros_like(rate).at[free_indices].set(direction)

        action = lambda direction: stiffness(free_direction(direction))[free_indices]
        free_rate, solver_valid, status, _, _, _ = _solve_action(
            action,
            rhs,
            policy,
        )
        rate = rate.at[free_indices].set(free_rate)
    else:
        solver_valid = jnp.asarray(True)
        status = jnp.asarray(0, dtype=jnp.int32)
    final_constraint = (stiffness(rate) + scale * bias_rate)[algebraic_indices]
    defect = jnp.linalg.norm(final_constraint)
    reference = jnp.linalg.norm(constrained[algebraic_indices])
    threshold = policy.tolerance.absolute + policy.tolerance.relative * reference
    valid = (
        solver_valid
        & jnp.all(jnp.isfinite(rate))
        & jnp.isfinite(defect)
        & (defect <= threshold)
    )
    udot = jax.jvp(
        lambda z: _coordinates_from_storage(prepared, z.reshape(shape)),
        (state,),
        (rate / scale,),
    )[1]
    return udot, (valid, status, defect, threshold, defect)


def semiconductor_transient(
    prepared: PreparedSemiconductorDevice,
    initial_operating_point: SemiconductorOperatingPoint,
    times: TimeGrid | ArrayLike,
    voltage_function: Callable[[Array], Array],
    /,
    *,
    policy: DAESolvePolicy | None = None,
    rate_linear_policy: LinearSolvePolicy | None = None,
    dense_reference_limit: int | None = None,
) -> SemiconductorTransientResult:
    """Native adaptive DAE integration in physical seconds.

    Named solver charts are transformed to extensive carrier, lattice,
    carrier-energy and surface-population storage before BDF differencing.
    Algebraic potential and material-interface traces remain storage-free.
    Index-one consistency recovers their rates, including displacement current.
    ``voltage_function`` must be JAX differentiable; segment discontinuities
    into separate runs. No rejected sample is reported as a valid current.
    """
    _require_dynamic_charge_storage(prepared)
    grid = (
        times
        if isinstance(times, TimeGrid)
        else TimeGrid(times, time_id="semiconductor:requested-times")
    )
    u0 = initial_operating_point.coordinates
    v0 = jnp.asarray(voltage_function(grid.t0))
    if v0.shape != initial_operating_point.voltages.shape:
        raise ValueError("voltage_function must return one SI voltage per terminal.")
    selected = _dae_policy(policy, u0.size)
    if selected.adaptive is None:
        raise ValueError("semiconductor_transient requires a native adaptive DAE policy.")
    rate_policy = _linear_policy(rate_linear_policy, u0.size, dense_reference_limit)
    problem = _dae_problem(prepared, u0, voltage_function)
    if policy is None:
        selected = eqx.tree_at(
            lambda value: value.nonlinear_method,
            selected,
            _dae_stage_method(problem, prepared.coloring, grid.t0),
        )
    solution = solve_dae(problem, grid, policy=selected)
    coordinates = jax.vmap(
        lambda state: _coordinates_from_storage(prepared, state.reshape(u0.shape))
    )(solution.states)
    rates, currents, conduction, charges, voltages, records = [], [], [], [], [], []
    for index in range(grid.num_times):
        time, u = solution.times[index], coordinates[index]
        volts, voltage_rate = jax.jvp(voltage_function, (time,), (jnp.ones_like(time),))
        rate, record = _consistent_rate(prepared, u, volts, voltage_rate, rate_policy)
        rates.append(rate)
        currents.append(prepared.terminal_current(u, rate))
        conduction.append(prepared.terminal_current(u))
        charges.append(prepared.terminal_charge(u))
        voltages.append(volts)
        records.append(record)
    evidence = _evidence(
        records, jnp.all(initial_operating_point.successful), (grid.num_times,)
    )
    total, dc, charge, rate = (
        jnp.stack(currents),
        jnp.stack(conduction),
        jnp.stack(charges),
        jnp.stack(rates),
    )
    valid = solution.valid & evidence.successful & jnp.all(jnp.isfinite(total), axis=-1)
    mask = valid[:, None]
    state_mask = valid.reshape(
        (grid.num_times,) + (1,) * initial_operating_point.coordinates.ndim
    )
    return SemiconductorTransientResult(
        solution.times,
        coordinates,
        jnp.where(state_mask, rate, jnp.nan),
        jnp.stack(voltages),
        jnp.where(mask, total, jnp.nan),
        jnp.where(mask, dc, jnp.nan),
        jnp.where(mask, total - dc, jnp.nan),
        jnp.where(mask, charge, jnp.nan),
        valid,
        solution,
        evidence,
    )


__all__ = [
    "SemiconductorLinearEvidence",
    "SemiconductorSmallSignalResult",
    "SemiconductorSensitivityResult",
    "SemiconductorTransientResult",
    "semiconductor_small_signal",
    "semiconductor_sensitivity",
    "semiconductor_transient",
]
