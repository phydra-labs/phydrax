#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...circuit import (
    AbstractImplicitCircuitLaw,
    CircuitDAEPlan,
    CircuitDAERunResult,
    CircuitElementEvaluation,
    CircuitElementStateLayout,
    CircuitOperatingPointResult,
    plan_circuit_operating_point,
    PreparedCircuitDAE,
    solve_circuit_dae,
)
from ...dynamics import DifferentialAlgebraicSystem, TimeGrid
from ...linalg import (
    ArraySpace,
    ILUPreconditionerBuilder,
    LinearSolvePolicy,
    PreconditioningPolicy,
)
from ...nonlinear import (
    FunctionRightNonlinearPreconditioner,
    JacobianPolicy,
    NewtonKrylov,
    NonlinearTermination,
    RightPreconditionedSystem,
    root,
)
from ...solver import DAEInitializationSpec, DAESolvePolicy
from ...sparse import (
    compile_sparse_jacobian,
    EdgeRelation,
    SparseCoordinateOperator,
    SparseDerivativePlan,
    SparsePattern,
)
from ._analysis import (
    _coordinates_from_storage,
    _dae_policy,
    _differential_mask,
    _linear_policy,
    _require_dynamic_charge_storage,
    _storage_coordinates,
)
from ._continuum import PreparedSemiconductorDevice


class SemiconductorCircuitLaw(AbstractImplicitCircuitLaw):
    """Classical semiconductor model as a native implicit circuit element.

    Terminal voltages/currents use SI and current is positive from the circuit
    node into the device. Auxiliary state uses the prepared named layout and
    extensive-storage chart. ``initialize_rate`` transforms physical coordinate
    rates into that chart. Internal electrostatic algebraic rates enter terminal
    displacement current, so coupled transient initialization requires an
    explicit consistent state/rate pair or native custom mask contract.
    """

    prepared: PreparedSemiconductorDevice

    def __init__(self, prepared: PreparedSemiconductorDevice, /):
        self.prepared = prepared
        roles = tuple(
            "differential" if value else "algebraic"
            for value in np.asarray(_differential_mask(prepared))
        )
        self.terminal_count = len(prepared.plan.terminal_names)
        self.input_names = ()
        # Contact potential rates are supplied by terminal voltage rates.
        # Internal electrostatic rates still enter the displacement current.
        self.voltage_rate_dependent = True
        self.state_layout = CircuitElementStateLayout(
            roles,
            rate_scale=jnp.ones((len(roles),)) / prepared.time_scale,
            residual_scale=jnp.full((len(roles),), np.sqrt(len(roles))),
        )
        self.law_id = canonical_fingerprint(
            {
                "kind": "semiconductor-continuum-circuit-law",
                "coordinates": "scaled-carrier-storage",
                "support": prepared.plan.support.source_id,
                "revision": prepared.plan.support.source_revision,
                "terminals": prepared.plan.terminal_names,
                "physics": array_tree_fingerprint(prepared.plan),
                "state_size": self.state_layout.size,
            }
        )

    def initialize(self, coordinates: ArrayLike, /) -> Array:
        value = jnp.asarray(coordinates, dtype=float)
        expected = self.prepared.layout.shape
        if value.shape != expected or bool(jnp.any(~jnp.isfinite(value))):
            raise ValueError(
                f"Circuit device coordinates must be finite with shape {expected}."
            )
        return _storage_coordinates(self.prepared, value).reshape(-1)

    def coordinates(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != (self.state_layout.size,):
            raise ValueError("Circuit device state has the wrong size.")
        return _coordinates_from_storage(
            self.prepared, value.reshape(self.prepared.layout.shape)
        )

    def initialize_rate(
        self, coordinates: ArrayLike, coordinate_rate: ArrayLike, /
    ) -> Array:
        value, rate = jnp.asarray(coordinates), jnp.asarray(coordinate_rate)
        expected = self.prepared.layout.shape
        if value.shape != expected or rate.shape != expected:
            raise ValueError(
                f"Circuit device coordinates and rates must have shape {expected}."
            )
        return jax.jvp(
            lambda u: _storage_coordinates(self.prepared, u), (value,), (rate,)
        )[1].reshape(-1)

    def evaluate(
        self,
        time,
        terminal_voltages,
        terminal_voltage_rates,
        state,
        state_rate,
        inputs,
        args,
        /,
    ) -> CircuitElementEvaluation:
        del time, inputs, args
        u, udot = jax.jvp(self.coordinates, (state,), (state_rate,))
        plan = self.prepared.plan
        contact_rate = (
            terminal_voltage_rates[jnp.maximum(plan.terminal_index, 0)]
            / plan.thermal_voltage
        )
        potential_rate = self.prepared.field(udot, "potential")
        potential_rate = jnp.where(plan.potential_mask, contact_rate, potential_rate)
        udot = self.prepared.layout.set(udot, "potential", potential_rate)
        for name in ("electron", "hole"):
            carrier_rate = self.prepared.field(udot, name)
            carrier_rate = jnp.where(plan.ohmic_mask, -contact_rate, carrier_rate)
            udot = self.prepared.layout.set(udot, name, carrier_rate)
        stored_rate = jnp.where(
            self.prepared.differential_mask, state_rate.reshape(u.shape), 0
        )
        residual = self.prepared.time_scale * (
            self.prepared.residual(u, terminal_voltages) + stored_rate
        )
        return CircuitElementEvaluation(
            self.prepared.terminal_current(u, udot), residual.reshape(-1)
        )


def _with_current_scale(
    prepared: PreparedCircuitDAE, current_scale: ArrayLike
) -> PreparedCircuitDAE:
    """Declare physical KCL scales without changing current/diagnostic units."""
    count = len(prepared.plan.layout.node_ids)
    scale = jnp.asarray(current_scale, dtype=prepared.system.residual_scale.dtype)
    if scale.shape not in ((), (count,)):
        raise ValueError("current_scale must be scalar or one value per non-ground node.")
    if bool(jnp.any(~jnp.isfinite(scale) | (scale <= 0))):
        raise ValueError("current_scale must be finite and positive, in amperes.")
    scales = prepared.system.residual_scale.at[:count].set(scale)
    identity = canonical_fingerprint(
        {
            "kind": "semiconductor-circuit-current-scaling",
            "prepared": prepared.prepared_id,
            "scales": array_tree_fingerprint(scales),
        }
    )
    old = prepared.plan
    plan = CircuitDAEPlan(
        old.circuit,
        old.layout,
        old.laws,
        old.input_layout,
        old.input_bindings,
        old.state_scale,
        old.rate_scale,
        scales,
        identity + "/plan",
    )
    previous = prepared.system
    system = DifferentialAlgebraicSystem(
        previous.residual,
        state_shape=previous.state_shape,
        structure=previous.structure,
        input_layout=previous.input_layout,
        state_scale=previous.state_scale,
        state_rate_scale=previous.state_rate_scale,
        residual_scale=scales,
        state_geometry=previous.state_geometry,
        system_id=identity + "/dae",
    )
    return PreparedCircuitDAE(plan, system, identity)


class _CircuitJacobian(StrictModule):
    """Mixed-mode sparse differentiation avoids coloring terminal sums as PDE rows."""

    kcl: SparseDerivativePlan
    internal: SparseDerivativePlan
    template: SparseCoordinateOperator

    def __call__(self, state, args):
        coefficients = jnp.concatenate(
            (self.kcl.coefficients(state, args), self.internal.coefficients(state, args))
        )
        return eqx.tree_at(lambda value: value.coefficients, self.template, coefficients)


def _circuit_jacobian(prepared, problem, initial, args):
    layout = prepared.plan.layout
    size, node_count = layout.size, len(layout.node_ids)
    rows, cols = [np.arange(size)], [np.arange(size)]

    def block(row_indices, column_indices):
        shape = (len(row_indices), len(column_indices))
        rows.append(np.broadcast_to(np.asarray(row_indices)[:, None], shape).reshape(-1))
        cols.append(
            np.broadcast_to(np.asarray(column_indices)[None, :], shape).reshape(-1)
        )

    for instance, law, (start, stop) in zip(
        prepared.plan.circuit.instances,
        prepared.plan.laws,
        layout.auxiliary_ranges,
        strict=True,
    ):
        nodes = [
            layout.node_ids.index(node)
            for node in instance.nodes
            if node != prepared.plan.circuit.ground
        ]
        block(nodes, nodes)
        if isinstance(law, SemiconductorCircuitLaw):
            device = law.prepared
            pattern = device.coloring.pattern
            rows.append(start + np.asarray(pattern.rows))
            cols.append(start + np.asarray(pattern.cols))
            terminal = np.asarray(device.plan.terminal_index)
            for index, node in enumerate(instance.nodes):
                if node == prepared.plan.circuit.ground:
                    continue
                node_index = layout.node_ids.index(node)
                contact = np.flatnonzero(terminal == index)
                contact_rows = np.asarray(
                    device.layout.node_indices(contact), dtype=np.int32
                ).reshape(-1)
                dependencies = np.unique(
                    np.asarray(pattern.cols)[
                        np.isin(np.asarray(pattern.rows), contact_rows)
                    ]
                )
                block([node_index], start + dependencies)
                block(contact_rows + start, [node_index])
        else:
            auxiliary = np.arange(start, stop)
            block(nodes, auxiliary)
            block(auxiliary, nodes)
            block(auxiliary, auxiliary)
    rows, cols = (
        np.concatenate(rows).astype(np.int32),
        np.concatenate(cols).astype(np.int32),
    )
    kcl = rows < node_count
    kcl_pattern = SparsePattern.from_coo(rows[kcl], cols[kcl], (node_count, size))
    internal_pattern = SparsePattern.from_coo(
        rows[~kcl] - node_count, cols[~kcl], (size - node_count, size)
    )
    kcl_derivative = compile_sparse_jacobian(
        lambda state, parameters: problem.evaluate(state, parameters)[0][:node_count],
        initial,
        source=problem.state_space,
        target=ArraySpace((node_count,), dtype=initial.dtype),
        sample_args=args,
        structure=kcl_pattern,
        compiler="native",
        mode="rev",
    )
    internal_derivative = compile_sparse_jacobian(
        lambda state, parameters: problem.evaluate(state, parameters)[0][node_count:],
        initial,
        source=problem.state_space,
        target=ArraySpace((size - node_count,), dtype=initial.dtype),
        sample_args=args,
        structure=internal_pattern,
        compiler="native",
        mode="fwd",
    )
    relation = EdgeRelation(
        jnp.concatenate((kcl_pattern.cols, internal_pattern.cols)),
        jnp.concatenate((kcl_pattern.rows, internal_pattern.rows + node_count)),
        source_size=size,
        target_size=size,
    )
    template = SparseCoordinateOperator(
        relation,
        jnp.zeros(relation.route_shape, dtype=initial.dtype),
        source=problem.state_space,
        target=problem.residual_space,
    )
    return _CircuitJacobian(kcl_derivative, internal_derivative, template)


def semiconductor_circuit_operating_point(
    prepared_circuit: PreparedCircuitDAE,
    initial_state: ArrayLike,
    /,
    *,
    current_scale: ArrayLike,
    args: Any = None,
    linear_policy: LinearSolvePolicy | None = None,
    dense_reference_limit: int | None = None,
    termination: NonlinearTermination | None = None,
) -> CircuitOperatingPointResult:
    """Coupled native Newton-Krylov solve, without all-state dense circuit AD.

    Assemble the circuit with CircuitElement(SemiconductorCircuitLaw(device))
    and prepare_circuit_dae. This uses native residual actions directly, not
    circuit operating_point_jacobian or the dense circuit small-signal path.
    Native nonlinear and KCL diagnostics are returned without hiding failure.
    ``current_scale`` is the required scalar or per-non-ground-node KCL scale in
    amperes. Explicit scaling prevents a nanoampere imbalance being certified
    against a dimensionless root tolerance as if it were a negligible ampere.
    """
    prepared_circuit = _with_current_scale(prepared_circuit, current_scale)
    size = prepared_circuit.plan.layout.size
    selected = _linear_policy(linear_policy, size, dense_reference_limit)
    if linear_policy is None:
        # MNA has structural zero pivots. This regularizes only the approximate
        # inverse; Newton still certifies the original, unshifted physical KCL.
        selected = eqx.tree_at(
            lambda value: value.preconditioning,
            selected,
            PreconditioningPolicy(
                ILUPreconditionerBuilder(
                    ordering="reverse-cuthill-mckee",
                    allow_pivot_replacement=True,
                    replacement_value=1.0,
                ),
                side="right",
                refresh="numeric",
            ),
            is_leaf=lambda value: value is None,
        )
    initial = jnp.asarray(initial_state, dtype=float)
    plan = plan_circuit_operating_point(prepared_circuit)
    latent = initial
    for law, (start, stop) in zip(
        prepared_circuit.plan.laws,
        prepared_circuit.plan.layout.auxiliary_ranges,
        strict=True,
    ):
        if isinstance(law, SemiconductorCircuitLaw):
            latent = latent.at[start:stop].set(
                law.coordinates(initial[start:stop]).reshape(-1)
            )

    def reconstruct(value, parameters):
        del parameters
        state = value
        for law, (start, stop) in zip(
            prepared_circuit.plan.laws,
            prepared_circuit.plan.layout.auxiliary_ranges,
            strict=True,
        ):
            if isinstance(law, SemiconductorCircuitLaw):
                stored = _storage_coordinates(
                    law.prepared,
                    value[start:stop].reshape(law.prepared.layout.shape),
                )
                state = state.at[start:stop].set(stored.reshape(-1))
        return state

    transformed = RightPreconditionedSystem(
        plan.problem,
        FunctionRightNonlinearPreconditioner(
            reconstruct,
            source=plan.problem.state_space,
            target=plan.problem.state_space,
            preconditioner_id=f"{plan.plan_id}/quasi-fermi",
        ),
    )
    jacobian = _circuit_jacobian(prepared_circuit, transformed.problem, latent, args)
    nonlinear = root(
        transformed,
        latent,
        args=args,
        method=NewtonKrylov(
            jacobian_policy=JacobianPolicy("explicit", operator=jacobian),
            linear_policy=selected,
        ),
        termination=(
            NonlinearTermination(absolute_step=0, relative_step=0, maximum_steps=150)
            if termination is None
            else termination
        ),
    )
    identity = canonical_fingerprint(
        {
            "kind": "semiconductor-circuit-operating-point",
            "plan": plan.plan_id,
            "transformation": transformed.transformation_id,
            "linear": nonlinear.provenance.linear_plan_id,
        }
    )
    state = jnp.asarray(nonlinear.state)
    diagnostics = prepared_circuit.diagnostics(
        jnp.asarray(0.0), state, jnp.zeros_like(state), args
    )
    return CircuitOperatingPointResult(state, nonlinear, diagnostics, identity)


def semiconductor_circuit_transient(
    prepared_circuit: PreparedCircuitDAE,
    initial_state: ArrayLike,
    initial_state_rate: ArrayLike,
    times: TimeGrid | ArrayLike,
    /,
    *,
    current_scale: ArrayLike,
    args: Any = None,
    initialization: DAEInitializationSpec | None = None,
    policy: DAESolvePolicy | None = None,
) -> CircuitDAERunResult:
    """Native adaptive, matrix-free mixed circuit/device descriptor integration.

    The initial pair is explicit because electrostatic internal rates enter
    KCL. By default native check-only initialization verifies that pair and
    reports INITIALIZATION_FAILED on inconsistency, with no fictitious repair.
    A caller with a structurally reduced circuit may supply a native custom
    free/fixed initialization contract instead. DC equilibria with stationary
    sources admit zero initial rates. Time-dependent source args follow native
    circuit conventions. Inspect result.solution.valid and rate_valid before
    using samples, and termination_status before continuing the trajectory.
    """
    for law in prepared_circuit.plan.laws:
        if isinstance(law, SemiconductorCircuitLaw):
            _require_dynamic_charge_storage(law.prepared)
    prepared_circuit = _with_current_scale(prepared_circuit, current_scale)
    grid = (
        times
        if isinstance(times, TimeGrid)
        else TimeGrid(times, time_id="semiconductor:coupled-times")
    )
    selected = _dae_policy(policy, prepared_circuit.plan.layout.size)
    if selected.adaptive is None:
        raise ValueError(
            "Coupled semiconductor transients require adaptive DAE integration."
        )
    for method in (selected.nonlinear_method, selected.initialization_method):
        if (
            not isinstance(method, NewtonKrylov)
            or method.jacobian_policy.mode == "explicit"
        ):
            raise ValueError(
                "Coupled semiconductor stages and initialization require matrix-free NewtonKrylov."
            )
        _linear_policy(method.linear_policy, prepared_circuit.plan.layout.size)
    spec = (
        DAEInitializationSpec.check_only() if initialization is None else initialization
    )
    return solve_circuit_dae(
        prepared_circuit,
        initial_state,
        grid,
        initial_state_rate=initial_state_rate,
        args=args,
        initialization=spec,
        policy=selected,
    )


__all__ = [
    "SemiconductorCircuitLaw",
    "semiconductor_circuit_operating_point",
    "semiconductor_circuit_transient",
]
