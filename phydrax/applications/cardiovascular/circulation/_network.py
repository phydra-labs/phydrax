#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....dynamics import (
    AcausalDAESource,
    compile_acausal_dae,
    DAEConnection,
    DAEStructuralPolicy,
    ReducedDAECompilation,
)
from ....linalg import DenseLU, LinearSolvePolicy
from ....nonlinear import (
    NewtonKrylov,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
    RobustRoot,
    root,
)
from ._components import PressureFlowComponent, StorageOwner


class PressureFlowConnection(StrictModule):
    """Directed equality connection between two hydraulic DAE ports."""

    left_component: str = eqx.field(static=True)
    left_port: str = eqx.field(static=True)
    right_component: str = eqx.field(static=True)
    right_port: str = eqx.field(static=True)
    connection_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_component: str,
        left_port: str,
        right_component: str,
        right_port: str,
        /,
    ) -> None:
        endpoints = tuple(
            str(value).strip()
            for value in (left_component, left_port, right_component, right_port)
        )
        if any(not value for value in endpoints):
            raise ValueError("Pressure/flow connection identifiers must be non-empty.")
        if endpoints[:2] == endpoints[2:]:
            raise ValueError("A pressure/flow port cannot connect to itself.")
        (
            self.left_component,
            self.left_port,
            self.right_component,
            self.right_port,
        ) = endpoints
        self.connection_id = canonical_fingerprint(
            {"kind": "pressure-flow-connection", "endpoints": list(endpoints)}
        )

    @property
    def left_port_id(self) -> str:
        return f"{self.left_component}.{self.left_port}"

    @property
    def right_port_id(self) -> str:
        return f"{self.right_component}.{self.right_port}"

    def as_dae_connection(self) -> DAEConnection:
        return DAEConnection(
            (self.left_port_id, self.right_port_id),
            (1, -1),
        )


class CirculationNetwork(StrictModule):
    """Validated acausal 0D circulation network and canonical DAE source."""

    components: tuple[PressureFlowComponent, ...]
    connections: tuple[PressureFlowConnection, ...]
    source: AcausalDAESource
    network_id: str = eqx.field(static=True)
    closed: bool = eqx.field(static=True)
    storage_ids: tuple[str, ...] = eqx.field(static=True)
    mechanics_storage_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        components: Sequence[PressureFlowComponent],
        connections: Sequence[PressureFlowConnection],
        /,
    ) -> None:
        component_values = tuple(components)
        connection_values = tuple(connections)
        if not component_values or any(
            not isinstance(value, PressureFlowComponent) for value in component_values
        ):
            raise ValueError("CirculationNetwork requires pressure/flow components.")
        if any(
            not isinstance(value, PressureFlowConnection) for value in connection_values
        ):
            raise TypeError("connections must contain PressureFlowConnection values.")
        by_name = {value.name: value for value in component_values}
        if len(by_name) != len(component_values):
            raise ValueError("Circulation component names must be unique.")
        all_ports = {
            f"{component.name}.{port.name}"
            for component in component_values
            for port in component.ports
        }
        used_ports: list[str] = []
        for connection in connection_values:
            if (
                connection.left_component not in by_name
                or connection.right_component not in by_name
            ):
                raise ValueError("Connection references an unknown component.")
            left = by_name[connection.left_component].port(connection.left_port)
            right = by_name[connection.right_component].port(connection.right_port)
            if (
                len(left.potentials) != 1
                or len(left.flows) != 1
                or len(right.potentials) != 1
                or len(right.flows) != 1
            ):
                raise ValueError(
                    "Cardiovascular hydraulic ports require one pressure and one flow."
                )
            used_ports.extend((connection.left_port_id, connection.right_port_id))
        if len(set(used_ports)) != len(used_ports):
            raise ValueError(
                "Each pressure/flow port may occur in at most one connection."
            )
        source = AcausalDAESource(
            tuple(value.dae_component for value in component_values),
            tuple(value.as_dae_connection() for value in connection_values),
        )
        circulation_storage = tuple(
            f"{component.name}.{variable}"
            for component in component_values
            if component.storage_owner is StorageOwner.CIRCULATION
            for variable in component.storage_variable_names
        )
        mechanics_storage = tuple(
            component.name
            for component in component_values
            if component.storage_owner is StorageOwner.MECHANICS
        )
        if set(circulation_storage) & set(mechanics_storage):
            raise ValueError("A storage state cannot have two owners.")
        self.components = component_values
        self.connections = connection_values
        self.source = source
        self.closed = bool(all_ports) and set(used_ports) == all_ports
        self.storage_ids = circulation_storage
        self.mechanics_storage_ids = mechanics_storage
        self.network_id = canonical_fingerprint(
            {
                "kind": "circulation-network",
                "components": [value.component_id for value in component_values],
                "connections": [value.connection_id for value in connection_values],
                "source": source.source_id,
                "closed": self.closed,
                "storage": list(circulation_storage),
                "mechanics_storage": list(mechanics_storage),
            }
        )

    def component(self, name: str, /) -> PressureFlowComponent:
        for value in self.components:
            if value.name == name:
                return value
        raise KeyError(f"Unknown circulation component {name!r}.")

    def replace_component(
        self,
        name: str,
        replacement: PressureFlowComponent,
        /,
    ) -> CirculationNetwork:
        if not isinstance(replacement, PressureFlowComponent):
            raise TypeError("replacement must be a PressureFlowComponent.")
        original = self.component(name)
        if replacement.name != original.name:
            raise ValueError("Replacement component must preserve the component name.")
        if tuple(port.name for port in replacement.ports) != tuple(
            port.name for port in original.ports
        ):
            raise ValueError("Replacement component must preserve the port names.")
        components = tuple(
            replacement if value.name == name else value for value in self.components
        )
        return CirculationNetwork(components, self.connections)


class ConsistentInitializationPlan(StrictModule):
    """Finite-capacity structural and nonlinear initialization policy."""

    structural_policy: DAEStructuralPolicy
    termination: NonlinearTermination
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_differentiations: int = 2,
        maximum_tears: int = 128,
        absolute_residual: float = 1.0e-9,
        relative_residual: float = 1.0e-9,
        maximum_steps: int = 64,
    ) -> None:
        structural = DAEStructuralPolicy(
            maximum_differentiations,
            maximum_tears,
            tearing="automatic",
        )
        termination = NonlinearTermination(
            absolute_residual=absolute_residual,
            relative_residual=relative_residual,
            maximum_steps=maximum_steps,
        )
        self.structural_policy = structural
        self.termination = termination
        self.plan_id = canonical_fingerprint(
            {
                "kind": "circulation-consistent-initialization-plan",
                "structural_policy": structural.policy_id,
                "absolute_residual": float(absolute_residual).hex(),
                "relative_residual": float(relative_residual).hex(),
                "maximum_steps": int(maximum_steps),
            }
        )


class PreparedConsistentInitialization(StrictModule):
    """Prepared fixed-shape DAE initialization state."""

    network: CirculationNetwork
    plan: ConsistentInitializationPlan
    compilation: ReducedDAECompilation
    default_state: Array
    prepared_id: str = eqx.field(static=True)


class ConsistentInitializationEvidence(StrictModule):
    residual: Array
    residual_norm: Array
    scaled_residual_norm: Array
    finite: Array
    successful: Array
    nonlinear_status: Array
    initialization_id: str = eqx.field(static=True)


class ConsistentInitializationResult(StrictModule):
    state: Array
    state_rate: Array
    jet: Any
    evidence: ConsistentInitializationEvidence
    nonlinear_result: NonlinearResult
    prepared: PreparedConsistentInitialization


def _default_state(
    network: CirculationNetwork,
    compilation: ReducedDAECompilation,
    /,
) -> Array:
    values = []
    for global_name in compilation.analysis.variable_names:
        component_name, variable_name = global_name.split(".", maxsplit=1)
        values.append(network.component(component_name).initial_value(variable_name))
    return jnp.stack(tuple(values))


def prepare_consistent_initialization(
    network: CirculationNetwork,
    plan: ConsistentInitializationPlan | None = None,
    /,
    *,
    args: Any = None,
) -> PreparedConsistentInitialization:
    """Compile the declared acausal network through the generic DAE compiler."""

    if not isinstance(network, CirculationNetwork):
        raise TypeError("network must be a CirculationNetwork.")
    if not network.closed:
        raise ValueError("Consistent initialization requires a closed network.")
    resolved_plan = ConsistentInitializationPlan() if plan is None else plan
    if not isinstance(resolved_plan, ConsistentInitializationPlan):
        raise TypeError("plan must be a ConsistentInitializationPlan or None.")
    compilation = compile_acausal_dae(
        network.source,
        resolved_plan.structural_policy,
        args=args,
    )
    default_state = _default_state(network, compilation)
    return PreparedConsistentInitialization(
        network,
        resolved_plan,
        compilation,
        default_state,
        canonical_fingerprint(
            {
                "kind": "prepared-circulation-consistent-initialization",
                "network": network.network_id,
                "plan": resolved_plan.plan_id,
                "compilation": compilation.compilation_id,
            }
        ),
    )


def initialize_consistent_state(
    prepared: PreparedConsistentInitialization,
    initial_state: ArrayLike | None = None,
    /,
    *,
    time: ArrayLike = 0.0,
    args: Any = None,
) -> ConsistentInitializationResult:
    """Hold differential states fixed and solve algebraic states/rates together."""

    if not isinstance(prepared, PreparedConsistentInitialization):
        raise TypeError("prepared must be PreparedConsistentInitialization.")
    time_ = jnp.asarray(time)
    if time_.shape != () or not bool(jnp.isfinite(time_)):
        raise ValueError("time must be one finite scalar.")
    state_seed = (
        prepared.default_state if initial_state is None else jnp.asarray(initial_state)
    )
    expected = prepared.compilation.system.state_shape
    if state_seed.shape != expected:
        raise ValueError(f"initial_state must have shape {expected}.")
    if not jnp.issubdtype(state_seed.dtype, jnp.inexact):
        state_seed = state_seed.astype(float)
    if not bool(jnp.all(jnp.isfinite(state_seed))):
        raise ValueError("initial_state must be finite.")
    differential = prepared.compilation.fixed_state_mask
    unknown_seed = jnp.where(differential, jnp.zeros_like(state_seed), state_seed)
    seed_state = jnp.where(differential, state_seed, unknown_seed)
    seed_rate = jnp.where(differential, unknown_seed, jnp.zeros_like(unknown_seed))
    seed_residual = prepared.compilation.residual_audit(
        time_, seed_state, seed_rate, args
    )
    residual_scale = jnp.maximum(jnp.abs(seed_residual), 1.0)

    def residual(unknown: Array, user_args: Any) -> Array:
        state = jnp.where(differential, state_seed, unknown)
        state_rate = jnp.where(differential, unknown, jnp.zeros_like(unknown))
        return (
            prepared.compilation.residual_audit(time_, state, state_rate, user_args)
            / residual_scale
        )

    problem = NonlinearSystemProblem(
        residual,
        problem_id=f"circulation-initialization:{prepared.prepared_id}",
    )
    method = (
        NewtonKrylov(linear_policy=LinearSolvePolicy(DenseLU()))
        if prepared.network.storage_ids
        else RobustRoot()
    )
    nonlinear_result = root(
        problem,
        unknown_seed,
        method=method,
        termination=prepared.plan.termination,
        args=args,
    )
    unknown = nonlinear_result.state
    state = jnp.where(differential, state_seed, unknown)
    state_rate = jnp.where(differential, unknown, jnp.zeros_like(unknown))
    residual_value = prepared.compilation.residual_audit(time_, state, state_rate, args)
    residual_norm = jnp.max(jnp.abs(residual_value))
    scaled_residual_norm = jnp.max(jnp.abs(residual_value) / residual_scale)
    finite = (
        jnp.all(jnp.isfinite(state))
        & jnp.all(jnp.isfinite(state_rate))
        & jnp.all(jnp.isfinite(residual_value))
    )
    successful = (
        nonlinear_result.successful
        & finite
        & (
            scaled_residual_norm
            <= prepared.plan.termination.residual_threshold(
                nonlinear_result.diagnostics.initial_residual_norm
            )
        )
    )
    evidence = ConsistentInitializationEvidence(
        residual_value,
        residual_norm,
        scaled_residual_norm,
        finite,
        successful,
        jnp.asarray(nonlinear_result.status),
        canonical_fingerprint(
            {
                "kind": "circulation-consistent-initialization-evidence",
                "prepared": prepared.prepared_id,
            }
        ),
    )
    return ConsistentInitializationResult(
        state,
        state_rate,
        prepared.compilation.reconstruction(state, state_rate),
        evidence,
        nonlinear_result,
        prepared,
    )


def circulation_state_values(
    result: ConsistentInitializationResult,
    /,
) -> dict[str, Array]:
    """Return zeroth-order global DAE values keyed by stable variable name."""

    if not isinstance(result, ConsistentInitializationResult):
        raise TypeError("result must be a ConsistentInitializationResult.")
    return {
        name: derivatives[0]
        for name, derivatives in zip(
            result.jet.variable_names,
            result.jet.derivatives,
            strict=True,
        )
    }


__all__ = [
    "CirculationNetwork",
    "ConsistentInitializationEvidence",
    "ConsistentInitializationPlan",
    "ConsistentInitializationResult",
    "PressureFlowConnection",
    "PreparedConsistentInitialization",
    "circulation_state_values",
    "initialize_consistent_state",
    "prepare_consistent_initialization",
]
