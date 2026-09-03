#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Gauge-fixed cardiac bidomain and optional torso finite-element DAE."""

from __future__ import annotations

from enum import IntFlag
from math import factorial, isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    SmallLinearSolvePlan,
    solve,
    solve_small_linear,
)


class BidomainSolveStatus(IntFlag):
    """Fail-closed status for one monolithic field step."""

    SUCCESS = 0
    NONFINITE = 1
    RESIDUAL_FAILURE = 2
    GAUGE_FAILURE = 4
    INCOMPATIBLE_EXTERNAL_SOURCE = 8
    PRECONDITIONER_FAILURE = 16


class HeartOnlyBidomainRoute(StrictModule, NonTrainableState):
    """Bidomain route without a surrounding volume conductor."""

    route_id: str = eqx.field(static=True)

    def __init__(self):
        self.route_id = canonical_fingerprint(
            {"kind": "cardiovascular-heart-only-bidomain-route"}
        )


class HeartTorsoBidomainRoute(StrictModule, NonTrainableState):
    """Torso volume conductor and fixed heart--torso interface support."""

    torso_node_ids: Array
    torso_element_ids: Array
    torso_nodes_mm: Array
    torso_elements: Array
    torso_conductivity_mS_per_mm: Array
    interface_ids: Array
    interface_node_pairs: Array
    interface_conductance_mS: Array
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        torso_node_ids: ArrayLike,
        torso_element_ids: ArrayLike,
        torso_nodes_mm: ArrayLike,
        torso_elements: ArrayLike,
        torso_conductivity_mS_per_mm: ArrayLike,
        interface_ids: ArrayLike,
        interface_node_pairs: ArrayLike,
        interface_conductance_mS: ArrayLike,
        /,
    ):
        node_ids = np.asarray(torso_node_ids, dtype=np.int64)
        element_ids = np.asarray(torso_element_ids, dtype=np.int64)
        nodes = np.asarray(torso_nodes_mm, dtype=float)
        elements = np.asarray(torso_elements, dtype=np.int32)
        identifiers = np.asarray(interface_ids, dtype=np.int64)
        pairs = np.asarray(interface_node_pairs, dtype=np.int32)
        conductance = np.asarray(interface_conductance_mS, dtype=float)
        _validate_mesh_ids(node_ids, element_ids, nodes, elements, "torso")
        dimension = nodes.shape[1]
        tensors = _conductivity_field(
            torso_conductivity_mS_per_mm,
            elements.shape[0],
            dimension,
            "torso_conductivity_mS_per_mm",
        )
        if identifiers.ndim != 1 or identifiers.size == 0:
            raise ValueError("interface_ids must be a non-empty fixed support vector.")
        if pairs.shape != (identifiers.size, 2) or conductance.shape != identifiers.shape:
            raise ValueError("Interface pairs and conductance must match interface IDs.")
        if np.any(identifiers < 0) or np.unique(identifiers).size != identifiers.size:
            raise ValueError("Interface IDs must be unique nonnegative stable integers.")
        if np.any(pairs[:, 1] < 0) or np.any(pairs[:, 1] >= nodes.shape[0]):
            raise ValueError("A torso interface index lies outside torso node capacity.")
        if not np.all(np.isfinite(conductance)) or np.any(conductance <= 0.0):
            raise ValueError("Interface conductances must be finite and positive.")
        self.torso_node_ids = jnp.asarray(node_ids)
        self.torso_element_ids = jnp.asarray(element_ids)
        self.torso_nodes_mm = jnp.asarray(nodes)
        self.torso_elements = jnp.asarray(elements)
        self.torso_conductivity_mS_per_mm = jnp.asarray(tensors)
        self.interface_ids = jnp.asarray(identifiers)
        self.interface_node_pairs = jnp.asarray(pairs)
        self.interface_conductance_mS = jnp.asarray(conductance)
        self.route_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-heart-torso-bidomain-route",
                "arrays": array_tree_fingerprint(
                    (
                        node_ids,
                        element_ids,
                        nodes,
                        elements,
                        tensors,
                        identifiers,
                        pairs,
                        conductance,
                    )
                ),
            }
        )


BidomainRoute = HeartOnlyBidomainRoute | HeartTorsoBidomainRoute


class BidomainFEMPlan(StrictModule, NonTrainableState):
    """Affine-P1 heart tensors, time policy, and explicit volume-conductor route."""

    route: BidomainRoute
    heart_node_ids: Array
    heart_element_ids: Array
    heart_nodes_mm: Array
    heart_elements: Array
    intracellular_conductivity_mS_per_mm: Array
    extracellular_conductivity_mS_per_mm: Array
    dt_ms: float = eqx.field(static=True)
    membrane_capacitance_uF_per_mm3: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    gauge_tolerance_mV: float = eqx.field(static=True)
    source_compatibility_tolerance_uA: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        route: BidomainRoute,
        heart_node_ids: ArrayLike,
        heart_element_ids: ArrayLike,
        heart_nodes_mm: ArrayLike,
        heart_elements: ArrayLike,
        intracellular_conductivity_mS_per_mm: ArrayLike,
        extracellular_conductivity_mS_per_mm: ArrayLike,
        /,
        *,
        dt_ms: float,
        membrane_capacitance_uF_per_mm3: float,
        residual_tolerance: float = 1.0e-9,
        gauge_tolerance_mV: float = 1.0e-9,
        source_compatibility_tolerance_uA: float = 1.0e-9,
    ):
        if not isinstance(route, (HeartOnlyBidomainRoute, HeartTorsoBidomainRoute)):
            raise TypeError("route must be a heart-only or heart--torso bidomain route.")
        node_ids = np.asarray(heart_node_ids, dtype=np.int64)
        element_ids = np.asarray(heart_element_ids, dtype=np.int64)
        nodes = np.asarray(heart_nodes_mm, dtype=float)
        elements = np.asarray(heart_elements, dtype=np.int32)
        _validate_mesh_ids(node_ids, element_ids, nodes, elements, "heart")
        dimension = nodes.shape[1]
        intracellular = _conductivity_field(
            intracellular_conductivity_mS_per_mm,
            elements.shape[0],
            dimension,
            "intracellular_conductivity_mS_per_mm",
        )
        extracellular = _conductivity_field(
            extracellular_conductivity_mS_per_mm,
            elements.shape[0],
            dimension,
            "extracellular_conductivity_mS_per_mm",
        )
        if isinstance(route, HeartTorsoBidomainRoute):
            if route.torso_nodes_mm.shape[1] != dimension:
                raise ValueError(
                    "Heart and torso meshes must have the same ambient dimension."
                )
            pairs = np.asarray(route.interface_node_pairs)
            if np.any(pairs[:, 0] < 0) or np.any(pairs[:, 0] >= nodes.shape[0]):
                raise ValueError(
                    "A heart interface index lies outside heart node capacity."
                )
        step = float(dt_ms)
        capacitance = float(membrane_capacitance_uF_per_mm3)
        residual = float(residual_tolerance)
        gauge = float(gauge_tolerance_mV)
        source = float(source_compatibility_tolerance_uA)
        if not isfinite(step) or step <= 0.0:
            raise ValueError("dt_ms must be finite and positive.")
        if not isfinite(capacitance) or capacitance <= 0.0:
            raise ValueError(
                "membrane_capacitance_uF_per_mm3 must be finite and positive."
            )
        if not all(
            isfinite(value) and value > 0.0 for value in (residual, gauge, source)
        ):
            raise ValueError("Bidomain evidence tolerances must be finite and positive.")
        self.route = route
        self.heart_node_ids = jnp.asarray(node_ids)
        self.heart_element_ids = jnp.asarray(element_ids)
        self.heart_nodes_mm = jnp.asarray(nodes)
        self.heart_elements = jnp.asarray(elements)
        self.intracellular_conductivity_mS_per_mm = jnp.asarray(intracellular)
        self.extracellular_conductivity_mS_per_mm = jnp.asarray(extracellular)
        self.dt_ms = step
        self.membrane_capacitance_uF_per_mm3 = capacitance
        self.residual_tolerance = residual
        self.gauge_tolerance_mV = gauge
        self.source_compatibility_tolerance_uA = source
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-bidomain-fem-plan",
                "route": route.route_id,
                "arrays": array_tree_fingerprint(
                    (node_ids, element_ids, nodes, elements, intracellular, extracellular)
                ),
                "dt_ms": step,
                "membrane_capacitance_uF_per_mm3": capacitance,
                "residual_tolerance": residual,
                "gauge_tolerance_mV": gauge,
                "source_compatibility_tolerance_uA": source,
            }
        )

    def prepare(self, /) -> "PreparedBidomainFEM":
        return prepare_bidomain_fem(self)


class PreparedBidomainFEM(StrictModule, NonTrainableState):
    """Assembled fixed-topology multi-field operator and block substrates."""

    plan: BidomainFEMPlan
    heart_mass_matrix: Array
    intracellular_stiffness: Array
    extracellular_stiffness: Array
    torso_mass_matrix: Array
    torso_stiffness: Array
    interface_coupling_matrix: Array
    gauge_weights: Array
    system_matrix: Array
    vm_preconditioner_block: Array
    potential_preconditioner_block: Array
    ungauged_nullspace_residual: Array
    prepared_id: str = eqx.field(static=True)

    @property
    def heart_node_count(self) -> int:
        return int(self.plan.heart_node_ids.shape[0])

    @property
    def torso_node_count(self) -> int:
        if isinstance(self.plan.route, HeartTorsoBidomainRoute):
            return int(self.plan.route.torso_node_ids.shape[0])
        return 0

    @property
    def field_size(self) -> int:
        return 2 * self.heart_node_count + self.torso_node_count + 1


class BidomainState(StrictModule):
    """Complete Vm, extracellular, optional torso, gauge multiplier, and time state."""

    transmembrane_voltage_mV: Array
    extracellular_potential_mV: Array
    torso_potential_mV: Array
    gauge_multiplier: Array
    time_ms: Array
    step_index: Array
    prepared_id: str = eqx.field(static=True)


class BidomainStepInputs(StrictModule):
    """Nodal source densities; ionic current is outward-positive."""

    ionic_current_uA_per_mm3: Array
    transmembrane_stimulus_uA_per_mm3: Array
    extracellular_stimulus_uA_per_mm3: Array
    torso_source_uA_per_mm3: Array


class BidomainBlockResidualEvidence(StrictModule):
    transmembrane_norm: Array
    extracellular_norm: Array
    torso_norm: Array
    gauge_norm: Array
    total_norm: Array
    relative_norm: Array


class BidomainGaugeEvidence(StrictModule):
    weighted_mean_potential_mV: Array
    constraint_residual: Array
    ungauged_nullspace_residual: Array
    fixed_gauge: Array


class BidomainInterfaceEvidence(StrictModule):
    potential_jump_norm_mV: Array
    interface_current_norm_uA: Array
    flux_balance_error_uA: Array
    supported: Array


class BidomainPreconditionerEvidence(StrictModule):
    minimum_absolute_diagonal: Array
    maximum_absolute_diagonal: Array
    diagonal_condition_estimate: Array
    input_residual_norm: Array
    preconditioned_action_defect_norm: Array
    relative_action_defect: Array
    finite: Array


class BidomainPreconditionerApplication(StrictModule):
    value: Array
    evidence: BidomainPreconditionerEvidence


class BidomainStepEvidence(StrictModule):
    block_residual: BidomainBlockResidualEvidence
    gauge: BidomainGaugeEvidence
    interface: BidomainInterfaceEvidence
    preconditioner: BidomainPreconditionerEvidence
    external_source_compatibility_uA: Array
    finite: Array
    status: Array
    successful: Array


class BidomainStepResult(StrictModule):
    state: BidomainState
    candidate_state: BidomainState
    evidence: BidomainStepEvidence


class MonodomainLimitEvidence(StrictModule):
    proportional_tensor_error: Array
    residual_norm: Array
    finite: Array
    successful: Array


class MonodomainLimitResult(StrictModule):
    transmembrane_voltage_mV: Array
    evidence: MonodomainLimitEvidence


def _validate_mesh_ids(
    node_ids: np.ndarray,
    element_ids: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
    name: str,
) -> None:
    if node_ids.ndim != 1 or node_ids.size < 2:
        raise ValueError(f"{name}_node_ids must contain at least two stable IDs.")
    if np.any(node_ids < 0) or np.unique(node_ids).size != node_ids.size:
        raise ValueError(f"{name} node IDs must be unique nonnegative integers.")
    if (
        nodes.ndim != 2
        or nodes.shape[0] != node_ids.size
        or nodes.shape[1] not in (1, 2, 3)
    ):
        raise ValueError(f"{name}_nodes_mm must have shape [node, dimension].")
    if not np.all(np.isfinite(nodes)):
        raise ValueError(f"{name} nodes must be finite.")
    if elements.ndim != 2 or elements.shape[0] == 0:
        raise ValueError(f"{name}_elements must be a non-empty simplex table.")
    if elements.shape[1] != nodes.shape[1] + 1:
        raise ValueError(f"{name} finite elements must be affine simplices.")
    if element_ids.shape != (elements.shape[0],):
        raise ValueError(f"{name}_element_ids must have one stable ID per element.")
    if np.any(element_ids < 0) or np.unique(element_ids).size != element_ids.size:
        raise ValueError(f"{name} element IDs must be unique nonnegative integers.")
    if np.any(elements < 0) or np.any(elements >= node_ids.size):
        raise ValueError(f"{name} element incidence lies outside node capacity.")
    if any(np.unique(cell).size != cell.size for cell in elements):
        raise ValueError(f"Every {name} simplex must contain distinct nodes.")


def _conductivity_field(
    values: ArrayLike,
    element_count: int,
    dimension: int,
    name: str,
) -> np.ndarray:
    tensors = np.asarray(values, dtype=float)
    if tensors.shape == (dimension, dimension):
        tensors = np.broadcast_to(tensors, (element_count, dimension, dimension)).copy()
    if tensors.shape != (element_count, dimension, dimension):
        raise ValueError(f"{name} must have shape [element, dimension, dimension].")
    if not np.all(np.isfinite(tensors)) or not np.allclose(
        tensors, np.swapaxes(tensors, -1, -2), rtol=1.0e-10, atol=1.0e-12
    ):
        raise ValueError(f"{name} tensors must be finite and symmetric.")
    if np.any(np.linalg.eigvalsh(tensors) <= 0.0):
        raise ValueError(f"{name} tensors must be positive definite.")
    return tensors


def _assemble_simplex_operators(
    nodes: np.ndarray,
    elements: np.ndarray,
    conductivity: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = nodes.shape[1]
    local_count = dimension + 1
    mass = np.zeros((nodes.shape[0], nodes.shape[0]), dtype=nodes.dtype)
    stiffness = np.zeros_like(mass)
    reference_gradients = np.concatenate(
        (-np.ones((1, dimension)), np.eye(dimension)), axis=0
    )
    jacobians = np.swapaxes(nodes[elements[:, 1:]] - nodes[elements[:, 0, None]], 1, 2)
    identities = np.broadcast_to(np.eye(dimension, dtype=nodes.dtype), jacobians.shape)
    inverse_result = solve_small_linear(
        SmallLinearSolvePlan(dimension),
        jnp.asarray(jacobians),
        jnp.asarray(identities),
    )
    if not bool(jnp.all(inverse_result.successful)):
        raise ValueError("Every bidomain simplex must have a nonsingular affine map.")
    inverse_jacobians = np.asarray(inverse_result.value)
    determinants = np.asarray(inverse_result.determinant)
    volumes = np.abs(determinants) / factorial(dimension)
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise ValueError("Every bidomain simplex must have positive measure.")
    gradients = contract("ia,eab->eib", reference_gradients, inverse_jacobians)
    local_stiffness = volumes[:, None, None] * contract(
        "eia,eab,ejb->eij", gradients, conductivity, gradients
    )
    mass_template = (np.ones((local_count, local_count)) + np.eye(local_count)) / (
        local_count * (local_count + 1)
    )
    local_mass = volumes[:, None, None] * mass_template[None, :, :]
    for element_index, cell in enumerate(elements):
        rows = np.repeat(cell, local_count)
        columns = np.tile(cell, local_count)
        np.add.at(mass, (rows, columns), local_mass[element_index].reshape(-1))
        np.add.at(
            stiffness,
            (rows, columns),
            local_stiffness[element_index].reshape(-1),
        )
    return mass, stiffness


def prepare_bidomain_fem(plan: BidomainFEMPlan, /) -> PreparedBidomainFEM:
    """Assemble the monolithic DAE, gauge constraint, and block preconditioner."""

    if not isinstance(plan, BidomainFEMPlan):
        raise TypeError("plan must be a BidomainFEMPlan.")
    heart_nodes = np.asarray(plan.heart_nodes_mm)
    heart_elements = np.asarray(plan.heart_elements)
    heart_mass, intracellular = _assemble_simplex_operators(
        heart_nodes,
        heart_elements,
        np.asarray(plan.intracellular_conductivity_mS_per_mm),
    )
    _, extracellular = _assemble_simplex_operators(
        heart_nodes,
        heart_elements,
        np.asarray(plan.extracellular_conductivity_mS_per_mm),
    )
    heart_count = heart_nodes.shape[0]
    if isinstance(plan.route, HeartTorsoBidomainRoute):
        torso_nodes = np.asarray(plan.route.torso_nodes_mm)
        torso_mass, torso_stiffness = _assemble_simplex_operators(
            torso_nodes,
            np.asarray(plan.route.torso_elements),
            np.asarray(plan.route.torso_conductivity_mS_per_mm),
        )
        torso_count = torso_nodes.shape[0]
        interface = np.zeros((heart_count, torso_count), dtype=heart_nodes.dtype)
        for (heart_node, torso_node), conductance in zip(
            np.asarray(plan.route.interface_node_pairs),
            np.asarray(plan.route.interface_conductance_mS),
        ):
            interface[heart_node, torso_node] += conductance
    else:
        torso_count = 0
        torso_mass = np.zeros((0, 0), dtype=heart_nodes.dtype)
        torso_stiffness = np.zeros((0, 0), dtype=heart_nodes.dtype)
        interface = np.zeros((heart_count, 0), dtype=heart_nodes.dtype)

    potential_size = heart_count + torso_count
    potential = np.zeros((potential_size, potential_size), dtype=heart_nodes.dtype)
    potential[:heart_count, :heart_count] = intracellular + extracellular
    if torso_count:
        heart_interface_diagonal = np.sum(interface, axis=1)
        torso_interface_diagonal = np.sum(interface, axis=0)
        potential[:heart_count, :heart_count] += np.diag(heart_interface_diagonal)
        potential[heart_count:, heart_count:] = torso_stiffness + np.diag(
            torso_interface_diagonal
        )
        potential[:heart_count, heart_count:] = -interface
        potential[heart_count:, :heart_count] = -interface.T
    heart_weights = np.sum(heart_mass, axis=1)
    torso_weights = np.sum(torso_mass, axis=1)
    gauge_weights = np.concatenate((heart_weights, torso_weights))
    gauge_weights = gauge_weights / np.sum(gauge_weights)
    ones = np.ones((potential_size,), dtype=heart_nodes.dtype)
    nullspace_residual = np.linalg.norm(potential @ ones)
    stabilized_potential = potential + np.outer(gauge_weights, gauge_weights)
    if np.linalg.matrix_rank(stabilized_potential) != potential_size:
        raise ValueError(
            "Bidomain extracellular topology is disconnected beyond one gauge mode."
        )

    vm_block = (
        plan.membrane_capacitance_uF_per_mm3 / plan.dt_ms * heart_mass + intracellular
    )
    field_size = 2 * heart_count + torso_count + 1
    system = np.zeros((field_size, field_size), dtype=heart_nodes.dtype)
    vm_slice = slice(0, heart_count)
    extracellular_slice = slice(heart_count, 2 * heart_count)
    torso_slice = slice(2 * heart_count, 2 * heart_count + torso_count)
    gauge_index = field_size - 1
    system[vm_slice, vm_slice] = vm_block
    system[vm_slice, extracellular_slice] = intracellular
    system[extracellular_slice, vm_slice] = intracellular
    system[
        heart_count : 2 * heart_count + torso_count,
        heart_count : 2 * heart_count + torso_count,
    ] = potential
    system[heart_count:gauge_index, gauge_index] = gauge_weights
    system[gauge_index, heart_count:gauge_index] = gauge_weights
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-cardiovascular-bidomain-fem",
            "plan": plan.plan_id,
            "operators": array_tree_fingerprint(
                (
                    heart_mass,
                    intracellular,
                    extracellular,
                    torso_mass,
                    torso_stiffness,
                    interface,
                    gauge_weights,
                    system,
                )
            ),
        }
    )
    return PreparedBidomainFEM(
        plan,
        jnp.asarray(heart_mass),
        jnp.asarray(intracellular),
        jnp.asarray(extracellular),
        jnp.asarray(torso_mass),
        jnp.asarray(torso_stiffness),
        jnp.asarray(interface),
        jnp.asarray(gauge_weights),
        jnp.asarray(system),
        jnp.asarray(vm_block),
        jnp.asarray(stabilized_potential),
        jnp.asarray(nullspace_residual),
        prepared_id,
    )


def initialize_bidomain_state(
    prepared: PreparedBidomainFEM,
    transmembrane_voltage_mV: ArrayLike,
    /,
    *,
    extracellular_potential_mV: ArrayLike | None = None,
    torso_potential_mV: ArrayLike | None = None,
) -> BidomainState:
    """Create a shape-checked state and project extracellular fields to the gauge."""

    if not isinstance(prepared, PreparedBidomainFEM):
        raise TypeError("prepared must be a PreparedBidomainFEM.")
    voltage = jnp.asarray(transmembrane_voltage_mV)
    if voltage.shape != (prepared.heart_node_count,):
        raise ValueError("transmembrane_voltage_mV changed the heart node layout.")
    if extracellular_potential_mV is None:
        extracellular = jnp.zeros_like(voltage)
    else:
        extracellular = jnp.asarray(extracellular_potential_mV, dtype=voltage.dtype)
    if extracellular.shape != voltage.shape:
        raise ValueError("extracellular_potential_mV changed the heart node layout.")
    if torso_potential_mV is None:
        torso = jnp.zeros((prepared.torso_node_count,), dtype=voltage.dtype)
    else:
        torso = jnp.asarray(torso_potential_mV, dtype=voltage.dtype)
    if torso.shape != (prepared.torso_node_count,):
        raise ValueError("torso_potential_mV changed the torso node layout.")
    potentials = jnp.concatenate((extracellular, torso))
    mean = prepared.gauge_weights @ potentials
    potentials = potentials - mean
    return BidomainState(
        voltage,
        potentials[: prepared.heart_node_count],
        potentials[prepared.heart_node_count :],
        jnp.asarray(0.0, dtype=voltage.dtype),
        jnp.asarray(0.0, dtype=voltage.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        prepared.prepared_id,
    )


def zero_bidomain_inputs(
    prepared: PreparedBidomainFEM, /, *, dtype=None
) -> BidomainStepInputs:
    resolved_dtype = prepared.heart_mass_matrix.dtype if dtype is None else dtype
    heart = jnp.zeros((prepared.heart_node_count,), dtype=resolved_dtype)
    torso = jnp.zeros((prepared.torso_node_count,), dtype=resolved_dtype)
    return BidomainStepInputs(heart, heart, heart, torso)


def _native_dense_solve(matrix: Array, right_hand_side: Array, /) -> Array:
    return solve(
        LinearSystem(DenseLinearOperator(matrix)),
        right_hand_side,
        policy=LinearSolvePolicy(DenseLU()),
    ).value


def _validate_state(prepared: PreparedBidomainFEM, state: BidomainState, /) -> None:
    if not isinstance(state, BidomainState) or state.prepared_id != prepared.prepared_id:
        raise ValueError("Bidomain state belongs to another prepared operator.")
    if (
        state.transmembrane_voltage_mV.shape != (prepared.heart_node_count,)
        or state.extracellular_potential_mV.shape != (prepared.heart_node_count,)
        or state.torso_potential_mV.shape != (prepared.torso_node_count,)
    ):
        raise ValueError("Bidomain state changed a fixed field layout.")


def _validate_inputs(
    prepared: PreparedBidomainFEM, inputs: BidomainStepInputs, /
) -> None:
    if not isinstance(inputs, BidomainStepInputs):
        raise TypeError("inputs must be BidomainStepInputs.")
    heart_shape = (prepared.heart_node_count,)
    if (
        inputs.ionic_current_uA_per_mm3.shape != heart_shape
        or inputs.transmembrane_stimulus_uA_per_mm3.shape != heart_shape
        or inputs.extracellular_stimulus_uA_per_mm3.shape != heart_shape
        or inputs.torso_source_uA_per_mm3.shape != (prepared.torso_node_count,)
    ):
        raise ValueError("Bidomain inputs changed a fixed heart or torso layout.")


def assemble_bidomain_right_hand_side(
    prepared: PreparedBidomainFEM,
    state: BidomainState,
    inputs: BidomainStepInputs,
    /,
) -> Array:
    """Assemble the monolithic implicit-Euler DAE right-hand side."""

    _validate_state(prepared, state)
    _validate_inputs(prepared, inputs)
    capacitance_rate = prepared.plan.membrane_capacitance_uF_per_mm3 / prepared.plan.dt_ms
    vm_right = capacitance_rate * (
        prepared.heart_mass_matrix @ state.transmembrane_voltage_mV
    ) + prepared.heart_mass_matrix @ (
        inputs.transmembrane_stimulus_uA_per_mm3 - inputs.ionic_current_uA_per_mm3
    )
    extracellular_right = (
        prepared.heart_mass_matrix @ inputs.extracellular_stimulus_uA_per_mm3
    )
    torso_right = prepared.torso_mass_matrix @ inputs.torso_source_uA_per_mm3
    return jnp.concatenate(
        (
            vm_right,
            extracellular_right,
            torso_right,
            jnp.zeros((1,), dtype=vm_right.dtype),
        )
    )


def apply_bidomain_block_preconditioner(
    prepared: PreparedBidomainFEM, residual: ArrayLike, /
) -> BidomainPreconditionerApplication:
    """Apply the reusable Vm/potential block inverse with rank-one gauge stabilization."""

    if not isinstance(prepared, PreparedBidomainFEM):
        raise TypeError("prepared must be a PreparedBidomainFEM.")
    value = jnp.asarray(residual)
    if value.shape != (prepared.field_size,):
        raise ValueError("residual changed the monolithic field layout.")
    heart_count = prepared.heart_node_count
    gauge_index = prepared.field_size - 1
    vm_value = _native_dense_solve(prepared.vm_preconditioner_block, value[:heart_count])
    potential_value = _native_dense_solve(
        prepared.potential_preconditioner_block,
        value[heart_count:gauge_index],
    )
    applied = jnp.concatenate((vm_value, potential_value, value[gauge_index:]))
    action_defect = prepared.system_matrix @ applied - value
    input_norm = jnp.linalg.norm(value)
    defect_norm = jnp.linalg.norm(action_defect)
    diagonal = jnp.abs(jnp.diag(prepared.system_matrix)[:-1])
    positive_diagonal = jnp.where(diagonal > 0.0, diagonal, jnp.inf)
    minimum = jnp.min(positive_diagonal)
    maximum = jnp.max(diagonal)
    condition = maximum / minimum
    relative = defect_norm / jnp.maximum(input_norm, jnp.finfo(value.dtype).tiny)
    finite = jnp.all(jnp.isfinite(applied)) & jnp.isfinite(relative)
    evidence = BidomainPreconditionerEvidence(
        minimum,
        maximum,
        condition,
        input_norm,
        defect_norm,
        relative,
        finite,
    )
    return BidomainPreconditionerApplication(applied, evidence)


def step_bidomain(
    prepared: PreparedBidomainFEM,
    state: BidomainState,
    inputs: BidomainStepInputs,
    /,
) -> BidomainStepResult:
    """Advance Vm and elliptic potential fields, accepting only qualified evidence."""

    if not isinstance(prepared, PreparedBidomainFEM):
        raise TypeError("prepared must be a PreparedBidomainFEM.")
    _validate_state(prepared, state)
    _validate_inputs(prepared, inputs)
    right = assemble_bidomain_right_hand_side(prepared, state, inputs)
    candidate = _native_dense_solve(prepared.system_matrix, right)
    residual = prepared.system_matrix @ candidate - right
    heart_count = prepared.heart_node_count
    torso_count = prepared.torso_node_count
    gauge_index = prepared.field_size - 1
    vm_residual = jnp.linalg.norm(residual[:heart_count])
    extracellular_residual = jnp.linalg.norm(residual[heart_count : 2 * heart_count])
    torso_residual = jnp.linalg.norm(
        residual[2 * heart_count : 2 * heart_count + torso_count]
    )
    gauge_residual = jnp.abs(residual[gauge_index])
    total_residual = jnp.linalg.norm(residual)
    relative_residual = total_residual / jnp.maximum(
        jnp.linalg.norm(right), jnp.finfo(candidate.dtype).tiny
    )
    potentials = candidate[heart_count:gauge_index]
    weighted_mean = prepared.gauge_weights @ potentials
    fixed_gauge = jnp.abs(weighted_mean) <= prepared.plan.gauge_tolerance_mV
    external_source_compatibility = jnp.abs(jnp.sum(right[heart_count:gauge_index]))
    compatible_source = (
        external_source_compatibility <= prepared.plan.source_compatibility_tolerance_uA
    )

    if isinstance(prepared.plan.route, HeartTorsoBidomainRoute):
        pairs = prepared.plan.route.interface_node_pairs
        conductance = prepared.plan.route.interface_conductance_mS
        heart_potential = candidate[heart_count : 2 * heart_count]
        torso_potential = candidate[2 * heart_count : gauge_index]
        jumps = heart_potential[pairs[:, 0]] - torso_potential[pairs[:, 1]]
        interface_current = conductance * jumps
        jump_norm = jnp.linalg.norm(jumps)
        current_norm = jnp.linalg.norm(interface_current)
        flux_balance = jnp.abs(jnp.sum(interface_current) + jnp.sum(-interface_current))
        interface_supported = jnp.asarray(True)
    else:
        jump_norm = jnp.asarray(0.0, dtype=candidate.dtype)
        current_norm = jnp.asarray(0.0, dtype=candidate.dtype)
        flux_balance = jnp.asarray(0.0, dtype=candidate.dtype)
        interface_supported = jnp.asarray(False)

    preconditioner = apply_bidomain_block_preconditioner(prepared, residual).evidence
    finite = (
        jnp.all(jnp.isfinite(candidate))
        & jnp.all(jnp.isfinite(residual))
        & jnp.isfinite(weighted_mean)
        & preconditioner.finite
    )
    residual_ok = relative_residual <= prepared.plan.residual_tolerance
    status = jnp.asarray(int(BidomainSolveStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        finite,
        status,
        jnp.bitwise_or(status, int(BidomainSolveStatus.NONFINITE)),
    )
    status = jnp.where(
        residual_ok,
        status,
        jnp.bitwise_or(status, int(BidomainSolveStatus.RESIDUAL_FAILURE)),
    )
    status = jnp.where(
        fixed_gauge,
        status,
        jnp.bitwise_or(status, int(BidomainSolveStatus.GAUGE_FAILURE)),
    )
    status = jnp.where(
        compatible_source,
        status,
        jnp.bitwise_or(status, int(BidomainSolveStatus.INCOMPATIBLE_EXTERNAL_SOURCE)),
    )
    status = jnp.where(
        preconditioner.finite,
        status,
        jnp.bitwise_or(status, int(BidomainSolveStatus.PRECONDITIONER_FAILURE)),
    )
    successful = status == int(BidomainSolveStatus.SUCCESS)
    candidate_state = BidomainState(
        candidate[:heart_count],
        candidate[heart_count : 2 * heart_count],
        candidate[2 * heart_count : gauge_index],
        candidate[gauge_index],
        state.time_ms + prepared.plan.dt_ms,
        state.step_index + jnp.asarray(1, dtype=jnp.int32),
        prepared.prepared_id,
    )
    accepted_state = jax.tree.map(
        lambda proposed, prior: jnp.where(successful, proposed, prior),
        candidate_state,
        state,
    )
    block_evidence = BidomainBlockResidualEvidence(
        vm_residual,
        extracellular_residual,
        torso_residual,
        gauge_residual,
        total_residual,
        relative_residual,
    )
    gauge_evidence = BidomainGaugeEvidence(
        weighted_mean,
        jnp.abs(weighted_mean),
        prepared.ungauged_nullspace_residual,
        fixed_gauge,
    )
    interface_evidence = BidomainInterfaceEvidence(
        jump_norm,
        current_norm,
        flux_balance,
        interface_supported,
    )
    evidence = BidomainStepEvidence(
        block_evidence,
        gauge_evidence,
        interface_evidence,
        preconditioner,
        external_source_compatibility,
        finite,
        status,
        successful,
    )
    return BidomainStepResult(accepted_state, candidate_state, evidence)


def step_proportional_monodomain_limit(
    prepared: PreparedBidomainFEM,
    state: BidomainState,
    inputs: BidomainStepInputs,
    extracellular_to_intracellular_ratio: float,
    /,
) -> MonodomainLimitResult:
    """Solve the analytic proportional-conductivity monodomain reduction.

    For ``sigma_e = ratio * sigma_i``, elimination of the gauge-fixed
    extracellular field gives ``sigma_m = ratio/(1+ratio) * sigma_i``.
    This route is intentionally unavailable for a torso-coupled problem.
    """

    if not isinstance(prepared.plan.route, HeartOnlyBidomainRoute):
        raise TypeError("The proportional monodomain limit requires a heart-only route.")
    _validate_state(prepared, state)
    _validate_inputs(prepared, inputs)
    ratio = float(extracellular_to_intracellular_ratio)
    if not isfinite(ratio) or ratio <= 0.0:
        raise ValueError(
            "extracellular_to_intracellular_ratio must be finite and positive."
        )
    extracellular_source_zero = jnp.all(inputs.extracellular_stimulus_uA_per_mm3 == 0.0)
    expected = ratio * prepared.intracellular_stiffness
    tensor_error = jnp.linalg.norm(
        prepared.extracellular_stiffness - expected
    ) / jnp.maximum(jnp.linalg.norm(expected), jnp.finfo(expected.dtype).tiny)
    capacitance_rate = prepared.plan.membrane_capacitance_uF_per_mm3 / prepared.plan.dt_ms
    effective = ratio / (1.0 + ratio) * prepared.intracellular_stiffness
    matrix = capacitance_rate * prepared.heart_mass_matrix + effective
    right = capacitance_rate * (
        prepared.heart_mass_matrix @ state.transmembrane_voltage_mV
    ) + prepared.heart_mass_matrix @ (
        inputs.transmembrane_stimulus_uA_per_mm3 - inputs.ionic_current_uA_per_mm3
    )
    voltage = _native_dense_solve(matrix, right)
    residual = jnp.linalg.norm(matrix @ voltage - right)
    finite = jnp.all(jnp.isfinite(voltage)) & jnp.isfinite(tensor_error)
    successful = (
        finite
        & extracellular_source_zero
        & (tensor_error <= prepared.plan.residual_tolerance)
        & (
            residual
            <= prepared.plan.residual_tolerance * jnp.maximum(jnp.linalg.norm(right), 1.0)
        )
    )
    evidence = MonodomainLimitEvidence(tensor_error, residual, finite, successful)
    return MonodomainLimitResult(voltage, evidence)


__all__ = [
    "BidomainBlockResidualEvidence",
    "BidomainFEMPlan",
    "BidomainGaugeEvidence",
    "BidomainInterfaceEvidence",
    "BidomainPreconditionerApplication",
    "BidomainPreconditionerEvidence",
    "BidomainRoute",
    "BidomainSolveStatus",
    "BidomainState",
    "BidomainStepEvidence",
    "BidomainStepInputs",
    "BidomainStepResult",
    "HeartOnlyBidomainRoute",
    "HeartTorsoBidomainRoute",
    "MonodomainLimitEvidence",
    "MonodomainLimitResult",
    "PreparedBidomainFEM",
    "apply_bidomain_block_preconditioner",
    "assemble_bidomain_right_hand_side",
    "initialize_bidomain_state",
    "prepare_bidomain_fem",
    "step_bidomain",
    "step_proportional_monodomain_limit",
    "zero_bidomain_inputs",
]
