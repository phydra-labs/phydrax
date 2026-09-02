#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from numbers import Integral
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..graph import GraphIR
from ..operators.quantum import (
    ChargeBasis,
    fluxonium_mode_problem,
    FluxoniumParameters,
    harmonic_mode_problem,
    HarmonicModeParameters,
    HilbertRegisterLayout,
    ModeReductionPolicy,
    OscillatorBasis,
    prepare_mode_reduction,
    PreparedModeReduction,
    refresh_mode_reduction,
    transmon_mode_problem,
    TransmonParameters,
)
from ._local_hamiltonian import LocalHamiltonian, LocalHamiltonianTerm
from ._quantum_control import (
    assemble_fixed_grid_local_hamiltonian,
    QuantumControlScheduleResult,
)


CircuitModeKind: TypeAlias = Literal["transmon", "fluxonium", "harmonic"]
CircuitBasis: TypeAlias = ChargeBasis | OscillatorBasis
CircuitParameters: TypeAlias = (
    TransmonParameters | FluxoniumParameters | HarmonicModeParameters
)


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _mode_policy_identity(policy: ModeReductionPolicy, /) -> dict[str, object]:
    return {
        "retained_dimension": policy.retained_dimension,
        "maximum_raw_dimension": policy.maximum_raw_dimension,
        "maximum_bytes": policy.maximum_bytes,
        "hermiticity_tolerance": policy.hermiticity_tolerance,
        "eigen_residual_tolerance": policy.eigen_residual_tolerance,
        "orthogonality_tolerance": policy.orthogonality_tolerance,
        "minimum_boundary_gap": policy.minimum_boundary_gap,
        "precision": policy.precision.policy_id,
        "tracking": {
            "degeneracy_absolute": policy.tracking.degeneracy_absolute,
            "degeneracy_relative": policy.tracking.degeneracy_relative,
            "minimum_overlap": policy.tracking.minimum_overlap,
            "minimum_assignment_margin": policy.tracking.minimum_assignment_margin,
            "orthogonality_tolerance": policy.tracking.orthogonality_tolerance,
            "maximum_dimension": policy.tracking.maximum_dimension,
        },
    }


class CircuitModePlacement(StrictModule):
    """One retained circuit mode placed at one processor graph node."""

    basis: CircuitBasis
    reduction_policy: ModeReductionPolicy
    wire_id: str = eqx.field(static=True)
    kind: CircuitModeKind = eqx.field(static=True)
    parameter_index: int = eqx.field(static=True)
    placement_id: str = eqx.field(static=True)

    def __init__(
        self,
        wire_id: str,
        kind: CircuitModeKind,
        basis: CircuitBasis,
        parameter_index: int,
        reduction_policy: ModeReductionPolicy,
        /,
        *,
        placement_id: str | None = None,
    ):
        wire = str(wire_id)
        if not wire:
            raise ValueError("wire_id must be nonempty.")
        if kind not in ("transmon", "fluxonium", "harmonic"):
            raise ValueError("Unknown circuit mode kind.")
        if kind == "transmon" and not isinstance(basis, ChargeBasis):
            raise TypeError("Transmon placements require ChargeBasis.")
        if kind in ("fluxonium", "harmonic") and not isinstance(basis, OscillatorBasis):
            raise TypeError("Fluxonium and harmonic placements require OscillatorBasis.")
        if isinstance(parameter_index, bool) or not isinstance(parameter_index, Integral):
            raise TypeError("parameter_index must be a non-negative integer.")
        parameter_index_ = int(parameter_index)
        if parameter_index_ < 0:
            raise ValueError("parameter_index must be non-negative.")
        if not isinstance(reduction_policy, ModeReductionPolicy):
            raise TypeError("reduction_policy must be a ModeReductionPolicy.")
        if reduction_policy.retained_dimension > basis.dimension:
            raise ValueError("Mode retention exceeds the selected raw basis dimension.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "circuit-mode-placement",
                    "wire": wire,
                    "mode_kind": kind,
                    "basis": basis.basis_id,
                    "parameter_index": parameter_index_,
                    "reduction": _mode_policy_identity(reduction_policy),
                }
            )
            if placement_id is None
            else str(placement_id)
        )
        if not identifier:
            raise ValueError("placement_id must be nonempty.")
        self.basis = basis
        self.reduction_policy = reduction_policy
        self.wire_id = wire
        self.kind = kind
        self.parameter_index = parameter_index_
        self.placement_id = identifier


class CircuitInteraction(StrictModule):
    """One factored interaction referencing projected mode operators and a strength."""

    target_indices: tuple[int, ...] = eqx.field(static=True)
    operator_names: tuple[str, ...] = eqx.field(static=True)
    strength_index: int = eqx.field(static=True)
    interaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        target_indices: Sequence[int],
        operator_names: Sequence[str],
        strength_index: int,
        /,
        *,
        interaction_id: str | None = None,
    ):
        targets = tuple(int(index) for index in target_indices)
        names = tuple(str(name) for name in operator_names)
        if (
            not targets
            or len(set(targets)) != len(targets)
            or any(index < 0 for index in targets)
        ):
            raise ValueError("target_indices must be unique and non-negative.")
        if len(names) != len(targets) or any(not name for name in names):
            raise ValueError("operator_names must align with target_indices.")
        if isinstance(strength_index, bool) or not isinstance(strength_index, Integral):
            raise TypeError("strength_index must be a non-negative integer.")
        strength = int(strength_index)
        if strength < 0:
            raise ValueError("strength_index must be non-negative.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "circuit-interaction",
                    "targets": list(targets),
                    "operators": list(names),
                    "strength_index": strength,
                }
            )
            if interaction_id is None
            else str(interaction_id)
        )
        if not identifier:
            raise ValueError("interaction_id must be nonempty.")
        self.target_indices = targets
        self.operator_names = names
        self.strength_index = strength
        self.interaction_id = identifier


class CircuitDrivePort(StrictModule):
    """One projected local operator exposed to an external control schedule."""

    mode_index: int = eqx.field(static=True)
    operator_name: str = eqx.field(static=True)
    scale_index: int = eqx.field(static=True)
    port_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_index: int,
        operator_name: str,
        scale_index: int,
        /,
        *,
        port_id: str | None = None,
    ):
        for name, value in (("mode_index", mode_index), ("scale_index", scale_index)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be a non-negative integer.")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative.")
        operator = str(operator_name)
        if not operator:
            raise ValueError("operator_name must be nonempty.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "circuit-drive-port",
                    "mode_index": int(mode_index),
                    "operator": operator,
                    "scale_index": int(scale_index),
                }
            )
            if port_id is None
            else str(port_id)
        )
        if not identifier:
            raise ValueError("port_id must be nonempty.")
        self.mode_index = int(mode_index)
        self.operator_name = operator
        self.scale_index = int(scale_index)
        self.port_id = identifier


class CircuitQEDDeviceSpec(StrictModule):
    """Typed circuit-mode semantics over one unbatched GraphIR topology."""

    topology: GraphIR
    placements: tuple[CircuitModePlacement, ...]
    edge_interactions: tuple[CircuitInteraction, ...]
    additional_interactions: tuple[CircuitInteraction, ...]
    drive_ports: tuple[CircuitDrivePort, ...]
    hbar: Array
    edge_active: tuple[bool, ...] = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: GraphIR,
        placements: Sequence[CircuitModePlacement],
        edge_interactions: Sequence[CircuitInteraction],
        /,
        *,
        additional_interactions: Sequence[CircuitInteraction] = (),
        drive_ports: Sequence[CircuitDrivePort] = (),
        hbar: ArrayLike = 1.0,
        spec_id: str | None = None,
    ):
        if not isinstance(topology, GraphIR):
            raise TypeError("topology must be a GraphIR.")
        topology.validate()
        if topology.num_graphs != 1:
            raise ValueError("CircuitQEDDeviceSpec requires one unbatched graph.")
        placements_ = tuple(placements)
        edges_ = tuple(edge_interactions)
        additional_ = tuple(additional_interactions)
        ports_ = tuple(drive_ports)
        if not placements_ or not all(
            isinstance(placement, CircuitModePlacement) for placement in placements_
        ):
            raise ValueError("placements must contain CircuitModePlacement values.")
        if len(placements_) != topology.num_nodes:
            raise ValueError("placements must align with graph nodes.")
        if len({placement.wire_id for placement in placements_}) != len(placements_):
            raise ValueError("Placement wire IDs must be unique.")
        if topology.node_mask is not None and not bool(jnp.all(topology.node_mask)):
            raise ValueError("Circuit device nodes must all be active.")
        if topology.senders is None or topology.receivers is None:
            if topology.num_edges:
                raise ValueError("Circuit topology edges require explicit endpoints.")
            senders = np.empty((0,), dtype=np.int32)
            receivers = np.empty((0,), dtype=np.int32)
        else:
            senders = np.asarray(topology.senders)
            receivers = np.asarray(topology.receivers)
        if len(edges_) != topology.num_edges or not all(
            isinstance(interaction, CircuitInteraction) for interaction in edges_
        ):
            raise ValueError("edge_interactions must align with graph edges.")
        edge_active = (
            tuple(True for _ in edges_)
            if topology.edge_mask is None
            else tuple(bool(value) for value in np.asarray(topology.edge_mask))
        )
        if len(edge_active) != len(edges_):
            raise ValueError("edge_mask must align with edge_interactions.")
        for index, interaction in enumerate(edges_):
            expected = (int(senders[index]), int(receivers[index]))
            if interaction.target_indices != expected:
                raise ValueError(
                    "Each edge interaction must follow its graph edge order."
                )
        if not all(isinstance(value, CircuitInteraction) for value in additional_):
            raise TypeError(
                "additional_interactions must contain CircuitInteraction values."
            )
        if not all(isinstance(value, CircuitDrivePort) for value in ports_):
            raise TypeError("drive_ports must contain CircuitDrivePort values.")
        for interaction in edges_ + additional_:
            if any(index >= len(placements_) for index in interaction.target_indices):
                raise ValueError("An interaction references an unknown mode index.")
        for port in ports_:
            if port.mode_index >= len(placements_):
                raise ValueError("A drive port references an unknown mode index.")
        hbar_ = jnp.asarray(hbar)
        if hbar_.shape != () or jnp.issubdtype(hbar_.dtype, jnp.complexfloating):
            raise TypeError("hbar must be one real scalar.")
        if not bool(jnp.isfinite(hbar_) & (hbar_ > 0.0)):
            raise ValueError("hbar must be finite and positive.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "circuit-qed-device-spec",
                    "topology": {
                        "n_node": array_tree_fingerprint(topology.n_node),
                        "n_edge": array_tree_fingerprint(topology.n_edge),
                        "senders": array_tree_fingerprint(jnp.asarray(senders)),
                        "receivers": array_tree_fingerprint(jnp.asarray(receivers)),
                        "edge_active": list(edge_active),
                    },
                    "placements": [placement.placement_id for placement in placements_],
                    "edge_interactions": [value.interaction_id for value in edges_],
                    "additional_interactions": [
                        value.interaction_id for value in additional_
                    ],
                    "drive_ports": [port.port_id for port in ports_],
                    "hbar_dtype": str(hbar_.dtype),
                }
            )
            if spec_id is None
            else str(spec_id)
        )
        if not identifier:
            raise ValueError("spec_id must be nonempty.")
        self.topology = topology
        self.placements = placements_
        self.edge_interactions = edges_
        self.additional_interactions = additional_
        self.drive_ports = ports_
        self.hbar = hbar_
        self.edge_active = edge_active
        self.spec_id = identifier


class CircuitQEDDeviceParameters(StrictModule):
    """Unique differentiable mode blocks, interaction strengths, and port scales."""

    mode_parameters: tuple[CircuitParameters, ...]
    interaction_strengths: Array
    drive_scales: Array

    def __init__(
        self,
        mode_parameters: Sequence[CircuitParameters],
        /,
        *,
        interaction_strengths: ArrayLike = (),
        drive_scales: ArrayLike = (),
    ):
        modes = tuple(mode_parameters)
        if not modes or not all(
            isinstance(
                value,
                (TransmonParameters, FluxoniumParameters, HarmonicModeParameters),
            )
            for value in modes
        ):
            raise ValueError(
                "mode_parameters must contain typed circuit mode parameters."
            )
        strengths = jnp.asarray(interaction_strengths)
        scales = jnp.asarray(drive_scales)
        if strengths.ndim != 1 or scales.ndim != 1:
            raise ValueError("interaction_strengths and drive_scales must be vectors.")
        if jnp.issubdtype(strengths.dtype, jnp.complexfloating) or jnp.issubdtype(
            scales.dtype, jnp.complexfloating
        ):
            raise TypeError("Interaction strengths and drive scales must be real.")
        if not jnp.issubdtype(strengths.dtype, jnp.inexact):
            strengths = strengths.astype(float)
        if not jnp.issubdtype(scales.dtype, jnp.inexact):
            scales = scales.astype(float)
        self.mode_parameters = modes
        self.interaction_strengths = strengths
        self.drive_scales = scales


class CircuitQEDDevicePolicy(StrictModule):
    """Global reduced-device resource policy."""

    maximum_hilbert_dimension: int = eqx.field(static=True)
    maximum_dense_entries: int = eqx.field(static=True)
    maximum_prepared_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_hilbert_dimension: int = 1 << 28,
        maximum_dense_entries: int = 1 << 26,
        maximum_prepared_bytes: int = 1 << 30,
    ):
        self.maximum_hilbert_dimension = _positive_integer(
            maximum_hilbert_dimension, "maximum_hilbert_dimension"
        )
        self.maximum_dense_entries = _positive_integer(
            maximum_dense_entries, "maximum_dense_entries"
        )
        self.maximum_prepared_bytes = _positive_integer(
            maximum_prepared_bytes, "maximum_prepared_bytes"
        )


class CircuitQEDDeviceCostEstimate(StrictModule):
    """Logical local, state, dense-Hamiltonian, and propagator storage."""

    hilbert_dimension: int = eqx.field(static=True)
    raw_mode_elements: int = eqx.field(static=True)
    projected_operator_elements: int = eqx.field(static=True)
    state_bytes: int = eqx.field(static=True)
    dense_entries: int = eqx.field(static=True)
    dense_bytes: int = eqx.field(static=True)
    full_propagator_bytes: int = eqx.field(static=True)
    dense_admissible: bool = eqx.field(static=True)


class CircuitQEDDevicePlan(StrictModule):
    """Static topology, layout, and cost plan for a reduced circuit-QED device."""

    spec: CircuitQEDDeviceSpec
    policy: CircuitQEDDevicePolicy
    layout: HilbertRegisterLayout
    cost: CircuitQEDDeviceCostEstimate
    required_mode_parameters: int = eqx.field(static=True)
    required_interaction_strengths: int = eqx.field(static=True)
    required_drive_scales: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class CircuitQEDDeviceDiagnostics(StrictModule):
    """Aggregated mode, parameter, topology, and resource evidence."""

    mode_valid: Array
    parameter_finite: Array
    topology_valid: Array
    resource_valid: Array
    finite: Array
    valid: Array


class PreparedCircuitQEDDevice(StrictModule):
    """Reduced modes and assembled local Hamiltonian terms for one device."""

    plan: CircuitQEDDevicePlan
    parameters: CircuitQEDDeviceParameters
    reductions: tuple[PreparedModeReduction, ...]
    drift: LocalHamiltonian
    drive_terms: tuple[LocalHamiltonianTerm, ...]
    diagnostics: CircuitQEDDeviceDiagnostics
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


def plan_circuit_qed_device(
    spec: CircuitQEDDeviceSpec,
    policy: CircuitQEDDevicePolicy | None = None,
    /,
) -> CircuitQEDDevicePlan:
    """Validate static device semantics and estimate every execution representation."""

    if not isinstance(spec, CircuitQEDDeviceSpec):
        raise TypeError("spec must be a CircuitQEDDeviceSpec.")
    selected = CircuitQEDDevicePolicy() if policy is None else policy
    if not isinstance(selected, CircuitQEDDevicePolicy):
        raise TypeError("policy must be a CircuitQEDDevicePolicy or None.")
    dimensions = tuple(
        placement.reduction_policy.retained_dimension for placement in spec.placements
    )
    layout = HilbertRegisterLayout(
        tuple(placement.wire_id for placement in spec.placements),
        dimensions,
    )
    hilbert = prod(dimensions)
    raw_elements = sum(placement.basis.dimension**2 for placement in spec.placements)
    projected_elements = sum(
        placement.reduction_policy.retained_dimension**2
        * (1 + len(_operator_names(placement.kind)))
        for placement in spec.placements
    )
    itemsize = np.dtype(jnp.complex128).itemsize
    dense_entries = hilbert * hilbert
    state_bytes = hilbert * itemsize
    dense_bytes = dense_entries * itemsize
    full_propagator_bytes = dense_bytes
    prepared_bytes = itemsize * (raw_elements + projected_elements)
    if hilbert > selected.maximum_hilbert_dimension:
        raise ValueError("Device Hilbert dimension exceeds maximum_hilbert_dimension.")
    if prepared_bytes > selected.maximum_prepared_bytes:
        raise ValueError("Prepared mode storage exceeds maximum_prepared_bytes.")
    dense_admissible = dense_entries <= selected.maximum_dense_entries
    cost = CircuitQEDDeviceCostEstimate(
        hilbert,
        raw_elements,
        projected_elements,
        state_bytes,
        dense_entries,
        dense_bytes,
        full_propagator_bytes,
        dense_admissible,
    )
    interactions = spec.edge_interactions + spec.additional_interactions
    required_modes = max(placement.parameter_index for placement in spec.placements) + 1
    required_strengths = (
        max(interaction.strength_index for interaction in interactions) + 1
        if interactions
        else 0
    )
    required_scales = (
        max(port.scale_index for port in spec.drive_ports) + 1 if spec.drive_ports else 0
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "circuit-qed-device-plan",
            "spec": spec.spec_id,
            "layout": layout.layout_id,
            "limits": {
                "maximum_hilbert_dimension": selected.maximum_hilbert_dimension,
                "maximum_dense_entries": selected.maximum_dense_entries,
                "maximum_prepared_bytes": selected.maximum_prepared_bytes,
            },
        }
    )
    return CircuitQEDDevicePlan(
        spec,
        selected,
        layout,
        cost,
        required_modes,
        required_strengths,
        required_scales,
        plan_id,
    )


def _operator_names(kind: CircuitModeKind, /) -> tuple[str, ...]:
    if kind == "transmon":
        return ("charge", "cos_phase", "sin_phase", "phase_raising")
    if kind == "fluxonium":
        return ("charge", "phase", "cos_phase", "sin_phase", "lowering")
    return ("charge", "phase", "lowering", "number")


def _mode_problem(
    placement: CircuitModePlacement,
    parameters: CircuitParameters,
    hbar: Array,
    /,
):
    problem_id = canonical_fingerprint(
        {
            "kind": "placed-circuit-mode-problem",
            "mode_kind": placement.kind,
            "basis": placement.basis.basis_id,
            "parameter_index": placement.parameter_index,
            "reduction": _mode_policy_identity(placement.reduction_policy),
        }
    )
    if placement.kind == "transmon":
        if not isinstance(parameters, TransmonParameters) or not isinstance(
            placement.basis, ChargeBasis
        ):
            raise TypeError("Transmon placement and parameter types do not match.")
        return transmon_mode_problem(
            parameters,
            placement.basis,
            hbar=hbar,
            problem_id=problem_id,
        )
    if placement.kind == "fluxonium":
        if not isinstance(parameters, FluxoniumParameters) or not isinstance(
            placement.basis, OscillatorBasis
        ):
            raise TypeError("Fluxonium placement and parameter types do not match.")
        return fluxonium_mode_problem(
            parameters,
            placement.basis,
            hbar=hbar,
            problem_id=problem_id,
        )
    if not isinstance(parameters, HarmonicModeParameters) or not isinstance(
        placement.basis, OscillatorBasis
    ):
        raise TypeError("Harmonic placement and parameter types do not match.")
    return harmonic_mode_problem(
        parameters,
        placement.basis,
        hbar=hbar,
        problem_id=problem_id,
    )


def _validate_parameters(
    plan: CircuitQEDDevicePlan,
    parameters: CircuitQEDDeviceParameters,
    /,
) -> None:
    if not isinstance(parameters, CircuitQEDDeviceParameters):
        raise TypeError("parameters must be CircuitQEDDeviceParameters.")
    if len(parameters.mode_parameters) != plan.required_mode_parameters:
        raise ValueError("mode_parameters count does not match the device plan.")
    if parameters.interaction_strengths.shape != (plan.required_interaction_strengths,):
        raise ValueError("interaction_strengths count does not match the device plan.")
    if parameters.drive_scales.shape != (plan.required_drive_scales,):
        raise ValueError("drive_scales count does not match the device plan.")


def _prepare_device(
    plan: CircuitQEDDevicePlan,
    parameters: CircuitQEDDeviceParameters,
    /,
    *,
    previous: PreparedCircuitQEDDevice | None,
    numeric_version: ArrayLike,
) -> PreparedCircuitQEDDevice:
    _validate_parameters(plan, parameters)
    reductions: list[PreparedModeReduction] = []
    cache: dict[tuple[str, str, str, int], PreparedModeReduction] = {}
    for index, placement in enumerate(plan.spec.placements):
        key = (
            placement.kind,
            placement.basis.basis_id,
            canonical_fingerprint(
                {
                    "kind": "mode-reduction-policy",
                    "policy": _mode_policy_identity(placement.reduction_policy),
                }
            ),
            placement.parameter_index,
        )
        if key in cache:
            reduced = cache[key]
        else:
            problem = _mode_problem(
                placement,
                parameters.mode_parameters[placement.parameter_index],
                plan.spec.hbar,
            )
            reduced = (
                prepare_mode_reduction(problem, policy=placement.reduction_policy)
                if previous is None
                else refresh_mode_reduction(previous.reductions[index], problem)
            )
            cache[key] = reduced
        reductions.append(reduced)

    drift_terms: list[LocalHamiltonianTerm] = []
    for placement, reduced in zip(plan.spec.placements, reductions, strict=True):
        relative_energies = reduced.energies - reduced.energies[0]
        diagonal = jnp.diag(relative_energies.astype(jnp.complex128))
        drift_terms.append(
            LocalHamiltonianTerm.from_product(
                (diagonal,),
                (placement.wire_id,),
                term_id=canonical_fingerprint(
                    {"kind": "circuit-mode-energy", "placement": placement.placement_id}
                ),
            )
        )

    all_interactions = plan.spec.edge_interactions + plan.spec.additional_interactions
    edge_count = len(plan.spec.edge_interactions)
    for interaction_index, interaction in enumerate(all_interactions):
        if (
            interaction_index < edge_count
            and not plan.spec.edge_active[interaction_index]
        ):
            continue
        factors = tuple(
            reductions[target].operator(name).matrix
            for target, name in zip(
                interaction.target_indices,
                interaction.operator_names,
                strict=True,
            )
        )
        strength = parameters.interaction_strengths[interaction.strength_index]
        factors = (strength * factors[0],) + factors[1:]
        drift_terms.append(
            LocalHamiltonianTerm.from_product(
                factors,
                tuple(
                    plan.spec.placements[index].wire_id
                    for index in interaction.target_indices
                ),
                term_id=interaction.interaction_id,
            )
        )

    drive_terms = tuple(
        LocalHamiltonianTerm.from_product(
            (
                parameters.drive_scales[port.scale_index]
                * reductions[port.mode_index].operator(port.operator_name).matrix,
            ),
            (plan.spec.placements[port.mode_index].wire_id,),
            term_id=port.port_id,
        )
        for port in plan.spec.drive_ports
    )
    drift = LocalHamiltonian(
        plan.layout,
        tuple(drift_terms),
        hamiltonian_id=canonical_fingerprint(
            {"kind": "circuit-qed-drift", "plan": plan.plan_id}
        ),
    )
    mode_valid = jnp.all(
        jnp.stack(tuple(reduction.diagnostics.valid for reduction in reductions))
    )
    parameter_leaves = jax.tree_util.tree_leaves(parameters.mode_parameters)
    parameter_finite = (
        jnp.all(jnp.isfinite(parameters.interaction_strengths))
        & jnp.all(jnp.isfinite(parameters.drive_scales))
        & jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(leaf)) for leaf in parameter_leaves))
        )
    )
    topology_valid = jnp.asarray(True)
    resource_valid = jnp.asarray(
        plan.cost.hilbert_dimension <= plan.policy.maximum_hilbert_dimension
    )
    finite = (
        parameter_finite
        & drift.finite
        & jnp.all(
            jnp.stack(tuple(term.finite for term in drive_terms) or (jnp.asarray(True),))
        )
    )
    valid = (
        mode_valid
        & topology_valid
        & resource_valid
        & finite
        & drift.valid
        & jnp.all(
            jnp.stack(tuple(term.valid for term in drive_terms) or (jnp.asarray(True),))
        )
    )
    diagnostics = CircuitQEDDeviceDiagnostics(
        mode_valid,
        parameter_finite,
        topology_valid,
        resource_valid,
        finite,
        valid,
    )
    version = jnp.asarray(numeric_version, dtype=jnp.int32)
    return PreparedCircuitQEDDevice(
        plan,
        parameters,
        tuple(reductions),
        drift,
        drive_terms,
        diagnostics,
        version,
        canonical_fingerprint(
            {"kind": "prepared-circuit-qed-device", "plan": plan.plan_id}
        ),
    )


def prepare_circuit_qed_device(
    spec: CircuitQEDDeviceSpec,
    parameters: CircuitQEDDeviceParameters,
    plan: CircuitQEDDevicePlan | None = None,
    /,
    *,
    policy: CircuitQEDDevicePolicy | None = None,
) -> PreparedCircuitQEDDevice:
    """Prepare local reductions and Hamiltonian terms for one device."""

    selected = plan_circuit_qed_device(spec, policy) if plan is None else plan
    if plan is not None:
        if policy is not None:
            raise ValueError("Specify plan or policy, not both.")
        if (
            not isinstance(plan, CircuitQEDDevicePlan)
            or plan.spec.spec_id != spec.spec_id
        ):
            raise ValueError("CircuitQEDDeviceSpec does not match the supplied plan.")
    return _prepare_device(
        selected,
        parameters,
        previous=None,
        numeric_version=jnp.asarray(0, dtype=jnp.int32),
    )


def refresh_circuit_qed_device(
    prepared: PreparedCircuitQEDDevice,
    parameters: CircuitQEDDeviceParameters,
    /,
) -> PreparedCircuitQEDDevice:
    """Refresh all numerical device leaves while preserving topology and labels."""

    if not isinstance(prepared, PreparedCircuitQEDDevice):
        raise TypeError("prepared must be a PreparedCircuitQEDDevice.")
    return _prepare_device(
        prepared.plan,
        parameters,
        previous=prepared,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
    )


def assemble_circuit_qed_hamiltonian(
    prepared: PreparedCircuitQEDDevice,
    controls: QuantumControlScheduleResult,
    /,
):
    """Bind sampled control coefficients to prepared circuit-QED drive ports."""

    if not isinstance(prepared, PreparedCircuitQEDDevice):
        raise TypeError("prepared must be a PreparedCircuitQEDDevice.")
    return assemble_fixed_grid_local_hamiltonian(
        prepared.drift,
        prepared.drive_terms,
        controls,
        hbar=prepared.plan.spec.hbar,
    )


__all__ = [
    "CircuitDrivePort",
    "CircuitInteraction",
    "CircuitModeKind",
    "CircuitModePlacement",
    "CircuitQEDDeviceCostEstimate",
    "CircuitQEDDeviceDiagnostics",
    "CircuitQEDDeviceParameters",
    "CircuitQEDDevicePlan",
    "CircuitQEDDevicePolicy",
    "CircuitQEDDeviceSpec",
    "PreparedCircuitQEDDevice",
    "assemble_circuit_qed_hamiltonian",
    "plan_circuit_qed_device",
    "prepare_circuit_qed_device",
    "refresh_circuit_qed_device",
]
