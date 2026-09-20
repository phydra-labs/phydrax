#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..ein import contract
from ._graph import AtomisticGraph
from ._system import PreparedAtomisticSystem


class ClassicalSpinState(StrictModule):
    """Classical unit-vector moments in the prepared atomistic site order."""

    directions: Array
    active_mask: Array
    hamiltonian_id: str = eqx.field(static=True)

    def __init__(
        self,
        directions: ArrayLike,
        active_mask: ArrayLike,
        hamiltonian_id: str,
        /,
        *,
        tolerance: float = 1.0e-8,
    ):
        values = jnp.asarray(directions)
        active = jnp.asarray(active_mask, dtype=jnp.bool_)
        identifier = str(hamiltonian_id)
        if values.ndim != 2 or values.shape[-1] != 3:
            raise ValueError("Classical spin directions must have shape (site, 3).")
        if active.shape != values.shape[:-1]:
            raise ValueError("Classical spin active_mask must match the site axis.")
        if not identifier:
            raise ValueError("hamiltonian_id must be non-empty.")
        residual = jnp.abs(jnp.sqrt(jnp.sum(values * values, axis=-1)) - 1.0)
        invalid = jnp.any(~jnp.isfinite(values)) | jnp.any(
            active & (residual > tolerance)
        )
        values = eqx.error_if(
            values,
            invalid,
            "Active classical spin directions must be finite unit vectors.",
        )
        self.directions = jnp.where(active[:, None], values, 0.0)
        self.active_mask = active
        self.hamiltonian_id = identifier


class ClassicalSpinHamiltonianPlan(StrictModule, NonTrainableState):
    """Once-per-bond classical spin support over one realized atomistic graph.

    The reduced Hamiltonian is

    ``-m_iᵀ (J I + Gamma) m_j - D·(m_i × m_j)`` per canonical bond,
    followed by ``-K (m·n)^2 - mu B·m`` per active site. Canonical bond
    orientation follows increasing stable particle ID, so changing a directed
    graph's storage order cannot change the DMI sign convention.
    """

    system: PreparedAtomisticSystem
    graph: AtomisticGraph
    bond_senders: Array
    bond_receivers: Array
    forward_edge_indices: Array
    reverse_edge_indices: Array
    site_mask: Array
    energy_unit: str = eqx.field(static=True)
    field_unit: str = eqx.field(static=True)
    moment_unit: str = eqx.field(static=True)
    validation_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: PreparedAtomisticSystem,
        graph: AtomisticGraph,
        /,
        *,
        energy_unit: str,
        field_unit: str,
        moment_unit: str,
        validation_tolerance: float = 1.0e-10,
    ):
        if not isinstance(system, PreparedAtomisticSystem):
            raise TypeError("system must be PreparedAtomisticSystem.")
        if not isinstance(graph, AtomisticGraph):
            raise TypeError("graph must be AtomisticGraph.")
        tolerance = float(validation_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("validation_tolerance must be finite and positive.")
        units = tuple(
            str(value).strip() for value in (energy_unit, field_unit, moment_unit)
        )
        if any(not value for value in units):
            raise ValueError("Spin energy, field, and moment units must be explicit.")
        if bool(np.any(np.asarray(graph.overflow))):
            raise ValueError("Classical spin preparation refuses graph overflow.")
        senders = np.asarray(graph.graph.senders, dtype=np.int64)
        receivers = np.asarray(graph.graph.receivers, dtype=np.int64)
        edge_mask = np.asarray(graph.graph.edge_mask, dtype=np.bool_)
        if senders.shape != receivers.shape or edge_mask.shape != senders.shape:
            raise ValueError("Atomistic graph edge storage is inconsistent.")
        if senders.ndim != 1:
            raise ValueError("Classical spin graph edge storage must be rank one.")
        if np.any(senders[edge_mask] == receivers[edge_mask]):
            raise ValueError(
                "Classical spin self-bonds require a distinct explicit model."
            )
        if np.any(senders[edge_mask] < 0) or np.any(receivers[edge_mask] < 0):
            raise ValueError("Atomistic graph contains negative active endpoints.")
        if np.any(senders[edge_mask] >= system.capacity) or np.any(
            receivers[edge_mask] >= system.capacity
        ):
            raise ValueError(
                "Atomistic graph endpoint exceeds the prepared system capacity."
            )
        graph_nodes = np.asarray(graph.graph.node_mask, dtype=np.bool_)
        site_mask = np.asarray(system.active_mask, dtype=np.bool_)
        if graph_nodes.shape != site_mask.shape or not np.array_equal(
            graph_nodes, site_mask
        ):
            raise ValueError(
                "Atomistic graph and spin system use different active sites."
            )

        particle_ids = np.asarray(system.plan.particle_ids, dtype=np.int64)
        displacement = np.asarray(graph.graph.edges["displacement"])
        active_indices = np.flatnonzero(edge_mask)
        by_oriented_pair: dict[tuple[int, int], list[int]] = {}
        for edge in active_indices:
            key = (int(senders[edge]), int(receivers[edge]))
            by_oriented_pair.setdefault(key, []).append(int(edge))
        if any(len(indices) != 1 for indices in by_oriented_pair.values()):
            raise ValueError(
                "Classical spin graph has duplicate directed bonds or ambiguous images."
            )
        forward: list[int] = []
        reverse: list[int] = []
        canonical_senders: list[int] = []
        canonical_receivers: list[int] = []
        for edge in active_indices:
            sender = int(senders[edge])
            receiver = int(receivers[edge])
            if particle_ids[sender] >= particle_ids[receiver]:
                continue
            reverse_key = (receiver, sender)
            if reverse_key not in by_oriented_pair:
                raise ValueError(
                    "Every classical spin bond requires one reverse graph edge."
                )
            reverse_edge = by_oriented_pair[reverse_key][0]
            scale = max(
                float(np.linalg.norm(displacement[edge])),
                float(np.linalg.norm(displacement[reverse_edge])),
                1.0,
            )
            if not np.allclose(
                displacement[edge],
                -displacement[reverse_edge],
                rtol=0.0,
                atol=tolerance * scale,
            ):
                raise ValueError(
                    "Reverse classical spin bonds have inconsistent geometry."
                )
            canonical_senders.append(sender)
            canonical_receivers.append(receiver)
            forward.append(int(edge))
            reverse.append(reverse_edge)
        if 2 * len(forward) != active_indices.size:
            raise ValueError(
                "Directed graph edges do not form unique reverse bond pairs."
            )

        self.system = system
        self.graph = graph
        self.bond_senders = jnp.asarray(canonical_senders, dtype=jnp.int32)
        self.bond_receivers = jnp.asarray(canonical_receivers, dtype=jnp.int32)
        self.forward_edge_indices = jnp.asarray(forward, dtype=jnp.int32)
        self.reverse_edge_indices = jnp.asarray(reverse, dtype=jnp.int32)
        self.site_mask = jnp.asarray(site_mask)
        self.energy_unit, self.field_unit, self.moment_unit = units
        self.validation_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "classical-spin-hamiltonian-plan",
                "system": system.prepared_id,
                "graph": graph.graph_id,
                "bonds": list(zip(canonical_senders, canonical_receivers, strict=True)),
                "energy_unit": units[0],
                "field_unit": units[1],
                "moment_unit": units[2],
                "validation_tolerance": tolerance,
            }
        )

    @property
    def site_count(self) -> int:
        return self.site_mask.size

    @property
    def bond_count(self) -> int:
        return self.bond_senders.size


class PreparedClassicalSpinHamiltonian(StrictModule):
    """Validated numeric spin coefficients on immutable site/bond support."""

    plan: ClassicalSpinHamiltonianPlan
    exchange: Array
    symmetric_exchange: Array
    dmi: Array
    anisotropy: Array
    anisotropy_axes: Array
    moments: Array
    magnetic_field: Array
    numeric_revision: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class ClassicalSpinEvaluation(StrictModule):
    """Energy ledger and exact effective field in the declared field unit."""

    total_energy: Array
    exchange_energy: Array
    symmetric_exchange_energy: Array
    dmi_energy: Array
    anisotropy_energy: Array
    zeeman_energy: Array
    effective_field: Array
    tangent_torque: Array
    magnetization: Array
    unit_norm_residual: Array
    ledger_residual: Array
    successful: Array
    hamiltonian_id: str = eqx.field(static=True)


def _finite_host(value: np.ndarray, name: str) -> None:
    if np.any(~np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values.")


def _numeric_coefficients(
    plan: ClassicalSpinHamiltonianPlan,
    *,
    exchange: ArrayLike | None,
    symmetric_exchange: ArrayLike | None,
    dmi: ArrayLike | None,
    anisotropy: ArrayLike | None,
    anisotropy_axes: ArrayLike | None,
    moments: ArrayLike,
    magnetic_field: ArrayLike | None,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    bonds = plan.bond_count
    sites = plan.site_count
    exchange_host = (
        np.zeros((bonds,), dtype=np.float64) if exchange is None else np.asarray(exchange)
    )
    gamma_host = (
        np.zeros((bonds, 3, 3), dtype=exchange_host.dtype)
        if symmetric_exchange is None
        else np.asarray(symmetric_exchange)
    )
    dmi_host = (
        np.zeros((bonds, 3), dtype=exchange_host.dtype)
        if dmi is None
        else np.asarray(dmi)
    )
    anisotropy_host = (
        np.zeros((sites,), dtype=exchange_host.dtype)
        if anisotropy is None
        else np.asarray(anisotropy)
    )
    axes_host = (
        np.tile(np.asarray([0.0, 0.0, 1.0]), (sites, 1))
        if anisotropy_axes is None
        else np.asarray(anisotropy_axes)
    )
    moments_host = np.asarray(moments)
    field_host = (
        np.zeros((sites, 3), dtype=exchange_host.dtype)
        if magnetic_field is None
        else np.asarray(magnetic_field)
    )
    expected = {
        "exchange": ((bonds,), exchange_host),
        "symmetric_exchange": ((bonds, 3, 3), gamma_host),
        "dmi": ((bonds, 3), dmi_host),
        "anisotropy": ((sites,), anisotropy_host),
        "anisotropy_axes": ((sites, 3), axes_host),
        "moments": ((sites,), moments_host),
        "magnetic_field": ((sites, 3), field_host),
    }
    for name, (shape, value) in expected.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}; got {value.shape}.")
        _finite_host(value, name)
    tolerance = plan.validation_tolerance
    transpose_defect = np.max(
        np.abs(gamma_host - np.swapaxes(gamma_host, -1, -2)), initial=0.0
    )
    trace_defect = np.max(np.abs(np.trace(gamma_host, axis1=-2, axis2=-1)), initial=0.0)
    gamma_scale = max(float(np.max(np.abs(gamma_host), initial=0.0)), 1.0)
    if max(float(transpose_defect), float(trace_defect)) > tolerance * gamma_scale:
        raise ValueError("Gamma exchange tensors must be symmetric and traceless.")
    active = np.asarray(plan.site_mask, dtype=np.bool_)
    if np.any(moments_host[active] <= 0.0):
        raise ValueError("Active classical magnetic moments must be positive.")
    relevant_axes = active & (np.abs(anisotropy_host) > 0.0)
    axis_norm = np.linalg.norm(axes_host, axis=-1)
    if np.any(relevant_axes & (np.abs(axis_norm - 1.0) > tolerance)):
        raise ValueError("Active anisotropy axes must be unit vectors when K is nonzero.")
    if np.any(~active & (np.abs(anisotropy_host) > 0.0)) or np.any(
        ~active & (np.linalg.norm(field_host, axis=-1) > 0.0)
    ):
        raise ValueError("Inactive spin sites cannot carry onsite spin coefficients.")
    safe_moments = np.where(active, moments_host, 1.0)
    safe_axes = np.where(active[:, None], axes_host, 0.0)
    return tuple(
        jnp.asarray(value)
        for value in (
            exchange_host,
            gamma_host,
            dmi_host,
            anisotropy_host,
            safe_axes,
            safe_moments,
            field_host,
        )
    )


def prepare_classical_spin_hamiltonian(
    plan: ClassicalSpinHamiltonianPlan,
    /,
    *,
    exchange: ArrayLike | None = None,
    symmetric_exchange: ArrayLike | None = None,
    dmi: ArrayLike | None = None,
    anisotropy: ArrayLike | None = None,
    anisotropy_axes: ArrayLike | None = None,
    moments: ArrayLike,
    magnetic_field: ArrayLike | None = None,
    numeric_revision: str = "0",
) -> PreparedClassicalSpinHamiltonian:
    if not isinstance(plan, ClassicalSpinHamiltonianPlan):
        raise TypeError("plan must be ClassicalSpinHamiltonianPlan.")
    revision = str(numeric_revision)
    if not revision:
        raise ValueError("numeric_revision must be non-empty.")
    values = _numeric_coefficients(
        plan,
        exchange=exchange,
        symmetric_exchange=symmetric_exchange,
        dmi=dmi,
        anisotropy=anisotropy,
        anisotropy_axes=anisotropy_axes,
        moments=moments,
        magnetic_field=magnetic_field,
    )
    arrays = {
        name: np.asarray(value)
        for name, value in zip(
            (
                "exchange",
                "symmetric_exchange",
                "dmi",
                "anisotropy",
                "anisotropy_axes",
                "moments",
                "magnetic_field",
            ),
            values,
            strict=True,
        )
    }
    identifier = canonical_fingerprint(
        {
            "kind": "prepared-classical-spin-hamiltonian",
            "plan": plan.plan_id,
            "numeric_revision": revision,
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    return PreparedClassicalSpinHamiltonian(plan, *values, revision, identifier)


def refresh_classical_spin_hamiltonian(
    prepared: PreparedClassicalSpinHamiltonian,
    /,
    **updates: ArrayLike,
) -> PreparedClassicalSpinHamiltonian:
    """Refresh coefficients without permitting a support or unit change."""

    if not isinstance(prepared, PreparedClassicalSpinHamiltonian):
        raise TypeError("prepared must be PreparedClassicalSpinHamiltonian.")
    allowed = {
        "exchange",
        "symmetric_exchange",
        "dmi",
        "anisotropy",
        "anisotropy_axes",
        "moments",
        "magnetic_field",
        "numeric_revision",
    }
    unknown = set(updates) - allowed
    if unknown:
        raise ValueError(f"Unknown classical spin refresh fields: {sorted(unknown)}.")
    values = {
        "exchange": prepared.exchange,
        "symmetric_exchange": prepared.symmetric_exchange,
        "dmi": prepared.dmi,
        "anisotropy": prepared.anisotropy,
        "anisotropy_axes": prepared.anisotropy_axes,
        "moments": prepared.moments,
        "magnetic_field": prepared.magnetic_field,
        "numeric_revision": prepared.numeric_revision,
    }
    values.update(updates)
    return prepare_classical_spin_hamiltonian(prepared.plan, **values)


def evaluate_classical_spin_hamiltonian(
    prepared: PreparedClassicalSpinHamiltonian,
    state: ClassicalSpinState | ArrayLike,
    /,
) -> ClassicalSpinEvaluation:
    if not isinstance(prepared, PreparedClassicalSpinHamiltonian):
        raise TypeError("prepared must be PreparedClassicalSpinHamiltonian.")
    if isinstance(state, ClassicalSpinState):
        if state.hamiltonian_id != prepared.prepared_id:
            raise ValueError("Classical spin state belongs to another Hamiltonian.")
        directions = state.directions
    else:
        state = ClassicalSpinState(
            state,
            prepared.plan.site_mask,
            prepared.prepared_id,
            tolerance=prepared.plan.validation_tolerance * 10.0,
        )
        directions = state.directions
    senders = prepared.plan.bond_senders
    receivers = prepared.plan.bond_receivers
    left = directions[senders]
    right = directions[receivers]
    exchange_energy = -jnp.sum(prepared.exchange * jnp.sum(left * right, axis=-1))
    gamma_right = contract(
        "bij,bj->bi", prepared.symmetric_exchange, right, backend="jax"
    )
    symmetric_energy = -jnp.sum(jnp.sum(left * gamma_right, axis=-1))
    dmi_energy = -jnp.sum(prepared.dmi * jnp.cross(left, right))
    projections = jnp.sum(directions * prepared.anisotropy_axes, axis=-1)
    anisotropy_energy = -jnp.sum(prepared.anisotropy * projections * projections)
    zeeman_energy = -jnp.sum(
        prepared.moments * jnp.sum(prepared.magnetic_field * directions, axis=-1)
    )
    total = (
        exchange_energy
        + symmetric_energy
        + dmi_energy
        + anisotropy_energy
        + zeeman_energy
    )

    exchange_matrix_right = prepared.exchange[:, None] * right + gamma_right
    gamma_left = contract("bji,bj->bi", prepared.symmetric_exchange, left, backend="jax")
    exchange_matrix_left = prepared.exchange[:, None] * left + gamma_left
    left_field = exchange_matrix_right - jnp.cross(prepared.dmi, right)
    right_field = exchange_matrix_left + jnp.cross(prepared.dmi, left)
    field_numerator = jnp.zeros_like(directions)
    field_numerator = field_numerator.at[senders].add(left_field)
    field_numerator = field_numerator.at[receivers].add(right_field)
    field_numerator = (
        field_numerator
        + (2.0 * prepared.anisotropy * projections)[:, None] * prepared.anisotropy_axes
    )
    effective = field_numerator / prepared.moments[:, None] + prepared.magnetic_field
    effective = jnp.where(prepared.plan.site_mask[:, None], effective, 0.0)
    torque = jnp.cross(directions, effective)
    magnetization = jnp.sum(prepared.moments[:, None] * directions, axis=0)
    norm_residual = jnp.max(
        jnp.where(
            prepared.plan.site_mask,
            jnp.abs(jnp.sqrt(jnp.sum(directions * directions, axis=-1)) - 1.0),
            0.0,
        ),
        initial=0.0,
    )
    reconstruction = (
        exchange_energy
        + symmetric_energy
        + dmi_energy
        + anisotropy_energy
        + zeeman_energy
    )
    ledger_residual = jnp.abs(total - reconstruction)
    successful = (
        jnp.all(jnp.isfinite(directions))
        & jnp.isfinite(total)
        & jnp.all(jnp.isfinite(effective))
        & (norm_residual <= 10.0 * prepared.plan.validation_tolerance)
    )
    return ClassicalSpinEvaluation(
        total,
        exchange_energy,
        symmetric_energy,
        dmi_energy,
        anisotropy_energy,
        zeeman_energy,
        effective,
        torque,
        magnetization,
        norm_residual,
        ledger_residual,
        successful,
        prepared.prepared_id,
    )


__all__ = [
    "ClassicalSpinEvaluation",
    "ClassicalSpinHamiltonianPlan",
    "ClassicalSpinState",
    "PreparedClassicalSpinHamiltonian",
    "evaluate_classical_spin_hamiltonian",
    "prepare_classical_spin_hamiltonian",
    "refresh_classical_spin_hamiltonian",
]
