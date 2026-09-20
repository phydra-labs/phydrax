#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared route action in finite character-projected orbit sectors."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ....linalg import (
    AbstractLinearOperator,
    ArraySpace,
    LinearCapabilityError,
    OperatorCapabilities,
    OperatorProperties,
)
from ....sparse import EdgeRelation
from ._compile import PreparedQuantumLattice
from ._operator import apply_compiled_to_coordinate
from ._orbit_sector import PreparedOrbitSectorBasis


class OrbitOperatorResourcePolicy(StrictModule):
    """Hard reduced-route and preparation-workspace limits."""

    maximum_routes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    invariance_tolerance: float = eqx.field(static=True)
    route_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_routes: int,
        maximum_workspace_bytes: int,
        invariance_tolerance: float = 1e-10,
        route_tolerance: float = 1e-14,
    ):
        routes = int(maximum_routes)
        workspace = int(maximum_workspace_bytes)
        invariance = float(invariance_tolerance)
        route = float(route_tolerance)
        if routes < 1 or workspace < 1:
            raise ValueError("Orbit-operator resource limits must be positive.")
        if (
            not np.isfinite(invariance)
            or invariance <= 0.0
            or not np.isfinite(route)
            or route < 0.0
        ):
            raise ValueError("Orbit-operator tolerances must be finite and valid.")
        self.maximum_routes = routes
        self.maximum_workspace_bytes = workspace
        self.invariance_tolerance = invariance
        self.route_tolerance = route
        self.policy_id = canonical_fingerprint(
            {
                "kind": "orbit-operator-resource-policy",
                "maximum_routes": routes,
                "maximum_workspace_bytes": workspace,
                "invariance_tolerance": invariance,
                "route_tolerance": route,
            }
        )


class OrbitOperatorEvidence(StrictModule):
    """Finite action-invariance and reduced-Hermiticity evidence."""

    invariance_residuals: Array
    maximum_invariance_residual: Array
    hermiticity_residual: Array
    accepted: Array
    group_order: int = eqx.field(static=True)
    direct_dimension: int = eqx.field(static=True)
    reduced_dimension: int = eqx.field(static=True)
    route_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class QuantumOrbitSectorOperator(AbstractLinearOperator):
    """Fixed-route quantum-lattice action in one character-projected sector."""

    prepared: PreparedQuantumLattice
    basis: PreparedOrbitSectorBasis
    relation: EdgeRelation
    amplitudes: Array
    evidence: OrbitOperatorEvidence
    resources: OrbitOperatorResourcePolicy = eqx.field(static=True)
    action_workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedQuantumLattice,
        basis: PreparedOrbitSectorBasis,
        relation: EdgeRelation,
        amplitudes: ArrayLike,
        evidence: OrbitOperatorEvidence,
        resources: OrbitOperatorResourcePolicy,
        /,
        *,
        action_workspace_bytes: int,
    ):
        if not isinstance(prepared, PreparedQuantumLattice):
            raise TypeError("prepared must be PreparedQuantumLattice.")
        if not isinstance(basis, PreparedOrbitSectorBasis):
            raise TypeError("basis must be PreparedOrbitSectorBasis.")
        if not isinstance(relation, EdgeRelation):
            raise TypeError("relation must be EdgeRelation.")
        if not isinstance(evidence, OrbitOperatorEvidence):
            raise TypeError("evidence must be OrbitOperatorEvidence.")
        if not isinstance(resources, OrbitOperatorResourcePolicy):
            raise TypeError("resources must be OrbitOperatorResourcePolicy.")
        values = jnp.asarray(amplitudes, dtype=jnp.complex128)
        if values.shape != relation.route_shape:
            raise ValueError("Orbit-route amplitudes must match the edge relation.")
        if (
            relation.source_size != basis.dimension
            or relation.target_size != basis.dimension
        ):
            raise ValueError("Orbit routes must act within the prepared orbit sector.")
        self.prepared = prepared
        self.basis = basis
        self.relation = relation
        self.amplitudes = values
        self.evidence = evidence
        self.resources = resources
        self.action_workspace_bytes = int(action_workspace_bytes)
        space = ArraySpace(
            (basis.dimension,),
            dtype=np.complex128,
            space_id=f"quantum-orbit-sector:{basis.basis_id}",
        )
        self.source = space
        self.target = space
        self.properties = OperatorProperties(
            self_adjoint=prepared.specification.self_adjoint and bool(evidence.accepted),
            evidence={"self_adjoint": "verified"}
            if prepared.specification.self_adjoint and bool(evidence.accepted)
            else None,
        )
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "matrix-free-quantum-orbit-sector-operator",
                "prepared": prepared.prepared_id,
                "basis": basis.basis_id,
                "relation": array_tree_fingerprint(
                    (relation.source_indices, relation.target_indices, relation.valid)
                ),
                "amplitudes": array_tree_fingerprint(values),
                "evidence": evidence.evidence_id,
                "resources": resources.policy_id,
            }
        )

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(vector)
        contributions = jnp.where(
            self.relation.valid,
            self.amplitudes * value[self.relation.source_indices],
            0.0j,
        )
        result = jnp.zeros((self.basis.dimension,), dtype=jnp.complex128)
        result = result.at[self.relation.target_indices].add(contributions)
        return self.target.validate(result)

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        contributions = jnp.where(
            self.relation.valid,
            self.amplitudes * value[self.relation.target_indices],
            0.0j,
        )
        result = jnp.zeros((self.basis.dimension,), dtype=jnp.complex128)
        result = result.at[self.relation.source_indices].add(contributions)
        return self.source.validate(result)

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        contributions = jnp.where(
            self.relation.valid,
            jnp.conj(self.amplitudes) * value[self.relation.target_indices],
            0.0j,
        )
        result = jnp.zeros((self.basis.dimension,), dtype=jnp.complex128)
        result = result.at[self.relation.source_indices].add(contributions)
        return self.source.validate(result)

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "QuantumOrbitSectorOperator intentionally forbids dense materialization."
        )


def _direct_action_columns(
    prepared: PreparedQuantumLattice,
    basis: PreparedOrbitSectorBasis,
    tolerance: float,
    /,
) -> list[dict[int, complex]]:
    direct = basis.base
    if (
        direct.site_ids != prepared.specification.site_ids
        or direct.site_dimensions != prepared.specification.local_dimensions
    ):
        raise ValueError(
            "Orbit-sector coordinates do not match the lattice specification."
        )
    columns: list[dict[int, complex]] = []
    for source in range(direct.dimension):
        coordinate = np.asarray(direct.coordinate(source), dtype=np.int32)
        outputs, amplitudes = apply_compiled_to_coordinate(prepared, coordinate)
        output_host = np.asarray(outputs, dtype=np.int32)
        amplitude_host = np.asarray(amplitudes, dtype=np.complex128)
        column: dict[int, complex] = {}
        for output, amplitude in zip(output_host, amplitude_host, strict=True):
            if abs(amplitude) <= tolerance:
                continue
            if not bool(np.asarray(direct.contains(output))):
                raise ValueError(
                    "Compiled action leaves the direct sector underlying the orbit basis."
                )
            target = int(np.asarray(direct.rank(output)))
            column[target] = column.get(target, 0.0 + 0.0j) + complex(amplitude)
        columns.append(
            {target: value for target, value in column.items() if abs(value) > tolerance}
        )
    return columns


def _column_difference_norm(
    left: dict[int, complex],
    right: dict[int, complex],
    /,
) -> float:
    keys = left.keys() | right.keys()
    if not keys:
        return 0.0
    numerator = np.sqrt(
        sum(abs(left.get(key, 0.0j) - right.get(key, 0.0j)) ** 2 for key in keys)
    )
    denominator = max(
        1.0,
        np.sqrt(sum(abs(left.get(key, 0.0j)) ** 2 for key in keys)),
        np.sqrt(sum(abs(right.get(key, 0.0j)) ** 2 for key in keys)),
    )
    return float(numerator / denominator)


def _invariance_residuals(
    columns: list[dict[int, complex]],
    basis: PreparedOrbitSectorBasis,
    /,
) -> np.ndarray:
    permutations = np.asarray(basis.action.permutations, dtype=np.int32)
    phases = np.asarray(basis.action.phases, dtype=np.complex128)
    residuals = np.zeros((basis.action.group_order,), dtype=np.float64)
    for group_index in range(basis.action.group_order):
        permutation = permutations[group_index]
        phase = phases[group_index]
        maximum = 0.0
        for source, column in enumerate(columns):
            left = {
                int(permutation[target]): amplitude * phase[target]
                for target, amplitude in column.items()
            }
            right = {
                target: phase[source] * amplitude
                for target, amplitude in columns[int(permutation[source])].items()
            }
            maximum = max(maximum, _column_difference_norm(left, right))
        residuals[group_index] = maximum
    return residuals


def _projected_matrix(
    columns: list[dict[int, complex]],
    basis: PreparedOrbitSectorBasis,
    /,
) -> np.ndarray:
    raw_to_orbit = np.asarray(basis.raw_to_orbit, dtype=np.int32)
    embedding = np.asarray(basis.embedding_coefficients, dtype=np.complex128)
    reduced = np.zeros((basis.dimension, basis.dimension), dtype=np.complex128)
    for direct_source, column in enumerate(columns):
        source = int(raw_to_orbit[direct_source])
        source_coefficient = embedding[direct_source]
        if source < 0 or abs(source_coefficient) <= basis.projection_tolerance:
            continue
        for direct_target, amplitude in column.items():
            target = int(raw_to_orbit[direct_target])
            target_coefficient = embedding[direct_target]
            if target < 0 or abs(target_coefficient) <= basis.projection_tolerance:
                continue
            reduced[target, source] += (
                np.conj(target_coefficient) * amplitude * source_coefficient
            )
    return reduced


def prepare_quantum_orbit_sector_operator(
    prepared: PreparedQuantumLattice,
    basis: PreparedOrbitSectorBasis,
    resources: OrbitOperatorResourcePolicy,
    /,
) -> QuantumOrbitSectorOperator:
    """Audit invariance and compile a coalesced reduced route operator."""
    if not isinstance(prepared, PreparedQuantumLattice):
        raise TypeError("prepared must be PreparedQuantumLattice.")
    if not isinstance(basis, PreparedOrbitSectorBasis):
        raise TypeError("basis must be PreparedOrbitSectorBasis.")
    if not isinstance(resources, OrbitOperatorResourcePolicy):
        raise TypeError("resources must be OrbitOperatorResourcePolicy.")

    direct_dense_bytes = (
        basis.base.dimension * basis.base.dimension * np.dtype(np.complex128).itemsize
    )
    reduced_dense_bytes = (
        basis.dimension * basis.dimension * np.dtype(np.complex128).itemsize
    )
    preparation_workspace = direct_dense_bytes + reduced_dense_bytes
    if preparation_workspace > resources.maximum_workspace_bytes:
        raise ValueError("Orbit-operator preparation exceeds maximum_workspace_bytes.")

    columns = _direct_action_columns(prepared, basis, resources.route_tolerance)
    invariance = _invariance_residuals(columns, basis)
    maximum_invariance = float(np.max(invariance)) if invariance.size else 0.0
    if maximum_invariance > resources.invariance_tolerance:
        raise ValueError(
            "Compiled quantum-lattice action is not invariant under the finite group."
        )

    reduced = _projected_matrix(columns, basis)
    hermiticity = float(
        np.linalg.norm(reduced - np.conj(reduced.T))
        / max(1.0, float(np.linalg.norm(reduced)))
    )
    accepted = maximum_invariance <= resources.invariance_tolerance
    if prepared.specification.self_adjoint:
        accepted = accepted and hermiticity <= resources.invariance_tolerance
        if not accepted:
            raise ValueError("Projected self-adjoint action failed Hermiticity audit.")

    targets, sources = np.nonzero(np.abs(reduced) > resources.route_tolerance)
    amplitudes = reduced[targets, sources]
    if amplitudes.size > resources.maximum_routes:
        raise ValueError("Reduced orbit action exceeds maximum_routes.")
    if amplitudes.size == 0:
        sources = np.zeros((1,), dtype=np.int32)
        targets = np.zeros((1,), dtype=np.int32)
        amplitudes = np.zeros((1,), dtype=np.complex128)
        valid = np.zeros((1,), dtype=np.bool_)
        route_count = 0
    else:
        sources = sources.astype(np.int32)
        targets = targets.astype(np.int32)
        valid = np.ones(amplitudes.shape, dtype=np.bool_)
        route_count = amplitudes.size
    action_workspace = (
        basis.dimension * np.dtype(np.complex128).itemsize
        + sources.nbytes
        + targets.nbytes
        + amplitudes.nbytes
        + valid.nbytes
    )
    if action_workspace > resources.maximum_workspace_bytes:
        raise ValueError("Reduced orbit action exceeds maximum_workspace_bytes.")

    evidence_id = canonical_fingerprint(
        {
            "kind": "orbit-operator-evidence",
            "prepared": prepared.prepared_id,
            "basis": basis.basis_id,
            "invariance": array_tree_fingerprint(invariance),
            "hermiticity": hermiticity,
            "routes": route_count,
        }
    )
    evidence = OrbitOperatorEvidence(
        invariance_residuals=jnp.asarray(invariance),
        maximum_invariance_residual=jnp.asarray(maximum_invariance),
        hermiticity_residual=jnp.asarray(hermiticity),
        accepted=jnp.asarray(accepted),
        group_order=basis.action.group_order,
        direct_dimension=basis.base.dimension,
        reduced_dimension=basis.dimension,
        route_count=route_count,
        evidence_id=evidence_id,
    )
    relation = EdgeRelation(
        sources,
        targets,
        source_size=basis.dimension,
        target_size=basis.dimension,
        valid=valid,
    )
    return QuantumOrbitSectorOperator(
        prepared,
        basis,
        relation,
        amplitudes,
        evidence,
        resources,
        action_workspace_bytes=action_workspace,
    )


__all__ = [
    "OrbitOperatorEvidence",
    "OrbitOperatorResourcePolicy",
    "QuantumOrbitSectorOperator",
    "prepare_quantum_orbit_sector_operator",
]
