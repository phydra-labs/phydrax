#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._register import HilbertRegisterLayout
from ._subspaces import BasisStateSubspace, DenseQuantumSubspace, QuantumSubspace


GaussSectorMethod: TypeAlias = Literal["exact-nullspace", "sparse-basis"]


class GaussConstraintNetwork(StrictModule):
    """Finite oriented Gauss network with optional signed matter/background charge.

    For incidence ``+1`` a link contributes its left generator, for ``-1`` it
    contributes minus its right generator.  The complete convention is
    ``G[v,a] = links[v,a] + matter[v,a] - background[v,a] I``.
    """

    layout: HilbertRegisterLayout
    vertex_link_incidence: Array
    incidence_pattern: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    link_left_generators: tuple[Array, ...]
    link_right_generators: tuple[Array, ...]
    matter_charge_generators: tuple[Array, ...]
    background_charges: Array
    local_hermiticity_residual: Array
    finite: Array
    valid: Array
    vertex_count: int = eqx.field(static=True)
    link_count: int = eqx.field(static=True)
    generator_count: int = eqx.field(static=True)
    link_wire_ids: tuple[str, ...] = eqx.field(static=True)
    matter_wire_ids: tuple[str, ...] = eqx.field(static=True)
    network_id: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)

    def __init__(
        self,
        vertex_link_incidence: ArrayLike,
        link_left_generators: Sequence[ArrayLike],
        link_right_generators: Sequence[ArrayLike],
        /,
        *,
        matter_charge_generators: Sequence[ArrayLike] | None = None,
        background_charges: ArrayLike | None = None,
        link_wire_ids: Sequence[str] | None = None,
        matter_wire_ids: Sequence[str] | None = None,
        hermiticity_tolerance: float = 1e-10,
    ):
        incidence_host = np.asarray(vertex_link_incidence)
        if (
            incidence_host.ndim != 2
            or incidence_host.shape[0] == 0
            or incidence_host.shape[1] == 0
        ):
            raise ValueError("vertex_link_incidence must have nonempty shape (V, E).")
        if np.any(~np.isin(incidence_host, (-1, 0, 1))):
            raise ValueError("Gauss incidence entries must be minus one, zero, or one.")
        vertex_count, link_count = map(int, incidence_host.shape)
        left = tuple(jnp.asarray(value) for value in link_left_generators)
        right = tuple(jnp.asarray(value) for value in link_right_generators)
        if len(left) != link_count or len(right) != link_count:
            raise ValueError("One left and right generator array is required per link.")
        if any(
            value.ndim != 3
            or value.shape[0] == 0
            or value.shape[1] != value.shape[2]
            or value.shape[1] == 0
            for value in left + right
        ):
            raise ValueError("Link generators require shape (generator, d, d).")
        generator_count = left[0].shape[0]
        if any(
            first.shape != second.shape or first.shape[0] != generator_count
            for first, second in zip(left, right, strict=True)
        ):
            raise ValueError("Left and right link generators must align exactly.")
        matter = (
            ()
            if matter_charge_generators is None
            else tuple(jnp.asarray(value) for value in matter_charge_generators)
        )
        if matter and (
            len(matter) != vertex_count
            or any(
                value.ndim != 3
                or value.shape[0] != generator_count
                or value.shape[1] != value.shape[2]
                or value.shape[1] == 0
                for value in matter
            )
        ):
            raise ValueError(
                "Matter generators require one shape (generator, d, d) array per vertex."
            )
        background = (
            jnp.zeros((vertex_count, generator_count), dtype=jnp.float64)
            if background_charges is None
            else jnp.asarray(background_charges)
        )
        if background.shape != (vertex_count, generator_count):
            raise ValueError("background_charges must have shape (vertex, generator).")
        if jnp.issubdtype(background.dtype, jnp.complexfloating):
            raise TypeError("background_charges must be real.")
        tolerance = float(hermiticity_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("hermiticity_tolerance must be finite and non-negative.")
        link_ids = (
            tuple(f"link:{index}" for index in range(link_count))
            if link_wire_ids is None
            else tuple(str(value) for value in link_wire_ids)
        )
        if not matter and matter_wire_ids is not None:
            raise ValueError("matter_wire_ids require matter_charge_generators.")
        matter_ids = (
            ()
            if not matter
            else tuple(f"matter:{index}" for index in range(vertex_count))
            if matter_wire_ids is None
            else tuple(str(value) for value in matter_wire_ids)
        )
        if len(link_ids) != link_count or len(matter_ids) != len(matter):
            raise ValueError(
                "Wire identifiers must align with links and matter vertices."
            )
        dimensions = tuple(value.shape[1] for value in left) + tuple(
            value.shape[1] for value in matter
        )
        layout = HilbertRegisterLayout(link_ids + matter_ids, dimensions)
        local_arrays = left + right + matter
        residuals = tuple(
            jnp.max(jnp.abs(value - jnp.conj(jnp.swapaxes(value, -1, -2))))
            for value in local_arrays
        )
        hermiticity = jnp.max(jnp.stack(residuals))
        finite = jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(value)) for value in local_arrays)
                + (jnp.all(jnp.isfinite(background)),)
            )
        )
        network_id = canonical_fingerprint(
            {
                "kind": "gauss-constraint-network",
                "layout": layout.layout_id,
                "incidence": array_tree_fingerprint(incidence_host.astype(np.int8)),
                "left_generators": tuple(
                    array_tree_fingerprint(np.asarray(value)) for value in left
                ),
                "right_generators": tuple(
                    array_tree_fingerprint(np.asarray(value)) for value in right
                ),
                "matter_generators": tuple(
                    array_tree_fingerprint(np.asarray(value)) for value in matter
                ),
                "background": array_tree_fingerprint(np.asarray(background)),
                "convention": "outgoing-left-minus-incoming-right-plus-matter-minus-background",
            }
        )
        self.layout = layout
        self.vertex_link_incidence = jnp.asarray(incidence_host, dtype=jnp.int8)
        self.incidence_pattern = tuple(tuple(row) for row in incidence_host)
        self.link_left_generators = left
        self.link_right_generators = right
        self.matter_charge_generators = matter
        self.background_charges = background
        self.local_hermiticity_residual = hermiticity
        self.finite = finite
        self.valid = finite & (hermiticity <= tolerance)
        self.vertex_count = vertex_count
        self.link_count = link_count
        self.generator_count = generator_count
        self.link_wire_ids = link_ids
        self.matter_wire_ids = matter_ids
        self.network_id = network_id
        self.convention = (
            "outgoing-left-minus-incoming-right-plus-matter-minus-background"
        )

    @property
    def physical_dimension(self) -> int:
        return self.layout.dimension

    def dense_generators(self, /, *, maximum_elements: int = 1 << 27) -> Array:
        """Materialize all G[v,a] after a fail-before-allocation guard."""

        dimension = self.physical_dimension
        required = self.vertex_count * self.generator_count * dimension * dimension
        maximum = int(maximum_elements)
        if maximum <= 0 or required > maximum:
            raise ValueError(
                f"Dense Gauss generators require {required} elements; capacity is {maximum}."
            )
        dtype = jnp.result_type(
            *(value.dtype for value in self.link_left_generators),
            *(value.dtype for value in self.matter_charge_generators),
            self.background_charges,
            1j,
        )
        identity = jnp.eye(dimension, dtype=dtype)
        rows = []
        for vertex in range(self.vertex_count):
            generators = []
            for generator in range(self.generator_count):
                value = (
                    -self.background_charges[vertex, generator].astype(dtype) * identity
                )
                for link in range(self.link_count):
                    orientation = self.incidence_pattern[vertex][link]
                    if orientation == 1:
                        local = self.link_left_generators[link][generator]
                        value = value + _embed_local_operator(
                            local, link, self.layout.local_dimensions, dtype
                        )
                    elif orientation == -1:
                        local = self.link_right_generators[link][generator]
                        value = value - _embed_local_operator(
                            local, link, self.layout.local_dimensions, dtype
                        )
                if self.matter_charge_generators:
                    local_matter = self.matter_charge_generators[vertex][generator]
                    value = value + _embed_local_operator(
                        local_matter,
                        self.link_count + vertex,
                        self.layout.local_dimensions,
                        dtype,
                    )
                generators.append(value)
            rows.append(jnp.stack(generators))
        return jnp.stack(rows)


class GaussSectorResourcePolicy(StrictModule):
    """Hard admission limits for finite physical-sector construction."""

    maximum_hilbert_dimension: int = eqx.field(static=True)
    maximum_dense_elements: int = eqx.field(static=True)
    maximum_basis_states: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_hilbert_dimension: int = 4096,
        maximum_dense_elements: int = 1 << 27,
        maximum_basis_states: int = 1 << 20,
        tolerance: float = 1e-10,
    ):
        dimension = int(maximum_hilbert_dimension)
        dense = int(maximum_dense_elements)
        basis = int(maximum_basis_states)
        tolerance_ = float(tolerance)
        if dimension <= 0 or dense <= 0 or basis <= 0:
            raise ValueError("Gauss-sector resource limits must be positive.")
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Gauss-sector tolerance must be finite and non-negative.")
        self.maximum_hilbert_dimension = dimension
        self.maximum_dense_elements = dense
        self.maximum_basis_states = basis
        self.tolerance = tolerance_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "gauss-sector-resource-policy",
                "maximum_hilbert_dimension": dimension,
                "maximum_dense_elements": dense,
                "maximum_basis_states": basis,
                "tolerance": tolerance_,
            }
        )


class GaussPhysicalSectorPlan(StrictModule):
    """Immutable structural admission and cost record for one physical sector."""

    resources: GaussSectorResourcePolicy
    method: str = eqx.field(static=True)
    physical_dimension: int = eqx.field(static=True)
    constraint_count: int = eqx.field(static=True)
    estimated_dense_elements: int = eqx.field(static=True)
    network_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class GaussPhysicalSectorEvidence(StrictModule):
    maximum_constraint_residual: Array
    projector_residual: Array
    finite: Array
    valid: Array
    physical_dimension: int = eqx.field(static=True)
    logical_dimension: int = eqx.field(static=True)
    method: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PreparedGaussPhysicalSector(StrictModule):
    """Prepared exact or computational-basis physical subspace."""

    plan: GaussPhysicalSectorPlan
    subspace: QuantumSubspace
    evidence: GaussPhysicalSectorEvidence
    prepared_id: str = eqx.field(static=True)


class GaussCommutatorEvidence(StrictModule):
    commutator_residuals: Array
    maximum_residual: Array
    finite: Array
    gauge_invariant: Array
    network_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def _embed_local_operator(
    operator: Array,
    wire: int,
    dimensions: tuple[int, ...],
    dtype: jnp.dtype,
    /,
) -> Array:
    value = jnp.asarray([[1.0]], dtype=dtype)
    for index, dimension in enumerate(dimensions):
        factor = (
            operator.astype(dtype) if index == wire else jnp.eye(dimension, dtype=dtype)
        )
        value = jnp.kron(value, factor)
    return value


def plan_gauss_physical_sector(
    network: GaussConstraintNetwork,
    /,
    *,
    method: GaussSectorMethod = "exact-nullspace",
    resources: GaussSectorResourcePolicy | None = None,
) -> GaussPhysicalSectorPlan:
    """Admit sector construction without materializing basis states or constraints."""

    if not isinstance(network, GaussConstraintNetwork):
        raise TypeError("network must be GaussConstraintNetwork.")
    if not bool(network.valid):
        raise ValueError("Gauss physical sectors require a valid constraint network.")
    if method not in ("exact-nullspace", "sparse-basis"):
        raise ValueError("Unknown Gauss physical-sector method.")
    selected = GaussSectorResourcePolicy() if resources is None else resources
    if not isinstance(selected, GaussSectorResourcePolicy):
        raise TypeError("resources must be GaussSectorResourcePolicy or None.")
    dimension = network.physical_dimension
    constraint_count = network.vertex_count * network.generator_count
    estimated = (
        (constraint_count + 2) * dimension * dimension
        if method == "exact-nullspace"
        else dimension * (network.layout.wire_count + constraint_count)
    )
    if dimension > selected.maximum_hilbert_dimension:
        raise ValueError("Gauss sector exceeds maximum_hilbert_dimension.")
    if method == "exact-nullspace" and estimated > selected.maximum_dense_elements:
        raise ValueError("Exact Gauss sector exceeds maximum_dense_elements.")
    if method == "sparse-basis" and dimension > selected.maximum_basis_states:
        raise ValueError("Sparse Gauss sector exceeds maximum_basis_states.")
    plan_id = canonical_fingerprint(
        {
            "kind": "gauss-physical-sector-plan",
            "network": network.network_id,
            "method": method,
            "resources": selected.policy_id,
            "physical_dimension": dimension,
            "constraint_count": constraint_count,
            "estimated_dense_elements": estimated,
        }
    )
    return GaussPhysicalSectorPlan(
        selected,
        method,
        dimension,
        constraint_count,
        estimated,
        network.network_id,
        plan_id,
    )


def _mixed_radix_levels(
    dimensions: tuple[int, ...], maximum_basis_states: int, /
) -> np.ndarray:
    dimension = prod(dimensions)
    if dimension > int(maximum_basis_states):
        raise ValueError("Basis enumeration exceeds maximum_basis_states.")
    indices = np.arange(dimension, dtype=np.int64)
    levels = np.empty((dimension, len(dimensions)), dtype=np.int32)
    remainder = indices.copy()
    for wire in range(len(dimensions) - 1, -1, -1):
        levels[:, wire] = remainder % dimensions[wire]
        remainder //= dimensions[wire]
    return levels


def _prepare_sparse_basis(
    network: GaussConstraintNetwork, plan: GaussPhysicalSectorPlan, /
) -> tuple[BasisStateSubspace, Array, Array]:
    local_arrays = (
        network.link_left_generators
        + network.link_right_generators
        + network.matter_charge_generators
    )
    offdiagonal = max(
        float(
            np.max(
                np.abs(
                    np.asarray(value)
                    - np.asarray(
                        tuple(np.diag(np.diag(matrix)) for matrix in np.asarray(value))
                    )
                )
            )
        )
        for value in local_arrays
    )
    if offdiagonal > plan.resources.tolerance:
        raise ValueError("sparse-basis sectors require diagonal local Gauss generators.")
    levels = _mixed_radix_levels(
        network.layout.local_dimensions, plan.resources.maximum_basis_states
    )
    residuals = np.zeros(
        (network.physical_dimension, network.vertex_count, network.generator_count),
        dtype=np.complex128,
    )
    incidence = np.asarray(network.vertex_link_incidence)
    background = np.asarray(network.background_charges)
    residuals -= background[None, :, :]
    for vertex in range(network.vertex_count):
        for link in range(network.link_count):
            orientation = int(incidence[vertex, link])
            if orientation == 0:
                continue
            source = (
                network.link_left_generators[link]
                if orientation == 1
                else network.link_right_generators[link]
            )
            diagonal = np.diagonal(np.asarray(source), axis1=-2, axis2=-1)
            residuals[:, vertex, :] += orientation * diagonal[:, levels[:, link]].T
        if network.matter_charge_generators:
            matter = np.diagonal(
                np.asarray(network.matter_charge_generators[vertex]),
                axis1=-2,
                axis2=-1,
            )
            residuals[:, vertex, :] += matter[:, levels[:, network.link_count + vertex]].T
    norms = np.max(np.abs(residuals), axis=(1, 2))
    selected = np.flatnonzero(norms <= plan.resources.tolerance)
    if selected.size == 0:
        raise ValueError("Gauss constraints have no physical basis states.")
    subspace = BasisStateSubspace(
        network.physical_dimension,
        selected,
        subspace_id=canonical_fingerprint(
            {
                "kind": "sparse-gauss-basis",
                "plan": plan.plan_id,
                "indices": selected,
            }
        ),
    )
    return subspace, jnp.asarray(np.max(norms[selected])), jnp.asarray(0.0)


def _prepare_exact_nullspace(
    network: GaussConstraintNetwork, plan: GaussPhysicalSectorPlan, /
) -> tuple[DenseQuantumSubspace, Array, Array]:
    generators = network.dense_generators(
        maximum_elements=plan.resources.maximum_dense_elements
    )
    host = np.asarray(generators).reshape(
        (plan.constraint_count, network.physical_dimension, network.physical_dimension)
    )
    stacked = host.reshape(
        (plan.constraint_count * network.physical_dimension, network.physical_dimension)
    )
    _, singular_values, right_adjoint = np.linalg.svd(stacked, full_matrices=True)
    rank = int(
        np.count_nonzero(
            singular_values
            > plan.resources.tolerance * max(1.0, float(singular_values[0]))
        )
    )
    isometry = np.conj(right_adjoint[rank:].T)
    if isometry.shape[1] == 0:
        raise ValueError("Gauss constraints have no numerical nullspace.")
    actions = host @ isometry
    maximum_residual = np.max(np.linalg.norm(actions, axis=1))
    projector = isometry @ np.conj(isometry.T)
    projector_residual = np.linalg.norm(projector @ projector - projector)
    subspace = DenseQuantumSubspace(
        jnp.asarray(isometry),
        tolerance=plan.resources.tolerance,
        maximum_entries=plan.resources.maximum_dense_elements,
        subspace_id=canonical_fingerprint(
            {
                "kind": "exact-gauss-nullspace",
                "plan": plan.plan_id,
                "logical_dimension": isometry.shape[1],
            }
        ),
    )
    return subspace, jnp.asarray(maximum_residual), jnp.asarray(projector_residual)


def prepare_gauss_physical_sector(
    plan: GaussPhysicalSectorPlan,
    network: GaussConstraintNetwork,
    /,
) -> PreparedGaussPhysicalSector:
    """Prepare a previously admitted physical sector from current numeric generators."""

    if not isinstance(plan, GaussPhysicalSectorPlan):
        raise TypeError("plan must be GaussPhysicalSectorPlan.")
    if not isinstance(network, GaussConstraintNetwork):
        raise TypeError("network must be GaussConstraintNetwork.")
    if plan.network_id != network.network_id:
        raise ValueError("Gauss physical-sector plan and network identities differ.")
    if plan.method == "sparse-basis":
        subspace, residual, projector_residual = _prepare_sparse_basis(network, plan)
    else:
        subspace, residual, projector_residual = _prepare_exact_nullspace(network, plan)
    finite = jnp.isfinite(residual) & jnp.isfinite(projector_residual)
    valid = (
        finite
        & subspace.evidence.valid
        & (residual <= plan.resources.tolerance)
        & (projector_residual <= plan.resources.tolerance)
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "gauss-physical-sector-evidence",
            "plan": plan.plan_id,
            "subspace": subspace.subspace_id,
        }
    )
    evidence = GaussPhysicalSectorEvidence(
        residual,
        projector_residual,
        finite,
        valid,
        plan.physical_dimension,
        subspace.logical_dimension,
        plan.method,
        evidence_id,
    )
    return PreparedGaussPhysicalSector(
        plan,
        subspace,
        evidence,
        canonical_fingerprint(
            {
                "kind": "prepared-gauss-physical-sector",
                "plan": plan.plan_id,
                "subspace": subspace.subspace_id,
            }
        ),
    )


def prepare_exact_physical_sector(
    network: GaussConstraintNetwork,
    /,
    *,
    resources: GaussSectorResourcePolicy | None = None,
) -> PreparedGaussPhysicalSector:
    return prepare_gauss_physical_sector(
        plan_gauss_physical_sector(
            network, method="exact-nullspace", resources=resources
        ),
        network,
    )


def prepare_sparse_physical_sector(
    network: GaussConstraintNetwork,
    /,
    *,
    resources: GaussSectorResourcePolicy | None = None,
) -> PreparedGaussPhysicalSector:
    return prepare_gauss_physical_sector(
        plan_gauss_physical_sector(network, method="sparse-basis", resources=resources),
        network,
    )


def gauss_commutator_evidence(
    network: GaussConstraintNetwork,
    operator: ArrayLike,
    /,
    *,
    operator_id: str,
    maximum_elements: int = 1 << 27,
    tolerance: float = 1e-10,
) -> GaussCommutatorEvidence:
    """Compute ``[G[v,a], O]`` for one finite dense operator."""

    if not isinstance(network, GaussConstraintNetwork):
        raise TypeError("network must be GaussConstraintNetwork.")
    value = jnp.asarray(operator)
    dimension = network.physical_dimension
    if value.shape != (dimension, dimension):
        raise ValueError("operator must span the complete Gauss-network Hilbert space.")
    identifier = str(operator_id)
    if not identifier:
        raise ValueError("operator_id must be nonempty.")
    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")
    generators = network.dense_generators(maximum_elements=maximum_elements)
    commutators = generators @ value - value @ generators
    residuals = jnp.linalg.norm(commutators, axis=(-2, -1))
    maximum = jnp.max(residuals)
    finite = jnp.all(jnp.isfinite(residuals))
    evidence_id = canonical_fingerprint(
        {
            "kind": "gauss-commutator-evidence",
            "network": network.network_id,
            "operator": identifier,
            "tolerance": tolerance_,
        }
    )
    return GaussCommutatorEvidence(
        residuals,
        maximum,
        finite,
        finite & (maximum <= tolerance_),
        network.network_id,
        identifier,
        evidence_id,
    )


def gauss_leakage(
    network: GaussConstraintNetwork,
    state: ArrayLike,
    /,
    *,
    maximum_elements: int = 1 << 27,
) -> Array:
    """Return normalized total squared Gauss-generator action on one state."""

    if not isinstance(network, GaussConstraintNetwork):
        raise TypeError("network must be GaussConstraintNetwork.")
    vector = jnp.asarray(state)
    if vector.shape != (network.physical_dimension,):
        raise ValueError("state must span the complete Gauss-network Hilbert space.")
    generators = network.dense_generators(maximum_elements=maximum_elements)
    actions = generators @ vector
    norm = jnp.real(jnp.vdot(vector, vector))
    norm = eqx.error_if(
        norm,
        ~jnp.isfinite(norm) | (norm <= 0.0),
        "Gauss leakage requires a finite nonzero state.",
    )
    return jnp.real(jnp.sum(jnp.abs(actions) ** 2)) / norm


__all__ = [
    "GaussCommutatorEvidence",
    "GaussConstraintNetwork",
    "GaussPhysicalSectorEvidence",
    "GaussPhysicalSectorPlan",
    "GaussSectorMethod",
    "GaussSectorResourcePolicy",
    "PreparedGaussPhysicalSector",
    "gauss_commutator_evidence",
    "gauss_leakage",
    "plan_gauss_physical_sector",
    "prepare_exact_physical_sector",
    "prepare_gauss_physical_sector",
    "prepare_sparse_physical_sector",
]
