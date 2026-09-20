#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Left-associated finite SU(2) coupling bases and projected operators."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ....tensor_network import su2_clebsch_gordan, su2_fusion


class SU2SectorResourcePolicy(StrictModule):
    maximum_product_dimension: int = eqx.field(static=True)
    maximum_sector_dimension: int = eqx.field(static=True)
    maximum_matrix_elements: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_product_dimension: int,
        maximum_sector_dimension: int,
        maximum_matrix_elements: int,
    ):
        values = tuple(
            (
                maximum_product_dimension,
                maximum_sector_dimension,
                maximum_matrix_elements,
            )
        )
        if any(value < 1 for value in values):
            raise ValueError("SU2 sector resource limits must be positive.")
        self.maximum_product_dimension = values[0]
        self.maximum_sector_dimension = values[1]
        self.maximum_matrix_elements = values[2]
        self.policy_id = canonical_fingerprint(
            {
                "kind": "su2-sector-resource-policy",
                "maximum_product_dimension": values[0],
                "maximum_sector_dimension": values[1],
                "maximum_matrix_elements": values[2],
            }
        )


class SU2CouplingTreePlan(StrictModule):
    """Fixed local doubled spins and one left-associated total-spin sector."""

    site_labels: tuple[str, ...] = eqx.field(static=True)
    local_twice_spins: tuple[int, ...] = eqx.field(static=True)
    total_twice_spin: int = eqx.field(static=True)
    coupling_paths: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    resources: SU2SectorResourcePolicy = eqx.field(static=True)
    product_dimension: int = eqx.field(static=True)
    multiplicity_dimension: int = eqx.field(static=True)
    sector_dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_labels: Sequence[str],
        local_twice_spins: Sequence[int],
        total_twice_spin: int,
        resources: SU2SectorResourcePolicy,
        /,
    ):
        labels = tuple(str(value) for value in site_labels)
        spins = tuple(local_twice_spins)
        total = int(total_twice_spin)
        if not labels or len(labels) != len(spins) or len(set(labels)) != len(labels):
            raise ValueError("SU2 site labels and local spins must align uniquely.")
        if len(spins) < 2 or any(value < 0 for value in spins) or total < 0:
            raise ValueError(
                "SU2 coupling requires at least two nonnegative doubled spins."
            )
        if not isinstance(resources, SU2SectorResourcePolicy):
            raise TypeError("resources must be SU2SectorResourcePolicy.")
        paths: list[tuple[int, ...]] = []

        def extend(current: int, index: int, path: tuple[int, ...]) -> None:
            if index == len(spins):
                if current == total:
                    paths.append(path)
                return
            for output in su2_fusion(current, spins[index]):
                extend(output, index + 1, path + (output,))

        for first_output in su2_fusion(spins[0], spins[1]):
            extend(first_output, 2, (first_output,))
        paths = sorted(set(paths))
        if not paths:
            raise ValueError("The requested total spin is unreachable.")
        product_dimension = prod(value + 1 for value in spins)
        sector_dimension = len(paths) * (total + 1)
        if product_dimension > resources.maximum_product_dimension:
            raise ValueError("SU2 product dimension exceeds resource admission.")
        if sector_dimension > resources.maximum_sector_dimension:
            raise ValueError("SU2 sector dimension exceeds resource admission.")
        if product_dimension * sector_dimension > resources.maximum_matrix_elements:
            raise ValueError("SU2 coupling transform exceeds matrix resource admission.")
        self.site_labels = labels
        self.local_twice_spins = spins
        self.total_twice_spin = total
        self.coupling_paths = tuple(paths)
        self.resources = resources
        self.product_dimension = product_dimension
        self.multiplicity_dimension = len(paths)
        self.sector_dimension = sector_dimension
        self.plan_id = canonical_fingerprint(
            {
                "kind": "left-associated-su2-coupling-tree-plan",
                "site_labels": labels,
                "local_twice_spins": spins,
                "total_twice_spin": total,
                "coupling_paths": tuple(paths),
                "resources": resources.policy_id,
                "convention": "doubled-spins-condon-shortley-left-associated",
            }
        )


class SU2SectorEvidence(StrictModule):
    orthonormality_residual: Array
    projector_idempotence_residual: Array
    multiplicity_dimension: int = eqx.field(static=True)
    sector_dimension: int = eqx.field(static=True)
    product_dimension: int = eqx.field(static=True)
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class PreparedSU2SectorBasis(StrictModule):
    """Exact Condon–Shortley transform from one coupled sector to product states."""

    plan: SU2CouplingTreePlan = eqx.field(static=True)
    product_coordinates: Array
    coupled_labels: Array
    transform: Array
    evidence: SU2SectorEvidence
    prepared_id: str = eqx.field(static=True)

    def to_product(self, sector_vector: ArrayLike, /) -> Array:
        value = jnp.asarray(sector_vector, dtype=self.transform.dtype)
        if value.shape != (self.plan.sector_dimension,):
            raise ValueError("sector_vector has the wrong dimension.")
        return jnp.conj(self.transform.T) @ value

    def from_product(self, product_vector: ArrayLike, /) -> Array:
        value = jnp.asarray(product_vector, dtype=self.transform.dtype)
        if value.shape != (self.plan.product_dimension,):
            raise ValueError("product_vector has the wrong dimension.")
        return self.transform @ value


def _product_coordinates(spins: tuple[int, ...], /) -> np.ndarray:
    local_dimensions = tuple(value + 1 for value in spins)
    return np.asarray(tuple(np.ndindex(*local_dimensions)), dtype=np.int32)


def prepare_su2_sector_basis(plan: SU2CouplingTreePlan, /) -> PreparedSU2SectorBasis:
    if not isinstance(plan, SU2CouplingTreePlan):
        raise TypeError("plan must be SU2CouplingTreePlan.")
    coordinates = _product_coordinates(plan.local_twice_spins)
    transform = np.zeros(
        (plan.sector_dimension, plan.product_dimension), dtype=np.float64
    )
    coupled_labels = []
    row = 0
    for path_index, path in enumerate(plan.coupling_paths):
        for total_m_index, total_m in enumerate(
            range(-plan.total_twice_spin, plan.total_twice_spin + 1, 2)
        ):
            coupled_labels.append((path_index, total_m_index))
            for product_index, coordinate in enumerate(coordinates):
                magnetic = tuple(
                    -spin + 2 * int(index)
                    for spin, index in zip(
                        plan.local_twice_spins, coordinate, strict=True
                    )
                )
                if sum(magnetic) != total_m:
                    continue
                intermediate_spin = path[0]
                intermediate_m = magnetic[0] + magnetic[1]
                first = su2_clebsch_gordan(
                    plan.local_twice_spins[0],
                    plan.local_twice_spins[1],
                    intermediate_spin,
                )
                amplitude = float(
                    first[
                        int(coordinate[0]),
                        int(coordinate[1]),
                        (intermediate_m + intermediate_spin) // 2,
                    ]
                )
                for site in range(2, len(plan.local_twice_spins)):
                    next_spin = path[site - 1]
                    table = su2_clebsch_gordan(
                        intermediate_spin,
                        plan.local_twice_spins[site],
                        next_spin,
                    )
                    next_m = intermediate_m + magnetic[site]
                    amplitude *= float(
                        table[
                            (intermediate_m + intermediate_spin) // 2,
                            int(coordinate[site]),
                            (next_m + next_spin) // 2,
                        ]
                    )
                    intermediate_spin = next_spin
                    intermediate_m = next_m
                transform[row, product_index] = amplitude
            row += 1
    gram = transform @ transform.T
    projector = transform.T @ transform
    orthonormality = float(
        np.linalg.norm(gram - np.eye(plan.sector_dimension))
        / max(1.0, np.linalg.norm(np.eye(plan.sector_dimension)))
    )
    idempotence = float(
        np.linalg.norm(projector @ projector - projector)
        / max(1.0, np.linalg.norm(projector))
    )
    finite = bool(np.all(np.isfinite(transform)))
    accepted = finite and orthonormality <= 1e-11 and idempotence <= 1e-11
    evidence = SU2SectorEvidence(
        orthonormality_residual=jnp.asarray(orthonormality),
        projector_idempotence_residual=jnp.asarray(idempotence),
        multiplicity_dimension=plan.multiplicity_dimension,
        sector_dimension=plan.sector_dimension,
        product_dimension=plan.product_dimension,
        finite=jnp.asarray(finite),
        accepted=jnp.asarray(accepted),
        plan_id=plan.plan_id,
        claim="finite-su2-coupled-sector-basis-in-condon-shortley-gauge",
    )
    return PreparedSU2SectorBasis(
        plan=plan,
        product_coordinates=jnp.asarray(coordinates),
        coupled_labels=jnp.asarray(coupled_labels, dtype=jnp.int32),
        transform=jnp.asarray(transform),
        evidence=evidence,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-su2-sector-basis",
                "plan": plan.plan_id,
                "transform": array_tree_fingerprint(transform),
            }
        ),
    )


class SU2ProjectedOperatorEvidence(StrictModule):
    commutator_residual: Array
    hermiticity_residual: Array
    finite: Array
    accepted: Array
    basis_id: str = eqx.field(static=True)
    operator_source_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class SU2ProjectedOperator(StrictModule):
    matrix: Array
    evidence: SU2ProjectedOperatorEvidence
    basis: PreparedSU2SectorBasis
    operator_id: str = eqx.field(static=True)

    def mv(self, vector: ArrayLike, /) -> Array:
        value = jnp.asarray(vector, dtype=self.matrix.dtype)
        if value.shape != (self.basis.plan.sector_dimension,):
            raise ValueError("Projected SU2 vector has the wrong dimension.")
        return self.matrix @ value


def project_product_operator_to_su2_sector(
    product_operator: ArrayLike,
    basis: PreparedSU2SectorBasis,
    /,
    *,
    operator_source_id: str,
    hermitian: bool,
    tolerance: float = 1e-10,
) -> SU2ProjectedOperator:
    if not isinstance(basis, PreparedSU2SectorBasis):
        raise TypeError("basis must be PreparedSU2SectorBasis.")
    matrix = np.asarray(product_operator)
    expected = (basis.plan.product_dimension, basis.plan.product_dimension)
    if matrix.shape != expected or not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"product_operator must be one finite matrix of shape {expected}."
        )
    if matrix.size > basis.plan.resources.maximum_matrix_elements:
        raise ValueError("Product operator exceeds matrix resource admission.")
    source = str(operator_source_id)
    if not source:
        raise ValueError("operator_source_id must be non-empty.")
    transform = np.asarray(basis.transform)
    projector = transform.T @ transform
    commutator = matrix @ projector - projector @ matrix
    commutator_residual = float(
        np.linalg.norm(commutator) / max(1.0, float(np.linalg.norm(matrix)))
    )
    if commutator_residual > float(tolerance):
        raise ValueError(
            "Product operator does not preserve the selected total-spin sector."
        )
    reduced = transform @ matrix @ transform.T
    hermiticity = float(
        np.linalg.norm(reduced - np.conj(reduced.T))
        / max(1.0, float(np.linalg.norm(reduced)))
    )
    finite = bool(np.all(np.isfinite(reduced)))
    accepted = (
        finite
        and commutator_residual <= tolerance
        and (not hermitian or hermiticity <= tolerance)
    )
    evidence = SU2ProjectedOperatorEvidence(
        commutator_residual=jnp.asarray(commutator_residual),
        hermiticity_residual=jnp.asarray(hermiticity),
        finite=jnp.asarray(finite),
        accepted=jnp.asarray(accepted),
        basis_id=basis.prepared_id,
        operator_source_id=source,
        claim="finite-dense-product-to-su2-sector-projection-reference",
    )
    return SU2ProjectedOperator(
        matrix=jnp.asarray(reduced),
        evidence=evidence,
        basis=basis,
        operator_id=canonical_fingerprint(
            {
                "kind": "su2-projected-operator",
                "basis": basis.prepared_id,
                "source": source,
                "matrix": array_tree_fingerprint(reduced),
            }
        ),
    )


__all__ = [
    "PreparedSU2SectorBasis",
    "SU2CouplingTreePlan",
    "SU2ProjectedOperator",
    "SU2ProjectedOperatorEvidence",
    "SU2SectorEvidence",
    "SU2SectorResourcePolicy",
    "prepare_su2_sector_basis",
    "project_product_operator_to_su2_sector",
]
