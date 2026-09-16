#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, LinearSubspace
from ...metrix.clifford import FiniteMetricIsometryGroup


class MagneticSymmetryOperation(StrictModule, NonTrainableState):
    """One caller-supplied spatial/magnetic operation.

    ``antiunitary=True`` includes time reversal. Polar vectors transform by R;
    axial spins transform by ``(-1)^theta det(R) R``. Momentum transforms by
    ``(-1)^theta R k``. The external magnetic field is axial and time odd,
    whereas a DMI vector is axial and time even.
    """

    spatial_matrix: Array
    axial_spin_matrix: Array
    momentum_matrix: Array
    magnetic_field_matrix: Array
    dmi_matrix: Array
    site_permutation: Array
    route_permutation: Array
    lattice_wraps: Array
    antiunitary: bool = eqx.field(static=True)
    operation_id: str = eqx.field(static=True)


class MagneticSymmetryRepresentationPlan(StrictModule, NonTrainableState):
    """Magnetic material semantics over an existing validated finite group."""

    group: FiniteMetricIsometryGroup
    operations: tuple[MagneticSymmetryOperation, ...]
    coefficient_representations: Array
    coefficient_count: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        group: FiniteMetricIsometryGroup,
        antiunitary: ArrayLike,
        site_permutations: ArrayLike,
        route_permutations: ArrayLike,
        lattice_wraps: ArrayLike,
        coefficient_representations: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-10,
        rank_tolerance: float = 1.0e-9,
    ):
        if not isinstance(group, FiniteMetricIsometryGroup):
            raise TypeError("group must be FiniteMetricIsometryGroup.")
        if group.algebra.dimension != 3:
            raise ValueError("Magnetic symmetry initially requires a 3D point group.")
        flags = np.asarray(antiunitary, dtype=bool)
        sites = np.asarray(site_permutations)
        routes = np.asarray(route_permutations)
        wraps = np.asarray(lattice_wraps)
        representations = np.asarray(coefficient_representations, dtype=complex)
        order = group.order
        if flags.shape != (order,):
            raise ValueError("antiunitary must have one flag per group operation.")
        if sites.ndim != 2 or sites.shape[0] != order:
            raise ValueError("site_permutations must have shape (group, site).")
        if routes.ndim != 2 or routes.shape[0] != order:
            raise ValueError("route_permutations must have shape (group, route).")
        if wraps.shape != (order, routes.shape[1], 3):
            raise ValueError("lattice_wraps must have shape (group, route, 3).")
        if representations.ndim != 3 or representations.shape[0] != order:
            raise ValueError(
                "coefficient_representations must have shape (group, coeff, coeff)."
            )
        coefficient_count = int(representations.shape[1])
        if coefficient_count < 1 or representations.shape[2] != coefficient_count:
            raise ValueError("Coefficient representations must be nonempty and square.")
        if np.any(~np.isfinite(representations)):
            raise ValueError("Coefficient representations must be finite.")
        tolerance_ = float(tolerance)
        rank_tolerance_ = float(rank_tolerance)
        if (
            not isfinite(tolerance_)
            or tolerance_ <= 0.0
            or not isfinite(rank_tolerance_)
            or rank_tolerance_ <= tolerance_
        ):
            raise ValueError("Magnetic symmetry tolerances are invalid.")
        for name, permutations in (("site", sites), ("route", routes)):
            expected = np.arange(permutations.shape[1])
            if not np.issubdtype(permutations.dtype, np.integer) or any(
                not np.array_equal(np.sort(row), expected) for row in permutations
            ):
                raise ValueError(f"Every {name} action must be a permutation.")
        sites = sites.astype(np.int32, copy=False)
        routes = routes.astype(np.int32, copy=False)
        if not np.issubdtype(wraps.dtype, np.integer):
            raise TypeError("Magnetic symmetry lattice wraps must be integers.")
        wraps = wraps.astype(np.int32, copy=False)
        table = group.multiplication_table
        for left in range(order):
            for right in range(order):
                product = table[left][right]
                if bool(flags[product]) != bool(flags[left] ^ flags[right]):
                    raise ValueError("Antiunitary flags must form a Z2 homomorphism.")
                if not np.array_equal(sites[product], sites[left][sites[right]]):
                    raise ValueError("Supplied magnetic site actions do not compose.")
                if not np.array_equal(routes[product], routes[left][routes[right]]):
                    raise ValueError("Supplied magnetic route actions do not compose.")
                composed_wraps = wraps[right] + wraps[left][routes[right]]
                if not np.array_equal(wraps[product], composed_wraps):
                    raise ValueError(
                        "Supplied canonical-route lattice wraps do not compose."
                    )
                expected_representation = representations[left] @ (
                    np.conj(representations[right])
                    if flags[left]
                    else representations[right]
                )
                if not np.allclose(
                    representations[product],
                    expected_representation,
                    rtol=0.0,
                    atol=tolerance_,
                ):
                    raise ValueError(
                        "Supplied unitary/antiunitary coefficient actions do not compose."
                    )
        identity = group.identity_index
        if flags[identity] or not np.array_equal(
            sites[identity], np.arange(sites.shape[1])
        ):
            raise ValueError("Magnetic identity must be unitary and fix every site.")
        if not np.array_equal(routes[identity], np.arange(routes.shape[1])) or np.any(
            wraps[identity] != 0
        ):
            raise ValueError("Magnetic identity must fix routes and lattice wraps.")
        matrices = np.asarray(group.matrices, dtype=float)
        operations = []
        for index, matrix in enumerate(matrices):
            determinant = float(np.linalg.det(matrix))
            if not np.isclose(abs(determinant), 1.0, atol=tolerance_):
                raise ValueError("Magnetic spatial operations must have determinant ±1.")
            time_sign = -1.0 if flags[index] else 1.0
            operations.append(
                MagneticSymmetryOperation(
                    jnp.asarray(matrix),
                    jnp.asarray(time_sign * determinant * matrix),
                    jnp.asarray(time_sign * matrix),
                    jnp.asarray(time_sign * determinant * matrix),
                    jnp.asarray(determinant * matrix),
                    jnp.asarray(sites[index]),
                    jnp.asarray(routes[index]),
                    jnp.asarray(wraps[index]),
                    bool(flags[index]),
                    canonical_fingerprint(
                        {
                            "kind": "magnetic-symmetry-operation",
                            "group": group.group_id,
                            "index": index,
                            "antiunitary": bool(flags[index]),
                        }
                    ),
                )
            )
        self.group = group
        self.operations = tuple(operations)
        self.coefficient_representations = jnp.asarray(representations)
        self.coefficient_count = coefficient_count
        self.tolerance = tolerance_
        self.rank_tolerance = rank_tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "magnetic-symmetry-representation-plan",
                "group": group.group_id,
                "antiunitary": flags.tolist(),
                "actions": array_tree_fingerprint(
                    {
                        "sites": sites,
                        "routes": routes,
                        "wraps": wraps,
                        "coefficient_representations": representations,
                    }
                ),
                "tolerance": tolerance_,
                "rank_tolerance": rank_tolerance_,
            }
        )


class MagneticSymmetryConstraintCertificate(StrictModule):
    plan: MagneticSymmetryRepresentationPlan
    invariant_subspace: LinearSubspace
    real_constraint_matrix: Array
    singular_values: Array
    cutoff: Array
    singular_gap: Array
    rank: Array
    nullity: Array
    composition_residual: Array
    certificate_id: str = eqx.field(static=True)

    def project(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape != (self.plan.coefficient_count,):
            raise ValueError("Magnetic coefficients have the wrong shape.")
        real = jnp.concatenate((jnp.real(values), jnp.imag(values)))
        basis = self.invariant_subspace.basis
        projected = basis @ (jnp.conj(basis.T) @ real)
        count = self.plan.coefficient_count
        return projected[:count] + 1.0j * projected[count:]

    def invariance_residual(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape != (self.plan.coefficient_count,):
            raise ValueError("Magnetic coefficients have the wrong shape.")
        residuals = []
        for index, operation in enumerate(self.plan.operations):
            source = jnp.conj(values) if operation.antiunitary else values
            residuals.append(
                jnp.max(
                    jnp.abs(
                        self.plan.coefficient_representations[index] @ source - values
                    ),
                    initial=0.0,
                )
            )
        return jnp.max(jnp.stack(residuals), initial=0.0)


def compile_magnetic_symmetry_constraints(
    plan: MagneticSymmetryRepresentationPlan,
    /,
) -> MagneticSymmetryConstraintCertificate:
    """Compile caller operations into a rank-certified real-linear subspace."""

    if not isinstance(plan, MagneticSymmetryRepresentationPlan):
        raise TypeError("plan must be MagneticSymmetryRepresentationPlan.")
    count = plan.coefficient_count
    identity = np.eye(2 * count)
    blocks = []
    representations = np.asarray(plan.coefficient_representations)
    for index, operation in enumerate(plan.operations):
        matrix = representations[index]
        real = matrix.real
        imag = matrix.imag
        block = (
            np.block([[real, imag], [imag, -real]])
            if operation.antiunitary
            else np.block([[real, -imag], [imag, real]])
        )
        blocks.append(block - identity)
    constraints = np.concatenate(blocks, axis=0)
    _, singular_values, vh = np.linalg.svd(constraints, full_matrices=True)
    scale = max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
    cutoff = plan.rank_tolerance * scale
    rank = int(np.count_nonzero(singular_values > cutoff))
    if singular_values.size:
        nearest = np.min(np.abs(singular_values - cutoff))
        if nearest <= plan.tolerance * scale:
            raise ValueError("Magnetic symmetry constraint rank is unresolved at cutoff.")
    nullity = 2 * count - rank
    basis = np.asarray(vh.conj().T[:, rank:], dtype=float)
    subspace = LinearSubspace(
        ArraySpace((2 * count,), dtype=float),
        jnp.asarray(basis),
        dimension=nullity,
        orthonormal=True,
        subspace_id=f"magnetic-invariants:{plan.plan_id}",
    )
    lower = singular_values[rank] if rank < singular_values.size else 0.0
    upper = singular_values[rank - 1] if rank else scale
    gap = float(upper - lower)
    certificate_id = canonical_fingerprint(
        {
            "kind": "magnetic-symmetry-constraint-certificate",
            "plan": plan.plan_id,
            "rank": rank,
            "nullity": nullity,
            "cutoff": cutoff,
        }
    )
    return MagneticSymmetryConstraintCertificate(
        plan,
        subspace,
        jnp.asarray(constraints),
        jnp.asarray(singular_values),
        jnp.asarray(cutoff),
        jnp.asarray(gap),
        jnp.asarray(rank, dtype=jnp.int32),
        jnp.asarray(nullity, dtype=jnp.int32),
        jnp.asarray(0.0),
        certificate_id,
    )


__all__ = [
    "MagneticSymmetryConstraintCertificate",
    "MagneticSymmetryOperation",
    "MagneticSymmetryRepresentationPlan",
    "compile_magnetic_symmetry_constraints",
]
