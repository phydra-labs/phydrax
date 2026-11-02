#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified geometric validity of mapped cells through Bernstein bounds.

The Jacobian determinant of a polynomial geometry map is itself a polynomial of
known degree on the reference cell. Converting it to Bernstein form gives
coefficients whose minimum and maximum enclose the determinant over the whole
cell (convex-hull property), while every corner coefficient equals the value at
that corner. Adaptive depth-first subdivision tightens the enclosure until the
determinant is proven strictly above the declared floor, an exact contact or
corner value decides the result, or no distinct binary64 interior vertex
remains. The last case remains UNRESOLVED.

Rational pyramids are certified through the collapsed coordinates
``(u, v, w) -> (u (1 - w) + w / 2, v (1 - w) + w / 2, w)``: the physical
determinant pulled back to the unit cube is the numerator of the rational
determinant divided by ``(1 - w)^2`` and is a tensor polynomial of degree
``(3k - 1, 3k - 1, 3k - 3)`` whose homogeneous Bernstein coefficients are bounded.
Certification requires a unique exact apex continuation of every mapped
coordinate and positivity of the collapsed-cube Jacobian determinant on the
closed cube, which covers every directional apex limit. A non-parallelogram base
gives a direction-dependent apex Jacobian; such cells remain valid, and
``apex_derivative_unique`` records the distinction because pointwise apex
tabulation refuses them (collapsed quadrature never samples the apex).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from functools import cache
from itertools import product
from typing import Any, assert_never

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import (
    polygon_simplicity_2d,
    PolygonSimplicityStatus,
    PredicateMode,
    PredicateSign,
    resolve_host_predicate_mode,
)
from .._strict import reconstruct_strict, register_strict_reconstruction, StrictModule
from .._trainable import NonTrainableState
from ..geometry._interval_enclosure import interval_divide
from ..linalg import (
    ArraySpace,
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    inverse_small_linear,
    SmallLinearSolvePlan,
)
from ..linalg._exact import determinant_exact, determinant_small_algebra
from ..typing import ConvertibleToArray
from ._cell_complex import PolyhedralConnectivity
from ._cell_geometry import (
    CellGeometrySpec,
    CellVertexGeometryElement,
    ScalarCoordinateElement,
)
from ._cell_mesh import CellMesh
from ._coordinate_enclosure import (
    add,
    affine_map,
    axes,
    bernstein_coefficients,
    BernsteinDomain,
    compose,
    constant,
    coordinate_execution_basis,
    coordinate_execution_map_expressions,
    coordinate_map_expressions,
    coordinate_point_rounding_error,
    coordinate_source_basis,
    CoordinateSourceBank,
    derivative,
    evaluate_polynomial,
    ExactScalar,
    Expression,
    linear_combinations,
    multiply,
    normalize_pyramid_reference_derivative,
    normalize_pyramid_reference_expression,
    Polynomial,
    polynomial_bounds,
    RationalPolynomial,
    scale,
    subtract,
    sum_polynomials,
)
from ._exact_algebraic_geometry import (
    coefficient_endpoint,
    coefficient_payload,
    RadicalElement,
)
from ._reference_cell import reference_cell_topology
from .fem._reference import FiniteElementSpec


class CellValidityStatus(IntEnum):
    CERTIFIED_VALID = 0
    INVALID = 1
    UNRESOLVED = 2


def cell_geometry_id(geometry: CellGeometrySpec, /) -> str:
    """Bind the complete coordinate map and its source/restriction ancestry."""
    restriction = geometry.restriction_source
    return canonical_fingerprint(
        {
            "kind": "cell-geometry",
            "layout": geometry.geometry_layout_id,
            "source_mesh": geometry.source_mesh_id,
            "source_topology": geometry.source_topology_id,
            "source_map": geometry.source_map_id,
            "restriction_source": (
                None if restriction is None else restriction.restriction_source_id
            ),
            "exact_coordinate_bank": (
                None
                if geometry.exact_coordinate_bank is None
                else tuple(
                    tuple(coefficient_payload(value) for value in row)
                    for row in geometry.exact_coordinate_bank
                )
            ),
            "content": array_tree_fingerprint(geometry),
        }
    )


def _small_or_factored_inverse(
    matrix: Array, source_id: str, target_id: str, operator_id: str, /
) -> tuple[Array, Array, Array]:
    """Inverse, signed determinant and solve evidence of square batches."""
    dimension = matrix.shape[-1]
    if dimension <= 4:
        result = inverse_small_linear(SmallLinearSolvePlan(dimension), matrix)
        return result.value, result.determinant, result.successful
    prepared = factorize(
        DenseLinearOperator(
            matrix,
            source=ArraySpace((dimension,), dtype=matrix.dtype, space_id=source_id),
            target=ArraySpace((dimension,), dtype=matrix.dtype, space_id=target_id),
            operator_id=operator_id,
        ),
        FactorizationPolicy("lu"),
    )
    inverse = prepared.materialize_inverse()
    determinant = prepared.determinant_sign() * jnp.exp(prepared.log_abs_determinant())
    return inverse.value, determinant, inverse.successful


def cell_jacobian_inverse_and_measure(
    jacobian: Array, /
) -> tuple[Array, Array, Array, Array, Array]:
    """Invert cell Jacobians ``(..., ambient, reference)`` for consumers.

    Returns the (pseudo-)inverse Jacobian ``(..., reference, ambient)``, the
    inverse metric ``(J^T J)^-1``, the measure ``sqrt(det(J^T J))``, the
    determinant (signed ``det(J)`` for full-dimensional cells, the measure for
    embedded ones) and the solve evidence.

    A full-dimensional cell evaluates ``det(J)`` itself, never the metric
    (Gram) determinant ``det(J^T J) = det(J)^2``: in binary64 the Gram
    determinant's rounding error is a small multiple of ``u prod |c_j|^2``
    (``c_j`` the columns of ``J``), so its sign is lost once the relative
    determinant ``r = |det(J)| / prod |c_j|`` falls below about ``sqrt(u)``,
    while ``det(J)`` keeps its sign down to a small multiple of ``u``. Embedded
    (manifold) cells keep the Gram determinant, whose square root is their
    measure.

    Each column is first scaled to unit length (``J = Jn D``), so the solve's
    conditioning depends only on ``r``, not on the cell's aspect ratio: every
    entry and cofactor of ``Jn``, and of the unit-diagonal ``Jn^T Jn``, is at
    most one in magnitude, hence ``cond_inf(Jn) <= d^2 / r`` and
    ``cond_inf(Jn^T Jn) <= d^2 / r^2`` for reference dimension ``d``. The
    evidence therefore holds for every full-dimensional cell whose relative
    determinant, and every embedded cell whose Gram-relative determinant
    ``r^2``, exceeds ``CELL_VALIDITY_DETERMINANT_FLOOR`` (below).
    """
    ambient = jacobian.shape[-2]
    dimension = jacobian.shape[-1]
    column_norms = jnp.sqrt(jnp.sum(jacobian * jacobian, axis=-2))
    nonzero = jnp.all(column_norms > 0.0, axis=-1)
    safe_norms = jnp.where(column_norms > 0.0, column_norms, 1.0)
    scale = jnp.prod(safe_norms, axis=-1)
    normalized = jacobian / safe_norms[..., None, :]
    if ambient == dimension:
        normalized_inverse, normalized_determinant, successful = (
            _small_or_factored_inverse(
                normalized,
                f"finite-element-reference-tangent:{dimension}",
                f"finite-element-physical-tangent:{dimension}",
                "finite-element-metric-jacobian",
            )
        )
        inverse_jacobian = normalized_inverse / safe_norms[..., :, None]
        determinant = normalized_determinant * scale
        inverse_metric = ein.contract(
            "...id,...jd->...ij", inverse_jacobian, inverse_jacobian
        )
        measure = jnp.abs(determinant)
    else:
        normalized_metric = ein.contract("...di,...dj->...ij", normalized, normalized)
        normalized_inverse_metric, normalized_gram, successful = (
            _small_or_factored_inverse(
                normalized_metric,
                f"finite-element-reference-tangent:{dimension}",
                f"finite-element-reference-tangent:{dimension}",
                f"finite-element-geometry-gram:{dimension}",
            )
        )
        inverse_metric = normalized_inverse_metric / (
            safe_norms[..., :, None] * safe_norms[..., None, :]
        )
        inverse_jacobian = ein.contract("...ij,...dj->...id", inverse_metric, jacobian)
        measure = jnp.sqrt(jnp.maximum(normalized_gram, 0.0)) * scale
        determinant = measure
    return inverse_jacobian, inverse_metric, measure, determinant, successful & nonzero


# CellValidityPolicy's default relative determinant floor, derived from the
# consumers' binary64 evaluation above so that a certified cell is never
# refused by them. The policy measures a full-dimensional map by its relative
# determinant r = |det J| / prod |c_j| and an embedded map by its Gram-relative
# determinant r^2 = det(J^T J) / prod |c_j|^2. The normalized solve refuses
# only a condition number above the small-solve limit M (1e12) or a
# max-entry-scaled determinant at or below its singular tolerance (1e-12).
# Full-dimensional cells (d <= 4): cond <= d^2 / r and the scaled determinant
# is at least r, so r > 16 / M = 1.6e-11 suffices. Embedded cells (reference
# dimension d <= 3): the scaled Gram determinant is r^2 and cond <= d^2 / r^2,
# so r^2 > max(9 / M, 1e-12) = 9e-12 suffices. The floor is the next power of
# two above both, 2^-35 (2.9e-11), far above the binary64 sign bound of det(J)
# (a few tens of u = 2^-53). Both measures are angle measures, independent of
# a cell's aspect ratio, so the floor does not limit thin cells.
_JACOBIAN_SOLVE_PLAN = SmallLinearSolvePlan(4)
CELL_VALIDITY_DETERMINANT_FLOOR = 2.0 ** math.ceil(
    math.log2(
        max(
            4**2 / _JACOBIAN_SOLVE_PLAN.maximum_condition,
            3**2 / _JACOBIAN_SOLVE_PLAN.maximum_condition,
            _JACOBIAN_SOLVE_PLAN.singular_tolerance,
        )
    )
)
# A polygon edge is degenerate when binary64 cannot resolve its direction:
# the computed edge vector carries an absolute error of at most
# 8 u * diameter (two rounded coordinate differences about the vertex
# centroid), so its direction and outward normal, which polygon consumers
# evaluate, have a relative error of 8 u * diameter / length. Requiring that
# to be at most 2^-20 gives length >= 2^23 u * diameter = 2^-30 * diameter
# (k = 2^23). The threshold is aspect-ratio independent of the polygon's
# thickness: a 1e-6-thick layer quad keeps every edge far above it.
POLYGON_EDGE_RESOLUTION = 2.0**-30


class CellValidityPolicy(StrictModule, NonTrainableState):
    """Degeneracy thresholds for Bernstein validity certification.

    ``relative_determinant_floor`` scales each cell's Hadamard bound of the
    evaluated determinant (the Gram determinant of embedded maps) and polygon
    areas and polyhedral star simplices. A determinant must be strictly above
    ``determinant_floor()``: the requested floor, but at least
    ``CELL_VALIDITY_DETERMINANT_FLOOR`` (2^-35, also the default), the bound
    below which finite-element geometry preparation
    (``cell_jacobian_inverse_and_measure``) may refuse a Jacobian, so that no
    certified cell is refused there. The measure is an angle measure,
    independent of a cell's aspect ratio. Polygon edges shorter than
    ``POLYGON_EDGE_RESOLUTION`` times the polygon diameter are degenerate
    whatever the floor. ``relative_planarity_tolerance`` bounds embedded
    polygon vertices' distance from their Newell plane relative to the polygon
    diameter.
    """

    relative_determinant_floor: float = eqx.field(static=True)
    relative_planarity_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative_determinant_floor: float = CELL_VALIDITY_DETERMINANT_FLOOR,
        relative_planarity_tolerance: float = 1.0e-10,
    ) -> None:
        floor = float(relative_determinant_floor)
        planarity = float(relative_planarity_tolerance)
        if not math.isfinite(floor) or floor < 0.0 or floor >= 1.0:
            raise ValueError("relative_determinant_floor must lie in [0, 1).")
        if not math.isfinite(planarity) or planarity < 0.0 or planarity >= 1.0:
            raise ValueError("relative_planarity_tolerance must lie in [0, 1).")
        self.relative_determinant_floor = floor
        self.relative_planarity_tolerance = planarity
        self.policy_id = canonical_fingerprint(
            {
                "kind": "cell-validity-policy",
                "relative_determinant_floor": floor,
                "relative_planarity_tolerance": planarity,
            }
        )

    def determinant_floor(self) -> float:
        """The floor a Jacobian determinant must clear to be certified."""
        return max(self.relative_determinant_floor, CELL_VALIDITY_DETERMINANT_FLOOR)


class CellValidityCertificate(StrictModule, NonTrainableState):
    """Independent exact-source and execution-map proofs of one geometry.

    ``status`` combines both proofs; determinant bounds and ``depth`` describe
    execution coefficients. Source bounds/depth remain separately inspectable.
    ``maximum_coefficient_rounding_error`` is scoped to the coordinate bank and
    has physical-coordinate units. ``maximum_element_coefficient_rounding_error``
    is dimensionless and describes element-basis coefficient shadows. Neither
    quantity estimates whole-map or Jacobian error.

    ``cell_map_digests`` names, per cell, the exact execution and source
    coordinate maps (cell kind and exact coefficients in their stored order)
    those proofs answer: ``cells x 2 x 16`` bytes, all-zero where no map record
    exists. It is not identity-bearing; ``certify_cell_geometry_validity(...,
    parent=...)`` reuses a cell's proof only for an identical digest.
    """

    status: Array
    determinant_lower: Array
    determinant_upper: Array
    depth: Array
    execution_status: Array
    source_status: Array
    source_determinant_lower: Array
    source_determinant_upper: Array
    source_depth: Array
    maximum_coefficient_rounding_error: Array
    maximum_element_coefficient_rounding_error: Array
    apex_derivative_unique: Array
    geometry: CellGeometrySpec
    mesh: CellMesh | None
    policy: CellValidityPolicy
    block_names: tuple[str, ...] = eqx.field(static=True)
    block_offsets: tuple[int, ...] = eqx.field(static=True)
    unsupported_block_names: tuple[str, ...] = eqx.field(static=True)
    certified_valid_count: int = eqx.field(static=True)
    invalid_count: int = eqx.field(static=True)
    unresolved_count: int = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    mesh_id: str | None = eqx.field(static=True)
    topology_id: str | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    cell_map_digests: bytes = eqx.field(static=True)

    def __init__(
        self,
        geometry: CellGeometrySpec | CellMesh,
        /,
        *,
        mesh: CellMesh | None = None,
        policy: CellValidityPolicy | None = None,
    ) -> None:
        """Compute validity; caller-supplied statuses are not proof premises."""
        computed = certify_cell_geometry_validity(geometry, mesh=mesh, policy=policy)
        self.execution_status = computed.execution_status
        self.status = computed.status
        self.determinant_lower = computed.determinant_lower
        self.determinant_upper = computed.determinant_upper
        self.depth = computed.depth
        self.source_status = computed.source_status
        self.source_determinant_lower = computed.source_determinant_lower
        self.source_determinant_upper = computed.source_determinant_upper
        self.source_depth = computed.source_depth
        self.maximum_coefficient_rounding_error = (
            computed.maximum_coefficient_rounding_error
        )
        self.maximum_element_coefficient_rounding_error = (
            computed.maximum_element_coefficient_rounding_error
        )
        self.apex_derivative_unique = computed.apex_derivative_unique
        self.geometry = computed.geometry
        self.mesh = computed.mesh
        self.policy = computed.policy
        self.block_names = computed.block_names
        self.block_offsets = computed.block_offsets
        self.unsupported_block_names = computed.unsupported_block_names
        self.certified_valid_count = computed.certified_valid_count
        self.invalid_count = computed.invalid_count
        self.unresolved_count = computed.unresolved_count
        self.geometry_id = computed.geometry_id
        self.geometry_layout_id = computed.geometry_layout_id
        self.mesh_id = computed.mesh_id
        self.topology_id = computed.topology_id
        self.policy_id = computed.policy_id
        self.certificate_id = computed.certificate_id
        self.cell_map_digests = computed.cell_map_digests

    @classmethod
    def _from_computed(
        cls,
        status: ConvertibleToArray,
        determinant_lower: ConvertibleToArray,
        determinant_upper: ConvertibleToArray,
        depth: ConvertibleToArray,
        /,
        *,
        source_proof: _BlockCertificate,
        maximum_coefficient_rounding_error: float,
        maximum_element_coefficient_rounding_error: float,
        apex_derivative_unique: ConvertibleToArray,
        geometry: CellGeometrySpec,
        mesh: CellMesh | None,
        policy: CellValidityPolicy,
        block_names: tuple[str, ...],
        block_offsets: tuple[int, ...],
        unsupported_block_names: tuple[str, ...],
        geometry_id: str,
        geometry_layout_id: str,
        policy_id: str,
        cell_map_digests: bytes | None = None,
    ) -> CellValidityCertificate:
        execution = _normalize_proof_arrays(
            status, determinant_lower, determinant_upper, depth
        )
        if not isinstance(source_proof, _BlockCertificate):
            raise TypeError("Source validity requires the canonical block proof.")
        source = _normalize_proof_arrays(
            source_proof.status,
            source_proof.lower,
            source_proof.upper,
            source_proof.depth,
        )
        if source.status.shape != execution.status.shape:
            raise ValueError("Source and execution proofs must cover the same cells.")
        apex_unique = np.asarray(apex_derivative_unique)
        if apex_unique.dtype != np.bool_ or apex_unique.shape != execution.status.shape:
            raise ValueError("Apex derivative evidence must be one bool per cell.")
        if source.status is execution.status:
            status_ = execution.status
        else:
            status_ = np.where(
                (source.status == CellValidityStatus.INVALID)
                | (execution.status == CellValidityStatus.INVALID),
                int(CellValidityStatus.INVALID),
                np.where(
                    (source.status == CellValidityStatus.UNRESOLVED)
                    | (execution.status == CellValidityStatus.UNRESOLVED),
                    int(CellValidityStatus.UNRESOLVED),
                    int(CellValidityStatus.CERTIFIED_VALID),
                ),
            ).astype(np.int32)
        lower, upper, depth_ = execution.lower, execution.upper, execution.depth
        rounding_error = float(maximum_coefficient_rounding_error)
        if not math.isfinite(rounding_error) or rounding_error < 0.0:
            raise ValueError(
                "Coordinate coefficient rounding error must be finite and nonnegative."
            )
        element_rounding_error = float(maximum_element_coefficient_rounding_error)
        if not math.isfinite(element_rounding_error) or element_rounding_error < 0.0:
            raise ValueError(
                "Element coefficient rounding error must be finite and nonnegative."
            )
        names = tuple(str(value) for value in block_names)
        offsets = tuple(int(value) for value in block_offsets)
        unsupported = tuple(str(value) for value in unsupported_block_names)
        if not isinstance(geometry, CellGeometrySpec) or not isinstance(
            policy, CellValidityPolicy
        ):
            raise TypeError(
                "Validity certificates retain CellGeometrySpec and CellValidityPolicy inputs."
            )
        if mesh is not None and not isinstance(mesh, CellMesh):
            raise TypeError("Validity certificate mesh must be CellMesh or None.")
        if (
            len(offsets) != len(names) + 1
            or not offsets
            or offsets[0] != 0
            or offsets[-1] != status_.size
            or any(stop < start for start, stop in zip(offsets[:-1], offsets[1:]))
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "Validity block offsets must partition cells in unique block order."
            )
        if not set(unsupported) <= set(names):
            raise ValueError("Unsupported validity blocks must be certificate blocks.")
        digest_bytes = 2 * _MAP_DIGEST_SIZE * status_.size
        if cell_map_digests is None:
            cell_map_digests = bytes(digest_bytes)
        if type(cell_map_digests) is not bytes or len(cell_map_digests) != digest_bytes:
            raise ValueError(
                "Cell map digests must be one execution and one source digest per cell."
            )
        geometry_id_ = cell_geometry_id(geometry)
        if (
            geometry_id_ != geometry_id
            or geometry.geometry_layout_id != geometry_layout_id
            or policy.policy_id != policy_id
        ):
            raise ValueError(
                "Validity certificate identity fields must match retained geometry and policy."
            )
        mesh_id_ = None if mesh is None else mesh.mesh_id
        topology_id_ = None if mesh is None else mesh.topology_id
        if mesh is not None and tuple(block.name for block in mesh.blocks) != names:
            raise ValueError(
                "Validity block names must follow the bound mesh block order."
            )
        execution_status = jnp.asarray(execution.status)
        values = {
            "execution_status": execution_status,
            "status": execution_status
            if status_ is execution.status
            else jnp.asarray(status_),
            "determinant_lower": jnp.asarray(lower),
            "determinant_upper": jnp.asarray(upper),
            "depth": jnp.asarray(depth_),
            "source_status": execution_status
            if source.status is execution.status
            else jnp.asarray(source.status),
            "source_determinant_lower": jnp.asarray(source.lower),
            "source_determinant_upper": jnp.asarray(source.upper),
            "source_depth": jnp.asarray(source.depth),
            "maximum_coefficient_rounding_error": jnp.asarray(
                rounding_error, dtype=jnp.float64
            ),
            "maximum_element_coefficient_rounding_error": jnp.asarray(
                element_rounding_error, dtype=jnp.float64
            ),
            "apex_derivative_unique": jnp.asarray(apex_unique, dtype=jnp.bool_),
            "geometry": geometry,
            "mesh": mesh,
            "policy": policy,
            "block_names": names,
            "block_offsets": offsets,
            "unsupported_block_names": unsupported,
            "certified_valid_count": int(
                np.count_nonzero(status_ == CellValidityStatus.CERTIFIED_VALID)
            ),
            "invalid_count": int(np.count_nonzero(status_ == CellValidityStatus.INVALID)),
            "unresolved_count": int(
                np.count_nonzero(status_ == CellValidityStatus.UNRESOLVED)
            ),
            "geometry_id": geometry_id_,
            "geometry_layout_id": geometry_layout_id,
            "mesh_id": mesh_id_,
            "topology_id": topology_id_,
            "policy_id": policy_id,
            "cell_map_digests": cell_map_digests,
        }
        values["certificate_id"] = canonical_fingerprint(
            {
                "kind": "cell-validity-certificate",
                "geometry": geometry_id_,
                "geometry_layout": geometry_layout_id,
                "mesh": mesh_id_,
                "topology": topology_id_,
                "policy": policy_id,
                "blocks": names,
                "block_offsets": offsets,
                "unsupported": unsupported,
                "determinant_lower": array_tree_fingerprint(lower),
                "determinant_upper": array_tree_fingerprint(upper),
                "status": array_tree_fingerprint(status_),
                "depth": array_tree_fingerprint(depth_),
                "execution_status": array_tree_fingerprint(execution.status),
                "source_status": array_tree_fingerprint(source.status),
                "source_determinant_lower": array_tree_fingerprint(source.lower),
                "source_determinant_upper": array_tree_fingerprint(source.upper),
                "source_depth": array_tree_fingerprint(source.depth),
                "maximum_coefficient_rounding_error": rounding_error,
                "maximum_element_coefficient_rounding_error": element_rounding_error,
            }
        )
        return reconstruct_strict(cls, values)

    @classmethod
    def from_stored_fields(
        cls,
        value: CellValidityCertificate,
        mesh: CellMesh | None,
        geometry: CellGeometrySpec,
        /,
        *,
        policy: CellValidityPolicy,
    ) -> CellValidityCertificate:
        """Recompute both proofs instead of trusting stored statuses or digests."""
        if type(value) is not cls:
            raise TypeError("value must be CellValidityCertificate.")
        if not isinstance(geometry, CellGeometrySpec):
            raise TypeError("geometry must be the renewed CellGeometrySpec.")
        if mesh is not None and not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be CellMesh or None.")
        if not isinstance(policy, CellValidityPolicy):
            raise TypeError("policy must be CellValidityPolicy.")
        rebuilt = certify_cell_geometry_validity(geometry, mesh=mesh, policy=policy)
        from ..geometry._mesh_certificates import _same_tree_content

        if not _same_tree_content(value, rebuilt):
            raise ValueError(
                "Stored cell validity differs from recomputed source and execution proofs."
            )
        return rebuilt

    @property
    def all_certified(self) -> bool:
        return self.certified_valid_count == self.status.shape[0]

    def stage_summary(self) -> dict[str, dict[int, int]]:
        """Cells decided per proof stage, derived from the recorded depths.

        Not identity-bearing: it is computed from fields the certificate id
        already binds. For the execution and source proofs, ``{depth: count}``
        counts decided (certified-valid or invalid) cells by depth: 0 is the
        exact affine determinant or the first Bernstein control-net screen, and
        ``k`` is ``k`` Bernstein subdivision levels. Unresolved cells are counted
        under key ``-1``.
        """
        summary: dict[str, dict[int, int]] = {}
        for name, status, depth in (
            ("execution", self.execution_status, self.depth),
            ("source", self.source_status, self.source_depth),
        ):
            status_ = np.asarray(status)
            depth_ = np.where(
                status_ == CellValidityStatus.UNRESOLVED, -1, np.asarray(depth)
            )
            levels, counts = np.unique(depth_, return_counts=True)
            summary[name] = {
                int(level): int(count)
                for level, count in zip(levels.tolist(), counts.tolist(), strict=True)
            }
        return summary

    def require_bound(
        self,
        geometry: CellGeometrySpec,
        /,
        *,
        mesh: CellMesh | None = None,
    ) -> None:
        from ..geometry._mesh_certificates import _same_tree_content

        if (
            not _same_tree_content(geometry, self.geometry)
            or cell_geometry_id(geometry) != self.geometry_id
            or geometry.geometry_layout_id != self.geometry_layout_id
            or (mesh is None) != (self.mesh is None)
            or (
                mesh is not None
                and (
                    not _same_tree_content(mesh, self.mesh)
                    or mesh.mesh_id != self.mesh_id
                    or mesh.topology_id != self.topology_id
                )
            )
        ):
            raise ValueError(
                "Validity certificate must be bound to these exact scientific inputs."
            )

    def require_current(
        self,
        geometry: CellGeometrySpec,
        /,
        *,
        mesh: CellMesh | None = None,
    ) -> None:
        """Require this computed proof to answer ``geometry`` (and ``mesh``).

        Nothing is recomputed: the operands must be the content this proof was
        computed from (:meth:`require_bound`) and the stored counts must be the
        ones its status array determines. :meth:`revalidate` recomputes both
        proofs.
        """
        self.require_bound(geometry, mesh=mesh)
        status = np.asarray(self.status)
        if (
            self.policy.policy_id != self.policy_id
            or int(np.count_nonzero(status == CellValidityStatus.CERTIFIED_VALID))
            != self.certified_valid_count
            or int(np.count_nonzero(status == CellValidityStatus.INVALID))
            != self.invalid_count
            or int(np.count_nonzero(status == CellValidityStatus.UNRESOLVED))
            != self.unresolved_count
        ):
            raise ValueError(
                "Validity certificate claims differ from its stored statuses or policy."
            )

    def revalidate(
        self,
        geometry: CellGeometrySpec,
        /,
        *,
        mesh: CellMesh | None = None,
    ) -> None:
        """Recompute both proofs; stored identities are not proof authority."""
        self.require_current(geometry, mesh=mesh)
        renewed_policy = CellValidityPolicy(
            relative_determinant_floor=self.policy.relative_determinant_floor,
            relative_planarity_tolerance=self.policy.relative_planarity_tolerance,
        )
        if renewed_policy.policy_id != self.policy_id:
            raise ValueError(
                "Validity certificate policy does not reproduce its controls."
            )
        fresh = certify_cell_geometry_validity(geometry, mesh=mesh, policy=self.policy)
        if (
            fresh.certificate_id != self.certificate_id
            or fresh.cell_map_digests != self.cell_map_digests
            or fresh.block_names != self.block_names
            or fresh.block_offsets != self.block_offsets
            or fresh.unsupported_block_names != self.unsupported_block_names
            or fresh.certified_valid_count != self.certified_valid_count
            or fresh.invalid_count != self.invalid_count
            or fresh.unresolved_count != self.unresolved_count
            or any(
                np.asarray(getattr(fresh, name)).dtype
                != np.asarray(getattr(self, name)).dtype
                or not np.array_equal(
                    np.asarray(getattr(fresh, name)), np.asarray(getattr(self, name))
                )
                for name in (
                    "status",
                    "execution_status",
                    "determinant_lower",
                    "determinant_upper",
                    "depth",
                    "source_status",
                    "source_determinant_lower",
                    "source_determinant_upper",
                    "source_depth",
                    "maximum_coefficient_rounding_error",
                    "maximum_element_coefficient_rounding_error",
                    "apex_derivative_unique",
                )
            )
        ):
            raise ValueError(
                "Validity certificate does not reproduce the current mathematical proof."
            )


register_strict_reconstruction(CellValidityCertificate)


_EPSILON = float(np.finfo(np.float64).eps)


# Bernstein parameter domains ------------------------------------------------


def _bernstein_1d(degree: int, points: np.ndarray, /) -> np.ndarray:
    powers = np.arange(degree + 1)
    binomial = np.asarray([math.comb(degree, value) for value in powers], np.float64)
    t = points[..., None]
    return binomial * t**powers * (1.0 - t) ** (degree - powers)


def _simplex_indices(degree: int, dimension: int, /) -> np.ndarray:
    return np.asarray(
        [
            index
            for index in product(range(degree + 1), repeat=dimension + 1)
            if sum(index) == degree
        ],
        dtype=np.int64,
    ).reshape(-1, dimension + 1)


def _bernstein_simplex(degree: int, dimension: int, points: np.ndarray, /) -> np.ndarray:
    indices = _simplex_indices(degree, dimension)
    barycentric = np.concatenate(
        (1.0 - np.sum(points, axis=-1, keepdims=True), points), axis=-1
    )
    multinomial = np.asarray(
        [
            math.factorial(degree) / math.prod(math.factorial(value) for value in row)
            for row in indices
        ],
        dtype=np.float64,
    )
    return multinomial * np.prod(
        barycentric[..., None, :] ** indices[None, :, :], axis=-1
    )


def _chebyshev_nodes(degree: int, /) -> np.ndarray:
    index = np.arange(degree + 1, dtype=np.float64)
    return 0.5 * (1.0 - np.cos((2.0 * index + 1.0) * np.pi / (2.0 * degree + 2.0)))


def _simplex_nodes(degree: int, dimension: int, /) -> np.ndarray:
    if degree == 0:
        return np.full((1, dimension), 1.0 / (dimension + 1.0))
    return _simplex_indices(degree, dimension)[:, 1:].astype(np.float64) / degree


def _tensor(values: tuple[np.ndarray, ...], /) -> np.ndarray:
    """Tensor product of per-axis basis tables sharing the leading point axis."""

    result = values[0]
    for value in values[1:]:
        result = (result[..., :, None] * value[..., None, :]).reshape(
            value.shape[:-1] + (-1,)
        )
    return result


def _grid(axes: tuple[np.ndarray, ...], /) -> np.ndarray:
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([value.reshape(-1) for value in mesh], axis=-1)


_INTERVAL_CHILDREN = ((0.0, 0.5), (0.5, 0.5))
_TRIANGLE_CHILDREN = (
    ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5)),
    ((0.5, 0.0), (1.0, 0.0), (0.5, 0.5)),
    ((0.0, 0.5), (0.5, 0.5), (0.0, 1.0)),
    ((0.5, 0.5), (0.0, 0.5), (0.5, 0.0)),
)


def _tetrahedron_children() -> tuple[np.ndarray, ...]:
    vertices = np.eye(4, 3, k=-1)
    middle = {
        (first, second): 0.5 * (vertices[first] + vertices[second])
        for first in range(4)
        for second in range(first + 1, 4)
    }
    corners = (
        (vertices[0], middle[0, 1], middle[0, 2], middle[0, 3]),
        (middle[0, 1], vertices[1], middle[1, 2], middle[1, 3]),
        (middle[0, 2], middle[1, 2], vertices[2], middle[2, 3]),
        (middle[0, 3], middle[1, 3], middle[2, 3], vertices[3]),
    )
    # Red refinement: the inner octahedron is split around diagonal m02-m13,
    # whose equatorial cycle is m01, m03, m23, m12.
    cycle = (middle[0, 1], middle[0, 3], middle[2, 3], middle[1, 2])
    inner = tuple(
        (middle[0, 2], middle[1, 3], cycle[index], cycle[(index + 1) % 4])
        for index in range(4)
    )
    return tuple(np.asarray(value) for value in (*corners, *inner))


def _simplex_child_maps(dimension: int, /) -> tuple[np.ndarray, np.ndarray]:
    if dimension == 2:
        children = tuple(np.asarray(value) for value in _TRIANGLE_CHILDREN)
    elif dimension == 3:
        children = _tetrahedron_children()
    elif dimension > 0:
        vertices = np.eye(dimension + 1, dimension, k=-1, dtype=np.float64)
        midpoint = 0.5 * (vertices[0] + vertices[1])
        first = vertices.copy()
        second = vertices.copy()
        first[1] = midpoint
        second[0] = midpoint
        children = (first, second)
    else:
        raise ValueError("Simplex subdivision requires positive dimension.")
    origins = np.stack([child[0] for child in children])
    matrices = np.stack([(child[1:] - child[0]).T for child in children])
    return origins, matrices


def _box_child_maps(dimension: int, /) -> tuple[np.ndarray, np.ndarray]:
    choices = tuple(product(_INTERVAL_CHILDREN, repeat=dimension))
    origins = np.asarray([[origin for origin, _ in choice] for choice in choices])
    matrices = np.stack(
        [np.diag([scale for _, scale in choice]) for choice in choices]
    ).reshape(-1, dimension, dimension)
    return origins, matrices


def _prism_child_maps() -> tuple[np.ndarray, np.ndarray]:
    triangle_origins, triangle_matrices = _simplex_child_maps(2)
    origins = []
    matrices = []
    for origin, matrix in zip(triangle_origins, triangle_matrices, strict=True):
        for start, axial_scale in _INTERVAL_CHILDREN:
            origins.append(np.concatenate((origin, (start,))))
            block = np.zeros((3, 3))
            block[:2, :2] = matrix
            block[2, 2] = axial_scale
            matrices.append(block)
    return np.asarray(origins), np.asarray(matrices)


@dataclass(frozen=True)
class _ReferenceSubdivision:
    """Immutable reference-cell subdivisions, independent of polynomial degree."""

    vertices: tuple[tuple[Fraction, ...], ...]
    origins: NDArray[np.float64]
    matrices: NDArray[np.float64]
    exact_origins: tuple[tuple[Fraction, ...], ...]
    exact_matrices: tuple[tuple[tuple[Fraction, ...], ...], ...]


@dataclass(frozen=True)
class _BernsteinPlan:
    """Host-only interpolation and subdivision tables of one parameter domain."""

    domain: BernsteinDomain
    nodes: np.ndarray
    coefficients_from_values: np.ndarray
    conversion_norm: float
    vertex_indices: np.ndarray
    subdivision: _ReferenceSubdivision


def _exact_domain_vertices(
    domain: BernsteinDomain, dimension: int, /
) -> tuple[tuple[Fraction, ...], ...]:
    match domain:
        case "box":
            return tuple(
                tuple(Fraction(value) for value in point)
                for point in product((0, 1), repeat=dimension)
            )
        case "simplex":
            return (
                tuple(Fraction(0) for _ in range(dimension)),
                *(
                    tuple(Fraction(int(axis == vertex)) for axis in range(dimension))
                    for vertex in range(dimension)
                ),
            )
        case "prism":
            return tuple(
                (Fraction(u), Fraction(v), Fraction(w))
                for u, v in ((0, 0), (1, 0), (0, 1))
                for w in (0, 1)
            )
        case _:
            raise ValueError(f"Unknown Bernstein parameter domain {domain!r}.")


@cache
def _reference_subdivision(
    domain: BernsteinDomain, dimension: int, /
) -> _ReferenceSubdivision:
    match domain:
        case "box":
            origins, matrices = _box_child_maps(dimension)
        case "simplex":
            origins, matrices = _simplex_child_maps(dimension)
        case "prism":
            if dimension != 3:
                raise ValueError("Prism subdivisions require three reference dimensions.")
            origins, matrices = _prism_child_maps()
        case _:
            assert_never(domain)
    origins.setflags(write=False)
    matrices.setflags(write=False)
    return _ReferenceSubdivision(
        _exact_domain_vertices(domain, dimension),
        origins,
        matrices,
        tuple(tuple(Fraction(float(value)) for value in row) for row in origins),
        tuple(
            tuple(tuple(Fraction(float(value)) for value in row) for row in child)
            for child in matrices
        ),
    )


def _bernstein_plan(
    domain: BernsteinDomain, degrees: tuple[int, ...], /
) -> _BernsteinPlan:
    match domain:
        case "box":
            axes = tuple(_chebyshev_nodes(degree) for degree in degrees)
            nodes = _grid(axes)
            basis = _tensor(
                tuple(
                    _bernstein_1d(degree, nodes[:, axis])
                    for axis, degree in enumerate(degrees)
                )
            )
            vertex_indices = np.ravel_multi_index(
                tuple(
                    np.asarray(values)
                    for values in zip(*product(*((0, degree) for degree in degrees)))
                ),
                tuple(degree + 1 for degree in degrees),
            )
        case "simplex":
            degree, dimension = degrees
            nodes = _simplex_nodes(degree, dimension)
            basis = _bernstein_simplex(degree, dimension, nodes)
            indices = _simplex_indices(degree, dimension)
            vertex_indices = np.flatnonzero(np.max(indices, axis=1) == degree)
        case "prism":
            triangle_degree, axial_degree = degrees
            triangle_nodes = _simplex_nodes(triangle_degree, 2)
            axial_nodes = _chebyshev_nodes(axial_degree)
            nodes = np.concatenate(
                (
                    np.repeat(triangle_nodes, axial_nodes.size, axis=0),
                    np.tile(axial_nodes, triangle_nodes.shape[0])[:, None],
                ),
                axis=1,
            )
            basis = _tensor(
                (
                    _bernstein_simplex(triangle_degree, 2, nodes[:, :2]),
                    _bernstein_1d(axial_degree, nodes[:, 2]),
                )
            )
            triangle_vertices = np.flatnonzero(
                np.max(_simplex_indices(triangle_degree, 2), axis=1) == triangle_degree
            )
            axial_vertices = np.unique(np.asarray((0, axial_degree)))
            vertex_indices = (
                triangle_vertices[:, None] * (axial_degree + 1) + axial_vertices[None, :]
            ).reshape(-1)
        case _:
            raise ValueError(f"Unknown Bernstein parameter domain {domain!r}.")
    # The conversion operator is applied to every piece of every level; it is
    # prepared once per (domain, degree) like other reference-element tables.
    conversion = np.linalg.solve(basis, np.eye(basis.shape[0]))
    vertex_indices = np.unique(vertex_indices)
    return _BernsteinPlan(
        domain,
        nodes,
        conversion,
        float(np.max(np.sum(np.abs(conversion), axis=1))),
        vertex_indices,
        _reference_subdivision(domain, nodes.shape[1]),
    )


def _determinant_route(
    cell_kind: str, degree: int, embedded: bool, /
) -> tuple[BernsteinDomain, tuple[int, ...]]:
    """Return the Bernstein domain and exact degree bound of the determinant."""

    scale = 2 if embedded else 1
    match cell_kind:
        case "interval":
            return "box", (scale * (degree - 1),)
        case "triangle":
            return "simplex", (scale * 2 * (degree - 1), 2)
        case "tetrahedron":
            return "simplex", (3 * (degree - 1), 3)
        case "quadrilateral":
            return "box", (scale * (2 * degree - 1),) * 2
        case "hexahedron":
            return "box", (3 * degree - 1,) * 3
        case "prism":
            return "prism", (3 * degree - 2, 3 * degree - 1)
        case "pyramid":
            return "box", (3 * degree - 1, 3 * degree - 1, 3 * degree - 3)
        case _:
            raise ValueError(f"No polynomial determinant route for {cell_kind!r}.")


def _reference_points(cell_kind: str, parameters: np.ndarray, /) -> np.ndarray:
    if cell_kind != "pyramid":
        return parameters
    height = parameters[..., 2:3]
    scale = 1.0 - height
    return np.concatenate(
        (
            parameters[..., :2] * scale + 0.5 * height,
            height,
        ),
        axis=-1,
    )


# Determinants ------------------------------------------------------------------


def _square_determinant(matrix: np.ndarray, /) -> np.ndarray:
    size = matrix.shape[-1]
    if size == 1:
        return matrix[..., 0, 0]
    if size == 2:
        return (
            matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]
        )
    return (
        matrix[..., 0, 0]
        * (matrix[..., 1, 1] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 1])
        - matrix[..., 0, 1]
        * (matrix[..., 1, 0] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 0])
        + matrix[..., 0, 2]
        * (matrix[..., 1, 0] * matrix[..., 2, 1] - matrix[..., 1, 1] * matrix[..., 2, 0])
    )


def _jacobian_determinant(
    jacobian: np.ndarray, magnitude: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Return the determinant and its Hadamard scale for (..., ambient, reference)."""

    column_scale = np.prod(np.linalg.norm(magnitude, axis=-2), axis=-1)
    if jacobian.shape[-2] == jacobian.shape[-1]:
        return _square_determinant(jacobian), column_scale
    gram = np.swapaxes(jacobian, -1, -2) @ jacobian
    return _square_determinant(gram), column_scale * column_scale


# Adaptive certification ----------------------------------------------------------


@dataclass(frozen=True)
class _BlockCertificate:
    status: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    depth: np.ndarray
    apex_derivative_unique: np.ndarray | None = None

    def apex_flags(self) -> np.ndarray:
        """Per-cell apex-derivative uniqueness; non-pyramid proofs are unique."""
        if self.apex_derivative_unique is None:
            return np.ones(self.status.shape, dtype=np.bool_)
        return self.apex_derivative_unique


def _normalize_proof_arrays(
    status: ConvertibleToArray,
    lower: ConvertibleToArray,
    upper: ConvertibleToArray,
    depth: ConvertibleToArray,
    /,
) -> _BlockCertificate:
    statuses = np.asarray(status, dtype=np.int32)
    lowers = np.asarray(lower, dtype=np.float64)
    uppers = np.asarray(upper, dtype=np.float64)
    depths = np.asarray(depth, dtype=np.int64)
    if statuses.ndim != 1 or any(
        value.shape != statuses.shape for value in (lowers, uppers, depths)
    ):
        raise ValueError("Validity certificate arrays must be aligned vectors.")
    if np.any(depths < 0):
        raise ValueError("Validity subdivision depths must be nonnegative.")
    if not np.all(np.isin(statuses, tuple(int(value) for value in CellValidityStatus))):
        raise ValueError("Validity status values must be CellValidityStatus codes.")
    return _BlockCertificate(statuses, lowers, uppers, depths)


def _evaluate_pieces(
    element: Any,
    cell_kind: str,
    plan: _BernsteinPlan,
    local: np.ndarray,
    cells: np.ndarray,
    origins: np.ndarray,
    matrices: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Bernstein coefficients and rounding margins for every piece."""

    node_count = plan.nodes.shape[0]
    dof_count = local.shape[1]
    if cells.size == 0:
        return (
            np.empty((0, plan.coefficients_from_values.shape[0]), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    parameters = origins[:, None, :] + ein.contract("pij,mj->pmi", matrices, plan.nodes)
    reference = _reference_points(cell_kind, parameters)
    _, gradients = element.tabulate(reference.reshape(-1, reference.shape[-1]))
    gradients = np.asarray(gradients, dtype=np.float64).reshape(
        (cells.size, node_count, dof_count, reference.shape[-1])
    )
    points = local[cells]
    jacobian = ein.contract("pmnk,pna->pmak", gradients, points)
    magnitude = ein.contract("pmnk,pna->pmak", np.abs(gradients), np.abs(points))
    values, scale = _jacobian_determinant(jacobian, magnitude)
    coefficient = ein.contract("rm,pm->pr", plan.coefficients_from_values, values)
    evaluation_error = 8.0 * (dof_count + reference.shape[-1] + 1) * _EPSILON
    margin = (
        2.0
        * plan.conversion_norm
        * (
            evaluation_error * np.max(scale, axis=1)
            + node_count * _EPSILON * np.max(np.abs(values), axis=1)
        )
    )
    return coefficient, margin


def _root_scale(
    element: Any, cell_kind: str, plan: _BernsteinPlan, local: np.ndarray, /
) -> np.ndarray:
    reference = _reference_points(cell_kind, plan.nodes)
    _, gradients = element.tabulate(reference)
    gradients = np.asarray(gradients, dtype=np.float64)
    magnitude = ein.contract("mnk,cna->cmak", np.abs(gradients), np.abs(local))
    column_scale = np.prod(np.linalg.norm(magnitude, axis=-2), axis=-1)
    embedded = local.shape[-1] != reference.shape[-1]
    return np.max(column_scale * column_scale if embedded else column_scale, axis=1)


def _child_vertices(
    origins: np.ndarray,
    matrices: np.ndarray,
    vertices: np.ndarray,
    /,
) -> np.ndarray:
    return origins[:, None, :] + ein.contract("cij,vj->cvi", matrices, vertices)


def _affine_determinant(
    jacobian: tuple[tuple[ExactScalar, ...], ...], dimension: int, /
) -> ExactScalar:
    ambient = len(jacobian)
    if ambient < dimension:
        raise ValueError(
            "Coordinate maps require at least as many physical as reference axes."
        )
    if ambient == dimension:
        return determinant_exact(jacobian)
    if ambient == 3 and dimension == 2:
        # Cauchy–Binet: det(JᵀJ) is the sum of squared 2×2 row minors.
        # This is the same exact Gram determinant, without constructing its
        # symmetric matrix and multiplying its already-reduced entries.
        minors = (
            jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0],
            jacobian[0][0] * jacobian[2][1] - jacobian[0][1] * jacobian[2][0],
            jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0],
        )
        return sum((value * value for value in minors), Fraction(0))
    gram = tuple(
        tuple(
            sum(
                (jacobian[axis][left] * jacobian[axis][right] for axis in range(ambient)),
                Fraction(0),
            )
            for right in range(dimension)
        )
        for left in range(dimension)
    )
    return determinant_exact(gram)


def _affine_quality(
    jacobian: tuple[tuple[ExactScalar, ...], ...],
    dimension: int,
    policy: CellValidityPolicy,
    /,
) -> tuple[CellValidityStatus, ExactScalar]:
    determinant = _affine_determinant(jacobian, dimension)
    squared_scale: ExactScalar = Fraction(1)
    for column in range(dimension):
        squared_scale *= sum(
            (values[column] * values[column] for values in jacobian), Fraction(0)
        )
    floor = Fraction(policy.determinant_floor())
    margin = (
        determinant - floor * squared_scale
        if len(jacobian) != dimension
        else determinant * determinant - floor * floor * squared_scale
    )
    status = (
        CellValidityStatus.INVALID
        if determinant <= 0 or margin <= 0
        else CellValidityStatus.CERTIFIED_VALID
    )
    return status, determinant


def _exact_affine_root(
    element: FiniteElementSpec,
    original: NDArray[np.float64],
    policy: CellValidityPolicy,
    /,
) -> tuple[CellValidityStatus, float, float] | None:
    affine = affine_map(
        coordinate_map_expressions(element, original), element.topological_dimension
    )
    if affine is None:
        return None
    status, determinant = _affine_quality(
        affine[1], element.topological_dimension, policy
    )
    return (
        status,
        coefficient_endpoint(determinant, upper=False, saturate=True),
        coefficient_endpoint(determinant, upper=True, saturate=True),
    )


def _exact_map_jacobian(
    expressions: tuple[Expression, ...], dimension: int, /
) -> tuple[tuple[Expression, ...], ...]:
    return tuple(
        tuple(derivative(expression, axis) for axis in range(dimension))
        for expression in expressions
    )


def _exact_map_determinant(
    jacobian: tuple[tuple[Expression, ...], ...], dimension: int, /
) -> Expression:
    if len(jacobian) < dimension:
        raise ValueError(
            "Coordinate maps require at least as many physical as reference axes."
        )
    if len(jacobian) == dimension:
        return determinant_small_algebra(
            jacobian, add=add, subtract=subtract, multiply=multiply
        )
    gram = tuple(
        tuple(
            sum_polynomials(
                tuple(multiply(row[left], row[right]) for row in jacobian),
                dimension,
            )
            for right in range(dimension)
        )
        for left in range(dimension)
    )
    return determinant_small_algebra(gram, add=add, subtract=subtract, multiply=multiply)


def _exact_hadamard_squared(
    jacobian: tuple[tuple[Expression, ...], ...], dimension: int, /
) -> Expression:
    bound: Expression = constant(1, dimension)
    for column in range(dimension):
        squared_norm = sum_polynomials(
            tuple(multiply(row[column], row[column]) for row in jacobian),
            dimension,
        )
        bound = multiply(bound, squared_norm)
    return bound


def _hadamard_margin_from_bounds(
    jacobian: tuple[tuple[Expression, ...], ...],
    domain: BernsteinDomain,
    dimension: int,
    determinant_lower: float,
    floor: Fraction,
    /,
) -> bool:
    """Prove a sufficient relative margin before expanding norm products."""
    if not math.isfinite(determinant_lower) or determinant_lower <= 0.0:
        return False
    norm_product_upper = Fraction(1)
    for column in range(dimension):
        norm_squared_upper = Fraction(0)
        for row in jacobian:
            lower, upper = polynomial_bounds(row[column], domain, dimension)
            if not math.isfinite(lower) or not math.isfinite(upper):
                return False
            magnitude = Fraction(max(abs(lower), abs(upper)))
            norm_squared_upper += magnitude * magnitude
        norm_product_upper *= norm_squared_upper
    determinant_bound = Fraction(determinant_lower)
    if len(jacobian) == dimension:
        return determinant_bound * determinant_bound > (
            floor * floor * norm_product_upper
        )
    return determinant_bound > floor * norm_product_upper


def _exact_polynomial_data(
    basis: tuple[Expression, ...],
    local: NDArray[np.float64],
    dimension: int,
    /,
) -> tuple[tuple[Expression, ...], Polynomial] | None:
    coefficients = tuple(
        tuple(Fraction(float(value)) for value in local[:, axis])
        for axis in range(local.shape[1])
    )
    expressions = linear_combinations(basis, coefficients)
    coordinates = tuple(
        expression for expression in expressions if isinstance(expression, dict)
    )
    if len(coordinates) != len(expressions):
        return None
    determinant = _exact_map_determinant(
        _exact_map_jacobian(expressions, dimension), dimension
    )
    if not isinstance(determinant, dict) or any(
        not isinstance(value, Fraction) for value in determinant.values()
    ):
        return None
    return expressions, determinant


def _exact_affine_arguments(
    origin: tuple[Fraction, ...],
    matrix: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[Polynomial, ...]:
    dimension = len(origin)
    variables = axes(dimension)
    return tuple(
        add(
            constant(origin[row], dimension),
            sum_polynomials(
                tuple(
                    scale(variables[column], matrix[row][column])
                    for column in range(dimension)
                ),
                dimension,
            ),
        )
        for row in range(dimension)
    )


def _exact_affine_image(
    origin: tuple[Fraction, ...],
    matrix: tuple[tuple[Fraction, ...], ...],
    point: tuple[Fraction, ...],
    /,
) -> tuple[Fraction, ...]:
    return tuple(
        origin[row]
        + sum(
            (matrix[row][column] * point[column] for column in range(len(point))),
            Fraction(0),
        )
        for row in range(len(origin))
    )


def _exact_child_affine(
    origin: tuple[Fraction, ...],
    matrix: tuple[tuple[Fraction, ...], ...],
    child_origin: tuple[Fraction, ...],
    child_matrix: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[tuple[Fraction, ...], tuple[tuple[Fraction, ...], ...]]:
    return (
        _exact_affine_image(origin, matrix, child_origin),
        tuple(
            tuple(
                sum(
                    (
                        matrix[row][inner] * child_matrix[inner][column]
                        for inner in range(len(origin))
                    ),
                    Fraction(0),
                )
                for column in range(len(origin))
            )
            for row in range(len(origin))
        ),
    )


def _bernstein_enclosure(controls: tuple[ExactScalar, ...], /) -> tuple[float, float]:
    return (
        coefficient_endpoint(min(controls), upper=False, saturate=True),
        coefficient_endpoint(max(controls), upper=True, saturate=True),
    )


def _polynomial_support_projection(
    polynomial: Polynomial,
    domain: BernsteinDomain,
    dimension: int,
    /,
) -> tuple[Polynomial, BernsteinDomain, int]:
    if dimension < 1 or any(len(exponent) != dimension for exponent in polynomial):
        raise ValueError("Bernstein dimension must match polynomial monomial dimensions.")
    active = tuple(
        axis
        for axis in range(dimension)
        if any(exponent[axis] for exponent in polynomial)
    )
    match domain:
        case "box":
            projected_domain: BernsteinDomain = "box"
        case "simplex":
            projected_domain = "box" if len(active) == 1 else "simplex"
        case "prism":
            if dimension != 3:
                raise ValueError("Prism Bernstein controls require three dimensions.")
            projected_domain = (
                "prism" if len(active) == 3 else "simplex" if active == (0, 1) else "box"
            )
        case _:
            assert_never(domain)
    if not active or len(active) == dimension:
        return polynomial, projected_domain, len(active)
    return (
        {
            tuple(exponent[axis] for axis in active): coefficient
            for exponent, coefficient in polynomial.items()
        },
        projected_domain,
        len(active),
    )


def _certify_exact_polynomial_cell(
    determinant: Polynomial,
    domain: BernsteinDomain,
    dimension: int,
    threshold: float,
    /,
) -> tuple[CellValidityStatus, float, float, int]:
    determinant, domain, dimension = _polynomial_support_projection(
        determinant, domain, dimension
    )
    threshold_exact = Fraction(threshold)
    if dimension == 0:
        value = next(iter(determinant.values()), Fraction(0))
        lower, upper = _bernstein_enclosure((value,))
        status = (
            CellValidityStatus.INVALID
            if value <= 0 or value < threshold_exact
            else CellValidityStatus.UNRESOLVED
            if value == threshold_exact
            else CellValidityStatus.CERTIFIED_VALID
        )
        return status, lower, upper, 0
    subdivision = _reference_subdivision(domain, dimension)
    root_controls = bernstein_coefficients(determinant, domain, dimension)
    root_lower, root_upper = _bernstein_enclosure(root_controls)
    vertices = subdivision.vertices
    child_origins = subdivision.exact_origins
    child_matrices = subdivision.exact_matrices
    zero = tuple(Fraction(0) for _ in range(dimension))
    identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(dimension))
        for row in range(dimension)
    )
    stack = [(determinant, root_controls, zero, identity, 0)]
    lower = np.inf
    upper = -np.inf
    unresolved = False
    maximum_depth = 0

    while stack:
        piece, controls, origin, matrix, level = stack.pop()
        maximum_depth = max(maximum_depth, level)
        piece_lower, piece_upper = _bernstein_enclosure(controls)
        corners = tuple(evaluate_polynomial(piece, point) for point in vertices)
        if any(value <= 0 or value < threshold_exact for value in corners):
            return CellValidityStatus.INVALID, root_lower, root_upper, maximum_depth
        if any(value == threshold_exact for value in corners):
            return CellValidityStatus.UNRESOLVED, root_lower, root_upper, maximum_depth
        if min(controls) > threshold_exact:
            lower = min(lower, piece_lower)
            upper = max(upper, piece_upper)
            continue

        parent_points = tuple(
            _exact_affine_image(origin, matrix, point) for point in vertices
        )
        parent_representations = {
            tuple(float(value) for value in point) for point in parent_points
        }
        for child_origin, child_matrix in zip(child_origins, child_matrices, strict=True):
            child_polynomial = compose(
                piece,
                _exact_affine_arguments(child_origin, child_matrix),
                dimension=dimension,
            )
            child_controls_ = bernstein_coefficients(child_polynomial, domain, dimension)
            maximum_depth = max(maximum_depth, level + 1)
            samples = tuple(
                evaluate_polynomial(child_polynomial, point) for point in vertices
            )
            if any(value <= 0 or value < threshold_exact for value in samples):
                return CellValidityStatus.INVALID, root_lower, root_upper, maximum_depth
            child_lower, child_upper = _bernstein_enclosure(child_controls_)
            if any(value == threshold_exact for value in samples):
                unresolved = True
                lower = min(lower, child_lower)
                upper = max(upper, child_upper)
                continue
            if min(child_controls_) > threshold_exact:
                lower = min(lower, child_lower)
                upper = max(upper, child_upper)
                continue

            child_global_origin, child_global_matrix = _exact_child_affine(
                origin, matrix, child_origin, child_matrix
            )
            child_points = tuple(
                _exact_affine_image(child_global_origin, child_global_matrix, point)
                for point in vertices
            )
            has_new_vertex = any(
                tuple(float(value) for value in point) not in parent_representations
                for point in child_points
            )
            if not has_new_vertex:
                unresolved = True
                lower = min(lower, child_lower)
                upper = max(upper, child_upper)
                continue
            stack.append(
                (
                    child_polynomial,
                    child_controls_,
                    child_global_origin,
                    child_global_matrix,
                    level + 1,
                )
            )

    status = (
        CellValidityStatus.UNRESOLVED
        if unresolved
        else CellValidityStatus.CERTIFIED_VALID
    )
    return status, float(lower), float(upper), maximum_depth


def _certify_numerical_polynomial_cell(
    element: FiniteElementSpec,
    cell_kind: str,
    local: NDArray[np.float64],
    original_local: NDArray[np.float64],
    policy: CellValidityPolicy,
    plan: _BernsteinPlan,
    threshold: float,
    /,
    *,
    exact_affine_available: bool,
) -> tuple[CellValidityStatus, float, float, int]:
    dimension = plan.nodes.shape[1]
    vertices = np.asarray(plan.subdivision.vertices, dtype=np.float64)
    child_count = plan.subdivision.origins.shape[0]
    lower = np.inf
    upper = -np.inf
    maximum_depth = 0
    invalid = False
    unresolved = False
    stack = [(np.zeros((dimension,)), np.eye(dimension), 0)]

    while stack:
        origin, matrix, level = stack.pop()
        coefficients, margins = _evaluate_pieces(
            element,
            cell_kind,
            plan,
            local[None, :, :],
            np.zeros((1,), dtype=np.int64),
            origin[None, :],
            matrix[None, :, :],
        )
        values = coefficients[0]
        margin = margins[0]
        piece_lower = float(np.min(values) - margin)
        piece_upper = float(np.max(values) + margin)
        corner = float(np.min(values[plan.vertex_indices]))
        maximum_depth = max(maximum_depth, level)

        if invalid:
            lower = min(lower, piece_lower)
            upper = max(upper, piece_upper)
            continue
        if piece_lower > threshold:
            lower = min(lower, piece_lower)
            upper = max(upper, piece_upper)
            continue
        if corner + margin < threshold:
            invalid = True
            lower = min(lower, piece_lower)
            upper = max(upper, piece_upper)
            continue
        if level == 0 and exact_affine_available:
            exact = _exact_affine_root(element, original_local, policy)
            if exact is not None:
                decision, lower, upper = exact
                invalid = decision == CellValidityStatus.INVALID
                unresolved = decision == CellValidityStatus.UNRESOLVED
                continue

        child_origins = origin[None, :] + ein.contract(
            "ij,cj->ci", matrix, plan.subdivision.origins
        )
        child_matrices = ein.contract("ij,cjk->cik", matrix, plan.subdivision.matrices)
        child_points = _child_vertices(child_origins, child_matrices, vertices)
        parent_points = origin[None, :] + ein.contract("ij,vj->vi", matrix, vertices)
        child_has_new_vertex = ~np.all(
            np.any(
                np.all(
                    child_points[:, :, None, :] == parent_points[None, None, :, :],
                    axis=-1,
                ),
                axis=-1,
            ),
            axis=-1,
        )
        if not np.all(child_has_new_vertex):
            unresolved = True
            lower = min(lower, piece_lower)
            upper = max(upper, piece_upper)
        for child in reversed(range(child_count)):
            if child_has_new_vertex[child]:
                stack.append((child_origins[child], child_matrices[child], level + 1))

    status = (
        CellValidityStatus.INVALID
        if invalid
        else CellValidityStatus.UNRESOLVED
        if unresolved
        else CellValidityStatus.CERTIFIED_VALID
    )
    return status, float(lower), float(upper), maximum_depth


def _certify_polynomial_cell(
    element: FiniteElementSpec,
    cell_kind: str,
    local: NDArray[np.float64],
    original_local: NDArray[np.float64],
    policy: CellValidityPolicy,
    plan: _BernsteinPlan,
    threshold: float,
    exact_basis: tuple[Expression, ...] | None,
    /,
) -> tuple[CellValidityStatus, float, float, int]:
    dimension = element.topological_dimension
    if exact_basis is not None and math.isfinite(threshold):
        exact_data = _exact_polynomial_data(exact_basis, original_local, dimension)
        if exact_data is not None:
            expressions, determinant = exact_data
            affine = affine_map(expressions, dimension)
            if affine is not None:
                status, exact_determinant = _affine_quality(affine[1], dimension, policy)
                return (
                    status,
                    coefficient_endpoint(exact_determinant, upper=False, saturate=True),
                    coefficient_endpoint(exact_determinant, upper=True, saturate=True),
                    0,
                )
            return _certify_exact_polynomial_cell(
                determinant, plan.domain, dimension, threshold
            )
    return _certify_numerical_polynomial_cell(
        element,
        cell_kind,
        local,
        original_local,
        policy,
        plan,
        threshold,
        exact_affine_available=exact_basis is not None,
    )


def _certify_polynomial_block(
    element: FiniteElementSpec,
    cell_kind: str,
    local: NDArray[np.float64],
    policy: CellValidityPolicy,
    /,
    *,
    original_local: NDArray[np.float64],
) -> _BlockCertificate:
    cell_kind = reference_cell_topology(cell_kind).family
    embedded = local.shape[-1] != element.topological_dimension
    if embedded and cell_kind not in ("interval", "triangle", "quadrilateral"):
        raise ValueError(f"Embedded {cell_kind} cells have no validity contract.")
    domain, degrees = _determinant_route(cell_kind, element.degree, embedded)
    plan = _bernstein_plan(domain, degrees)
    threshold = policy.determinant_floor() * _root_scale(element, cell_kind, plan, local)
    exact_basis: tuple[Expression, ...] | None = None
    if cell_kind != "pyramid":
        try:
            exact_basis = coordinate_source_basis(element)
        except ValueError:
            # Opaque and rational source bases use the outward numerical route.
            exact_basis = None
    status = np.empty((local.shape[0],), dtype=np.int32)
    lower = np.empty((local.shape[0],), dtype=np.float64)
    upper = np.empty((local.shape[0],), dtype=np.float64)
    depth = np.empty((local.shape[0],), dtype=np.int64)
    for cell in range(local.shape[0]):
        (
            status[cell],
            lower[cell],
            upper[cell],
            depth[cell],
        ) = _certify_polynomial_cell(
            element,
            cell_kind,
            local[cell],
            original_local[cell],
            policy,
            plan,
            float(threshold[cell]),
            exact_basis,
        )
    return _BlockCertificate(status, lower, upper, depth)


type _ExactCellValidity = tuple[CellValidityStatus, float, float, int]


def _source_proof_domain(cell_kind: str, /) -> BernsteinDomain | None:
    match reference_cell_topology(cell_kind).family:
        case "interval" | "quadrilateral" | "hexahedron" | "pyramid":
            return "box"
        case "triangle" | "tetrahedron":
            return "simplex"
        case "prism":
            return "prism"
        case _:
            return None


def _source_denominator_certificate(
    denominator: Mapping[tuple[int, ...], ExactScalar],
    domain: BernsteinDomain,
    dimension: int,
    /,
) -> tuple[int, _ExactCellValidity]:
    vertices = _reference_subdivision(domain, dimension).vertices
    center = tuple(
        sum((point[axis] for point in vertices), Fraction(0)) / len(vertices)
        for axis in range(dimension)
    )
    center_value = evaluate_polynomial(denominator, center)
    if center_value == 0:
        return 0, (CellValidityStatus.UNRESOLVED, -math.inf, math.inf, 0)
    sign = 1 if center_value > 0 else -1
    certificate = _certify_exact_polynomial_cell(
        scale(denominator, sign), domain, dimension, 0.0
    )
    return sign, certificate


def _source_denominators_defined(
    expressions: tuple[Expression, ...],
    domain: BernsteinDomain,
    dimension: int,
    /,
) -> tuple[bool, int]:
    seen: list[Mapping[tuple[int, ...], ExactScalar]] = []
    depth = 0
    for expression in expressions:
        if not isinstance(expression, RationalPolynomial):
            continue
        denominator = expression.denominator
        if any(denominator == previous for previous in seen):
            continue
        seen.append(denominator)
        _, certificate = _source_denominator_certificate(denominator, domain, dimension)
        depth = max(depth, certificate[3])
        if certificate[0] is not CellValidityStatus.CERTIFIED_VALID:
            return False, depth
    return True, depth


def _certify_positive_source_expression(
    expression: Expression,
    domain: BernsteinDomain,
    dimension: int,
    /,
) -> _ExactCellValidity:
    if not isinstance(expression, RationalPolynomial):
        return _certify_exact_polynomial_cell(expression, domain, dimension, 0.0)
    sign, denominator = _source_denominator_certificate(
        expression.denominator, domain, dimension
    )
    if denominator[0] is not CellValidityStatus.CERTIFIED_VALID:
        return CellValidityStatus.UNRESOLVED, -math.inf, math.inf, denominator[3]
    numerator = _certify_exact_polynomial_cell(
        scale(expression.numerator, sign), domain, dimension, 0.0
    )
    lower, upper = interval_divide(
        (
            np.asarray(numerator[1], dtype=np.float64),
            np.asarray(numerator[2], dtype=np.float64),
        ),
        (
            np.asarray(denominator[1], dtype=np.float64),
            np.asarray(denominator[2], dtype=np.float64),
        ),
    )
    return (
        numerator[0],
        float(lower),
        float(upper),
        max(numerator[3], denominator[3]),
    )


def _certify_nonlinear_source_map(
    expressions: tuple[Expression, ...],
    cell_kind: str,
    domain: BernsteinDomain,
    dimension: int,
    policy: CellValidityPolicy,
    /,
) -> _ExactCellValidity:
    """Prove the effective map and its pointwise relative Hadamard margin."""
    if reference_cell_topology(cell_kind).family == "pyramid":
        normalized = tuple(
            normalize_pyramid_reference_expression(expression)
            for expression in expressions
        )
        if any(value.apex_value is None for value in normalized):
            return CellValidityStatus.UNRESOLVED, math.nan, math.nan, 0
        normalized_derivatives = tuple(
            tuple(
                normalize_pyramid_reference_derivative(expression, axis)
                for axis in range(dimension)
            )
            for expression in expressions
        )
        # Normalized Jacobian entries live on the closed collapsed cube; a
        # direction-dependent apex value is admitted (see apex_derivative_unique).
        jacobian = tuple(
            tuple(value.expression for value in row) for row in normalized_derivatives
        )
        proof_coordinates = tuple(value.expression for value in normalized)
    else:
        proof_coordinates = expressions
        jacobian = _exact_map_jacobian(expressions, dimension)
    defined, denominator_depth = _source_denominators_defined(
        proof_coordinates + tuple(value for row in jacobian for value in row),
        domain,
        dimension,
    )
    if not defined:
        return CellValidityStatus.UNRESOLVED, math.nan, math.nan, denominator_depth
    determinant = _exact_map_determinant(jacobian, dimension)
    result = _certify_positive_source_expression(determinant, domain, dimension)
    status, lower, upper, depth = result
    depth = max(depth, denominator_depth)
    if status is not CellValidityStatus.CERTIFIED_VALID:
        return status, lower, upper, depth
    floor = Fraction(policy.determinant_floor())
    if _hadamard_margin_from_bounds(jacobian, domain, dimension, lower, floor):
        return status, lower, upper, depth
    scale_squared = _exact_hadamard_squared(jacobian, dimension)
    margin = (
        subtract(
            multiply(determinant, determinant),
            scale(scale_squared, floor * floor),
        )
        if len(jacobian) == dimension
        else subtract(determinant, scale(scale_squared, floor))
    )
    margin_certificate = _certify_positive_source_expression(margin, domain, dimension)
    return margin_certificate[0], lower, upper, max(depth, margin_certificate[3])


# One cell's proof (status, determinant lower/upper endpoint, depth) is a pure
# function of its exact coordinate map, cell kind and validity policy.
type _CellProof = tuple[int, float, float, int]
type _CellProofs = dict[bytes, _CellProof]
_MAP_DIGEST_SIZE = 16


def _polynomial_digest_terms(
    polynomial: Mapping[tuple[int, ...], ExactScalar], parts: list[object], /
) -> bool:
    parts.append(len(polynomial))
    for exponent, coefficient in polynomial.items():
        if type(coefficient) is Fraction:
            parts.append((exponent, coefficient.numerator, coefficient.denominator))
        elif type(coefficient) is RadicalElement:
            parts.append((exponent, "radical", coefficient.terms))
        else:
            return False
    return True


def _map_digest(cell_kind: str, expressions: tuple[Expression, ...], /) -> bytes | None:
    """Digest a cell's exact coordinate map, or ``None`` when it has no record.

    Coefficients are encoded with their exact type and in stored order, so an
    equal digest means the prover receives identical operands.
    """
    parts: list[object] = [cell_kind, len(expressions)]
    for expression in expressions:
        if isinstance(expression, RationalPolynomial):
            parts.append("rational")
            if not _polynomial_digest_terms(
                expression.numerator, parts
            ) or not _polynomial_digest_terms(expression.denominator, parts):
                return None
        else:
            parts.append("polynomial")
            if not _polynomial_digest_terms(expression, parts):
                return None
    return hashlib.blake2b(
        repr(parts).encode("ascii"), digest_size=_MAP_DIGEST_SIZE
    ).digest()


def _pyramid_apex_unique(expressions: tuple[Expression, ...], dimension: int, /) -> bool:
    return all(
        normalize_pyramid_reference_derivative(expression, axis).apex_value is not None
        for expression in expressions
        for axis in range(dimension)
    )


def _certify_source_block(
    element: ScalarCoordinateElement,
    cell_kind: str,
    coordinates: CoordinateSourceBank,
    routes: NDArray[np.int64],
    policy: CellValidityPolicy,
    proofs: _CellProofs,
    /,
    *,
    execution_map: bool = False,
) -> tuple[_BlockCertificate, bool, NDArray[np.uint8]]:
    """Prove an exact-source or binary64-shadow coordinate map.

    ``proofs`` holds proofs of ``policy`` keyed by exact map digest; a cell
    whose map is recorded there reuses its proof, and every proof made here
    is recorded. Also returns each cell's map digest (zero when unrecorded).
    """
    count = routes.shape[0]
    dimension = reference_cell_topology(cell_kind).dimension
    status = np.full(count, CellValidityStatus.UNRESOLVED, dtype=np.int32)
    lower = np.full(count, np.nan, dtype=np.float64)
    upper = np.full(count, np.nan, dtype=np.float64)
    depth = np.zeros(count, dtype=np.int64)
    apex_unique = np.ones(count, dtype=np.bool_)
    digests = np.zeros((count, _MAP_DIGEST_SIZE), dtype=np.uint8)
    pyramid = reference_cell_topology(cell_kind).family == "pyramid"
    domain = _source_proof_domain(cell_kind)
    has_unsupported_map = False
    for row in range(routes.shape[0]):
        local = tuple(
            coordinates[int(routes[row, column])] for column in range(routes.shape[1])
        )
        expressions = (
            coordinate_execution_map_expressions(element, local, cell_row=row)
            if execution_map
            else coordinate_map_expressions(element, local, cell_row=row)
        )
        # Without a proof domain an unsupported nonaffine map is reported, so
        # only maps with a domain carry reusable records.
        digest = None if domain is None else _map_digest(cell_kind, expressions)
        if digest is not None:
            digests[row] = np.frombuffer(digest, dtype=np.uint8)
            recorded = proofs.get(digest)
            if recorded is not None:
                status[row], lower[row], upper[row], depth[row] = recorded
                if pyramid and affine_map(expressions, dimension) is None:
                    apex_unique[row] = _pyramid_apex_unique(expressions, dimension)
                continue
        affine = affine_map(expressions, dimension)
        if affine is None:
            if domain is None:
                has_unsupported_map = True
                continue
            status[row], lower[row], upper[row], depth[row] = (
                _certify_nonlinear_source_map(
                    expressions, cell_kind, domain, dimension, policy
                )
            )
            if pyramid:
                apex_unique[row] = _pyramid_apex_unique(expressions, dimension)
        else:
            _, jacobian = affine
            status[row], determinant = _affine_quality(jacobian, dimension, policy)
            lower[row] = coefficient_endpoint(determinant, upper=False, saturate=True)
            upper[row] = coefficient_endpoint(determinant, upper=True, saturate=True)
        if digest is not None:
            proofs[digest] = (
                int(status[row]),
                float(lower[row]),
                float(upper[row]),
                int(depth[row]),
            )
    return (
        _BlockCertificate(status, lower, upper, depth, apex_unique),
        has_unsupported_map,
        digests,
    )


@dataclass(frozen=True, slots=True)
class ValidatedCellValidityParent:
    """Host-only mathematical replay of one parent validity proof."""

    certificate: CellValidityCertificate

    def __post_init__(self) -> None:
        self.certificate.revalidate(self.certificate.geometry, mesh=self.certificate.mesh)


def _parent_proofs(
    parent: CellValidityCertificate
    | ValidatedCellValidityParent
    | tuple[CellValidityCertificate | ValidatedCellValidityParent, ...]
    | None,
    policy: CellValidityPolicy,
    /,
) -> _CellProofs:
    """Index the recorded execution and source proofs of parent certificates."""
    match parent:
        case None:
            parents: tuple[
                CellValidityCertificate | ValidatedCellValidityParent, ...
            ] = ()
        case CellValidityCertificate() | ValidatedCellValidityParent():
            parents = (parent,)
        case tuple():
            parents = parent
        case _:
            raise TypeError(
                "parent must be CellValidityCertificate, a tuple of them, or None."
            )
    proofs: _CellProofs = {}
    for value in parents:
        if type(value) is ValidatedCellValidityParent:
            certificate = value.certificate
            certificate.require_current(certificate.geometry, mesh=certificate.mesh)
        elif type(value) is CellValidityCertificate:
            certificate = value
            certificate.revalidate(certificate.geometry, mesh=certificate.mesh)
        else:
            raise TypeError(
                "parent must hold exact validity certificates or validations."
            )
        if certificate.policy_id != policy.policy_id:
            raise ValueError("Parent validity proofs must use the same validity policy.")
        table = np.frombuffer(certificate.cell_map_digests, dtype=np.uint8).reshape(
            -1, 2, _MAP_DIGEST_SIZE
        )
        for column, arrays in enumerate(
            (
                (
                    certificate.execution_status,
                    certificate.determinant_lower,
                    certificate.determinant_upper,
                    certificate.depth,
                ),
                (
                    certificate.source_status,
                    certificate.source_determinant_lower,
                    certificate.source_determinant_upper,
                    certificate.source_depth,
                ),
            )
        ):
            digests = table[:, column]
            rows = np.flatnonzero(np.any(digests != 0, axis=1))
            status, lower, upper, depth = (
                np.asarray(value)[rows].tolist() for value in arrays
            )
            for row, values in zip(
                rows.tolist(), zip(status, lower, upper, depth, strict=True), strict=True
            ):
                proofs.setdefault(digests[row].tobytes(), values)
    return proofs


def _polynomial_coefficient_difference(
    first: Mapping[tuple[int, ...], ExactScalar],
    second: Mapping[tuple[int, ...], ExactScalar],
    /,
) -> ExactScalar:
    return max(
        (
            abs(first.get(exponent, Fraction(0)) - second.get(exponent, Fraction(0)))
            for exponent in first.keys() | second.keys()
        ),
        default=Fraction(0),
    )


def _expression_evidence_equal(first: Expression, second: Expression, /) -> bool:
    """Compare exact coefficients and rational denominator-domain evidence."""
    if isinstance(first, RationalPolynomial):
        return (
            isinstance(second, RationalPolynomial)
            and dict(first.numerator) == dict(second.numerator)
            and dict(first.denominator) == dict(second.denominator)
        )
    return not isinstance(second, RationalPolynomial) and dict(first) == dict(second)


def _expression_sequence_evidence_equal(
    first: tuple[Expression, ...], second: tuple[Expression, ...], /
) -> bool:
    return len(first) == len(second) and all(
        _expression_evidence_equal(left, right)
        for left, right in zip(first, second, strict=True)
    )


def _element_basis_rounding_error(
    source: tuple[Expression, ...],
    execution: tuple[Expression, ...],
    dimension: int,
    /,
) -> ExactScalar:
    """Return a dimensionless coefficient discrepancy between effective bases."""
    if len(source) != len(execution):
        return Fraction(1)
    maximum: ExactScalar = Fraction(0)
    unit = constant(1, dimension)
    for source_value, execution_value in zip(source, execution, strict=True):
        if isinstance(source_value, RationalPolynomial) and isinstance(
            execution_value, RationalPolynomial
        ):
            error = max(
                _polynomial_coefficient_difference(
                    source_value.numerator, execution_value.numerator
                ),
                _polynomial_coefficient_difference(
                    source_value.denominator, execution_value.denominator
                ),
            )
        elif not isinstance(source_value, RationalPolynomial) and not isinstance(
            execution_value, RationalPolynomial
        ):
            error = _polynomial_coefficient_difference(source_value, execution_value)
        else:
            source_numerator = (
                source_value.numerator
                if isinstance(source_value, RationalPolynomial)
                else source_value
            )
            source_denominator = (
                source_value.denominator
                if isinstance(source_value, RationalPolynomial)
                else unit
            )
            execution_numerator = (
                execution_value.numerator
                if isinstance(execution_value, RationalPolynomial)
                else execution_value
            )
            execution_denominator = (
                execution_value.denominator
                if isinstance(execution_value, RationalPolynomial)
                else unit
            )
            difference = subtract(
                multiply(source_numerator, execution_denominator),
                multiply(execution_numerator, source_denominator),
            )
            if isinstance(difference, RationalPolynomial):
                raise RuntimeError(
                    "Cross-multiplied basis difference must be polynomial."
                )
            error = max(
                (abs(value) for value in difference.values()),
                default=Fraction(0),
            )
        maximum = max(maximum, error)
    return maximum


# Star-decomposed variable-topology cells --------------------------------------------


@dataclass(frozen=True)
class PolyhedralStarTables:
    """Host CSR tables of the face-centroid star decomposition of polyhedra.

    Star entry ``e`` is the tetrahedron ``(cell centroid, face centroid,
    star_first[e], star_second[e])`` whose base edge follows the outward
    orientation of face ``star_face[e]`` in cell ``star_cell[e]``;
    ``star_sign[e]`` orients the stored face loop outward. Each cell edge
    appears in exactly two entries; ``edge_pairs`` lists them with the first
    entry fixing the edge direction.
    """

    face_sizes: np.ndarray
    face_corner_face: np.ndarray
    face_corner_vertex: np.ndarray
    face_corner_next: np.ndarray
    cell_vertex_cell: np.ndarray
    cell_vertex_values: np.ndarray
    cell_vertex_counts: np.ndarray
    star_cell: np.ndarray
    star_face: np.ndarray
    star_first: np.ndarray
    star_second: np.ndarray
    star_sign: np.ndarray
    edge_pairs: np.ndarray


def polyhedral_star_tables(
    connectivity: PolyhedralConnectivity, /
) -> PolyhedralStarTables:
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("connectivity must be PolyhedralConnectivity.")
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    cell_face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    cell_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.float64)
    cell_vertex_offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int64)
    cell_vertices = np.asarray(connectivity.cell_vertex_values, dtype=np.int64)
    face_sizes = np.diff(face_offsets)
    corner_face = np.repeat(np.arange(face_sizes.size), face_sizes)
    corner_local = np.arange(face_values.size) - face_offsets[corner_face]
    corner_next = face_values[
        face_offsets[corner_face] + (corner_local + 1) % face_sizes[corner_face]
    ]
    incidence_cell = np.repeat(
        np.arange(cell_face_offsets.size - 1), np.diff(cell_face_offsets)
    )
    entry_counts = face_sizes[cell_faces]
    entry_offsets = np.concatenate(((0,), np.cumsum(entry_counts)))
    entry_incidence = np.repeat(np.arange(cell_faces.size), entry_counts)
    entry_corner = face_offsets[cell_faces[entry_incidence]] + (
        np.arange(entry_offsets[-1]) - entry_offsets[entry_incidence]
    )
    outward = cell_signs[entry_incidence] > 0.0
    first = np.where(outward, face_values[entry_corner], corner_next[entry_corner])
    second = np.where(outward, corner_next[entry_corner], face_values[entry_corner])
    star_cell = incidence_cell[entry_incidence]
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    order = np.lexsort((first, high, low, star_cell))
    if order.size % 2 or not (
        np.array_equal(star_cell[order[0::2]], star_cell[order[1::2]])
        and np.array_equal(low[order[0::2]], low[order[1::2]])
        and np.array_equal(high[order[0::2]], high[order[1::2]])
        and np.array_equal(first[order[0::2]], second[order[1::2]])
    ):
        raise ValueError(
            "Polyhedral cells must be closed, consistently oriented surfaces."
        )
    return PolyhedralStarTables(
        face_sizes=face_sizes,
        face_corner_face=corner_face,
        face_corner_vertex=face_values,
        face_corner_next=corner_next,
        cell_vertex_cell=np.repeat(
            np.arange(cell_vertex_offsets.size - 1), np.diff(cell_vertex_offsets)
        ),
        cell_vertex_values=cell_vertices,
        cell_vertex_counts=np.diff(cell_vertex_offsets),
        star_cell=star_cell,
        star_face=cell_faces[entry_incidence],
        star_first=first,
        star_second=second,
        star_sign=np.where(outward, 1.0, -1.0),
        edge_pairs=np.stack((order[0::2], order[1::2]), axis=1),
    )


def _segment_mean(values: np.ndarray, segments: np.ndarray, count: int, /) -> np.ndarray:
    totals = np.zeros((count,) + values.shape[1:])
    np.add.at(totals, segments, values)
    sizes = np.bincount(segments, minlength=count).astype(np.float64)
    return totals / sizes.reshape((-1,) + (1,) * (values.ndim - 1))


def _star_status(
    determinants: np.ndarray,
    margins: np.ndarray,
    segments: np.ndarray,
    thresholds: np.ndarray,
    total_measure: np.ndarray,
    /,
) -> _BlockCertificate:
    """Classify cells from exact star-simplex determinants.

    All star simplices above the floor certify an invertible piecewise-affine
    star map. A total signed measure below the floor proves inversion or
    degeneracy. Anything else is not star-shaped about the chosen center, which
    is not a validity disproof, so it remains UNRESOLVED.
    """

    count = thresholds.size
    lower = np.full((count,), np.inf)
    upper = np.full((count,), -np.inf)
    np.minimum.at(lower, segments, determinants - margins)
    np.maximum.at(upper, segments, determinants + margins)
    status = np.where(
        lower > thresholds,
        CellValidityStatus.CERTIFIED_VALID,
        np.where(
            total_measure < thresholds,
            CellValidityStatus.INVALID,
            CellValidityStatus.UNRESOLVED,
        ),
    ).astype(np.int32)
    return _BlockCertificate(status, lower, upper, np.zeros((count,), dtype=np.int64))


def _polygon_measure(points: np.ndarray, policy: CellValidityPolicy, /) -> tuple:
    """Determinant enclosure, floor, planarity flags, and planar loops of polygons.

    The determinant is twice the signed shoelace area about the vertex centroid
    (planar) or the squared Newell vector area (embedded, Gram convention).
    Embedded loops are measured against their Newell plane through the vertex
    centroid and projected by dropping the dominant normal axis, which keeps the
    projected coordinates exact. Returns ``(determinant, margin, threshold,
    nonplanar, ambiguous_planarity, planar_points)``.
    """

    cell_count, corner_count, ambient = points.shape
    center = np.mean(points, axis=1, keepdims=True)
    first = points - center
    second = np.roll(first, -1, axis=1)
    scale = np.sum(
        np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1), axis=1
    )
    rounding = 2.0 * (corner_count + 8) * _EPSILON * scale
    floor = policy.relative_determinant_floor
    if ambient == 2:
        area = np.sum(
            first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0], axis=1
        )
        planar = np.zeros((cell_count,), dtype=np.bool_)
        return area, rounding, floor * scale, planar, planar, points
    area = np.sum(np.cross(first, second), axis=1)
    magnitude = np.linalg.norm(area, axis=-1)
    guarded = np.maximum(magnitude, np.finfo(np.float64).tiny)
    # Vertex offsets from the Newell plane; the normal direction inherits the
    # relative rounding of the area vector.
    offset = np.max(
        np.abs(np.sum(first * (area / guarded[:, None])[:, None], axis=-1)), axis=1
    )
    slack = np.max(np.linalg.norm(first, axis=-1), axis=1) * (
        16.0 * _EPSILON + rounding / guarded
    )
    diameter = np.linalg.norm(np.max(points, axis=1) - np.min(points, axis=1), axis=-1)
    allowed = policy.relative_planarity_tolerance * diameter
    axes = np.asarray(((1, 2), (0, 2), (0, 1)))[np.argmax(np.abs(area), axis=-1)]
    return (
        magnitude * magnitude,
        2.0 * magnitude * rounding + rounding * rounding,
        floor * scale * scale,
        offset - slack > allowed,
        offset + slack > allowed,
        np.take_along_axis(points, axes[:, None, :], axis=2),
    )


def _certify_polygon_block(points: np.ndarray, policy: CellValidityPolicy, /) -> Any:
    """Certify polygon cells without assuming star-shapedness.

    A planar polygon is valid iff its vertices are finite and distinct, its
    boundary is simple, it is counterclockwise, and twice its area exceeds the
    policy floor. An embedded polygon must instead lie within
    ``relative_planarity_tolerance`` of its Newell plane, be simple in its
    dominant-axis projection, and have a squared vector area above the floor.
    """

    cell_count, corner_count, ambient = points.shape
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    finite = np.all(np.isfinite(points), axis=(1, 2))
    safe = np.where(finite[:, None, None], points, 0.0)
    order = np.lexsort(np.moveaxis(safe, 2, 0), axis=-1)
    ordered = np.take_along_axis(safe, order[..., None], axis=1)
    repeated = np.any(np.all(ordered[:, 1:] == ordered[:, :-1], axis=-1), axis=1)
    centered = safe - np.mean(safe, axis=1, keepdims=True)
    following = np.roll(centered, -1, axis=1)
    edge_vectors = following - centered
    edge_lengths = np.linalg.norm(edge_vectors, axis=-1)
    edge_roundoff = (
        4.0 * _EPSILON * np.linalg.norm(np.abs(centered) + np.abs(following), axis=-1)
    )
    diameter = np.linalg.norm(np.max(safe, axis=1) - np.min(safe, axis=1), axis=-1)
    edge_floor = POLYGON_EDGE_RESOLUTION * diameter[:, None]
    short_edge = np.any(edge_lengths + edge_roundoff < edge_floor, axis=1)
    uncertain_edge = np.any(edge_lengths - edge_roundoff <= edge_floor, axis=1)
    determinant, margin, threshold, nonplanar, ambiguous, planar = _polygon_measure(
        safe, policy
    )
    invalid = ~finite | repeated | nonplanar | short_edge
    candidates = np.flatnonzero(~invalid)
    simplicity = np.full((cell_count,), PolygonSimplicityStatus.SIMPLE, dtype=np.int8)
    # Embedded polygons carry no ambient orientation convention.
    orientation = np.full((cell_count,), PredicateSign.POSITIVE, dtype=np.int8)
    if candidates.size:
        result = polygon_simplicity_2d(planar[candidates], mode=mode)
        simplicity[candidates] = np.asarray(result.status)
        if ambient == 2:
            orientation[candidates] = np.asarray(result.orientation)
    simple = simplicity == PolygonSimplicityStatus.SIMPLE
    lower = determinant - margin
    upper = determinant + margin
    invalid |= (simplicity == PolygonSimplicityStatus.SELF_INTERSECTING) | (
        simple & ((orientation == PredicateSign.NEGATIVE) | (upper < threshold))
    )
    # A certified simple loop has exactly nonzero area, which meets a zero floor.
    above = (lower > threshold) | (simple & (threshold == 0.0))
    unresolved = (
        ambiguous
        | (uncertain_edge & ~short_edge)
        | ~simple
        | (orientation == PredicateSign.UNCERTAIN)
        | ~above
    )
    status = np.where(
        invalid,
        CellValidityStatus.INVALID,
        np.where(
            unresolved,
            CellValidityStatus.UNRESOLVED,
            CellValidityStatus.CERTIFIED_VALID,
        ),
    ).astype(np.int32)
    return _BlockCertificate(
        status,
        np.where(finite, lower, np.nan),
        np.where(finite, upper, np.nan),
        np.zeros((cell_count,), dtype=np.int64),
    )


def _certify_polyhedral_cells(
    tables: PolyhedralStarTables,
    coordinates: np.ndarray,
    cells: np.ndarray,
    policy: CellValidityPolicy,
    /,
) -> _BlockCertificate:
    """Certify polyhedral cells ``cells`` (global connectivity rows) of one block."""

    cell_total = tables.cell_vertex_counts.size
    face_total = tables.face_sizes.size
    center = _segment_mean(
        coordinates[tables.cell_vertex_values], tables.cell_vertex_cell, cell_total
    )
    face_center = _segment_mean(
        coordinates[tables.face_corner_vertex], tables.face_corner_face, face_total
    )
    selected = np.isin(tables.star_cell, cells)
    star_cell = tables.star_cell[selected]
    apex = center[star_cell]
    columns = np.stack(
        (
            face_center[tables.star_face[selected]] - apex,
            coordinates[tables.star_first[selected]] - apex,
            coordinates[tables.star_second[selected]] - apex,
        ),
        axis=-1,
    )
    determinants = _square_determinant(columns)
    scale = np.prod(np.linalg.norm(columns, axis=-2), axis=-1)
    local = np.searchsorted(cells, star_cell)
    thresholds = np.zeros((cells.size,))
    np.maximum.at(thresholds, local, policy.determinant_floor() * scale)
    total = np.zeros((cells.size,))
    np.add.at(total, local, determinants)
    return _star_status(determinants, 32.0 * _EPSILON * scale, local, thresholds, total)


# Public entry point ----------------------------------------------------------------


def _polyhedral_coordinates(
    mesh: CellMesh, routes: tuple[np.ndarray, ...], values: np.ndarray, /
) -> np.ndarray:
    """Gather vertex-geometry rows of polyhedral blocks into mesh-vertex order."""

    points = np.full((mesh.coordinates.shape[0], values.shape[1]), np.nan)
    for block, route in zip(mesh.blocks, routes, strict=True):
        if block.cell_kind == "polyhedron":
            vertices = np.asarray(block.vertices, dtype=np.int64)
            valid = np.asarray(block.vertex_valid, dtype=np.bool_)
            points[vertices[valid]] = values[route[valid]]
    return points


def certify_cell_geometry_validity(
    geometry: CellMesh | CellGeometrySpec,
    /,
    *,
    mesh: CellMesh | None = None,
    policy: CellValidityPolicy | None = None,
    parent: CellValidityCertificate
    | ValidatedCellValidityParent
    | tuple[CellValidityCertificate | ValidatedCellValidityParent, ...]
    | None = None,
) -> CellValidityCertificate:
    """Certify determinant positivity of every mapped cell with Bernstein bounds.

    A ``CellMesh`` is certified with its affine vertex geometry. A
    ``CellGeometrySpec`` is certified in mesh block order when ``mesh`` is given
    (required for polyhedral blocks) and in its own sorted block order otherwise.

    ``parent`` passes earlier certificates of the same ``policy`` explicitly. A
    cell whose exact coordinate map (cell kind and exact coefficients) equals a
    map a parent proved reuses that recorded proof, matched by map digest and
    never by cell index; every other cell is proved. The result is identical
    to the certificate of a full proof.
    """

    policy_ = CellValidityPolicy() if policy is None else policy
    if not isinstance(policy_, CellValidityPolicy):
        raise TypeError("policy must be CellValidityPolicy or None.")
    proofs = _parent_proofs(parent, policy_)
    if isinstance(geometry, CellMesh):
        if mesh is not None:
            raise ValueError("mesh is implied when certifying a CellMesh.")
        mesh_ = geometry
        spec = CellGeometrySpec.affine(geometry)
    elif isinstance(geometry, CellGeometrySpec):
        mesh_ = mesh
        spec = geometry
    else:
        raise TypeError("geometry must be CellMesh or CellGeometrySpec.")
    if mesh_ is None:
        elements = spec.elements
        routes = tuple(np.asarray(value, dtype=np.int64) for value in spec.geometry_dofs)
        names = spec.block_names
        kinds = tuple(element.cell_kind for element in elements)
    else:
        if not isinstance(mesh_, CellMesh):
            raise TypeError("mesh must be CellMesh or None.")
        resolved, resolved_routes, _ = spec.resolve(mesh_)
        elements = resolved
        routes = tuple(np.asarray(value, dtype=np.int64) for value in resolved_routes)
        names = tuple(block.name for block in mesh_.blocks)
        kinds = tuple(block.cell_kind for block in mesh_.blocks)
    values = np.asarray(spec.coordinates, dtype=np.float64)
    execution_coordinates: CoordinateSourceBank = tuple(
        tuple(Fraction(float(value)) for value in row) for row in values
    )
    source_coordinates = (
        spec.source_coordinates() if spec.exact_coordinate_bank is not None else None
    )
    source_bank = (
        execution_coordinates if source_coordinates is None else source_coordinates
    )
    distinct_source = (
        source_coordinates is not None and source_coordinates != execution_coordinates
    )
    coefficient_rounding_error = (
        max(
            (
                coordinate_point_rounding_error(point, values[row])
                for row, point in enumerate(source_coordinates)
            ),
            default=Fraction(0),
        )
        if source_coordinates is not None
        else Fraction(0)
    )
    element_rounding_error: ExactScalar = Fraction(0)
    unsupported: set[str] = set()
    for name, element, route in zip(names, elements, routes, strict=True):
        if not isinstance(element, ScalarCoordinateElement):
            continue
        for row in range(route.shape[0]):
            try:
                source_basis = coordinate_source_basis(element, cell_row=row)
                execution_basis = coordinate_execution_basis(element, cell_row=row)
            except (OverflowError, TypeError, ValueError):
                distinct_source = True
                unsupported.add(name)
                continue
            distinct_source = distinct_source or not _expression_sequence_evidence_equal(
                source_basis, execution_basis
            )
            element_rounding_error = max(
                element_rounding_error,
                _element_basis_rounding_error(
                    source_basis,
                    execution_basis,
                    element.topological_dimension,
                ),
            )
    tables = None
    polyhedral_points = None
    cursor = 0
    offsets = [0]
    blocks: list[_BlockCertificate] = []
    source_blocks: list[_BlockCertificate] = []
    digests = np.zeros(
        (sum(route.shape[0] for route in routes), 2, _MAP_DIGEST_SIZE), dtype=np.uint8
    )
    for name, kind, element, route in zip(names, kinds, elements, routes, strict=True):
        cell_count = route.shape[0]
        if isinstance(element, CellVertexGeometryElement) and kind == "polygon":
            blocks.append(_certify_polygon_block(values[route], policy_))
        elif isinstance(element, CellVertexGeometryElement):
            if mesh_ is None:
                raise ValueError("Polyhedral validity certification requires the mesh.")
            if tables is None:
                connectivity = mesh_.connectivity
                if not isinstance(connectivity, PolyhedralConnectivity):
                    raise TypeError(
                        "Polyhedral geometry requires polyhedral mesh connectivity."
                    )
                tables = polyhedral_star_tables(connectivity)
                polyhedral_points = _polyhedral_coordinates(mesh_, routes, values)
            if polyhedral_points is None:
                raise RuntimeError(
                    "Polyhedral coordinate preparation did not produce coordinates."
                )
            blocks.append(
                _certify_polyhedral_cells(
                    tables,
                    polyhedral_points,
                    np.arange(cursor, cursor + cell_count),
                    policy_,
                )
            )
        elif isinstance(element, ScalarCoordinateElement):
            try:
                certificate, has_unsupported_map, block_digests = _certify_source_block(
                    element,
                    kind,
                    execution_coordinates,
                    route,
                    policy_,
                    proofs,
                    execution_map=True,
                )
            except (OverflowError, TypeError, ValueError):
                has_unsupported_map = True
                certificate = _BlockCertificate(
                    np.full((cell_count,), CellValidityStatus.UNRESOLVED, np.int32),
                    np.full((cell_count,), np.nan),
                    np.full((cell_count,), np.nan),
                    np.zeros((cell_count,), dtype=np.int64),
                )
            else:
                digests[cursor : cursor + cell_count, 0] = block_digests
            blocks.append(certificate)
            if has_unsupported_map:
                unsupported.add(name)
        else:
            unsupported.add(name)
            blocks.append(
                _BlockCertificate(
                    np.full((cell_count,), CellValidityStatus.UNRESOLVED, np.int32),
                    np.full((cell_count,), np.nan),
                    np.full((cell_count,), np.nan),
                    np.zeros((cell_count,), dtype=np.int64),
                )
            )
        if distinct_source:
            if isinstance(element, ScalarCoordinateElement):
                try:
                    source_certificate, source_unsupported, block_digests = (
                        _certify_source_block(
                            element, kind, source_bank, route, policy_, proofs
                        )
                    )
                except (OverflowError, TypeError, ValueError):
                    source_unsupported = True
                    source_certificate = _BlockCertificate(
                        np.full((cell_count,), CellValidityStatus.UNRESOLVED, np.int32),
                        np.full((cell_count,), np.nan),
                        np.full((cell_count,), np.nan),
                        np.zeros((cell_count,), dtype=np.int64),
                    )
                else:
                    digests[cursor : cursor + cell_count, 1] = block_digests
                if source_unsupported:
                    unsupported.add(name)
            elif (
                isinstance(element, CellVertexGeometryElement)
                and source_coordinates is not None
                and source_coordinates != execution_coordinates
            ):
                if mesh_ is None:
                    raise ValueError(
                        "Exact variable-topology source validity requires the bound mesh."
                    )
                from ._cell_geometry_source_validity import (
                    certify_exact_variable_topology_block,
                )

                source_certificate = certify_exact_variable_topology_block(
                    mesh_,
                    dict(zip(names, routes, strict=True)),
                    source_coordinates,
                    policy_,
                    block_name=name,
                    cell_rows=np.arange(cell_count, dtype=np.int64),
                )
            else:
                source_certificate = blocks[-1]
            source_blocks.append(source_certificate)
        cursor += cell_count
        offsets.append(cursor)
    execution_proof = _BlockCertificate(
        np.concatenate([block.status for block in blocks]),
        np.concatenate([block.lower for block in blocks]),
        np.concatenate([block.upper for block in blocks]),
        np.concatenate([block.depth for block in blocks]),
        np.concatenate([block.apex_flags() for block in blocks]),
    )
    source_proof = (
        _BlockCertificate(
            np.concatenate([block.status for block in source_blocks]),
            np.concatenate([block.lower for block in source_blocks]),
            np.concatenate([block.upper for block in source_blocks]),
            np.concatenate([block.depth for block in source_blocks]),
            np.concatenate([block.apex_flags() for block in source_blocks]),
        )
        if distinct_source
        else execution_proof
    )
    if not distinct_source:
        digests[:, 1] = digests[:, 0]
    return CellValidityCertificate._from_computed(
        execution_proof.status,
        execution_proof.lower,
        execution_proof.upper,
        execution_proof.depth,
        source_proof=source_proof,
        maximum_coefficient_rounding_error=coefficient_endpoint(
            coefficient_rounding_error, upper=True, saturate=True
        ),
        maximum_element_coefficient_rounding_error=coefficient_endpoint(
            element_rounding_error, upper=True, saturate=True
        ),
        apex_derivative_unique=execution_proof.apex_flags() & source_proof.apex_flags(),
        geometry=spec,
        mesh=mesh_,
        policy=policy_,
        block_names=names,
        block_offsets=tuple(offsets),
        unsupported_block_names=tuple(name for name in names if name in unsupported),
        geometry_id=cell_geometry_id(spec),
        geometry_layout_id=spec.geometry_layout_id,
        policy_id=policy_.policy_id,
        cell_map_digests=digests.tobytes(),
    )


__all__ = [
    "CellValidityCertificate",
    "CellValidityPolicy",
    "CellValidityStatus",
    "certify_cell_geometry_validity",
]
