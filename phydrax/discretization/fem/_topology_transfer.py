#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any, final, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._polynomial._cubature import cubature_rule_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    estimate_condition_number,
    OperatorCapabilities,
    OperatorProperties,
    prepare_sparse_factorization,
    PreparedSparseFactorization,
    refresh_sparse_factorization,
    SparseFactorizationPlan,
    SparseFactorizationPolicy,
    SparseFactorizationStatus,
    SpectralEstimate,
)
from ...linalg._operators import _generic_adjoint, _materialize_by_basis
from ...sparse import (
    EdgeRelation,
    linear_apply,
    linear_transpose_apply,
    RowRelation,
    SparseLinearMap,
)
from .._cell_mesh import CellMesh
from ._generic import FiniteElementDiscretization
from ._reference import FiniteElementSpec, lagrange_element


if TYPE_CHECKING:
    from ...geometry import PreparedCommonRefinement


# Invariant claims are certified in the coefficient dtype up to this many ulps of
# the local row scale.
_CLAIM_ULPS = 64.0

_SIMPLEX_KINDS = {2: "triangle", 3: "tetrahedron"}
_LAGRANGE_FAMILIES = ("Lagrange", "DiscontinuousLagrange")


def _claim_tolerance(dtype: np.dtype, scale: np.ndarray, /) -> np.ndarray:
    return _CLAIM_ULPS * np.finfo(dtype).eps * np.maximum(scale, 1.0)


def _block_action(action: Callable[[Array], Array], values: ArrayLike, /) -> Array:
    """Apply a canonical-column action to ``(dofs, *payload)`` values."""

    array = jnp.asarray(values)
    if array.ndim == 0:
        raise ValueError("Transfer inputs require a leading DOF axis.")
    block = action(array.reshape((array.shape[0], -1)))
    return block.reshape((block.shape[0],) + array.shape[1:])


class FiniteElementTopologyTransfer(StrictModule, NonTrainableState):
    """Primal coefficient transfer between two finite-element topologies.

    ``primal`` maps source coefficients to target coefficients (target DOFs by source
    DOFs): a ``SparseLinearMap`` for local stencils, or a linear operator such as
    ``FiniteElementL2Projection`` whose action couples every DOF. Its algebraic
    transpose is the raw dual pullback used for residuals and loads;
    ``hilbert_adjoint``, when present, is the adjoint with respect to the declared
    source and target inner products. Trailing payload axes of ``apply`` and
    ``pullback`` inputs are carried through as one column block.

    Every invariant claim is certified at construction: constant preservation from
    row sums (sparse coefficients, otherwise the action on ones), positivity from
    sparse coefficients only, linear preservation from supplied coordinates, and
    conservation from supplied DOF measures. Tolerances scale with
    ``action_condition``, the relative roundoff amplification of ``primal`` actions
    (for example the condition estimate of an inner mass solve).
    """

    primal: SparseLinearMap | AbstractLinearOperator
    hilbert_adjoint: SparseLinearMap | AbstractLinearOperator | None
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    preserves_constants: bool = eqx.field(static=True)
    preserves_linear: bool = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    positivity_preserving: bool = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        primal: SparseLinearMap | AbstractLinearOperator,
        source_topology_id: str,
        target_topology_id: str,
        /,
        *,
        hilbert_adjoint: SparseLinearMap | AbstractLinearOperator | None = None,
        preserves_constants: bool = False,
        preserves_linear: bool = False,
        conservative: bool = False,
        positivity_preserving: bool = False,
        action_condition: float = 1.0,
        source_coordinates: ArrayLike | None = None,
        target_coordinates: ArrayLike | None = None,
        source_measures: ArrayLike | None = None,
        target_measures: ArrayLike | None = None,
    ) -> None:
        if not isinstance(primal, AbstractLinearOperator):
            raise TypeError("primal must be a SparseLinearMap or linear operator.")
        source_space = primal.source
        target_space = primal.target
        if (
            primal.batch_shape
            or not isinstance(source_space, ArraySpace)
            or not isinstance(target_space, ArraySpace)
            or len(source_space.shape) != 1
            or len(target_space.shape) != 1
        ):
            raise ValueError(
                "Topology transfer primal must be one unbatched target-by-source map."
            )
        source_id = canonical_identifier(source_topology_id, "source_topology_id")
        target_id = canonical_identifier(target_topology_id, "target_topology_id")
        if hilbert_adjoint is not None and not isinstance(
            hilbert_adjoint, AbstractLinearOperator
        ):
            raise TypeError("hilbert_adjoint must be a linear operator or None.")
        if hilbert_adjoint is not None and (
            hilbert_adjoint.source.size != primal.target.size
            or hilbert_adjoint.target.size != primal.source.size
        ):
            raise ValueError("hilbert_adjoint must map target DOFs back to source DOFs.")
        condition = float(action_condition)
        if not isfinite(condition) or condition < 1.0:
            raise ValueError("action_condition must be finite and at least one.")
        claims = tuple(
            bool(value)
            for value in (
                preserves_constants,
                preserves_linear,
                conservative,
                positivity_preserving,
            )
        )
        _certify_claims(
            primal,
            source_space,
            *claims,
            action_condition=condition,
            source_coordinates=source_coordinates,
            target_coordinates=target_coordinates,
            source_measures=source_measures,
            target_measures=target_measures,
        )
        self.primal = primal
        self.hilbert_adjoint = hilbert_adjoint
        self.source_topology_id = source_id
        self.target_topology_id = target_id
        (
            self.preserves_constants,
            self.preserves_linear,
            self.conservative,
            self.positivity_preserving,
        ) = claims
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "finite-element-topology-transfer",
                "source_topology": source_id,
                "target_topology": target_id,
                "primal": primal.operator_id,
                "coefficients": array_tree_fingerprint(np.asarray(primal.coefficients))
                if isinstance(primal, SparseLinearMap)
                else None,
                "hilbert_adjoint": None
                if hilbert_adjoint is None
                else hilbert_adjoint.operator_id,
                "preserves_constants": claims[0],
                "preserves_linear": claims[1],
                "conservative": claims[2],
                "positivity_preserving": claims[3],
            }
        )

    @property
    def source_size(self) -> int:
        return self.primal.source.size

    @property
    def target_size(self) -> int:
        return self.primal.target.size

    def apply(self, values: ArrayLike, /) -> Array:
        """Transfer source coefficients, preserving trailing payload axes."""

        return _block_action(self.primal.mv_block, values)

    def pullback(self, dual: ArrayLike, /) -> Array:
        """Pull target duals back through the algebraic transpose of the primal map."""

        return _block_action(self.primal.transpose_mv_block, dual)


def _row_scales(
    primal: AbstractLinearOperator,
    positivity_preserving: bool,
    dtype: np.dtype,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Row sums and absolute row sums (action magnitude without coefficients)."""

    ones = jnp.ones((primal.source.size,), dtype=dtype)
    row_scale = np.asarray(_block_action(primal.mv_block, ones), dtype=dtype)
    if not isinstance(primal, SparseLinearMap):
        if not np.all(np.isfinite(row_scale)):
            raise ValueError("Topology transfer actions must be finite.")
        if positivity_preserving:
            raise ValueError(
                "Positivity preservation is certified only from sparse coefficients."
            )
        return row_scale, np.abs(row_scale)
    valid = np.asarray(primal.relation.valid, dtype=np.bool_)
    coefficients = np.where(valid, np.asarray(primal.coefficients), 0.0)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Topology transfer coefficients must be finite.")
    if positivity_preserving and np.any(coefficients < 0.0):
        raise ValueError("Transfer claims positivity preservation with negative weights.")
    absolute_scale = np.asarray(
        SparseLinearMap(
            primal.relation,
            jnp.abs(primal.coefficients),
            operator_id=f"{primal.operator_id}:absolute",
        ).mv(ones),
        dtype=dtype,
    )
    return row_scale, absolute_scale


def _certify_claims(
    primal: AbstractLinearOperator,
    source_space: ArraySpace,
    preserves_constants: bool,
    preserves_linear: bool,
    conservative: bool,
    positivity_preserving: bool,
    /,
    *,
    action_condition: float,
    source_coordinates: ArrayLike | None,
    target_coordinates: ArrayLike | None,
    source_measures: ArrayLike | None,
    target_measures: ArrayLike | None,
) -> None:
    dtype = np.dtype(source_space.dtype)
    row_scale, absolute_scale = _row_scales(primal, positivity_preserving, dtype)

    def exceeds(defect: np.ndarray, scale: np.ndarray) -> bool:
        return bool(
            np.any(np.abs(defect) > action_condition * _claim_tolerance(dtype, scale))
        )

    if preserves_constants and exceeds(row_scale - 1.0, absolute_scale):
        raise ValueError(
            "Transfer claims constant preservation but rows do not sum to one."
        )
    if preserves_linear:
        if source_coordinates is None or target_coordinates is None:
            raise ValueError(
                "Linear preservation requires source and target coordinates to certify."
            )
        source = np.asarray(source_coordinates, dtype=dtype)
        target = np.asarray(target_coordinates, dtype=dtype)
        if (
            source.ndim != 2
            or target.ndim != 2
            or source.shape[0] != primal.source.size
            or target.shape != (primal.target.size, source.shape[1])
        ):
            raise ValueError("Transfer coordinates must align with source/target DOFs.")
        mapped = np.asarray(_block_action(primal.mv_block, source), dtype=dtype)
        scale = absolute_scale[:, None] * np.max(np.abs(source), initial=1.0)
        if exceeds(mapped - target, scale):
            raise ValueError("Transfer claims linear preservation but moves coordinates.")
    if conservative:
        if source_measures is None or target_measures is None:
            raise ValueError("Conservation requires source and target DOF measures.")
        source_mass = np.asarray(source_measures, dtype=dtype)
        target_mass = np.asarray(target_measures, dtype=dtype)
        if source_mass.shape != (primal.source.size,) or target_mass.shape != (
            primal.target.size,
        ):
            raise ValueError("Transfer measures must align with source/target DOFs.")
        pulled = np.asarray(
            _block_action(primal.transpose_mv_block, target_mass), dtype=dtype
        )
        scale = np.full(
            pulled.shape, np.sum(np.abs(target_mass)) + np.sum(np.abs(source_mass))
        )
        if exceeds(pulled - source_mass, scale):
            raise ValueError("Transfer claims conservation but changes DOF measures.")


def vertex_interpolation_transfer(
    source_rows: ArrayLike,
    weights: ArrayLike,
    valid: ArrayLike,
    /,
    *,
    source_size: int,
    source_topology_id: str,
    target_topology_id: str,
    hilbert_adjoint: SparseLinearMap | AbstractLinearOperator | None = None,
    preserves_linear: bool = False,
    conservative: bool = False,
    source_coordinates: ArrayLike | None = None,
    target_coordinates: ArrayLike | None = None,
    source_measures: ArrayLike | None = None,
    target_measures: ArrayLike | None = None,
) -> FiniteElementTopologyTransfer:
    """Build one fixed-width row-stencil transfer with O(targets x width) storage.

    Row ``t`` evaluates target DOF ``t`` as the weighted sum of the valid source DOF
    positions ``source_rows[t]``. Constant and positivity preservation are derived
    from the weights; linear preservation and conservation are certified from the
    supplied coordinates and measures.
    """

    rows = np.asarray(source_rows)
    coefficients = np.asarray(weights)
    route_valid = np.asarray(valid)
    if not np.issubdtype(rows.dtype, np.integer):
        raise TypeError("source_rows must have an integer dtype.")
    if route_valid.dtype != np.bool_:
        raise TypeError("valid must be a boolean mask.")
    if not np.issubdtype(coefficients.dtype, np.floating):
        raise TypeError("weights must have a real floating dtype.")
    size = int(source_size)
    if (
        rows.ndim != 2
        or rows.shape[0] == 0
        or rows.shape[1] == 0
        or coefficients.shape != rows.shape
        or route_valid.shape != rows.shape
        or size <= 0
    ):
        raise ValueError("Interpolation rows, weights, and validity must share (n, w).")
    if not np.all(np.any(route_valid, axis=1)):
        raise ValueError("Every interpolation target requires one valid source route.")
    routed = rows[route_valid]
    if np.any(routed < 0) or np.any(routed >= size):
        raise ValueError("Valid interpolation routes must address source DOFs.")
    if not np.all(np.isfinite(coefficients[route_valid])):
        raise ValueError("Valid interpolation weights must be finite.")
    safe_rows = np.where(route_valid, rows, 0).astype(np.int32)
    safe_weights = np.where(route_valid, coefficients, 0.0)
    tolerance = _claim_tolerance(coefficients.dtype, np.sum(np.abs(safe_weights), axis=1))
    preserves_constants = bool(
        np.all(np.abs(np.sum(safe_weights, axis=1) - 1.0) <= tolerance)
    )
    source_id = canonical_identifier(source_topology_id, "source_topology_id")
    target_id = canonical_identifier(target_topology_id, "target_topology_id")
    relation = RowRelation(safe_rows, source_size=size, valid=route_valid)
    primal = SparseLinearMap(
        relation,
        jnp.asarray(safe_weights),
        operator_id=canonical_fingerprint(
            {
                "kind": "finite-element-vertex-interpolation",
                "source_topology": source_id,
                "target_topology": target_id,
                "source_size": size,
                "source_rows": array_tree_fingerprint(safe_rows),
                "valid": array_tree_fingerprint(route_valid),
            }
        ),
    )
    return FiniteElementTopologyTransfer(
        primal,
        source_id,
        target_id,
        hilbert_adjoint=hilbert_adjoint,
        preserves_constants=preserves_constants,
        preserves_linear=preserves_linear,
        conservative=conservative,
        positivity_preserving=bool(np.all(safe_weights >= 0.0)),
        source_coordinates=source_coordinates,
        target_coordinates=target_coordinates,
        source_measures=source_measures,
        target_measures=target_measures,
    )


# Target mass policy: the fill-reducing ordering and symbolic Cholesky pattern are
# planned once per target DOF structure and reused by every numeric refresh.
_TARGET_MASS_FACTORIZATION = SparseFactorizationPolicy(
    "cholesky", ordering="reverse-cuthill-mckee"
)
_L2_ARTIFACT_TOKEN = object()


@final
class PreparedL2ProjectionTarget(StrictModule, NonTrainableState):
    """Prepared target space of Galerkin L2 projections onto one FE field.

    Built by :func:`prepare_l2_projection_target`. Owns the target mass ``M_T`` of
    the scalar Lagrange field ``field_name`` of ``discretization``, its sparse
    Cholesky ``factorization`` (the symbolic reverse Cuthill-McKee ordering and
    fill pattern in ``factorization.plan``, the numeric factor, its status and
    pivot diagnostics), the Golub-Kahan estimate ``mass_condition``, and
    ``dof_measures`` (integrals of the target basis functions). One artifact serves
    every source field and every common refinement onto this target, and payload
    axes of its transfers are solved as one multi-right-hand-side block.

    ``structure_id`` identifies the target DOF structure and symbolic factor
    pattern; it is unchanged by :func:`refresh_l2_projection_target`, and the
    compiled factorization and projection kernels depend only on this structure.
    ``target_id`` additionally identifies the target geometry and field.
    """

    discretization: FiniteElementDiscretization
    mass: SparseLinearMap
    factorization: PreparedSparseFactorization
    mass_condition: SpectralEstimate
    dof_measures: Array
    field_name: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteElementDiscretization,
        mass: SparseLinearMap,
        factorization: PreparedSparseFactorization,
        mass_condition: SpectralEstimate,
        dof_measures: ArrayLike,
        /,
        *,
        field_name: str,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _L2_ARTIFACT_TOKEN:
            raise TypeError(
                "PreparedL2ProjectionTarget is constructed by "
                "prepare_l2_projection_target or refresh_l2_projection_target."
            )
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be a FiniteElementDiscretization.")
        if not isinstance(mass, SparseLinearMap):
            raise TypeError("mass must be a SparseLinearMap.")
        if not isinstance(factorization, PreparedSparseFactorization):
            raise TypeError("factorization must be a PreparedSparseFactorization.")
        if not isinstance(mass_condition, SpectralEstimate):
            raise TypeError("mass_condition must be a SpectralEstimate.")
        name = str(field_name)
        dof_map = discretization.dof_maps[discretization._field_index(name)]
        size = dof_map.global_dof_count
        measures = jnp.asarray(dof_measures)
        if (
            mass.batch_shape
            or mass.input_shape != (size,)
            or mass.output_shape != (size,)
            or factorization.plan.shape != (size, size)
            or factorization.batch_shape
            or factorization.plan.kind != "cholesky"
            or measures.shape != (size,)
        ):
            raise ValueError(
                "The target mass, its Cholesky factorization, and the DOF measures "
                f"must be unbatched over the {size} DOFs of field {name!r}."
            )
        if not (
            mass.properties.certifies("self_adjoint")
            and mass.properties.certifies("positive_definite")
        ):
            raise ValueError(
                "The target mass must certify self-adjoint positive definiteness."
            )
        # Preparation boundary: one host decision on the prepared factor status.
        status = SparseFactorizationStatus(int(factorization.status))
        if status is not SparseFactorizationStatus.SUCCESS:
            raise ValueError(
                f"Target mass Cholesky factorization failed ({status.name})."
            )
        if not isfinite(float(mass_condition.value)):
            raise ValueError("The target mass condition estimate is not finite.")
        self.discretization = discretization
        self.mass = mass
        self.factorization = factorization
        self.mass_condition = mass_condition
        self.dof_measures = measures
        self.field_name = name
        self.structure_id = canonical_fingerprint(
            {
                "kind": "finite-element-l2-projection-target-structure",
                "dof_map": dof_map.dof_map_id,
                "factorization_plan": factorization.plan.plan_id,
            }
        )
        self.target_id = canonical_fingerprint(
            {
                "kind": "finite-element-l2-projection-target",
                "structure": self.structure_id,
                "target": discretization.prepared_id,
                "field": name,
            }
        )


@final
class FiniteElementL2Projection(AbstractLinearOperator):
    """Galerkin L2 projection ``M_T^{-1} B`` between non-matching FE spaces.

    ``mixed_mass`` is ``B`` (target DOFs by source DOFs, integrals of target times
    source basis functions over the common refinement) and ``prepared_target``
    owns the target mass ``M_T``, its prepared sparse Cholesky factor, and the
    solve evidence (factor status, pivot diagnostics, condition estimate). ``mv``
    solves ``M_T x = B u`` and ``transpose_mv`` applies the algebraic transpose
    ``B^T M_T^{-1}``; trailing payload axes are solved as one
    multi-right-hand-side block.
    """

    _fused_block_action_kind = "fused"

    mixed_mass: SparseLinearMap
    prepared_target: PreparedL2ProjectionTarget

    def __init__(
        self,
        mixed_mass: SparseLinearMap,
        prepared_target: PreparedL2ProjectionTarget,
        /,
        *,
        operator_id: str,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _L2_ARTIFACT_TOKEN:
            raise TypeError(
                "FiniteElementL2Projection is constructed by "
                "prepare_l2_projection_transfer."
            )
        if not isinstance(mixed_mass, SparseLinearMap):
            raise TypeError("mixed_mass must be a SparseLinearMap.")
        if not isinstance(prepared_target, PreparedL2ProjectionTarget):
            raise TypeError("prepared_target must be a PreparedL2ProjectionTarget.")
        if (
            mixed_mass.batch_shape
            or len(mixed_mass.input_shape) != 1
            or mixed_mass.output_shape != prepared_target.mass.output_shape
        ):
            raise ValueError(
                "L2 projection needs an unbatched mixed mass from source DOFs onto "
                "the prepared target DOFs."
            )
        identifier = canonical_identifier(operator_id, "operator_id")
        self.mixed_mass = mixed_mass
        self.prepared_target = prepared_target
        self.source = mixed_mass.source
        self.target = mixed_mass.target
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True
        )
        self.batch_shape = ()
        self.operator_id = identifier

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        mixed = _mixed_mass_action(
            self.mixed_mass.relation, self.mixed_mass.coefficients, jnp.asarray(vector)
        )
        return _target_mass_solve(self.prepared_target.factorization, mixed)

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        # M_T is symmetric, so the transpose solve reuses the same factor.
        solved = _target_mass_solve(
            self.prepared_target.factorization, jnp.asarray(vector)
        )
        return _mixed_mass_transpose_action(
            self.mixed_mass.relation, self.mixed_mass.coefficients, solved
        )

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        return _materialize_by_basis(self)


# Module-level compiled kernels. Their arguments carry only structural static
# metadata (relation sizes, the symbolic factor plan), never numeric identities,
# so a compilation depends on shapes and structure alone. The level-scheduled
# triangular solves dominate compilation and are keyed by the target structure
# and payload width only: every transfer onto a prepared target, and every
# refresh of it, reuses them; a new transfer compiles only its sparse gathers.
@eqx.filter_jit
def _target_mass_solve(
    factorization: PreparedSparseFactorization, values: Array, /
) -> Array:
    block = values.reshape((values.shape[0], -1))
    return factorization.solve(block).value.reshape(values.shape)


_mixed_mass_action = eqx.filter_jit(linear_apply)
_mixed_mass_transpose_action = eqx.filter_jit(linear_transpose_apply)

# Numeric Cholesky of the target mass on a fixed symbolic plan. The mass routes
# are traced here, so the native refresh gathers values through the plan's
# retained route-to-CSR scatter instead of revalidating the pattern on the host.
_factor_target_mass = eqx.filter_jit(refresh_sparse_factorization)
_mass_condition = eqx.filter_jit(estimate_condition_number)


def _validated_overlaps(
    refinement: PreparedCommonRefinement,
    source: CellMesh,
    target: CellMesh,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overlap simplices ``(S, d+1, d)`` with their source and target cells."""

    from ...geometry import CommonRefinementStatus, PreparedCommonRefinement

    if not isinstance(refinement, PreparedCommonRefinement):
        raise TypeError("refinement must be a PreparedCommonRefinement.")
    if not refinement.succeeded:
        raise ValueError(
            "L2 projection requires a successful common refinement; got "
            f"{CommonRefinementStatus(refinement.status).name}: "
            f"{refinement.evidence.reason}"
        )
    if refinement.source_mesh_id != source.mesh_id or (
        refinement.target_mesh_id != target.mesh_id
    ):
        raise ValueError(
            "The common refinement does not join the source and target meshes."
        )
    if refinement.simplices is None or refinement.simplex_offsets is None:
        raise ValueError(
            "L2 projection requires CommonRefinementPolicy(overlap_simplices=True)."
        )
    offsets = np.asarray(refinement.simplex_offsets, dtype=np.int64)
    simplices = np.asarray(refinement.simplices, dtype=np.float64)
    if simplices.shape[0] == 0:
        raise ValueError("The common refinement has no overlap simplices.")
    entries = np.repeat(np.arange(offsets.size - 1, dtype=np.int64), np.diff(offsets))
    return (
        simplices,
        np.asarray(refinement.source_cells, dtype=np.int64)[entries],
        np.asarray(refinement.target_cells, dtype=np.int64)[entries],
    )


def _projection_elements(
    discretization: FiniteElementDiscretization,
    field_index: int,
    role: str,
    /,
) -> tuple[FiniteElementSpec, ...]:
    """Scalar Lagrange elements on affine simplices matching the refined mesh."""

    mesh = discretization.mesh
    coordinates = np.asarray(discretization.default_runtime.coordinates)
    mesh_coordinates = np.asarray(mesh.coordinates)
    elements = discretization.elements[field_index]
    for block, element, coordinate_element, routes in zip(
        mesh.blocks,
        elements,
        discretization.coordinate_elements,
        discretization.coordinate_dofs,
        strict=True,
    ):
        if (
            element.family not in _LAGRANGE_FAMILIES
            or element.cell_kind not in _SIMPLEX_KINDS.values()
            or element.mapping != "identity"
            or element.value_shape
        ):
            raise ValueError(
                "L2 projection supports scalar Lagrange elements on triangles and "
                f"tetrahedra; {role} block {block.name!r} has {element.family} on "
                f"{element.cell_kind}."
            )
        if (
            coordinate_element.element_id
            != lagrange_element(block.cell_kind, 1).element_id
        ):
            raise ValueError(
                f"L2 projection requires affine geometry; {role} block {block.name!r} "
                "uses a higher-order coordinate map."
            )
        if not np.array_equal(
            coordinates[np.asarray(routes)], mesh_coordinates[np.asarray(block.vertices)]
        ):
            raise ValueError(f"The {role} FE geometry differs from its refined mesh.")
    return elements


# Host-only immutable preparation: the affine cell maps, overlap pullbacks, and
# element integrals below are NumPy (batched LAPACK determinants/solves of d x d
# frames, d <= 3). The runtime action is the prepared sparse factorization.


def _affine_frames(
    discretization: FiniteElementDiscretization, /
) -> tuple[np.ndarray, np.ndarray]:
    """Origins ``(C, d)`` and edge-column frames ``(C, d, d)`` of every cell.

    Cells follow the concatenated block order; the P1 coordinate element (checked
    by ``_projection_elements``) maps ``x = origin + frame @ xi``.
    """

    coordinates = np.asarray(discretization.default_runtime.coordinates, np.float64)
    vertices = np.concatenate(
        tuple(
            coordinates[np.asarray(routes)] for routes in discretization.coordinate_dofs
        )
    )
    return vertices[:, 0, :], np.swapaxes(vertices[:, 1:, :] - vertices[:, :1, :], -1, -2)


@eqx.filter_jit
def _tabulate_values(element: FiniteElementSpec, points: Array, /) -> Array:
    return element.tabulate(points)[0]


def _oriented_basis(
    element: FiniteElementSpec, orientation: np.ndarray, points: np.ndarray, /
) -> np.ndarray:
    """Basis values ``(..., Q, n)`` at reference points ``(..., Q, d)``.

    The element tabulation runs as one compiled call instead of an eager dispatch
    per primitive. Points are padded with the reference origin to a power-of-two
    count, so overlap sets of varying size (every new refinement) reuse a
    logarithmic number of compilations; the padded rows are discarded.
    """

    flat = points.reshape((-1, points.shape[-1]))
    padded = np.zeros(
        (1 << max(flat.shape[0] - 1, 0).bit_length(), flat.shape[1]), dtype=np.float64
    )
    padded[: flat.shape[0]] = flat
    # ty: ignore[invalid-argument-type]
    values = np.asarray(_tabulate_values(element, padded), dtype=np.float64)
    basis = values[: flat.shape[0]].reshape(
        points.shape[:-1] + (element.local_dof_count,)
    )
    return basis * orientation[..., None, :]


def _overlap_quadrature(
    simplices: np.ndarray, degree: int, /
) -> tuple[np.ndarray, np.ndarray, str]:
    """Physical points ``(S, Q, d)`` and weights ``(S, Q)`` exact to ``degree``."""

    dimension = simplices.shape[-1]
    # ty: ignore[invalid-argument-type]
    rule = cubature_rule_data(_SIMPLEX_KINDS[dimension], degree)
    frames = np.swapaxes(simplices[:, 1:, :] - simplices[:, :1, :], -1, -2)
    # Overlap simplices are fanned and positively oriented up to rounding; the
    # signed measure keeps each fan an exact partition of its overlap.
    measures = np.linalg.det(frames)
    points = simplices[:, None, 0, :] + np.asarray(rule.points)[None] @ np.swapaxes(
        frames, -1, -2
    )
    weights = measures[:, None] * np.asarray(rule.weights)[None, :]
    return points, weights, rule.rule_id


def _overlap_basis(
    discretization: FiniteElementDiscretization,
    field_index: int,
    cells: np.ndarray,
    points: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Padded DOF routes, validity, and oriented basis values of the owning cells."""

    dof_map = discretization.dof_maps[field_index]
    blocks = discretization.mesh.blocks
    starts = np.cumsum((0,) + tuple(block.cell_count for block in blocks))
    width = max(routes.shape[1] for routes in dof_map.cell_dofs)
    count, quadrature, _ = points.shape
    routes = np.zeros((count, width), dtype=np.int32)
    valid = np.zeros((count, width), dtype=np.bool_)
    values = np.zeros((count, quadrature, width), dtype=np.float64)
    origins, frames = _affine_frames(discretization)
    # x = origin + frame @ xi on the owning affine cell; one batched solve pulls
    # every quadrature point of an overlap simplex back to reference coordinates.
    reference = np.linalg.solve(
        frames[cells][:, None], (points - origins[cells][:, None, :])[..., None]
    )[..., 0]
    for block_index, element in enumerate(discretization.elements[field_index]):
        rows = np.flatnonzero(
            (cells >= starts[block_index]) & (cells < starts[block_index + 1])
        )
        if rows.size == 0:
            continue
        local = cells[rows] - starts[block_index]
        size = element.local_dof_count
        routes[rows, :size] = np.asarray(dof_map.cell_dofs[block_index])[local]
        valid[rows, :size] = True
        values[rows, :, :size] = _oriented_basis(
            element,
            np.asarray(dof_map.orientations[block_index])[local],
            reference[rows],
        )
    return routes, valid, values


def _cell_integrals(
    discretization: FiniteElementDiscretization,
    field_index: int,
    /,
    *,
    mass: bool,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    """Integrals of every global basis function and, optionally, mass COO triples.

    The per-block rule is exact for degree ``2p`` (mass) or ``p`` (integrals).
    """

    dof_map = discretization.dof_maps[field_index]
    _, frames = _affine_frames(discretization)
    determinants = np.abs(np.linalg.det(frames))
    size = dof_map.global_dof_count
    measures = np.zeros((size,), dtype=np.float64)
    rows, columns, values = [], [], []
    start = 0
    for block_index, element in enumerate(discretization.elements[field_index]):
        routes = np.asarray(dof_map.cell_dofs[block_index])
        stop = start + routes.shape[0]
        # ty: ignore[invalid-argument-type]
        rule = cubature_rule_data(element.cell_kind, (2 if mass else 1) * element.degree)
        basis = _oriented_basis(
            element,
            np.asarray(dof_map.orientations[block_index]),
            np.asarray(rule.points)[None],
        )
        weights = determinants[start:stop, None] * np.asarray(rule.weights)[None, :]
        weighted = weights[:, :, None] * basis
        measures += np.bincount(
            routes.reshape((-1,)),
            weights=np.sum(weighted, axis=1).reshape((-1,)),
            minlength=size,
        )
        if mass:
            local = np.swapaxes(weighted, 1, 2) @ basis
            rows.append(np.broadcast_to(routes[:, :, None], local.shape).reshape((-1,)))
            columns.append(
                np.broadcast_to(routes[:, None, :], local.shape).reshape((-1,))
            )
            values.append(local.reshape((-1,)))
        start = stop
    if not mass:
        return measures, None
    return measures, (
        np.concatenate(rows),
        np.concatenate(columns),
        np.concatenate(values),
    )


def _coalesced_map(
    target_routes: np.ndarray,
    source_routes: np.ndarray,
    values: np.ndarray,
    /,
    *,
    target_size: int,
    source_size: int,
    properties: OperatorProperties,
    operator_id: str,
) -> SparseLinearMap:
    """Sum duplicate routes in canonical (target, source) order.

    The stable sort keeps each duplicate group in input order, so the reduction is
    deterministic for a deterministic input ordering.
    """

    order = np.lexsort((source_routes, target_routes))
    targets = target_routes[order]
    sources = source_routes[order]
    first = np.ones((order.size,), dtype=np.bool_)
    first[1:] = (np.diff(targets) != 0) | (np.diff(sources) != 0)
    starts = np.flatnonzero(first)
    relation = EdgeRelation(
        sources[starts].astype(np.int32),
        targets[starts].astype(np.int32),
        source_size=source_size,
        target_size=target_size,
    )
    return SparseLinearMap(
        relation,
        jnp.asarray(np.add.reduceat(values[order], starts)),
        properties=properties,
        operator_id=operator_id,
    )


def _target_mass(
    target: FiniteElementDiscretization, field_index: int, /
) -> tuple[SparseLinearMap, np.ndarray]:
    """Target mass ``M_T`` and the integrals of every target basis function."""

    measures, triples = _cell_integrals(target, field_index, mass=True)
    dof_map = target.dof_maps[field_index]
    mass = _coalesced_map(
        # ty: ignore[not-iterable]
        *triples,
        target_size=dof_map.global_dof_count,
        source_size=dof_map.global_dof_count,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        ),
        # Structural identity: the mass of one DOF map keeps its operator (and
        # compiled-kernel) identity across numeric geometry refreshes.
        operator_id=canonical_fingerprint(
            {"kind": "finite-element-scalar-mass", "dof_map": dof_map.dof_map_id}
        ),
    )
    return mass, measures


def _factored_target(
    target: FiniteElementDiscretization,
    field_name: str,
    plan: SparseFactorizationPlan | None,
    /,
) -> PreparedL2ProjectionTarget:
    """Assemble, factor, and condition-estimate the target mass on one plan.

    ``plan`` is ``None`` for a cold preparation (symbolic analysis of the fresh
    pattern) and the retained symbolic plan for a numeric refresh.
    """

    mass, measures = _target_mass(target, target._field_index(field_name))
    symbolic = (
        prepare_sparse_factorization(mass, _TARGET_MASS_FACTORIZATION)
        if plan is None
        else plan
    )
    return PreparedL2ProjectionTarget(
        target,
        mass,
        _factor_target_mass(symbolic, mass),
        _mass_condition(mass),
        measures,
        field_name=field_name,
        _construction_token=_L2_ARTIFACT_TOKEN,
    )


def prepare_l2_projection_target(
    target: FiniteElementDiscretization,
    /,
    *,
    field_name: str,
) -> PreparedL2ProjectionTarget:
    """Prepare the target space of Galerkin L2 projections onto one FE field.

    ``field_name`` names a scalar Lagrange field (continuous or discontinuous, any
    degree) of ``target`` on affine triangles or tetrahedra. The target mass is
    integrated exactly, its symbolic Cholesky pattern is planned under a reverse
    Cuthill-McKee ordering, and the numeric factor and condition estimate are
    computed once. Pass the result to :func:`prepare_l2_projection_transfer` for
    every source field and common refinement onto this target, and to
    :func:`refresh_l2_projection_target` when the target geometry moves with an
    unchanged DOF structure. Unsupported elements, a failed factorization, and a
    non-finite condition estimate raise ``ValueError``.
    """

    if not isinstance(target, FiniteElementDiscretization):
        raise TypeError("target must be a FiniteElementDiscretization.")
    name = str(field_name)
    _projection_elements(target, target._field_index(name), "target")
    return _factored_target(target, name, None)


def refresh_l2_projection_target(
    prepared: PreparedL2ProjectionTarget,
    target: FiniteElementDiscretization,
    /,
) -> PreparedL2ProjectionTarget:
    """Refactor a prepared L2 projection target for moved target geometry.

    ``target`` must carry the prepared field with the same DOF structure (equal
    ``dof_map_id``: mesh topology, elements, and DOF routes), for example the same
    finite-element plan prepared on a moved mesh. Only numeric values change: the
    target mass is reassembled, the retained symbolic plan is refactored through
    the native sparse factorization refresh, and the compiled factorization and
    projection kernels are reused. A changed DOF structure raises ``ValueError``;
    prepare a new target instead.
    """

    if not isinstance(prepared, PreparedL2ProjectionTarget):
        raise TypeError("prepared must be a PreparedL2ProjectionTarget.")
    if not isinstance(target, FiniteElementDiscretization):
        raise TypeError("target must be a FiniteElementDiscretization.")
    name = prepared.field_name
    index = target._field_index(name)
    _projection_elements(target, index, "target")
    previous = prepared.discretization
    if (
        target.dof_maps[index].dof_map_id
        != previous.dof_maps[previous._field_index(name)].dof_map_id
    ):
        raise ValueError(
            "Refreshing an L2 projection target requires an unchanged DOF structure; "
            "prepare a new target instead."
        )
    return _factored_target(target, name, prepared.factorization.plan)


def _coverage_bound(refinement: PreparedCommonRefinement, /) -> float:
    """Largest accepted relative coverage defect of any source or target cell."""

    evidence = refinement.evidence
    return max(
        float(
            np.max(
                np.asarray(evidence.source_coverage_tolerances)
                / np.asarray(refinement.source_measures)
            )
        ),
        float(
            np.max(
                np.asarray(evidence.target_coverage_tolerances)
                / np.asarray(refinement.target_measures)
            )
        ),
    )


def _coverage_claims(refinement: PreparedCommonRefinement, /) -> tuple[bool, bool]:
    """Whether every target cell, and every source cell, is certified covered."""

    from ...geometry import CommonRefinementCoverage

    match refinement.policy.coverage:
        case CommonRefinementCoverage.COMPLETE:
            return True, True
        case CommonRefinementCoverage.TARGET:
            return True, False
        case CommonRefinementCoverage.SOURCE:
            return False, True
        case CommonRefinementCoverage.PARTIAL:
            return False, False
        case coverage:
            raise ValueError(f"Unknown common-refinement coverage {coverage!r}.")


def prepare_l2_projection_transfer(
    source: FiniteElementDiscretization,
    target: PreparedL2ProjectionTarget,
    refinement: PreparedCommonRefinement,
    /,
    *,
    field_name: str,
) -> FiniteElementTopologyTransfer:
    """Prepare the Galerkin L2 projection of one FE field onto a non-matching mesh.

    ``target`` is a :func:`prepare_l2_projection_target` artifact and
    ``refinement`` the common refinement of ``source.mesh`` and the target mesh
    prepared with ``CommonRefinementPolicy(overlap_simplices=True)``. The source
    field ``field_name`` is a scalar Lagrange field (continuous or discontinuous,
    any degree) on affine triangles or tetrahedra; component axes of the field
    are carried as payload. Only the mixed mass ``B`` is assembled here, exactly
    on the overlap simplices with a rule for the product of source and target
    degrees; the prepared target factor is shared (``FiniteElementL2Projection``).
    The transfer applies ``M_T^{-1} B`` and pulls duals back through
    ``B^T M_T^{-1}``.

    Claims follow the certified coverage: constants (and, for degrees >= 1, linear
    fields) are preserved when every target cell is covered, and the integral is
    conserved when every source cell is covered. Claims are certified to the
    refinement's accepted coverage accuracy (its per-cell coverage tolerances),
    amplified by the basis Lebesgue constant and the target mass condition
    estimate. Positivity is never claimed. Failed or mismatched refinements and
    unsupported elements raise ``ValueError``.
    """

    if not isinstance(source, FiniteElementDiscretization):
        raise TypeError("source must be a FiniteElementDiscretization.")
    if not isinstance(target, PreparedL2ProjectionTarget):
        raise TypeError("target must be a PreparedL2ProjectionTarget.")
    source_name = str(field_name)
    source_index = source._field_index(source_name)
    space = target.discretization
    target_index = space._field_index(target.field_name)
    simplices, source_cells, target_cells = _validated_overlaps(
        refinement, source.mesh, space.mesh
    )
    source_elements = _projection_elements(source, source_index, "source")
    target_elements = space.elements[target_index]
    points, weights, rule_id = _overlap_quadrature(
        simplices,
        max(element.degree for element in source_elements)
        + max(element.degree for element in target_elements),
    )
    target_routes, target_valid, target_basis = _overlap_basis(
        space, target_index, target_cells, points
    )
    source_routes, source_valid, source_basis = _overlap_basis(
        source, source_index, source_cells, points
    )
    local = np.swapaxes(weights[:, :, None] * target_basis, 1, 2) @ source_basis
    routed = target_valid[:, :, None] & source_valid[:, None, :]
    source_dofs = source.dof_maps[source_index]
    target_dofs = space.dof_maps[target_index]
    identity = {
        "refinement": refinement.refinement_id,
        "source": source.prepared_id,
        "source_field": source_name,
        "target": target.target_id,
        "rule": rule_id,
    }
    mixed_mass = _coalesced_map(
        np.broadcast_to(target_routes[:, :, None], local.shape)[routed],
        np.broadcast_to(source_routes[:, None, :], local.shape)[routed],
        local[routed],
        target_size=target_dofs.global_dof_count,
        source_size=source_dofs.global_dof_count,
        properties=OperatorProperties(),
        operator_id=canonical_fingerprint(
            {"kind": "finite-element-mixed-mass", **identity}
        ),
    )
    projection = FiniteElementL2Projection(
        mixed_mass,
        target,
        operator_id=canonical_fingerprint(
            {
                "kind": "finite-element-l2-projection",
                **identity,
                "mixed_mass": array_tree_fingerprint(np.asarray(mixed_mass.coefficients)),
            }
        ),
        _construction_token=_L2_ARTIFACT_TOKEN,
    )
    target_covered, source_covered = _coverage_claims(refinement)
    linear = (
        target_covered
        and min(element.degree for element in source_elements) >= 1
        and min(element.degree for element in target_elements) >= 1
    )
    # Claims hold to the accuracy the refinement certified: an accepted relative
    # coverage defect r perturbs B by at most r times the basis Lebesgue constant,
    # and the target solve amplifies that by the mass condition.
    lebesgue = max(
        float(np.max(np.sum(np.abs(target_basis), axis=-1))),
        float(np.max(np.sum(np.abs(source_basis), axis=-1))),
    )
    coverage = _coverage_bound(refinement) / (_CLAIM_ULPS * np.finfo(np.float64).eps)
    condition = float(target.mass_condition.value)
    return FiniteElementTopologyTransfer(
        projection,
        source.mesh.topology_id,
        space.mesh.topology_id,
        preserves_constants=target_covered,
        preserves_linear=linear,
        conservative=source_covered,
        action_condition=max(condition, 1.0) * max(lebesgue, 1.0) * (1.0 + coverage),
        source_coordinates=source_dofs.dof_coordinates if linear else None,
        target_coordinates=target_dofs.dof_coordinates if linear else None,
        source_measures=_cell_integrals(source, source_index, mass=False)[0]
        if source_covered
        else None,
        target_measures=target.dof_measures if source_covered else None,
    )


__all__ = [
    "FiniteElementL2Projection",
    "FiniteElementTopologyTransfer",
    "PreparedL2ProjectionTarget",
    "prepare_l2_projection_target",
    "prepare_l2_projection_transfer",
    "refresh_l2_projection_target",
    "vertex_interpolation_transfer",
]
