#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from math import isfinite
from typing import Any, final, Literal, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from ..._coordinate_measure import (
    CoordinateFrame,
    CoordinateMeasure,
    CoordinateMeasurePlan,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._polynomial._cubature import cubature_rule_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...linalg import (
    AbstractLinearOperator,
    adjoint,
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
    transpose,
)
from ...linalg._operators import _generic_adjoint, _materialize_by_basis
from ...linalg._sparse_rank import sparse_support_rank, SparseSupportRank
from ...sparse import (
    EdgeRelation,
    linear_apply,
    linear_transpose_apply,
    RowRelation,
    SparseCoordinateOperator,
    SparseLinearMap,
)
from ...typing import checked, parse
from .._cell_mesh import CellMesh
from .._spaces import DiscreteFieldSpace
from .._topology_epoch import TopologyEpoch, TopologyEpochTransition
from .._transfer import FieldTransfer, TransferProperties
from ._generic import FiniteElementDiscretization
from ._reference import FiniteElementSpec, lagrange_element


if TYPE_CHECKING:
    from ...geometry import PreparedCommonRefinement
    from .._distributed_field import DistributedSparseTransfer


# Invariant claims are certified in the coefficient dtype up to this many ulps of
# the local row scale.
_CLAIM_ULPS = 64.0

_SIMPLEX_KINDS = {2: "triangle", 3: "tetrahedron"}
_LAGRANGE_FAMILIES = ("Lagrange", "DiscontinuousLagrange")


def _claim_tolerance(dtype: np.dtype, scale: np.ndarray, /) -> np.ndarray:
    return _CLAIM_ULPS * np.finfo(dtype).eps * np.maximum(scale, 1.0)


def _measure_defect_bound(
    dtype: np.dtype,
    action_condition: float,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    /,
) -> float:
    """Certified per-DOF bound on ``|P^T target_mass - source_mass|``.

    Conservation is a relative statement about the measures themselves, so the
    bound scales with their total magnitude and carries no unit floor (DOF
    measures of 1e-9 m^3 are as conservative as those of 1 m^3).
    """
    total = np.sum(np.abs(target_mass)) + np.sum(np.abs(source_mass))
    return float(action_condition * _CLAIM_ULPS * np.finfo(dtype).eps * total)


def _block_action(action: Callable[[Array], Array], values: ArrayLike, /) -> Array:
    """Apply a canonical-column action to ``(dofs, *payload)`` values."""

    array = jnp.asarray(values)
    if array.ndim == 0:
        raise ValueError("Transfer inputs require a leading DOF axis.")
    block = action(array.reshape((array.shape[0], -1)))
    return block.reshape((block.shape[0],) + array.shape[1:])


def _project_exact_transfer_values(
    values: list[Fraction], dtype: np.dtype, /
) -> np.ndarray:
    """Project exact nonzero entries without silently losing their support."""
    try:
        projected = np.asarray([float(value) for value in values], dtype=dtype)
    except OverflowError as error:
        raise ValueError("Exact transfer coefficients exceed execution range.") from error
    if not np.all(np.isfinite(projected)) or any(
        value != 0 and projected[index] == 0 for index, value in enumerate(values)
    ):
        raise ValueError("Exact transfer coefficients lose finite execution support.")
    return projected


type NestedNodalTransferSemantics = Literal["interpolate", "density", "intensive"]


def _sparse_transfer_support(primal: AbstractLinearOperator, /) -> SparseSupportRank:
    if not isinstance(primal, (SparseLinearMap, SparseCoordinateOperator)):
        raise ValueError("Support reporting requires an explicitly sparse primal map.")
    if isinstance(primal, SparseCoordinateOperator) and primal.block_shape is not None:
        raise ValueError("Support reporting requires scalar sparse coordinates.")
    return sparse_support_rank(primal.sparse_storage())


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
    action_condition: float = eqx.field(static=True)
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
        self.action_condition = condition
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
                "action_condition": condition,
            }
        )

    @property
    def source_size(self) -> int:
        return self.primal.source.size

    @property
    def target_size(self) -> int:
        return self.primal.target.size

    def support_report(self, /) -> SparseSupportRank:
        """Prepare host sparse support and rank bounds without a global dense matrix.

        The singleton-row lower bound and structural upper bound certify the rank
        of the stored execution matrix. ``exact_rank`` is set only when they meet.
        """
        return _sparse_transfer_support(self.primal)

    def distributed(
        self,
        source_owner: ArrayLike,
        target_owner: ArrayLike,
        part_count: int,
        /,
    ) -> DistributedSparseTransfer:
        """Prepare owner-local primal/transpose routes when the map is sparse."""
        from .._distributed_field import DistributedSparseTransfer

        if not isinstance(self.primal, (SparseLinearMap, SparseCoordinateOperator)):
            raise ValueError(
                "Owner-local topology transfer requires an explicit sparse primal map."
            )
        if (
            isinstance(self.primal, SparseCoordinateOperator)
            and self.primal.block_shape is not None
        ):
            raise ValueError(
                "Owner-local topology transfer requires scalar sparse coordinates."
            )
        return DistributedSparseTransfer(
            self.primal.relation,
            self.primal.coefficients,
            source_owner,
            target_owner,
            part_count,
            operator_id=self.transfer_id,
        )

    def apply(self, values: ArrayLike, /) -> Array:
        """Transfer source coefficients, preserving trailing payload axes."""

        return _block_action(self.primal.mv_block, values)

    def pullback(self, dual: ArrayLike, /) -> Array:
        """Pull target duals back through the algebraic transpose of the primal map."""

        return _block_action(self.primal.transpose_mv_block, dual)

    def epoch_transition(
        self,
        source_field: DiscreteFieldSpace,
        target_field: DiscreteFieldSpace,
        source_epoch: TopologyEpoch,
        target_epoch: TopologyEpoch,
        source_measures: ArrayLike,
        target_measures: ArrayLike,
        /,
    ) -> TopologyEpochTransition:
        """Bind this certified transfer as an explicit topology-epoch transition.

        The epochs must realize exactly this transfer's source and target
        topologies. Field-transfer properties are this artifact's certified claims,
        never caller assertions; the raw transpose is the dual pullback and the
        Hilbert adjoint is the declared one, else the adjoint under the coefficient
        pairings. Only conservative transfers qualify; the measures are the DOF
        integrals the transition reports content against, and they must satisfy
        this transfer's conservation certificate, whose per-DOF measure-defect
        bound the transition carries into its content tolerance.
        """

        if not isinstance(source_epoch, TopologyEpoch) or not isinstance(
            target_epoch, TopologyEpoch
        ):
            raise TypeError("Epoch transitions need TopologyEpoch endpoints.")
        if (
            source_epoch.topology_id != self.source_topology_id
            or target_epoch.topology_id != self.target_topology_id
        ):
            raise ValueError("Topology epochs do not realize this transfer's topologies.")
        if not self.conservative:
            raise ValueError(
                "Only a certified conservative transfer forms a topology transition."
            )
        source_space = self.primal.source
        if not isinstance(source_space, ArraySpace):
            raise RuntimeError("Topology transfer primal lost its ArraySpace source.")
        dtype = np.dtype(source_space.dtype)
        source_mass = np.asarray(source_measures, dtype=dtype)
        target_mass = np.asarray(target_measures, dtype=dtype)
        if source_mass.shape != (self.source_size,) or target_mass.shape != (
            self.target_size,
        ):
            raise ValueError("Transition measures must align with source/target DOFs.")
        bound = _measure_defect_bound(
            dtype, self.action_condition, source_mass, target_mass
        )
        pulled = np.asarray(
            _block_action(self.primal.transpose_mv_block, target_mass), dtype=dtype
        )
        if np.any(np.abs(pulled - source_mass) > bound):
            raise ValueError(
                "Transition measures are not conserved by this transfer within its "
                "certified bound."
            )
        exact_on = ("constants", "linears") if self.preserves_linear else ("constants",)
        transfer = FieldTransfer(
            source_field,
            target_field,
            self.primal,
            dual_pullback_operator=transpose(self.primal),
            hilbert_adjoint_operator=(
                adjoint(self.primal)
                if self.hilbert_adjoint is None
                else self.hilbert_adjoint
            ),
            properties=TransferProperties(
                constant_preserving=self.preserves_constants,
                conservative=True,
                positivity_preserving=self.positivity_preserving,
                adjoint_paired=True,
                differentiable_geometry=False,
                exact_on=exact_on if self.preserves_constants else (),
            ),
            transfer_id=self.transfer_id,
        )
        return TopologyEpochTransition(
            source_epoch,
            target_epoch,
            transfer,
            source_measures,
            target_measures,
            measure_defect_bound=bound,
        )


def _row_scales(
    primal: AbstractLinearOperator,
    positivity_preserving: bool,
    dtype: np.dtype,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Row sums and absolute row sums (action magnitude without coefficients)."""

    ones = jnp.ones((primal.source.size,), dtype=dtype)
    row_scale = np.asarray(_block_action(primal.mv_block, ones), dtype=dtype)
    if not isinstance(primal, (SparseLinearMap, SparseCoordinateOperator)):
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
    if isinstance(primal, SparseCoordinateOperator):
        if primal.block_shape is not None:
            raise ValueError("Scalar topology claims require scalar sparse coefficients.")
        absolute_operator = SparseCoordinateOperator(
            primal.relation,
            jnp.abs(primal.coefficients),
            source=primal.source,
            target=primal.target,
            operator_id=f"{primal.operator_id}:absolute",
        )
    else:
        absolute_operator = SparseLinearMap(
            primal.relation,
            jnp.abs(primal.coefficients),
            operator_id=f"{primal.operator_id}:absolute",
        )
    absolute_scale = np.asarray(absolute_operator.mv(ones), dtype=dtype)
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
        bound = _measure_defect_bound(dtype, action_condition, source_mass, target_mass)
        if np.any(np.abs(pulled - source_mass) > bound):
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


def nested_nodal_transfer(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    transition: Any,
    field_name: str,
    /,
    *,
    semantics: NestedNodalTransferSemantics,
    source_cell_classes: ArrayLike | None = None,
    target_cell_classes: ArrayLike | None = None,
) -> FiniteElementTopologyTransfer:
    """Transfer a named scalar nodal field using bound exact cell ancestry.

    ``interpolate`` evaluates source coefficients at target nodal functionals.
    It makes no conservation or history/irreversibility claim. ``density`` and
    ``intensive`` use physical-mass projection; only density declares conservation.
    Projection requires certified-valid three-dimensional Cartesian geometry and
    exact rational polynomial coordinate maps and reference-chart moments;
    rational-function measures are refused rather than assigned an unproved bound.
    Optional source/target cell classes certify that no projection route mixes
    declared regions; combined with density conservation this proves each
    region's content independently.
    """
    from .. import _coordinate_enclosure as algebra
    from .._cell_geometry import coordinate_effective_basis
    from .._cell_geometry_transition_types import CellGeometryTransition
    from .._nested_reference import nested_cell_reference_maps
    from .._nested_reference_types import (
        _reference_cell_contains,
        ReferenceCellMapWitnesses,
    )
    from ._de_rham import (
        _index_nested_reference_maps,
        _retained_identity_reference_maps,
        _sparse_operator,
    )
    from ._partition_form_moments import _affine, _inverse_map
    from ._reference import discontinuous_element, lagrange_reference_node_data

    if not isinstance(source, FiniteElementDiscretization) or not isinstance(
        target, FiniteElementDiscretization
    ):
        raise TypeError(
            "Nested nodal transfer requires prepared finite-element endpoints."
        )
    if not isinstance(transition, CellGeometryTransition):
        raise TypeError("Nested nodal transfer requires a bound CellGeometryTransition.")
    semantics = parse(semantics, NestedNodalTransferSemantics, "semantics")
    transition.require_bound(
        source.mesh, transition.source_geometry, target.mesh, transition.target_geometry
    )
    for discretization, geometry in (
        (source, transition.source_geometry),
        (target, transition.target_geometry),
    ):
        if (
            geometry.geometry_layout_id
            != discretization.default_runtime.geometry_layout_id
            or not np.array_equal(
                np.asarray(geometry.resolve(discretization.mesh)[2]),
                np.asarray(discretization.default_runtime.coordinates),
            )
        ):
            raise ValueError("Nodal transfer geometry differs from its bound transition.")
    if (
        source.default_runtime.coordinate_frame != target.default_runtime.coordinate_frame
        or _coordinate_measure_key(source) != _coordinate_measure_key(target)
    ):
        raise ValueError(
            "Nodal transfer endpoints require matching scientific axes and measure."
        )
    if np.dtype(source.coefficient_dtype) != np.dtype(target.coefficient_dtype):
        raise ValueError("Nodal transfer endpoints require the same coefficient dtype.")
    source_index, target_index = (
        source._field_index(field_name),
        target._field_index(field_name),
    )
    source_dofs, target_dofs = (
        source.dof_maps[source_index],
        target.dof_maps[target_index],
    )
    if source_dofs.component_shape != target_dofs.component_shape:
        raise ValueError("Nodal transfer field component axes differ.")
    source_elements, target_elements = (
        source.elements[source_index],
        target.elements[target_index],
    )
    conformity = source_elements[0].conformity
    if conformity not in ("H1", "L2"):
        raise ValueError("Nodal transfer requires one scalar H1 or discontinuous field.")
    for element in (*source_elements, *target_elements):
        if (
            element.conformity != conformity
            or element.form_basis is not None
            or element.value_spec.value_spec_id
            != source_elements[0].value_spec.value_spec_id
            or element.degree not in ((1, 2) if conformity == "H1" else (0, 1))
        ):
            raise ValueError(
                "Nodal transfer field spaces have incompatible scientific identities."
            )
        expected = (
            lagrange_element(element.cell_kind, element.degree)
            if conformity == "H1"
            else discontinuous_element(element.cell_kind, element.degree)
        )
        if element.element_id != expected.element_id:
            raise ValueError(
                "Nodal transfer requires canonical point-evaluation elements."
            )
    if semantics in ("density", "intensive") and conformity != "L2":
        raise ValueError(
            "Physical-mass projection requires a discontinuous scalar field."
        )

    partition = transition.coarsening_partition
    partition_ids = frozenset(
        int(identifier)
        for group in (() if partition is None else partition.maps)
        for identifier in np.asarray(group.source_cell_ids)
    )
    direct = transition.reference_map
    records = (
        []
        if direct is None
        else [direct]
        if isinstance(direct, ReferenceCellMapWitnesses)
        else list(direct)
    )
    if transition.refinement is not None:
        records.append(transition.refinement)
    covered = set(partition_ids)
    for record in records:
        identifiers = getattr(
            record, "target_cell_ids", getattr(record, "fine_cell_ids", ())
        )
        covered.update(int(value) for value in np.asarray(identifiers))
    records.extend(_retained_identity_reference_maps(source, target, covered))
    reference_maps, parents = _index_nested_reference_maps(
        tuple(records), source.mesh, target.mesh, partition_target_ids=partition_ids
    )
    source_offsets = np.cumsum((0, *(block.cell_count for block in source.mesh.blocks)))
    source_lookup = {
        int(identifier): (block_index, row)
        for block_index, block in enumerate(source.mesh.blocks)
        for row, identifier in enumerate(np.asarray(block.global_ids))
    }
    partitions: dict[int, list[Any]] = {}
    forward_partitions: dict[int, list[Any]] = {}
    if partition is not None:
        for group in partition.maps:
            for mapping in nested_cell_reference_maps(group):
                forward_partitions.setdefault(mapping.source_cell_id, []).append(mapping)
                a, matrix = _affine(
                    mapping.target_cell_kind, mapping.source_reference_vertices
                )
                inverse = _inverse_map(
                    mapping.target_cell_id,
                    mapping.source_cell_id,
                    mapping.target_cell_kind,
                    mapping.source_cell_kind,
                    a,
                    matrix,
                )
                partitions.setdefault(mapping.source_cell_id, []).append(inverse)
        for mappings in partitions.values():
            mappings.sort(key=lambda mapping: mapping.source_cell_id)
    region_scope_id = None
    if (source_cell_classes is None) != (target_cell_classes is None):
        raise ValueError("Region certification requires both endpoint cell-class arrays.")
    if source_cell_classes is not None and target_cell_classes is not None:
        if semantics not in ("density", "intensive"):
            raise ValueError(
                "Cell-class certification applies to physical-mass projection."
            )
        source_classes = np.asarray(source_cell_classes)
        target_classes = np.asarray(target_cell_classes)
        if (
            source_classes.shape != (int(source_offsets[-1]),)
            or target_classes.shape
            != (sum(block.cell_count for block in target.mesh.blocks),)
            or source_classes.dtype.kind not in "iu"
            or target_classes.dtype.kind not in "iu"
        ):
            raise ValueError(
                "Cell-class arrays must be integer and aligned with endpoint cells."
            )
        offset = 0
        for block in target.mesh.blocks:
            for cell, identifier in enumerate(np.asarray(block.global_ids)):
                row = offset + cell
                if parents[row] >= 0:
                    source_rows = (int(parents[row]),)
                else:
                    source_rows = tuple(
                        int(source_offsets[source_lookup[record.target_cell_id][0]])
                        + source_lookup[record.target_cell_id][1]
                        for record in forward_partitions.get(int(identifier), ())
                    )
                if not source_rows or any(
                    source_classes[source_row] != target_classes[row]
                    for source_row in source_rows
                ):
                    raise ValueError(
                        "Nested physical-mass projection crosses a declared region."
                    )
            offset += block.cell_count
        region_scope_id = canonical_fingerprint(
            {
                "source_classes": array_tree_fingerprint(source_classes),
                "target_classes": array_tree_fingerprint(target_classes),
            }
        )
    if semantics in ("density", "intensive"):
        return _nested_nodal_density_transfer(
            source,
            target,
            transition,
            source_index,
            target_index,
            reference_maps,
            parents,
            forward_partitions,
            source_offsets,
            source_lookup,
            semantics == "density",
            region_scope_id,
        )
    if partition is not None and conformity == "L2":
        raise ValueError(
            "Discontinuous coarsening requires the declared density projection."
        )

    rows: list[int] = []
    columns: list[int] = []
    values: list[Fraction] = []
    owners: set[int] = set()
    source_basis_cache: dict[str, tuple[Any, ...]] = {}
    target_offset = 0
    for block_index, (block, element) in enumerate(
        zip(target.mesh.blocks, target_elements, strict=True)
    ):
        nodes = lagrange_reference_node_data(element)
        for cell, identifier in enumerate(np.asarray(block.global_ids)):
            target_row = target_offset + cell
            target_routes = np.asarray(
                target_dofs.cell_dofs[block_index][cell], dtype=np.int64
            )
            parent = int(parents[target_row])
            direct_map = reference_maps.get(target_row)
            for local, global_row in enumerate(target_routes):
                if int(global_row) in owners:
                    continue
                node = nodes[local].exact_point
                if node is None or any(not isinstance(value, Fraction) for value in node):
                    raise ValueError(
                        "Nested nodal transfer requires exact rational reference nodes."
                    )
                if parent >= 0:
                    if direct_map is None:
                        raise ValueError("A target cell lacks its exact reference map.")
                    source_block = int(
                        np.searchsorted(source_offsets[1:], parent, side="right")
                    )
                    source_cell = parent - int(source_offsets[source_block])
                    point = direct_map.evaluate(node)
                else:
                    candidates = partitions.get(int(identifier), ())
                    selected = None
                    for candidate in candidates:
                        point_ = candidate.evaluate(node)
                        if all(
                            isinstance(value, Fraction) for value in point_
                        ) and _reference_cell_contains(
                            candidate.source_cell_kind, point_
                        ):
                            selected = (candidate, point_)
                            break
                    if selected is None:
                        raise ValueError(
                            "Coarsened nodal functional lacks a source-cell owner."
                        )
                    candidate, point = selected
                    source_block, source_cell = source_lookup[candidate.source_cell_id]
                source_element = source_elements[source_block]
                if source_element.element_id not in source_basis_cache:
                    source_basis_cache[source_element.element_id] = (
                        (
                            algebra.constant(
                                Fraction(1), source_element.topological_dimension
                            ),
                        )
                        if source_element.degree == 0
                        else coordinate_effective_basis(
                            lagrange_element(
                                source_element.cell_kind, source_element.degree
                            )
                        )
                    )
                exact_values = tuple(
                    algebra.evaluate_expression(function, point)
                    for function in source_basis_cache[source_element.element_id]
                )
                source_routes = np.asarray(
                    source_dofs.cell_dofs[source_block][source_cell], dtype=np.int64
                )
                for column, value in zip(source_routes, exact_values, strict=True):
                    if not isinstance(value, Fraction):
                        raise ValueError(
                            "Nested nodal interpolation requires rational exact coefficients."
                        )
                    if value:
                        rows.append(int(global_row))
                        columns.append(int(column))
                        values.append(value)
                owners.add(int(global_row))
        target_offset += block.cell_count
    if len(owners) != target_dofs.global_dof_count:
        raise ValueError("Nested nodal transfer has incomplete target support.")
    dtype = np.dtype(target.coefficient_dtype)
    operator = _sparse_operator(
        np.asarray(rows, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        jnp.asarray(_project_exact_transfer_values(values, dtype), dtype=dtype),
        ArraySpace((source_dofs.global_dof_count,), dtype=dtype),
        ArraySpace((target_dofs.global_dof_count,), dtype=dtype),
        canonical_fingerprint(
            {
                "kind": "nested-nodal-interpolation",
                "source_field": source.field_spaces[source_index].field_space_id,
                "target_field": target.field_spaces[target_index].field_space_id,
                "geometry_transition": transition.transition_id,
            }
        ),
    )
    return FiniteElementTopologyTransfer(
        operator,
        source.mesh.topology_id,
        target.mesh.topology_id,
        preserves_constants=True,
    )


def _nested_nodal_density_transfer(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    transition: Any,
    source_index: int,
    target_index: int,
    reference_maps: dict[int, Any],
    parents: np.ndarray,
    partitions: dict[int, list[Any]],
    source_offsets: np.ndarray,
    source_lookup: dict[int, tuple[int, int]],
    conservative: bool,
    region_scope_id: str | None,
    /,
) -> FiniteElementTopologyTransfer:
    """Cell-local exact physical L2 projection of discontinuous nodal fields."""
    from ...linalg._exact import solve_exact
    from .. import _coordinate_enclosure as algebra
    from .._cell_geometry import coordinate_effective_basis
    from .._cell_geometry_transition_types import _geometry_map_expressions
    from .._cell_geometry_validity import certify_cell_geometry_validity
    from ._de_rham import _sparse_operator
    from ._exact_form_moments import (
        _determinant,
        _entity_simplices,
        _reference_arguments,
        _simplex_chart_arguments,
        _simplex_determinant,
        _standard_simplex_polynomial_integral,
    )

    if source.mesh.topological_dimension != 3 or source.mesh.ambient_dimension != 3:
        raise ValueError(
            "Nested physical-mass projection requires a three-dimensional volume."
        )
    if _coordinate_measure_key(source)[0] != "cartesian":
        raise ValueError(
            "Nested physical-mass projection currently requires Cartesian measure."
        )
    if np.dtype(source.coefficient_dtype) != np.dtype(target.coefficient_dtype):
        raise ValueError(
            "Nested projection endpoints require the same coefficient dtype."
        )
    for mesh, geometry in (
        (source.mesh, transition.source_geometry),
        (target.mesh, transition.target_geometry),
    ):
        certificate = certify_cell_geometry_validity(geometry, mesh=mesh)
        if certificate.invalid_count or certificate.unresolved_count:
            raise ValueError(
                "Nested physical-mass projection requires certified valid geometry."
            )
    source_resolved = transition.source_geometry.resolve(source.mesh)
    target_resolved = transition.target_geometry.resolve(target.mesh)
    source_maps = _geometry_map_expressions(
        source.mesh, transition.source_geometry, (source_resolved[0], source_resolved[1])
    )
    target_maps = _geometry_map_expressions(
        target.mesh, transition.target_geometry, (target_resolved[0], target_resolved[1])
    )
    basis_cache: dict[str, tuple[Any, ...]] = {}
    determinant_cache: dict[tuple[str, int], Any] = {}

    def basis(element: FiniteElementSpec, /) -> tuple[Any, ...]:
        if element.element_id not in basis_cache:
            basis_cache[element.element_id] = (
                (algebra.constant(Fraction(1), element.topological_dimension),)
                if element.degree == 0
                else coordinate_effective_basis(
                    lagrange_element(element.cell_kind, element.degree)
                )
            )
        return basis_cache[element.element_id]

    def determinant(maps: Any, endpoint: str, identifier: int, kind: str, /) -> Any:
        key = (endpoint, identifier)
        if key not in determinant_cache:
            expressions = maps.expressions(identifier)
            value = _determinant(
                tuple(
                    tuple(algebra.derivative(expression, axis) for axis in range(3))
                    for expression in expressions
                ),
                3,
            )
            if isinstance(value, algebra.RationalPolynomial):
                raise ValueError(
                    "Rational coordinate measures need bounded integration evidence."
                )
            center = (
                (Fraction(1, 3), Fraction(1, 3), Fraction(1, 2))
                if kind == "prism"
                else (Fraction(1, 4),) * 3
            )
            sign = algebra.evaluate_expression(value, center)
            if not isinstance(sign, Fraction) or not sign:
                raise ValueError(
                    "Physical-mass orientation lacks an exact rational sign."
                )
            determinant_cache[key] = value if sign > 0 else algebra.scale(value, -1)
        return determinant_cache[key]

    def integrate(expression: Any, kind: str, /) -> Fraction:
        if kind not in ("prism", "tetrahedron", "simplex:3"):
            raise ValueError(
                "Nested physical-mass projection requires prism or tetrahedron cells."
            )
        family = "prism-trimmed" if kind == "prism" else "trimmed"
        total = Fraction(0)
        for simplex in _entity_simplices(3, family):
            local = algebra.compose_expression(
                expression, _simplex_chart_arguments(simplex, 3)
            )
            if isinstance(local, algebra.RationalPolynomial) or any(
                not isinstance(value, Fraction) for value in local.values()
            ):
                raise ValueError(
                    "Physical-mass moments require exact rational polynomials."
                )
            total += _standard_simplex_polynomial_integral(local, 3) * abs(
                _simplex_determinant(simplex)
            )
        if not isinstance(total, Fraction):
            raise RuntimeError(
                "Rational polynomial integration returned a non-rational moment."
            )
        return total

    def cell_measures(
        discretization: FiniteElementDiscretization,
        field_index: int,
        maps: Any,
        endpoint: str,
        /,
    ) -> list[Fraction]:
        dofs = discretization.dof_maps[field_index]
        result = [Fraction(0)] * dofs.global_dof_count
        for block_index, (block, element) in enumerate(
            zip(
                discretization.mesh.blocks,
                discretization.elements[field_index],
                strict=True,
            )
        ):
            functions = basis(element)
            for cell, identifier in enumerate(np.asarray(block.global_ids)):
                weight = determinant(maps, endpoint, int(identifier), block.cell_kind)
                routes = np.asarray(dofs.cell_dofs[block_index][cell], dtype=np.int64)
                for row, function in zip(routes, functions, strict=True):
                    result[int(row)] += integrate(
                        algebra.multiply(function, weight), block.cell_kind
                    )
        return result

    source_dofs, target_dofs = (
        source.dof_maps[source_index],
        target.dof_maps[target_index],
    )
    source_measure = cell_measures(source, source_index, source_maps, "source")
    target_measure = cell_measures(target, target_index, target_maps, "target")
    rows: list[int] = []
    columns: list[int] = []
    exact_values: list[Fraction] = []
    condition = 1.0
    target_offset = 0
    for block_index, (block, element) in enumerate(
        zip(target.mesh.blocks, target.elements[target_index], strict=True)
    ):
        target_basis = basis(element)
        for cell, identifier_ in enumerate(np.asarray(block.global_ids)):
            identifier = int(identifier_)
            target_row = target_offset + cell
            target_routes = np.asarray(
                target_dofs.cell_dofs[block_index][cell], dtype=np.int64
            )
            target_weight = determinant(
                target_maps, "target", identifier, block.cell_kind
            )
            mass = [
                [
                    integrate(
                        algebra.multiply(algebra.multiply(a, b), target_weight),
                        block.cell_kind,
                    )
                    for b in target_basis
                ]
                for a in target_basis
            ]
            condition = max(
                condition, float(np.linalg.cond(np.asarray(mass, dtype=np.float64)))
            )
            if not np.isfinite(condition):
                raise ValueError("Nested projection target mass is rank deficient.")
            if int(parents[target_row]) >= 0:
                parent = int(parents[target_row])
                source_block = int(
                    np.searchsorted(source_offsets[1:], parent, side="right")
                )
                source_cell = parent - int(source_offsets[source_block])
                pieces = ((source_block, source_cell, reference_maps[target_row], False),)
            else:
                maps = partitions.get(identifier, ())
                if not maps:
                    raise ValueError(
                        "Coarsened density target has no exact child partition."
                    )
                pieces = tuple(
                    (*source_lookup[mapping.target_cell_id], mapping, True)
                    for mapping in maps
                )
            for source_block, source_cell, mapping, coarsened in pieces:
                source_element = source.elements[source_index][source_block]
                source_basis = basis(source_element)
                source_routes = np.asarray(
                    source_dofs.cell_dofs[source_block][source_cell], dtype=np.int64
                )
                arguments = _reference_arguments(mapping, 3)
                if coarsened:
                    fine_block = source.mesh.blocks[source_block]
                    fine_id = int(np.asarray(fine_block.global_ids)[source_cell])
                    weight = determinant(
                        source_maps, "source", fine_id, fine_block.cell_kind
                    )
                    test_basis = tuple(
                        algebra.compose_expression(value, arguments)
                        for value in target_basis
                    )
                    trial_basis = source_basis
                    integration_kind = fine_block.cell_kind
                else:
                    weight = target_weight
                    test_basis = target_basis
                    trial_basis = tuple(
                        algebra.compose_expression(value, arguments)
                        for value in source_basis
                    )
                    integration_kind = block.cell_kind
                mixed = [
                    [
                        integrate(
                            algebra.multiply(algebra.multiply(a, b), weight),
                            integration_kind,
                        )
                        for b in trial_basis
                    ]
                    for a in test_basis
                ]
                projection = solve_exact(
                    mass,
                    mixed,
                    singular_message="Nested target physical mass is rank deficient.",
                )
                for local_row, global_row in enumerate(target_routes):
                    for local_column, global_column in enumerate(source_routes):
                        value = projection[local_row][local_column]
                        if value:
                            rows.append(int(global_row))
                            columns.append(int(global_column))
                            exact_values.append(value)
        target_offset += block.cell_count
    if conservative:
        pulled = [Fraction(0)] * source_dofs.global_dof_count
        for row, column, value in zip(rows, columns, exact_values, strict=True):
            pulled[column] += target_measure[row] * value
        if pulled != source_measure:
            raise ValueError(
                "Nested density projection does not conserve exact physical content."
            )
    dtype = np.dtype(target.coefficient_dtype)
    operator = _sparse_operator(
        np.asarray(rows, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        jnp.asarray(_project_exact_transfer_values(exact_values, dtype), dtype=dtype),
        ArraySpace((source_dofs.global_dof_count,), dtype=dtype),
        ArraySpace((target_dofs.global_dof_count,), dtype=dtype),
        canonical_fingerprint(
            {
                "kind": "nested-physical-mass-projection",
                "source_field": source.field_spaces[source_index].field_space_id,
                "target_field": target.field_spaces[target_index].field_space_id,
                "geometry_transition": transition.transition_id,
                "conservative": conservative,
                "region_scope": region_scope_id,
            }
        ),
    )
    return FiniteElementTopologyTransfer(
        operator,
        source.mesh.topology_id,
        target.mesh.topology_id,
        preserves_constants=True,
        conservative=conservative,
        action_condition=condition,
        source_measures=_project_exact_transfer_values(source_measure, dtype)
        if conservative
        else None,
        target_measures=_project_exact_transfer_values(target_measure, dtype)
        if conservative
        else None,
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

    @checked
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

    @checked
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
            or element.value_spec.form_type.degree != 0
            or element.value_spec.proxy != "scalar"
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


def _coordinate_measure_key(
    discretization: FiniteElementDiscretization, /
) -> tuple[CoordinateMeasure, int | None, CoordinateFrame]:
    measure = discretization.coordinate_measure
    return (
        ("cartesian", None, "reference")
        if measure is None
        else (measure.kind, measure.radial_axis, measure.frame)
    )


def _measure_degree(coordinate_measure: CoordinateMeasurePlan | None, /) -> int:
    """Polynomial degree the rotational factor (1, r, or r²) adds to an integrand."""

    if coordinate_measure is None or coordinate_measure.kind == "cartesian":
        return 0
    return 2 if coordinate_measure.kind == "spherical-radial" else 1


def _measure_weights(
    points: np.ndarray,
    weights: np.ndarray,
    /,
    *,
    coordinate_measure: CoordinateMeasurePlan | None,
    coordinate_frame: CoordinateFrame,
) -> np.ndarray:
    """Metric-weighted physical quadrature times the declared rotational factor."""

    if coordinate_measure is None:
        return weights
    coordinate_measure.validate_coordinates(points)
    return weights * np.asarray(
        coordinate_measure.factor(points, frame=coordinate_frame), dtype=np.float64
    )


def _overlap_quadrature(
    simplices: np.ndarray,
    degree: int,
    /,
    *,
    coordinate_measure: CoordinateMeasurePlan | None,
    coordinate_frame: CoordinateFrame,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Physical points/weights with the declared coordinate measure."""
    dimension = simplices.shape[-1]
    # ty: ignore[invalid-argument-type]
    rule = cubature_rule_data(
        _SIMPLEX_KINDS[dimension], degree + _measure_degree(coordinate_measure)
    )
    frames = np.swapaxes(simplices[:, 1:, :] - simplices[:, :1, :], -1, -2)
    measures = np.linalg.det(frames)
    points = simplices[:, None, 0, :] + np.asarray(rule.points)[None] @ np.swapaxes(
        frames, -1, -2
    )
    weights = measures[:, None] * np.asarray(rule.weights)[None, :]
    if coordinate_measure is None:
        return points, weights, rule.rule_id
    weights = _measure_weights(
        points,
        weights,
        coordinate_measure=coordinate_measure,
        coordinate_frame=coordinate_frame,
    )
    rule_id = canonical_fingerprint(
        {
            "kind": "coordinate-measure-overlap-rule",
            "cubature": rule.rule_id,
            "coordinate_measure": coordinate_measure.plan_id,
        }
    )
    return points, weights, rule_id


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

    The per-block rule is exact for degree ``2p`` (mass) or ``p`` (integrals)
    times the rotational factor of the declared coordinate measure, so the target
    mass, its integrals, and the overlap mixed mass share one measure.
    """

    dof_map = discretization.dof_maps[field_index]
    origins, frames = _affine_frames(discretization)
    determinants = np.abs(np.linalg.det(frames))
    coordinate_measure = discretization.coordinate_measure
    coordinate_frame = discretization.default_runtime.coordinate_frame
    measure_degree = _measure_degree(coordinate_measure)
    size = dof_map.global_dof_count
    measures = np.zeros((size,), dtype=np.float64)
    rows, columns, values = [], [], []
    start = 0
    for block_index, element in enumerate(discretization.elements[field_index]):
        routes = np.asarray(dof_map.cell_dofs[block_index])
        stop = start + routes.shape[0]
        # ty: ignore[invalid-argument-type]
        rule = cubature_rule_data(
            element.cell_kind, (2 if mass else 1) * element.degree + measure_degree
        )
        basis = _oriented_basis(
            element,
            np.asarray(dof_map.orientations[block_index]),
            np.asarray(rule.points)[None],
        )
        points = origins[start:stop, None, :] + np.asarray(rule.points)[
            None
        ] @ np.swapaxes(frames[start:stop], -1, -2)
        weights = _measure_weights(
            points,
            determinants[start:stop, None] * np.asarray(rule.weights)[None, :],
            coordinate_measure=coordinate_measure,
            coordinate_frame=coordinate_frame,
        )
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
    if (
        source.default_runtime.coordinate_frame != space.default_runtime.coordinate_frame
        or _coordinate_measure_key(source) != _coordinate_measure_key(space)
    ):
        raise ValueError("L2 projection requires matching coordinate frame and measure.")
    simplices, source_cells, target_cells = _validated_overlaps(
        refinement, source.mesh, space.mesh
    )
    source_elements = _projection_elements(source, source_index, "source")
    target_elements = space.elements[target_index]
    points, weights, rule_id = _overlap_quadrature(
        simplices,
        max(element.degree for element in source_elements)
        + max(element.degree for element in target_elements),
        coordinate_measure=space.coordinate_measure,
        coordinate_frame=space.default_runtime.coordinate_frame,
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
    "NestedNodalTransferSemantics",
    "nested_nodal_transfer",
    "prepare_l2_projection_target",
    "prepare_l2_projection_transfer",
    "refresh_l2_projection_target",
    "vertex_interpolation_transfer",
]
