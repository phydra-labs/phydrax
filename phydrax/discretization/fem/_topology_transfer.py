#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import factorial, isfinite
from typing import assert_never, final, Literal, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

import phydrax.discretization._mapped_geometry_measure as mapped_measure_owner
import phydrax.discretization._periodic_exact as periodic_exact_owner
import phydrax.discretization._source_geometry_realization as source_realization_owner
from phydrax.discretization.fem._reference_types import FiniteElementSpec
from phydrax.geometry._rational import fraction_endpoint

from ... import ein, geometry as imported_geometry, linalg as imported_linalg
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._polynomial._cubature import cubature_rule_data
from ..._resource_ticket import HostBatchTicket
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...exterior import (
    _algebra as imported_exterior__algebra,
    _form_type as imported_exterior__form_type,
)
from ...linalg import (
    AbstractLinearOperator,
    adjoint,
    ArraySpace,
    DenseLinearOperator,
    determinant_small_linear,
    estimate_condition_number,
    FactorizationPolicy,
    factorize,
    FunctionLinearOperator,
    OperatorCapabilities,
    OperatorProperties,
    prepare_sparse_factorization,
    PreparedFactorization,
    PreparedSparseFactorization,
    refresh_sparse_factorization,
    SmallLinearSolvePlan,
    SparseFactorizationPlan,
    SparseFactorizationPolicy,
    SparseFactorizationStatus,
    SpectralEstimate,
    transpose,
)
from ...linalg._operators import generic_adjoint, materialize_by_basis
from ...meshing import _measurements as imported_meshing__measurements
from ...sparse import (
    EdgeRelation,
    linear_apply,
    linear_transpose_apply,
    RowRelation,
    SparseLinearMap,
)
from ...typing import checked, Dim, Float, parse, validate
from .. import (
    _cell_geometry as imported_cell_geometry,
    _cell_geometry_validity as imported_cell_geometry_validity,
    _coordinate_enclosure as imported_coordinate_enclosure,
    _nested_reference as imported_nested_reference,
    _reference_cell as imported_reference_cell,
)
from .._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from .._cell_geometry import (
    BarycentricCellGeometryElement,
    CellGeometrySpec,
    coordinate_lagrange_element,
    RestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from .._cell_geometry_transition_types import (
    CellGeometryTransition,
    CellGeometryTransitionError,
    NestedReferenceWitnesses,
)
from .._cell_mesh import CellMesh
from .._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from .._coordinate_enclosure import (
    coordinate_estimate_bank,
    coordinate_polynomials_estimate,
    CoordinateSourceBank,
    Expression,
    Polynomial,
)
from .._exact_algebraic_geometry import rational_coefficient
from .._mapped_geometry_measure import (
    mapped_root_identities,
    PreparedMappedSqrtIntegral,
)
from .._nested_reference import rooted_nested_reference_pairs
from .._nested_reference_types import NestedReferencePair, PolynomialReferencePair
from .._source_geometry_realization import SourceGeometryRealization
from .._spaces import DiscreteFieldSpace
from .._topology_epoch import (
    FieldEpochTransition,
    TopologyEpoch,
    TopologyEpochTransition,
)
from .._transfer import (
    FieldTransfer,
    TransferGeometryBinding,
    TransferProperties,
    TransferSemantics,
)
from . import (
    _mapped_form_transfer as imported_mapped_form_transfer,
    _reference as imported_reference,
)
from ._assembly import tetrahedral_face_vertices
from ._generic import (
    FiniteElementDiscretization,
    FiniteElementDofMap,
    FiniteElementTransferDiscretization,
)


if TYPE_CHECKING:
    from ...meshing._measurements import NativeExecutionRecord


# Invariant claims are certified in the coefficient dtype up to this many ulps of
# the local row scale.
CLAIM_ULPS = 64.0

_SIMPLEX_KINDS = {2: "triangle", 3: "tetrahedron"}
_AFFINE_MEASURE_PLANS = {
    dimension: SmallLinearSolvePlan(dimension) for dimension in _SIMPLEX_KINDS
}

type MappedDgCell = tuple[FiniteElementSpec, NDArray[np.int64], tuple[Polynomial, ...]]
type _NestedFiniteElementDiscretization = (
    FiniteElementDiscretization | FiniteElementTransferDiscretization
)
type _CompatibleFieldBinding = tuple[
    str,
    int,
    str,
    str,
    tuple[str, ...],
    tuple[int, ...],
    str,
]


def _compatible_field_binding(
    discretization: _NestedFiniteElementDiscretization, field_name: str, /
) -> _CompatibleFieldBinding:
    name = canonical_identifier(field_name, "field_name")
    index = discretization.field_index(name)
    field = discretization.construction_plan.fields[index]
    dof_map = discretization.dof_maps[index]
    space = discretization.field_spaces[index].vector_space
    if not isinstance(space, ArraySpace):
        raise TypeError(
            "compatible_field_binding.configuration must provide ArraySpace coefficients; got Compatible transfer fields require ArraySpace coefficients."
        )
    return (
        name,
        index,
        field.field_spec_id,
        dof_map.dof_map_id,
        tuple(element.element_id for element in discretization.elements[index]),
        space.shape,
        space.dtype.str,
    )


@dataclass(frozen=True)
class _MappedNestedCompatiblePreparation:
    source_prepared_id: str
    target_prepared_id: str
    source_revision: str
    target_revision: str
    source_geometry_id: str
    target_geometry_id: str
    bindings: tuple[tuple[_CompatibleFieldBinding, _CompatibleFieldBinding], ...]
    pairs: tuple[NestedReferencePair | PolynomialReferencePair, ...]
    geometry_bound: float

    def require(
        self,
        source: _NestedFiniteElementDiscretization,
        target: _NestedFiniteElementDiscretization,
        source_geometry: CellGeometrySpec,
        target_geometry: CellGeometrySpec,
        field_name: str,
        /,
    ) -> None:
        binding = (
            _compatible_field_binding(source, field_name),
            _compatible_field_binding(target, field_name),
        )
        if (
            source.prepared_id != self.source_prepared_id
            or target.prepared_id != self.target_prepared_id
            or source.mesh.numeric_version != self.source_revision
            or target.mesh.numeric_version != self.target_revision
            or _coordinate_geometry_id(source_geometry) != self.source_geometry_id
            or _coordinate_geometry_id(target_geometry) != self.target_geometry_id
            or binding not in self.bindings
        ):
            raise ValueError(
                "_MappedNestedCompatiblePreparation.binding must match the prepared mesh revisions, coordinate maps, and field bindings; got "
                f"source=({source.prepared_id!r}, {source.mesh.numeric_version!r}, {_coordinate_geometry_id(source_geometry)!r}), "
                f"target=({target.prepared_id!r}, {target.mesh.numeric_version!r}, {_coordinate_geometry_id(target_geometry)!r}), binding={binding!r}"
            )


def _simplex_cubature_kind(kind: str, /) -> Literal["triangle", "tetrahedron"]:
    if kind == "triangle":
        return "triangle"
    if kind == "tetrahedron":
        return "tetrahedron"
    raise ValueError(
        f"simplex_cubature_kind.kind must support the declared value; got Simplex finite-element cubature does not support {kind!r}."
    )


def _claim_tolerance(dtype: np.dtype, scale: np.ndarray, /) -> np.ndarray:
    return CLAIM_ULPS * np.finfo(dtype).eps * np.maximum(scale, 1.0)


def measure_defect_bound(
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
    return float(action_condition * CLAIM_ULPS * np.finfo(dtype).eps * total)


def _block_action(action: Callable[[Array], Array], values: ArrayLike, /) -> Array:
    """Apply a declared-column action to ``(dofs, *payload)`` values."""

    array = jnp.asarray(values)
    if array.ndim == 0:
        raise ValueError(
            "block_action.configuration must provide a leading DOF axis; got Transfer inputs require a leading DOF axis."
        )
    block = action(array.reshape((array.shape[0], -1)))
    return block.reshape((block.shape[0],) + array.shape[1:])


class _TransferSourceDofDim(Dim):
    """Source coefficient rows used to certify a topology transfer."""


class _TransferTargetDofDim(Dim):
    """Target coefficient rows used to certify a topology transfer."""


class _TransferPhysicalDim(Dim):
    """Physical coordinate components shared by a transfer's endpoints."""


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
    (for example the condition estimate of an inner mass solve). Coordinates and
    measures needed by declared claims remain dynamic leaves so stored-field
    reconstruction can certify those claims again.
    """

    __strict_contract__ = True

    primal: SparseLinearMap | AbstractLinearOperator
    hilbert_adjoint: SparseLinearMap | AbstractLinearOperator | None
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    preserves_constants: bool = eqx.field(static=True)
    preserves_linear: bool = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    positivity_preserving: bool = eqx.field(static=True)
    action_condition: float = eqx.field(static=True)
    semantics: TransferSemantics = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    source_coordinates: Float[_TransferSourceDofDim, _TransferPhysicalDim] | None
    target_coordinates: Float[_TransferTargetDofDim, _TransferPhysicalDim] | None
    source_measures: Float[_TransferSourceDofDim] | None
    target_measures: Float[_TransferTargetDofDim] | None

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[FiniteElementTopologyTransfer],
        value: FiniteElementTopologyTransfer,
        /,
    ) -> FiniteElementTopologyTransfer:
        """Reconstruct the transfer and recertify its retained invariant inputs."""
        rebuilt = cls(
            value.primal,
            value.source_topology_id,
            value.target_topology_id,
            hilbert_adjoint=value.hilbert_adjoint,
            preserves_constants=value.preserves_constants,
            preserves_linear=value.preserves_linear,
            conservative=value.conservative,
            positivity_preserving=value.positivity_preserving,
            action_condition=value.action_condition,
            semantics=value.semantics,
            source_coordinates=value.source_coordinates,
            target_coordinates=value.target_coordinates,
            source_measures=value.source_measures,
            target_measures=value.target_measures,
        )
        validate(rebuilt)
        return rebuilt

    @checked
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
        semantics: TransferSemantics = "interpolation",
        source_coordinates: ArrayLike | None = None,
        target_coordinates: ArrayLike | None = None,
        source_measures: ArrayLike | None = None,
        target_measures: ArrayLike | None = None,
    ) -> None:

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
                "FiniteElementTopologyTransfer.primal must be one unbatched target-by-source map; got Topology transfer primal must be one unbatched target-by-source map."
            )
        source_id = canonical_identifier(source_topology_id, "source_topology_id")
        target_id = canonical_identifier(target_topology_id, "target_topology_id")

        if hilbert_adjoint is not None and (
            hilbert_adjoint.source.size != primal.target.size
            or hilbert_adjoint.target.size != primal.source.size
        ):
            raise ValueError(
                "FiniteElementTopologyTransfer.hilbert_adjoint must map target DOFs back to source DOFs; got hilbert_adjoint must map target DOFs back to source DOFs."
            )
        condition = float(action_condition)
        if not isfinite(condition) or condition < 1.0:
            raise ValueError(
                "FiniteElementTopologyTransfer.action_condition must be finite and at least one; got action_condition must be finite and at least one."
            )
        semantics_ = parse(semantics, TransferSemantics, "semantics")
        claims = tuple(
            bool(value)
            for value in (
                preserves_constants,
                preserves_linear,
                conservative,
                positivity_preserving,
            )
        )
        source_coordinates_ = (
            None
            if not claims[1] or source_coordinates is None
            else jnp.asarray(source_coordinates, dtype=source_space.dtype)
        )
        target_coordinates_ = (
            None
            if not claims[1] or target_coordinates is None
            else jnp.asarray(target_coordinates, dtype=source_space.dtype)
        )
        source_measures_ = (
            None
            if not claims[2] or source_measures is None
            else jnp.asarray(source_measures, dtype=source_space.dtype)
        )
        target_measures_ = (
            None
            if not claims[2] or target_measures is None
            else jnp.asarray(target_measures, dtype=source_space.dtype)
        )
        _certify_claims(
            primal,
            source_space,
            *claims,
            action_condition=condition,
            source_coordinates=source_coordinates_,
            target_coordinates=target_coordinates_,
            source_measures=source_measures_,
            target_measures=target_measures_,
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
        self.semantics = semantics_
        self.source_coordinates = source_coordinates_
        self.target_coordinates = target_coordinates_
        self.source_measures = source_measures_
        self.target_measures = target_measures_
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
                "semantics": semantics_,
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

        match self.primal:
            case SparseLinearMap() as primal:
                return _mixed_mass_action(
                    primal.relation, primal.coefficients, jnp.asarray(values)
                )
            case primal:
                return _block_action(primal.mv_block, values)

    def pullback(self, dual: ArrayLike, /) -> Array:
        """Pull target duals back through the algebraic transpose of the primal map."""

        match self.primal:
            case SparseLinearMap() as primal:
                return _mixed_mass_transpose_action(
                    primal.relation, primal.coefficients, jnp.asarray(dual)
                )
            case primal:
                return _block_action(primal.transpose_mv_block, dual)

    @checked
    def epoch_transition(
        self,
        source_field: DiscreteFieldSpace,
        target_field: DiscreteFieldSpace,
        source_epoch: TopologyEpoch,
        target_epoch: TopologyEpoch,
        source_measures: ArrayLike,
        target_measures: ArrayLike,
        /,
        *,
        geometry: TransferGeometryBinding,
    ) -> TopologyEpochTransition:
        """Bind this certified transfer as an explicit topology-epoch transition.

        The epochs must realize exactly this transfer's source and target
        topologies. Field-transfer properties are this artifact's certified claims,
        never caller assertions; the raw transpose is the dual pullback and the
        Hilbert adjoint is the declared one, else the adjoint under the coefficient
        pairings. Only conservative transfers qualify; the measures are the DOF
        integrals the transition reports content against, and they must satisfy
        this transfer's conservation certificate, whose per-DOF measure-defect
        bound the transition carries into its content tolerance. ``geometry`` binds
        the source/target
        coordinate maps the transfer was prepared on; a ``nested-interpolation``
        transfer requires its algebraic-restriction binding.
        """

        if (
            source_epoch.topology_id != self.source_topology_id
            or target_epoch.topology_id != self.target_topology_id
        ):
            raise ValueError(
                "FiniteElementTopologyTransfer.epoch_transition.configuration must realize this transfer's topologies; got Topology epochs do not realize this transfer's topologies."
            )
        if not self.conservative:
            raise ValueError(
                "FiniteElementTopologyTransfer.epoch_transition.configuration must certify conservation before binding a topology transition; got Only a certified conservative transfer forms a topology transition."
            )
        source_space = self.primal.source
        if not isinstance(source_space, ArraySpace):
            raise RuntimeError(
                "FiniteElementTopologyTransfer.epoch_transition.configuration must have an ArraySpace source; got Topology transfer primal lost its ArraySpace source."
            )
        dtype = np.dtype(source_space.dtype)
        source_mass = np.asarray(source_measures, dtype=dtype)
        target_mass = np.asarray(target_measures, dtype=dtype)
        if source_mass.shape != (self.source_size,) or target_mass.shape != (
            self.target_size,
        ):
            raise ValueError(
                "FiniteElementTopologyTransfer.epoch_transition.configuration must align with source/target DOFs; got Transition measures must align with source/target DOFs."
            )
        bound = measure_defect_bound(
            dtype, self.action_condition, source_mass, target_mass
        )
        pulled = np.asarray(
            _block_action(self.primal.transpose_mv_block, target_mass), dtype=dtype
        )
        if np.any(np.abs(pulled - source_mass) > bound):
            raise ValueError(
                "FiniteElementTopologyTransfer.epoch_transition.measures must conserve source measures within the certified bound; got "
                f"defect={np.abs(pulled - source_mass)!r}, bound={bound!r}"
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
                nested=self.semantics == "nested-interpolation",
                adjoint_paired=True,
                differentiable_geometry=False,
                exact_on=exact_on if self.preserves_constants else (),
                semantics=self.semantics,
            ),
            geometry=geometry,
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
    row_scale = np.asarray(
        _mixed_mass_action(primal.relation, primal.coefficients, ones)
        if isinstance(primal, SparseLinearMap)
        else _block_action(primal.mv_block, ones),
        dtype=dtype,
    )
    if not isinstance(primal, SparseLinearMap):
        if not np.all(np.isfinite(row_scale)):
            raise ValueError(
                "row_scales.configuration must be finite; got Topology transfer actions must be finite."
            )
        if positivity_preserving:
            raise ValueError(
                "row_scales.configuration must provide sparse coefficients for positivity certification; got Positivity preservation is certified only from sparse coefficients."
            )
        return row_scale, np.abs(row_scale)
    valid = np.asarray(primal.relation.valid, dtype=np.bool_)
    coefficients = np.where(valid, np.asarray(primal.coefficients), 0.0)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(
            "row_scales.configuration must be finite; got Topology transfer coefficients must be finite."
        )
    if positivity_preserving and np.any(coefficients < 0.0):
        raise ValueError(
            "row_scales.configuration must have nonnegative weights for positivity preservation; got Transfer claims positivity preservation with negative weights."
        )
    absolute_scale = np.asarray(
        _mixed_mass_action(
            primal.relation,
            np.abs(np.asarray(primal.coefficients)),
            ones,
        ),
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
            "certify_claims.configuration must sum to one; got Transfer claims constant preservation but rows do not sum to one."
        )
    if preserves_linear:
        if source_coordinates is None or target_coordinates is None:
            raise ValueError(
                "certify_claims.configuration must provide source and target coordinates to certify; got Linear preservation requires source and target coordinates to certify."
            )
        source = np.asarray(source_coordinates, dtype=dtype)
        target = np.asarray(target_coordinates, dtype=dtype)
        if (
            source.ndim != 2
            or target.ndim != 2
            or source.shape[0] != primal.source.size
            or target.shape != (primal.target.size, source.shape[1])
        ):
            raise ValueError(
                "certify_claims.configuration must align with source/target DOFs; got Transfer coordinates must align with source/target DOFs."
            )
        mapped = np.asarray(
            _mixed_mass_action(primal.relation, primal.coefficients, source)
            if isinstance(primal, SparseLinearMap)
            else _block_action(primal.mv_block, source),
            dtype=dtype,
        )
        scale = absolute_scale[:, None] * np.max(np.abs(source), initial=1.0)
        if exceeds(mapped - target, scale):
            raise ValueError(
                "certify_claims.configuration must preserve source coordinates for a linear-preservation claim; got Transfer claims linear preservation but moves coordinates."
            )
    if conservative:
        if source_measures is None or target_measures is None:
            raise ValueError(
                "certify_claims.configuration must provide source and target DOF measures; got Conservation requires source and target DOF measures."
            )
        source_mass = np.asarray(source_measures, dtype=dtype)
        target_mass = np.asarray(target_measures, dtype=dtype)
        if source_mass.shape != (primal.source.size,) or target_mass.shape != (
            primal.target.size,
        ):
            raise ValueError(
                "certify_claims.configuration must align with source/target DOFs; got Transfer measures must align with source/target DOFs."
            )
        pulled = np.asarray(
            _mixed_mass_transpose_action(
                primal.relation, primal.coefficients, target_mass
            )
            if isinstance(primal, SparseLinearMap)
            else _block_action(primal.transpose_mv_block, target_mass),
            dtype=dtype,
        )
        bound = measure_defect_bound(dtype, action_condition, source_mass, target_mass)
        if np.any(np.abs(pulled - source_mass) > bound):
            raise ValueError(
                "certify_claims.configuration must preserve DOF measures for a conservation claim; got Transfer claims conservation but changes DOF measures."
            )


@checked
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
        raise TypeError(
            "vertex_interpolation_transfer.source_rows must have an integer dtype; got source_rows must have an integer dtype."
        )
    if route_valid.dtype != np.bool_:
        raise TypeError(
            "vertex_interpolation_transfer.valid must be a boolean mask; got valid must be a boolean mask."
        )
    if not np.issubdtype(coefficients.dtype, np.floating):
        raise TypeError(
            "vertex_interpolation_transfer.weights must have a real floating dtype; got weights must have a real floating dtype."
        )
    size = int(source_size)
    if (
        rows.ndim != 2
        or rows.shape[0] == 0
        or rows.shape[1] == 0
        or coefficients.shape != rows.shape
        or route_valid.shape != rows.shape
        or size <= 0
    ):
        raise ValueError(
            "vertex_interpolation_transfer.weights must share (n, w); got Interpolation rows, weights, and validity must share (n, w)."
        )
    if not np.all(np.any(route_valid, axis=1)):
        raise ValueError(
            "vertex_interpolation_transfer.valid must provide one valid source route; got Every interpolation target requires one valid source route."
        )
    routed = rows[route_valid]
    if np.any(routed < 0) or np.any(routed >= size):
        raise ValueError(
            "vertex_interpolation_transfer.configuration must address source DOFs; got Valid interpolation routes must address source DOFs."
        )
    if not np.all(np.isfinite(coefficients[route_valid])):
        raise ValueError(
            "vertex_interpolation_transfer.weights must be finite; got Valid interpolation weights must be finite."
        )
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


class _CompatibleProjectionTarget(StrictModule, NonTrainableState):
    """Target-only differential image and mass-weighted constraint lift.

    The SVD factors belong to native linalg. They and their inverse actions are
    prepared once, independently of source topology, overlap, or field payload.
    Differential coordinates are cellwise L2-orthonormal polynomial moments.
    """

    differential: Array
    constraints: Array
    whitening: Array
    content_scale: Array
    image_inverse: Array
    lift: Array
    image_factor: PreparedFactorization
    schur_factor: PreparedFactorization
    degree: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    condition: float = eqx.field(static=True)

    def __init__(
        self,
        differential: ArrayLike,
        constraints: ArrayLike,
        whitening: ArrayLike,
        content_scale: ArrayLike,
        image_inverse: ArrayLike,
        lift: ArrayLike,
        image_factor: PreparedFactorization,
        schur_factor: PreparedFactorization,
        degree: int,
        component_count: int,
        condition: float,
        /,
    ) -> None:
        self.differential = jnp.asarray(differential)
        self.constraints = jnp.asarray(constraints)
        self.whitening = jnp.asarray(whitening)
        self.content_scale = jnp.asarray(content_scale)
        self.image_inverse = jnp.asarray(image_inverse)
        self.lift = jnp.asarray(lift)
        self.image_factor = image_factor
        self.schur_factor = schur_factor
        self.degree = degree
        self.component_count = component_count
        self.condition = condition


@final
class PreparedL2ProjectionTarget(StrictModule, NonTrainableState):
    """Prepared target space of Galerkin L2 projections onto one FE field.

    Built by :func:`prepare_l2_projection_target`. Owns the target mass ``M_T`` of
    the field ``field_name`` of ``discretization`` (scalar Lagrange or
    Piola-mapped H(curl)/H(div)), its sparse Cholesky ``factorization`` (the
    symbolic reverse Cuthill-McKee ordering and fill pattern in
    ``factorization.plan``, the numeric factor, its status and pivot
    diagnostics), the Golub-Kahan estimate ``mass_condition``, and
    ``dof_measures`` (integrals of the target basis functions, one vector per DOF
    for vector-valued fields). One artifact serves
    every source field and every common refinement onto this target, and payload
    axes of its transfers are solved as one multi-right-hand-side block.

    For Piola fields ``compatible`` additionally owns full polynomial
    differential moments, L2-orthonormal cell tests, native SVD factors for the
    differential image and mass-weighted constraints, and their cached lift
    actions. These target-only artifacts are reused by constrained compatible
    field transfers; plain :func:`prepare_l2_projection_transfer` remains L2.

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
    compatible: _CompatibleProjectionTarget | None
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
        constraint_policy: FactorizationPolicy | None = None,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _L2_ARTIFACT_TOKEN:
            raise TypeError(
                "PreparedL2ProjectionTarget._construction_token must come from prepare_l2_projection_target or refresh_l2_projection_target; got "
                f"{type(_construction_token).__name__}"
            )
        name = str(field_name)
        index = discretization.field_index(name)
        dof_map = discretization.dof_maps[index]
        size = dof_map.global_dof_count
        value_shape = discretization.elements[index][0].value_shape
        measures = jnp.asarray(dof_measures)
        if (
            mass.batch_shape
            or mass.input_shape != (size,)
            or mass.output_shape != (size,)
            or factorization.plan.shape != (size, size)
            or factorization.batch_shape
            or factorization.plan.kind != "cholesky"
            or measures.shape != (size,) + value_shape
        ):
            raise ValueError(
                f"PreparedL2ProjectionTarget.mass must have unbatched ({size},) input/output axes, an unbatched ({size}, {size}) Cholesky factorization, and measures shaped {(size,) + value_shape} for field {name!r}; got "
                f"mass batch={mass.batch_shape!r}, input={mass.input_shape!r}, output={mass.output_shape!r}, factor shape={factorization.plan.shape!r}, batch={factorization.batch_shape!r}, kind={factorization.plan.kind!r}, measures={measures.shape!r}"
            )
        if not (
            mass.properties.certifies("self_adjoint")
            and mass.properties.certifies("positive_definite")
        ):
            raise ValueError(
                "PreparedL2ProjectionTarget.mass must certify self-adjoint positive definiteness; got The target mass must certify self-adjoint positive definiteness."
            )
        # Preparation boundary: one host decision on the prepared factor status.
        status = SparseFactorizationStatus(int(factorization.status))
        if status is not SparseFactorizationStatus.SUCCESS:
            raise ValueError(
                f"PreparedL2ProjectionTarget.mass must pass the declared scientific certificate; got Target mass Cholesky factorization failed ({status.name})."
            )
        if not isfinite(float(mass_condition.value)):
            raise ValueError(
                "PreparedL2ProjectionTarget.mass must be finite; got The target mass condition estimate is not finite."
            )
        self.discretization = discretization
        self.mass = mass
        self.factorization = factorization
        self.mass_condition = mass_condition
        self.dof_measures = measures
        self.compatible = (
            None
            if not value_shape
            else _prepare_compatible_target(
                discretization,
                index,
                factorization,
                np.asarray(measures),
                FactorizationPolicy("svd")
                if constraint_policy is None
                else constraint_policy,
            )
        )
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
                "FiniteElementL2Projection._construction_token must come from prepare_l2_projection_transfer; got "
                f"{type(_construction_token).__name__}"
            )
        if (
            mixed_mass.batch_shape
            or len(mixed_mass.input_shape) != 1
            or mixed_mass.output_shape != prepared_target.mass.output_shape
        ):
            raise ValueError(
                "FiniteElementL2Projection.mixed_mass must be unbatched with one source DOF axis and the prepared target output shape; got "
                f"batch={mixed_mass.batch_shape!r}, input={mixed_mass.input_shape!r}, output={mixed_mass.output_shape!r}, target={prepared_target.mass.output_shape!r}"
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

    def mv(self, vector: ArrayLike, /) -> Array:
        mixed = _mixed_mass_action(
            self.mixed_mass.relation, self.mixed_mass.coefficients, jnp.asarray(vector)
        )
        return _target_mass_solve(self.prepared_target.factorization, mixed)

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        # M_T is symmetric, so the transpose solve reuses the same factor.
        solved = _target_mass_solve(
            self.prepared_target.factorization, jnp.asarray(vector)
        )
        return _mixed_mass_transpose_action(
            self.mixed_mass.relation, self.mixed_mass.coefficients, solved
        )

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        return generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        return materialize_by_basis(self)


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
    refinement: imported_geometry.PreparedCommonRefinement,
    source: CellMesh,
    target: CellMesh,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overlap simplices ``(S, d+1, d)`` with their source and target cells."""

    if not refinement.succeeded:
        raise ValueError(
            "validated_overlaps.refinement.status must indicate successful common refinement; got "
            f"{imported_geometry.CommonRefinementStatus(refinement.status).name}: "
            f"{refinement.evidence.reason}"
        )
    if refinement.source_mesh_id != source.mesh_id or (
        refinement.target_mesh_id != target.mesh_id
    ):
        raise ValueError(
            "validated_overlaps.refinement must join the source and target meshes; got The common refinement does not join the source and target meshes."
        )
    if refinement.simplices is None or refinement.simplex_offsets is None:
        raise ValueError(
            "validated_overlaps.configuration must provide CommonRefinementPolicy(overlap_simplices=True); got L2 projection requires CommonRefinementPolicy(overlap_simplices=True)."
        )
    offsets = np.asarray(refinement.simplex_offsets, dtype=np.int64)
    simplices = np.asarray(refinement.simplices, dtype=np.float64)
    if simplices.shape[0] == 0:
        raise ValueError(
            "validated_overlaps.refinement must select a supported declared route; got The common refinement has no overlap simplices."
        )
    entries = np.repeat(np.arange(offsets.size - 1, dtype=np.int64), np.diff(offsets))
    return (
        simplices,
        np.asarray(refinement.source_cells, dtype=np.int64)[entries],
        np.asarray(refinement.target_cells, dtype=np.int64)[entries],
    )


def _require_affine_simplices(
    discretization: _NestedFiniteElementDiscretization, role: str, operation: str, /
) -> None:
    """Refuse any block that is not an affine simplex realized on its mesh."""

    mesh = discretization.mesh
    coordinates = np.asarray(discretization.default_runtime.coordinates)
    mesh_coordinates = np.asarray(mesh.coordinates)
    for block, coordinate_element, routes in zip(
        mesh.blocks,
        discretization.coordinate_elements,
        discretization.coordinate_dofs,
        strict=True,
    ):
        if block.cell_kind not in _SIMPLEX_KINDS.values():
            raise ValueError(
                "require_affine_simplices.block.cell_kind must be triangle or tetrahedron; got "
                f"{block.cell_kind!r} in {role} block {block.name!r} for {operation}"
            )
        canonical = coordinate_lagrange_element(block.cell_kind, 1)
        match coordinate_element:
            case BarycentricCellGeometryElement():
                affine_element = coordinate_element
            case FiniteElementSpec() if (
                coordinate_element.element_id == canonical.element_id
            ):
                affine_element = coordinate_element
            case _:
                raise ValueError(
                    "require_affine_simplices.coordinate_element must define affine simplex geometry; got "
                    f"{type(coordinate_element).__name__} in {role} block {block.name!r} for {operation}, element={coordinate_element!r}"
                )
        reference_values = np.asarray(
            affine_element.tabulate(canonical.reference_nodes)[0]
        )
        represented = reference_values @ coordinates[np.asarray(routes)]
        if not np.array_equal(represented, mesh_coordinates[np.asarray(block.vertices)]):
            raise ValueError(
                f"require_affine_simplices.role must agree with the declared geometry, topology, and field layout; got The {role} FE geometry differs from its refined mesh."
            )


def _require_coefficient_affine_simplices(
    discretization: _NestedFiniteElementDiscretization,
    geometry: CellGeometrySpec,
    role: str,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Authenticate retained affine maps and their published corner carriers."""
    restriction = geometry.restriction_source
    published_roots = (
        restriction is not None
        and restriction.carrier_publication == "published_affine_roots"
    )
    if published_roots:
        imported_cell_geometry.require_published_affine_carrier(
            geometry, discretization.mesh, budget=budget
        )
    cells = mapped_measure_owner.mapped_geometry_cells(
        discretization.mesh, geometry, budget=budget
    )
    mesh_coordinates = np.asarray(discretization.mesh.coordinates)
    runtime_coordinates = np.asarray(discretization.default_runtime.coordinates)
    start = 0
    for block, element, routes in zip(
        discretization.mesh.blocks,
        discretization.coordinate_elements,
        discretization.coordinate_dofs,
        strict=True,
    ):
        canonical = coordinate_lagrange_element(block.cell_kind, 1)
        if not isinstance(element, BarycentricCellGeometryElement):
            if (
                not isinstance(element, FiniteElementSpec)
                or element.element_id != canonical.element_id
            ):
                raise ValueError(
                    "coefficient_affine_simplices.element must retain an affine P1 simplex source; got another coordinate map."
                )
            represented = (
                np.asarray(element.tabulate(canonical.reference_nodes)[0])
                @ (runtime_coordinates[np.asarray(routes)])
            )
            if not published_roots and not np.array_equal(
                represented, mesh_coordinates[np.asarray(block.vertices)]
            ):
                raise ValueError(
                    f"coefficient_affine_simplices.carrier must bind the {role} coordinate map; got different mesh corners."
                )
        for row, (cell_element, local) in zip(
            np.asarray(block.vertices),
            cells[start : start + block.cell_count],
            strict=True,
        ):
            corners = imported_coordinate_enclosure.affine_simplex_corner_images(
                cell_element, local, budget=budget
            )
            if corners is None:
                raise ValueError(
                    "coefficient_affine_simplices.source must prove the retained source affine throughout; got unproved source affinity."
                )
            if (
                not published_roots
                and isinstance(element, BarycentricCellGeometryElement)
                and not np.array_equal(
                    np.asarray(
                        [
                            imported_coordinate_enclosure.rounded_point(point)
                            for point in corners
                        ],
                        dtype=np.float64,
                    ),
                    mesh_coordinates[row],
                )
            ):
                raise ValueError(
                    f"coefficient_affine_simplices.carrier must bind the {role} rational corner images; got different mesh corners."
                )
        start += block.cell_count


def _projection_field_elements(
    discretization: FiniteElementDiscretization,
    field_index: int,
    role: str,
    /,
) -> tuple[FiniteElementSpec, ...]:
    """Return supported scalar Lagrange or Piola-mapped projection elements."""

    elements = discretization.elements[field_index]
    for block, element in zip(discretization.mesh.blocks, elements, strict=True):
        match element.mapping:
            case "identity":
                supported = (
                    element.value_spec.form_type.degree == 0 and not element.value_shape
                )
            case "covariant_piola" | "contravariant_piola":
                supported = element.form_basis is not None and element.value_shape == (
                    element.topological_dimension,
                )
            case _:
                supported = False
        if element.cell_kind not in _SIMPLEX_KINDS.values() or not supported:
            raise ValueError(
                "projection_elements.element must be scalar Lagrange or Piola-mapped H(curl)/H(div) on a triangle or tetrahedron; got "
                f"role={role!r}, block={block.name!r}, family={element.family!r}, cell_kind={element.cell_kind!r}, mapping={element.mapping!r}, value_shape={element.value_shape!r}, form_degree={element.value_spec.form_type.degree!r}, form_basis_present={element.form_basis is not None}"
            )
    return elements


def _polynomial_degree(element: FiniteElementSpec, /) -> int:
    if element.form_basis is not None:
        return max((sum(alpha) for alpha in element.form_basis.exponents), default=0)
    return element.degree


# Host-only immutable preparation: the affine cell maps, overlap pullbacks, and
# element integrals below are NumPy (batched LAPACK determinants/solves of d x d
# frames, d <= 3). The runtime action is the prepared sparse factorization.


def _affine_frames(
    discretization: _NestedFiniteElementDiscretization, /
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical affine-cell origins and edge-column frames.

    Cells follow concatenated mesh block order. Affine admission verifies that
    coordinate-map corner images match the mesh vertices. Retained coefficient
    rows can be nonnodal and must not be reinterpreted as physical vertices.
    """

    coordinates = np.asarray(discretization.mesh.coordinates, dtype=np.float64)
    vertices = np.concatenate(
        tuple(
            coordinates[np.asarray(block.vertices)]
            for block in discretization.mesh.blocks
        )
    )
    return vertices[:, 0, :], np.swapaxes(vertices[:, 1:, :] - vertices[:, :1, :], -1, -2)


def _affine_measure_densities(frames: np.ndarray, /) -> np.ndarray:
    """Physical measure per unit reference measure of affine cells ``(C, m, d)``.

    Full-dimensional frames use ``|det F|``. An embedded affine immersion
    (``m > d``) has the algebraic area-formula density ``sqrt(det(F^T F))``, the same
    measure the exterior owner uses to map densities and fluxes.
    """
    if frames.shape[-2] == frames.shape[-1]:
        return np.abs(np.linalg.det(frames))
    if frames.shape[-2] < frames.shape[-1]:
        raise ValueError(
            "affine_measure_densities.frames must immerse their reference cell; got Affine cell frames must immerse their reference cell."
        )
    gram = np.swapaxes(frames, -1, -2) @ frames
    determinant = determinant_small_linear(_AFFINE_MEASURE_PLANS[frames.shape[-1]], gram)
    return np.sqrt(np.abs(np.asarray(jax.device_get(determinant))))


def _overlap_quadrature(
    simplices: np.ndarray, degree: int, /
) -> tuple[np.ndarray, np.ndarray, str]:
    """Physical points ``(S, Q, d)`` and weights ``(S, Q)`` algebraic to ``degree``."""

    dimension = simplices.shape[-1]
    rule = cubature_rule_data(_simplex_cubature_kind(_SIMPLEX_KINDS[dimension]), degree)
    frames = np.swapaxes(simplices[:, 1:, :] - simplices[:, :1, :], -1, -2)
    # Overlap simplices are fanned and positively oriented up to rounding; the
    # signed measure keeps each fan an algebraic partition of its overlap.
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Padded DOF routes, validity, and mapped basis of the cells.

    Values are the oriented Piola-mapped basis ``(S, Q, n, V)`` (``V = 1`` for
    scalar fields) and its curl/divergence ``(S, Q, n, W)``, ``None`` for scalar
    fields.
    """

    dof_map = discretization.dof_maps[field_index]
    elements = discretization.elements[field_index]
    blocks = discretization.mesh.blocks
    starts = np.cumsum((0,) + tuple(block.cell_count for block in blocks))
    width = max(routes.shape[1] for routes in dof_map.cell_dofs)
    count, quadrature, _ = points.shape
    value_width = elements[0].value_shape[0] if elements[0].value_shape else 1
    routes = np.zeros((count, width), dtype=np.int32)
    valid = np.zeros((count, width), dtype=np.bool_)
    values = np.zeros((count, quadrature, width, value_width), dtype=np.float64)
    differential: np.ndarray | None = None
    origins, frames = _affine_frames(discretization)
    # x = origin + frame @ xi on the affine cell; one batched solve pulls
    # every quadrature point of an overlap simplex back to reference coordinates.
    reference = np.linalg.solve(
        frames[cells][:, None], (points - origins[cells][:, None, :])[..., None]
    )[..., 0]
    for block_index, element in enumerate(elements):
        rows = np.flatnonzero(
            (cells >= starts[block_index]) & (cells < starts[block_index + 1])
        )
        if rows.size == 0:
            continue
        local = cells[rows] - starts[block_index]
        size = element.local_dof_count
        routes[rows, :size] = np.asarray(dof_map.cell_dofs[block_index])[local]
        valid[rows, :size] = True
        mapped, derivative = _mapped_basis(
            element,
            frames[cells[rows]],
            np.asarray(dof_map.orientations[block_index])[local],
            reference[rows],
            transformation=_basis_transformation(
                discretization, field_index, block_index, local
            ),
        )
        values[rows, :, :size] = mapped
        if derivative is not None:
            if differential is None:
                differential = np.zeros(
                    (count, quadrature, width, derivative.shape[-1]), dtype=np.float64
                )
            differential[rows, :, :size] = derivative
    return routes, valid, values, differential


def _cell_integrals(
    discretization: _NestedFiniteElementDiscretization,
    field_index: int,
    /,
    *,
    mass: bool,
    cell_measures: NDArray[np.float64] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    """Integrals of every global basis function and, optionally, mass COO triples.

    Integrals are ``(N,)`` for scalar fields and ``(N, V)`` for Piola-mapped
    vector fields; the mass pairs values by their Euclidean inner product. The
    per-block rule is algebraic for degree ``2p`` (mass) or ``p`` (integrals) of the
    basis polynomial degree ``p``.
    """

    dof_map = discretization.dof_maps[field_index]
    elements = discretization.elements[field_index]
    if cell_measures is None:
        _, frames = _affine_frames(discretization)
        determinants = _affine_measure_densities(frames)
    else:
        count = sum(block.cell_count for block in discretization.mesh.blocks)
        dimension = discretization.mesh.topological_dimension
        if cell_measures.shape != (count,) or any(
            element.mapping != "identity" or element.value_shape for element in elements
        ):
            raise ValueError(
                "cell_integrals.cell_measures must bind scalar affine reference bases in mesh cell order; got incompatible measures or elements."
            )
        frames = np.broadcast_to(
            np.eye(dimension, dtype=np.float64), (count, dimension, dimension)
        )
        determinants = cell_measures * factorial(dimension)
    size = dof_map.global_dof_count
    value_shape = elements[0].value_shape
    measures = np.zeros((size, value_shape[0] if value_shape else 1), dtype=np.float64)
    rows, columns, values = [], [], []
    start = 0
    for block_index, element in enumerate(elements):
        routes = np.asarray(dof_map.cell_dofs[block_index])
        stop = start + routes.shape[0]
        rule = cubature_rule_data(
            _simplex_cubature_kind(element.cell_kind),
            (2 if mass else 1) * _polynomial_degree(element),
        )
        basis, _ = _mapped_basis(
            element,
            frames[start:stop],
            np.asarray(dof_map.orientations[block_index]),
            np.asarray(rule.points, dtype=np.float64),
            transformation=_basis_transformation(
                discretization, field_index, block_index
            ),
        )
        weights = determinants[start:stop, None] * np.asarray(rule.weights)[None, :]
        integrals = np.asarray(ein.contract("cq,cqkv->ckv", weights, basis))
        # Unbuffered accumulation in route order, deterministic for fixed routes.
        np.add.at(
            measures, routes.reshape((-1,)), integrals.reshape((-1, measures.shape[1]))
        )
        if mass:
            local = np.asarray(ein.contract("cq,cqiv,cqjv->cij", weights, basis, basis))
            rows.append(np.broadcast_to(routes[:, :, None], local.shape).reshape((-1,)))
            columns.append(
                np.broadcast_to(routes[:, None, :], local.shape).reshape((-1,))
            )
            values.append(local.reshape((-1,)))
        start = stop
    measures = measures.reshape((size,) + value_shape)
    if not mass:
        return measures, None
    return measures, (
        np.concatenate(rows),
        np.concatenate(columns),
        np.concatenate(values),
    )


def coalesced_map(
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
    """Sum duplicate routes in declared (target, source) order.

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
    if triples is None:
        raise ValueError(
            "target_mass.configuration must provide cell mass entries; got Target mass assembly requires cell mass entries."
        )
    dof_map = target.dof_maps[field_index]
    mass = coalesced_map(
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
            {
                "kind": "finite-element-scalar-mass"
                if not target.elements[field_index][0].value_shape
                else "finite-element-piola-mass",
                "dof_map": dof_map.dof_map_id,
            }
        ),
    )
    return mass, measures


def _factored_target(
    target: FiniteElementDiscretization,
    field_name: str,
    plan: SparseFactorizationPlan | None,
    /,
    *,
    constraint_policy: FactorizationPolicy,
) -> PreparedL2ProjectionTarget:
    """Assemble, factor, and condition-estimate the target mass on one plan.

    ``plan`` is ``None`` for a cold preparation (symbolic analysis of the fresh
    pattern) and the retained symbolic plan for a numeric refresh.
    """

    mass, measures = _target_mass(target, target.field_index(field_name))
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
        constraint_policy=constraint_policy,
        _construction_token=_L2_ARTIFACT_TOKEN,
    )


@checked
def prepare_l2_projection_target(
    target: FiniteElementDiscretization,
    /,
    *,
    field_name: str,
    constraint_policy: FactorizationPolicy | None = None,
) -> PreparedL2ProjectionTarget:
    """Prepare the target space of Galerkin L2 projections onto one FE field.

    ``field_name`` names a scalar Lagrange field (continuous or discontinuous, any
    degree) or a Piola-mapped H(curl)/H(div) field of ``target`` on affine
    triangles or tetrahedra. The target mass is
    integrated exactly, its symbolic Cholesky pattern is planned under a reverse
    Cuthill-McKee ordering, and the numeric factor and condition estimate are
    computed once. Pass the result to :func:`prepare_l2_projection_transfer` for
    every source field and common refinement onto this target, and to
    :func:`refresh_l2_projection_target` when the target geometry moves with an
    unchanged DOF structure. Unsupported elements, a failed factorization, and a
    non-finite condition estimate raise ``ValueError``.
    Piola differential image/constraint factors use native dense SVD, bounded by
    ``constraint_policy``'s existing linalg materialization/workspace budgets.
    The default is ``FactorizationPolicy("svd")``; larger declared numerical
    budgets can be supplied without changing transfer semantics or certificates.
    """

    name = str(field_name)
    _projection_field_elements(target, target.field_index(name), "target")
    _require_affine_simplices(target, "target", "L2 projection")
    policy = (
        FactorizationPolicy("svd") if constraint_policy is None else constraint_policy
    )
    if policy.kind != "svd":
        raise ValueError(
            "prepare_l2_projection_target.constraint_policy must be a native SVD factorization policy; got Compatible constraint_policy must be a native SVD factorization policy."
        )
    return _factored_target(target, name, None, constraint_policy=policy)


@checked
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

    name = prepared.field_name
    index = target.field_index(name)
    _projection_field_elements(target, index, "target")
    _require_affine_simplices(target, "target", "L2 projection")
    previous = prepared.discretization
    if (
        target.dof_maps[index].dof_map_id
        != previous.dof_maps[previous.field_index(name)].dof_map_id
    ):
        raise ValueError(
            "refresh_l2_projection_target.dof_map_id must match the prepared field DOF structure; got "
            f"{target.dof_maps[index].dof_map_id!r}, prepared={previous.dof_maps[previous.field_index(name)].dof_map_id!r}, field={name!r}"
        )
    return _factored_target(
        target,
        name,
        prepared.factorization.plan,
        constraint_policy=FactorizationPolicy("svd")
        if prepared.compatible is None
        else prepared.compatible.image_factor.policy,
    )


def _coverage_bound(refinement: imported_geometry.PreparedCommonRefinement, /) -> float:
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


def _coverage_claims(
    refinement: imported_geometry.PreparedCommonRefinement, /
) -> tuple[bool, bool]:
    """Whether every target cell, and every source cell, is certified covered."""

    match refinement.policy.coverage:
        case imported_geometry.CommonRefinementCoverage.COMPLETE:
            return True, True
        case imported_geometry.CommonRefinementCoverage.TARGET:
            return True, False
        case imported_geometry.CommonRefinementCoverage.SOURCE:
            return False, True
        case imported_geometry.CommonRefinementCoverage.PARTIAL:
            return False, False
        case coverage:
            raise ValueError(
                f"coverage_claims.refinement must select a supported declared route; got Unknown common-refinement coverage {coverage!r}."
            )


@dataclass(frozen=True, slots=True)
class _ProjectionOverlap:
    """Host overlap data of one assembled projection, reused by its evidence."""

    projection: FiniteElementL2Projection
    target_cells: np.ndarray
    points: np.ndarray
    weights: np.ndarray
    source_routes: np.ndarray
    source_valid: np.ndarray
    source_differential: np.ndarray | None
    source_measures: np.ndarray | None = None


def _assembled_projection(
    source: FiniteElementDiscretization,
    target: PreparedL2ProjectionTarget,
    refinement: imported_geometry.PreparedCommonRefinement,
    field_name: str,
    /,
) -> tuple[FiniteElementTopologyTransfer, _ProjectionOverlap]:
    """The certified projection transfer and the overlap data it was built from."""

    source_name = str(field_name)
    source_index = source.field_index(source_name)
    space = target.discretization
    target_index = space.field_index(target.field_name)
    simplices, source_cells, target_cells = _validated_overlaps(
        refinement, source.mesh, space.mesh
    )
    source_elements = _projection_field_elements(source, source_index, "source")
    _require_affine_simplices(source, "source", "L2 projection")
    target_elements = space.elements[target_index]
    if (
        len(
            {
                element.value_spec.value_spec_id
                for element in source_elements + target_elements
            }
        )
        != 1
    ):
        raise ValueError(
            "assembled_projection.configuration must provide one scientific form type, twist, and physical proxy; got L2 projection requires one scientific form type, twist, and physical proxy."
        )
    if (source_elements[0].mapping, source_elements[0].value_shape) != (
        target_elements[0].mapping,
        target_elements[0].value_shape,
    ):
        raise ValueError(
            "assembled_projection.elements must share one Piola mapping and value shape; got "
            f"source=({source_elements[0].mapping!r}, {source_elements[0].value_shape!r}), target=({target_elements[0].mapping!r}, {target_elements[0].value_shape!r})"
        )
    points, weights, rule_id = _overlap_quadrature(
        simplices,
        max(_polynomial_degree(element) for element in source_elements)
        + max(_polynomial_degree(element) for element in target_elements),
    )
    target_routes, target_valid, target_basis, _ = _overlap_basis(
        space, target_index, target_cells, points
    )
    source_routes, source_valid, source_basis, source_differential = _overlap_basis(
        source, source_index, source_cells, points
    )
    local = np.asarray(
        ein.contract("sq,sqiv,sqkv->sik", weights, target_basis, source_basis)
    )
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
    mixed_mass = coalesced_map(
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
    # Nodal constant/linear/content claims are scalar semantics; Piola fields
    # certify reproduction and vector content in their transfer evidence.
    scalar = all(
        element.mapping == "identity"
        and not element.value_shape
        and element.representation == "point_value"
        for element in source_elements + target_elements
    )
    constants = target_covered and scalar
    linear = (
        constants
        and min(element.degree for element in source_elements) >= 1
        and min(element.degree for element in target_elements) >= 1
        and not source_dofs.association.startswith("quotient_")
        and not target_dofs.association.startswith("quotient_")
    )
    conservative = source_covered and scalar
    # Claims hold to the accuracy the refinement certified: an accepted relative
    # coverage defect r perturbs B by at most r times the basis Lebesgue constant,
    # and the target solve amplifies that by the mass condition.
    lebesgue = max(
        float(np.max(np.sum(np.abs(target_basis), axis=2))),
        float(np.max(np.sum(np.abs(source_basis), axis=2))),
    )
    coverage = _coverage_bound(refinement) / (CLAIM_ULPS * np.finfo(np.float64).eps)
    condition = float(target.mass_condition.value)
    transfer = FiniteElementTopologyTransfer(
        projection,
        source.mesh.topology_id,
        space.mesh.topology_id,
        preserves_constants=constants,
        preserves_linear=linear,
        conservative=conservative,
        action_condition=max(condition, 1.0) * max(lebesgue, 1.0) * (1.0 + coverage),
        semantics="l2-projection",
        source_coordinates=source_dofs.dof_coordinates if linear else None,
        target_coordinates=target_dofs.dof_coordinates if linear else None,
        source_measures=_cell_integrals(source, source_index, mass=False)[0]
        if conservative
        else None,
        target_measures=target.dof_measures if conservative else None,
    )
    return transfer, _ProjectionOverlap(
        projection,
        target_cells,
        points,
        weights,
        source_routes,
        source_valid,
        source_differential,
    )


@checked
def prepare_l2_projection_transfer(
    source: FiniteElementDiscretization,
    target: PreparedL2ProjectionTarget,
    refinement: imported_geometry.PreparedCommonRefinement,
    /,
    *,
    field_name: str,
) -> FiniteElementTopologyTransfer:
    """Prepare the Galerkin L2 projection of one FE field onto a non-matching mesh.

    ``target`` is a :func:`prepare_l2_projection_target` artifact and
    ``refinement`` the common refinement of ``source.mesh`` and the target mesh
    prepared with ``CommonRefinementPolicy(overlap_simplices=True)``. The source
    field ``field_name`` is a scalar Lagrange field (continuous or discontinuous,
    any degree) or a Piola-mapped H(curl)/H(div) field on affine triangles or
    tetrahedra, projected onto a target field of the same mapping; component
    axes of scalar fields are carried as payload. Only the mixed mass ``B`` is
    assembled here, exactly on the overlap simplices with a rule for the product
    of source and target polynomial degrees, pairing covariant or contravariant
    Piola values of both cells; the prepared target factor is shared
    (``FiniteElementL2Projection``). The transfer applies ``M_T^{-1} B`` and
    pulls duals back through ``B^T M_T^{-1}``.

    Scalar claims follow the certified coverage: constants (and, for degrees
    >= 1, linear fields) are preserved when every target cell is covered, and the
    integral is conserved when every source cell is covered. Claims are certified
    to the refinement's accepted coverage accuracy (its per-cell coverage
    tolerances), amplified by the basis Lebesgue constant and the target mass
    condition estimate. Positivity is never claimed. Piola-mapped fields make no
    nodal claims; :func:`prepare_projection_field_transfer` certifies their
    reproduction and content. Failed or mismatched refinements and unsupported
    elements raise ``ValueError``.
    """

    return _assembled_projection(source, target, refinement, field_name)[0]


def _differential_tests(points: np.ndarray, degree: int, /) -> np.ndarray:
    exponents = np.asarray(
        [
            powers
            for powers in product(range(degree + 1), repeat=points.shape[-1])
            if sum(powers) <= degree
        ],
        dtype=np.int64,
    )
    return np.prod(points[..., None, :] ** exponents, axis=-1)


def _native_generalized_inverse(
    matrix: np.ndarray, kind: str, policy: FactorizationPolicy, /
) -> tuple[PreparedFactorization, np.ndarray, float]:
    factor = factorize(
        DenseLinearOperator(
            matrix,
            operator_id=canonical_fingerprint(
                {"kind": kind, "matrix": array_tree_fingerprint(matrix)}
            ),
        ),
        policy,
    )
    inverse = factor.materialize_pseudoinverse()
    if not bool(inverse.successful) or not np.all(np.isfinite(np.asarray(inverse.value))):
        raise ValueError(
            f"native_generalized_inverse.kind must pass the declared scientific certificate; got Compatible projection {kind} factorization failed."
        )
    rank = int(factor.rank())
    spectrum = np.asarray(factor.singular_values())
    condition = 1.0 if rank == 0 else float(spectrum[0] / spectrum[rank - 1])
    return factor, np.asarray(inverse.value), condition


def _prepare_compatible_target(
    target: FiniteElementDiscretization,
    field_index: int,
    mass_factor: PreparedSparseFactorization,
    measures: np.ndarray,
    policy: FactorizationPolicy,
    /,
) -> _CompatibleProjectionTarget:
    """Prepare full polynomial differential moments, not just cell means."""

    elements = target.elements[field_index]
    degree = max(max(_polynomial_degree(element) - 1, 0) for element in elements)
    dimension = target.mesh.topological_dimension
    components = 3 if elements[0].mapping == "covariant_piola" and dimension == 3 else 1
    _, frames = _affine_frames(target)
    # Tiny reference-polynomial Gram factors are immutable host preparation,
    # like the affine-frame solves. Global rank/constraint solves use native linalg.
    rule = cubature_rule_data(
        _simplex_cubature_kind(elements[0].cell_kind), 2 * degree + 2
    )
    reference = np.asarray(rule.points, dtype=np.float64)
    tests = _differential_tests(reference, degree)
    gram = np.asarray(ein.contract("q,qp,qr->pr", np.asarray(rule.weights), tests, tests))
    cholesky = np.linalg.cholesky(gram)
    reference_whitening = np.linalg.solve(cholesky, np.eye(tests.shape[-1]))
    whitening = (
        reference_whitening[None]
        / np.sqrt(_affine_measure_densities(frames))[:, None, None]
    )
    count, width = frames.shape[0], tests.shape[-1] * components
    dof_map = target.dof_maps[field_index]
    constraint_count = count * width + measures.shape[-1]
    policy.materialization.require(
        max(constraint_count * dof_map.global_dof_count, constraint_count**2),
        jnp.dtype("float64"),
    )
    differential = np.zeros((count * width, dof_map.global_dof_count))
    start = 0
    for block_index, element in enumerate(elements):
        routes = np.asarray(dof_map.cell_dofs[block_index], dtype=np.int64)
        stop = start + routes.shape[0]
        _, derivative = _mapped_basis(
            element,
            frames[start:stop],
            np.asarray(dof_map.orientations[block_index]),
            reference,
            transformation=_basis_transformation(target, field_index, block_index),
        )
        if derivative is None:
            raise ValueError(
                "prepare_compatible_target.configuration must provide a differential field; got Compatible projection requires a differential field."
            )
        weights = (
            _affine_measure_densities(frames[start:stop])[:, None]
            * np.asarray(rule.weights)[None]
        )
        moments = np.asarray(ein.contract("cq,qp,cqnw->cpwn", weights, tests, derivative))
        moments = np.asarray(
            ein.contract("cpr,crwn->cpwn", whitening[start:stop], moments)
        ).reshape((stop - start, width, element.local_dof_count))
        rows = np.arange(start * width, stop * width).reshape((-1, width))
        np.add.at(
            differential,
            (rows[:, :, None], routes[:, None, :]),
            moments,
        )
        start = stop
    image_factor, image_inverse, image_condition = _native_generalized_inverse(
        differential, "finite-element-differential-image", policy
    )
    norms = np.linalg.norm(measures, axis=0)
    content_scale = 1.0 / np.maximum(norms, np.finfo(np.float64).tiny)
    constraints = np.concatenate((differential, (measures * content_scale).T), axis=0)
    mass_inverse = np.asarray(_target_mass_solve(mass_factor, jnp.asarray(constraints.T)))
    schur = constraints @ mass_inverse
    schur = 0.5 * (schur + schur.T)
    schur_factor, schur_inverse, schur_condition = _native_generalized_inverse(
        schur, "finite-element-compatible-schur", policy
    )
    return _CompatibleProjectionTarget(
        differential,
        constraints,
        whitening,
        content_scale,
        image_inverse,
        mass_inverse @ schur_inverse,
        image_factor,
        schur_factor,
        degree,
        components,
        max(image_condition, schur_condition, 1.0),
    )


class _CompatibleProjection(AbstractLinearOperator):
    """Mass-minimal correction with a realizable differential companion.

    In 3D the companion is the L2 projection of curl into the target discrete
    curl image; independent cellwise curl projections need not have continuous
    normal traces and cannot in general be curls. H(div), and planar H(curl),
    enforce the raw cellwise differential projection instead, and preparation
    refuses it if the supplied mesh/quotient makes it unrealizable.
    """

    _fused_block_action_kind = "fused"
    base: FiniteElementL2Projection
    raw_differential: SparseLinearMap
    source_measures: Array
    target_data: _CompatibleProjectionTarget
    project_image: bool = eqx.field(static=True)

    def __init__(
        self,
        base: FiniteElementL2Projection,
        raw_differential: SparseLinearMap,
        source_measures: np.ndarray,
        /,
        *,
        project_image: bool,
    ) -> None:
        data = base.prepared_target.compatible
        if data is None:
            raise ValueError(
                "_CompatibleProjection.configuration must agree with the declared geometry, topology, and field layout; got Compatible target differential factors are missing."
            )
        self.base = base
        self.raw_differential = raw_differential
        self.source_measures = jnp.asarray(source_measures)
        self.target_data = data
        self.project_image = project_image
        self.source, self.target = base.source, base.target
        self.batch_shape = ()
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True
        )
        self.operator_id = canonical_fingerprint(
            {
                "kind": "finite-element-compatible-projection",
                "base": base.operator_id,
                "differential": raw_differential.operator_id,
                "image_factor": data.image_factor.factorization_id,
                "schur_factor": data.schur_factor.factorization_id,
                "project_image": project_image,
            }
        )

    def companion(self, values: Array, /) -> Array:
        raw = self.raw_differential.mv(values)
        data = self.target_data
        return (
            data.differential @ (data.image_inverse @ raw) if self.project_image else raw
        )

    def constraint_rhs(self, values: Array, /) -> Array:
        content = (self.source_measures * self.target_data.content_scale).T @ values
        return jnp.concatenate((self.companion(values), content), axis=0)

    def constraint_transpose(self, dual: Array, /) -> Array:
        data = self.target_data
        size = data.differential.shape[0]
        differential = dual[:size]
        if self.project_image:
            differential = data.image_inverse.T @ (data.differential.T @ differential)
        return (
            self.raw_differential.transpose_mv(differential)
            + (self.source_measures * data.content_scale) @ dual[size:]
        )

    def mv(self, vector: ArrayLike, /) -> Array:
        values = jnp.asarray(vector)
        projected = self.base.mv(values)
        return projected + self.target_data.lift @ (
            self.constraint_rhs(values) - self.target_data.constraints @ projected
        )

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        values = jnp.asarray(vector)
        dual = self.target_data.lift.T @ values
        return self.base.transpose_mv(
            values - self.target_data.constraints.T @ dual
        ) + self.constraint_transpose(dual)

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        return generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        return materialize_by_basis(self)


def _polynomial_probes(element: FiniteElementSpec, points: np.ndarray, /) -> np.ndarray:
    """Probe fields ``(..., v, P)`` at points ``(..., d)`` in every mapped space.

    Scalar spaces use the constant plus the coordinates when the degree admits
    them. Piola spaces use constants plus ``x`` for H(div), or rotations for
    H(curl). Constant-order full form spaces use only constants; higher-order
    compatible spaces contain the appropriate linear probe.
    """

    if element.mapping == "identity":
        ones = np.ones(points.shape[:-1] + (1, 1), dtype=np.float64)
        if _polynomial_degree(element) == 0:
            return ones
        return np.concatenate((ones, points[..., None, :]), axis=-1)
    dimension = points.shape[-1]
    constants = np.broadcast_to(
        np.eye(dimension), points.shape[:-1] + (dimension, dimension)
    )
    if _polynomial_degree(element) == 0:
        return constants
    match element.mapping:
        case "contravariant_piola":
            linear = points[..., None]
        case "covariant_piola" if dimension == 2:
            linear = np.stack((-points[..., 1], points[..., 0]), axis=-1)[..., None]
        case "covariant_piola":
            linear = np.stack(
                tuple(
                    np.cross(np.broadcast_to(axis, points.shape), points)
                    for axis in np.eye(3)
                ),
                axis=-1,
            )
        case mapping:
            raise ValueError(
                f"polynomial_probes.configuration must select a supported declared route; got Mapping {mapping!r} has no compatible probe fields."
            )
    return np.concatenate((constants, linear), axis=-1)


def _periodic_probe_mask(
    discretization: FiniteElementDiscretization, field_index: int, /
) -> np.ndarray:
    """Admit affine probes by physical seam traces, independently of FE routes.

    Affine trace differences vanish on an entire affine seam precisely when
    they vanish at its spanning facet corners. These checks use declared
    representative-to-copy isometries and vertex permutations, never the
    assembled basis/routes being certified.
    """
    mesh = discretization.mesh
    element = discretization.elements[field_index][0]
    dimension = mesh.topological_dimension
    count = dimension + (
        1 if element.mapping == "contravariant_piola" or dimension == 2 else dimension
    )
    periodic = mesh.periodic_topology
    if periodic is None:
        return np.ones(count, dtype=np.bool_)
    degree = dimension - 1
    orbit = np.asarray(periodic.orbits(degree)[0], dtype=np.int64)
    representatives = np.asarray(periodic.orbit_representatives(degree), dtype=np.int64)
    copies = np.flatnonzero(np.arange(orbit.size) != representatives[orbit])
    if copies.size == 0:
        return np.ones(count, dtype=np.bool_)
    connectivity = mesh.connectivity
    match (dimension, connectivity):
        case (3, TetrahedralConnectivity() | PolyhedralConnectivity()):
            entities = tetrahedral_face_vertices(connectivity)
        case (2, PolygonalConnectivity()):
            entities = np.asarray(connectivity.edges)
        case _:
            raise ValueError(
                "periodic_probe_mask.configuration must provide simplex facet connectivity; got Periodic compatible probes require simplex facet connectivity."
            )
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    permutations = periodic.entity_vertex_permutations(mesh, degree)
    source_corners = coordinates[entities[representatives[orbit[copies]]]]
    source_corners = np.take_along_axis(
        source_corners,
        np.stack([permutations[index] for index in copies])[..., None],
        axis=1,
    )
    copy_corners = coordinates[entities[copies]]
    rotations = periodic.orbit_isometries(degree)[copies, :dimension, :dimension]
    source_values = _polynomial_probes(element, source_corners)
    copy_values = _polynomial_probes(element, copy_corners)
    expected = np.asarray(ein.contract("fij,fvjp->fvip", rotations, source_values))
    difference = copy_values - expected
    tangents = copy_corners[:, 1:] - copy_corners[:, :1]
    if element.mapping == "contravariant_piola":
        normals = (
            np.cross(tangents[:, 0], tangents[:, 1])
            if dimension == 3
            else np.stack((tangents[:, 0, 1], -tangents[:, 0, 0]), axis=-1)
        )
        directions = normals[:, None]
    else:
        directions = tangents
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    trace_defect = np.asarray(ein.contract("fvdp,fkd->fvkp", difference, directions))
    scale = np.maximum(
        np.max(np.abs(copy_values), axis=(0, 1, 2)),
        np.max(np.abs(expected), axis=(0, 1, 2)),
    )
    return np.max(np.abs(trace_defect), axis=(0, 1, 2)) <= (
        CLAIM_ULPS
        * np.finfo(np.float64).eps
        * np.maximum(scale, np.finfo(np.float64).tiny)
    )


def _probe_coefficients(
    discretization: FiniteElementDiscretization, field_index: int, /
) -> np.ndarray:
    """Global coefficients ``(N, P)`` of the polynomial probes in one Piola field.

    The probes lie in every cell's local space, so the per-cell L2 projection
    reproduces them exactly and every cell sharing a DOF yields its coefficient.
    """

    dof_map = discretization.dof_maps[field_index]
    origins, frames = _affine_frames(discretization)
    determinants = _affine_measure_densities(frames)
    blocks = []
    start = 0
    for block_index, element in enumerate(discretization.elements[field_index]):
        routes = np.asarray(dof_map.cell_dofs[block_index], dtype=np.int64)
        stop = start + routes.shape[0]
        rule = cubature_rule_data(
            _simplex_cubature_kind(element.cell_kind), 2 * _polynomial_degree(element)
        )
        reference = np.asarray(rule.points, dtype=np.float64)
        basis, _ = _mapped_basis(
            element,
            frames[start:stop],
            np.asarray(dof_map.orientations[block_index]),
            reference,
            transformation=_basis_transformation(
                discretization, field_index, block_index
            ),
        )
        physical = origins[start:stop, None, :] + reference[None] @ np.swapaxes(
            frames[start:stop], -1, -2
        )
        weights = determinants[start:stop, None] * np.asarray(rule.weights)[None, :]
        mass = np.asarray(ein.contract("cq,cqiv,cqjv->cij", weights, basis, basis))
        load = np.asarray(
            ein.contract(
                "cq,cqiv,cqvp->cip",
                weights,
                basis,
                _polynomial_probes(element, physical),
            )
        )
        # Host-only preparation: batched LAPACK solves of the local element
        # masses, as for the nested local reconstruction.
        local = np.linalg.solve(mass, load)
        blocks.append((routes.reshape((-1,)), local.reshape((-1, local.shape[-1]))))
        start = stop
    coefficients = np.zeros(
        (dof_map.global_dof_count, blocks[0][1].shape[-1]), dtype=np.float64
    )
    for routes, values in blocks:
        coefficients[routes] = values
    return coefficients


def _cell_differentials(
    discretization: FiniteElementDiscretization, field_index: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Padded routes ``(C, n)``, validity, and cell integrals ``(C, n, W)`` of the
    curl/divergence of every oriented basis function of one Piola field."""

    dof_map = discretization.dof_maps[field_index]
    _, frames = _affine_frames(discretization)
    determinants = _affine_measure_densities(frames)
    width = max(routes.shape[1] for routes in dof_map.cell_dofs)
    routes = np.zeros((frames.shape[0], width), dtype=np.int64)
    valid = np.zeros((frames.shape[0], width), dtype=np.bool_)
    integrals: np.ndarray | None = None
    start = 0
    for block_index, element in enumerate(discretization.elements[field_index]):
        block_routes = np.asarray(dof_map.cell_dofs[block_index], dtype=np.int64)
        stop = start + block_routes.shape[0]
        size = element.local_dof_count
        rule = cubature_rule_data(
            _simplex_cubature_kind(element.cell_kind), _polynomial_degree(element)
        )
        _, differential = _mapped_basis(
            element,
            frames[start:stop],
            np.asarray(dof_map.orientations[block_index]),
            np.asarray(rule.points, dtype=np.float64),
            transformation=_basis_transformation(
                discretization, field_index, block_index
            ),
        )
        if differential is None:
            raise ValueError(
                "cell_differentials.configuration must select a supported declared route; got Scalar fields have no curl or divergence."
            )
        weights = determinants[start:stop, None] * np.asarray(rule.weights)[None, :]
        if integrals is None:
            integrals = np.zeros(
                (frames.shape[0], width, differential.shape[-1]), dtype=np.float64
            )
        integrals[start:stop, :size] = np.asarray(
            ein.contract("cq,cqkw->ckw", weights, differential)
        )
        routes[start:stop, :size] = block_routes
        valid[start:stop, :size] = True
        start = stop
    if integrals is None:
        raise ValueError(
            "cell_differentials.configuration must provide at least one cell; got A Piola field needs at least one cell."
        )
    return routes, valid, integrals


def _relative(defect: np.ndarray, reference: np.ndarray, /) -> float:
    scale = max(float(np.max(np.abs(reference), initial=0.0)), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(defect), initial=0.0)) / scale


def _constrained_projection(
    source: FiniteElementDiscretization,
    source_index: int,
    target: PreparedL2ProjectionTarget,
    refinement: imported_geometry.PreparedCommonRefinement | None,
    overlap: _ProjectionOverlap,
    /,
) -> FiniteElementTopologyTransfer:
    space = target.discretization
    target_index = space.field_index(target.field_name)
    element = _field_signature(source, source_index, space, target_index)
    if refinement is not None and not all(_coverage_claims(refinement)):
        raise ValueError(
            "constrained_projection.source must provide full source and target coverage; got Compatible projection requires full source and target coverage."
        )
    if refinement is None and overlap.source_measures is None:
        raise ValueError(
            "constrained_projection.source must provide its integrated source content measures; got Reference coarsening requires its integrated source content measures."
        )
    data = target.compatible
    derivative = overlap.source_differential
    if data is None or derivative is None:
        raise ValueError(
            "constrained_projection.configuration must provide prepared Piola differential factors; got Compatible projection requires prepared Piola differential factors."
        )
    origins, frames = _affine_frames(space)
    cells = overlap.target_cells
    reference = _pullback(origins[cells], frames[cells], overlap.points)
    tests = _differential_tests(reference, data.degree)
    tests = np.asarray(
        ein.contract("spr,sqr->sqp", np.asarray(data.whitening)[cells], tests)
    )
    moments = np.asarray(
        ein.contract("sq,sqp,sqnw->spwn", overlap.weights, tests, derivative)
    ).reshape((cells.size, -1, overlap.source_routes.shape[1]))
    width = moments.shape[1]
    rows = cells[:, None] * width + np.arange(width)[None]
    valid = np.broadcast_to(overlap.source_valid[:, None, :], moments.shape)
    raw = coalesced_map(
        np.broadcast_to(rows[:, :, None], moments.shape)[valid],
        np.broadcast_to(overlap.source_routes[:, None, :], moments.shape)[valid],
        moments[valid],
        target_size=data.differential.shape[0],
        source_size=source.dof_maps[source_index].global_dof_count,
        properties=OperatorProperties(),
        operator_id=canonical_fingerprint(
            {
                "kind": "finite-element-common-refinement-differential",
                "base": overlap.projection.operator_id,
                "degree": data.degree,
            }
        ),
    )
    projection = _CompatibleProjection(
        overlap.projection,
        raw,
        _cell_integrals(source, source_index, mass=False)[0]
        if overlap.source_measures is None
        else overlap.source_measures,
        project_image=element.mapping == "covariant_piola"
        and space.mesh.topological_dimension == 3,
    )
    return FiniteElementTopologyTransfer(
        projection,
        source.mesh.topology_id,
        space.mesh.topology_id,
        semantics=_nested_semantics(element, True),
        action_condition=max(float(target.mass_condition.value), data.condition, 1.0)
        * max(_frame_condition(frames), 1.0),
    )


def _compatible_projection_evidence(
    source: FiniteElementDiscretization,
    source_index: int,
    target: PreparedL2ProjectionTarget,
    refinement: imported_geometry.PreparedCommonRefinement | None,
    transfer: FiniteElementTopologyTransfer,
    overlap: _ProjectionOverlap,
    /,
) -> FiniteElementTransferEvidence:
    """All-column commuting/content certificate of the constrained projection.

    In H(div) the companion is the cellwise L2 divergence projection. In 3D
    H(curl) it is the L2 projection into the conforming target discrete curl
    image, not an unrealizable discontinuous cellwise vector projection.
    ``cellwise-curl-realizability`` reports that distinction without weakening
    or renaming the required commuting residual. No DOFs are sampled.
    """

    space = target.discretization
    target_index = space.field_index(target.field_name)
    projection = transfer.primal
    if not isinstance(projection, _CompatibleProjection):
        raise TypeError(
            "compatible_projection_evidence.configuration must provide a constrained projection; got Compatible evidence needs a constrained projection."
        )
    differential = overlap.source_differential
    if differential is None:
        raise ValueError(
            "compatible_projection_evidence.source must provide a Piola source field; got Compatible projection evidence needs a Piola source field."
        )
    if refinement is None and overlap.source_measures is None:
        raise ValueError(
            "compatible_projection_evidence.source must provide integrated source content measures; got Reference coarsening evidence requires integrated source content measures."
        )
    coefficient_space = projection.source
    if not isinstance(coefficient_space, ArraySpace):
        raise TypeError(
            "compatible_projection_evidence.source must provide an ArraySpace source; got Compatible projection evidence requires an ArraySpace source."
        )

    def apply(values: ArrayLike, /) -> np.ndarray:
        return np.asarray(_block_action(projection.mv_block, values), dtype=np.float64)

    def pullback(values: ArrayLike, /) -> np.ndarray:
        return np.asarray(
            _block_action(projection.transpose_mv_block, values), dtype=np.float64
        )

    probe_mask = _periodic_probe_mask(source, source_index) & _periodic_probe_mask(
        space, target_index
    )
    probes = _probe_coefficients(source, source_index)[:, probe_mask]
    target_probes = _probe_coefficients(space, target_index)[:, probe_mask]
    probe_count = int(np.count_nonzero(probe_mask))
    if probe_count:
        projected = apply(probes)
        loads = np.asarray(_block_action(overlap.projection.mixed_mass.mv_block, probes))
        base_probes = np.asarray(_block_action(overlap.projection.mv_block, probes))
        residual = np.asarray(_block_action(target.mass.mv_block, base_probes)) - loads
    source_measures = (
        _cell_integrals(source, source_index, mass=False)[0]
        if overlap.source_measures is None
        else overlap.source_measures
    )

    routes, valid, integrals = _cell_differentials(space, target_index)
    width = integrals.shape[-1]
    target_content = np.zeros((target.mass.output_shape[0], width), dtype=np.float64)
    np.add.at(target_content, routes[valid], integrals[valid])
    overlap_integrals = np.asarray(
        ein.contract("sq,sqjw->sjw", overlap.weights, differential)
    )
    source_size = source.dof_maps[source_index].global_dof_count
    source_content = np.zeros((source_size, width), dtype=np.float64)
    np.add.at(
        source_content,
        overlap.source_routes[overlap.source_valid],
        overlap_integrals[overlap.source_valid],
    )

    data = projection.target_data
    constraint_count = data.constraints.shape[0]
    differential_count = data.differential.shape[0]
    commuting_error = commuting_scale = 0.0
    raw_error = raw_scale = 0.0
    # algebraic algebraic transpose checks every source column, in bounded RHS
    # blocks, rather than estimating the invariant from a selection of DOFs.
    for start in range(0, differential_count, 32):
        stop = min(start + 32, differential_count)
        units = np.zeros((constraint_count, stop - start))
        units[np.arange(start, stop), np.arange(stop - start)] = 1.0
        transferred = pullback(np.asarray(data.constraints)[start:stop].T)
        expected = np.asarray(projection.constraint_transpose(jnp.asarray(units)))
        raw = np.asarray(
            projection.raw_differential.transpose_mv_block(
                jnp.asarray(units[:differential_count])
            )
        )
        commuting_error = max(
            commuting_error, float(np.max(np.abs(transferred - expected)))
        )
        commuting_scale = max(commuting_scale, float(np.max(np.abs(expected))))
        raw_error = max(raw_error, float(np.max(np.abs(transferred - raw))))
        raw_scale = max(raw_scale, float(np.max(np.abs(raw))))

    match source.elements[source_index][0].mapping:
        case "contravariant_piola":
            content_name = "divergence-content"
        case "covariant_piola":
            content_name = "curl-content"
        case mapping:
            raise ValueError(
                f"compatible_projection_evidence.configuration must use a covariant or contravariant Piola map; got Mapping {mapping!r} is not a compatible Piola map."
            )
    certified = {
        "coverage": 0.0 if refinement is None else _coverage_bound(refinement),
        "commuting": commuting_error / max(commuting_scale, np.finfo(np.float64).tiny),
        content_name: _relative(
            pullback(target_content) - source_content, source_content
        ),
        "content": _relative(
            pullback(target.dof_measures) - source_measures, source_measures
        ),
    }
    estimates: dict[str, float] = {}
    if probe_count:
        certified["reproduction"] = _relative(projected - target_probes, target_probes)
        estimates["solve-residual"] = _relative(residual, loads)
    if (
        source.mesh.periodic_topology is not None
        or space.mesh.periodic_topology is not None
    ):
        estimates["polynomial-probes"] = float(probe_count)
    if projection.project_image:
        estimates["cellwise-curl-realizability"] = raw_error / max(
            raw_scale, np.finfo(np.float64).tiny
        )
    tolerance = (
        CLAIM_ULPS
        * float(np.finfo(np.dtype(coefficient_space.dtype)).eps)
        * transfer.action_condition
    )
    return FiniteElementTransferEvidence(
        certified, max(tolerance, certified["coverage"]), estimates=estimates
    )


@final
class FiniteElementTransferEvidence(StrictModule, NonTrainableState):
    """Certificate of one prepared finite-element field transfer.

    ``defects`` are certified relative defects in declared name order.
    ``containment`` is the largest barycentric excursion of a target cell outside
    its parent chart; ``reproduction`` the residual of reconstructing every
    source basis function (or, for projections, every reproducible polynomial
    probe) in the target space; ``continuity`` the disagreement of shared target
    DOFs computed from different cells (trace continuity and
    orientation); ``commuting`` the curl or divergence of the transferred field
    against the transferred curl or divergence; ``coverage`` the accepted
    common-refinement coverage of a projection; ``content`` the conserved total
    integral of a projected vector field. ``passed`` holds when every defect is
    at most ``tolerance``.

    ``estimates`` are measured, uncertified relative quantities in declared
    name order, including the uncorrected mass-solve residual and the
    unrealizability of a discontinuous cellwise curl projection. The 3D curl
    companion is L2 projection into the target discrete curl image. Estimates
    inform the consumer and never decide ``passed``.

    ``bounds`` are mathematically certified absolute error bounds in the
    quantity's physical units, distinct from relative defects and uncertified
    diagnostics. In particular ``bound("content")`` bounds every source column
    of the conserved-measure functional, including integration/publication
    errors, and is carried into the epoch content ledger.
    """

    defects: tuple[tuple[str, float], ...] = eqx.field(static=True)
    estimates: tuple[tuple[str, float], ...] = eqx.field(static=True)
    bounds: tuple[tuple[str, float], ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    execution_evidence: NativeExecutionRecord | None

    @checked
    def __init__(
        self,
        defects: Mapping[str, float],
        tolerance: float,
        /,
        *,
        estimates: Mapping[str, float] | None = None,
        bounds: Mapping[str, float] | None = None,
        execution_evidence: imported_meshing__measurements.NativeExecutionRecord
        | None = None,
    ) -> None:
        records = _evidence_records(defects, "defect")
        measured = _evidence_records({} if estimates is None else estimates, "estimate")
        certified_bounds = _evidence_records({} if bounds is None else bounds, "bound")
        bound = float(tolerance)
        if not records:
            raise ValueError(
                "FiniteElementTransferEvidence.configuration must provide at least one certified defect; got Transfer evidence needs at least one certified defect."
            )
        if not isfinite(bound) or bound < 0.0:
            raise ValueError(
                "FiniteElementTransferEvidence.tolerance must be finite and nonnegative; got Transfer evidence tolerance must be finite and nonnegative."
            )
        if {name for name, _ in records} & {name for name, _ in measured}:
            raise ValueError(
                "FiniteElementTransferEvidence.configuration must keep certified defects disjoint from estimates; got A transfer quantity is either certified or estimated."
            )
        if {name for name, _ in certified_bounds} & {name for name, _ in measured}:
            raise ValueError(
                "FiniteElementTransferEvidence.configuration must keep absolute bounds disjoint from estimates; got An absolute error quantity is either bounded or estimated."
            )
        if execution_evidence is not None:
            execution_evidence.require_valid()
            if int(np.asarray(execution_evidence.status)) != 0:
                raise ValueError(
                    "FiniteElementTransferEvidence.configuration must have successful execution evidence before certifying a transfer; got A refused source execution allowance cannot certify a field transfer."
                )
        self.execution_evidence = execution_evidence
        self.defects = records
        self.estimates = measured
        self.bounds = certified_bounds
        self.tolerance = bound
        self.passed = all(value <= bound for _, value in records)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "finite-element-transfer-evidence",
                "defects": [list(record) for record in records],
                "estimates": [list(record) for record in measured],
                "bounds": [list(record) for record in certified_bounds],
                "tolerance": bound,
                "execution_evidence": None
                if execution_evidence is None
                else array_tree_fingerprint(execution_evidence),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[FiniteElementTransferEvidence],
        value: FiniteElementTransferEvidence,
        /,
    ) -> FiniteElementTransferEvidence:
        """Reconstruct persisted quantities and verify their evidence identity."""
        if type(value) is not cls:
            raise TypeError(
                f"FiniteElementTransferEvidence.stored_fields must have the registered {cls.__qualname__} type; got {type(value).__qualname__}"
            )
        rebuilt = cls(
            dict(value.defects),
            value.tolerance,
            estimates=dict(value.estimates),
            bounds=dict(value.bounds),
            execution_evidence=value.execution_evidence,
        )
        if rebuilt.evidence_id != value.evidence_id or rebuilt.passed != value.passed:
            raise ValueError(
                "FiniteElementTransferEvidence.stored_fields must retain certified quantities and execution evidence; got changed evidence identity or acceptance"
            )
        return rebuilt

    def defect(self, name: str, /) -> float:
        for key, value in self.defects:
            if key == name:
                return value
        raise KeyError(
            f"FiniteElementTransferEvidence.defect.name must select a supported declared route; got Transfer evidence has no {name!r} defect."
        )

    def estimate(self, name: str, /) -> float:
        for key, value in self.estimates:
            if key == name:
                return value
        raise KeyError(
            f"FiniteElementTransferEvidence.estimate.name must select a supported declared route; got Transfer evidence has no {name!r} estimate."
        )

    def bound(self, name: str, /) -> float:
        for key, value in self.bounds:
            if key == name:
                return value
        raise KeyError(
            f"FiniteElementTransferEvidence.bound.name must select a supported declared route; got Transfer evidence has no {name!r} absolute bound."
        )


def _evidence_records(
    values: Mapping[str, float], kind: str, /
) -> tuple[tuple[str, float], ...]:
    records = tuple(
        sorted(
            (canonical_identifier(name, f"{kind} name"), float(value))
            for name, value in values.items()
        )
    )
    if any(not isfinite(value) or value < 0.0 for _, value in records):
        raise ValueError(
            f"evidence_records.kind must be finite and nonnegative; got Transfer {kind}s must be finite and nonnegative."
        )
    return records


def _payload_operator(
    primal: AbstractLinearOperator, source: ArraySpace, target: ArraySpace, /
) -> AbstractLinearOperator:
    """The DOF-axis transfer acting on field arrays with trailing component axes."""

    if primal.source.compatible(source) and primal.target.compatible(target):
        return primal

    def action(values: Array) -> Array:
        return _block_action(primal.mv_block, values)

    def transpose_action(values: Array) -> Array:
        return _block_action(primal.transpose_mv_block, values)

    return FunctionLinearOperator(
        action,
        source=source,
        target=target,
        transpose_action=transpose_action,
        operator_id=canonical_fingerprint(
            {
                "kind": "finite-element-payload-transfer",
                "primal": primal.operator_id,
                "source_shape": list(source.shape),
                "target_shape": list(target.shape),
            }
        ),
    )


@final
class FiniteElementFieldTransfer(StrictModule, NonTrainableState):
    """One finite-element field transfer with semantics, geometry, and evidence.

    ``transfer`` owns the certified primal map and claims; ``source_field`` and
    ``target_field`` are the represented field spaces it maps (their DOF functionals
    selected the route, never array widths); ``geometry`` binds the coordinate
    maps; ``evidence`` certifies nesting, continuity, and commuting defects.
    ``source_measures``/``target_measures`` are the DOF integrals of a
    conservative scalar transfer. :meth:`epoch_transition` is the
    ``CompositionRebind`` physical-remap transport of this transfer.
    """

    transfer: FiniteElementTopologyTransfer
    source_field: DiscreteFieldSpace
    target_field: DiscreteFieldSpace
    source_measures: Array | None
    target_measures: Array | None
    geometry: TransferGeometryBinding
    evidence: FiniteElementTransferEvidence
    transfer_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        transfer: FiniteElementTopologyTransfer,
        source_field: DiscreteFieldSpace,
        target_field: DiscreteFieldSpace,
        geometry: TransferGeometryBinding,
        evidence: FiniteElementTransferEvidence,
        /,
        *,
        source_measures: ArrayLike | None = None,
        target_measures: ArrayLike | None = None,
    ) -> None:

        spaces = (source_field.vector_space, target_field.vector_space)
        if (
            not isinstance(spaces[0], ArraySpace)
            or not isinstance(spaces[1], ArraySpace)
            or spaces[0].shape[:1] != (transfer.source_size,)
            or spaces[1].shape[:1] != (transfer.target_size,)
            or spaces[0].shape[1:] != spaces[1].shape[1:]
        ):
            raise ValueError(
                "FiniteElementFieldTransfer.field_spaces must lead with the source/target DOF axes and share component axes; got "
                f"source={spaces[0]!r}, target={spaces[1]!r}, source_size={transfer.source_size}, target_size={transfer.target_size}"
            )
        if (source_measures is None) != (target_measures is None) or (
            source_measures is None
        ) == transfer.conservative:
            raise ValueError(
                "FiniteElementFieldTransfer.configuration must provide both measures if and only if the transfer is conservative; got Exactly the conservative transfers carry source and target measures."
            )
        measures = (
            None
            if source_measures is None
            else jnp.asarray(source_measures).reshape((transfer.source_size,)),
            None
            if target_measures is None
            else jnp.asarray(target_measures).reshape((transfer.target_size,)),
        )
        if (
            geometry.source_topology_id != transfer.source_topology_id
            or geometry.target_topology_id != transfer.target_topology_id
        ):
            raise ValueError(
                "FiniteElementFieldTransfer.transfer must realize the field transfer's topologies; got Geometry binding does not realize the field transfer's topologies."
            )
        self.transfer = transfer
        self.source_field = source_field
        self.target_field = target_field
        self.source_measures, self.target_measures = measures
        self.geometry = geometry
        self.evidence = evidence
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "finite-element-field-transfer",
                "transfer": transfer.transfer_id,
                "source_field": source_field.field_space_id,
                "target_field": target_field.field_space_id,
                "geometry": geometry.binding_id,
                "evidence": evidence.evidence_id,
            }
        )

    @property
    def semantics(self) -> TransferSemantics:
        return self.transfer.semantics

    def _field_transfer(self, conservative: bool, /) -> FieldTransfer:
        source_space = self.source_field.vector_space
        target_space = self.target_field.vector_space
        if not isinstance(source_space, ArraySpace) or not isinstance(
            target_space, ArraySpace
        ):
            raise RuntimeError(
                "FiniteElementFieldTransfer._field_transfer.configuration must have ArraySpace coefficient spaces; got Finite-element field spaces lost their ArraySpace."
            )
        primal = _payload_operator(self.transfer.primal, source_space, target_space)
        declared = self.transfer.hilbert_adjoint
        hilbert = (
            adjoint(primal)
            if declared is None
            else _payload_operator(declared, target_space, source_space)
        )
        transfer = self.transfer
        exact_on = (
            ("constants", "linears") if transfer.preserves_linear else ("constants",)
        )
        return FieldTransfer(
            self.source_field,
            self.target_field,
            primal,
            dual_pullback_operator=transpose(primal),
            hilbert_adjoint_operator=hilbert,
            properties=TransferProperties(
                constant_preserving=transfer.preserves_constants,
                conservative=conservative,
                positivity_preserving=transfer.positivity_preserving,
                nested=transfer.semantics == "nested-interpolation",
                adjoint_paired=True,
                differentiable_geometry=False,
                exact_on=exact_on if transfer.preserves_constants else (),
                semantics=transfer.semantics,
            ),
            geometry=self.geometry,
            transfer_id=self.transfer_id,
        )

    def field_transfer(self) -> FieldTransfer:
        """The generic field transfer with this transfer's certified claims."""

        return self._field_transfer(self.transfer.conservative)

    @checked
    def epoch_transition(
        self, source_epoch: TopologyEpoch, target_epoch: TopologyEpoch, /
    ) -> TopologyEpochTransition | FieldEpochTransition:
        """Bind this transfer between two topology epochs for a composition rebind.

        A conservative transfer with positive DOF measures forms a
        :class:`TopologyEpochTransition` whose content ledger carries its
        certified measure-defect bound (component axes share the DOF measure).
        Every other transfer forms a :class:`FieldEpochTransition` whose transport
        succeeds only if this transfer's evidence passed. Both expose
        ``composition_transport(source_entry, target_entry)``.
        """

        transfer = self.transfer
        if (
            source_epoch.topology_id != transfer.source_topology_id
            or target_epoch.topology_id != transfer.target_topology_id
        ):
            raise ValueError(
                "FiniteElementTopologyTransfer.epoch_transition.configuration must realize this transfer's topologies; got Topology epochs do not realize this transfer's topologies."
            )
        if (
            source_epoch.geometry_id != self.geometry.source_geometry_id
            or target_epoch.geometry_id != self.geometry.target_geometry_id
        ):
            raise ValueError(
                "FiniteElementFieldTransfer.epoch_transition.configuration must realize this transfer's represented geometry; got Topology epochs do not realize this transfer's represented geometry."
            )
        source_measures, target_measures = self.source_measures, self.target_measures
        if (
            source_measures is None
            or target_measures is None
            or not bool(jnp.all(source_measures > 0.0))
            or not bool(jnp.all(target_measures > 0.0))
        ):
            return FieldEpochTransition(
                source_epoch,
                target_epoch,
                self._field_transfer(False),
                evidence_passed=self.evidence.passed,
                evidence_id=self.evidence.evidence_id,
            )
        source_mass = np.asarray(source_measures, dtype=np.float64)
        target_mass = np.asarray(target_measures, dtype=np.float64)
        components = self.source_field.vector_space.size // transfer.source_size
        bound = measure_defect_bound(
            np.dtype(np.float64), transfer.action_condition, source_mass, target_mass
        )
        certified_content = dict(self.evidence.bounds).get("content")
        if certified_content is not None:
            bound = max(bound, certified_content)
        return TopologyEpochTransition(
            source_epoch,
            target_epoch,
            self._field_transfer(True),
            np.repeat(source_mass, components),
            np.repeat(target_mass, components),
            measure_defect_bound=bound,
        )


def _field_signature(
    source: _NestedFiniteElementDiscretization,
    source_index: int,
    target: _NestedFiniteElementDiscretization,
    target_index: int,
    /,
) -> FiniteElementSpec:
    """The one element family both fields use on every block, or a refusal."""

    elements = source.elements[source_index] + target.elements[target_index]
    signatures = {
        (
            element.family,
            element.degree,
            element.conformity,
            element.representation,
            element.mapping,
            element.value_shape,
            element.cell_kind,
            element.value_spec.value_spec_id,
            element.tabulator_id,
        )
        for element in elements
    }
    if len(signatures) != 1:
        raise ValueError(
            "field_signature.elements must share one family, degree, conformity, representation, mapping, value shape, cell kind, value specification, and tabulator on both meshes; got "
            f"{sorted(signatures)!r}"
        )
    element = elements[0]
    if element.mapping == "identity" and element.value_shape:
        raise ValueError(
            "field_signature.configuration must select a supported declared route; got Identity-mapped vector elements have no nested transfer."
        )
    if element.form_basis is None and element.mapping != "identity":
        raise ValueError(
            "field_signature.configuration must provide declared form-basis metadata; got Mapped finite-element transfer requires declared form-basis metadata."
        )
    return element


def _cell_routes(
    discretization: _NestedFiniteElementDiscretization, field_index: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated-block DOF routes ``(C, n)`` and orientation signs ``(C, n)``."""

    dof_map = discretization.dof_maps[field_index]
    return (
        np.concatenate([np.asarray(routes) for routes in dof_map.cell_dofs]).astype(
            np.int64
        ),
        np.concatenate(
            [np.asarray(signs, dtype=np.float64) for signs in dof_map.orientations]
        ),
    )


def _basis_transformation(
    discretization: _NestedFiniteElementDiscretization,
    field_index: int,
    block_index: int,
    cells: np.ndarray | None = None,
    /,
) -> np.ndarray | None:
    transformations = discretization.dof_maps[field_index].cell_transforms
    block = np.asarray(transformations[block_index], dtype=np.float64)
    return block if cells is None else block[cells]


def _cell_transformations(
    discretization: _NestedFiniteElementDiscretization, field_index: int, /
) -> np.ndarray | None:
    transformations = discretization.dof_maps[field_index].cell_transforms
    return np.concatenate(
        [np.asarray(block, dtype=np.float64) for block in transformations]
    )


@eqx.filter_jit
def _tabulate_basis(element: FiniteElementSpec, points: Array, /) -> tuple[Array, Array]:
    return element.tabulate(points)


def tabulated(
    element: FiniteElementSpec, points: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Reference basis values and gradients at points ``(..., d)``.

    Points are padded to a power-of-two count so varying cell counts reuse a
    logarithmic number of compilations; padded rows are discarded.
    """

    flat = points.reshape((-1, points.shape[-1]))
    padded = np.zeros(
        (1 << max(flat.shape[0] - 1, 0).bit_length(), flat.shape[1]), dtype=np.float64
    )
    padded[: flat.shape[0]] = flat
    values, gradients = _tabulate_basis(element, jnp.asarray(padded))
    values_ = np.asarray(values, dtype=np.float64)[: flat.shape[0]]
    gradients_ = np.asarray(gradients, dtype=np.float64)[: flat.shape[0]]
    return (
        values_.reshape(points.shape[:-1] + values_.shape[1:]),
        gradients_.reshape(points.shape[:-1] + gradients_.shape[1:]),
    )


def _mapped_basis(
    element: FiniteElementSpec,
    frames: np.ndarray,
    orientation: np.ndarray,
    points: np.ndarray,
    /,
    *,
    transformation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Oriented physical basis ``(C, Q, n, V)`` and its curl/divergence ``(C, Q, n, W)``.

    Reference points are ``(C, Q, d)`` per cell or ``(Q, d)`` shared by every
    cell. Mapping uses the element's declared form value specification, including
    its twist and physical proxy. full cell transforms map global routed
    coefficients to reference coefficients and include entity orientation.
    """

    values, gradients = tabulated(element, points)
    if points.ndim == 2:
        values = np.broadcast_to(values, (frames.shape[0],) + values.shape)
        gradients = np.broadcast_to(gradients, (frames.shape[0],) + gradients.shape)
    if element.form_basis is not None:
        mapped = np.asarray(
            imported_exterior__algebra.map_reference_values(
                values, element.value_spec, jnp.asarray(frames[:, None, None])
            )
        )
        mapped = mapped.reshape((*mapped.shape[:3], -1))
        degree = element.value_spec.form_type.degree
        differential = None
        if degree < element.topological_dimension:
            component_gradient = jnp.stack(
                tuple(
                    imported_exterior__algebra.vector_to_form(
                        jnp.asarray(gradients[..., axis]), element.value_spec
                    )
                    for axis in range(element.topological_dimension)
                ),
                axis=-1,
            )
            scientific = element.value_spec.form_type
            derivative_type = scientific.exterior_derivative_type()
            proxy = (
                "density"
                if derivative_type.degree == derivative_type.dimension
                else "circulation"
                if derivative_type.degree == 1
                else "flux"
            )
            derivative_spec = imported_exterior__form_type.FormValueSpec(
                derivative_type, proxy=proxy
            )
            derivative = imported_exterior__algebra.form_to_vector(
                imported_exterior__algebra.exterior_derivative_from_jacobian(
                    component_gradient, scientific
                ),
                derivative_spec,
            )
            differential = np.asarray(
                imported_exterior__algebra.map_reference_values(
                    derivative, derivative_spec, jnp.asarray(frames[:, None, None])
                )
            )
            differential = differential.reshape((*differential.shape[:3], -1))
        if transformation is not None:
            mapped = np.asarray(ein.contract("cji,cqjv->cqiv", transformation, mapped))
            if differential is not None:
                differential = np.asarray(
                    ein.contract("cji,cqjw->cqiw", transformation, differential)
                )
        else:
            mapped = mapped * orientation[:, None, :, None]
            if differential is not None:
                differential = differential * orientation[:, None, :, None]
        return mapped, differential
    if element.mapping != "identity":
        raise ValueError(
            "field_signature.configuration must provide declared form-basis metadata; got Mapped finite-element transfer requires declared form-basis metadata."
        )
    mapped = values[..., None]
    if transformation is not None:
        mapped = np.asarray(ein.contract("cji,cqjv->cqiv", transformation, mapped))
    else:
        mapped = mapped * orientation[:, None, :, None]
    return mapped, None


def _parent_witness(
    parent_cells: ArrayLike | None, target_count: int, source_count: int, /
) -> np.ndarray:
    if parent_cells is None:
        raise ValueError(
            "parent_witness.parent_cells must provide parent_cells; got Nested refinement requires parent_cells."
        )
    parents = np.asarray(parent_cells)
    if not np.issubdtype(parents.dtype, np.integer):
        raise TypeError(
            "parent_witness.parent_cells must have an integer dtype; got parent_cells must have an integer dtype."
        )
    if parents.shape != (target_count,):
        raise ValueError(
            "parent_witness.parent_cells must provide one source cell per target cell; got parent_cells needs one source cell per target cell."
        )
    if np.any(parents < 0) or np.any(parents >= source_count):
        raise ValueError(
            "parent_witness.parent_cells must address source cells; got parent_cells must address source cells."
        )
    return parents.astype(np.int64)


def _reference_vertices(dimension: int, /) -> np.ndarray:
    return np.concatenate(
        (np.zeros((1, dimension)), np.eye(dimension, dtype=np.float64)), axis=0
    )


def _pullback(
    origins: np.ndarray, frames: np.ndarray, points: np.ndarray, /
) -> np.ndarray:
    """Reference coordinates ``(C, P, d)`` of physical points in affine cells."""

    return np.linalg.solve(frames[:, None], (points - origins[:, None, :])[..., None])[
        ..., 0
    ]


def _containment_defect(
    target_origins: np.ndarray,
    target_frames: np.ndarray,
    parent_origins: np.ndarray,
    parent_frames: np.ndarray,
    /,
) -> float:
    """Largest barycentric excursion of target corners outside their parents.

    Affine simplices are convex, so corner containment certifies containment of
    the whole target cell and algebraic restriction of the parent coordinate map.
    """

    corners = _reference_vertices(target_frames.shape[-1])
    physical = target_origins[:, None, :] + corners[None] @ np.swapaxes(
        target_frames, -1, -2
    )
    local = _pullback(parent_origins, parent_frames, physical)
    barycentric = np.concatenate(
        (1.0 - np.sum(local, axis=-1, keepdims=True), local), axis=-1
    )
    return float(max(0.0, np.max(-barycentric), np.max(barycentric - 1.0)))


def _frame_condition(frames: np.ndarray, /) -> float:
    # Host-only preparation of statically tiny d x d frames (d <= 3).
    singular = np.linalg.svd(frames, compute_uv=False)
    return float(np.max(singular[:, 0] / singular[:, -1]))


def _row_disagreement(
    columns: np.ndarray,
    values: np.ndarray,
    reference_columns: np.ndarray,
    reference_values: np.ndarray,
    column_count: int,
    /,
) -> float:
    """Largest entry of ``rows - reference_rows`` as sparse rows over ``column_count``.

    Row ``k`` of each operand is the pair ``(columns[k], values[k])``; two rows
    agree when their coefficients agree on every column, whatever local order or
     cell produced them.
    """

    rows = np.arange(columns.shape[0], dtype=np.int64)[:, None]
    keys = np.concatenate(
        (
            (rows * column_count + columns).reshape((-1,)),
            (rows * column_count + reference_columns).reshape((-1,)),
        )
    )
    signed = np.concatenate((values.reshape((-1,)), -reference_values.reshape((-1,))))
    _, inverse = np.unique(keys, return_inverse=True)
    difference = np.bincount(inverse.reshape((-1,)), weights=signed)
    scale = max(
        float(np.max(np.abs(reference_values), initial=0.0)),
        np.finfo(np.float64).tiny,
    )
    return float(np.max(np.abs(difference), initial=0.0)) / scale


def owned_rows(
    target_routes: np.ndarray,
    source_routes: np.ndarray,
    local: np.ndarray,
    target_size: int,
    source_size: int,
    /,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Global rows from the first cell of each DOF and their continuity defect.

    Every other cell sharing a DOF must produce the same row; the defect measures
    trace continuity and orientation consistency of the local transfers.
    """

    width = target_routes.shape[1]
    flat = target_routes.reshape((-1,))
    dofs, first = np.unique(flat, return_index=True)
    if not np.array_equal(dofs, np.arange(target_size)):
        raise ValueError(
            "owned_rows.configuration must cover every target DOF; got Target cells do not cover every target DOF."
        )
    owner_cells, owner_slots = np.divmod(first, width)
    rows = source_routes[owner_cells]
    coefficients = local[owner_cells, owner_slots]
    occurrence_columns = np.repeat(source_routes, width, axis=0)
    continuity = _row_disagreement(
        occurrence_columns,
        local.reshape((-1, local.shape[-1])),
        rows[flat],
        coefficients[flat],
        source_size,
    )
    return rows, coefficients, continuity


def _local_nested_transfer(
    element: FiniteElementSpec,
    target_frames: np.ndarray,
    target_orientation: np.ndarray,
    reference_points: np.ndarray,
    reference_weights: np.ndarray,
    parent_frames: np.ndarray,
    parent_orientation: np.ndarray,
    parent_points: np.ndarray,
    /,
    *,
    target_transformation: np.ndarray | None = None,
    parent_transformation: np.ndarray | None = None,
    maximum_work: int = 100_000_000,
    maximum_storage_bytes: int = 256_000_000,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, dict[str, float], float]:
    """algebraic local reconstruction ``M^{-1} B`` of parent basis functions per child.

    Returns the local maps ``(C, n_target, n_source)``, their reproduction and
    commuting defects, and the local mass condition that scales the certificate.
    """

    cells = target_frames.shape[0]

    preparation = imported_mapped_form_transfer.PreparationWork(
        maximum_work, maximum_storage_bytes, budget=budget
    )
    width = element.local_dof_count
    value_width = max(element.topological_dimension if element.value_shape else 1, 1)
    samples = reference_points.shape[0]
    preparation.charge(
        cells * (width**3 + 2 * samples * width**2 * value_width),
        cells
        * (
            3 * width**2
            + 2 * samples * width * value_width * (element.topological_dimension + 2)
        ),
    )
    target_basis, target_differential = _mapped_basis(
        element,
        target_frames,
        target_orientation,
        np.broadcast_to(reference_points, (cells,) + reference_points.shape),
        transformation=target_transformation,
    )
    source_basis, source_differential = _mapped_basis(
        element,
        parent_frames,
        parent_orientation,
        parent_points,
        transformation=parent_transformation,
    )
    weights = _affine_measure_densities(target_frames)[:, None] * reference_weights[None]
    mass = np.asarray(
        ein.contract("cq,cqiv,cqjv->cij", weights, target_basis, target_basis)
    )
    mixed = np.asarray(
        ein.contract("cq,cqiv,cqkv->cik", weights, target_basis, source_basis)
    )
    if width <= 4:
        solved = imported_linalg.solve_small_linear(
            imported_linalg.SmallLinearSolvePlan(width), mass, mixed
        )
        condition = float(np.max(np.asarray(solved.condition_estimate)))
    else:
        factor = factorize(DenseLinearOperator(mass), FactorizationPolicy("svd"))
        if np.any(np.asarray(factor.rank()) != width):
            raise ValueError(
                "local_nested_transfer.element must have full rank; got Nested finite-element local mass is rank deficient."
            )
        spectrum = np.asarray(factor.singular_values())
        condition = float(np.max(spectrum[..., 0] / spectrum[..., -1]))
        solved = factor.solve(
            jnp.asarray(mixed), rhs_layout=imported_linalg.RHSLayout((mixed.shape[-1],))
        )
    if not bool(np.all(np.asarray(solved.successful))):
        raise ValueError(
            "local_nested_transfer.element must pass the declared scientific certificate; got Nested finite-element native local mass solve failed."
        )
    local = np.asarray(solved.value, dtype=np.float64)
    if not np.all(np.isfinite(local)) or not np.isfinite(condition) or condition > 1e12:
        raise ValueError(
            "local_nested_transfer.element must meet the declared conditioning bound; got Nested finite-element local mass is ill conditioned."
        )
    reconstructed = np.asarray(ein.contract("cqiv,cik->cqkv", target_basis, local))
    scale = max(float(np.max(np.abs(source_basis))), np.finfo(np.float64).tiny)
    defects = {
        "reproduction": float(np.max(np.abs(reconstructed - source_basis))) / scale
    }
    if target_differential is not None and source_differential is not None:
        target_integrals = np.asarray(
            ein.contract("cq,cqiw->ciw", weights, target_differential)
        )
        source_integrals = np.asarray(
            ein.contract("cq,cqkw->ckw", weights, source_differential)
        )
        transferred = np.asarray(ein.contract("ciw,cik->ckw", target_integrals, local))
        reference = max(
            float(np.max(np.abs(source_integrals))),
            float(np.max(np.abs(target_integrals))) * float(np.max(np.abs(local))),
            np.finfo(np.float64).tiny,
        )
        defects["commuting"] = (
            float(np.max(np.abs(transferred - source_integrals))) / reference
        )
    return local, defects, condition


def _nested_semantics(element: FiniteElementSpec, passed: bool, /) -> TransferSemantics:
    match element.mapping:
        case "identity":
            return "nested-interpolation" if passed else "interpolation"
        case "density" | "exterior" if element.form_basis is not None:
            return "nested-interpolation" if passed else "interpolation"
        case "covariant_piola":
            return "covariant-piola"
        case "contravariant_piola":
            return "contravariant-piola"
        case mapping:
            raise ValueError(
                f"nested_semantics.element must select a supported declared route; got Unsupported finite-element mapping {mapping!r}."
            )


def _single_coordinate_element(
    discretization: _NestedFiniteElementDiscretization, role: str, /
) -> FiniteElementSpec:
    """The one simplex Lagrange coordinate element shared by every block."""

    elements = {
        element.element_id: element for element in discretization.coordinate_elements
    }
    kinds = {block.cell_kind for block in discretization.mesh.blocks}
    element = next(iter(elements.values()))
    if (
        len(elements) != 1
        or not kinds <= set(_SIMPLEX_KINDS.values())
        or not isinstance(element, FiniteElementSpec)
        or element.element_id
        != coordinate_lagrange_element(element.cell_kind, element.degree).element_id
    ):
        raise ValueError(
            "single_coordinate_element.coordinate_elements must share one simplex Lagrange element across all blocks; got "
            f"role={role!r}, element_ids={tuple(elements)!r}, cell_kinds={sorted(kinds)!r}, element_type={type(element).__name__}"
        )
    return element


def _map_restriction_defect(
    source: _NestedFiniteElementDiscretization,
    target: _NestedFiniteElementDiscretization,
    parents: np.ndarray,
    origins: np.ndarray,
    frames: np.ndarray,
    /,
) -> float:
    """Relative disagreement of the target map with the restricted parent map.

    Every target coordinate node is compared with the parent coordinate map at
    the node's parent reference point; zero certifies that the target geometry is
    the algebraic restriction the reference witnesses describe.
    """

    source_element = _single_coordinate_element(source, "source")
    target_element = _single_coordinate_element(target, "target")
    source_points = np.asarray(source.default_runtime.coordinates, dtype=np.float64)
    target_points = np.asarray(target.default_runtime.coordinates, dtype=np.float64)
    source_routes = np.concatenate(
        [np.asarray(routes) for routes in source.coordinate_dofs]
    )
    target_routes = np.concatenate(
        [np.asarray(routes) for routes in target.coordinate_dofs]
    )
    nodes = np.asarray(target_element.reference_nodes, dtype=np.float64)
    parent_points = origins[:, None, :] + nodes[None] @ np.swapaxes(frames, -1, -2)
    values = np.asarray(
        source_element.tabulate(parent_points.reshape((-1, nodes.shape[1])))[0],
        dtype=np.float64,
    ).reshape(parent_points.shape[:2] + (source_element.local_dof_count,))
    restricted = np.asarray(
        ein.contract("cnm,cmD->cnD", values, source_points[source_routes[parents]])
    )
    scale = max(float(np.max(np.abs(source_points))), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(restricted - target_points[target_routes]))) / scale


class _NestedAffineGeometry(StrictModule):
    parents: NDArray[np.int64]
    target_origins: NDArray[np.float64]
    target_frames: NDArray[np.float64]
    parent_origins: NDArray[np.float64]
    parent_frames: NDArray[np.float64]
    geometry_defects: dict[str, float]
    affine: bool


def _nested_affine_geometry(
    source: _NestedFiniteElementDiscretization,
    target: _NestedFiniteElementDiscretization,
    parent_cells: ArrayLike | None,
    parent_reference_vertices: ArrayLike | None,
    actual_source_geometry: CellGeometrySpec,
    actual_target_geometry: CellGeometrySpec,
    affine_coefficient_actions: bool,
    *,
    budget: CoordinateEnclosureBudget,
) -> _NestedAffineGeometry:
    geometry_defects: dict[str, float] = {}
    if parent_reference_vertices is None:
        _require_affine_simplices(source, "source", "Nested transfer")
        _require_affine_simplices(target, "target", "Nested transfer")
        affine = True
        source_origins, source_frames = _affine_frames(source)
        target_origins, target_frames = _affine_frames(target)
        parents = _parent_witness(
            parent_cells, target_frames.shape[0], source_frames.shape[0]
        )
        parent_origins, parent_frames = source_origins[parents], source_frames[parents]
    else:
        # Reference-relative child maps: the target reference simplex maps into the
        # parent reference cell, whose own frame is the identity.
        affine = all(
            element_.element_id
            == coordinate_lagrange_element(block.cell_kind, 1).element_id
            or isinstance(element_, BarycentricCellGeometryElement)
            for block, element_ in zip(
                source.mesh.blocks, source.coordinate_elements, strict=True
            )
        )
        corners = np.asarray(parent_reference_vertices, dtype=np.float64)
        dimension = target.mesh.topological_dimension
        count = sum(block.cell_count for block in target.mesh.blocks)
        if corners.shape != (count, dimension + 1, dimension) or not np.all(
            np.isfinite(corners)
        ):
            raise ValueError(
                "nested_affine_geometry.target must be finite (target cells, d+1, d); got parent_reference_vertices must be finite (target cells, d+1, d)."
            )
        parents = _parent_witness(
            parent_cells, count, sum(block.cell_count for block in source.mesh.blocks)
        )
        target_origins = corners[:, 0]
        target_frames = np.swapaxes(corners[:, 1:] - corners[:, :1], -1, -2)
        parent_origins = np.zeros_like(target_origins)
        parent_frames = np.broadcast_to(np.eye(dimension), target_frames.shape).copy()
        if affine_coefficient_actions:
            pairs = imported_nested_reference.nested_reference_pairs(
                source.mesh,
                actual_source_geometry,
                target.mesh,
                actual_target_geometry,
                parent_cells=parents,
                parent_reference_vertices=corners,
                budget=budget,
            )
            bound = mapped_measure_owner.certify_nested_geometry_pairs(
                source.mesh,
                actual_source_geometry,
                target.mesh,
                actual_target_geometry,
                pairs,
                budget=budget,
            )
            if bound != 0.0:
                raise ValueError(
                    "nested_affine_geometry.coefficient_action must preserve the source map algebraically under its reference witnesses; got a nonzero algebraic difference."
                )
            geometry_defects["geometry"] = bound
        else:
            geometry_defects["geometry"] = _map_restriction_defect(
                source, target, parents, target_origins, target_frames
            )
    if not affine_coefficient_actions and (
        actual_source_geometry.restriction_source is not None
        or actual_target_geometry.restriction_source is not None
    ):
        source_roots = mapped_root_identities(source.mesh, actual_source_geometry)
        target_roots = mapped_root_identities(target.mesh, actual_target_geometry)
        if any(
            source_roots[parent] != target_roots[row]
            for row, parent in enumerate(parents)
        ):
            raise ValueError(
                "nested_affine_geometry.parent_cells must preserve scientific root identities; got "
                f"source_roots={source_roots!r}, target_roots={target_roots!r}, parents={parents!r}"
            )
    return _NestedAffineGeometry(
        parents,
        target_origins,
        target_frames,
        parent_origins,
        parent_frames,
        geometry_defects,
        affine,
    )


def _prepare_affine_nested_field_transfer(
    source: _NestedFiniteElementDiscretization,
    target: _NestedFiniteElementDiscretization,
    parent_cells: ArrayLike | None,
    *,
    name: str,
    source_index: int,
    target_index: int,
    parent_reference_vertices: ArrayLike | None,
    actual_source_geometry: CellGeometrySpec,
    actual_target_geometry: CellGeometrySpec,
    affine_coefficient_actions: bool,
    maximum_work: int,
    maximum_storage_bytes: int,
    budget: CoordinateEnclosureBudget,
) -> FiniteElementFieldTransfer:
    element = _field_signature(source, source_index, target, target_index)
    prepared = _nested_affine_geometry(
        source,
        target,
        parent_cells,
        parent_reference_vertices,
        actual_source_geometry,
        actual_target_geometry,
        affine_coefficient_actions,
        budget=budget,
    )
    parents = prepared.parents
    target_origins, target_frames = prepared.target_origins, prepared.target_frames
    parent_origins, parent_frames = prepared.parent_origins, prepared.parent_frames
    geometry_defects, affine = prepared.geometry_defects, prepared.affine
    degree = _polynomial_degree(element)
    rule = cubature_rule_data(_simplex_cubature_kind(element.cell_kind), 2 * degree)
    reference_points = np.asarray(rule.points, dtype=np.float64)
    physical = target_origins[:, None, :] + reference_points[None] @ np.swapaxes(
        target_frames, -1, -2
    )
    target_routes, target_orientation = _cell_routes(target, target_index)
    source_routes, source_orientation = _cell_routes(source, source_index)
    source_transformation = _cell_transformations(source, source_index)
    local, defects, mass_condition = _local_nested_transfer(
        element,
        target_frames,
        target_orientation,
        reference_points,
        np.asarray(rule.weights, dtype=np.float64),
        parent_frames,
        source_orientation[parents],
        _pullback(parent_origins, parent_frames, physical),
        target_transformation=_cell_transformations(target, target_index),
        parent_transformation=None
        if source_transformation is None
        else source_transformation[parents],
        maximum_work=maximum_work,
        maximum_storage_bytes=maximum_storage_bytes,
        budget=budget,
    )
    source_dofs = source.dof_maps[source_index]
    target_dofs = target.dof_maps[target_index]
    rows, coefficients, continuity = owned_rows(
        target_routes,
        source_routes[parents],
        local,
        target_dofs.global_dof_count,
        source_dofs.global_dof_count,
    )
    containment = _containment_defect(
        target_origins, target_frames, parent_origins, parent_frames
    )
    lagrange = (
        element.mapping == "identity"
        and not element.value_shape
        and element.representation == "point_value"
    )
    source_measures, target_measures = None, None
    if lagrange and affine and affine_coefficient_actions:
        cell_measures: list[NDArray[np.float64]] = []
        for space, geometry in (
            (source, actual_source_geometry),
            (target, actual_target_geometry),
        ):
            measured_cells, errors, _ = mapped_measure_owner.certified_cell_measures(
                space.mesh, geometry, maximum_work=maximum_work, budget=budget
            )
            cell_measures.append(measured_cells)
            defects["measure_rounding"] = max(
                defects.get("measure_rounding", 0.0),
                float(np.sum(errors))
                / max(float(np.sum(measured_cells)), np.finfo(np.float64).tiny),
            )
        source_measures = _cell_integrals(
            source, source_index, mass=False, cell_measures=cell_measures[0]
        )[0]
        target_measures = _cell_integrals(
            target, target_index, mass=False, cell_measures=cell_measures[1]
        )[0]
        restricted_measures = np.zeros_like(source_measures)
        np.add.at(
            restricted_measures,
            rows.reshape(-1),
            (coefficients * target_measures[:, None]).reshape(-1),
        )
        defects["conservation"] = _relative(
            restricted_measures - source_measures, source_measures
        )
    amplification = max(
        mass_condition, _frame_condition(target_frames), _frame_condition(parent_frames)
    ) * float(local.shape[1] * local.shape[2])
    evidence = FiniteElementTransferEvidence(
        {
            "containment": containment,
            "continuity": continuity,
            **geometry_defects,
            **defects,
        },
        CLAIM_ULPS * float(np.finfo(np.float64).eps) * amplification,
    )
    passed = evidence.passed
    identity = {
        "source": source.prepared_id,
        "target": target.prepared_id,
        "field": name,
        "parents": array_tree_fingerprint(parents),
    }
    primal = SparseLinearMap(
        RowRelation(
            rows.astype(np.int32),
            source_size=source_dofs.global_dof_count,
            valid=np.ones(rows.shape, dtype=np.bool_),
        ),
        jnp.asarray(coefficients),
        operator_id=canonical_fingerprint(
            {"kind": "finite-element-nested-transfer", **identity}
        ),
    )
    claims = passed and lagrange
    measured = claims and affine
    if not measured:
        source_measures, target_measures = None, None
    elif not affine_coefficient_actions:
        source_measures = _cell_integrals(source, source_index, mass=False)[0]
        target_measures = _cell_integrals(target, target_index, mass=False)[0]
    linear = (
        measured
        and element.degree >= 1
        and not source_dofs.association.startswith("quotient_")
        and not target_dofs.association.startswith("quotient_")
    )
    transfer = FiniteElementTopologyTransfer(
        primal,
        source.mesh.topology_id,
        target.mesh.topology_id,
        preserves_constants=claims,
        preserves_linear=linear,
        conservative=measured,
        positivity_preserving=claims and bool(np.all(coefficients >= 0.0)),
        action_condition=max(amplification, 1.0),
        semantics=_nested_semantics(element, passed),
        source_coordinates=source_dofs.dof_coordinates if linear else None,
        target_coordinates=target_dofs.dof_coordinates if linear else None,
        source_measures=source_measures,
        target_measures=target_measures,
    )
    nesting = max(containment, geometry_defects.get("geometry", 0.0))
    geometry = TransferGeometryBinding(
        _coordinate_geometry_id(actual_source_geometry),
        _coordinate_geometry_id(actual_target_geometry),
        "exact-restriction"
        if nesting <= evidence.tolerance
        else "bounded-reconstruction",
        source_topology_id=source.mesh.topology_id,
        target_topology_id=target.mesh.topology_id,
        coverage_defect=0.0 if nesting <= evidence.tolerance else nesting,
    )
    return FiniteElementFieldTransfer(
        transfer,
        source.field_spaces[source_index],
        target.field_spaces[target_index],
        geometry,
        evidence,
        source_measures=source_measures,
        target_measures=target_measures,
    )


def _reference_coarsening_admissible(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    source_index: int,
    target_index: int,
) -> bool:
    return not all(
        element.form_basis is not None
        for element in source.elements[source_index] + target.elements[target_index]
    ) and all(
        block.cell_kind in _SIMPLEX_KINDS.values()
        and (
            coordinate.element_id
            == coordinate_lagrange_element(block.cell_kind, 1).element_id
            or isinstance(coordinate, BarycentricCellGeometryElement)
        )
        for space in (source, target)
        for block, coordinate in zip(
            space.mesh.blocks, space.coordinate_elements, strict=True
        )
    )


def _prepare_nested_field_transfer(
    source: _NestedFiniteElementDiscretization,
    target: _NestedFiniteElementDiscretization,
    parent_cells: ArrayLike | None,
    /,
    *,
    field_name: str,
    parent_reference_vertices: ArrayLike | None = None,
    source_geometry: CellGeometrySpec | None = None,
    target_geometry: CellGeometrySpec | None = None,
    geometry_transition: CellGeometryTransition | None = None,
    coarsening_witnesses: NestedReferenceWitnesses | None = None,
    maximum_work: int = 100_000_000,
    maximum_storage_bytes: int = 256_000_000,
    budget: CoordinateEnclosureBudget,
) -> FiniteElementFieldTransfer:
    """Dispatch certified affine, mapped, and retained-reference field transfers."""
    name = str(field_name)
    source_index = source.field_index(name)
    target_index = target.field_index(name)
    actual_source_geometry = bound_coordinate_spec(
        source, source_geometry, "source", budget=budget
    )
    actual_target_geometry = bound_coordinate_spec(
        target, target_geometry, "target", budget=budget
    )
    match source, target:
        case FiniteElementDiscretization(), FiniteElementDiscretization() if (
            coarsening_witnesses is not None
            and _reference_coarsening_admissible(
                source, target, source_index, target_index
            )
        ):
            return _prepare_reference_coarsening_projection(
                source,
                target,
                actual_source_geometry,
                actual_target_geometry,
                field_name=name,
                coarsening_witnesses=coarsening_witnesses,
                budget=budget,
            )
        case _:
            pass
    coefficient_action_geometry = any(
        isinstance(element, BarycentricCellGeometryElement)
        for space in (source, target)
        for element in space.coordinate_elements
    )
    affine_coefficient_actions = (
        coefficient_action_geometry
        and coarsening_witnesses is None
        and geometry_transition is None
        and parent_reference_vertices is not None
    )
    if affine_coefficient_actions:
        _require_coefficient_affine_simplices(
            source, actual_source_geometry, "source", budget=budget
        )
        _require_coefficient_affine_simplices(
            target, actual_target_geometry, "target", budget=budget
        )
    mapped = (
        any(
            block.cell_kind not in _SIMPLEX_KINDS.values()
            for space in (source, target)
            for block in space.mesh.blocks
        )
        or any(
            isinstance(element, RestrictedCellGeometryElement)
            or isinstance(element, BarycentricCellGeometryElement)
            and not affine_coefficient_actions
            for space in (source, target)
            for element in space.coordinate_elements
        )
        or coarsening_witnesses is not None
        or geometry_transition is not None
        or (
            parent_reference_vertices is not None
            and all(
                element.form_basis is not None
                for element in source.elements[source_index]
                + target.elements[target_index]
            )
        )
    )
    if mapped:
        source_elements = source.elements[source_index]
        target_elements = target.elements[target_index]
        kinds = {element.conformity for element in source_elements + target_elements}
        if all(
            element.form_basis is not None
            for element in source_elements + target_elements
        ):
            return _prepare_mapped_nested_compatible_transfer(
                source,
                target,
                actual_source_geometry,
                actual_target_geometry,
                field_name=name,
                parent_cells=parent_cells,
                parent_reference_vertices=parent_reference_vertices,
                geometry_transition=geometry_transition,
                coarsening_witnesses=coarsening_witnesses,
                maximum_work=maximum_work,
                maximum_storage_bytes=maximum_storage_bytes,
                budget=budget,
            )
        match sorted(kinds):
            case ["H1"]:
                prepare = _prepare_mapped_nested_h1_transfer
            case ["L2"]:
                prepare = prepare_mapped_nested_dg_transfer
            case ["Hcurl"] | ["Hdiv"]:
                prepare = _prepare_mapped_nested_compatible_transfer
            case _:
                raise ValueError(
                    "prepare_nested_field_transfer.configuration must preserve field-family conformity; got Mapped nested transfer cannot change field-family conformity."
                )
        return prepare(
            source,
            target,
            actual_source_geometry,
            actual_target_geometry,
            field_name=name,
            parent_cells=parent_cells,
            parent_reference_vertices=parent_reference_vertices,
            geometry_transition=geometry_transition,
            budget=budget,
        )
    return _prepare_affine_nested_field_transfer(
        source,
        target,
        parent_cells,
        name=name,
        source_index=source_index,
        target_index=target_index,
        parent_reference_vertices=parent_reference_vertices,
        actual_source_geometry=actual_source_geometry,
        actual_target_geometry=actual_target_geometry,
        affine_coefficient_actions=affine_coefficient_actions,
        maximum_work=maximum_work,
        maximum_storage_bytes=maximum_storage_bytes,
        budget=budget,
    )


def _prepare_projection_field_transfer(
    source: FiniteElementDiscretization,
    target: PreparedL2ProjectionTarget,
    refinement: imported_geometry.PreparedCommonRefinement,
    /,
    *,
    field_name: str,
    budget: CoordinateEnclosureBudget,
) -> FiniteElementFieldTransfer:
    """Prepare a non-nested scalar or compatible vector projection.

    Scalar fields use ordinary Galerkin L2 projection. Piola fields require one
    identical family/degree on both affine meshes and full common-refinement
    coverage. They minimize L2 field error subject to conserved vector content
    and a commuting differential: cellwise polynomial L2 divergence in H(div),
    cellwise L2 curl in 2D, and L2 projection into the target discrete curl image
    in 3D H(curl). An independently projected discontinuous cellwise 3D curl may
    not be realizable; its defect is reported, never claimed as commuting.
    Native linalg rank-revealing factors and constraint lifts are target-owned
    and reused deterministically for every source/overlap and payload. Evidence
    checks every source basis column; failed realizability or commuting/content
    certificates refuse the route. Curved nonnested geometry is not admitted.
    """

    transfer, overlap = _assembled_projection(source, target, refinement, field_name)
    name = str(field_name)
    source_index = source.field_index(name)
    space = target.discretization
    coverage = _coverage_bound(refinement)
    if source.elements[source_index][0].mapping == "identity":
        target_index = space.field_index(target.field_name)
        # Scalar probes are constant-first; only probes in both spaces must be
        # reproduced, and affine probes are not periodic functions.
        periodic = (
            source.mesh.periodic_topology is not None
            or space.mesh.periodic_topology is not None
        )
        source_probes = _probe_coefficients(source, source_index)
        target_probes = _probe_coefficients(space, target_index)
        common = 1 if periodic else min(source_probes.shape[1], target_probes.shape[1])
        projected = np.asarray(
            _block_action(transfer.primal.mv_block, source_probes[:, :common]),
            dtype=np.float64,
        )
        certified = {
            "coverage": coverage,
            "reproduction": _relative(
                projected - target_probes[:, :common], target_probes[:, :common]
            ),
        }
        estimates = {"polynomial-probes": float(common)} if periodic else {}
        tolerance = (
            CLAIM_ULPS * float(np.finfo(np.float64).eps) * transfer.action_condition
        )
        evidence = FiniteElementTransferEvidence(
            certified, max(tolerance, coverage), estimates=estimates
        )
    else:
        transfer = _constrained_projection(
            source, source_index, target, refinement, overlap
        )
        evidence = _compatible_projection_evidence(
            source, source_index, target, refinement, transfer, overlap
        )
        if not evidence.passed:
            raise ValueError(
                "prepare_l2_projection_transfer.evidence must satisfy commuting and content constraints within tolerance; got "
                f"defects={evidence.defects!r}, tolerance={evidence.tolerance!r}"
            )
    return FiniteElementFieldTransfer(
        transfer,
        source.field_spaces[source_index],
        space.field_spaces[space.field_index(target.field_name)],
        TransferGeometryBinding(
            _coordinate_geometry_id(
                bound_coordinate_spec(source, None, "source", budget=budget)
            ),
            _coordinate_geometry_id(
                bound_coordinate_spec(space, None, "target", budget=budget)
            ),
            "bounded-reconstruction",
            source_topology_id=source.mesh.topology_id,
            target_topology_id=space.mesh.topology_id,
            coverage_defect=coverage,
        ),
        evidence,
        source_measures=_cell_integrals(source, source_index, mass=False)[0]
        if transfer.conservative
        else None,
        target_measures=target.dof_measures if transfer.conservative else None,
    )


__all__ = [
    "FiniteElementFieldTransfer",
    "FiniteElementL2Projection",
    "FiniteElementTopologyTransfer",
    "FiniteElementTransferEvidence",
    "PreparedL2ProjectionTarget",
    "prepare_l2_projection_target",
    "prepare_l2_projection_transfer",
    "prepare_nested_field_transfer",
    "SourceRealizationFieldSemantics",
    "prepare_source_realization_field_transfer",
    "prepare_projection_field_transfer",
    "refresh_l2_projection_target",
    "vertex_interpolation_transfer",
]


def _mapped_dg_source_basis(
    element: FiniteElementSpec, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...]:
    """Recover the represented declared scalar DG source, never a width alias."""

    if (
        element.conformity != "L2"
        or element.mapping != "identity"
        or element.value_shape
        or element.representation != "point_value"
    ):
        raise ValueError(
            "mapped_dg_source_basis.configuration must provide scalar point-value cell functionals; got Mapped nested DG transfer requires scalar point-value cell functionals."
        )
    basis = imported_coordinate_enclosure.source_basis(element, budget=budget)
    if basis is None:
        raise ValueError(
            "mapped_dg_source_basis.configuration must select a supported declared route; got Mapped DG field source has no algebraic declared expression."
        )
    return basis


def mapped_dg_cells(
    discretization: _NestedFiniteElementDiscretization,
    field_index: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[MappedDgCell, ...]:
    dofs = discretization.dof_maps[field_index]
    result: list[MappedDgCell] = []
    for element, routes, signs in zip(
        discretization.elements[field_index],
        dofs.cell_dofs,
        dofs.orientations,
        strict=True,
    ):
        basis = _mapped_dg_source_basis(element, budget=budget)
        for route, sign in zip(np.asarray(routes), np.asarray(signs), strict=True):
            if np.any(sign != 1):
                raise ValueError(
                    "mapped_dg_cells.configuration must use declared identity orientation; got Scalar DG routes must use declared identity orientation."
                )
            result.append((element, np.asarray(route, dtype=np.int64), basis))
    all_routes = np.concatenate([route for _, route, _ in result])
    if (
        all_routes.size != dofs.global_dof_count
        or np.unique(all_routes).size != all_routes.size
    ):
        raise ValueError(
            "mapped_dg_cells.configuration must provide unshared full cell-local DOF routes; got Mapped DG transfer requires unshared full cell-local DOF routes."
        )
    return tuple(result)


class _EmbeddedDgDensity(NamedTuple):
    element: ScalarCoordinateElement
    coefficients: CoordinateSourceBank
    preparation: PreparedMappedSqrtIntegral


def _mapped_dg_integral(
    weight: Expression,
    density: Expression | _EmbeddedDgDensity,
    kind: str,
    work: list[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]:

    if isinstance(density, _EmbeddedDgDensity):
        if isinstance(weight, imported_coordinate_enclosure.RationalPolynomial):
            raise ValueError(
                "mapped_dg_integral.configuration must provide declared polynomial reference weights; got Prepared embedded DG functionals require declared polynomial reference weights."
            )
        return density.preparation.integral(weight, budget=budget)
    return mapped_measure_owner.integrate_mapped_polynomial(
        imported_coordinate_enclosure.expression_multiply(density, weight, budget=budget),
        kind,
        budget=budget,
    )


def _mapped_dg_forms(
    first: Sequence[Expression],
    second: Sequence[Expression],
    density: Expression | _EmbeddedDgDensity,
    kind: str,
    work: list[int] | None = None,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    integration_work = [0, 100_000_000] if work is None else work
    values = np.empty((len(first), len(second)), dtype=np.float64)
    errors = np.empty_like(values)
    # Products and their polynomial memos are consumed by this local form.
    # Only numerical values and outward errors survive into the transfer.
    with budget.polynomial_preparation_scope():
        for i, a in enumerate(first):
            for j, b in enumerate(second):
                values[i, j], errors[i, j] = _mapped_dg_integral(
                    imported_coordinate_enclosure.expression_multiply(
                        a, b, budget=budget
                    ),
                    density,
                    kind,
                    integration_work,
                    budget=budget,
                )
    return values, errors


def _mapped_dg_constant_bound(
    cells: Sequence[MappedDgCell],
    weights: Sequence[Fraction],
    *,
    budget: CoordinateEnclosureBudget,
) -> float:

    bound = 0.0
    for element, route, basis in cells:
        polynomial = imported_coordinate_enclosure.add(
            imported_coordinate_enclosure.sum_polynomials(
                tuple(
                    imported_coordinate_enclosure.scale(
                        term, weights[int(dof)], budget=budget
                    )
                    for dof, term in zip(route, basis, strict=True)
                ),
                budget=budget,
            ),
            imported_coordinate_enclosure.constant(
                -1, element.topological_dimension, budget=budget
            ),
            budget=budget,
        )
        if polynomial:
            domain = (
                "simplex"
                if element.cell_kind in ("interval", "triangle", "tetrahedron")
                else "prism"
                if element.cell_kind == "prism"
                else "box"
            )
            lower, upper = imported_coordinate_enclosure.polynomial_bounds(
                polynomial, domain, element.topological_dimension, budget=budget
            )
            bound = max(bound, abs(lower), abs(upper))
    return bound


def _mapped_dg_embedded_density(
    element: ScalarCoordinateElement,
    coefficients: CoordinateSourceBank,
    goal: Fraction,
    work: list[int],
    cache: dict[tuple[str, CoordinateSourceBank], _EmbeddedDgDensity],
    *,
    budget: CoordinateEnclosureBudget,
) -> _EmbeddedDgDensity:

    key = element.element_id, coefficients
    if key not in cache:
        preparation = mapped_measure_owner.certified_sqrt_prepare_mapped_integral(
            element, coefficients, goal, work, budget=budget
        )
        cache[key] = _EmbeddedDgDensity(element, coefficients, preparation)
    return cache[key]


class _MappedDgInput(StrictModule):
    name: str
    source_index: int
    target_index: int
    source_cells: tuple[MappedDgCell, ...]
    target_cells: tuple[MappedDgCell, ...]
    pairs: tuple[NestedReferencePair | PolynomialReferencePair, ...]
    geometry_bound: float
    source_geometry_cells: tuple[
        tuple[ScalarCoordinateElement, CoordinateSourceBank], ...
    ]
    target_geometry_cells: tuple[
        tuple[ScalarCoordinateElement, CoordinateSourceBank], ...
    ]


class _MappedDgForms(StrictModule):
    source_measures: NDArray[np.float64]
    target_measures: NDArray[np.float64]
    source_errors: NDArray[np.float64]
    target_errors: NDArray[np.float64]
    masses: list[NDArray[np.float64]]
    mixed: list[dict[int, NDArray[np.float64]]]
    groups: dict[tuple[str, str, str], list[int]]
    integration_error: float
    integration_work: list[int]
    density_cache: dict[tuple[str, CoordinateSourceBank], _EmbeddedDgDensity]
    uniform_goal: Fraction


class _MappedDgSolution(StrictModule):
    rows: NDArray[np.int64]
    columns: NDArray[np.int64]
    coefficients: NDArray[np.float64]
    condition: float
    solve_defect: float
    minimum_rank: float


class _MappedDgCertificate(StrictModule):
    evidence: FiniteElementTransferEvidence
    amplification: float
    geometry_scale: float


def _mapped_dg_input(
    source: _NestedFiniteElementDiscretization,
    target: _NestedFiniteElementDiscretization,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    *,
    field_name: str,
    parent_cells: ArrayLike | None,
    parent_reference_vertices: ArrayLike | None,
    geometry_transition: CellGeometryTransition | None,
    budget: CoordinateEnclosureBudget,
) -> _MappedDgInput:
    name = canonical_identifier(field_name, "field_name")
    source_index, target_index = source.field_index(name), target.field_index(name)
    source_cells, target_cells = (
        mapped_dg_cells(source, source_index, budget=budget),
        mapped_dg_cells(target, target_index, budget=budget),
    )
    source_space = source.field_spaces[source_index].vector_space
    target_space = target.field_spaces[target_index].vector_space
    if not isinstance(source_space, ArraySpace) or not isinstance(
        target_space, ArraySpace
    ):
        raise TypeError(
            "mapped_dg_input.configuration must provide array coefficient spaces; got Mapped DG transfer requires array coefficient spaces."
        )
    if source_space.shape[1:] != target_space.shape[1:]:
        raise ValueError(
            "mapped_dg_input.source must agree with the declared geometry, topology, and field layout; got Mapped DG source and target component payloads differ."
        )
    pairs, geometry_bound = rooted_nested_reference_pairs(
        source.mesh,
        source_geometry,
        target.mesh,
        target_geometry,
        parent_cells=parent_cells,
        parent_reference_vertices=parent_reference_vertices,
        geometry_transition=geometry_transition,
        budget=budget,
    )
    source_geometry_cells = mapped_measure_owner.mapped_geometry_cells(
        source.mesh, source_geometry, budget=budget
    )
    target_geometry_cells = mapped_measure_owner.mapped_geometry_cells(
        target.mesh, target_geometry, budget=budget
    )
    return _MappedDgInput(
        name,
        source_index,
        target_index,
        source_cells,
        target_cells,
        pairs,
        geometry_bound,
        source_geometry_cells,
        target_geometry_cells,
    )


def _mapped_dg_integration_goal(
    source_cells: tuple[MappedDgCell, ...],
    target_cells: tuple[MappedDgCell, ...],
    pairs: tuple[NestedReferencePair | PolynomialReferencePair, ...],
    embedded: bool,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    uniform_goal = Fraction(0)
    if embedded:
        functional_count = (
            sum(len(basis) for _, _, basis in source_cells)
            + sum(len(basis) + len(basis) ** 2 for _, _, basis in target_cells)
            + sum(
                len(source_cells[pair.source_cell][2])
                * len(target_cells[pair.target_cell][2])
                for pair in pairs
            )
        )
        amplification_lower = max(len(basis) ** 2 for _, _, basis in target_cells)
        basis_sup = max(
            (
                abs(
                    rational_coefficient(
                        value, role="mapped_dg.integration_goal.reference_basis"
                    )
                )
                for element, _, basis in (*source_cells, *target_cells)
                for polynomial in basis
                for value in imported_coordinate_enclosure.bernstein_coefficients(
                    polynomial,
                    "simplex" if element.cell_kind == "triangle" else "box",
                    2,
                    budget=budget,
                )
            ),
            default=Fraction(1),
        )
        # The existing native mass-condition amplification is at least k**2.
        # Publication errors remain explicit in the source all-column proof.
        uniform_goal = (
            Fraction(CLAIM_ULPS)
            * Fraction(float(np.finfo(np.float64).eps))
            * amplification_lower
            / (16 * functional_count * max(basis_sup, Fraction(1)) ** 2)
        )
    return uniform_goal


def _mapped_dg_cell_forms(
    source: _NestedFiniteElementDiscretization,
    target: _NestedFiniteElementDiscretization,
    prepared: _MappedDgInput,
    *,
    budget: CoordinateEnclosureBudget,
) -> _MappedDgForms:
    source_index, target_index = prepared.source_index, prepared.target_index
    source_cells, target_cells = prepared.source_cells, prepared.target_cells
    pairs = prepared.pairs
    source_geometry_cells = prepared.source_geometry_cells
    target_geometry_cells = prepared.target_geometry_cells
    source_size = source.dof_maps[source_index].global_dof_count
    target_size = target.dof_maps[target_index].global_dof_count
    source_measures, target_measures = np.zeros(source_size), np.zeros(target_size)
    source_errors, target_errors = np.zeros(source_size), np.zeros(target_size)
    masses: list[NDArray[np.float64]] = []
    mixed: list[dict[int, NDArray[np.float64]]] = [dict() for _ in target_cells]
    groups: dict[tuple[str, str, str], list[int]] = {}
    integration_error = 0.0
    integration_work = [0, 100_000_000]
    embedded = (
        source.mesh.topological_dimension == 2 and source.mesh.ambient_dimension == 3
    )
    density_cache: dict[tuple[str, CoordinateSourceBank], _EmbeddedDgDensity] = {}
    uniform_goal = _mapped_dg_integration_goal(
        source_cells, target_cells, pairs, embedded, budget=budget
    )
    for cells, geometries, measures, errors, is_target in (
        (source_cells, source_geometry_cells, source_measures, source_errors, False),
        (target_cells, target_geometry_cells, target_measures, target_errors, True),
    ):
        for cell, ((element, route, basis), (geometry_element, local)) in enumerate(
            zip(cells, geometries, strict=True)
        ):
            density = (
                _mapped_dg_embedded_density(
                    geometry_element,
                    local,
                    uniform_goal,
                    integration_work,
                    density_cache,
                    budget=budget,
                )
                if embedded
                else mapped_measure_owner.mapped_density_expression(
                    geometry_element, local, budget=budget
                )
            )
            chart_basis = (
                tuple(
                    imported_coordinate_enclosure.compose(
                        term,
                        imported_coordinate_enclosure.chart_arguments(
                            element.cell_kind,
                            element.topological_dimension,
                            budget=budget,
                        ),
                        budget=budget,
                    )
                    for term in basis
                )
                if element.cell_kind != "pyramid"
                else basis
            )
            for dof, term in zip(route, chart_basis, strict=True):
                measures[dof], errors[dof] = _mapped_dg_integral(
                    term, density, element.cell_kind, integration_work, budget=budget
                )
            if is_target:
                mass, error = _mapped_dg_forms(
                    chart_basis,
                    chart_basis,
                    density,
                    element.cell_kind,
                    integration_work,
                    budget=budget,
                )
                masses.append(mass)
                integration_error += float(np.sum(error))
                root_element = geometry_element
                while isinstance(
                    root_element, imported_cell_geometry.RestrictedCellGeometryElement
                ):
                    root_element = root_element.source_element
                key = (
                    element.element_id,
                    geometry_element.cell_kind,
                    root_element.element_id,
                )
                groups.setdefault(key, []).append(cell)
    return _MappedDgForms(
        source_measures,
        target_measures,
        source_errors,
        target_errors,
        masses,
        mixed,
        groups,
        integration_error,
        integration_work,
        density_cache,
        uniform_goal,
    )


def _mapped_dg_patch_forms(
    source: _NestedFiniteElementDiscretization,
    prepared: _MappedDgInput,
    forms: _MappedDgForms,
    *,
    budget: CoordinateEnclosureBudget,
) -> _MappedDgForms:
    source_cells, target_cells = prepared.source_cells, prepared.target_cells
    pairs = prepared.pairs
    source_geometry_cells = prepared.source_geometry_cells
    target_geometry_cells = prepared.target_geometry_cells
    source_measures, target_measures = forms.source_measures, forms.target_measures
    source_errors, target_errors = forms.source_errors, forms.target_errors
    masses, mixed, groups = forms.masses, forms.mixed, forms.groups
    integration_error, integration_work = forms.integration_error, forms.integration_work
    density_cache, uniform_goal = forms.density_cache, forms.uniform_goal
    embedded = (
        source.mesh.topological_dimension == 2 and source.mesh.ambient_dimension == 3
    )
    for pair in pairs:
        se, sr, sb = source_cells[pair.source_cell]
        te, tr, tb = target_cells[pair.target_cell]
        fine_element, fine_local = (
            source_geometry_cells[pair.source_cell]
            if pair.fine_is_source
            else target_geometry_cells[pair.target_cell]
        )
        fine_kind = fine_element.cell_kind
        match pair:
            case PolynomialReferencePair():
                if pair.fine_is_source:
                    first = tuple(
                        imported_coordinate_enclosure.compose(
                            value, pair.arguments, budget=budget
                        )
                        for value in tb
                    )
                    second = tuple(sb)
                else:
                    first = tuple(tb)
                    second = tuple(
                        imported_coordinate_enclosure.compose(
                            value, pair.arguments, budget=budget
                        )
                        for value in sb
                    )
            case NestedReferencePair() if pair.fine_is_source:
                first = mapped_measure_owner.mapped_expression_restriction(
                    tb, te.cell_kind, fine_kind, pair.matrix, pair.offset, budget=budget
                )
                second = mapped_measure_owner.mapped_expression_restriction(
                    sb,
                    se.cell_kind,
                    fine_kind,
                    np.eye(se.topological_dimension),
                    np.zeros(se.topological_dimension),
                    budget=budget,
                )
            case NestedReferencePair():
                first = mapped_measure_owner.mapped_expression_restriction(
                    tb,
                    te.cell_kind,
                    fine_kind,
                    np.eye(te.topological_dimension),
                    np.zeros(te.topological_dimension),
                    budget=budget,
                )
                second = mapped_measure_owner.mapped_expression_restriction(
                    sb, se.cell_kind, fine_kind, pair.matrix, pair.offset, budget=budget
                )
            case _:
                assert_never(pair)
        density = (
            _mapped_dg_embedded_density(
                fine_element,
                fine_local,
                uniform_goal,
                integration_work,
                density_cache,
                budget=budget,
            )
            if embedded
            else mapped_measure_owner.mapped_density_expression(
                fine_element, fine_local, budget=budget
            )
        )
        form, error = _mapped_dg_forms(
            first, second, density, fine_kind, integration_work, budget=budget
        )
        mixed[pair.target_cell][pair.source_cell] = form
        integration_error += float(np.sum(error))
    return _MappedDgForms(
        source_measures,
        target_measures,
        source_errors,
        target_errors,
        masses,
        mixed,
        groups,
        integration_error,
        integration_work,
        density_cache,
        uniform_goal,
    )


def _solve_mapped_dg_forms(
    prepared: _MappedDgInput,
    forms: _MappedDgForms,
) -> _MappedDgSolution:
    source_cells, target_cells = prepared.source_cells, prepared.target_cells
    masses, mixed, groups = forms.masses, forms.mixed, forms.groups
    rows, columns, coefficients = [], [], []
    condition, solve_defect, minimum_rank = 1.0, 0.0, float("inf")
    for key, cells in groups.items():
        mass = np.stack([masses[cell] for cell in cells])
        factor = factorize(
            DenseLinearOperator(
                mass,
                operator_id=canonical_fingerprint(
                    {
                        "kind": "mapped-dg-local-mass",
                        "family": list(key),
                        "values": array_tree_fingerprint(mass),
                    }
                ),
            ),
            FactorizationPolicy("svd"),
        )
        ranks = np.asarray(factor.rank())
        minimum_rank = min(minimum_rank, float(np.min(ranks)))
        if np.any(ranks != mass.shape[-1]):
            raise ValueError(
                "solve_mapped_dg_forms.configuration must have full rank; got Mapped DG physical mass is rank deficient."
            )
        spectrum = np.asarray(factor.singular_values())
        condition = max(condition, float(np.max(spectrum[..., 0] / spectrum[..., -1])))
        # The native factor is batched by cell family; RHS widths may vary by patch.
        width = max(sum(form.shape[1] for form in mixed[cell].values()) for cell in cells)
        rhs = np.zeros((len(cells), mass.shape[-1], width))
        ordered = []
        for batch, cell in enumerate(cells):
            parents = sorted(mixed[cell])
            right = np.concatenate([mixed[cell][parent] for parent in parents], axis=1)
            rhs[batch, :, : right.shape[1]] = right
            ordered.append((parents, right.shape[1]))
        solved = factor.solve(jnp.asarray(rhs))
        values = np.asarray(solved.value)
        if not bool(np.all(np.asarray(solved.successful))) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "solve_mapped_dg_forms.configuration must pass the declared scientific certificate; got Mapped DG native local mass solve failed."
            )
        solve_defect = max(solve_defect, float(np.max(np.abs(mass @ values - rhs))))
        for batch, cell in enumerate(cells):
            parents, count = ordered[batch]
            source_routes = np.concatenate(
                [source_cells[parent][1] for parent in parents]
            )
            target_routes = target_cells[cell][1]
            rows.append(np.repeat(target_routes, count))
            columns.append(np.tile(source_routes, target_routes.size))
            coefficients.append(values[batch, :, :count].reshape(-1))
    rows, columns, coefficients = (
        np.concatenate(rows),
        np.concatenate(columns),
        np.concatenate(coefficients),
    )
    return _MappedDgSolution(
        rows, columns, coefficients, condition, solve_defect, minimum_rank
    )


def _certify_mapped_dg_transfer(
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    prepared: _MappedDgInput,
    forms: _MappedDgForms,
    solved: _MappedDgSolution,
    primal: SparseLinearMap,
    *,
    budget: CoordinateEnclosureBudget,
) -> _MappedDgCertificate:
    source_cells, target_cells = prepared.source_cells, prepared.target_cells
    geometry_bound = prepared.geometry_bound
    source_measures, target_measures = forms.source_measures, forms.target_measures
    source_errors, target_errors = forms.source_errors, forms.target_errors
    masses = forms.masses
    integration_error = forms.integration_error
    rows, columns, coefficients = solved.rows, solved.columns, solved.coefficients
    condition, solve_defect, minimum_rank = (
        solved.condition,
        solved.solve_defect,
        solved.minimum_rank,
    )
    source_size, target_size = source_measures.size, target_measures.size
    amplification = max(condition * max(mass.shape[-1] ** 2 for mass in masses), 1.0)
    constant = np.asarray(primal.mv(jnp.ones(source_size)))
    exact_row_sums = [Fraction(0) for _ in range(target_size)]
    for row, value in zip(rows, coefficients, strict=True):
        exact_row_sums[int(row)] += Fraction(float(value))
    constant_bound = max(
        _mapped_dg_constant_bound(
            source_cells, [Fraction(1)] * source_size, budget=budget
        ),
        _mapped_dg_constant_bound(target_cells, exact_row_sums, budget=budget),
    )

    exact_content = [-Fraction(float(value)) for value in source_measures]
    for row, column, value in zip(rows, columns, coefficients, strict=True):
        exact_content[int(column)] += Fraction(float(value)) * Fraction(
            float(target_measures[int(row)])
        )
    coefficient_content_bound = max(
        fraction_endpoint(abs(value), upper=True, saturate=True)
        for value in exact_content
    )
    scale = max(
        float(np.sum(np.abs(source_measures))),
        float(np.sum(np.abs(target_measures))),
        1.0,
    )
    # Absolute coefficientwise content uncertainty uses |P|, not a signed transpose.
    propagated = np.zeros(source_size)
    np.add.at(propagated, columns, np.abs(coefficients) * target_errors[rows])
    summation_roundoff = np.finfo(np.float64).eps * (coefficients.size + 2)
    publication_bound = float(
        np.nextafter(
            np.max(source_errors + propagated, initial=0) / (1 - summation_roundoff),
            np.inf,
        )
    )
    integration_roundoff = np.finfo(np.float64).eps * (
        sum(matrix.size for matrix in masses) + coefficients.size + 2
    )
    integration_error = float(
        np.nextafter(integration_error / (1 - integration_roundoff), np.inf)
    )
    tolerance = CLAIM_ULPS * np.finfo(np.float64).eps * amplification
    geometry_scale = max(
        float(np.max(np.abs(np.asarray(source_geometry.coordinates)))),
        float(np.max(np.abs(np.asarray(target_geometry.coordinates)))),
        1.0,
    )
    evidence = FiniteElementTransferEvidence(
        {
            "constants": float(np.max(np.abs(constant - 1))),
            "content": coefficient_content_bound / scale,
            "constants_continuum": constant_bound,
            "geometry": geometry_bound / geometry_scale,
            "mass_solve": solve_defect / scale,
            "integration": (integration_error + publication_bound) / scale,
        },
        tolerance,
        bounds={
            "integration": integration_error,
            "constants": constant_bound,
            "geometry": geometry_bound,
            "content": float(
                np.nextafter(
                    coefficient_content_bound + publication_bound + tolerance * scale,
                    np.inf,
                )
            ),
        },
        estimates={"minimum_local_rank": minimum_rank, "mass_condition": condition},
    )
    if not evidence.passed:
        raise ValueError(
            "certify_mapped_dg_transfer.configuration must pass the declared scientific certificate; got Mapped nested DG transfer failed its all-column physical content certificate."
        )
    return _MappedDgCertificate(evidence, amplification, geometry_scale)


def _publish_mapped_dg_transfer(
    source: _NestedFiniteElementDiscretization,
    target: _NestedFiniteElementDiscretization,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    prepared: _MappedDgInput,
    forms: _MappedDgForms,
    solved: _MappedDgSolution,
    *,
    budget: CoordinateEnclosureBudget,
) -> FiniteElementFieldTransfer:
    name = prepared.name
    source_index, target_index = prepared.source_index, prepared.target_index
    source_measures, target_measures = forms.source_measures, forms.target_measures
    source_size, target_size = source_measures.size, target_measures.size
    rows, columns, coefficients = solved.rows, solved.columns, solved.coefficients
    primal = coalesced_map(
        rows,
        columns,
        coefficients,
        target_size=target_size,
        source_size=source_size,
        properties=OperatorProperties(),
        operator_id=canonical_fingerprint(
            {
                "kind": "mapped-nested-dg",
                "source": source.prepared_id,
                "target": target.prepared_id,
                "field": name,
                "source_geometry": imported_cell_geometry_validity.cell_geometry_id(
                    source_geometry
                ),
                "target_geometry": imported_cell_geometry_validity.cell_geometry_id(
                    target_geometry
                ),
                "coefficients": array_tree_fingerprint(coefficients),
            }
        ),
    )
    certificate = _certify_mapped_dg_transfer(
        source_geometry, target_geometry, prepared, forms, solved, primal, budget=budget
    )
    evidence, amplification = certificate.evidence, certificate.amplification
    geometry_scale, geometry_bound = certificate.geometry_scale, prepared.geometry_bound
    transfer = FiniteElementTopologyTransfer(
        primal,
        source.mesh.topology_id,
        target.mesh.topology_id,
        preserves_constants=True,
        conservative=True,
        positivity_preserving=bool(np.all(coefficients >= 0)),
        action_condition=amplification,
        semantics="l2-projection",
        source_measures=source_measures,
        target_measures=target_measures,
    )
    binding = TransferGeometryBinding(
        imported_cell_geometry_validity.cell_geometry_id(source_geometry),
        imported_cell_geometry_validity.cell_geometry_id(target_geometry),
        "exact-restriction" if geometry_bound == 0 else "bounded-reconstruction",
        source_topology_id=source.mesh.topology_id,
        target_topology_id=target.mesh.topology_id,
        coverage_defect=geometry_bound / geometry_scale,
    )
    return FiniteElementFieldTransfer(
        transfer,
        source.field_spaces[source_index],
        target.field_spaces[target_index],
        binding,
        evidence,
        source_measures=source_measures,
        target_measures=target_measures,
    )


@checked
def prepare_mapped_nested_dg_transfer(
    source: _NestedFiniteElementDiscretization,
    target: _NestedFiniteElementDiscretization,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    /,
    *,
    field_name: str,
    parent_cells: ArrayLike | None = None,
    parent_reference_vertices: ArrayLike | None = None,
    geometry_transition: CellGeometryTransition | None = None,
    budget: CoordinateEnclosureBudget,
) -> FiniteElementFieldTransfer:
    """Prepare physical cell-local L2 DG projection over certified nested patches.

    Integration uses algebraic declared expressions (including collapsed pyramid
    charts), with explicit outward publication errors. Local masses are batched
    by represented field and geometry descriptors and solved by native SVD factors.
    The sparse primal and its algebraic transpose preserve all payload axes.
    """
    with budget.measurement(), budget.temporary_scope():
        try:
            prepared = _mapped_dg_input(
                source,
                target,
                source_geometry,
                target_geometry,
                field_name=field_name,
                parent_cells=parent_cells,
                parent_reference_vertices=parent_reference_vertices,
                geometry_transition=geometry_transition,
                budget=budget,
            )
            forms = _mapped_dg_cell_forms(source, target, prepared, budget=budget)
            forms = _mapped_dg_patch_forms(source, prepared, forms, budget=budget)
            solved = _solve_mapped_dg_forms(prepared, forms)
            return _publish_mapped_dg_transfer(
                source,
                target,
                source_geometry,
                target_geometry,
                prepared,
                forms,
                solved,
                budget=budget,
            )
        except CoordinateEnclosureResourceError as error:
            if (
                source.mesh.topological_dimension == 2
                and source.mesh.ambient_dimension == 3
            ):
                raise CellGeometryTransitionError(
                    "resource_limit",
                    "prepare_mapped_nested_dg_transfer.budget must fit the shared coefficient work and storage budget; got Mapped DG algebraic coefficient preparation exhausted its shared work/storage budget.",
                    measured=float(error.requested),
                    limit=float(error.limit),
                ) from error
            raise


@checked
def bound_coordinate_spec(
    discretization: _NestedFiniteElementDiscretization,
    geometry: imported_cell_geometry.CellGeometrySpec | None,
    role: str,
    *,
    budget: CoordinateEnclosureBudget,
) -> imported_cell_geometry.CellGeometrySpec:

    if geometry is None:
        if any(
            isinstance(
                element,
                (
                    imported_cell_geometry.BarycentricCellGeometryElement,
                    imported_cell_geometry.RestrictedCellGeometryElement,
                ),
            )
            for element in discretization.coordinate_elements
        ):
            raise ValueError(
                f"bound_coordinate_spec.role must provide its represented coordinate specification; got Mapped nested {role} requires its represented coordinate specification."
            )
        geometry = imported_cell_geometry.CellGeometrySpec(
            {
                block.name: element
                for block, element in zip(
                    discretization.mesh.blocks,
                    discretization.coordinate_elements,
                    strict=True,
                )
            },
            {
                block.name: routes
                for block, routes in zip(
                    discretization.mesh.blocks,
                    discretization.coordinate_dofs,
                    strict=True,
                )
            },
            discretization.default_runtime.coordinates,
            budget=budget,
        )

    elements, routes, values = geometry.resolve(discretization.mesh, budget=budget)
    if any(
        actual.element_id != expected.element_id
        or not np.array_equal(actual_routes, expected_routes)
        for actual, expected, actual_routes, expected_routes in zip(
            elements,
            discretization.coordinate_elements,
            routes,
            discretization.coordinate_dofs,
            strict=True,
        )
    ) or not np.array_equal(values, discretization.default_runtime.coordinates):
        raise ValueError(
            f"bound_coordinate_spec.geometry must agree with the declared geometry, topology, and field layout; got The {role} nested geometry differs from the represented prepared field map."
        )
    return geometry


def _reference_contains(
    kind: str,
    points: NDArray[np.float64],
    tolerance: float,
) -> NDArray[np.bool_]:
    if kind in ("triangle", "tetrahedron"):
        return np.all(points >= -tolerance, axis=-1) & (
            points.sum(axis=-1) <= 1.0 + tolerance
        )
    if kind == "prism":
        return (
            np.all(points >= -tolerance, axis=-1)
            & (points[:, :2].sum(axis=-1) <= 1.0 + tolerance)
            & (points[:, 2] <= 1.0 + tolerance)
        )
    if kind == "pyramid":
        height = points[:, 2]
        return (
            (height >= -tolerance)
            & (height <= 1.0 + tolerance)
            & np.all(points[:, :2] >= 0.5 * height[:, None] - tolerance, axis=-1)
            & np.all(points[:, :2] <= 1.0 - 0.5 * height[:, None] + tolerance, axis=-1)
        )
    if kind in ("interval", "quadrilateral", "hexahedron"):
        return np.all((points >= -tolerance) & (points <= 1.0 + tolerance), axis=-1)
    raise ValueError(
        f"reference_contains.kind must select a supported declared route; got No supported nested reference containment for {kind!r}."
    )


def _mapped_h1_functional_rows(
    source: _NestedFiniteElementDiscretization,
    target: _NestedFiniteElementDiscretization,
    source_elements: tuple[FiniteElementSpec, ...],
    target_elements: tuple[FiniteElementSpec, ...],
    source_dofs: FiniteElementDofMap,
    target_dofs: FiniteElementDofMap,
    pairs: tuple[NestedReferencePair, ...],
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Evaluate oriented nodal functionals and reconcile shared target rows."""
    source_starts = np.cumsum(
        (0,) + tuple(block.cell_count for block in source.mesh.blocks)
    )
    target_starts = np.cumsum(
        (0,) + tuple(block.cell_count for block in target.mesh.blocks)
    )
    candidate_targets: list[int] = []
    candidate_sources: list[NDArray[np.int64]] = []
    candidate_values: list[NDArray[np.float64]] = []
    frame_condition = 1.0
    for pair in pairs:
        source_block = int(
            np.searchsorted(source_starts, pair.source_cell, side="right") - 1
        )
        target_block = int(
            np.searchsorted(target_starts, pair.target_cell, side="right") - 1
        )
        source_local = pair.source_cell - int(source_starts[source_block])
        target_local = pair.target_cell - int(target_starts[target_block])
        source_element, target_element = (
            source_elements[source_block],
            target_elements[target_block],
        )
        nodes = np.asarray(target_element.reference_nodes, dtype=np.float64)
        condition = _frame_condition(pair.matrix[None])
        frame_condition = max(frame_condition, condition)
        if pair.fine_is_source:
            solve = imported_linalg.solve_small_linear(
                imported_linalg.SmallLinearSolvePlan(pair.matrix.shape[0]),
                pair.matrix,
                (nodes - pair.offset).T,
            )
            if not bool(np.all(np.asarray(solve.successful))):
                raise ValueError(
                    "prepare_mapped_nested_h1_transfer.configuration must have a nonsingular chart with resolved orientation; got Mapped H1 coarsening has a singular reference chart."
                )
            points = np.asarray(solve.value, dtype=np.float64).T
            selected = _reference_contains(
                source_element.cell_kind,
                points,
                CLAIM_ULPS * np.finfo(np.float64).eps * condition,
            )
        else:
            points = nodes @ pair.matrix.T + pair.offset
            selected = np.ones(nodes.shape[0], dtype=np.bool_)
        if not np.any(selected):
            continue
        values, _ = tabulated(source_element, points[selected])
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "prepare_mapped_nested_h1_transfer.source must be finite; got Mapped H1 source nodal functionals are not finite."
            )
        columns = np.asarray(source_dofs.cell_dofs[source_block])[source_local]
        source_signs = np.asarray(source_dofs.orientations[source_block])[source_local]
        targets = np.asarray(target_dofs.cell_dofs[target_block])[target_local, selected]
        target_signs = np.asarray(target_dofs.orientations[target_block])[
            target_local, selected
        ]
        candidate_targets.extend(targets.tolist())
        candidate_sources.extend(np.broadcast_to(columns, values.shape))
        candidate_values.extend(values * source_signs[None] / target_signs[:, None])
    width = max(element.local_dof_count for element in source_elements)
    columns = np.zeros((len(candidate_targets), width), dtype=np.int64)
    coefficients = np.zeros(columns.shape, dtype=np.float64)
    for row, (indices, values) in enumerate(
        zip(candidate_sources, candidate_values, strict=True)
    ):
        columns[row, : indices.size] = indices
        coefficients[row, : values.size] = values
    rows, values, continuity = owned_rows(
        np.asarray(candidate_targets, dtype=np.int64)[:, None],
        columns,
        coefficients[:, None],
        target_dofs.global_dof_count,
        source_dofs.global_dof_count,
    )
    return rows, values, continuity, frame_condition, width


def _prepare_mapped_nested_h1_transfer(
    source: _NestedFiniteElementDiscretization,
    target: _NestedFiniteElementDiscretization,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    /,
    *,
    field_name: str,
    parent_cells: ArrayLike | None = None,
    parent_reference_vertices: ArrayLike | None = None,
    geometry_transition: CellGeometryTransition | None = None,
    budget: CoordinateEnclosureBudget,
) -> FiniteElementFieldTransfer:
    """represented target nodal functionals, including full-patch coarsening.

    Coarsening and family charts need not have a field-space inclusion. This
    route therefore certifies nodal interpolation and shared trace agreement,
    never claims full function reproduction or conserved H1 content.
    """

    name = canonical_identifier(field_name, "field_name")
    source_index, target_index = source.field_index(name), target.field_index(name)
    source_elements, target_elements = (
        source.elements[source_index],
        target.elements[target_index],
    )
    degrees = {element.degree for element in source_elements + target_elements}
    if len(degrees) != 1 or any(
        element.conformity != "H1"
        or element.value_shape
        or element.mapping != "identity"
        or element.element_id
        != imported_reference.lagrange_element(block.cell_kind, element.degree).element_id
        for space, elements in ((source, source_elements), (target, target_elements))
        for block, element in zip(space.mesh.blocks, elements, strict=True)
    ):
        raise ValueError(
            "prepare_mapped_nested_h1_transfer.configuration must provide one degree of declared nodal Lagrange fields; got Mapped nested H1 transfer requires one degree of declared nodal Lagrange fields."
        )
    source_space = source.field_spaces[source_index].vector_space
    target_space = target.field_spaces[target_index].vector_space
    if not isinstance(source_space, ArraySpace) or not isinstance(
        target_space, ArraySpace
    ):
        raise TypeError(
            "prepare_mapped_nested_h1_transfer.configuration must provide array coefficient spaces; got Mapped nested H1 transfer requires array coefficient spaces."
        )
    if source_space.shape[1:] != target_space.shape[1:]:
        raise ValueError(
            "prepare_mapped_nested_h1_transfer.configuration must agree with the declared geometry, topology, and field layout; got Mapped nested H1 component payloads differ."
        )
    pairs = imported_nested_reference.nested_reference_pairs(
        source.mesh,
        source_geometry,
        target.mesh,
        target_geometry,
        parent_cells=parent_cells,
        parent_reference_vertices=parent_reference_vertices,
        geometry_transition=geometry_transition,
        budget=budget,
    )
    geometry_bound = mapped_measure_owner.certify_nested_geometry_pairs(
        source.mesh, source_geometry, target.mesh, target_geometry, pairs, budget=budget
    )
    source_dofs, target_dofs = (
        source.dof_maps[source_index],
        target.dof_maps[target_index],
    )
    rows, values, continuity, frame_condition, width = _mapped_h1_functional_rows(
        source,
        target,
        source_elements,
        target_elements,
        source_dofs,
        target_dofs,
        pairs,
    )
    condition = (
        frame_condition * max(float(np.max(np.sum(np.abs(values), axis=-1))), 1.0) * width
    )
    coordinate_scale = max(
        float(np.max(np.abs(np.asarray(source_geometry.coordinates)))),
        float(np.max(np.abs(np.asarray(target_geometry.coordinates)))),
        1.0,
    )
    tolerance = CLAIM_ULPS * np.finfo(np.float64).eps * max(condition, 1.0)
    evidence = FiniteElementTransferEvidence(
        {
            "geometry": geometry_bound / coordinate_scale,
            "continuity": continuity,
            "constants": float(np.max(np.abs(values.sum(axis=-1) - 1.0))),
        },
        tolerance,
        bounds={"geometry": geometry_bound},
    )
    if not evidence.passed:
        raise ValueError(
            f"prepare_mapped_nested_h1_transfer.configuration must pass the declared scientific certificate; got Mapped H1 nodal/trace transfer failed: {evidence.defects}."
        )
    primal = SparseLinearMap(
        RowRelation(rows.astype(np.int32), source_size=source_dofs.global_dof_count),
        values,
        operator_id=canonical_fingerprint(
            {
                "kind": "mapped-nested-h1-functional-transfer",
                "source": source.prepared_id,
                "target": target.prepared_id,
                "field": name,
                "source_geometry": imported_cell_geometry_validity.cell_geometry_id(
                    source_geometry
                ),
                "target_geometry": imported_cell_geometry_validity.cell_geometry_id(
                    target_geometry
                ),
                "coefficients": array_tree_fingerprint(values),
                "routes": array_tree_fingerprint(rows),
            }
        ),
    )
    transfer = FiniteElementTopologyTransfer(
        primal,
        source.mesh.topology_id,
        target.mesh.topology_id,
        preserves_constants=True,
        positivity_preserving=bool(np.all(values >= 0.0)),
        semantics="interpolation",
        action_condition=max(condition, 1.0),
    )
    return FiniteElementFieldTransfer(
        transfer,
        source.field_spaces[source_index],
        target.field_spaces[target_index],
        TransferGeometryBinding(
            imported_cell_geometry_validity.cell_geometry_id(source_geometry),
            imported_cell_geometry_validity.cell_geometry_id(target_geometry),
            "exact-restriction",
            source_topology_id=source.mesh.topology_id,
            target_topology_id=target.mesh.topology_id,
        ),
        evidence,
    )


def _coordinate_geometry_id(geometry: CellGeometrySpec) -> str:

    return imported_cell_geometry_validity.cell_geometry_id(geometry)


def _mapped_nested_compatible_preparation(
    source: _NestedFiniteElementDiscretization,
    target: _NestedFiniteElementDiscretization,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    field_names: Sequence[str],
    /,
    *,
    parent_cells: ArrayLike | None = None,
    parent_reference_vertices: ArrayLike | None = None,
    geometry_transition: CellGeometryTransition | None = None,
    coarsening_witnesses: NestedReferenceWitnesses | None = None,
    budget: CoordinateEnclosureBudget,
) -> _MappedNestedCompatiblePreparation:
    names = tuple(canonical_identifier(name, "field_name") for name in field_names)
    if not names or len(set(names)) != len(names):
        raise ValueError(
            "mapped_nested_compatible_preparation.configuration must provide unique ordered field roles; got Compatible field preparation requires unique ordered field roles."
        )
    bindings = tuple(
        (
            _compatible_field_binding(source, name),
            _compatible_field_binding(target, name),
        )
        for name in names
    )
    for (source_binding, target_binding), name in zip(bindings, names, strict=True):
        source_index, target_index = source_binding[1], target_binding[1]
        conformities = {
            element.conformity
            for element in source.elements[source_index] + target.elements[target_index]
        }
        if conformities not in ({"Hcurl"}, {"Hdiv"}) or any(
            element.form_basis is None
            for element in source.elements[source_index] + target.elements[target_index]
        ):
            raise ValueError(
                f"mapped_nested_compatible_preparation.configuration must use a compatible form-basis field role; got Field {name!r} is not a declared compatible finite-element role."
            )
    pairs, geometry_bound = rooted_nested_reference_pairs(
        source.mesh,
        source_geometry,
        target.mesh,
        target_geometry,
        parent_cells=parent_cells,
        parent_reference_vertices=parent_reference_vertices,
        geometry_transition=geometry_transition,
        coarsening_witnesses=coarsening_witnesses,
        budget=budget,
    )
    return _MappedNestedCompatiblePreparation(
        source.prepared_id,
        target.prepared_id,
        source.mesh.numeric_version,
        target.mesh.numeric_version,
        _coordinate_geometry_id(source_geometry),
        _coordinate_geometry_id(target_geometry),
        bindings,
        pairs,
        geometry_bound,
    )


def _prepare_nested_compatible_field_transfers(
    source: _NestedFiniteElementDiscretization,
    target: _NestedFiniteElementDiscretization,
    parent_cells: ArrayLike | None,
    /,
    *,
    field_names: Sequence[str],
    parent_reference_vertices: ArrayLike | None = None,
    source_geometry: CellGeometrySpec | None = None,
    target_geometry: CellGeometrySpec | None = None,
    geometry_transition: CellGeometryTransition | None = None,
    coarsening_witnesses: NestedReferenceWitnesses | None = None,
    maximum_work: int = 100_000_000,
    maximum_storage_bytes: int = 256_000_000,
    budget: CoordinateEnclosureBudget,
) -> tuple[FiniteElementFieldTransfer, ...]:
    """Prepare ordered compatible fields from one authenticated nested substrate."""

    actual_source_geometry = bound_coordinate_spec(
        source, source_geometry, "source", budget=budget
    )
    actual_target_geometry = bound_coordinate_spec(
        target, target_geometry, "target", budget=budget
    )
    names = tuple(field_names)
    preparation = _mapped_nested_compatible_preparation(
        source,
        target,
        actual_source_geometry,
        actual_target_geometry,
        names,
        parent_cells=parent_cells,
        parent_reference_vertices=parent_reference_vertices,
        geometry_transition=geometry_transition,
        coarsening_witnesses=coarsening_witnesses,
        budget=budget,
    )
    return tuple(
        _prepare_mapped_nested_compatible_transfer(
            source,
            target,
            actual_source_geometry,
            actual_target_geometry,
            field_name=name,
            maximum_work=maximum_work,
            maximum_storage_bytes=maximum_storage_bytes,
            _reference_preparation=preparation,
            budget=budget,
        )
        for name in names
    )


def _prepare_mapped_nested_compatible_transfer(
    source: _NestedFiniteElementDiscretization,
    target: _NestedFiniteElementDiscretization,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    /,
    *,
    field_name: str,
    parent_cells: ArrayLike | None = None,
    parent_reference_vertices: ArrayLike | None = None,
    geometry_transition: CellGeometryTransition | None = None,
    coarsening_witnesses: NestedReferenceWitnesses | None = None,
    maximum_work: int = 100_000_000,
    maximum_storage_bytes: int = 256_000_000,
    _reference_preparation: _MappedNestedCompatiblePreparation | None = None,
    budget: CoordinateEnclosureBudget,
) -> FiniteElementFieldTransfer:

    source_index = source.field_index(field_name)
    target_index = target.field_index(field_name)
    preparation = _reference_preparation
    if preparation is None:
        preparation = _mapped_nested_compatible_preparation(
            source,
            target,
            source_geometry,
            target_geometry,
            (field_name,),
            parent_cells=parent_cells,
            parent_reference_vertices=parent_reference_vertices,
            geometry_transition=geometry_transition,
            coarsening_witnesses=coarsening_witnesses,
            budget=budget,
        )
    preparation.require(source, target, source_geometry, target_geometry, field_name)
    pairs, geometry_bound = preparation.pairs, preparation.geometry_bound
    actions = imported_mapped_form_transfer.prepare_form_patch_actions(
        source,
        target,
        source_index,
        target_index,
        pairs,
        maximum_work=maximum_work,
        maximum_storage_bytes=maximum_storage_bytes,
        budget=budget,
    )
    candidate_targets = actions.target_rows
    candidate_sources = actions.source_rows
    local = actions.coefficients
    defects, estimates = actions.defects, actions.estimates
    source_dofs = source.dof_maps[source_index]
    target_dofs = target.dof_maps[target_index]
    element = source.elements[source_index][0]
    rows, coefficients, continuity = owned_rows(
        candidate_targets[:, None],
        candidate_sources,
        local[:, None],
        target_dofs.global_dof_count,
        source_dofs.global_dof_count,
    )
    scale = max(
        float(np.max(np.abs(np.asarray(source_geometry.coordinates)))),
        float(np.max(np.abs(np.asarray(target_geometry.coordinates)))),
        1.0,
    )
    condition = max(estimates["moment_condition"], 1.0) * local.shape[1]
    evidence = FiniteElementTransferEvidence(
        {
            "geometry": geometry_bound / scale,
            "continuity": continuity,
            **defects,
        },
        CLAIM_ULPS * np.finfo(np.float64).eps * condition,
        bounds={"geometry": geometry_bound, "moments": defects["integration"]},
        estimates=estimates,
    )
    if not evidence.passed:
        raise ValueError(
            "mapped_nested_compatible_transfer.evidence must satisfy the field certificate within tolerance; got "
            f"field={field_name!r}, defects={evidence.defects!r}, tolerance={evidence.tolerance!r}"
        )
    primal = SparseLinearMap(
        RowRelation(rows.astype(np.int32), source_size=source_dofs.global_dof_count),
        coefficients,
        operator_id=canonical_fingerprint(
            {
                "kind": "mapped-nested-compatible-moment-transfer",
                "source": source.prepared_id,
                "target": target.prepared_id,
                "field": field_name,
                "source_geometry": _coordinate_geometry_id(source_geometry),
                "target_geometry": _coordinate_geometry_id(target_geometry),
                "coefficients": array_tree_fingerprint(coefficients),
                "routes": array_tree_fingerprint(rows),
            }
        ),
    )
    transfer = FiniteElementTopologyTransfer(
        primal,
        source.mesh.topology_id,
        target.mesh.topology_id,
        semantics=_nested_semantics(element, True),
        action_condition=condition,
    )
    return FiniteElementFieldTransfer(
        transfer,
        source.field_spaces[source_index],
        target.field_spaces[target_index],
        TransferGeometryBinding(
            _coordinate_geometry_id(source_geometry),
            _coordinate_geometry_id(target_geometry),
            "exact-restriction" if geometry_bound == 0 else "bounded-reconstruction",
            source_topology_id=source.mesh.topology_id,
            target_topology_id=target.mesh.topology_id,
        ),
        evidence,
    )


def _reference_join_isometry(
    source: CellMesh,
    target: CellMesh,
    pair: NestedReferencePair,
    expected: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Verify a chart image from scientific shared vertices and group members."""

    source_starts = np.cumsum((0,) + tuple(block.cell_count for block in source.blocks))
    target_starts = np.cumsum((0,) + tuple(block.cell_count for block in target.blocks))
    source_block = int(np.searchsorted(source_starts, pair.source_cell, side="right") - 1)
    target_block = int(np.searchsorted(target_starts, pair.target_cell, side="right") - 1)
    source_vertices = np.asarray(source.blocks[source_block].vertices)[
        pair.source_cell - source_starts[source_block]
    ]
    target_vertices = np.asarray(target.blocks[target_block].vertices)[
        pair.target_cell - target_starts[target_block]
    ]
    actual = np.asarray(source.coordinates)[source_vertices]
    scale = max(float(np.max(np.abs(actual))), float(np.max(np.abs(expected))), 1.0)
    tolerance = CLAIM_ULPS * np.finfo(np.float64).eps * scale
    if np.max(np.abs(actual - expected)) <= tolerance:
        return np.eye(actual.shape[1]), np.zeros(actual.shape[1])
    first, second = source.periodic_topology, target.periodic_topology
    if (
        first is None
        or second is None
        or periodic_exact_owner.identification_id(first.cell)
        != periodic_exact_owner.identification_id(second.cell)
    ):
        raise ValueError(
            "reference_join_isometry.configuration must agree with the declared geometry, topology, and field layout; got Reference coarsening charts differ without one declared periodic group."
        )
    source_vertex_ids = np.asarray(source.vertex_global_ids)[source_vertices]
    target_vertex_ids = np.asarray(target.vertex_global_ids)[target_vertices]
    common = np.intersect1d(source_vertex_ids, target_vertex_ids)
    if common.size == 0:
        raise ValueError(
            "reference_join_isometry.configuration must select a supported declared route; got Periodic chart relocation has no shared scientific anchor vertex."
        )
    identifier = common[0]
    source_vertex = int(
        source_vertices[np.flatnonzero(source_vertex_ids == identifier)[0]]
    )
    target_vertex = int(
        target_vertices[np.flatnonzero(target_vertex_ids == identifier)[0]]
    )
    source_orbit = np.asarray(first.orbits(0)[0])
    target_orbit = np.asarray(second.orbits(0)[0])
    source_representative = int(
        np.asarray(first.orbit_representatives(0))[source_orbit[source_vertex]]
    )
    target_representative = int(
        np.asarray(second.orbit_representatives(0))[target_orbit[target_vertex]]
    )
    if int(np.asarray(source.vertex_global_ids)[source_representative]) != int(
        np.asarray(target.vertex_global_ids)[target_representative]
    ):
        raise ValueError(
            "reference_join_isometry.configuration must agree with the declared geometry, topology, and field layout; got Periodic chart relocation has different scientific representative anchors."
        )
    dimension = actual.shape[1]
    source_map = first.orbit_isometries(0)[source_vertex]
    target_map = second.orbit_isometries(0)[target_vertex]
    rotation = target_map[:dimension, :dimension] @ source_map[:dimension, :dimension].T
    shift = (
        target_map[:dimension, dimension] - rotation @ source_map[:dimension, dimension]
    )
    if np.max(np.abs(actual @ rotation.T + shift - expected)) > tolerance:
        raise ValueError(
            "reference_join_isometry.configuration must realize the coarsening chart; got Stored periodic isometries do not realize the coarsening chart."
        )
    return rotation, shift


def _prepare_reference_coarsening_projection(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    /,
    *,
    field_name: str,
    coarsening_witnesses: NestedReferenceWitnesses,
    budget: CoordinateEnclosureBudget,
) -> FiniteElementFieldTransfer:
    """Conservative/commuting projection on the represented retained reference join."""

    _require_coefficient_affine_simplices(
        source, source_geometry, "source", budget=budget
    )
    _require_coefficient_affine_simplices(
        target, target_geometry, "target", budget=budget
    )
    source_index, target_index = (
        source.field_index(field_name),
        target.field_index(field_name),
    )
    element = _field_signature(source, source_index, target, target_index)
    pairs = imported_nested_reference.nested_reference_pairs(
        source.mesh,
        source_geometry,
        target.mesh,
        target_geometry,
        coarsening_witnesses=coarsening_witnesses,
        budget=budget,
    )
    if not all(pair.fine_is_source for pair in pairs):
        raise ValueError(
            "prepare_reference_coarsening_projection.source must provide fine-source integration pieces; got Reference coarsening projection needs fine-source integration pieces."
        )
    source_routes, source_signs = _cell_routes(source, source_index)
    target_routes, target_signs = _cell_routes(target, target_index)
    target_origins, target_frames = _affine_frames(target)
    parents = np.asarray([pair.target_cell for pair in pairs], dtype=np.int64)
    fine_cells = np.asarray([pair.source_cell for pair in pairs], dtype=np.int64)
    matrices = np.stack([pair.matrix for pair in pairs])
    offsets = np.stack([pair.offset for pair in pairs])
    fine_frames = target_frames[parents] @ matrices
    fine_origins = target_origins[parents] + np.asarray(
        ein.contract("pdm,pm->pd", target_frames[parents], offsets)
    )
    vertices = np.asarray(
        imported_reference_cell.reference_cell_topology(element.cell_kind).vertices
    )
    for row, pair in enumerate(pairs):
        expected = fine_origins[row] + vertices @ fine_frames[row].T
        _reference_join_isometry(source.mesh, target.mesh, pair, expected)
    rule = cubature_rule_data(
        _simplex_cubature_kind(element.cell_kind), 2 * _polynomial_degree(element)
    )
    points, rule_weights = np.asarray(rule.points), np.asarray(rule.weights)
    coarse_points = offsets[:, None] + points[None] @ np.swapaxes(matrices, -1, -2)
    physical = fine_origins[:, None] + points[None] @ np.swapaxes(fine_frames, -1, -2)
    weights = _affine_measure_densities(fine_frames)[:, None] * rule_weights[None]
    source_transformations = _cell_transformations(source, source_index)
    target_transformations = _cell_transformations(target, target_index)
    source_basis, source_differential = _mapped_basis(
        element,
        fine_frames,
        source_signs[fine_cells],
        points,
        transformation=None
        if source_transformations is None
        else source_transformations[fine_cells],
    )
    target_basis, _ = _mapped_basis(
        element,
        target_frames[parents],
        target_signs[parents],
        coarse_points,
        transformation=None
        if target_transformations is None
        else target_transformations[parents],
    )
    local = np.asarray(
        ein.contract("pq,pqiv,pqjv->pij", weights, target_basis, source_basis)
    )
    source_size = source.dof_maps[source_index].global_dof_count
    target_size = target.dof_maps[target_index].global_dof_count
    identity = {
        "kind": "authoritative-reference-coarsening",
        "source": source.prepared_id,
        "target": target.prepared_id,
        "field": field_name,
        "fine_ids": array_tree_fingerprint(
            np.asarray(coarsening_witnesses.fine_cell_ids)
        ),
        "coarse_ids": array_tree_fingerprint(
            np.asarray(coarsening_witnesses.coarse_cell_ids)
        ),
        "reference": array_tree_fingerprint(
            np.asarray(coarsening_witnesses.fine_reference_vertices)
        ),
    }
    mixed = coalesced_map(
        np.broadcast_to(target_routes[parents, :, None], local.shape).reshape(-1),
        np.broadcast_to(source_routes[fine_cells, None, :], local.shape).reshape(-1),
        local.reshape(-1),
        target_size=target_size,
        source_size=source_size,
        properties=OperatorProperties(),
        operator_id=canonical_fingerprint(identity),
    )
    # Coefficient-image authentication above already admits the target geometry.
    _projection_field_elements(target, target_index, "target")
    prepared = _factored_target(
        target, field_name, None, constraint_policy=FactorizationPolicy("svd")
    )
    projection = FiniteElementL2Projection(
        mixed,
        prepared,
        operator_id=canonical_fingerprint(
            {"kind": "reference-coarsening-projection", "mixed": mixed.operator_id}
        ),
        _construction_token=_L2_ARTIFACT_TOKEN,
    )
    value_width = source_basis.shape[-1]
    source_measures = np.zeros((source_size, value_width))
    integrals = np.asarray(ein.contract("pq,pqiv->piv", weights, source_basis))
    np.add.at(source_measures, source_routes[fine_cells], integrals)
    scalar = element.mapping == "identity"
    if scalar:
        source_measures = source_measures[:, 0]
        transfer = FiniteElementTopologyTransfer(
            projection,
            source.mesh.topology_id,
            target.mesh.topology_id,
            preserves_constants=True,
            conservative=True,
            semantics="l2-projection",
            action_condition=max(float(prepared.mass_condition.value), 1.0),
            source_measures=source_measures,
            target_measures=prepared.dof_measures,
        )
        pulled = np.asarray(transfer.pullback(prepared.dof_measures))
        content = _relative(pulled - source_measures, source_measures)
        evidence = FiniteElementTransferEvidence(
            {"coverage": 0.0, "content": content},
            CLAIM_ULPS * np.finfo(np.float64).eps * transfer.action_condition,
            bounds={"content": float(np.max(np.abs(pulled - source_measures)))},
        )
        if not evidence.passed:
            raise ValueError(
                "prepare_reference_coarsening_projection.configuration must pass the declared scientific certificate; got Scalar reference coarsening failed conservative content."
            )
        return FiniteElementFieldTransfer(
            transfer,
            source.field_spaces[source_index],
            target.field_spaces[target_index],
            TransferGeometryBinding(
                _coordinate_geometry_id(source_geometry),
                _coordinate_geometry_id(target_geometry),
                "exact-restriction",
                source_topology_id=source.mesh.topology_id,
                target_topology_id=target.mesh.topology_id,
            ),
            evidence,
            source_measures=source_measures,
            target_measures=prepared.dof_measures,
        )
    overlap = _ProjectionOverlap(
        projection,
        parents,
        physical,
        weights,
        source_routes[fine_cells],
        np.ones(source_routes[fine_cells].shape, dtype=np.bool_),
        source_differential,
        source_measures,
    )
    transfer = _constrained_projection(source, source_index, prepared, None, overlap)
    evidence = _compatible_projection_evidence(
        source, source_index, prepared, None, transfer, overlap
    )
    if not evidence.passed:
        raise ValueError(
            f"prepare_reference_coarsening_projection.configuration must pass the declared scientific certificate; got Compatible reference coarsening failed: {evidence.defects}."
        )
    return FiniteElementFieldTransfer(
        transfer,
        source.field_spaces[source_index],
        target.field_spaces[target_index],
        TransferGeometryBinding(
            _coordinate_geometry_id(source_geometry),
            _coordinate_geometry_id(target_geometry),
            "exact-restriction",
            source_topology_id=source.mesh.topology_id,
            target_topology_id=target.mesh.topology_id,
        ),
        evidence,
    )


type SourceRealizationFieldSemantics = Literal[
    "intensive", "conservative-density", "material-compatible"
]


def _realization_field_binding(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    realization: SourceGeometryRealization,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> TransferGeometryBinding:

    source_realization_owner.require_realization_reference_complex(
        source.mesh, target.mesh
    )
    realization.require_current()
    bound_coordinate_spec(source, source_geometry, "source", budget=budget)
    bound_coordinate_spec(target, target_geometry, "target", budget=budget)
    realization.source_embedding.binding.require(source.mesh, source_geometry)
    realization.target_embedding.binding.require(target.mesh, target_geometry)
    transition = realization.transition
    if transition.evidence.kind != "source_realization" or (
        transition.source_geometry_id
        != imported_cell_geometry_validity.cell_geometry_id(source_geometry)
        or transition.target_geometry_id
        != imported_cell_geometry_validity.cell_geometry_id(target_geometry)
    ):
        raise ValueError(
            "realization_field_binding.source must provide the represented source-realization maps; got Material field transfer requires the represented source-realization maps."
        )
    return TransferGeometryBinding(
        transition.source_geometry_id,
        transition.target_geometry_id,
        "source-realization",
        source_topology_id=source.mesh.topology_id,
        target_topology_id=target.mesh.topology_id,
        coverage_defect=None,
    )


def _realization_reference_field_transfer(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    binding: TransferGeometryBinding,
    field_name: str,
    semantics: SourceRealizationFieldSemantics,
    /,
) -> FiniteElementFieldTransfer:
    si, ti = source.field_index(field_name), target.field_index(field_name)
    old_map, new_map = source.dof_maps[si], target.dof_maps[ti]
    conformities = {
        value.conformity for value in source.elements[si] + target.elements[ti]
    }
    compatible = semantics == "material-compatible"
    if compatible:
        if conformities not in ({"Hcurl"}, {"Hdiv"}):
            raise ValueError(
                "realization_reference_field_transfer.configuration must provide an represented H(curl) or H(div) field; got Material-compatible realization requires an represented H(curl) or H(div) field."
            )
        meaning: TransferSemantics = (
            "covariant-piola" if conformities == {"Hcurl"} else "contravariant-piola"
        )
    else:
        if not conformities <= {"H1", "L2"}:
            raise ValueError(
                "realization_reference_field_transfer.configuration must provide scalar H1 or DG reference functionals; got Intensive material pullback requires scalar H1 or DG reference functionals."
            )
        meaning = "material-pullback"
    correspondence: dict[int, int] = {}
    for (
        old_element,
        new_element,
        old_routes,
        new_routes,
        old_transforms,
        new_transforms,
    ) in zip(
        source.elements[si],
        target.elements[ti],
        old_map.cell_dofs,
        new_map.cell_dofs,
        old_map.cell_transforms,
        new_map.cell_transforms,
        strict=True,
    ):
        if old_element.element_id != new_element.element_id or not np.array_equal(
            np.asarray(old_transforms), np.asarray(new_transforms)
        ):
            raise ValueError(
                "realization_reference_field_transfer.configuration must preserve the oriented reference field functionals; got Geometry realization changes the oriented reference field functionals."
            )
        if not compatible and (
            old_element.mapping != "identity" or old_element.value_shape
        ):
            raise ValueError(
                "realization_reference_field_transfer.configuration must provide represented scalar reference elements; got Intensive material pullback requires represented scalar reference elements."
            )
        for old, new in zip(
            np.asarray(old_routes).flat, np.asarray(new_routes).flat, strict=True
        ):
            row, column = int(new), int(old)
            previous = correspondence.get(row)
            if previous is not None and previous != column:
                raise ValueError(
                    "realization_reference_field_transfer.configuration must agree with the declared geometry, topology, and field layout; got Reference material transport disagrees at a shared field functional."
                )
            correspondence[row] = column
    if set(correspondence) != set(range(new_map.global_dof_count)) or set(
        correspondence.values()
    ) != set(range(old_map.global_dof_count)):
        raise ValueError(
            "realization_reference_field_transfer.source must cover every source and target functional; got Material reference transport does not cover every source and target functional."
        )
    rows = np.arange(new_map.global_dof_count, dtype=np.int32)
    columns = np.asarray(
        [correspondence[row] for row in range(new_map.global_dof_count)], dtype=np.int32
    )
    primal = coalesced_map(
        rows,
        columns,
        np.ones(rows.size, dtype=np.float64),
        target_size=new_map.global_dof_count,
        source_size=old_map.global_dof_count,
        properties=OperatorProperties(),
        operator_id=canonical_fingerprint(
            {
                "kind": "source-realization-material-functionals",
                "source": source.prepared_id,
                "target": target.prepared_id,
                "field": field_name,
                "meaning": meaning,
                "reference_owners": tuple(
                    value.element_id for value in source.elements[si]
                ),
                "source_routes": array_tree_fingerprint(columns),
            }
        ),
    )
    evidence = FiniteElementTransferEvidence(
        {
            "reference_reproduction": 0.0,
            "continuity": 0.0,
            **({"commuting": 0.0} if compatible else {"constants": 0.0}),
        },
        0.0,
    )
    transfer = FiniteElementTopologyTransfer(
        primal,
        source.mesh.topology_id,
        target.mesh.topology_id,
        conservative=False,
        preserves_constants=not compatible,
        positivity_preserving=not compatible,
        semantics=meaning,
    )
    return FiniteElementFieldTransfer(
        transfer, source.field_spaces[si], target.field_spaces[ti], binding, evidence
    )


def _realization_density_forms(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    realization: SourceGeometryRealization,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    field_name: str,
    maximum_work: int,
    maximum_storage_bytes: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> FiniteElementFieldTransfer:

    si, ti = source.field_index(field_name), target.field_index(field_name)
    ledger = budget
    with ledger.activate():
        sc, tc = (
            mapped_dg_cells(source, si, budget=budget),
            mapped_dg_cells(target, ti, budget=budget),
        )
    sg, tg = (
        mapped_measure_owner.mapped_geometry_cells(
            source.mesh, source_geometry, budget=budget
        ),
        mapped_measure_owner.mapped_geometry_cells(
            target.mesh, target_geometry, budget=budget
        ),
    )
    sm = np.zeros(source.dof_maps[si].global_dof_count, dtype=np.float64)
    tm = np.zeros(target.dof_maps[ti].global_dof_count, dtype=np.float64)
    se, te = np.zeros_like(sm), np.zeros_like(tm)
    masses: list[NDArray[np.float64]] = []
    mixed: list[dict[int, NDArray[np.float64]]] = []
    errors = 0.0
    old_values = np.asarray(realization.source_cell_measures, dtype=np.float64)
    new_values = np.asarray(realization.target_cell_measures, dtype=np.float64)
    old_errors = np.asarray(realization.source_measure_errors, dtype=np.float64)
    new_errors = np.asarray(realization.target_measure_errors, dtype=np.float64)
    work = [0, maximum_work]
    for cell, (
        (old_element, old_route, old_basis),
        (new_element, new_route, new_basis),
    ) in enumerate(zip(sc, tc, strict=True)):
        if old_element.element_id != new_element.element_id:
            raise ValueError(
                "realization_density_forms.configuration must preserve the reference field element; got Realization density transfer changes its reference field owner."
            )
        with ledger.activate(), ledger.temporary_scope():
            embedded = (
                source.mesh.topological_dimension == 2
                and source.mesh.ambient_dimension == 3
            )
            densities: tuple[Expression, Expression] | None = None
            if not embedded and (
                len(old_basis) != 1
                or old_basis[0]
                != imported_coordinate_enclosure.constant(
                    1, old_element.topological_dimension, budget=budget
                )
            ):
                densities = (
                    mapped_measure_owner.mapped_density_expression(
                        *sg[cell], budget=budget
                    ),
                    mapped_measure_owner.mapped_density_expression(
                        *tg[cell], budget=budget
                    ),
                )

            def integral(side: int, weight: Polynomial) -> tuple[float, float]:
                if embedded:
                    geometry_element, values = (sg if side == 0 else tg)[cell]
                    return mapped_measure_owner.certified_sqrt_mapped_integral_state(
                        geometry_element,
                        values,
                        weight,
                        1e-12,
                        1e-12,
                        10000,
                        32,
                        work,
                        budget=budget,
                    )
                if densities is None:
                    raise ValueError(
                        "integral.configuration must include prepared physical densities for nonconstant material projection; got Nonconstant material projection lacks its prepared physical density."
                    )

                return mapped_measure_owner.integrate_mapped_polynomial(
                    imported_coordinate_enclosure.expression_multiply(
                        densities[side], weight, budget=budget
                    ),
                    old_element.cell_kind,
                    budget=budget,
                )

            # DG0 consumes the already-enclosed physical inventories directly.
            if len(old_basis) == len(new_basis) == 1 and old_basis[0] == new_basis[
                0
            ] == imported_coordinate_enclosure.constant(
                1, old_element.topological_dimension, budget=budget
            ):
                old_value, old_error = old_values[cell], old_errors[cell]
                new_value, new_error = new_values[cell], new_errors[cell]
                sm[old_route], se[old_route] = old_value, old_error
                tm[new_route], te[new_route] = new_value, new_error
                mass = np.asarray(((new_value,),), dtype=np.float64)
                form = np.asarray(((old_value,),), dtype=np.float64)
                errors += old_error + new_error
            else:
                old_basis = (
                    old_basis
                    if old_element.cell_kind == "pyramid"
                    else tuple(
                        imported_coordinate_enclosure.compose(
                            value,
                            imported_coordinate_enclosure.chart_arguments(
                                old_element.cell_kind,
                                old_element.topological_dimension,
                                budget=budget,
                            ),
                            budget=budget,
                        )
                        for value in old_basis
                    )
                )
                new_basis = (
                    new_basis
                    if new_element.cell_kind == "pyramid"
                    else tuple(
                        imported_coordinate_enclosure.compose(
                            value,
                            imported_coordinate_enclosure.chart_arguments(
                                new_element.cell_kind,
                                new_element.topological_dimension,
                                budget=budget,
                            ),
                            budget=budget,
                        )
                        for value in new_basis
                    )
                )
                for route, basis, measures, measure_errors, side in (
                    (old_route, old_basis, sm, se, 0),
                    (new_route, new_basis, tm, te, 1),
                ):
                    for dof, weight in zip(route, basis, strict=True):
                        measures[dof], measure_errors[dof] = integral(side, weight)
                mass = np.empty((len(new_basis), len(new_basis)), dtype=np.float64)
                form = np.empty((len(new_basis), len(old_basis)), dtype=np.float64)
                for first, a in enumerate(new_basis):
                    for second, b in enumerate(new_basis):
                        mass[first, second], error = integral(
                            1, imported_coordinate_enclosure.multiply(a, b, budget=budget)
                        )
                        errors += error
                    for second, b in enumerate(old_basis):
                        form[first, second], error = integral(
                            0, imported_coordinate_enclosure.multiply(a, b, budget=budget)
                        )
                        errors += error
            masses.append(mass)
            mixed.append({cell: form})
    # Lazy: fem._topology_transfer <-> _surface_chart_transfer import cycle.
    from . import _surface_chart_transfer as imported_surface_chart_transfer

    return imported_surface_chart_transfer.finish_material_dg_projection(
        source,
        target,
        realization,
        field_name=field_name,
        source_cells=sc,
        target_cells=tc,
        source_measures=sm,
        target_measures=tm,
        source_errors=se,
        target_errors=te,
        masses=masses,
        mixed=mixed,
        integration_error=errors,
        semantics="conservative-density",
        budgets={
            "absolute_tolerance": 1e-12,
            "relative_tolerance": 1e-12,
            "maximum_work": maximum_work,
            "maximum_subcells": 10000,
            "maximum_binomial_terms": 32,
        },
    )


def _prepare_source_realization_field_transfer(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    realization: SourceGeometryRealization,
    /,
    *,
    field_name: str,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    semantics: SourceRealizationFieldSemantics,
    maximum_work: int = 100_000_000,
    maximum_storage_bytes: int = 512 * 1024**2,
    budget: CoordinateEnclosureBudget,
) -> FiniteElementFieldTransfer:
    """Material pullback or physical-density projection across changed domains.

    No world-point reproduction or common physical coverage is inferred.
    Compatible fields retain their represented oriented reference moments and Piola
    interpretation; conservative scalar DG carries old/new physical inventories.
    """
    meaning = parse(semantics, SourceRealizationFieldSemantics, "semantics")
    if maximum_work < 2 or maximum_storage_bytes < 1:
        raise ValueError(
            "prepare_source_realization_field_transfer.realization must provide positive work and storage budgets; got Source-realization transfer requires positive work and storage budgets."
        )
    binding = _realization_field_binding(
        source, target, realization, source_geometry, target_geometry, budget=budget
    )
    if meaning == "conservative-density":
        return _realization_density_forms(
            source,
            target,
            realization,
            source_geometry,
            target_geometry,
            field_name,
            maximum_work,
            maximum_storage_bytes,
            budget=budget,
        )
    return _realization_reference_field_transfer(
        source, target, binding, field_name, meaning
    )


@checked
def prepare_nested_field_transfer(
    source: _NestedFiniteElementDiscretization,
    target: _NestedFiniteElementDiscretization,
    parent_cells: ArrayLike | None,
    /,
    *,
    field_name: str,
    parent_reference_vertices: ArrayLike | None = None,
    source_geometry: CellGeometrySpec | None = None,
    target_geometry: CellGeometrySpec | None = None,
    geometry_transition: CellGeometryTransition | None = None,
    coarsening_witnesses: NestedReferenceWitnesses | None = None,
    maximum_work: int = 100_000_000,
    maximum_storage_bytes: int = 256_000_000,
    budget: CoordinateEnclosureBudget | None = None,
) -> FiniteElementFieldTransfer:
    """Prepare the algebraic transfer of one FE field onto a nested refinement.

    ``parent_cells[t]`` is the source cell (concatenated block order) containing
    target cell ``t``, the parent chart of a refinement lineage. The route is
    selected by the field's element family and DOF functionals on both meshes,
    never by array widths; an incompatible family raises ``ValueError``.

    Without ``parent_reference_vertices`` both meshes must be affine simplices and
    the child maps are recovered from physical corners. With it (``(C, d+1, d)``
    target corners in parent reference coordinates, e.g.
    ``MeshAdaptationResult.parent_reference_vertices``) the transfer is built in
    reference coordinates, so curved Lagrange coordinate maps are admitted; the
    ``geometry`` defect then certifies that the target map is the algebraic
    restriction of the parent map at every target coordinate node.

    Each target cell reconstructs the parent's basis functions in its own space,
    ``M^{-1} B`` with the target Piola-mapped basis (identity for H1/L2 Lagrange,
    covariant for H(curl) Nedelec, contravariant for H(div) RT/BDM).
    Where the target space contains the source space this equals applying the
    target DOF functionals (nodal values, edge circulation or face flux moments)
    and reproduces every source field exactly. The evidence certifies it:
    ``containment`` (the chart nests), ``reproduction`` (the space inclusion
    holds), ``continuity`` (shared DOFs agree across cells, validating trace
    continuity and orientation), and ``commuting`` for compatible fields (the
    curl/divergence of the transfer equals the transferred curl/divergence).
    Scalar Lagrange transfers then certify constant, linear, positivity, and
    content conservation claims on affine maps (constants only on curved maps,
    whose content measures this route does not integrate); a failed certificate
    claims nothing and its epoch transition refuses the rebind.

    Affine coefficient actions retain their rational source banks and material maps.
    Refinement requires zero algebraic map difference under a full reference
    partition; physical inventory weights come from the source-owned cell measure,
    never determinants of rounded carrier corners. Carrier-affine projection and
    whole-patch reference coarsening retain their bitwise FE/mesh agreement guard.

    Declared mixed/restricted maps use explicit ``source_geometry`` and
    ``target_geometry`` and immediate-parent reference witnesses. A bound
    ``geometry_transition`` supplies full coarsening patches when
    ``parent_cells`` is ``None``. H1 evaluates represented target nodal functionals
    (without claiming general coarsening function/content reproduction); DG
    projects with algebraic/enclosed physical forms and a conserved content ledger.
    Restricted coordinate routes remain source coefficients. Declared form
    fields apply their full entity functionals on certified reference
    patches, including full-source coarsening. Compatible preparation obeys
    cumulative ``maximum_work`` and ``maximum_storage_bytes`` bounds.

    A supplied budget retains request accounting; otherwise the call owns one
    ticket bounded by maximum_work and maximum_storage_bytes.
    """
    if budget is not None:
        return _prepare_nested_field_transfer(
            source,
            target,
            parent_cells,
            field_name=field_name,
            parent_reference_vertices=parent_reference_vertices,
            source_geometry=source_geometry,
            target_geometry=target_geometry,
            geometry_transition=geometry_transition,
            coarsening_witnesses=coarsening_witnesses,
            maximum_work=maximum_work,
            maximum_storage_bytes=maximum_storage_bytes,
            budget=budget,
        )
    with HostBatchTicket(
        maximum_work,
        queries_upper=field_transfer_query_upper(
            maximum_work, source_geometry, target_geometry
        ),
        bytes_upper=maximum_storage_bytes,
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=maximum_work,
            maximum_memory_bytes=maximum_storage_bytes,
        )
        with budget.activate():
            return _prepare_nested_field_transfer(
                source,
                target,
                parent_cells,
                field_name=field_name,
                parent_reference_vertices=parent_reference_vertices,
                source_geometry=source_geometry,
                target_geometry=target_geometry,
                geometry_transition=geometry_transition,
                coarsening_witnesses=coarsening_witnesses,
                maximum_work=maximum_work,
                maximum_storage_bytes=maximum_storage_bytes,
                budget=budget,
            )


@checked
def prepare_nested_compatible_field_transfers(
    source: _NestedFiniteElementDiscretization,
    target: _NestedFiniteElementDiscretization,
    parent_cells: ArrayLike | None,
    /,
    *,
    field_names: Sequence[str],
    parent_reference_vertices: ArrayLike | None = None,
    source_geometry: CellGeometrySpec | None = None,
    target_geometry: CellGeometrySpec | None = None,
    geometry_transition: CellGeometryTransition | None = None,
    coarsening_witnesses: NestedReferenceWitnesses | None = None,
    maximum_work: int = 100_000_000,
    maximum_storage_bytes: int = 256_000_000,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[FiniteElementFieldTransfer, ...]:
    """Prepare field-ordered compatible transfers on the declared nested geometry.

    A supplied coordinate budget retains request accounting; otherwise the call
    owns one ticket bounded by maximum_work and maximum_storage_bytes.
    """
    if budget is not None:
        return _prepare_nested_compatible_field_transfers(
            source,
            target,
            parent_cells,
            field_names=field_names,
            parent_reference_vertices=parent_reference_vertices,
            source_geometry=source_geometry,
            target_geometry=target_geometry,
            geometry_transition=geometry_transition,
            coarsening_witnesses=coarsening_witnesses,
            maximum_work=maximum_work,
            maximum_storage_bytes=maximum_storage_bytes,
            budget=budget,
        )
    with HostBatchTicket(
        maximum_work,
        queries_upper=field_transfer_query_upper(
            maximum_work, source_geometry, target_geometry
        ),
        bytes_upper=maximum_storage_bytes,
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=maximum_work,
            maximum_memory_bytes=maximum_storage_bytes,
        )
        with budget.activate():
            return _prepare_nested_compatible_field_transfers(
                source,
                target,
                parent_cells,
                field_names=field_names,
                parent_reference_vertices=parent_reference_vertices,
                source_geometry=source_geometry,
                target_geometry=target_geometry,
                geometry_transition=geometry_transition,
                coarsening_witnesses=coarsening_witnesses,
                maximum_work=maximum_work,
                maximum_storage_bytes=maximum_storage_bytes,
                budget=budget,
            )


@checked
def prepare_source_realization_field_transfer(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    realization: SourceGeometryRealization,
    /,
    *,
    field_name: str,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    semantics: SourceRealizationFieldSemantics,
    maximum_work: int = 100_000_000,
    maximum_storage_bytes: int = 512 * 1024**2,
    budget: CoordinateEnclosureBudget | None = None,
) -> FiniteElementFieldTransfer:
    """Prepare the request with one explicitly bounded host algebraic-work ticket."""
    if budget is not None:
        return _prepare_source_realization_field_transfer(
            source,
            target,
            realization,
            field_name=field_name,
            source_geometry=source_geometry,
            target_geometry=target_geometry,
            semantics=semantics,
            maximum_work=maximum_work,
            maximum_storage_bytes=maximum_storage_bytes,
            budget=budget,
        )
    with HostBatchTicket(
        maximum_work,
        queries_upper=field_transfer_query_upper(
            maximum_work, source_geometry, target_geometry
        ),
        bytes_upper=maximum_storage_bytes,
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=maximum_work,
            maximum_memory_bytes=maximum_storage_bytes,
        )
        with budget.activate():
            return _prepare_source_realization_field_transfer(
                source,
                target,
                realization,
                field_name=field_name,
                source_geometry=source_geometry,
                target_geometry=target_geometry,
                semantics=semantics,
                maximum_work=maximum_work,
                maximum_storage_bytes=maximum_storage_bytes,
                budget=budget,
            )


@checked
def prepare_projection_field_transfer(
    source: FiniteElementDiscretization,
    target: PreparedL2ProjectionTarget,
    refinement: imported_geometry.PreparedCommonRefinement,
    /,
    *,
    field_name: str,
    budget: CoordinateEnclosureBudget | None = None,
) -> FiniteElementFieldTransfer:
    """Prepare projection and algebraic source bindings within one explicit host request."""
    if budget is not None:
        return _prepare_projection_field_transfer(
            source,
            target,
            refinement,
            field_name=field_name,
            budget=budget,
        )
    sources = []
    for space in (source, target.discretization):
        coordinates = np.asarray(space.default_runtime.coordinates, dtype=np.float64)
        for element, routes in zip(
            space.coordinate_elements, space.coordinate_dofs, strict=True
        ):
            for route in np.asarray(routes):
                estimate = coordinate_polynomials_estimate(element, coordinates[route])
                if estimate is not None:
                    sources.append(estimate)
    estimate = coordinate_estimate_bank(tuple(sources))
    with HostBatchTicket(estimate.work_upper, bytes_upper=estimate.bytes_upper) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=estimate.work_upper,
            maximum_memory_bytes=estimate.bytes_upper,
        )
        with ledger.activate():
            return _prepare_projection_field_transfer(
                source,
                target,
                refinement,
                field_name=field_name,
                budget=ledger,
            )


def field_transfer_query_upper(
    maximum_work: int,
    source_geometry: CellGeometrySpec | None,
    target_geometry: CellGeometrySpec | None,
    /,
) -> int:
    # Every completed positive-query source preparation performs positive algebraic
    # work. The extra preparation covers its fail-fast refusal/invalidity prefix;
    # copied restriction vertices need not themselves consume coefficient work.
    source_queries = sum(
        geometry.exact_source.preparation_query_upper()
        for geometry in (source_geometry, target_geometry)
        if geometry is not None and geometry.exact_source is not None
    )
    return source_queries * (maximum_work + 1)
