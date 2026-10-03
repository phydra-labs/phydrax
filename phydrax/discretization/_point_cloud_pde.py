#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import assert_never, final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..linalg import (
    AbstractLinearOperator,
    AbstractPreconditioner,
    AbstractSparseLinearOperator,
    AdditiveSubspaceCorrectionPreconditioner,
    ArraySpace,
    DensePropertyVerificationPolicy,
    DiagonalLinearOperator,
    DifferentiationPolicy,
    FailurePolicy,
    GaussSeidelPreconditionerBuilder,
    GeneralizedLSMR,
    GMRES,
    ILUPreconditionerBuilder,
    LeastSquaresProblem,
    LinearCapabilityError,
    LinearSolveControl,
    LinearSolveDiagnostics,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    plan_sparse_assembly,
    PreconditioningPolicy,
    prepare,
    prepare_sparse_assembly,
    PreparedLinearSolve,
    PreparedSparseAssembly,
    refresh,
    refresh_sparse_assembly,
    solve,
    SolveResourcePolicy,
    SparseAssemblyPolicy,
    SparseFactorizationPolicy,
    SparseFactorizationPreconditionerBuilder,
    SubspaceCorrectionTerm,
    TolerancePolicy,
    transpose,
    verify_dense_properties,
)
from ..linalg.eigen import (
    general_eigensolve,
    GeneralEigenproblem,
    GeneralEigenResourcePolicy,
    GeneralEigenSelection,
    GeneralEigenSolvePolicy,
    GeneralEigenSolveResult,
    GeneralEigenSolveStatus,
    GeneralEigenTolerancePolicy,
    prepare_general_eigensolve,
    PreparedGeneralEigenSolve,
    refresh_general_eigensolve,
    RestartedArnoldi,
    ShiftInvertTransform,
)
from ..sparse import RowRelation, SparseCoordinateOperator
from ..typing import checked, ConvertibleToArray, parse
from ._point_cloud import PreparedPointCloudDiscretization
from .meshfree._boundary import (
    discretization_family,
    MultiIndex,
    PointDerivativeFamily,
    PointInterfaceCondition,
    PointSBPDerivatives,
    prepare_point_family,
    PreparedPointGhostLayer,
    PreparedPointSideSupport,
    sbp_identity_residuals,
)
from .meshfree._stencils import LocalStencilPolicy, LocalStencilReport
from .spatial import MortonAddressPlan
from .spatial._morton import _domain_point_coordinates


if TYPE_CHECKING:
    from ..conditions import Periodic
    from ..domain import PeriodicIdentification
    from .meshfree._multilevel import (
        MeshfreeCoarseningPolicy,
        MeshfreeHierarchyPlan,
        PreparedMeshfreeHierarchy,
    )

PointBoundaryKind: TypeAlias = Literal["dirichlet", "neumann", "robin", "periodic"]
PointDiffusionForm: TypeAlias = Literal["collocated", "dissipative"]
PointDiffusivityKind: TypeAlias = Literal["scalar", "tensor"]
PointNeumannCompatibility: TypeAlias = Literal["refuse", "project"]
PointPoissonRoute: TypeAlias = Literal[
    "square-collocation", "ghost-collocation", "oversampled-least-squares"
]
PointStabilityPolicy: TypeAlias = Literal["require-assessment", "diagnostic"]
PointStabilityOutcome: TypeAlias = Literal[
    "admitted", "nonpositive-real-part", "stability-unassessed"
]
PointStabilityRefresh: TypeAlias = Literal["reassess", "reuse-within-perturbation"]

# Below this many points ILU-GMRES converges in O(20) iterations and is cheaper
# to prepare; at and above it the native meshfree multigrid cycle is the
# default (measured 2-D Dirichlet iterations: ILU 31/64 vs multigrid 9-15 at
# 4096/16384 points).
_MULTILEVEL_DEFAULT_POINTS = 2048


def _module_identity(module: object, /) -> dict[str, object]:
    """Deterministic identity of a static policy tree and its numeric leaves."""
    return {
        "structure": str(jax.tree_util.tree_structure(module)),
        "leaves": array_tree_fingerprint(module),
    }


def _host_rows(rows: ArrayLike, name: str, /) -> np.ndarray:
    value = np.asarray(rows)
    if value.ndim != 1 or value.size == 0 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must be a nonempty integer vector.")
    if np.any(value < 0) or np.unique(value).size != value.size:
        raise ValueError(f"{name} must contain unique nonnegative row indices.")
    return value.astype(np.int32)


def _host_values(values: ArrayLike, count: int, name: str, /) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    if value.ndim > 1 or (value.ndim == 1 and value.shape != (count,)):
        raise ValueError(f"{name} must be a scalar or have one value per row.")
    value = np.broadcast_to(value, (count,)).copy()
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite.")
    return value


def _host_normals(normals: ConvertibleToArray, count: int, name: str, /) -> np.ndarray:
    value = np.asarray(normals, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != count:
        raise ValueError(f"{name} must have shape (rows, dimension).")
    lengths = np.linalg.norm(value, axis=1)
    if not np.all(np.isfinite(value)) or np.any(lengths <= 0.0):
        raise ValueError(f"{name} must be finite and nonzero.")
    return value / lengths[:, None]


@final
class _PointSeam(StrictModule):
    """One canonical periodic identification bound to a point-array axis."""

    identification_id: str = eqx.field(static=True)
    revision: str = eqx.field(static=True)
    relation_id: str | None = eqx.field(static=True)
    coordinate: tuple[str, int | None] = eqx.field(static=True)
    axis: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)


def _bound_seam(
    seam: PeriodicIdentification | Periodic,
    coordinates: Sequence[tuple[str, int | None]] | None,
    tolerance: float,
    /,
) -> tuple[_PointSeam, np.ndarray | None]:
    """Bind a seam to point axes; return its record and any relation target ``g``.

    Point seam rows realize ``u(upper) - u(lower) = g`` with homogeneous conormal
    flux balance: a `Periodic` relation is admitted only as that scalar
    same-field order-zero relation with identity transport and a constant real
    target. Antiperiodic, Bloch, event-linear, and jet relations are refused.
    """
    from ..conditions import JetAction, Periodic
    from ..domain import PeriodicIdentification

    target: np.ndarray | None = None
    relation_id: str | None = None
    if isinstance(seam, Periodic):
        plain = JetAction.derivative(0).action_id
        if not seam.identity_transport:
            raise ValueError("Point seam rows support identity seam transport only.")
        if (
            not seam.same_field
            or seam.value.shape != ()
            or seam.source_action.action_id != plain
            or seam.target_action.action_id != plain
        ):
            raise ValueError(
                "Point seam rows realize one scalar same-field order-zero trace relation."
            )
        if seam.target_constant is None or jnp.iscomplexobj(seam.target_constant):
            raise ValueError("Point seam rows need a constant real seam target.")
        identification = seam.identification
        target = np.asarray(seam.target_constant, dtype=np.float64)
        relation_id = seam.condition_id
    elif isinstance(seam, PeriodicIdentification):
        identification = seam
    else:
        raise TypeError("seam must be a PeriodicIdentification or conditions.Periodic.")
    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("seam_tolerance must be finite and nonnegative.")
    axes, _, _ = _domain_point_coordinates(identification.domain, coordinates)
    coordinate = (identification.label, identification.component)
    if coordinate not in axes:
        raise ValueError(f"Identified coordinate {coordinate!r} is not a point axis.")
    record = _PointSeam(
        identification_id=identification.identification_id,
        revision=identification.revision,
        relation_id=relation_id,
        coordinate=coordinate,
        axis=axes.index(coordinate),
        dimension=len(axes),
        lower=identification.lower,
        upper=identification.upper,
        tolerance=threshold,
    )
    return record, target


@final
class PointBoundaryCondition(StrictModule):
    """One boundary entity: a kind, its equation rows, data, and geometry.

    ``normals`` are physical outward unit normals and ``measure`` the physical
    boundary measure of each row; neither is inferred from coordinates. ``side``
    names the material side whose one-sided support evaluates conormal
    derivatives at rows shared by several sides.

    A periodic entity is the closed-box point realization of one canonical
    seam, ``seam``: a `PeriodicIdentification`, or a `conditions.Periodic`
    relation on it that owns the seam target ``g`` (then omit ``values``).
    ``rows`` are target (upper-face) rows and ``partners`` their source
    (lower-face) images, so value rows enforce the canonical
    ``u(upper) - u(lower) = g`` and partner rows enforce conormal-flux balance.
    Normals are the identification's target-face normal and are not declared.
    ``coordinates`` binds point axes to the domain coordinates as in
    `MortonAddressPlan.from_periodic_identifications`. Preparation certifies the
    pairing against the row coordinates: rows on the upper face, partners on
    the lower face at equal transverse coordinates (the face map), within
    ``seam_tolerance`` times the larger of the period and the cloud extent.
    Only identity transport is realized; other transports are refused.
    """

    rows: Array
    values: Array
    robin_coefficient: Array | None
    normals: Array | None
    measure: Array | None
    partners: Array | None
    kind: PointBoundaryKind = eqx.field(static=True)
    label: str = eqx.field(static=True)
    component: int = eqx.field(static=True)
    side: str | None = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)
    seam: _PointSeam | None

    def __init__(
        self,
        kind: PointBoundaryKind,
        rows: ArrayLike,
        values: ArrayLike | None = None,
        /,
        *,
        label: str,
        component: int = 0,
        normals: ConvertibleToArray | None = None,
        measure: ArrayLike | None = None,
        robin_coefficient: ArrayLike | None = None,
        side: str | None = None,
        partners: ArrayLike | None = None,
        seam: PeriodicIdentification | Periodic | None = None,
        coordinates: Sequence[tuple[str, int | None]] | None = None,
        seam_tolerance: float = 1e-10,
    ) -> None:
        kind = parse(kind, PointBoundaryKind, "kind")
        name = canonical_identifier(label, "label")
        if isinstance(component, bool) or not isinstance(component, int) or component < 0:
            raise ValueError("component must be a nonnegative integer.")
        index = _host_rows(rows, "rows")
        direction = (
            None if normals is None else _host_normals(normals, index.size, "normals")
        )
        weights = (
            None if measure is None else _host_values(measure, index.size, "measure")
        )
        if weights is not None and np.any(weights <= 0.0):
            raise ValueError("Boundary measure must be positive.")
        seamless = seam is None and coordinates is None
        coefficient: np.ndarray | None = None
        images: np.ndarray | None = None
        record: _PointSeam | None = None
        target: np.ndarray | None = None
        match kind:
            case "dirichlet":
                if robin_coefficient is not None or partners is not None or not seamless:
                    raise ValueError(
                        "Dirichlet rows accept no Robin data, partners, or seam."
                    )
            case "neumann":
                if robin_coefficient is not None or partners is not None or not seamless:
                    raise ValueError(
                        "Neumann rows accept no Robin data, partners, or seam."
                    )
                if direction is None:
                    raise ValueError("Neumann rows require physical normals.")
            case "robin":
                if robin_coefficient is None or partners is not None or not seamless:
                    raise ValueError("Robin boundaries require robin_coefficient only.")
                if direction is None:
                    raise ValueError("Robin rows require physical normals.")
                coefficient = _host_values(
                    robin_coefficient, index.size, "Robin coefficients"
                )
                if np.any(coefficient < 0.0):
                    raise ValueError("Robin coefficients must be finite and nonnegative.")
                if not np.any(coefficient > 0.0):
                    raise ValueError(
                        "Zero-coefficient Robin is Neumann; choose neumann with an explicit compatibility policy."
                    )
            case "periodic":
                if robin_coefficient is not None or partners is None or seam is None:
                    raise ValueError(
                        "Periodic rows require a canonical seam and partners, and no Robin data."
                    )
                if direction is not None:
                    raise ValueError(
                        "Periodic row normals are the seam's target-face normal; omit normals."
                    )
                record, target = _bound_seam(seam, coordinates, seam_tolerance)
                if target is not None and values is not None:
                    raise ValueError(
                        "A conditions.Periodic relation owns the seam target; omit values."
                    )
                direction = np.zeros((index.size, record.dimension), dtype=np.float64)
                direction[:, record.axis] = 1.0
                images = _host_rows(partners, "partners")
                if images.size != index.size or np.intersect1d(images, index).size:
                    raise ValueError("Periodic partners must pair disjointly with rows.")
            case _:
                assert_never(kind)
        data = _host_values(
            (0.0 if values is None else values) if target is None else target,
            index.size,
            "Point boundary values",
        )
        side_ = None if side is None else canonical_identifier(side, "side")
        self.rows = jnp.asarray(index)
        self.values = jnp.asarray(data)
        self.robin_coefficient = None if coefficient is None else jnp.asarray(coefficient)
        self.normals = None if direction is None else jnp.asarray(direction)
        self.measure = None if weights is None else jnp.asarray(weights)
        self.partners = None if images is None else jnp.asarray(images)
        self.kind = kind
        self.label = name
        self.component = component
        self.side = side_
        self.seam = record
        identity: dict[str, object] = {
            "kind": "point-boundary-condition",
            "boundary_kind": kind,
            "label": name,
            "component": component,
            "side": side_,
            "rows": array_tree_fingerprint(index),
            "values": array_tree_fingerprint(data),
            "robin": None if coefficient is None else array_tree_fingerprint(coefficient),
            "normals": None if direction is None else array_tree_fingerprint(direction),
            "measure": None if weights is None else array_tree_fingerprint(weights),
            "partners": None if images is None else array_tree_fingerprint(images),
        }
        if record is not None:
            identity["seam"] = {
                "identification": record.identification_id,
                "revision": record.revision,
                "relation": record.relation_id,
                "coordinate": list(record.coordinate),
                "axis": record.axis,
                "tolerance": record.tolerance,
            }
        self.condition_id = canonical_fingerprint(identity)

    @property
    def owned_rows(self) -> np.ndarray:
        """Host rows whose equations this entity owns (partners included)."""
        rows = np.asarray(self.rows)
        if self.partners is None:
            return rows
        return np.concatenate((rows, np.asarray(self.partners)))

    @property
    def flux(self) -> bool:
        return self.kind != "dirichlet"


@final
class PointBoundaryPlan(StrictModule):
    """Canonical per-entity boundary declaration over equation rows.

    Every (row, component) is owned by at most one entity; corners therefore
    have explicit ownership rather than an incidental overwrite order. An empty
    declaration is valid only for clouds without boundary rows (for example a
    fully periodic cloud); every connected component then floats and receives
    an explicit gauge in the consuming plan.
    """

    conditions: tuple[PointBoundaryCondition, ...]
    row_count: int = eqx.field(static=True)
    components: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        conditions: Sequence[PointBoundaryCondition],
        /,
        *,
        row_count: int,
        components: int = 1,
    ) -> None:
        entities = tuple(conditions)
        if any(not isinstance(entity, PointBoundaryCondition) for entity in entities):
            raise TypeError("conditions must be PointBoundaryCondition values.")
        labels = tuple(entity.label for entity in entities)
        if len(set(labels)) != len(labels):
            raise ValueError("Boundary condition labels must be unique.")
        if row_count < 1 or components < 1:
            raise ValueError("row_count and components must be positive.")
        owner = np.full((components, row_count), -1, dtype=np.int64)
        dimensions = {
            entity.normals.shape[1] for entity in entities if entity.normals is not None
        }
        if len(dimensions) > 1:
            raise ValueError("Boundary normals must share one spatial dimension.")
        for number, entity in enumerate(entities):
            if entity.component >= components:
                raise ValueError(f"Condition {entity.label!r} names an absent component.")
            rows = entity.owned_rows
            if np.any(rows >= row_count):
                raise ValueError(f"Condition {entity.label!r} exceeds row_count.")
            clash = owner[entity.component, rows]
            if np.any(clash >= 0):
                other = labels[int(clash[clash >= 0][0])]
                raise ValueError(
                    f"Rows are owned by both {other!r} and {entity.label!r}; declare corner ownership once."
                )
            owner[entity.component, rows] = number
        self.conditions = entities
        self.row_count = row_count
        self.components = components
        self.plan_id = canonical_fingerprint(
            {
                "kind": "point-boundary-plan",
                "rows": row_count,
                "components": components,
                "conditions": tuple(entity.condition_id for entity in entities),
            }
        )

    def condition(self, label: str, /) -> PointBoundaryCondition:
        for entity in self.conditions:
            if entity.label == label:
                return entity
        raise ValueError(f"Unknown boundary condition {label!r}.")

    def owned(self, component: int = 0, /) -> np.ndarray:
        mask = np.zeros(self.row_count, dtype=np.bool_)
        for entity in self.conditions:
            if entity.component == component:
                mask[entity.owned_rows] = True
        return mask

    def row_values(
        self,
        component: int = 0,
        /,
        *,
        overrides: Mapping[str, ArrayLike] | None = None,
    ) -> Array:
        """Boundary right-hand sides placed on owned rows; partner rows carry zero."""
        replaced = {} if overrides is None else dict(overrides)
        unknown = set(replaced) - {entity.label for entity in self.conditions}
        if unknown:
            raise ValueError(f"Unknown boundary overrides {sorted(unknown)}.")
        values = jnp.zeros((self.row_count,), dtype=jnp.float64)
        for entity in self.conditions:
            if entity.component != component:
                continue
            data = (
                entity.values
                if entity.label not in replaced
                else jnp.broadcast_to(
                    jnp.asarray(replaced[entity.label], dtype=jnp.float64),
                    entity.values.shape,
                )
            )
            data = eqx.error_if(
                data, jnp.any(~jnp.isfinite(data)), "Boundary values must be finite."
            )
            values = values.at[entity.rows].set(data)
        return values


@final
class PointSBPReport(StrictModule, NonTrainableState):
    maximum_green_residual: float = eqx.field(static=True)
    maximum_conservation_residual: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)
    evidence_scope: Literal["full-sparse-coefficient-identity"] = eqx.field(
        static=True, default="full-sparse-coefficient-identity"
    )


def point_sbp_report(
    discretization: PreparedPointCloudDiscretization,
    /,
    *,
    tolerance: float = 1e-8,
    maximum_coefficients: int = 1_000_000,
) -> PointSBPReport:
    """Check every coefficient of MD + DᵀM = Bn, at host preparation.

    The conservation check is 1ᵀMD = 1ᵀBn, not a zero boundary flux claim.
    Resource refusal never substitutes a finite set of polynomial probes.
    """
    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("SBP tolerance must be finite and nonnegative.")
    budget = int(maximum_coefficients)
    count, width = discretization.relation.source_indices.shape
    if budget < 1 or 2 * count * width + count > budget:
        raise ValueError("Full sparse SBP identity exceeds maximum_coefficients.")
    boundary_weights = discretization.plan.boundary_quadrature_weights
    if boundary_weights is None:
        raise ValueError("Point SBP evidence requires boundary_quadrature_weights.")
    maximum_green, maximum_conservation = sbp_identity_residuals(
        np.asarray(discretization.relation.source_indices),
        np.asarray(discretization.relation.valid),
        tuple(np.asarray(first) for first, _ in discretization.derivative_weights),
        np.asarray(discretization.quadrature_weights),
        np.asarray(boundary_weights),
        np.asarray(discretization.plan.boundary_normals),
    )
    identifier = canonical_fingerprint(
        {
            "kind": "point-sbp-report",
            "discretization": discretization.prepared_id,
            "green": maximum_green,
            "conservation": maximum_conservation,
            "tolerance": threshold,
        }
    )
    return PointSBPReport(
        maximum_green,
        maximum_conservation,
        maximum_green <= threshold and maximum_conservation <= threshold,
        identifier,
    )


def point_solve_space(discretization: PreparedPointCloudDiscretization, /) -> ArraySpace:
    """Euclidean stiffness coordinates of scalar point equations."""
    return ArraySpace(
        discretization.state_shape,
        dtype=jnp.float64,
        space_id=f"{discretization.prepared_id}:point-scalar-coordinates",
    )


def _unit(dimension: int, *axes: int) -> MultiIndex:
    return tuple(sum(axis == d for axis in axes) for d in range(dimension))


@final
class PointDiffusivityEvidence(StrictModule, NonTrainableState):
    """Native finite/symmetric/definiteness evidence of a diffusivity field."""

    minimum_eigenvalue: Array
    maximum_eigenvalue: Array
    maximum_condition: Array
    symmetry_defect: Array
    finite: Array
    positive_definite: Array
    kind: PointDiffusivityKind = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.finite & self.positive_definite


def _diffusivity_evidence(
    coefficient: Array, used: Array, kind: PointDiffusivityKind, /
) -> PointDiffusivityEvidence:
    """Evidence over used entries; unused side entries are excluded, not repaired."""
    match kind:
        case "scalar":
            finite = jnp.all(jnp.where(used, jnp.isfinite(coefficient), True))
            minimum = jnp.min(jnp.where(used, coefficient, jnp.inf))
            maximum = jnp.max(jnp.where(used, coefficient, -jnp.inf))
            return PointDiffusivityEvidence(
                minimum_eigenvalue=minimum,
                maximum_eigenvalue=maximum,
                maximum_condition=maximum / minimum,
                symmetry_defect=jnp.asarray(0.0, dtype=coefficient.dtype),
                finite=finite,
                positive_definite=finite & (minimum > 0.0),
                kind=kind,
            )
        case "tensor":
            evidence = verify_dense_properties(
                coefficient,
                policy=DensePropertyVerificationPolicy(require_positive_definite=True),
            )
            return PointDiffusivityEvidence(
                minimum_eigenvalue=jnp.min(
                    jnp.where(used[..., None], evidence.eigenvalues, jnp.inf)
                ),
                maximum_eigenvalue=jnp.max(
                    jnp.where(used[..., None], evidence.eigenvalues, -jnp.inf)
                ),
                maximum_condition=jnp.max(
                    jnp.where(used, evidence.condition_estimate, 0.0)
                ),
                symmetry_defect=jnp.max(jnp.where(used, evidence.hermitian_defect, 0.0)),
                finite=jnp.all(jnp.where(used, evidence.finite, True)),
                positive_definite=jnp.all(
                    jnp.where(used, evidence.positive_definite, True)
                ),
                kind=kind,
            )
        case _:
            assert_never(kind)


def _coefficient_field(
    value: ArrayLike, count: int, dimension: int, kind: PointDiffusivityKind, /
) -> Array:
    coefficient = jnp.asarray(value, dtype=jnp.float64)
    match kind:
        case "scalar":
            if coefficient.ndim > 1 or coefficient.shape not in ((), (count,)):
                raise ValueError("Scalar diffusivity must be a scalar or (points,).")
            return jnp.broadcast_to(coefficient, (count,))
        case "tensor":
            tensor = (dimension, dimension)
            if coefficient.shape not in (tensor, (count, *tensor)):
                raise ValueError(
                    "Tensor diffusivity must have shape (dimension, dimension) or (points, dimension, dimension)."
                )
            return jnp.broadcast_to(coefficient, (count, *tensor))
        case _:
            assert_never(kind)


@final
class PointDiffusionOperator(StrictModule, NonTrainableState):
    """Collocated +div(K grad), or the quadrature-adjoint action -M⁻¹ ΣDᵢᵀMKᵢⱼDⱼ.

    ``K`` is a declared scalar or symmetric tensor field whose finiteness,
    symmetry, and positive definiteness are verified by native dense property
    evidence; invalid fields are refused, never symmetrized or shifted.
    Collocation applies the variable-coefficient product rule. Dissipative
    energy is nonpositive in the quadrature pairing, not necessarily in ℓ².
    Energy stability alone does not imply continuum consistency for arbitrary
    quadrature/non-SBP derivatives; no approximation order is certified here.

    With side support, each side supplies its own coefficient field and
    one-sided stencils; rows shared by several sides carry no bulk equation
    (an interface or boundary condition owns them).
    """

    discretization: PreparedPointCloudDiscretization
    diffusivity: Array
    families: tuple[PointDerivativeFamily, ...]
    home: Array
    evidence: PointDiffusivityEvidence
    sides: PreparedPointSideSupport | None
    sbp: PointSBPDerivatives | None
    form: PointDiffusionForm = eqx.field(static=True)
    kind: PointDiffusivityKind = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        discretization: PreparedPointCloudDiscretization,
        diffusivity: ArrayLike | tuple[ArrayLike, ...] | list[ArrayLike] = 1.0,
        /,
        *,
        form: PointDiffusionForm = "dissipative",
        kind: PointDiffusivityKind = "scalar",
        sides: PreparedPointSideSupport | None = None,
        sbp: PointSBPDerivatives | None = None,
    ) -> None:
        form = parse(form, PointDiffusionForm, "form")
        kind = parse(kind, PointDiffusivityKind, "kind")
        if sbp is not None and (form != "dissipative" or sides is not None):
            raise ValueError(
                "Prepared SBP derivatives require single-sided dissipative diffusion."
            )
        count = discretization.state_shape[0]
        dimension = discretization.spatial_dimension
        if sides is None:
            if isinstance(diffusivity, (tuple, list)):
                raise TypeError("Per-side diffusivities require side support.")
            fields: tuple[ArrayLike, ...] = (diffusivity,)
            families = (
                discretization_family(discretization)
                if sbp is None
                else sbp.bind(discretization),
            )
            home = jnp.ones((1, count), dtype=jnp.bool_)
            used = home
        else:
            if not isinstance(sides, PreparedPointSideSupport):
                raise TypeError("sides must be PreparedPointSideSupport.")
            if sides.plan.discretization.prepared_id != discretization.prepared_id:
                raise ValueError("Side support must belong to this point cloud.")
            if not isinstance(diffusivity, (tuple, list)) or len(diffusivity) != len(
                sides.sides
            ):
                raise ValueError("Side support requires one diffusivity field per side.")
            fields = tuple(diffusivity)
            families = sides.families
            used = sides.evidence.membership.T
            home = used & ~sides.evidence.one_sided[None, :]
        coefficient = jnp.stack(
            [_coefficient_field(field, count, dimension, kind) for field in fields]
        )
        evidence = _diffusivity_evidence(coefficient, used, kind)
        coefficient = eqx.error_if(
            coefficient,
            ~evidence.successful,
            "Point diffusivity must be finite, symmetric, and positive definite.",
        )
        self.discretization = discretization
        self.diffusivity = coefficient
        self.families = tuple(families)
        self.home = home
        self.evidence = evidence
        self.sides = sides
        self.sbp = sbp
        self.form = form
        self.kind = kind
        self.operator_id = canonical_fingerprint(
            {
                "kind": "point-diffusion",
                "discretization": discretization.prepared_id,
                "sides": None if sides is None else sides.support_id,
                "form": form,
                "coefficient_kind": kind,
                "diffusivity": array_tree_fingerprint(coefficient),
                "sbp": None if sbp is None else sbp.result_id,
            }
        )

    @property
    def space(self) -> ArraySpace:
        return point_solve_space(self.discretization)

    def rebind(
        self,
        discretization: PreparedPointCloudDiscretization,
        diffusivity: ArrayLike,
        /,
    ) -> PointDiffusionOperator:
        """Bind refreshed stage geometry and a coefficient field, traceably.

        ``discretization`` must share this operator's point layout (for
        example a refreshed stage geometry). The diffusivity evidence is
        recomputed natively and published, not raised: consumers inside a
        traced stage must gate on ``evidence.successful``. Side-supported
        operators carry per-side stencils and are rebuilt, not rebound.
        """
        if not isinstance(discretization, PreparedPointCloudDiscretization):
            raise TypeError("discretization must be PreparedPointCloudDiscretization.")
        if self.sbp is not None:
            raise ValueError(
                "SBP diffusion must be rebuilt with derivatives admitted for refreshed geometry."
            )
        if self.sides is not None:
            raise ValueError("Side-supported diffusion is rebuilt, not rebound.")
        count = discretization.state_shape[0]
        if count != self.discretization.state_shape[0]:
            raise ValueError("A rebound discretization must keep the point layout.")
        coefficient = _coefficient_field(
            diffusivity, count, discretization.spatial_dimension, self.kind
        )[None]
        evidence = _diffusivity_evidence(coefficient, self.home, self.kind)
        # operator_id stays the structural identity of the prepared operator:
        # a traced coefficient cannot be fingerprinted inside a stage.
        return eqx.tree_at(
            lambda item: (
                item.discretization,
                item.families,
                item.diffusivity,
                item.evidence,
            ),
            self,
            (
                discretization,
                (discretization_family(discretization),),
                coefficient,
                evidence,
            ),
        )

    def entries(self, side: int, /) -> dict[tuple[int, int], Array]:
        """Structurally nonzero coefficient entries ``K_ij`` of one side."""
        dimension = self.discretization.spatial_dimension
        match self.kind:
            case "scalar":
                return {(a, a): self.diffusivity[side] for a in range(dimension)}
            case "tensor":
                return {
                    (i, j): self.diffusivity[side, :, i, j]
                    for i in range(dimension)
                    for j in range(dimension)
                }
            case _:
                assert_never(self.kind)

    def mv(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.ndim < 1 or value.shape[0] != self.discretization.state_shape[0]:
            raise ValueError("Point diffusion values must begin with the point count.")
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        mass = self.discretization.quadrature_weights.reshape(shape)
        dimension = self.discretization.spatial_dimension
        output = jnp.zeros_like(value)
        for side, family in enumerate(self.families):
            home = self.home[side].reshape(shape)
            for (i, j), coefficient in self.entries(side).items():
                k = coefficient.reshape(shape)
                derivative = family.apply(value, _unit(dimension, j))
                if self.form == "dissipative":
                    output = (
                        output
                        - family.transpose_apply(
                            jnp.where(home, mass * k * derivative, 0.0),
                            _unit(dimension, i),
                        )
                        / mass
                    )
                else:
                    gradient = family.apply(coefficient, _unit(dimension, i)).reshape(
                        shape
                    )
                    term = (
                        k * family.apply(value, _unit(dimension, i, j))
                        + gradient * derivative
                    )
                    output = output + jnp.where(home, term, 0.0)
        return output

    def energy_rate(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        return jnp.real(
            jnp.vdot(
                value,
                self.discretization.quadrature_weights.reshape(shape) * self.mv(value),
            )
        )

    def stiffness(self) -> AbstractLinearOperator:
        """Native sparse-composable positive-sign elliptic operator.

        Dissipative returns -M L; collocated returns -L.
        """
        space = self.space
        dimension = self.discretization.spatial_dimension
        terms: list[AbstractLinearOperator] = []
        for side, family in enumerate(self.families):
            home = self.home[side].astype(jnp.float64)
            entries = self.entries(side)
            if self.form == "dissipative":
                mass = self.discretization.quadrature_weights * home
                for (i, j), coefficient in entries.items():
                    terms.append(
                        transpose(
                            family.operator(
                                _unit(dimension, i), source=space, target=space
                            )
                        )
                        @ family.operator(
                            _unit(dimension, j),
                            source=space,
                            target=space,
                            coefficients=mass * coefficient,
                        )
                    )
                continue
            weights = jnp.zeros_like(family.weights[0])
            for (i, j), coefficient in entries.items():
                gradient = family.apply(coefficient, _unit(dimension, i))
                weights = (
                    weights
                    - (home * coefficient)[:, None]
                    * family.weights_for(_unit(dimension, i, j))
                    - (home * gradient)[:, None] * family.weights_for(_unit(dimension, j))
                )
            terms.append(
                SparseCoordinateOperator(
                    family.relation,
                    weights,
                    source=space,
                    target=space,
                    operator_id=f"{self.operator_id}:collocated:{side}",
                )
            )
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return result

    def conormal_weights(self, side: int, normals: ArrayLike, /) -> Array:
        """Row weights of ``n · K grad`` on one side's stencils; zero normals give zero rows."""
        direction = jnp.asarray(normals, dtype=jnp.float64)
        dimension = self.discretization.spatial_dimension
        family = self.families[side]
        weights = jnp.zeros_like(family.weights[0])
        for (i, j), coefficient in self.entries(side).items():
            weights = weights + (direction[:, i] * coefficient)[
                :, None
            ] * family.weights_for(_unit(dimension, j))
        return weights


def _image_relation(
    relation: RowRelation, weights: Array, targets: np.ndarray, sources: np.ndarray, /
) -> tuple[RowRelation, Array]:
    """Copy route rows ``sources`` onto rows ``targets``; all other rows are empty."""
    indices = np.zeros(relation.route_shape, dtype=np.int32)
    valid = np.zeros(relation.route_shape, dtype=np.bool_)
    indices[targets] = np.asarray(relation.source_indices)[sources]
    valid[targets] = np.asarray(relation.valid)[sources]
    image = jnp.zeros_like(weights).at[targets].set(weights[sources])
    return RowRelation(indices, source_size=relation.source_size, valid=valid), image


def _selection(
    count: int, source_count: int, targets: np.ndarray, sources: np.ndarray, /
) -> RowRelation:
    indices = np.zeros((count, 1), dtype=np.int32)
    valid = np.zeros((count, 1), dtype=np.bool_)
    indices[targets, 0] = sources
    valid[targets, 0] = True
    return RowRelation(indices, source_size=source_count, valid=valid)


@final
class _RowLayout(StrictModule, NonTrainableState):
    """Host classification of equation rows for one scalar elliptic plan."""

    dirichlet: Array
    flux: Array
    robin: Array
    replica: Array
    replica_images: Array
    partner: Array
    partner_images: Array
    bulk: Array
    flux_normals: Array
    image_normals: Array


def _validate_point_seam(
    condition: PointBoundaryCondition,
    points: np.ndarray,
    address: MortonAddressPlan,
    /,
) -> None:
    """Certify a closed-box seam realization against its canonical identification.

    A periodic address realizes the seam half-open with one representative per
    orbit, so seam rows there would identify the seam twice. Otherwise every
    target row lies on the upper face and its partner is its face-map image on
    the lower face: equal transverse coordinates within the declared tolerance.
    Rows and partners are unique and disjoint, so the pairing is a bijection.
    """
    seam = condition.seam
    if seam is None or condition.partners is None:
        raise RuntimeError(f"Periodic condition {condition.label!r} lost its seam.")
    if points.ndim != 2 or points.shape[1] != seam.dimension:
        raise ValueError(
            f"Periodic condition {condition.label!r} binds {seam.dimension} point "
            f"coordinates; rows have shape {points.shape}."
        )
    if address.periodic_axes[seam.axis]:
        raise ValueError(
            f"Periodic condition {condition.label!r}: the periodic cloud address already "
            "identifies this seam half-open; seam rows need a closed-box (non-periodic) axis."
        )
    target = points[np.asarray(condition.rows)]
    source = points[np.asarray(condition.partners)]
    extent = float(np.max(np.ptp(points, axis=0)))
    tolerance = seam.tolerance * max(seam.upper - seam.lower, extent)
    if np.any(np.abs(target[:, seam.axis] - seam.upper) > tolerance):
        raise ValueError(
            f"Periodic condition {condition.label!r}: rows must lie on the target "
            f"(upper) face {seam.coordinate!r} = {seam.upper}."
        )
    if np.any(np.abs(source[:, seam.axis] - seam.lower) > tolerance):
        raise ValueError(
            f"Periodic condition {condition.label!r}: partners must lie on the source "
            f"(lower) face {seam.coordinate!r} = {seam.lower}."
        )
    if np.any(np.abs(np.delete(target - source, seam.axis, axis=1)) > tolerance):
        raise ValueError(
            f"Periodic condition {condition.label!r}: each partner must be the face-map "
            "image of its row (equal transverse coordinates)."
        )


def _row_layout(
    boundary: PointBoundaryPlan,
    interfaces: tuple[PointInterfaceCondition, ...],
    sides: tuple[str, ...],
    membership: np.ndarray,
    dimension: int,
    /,
    *,
    row_points: np.ndarray,
    address: MortonAddressPlan,
    component: int = 0,
) -> _RowLayout:
    """Classify rows and place per-side conormal normals (zero where unused).

    ``flux_normals[s]`` holds the normal each flux row applies to side ``s``'s
    conormal; ``image_normals[s]`` holds normals of periodic replica rows whose
    conormal is copied onto their partner rows. ``row_points`` are the equation
    row coordinates against which periodic seams are certified.
    """
    count = boundary.row_count
    side_count = len(sides)
    dirichlet = np.zeros(count, dtype=np.bool_)
    flux = np.zeros(count, dtype=np.bool_)
    robin = np.zeros(count, dtype=np.float64)
    replica = np.zeros(count, dtype=np.bool_)
    replica_images = np.zeros(count, dtype=np.int32)
    partner = np.zeros(count, dtype=np.bool_)
    partner_images = np.zeros(count, dtype=np.int32)
    flux_normals = np.zeros((side_count, count, dimension), dtype=np.float64)
    image_normals = np.zeros((side_count, count, dimension), dtype=np.float64)

    def evaluation_side(condition: PointBoundaryCondition, row: np.ndarray) -> np.ndarray:
        if condition.side is not None:
            if condition.side not in sides:
                raise ValueError(f"Condition {condition.label!r} names an unknown side.")
            side = sides.index(condition.side)
            if not np.all(membership[row, side]):
                raise ValueError(
                    f"Condition {condition.label!r} evaluates side {condition.side!r} outside its points."
                )
            return np.full(row.size, side)
        shared = np.count_nonzero(membership[row], axis=1) != 1
        if np.any(shared):
            raise ValueError(
                f"Condition {condition.label!r} has rows on several sides; declare side."
            )
        return np.argmax(membership[row], axis=1)

    for condition in boundary.conditions:
        if condition.component != component:
            continue
        rows = np.asarray(condition.rows)
        match condition.kind:
            case "dirichlet":
                dirichlet[rows] = True
            case "neumann" | "robin":
                normals = np.asarray(condition.normals)
                side = evaluation_side(condition, rows)
                flux[rows] = True
                flux_normals[side, rows] = normals
                if condition.kind == "robin":
                    robin[rows] = np.asarray(condition.robin_coefficient)
            case "periodic":
                _validate_point_seam(condition, row_points, address)
                normals = np.asarray(condition.normals)
                images = np.asarray(condition.partners)
                replica[rows] = True
                replica_images[rows] = images
                partner[images] = True
                partner_images[images] = rows
                flux[images] = True
                flux_normals[evaluation_side(condition, images), images] = -normals
                image_normals[evaluation_side(condition, rows), rows] = normals
            case _:
                assert_never(condition.kind)
    for interface in interfaces:
        rows = np.asarray(interface.rows)
        if interface.minus not in sides or interface.plus not in sides:
            raise ValueError(f"Interface {interface.label!r} names undeclared sides.")
        minus, plus = sides.index(interface.minus), sides.index(interface.plus)
        if not np.all(membership[rows, minus] & membership[rows, plus]):
            raise ValueError(
                f"Interface {interface.label!r} rows must belong to both of its sides."
            )
        if np.any(flux[rows] | dirichlet[rows] | replica[rows]):
            raise ValueError(
                f"Interface {interface.label!r} rows are already owned by another condition."
            )
        normals = np.asarray(interface.normals)
        flux[rows] = True
        flux_normals[minus, rows] = normals
        flux_normals[plus, rows] = -normals
    bulk = ~(dirichlet | flux | replica)
    shared = np.count_nonzero(membership, axis=1) > 1
    if np.any(bulk & shared):
        raise ValueError(
            "Rows shared by several sides need an interface or boundary condition."
        )
    if not np.any(bulk):
        raise ValueError("Poisson requires at least one bulk (interior) equation row.")
    return _RowLayout(
        dirichlet=jnp.asarray(dirichlet),
        flux=jnp.asarray(flux),
        robin=jnp.asarray(robin),
        replica=jnp.asarray(replica),
        replica_images=jnp.asarray(replica_images),
        partner=jnp.asarray(partner),
        partner_images=jnp.asarray(partner_images),
        bulk=jnp.asarray(bulk),
        flux_normals=jnp.asarray(flux_normals),
        image_normals=jnp.asarray(image_normals),
    )


def _ghost_row_layout(
    layout: _RowLayout, ghosts: PreparedPointGhostLayer, /
) -> _RowLayout:
    """Extend a single-sided cloud layout by one boundary-condition row per ghost.

    Neumann and Robin cloud rows become bulk PDE rows; ghost row ``N + g``
    carries the condition of cloud row ``ghosts.plan.rows[g]`` with its
    outward normal and Robin anchoring coefficient.
    """
    rows = np.asarray(ghosts.plan.rows)
    if np.any(np.asarray(layout.replica) | np.asarray(layout.partner)):
        raise ValueError("Boundary ghosts do not serve periodic rows.")
    if not np.array_equal(np.sort(rows), np.flatnonzero(np.asarray(layout.flux))) or not (
        np.array_equal(
            np.asarray(layout.flux_normals[0])[rows], np.asarray(ghosts.plan.normals)
        )
    ):
        raise ValueError(
            "The ghost layer must extend exactly this boundary's Neumann and Robin rows with their normals."
        )
    dimension = layout.flux_normals.shape[2]
    padding = np.zeros(rows.size, dtype=np.bool_)
    dirichlet = np.asarray(layout.dirichlet)
    total = dirichlet.size + rows.size
    flux_normals = np.zeros((1, total, dimension), dtype=np.float64)
    flux_normals[0, dirichlet.size :] = np.asarray(ghosts.plan.normals)
    return _RowLayout(
        dirichlet=jnp.asarray(np.concatenate((dirichlet, padding))),
        flux=jnp.asarray(np.concatenate((np.zeros_like(dirichlet), ~padding))),
        robin=jnp.asarray(
            np.concatenate((np.zeros(dirichlet.size), np.asarray(layout.robin)[rows]))
        ),
        replica=jnp.zeros((total,), dtype=jnp.bool_),
        replica_images=jnp.zeros((total,), dtype=jnp.int32),
        partner=jnp.zeros((total,), dtype=jnp.bool_),
        partner_images=jnp.zeros((total,), dtype=jnp.int32),
        bulk=jnp.asarray(np.concatenate((~dirichlet, padding))),
        flux_normals=jnp.asarray(flux_normals),
        image_normals=jnp.zeros((1, total, dimension), dtype=jnp.float64),
    )


@final
class PointCollocationPlan(StrictModule):
    """Independent target rows for oversampled least-squares collocation.

    ``row_measure`` is the physical volume measure of bulk target rows;
    boundary rows use their condition's physical boundary measure times the
    declared ``boundary_weight``. These row weights define the discrete
    objective ``sum_t w_t r_t²``, which differs from square collocation.
    """

    points: Array
    row_measure: Array
    boundary_mask: Array
    stencil: LocalStencilPolicy | None
    boundary_weight: float = eqx.field(static=True)
    maximum_weight_ratio: float = eqx.field(static=True)
    neighbors: int | None = eqx.field(static=True)
    maximum_candidates: int | None = eqx.field(static=True)
    target_chunk_size: int | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        row_measure: ArrayLike,
        /,
        *,
        boundary_mask: ArrayLike,
        boundary_weight: float,
        maximum_weight_ratio: float = 1e10,
        stencil: LocalStencilPolicy | None = None,
        neighbors: int | None = None,
        maximum_candidates: int | None = None,
        target_chunk_size: int | None = None,
    ) -> None:
        targets = np.asarray(points, dtype=np.float64)
        if targets.ndim != 2 or not np.all(np.isfinite(targets)):
            raise ValueError("Collocation points must be finite (targets, dimension).")
        measure = np.asarray(row_measure, dtype=np.float64)
        mask = np.asarray(boundary_mask)
        if mask.dtype != np.bool_ or mask.shape != targets.shape[:1]:
            raise ValueError("boundary_mask must be Boolean with one entry per target.")
        if measure.shape != targets.shape[:1] or np.any(~np.isfinite(measure)):
            raise ValueError("row_measure must be finite with one entry per target.")
        if np.any(measure[~mask] <= 0.0):
            raise ValueError("Bulk collocation rows need positive physical measure.")
        weight = float(boundary_weight)
        ratio = float(maximum_weight_ratio)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("boundary_weight must be finite and positive.")
        if not np.isfinite(ratio) or ratio < 1.0:
            raise ValueError("maximum_weight_ratio must be finite and at least one.")
        if stencil is not None and not isinstance(stencil, LocalStencilPolicy):
            raise TypeError("stencil must be a LocalStencilPolicy.")
        self.points = jnp.asarray(targets)
        self.row_measure = jnp.asarray(np.where(mask, 0.0, measure))
        self.boundary_mask = jnp.asarray(mask)
        self.stencil = stencil
        self.boundary_weight = weight
        self.maximum_weight_ratio = ratio
        self.neighbors = neighbors
        self.maximum_candidates = maximum_candidates
        self.target_chunk_size = target_chunk_size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "point-collocation-plan",
                "points": array_tree_fingerprint(targets),
                "measure": array_tree_fingerprint(np.where(mask, 0.0, measure)),
                "boundary": array_tree_fingerprint(mask),
                "boundary_weight": weight,
                "ratio": ratio,
                "stencil": None if stencil is None else _module_identity(stencil),
                "neighbors": neighbors,
                "candidates": maximum_candidates,
                "chunk": target_chunk_size,
            }
        )

    def prepare(
        self, discretization: PreparedPointCloudDiscretization, /
    ) -> PreparedPointCollocation:
        return PreparedPointCollocation(self, discretization)


@final
class PreparedPointCollocation(StrictModule, NonTrainableState):
    """Admitted value and derivative stencils from cloud sources to target rows."""

    plan: PointCollocationPlan
    family: PointDerivativeFamily
    report: LocalStencilReport
    discretization_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PointCollocationPlan,
        discretization: PreparedPointCloudDiscretization,
        /,
    ) -> None:
        if not isinstance(plan, PointCollocationPlan) or not isinstance(
            discretization, PreparedPointCloudDiscretization
        ):
            raise TypeError(
                "A PointCollocationPlan and prepared point cloud are required."
            )
        sources = np.asarray(discretization.points)
        targets = np.asarray(plan.points)
        if targets.shape[1] != sources.shape[1]:
            raise ValueError("Collocation targets must share the cloud dimension.")
        if targets.shape[0] < sources.shape[0]:
            raise ValueError(
                "Oversampled collocation requires at least as many target rows as unknowns."
            )
        policy = discretization.plan.stencil if plan.stencil is None else plan.stencil
        family, report, _ = prepare_point_family(
            discretization,
            targets,
            np.arange(targets.shape[0]),
            source_active=np.ones(sources.shape[0], dtype=np.bool_),
            neighbors=discretization.plan.neighbors
            if plan.neighbors is None
            else plan.neighbors,
            policy=policy,
            value=True,
            maximum_candidates=plan.maximum_candidates,
            target_chunk_size=plan.target_chunk_size,
            label=f"{plan.plan_id}:collocation",
        )
        self.plan = plan
        self.family = family
        self.report = report
        self.discretization_id = discretization.prepared_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-point-collocation",
                "plan": plan.plan_id,
                "cloud": discretization.prepared_id,
                "family": family.family_id,
            }
        )

    @property
    def target_count(self) -> int:
        return self.plan.points.shape[0]


def _unknown_components(
    families: Sequence[PointDerivativeFamily],
    pairs: np.ndarray,
    count: int,
    /,
    *,
    square: bool,
) -> np.ndarray:
    """Connected components of the unknown graph from stencil routes and identifications."""
    rows: list[np.ndarray] = [pairs[:, 0]]
    columns: list[np.ndarray] = [pairs[:, 1]]
    for family in families:
        valid = np.asarray(family.relation.valid)
        indices = np.asarray(family.relation.source_indices)
        if square:
            rows.append(np.repeat(np.arange(count), indices.shape[1])[valid.reshape(-1)])
            columns.append(indices.reshape(-1)[valid.reshape(-1)])
        else:
            # A rectangular row couples all of its unknowns; join them through
            # its first route.
            anchor = np.repeat(indices[:, :1], indices.shape[1], axis=1)
            rows.append(anchor[valid])
            columns.append(indices[valid])
    row = np.concatenate(rows)
    column = np.concatenate(columns)
    graph = coo_array(
        (np.ones(row.size, dtype=np.int8), (row, column)), shape=(count, count)
    )
    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int32)


def _components_and_gauges(
    families: Sequence[PointDerivativeFamily],
    layout: _RowLayout,
    count: int,
    route: PointPoissonRoute,
    gauges: Sequence[int] | None,
    /,
    *,
    identified: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Component labels of the unknowns and one gauge per floating component.

    A component floats when no Dirichlet or positive-Robin row constrains it;
    each floating component's constant is in the right nullspace.
    ``identified`` lists further unknown pairs coupled by an equation row that
    no stencil route records (a ghost and its boundary row).
    """
    replica = np.asarray(layout.replica)
    square = route != "oversampled-least-squares"
    pairs = (
        np.concatenate(
            (
                np.stack(
                    (np.asarray(layout.replica_images)[replica], np.flatnonzero(replica)),
                    axis=1,
                ),
                np.zeros((0, 2), dtype=np.int64) if identified is None else identified,
            )
        )
        if square
        else np.zeros((0, 2), dtype=np.int64)
    )
    labels = _unknown_components(families, pairs, count, square=square)
    anchored_rows = np.asarray(layout.dirichlet) | (np.asarray(layout.robin) > 0.0)
    if square:
        anchored = set(labels[anchored_rows].tolist())
    else:
        sources = np.asarray(families[0].relation.source_indices)[anchored_rows]
        valid = np.asarray(families[0].relation.valid)[anchored_rows]
        anchored = set(labels[sources[valid]].tolist())
    floating = sorted(
        (c for c in set(labels.tolist()) if c not in anchored),
        key=lambda c: int(np.flatnonzero(labels == c)[0]),
    )
    bulk = np.asarray(layout.bulk)
    if not square:
        if floating:
            raise ValueError(
                "Oversampled least squares refuses floating components; declare Dirichlet or Robin rows."
            )
        if gauges is not None:
            raise ValueError("Oversampled least squares takes no gauges.")
        return labels, ()
    if gauges is None:
        return labels, tuple(
            int(np.flatnonzero((labels == c) & bulk)[0]) for c in floating
        )
    selected = tuple(int(g) for g in gauges)
    if any(g < 0 or g >= count or not bulk[g] for g in selected):
        raise ValueError("Gauge must select an interior point.")
    if sorted(labels[list(selected)].tolist()) != sorted(floating):
        raise ValueError(
            "Gauges must select exactly one interior row per floating component."
        )
    return labels, tuple(sorted(selected, key=lambda g: floating.index(int(labels[g]))))


def _validate_row_weights(
    boundary: PointBoundaryPlan, collocation: PreparedPointCollocation, /
) -> None:
    weights = np.asarray(
        _least_squares_weights(boundary, collocation, collocation.target_count)
    )
    if weights.min() <= 0.0 or (
        weights.max() / weights.min() > collocation.plan.maximum_weight_ratio
    ):
        raise ValueError(
            "Oversampled row weights must be positive within maximum_weight_ratio."
        )


@final
class PointCloudPoissonResult(StrictModule):
    """Solution, original-equation residuals, and route evidence.

    ``values`` holds one value per cloud point. On the ghost route
    ``ghost_values`` holds the ghost unknowns (the declared extension) and
    ``ghost_extension_defect`` the extension-law evidence
    ``max_g |u_g - (E u)(x_g)|`` (see ``PointGhostLayerPlan``); both are
    ``None`` on other routes. Residual norms are over every equation row,
    ghost boundary rows included.
    """

    values: Array
    residual_norm: Array
    residual_tolerance: Array
    boundary_residual_norm: Array
    compatible: Array
    compatibility_residual: Array
    component_compatibility_residual: Array
    source_correction: Array
    gauge_residual: Array
    linear_result: LinearSolveResult
    correction_linear_result: LinearSolveResult | None
    diffusivity_evidence: PointDiffusivityEvidence
    ghost_values: Array | None
    ghost_extension_defect: Array | None
    route: PointPoissonRoute = eqx.field(static=True)
    continuum_consistent: bool = eqx.field(static=True)

    @property
    def algebraically_successful(self) -> Array:
        correction_ok = (
            jnp.asarray(True)
            if self.correction_linear_result is None
            else self.correction_linear_result.successful
        )
        accepted = (
            self.linear_result.successful
            & correction_ok
            & self.diffusivity_evidence.successful
            & jnp.isfinite(self.residual_norm)
        )
        match self.route:
            case "square-collocation" | "ghost-collocation":
                return (
                    accepted
                    & (self.residual_norm <= self.residual_tolerance)
                    & (self.boundary_residual_norm <= self.residual_tolerance)
                    & (self.gauge_residual <= self.residual_tolerance)
                )
            case "oversampled-least-squares":
                # The weighted residual of an oversampled system is not zero;
                # acceptance is the native least-squares status.
                return accepted
            case _:
                assert_never(self.route)

    @property
    def successful(self) -> Array:
        """Scientific acceptance additionally requires an authorized continuum realization."""
        return self.continuum_consistent & self.algebraically_successful

    @property
    def status(self) -> Array:
        return self.linear_result.status

    @property
    def diagnostics(self) -> LinearSolveDiagnostics:
        return self.linear_result.diagnostics


@final
class PointCloudPoissonPlan(StrictModule):
    """Canonical scalar elliptic plan ``-div(K grad u) = f`` on a point cloud.

    Square collocation owns one equation per cloud point; the oversampled route
    owns one weighted least-squares row per independent target. Boundary,
    interface, gauge, compatibility, linear-solve, preconditioning, hierarchy,
    assembly, stability, and precision choices are all bound into ``plan_id``.

    Square collocation defaults to ``stability="require-assessment"``: on
    irregular clouds collocated least-squares stencils can produce spurious
    eigenvalues with nonpositive real part and wrong solutions, so preparation
    fails closed unless a shift-invert Arnoldi assessment (``stability_assessment``)
    finds no such eigenvalue near zero. ``"diagnostic"`` records the same
    evidence and proceeds. The default ``assembly_policy`` and the default
    assessment's sparse-factor limits scale with the declared row count and
    stencil width.

    ``form="dissipative"`` retains volume test equations at natural boundary
    nodes, integrates Neumann loads with the declared boundary quadrature,
    and adds the Robin boundary mass. Essential rows use Dirichlet lifting.
    ``sbp`` explicitly binds admitted ``PointSBPDerivatives`` to their original
    cloud and quadrature. Algebraic Green identities alone do not establish
    continuum stability: only the native tensor SBP owning bridge authorizes
    this weak continuum realization. Otherwise raw values and
    ``algebraically_successful`` remain available, but ``continuum_consistent``
    and scientific ``successful`` are false.

    ``ghosts`` selects the PDE+BC boundary route (``"ghost-collocation"``):
    every Neumann and Robin row keeps its PDE equation, its boundary condition
    moves to the row of one ghost unknown outside the domain, and the unknowns
    are the cloud values followed by the ghost values. Boundary-condition rows
    are divided by their ghost offset ``δ_b`` so every row carries the units of
    the PDE; the residual and its tolerance are over these declared rows. The
    bulk source is therefore also required at flux boundary points. Variable
    diffusivity is data on the cloud: its gradient uses the cloud's own
    stencils, while derivatives of the unknown use ghost-extended stencils.
    """

    discretization: PreparedPointCloudDiscretization
    boundary: PointBoundaryPlan
    sides: PreparedPointSideSupport | None
    sbp: PointSBPDerivatives | None
    interfaces: tuple[PointInterfaceCondition, ...]
    collocation: PreparedPointCollocation | None
    ghosts: PreparedPointGhostLayer | None
    linear_policy: LinearSolvePolicy
    hierarchy: PreparedMeshfreeHierarchy | None
    assembly_policy: SparseAssemblyPolicy
    stability_assessment: GeneralEigenSolvePolicy
    layout: _RowLayout
    component_labels: Array
    form: PointDiffusionForm = eqx.field(static=True)
    diffusivity_kind: PointDiffusivityKind = eqx.field(static=True)
    compatibility: PointNeumannCompatibility = eqx.field(static=True)
    gauges: tuple[int, ...] = eqx.field(static=True)
    route: PointPoissonRoute = eqx.field(static=True)
    stability: PointStabilityPolicy = eqx.field(static=True)
    stability_refresh: PointStabilityRefresh = eqx.field(static=True)
    continuum_consistent: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: PreparedPointCloudDiscretization,
        boundary: PointBoundaryPlan,
        /,
        *,
        form: PointDiffusionForm = "collocated",
        diffusivity: PointDiffusivityKind = "scalar",
        sides: PreparedPointSideSupport | None = None,
        interfaces: Sequence[PointInterfaceCondition] = (),
        collocation: PreparedPointCollocation | None = None,
        linear_policy: LinearSolvePolicy | None = None,
        hierarchy: PreparedMeshfreeHierarchy | None = None,
        compatibility: PointNeumannCompatibility = "refuse",
        assembly_policy: SparseAssemblyPolicy | None = None,
        gauges: Sequence[int] | None = None,
        stability: PointStabilityPolicy = "require-assessment",
        stability_assessment: GeneralEigenSolvePolicy | None = None,
        stability_refresh: PointStabilityRefresh = "reassess",
        ghosts: PreparedPointGhostLayer | None = None,
        sbp: PointSBPDerivatives | None = None,
    ) -> None:
        if not isinstance(
            discretization, PreparedPointCloudDiscretization
        ) or not isinstance(boundary, PointBoundaryPlan):
            raise TypeError(
                "Poisson requires a prepared point cloud and PointBoundaryPlan."
            )
        form_ = parse(form, PointDiffusionForm, "form")
        kind = parse(diffusivity, PointDiffusivityKind, "diffusivity")
        compatibility_ = parse(compatibility, PointNeumannCompatibility, "compatibility")
        stability_ = parse(stability, PointStabilityPolicy, "stability")
        stability_refresh_ = parse(
            stability_refresh, PointStabilityRefresh, "stability_refresh"
        )
        if sbp is not None and (form_ != "dissipative" or sides is not None):
            raise ValueError(
                "Prepared SBP derivatives require single-sided dissipative Poisson."
            )
        if sbp is not None:
            sbp.bind(discretization)
        continuum_consistent = form_ != "dissipative" or (
            sbp is not None and sbp.stable_realization
        )
        if stability_assessment is not None and not isinstance(
            stability_assessment, GeneralEigenSolvePolicy
        ):
            raise TypeError(
                "stability_assessment must be a GeneralEigenSolvePolicy or None."
            )
        interfaces_ = tuple(interfaces)
        if any(not isinstance(i, PointInterfaceCondition) for i in interfaces_):
            raise TypeError("interfaces must be PointInterfaceCondition values.")
        if interfaces_ and sides is None:
            raise ValueError("Interface conditions require PreparedPointSideSupport.")
        if sides is not None and (
            not isinstance(sides, PreparedPointSideSupport)
            or sides.plan.discretization.prepared_id != discretization.prepared_id
        ):
            raise ValueError("sides must be side support prepared on this point cloud.")
        count = discretization.state_shape[0]
        dimension = discretization.spatial_dimension
        if collocation is None:
            route: PointPoissonRoute = "square-collocation"
            rows = count
            declared = np.asarray(discretization.plan.boundary_mask)
        else:
            if not isinstance(collocation, PreparedPointCollocation):
                raise TypeError("collocation must be PreparedPointCollocation.")
            if collocation.discretization_id != discretization.prepared_id:
                raise ValueError("Collocation targets were prepared for another cloud.")
            if form_ != "collocated" or sides is not None:
                raise ValueError(
                    "Oversampled least squares supports single-sided collocated form only."
                )
            route = "oversampled-least-squares"
            rows = collocation.target_count
            declared = np.asarray(collocation.plan.boundary_mask)
        if ghosts is not None:
            if not isinstance(ghosts, PreparedPointGhostLayer):
                raise TypeError("ghosts must be PreparedPointGhostLayer.")
            if ghosts.discretization_id != discretization.prepared_id:
                raise ValueError("The ghost layer was prepared for another cloud.")
            if collocation is not None or sides is not None or form_ != "collocated":
                raise ValueError(
                    "Boundary ghosts extend single-sided collocated square equations only."
                )
            route = "ghost-collocation"
        if boundary.row_count != rows or boundary.components != 1:
            raise ValueError("Boundary plan must declare one scalar component per row.")
        membership = (
            np.ones((rows, 1), dtype=np.bool_)
            if sides is None
            else np.asarray(sides.evidence.membership)
        )
        names = ("cloud",) if sides is None else sides.sides
        if sides is None and any(c.side is not None for c in boundary.conditions):
            raise ValueError("A boundary side requires PreparedPointSideSupport.")
        layout = _row_layout(
            boundary,
            interfaces_,
            names,
            membership,
            dimension,
            row_points=np.asarray(
                discretization.plan.points
                if collocation is None
                else collocation.plan.points
            ),
            address=discretization.plan.address,
        )
        if form_ == "dissipative":
            if (
                sides is not None
                or interfaces_
                or any(condition.kind == "periodic" for condition in boundary.conditions)
            ):
                raise ValueError(
                    "Weak dissipative Poisson requires single-sided natural or essential boundaries, not seam or interface row equations."
                )
            for condition in boundary.conditions:
                if condition.kind in ("neumann", "robin"):
                    measure = _natural_boundary_measure(discretization, condition)
                    if sbp is not None:
                        declared_measure = discretization.plan.boundary_quadrature_weights
                        if declared_measure is None:
                            raise ValueError(
                                "SBP natural boundaries require the cloud boundary quadrature."
                            )
                        if not np.array_equal(
                            np.asarray(measure),
                            np.asarray(declared_measure)[np.asarray(condition.rows)],
                        ) or not np.allclose(
                            np.asarray(condition.normals),
                            np.asarray(discretization.plan.boundary_normals)[
                                np.asarray(condition.rows)
                            ],
                            rtol=0.0,
                            atol=sbp.tolerance,
                        ):
                            raise ValueError(
                                "Natural boundary measure and orientation must match the SBP quadrature."
                            )
        owned = boundary.owned(0)
        if not np.array_equal(owned, declared):
            raise ValueError(
                "Boundary conditions must own exactly the declared boundary rows."
            )
        if (
            not np.any(owned)
            and form_ == "collocated"
            and not all(discretization.plan.address.periodic_axes)
        ):
            # Collocated rows on a bounded cloud without boundary rows have more
            # than a constant nullspace (e.g. affine functions); no gauge fixes
            # it. The dissipative form's symmetric G^T M K G instead imposes
            # the natural (weak homogeneous conormal) boundary condition.
            raise ValueError(
                "A boundary-free collocated Poisson plan requires a fully periodic cloud address."
            )
        unknowns = count if ghosts is None else ghosts.row_count
        identified: np.ndarray | None = None
        if ghosts is not None:
            layout = _ghost_row_layout(layout, ghosts)
            families: tuple[PointDerivativeFamily, ...] = (ghosts.family,)
            identified = np.stack(
                (count + np.arange(ghosts.ghost_count), np.asarray(ghosts.plan.rows)),
                axis=1,
            )
        elif collocation is None:
            families = (
                (discretization_family(discretization),)
                if sides is None
                else sides.families
            )
        else:
            _validate_row_weights(boundary, collocation)
            families = (collocation.family,)
        labels, gauge_rows = _components_and_gauges(
            families, layout, unknowns, route, gauges, identified=identified
        )
        assembly_policy_ = (
            _capacity_assembly_policy(max(rows, unknowns), families)
            if assembly_policy is None
            else assembly_policy
        )
        if not isinstance(assembly_policy_, SparseAssemblyPolicy):
            raise TypeError("assembly_policy must be SparseAssemblyPolicy.")
        if (
            linear_policy is None
            and hierarchy is None
            and route != "oversampled-least-squares"
            and unknowns >= _MULTILEVEL_DEFAULT_POINTS
        ):
            # Scalable default: the native meshfree hierarchy that eliminates
            # every decoupled identity row, as hierarchy_plan() declares.
            from .meshfree._multilevel import MeshfreeHierarchyPlan

            hierarchy = MeshfreeHierarchyPlan(
                discretization.points if ghosts is None else ghosts.points,
                boundary=jnp.asarray(_hierarchy_boundary(layout, gauge_rows, route)),
                eliminated=jnp.asarray(_eliminated_rows(layout, gauge_rows)),
                stable_ids=None if ghosts is None else ghosts.point_ids,
            ).prepare(_solve_space(discretization, ghosts))
        policy = (
            _default_policy(route, unknowns, hierarchy, assembly_policy_)
            if linear_policy is None
            else linear_policy
        )
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a native LinearSolvePolicy.")
        assessment = (
            collocation_stability_assessment(
                max(rows, unknowns),
                max(family.relation.route_shape[-1] for family in families),
                discretization.spatial_dimension,
                preconditioned=policy.preconditioning is not None,
                resources=policy.resources,
            )
            if stability_assessment is None
            else stability_assessment
        )
        if hierarchy is not None:
            _validate_hierarchy(
                hierarchy,
                discretization.points if ghosts is None else ghosts.points,
                layout,
                gauge_rows,
                route,
            )
        identifier = canonical_fingerprint(
            {
                "kind": "point-poisson-plan",
                "discretization": discretization.prepared_id,
                "boundary": boundary.plan_id,
                "sides": None if sides is None else sides.support_id,
                "interfaces": tuple(i.condition_id for i in interfaces_),
                "collocation": None if collocation is None else collocation.prepared_id,
                "ghosts": None if ghosts is None else ghosts.prepared_id,
                "sbp": None if sbp is None else sbp.result_id,
                "route": route,
                "form": form_,
                "continuum_consistent": continuum_consistent,
                "diffusivity": kind,
                "compatibility": compatibility_,
                "gauges": gauge_rows,
                "linear_policy": _module_identity(policy),
                "hierarchy": None if hierarchy is None else hierarchy.hierarchy_id,
                "assembly_policy": _module_identity(assembly_policy_),
                "stability": stability_,
                "stability_assessment": _module_identity(assessment),
                "stability_refresh": stability_refresh_,
                "precision": np.dtype(np.float64).name,
            }
        )
        self.discretization = discretization
        self.sbp = sbp
        self.boundary = boundary
        self.sides = sides
        self.interfaces = interfaces_
        self.collocation = collocation
        self.ghosts = ghosts
        self.linear_policy = policy
        self.hierarchy = hierarchy
        self.assembly_policy = assembly_policy_
        self.stability_assessment = assessment
        self.layout = layout
        self.component_labels = jnp.asarray(labels)
        self.form = form_
        self.continuum_consistent = continuum_consistent
        self.diffusivity_kind = kind
        self.compatibility = compatibility_
        self.gauges = gauge_rows
        self.route = route
        self.stability = stability_
        self.stability_refresh = stability_refresh_
        self.plan_id = identifier

    @property
    def solve_space(self) -> ArraySpace:
        return _solve_space(self.discretization, self.ghosts)

    @property
    def eliminated_rows(self) -> Array:
        """Decoupled identity rows of the solve operator: Dirichlet and gauge rows.

        A meshfree hierarchy gives these coordinates empty prolongation rows, so
        their identity equations never enter a Galerkin coarse operator
        (measured: retaining a 256-point Dirichlet ring on every level gave
        operator complexity 6.45 at 4096 points and nonpositive coarse
        diagonals).
        """
        return jnp.asarray(_eliminated_rows(self.layout, self.gauges))

    def hierarchy_plan(
        self, policy: MeshfreeCoarseningPolicy | None = None, /
    ) -> MeshfreeHierarchyPlan:
        """Meshfree hierarchy over the solve unknowns eliminating ``eliminated_rows``.

        Square collocation also declares its remaining non-bulk rows (Neumann,
        Robin, interface, periodic) as ``boundary``; the coarsening policy's
        ``boundary_retention_levels`` decides on how many levels they are kept.
        The ghost route's boundary rows carry PDE units and coarsen like bulk
        rows.
        """
        from .meshfree._multilevel import MeshfreeHierarchyPlan

        boundary = jnp.asarray(_hierarchy_boundary(self.layout, self.gauges, self.route))
        match self.route:
            case "square-collocation":
                return MeshfreeHierarchyPlan(
                    self.discretization.points,
                    boundary=boundary,
                    eliminated=self.eliminated_rows,
                    policy=policy,
                )
            case "ghost-collocation":
                if self.ghosts is None:
                    raise RuntimeError("The ghost route always carries its ghost layer.")
                return MeshfreeHierarchyPlan(
                    self.ghosts.points,
                    boundary=boundary,
                    eliminated=self.eliminated_rows,
                    stable_ids=self.ghosts.point_ids,
                    policy=policy,
                )
            case "oversampled-least-squares":
                raise ValueError(
                    "Meshfree hierarchies precondition square collocation only."
                )
            case _:
                assert_never(self.route)

    def prepare(
        self, diffusivity: ArrayLike | tuple[ArrayLike, ...] | list[ArrayLike] = 1.0, /
    ) -> PreparedPointCloudPoisson:
        return PreparedPointCloudPoisson(self, diffusivity)


def _capacity_assembly_policy(
    rows: int, families: Sequence[PointDerivativeFamily], /
) -> SparseAssemblyPolicy:
    """Sparse assembly limits scaled by the declared rows and stencil width.

    Every product assembled for this plan (the dissipative ``Dᵀ M K D``, the
    Dirichlet/gauge eliminations, the stability restriction, and the
    meshfree hierarchy's Galerkin ``Rᵀ A P``) contracts operands whose rows hold
    at most ``w`` stencil entries, so ``rows · w²`` bounds the contributions and
    entries of one product (measured: the 16384-point, 20-neighbor Galerkin
    level-1 product needs 4.02e6 of the 6.55e6 allowed). The workspace charges
    the composition recipe's ten 64-bit index arrays per contribution plus
    output storage per entry. The former fixed limits remain the floor.
    """
    width = max(family.relation.route_shape[-1] for family in families)
    entries = rows * width * width
    floor = SparseAssemblyPolicy()
    return SparseAssemblyPolicy(
        max_nnz=max(floor.max_nnz, entries),
        max_bytes=max(floor.max_bytes, 24 * entries + 8 * (rows + 1)),
        max_contributions=max(floor.max_contributions, entries),
        max_workspace_bytes=max(floor.max_workspace_bytes, 104 * entries),
    )


def collocation_stability_assessment(
    rows: int,
    width: int,
    dimension: int,
    /,
    *,
    preconditioned: bool,
    resources: SolveResourcePolicy | None = None,
    restart: int = 40,
) -> GeneralEigenSolvePolicy:
    """Default spectral assessment selected deterministically by declared capacity.

    Krylov–Schur shift-invert Arnoldi on the eigenvalues nearest zero: the low
    end of an elliptic spectrum, where spurious collocation modes appear
    (measured on a random 1024-point disk: GMLS degree 2 / 16 neighbors has 8 of
    its 16 nearest eigenvalues with Re <= 0; PHS degree 3 / 30 neighbors has min
    Re 11.58 ≈ kλ₁).

    ``rows`` is the solve dimension and ``width`` the widest operator row (for
    ``c`` coupled components, ``c`` times the stencil width). The fill of a
    fill-reducing factor of a ``w``-wide local operator over ``R`` rows is
    estimated from the nested-dissection growth of its spatial ``dimension``:
    ``R w`` in 1-D, ``R w ceil(log2 R)`` in 2-D, and ``2 w R^((2d-2)/d)`` for
    ``d >= 3``. While that estimate fits the fixed sparse-factorization limits,
    or when the plan declares no preconditioner (``preconditioned=False``), the
    shift-invert transform is one prepared sparse LU with limits scaled to the
    estimate and sixteen eigenvalues are assessed. Above it the transform is
    iterative: device-bound GMRES (relative tolerance 1e-8) on the assessed
    block, preconditioned by the plan's own prepared solve preconditioner
    restricted to the assessed rows (`assess_square_collocation` with
    ``transform_preconditioner=``), so no second factorization is built
    (measured: the 3-D 14.7k-point ghost operator exceeded 8e9 symbolic LU work
    after a 2038 s preparation). Each transformed action is then one
    preconditioned solve, so the iterative route assesses the eight nearest
    eigenvalues (measured on the random 4096-point PHS disk: 38 actions for
    eight against 61 for sixteen, the same minimum real part 11.560193 as LU).
    Inexact inner solves cannot weaken the evidence: every backward error is
    the residual of the original operator. ``resources`` is the declared solve's
    resource policy: the reused preconditioner is charged against the same
    budgets as in the solve (measured: the 3-D 14.7k-point Neumann ghost
    V-cycle retains 283 MB against the 256 MB default). ``restart`` is the
    iterative transform's GMRES restart; a caller whose own solve declares a
    larger restart passes it, so the transform reuses the Krylov memory that
    ``resources`` already budgets instead of restarting a long inner solve.
    """
    if rows < 1 or width < 1 or dimension < 1 or restart < 1:
        raise ValueError("rows, width, dimension, and restart must be positive.")
    if dimension <= 1:
        estimate = rows * width
    elif dimension == 2:
        estimate = rows * width * max(math.ceil(math.log2(rows)), 1)
    else:
        estimate = 2 * width * math.ceil(rows ** ((2 * dimension - 2) / dimension))
    floor = SparseFactorizationPolicy()
    transform_solve: LinearSolvePolicy | SparseFactorizationPolicy
    iterative = preconditioned and estimate > floor.max_factor_nnz
    if iterative:
        transform_solve = LinearSolvePolicy(
            GMRES(restart=restart, stagnation_iterations=40),
            tolerance=TolerancePolicy(relative=1e-8, absolute=0.0, max_steps=400),
            differentiation=DifferentiationPolicy("none"),
            failure=FailurePolicy("status"),
            require_device_binding=True,
            resources=resources,
        )
    else:
        entries = max(floor.max_factor_nnz, estimate)
        ratio = entries / floor.max_factor_nnz
        transform_solve = SparseFactorizationPolicy(
            "lu",
            ordering="approximate-minimum-degree",
            max_factor_nnz=entries,
            max_factor_bytes=math.ceil(ratio * floor.max_factor_bytes),
            max_symbolic_work=math.ceil(ratio * floor.max_symbolic_work),
        )
    return GeneralEigenSolvePolicy(
        RestartedArnoldi(restart="krylov-schur"),
        transform=ShiftInvertTransform(0.0),
        selection=GeneralEigenSelection.closest(0.0, 8 if iterative else 16),
        max_steps=400,
        transform_solve=transform_solve,
        resources=GeneralEigenResourcePolicy(max_dimension=max(rows, 4096)),
        vectors="right",
        failure=FailurePolicy("status"),
    )


def _solve_space(
    discretization: PreparedPointCloudDiscretization,
    ghosts: PreparedPointGhostLayer | None,
    /,
) -> ArraySpace:
    """Euclidean stiffness coordinates of the solved unknowns."""
    if ghosts is None:
        return point_solve_space(discretization)
    return ArraySpace(
        (ghosts.row_count,),
        dtype=jnp.float64,
        space_id=f"{ghosts.prepared_id}:ghost-scalar-coordinates",
    )


def _eliminated_rows(layout: _RowLayout, gauges: tuple[int, ...], /) -> np.ndarray:
    """Dirichlet and gauge rows: identity equations decoupled in the solve operator."""
    eliminated = np.asarray(layout.dirichlet).copy()
    eliminated[list(gauges)] = True
    return eliminated


def _hierarchy_boundary(
    layout: _RowLayout, gauges: tuple[int, ...], route: PointPoissonRoute, /
) -> np.ndarray:
    """Non-bulk rows that are not eliminated; none for the ghost route."""
    match route:
        case "square-collocation" | "oversampled-least-squares":
            return ~np.asarray(layout.bulk) & ~_eliminated_rows(layout, gauges)
        case "ghost-collocation":
            return np.zeros(np.asarray(layout.bulk).shape, dtype=np.bool_)
        case _:
            assert_never(route)


def _default_policy(
    route: PointPoissonRoute,
    count: int,
    hierarchy: PreparedMeshfreeHierarchy | None,
    assembly: SparseAssemblyPolicy,
    /,
) -> LinearSolvePolicy:
    match route:
        case "square-collocation" | "ghost-collocation":
            if hierarchy is None:
                preconditioning = PreconditioningPolicy(
                    ILUPreconditionerBuilder(), refresh="numeric"
                )
            else:
                from .meshfree._multilevel import meshfree_multigrid_builder

                levels = len(hierarchy.transfers)
                preconditioning = PreconditioningPolicy(
                    meshfree_multigrid_builder(
                        hierarchy,
                        smoothers=tuple(
                            GaussSeidelPreconditionerBuilder(direction="symmetric")
                            for _ in range(levels)
                        )
                        if levels
                        else None,
                        coarse_solver=SparseFactorizationPreconditionerBuilder(),
                        assembly=assembly,
                        refresh_mode="reuse-transfers",
                    )
                )
            return LinearSolvePolicy(
                GMRES(restart=min(40, count)),
                tolerance=TolerancePolicy(relative=1e-9, absolute=1e-10, max_steps=1000),
                preconditioning=preconditioning,
                failure=FailurePolicy("error"),
            )
        case "oversampled-least-squares":
            if hierarchy is not None:
                raise ValueError(
                    "Meshfree hierarchies precondition square collocation only."
                )
            return LinearSolvePolicy(
                GeneralizedLSMR(condition_limit=1e10),
                tolerance=TolerancePolicy(
                    relative=1e-10, absolute=1e-12, max_steps=50 * count
                ),
                failure=FailurePolicy("status"),
            )
        case _:
            assert_never(route)


def _validate_hierarchy(
    hierarchy: PreparedMeshfreeHierarchy,
    points: Array,
    layout: _RowLayout,
    gauges: tuple[int, ...],
    route: PointPoissonRoute,
    /,
) -> None:
    from .meshfree._multilevel import PreparedMeshfreeHierarchy

    if not isinstance(hierarchy, PreparedMeshfreeHierarchy):
        raise TypeError("hierarchy must be PreparedMeshfreeHierarchy.")
    if route == "oversampled-least-squares":
        raise ValueError("Meshfree hierarchies precondition square collocation only.")
    fine = hierarchy.spaces[0]
    if fine.shape != (points.shape[0],) or not np.array_equal(
        np.asarray(hierarchy.plan.points), np.asarray(points)
    ):
        raise ValueError("Hierarchy must be prepared on this plan's scalar unknowns.")
    required = _eliminated_rows(layout, gauges)
    if not np.all(np.asarray(hierarchy.plan.eliminated)[required]):
        raise ValueError(
            "Hierarchy must declare every Dirichlet and gauge identity row eliminated; "
            "use plan.hierarchy_plan()."
        )


def _least_squares_weights(
    boundary: PointBoundaryPlan, collocation: PreparedPointCollocation, rows: int, /
) -> Array:
    weights = jnp.asarray(collocation.plan.row_measure)
    for condition in boundary.conditions:
        if condition.measure is None:
            raise ValueError("Oversampled boundary rows require physical measure.")
        weights = weights.at[condition.rows].set(
            collocation.plan.boundary_weight * condition.measure
        )
        if condition.partners is not None:
            weights = weights.at[condition.partners].set(
                collocation.plan.boundary_weight * condition.measure
            )
    if weights.shape != (rows,):
        raise ValueError("Least-squares weights must cover every target row.")
    return weights


def _natural_boundary_measure(
    discretization: PreparedPointCloudDiscretization,
    condition: PointBoundaryCondition,
    /,
) -> Array:
    """Physical boundary quadrature for a weak natural load or Robin mass."""
    measure = condition.measure
    if measure is None:
        boundary_measure = discretization.plan.boundary_quadrature_weights
        if boundary_measure is None:
            raise ValueError(
                "Weak Neumann and Robin conditions require physical boundary quadrature."
            )
        measure = boundary_measure[condition.rows]
    measure = eqx.error_if(
        measure,
        jnp.any(~jnp.isfinite(measure) | (measure <= 0.0)),
        "Weak natural boundary measures must be finite and positive.",
    )
    return measure


def _square_operators(
    plan: PointCloudPoissonPlan, diffusion: PointDiffusionOperator
) -> tuple[AbstractLinearOperator, AbstractLinearOperator]:
    """Physical square equations and the Dirichlet-eliminated, gauged solve operator."""
    layout = plan.layout
    space = plan.solve_space
    count = space.shape[0]
    stiffness = diffusion.stiffness()
    physical: AbstractLinearOperator
    if plan.form == "dissipative":
        # Natural nodes retain their volume test equations. Only essential
        # rows are replaced; Neumann data belongs to the integrated load.
        physical = (
            DiagonalLinearOperator((~layout.dirichlet).astype(jnp.float64), space=space)
            @ stiffness
        )
        diagonal = layout.dirichlet.astype(jnp.float64)
        for condition in plan.boundary.conditions:
            if condition.kind == "robin":
                coefficient = condition.robin_coefficient
                if coefficient is None:
                    raise ValueError(
                        "Robin conditions require their boundary coefficient."
                    )
                diagonal = diagonal.at[condition.rows].add(
                    _natural_boundary_measure(plan.discretization, condition)
                    * coefficient
                )
        physical = physical + DiagonalLinearOperator(diagonal, space=space)
        return physical, _eliminated(plan, physical)
    physical = (
        DiagonalLinearOperator(layout.bulk.astype(jnp.float64), space=space) @ stiffness
    )
    for side, family in enumerate(diffusion.families):
        if bool(np.any(np.asarray(layout.flux_normals[side]) != 0.0)):
            physical = physical + SparseCoordinateOperator(
                family.relation,
                diffusion.conormal_weights(side, layout.flux_normals[side]),
                source=space,
                target=space,
                operator_id=f"{plan.plan_id}:conormal:{side}",
            )
        image_rows = np.flatnonzero(
            np.any(np.asarray(layout.image_normals[side]) != 0.0, axis=1)
        )
        if image_rows.size:
            targets = np.asarray(layout.replica_images)[image_rows]
            relation, weights = _image_relation(
                family.relation,
                diffusion.conormal_weights(side, layout.image_normals[side]),
                targets,
                image_rows,
            )
            physical = physical + SparseCoordinateOperator(
                relation,
                weights,
                source=space,
                target=space,
                operator_id=f"{plan.plan_id}:periodic-image:{side}",
            )
    replica = np.asarray(layout.replica)
    if np.any(replica):
        rows = np.flatnonzero(replica)
        physical = physical - SparseCoordinateOperator(
            _selection(count, count, rows, np.asarray(layout.replica_images)[rows]),
            jnp.ones((count, 1), dtype=jnp.float64),
            source=space,
            target=space,
            operator_id=f"{plan.plan_id}:periodic-replica",
        )
    diagonal = (
        layout.dirichlet.astype(jnp.float64)
        + layout.robin
        + layout.replica.astype(jnp.float64)
    )
    physical = physical + DiagonalLinearOperator(diagonal, space=space)
    return physical, _eliminated(plan, physical)


def _eliminated(
    plan: PointCloudPoissonPlan, physical: AbstractLinearOperator, /
) -> AbstractLinearOperator:
    """Dirichlet-column elimination and gauge-row replacement of square equations."""
    layout = plan.layout
    space = plan.solve_space
    solve_operator = physical
    if bool(np.any(np.asarray(layout.dirichlet))):
        solve_operator = physical @ DiagonalLinearOperator(
            (~layout.dirichlet).astype(jnp.float64), space=space
        ) + DiagonalLinearOperator(layout.dirichlet.astype(jnp.float64), space=space)
    if plan.gauges:
        gauge = (
            jnp.zeros(space.shape, dtype=jnp.float64)
            .at[jnp.asarray(plan.gauges)]
            .set(1.0)
        )
        solve_operator = DiagonalLinearOperator(
            1.0 - gauge, space=space
        ) @ solve_operator + DiagonalLinearOperator(gauge, space=space)
    return solve_operator


def _ghost_operators(
    plan: PointCloudPoissonPlan, diffusion: PointDiffusionOperator, /
) -> tuple[AbstractLinearOperator, AbstractLinearOperator]:
    """Ghost-extended PDE rows and offset-scaled boundary rows over cloud+ghost unknowns.

    Cloud row ``i`` (non-Dirichlet) is ``-(K : D² u + (div K) · D u)`` with
    ghost-extended stencils ``D`` and the coefficient gradient from the cloud's
    own stencils. Ghost row ``N + g`` is ``(n_b · K_b D u(x_b) + α_b u_b) / δ_b``.
    """
    ghosts = plan.ghosts
    if ghosts is None:
        raise RuntimeError("The ghost route always carries its ghost layer.")
    layout = plan.layout
    space = plan.solve_space
    family = ghosts.family
    cloud = diffusion.families[0]
    dimension = plan.discretization.spatial_dimension
    total = ghosts.row_count
    rows = np.asarray(ghosts.plan.rows)
    ghost_rows = ghosts.cloud_count + np.arange(ghosts.ghost_count)
    padding = jnp.zeros((ghosts.ghost_count,), dtype=jnp.float64)
    bulk = layout.bulk.astype(jnp.float64)
    scaled_normals = (
        jnp.zeros((total, dimension), dtype=jnp.float64)
        .at[rows]
        .set(ghosts.plan.normals / ghosts.offsets[:, None])
    )
    weights = jnp.zeros_like(family.weights[0])
    conormal = jnp.zeros_like(family.weights[0])
    for (i, j), coefficient in diffusion.entries(0).items():
        k = jnp.concatenate((coefficient, padding))
        gradient = jnp.concatenate(
            (cloud.apply(coefficient, _unit(dimension, i)), padding)
        )
        weights = (
            weights
            - (bulk * k)[:, None] * family.weights_for(_unit(dimension, i, j))
            - (bulk * gradient)[:, None] * family.weights_for(_unit(dimension, j))
        )
        conormal = conormal + (scaled_normals[:, i] * k)[:, None] * family.weights_for(
            _unit(dimension, j)
        )
    physical: AbstractLinearOperator = SparseCoordinateOperator(
        family.relation,
        weights,
        source=space,
        target=space,
        operator_id=f"{plan.plan_id}:ghost-pde",
    )
    relation, image = _image_relation(family.relation, conormal, ghost_rows, rows)
    physical = physical + SparseCoordinateOperator(
        relation,
        image,
        source=space,
        target=space,
        operator_id=f"{plan.plan_id}:ghost-boundary",
    )
    if bool(np.any(np.asarray(layout.robin) > 0.0)):
        coupling = (
            jnp.zeros((total, 1), dtype=jnp.float64)
            .at[ghost_rows, 0]
            .set(layout.robin[ghost_rows] / ghosts.offsets)
        )
        physical = physical + SparseCoordinateOperator(
            _selection(total, total, ghost_rows, rows),
            coupling,
            source=space,
            target=space,
            operator_id=f"{plan.plan_id}:ghost-robin",
        )
    physical = physical + DiagonalLinearOperator(
        layout.dirichlet.astype(jnp.float64), space=space
    )
    return physical, _eliminated(plan, physical)


def _oversampled_operator(
    plan: PointCloudPoissonPlan, diffusion: PointDiffusionOperator
) -> AbstractLinearOperator:
    """Rectangular target equations ``A: cloud values -> target rows``."""
    collocation = plan.collocation
    if collocation is None:
        raise ValueError("Oversampled route requires prepared collocation targets.")
    layout = plan.layout
    family = collocation.family
    dimension = plan.discretization.spatial_dimension
    source = plan.solve_space
    target = ArraySpace(
        (collocation.target_count,),
        dtype=jnp.float64,
        space_id=f"{collocation.prepared_id}:target-rows",
    )
    value = _unit(dimension)
    bulk = layout.bulk.astype(jnp.float64)
    weights = jnp.zeros_like(family.weights[0])
    normals = layout.flux_normals[0]
    for (i, j), coefficient in diffusion.entries(0).items():
        k_rows = family.apply(coefficient, value)
        gradient_rows = family.apply(coefficient, _unit(dimension, i))
        weights = (
            weights
            - (bulk * k_rows)[:, None] * family.weights_for(_unit(dimension, i, j))
            - (bulk * gradient_rows)[:, None] * family.weights_for(_unit(dimension, j))
            + (normals[:, i] * k_rows)[:, None] * family.weights_for(_unit(dimension, j))
        )
    diagonal = (
        layout.dirichlet.astype(jnp.float64)
        + layout.robin
        + layout.replica.astype(jnp.float64)
    )
    weights = weights + diagonal[:, None] * family.weights_for(value)
    operator: AbstractLinearOperator = SparseCoordinateOperator(
        family.relation,
        weights,
        source=source,
        target=target,
        operator_id=f"{plan.plan_id}:oversampled",
    )
    replica_rows = np.flatnonzero(np.asarray(layout.replica))
    if replica_rows.size:
        images = np.asarray(layout.replica_images)[replica_rows]
        relation, image_values = _image_relation(
            family.relation, family.weights_for(value), replica_rows, images
        )
        operator = operator - SparseCoordinateOperator(
            relation, image_values, source=source, target=target
        )
        image_normals = layout.image_normals[0]
        conormal = jnp.zeros_like(family.weights[0])
        for (i, j), coefficient in diffusion.entries(0).items():
            conormal = conormal + (
                image_normals[:, i] * family.apply(coefficient, value)
            )[:, None] * family.weights_for(_unit(dimension, j))
        relation, image_flux = _image_relation(
            family.relation, conormal, images, replica_rows
        )
        operator = operator + SparseCoordinateOperator(
            relation, image_flux, source=source, target=target
        )
    return operator


@final
class PointCollocationStability(StrictModule, NonTrainableState):
    """Spectral admission evidence of the square elliptic solve operator.

    The assessed operator is the native solve operator restricted to the rows
    that are not eliminated identity rows; Dirichlet and gauge rows contribute
    the exact eigenvalue 1. The declared general eigensolve
    (`phydrax.linalg.eigen.general_eigensolve`; by default Krylov–Schur
    shift-invert Arnoldi through one prepared sparse LU) estimates the eigenvalues nearest
    the assessment shift, and every converged estimate ``ρ`` carries a rigorous
    backward error ``ε``: ``ρ`` is an exact eigenvalue of an operator within
    spectral-norm distance ``ε``. ``"nonpositive-real-part"`` therefore states
    that the solve operator lies within ``maximum_backward_error`` of one with
    a nonpositive-real eigenvalue. ``"admitted"`` states that no converged
    estimate near the shift has ``Re <= 0``: an estimate over the assessed
    modes, not a certificate of the whole spectrum. ``"stability-unassessed"``
    records an assessment that did not finish within its declared work or
    resources (``unassessed_reason``).

    ``reused`` marks evidence carried over a coefficient refresh without a new
    eigensolve (``stability_refresh="reuse-within-perturbation"``). With
    ``perturbation_bound`` ``δ ≥ ‖A' - A‖₂`` (``sqrt(‖ΔA‖₁ ‖ΔA‖∞)`` against the
    last assessed operator), a pair with backward error ``ε`` for ``A`` has
    backward error at most ``ε + δ`` for ``A'``; only pairs whose combined
    error still meets the assessment's convergence tolerance are carried, and
    ``maximum_backward_error`` reports the combined bound. A carried
    ``"nonpositive-real-part"`` keeps its rigorous meaning. ``"admitted"`` is
    carried only for a certified self-adjoint operator (``"bauer-fike"``
    enclosure), where every carried ``ρ - (ε + δ)`` must stay positive.

    Center dominance ``a_ii / sum_j |a_ij|`` of the physical bulk rows is a
    diagnostic only: a nonpositive diagonal marks nearly coincident
    least-squares stencils but neither implies nor excludes spectral
    instability (a stable random-cloud PHS operator measured one bulk row at
    -4e-4).
    """

    spectrum: GeneralEigenSolveResult | None
    policy: PointStabilityPolicy = eqx.field(static=True)
    outcome: PointStabilityOutcome = eqx.field(static=True)
    assessed_rows: int = eqx.field(static=True)
    converged_count: int = eqx.field(static=True)
    nonpositive_count: int = eqx.field(static=True)
    minimum_real_part: float = eqx.field(static=True)
    maximum_backward_error: float = eqx.field(static=True)
    unassessed_reason: str | None = eqx.field(static=True)
    bulk_rows: int = eqx.field(static=True)
    nonpositive_diagonal_rows: int = eqx.field(static=True)
    minimum_center_dominance: float = eqx.field(static=True)
    median_center_dominance: float = eqx.field(static=True)
    reused: bool = eqx.field(static=True)
    perturbation_bound: float = eqx.field(static=True)

    @property
    def admitted(self) -> bool:
        return self.outcome == "admitted"


@final
class PointCollocationStabilityRefusal(ValueError):
    """Square collocation refused under ``stability="require-assessment"``."""

    stability: PointCollocationStability

    def __init__(self, stability: PointCollocationStability, /) -> None:
        match stability.outcome:
            case "nonpositive-real-part":
                message = (
                    f"Square point collocation refused: the spectral assessment found "
                    f"{stability.nonpositive_count} of {stability.converged_count} "
                    "converged eigenvalues nearest the assessment shift with "
                    "nonpositive real "
                    f"part (minimum real part {stability.minimum_real_part:.4g}, "
                    f"backward error <= {stability.maximum_backward_error:.2g}) over "
                    f"{stability.assessed_rows} assessed rows; such square collocation "
                    "is spectrally unstable. Use cubic-augmented phs-rbf-fd stencils "
                    "with about three times the polynomial basis size, the "
                    "dissipative form, the oversampled least-squares route, "
                    "declared boundary ghosts (PointGhostLayerPlan) for Neumann or "
                    "Robin rows, or stability='diagnostic' to proceed with "
                    "recorded evidence."
                )
            case "stability-unassessed":
                message = (
                    "Square point collocation refused (stability-unassessed): "
                    f"{stability.unassessed_reason}. Declare a stability_assessment "
                    "with larger work or resource limits, or stability='diagnostic' "
                    "to proceed with recorded evidence."
                )
            case "admitted":
                raise ValueError("An admitted stability assessment is not a refusal.")
            case _:
                assert_never(stability.outcome)
        super().__init__(message)
        self.stability = stability


@final
class PreparedCollocationAssessment(StrictModule, NonTrainableState):
    """Restricted solve operator and its prepared eigensolve, reused on refresh.

    ``spectrum`` is the last eigensolve and ``reference`` the restricted
    operator's values it assessed: the warm start and the perturbation bound of
    a later refresh are taken against them.
    """

    assembly: PreparedSparseAssembly
    eigensolve: PreparedGeneralEigenSolve
    spectrum: GeneralEigenSolveResult
    reference: Array


def _center_dominance(
    operator: AbstractLinearOperator, bulk: Array, /
) -> tuple[int, int, float, float]:
    """Bulk rows, nonpositive-diagonal rows, and min/median center dominance."""
    if not isinstance(operator, AbstractSparseLinearOperator):
        raise RuntimeError("Assembled point equations must be canonical sparse.")
    storage = operator.sparse_storage()
    count = storage.shape[0]
    values = np.asarray(storage.values)
    columns = np.asarray(storage.indices)
    rows = np.repeat(np.arange(count), np.diff(np.asarray(storage.indptr)))
    on_diagonal = columns == rows
    diagonal = np.bincount(
        rows[on_diagonal], weights=values[on_diagonal], minlength=count
    )
    magnitude = np.bincount(rows, weights=np.abs(values), minlength=count)
    bulk_rows = np.flatnonzero(np.asarray(bulk))
    dominance = diagonal[bulk_rows] / np.where(
        magnitude[bulk_rows] > 0.0, magnitude[bulk_rows], 1.0
    )
    return (
        bulk_rows.size,
        int(np.count_nonzero(diagonal[bulk_rows] <= 0.0)),
        float(np.min(dominance)) if dominance.size else 1.0,
        float(np.median(dominance)) if dominance.size else 1.0,
    )


def _fitted_assessment(
    policy: GeneralEigenSolvePolicy, dimension: int, /
) -> GeneralEigenSolvePolicy:
    """The declared assessment with its count capped to what ``dimension`` admits."""
    count = policy.selection.count
    method = policy.method
    if count is None or not isinstance(method, RestartedArnoldi):
        return policy
    match method.restart:
        case "krylov-schur":
            admitted = dimension - 1
        case "block-ritz":
            admitted = min(dimension - 2, dimension // 2)
        case _:
            assert_never(method.restart)
    if count <= admitted:
        return policy
    selection = GeneralEigenSelection(
        policy.selection.kind, count=admitted, target=policy.selection.target
    )
    return eqx.tree_at(lambda item: item.selection, policy, selection)


def _restricted_transform(
    assessment: GeneralEigenSolvePolicy,
    restriction: SparseCoordinateOperator,
    restricted: AbstractLinearOperator,
    preconditioner: AbstractPreconditioner | None,
    assembly_policy: SparseAssemblyPolicy,
    /,
) -> GeneralEigenSolvePolicy:
    """Precondition an iterative shift-invert transform with ``R P Rᵀ``.

    With the eliminated identity rows ordered last the solve operator is
    ``A = [[B, C], [0, I]]``, so ``A⁻¹ = [[B⁻¹, -B⁻¹ C], [0, I]]`` and
    ``R A⁻¹ Rᵀ = B⁻¹`` exactly: restricting the solve's prepared approximate
    inverse ``P ≈ A⁻¹`` gives an approximate inverse of the assessed block ``B``
    without a second preparation.
    """
    transform_solve = assessment.transform_solve
    if preconditioner is None:
        return assessment
    if (
        not isinstance(transform_solve, LinearSolvePolicy)
        or transform_solve.preconditioning is not None
    ):
        raise ValueError(
            "transform_preconditioner requires an unpreconditioned LinearSolvePolicy "
            "transform_solve in the assessment."
        )
    if not preconditioner.space.compatible(restriction.source):
        raise ValueError("transform_preconditioner must act on the solve space.")
    # The term's local setup block Rᵀ B R is assembled under the plan's
    # capacity-scaled limits, not the fixed defaults.
    term = SubspaceCorrectionTerm(
        transpose(restriction), restriction, preconditioner, assembly=assembly_policy
    )
    restricted_preconditioner = AdditiveSubspaceCorrectionPreconditioner(
        restricted, (term,)
    )
    return eqx.tree_at(
        lambda item: item.transform_solve,
        assessment,
        eqx.tree_at(
            lambda item: item.preconditioning,
            transform_solve,
            PreconditioningPolicy(restricted_preconditioner),
            is_leaf=lambda value: value is None,
        ),
    )


def _perturbation_bound(operator: AbstractLinearOperator, reference: Array, /) -> float:
    """``sqrt(‖ΔA‖₁ ‖ΔA‖∞) >= ‖ΔA‖₂`` on one unchanged sparse pattern."""
    if not isinstance(operator, AbstractSparseLinearOperator):
        raise RuntimeError("Assessed operators must be canonical sparse.")
    storage = operator.sparse_storage()
    difference = np.abs(np.asarray(storage.values) - np.asarray(reference))
    count = storage.shape[0]
    rows = np.repeat(np.arange(count), np.diff(np.asarray(storage.indptr)))
    row_sums = np.bincount(rows, weights=difference, minlength=count)
    column_sums = np.bincount(
        np.asarray(storage.indices), weights=difference, minlength=storage.shape[1]
    )
    return float(np.sqrt(np.max(row_sums) * np.max(column_sums)))


def _carried_evidence(
    spectrum: GeneralEigenSolveResult,
    bound: float,
    tolerance: GeneralEigenTolerancePolicy,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairs whose combined backward error ``ε + δ`` still meets the tolerance.

    A unit Ritz vector's residual scale ``‖A x‖ + |ρ|`` can shrink by at most
    ``δ`` under the perturbation, so the threshold uses ``scale - δ``.
    """
    diagnostics = spectrum.diagnostics
    residuals = np.asarray(diagnostics.right_residual_norms)
    relative = np.asarray(diagnostics.right_relative_residuals)
    scales = np.where(relative > 0, residuals / np.where(relative > 0, relative, 1), 0)
    errors = np.asarray(diagnostics.backward_errors) + bound
    threshold = tolerance.absolute + tolerance.relative * np.maximum(scales - bound, 0)
    return np.asarray(diagnostics.converged_mask) & (errors <= threshold), errors


def _reusable(
    spectrum: GeneralEigenSolveResult,
    carried: np.ndarray,
    errors: np.ndarray,
    self_adjoint: bool,
    /,
) -> bool:
    real = np.real(np.asarray(spectrum.eigenvalues))
    if np.any(carried & (real <= 0.0)):
        return True
    converged = np.asarray(spectrum.diagnostics.converged_mask)
    return (
        self_adjoint
        and spectrum.diagnostics.enclosure == "bauer-fike"
        and int(spectrum.status) == int(GeneralEigenSolveStatus.SUCCESS)
        and bool(np.all(carried == converged))
        and bool(np.all(real[carried] - errors[carried] > 0.0))
    )


def _spectral_assessment(
    solved: AbstractLinearOperator,
    space: ArraySpace,
    rows: np.ndarray,
    assessment: GeneralEigenSolvePolicy,
    assembly_policy: SparseAssemblyPolicy,
    problem_id: str,
    prior: PreparedCollocationAssessment | None,
    preconditioner: AbstractPreconditioner | None,
    refresh_policy: PointStabilityRefresh,
    /,
) -> tuple[
    PreparedCollocationAssessment | None,
    GeneralEigenSolveResult | None,
    str | None,
    float | None,
]:
    """Prepared state, spectrum, unassessed reason, and the reuse bound (None: fresh)."""
    if rows.size < 2:
        return None, None, f"only {rows.size} assessed row", None
    restriction = SparseCoordinateOperator(
        _selection(rows.size, space.shape[0], np.arange(rows.size), rows),
        jnp.ones((rows.size, 1), dtype=jnp.float64),
        source=space,
        target=ArraySpace((rows.size,)),
        operator_id=f"{problem_id}:stability-restriction",
    )
    restricted = restriction @ solved @ transpose(restriction)
    assembly = (
        prepare_sparse_assembly(
            plan_sparse_assembly(restricted, assembly_policy), restricted
        )
        if prior is None
        else refresh_sparse_assembly(prior.assembly, restricted)
    )
    match refresh_policy:
        case "reuse-within-perturbation":
            if prior is not None:
                bound = _perturbation_bound(assembly.operator, prior.reference)
                carried, errors = _carried_evidence(
                    prior.spectrum, bound, assessment.tolerance
                )
                self_adjoint = assembly.operator.properties.certifies("self_adjoint")
                if _reusable(prior.spectrum, carried, errors, self_adjoint):
                    reused = eqx.tree_at(lambda item: item.assembly, prior, assembly)
                    return reused, prior.spectrum, None, bound
        case "reassess":
            pass
        case _:
            assert_never(refresh_policy)
    problem = GeneralEigenproblem(assembly.operator, problem_id=f"{problem_id}:stability")
    if prior is None or preconditioner is not None:
        # An iterative transform binds the solve's current preconditioner, so it
        # is re-prepared (cheaply: no factorization) with the refreshed one.
        policy = _restricted_transform(
            _fitted_assessment(assessment, rows.size),
            restriction,
            assembly.operator,
            preconditioner,
            assembly_policy,
        )
        try:
            eigensolve = prepare_general_eigensolve(problem, policy)
        except LinearCapabilityError as error:
            # A resource refusal of the assessment is evidence, not a failure to
            # hide: it becomes the explicit stability-unassessed outcome.
            return None, None, str(error), None
        if prior is not None:
            eigensolve = refresh_general_eigensolve(
                eigensolve, problem, warm_start=prior.spectrum
            )
    else:
        eigensolve = refresh_general_eigensolve(
            prior.eigensolve, problem, warm_start=prior.spectrum
        )
    spectrum = general_eigensolve(eigensolve)
    if not isinstance(assembly.operator, AbstractSparseLinearOperator):
        raise RuntimeError("Assessed operators must be canonical sparse.")
    reference = assembly.operator.sparse_storage().values
    return (
        PreparedCollocationAssessment(assembly, eigensolve, spectrum, reference),
        spectrum,
        None,
        None,
    )


def _stability_outcome(
    spectrum: GeneralEigenSolveResult | None,
    reason: str | None,
    carried: tuple[np.ndarray, np.ndarray] | None,
    /,
) -> tuple[PointStabilityOutcome, str | None, int, int, float, float]:
    """Outcome, unassessed reason, converged/nonpositive counts, min Re, max backward error.

    ``carried`` (mask, combined errors) replaces the converged mask and backward
    errors of reused evidence.
    """
    if spectrum is None:
        return "stability-unassessed", reason, 0, 0, float("nan"), float("nan")
    diagnostics = spectrum.diagnostics
    if carried is None:
        converged = np.asarray(diagnostics.converged_mask)
        all_errors = np.asarray(diagnostics.backward_errors)
    else:
        converged, all_errors = carried
    real = np.real(np.asarray(spectrum.eigenvalues))[converged]
    errors = all_errors[converged]
    nonpositive = int(np.count_nonzero(real <= 0.0))
    status = GeneralEigenSolveStatus(int(spectrum.status))
    if nonpositive:
        outcome: PointStabilityOutcome = "nonpositive-real-part"
    elif status == GeneralEigenSolveStatus.SUCCESS:
        outcome = "admitted"
    else:
        outcome = "stability-unassessed"
        reason = (
            f"eigensolve status {status.name} with {real.size} of "
            f"{converged.size} eigenvalue estimates converged after "
            f"{int(diagnostics.arnoldi_action_count)} transformed actions "
            f"(factorization status {int(diagnostics.factorization_status)})"
        )
    return (
        outcome,
        reason,
        real.size,
        nonpositive,
        float(np.min(real)) if real.size else float("nan"),
        float(np.max(errors)) if errors.size else float("nan"),
    )


@checked
def assess_square_collocation(
    physical: AbstractLinearOperator,
    solved: AbstractLinearOperator,
    /,
    *,
    space: ArraySpace,
    eliminated: ArrayLike,
    bulk: ArrayLike,
    policy: PointStabilityPolicy,
    assessment: GeneralEigenSolvePolicy,
    assembly_policy: SparseAssemblyPolicy,
    problem_id: str,
    prior: PreparedCollocationAssessment | None = None,
    transform_preconditioner: AbstractPreconditioner | None = None,
    refresh: PointStabilityRefresh = "reassess",
) -> tuple[PreparedCollocationAssessment | None, PointCollocationStability]:
    """Spectral admission of one square point-collocation solve operator.

    ``solved`` is the native solve operator on ``space``; ``eliminated`` marks
    its decoupled identity coordinates (lifted Dirichlet and gauge rows), which
    are excluded from the assessment because they contribute the exact
    eigenvalue 1. ``physical`` and ``bulk`` (one flag per physical row) only feed
    the center-dominance diagnostics. ``prior`` reuses the symbolic restriction
    assembly and the prepared eigensolve of an earlier call on the same
    pattern. ``transform_preconditioner`` is the solve's own prepared
    approximate inverse of ``solved``; an iterative (``LinearSolvePolicy``)
    shift-invert transform is then preconditioned by its restriction to the
    assessed rows, which is an approximate inverse of the assessed block. Under
    ``policy="require-assessment"`` anything but ``"admitted"`` raises
    `PointCollocationStabilityRefusal`; ``"diagnostic"`` returns the evidence.
    With a ``prior``, ``refresh="reassess"`` reruns the eigensolve warm-started
    from the prior converged eigenvectors; ``"reuse-within-perturbation"``
    first tries to carry the prior evidence across the certified perturbation
    bound (see `PointCollocationStability`) and reassesses otherwise. Scalar and
    coupled (component-major) square collocation share this one admission.
    """
    policy_ = parse(policy, PointStabilityPolicy, "policy")
    refresh_ = parse(refresh, PointStabilityRefresh, "refresh")
    eliminated_ = np.asarray(eliminated)
    if eliminated_.shape != space.shape or eliminated_.dtype != np.bool_:
        raise ValueError("eliminated must hold one Boolean per solve coordinate.")
    if not problem_id:
        raise ValueError("problem_id must be non-empty.")
    rows = np.flatnonzero(~eliminated_)
    prepared, spectrum, reason, bound = _spectral_assessment(
        solved,
        space,
        rows,
        assessment,
        assembly_policy,
        problem_id,
        prior,
        transform_preconditioner,
        refresh_,
    )
    carried = (
        None
        if bound is None or spectrum is None
        else _carried_evidence(spectrum, bound, assessment.tolerance)
    )
    outcome, reason, converged, nonpositive, minimum, backward = _stability_outcome(
        spectrum, reason, carried
    )
    bulk_rows, nonpositive_diagonal, minimum_dominance, median_dominance = (
        _center_dominance(physical, jnp.asarray(bulk, dtype=jnp.bool_))
    )
    stability = PointCollocationStability(
        spectrum=spectrum,
        policy=policy_,
        outcome=outcome,
        assessed_rows=rows.size,
        converged_count=converged,
        nonpositive_count=nonpositive,
        minimum_real_part=minimum,
        maximum_backward_error=backward,
        unassessed_reason=reason,
        bulk_rows=bulk_rows,
        nonpositive_diagonal_rows=nonpositive_diagonal,
        minimum_center_dominance=minimum_dominance,
        median_center_dominance=median_dominance,
        reused=bound is not None,
        perturbation_bound=0.0 if bound is None else bound,
    )
    match policy_:
        case "require-assessment":
            if not stability.admitted:
                raise PointCollocationStabilityRefusal(stability)
        case "diagnostic":
            pass
        case _:
            assert_never(policy_)
    return prepared, stability


def _assess_square_operator(
    plan: PointCloudPoissonPlan,
    physical: AbstractLinearOperator,
    solved: AbstractLinearOperator,
    linear_solve: PreparedLinearSolve,
    prior: PreparedCollocationAssessment | None,
    /,
) -> tuple[PreparedCollocationAssessment | None, PointCollocationStability | None]:
    match plan.route:
        case "square-collocation" | "ghost-collocation":
            transform_solve = plan.stability_assessment.transform_solve
            state = linear_solve.preconditioning_state
            # An unpreconditioned iterative transform reuses the solve's own
            # prepared preconditioner; a declared one is used as declared.
            reused = (
                state.action
                if state is not None
                and isinstance(transform_solve, LinearSolvePolicy)
                and transform_solve.preconditioning is None
                else None
            )
            return assess_square_collocation(
                physical,
                solved,
                space=plan.solve_space,
                eliminated=_eliminated_rows(plan.layout, plan.gauges),
                bulk=plan.layout.bulk,
                policy=plan.stability,
                assessment=plan.stability_assessment,
                assembly_policy=plan.assembly_policy,
                problem_id=plan.plan_id,
                prior=prior,
                transform_preconditioner=reused,
                refresh=plan.stability_refresh,
            )
        case "oversampled-least-squares":
            return None, None
        case _:
            assert_never(plan.route)


@final
class PreparedPointCloudPoisson(StrictModule, NonTrainableState):
    """Reusable sparse assembly and native solve; preparation is host-side.

    Each floating component (no Dirichlet or positive-Robin row) carries one
    explicit point gauge. Compatibility is checked per component against the
    complete original algebraic equations. Projection changes each floating
    component's bulk source by a reported constant, determined from its
    removed equation—not a quadrature integral. Additional nullspaces cause
    native solve refusal.

    Collocated form targets the continuum product-rule equation. Dissipative
    form retains its energy-stable quadrature-adjoint algebraic contract with
    natural integrated boundary loads. Scientific success additionally requires
    the plan's explicit admitted ``sbp`` binding; algebraic success remains
    available through ``linear_result`` and the original-equation residuals.
    The oversampled route minimizes ``sum_t w_t r_t²`` with declared weights.
    """

    plan: PointCloudPoissonPlan
    diffusion: PointDiffusionOperator
    physical_assembly: PreparedSparseAssembly
    assembly: PreparedSparseAssembly
    linear_solve: PreparedLinearSolve
    row_weights: Array | None
    stability: PointCollocationStability | None
    spectral_assessment: PreparedCollocationAssessment | None

    @checked
    def __init__(
        self,
        plan: PointCloudPoissonPlan,
        diffusivity: ArrayLike | tuple[ArrayLike, ...] | list[ArrayLike] = 1.0,
        /,
    ) -> None:
        diffusion = _diffusion(plan, diffusivity)
        physical, solved, weights = _operators(plan, diffusion)
        physical_assembly = prepare_sparse_assembly(
            plan_sparse_assembly(physical, plan.assembly_policy), physical
        )
        assembly = (
            physical_assembly
            if solved is physical
            else prepare_sparse_assembly(
                plan_sparse_assembly(solved, plan.assembly_policy), solved
            )
        )
        linear_solve = prepare(
            _problem(plan, assembly.operator, weights), plan.linear_policy
        )
        spectral_assessment, stability = _assess_square_operator(
            plan, physical_assembly.operator, assembly.operator, linear_solve, None
        )
        self.plan = plan
        self.diffusion = diffusion
        self.physical_assembly = physical_assembly
        self.assembly = assembly
        self.linear_solve = linear_solve
        self.row_weights = weights
        self.stability = stability
        self.spectral_assessment = spectral_assessment

    @property
    def hierarchy(self) -> PreparedMeshfreeHierarchy | None:
        return self.plan.hierarchy

    def refresh(
        self, diffusivity: ArrayLike | tuple[ArrayLike, ...] | list[ArrayLike], /
    ) -> PreparedPointCloudPoisson:
        diffusion = _diffusion(self.plan, diffusivity)
        physical, solved, weights = _operators(self.plan, diffusion)
        physical_assembly = refresh_sparse_assembly(self.physical_assembly, physical)
        assembly = (
            physical_assembly
            if solved is physical
            else refresh_sparse_assembly(self.assembly, solved)
        )
        linear_solve = refresh(
            self.linear_solve, _problem(self.plan, assembly.operator, weights)
        )
        spectral_assessment, stability = _assess_square_operator(
            self.plan,
            physical_assembly.operator,
            assembly.operator,
            linear_solve,
            self.spectral_assessment,
        )
        return eqx.tree_at(
            lambda p: (
                p.diffusion,
                p.physical_assembly,
                p.assembly,
                p.linear_solve,
                p.stability,
                p.spectral_assessment,
            ),
            self,
            (
                diffusion,
                physical_assembly,
                assembly,
                linear_solve,
                stability,
                spectral_assessment,
            ),
            is_leaf=lambda value: value is None,
        )

    def physical_rhs(
        self,
        source: ArrayLike,
        /,
        *,
        boundary_values: Mapping[str, ArrayLike] | None = None,
    ) -> Array:
        """Original right-hand side: scaled bulk source, boundary and interface data.

        On the ghost route the cloud rows carry the source (Dirichlet rows
        their values) and ghost rows their boundary data divided by ``δ_b``.
        """
        layout = self.plan.layout
        source_ = jnp.asarray(source, dtype=jnp.float64)
        if source_.shape != (self.plan.boundary.row_count,):
            raise ValueError("Poisson source must have one value per equation row.")
        source_ = eqx.error_if(
            source_,
            jnp.any(~jnp.isfinite(source_)),
            "Poisson source and boundary values must be finite.",
        )
        values = self.plan.boundary.row_values(0, overrides=boundary_values)
        for interface in self.plan.interfaces:
            values = values.at[interface.rows].set(interface.flux_jump)
        scale = (
            self.plan.discretization.quadrature_weights
            if self.plan.form == "dissipative"
            else jnp.ones_like(source_)
        )
        ghosts = self.plan.ghosts
        if self.plan.form == "dissipative":
            rhs = scale * source_
            for condition in self.plan.boundary.conditions:
                if condition.kind in ("neumann", "robin"):
                    rhs = rhs.at[condition.rows].add(
                        _natural_boundary_measure(self.plan.discretization, condition)
                        * values[condition.rows]
                    )
            return jnp.where(layout.dirichlet, values, rhs)
        if ghosts is None:
            return jnp.where(layout.bulk, scale * source_, values)
        count = ghosts.cloud_count
        return jnp.concatenate(
            (
                jnp.where(layout.bulk[:count], source_, values),
                values[ghosts.plan.rows] / ghosts.offsets,
            )
        )

    def solve(
        self,
        source: ArrayLike,
        /,
        *,
        boundary_values: Mapping[str, ArrayLike] | None = None,
    ) -> PointCloudPoissonResult:
        physical_rhs = self.physical_rhs(source, boundary_values=boundary_values)
        return _compiled_solve(self, physical_rhs)

    def _solve_oversampled(self, physical_rhs: Array, /) -> PointCloudPoissonResult:
        weights = self.row_weights
        if weights is None:
            raise ValueError("Oversampled solves require declared row weights.")
        linear_result = solve(self.linear_solve, physical_rhs)
        solution = linear_result.value
        residual = self.physical_assembly.operator.mv(solution) - physical_rhs
        norm = jnp.sqrt(jnp.sum(weights * residual * residual))
        zero = jnp.zeros((0,), dtype=jnp.float64)
        return PointCloudPoissonResult(
            values=solution,
            residual_norm=norm,
            residual_tolerance=jnp.asarray(jnp.inf, dtype=jnp.float64),
            boundary_residual_norm=jnp.linalg.norm(
                jnp.where(self.plan.layout.bulk, 0.0, residual)
            ),
            compatible=jnp.asarray(True),
            compatibility_residual=jnp.asarray(0.0, dtype=jnp.float64),
            component_compatibility_residual=zero,
            source_correction=jnp.zeros_like(physical_rhs),
            gauge_residual=jnp.asarray(0.0, dtype=jnp.float64),
            linear_result=linear_result,
            correction_linear_result=None,
            diffusivity_evidence=self.diffusion.evidence,
            ghost_values=None,
            ghost_extension_defect=None,
            route=self.plan.route,
            continuum_consistent=True,
        )

    def _solve_square(self, physical_rhs: Array, /) -> PointCloudPoissonResult:
        plan = self.plan
        layout = plan.layout
        operator = self.physical_assembly.operator
        tolerance = plan.linear_policy.tolerance
        # The selected failure mode governs original-equation acceptance too:
        # "error" refuses by raising; "status" publishes successful=False.
        raising = plan.linear_policy.failure.mode == "error"
        threshold = tolerance.absolute + tolerance.relative * jnp.linalg.norm(
            physical_rhs
        )
        rhs = physical_rhs
        control = None
        dirichlet = any(c.kind == "dirichlet" for c in plan.boundary.conditions)
        if dirichlet:
            lift = jnp.where(layout.dirichlet, physical_rhs, 0.0)
            rhs = jnp.where(
                layout.dirichlet, physical_rhs, physical_rhs - operator.mv(lift)
            )
        if self.linear_solve.plan.backend == "native-krylov" and (
            dirichlet or plan.gauges
        ):
            # Stop against the requested original-equation tolerance: lifting
            # can amplify the algebraic RHS norm, and a gauge replaces a
            # physical row the Krylov solve never sees. With a constant left
            # kernel that row's residual is minus the sum of the others, so
            # ||r|| <= sqrt(n) ||r_solved|| and the stop is tightened by sqrt(n).
            stop = (
                threshold / np.sqrt(physical_rhs.shape[0]) if plan.gauges else threshold
            )
            control = LinearSolveControl(relative_tolerance=0.0, absolute_tolerance=stop)
        gauges = jnp.asarray(plan.gauges, dtype=jnp.int32)
        if plan.gauges:
            rhs = rhs.at[gauges].set(0.0)
        linear_result = solve(self.linear_solve, rhs, control=control)
        solution = linear_result.value
        residual_before = operator.mv(solution) - physical_rhs
        labels = plan.component_labels
        floating = labels[gauges] if plan.gauges else jnp.zeros((0,), jnp.int32)
        member = labels[None, :] == floating[:, None]
        component_residual = jnp.sqrt(
            jnp.sum(jnp.where(member, residual_before[None, :] ** 2, 0.0), axis=1)
        )
        compatibility_residual = jnp.linalg.norm(residual_before)
        compatible = compatibility_residual <= threshold
        correction = jnp.zeros_like(physical_rhs)
        correction_result = None
        if plan.gauges and plan.compatibility == "project":
            scale = (
                plan.discretization.quadrature_weights
                if plan.form == "dissipative"
                else jnp.ones_like(physical_rhs)
            )
            # Floating components are disconnected blocks of the gauged
            # operator, so one solve returns every component's response.
            source_rows = ~layout.dirichlet if plan.form == "dissipative" else layout.bulk
            direction = jnp.where(source_rows & jnp.any(member, axis=0), scale, 0.0)
            correction_result = solve(self.linear_solve, direction.at[gauges].set(0.0))
            response = correction_result.value
            denominator = (operator.mv(response) - direction)[gauges]
            degenerate = jnp.any(~jnp.isfinite(denominator)) | jnp.any(
                jnp.abs(denominator)
                <= jnp.finfo(jnp.float64).eps * jnp.linalg.norm(direction)
            )
            if raising:
                denominator = eqx.error_if(
                    denominator,
                    degenerate,
                    "Neumann source projection direction is algebraically incompatible with the left nullspace.",
                )
            # Under status failure a degenerate direction applies no shift; the
            # original residual check below then reports the unresolved equation.
            shifts = jnp.where(
                degenerate,
                0.0,
                -residual_before[gauges] / jnp.where(degenerate, 1.0, denominator),
            )
            shift_field = jnp.sum(jnp.where(member, shifts[:, None], 0.0), axis=0)
            correction = jnp.where(source_rows, shift_field, 0.0)
            solution = solution + shift_field * response
            physical_rhs = physical_rhs + scale * correction
        residual = operator.mv(solution) - physical_rhs
        norm = jnp.linalg.norm(residual)
        residual_tolerance = tolerance.absolute + tolerance.relative * jnp.linalg.norm(
            physical_rhs
        )
        if raising:
            solution = eqx.error_if(
                solution,
                ~jnp.isfinite(norm) | (norm > residual_tolerance),
                "Point Poisson refused: incompatible Neumann data or unresolved physical/boundary equations.",
            )
        gauge = (
            jnp.max(jnp.abs(solution[gauges]))
            if plan.gauges
            else jnp.asarray(0.0, dtype=jnp.float64)
        )
        return PointCloudPoissonResult(
            values=solution,
            residual_norm=norm,
            residual_tolerance=residual_tolerance,
            boundary_residual_norm=jnp.linalg.norm(jnp.where(layout.bulk, 0.0, residual)),
            compatible=compatible,
            compatibility_residual=compatibility_residual,
            component_compatibility_residual=component_residual,
            source_correction=correction,
            gauge_residual=gauge,
            linear_result=linear_result,
            correction_linear_result=correction_result,
            diffusivity_evidence=self.diffusion.evidence,
            ghost_values=None,
            ghost_extension_defect=None,
            route=plan.route,
            continuum_consistent=plan.continuum_consistent,
        )


def _solve(
    prepared: PreparedPointCloudPoisson, physical_rhs: Array, /
) -> PointCloudPoissonResult:
    match prepared.plan.route:
        case "square-collocation":
            return prepared._solve_square(physical_rhs)
        case "ghost-collocation":
            ghosts = prepared.plan.ghosts
            if ghosts is None:
                raise RuntimeError("The ghost route always carries its ghost layer.")
            result = prepared._solve_square(physical_rhs)
            extended = result.values
            count = ghosts.cloud_count
            return eqx.tree_at(
                lambda r: (
                    r.values,
                    r.source_correction,
                    r.ghost_values,
                    r.ghost_extension_defect,
                ),
                result,
                (
                    extended[:count],
                    result.source_correction[:count],
                    extended[count:],
                    ghosts.extension_defect(extended),
                ),
                is_leaf=lambda value: value is None,
            )
        case "oversampled-least-squares":
            return prepared._solve_oversampled(physical_rhs)
        case _:
            assert_never(prepared.plan.route)


# One stable compiled entry: the native Krylov loops are traced once per
# prepared structure instead of being re-traced by every eager solve call.
_compiled_solve = eqx.filter_jit(_solve)


def _diffusion(
    plan: PointCloudPoissonPlan,
    diffusivity: ArrayLike | tuple[ArrayLike, ...] | list[ArrayLike],
    /,
) -> PointDiffusionOperator:
    return PointDiffusionOperator(
        plan.discretization,
        diffusivity,
        form=plan.form,
        kind=plan.diffusivity_kind,
        sides=plan.sides,
        sbp=plan.sbp,
    )


def _operators(
    plan: PointCloudPoissonPlan, diffusion: PointDiffusionOperator, /
) -> tuple[AbstractLinearOperator, AbstractLinearOperator, Array | None]:
    match plan.route:
        case "square-collocation":
            physical, solved = _square_operators(plan, diffusion)
            return physical, solved, None
        case "ghost-collocation":
            physical, solved = _ghost_operators(plan, diffusion)
            return physical, solved, None
        case "oversampled-least-squares":
            collocation = plan.collocation
            if collocation is None:
                raise ValueError("Oversampled route requires collocation targets.")
            operator = _oversampled_operator(plan, diffusion)
            weights = _least_squares_weights(
                plan.boundary, collocation, collocation.target_count
            )
            return operator, operator, weights
        case _:
            assert_never(plan.route)


def _problem(
    plan: PointCloudPoissonPlan,
    operator: AbstractLinearOperator,
    weights: Array | None,
    /,
) -> LinearSystem | LeastSquaresProblem:
    match plan.route:
        case "square-collocation" | "ghost-collocation":
            return LinearSystem(operator, problem_id=plan.plan_id)
        case "oversampled-least-squares":
            return LeastSquaresProblem(operator, weights=weights, problem_id=plan.plan_id)
        case _:
            assert_never(plan.route)


__all__ = [
    "PointBoundaryCondition",
    "PointBoundaryKind",
    "PointBoundaryPlan",
    "PointCloudPoissonPlan",
    "PointCloudPoissonResult",
    "PointCollocationPlan",
    "PointCollocationStability",
    "PointCollocationStabilityRefusal",
    "PointDiffusionForm",
    "PointDiffusionOperator",
    "PointDiffusivityEvidence",
    "PointDiffusivityKind",
    "PointNeumannCompatibility",
    "PointPoissonRoute",
    "PointStabilityOutcome",
    "PointStabilityPolicy",
    "PointStabilityRefresh",
    "PointSBPReport",
    "PreparedCollocationAssessment",
    "PreparedPointCloudPoisson",
    "PreparedPointCollocation",
    "assess_square_collocation",
    "collocation_stability_assessment",
    "point_sbp_report",
]
