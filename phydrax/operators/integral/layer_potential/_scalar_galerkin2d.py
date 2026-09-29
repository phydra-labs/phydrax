#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualified native Galerkin boundary operator for 2-D scalar Laplace problems.

The product lives on a closed simple polygon with straight panels. Its trace
spaces are the continuous piecewise-linear (P1) Dirichlet trace and the
piecewise-constant (DP0) conormal trace, paired through physical arc length.
It prepares the weak single layer ``V`` (DP0 x DP0), the weak double layer
``K`` (DP0 test x P1 trial), the mixed, P1 and DP0 mass maps, and the exact
panel-length functional ``m``. The kernel is ``G = -log|x - y|/(2π)`` and every
normal points from the bounded interior to the unbounded exterior.

A bounded exterior harmonic field satisfies ``u = c_inf + D φ - S q`` with
``∫ q ds = 0``; its exterior trace gives the weak relation
``(M/2 - K) φ + V q - m c_inf = 0``. The square bordered exterior
Dirichlet-to-Neumann system couples that relation to the zero-total-conormal
row with the far-field constant as an explicit unknown.
"""

from __future__ import annotations

import math
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ...._admissibility import guard_derivative_validity, refuse_derivative_dependencies
from ...._differentiation import (
    DerivativeRoute,
    DerivativeSurface,
    OwnerDerivativeCapability,
)
from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ...._tree_math import uncancelled_direction
from ...._validation import canonical_identifier, positive_integer
from ....discretization._boundary_trace_space import (
    boundary_geometry_revision,
    BoundaryTraceSpaceCapability,
    CauchyTraceCapability,
)
from ....geometry._polygon import signed_area2
from ....linalg import (
    AbstractLinearOperator,
    ArraySpace,
    BlockLinearOperator,
    BlockSpace,
    DenseLinearOperator,
    DiagonalPairing,
    DifferentiationMode,
    DifferentiationPolicy,
    DualSpace,
    FailurePolicy,
    GMRES,
    JacobiPreconditionerBuilder,
    LinearCapabilityError,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorCapabilities,
    OperatorPairing,
    OperatorProperties,
    PCG,
    PreconditioningPolicy,
    prepare,
    PreparedLinearSolve,
    solve,
    TolerancePolicy,
)
from ....linalg._operators import _AbstractCostedLinearOperator
from ....sparse import EdgeRelation, SparseCoordinateOperator
from ....typing import (
    AnyDim,
    as_array,
    as_host_array,
    Bool,
    ConvertibleToArray,
    Dim,
    Float64,
    HostFloat64,
    Identifier,
    Int32,
    Int64,
    parse,
    Scalar,
    Size,
)
from ._galerkin_quadrature2d import (
    EXCEPTION_ENTRY_BYTES,
    PAIR_CLASS_METHODS_2D,
    PAIR_CLASS_NAMES_2D,
    PanelGeometry2D,
    PanelLayerKind2D,
    PanelPairData2D,
    prepare_panel_pairs_2d,
    regular_pair_values,
    regular_panel_rule,
    segment_distances,
    segment_layer_moments,
)
from ._laplace2d import LaplaceLayerKernel2D
from ._qualification import BoundarySupportEnvelope


ScalarTraceSide2D: TypeAlias = Literal["interior", "exterior"]
CurveTraversal2D: TypeAlias = Literal["counterclockwise", "clockwise"]
BoundaryTraceRepresentation2D: TypeAlias = Literal["continuous-p1", "dp0"]
ExteriorFarField2D: TypeAlias = Literal["bounded", "decaying"]

_EPSILON = float(np.finfo(np.float64).eps)
_FLOAT_BYTES = np.dtype(np.float64).itemsize
_FIELD_CHUNK = 64
_UNSUPPORTED_CLAIMS = {
    "continuum-error": "No continuum discretization-error estimator is implemented.",
    "open-curves": "Only closed curves with a bounded interior are prepared.",
    "multiply-connected-boundaries": "Exactly one closed curve is prepared.",
    "curved-geometry": "Panels are exact straight segments; curved charts are refused.",
    "moving-geometry": "Geometry is fixed at preparation; no motion route exists.",
    "helmholtz": "Only the Laplace kernel -log(r)/(2*pi) is assembled.",
    "hypersingular-operator": "The hypersingular map W is not assembled in 2-D.",
    "fast-multipole-galerkin-action": "Only the blocked-direct Galerkin action exists.",
    "geometry-derivatives": (
        "Curve vertices, panels, pair classes, and singular corrections are fixed at "
        "host preparation; derivatives admit densities and Dirichlet data only."
    ),
    "field-target-derivatives": (
        "Field-evaluation target positions are not differentiable: the closed-form "
        "panel moments mask panel-line coordinates."
    ),
    "undeclared-far-field": "Exterior solves require a declared far-field condition.",
}
_SUPPORTED_CLAIMS = (
    "finite-execution",
    "weak-single-layer-dp0-dp0",
    "weak-double-layer-dp0-p1",
    "arc-length-trace-pairings",
    "operator-transpose-and-adjoint",
    "quadrature-error-bound",
    "bordered-bounded-exterior-dirichlet-to-neumann",
    "declared-l2-trace-projection",
    "exact-straight-panel-field-evaluation",
    "explicit-policy-dense-materialization",
    "fixed-geometry-density-derivatives",
    "dirichlet-data-implicit-derivative",
)
_FIXED_STRUCTURE_REFUSALS = {
    "geometry": (
        "curve vertices, panels, normals, and lengths are fixed at host preparation; "
        "pair classes and singular corrections are prepared once and never traced"
    ),
    "kernel": "the Laplace kernel -log(r)/(2*pi) is fixed and has no parameters",
    "quadrature": (
        "regular rules, pair classification, and adaptive singular corrections are "
        "host-prepared and not differentiable"
    ),
}
_TARGET_REFUSAL = (
    "evaluate_field target positions are not qualified: the closed-form "
    "straight-panel moments mask panel-line coordinates (a target on the extension "
    "line of a panel), where the traced derivative is not the field gradient"
)


def _support_envelope() -> BoundarySupportEnvelope:
    """The exact candidate support tuple of this product; it is not a qualification."""
    return BoundarySupportEnvelope(
        geometry_id="closed-simple-straight-panel-polygon-2d",
        trace_id="continuous-p1-dirichlet+dp0-conormal",
        pde_formulation_id="laplace-weak-v-k+bordered-bounded-exterior-dtn",
        provider_id="native-blocked-direct-2d",
        precision_id="float64-accumulate-float64",
        differentiation_id="rhs-only+fixed-geometry-linear-density-actions",
        platform_id="jax-cpu",
        claims=(*_SUPPORTED_CLAIMS, *_UNSUPPORTED_CLAIMS),
        unsupported_claims=_UNSUPPORTED_CLAIMS,
        stop_ship_conditions=(
            "quadrature-tolerance-exceeded",
            "resource-preflight-failed",
            "exterior-solve-not-accepted",
        ),
    )


class _VertexDim(Dim, minimum=3):
    """Closed-curve vertices, the continuous P1 coefficients."""


class _PanelDim(Dim, minimum=3):
    """Straight panels, the DP0 coefficients."""


class _CoefficientDim(Dim, minimum=3):
    """Coefficients of one boundary trace space."""


class _TargetDim(Dim, minimum=1):
    """Field-evaluation targets."""


class _SampleDim(Dim, minimum=1):
    """Trace samples on each panel."""


def _curve_geometry(
    vertices: np.ndarray, /
) -> tuple[PanelGeometry2D, CurveTraversal2D, float]:
    successors = np.roll(vertices, -1, axis=0)
    edges = successors - vertices
    lengths = np.linalg.norm(edges, axis=1)
    scale = max(float(np.max(np.abs(vertices))), float(np.max(lengths)), 1.0e-300)
    tolerance = 64.0 * _EPSILON * scale
    if np.any(lengths <= tolerance):
        raise ValueError("Closed polygonal curves cannot contain zero-length panels.")
    area = signed_area2(vertices)
    if abs(area) <= 128.0 * _EPSILON * scale * scale:
        raise ValueError("Closed polygonal curves must enclose a positive area.")
    tangents = edges / lengths[:, None]
    orientation = 1.0 if area > 0.0 else -1.0
    normals = orientation * np.stack((tangents[:, 1], -tangents[:, 0]), axis=1)
    traversal: CurveTraversal2D = "counterclockwise" if area > 0.0 else "clockwise"
    return PanelGeometry2D(vertices, tangents, normals, lengths), traversal, tolerance


def _validate_simple_curve(geometry: PanelGeometry2D, tolerance: float, /) -> None:
    """Refuse fold-back corners and touching or crossing non-adjacent panels."""
    count = geometry.lengths.shape[0]
    previous = np.roll(geometry.tangents, 1, axis=0)
    if np.any(-np.sum(previous * geometry.tangents, axis=1) >= 1.0 - 64.0 * _EPSILON):
        raise ValueError("Closed polygonal curves cannot fold back at a vertex.")
    ends = geometry.starts + geometry.lengths[:, None] * geometry.tangents
    columns = np.arange(count)
    rows_per_block = max(1, (1 << 22) // count)
    for first in range(0, count, rows_per_block):
        rows = np.arange(first, min(first + rows_per_block, count))
        distances = segment_distances(
            geometry.starts[rows][:, None, :],
            ends[rows][:, None, :],
            geometry.starts[None, :, :],
            ends[None, :, :],
        )
        offset = (columns[None, :] - rows[:, None]) % count
        separated = (offset > 1) & (offset < count - 1)
        if np.any(separated & (distances <= tolerance)):
            raise ValueError(
                "Closed polygonal curves must be simple; non-adjacent panels touch or cross."
            )


class ClosedPolygonalCurve2D(StrictModule, NonTrainableState):
    """Closed simple polygon with interior-to-exterior panel normals.

    Vertices are the continuous P1 coefficients in declared order; panel ``k``
    joins vertex ``k`` to vertex ``k + 1`` cyclically and carries one DP0
    coefficient. Either traversal is accepted without reordering, and every
    normal points from the bounded interior to the unbounded exterior. Panels
    are exact straight segments, so no curved-boundary approximation is made or
    claimed. A Nyström ``BoundaryPanelization2D`` on arbitrary charts is not a
    polygonal Galerkin support; the declaring geometry identity enters through
    ``source_id``.
    """

    __strict_contract__ = True

    vertices: Float64[_VertexDim, Literal[2]]
    panel_vertices: Int32[_PanelDim, Literal[2]]
    lengths: Float64[_PanelDim]
    tangents: Float64[_PanelDim, Literal[2]]
    normals: Float64[_PanelDim, Literal[2]]
    traversal: CurveTraversal2D = eqx.field(static=True)
    source_id: Identifier = eqx.field(static=True)
    vertex_count: Size[_VertexDim] = eqx.field(static=True)
    panel_count: Size[_PanelDim] = eqx.field(static=True)
    curve_id: Identifier = eqx.field(static=True)

    def __init__(self, vertices: ConvertibleToArray, /, *, source_id: str) -> None:
        source = canonical_identifier(source_id, "source_id")
        host = as_host_array(vertices, HostFloat64[AnyDim, Literal[2]], "vertices")
        if host.shape[0] < 3:
            raise ValueError("A closed polygonal curve requires at least three vertices.")
        if not np.all(np.isfinite(host)):
            raise ValueError("Closed polygonal curve vertices must be finite.")
        geometry, traversal, tolerance = _curve_geometry(host)
        _validate_simple_curve(geometry, tolerance)
        count = host.shape[0]
        indices = np.arange(count, dtype=np.int32)
        vertices_ = jnp.asarray(host, dtype=jnp.float64)
        self.vertices = vertices_
        self.panel_vertices = jnp.asarray(
            np.stack((indices, np.roll(indices, -1)), axis=1), dtype=jnp.int32
        )
        self.lengths = jnp.asarray(geometry.lengths, dtype=jnp.float64)
        self.tangents = jnp.asarray(geometry.tangents, dtype=jnp.float64)
        self.normals = jnp.asarray(geometry.normals, dtype=jnp.float64)
        self.traversal = traversal
        self.source_id = source
        self.vertex_count = count
        self.panel_count = count
        self.curve_id = canonical_fingerprint(
            {
                "kind": "closed-polygonal-curve-2d",
                "source_id": source,
                "vertices": array_tree_fingerprint(vertices_),
            }
        )


def _host_geometry(curve: ClosedPolygonalCurve2D, /) -> PanelGeometry2D:
    return PanelGeometry2D(
        starts=np.asarray(curve.vertices, dtype=np.float64),
        tangents=np.asarray(curve.tangents, dtype=np.float64),
        normals=np.asarray(curve.normals, dtype=np.float64),
        lengths=np.asarray(curve.lengths, dtype=np.float64),
    )


class ScalarTraceConvention2D(StrictModule, NonTrainableState):
    """Outward-normal trace, jump, and far-field convention for a closed curve.

    ``interior`` is the bounded side and ``exterior`` the unbounded side. The
    normal points from interior to exterior, ``gamma0`` is the Dirichlet trace
    and ``gamma1`` differentiates along that normal on both sides. With
    ``G = -log|x - y|/(2π)`` and ``D`` using the source normal,
    ``gamma0^± D = K ± I/2`` and ``gamma1^± S = K' ∓ I/2``. A bounded exterior
    harmonic field is ``u = c_inf + D(gamma0^+ u) - S(gamma1^+ u)`` with
    ``∫ gamma1^+ u ds = 0``, so its exterior Dirichlet trace satisfies
    ``(I/2 - K) gamma0^+ u + V gamma1^+ u - c_inf = 0``. An interior harmonic
    field is ``u = S(gamma1^- u) - D(gamma0^- u)``, whose trace satisfies
    ``(I/2 + K) gamma0^- u - V gamma1^- u = 0``; the two relations differ.
    """

    ambient_dimension: int = eqx.field(static=True)
    boundary_dimension: int = eqx.field(static=True)
    interior: str = eqx.field(static=True)
    exterior: str = eqx.field(static=True)
    normal_orientation: str = eqx.field(static=True)
    dirichlet_trace: str = eqx.field(static=True)
    conormal_trace: str = eqx.field(static=True)
    fundamental_solution: str = eqx.field(static=True)
    exterior_representation: str = eqx.field(static=True)
    interior_representation: str = eqx.field(static=True)
    far_field: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)

    def __init__(self) -> None:
        self.ambient_dimension = 2
        self.boundary_dimension = 1
        self.interior = "bounded-side"
        self.exterior = "unbounded-side"
        self.normal_orientation = "interior-to-exterior"
        self.dirichlet_trace = "gamma0:u"
        self.conormal_trace = "gamma1:outward-normal-derivative"
        self.fundamental_solution = "-log(r)/(2*pi)"
        self.exterior_representation = "u=c_inf+D(gamma0^+u)-S(gamma1^+u)"
        self.interior_representation = "u=S(gamma1^-u)-D(gamma0^-u)"
        self.far_field = "bounded-iff-integral(gamma1^+u)=0;u->c_inf+O(1/|x|)"
        self.convention_id = canonical_fingerprint(
            {
                "kind": "closed-scalar-trace-convention-2d",
                "normal": self.normal_orientation,
                "fundamental_solution": self.fundamental_solution,
                "gamma0_D": {"interior": -0.5, "exterior": 0.5},
                "gamma1_S": {"interior": 0.5, "exterior": -0.5},
                "exterior": self.exterior_representation,
                "interior": self.interior_representation,
                "far_field": self.far_field,
            }
        )

    def double_layer_dirichlet_jump(self, side: ScalarTraceSide2D, /) -> float:
        """Return the identity coefficient in ``gamma0 D`` on ``side``."""
        match parse(side, ScalarTraceSide2D, "side"):
            case "interior":
                return -0.5
            case "exterior":
                return 0.5
            case invalid:
                assert_never(invalid)

    def single_layer_neumann_jump(self, side: ScalarTraceSide2D, /) -> float:
        """Return the identity coefficient in ``gamma1 S`` on ``side``."""
        match parse(side, ScalarTraceSide2D, "side"):
            case "interior":
                return 0.5
            case "exterior":
                return -0.5
            case invalid:
                assert_never(invalid)


class BoundaryTraceSpace2D(StrictModule, NonTrainableState):
    """Coefficient space of one boundary trace with its arc-length Gram map.

    ``mass`` maps coefficients to their dual, ``(M u)_i = ∫ u v_i ds`` for basis
    functions ``v_i``, and ``integral_weights`` is the covector ``∫ v_i ds``.
    ``riesz_solve`` is the prepared native inverse of a non-diagonal Gram map
    (continuous P1); it is ``None`` for DP0, whose Gram map is the diagonal of
    panel lengths.
    """

    __strict_contract__ = True

    representation: BoundaryTraceRepresentation2D = eqx.field(static=True)
    quantity: str = eqx.field(static=True)
    sobolev_conformity: str = eqx.field(static=True)
    vector_space: ArraySpace
    mass: AbstractLinearOperator
    integral_weights: Float64[_CoefficientDim]
    riesz_solve: PreparedLinearSolve | None
    dimension: Size[_CoefficientDim] = eqx.field(static=True)


class ScalarBoundarySpaces2D(StrictModule, NonTrainableState):
    """Canonical P1 Dirichlet and DP0 conormal traces of one closed polygon.

    ``mixed_mass`` maps P1 coefficients into the DP0 dual,
    ``(M φ)_i = ∫_{panel i} φ ds``; it is the physical duality pairing between
    the two trace spaces. ``far_field_space`` carries the exterior far-field
    constant of the bordered exterior formulation.
    """

    curve: ClosedPolygonalCurve2D
    dirichlet_trace: BoundaryTraceSpace2D
    conormal_trace: BoundaryTraceSpace2D
    mixed_mass: AbstractLinearOperator
    far_field_space: ArraySpace
    spaces_id: str = eqx.field(static=True)

    def pair(self, conormal: ArrayLike, dirichlet: ArrayLike, /) -> Array:
        """Physical duality ``∫ q φ ds`` of conormal and Dirichlet coefficients."""
        values = self.conormal_trace.vector_space.validate(conormal)
        return jnp.dot(values, self.mixed_mass.mv(dirichlet))

    def cauchy_trace_capability(self) -> CauchyTraceCapability:
        """Publish the P1 Dirichlet / DP0 conormal Cauchy data of this curve.

        The conormal trace is the Neumann quantity along the interior-to-exterior
        normal of the bounded interior, whichever traversal was declared; both
        native spaces already carry their arc-length Gram pairings and
        ``mixed_mass`` is the duality.
        """
        convention = ScalarTraceConvention2D()
        revision = boundary_geometry_revision(
            self.curve.vertices, self.curve.panel_vertices
        )
        dirichlet = self.dirichlet_trace.vector_space
        conormal = self.conormal_trace.vector_space
        return CauchyTraceCapability(
            BoundaryTraceSpaceCapability(
                owner_id=self.spaces_id,
                quantity="dirichlet",
                representation="continuous-p1",
                coefficient_space=dirichlet,
                gram_space=dirichlet,
                mass=self.dirichlet_trace.mass,
                ambient_dimension=2,
                revision_id=revision,
            ),
            BoundaryTraceSpaceCapability(
                owner_id=self.spaces_id,
                quantity="neumann",
                representation="dp0",
                coefficient_space=conormal,
                gram_space=conormal,
                mass=self.conormal_trace.mass,
                ambient_dimension=2,
                revision_id=revision,
            ),
            self.mixed_mass,
            interior=convention.interior,
            convention_id=convention.convention_id,
        )


def _gram_policy(tolerance: float, size: int, /) -> LinearSolvePolicy:
    return LinearSolvePolicy(
        PCG(),
        tolerance=TolerancePolicy(relative=tolerance, absolute=0.0, max_steps=4 * size),
        preconditioning=PreconditioningPolicy(JacobiPreconditionerBuilder()),
        differentiation=DifferentiationPolicy("rhs-only"),
        failure=FailurePolicy("status"),
    )


def _p1_trace_space(
    curve: ClosedPolygonalCurve2D, tolerance: float, /
) -> BoundaryTraceSpace2D:
    lengths = np.asarray(curve.lengths)
    first = np.asarray(curve.panel_vertices[:, 0])
    second = np.asarray(curve.panel_vertices[:, 1])
    count = curve.vertex_count
    relation = EdgeRelation(
        np.concatenate((first, second, first, second)),
        np.concatenate((first, first, second, second)),
        source_size=count,
        target_size=count,
    )
    values = jnp.asarray(
        np.concatenate((lengths / 3.0, lengths / 6.0, lengths / 6.0, lengths / 3.0)),
        dtype=jnp.float64,
    )
    space_id = canonical_fingerprint(
        {"kind": "closed-curve-continuous-p1-trace-2d", "curve": curve.curve_id}
    )
    coordinates = ArraySpace(
        (count,), dtype=np.float64, space_id=f"{space_id}:coordinates"
    )
    gram = SparseCoordinateOperator(
        relation,
        values,
        source=coordinates,
        target=coordinates,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
        operator_id=f"{space_id}:gram",
    )
    riesz_solve = prepare(
        LinearSystem(gram, problem_id=f"{space_id}:gram-system"),
        _gram_policy(tolerance, count),
    )
    vector_space = ArraySpace(
        (count,),
        dtype=np.float64,
        pairing=OperatorPairing(gram, prepared_inverse=riesz_solve),
        space_id=space_id,
    )
    return BoundaryTraceSpace2D(
        representation="continuous-p1",
        quantity="dirichlet-trace",
        sobolev_conformity="H^(1/2)(Gamma)",
        vector_space=vector_space,
        mass=SparseCoordinateOperator(
            relation,
            values,
            source=vector_space,
            target=DualSpace(vector_space),
            operator_id=f"{space_id}:mass",
        ),
        integral_weights=jnp.asarray(0.5 * (lengths + np.roll(lengths, 1))),
        riesz_solve=riesz_solve,
        dimension=count,
    )


def _dp0_trace_space(curve: ClosedPolygonalCurve2D, /) -> BoundaryTraceSpace2D:
    count = curve.panel_count
    panels = np.arange(count)
    space_id = canonical_fingerprint(
        {"kind": "closed-curve-dp0-trace-2d", "curve": curve.curve_id}
    )
    vector_space = ArraySpace(
        (count,),
        dtype=np.float64,
        pairing=DiagonalPairing(curve.lengths, pairing_id=f"{space_id}:arc-length"),
        space_id=space_id,
    )
    return BoundaryTraceSpace2D(
        representation="dp0",
        quantity="conormal-trace",
        sobolev_conformity="H^(-1/2)(Gamma)",
        vector_space=vector_space,
        mass=SparseCoordinateOperator(
            EdgeRelation(panels, panels, source_size=count, target_size=count),
            curve.lengths,
            source=vector_space,
            target=DualSpace(vector_space),
            operator_id=f"{space_id}:mass",
        ),
        integral_weights=curve.lengths,
        riesz_solve=None,
        dimension=count,
    )


def _prepare_boundary_spaces(
    curve: ClosedPolygonalCurve2D, gram_tolerance: float, /
) -> ScalarBoundarySpaces2D:
    dirichlet = _p1_trace_space(curve, gram_tolerance)
    conormal = _dp0_trace_space(curve)
    count = curve.panel_count
    panels = np.arange(count)
    half = 0.5 * np.asarray(curve.lengths)
    mixed_id = canonical_fingerprint(
        {
            "kind": "closed-curve-mixed-mass-2d",
            "dirichlet": dirichlet.vector_space.space_id,
            "conormal": conormal.vector_space.space_id,
        }
    )
    mixed = SparseCoordinateOperator(
        EdgeRelation(
            np.asarray(curve.panel_vertices).T.reshape((-1,)),
            np.concatenate((panels, panels)),
            source_size=curve.vertex_count,
            target_size=count,
        ),
        jnp.asarray(np.concatenate((half, half)), dtype=jnp.float64),
        source=dirichlet.vector_space,
        target=DualSpace(conormal.vector_space),
        operator_id=mixed_id,
    )
    far_field = ArraySpace(
        (1,),
        dtype=np.float64,
        space_id=canonical_fingerprint(
            {"kind": "exterior-far-field-constant-2d", "curve": curve.curve_id}
        ),
    )
    return ScalarBoundarySpaces2D(
        curve=curve,
        dirichlet_trace=dirichlet,
        conormal_trace=conormal,
        mixed_mass=mixed,
        far_field_space=far_field,
        spaces_id=canonical_fingerprint(
            {
                "kind": "closed-curve-scalar-boundary-spaces-2d",
                "dirichlet": dirichlet.vector_space.space_id,
                "conormal": conormal.vector_space.space_id,
                "mixed": mixed_id,
                "far_field": far_field.space_id,
            }
        ),
    )


def _positive_float(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


class ScalarLaplaceGalerkinPolicy2D(StrictModule, NonTrainableState):
    """Accuracy, blocking, and resource policy for the 2-D Laplace Galerkin operator.

    ``tolerance`` bounds the estimated quadrature error of every panel-pair
    entry, normalized by the panel measures: ``V_ij`` by ``h_i h_j`` and each
    ``K`` hat moment by ``h_i`` (both are then panel-averaged kernel values).
    Regular pairs use the order-``regular_order`` tensor Gauss--Legendre rule
    at run time when they are at least ``near_ratio`` times the larger panel
    length apart and the certification sweep meets ``tolerance``.
    """

    regular_order: int = eqx.field(static=True)
    near_ratio: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    max_depth: int = eqx.field(static=True)
    max_adaptive_intervals: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    max_exception_pairs: int = eqx.field(static=True)
    max_preparation_workspace_bytes: int = eqx.field(static=True)
    max_resident_bytes: int = eqx.field(static=True)
    gram_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        regular_order: int = 6,
        near_ratio: float = 2.0,
        tolerance: float = 1.0e-10,
        max_depth: int = 48,
        max_adaptive_intervals: int = 2_000_000,
        block_size: int = 32,
        max_exception_pairs: int = 4_000_000,
        max_preparation_workspace_bytes: int = 128 * 1024 * 1024,
        max_resident_bytes: int = 256 * 1024 * 1024,
        gram_tolerance: float = 1.0e-13,
    ) -> None:
        order = positive_integer(regular_order, "regular_order")
        if order < 2:
            raise ValueError("regular_order must be at least two.")
        ratio = _positive_float(near_ratio, "near_ratio")
        accuracy = _positive_float(tolerance, "tolerance")
        gram = _positive_float(gram_tolerance, "gram_tolerance")
        if gram >= 1.0:
            raise ValueError("gram_tolerance must be smaller than one.")
        limits = (
            positive_integer(max_depth, "max_depth"),
            positive_integer(max_adaptive_intervals, "max_adaptive_intervals"),
            positive_integer(block_size, "block_size"),
            positive_integer(max_exception_pairs, "max_exception_pairs"),
            positive_integer(
                max_preparation_workspace_bytes, "max_preparation_workspace_bytes"
            ),
            positive_integer(max_resident_bytes, "max_resident_bytes"),
        )
        self.regular_order = order
        self.near_ratio = ratio
        self.tolerance = accuracy
        (
            self.max_depth,
            self.max_adaptive_intervals,
            self.block_size,
            self.max_exception_pairs,
            self.max_preparation_workspace_bytes,
            self.max_resident_bytes,
        ) = limits
        self.gram_tolerance = gram
        self.policy_id = canonical_fingerprint(
            {
                "kind": "scalar-laplace-galerkin-policy-2d",
                "regular_order": order,
                "near_ratio": ratio,
                "tolerance": accuracy,
                "limits": list(limits),
                "gram_tolerance": gram,
            }
        )


class ScalarLaplaceGalerkinReport2D(StrictModule, NonTrainableState):
    """Pair classification, quadrature error, work, and resource evidence.

    Classes are ``coincident``, ``shared-endpoint``, ``near`` (including regular
    pairs promoted by the certification sweep), and ``regular``. Errors are the
    normalized estimates described by the policy; ``evaluations`` counts kernel
    or closed-form inner-integral evaluations per class. ``support`` is the exact
    candidate support tuple with its explicit unsupported claims; it is not a
    qualification. No continuum discretization error is estimated.
    """

    __strict_contract__ = True

    curve_id: Identifier = eqx.field(static=True)
    policy_id: Identifier = eqx.field(static=True)
    kernel_id: Identifier = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    fundamental_solution: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    panel_count: int = eqx.field(static=True)
    vertex_count: int = eqx.field(static=True)
    pair_class_names: tuple[str, str, str, str] = eqx.field(static=True)
    pair_class_methods: tuple[str, str, str, str] = eqx.field(static=True)
    pair_counts: tuple[int, int, int, int] = eqx.field(static=True)
    promoted_pair_count: int = eqx.field(static=True)
    maximum_errors: Float64[Literal[4]]
    tolerance: float = eqx.field(static=True)
    pair_class_supported: Bool[Literal[4]]
    evaluations: Int64[Literal[4]]
    maximum_adaptive_depth: int = eqx.field(static=True)
    exception_count: int = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    action_workspace_bytes_per_rhs: int = eqx.field(static=True)
    materializable: bool = eqx.field(static=True)
    continuum_discretization_error_estimated: bool = eqx.field(static=True)
    finite: Bool[Scalar]
    accuracy_supported: Bool[Scalar]
    support: BoundarySupportEnvelope
    report_id: Identifier = eqx.field(static=True)


class _BlockedLayerOperator2D(_AbstractCostedLinearOperator):
    """Blocked-direct weak V or K action with sparse non-regular corrections.

    Regular pairs are summed block by block with the prepared tensor rule;
    exception pairs are masked there and applied from their stored values. No
    panel-pair tensor is retained between blocks.
    """

    points: Array
    weights: Array
    normals: Array
    valid: Array
    basis: Array
    exception_keys: Array
    panel_vertices: Array
    correction: SparseCoordinateOperator
    layer: PanelLayerKind2D = eqx.field(static=True)
    panel_count: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    action_workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        layer: PanelLayerKind2D,
        curve: ClosedPolygonalCurve2D,
        pairs: PanelPairData2D,
        rule: tuple[np.ndarray, np.ndarray, np.ndarray],
        /,
        *,
        source: ArraySpace,
        target: DualSpace,
        block_size: int,
        operator_id: str,
    ) -> None:
        points, weights, hats = rule
        count = curve.panel_count
        padded = -(-count // block_size) * block_size
        padding = padded - count
        order = points.shape[1]
        match layer:
            case "single":
                basis = np.ones((order, 1), dtype=np.float64)
                correction = _single_correction(pairs, count, source, target)
            case "double":
                basis = hats
                correction = _double_correction(pairs, curve, source, target)
            case _:
                assert_never(layer)
        self.points = jnp.asarray(
            np.pad(points, ((0, padding), (0, 0), (0, 0)), mode="edge")
        )
        self.weights = jnp.asarray(np.pad(weights, ((0, padding), (0, 0))))
        self.normals = jnp.asarray(
            np.pad(np.asarray(curve.normals), ((0, padding), (0, 0)), mode="edge")
        )
        self.valid = jnp.arange(padded) < count
        self.basis = jnp.asarray(basis, dtype=jnp.float64)
        self.exception_keys = pairs.exception_keys
        self.panel_vertices = curve.panel_vertices
        self.correction = correction
        self.layer = layer
        self.panel_count = count
        self.block_size = block_size
        tensor = block_size * order * block_size * order
        self.action_workspace_bytes = _FLOAT_BYTES * (
            10 * tensor + 4 * block_size * block_size * basis.shape[1] + 3 * padded
        )
        self.source = source
        self.target = target
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True, diagonal_assembly=False
        )
        self.batch_shape = ()
        self.operator_id = operator_id

    @property
    def _padded(self) -> int:
        return self.points.shape[0]

    def _pair_block(self, target_start: Array, source_start: Array, /) -> Array:
        size = self.block_size
        offsets = jnp.arange(size)
        targets = target_start + offsets
        sources = source_start + offsets
        keys = targets[:, None].astype(jnp.int64) * self.panel_count + sources[None, :]
        positions = jnp.searchsorted(self.exception_keys, keys)
        safe = jnp.minimum(positions, self.exception_keys.shape[0] - 1)
        exceptional = self.exception_keys[safe] == keys
        target_valid = jax.lax.dynamic_slice_in_dim(self.valid, target_start, size)
        source_valid = jax.lax.dynamic_slice_in_dim(self.valid, source_start, size)
        active = target_valid[:, None] & source_valid[None, :] & ~exceptional
        return regular_pair_values(
            jax.lax.dynamic_slice_in_dim(self.points, target_start, size),
            jax.lax.dynamic_slice_in_dim(self.weights, target_start, size),
            jax.lax.dynamic_slice_in_dim(self.points, source_start, size),
            jax.lax.dynamic_slice_in_dim(self.weights, source_start, size),
            jax.lax.dynamic_slice_in_dim(self.normals, source_start, size),
            self.basis,
            active,
            self.layer,
        )

    def _regular_rows(self, density: Array, /) -> Array:
        size = self.block_size
        blocks = self._padded // size

        def target_step(target_block: Array, rows: Array) -> Array:
            start = target_block * size

            def source_step(source_block: Array, total: Array) -> Array:
                source_start = source_block * size
                values = self._pair_block(start, source_start)
                local = jax.lax.dynamic_slice_in_dim(density, source_start, size)
                return total + ein.contract("tsk,sk->t", values, local, backend="jax")

            total = jax.lax.fori_loop(
                0, blocks, source_step, jnp.zeros((size,), dtype=density.dtype)
            )
            return jax.lax.dynamic_update_slice_in_dim(rows, total, start, 0)

        return jax.lax.fori_loop(
            0, blocks, target_step, jnp.zeros((self._padded,), dtype=density.dtype)
        )

    def _regular_columns(self, weights: Array, /) -> Array:
        size = self.block_size
        blocks = self._padded // size
        components = self.basis.shape[1]

        def source_step(source_block: Array, columns: Array) -> Array:
            start = source_block * size

            def target_step(target_block: Array, total: Array) -> Array:
                target_start = target_block * size
                values = self._pair_block(target_start, start)
                local = jax.lax.dynamic_slice_in_dim(weights, target_start, size)
                return total + ein.contract("tsk,t->sk", values, local, backend="jax")

            total = jax.lax.fori_loop(
                0,
                blocks,
                target_step,
                jnp.zeros((size, components), dtype=weights.dtype),
            )
            return jax.lax.dynamic_update_slice_in_dim(columns, total, start, 0)

        return jax.lax.fori_loop(
            0,
            blocks,
            source_step,
            jnp.zeros((self._padded, components), dtype=weights.dtype),
        )

    def _panel_density(self, value: Array, /) -> Array:
        match self.layer:
            case "single":
                density = value[:, None]
            case "double":
                density = value[self.panel_vertices]
            case _:
                assert_never(self.layer)
        return jnp.pad(density, ((0, self._padded - self.panel_count), (0, 0)))

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(vector)
        rows = self._regular_rows(self._panel_density(value))[: self.panel_count]
        return self.target.validate(rows + self.correction.mv(value))

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        padded = jnp.pad(value, (0, self._padded - self.panel_count))
        columns = self._regular_columns(padded)[: self.panel_count]
        match self.layer:
            case "single":
                regular = columns[:, 0]
            case "double":
                regular = (
                    jnp.zeros((self.source.size,), dtype=columns.dtype)
                    .at[self.panel_vertices.reshape((-1,))]
                    .add(columns.reshape((-1,)))
                )
            case _:
                assert_never(self.layer)
        return self.source.validate(regular + self.correction.transpose_mv(value))

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        covector = self.target.riesz(vector)
        return self.source.inverse_riesz(self.transpose_mv(covector))

    def _materialize(self, /) -> Array:
        size = self.block_size
        blocks = self._padded // size
        components = self.basis.shape[1]

        def target_step(target_block: Array, matrix: Array) -> Array:
            def source_step(source_block: Array, matrix: Array) -> Array:
                values = self._pair_block(target_block * size, source_block * size)
                return jax.lax.dynamic_update_slice(
                    matrix, values, (target_block * size, source_block * size, 0)
                )

            return jax.lax.fori_loop(0, blocks, source_step, matrix)

        matrix = jax.lax.fori_loop(
            0,
            blocks,
            target_step,
            jnp.zeros((self._padded, self._padded, components), dtype=jnp.float64),
        )[: self.panel_count, : self.panel_count]
        match self.layer:
            case "single":
                regular = matrix[..., 0]
            case "double":
                regular = (
                    jnp.zeros((self.panel_count, self.source.size), dtype=matrix.dtype)
                    .at[:, self.panel_vertices[:, 0]]
                    .add(matrix[..., 0])
                    .at[:, self.panel_vertices[:, 1]]
                    .add(matrix[..., 1])
                )
            case _:
                assert_never(self.layer)
        return regular + self.correction.as_dense()

    def _action_workspace_cost(self, /) -> tuple[int, str]:
        return self.action_workspace_bytes, "blocked-direct-panel-galerkin-action"


def _single_correction(
    pairs: PanelPairData2D,
    count: int,
    source: ArraySpace,
    target: DualSpace,
    /,
) -> SparseCoordinateOperator:
    return SparseCoordinateOperator(
        EdgeRelation(pairs.sources, pairs.targets, source_size=count, target_size=count),
        pairs.single_values,
        source=source,
        target=target,
    )


def _double_correction(
    pairs: PanelPairData2D,
    curve: ClosedPolygonalCurve2D,
    source: ArraySpace,
    target: DualSpace,
    /,
) -> SparseCoordinateOperator:
    hats = curve.panel_vertices[pairs.sources]
    return SparseCoordinateOperator(
        EdgeRelation(
            jnp.concatenate((hats[:, 0], hats[:, 1])),
            jnp.concatenate((pairs.targets, pairs.targets)),
            source_size=curve.vertex_count,
            target_size=curve.panel_count,
        ),
        jnp.concatenate((pairs.double_values[:, 0], pairs.double_values[:, 1])),
        source=source,
        target=target,
    )


class ScalarLaplaceFieldEvaluation2D(StrictModule):
    """Off-boundary field values with side-membership evidence.

    Potentials are exact straight-panel integrals of the discrete densities.
    ``winding_numbers`` is ``-D[1]``: one inside the curve and zero outside.
    ``side_valid`` requires the declared side and a positive boundary distance.
    """

    __strict_contract__ = True

    values: Float64[_TargetDim]
    winding_numbers: Float64[_TargetDim]
    boundary_distances: Float64[_TargetDim]
    side_valid: Bool[_TargetDim]
    accepted: Bool[Scalar]
    side: ScalarTraceSide2D = eqx.field(static=True)
    method: str = eqx.field(static=True)


def _curve_potentials(
    curve: ClosedPolygonalCurve2D,
    points: Array,
    dirichlet: Array,
    conormal: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    count = points.shape[0]
    padded = -(-count // _FIELD_CHUNK) * _FIELD_CHUNK
    padded_points = jnp.concatenate(
        (points, jnp.broadcast_to(points[:1], (padded - count, 2)))
    )
    starts = curve.vertices[curve.panel_vertices[:, 0]]
    start_values = dirichlet[curve.panel_vertices[:, 0]]
    end_values = dirichlet[curve.panel_vertices[:, 1]]

    def evaluate(block: Array) -> Array:
        single, start, end = segment_layer_moments(
            block[:, None, :],
            starts[None, :, :],
            curve.tangents[None, :, :],
            curve.normals[None, :, :],
            curve.lengths[None, :],
        )
        relative = block[:, None, :] - starts[None, :, :]
        along = jnp.clip(
            jnp.sum(relative * curve.tangents[None, :, :], axis=-1), 0.0, curve.lengths
        )
        closest = starts[None, :, :] + along[..., None] * curve.tangents[None, :, :]
        distance = jnp.min(jnp.linalg.norm(block[:, None, :] - closest, axis=-1), axis=1)
        return jnp.stack(
            (
                single @ conormal,
                start @ start_values + end @ end_values,
                -jnp.sum(start + end, axis=1),
                distance,
            ),
            axis=1,
        )

    values = jax.lax.map(evaluate, padded_points.reshape((-1, _FIELD_CHUNK, 2)))
    values = values.reshape((padded, 4))[:count]
    return values[:, 0], values[:, 1], values[:, 2], values[:, 3]


class ScalarLaplaceGalerkin2D(StrictModule, NonTrainableState):
    """Prepared weak scalar Laplace boundary operators on a closed polygon.

    ``single_layer`` is ``V_ij = ∫∫ ψ_i G ψ_j`` (DP0 x DP0) and ``double_layer``
    is ``K_iv = ∫∫ ψ_i ∂G/∂n_y φ_v`` (DP0 test x continuous P1 trial); both map
    into the DP0 dual. ``exterior_relation`` is the rectangular weak exterior
    Cauchy relation from ``(dirichlet_trace, conormal, far_field_constant)`` to
    ``(exterior_boundary_equation, total_conormal)``:
    ``((M/2 - K) φ + V q - m c, m^T q)``. The kernel identity is the existing
    ``LaplaceLayerKernel2D`` identity, recorded as ``report.kernel_id``. Geometry,
    pair classification, and singular corrections are fixed at preparation and
    reused for any density.

    ``derivative_capability`` admits the densities ``dirichlet`` and
    ``conormal`` and the ``far_field_constant`` as direct ``INPUT`` derivatives:
    ``single_layer``, ``double_layer``, ``exterior_relation``, and
    ``evaluate_field`` are linear in them on the fixed prepared geometry, so
    their JVP is the action itself and their VJP is the transpose action.
    ``geometry``, ``kernel``, ``quadrature``, and field ``targets`` are refused
    with their reasons; ``evaluate_field`` raises a ``derivative-unsupported``
    error at trace time for target or curve tangents.
    """

    curve: ClosedPolygonalCurve2D
    spaces: ScalarBoundarySpaces2D
    convention: ScalarTraceConvention2D
    single_layer: AbstractLinearOperator
    double_layer: AbstractLinearOperator
    exterior_relation: BlockLinearOperator
    policy: ScalarLaplaceGalerkinPolicy2D
    report: ScalarLaplaceGalerkinReport2D
    derivative_capability: OwnerDerivativeCapability
    prepared_id: str = eqx.field(static=True)

    def evaluate_field(
        self,
        targets: ConvertibleToArray,
        /,
        *,
        side: ScalarTraceSide2D,
        dirichlet: ArrayLike,
        conormal: ArrayLike,
        far_field_constant: ArrayLike | None = None,
    ) -> ScalarLaplaceFieldEvaluation2D:
        """Evaluate the declared-side Green representation off the boundary.

        Exterior: ``u = c_inf + D φ - S q``; interior: ``u = S q - D φ``. The
        far-field constant is required exactly for the exterior side. Values,
        winding numbers, and distances are linear in or independent of the
        densities; target and curve derivatives are refused.
        """
        side_ = parse(side, ScalarTraceSide2D, "side")
        points = as_array(targets, Float64[_TargetDim, Literal[2]], "targets")
        phi = self.spaces.dirichlet_trace.vector_space.validate(dirichlet)
        q = self.spaces.conormal_trace.vector_space.validate(conormal)
        single, double, winding, distance = _curve_potentials(self.curve, points, phi, q)
        match side_:
            case "exterior":
                if far_field_constant is None:
                    raise ValueError("Exterior fields require far_field_constant.")
                constant = as_array(
                    far_field_constant, Float64[Scalar], "far_field_constant"
                )
                values = constant + double - single
                side_match = jnp.abs(winding) < 0.5
            case "interior":
                if far_field_constant is not None:
                    raise ValueError("Interior fields take no far_field_constant.")
                values = single - double
                side_match = jnp.abs(winding - 1.0) < 0.5
            case _:
                assert_never(side_)
        scale = jnp.maximum(
            jnp.max(jnp.abs(self.curve.vertices)), jnp.max(jnp.abs(points))
        )
        valid = side_match & (distance > 64.0 * _EPSILON * scale)
        outputs = refuse_derivative_dependencies(
            refuse_derivative_dependencies(
                (values, winding, distance),
                (points,),
                message=f"ScalarLaplaceGalerkin2D refuses 'targets': {_TARGET_REFUSAL}",
            ),
            (self.curve,),
            message=(
                "ScalarLaplaceGalerkin2D refuses 'geometry': "
                f"{_FIXED_STRUCTURE_REFUSALS['geometry']}"
            ),
        )
        return ScalarLaplaceFieldEvaluation2D(
            values=outputs[0],
            winding_numbers=outputs[1],
            boundary_distances=outputs[2],
            side_valid=valid,
            accepted=jnp.all(valid),
            side=side_,
            method="exact-straight-panel-closed-form",
        )


def _exterior_relation(
    spaces: ScalarBoundarySpaces2D,
    single: AbstractLinearOperator,
    double: AbstractLinearOperator,
    /,
) -> BlockLinearOperator:
    dirichlet = spaces.dirichlet_trace.vector_space
    conormal = spaces.conormal_trace.vector_space
    far_field = spaces.far_field_space
    rows = DualSpace(conormal)
    total_row = DualSpace(far_field)
    lengths = spaces.conormal_trace.integral_weights
    return BlockLinearOperator(
        (
            (
                0.5 * spaces.mixed_mass - double,
                single,
                DenseLinearOperator(-lengths[:, None], source=far_field, target=rows),
            ),
            (
                None,
                DenseLinearOperator(lengths[None, :], source=conormal, target=total_row),
                None,
            ),
        ),
        source=BlockSpace(
            (dirichlet, conormal, far_field),
            names=("dirichlet_trace", "conormal", "far_field_constant"),
        ),
        target=BlockSpace(
            (rows, total_row), names=("exterior_boundary_equation", "total_conormal")
        ),
        operator_id=canonical_fingerprint(
            {
                "kind": "scalar-laplace-exterior-cauchy-relation-2d",
                "spaces": spaces.spaces_id,
                "single_layer": single.operator_id,
                "double_layer": double.operator_id,
            }
        ),
    )


def _assembly_report(
    curve: ClosedPolygonalCurve2D,
    policy: ScalarLaplaceGalerkinPolicy2D,
    pairs: PanelPairData2D,
    kernel_id: str,
    operators: tuple[_BlockedLayerOperator2D, _BlockedLayerOperator2D],
    resident_bytes: int,
    /,
) -> ScalarLaplaceGalerkinReport2D:
    counts = jnp.asarray(pairs.pair_counts)
    supported = jnp.isfinite(pairs.maximum_errors) & (
        (counts == 0) | (pairs.maximum_errors <= policy.tolerance)
    )
    finite = (
        jnp.all(jnp.isfinite(pairs.single_values))
        & jnp.all(jnp.isfinite(pairs.double_values))
        & jnp.all(jnp.isfinite(operators[0].points))
    )
    return ScalarLaplaceGalerkinReport2D(
        curve_id=curve.curve_id,
        policy_id=policy.policy_id,
        kernel_id=kernel_id,
        pde="-Delta(u)=0",
        fundamental_solution="-log(r)/(2*pi)",
        provider="native-blocked-direct",
        panel_count=curve.panel_count,
        vertex_count=curve.vertex_count,
        pair_class_names=PAIR_CLASS_NAMES_2D,
        pair_class_methods=PAIR_CLASS_METHODS_2D,
        pair_counts=pairs.pair_counts,
        promoted_pair_count=pairs.promoted_count,
        maximum_errors=pairs.maximum_errors,
        tolerance=policy.tolerance,
        pair_class_supported=supported,
        evaluations=pairs.evaluations,
        maximum_adaptive_depth=pairs.maximum_depth,
        exception_count=pairs.exception_keys.shape[0],
        preparation_workspace_bytes=pairs.preparation_workspace_bytes,
        resident_bytes=resident_bytes,
        action_workspace_bytes_per_rhs=max(
            operator.action_workspace_bytes for operator in operators
        ),
        materializable=True,
        continuum_discretization_error_estimated=False,
        finite=finite,
        accuracy_supported=finite & jnp.all(supported),
        support=_support_envelope(),
        report_id=canonical_fingerprint(
            {
                "kind": "scalar-laplace-galerkin-report-2d",
                "curve": curve.curve_id,
                "policy": policy.policy_id,
                "pair_counts": list(pairs.pair_counts),
                "promoted": pairs.promoted_count,
                "errors": array_tree_fingerprint(pairs.maximum_errors),
                "evaluations": array_tree_fingerprint(pairs.evaluations),
            }
        ),
    )


def _resident_bytes(
    curve: ClosedPolygonalCurve2D,
    policy: ScalarLaplaceGalerkinPolicy2D,
    exception_bytes: int,
    /,
) -> int:
    padded = -(-curve.panel_count // policy.block_size) * policy.block_size
    rule_bytes = 2 * padded * (3 * policy.regular_order + 3) * _FLOAT_BYTES
    trace_bytes = 12 * curve.vertex_count * _FLOAT_BYTES
    return rule_bytes + trace_bytes + 2 * exception_bytes


def prepare_scalar_laplace_galerkin_2d(
    curve: ClosedPolygonalCurve2D,
    /,
    *,
    policy: ScalarLaplaceGalerkinPolicy2D | None = None,
) -> ScalarLaplaceGalerkin2D:
    """Prepare weak V, K, trace spaces, and the exterior relation on a polygon."""
    selected = ScalarLaplaceGalerkinPolicy2D() if policy is None else policy
    if not isinstance(selected, ScalarLaplaceGalerkinPolicy2D):
        raise TypeError("policy must be a ScalarLaplaceGalerkinPolicy2D or None.")
    if not isinstance(curve, ClosedPolygonalCurve2D):
        raise TypeError("curve must be a ClosedPolygonalCurve2D.")
    minimum = _resident_bytes(
        curve, selected, 3 * curve.panel_count * EXCEPTION_ENTRY_BYTES
    )
    if minimum > selected.max_resident_bytes:
        raise ValueError("[resident-bytes] Galerkin state exceeds max_resident_bytes.")
    geometry = _host_geometry(curve)
    pairs = prepare_panel_pairs_2d(
        geometry,
        regular_order=selected.regular_order,
        near_ratio=selected.near_ratio,
        tolerance=selected.tolerance,
        max_depth=selected.max_depth,
        max_adaptive_intervals=selected.max_adaptive_intervals,
        max_exception_pairs=selected.max_exception_pairs,
        max_preparation_workspace_bytes=selected.max_preparation_workspace_bytes,
    )
    resident = _resident_bytes(curve, selected, pairs.resident_bytes)
    if resident > selected.max_resident_bytes:
        raise ValueError("[resident-bytes] Galerkin state exceeds max_resident_bytes.")
    spaces = _prepare_boundary_spaces(curve, selected.gram_tolerance)
    rule = regular_panel_rule(geometry, selected.regular_order)
    kernel_id = LaplaceLayerKernel2D().kernel_id
    rows = DualSpace(spaces.conormal_trace.vector_space)
    single, double = (
        _BlockedLayerOperator2D(
            layer,
            curve,
            pairs,
            rule,
            source=source,
            target=rows,
            block_size=selected.block_size,
            operator_id=canonical_fingerprint(
                {
                    "kind": f"scalar-laplace-weak-{layer}-layer-2d",
                    "spaces": spaces.spaces_id,
                    "kernel": kernel_id,
                    "policy": selected.policy_id,
                }
            ),
        )
        for layer, source in (
            ("single", spaces.conormal_trace.vector_space),
            ("double", spaces.dirichlet_trace.vector_space),
        )
    )
    report = _assembly_report(
        curve, selected, pairs, kernel_id, (single, double), resident
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "scalar-laplace-galerkin-2d",
            "spaces": spaces.spaces_id,
            "report": report.report_id,
        }
    )
    return ScalarLaplaceGalerkin2D(
        curve=curve,
        spaces=spaces,
        convention=ScalarTraceConvention2D(),
        single_layer=single,
        double_layer=double,
        exterior_relation=_exterior_relation(spaces, single, double),
        policy=selected,
        report=report,
        derivative_capability=OwnerDerivativeCapability(
            prepared_id,
            admitted={
                "conormal": DerivativeSurface.INPUT,
                "dirichlet": DerivativeSurface.INPUT,
                "far_field_constant": DerivativeSurface.INPUT,
            },
            refused={**_FIXED_STRUCTURE_REFUSALS, "targets": _TARGET_REFUSAL},
            route=DerivativeRoute.DIRECT,
            conditions=("prepared-geometry-fixed",),
        ),
        prepared_id=prepared_id,
    )


class PreparedExteriorLaplaceDirichlet2D(StrictModule, NonTrainableState):
    """Square bordered exterior Dirichlet-to-Neumann system with far-field data.

    Unknowns are the DP0 conormal ``q`` and the far-field constant ``c`` (block
    names ``conormal`` and ``far_field_constant``). The equations, in the same
    order, are the exterior boundary integral equation
    ``(M/2 - K) φ + V q - m c = 0`` tested with DP0 and the bounded-field
    compatibility ``m^T q = 0``. Each equation is applied through the inverse
    Riesz map of its declared row space (DP0 arc-length mass, Euclidean scalar),
    so the square operator is an endomorphism of the unknown block space. The
    system is nonsingular because ``V`` is positive definite on zero-mean DP0
    densities and ``m`` is nonzero. ``decaying`` additionally requires the
    solved ``|c|`` to meet ``far_field_tolerance``; a nonzero constant is
    reported, never silently accepted.

    ``derivative_capability`` admits ``dirichlet`` as an implicit
    ``SOLVER_ARGUMENT`` under ``rhs-only`` differentiation, conditioned on a
    converged solve, an accepted result, and the fixed prepared geometry; under
    ``none`` nothing is admitted and the route is stopped. Geometry, kernel,
    and quadrature derivatives are always refused.
    """

    galerkin: ScalarLaplaceGalerkin2D
    operator: BlockLinearOperator
    data_operator: AbstractLinearOperator
    linear: PreparedLinearSolve
    derivative_capability: OwnerDerivativeCapability
    far_field: ExteriorFarField2D = eqx.field(static=True)
    far_field_tolerance: float | None = eqx.field(static=True)
    certification_tolerance: float = eqx.field(static=True)
    equations: tuple[str, str] = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)


class ExteriorLaplaceDirichletResult2D(StrictModule):
    """Exterior conormal, far-field constant, native status, and certification.

    ``equation_residual`` is the DP0-dual norm of the original weak exterior
    equation, recomputed after the solve; ``equation_scale`` is the sum of its
    term norms at the solution and at the response to the uncancelled
    (sign-scrambled) Dirichlet terms, so an exact datum whose terms cancel is
    measured against those terms. ``total_conormal`` is ``∫ q ds``, gated
    against ``∫ (|q| + |q_s|) ds`` with ``q_s`` that same response. ``accepted``
    is primal acceptance; ``derivative_valid`` additionally requires the solve's own
    derivative contract and is ``False`` when ``derivative_capability`` is
    stopped. Floating outputs, including ``linear.value``, carry NaN tangents
    (``status`` failure) or raise (``error`` failure) when not accepted.
    """

    __strict_contract__ = True

    conormal: Float64[_PanelDim]
    far_field_constant: Float64[Scalar]
    linear: LinearSolveResult
    equation_residual: Float64[Scalar]
    equation_scale: Float64[Scalar]
    total_conormal: Float64[Scalar]
    far_field_satisfied: Bool[Scalar]
    equations_certified: Bool[Scalar]
    accepted: Bool[Scalar]
    derivative_valid: Bool[Scalar]
    derivative_capability: OwnerDerivativeCapability
    far_field: ExteriorFarField2D = eqx.field(static=True)


def _far_field_tolerance(
    far_field: ExteriorFarField2D, tolerance: float | None, /
) -> float | None:
    match far_field:
        case "bounded":
            if tolerance is not None:
                raise ValueError("A bounded far field takes no far_field_tolerance.")
            return None
        case "decaying":
            if tolerance is None:
                raise ValueError("A decaying far field requires far_field_tolerance.")
            return _positive_float(tolerance, "far_field_tolerance")
        case _:
            assert_never(far_field)


def _exterior_policy(size: int, /) -> LinearSolvePolicy:
    return LinearSolvePolicy(
        GMRES(restart=min(size, 128), stagnation_iterations=min(size, 128)),
        tolerance=TolerancePolicy(relative=1.0e-12, absolute=0.0, max_steps=8 * size),
        differentiation=DifferentiationPolicy("rhs-only"),
        failure=FailurePolicy("status"),
    )


def prepare_exterior_laplace_dirichlet_2d(
    galerkin: ScalarLaplaceGalerkin2D,
    /,
    *,
    far_field: ExteriorFarField2D = "bounded",
    far_field_tolerance: float | None = None,
    certification_tolerance: float = 1.0e-8,
    linear: LinearSolvePolicy | None = None,
) -> PreparedExteriorLaplaceDirichlet2D:
    """Prepare the bordered exterior Dirichlet-to-Neumann system once.

    The default solve is matrix-free GMRES on the blocked-direct actions. A
    dense direct policy materializes only within its own MaterializationPolicy.
    Only ``rhs-only`` (the default; implicit Dirichlet-data derivatives) or
    ``none`` differentiation is admitted: the prepared ``V``/``K`` operators are
    fixed geometry, so operator, kernel, and geometry derivatives are refused.
    """
    if not isinstance(galerkin, ScalarLaplaceGalerkin2D):
        raise TypeError("galerkin must be a ScalarLaplaceGalerkin2D.")
    selected_far_field = parse(far_field, ExteriorFarField2D, "far_field")
    decay = _far_field_tolerance(selected_far_field, far_field_tolerance)
    certification = _positive_float(certification_tolerance, "certification_tolerance")
    spaces = galerkin.spaces
    conormal = spaces.conormal_trace.vector_space
    far = spaces.far_field_space
    unknowns = BlockSpace((conormal, far), names=("conormal", "far_field_constant"))
    policy = _exterior_policy(unknowns.size) if linear is None else linear
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear must be a LinearSolvePolicy or None.")
    if not bool(galerkin.report.accuracy_supported):
        raise ValueError(
            "[quadrature] Galerkin quadrature evidence does not support solving."
        )
    count = conormal.size
    panels = np.arange(count)
    riesz = SparseCoordinateOperator(
        EdgeRelation(panels, panels, source_size=count, target_size=count),
        1.0 / spaces.conormal_trace.integral_weights,
        source=DualSpace(conormal),
        target=conormal,
    )
    lengths = spaces.conormal_trace.integral_weights
    operator = BlockLinearOperator(
        (
            (
                riesz @ galerkin.single_layer,
                DenseLinearOperator(-jnp.ones((count, 1)), source=far, target=conormal),
            ),
            (DenseLinearOperator(lengths[None, :], source=conormal, target=far), None),
        ),
        source=unknowns,
        target=unknowns,
        operator_id=canonical_fingerprint(
            {
                "kind": "scalar-laplace-bordered-exterior-dirichlet-2d",
                "galerkin": galerkin.prepared_id,
            }
        ),
    )
    trace_block = galerkin.exterior_relation.blocks[0][0]
    if trace_block is None:
        raise RuntimeError("The exterior relation has no Dirichlet-trace block.")
    formulation_id = canonical_fingerprint(
        {
            "kind": "prepared-exterior-laplace-dirichlet-2d",
            "operator": operator.operator_id,
            "far_field": selected_far_field,
            "far_field_tolerance": decay,
            "certification_tolerance": certification,
        }
    )
    capability = _exterior_derivative_capability(
        formulation_id, policy.differentiation.mode
    )
    return PreparedExteriorLaplaceDirichlet2D(
        galerkin=galerkin,
        operator=operator,
        data_operator=riesz @ trace_block,
        linear=prepare(LinearSystem(operator, problem_id=formulation_id), policy),
        derivative_capability=capability,
        far_field=selected_far_field,
        far_field_tolerance=decay,
        certification_tolerance=certification,
        equations=(
            "dp0-riesz:(M/2-K)phi+Vq-m*c=0",
            "total-conormal:m^T q=0",
        ),
        formulation_id=formulation_id,
    )


def _exterior_derivative_capability(
    formulation_id: str, mode: DifferentiationMode, /
) -> OwnerDerivativeCapability:
    match mode:
        case "rhs-only":
            return OwnerDerivativeCapability(
                formulation_id,
                admitted={"dirichlet": DerivativeSurface.SOLVER_ARGUMENT},
                refused=_FIXED_STRUCTURE_REFUSALS,
                route=DerivativeRoute.IMPLICIT,
                conditions=(
                    "accepted-result",
                    "prepared-geometry-fixed",
                    "solve-converged",
                ),
            )
        case "none":
            return OwnerDerivativeCapability(
                formulation_id,
                admitted={},
                refused={
                    **_FIXED_STRUCTURE_REFUSALS,
                    "dirichlet": (
                        "differentiation mode 'none' admits no derivative of the "
                        "exterior Dirichlet-to-Neumann solve"
                    ),
                },
                route=DerivativeRoute.STOPPED,
                conditions=("prepared-geometry-fixed",),
            )
        case "mathematical":
            raise ValueError(
                "The exterior Galerkin solve admits only rhs-only or none "
                "differentiation: the prepared V/K operators are fixed geometry, so "
                "operator, kernel, and geometry derivatives are not qualified."
            )
        case "algorithmic":
            raise ValueError(
                "The exterior Galerkin solve admits only rhs-only or none "
                "differentiation: an unrolled Krylov derivative is not the "
                "solution-map derivative and is not qualified."
            )
        case invalid:
            assert_never(invalid)


def _dual_norm(rows: Array, lengths: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(rows * rows / lengths))


def uncancelled_exterior_response(
    prepared: PreparedExteriorLaplaceDirichlet2D, dirichlet: Array, /
) -> tuple[Array, Array, Array, Array]:
    """The exterior solution for the uncancelled direction of ``dirichlet``.

    Returns ``(s * phi, q_s, c_s, successful)`` for fixed Rademacher signs
    ``s`` (``uncancelled_direction``): the conormal and far-field constant the
    individual Dirichlet terms drive before they cancel. A Dirichlet datum whose
    exterior response vanishes (a constant, whose double-layer and identity
    terms cancel) is solved to roundoff of those terms, so ``q_s`` is the
    reference magnitude of the solved conormal and of its total ``m^T q``.
    The response is evidence scale only; it carries no derivative.
    """
    direction = jax.lax.stop_gradient(uncancelled_direction(dirichlet))
    response = solve(
        prepared.linear,
        (-prepared.data_operator.mv(direction), jnp.zeros((1,), dtype=jnp.float64)),
    )
    conormal, constant = jax.lax.stop_gradient(response.value)
    return direction, conormal, constant, response.successful


def _guard_exterior_derivatives[T](
    prepared: PreparedExteriorLaplaceDirichlet2D,
    outputs: T,
    phi: Array,
    derivative_valid: Array,
    /,
) -> T:
    """Poison non-accepted derivatives, or refuse all of them when stopped."""
    capability = prepared.derivative_capability
    if capability.admits("dirichlet"):
        return guard_derivative_validity(
            outputs,
            derivative_valid,
            dependencies=(phi,),
            failure=prepared.linear.plan.policy.failure.mode,
            message=(
                "The exterior Galerkin solve was not accepted (solve, equation "
                "certification, or declared far field failed); its Dirichlet-data "
                "derivative is invalid."
            ),
        )
    reason = dict(capability.refused)["dirichlet"]
    return refuse_derivative_dependencies(
        outputs,
        (phi,),
        message=f"owner {capability.owner_id} refuses 'dirichlet': {reason}",
    )


def solve_exterior_laplace_dirichlet_2d(
    prepared: PreparedExteriorLaplaceDirichlet2D,
    dirichlet: ArrayLike,
    /,
) -> ExteriorLaplaceDirichletResult2D:
    """Solve for the exterior conormal and far-field constant of P1 Dirichlet data.

    Acceptance requires native solve success, the original weak equations to
    hold within ``certification_tolerance`` of their term scale, zero total
    conormal within the same relative tolerance, and the declared far field.
    Under ``rhs-only`` the floating outputs are differentiable in ``dirichlet``
    exactly when ``derivative_valid``; otherwise their tangents are NaN (or the
    ``error`` failure policy raises). Under ``none`` any ``dirichlet``
    derivative raises ``derivative-unsupported`` at trace time.
    """
    if not isinstance(prepared, PreparedExteriorLaplaceDirichlet2D):
        raise TypeError("prepared must be a PreparedExteriorLaplaceDirichlet2D.")
    galerkin = prepared.galerkin
    phi = galerkin.spaces.dirichlet_trace.vector_space.validate(dirichlet)
    rhs = (-prepared.data_operator.mv(phi), jnp.zeros((1,), dtype=jnp.float64))
    result = solve(prepared.linear, rhs)
    conormal, constant = result.value
    lengths = galerkin.spaces.conormal_trace.integral_weights
    rows, total = galerkin.exterior_relation.mv((phi, conormal, constant))
    trace_block = galerkin.exterior_relation.blocks[0][0]
    if trace_block is None:
        raise RuntimeError("The exterior relation has no Dirichlet-trace block.")
    uncancelled, conormal_terms, constant_terms, terms_solved = (
        uncancelled_exterior_response(prepared, phi)
    )
    # Every term norm is taken both at the solution and at the uncancelled
    # response: an exact datum whose terms cancel is measured against them.
    scale = (
        _dual_norm(trace_block.mv(phi), lengths)
        + _dual_norm(galerkin.single_layer.mv(conormal), lengths)
        + jnp.abs(constant[0]) * jnp.sqrt(jnp.sum(lengths))
        + _dual_norm(trace_block.mv(uncancelled), lengths)
        + _dual_norm(galerkin.single_layer.mv(conormal_terms), lengths)
        + jnp.abs(constant_terms[0]) * jnp.sqrt(jnp.sum(lengths))
    )
    total_scale = jnp.sum(lengths * (jnp.abs(conormal) + jnp.abs(conormal_terms)))
    residual = _dual_norm(rows, lengths)
    tolerance = prepared.certification_tolerance
    certified = (
        jnp.isfinite(residual)
        & jnp.all(jnp.isfinite(conormal))
        & terms_solved
        & (residual <= tolerance * scale)
        & (jnp.abs(total[0]) <= tolerance * total_scale)
    )
    match prepared.far_field:
        case "bounded":
            far_field_ok = jnp.asarray(True)
        case "decaying":
            if prepared.far_field_tolerance is None:
                raise RuntimeError("A decaying far field lost its tolerance.")
            far_field_ok = jnp.abs(constant[0]) <= prepared.far_field_tolerance
        case _:
            assert_never(prepared.far_field)
    accepted = result.successful & certified & far_field_ok
    derivative_valid = accepted & result.derivative_valid
    value, residual, scale, total = _guard_exterior_derivatives(
        prepared, (result.value, residual, scale, total), phi, derivative_valid
    )
    conormal, constant = value
    return ExteriorLaplaceDirichletResult2D(
        conormal=conormal,
        far_field_constant=constant[0],
        linear=eqx.tree_at(lambda linear: linear.value, result, value),
        equation_residual=residual,
        equation_scale=scale,
        total_conormal=total[0],
        far_field_satisfied=far_field_ok,
        equations_certified=certified,
        accepted=accepted,
        derivative_valid=derivative_valid,
        derivative_capability=prepared.derivative_capability,
        far_field=prepared.far_field,
    )


class TraceProjectionResult2D(StrictModule):
    """Declared L2 projection of one sampled trace with its measured defect.

    ``projection_defect`` is ``||f - Π f||_{L2(Γ)}`` evaluated with the
    projection's own sampling rule and ``trace_norm`` is ``||f||`` on that rule.
    """

    __strict_contract__ = True

    coefficients: Float64[_CoefficientDim]
    projection_defect: Float64[Scalar]
    trace_norm: Float64[Scalar]
    relative_defect: Float64[Scalar]
    solve: LinearSolveResult | None
    successful: Bool[Scalar]
    representation: BoundaryTraceRepresentation2D = eqx.field(static=True)
    method: str = eqx.field(static=True)


class _P1TraceProjectionOperator2D(_AbstractCostedLinearOperator):
    """L2 projection of samples onto continuous P1; fails closed on its Gram solve."""

    load: SparseCoordinateOperator
    gram_solve: PreparedLinearSolve

    def __init__(
        self,
        load: SparseCoordinateOperator,
        gram_solve: PreparedLinearSolve,
        /,
        *,
        target: ArraySpace,
        operator_id: str,
    ) -> None:
        self.load = load
        self.gram_solve = gram_solve
        self.source = load.source
        self.target = target
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=False, diagonal_assembly=False
        )
        self.batch_shape = ()
        self.operator_id = operator_id

    def _gram_inverse(self, value: Array, /) -> Array:
        result = solve(self.gram_solve, value)
        return eqx.error_if(
            result.value,
            ~result.successful,
            "The P1 trace-projection Gram solve failed.",
        )

    def mv(self, vector: ArrayLike, /) -> Array:
        return self.target.validate(self._gram_inverse(self.load.mv(vector)))

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self._gram_inverse(self.target.validate(vector))
        return self.source.validate(self.load.transpose_mv(value))

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        covector = self.target.riesz(vector)
        return self.source.inverse_riesz(self.transpose_mv(covector))

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "Trace projections apply a Gram solve; no dense form."
        )

    def _action_workspace_cost(self, /) -> tuple[int, str]:
        size = self.target.size
        return 8 * size * _FLOAT_BYTES + self.source.size * _FLOAT_BYTES, (
            "sparse-load-plus-prepared-gram-solve"
        )


class BoundaryTraceProjection2D(StrictModule, NonTrainableState):
    """Declared L2 projections of sampled boundary traces onto P1 and DP0.

    Samples are values of a possibly higher-order trace at ``sample_points``
    (Gauss--Legendre points of each panel). ``project_dirichlet`` solves
    ``M c = ∫ f φ_v ds`` with the prepared P1 Gram solve; ``project_conormal``
    returns panel means. Loads are exact for per-panel polynomial traces of
    degree at most ``exact_polynomial_degree``. The result is always an L2
    projection with its measured defect, never the exact higher-order trace.
    ``dirichlet_projection`` is the same P1 projection as a linear operator
    whose Hilbert adjoint is P1 evaluation at the samples.
    """

    __strict_contract__ = True

    spaces: ScalarBoundarySpaces2D
    sample_points: Float64[_PanelDim, _SampleDim, Literal[2]]
    sample_weights: Float64[_PanelDim, _SampleDim]
    hat_values: Float64[_SampleDim, Literal[2]]
    sample_space: ArraySpace
    load: SparseCoordinateOperator
    dirichlet_projection: AbstractLinearOperator
    gram_solve: PreparedLinearSolve
    order: int = eqx.field(static=True)
    exact_polynomial_degree: int = eqx.field(static=True)
    projection_id: Identifier = eqx.field(static=True)

    def _result(
        self,
        samples: Array,
        reconstructed: Array,
        coefficients: Array,
        solve_result: LinearSolveResult | None,
        representation: BoundaryTraceRepresentation2D,
        /,
    ) -> TraceProjectionResult2D:
        difference = samples - reconstructed
        defect = jnp.sqrt(jnp.sum(self.sample_weights * difference * difference))
        norm = jnp.sqrt(jnp.sum(self.sample_weights * samples * samples))
        finite = jnp.all(jnp.isfinite(coefficients)) & jnp.isfinite(defect)
        successful = finite if solve_result is None else finite & solve_result.successful
        return TraceProjectionResult2D(
            coefficients=coefficients,
            projection_defect=defect,
            trace_norm=norm,
            relative_defect=jnp.where(
                norm > 0.0, defect / jnp.where(norm > 0.0, norm, 1.0), 0.0
            ),
            solve=solve_result,
            successful=successful,
            representation=representation,
            method="l2-projection",
        )

    def project_dirichlet(self, samples: ArrayLike, /) -> TraceProjectionResult2D:
        """L2-project sampled Dirichlet data onto continuous P1."""
        values = self.sample_space.validate(samples)
        result = solve(self.gram_solve, self.load.mv(values))
        coefficients = result.value
        vertices = self.spaces.curve.panel_vertices
        reconstructed = (
            coefficients[vertices[:, 0]][:, None] * self.hat_values[None, :, 0]
            + coefficients[vertices[:, 1]][:, None] * self.hat_values[None, :, 1]
        )
        return self._result(values, reconstructed, coefficients, result, "continuous-p1")

    def project_conormal(self, samples: ArrayLike, /) -> TraceProjectionResult2D:
        """L2-project sampled conormal data onto DP0 panel means."""
        values = self.sample_space.validate(samples)
        coefficients = jnp.sum(self.sample_weights * values, axis=1) / (
            self.spaces.curve.lengths
        )
        return self._result(values, coefficients[:, None], coefficients, None, "dp0")


def prepare_boundary_trace_projection_2d(
    spaces: ScalarBoundarySpaces2D,
    /,
    *,
    order: int = 8,
) -> BoundaryTraceProjection2D:
    """Prepare sampling points and the P1/DP0 L2 projections of boundary traces."""
    if not isinstance(spaces, ScalarBoundarySpaces2D):
        raise TypeError("spaces must be ScalarBoundarySpaces2D.")
    samples_per_panel = positive_integer(order, "order")
    gram_solve = spaces.dirichlet_trace.riesz_solve
    if gram_solve is None:
        raise RuntimeError("The continuous P1 trace space has no prepared Gram solve.")
    curve = spaces.curve
    geometry = _host_geometry(curve)
    points, weights, hats = regular_panel_rule(geometry, samples_per_panel)
    count = curve.panel_count
    projection_id = canonical_fingerprint(
        {
            "kind": "boundary-trace-l2-projection-2d",
            "spaces": spaces.spaces_id,
            "order": samples_per_panel,
        }
    )
    sample_space = ArraySpace(
        (count, samples_per_panel),
        dtype=np.float64,
        pairing=DiagonalPairing(jnp.asarray(weights), pairing_id=f"{projection_id}:rule"),
        space_id=f"{projection_id}:samples",
    )
    flat = np.arange(count * samples_per_panel).reshape((count, samples_per_panel))
    vertices = np.asarray(curve.panel_vertices)
    load = SparseCoordinateOperator(
        EdgeRelation(
            np.concatenate((flat.reshape((-1,)), flat.reshape((-1,)))),
            np.concatenate(
                (
                    np.repeat(vertices[:, 0], samples_per_panel),
                    np.repeat(vertices[:, 1], samples_per_panel),
                )
            ),
            source_size=count * samples_per_panel,
            target_size=curve.vertex_count,
        ),
        jnp.asarray(
            np.concatenate(
                (
                    (weights * hats[None, :, 0]).reshape((-1,)),
                    (weights * hats[None, :, 1]).reshape((-1,)),
                )
            ),
            dtype=jnp.float64,
        ),
        source=sample_space,
        target=DualSpace(spaces.dirichlet_trace.vector_space),
        operator_id=f"{projection_id}:p1-load",
    )
    return BoundaryTraceProjection2D(
        spaces=spaces,
        sample_points=jnp.asarray(points),
        sample_weights=jnp.asarray(weights),
        hat_values=jnp.asarray(hats),
        sample_space=sample_space,
        load=load,
        dirichlet_projection=_P1TraceProjectionOperator2D(
            load,
            gram_solve,
            target=spaces.dirichlet_trace.vector_space,
            operator_id=f"{projection_id}:p1-projection",
        ),
        gram_solve=gram_solve,
        order=samples_per_panel,
        exact_polynomial_degree=2 * samples_per_panel - 2,
        projection_id=projection_id,
    )


__all__ = [
    "BoundaryTraceProjection2D",
    "BoundaryTraceRepresentation2D",
    "BoundaryTraceSpace2D",
    "ClosedPolygonalCurve2D",
    "CurveTraversal2D",
    "ExteriorFarField2D",
    "ExteriorLaplaceDirichletResult2D",
    "prepare_boundary_trace_projection_2d",
    "prepare_exterior_laplace_dirichlet_2d",
    "prepare_scalar_laplace_galerkin_2d",
    "PreparedExteriorLaplaceDirichlet2D",
    "ScalarBoundarySpaces2D",
    "ScalarLaplaceFieldEvaluation2D",
    "ScalarLaplaceGalerkin2D",
    "ScalarLaplaceGalerkinPolicy2D",
    "ScalarLaplaceGalerkinReport2D",
    "ScalarTraceConvention2D",
    "ScalarTraceSide2D",
    "solve_exterior_laplace_dirichlet_2d",
    "TraceProjectionResult2D",
]
