#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provide native overset connectivity for a `MeshAssembly` and its moving registrations.

Prepared overset connectivity binds the certified cell parts of one assembly revision,
their explicitly declared solid walls and overset (artificial) outer boundaries, and one
donor-resolution policy. Preparation is host-side and runs in fixed phases:

1. *Hole cutting.* Declared PLC walls use bounded BVH ray candidates and crossing-parity
predicates evaluated with encoded-coefficient arithmetic and symbolic perturbation,
counting edge/vertex hits once. When a native B-Rep query is registered, its membership
results and explicit unresolved statuses determine vertex classification instead.
Contacts are blanked and reported. Represented facet/cell crossing predicates and
enclosed foreign wall vertices detect cut cells even without any enclosed mesh vertex;
these cells are blanked and reported as under-resolved cuts.
2. *Fringe layers.* Unblanked vertices of blanked cells and the vertices of a part's
overset boundary seed fringe layer one; each further layer adds the unclassified
vertices of cells touching the previous layer. Fringe vertices are receptors, cells with
a receptor vertex are fringe cells, and only cells whose every vertex is active are
donor-admissible, so donors never depend on other receptors. A part's own wall vertices
are protected: a wall vertex that is not active is a reported conflict.
3. *Donor resolution.* Receptors are located by the donor part's inverse cell map
restricted to admissible cells. Among containing donor parts the policy decides
deterministically (finest donor cell, then declared part priority, then part name, then
donor cell ID). A receptor without an admissible donor is an orphan with its reason.

Interpolative transfer is built as `PreparedFieldQuery` routes on the donor field
reconstruction, retaining its pointwise evidence and algebraic transpose, and is never
conservative. The conservative mode is a separate common-refinement remap of explicitly
selected cell averages. Motion refits the wall hierarchies, repeats hole cutting and
donor resolution, and commits a new registration epoch through `CompositionRebind` only
when every requested receptor and protected wall vertex satisfies the coverage contract
and the state transfer of uncovered vertices passes; otherwise the previous composition
is returned.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import IntEnum
from fractions import Fraction
from math import inf
from typing import assert_never, final, Literal, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._field_query as field_query_owner
import phydrax.discretization._reference_cell_types as reference_cell_types_owner
import phydrax.discretization._simplicial_locator as simplicial_locator_owner
import phydrax.discretization._views as views_owner
import phydrax.discretization.finite_volume._automatic_remap as automatic_remap_owner
import phydrax.discretization.finite_volume._remap_evidence as remap_evidence_owner
import phydrax.discretization.finite_volume._unstructured as unstructured_owner
import phydrax.geometry._contracts as contracts_owner
import phydrax.geometry._supermesh as supermesh_owner
from phydrax.discretization._reference_cell_types import CellGeometryElement
from phydrax.geometry._rational import fraction_endpoint
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory

from .._bvh import (
    bvh_overlap_pair_blocks,
    PackedBVH,
    prepare_bvh,
    refit_packed_bvh_bounds,
)
from .._execution_runtime import ExecutionGroup
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import (
    orient2d,
    orient3d,
    PredicateMode,
    PredicateResult,
    resolve_host_predicate_mode,
    segment_intersections_2d,
)
from .._meshcore import (
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
    triangle_intersection_classes,
)
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_geometry import (
    cell_geometry_preparation_query_upper,
    CellVertexGeometryElement,
    RestrictedCellGeometryElement,
)
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_enclosure import (
    coordinate_polynomials,
    polynomial_bounds,
    restrict_coordinates,
)
from ..discretization._exact_plc_geometry import translated_plc_layout_matches
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._mapped_locator import PreparedMappedCellLocator
from ..discretization._polyhedral_locator import PreparedPolyhedralCellLocator
from ..discretization._reference_cell import reference_cell_topology
from ..discretization._simplicial_locator import LocatedCellMap
from ..discretization._view_support import (
    mapped_mesh_support_query,
    MappedCellBoundaryMap,
    MappedMeshSupportKernel,
)
from ..discretization._views import transpose_duality_evidence
from ..discretization.fem import (
    FiniteElementDiscretization,
    FiniteElementFieldReconstructionKernel,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    FiniteElementSpec,
    lagrange_element,
    prepare_finite_element_field_reconstruction,
    PreparedFiniteElementCellMap,
)
from ..discretization.finite_volume._field_view import (
    UnstructuredFiniteVolumeFieldReconstructionKernel,
)
from ..discretization.finite_volume._triangle_fv import TriangleFiniteVolumeDiscretization
from ..geometry._interval_enclosure import Interval, interval_add, interval_multiply
from ..geometry.brep._projection_contracts import BRepProjectionStatus
from ..geometry.brep._query import BRepContainmentResult, PreparedBRepQuery
from ..geometry.brep._query_types import BRepQueryBudget, BRepQueryResourceError
from ..lifecycle import (
    commit_composition_rebind,
    Composition,
    CompositionEntry,
    CompositionRebind,
    CompositionRebindReceipt,
    CompositionTransport,
)
from ..linalg import SmallLinearSolvePlan, solve_small_linear
from ..linalg._small_batched import (
    exact_small_linear_action_bounds,
    prepare_exact_small_linear_actions,
)
from ..typing import checked, parse
from ._assembly import MeshAssembly
from ._coupling import (
    CouplingSearchEvidence,
    CouplingSearchStatus,
    encoded_image_isometry,
    MeshCoupling,
    MeshCouplingKind,
    OversetCoupling,
    OversetValueAction,
)
from ._measurements import (
    measure_phase,
    NativeMeshingPhaseRecorder,
    phase_started,
    record_elapsed,
)
from ._mesh_part import MeshPart
from ._organization_records import MeshAttribute, MeshLabel, MeshPatch, MeshZone
from ._result import CellMeshingResult
from ._scope import MeshingScope


if TYPE_CHECKING:
    from . import _overset_exchange as overset_exchange_owner


type OversetDonorPriority = Literal["finest-donor", "part-priority"]

_OWNER = "phydrax.meshing.overset"
_CONNECTIVITY_ENTRY = "overset:connectivity"
_SIMPLICES = {2: "triangle", 3: "tetrahedron"}
_SEARCH_METHOD = "native-overset-authoritative-hole-cut-admissible-donor"
_UNBOUNDED_EXECUTION_COUNTER = (1 << 64) - 1


class OversetVertexStatus(IntEnum):
    """Blanking of one part vertex."""

    ACTIVE = 0
    RECEPTOR = 1
    HOLE = 2


class OversetCellStatus(IntEnum):
    """Blanking of one part cell; only `ACTIVE` cells are donor-admissible."""

    ACTIVE = 0
    FRINGE = 1
    HOLE = 2


def _part_entry(name: str, /) -> str:
    return f"overset:part:{name}"


# ----------------------------------------------------------------- declarations


@final
class OversetPartSpec(StrictModule, NonTrainableState):
    """Declare the overset role of one assembly part.

    `wall` names the facets (dimension `d - 1`) of the part's closed solid wall: it cuts
    holes in every other part and its vertices are protected in this part. `boundary`
    names the facets of the part's overset (artificial) outer boundary, whose vertices
    are receptors. Neither is inferred from names or coordinates. `priority` breaks
    donor ties under the policy (larger wins), and `owner_rank` is the execution rank
    responsible for the part's donor packets.
    `excluded` is a full-dimensional cell scope never admitted as a donor. `solid_query`
    optionally supplies native B-Rep or closed mapped domain membership used for
    classification. Possible wall/cell contacts lacking an intersection certificate are
    conservatively blanked and exposed as ambiguous cells.
    Paired `image_rotation` and `image_translation` define one registered support image,
    `x_world = R x_source + t`, for donors, receptors and walls. The certified source
    carrier, coordinate axes and stable entity identities remain the basis for
    source-bound maps; this pose is not a periodic quotient.
    """

    part_name: str = eqx.field(static=True)
    wall: MeshingScope | None
    boundary: MeshingScope | None
    excluded: MeshingScope | None
    solid_query: PreparedBRepQuery | contracts_owner.CompiledGeometry | None
    image_rotation: Array | None
    image_translation: Array | None
    solid_index: int = eqx.field(static=True)
    priority: int = eqx.field(static=True)
    owner_rank: int = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        part_name: str,
        /,
        *,
        wall: MeshingScope | None = None,
        boundary: MeshingScope | None = None,
        excluded: MeshingScope | None = None,
        solid_query: PreparedBRepQuery | contracts_owner.CompiledGeometry | None = None,
        solid_index: int = 0,
        priority: int = 0,
        owner_rank: int = 0,
        image_rotation: ArrayLike | None = None,
        image_translation: ArrayLike | None = None,
    ) -> None:
        name = str(part_name).strip()
        if not name:
            raise ValueError(
                "OversetPartSpec.part_name must be non-empty; got an empty name."
            )
        for scope, label in (
            (wall, "wall"),
            (boundary, "boundary"),
            (excluded, "excluded"),
        ):
            if scope is not None and scope.source_id != name:
                raise ValueError(
                    f"{'OversetPartSpec.' + label + '.source_id'} must match part_name; got a scope for another part."
                )
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise TypeError(
                "OversetPartSpec.priority must be an integer excluding bool; got a bool or non-integer value."
            )
        if isinstance(owner_rank, bool) or not isinstance(owner_rank, int):
            raise TypeError(
                "OversetPartSpec.owner_rank must be an integer excluding bool; got a bool or non-integer value."
            )
        if owner_rank < 0:
            raise ValueError(
                "OversetPartSpec.owner_rank must be non-negative; got a negative rank."
            )
        match solid_query:
            case contracts_owner.CompiledGeometry(kernel=MappedMeshSupportKernel()):
                pass
            case contracts_owner.CompiledGeometry():
                raise TypeError(
                    "OversetPartSpec.solid_query.kernel must be MappedMeshSupportKernel with whole mapped-cell support; got another compiled wall kernel."
                )
            case PreparedBRepQuery() | None:
                pass
            case _ as unknown:
                assert_never(unknown)
        if (
            isinstance(solid_index, bool)
            or not isinstance(solid_index, int)
            or solid_index < 0
        ):
            raise ValueError(
                "OversetPartSpec.solid_index must be a non-negative integer excluding bool; got a bool, non-integer, or negative value."
            )
        match solid_query:
            case None:
                invalid_solid_index = False
            case PreparedBRepQuery():
                invalid_solid_index = wall is None or solid_index >= len(
                    solid_query.solid_faces
                )
            case contracts_owner.CompiledGeometry():
                invalid_solid_index = wall is None or solid_index != 0
            case _ as unknown:
                assert_never(unknown)
        if invalid_solid_index:
            raise ValueError(
                "OversetPartSpec.solid_query must pair with a protected wall and a solid_index supported by its query type; got a missing wall or unsupported solid index."
            )
        if (image_rotation is None) != (image_translation is None):
            raise ValueError(
                "OversetPartSpec.image_rotation must be provided together with image_translation or both must be None; got only one image component."
            )
        rotation, translation = None, None
        if image_rotation is not None and image_translation is not None:
            rotation = np.asarray(image_rotation, dtype=np.float64)
            translation = np.asarray(image_translation, dtype=np.float64)
            if (
                rotation.ndim != 2
                or rotation.shape[0] not in (2, 3)
                or rotation.shape[1] != rotation.shape[0]
                or translation.shape != (rotation.shape[0],)
                or not np.all(np.isfinite(rotation))
                or not np.all(np.isfinite(translation))
            ):
                raise ValueError(
                    "OversetPartSpec.image_rotation must be a finite 2D or 3D square matrix paired with an ambient-dimensional image_translation; got invalid ranks, shapes, dimensions, or coefficients."
                )
            if not np.allclose(
                rotation.T @ rotation,
                np.eye(rotation.shape[0], dtype=np.float64),
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError(
                    "OversetPartSpec.image_rotation must satisfy the Euclidean Gram identity R.T @ R = I within absolute tolerance 1e-12; got coefficients that fail the isometry check."
                )
        self.part_name = name
        self.wall = wall
        self.boundary = boundary
        self.excluded = excluded
        self.solid_query, self.solid_index = solid_query, solid_index
        self.image_rotation = (
            None if rotation is None else jax.device_put(np.asarray(rotation))
        )
        self.image_translation = (
            None if translation is None else jax.device_put(np.asarray(translation))
        )
        self.priority = priority
        self.owner_rank = owner_rank
        self.spec_id = canonical_fingerprint(
            {
                "kind": "overset-part-spec",
                "part": name,
                "wall": None if wall is None else wall.scope_id,
                "boundary": None if boundary is None else boundary.scope_id,
                "excluded": None if excluded is None else excluded.scope_id,
                "solid_query": _solid_query_id(solid_query),
                "solid_index": solid_index,
                "priority": priority,
                "owner_rank": owner_rank,
                "image": array_tree_fingerprint((rotation, translation)),
            }
        )

    @property
    def image_isometry_exact(self) -> bool:
        """Encoded Euclidean Gram proof; never a source/group equivalence claim."""
        return self.image_rotation is None or encoded_image_isometry(
            np.asarray(self.image_rotation)
        )

    @checked
    def rebound(
        self,
        part: MeshPart,
        /,
        *,
        solid_query: PreparedBRepQuery | contracts_owner.CompiledGeometry | None = None,
    ) -> OversetPartSpec:
        """Rebind the wall and boundary scopes to a moved revision of the part."""
        if part.name != self.part_name:
            raise ValueError(
                "OversetPartSpec.rebound.part.name must equal OversetPartSpec.part_name; got a revision of another part."
            )
        return OversetPartSpec(
            self.part_name,
            wall=None
            if self.wall is None
            else part.scope(
                self.wall.entity_dimension,
                self.wall.entity_ids,
                entity_set_id=self.wall.entity_set_id,
            ),
            boundary=None
            if self.boundary is None
            else part.scope(
                self.boundary.entity_dimension,
                self.boundary.entity_ids,
                entity_set_id=self.boundary.entity_set_id,
            ),
            excluded=None
            if self.excluded is None
            else part.scope(
                self.excluded.entity_dimension,
                self.excluded.entity_ids,
                entity_set_id=self.excluded.entity_set_id,
            ),
            solid_query=self.solid_query if solid_query is None else solid_query,
            solid_index=self.solid_index,
            priority=self.priority,
            owner_rank=self.owner_rank,
            image_rotation=self.image_rotation,
            image_translation=self.image_translation,
        )


def _image_points(
    spec: OversetPartSpec, points: np.ndarray, /, *, inverse: bool = False
) -> np.ndarray:
    """Evaluate the authored source/world pose without replacing its carrier."""
    if spec.image_rotation is None or spec.image_translation is None:
        return points
    rotation, translation = (
        np.asarray(spec.image_rotation),
        np.asarray(spec.image_translation),
    )
    if not inverse:
        return points @ rotation.T + translation
    if points.size == 0:
        return points
    dimension = rotation.shape[0]
    right = (points - translation).reshape((-1, dimension)).T
    solved = solve_small_linear(SmallLinearSolvePlan(dimension), rotation, right)
    if not bool(np.asarray(solved.successful)):
        raise MeshingFailure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "OversetPartSpec.image_rotation must yield a successful native pullback solve; got a failed linear solve.",
            stage="image-query",
            achieved=(
                ("image_solve_status", float(np.asarray(solved.status))),
                ("image_solve_condition", float(np.asarray(solved.condition_estimate))),
            ),
        )
    return np.asarray(solved.value).T.reshape(points.shape)


def _relative_image_system(
    source: OversetPartSpec,
    target: OversetPartSpec,
    dimension: int,
    /,
) -> (
    tuple[
        tuple[tuple[Fraction, ...], ...],
        tuple[tuple[Fraction, ...], ...],
    ]
    | None
):
    """Return the authored source-to-target coefficient system, or an identity."""
    if source.image_rotation is None and target.image_rotation is None:
        return None
    source_rotation = (
        np.eye(dimension, dtype=np.float64)
        if source.image_rotation is None
        else np.asarray(source.image_rotation)
    )
    target_rotation = (
        np.eye(dimension, dtype=np.float64)
        if target.image_rotation is None
        else np.asarray(target.image_rotation)
    )
    source_translation = (
        np.zeros(dimension, dtype=np.float64)
        if source.image_translation is None
        else np.asarray(source.image_translation)
    )
    target_translation = (
        np.zeros(dimension, dtype=np.float64)
        if target.image_translation is None
        else np.asarray(target.image_translation)
    )
    if np.array_equal(source_rotation, target_rotation) and np.array_equal(
        source_translation, target_translation
    ):
        return None
    matrix = tuple(
        tuple(Fraction(float(value)) for value in row) for row in target_rotation
    )
    right = tuple(
        tuple(Fraction(float(value)) for value in source_rotation[axis])
        + (
            Fraction(float(source_translation[axis]))
            - Fraction(float(target_translation[axis])),
        )
        for axis in range(dimension)
    )
    return matrix, right


def _relative_image_actions(
    source: OversetPartSpec,
    target: OversetPartSpec,
    dimension: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Fraction, ...], ...] | None:
    """Return every requested source-to-target action using encoded-coefficient arithmetic
    on the authored binary64 maps.
    """
    system = _relative_image_system(source, target, dimension)
    if system is None:
        return None
    solved = prepare_exact_small_linear_actions(*system, budget=budget)
    if not solved.successful or solved.actions is None:
        raise ValueError(
            "_relative_image_actions.coordinate_map must be invertible; got an unsuccessful solve or missing actions."
        )
    return solved.actions


def _relative_image(
    source: OversetPartSpec,
    target: OversetPartSpec,
    dimension: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    actions = _relative_image_actions(source, target, dimension, budget=budget)
    if actions is None:
        return None, None
    values = np.asarray(
        [[float(value) for value in row] for row in actions], dtype=np.float64
    )
    return values[:, :dimension], values[:, dimension]


def _relative_points(
    source: OversetPartSpec, target: OversetPartSpec, points: np.ndarray, /
) -> np.ndarray:
    if source.image_rotation is None and target.image_rotation is None:
        return points
    if (
        source.image_rotation is not None
        and target.image_rotation is not None
        and source.image_translation is not None
        and target.image_translation is not None
        and np.array_equal(source.image_rotation, target.image_rotation)
        and np.array_equal(source.image_translation, target.image_translation)
    ):
        return points
    return _image_points(target, _image_points(source, points), inverse=True)


def _image_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    source: OversetPartSpec,
    target: OversetPartSpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Interval:
    """Return outward bounds for the entire source using encoded-coefficient arithmetic for
    encoded-map solve actions.
    """
    dimension = lower.shape[-1]
    actions = _relative_image_actions(source, target, dimension, budget=budget)
    if actions is None:
        return lower, upper
    low, high = np.zeros_like(lower), np.zeros_like(upper)
    for source_axis in range(dimension):
        coefficient_low = np.asarray(
            [
                fraction_endpoint(row[source_axis], upper=False, saturate=True)
                for row in actions
            ],
            dtype=np.float64,
        )
        coefficient_high = np.asarray(
            [
                fraction_endpoint(row[source_axis], upper=True, saturate=True)
                for row in actions
            ],
            dtype=np.float64,
        )
        term = interval_multiply(
            (coefficient_low, coefficient_high),
            (lower[..., source_axis, None], upper[..., source_axis, None]),
        )
        low, high = interval_add((low, high), term)
    translation_low = np.asarray(
        [
            fraction_endpoint(row[dimension], upper=False, saturate=True)
            for row in actions
        ],
        dtype=np.float64,
    )
    translation_high = np.asarray(
        [fraction_endpoint(row[dimension], upper=True, saturate=True) for row in actions],
        dtype=np.float64,
    )
    return interval_add((low, high), (translation_low, translation_high))


@final
class OversetPolicy(StrictModule, NonTrainableState):
    """Specify fringe depth, donor resolution and resource bounds for an overset assembly.

    `donor_priority="finest-donor"` prefers the containing donor cell of smallest
    measure, then the larger part priority; `"part-priority"` reverses the first two
    keys. Remaining ties are broken by part name and donor cell ID.
    `maximum_wall_candidate_pairs` is one shared wall allowance for ray/facet pairs,
    cell/wall broad-phase pairs and native B-Rep membership-query operations;
    `maximum_donor_candidate_pairs` bounds worst-case prepared point/cell routes across
    competing parts before allocation. Exhaustion never publishes a partial
    classification or silently ignores an unresolved competitor.
    """

    fringe_layers: int = eqx.field(static=True)
    donor_priority: OversetDonorPriority = eqx.field(static=True)
    location_policy: simplicial_locator_owner.SimplicialLocationPolicy | None
    maximum_wall_candidate_pairs: int = eqx.field(static=True)
    maximum_donor_candidate_pairs: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        fringe_layers: int = 2,
        donor_priority: OversetDonorPriority = "finest-donor",
        location_policy: simplicial_locator_owner.SimplicialLocationPolicy | None = None,
        maximum_wall_candidate_pairs: int = 50_000_000,
        maximum_donor_candidate_pairs: int = 50_000_000,
    ) -> None:
        for value, label in (
            (fringe_layers, "fringe_layers"),
            (maximum_wall_candidate_pairs, "maximum_wall_candidate_pairs"),
            (maximum_donor_candidate_pairs, "maximum_donor_candidate_pairs"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{'OversetPolicy.' + label} must be an integer excluding bool; got a bool or non-integer value."
                )
            if value < 1:
                raise ValueError(
                    f"{'OversetPolicy.' + label} must be an integer greater than zero; got a value below one."
                )
        priority = parse(donor_priority, OversetDonorPriority, "donor_priority")
        self.fringe_layers = fringe_layers
        self.donor_priority = priority
        self.location_policy = location_policy
        self.maximum_wall_candidate_pairs = maximum_wall_candidate_pairs
        self.maximum_donor_candidate_pairs = maximum_donor_candidate_pairs
        self.policy_id = canonical_fingerprint(
            {
                "kind": "overset-policy",
                "fringe_layers": fringe_layers,
                "donor_priority": priority,
                "location": None
                if location_policy is None
                else location_policy.policy_id,
                "maximum_wall_candidate_pairs": maximum_wall_candidate_pairs,
                "maximum_donor_candidate_pairs": maximum_donor_candidate_pairs,
            }
        )

    def native_limits(self) -> NativeExecutionLimits:
        """Preserve unbounded mechanical axes; pair limits remain domain-owned."""
        return NativeExecutionLimits(
            _UNBOUNDED_EXECUTION_COUNTER,
            _UNBOUNDED_EXECUTION_COUNTER,
            _UNBOUNDED_EXECUTION_COUNTER,
            _UNBOUNDED_EXECUTION_COUNTER,
            inf,
        )


# -------------------------------------------------------------------- evidence


@final
class OversetPartBlanking(StrictModule, NonTrainableState):
    """Store hole/fringe classification for one part in mesh row order.

    `vertex_status`/`cell_status` hold `OversetVertexStatus`/`OversetCellStatus` codes
    and `fringe_layer` the receptor layer (0 for non-receptors). `wall_contact` marks
    boundary vertices determined by the wall query, using certain on-wall predicates for
    PLC walls; `ambiguous_cells` marks blanked cuts without an enclosed vertex,
    including possible whole-source-box contacts lacking an intersection proof; and
    `protected_conflict` marks own-wall vertices that are not active.
    """

    vertex_ids: Array
    vertex_status: Array
    fringe_layer: Array
    wall_contact: Array
    protected_conflict: Array
    cell_ids: Array
    cell_status: Array
    ambiguous_cells: Array
    part_name: str = eqx.field(static=True)
    part_id: str = eqx.field(static=True)
    blanking_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        part: MeshPart,
        vertex_status: np.ndarray,
        fringe_layer: np.ndarray,
        wall_contact: np.ndarray,
        protected_conflict: np.ndarray,
        cell_status: np.ndarray,
        ambiguous_cells: np.ndarray,
        /,
    ) -> None:
        mesh = _cell_result(part).mesh
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        cell_ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
        )
        arrays = (
            np.asarray(vertex_status, dtype=np.int8),
            np.asarray(fringe_layer, dtype=np.int32),
            np.asarray(wall_contact, dtype=np.bool_),
            np.asarray(protected_conflict, dtype=np.bool_),
            np.asarray(cell_status, dtype=np.int8),
            np.asarray(ambiguous_cells, dtype=np.bool_),
        )
        if any(value.shape != vertex_ids.shape for value in arrays[:4]) or any(
            value.shape != cell_ids.shape for value in arrays[4:]
        ):
            raise ValueError(
                "OversetPartBlanking.records must hold one status record per vertex_id and cell_id; got a vertex or cell array shape mismatch."
            )
        self.vertex_ids = jax.device_put(np.asarray(vertex_ids))
        self.vertex_status = jax.device_put(np.asarray(arrays[0]))
        self.fringe_layer = jax.device_put(np.asarray(arrays[1]))
        self.wall_contact = jax.device_put(np.asarray(arrays[2]))
        self.protected_conflict = jax.device_put(np.asarray(arrays[3]))
        self.cell_ids = jax.device_put(np.asarray(cell_ids))
        self.cell_status = jax.device_put(np.asarray(arrays[4]))
        self.ambiguous_cells = jax.device_put(np.asarray(arrays[5]))
        self.part_name = part.name
        self.part_id = part.part_id
        self.blanking_id = canonical_fingerprint(
            {
                "kind": "overset-part-blanking",
                "part": part.part_id,
                "arrays": array_tree_fingerprint(arrays),
            }
        )

    def vertex_ids_with(self, status: OversetVertexStatus, /) -> np.ndarray:
        """Stable IDs of the vertices with one status, in ascending order."""
        if not isinstance(status, OversetVertexStatus):
            raise TypeError(
                "OversetPartBlanking.vertex_ids_with.status must be OversetVertexStatus; got another status type."
            )
        mask = np.asarray(self.vertex_status) == int(status)
        return np.sort(np.asarray(self.vertex_ids)[mask])

    def cell_ids_with(self, status: OversetCellStatus, /) -> np.ndarray:
        """Stable IDs of the cells with one status, in ascending order."""
        if not isinstance(status, OversetCellStatus):
            raise TypeError(
                "OversetPartBlanking.cell_ids_with.status must be OversetCellStatus; got another status type."
            )
        mask = np.asarray(self.cell_status) == int(status)
        return np.sort(np.asarray(self.cell_ids)[mask])


@final
class OversetReceptorEvidence(StrictModule, NonTrainableState):
    """Donor outcome of every receptor of one part, in ascending stable-ID order.

    `status` holds `CouplingSearchStatus` codes: `OUTSIDE` (no donor part covers
    the receptor), `EXCLUDED_DONOR` (only blanked or fringe donor cells contain
    it), `RESOURCE_EXCEEDED` and `UNRESOLVED` (inverse-map failure) are orphans.
    `donor_parts` indexes the assembly parts (-1 for orphans), `donor_cells` holds the donor
    cell stable ID, `candidate_parts` the number of parts with an admissible
    containing cell, `residuals` the inverse-map residual and `donor_measures`
    the measure of the selected donor cell.
    """

    receptor_ids: Array
    receptor_rows: Array
    status: Array
    donor_parts: Array
    donor_cells: Array
    candidate_parts: Array
    residuals: Array
    donor_measures: Array
    part_name: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        part_name: str,
        receptor_ids: np.ndarray,
        receptor_rows: np.ndarray,
        status: np.ndarray,
        donor_parts: np.ndarray,
        donor_cells: np.ndarray,
        candidate_parts: np.ndarray,
        residuals: np.ndarray,
        donor_measures: np.ndarray,
        /,
    ) -> None:
        arrays = (
            np.asarray(receptor_ids, dtype=np.int64),
            np.asarray(receptor_rows, dtype=np.int32),
            np.asarray(status, dtype=np.int32),
            np.asarray(donor_parts, dtype=np.int32),
            np.asarray(donor_cells, dtype=np.int64),
            np.asarray(candidate_parts, dtype=np.int32),
            np.asarray(residuals, dtype=np.float64),
            np.asarray(donor_measures, dtype=np.float64),
        )
        count = arrays[0].shape
        if len(count) != 1 or any(value.shape != count for value in arrays):
            raise ValueError(
                "OversetReceptorEvidence.records must hold one record per receptor_id; got inconsistent array lengths or a non-vector receptor ID array."
            )
        if np.any(np.diff(arrays[0]) <= 0):
            raise ValueError(
                "OversetReceptorEvidence.receptor_ids must be strictly ascending and unique; got a duplicate or descending stable ID."
            )
        (
            self.receptor_ids,
            self.receptor_rows,
            self.status,
            self.donor_parts,
            self.donor_cells,
            self.candidate_parts,
            self.residuals,
            self.donor_measures,
        ) = (jax.device_put(np.asarray(value)) for value in arrays)
        self.part_name = str(part_name)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "overset-receptor-evidence",
                "part": self.part_name,
                "arrays": array_tree_fingerprint(arrays),
            }
        )

    @property
    def found(self) -> Array:
        return self.status == int(CouplingSearchStatus.FOUND)

    @property
    def orphan_ids(self) -> np.ndarray:
        return np.asarray(self.receptor_ids)[~np.asarray(self.found)]


@final
class OversetDonorPacket(StrictModule, NonTrainableState):
    """Describe receptors served by one donor block, keyed by scientific IDs.

    `receptor_slots` index the receptor evidence in ascending stable-ID order.
    `donor_block` names the inverse chart used for the donor, and `donor_cell_rows` are
    local to that block; `donor_cells` retain source scientific cell IDs. Query points,
    both part revisions, block identity and owner ranks bind the packet. Geometry-only
    packets do not declare a synthetic interpolation family.
    `points` are donor-source query sites, `target_points` are receptor sites in the
    target's stored source chart, and `receptor_points` are registered-world sites.
    `rotation`/`translation` map donor-source into target-source axes.
    """

    receptor_ids: Array
    receptor_slots: Array
    points: Array
    receptor_points: Array
    target_points: Array
    rotation: Array | None
    translation: Array | None
    donor_cells: Array
    donor_cell_rows: Array
    donor_block: str = eqx.field(static=True)
    receptor_part: str = eqx.field(static=True)
    donor_part: str = eqx.field(static=True)
    receptor_rank: int = eqx.field(static=True)
    donor_rank: int = eqx.field(static=True)
    receptor_revision: str = eqx.field(static=True)
    donor_revision: str = eqx.field(static=True)
    packet_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        receptor_part: OversetPartSpec,
        donor_part: OversetPartSpec,
        receptor_ids: np.ndarray,
        receptor_slots: np.ndarray,
        points: np.ndarray,
        donor_cells: np.ndarray,
        donor_cell_rows: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        receptor_revision: str,
        donor_revision: str,
        donor_block: str,
        receptor_points: np.ndarray | None = None,
        target_points: np.ndarray | None = None,
    ) -> None:
        arrays = (
            np.asarray(receptor_ids, dtype=np.int64),
            np.asarray(receptor_slots, dtype=np.int32),
            np.asarray(points, dtype=np.float64),
            np.asarray(donor_cells, dtype=np.int64),
            np.asarray(donor_cell_rows, dtype=np.int32),
        )
        count = arrays[0].shape[0]
        if (
            count == 0
            or arrays[1].shape != (count,)
            or arrays[2].ndim != 2
            or arrays[2].shape[0] != count
            or arrays[3].shape != (count,)
            or arrays[4].shape != (count,)
        ):
            raise ValueError(
                "OversetDonorPacket.records must contain at least one served receptor and one aligned record in every packet array; got an empty packet or mismatched array shape."
            )
        if (
            np.any(np.diff(arrays[0]) <= 0)
            or np.any(np.diff(arrays[1]) <= 0)
            or np.any(arrays[0] < 0)
            or np.any(arrays[1] < 0)
            or np.any(arrays[3] < 0)
            or np.any(arrays[4] < 0)
            or not np.all(np.isfinite(arrays[2]))
            or not receptor_revision
            or not donor_revision
        ):
            raise ValueError(
                "OversetDonorPacket.identities must use ascending unique non-negative receptor and donor IDs, non-negative block and local IDs, finite points, and non-empty revision IDs; got an ordering, range, finiteness, or revision violation."
            )
        world = (
            _image_points(donor_part, arrays[2])
            if receptor_points is None
            else np.asarray(receptor_points, dtype=np.float64)
        )
        target = (
            _image_points(receptor_part, world, inverse=True)
            if target_points is None
            else np.asarray(target_points, dtype=np.float64)
        )
        if (
            world.shape != arrays[2].shape
            or target.shape != arrays[2].shape
            or not np.all(np.isfinite(world))
            or not np.all(np.isfinite(target))
            or not np.allclose(
                _image_points(donor_part, arrays[2]), world, rtol=0, atol=1e-10
            )
            or not np.allclose(
                _image_points(receptor_part, target), world, rtol=0, atol=1e-10
            )
        ):
            raise ValueError(
                "OversetDonorPacket.points must have matching shapes and map donor and receptor coordinates to the same finite registered-world points within absolute tolerance 1e-10; got a shape, finiteness, or image-coordinate mismatch."
            )
        rotation, translation = _relative_image(
            donor_part, receptor_part, arrays[2].shape[1], budget=budget
        )
        self.receptor_points, self.target_points = (
            jax.device_put(np.asarray(world)),
            jax.device_put(np.asarray(target)),
        )
        self.rotation = None if rotation is None else jax.device_put(np.asarray(rotation))
        self.translation = (
            None if translation is None else jax.device_put(np.asarray(translation))
        )
        (
            self.receptor_ids,
            self.receptor_slots,
            self.points,
            self.donor_cells,
            self.donor_cell_rows,
        ) = (jax.device_put(np.asarray(value)) for value in arrays)
        self.donor_block = str(donor_block)
        if not self.donor_block:
            raise ValueError(
                "OversetDonorPacket.donor_block must be non-empty; got an empty donor block name."
            )
        self.receptor_part = receptor_part.part_name
        self.donor_part = donor_part.part_name
        self.receptor_rank = receptor_part.owner_rank
        self.donor_rank = donor_part.owner_rank
        self.receptor_revision = str(receptor_revision)
        self.donor_revision = str(donor_revision)
        self.packet_id = canonical_fingerprint(
            {
                "kind": "overset-donor-packet",
                "receptor": receptor_part.part_name,
                "donor": donor_part.part_name,
                "donor_block": self.donor_block,
                "ranks": [receptor_part.owner_rank, donor_part.owner_rank],
                "receptor_revision": self.receptor_revision,
                "donor_revision": self.donor_revision,
                "specs": [receptor_part.spec_id, donor_part.spec_id],
                "arrays": array_tree_fingerprint(arrays),
                "world_target_points": array_tree_fingerprint((world, target)),
            }
        )


@final
class _PartGeometry(StrictModule, NonTrainableState):
    """Store source-bound charts and boxes enclosing every coordinate source in source cell
    order.
    """

    discretization: FiniteElementDiscretization | None
    locators: tuple[PreparedMappedCellLocator | PreparedPolyhedralCellLocator, ...]
    block_names: tuple[str, ...] = eqx.field(static=True)
    block_offsets: tuple[int, ...] = eqx.field(static=True)
    cell_measures: Array
    cell_vertices: Array
    cell_lower: Array
    cell_upper: Array
    affine_cells: Array
    wall_facets: Array | None
    wall_bvh: PackedBVH | None
    cell_ids: Array
    part_id: str = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)


class OversetConnectivityError(ValueError):
    """Report that overset connectivity fails coverage for at least one requested entity;
    see ``connectivity``.
    """

    def __init__(
        self,
        message: str,
        connectivity: OversetConnectivity,
        /,
        *,
        field_evidence: tuple[OversetReceptorEvidence, ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.connectivity = connectivity
        self.field_evidence = field_evidence


# ------------------------------------------------------------ host preparation
@contextmanager
def _field_packet_coordinate_phase(
    connectivity: OversetConnectivity, /
) -> Iterator[CoordinateEnclosureBudget]:
    """Bound every potential packet image solve from its authored coefficients."""
    work_upper = bytes_upper = 0
    for receptor, part in enumerate(connectivity.assembly.parts):
        dimension = _cell_result(part).mesh.ambient_dimension
        for donor, geometry in enumerate(connectivity.geometries):
            if donor == receptor:
                continue
            system = _relative_image_system(
                connectivity.specs[donor], connectivity.specs[receptor], dimension
            )
            if system is None:
                continue
            work, storage = exact_small_linear_action_bounds(*system)
            repetitions = len(geometry.block_names)
            work_upper += repetitions * work
            bytes_upper += repetitions * storage
    with HostBatchTicket(work_upper, bytes_upper=bytes_upper) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=work_upper,
            maximum_memory_bytes=bytes_upper,
        )
        with budget.activate():
            yield budget


@contextmanager
@checked
def coordinate_phase(
    *,
    scope: NativeExecutionScope,
    queries_upper: int = 0,
    cavity_upper: int = 0,
) -> Iterator[CoordinateEnclosureBudget]:
    """Reserve the remaining source envelope and settle before native hierarchy work."""
    allowance = scope.remaining()
    with scope.batch(
        allowance.remaining_work_units,
        queries_upper=min(queries_upper, allowance.remaining_geometry_queries),
        cavity_upper=min(cavity_upper, allowance.remaining_cavity_cells),
        bytes_upper=allowance.remaining_managed_bytes,
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=_UNBOUNDED_EXECUTION_COUNTER,
            maximum_memory_bytes=_UNBOUNDED_EXECUTION_COUNTER,
        )
        with budget.activate(), ticket.native_target():
            yield budget


def _mapped_solid_kernel(
    query: contracts_owner.CompiledGeometry, /
) -> MappedMeshSupportKernel:
    kernel = query.kernel
    if not isinstance(kernel, MappedMeshSupportKernel):
        raise TypeError(
            "_mapped_solid_kernel.kernel must be MappedMeshSupportKernel with whole source-bound cell support; got another kernel type."
        )
    return kernel


def _represented_linear_element(element: CellGeometryElement, /) -> bool:
    match element:
        case CellVertexGeometryElement():
            return True
        case FiniteElementSpec() | RestrictedCellGeometryElement():
            return element.degree == 1
        case _:
            raise TypeError(
                "_represented_linear_element.element.coordinate_source must be vertex coordinates or a finite-element coordinate source; got an unsupported represented-domain source."
            )


def _solid_query_id(
    query: PreparedBRepQuery | contracts_owner.CompiledGeometry | None, /
) -> str | None:
    if query is None:
        return None
    if isinstance(query, PreparedBRepQuery):
        return query.query_id
    kernel = _mapped_solid_kernel(query)
    return canonical_fingerprint(
        {
            "kind": "overset-mapped-solid",
            "source": kernel.locator.source_binding_id,
            "state": array_tree_fingerprint(query.state),
        }
    )


def _cell_result(part: MeshPart, /) -> CellMeshingResult:
    carrier = part.carrier
    if not isinstance(carrier, CellMeshingResult):
        raise TypeError(
            "_cell_result.carrier must be CellMeshingResult; got another carrier type."
        )
    mesh = carrier.mesh
    mesh.require_dense("native overset connectivity")
    supported = {
        2: {"triangle", "quadrilateral"},
        3: {"tetrahedron", "prism", "hexahedron", "pyramid", "polyhedron"},
    }
    if (
        mesh.ambient_dimension not in supported
        or mesh.topological_dimension != mesh.ambient_dimension
        or any(
            block.cell_kind not in supported[mesh.ambient_dimension]
            for block in mesh.blocks
        )
    ):
        raise ValueError(
            "_cell_result.mesh must use supported regular full-dimensional mapped or planar polyhedral cells; got an unsupported dimension, codimension, or cell kind."
        )
    return carrier


def _signs(result: PredicateResult, /) -> np.ndarray:
    if not bool(np.all(np.asarray(result.certain))):
        raise RuntimeError(
            "_signs.result.certain must mark every orientation predicate certain; got at least one uncertain predicate."
        )
    return np.asarray(result.signs, dtype=np.int8)


def _mode() -> PredicateMode:
    return resolve_host_predicate_mode(PredicateMode.EXACT)


@checked
def facet_rows(part: MeshPart, scope: MeshingScope, /) -> np.ndarray:
    """Vertex rows of the declared boundary facets of one part."""
    part.require_scope(scope)
    mesh = _cell_result(part).mesh
    dimension = mesh.ambient_dimension
    if scope.entity_dimension != dimension - 1:
        raise ValueError(
            "facet_rows.scope.entity_dimension must equal dimension - 1; got a non-facet scope."
        )
    identifiers = np.asarray(mesh.entity_set(dimension - 1).entity_ids)
    order = np.argsort(identifiers, kind="stable")
    rows = order[np.searchsorted(identifiers[order], np.asarray(scope.entity_ids))]
    connectivity = mesh.connectivity
    match connectivity:
        case PolygonalConnectivity():
            facets = np.asarray(connectivity.edges, dtype=np.int64)
            boundary = np.asarray(connectivity.boundary_edges)
        case PolyhedralConnectivity():
            offsets = np.asarray(connectivity.face_vertex_offsets)
            values = np.asarray(connectivity.face_vertex_values)
            loops = [values[offsets[row] : offsets[row + 1]] for row in rows]
            boundary = np.asarray(connectivity.face_neighbor) < 0
            if not np.all(boundary[rows]):
                raise ValueError(
                    "facet_rows.scope must contain only exterior mesh facets; got at least one interior facet."
                )
            width = max((len(loop) for loop in loops), default=0)
            return np.asarray(
                [np.pad(loop, (0, width - len(loop)), mode="edge") for loop in loops],
                dtype=np.int64,
            ).reshape((len(loops), width))
        case TetrahedralConnectivity() | HexahedralConnectivity():
            facets = np.asarray(connectivity.faces, dtype=np.int64)
            boundary = np.asarray(connectivity.boundary_faces)
        case _:
            raise TypeError(
                "facet_rows.connectivity must provide polygonal or spatial facet incidence; got an unsupported incidence kind."
            )
    if not np.all(boundary[rows]):
        raise ValueError(
            f"facet_rows.scope must contain only mesh boundary facets for part {part.name!r}; got at least one interior facet."
        )
    return facets[rows]


def _facet_loops(facets: np.ndarray, /) -> tuple[np.ndarray, ...]:
    return tuple(row[np.concatenate(([True], row[1:] != row[:-1]))] for row in facets)


def _require_closed(facets: np.ndarray, name: str, /) -> None:
    """Require a closed wall to have two incident facets at every represented ridge."""
    loops = _facet_loops(facets)
    if not loops:
        raise ValueError(
            f"_require_closed.facets must contain wall facets for {name!r}; got an empty wall."
        )
    if len(loops[0]) == 2:
        ridges = np.concatenate(loops).reshape((-1, 1))
    else:
        ridges = np.sort(
            np.concatenate([np.stack((row, np.roll(row, -1)), axis=1) for row in loops]),
            axis=1,
        )
    _, counts = np.unique(ridges, axis=0, return_counts=True)
    if np.any(counts != 2):
        raise ValueError(
            f"_require_closed.facets must form a closed wall for {name!r} with two incident facets per ridge; got an incidence count other than two."
        )


def _triangulate_loops(facets: np.ndarray, dimension: int, /) -> np.ndarray:
    loops = _facet_loops(facets)
    if dimension == 2:
        return np.asarray(loops, dtype=np.int64).reshape((-1, 2))
    return np.asarray(
        [row[[0, local, local + 1]] for row in loops for local in range(1, len(row) - 1)],
        dtype=np.int64,
    ).reshape((-1, 3))


def _box_bounds(corners: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    return np.min(corners, axis=1), np.max(corners, axis=1)


def _mapped_wall_bounds(
    query: contracts_owner.CompiledGeometry, /, *, budget: CoordinateEnclosureBudget
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose every represented atlas facet using encoded-coefficient arithmetic for
    source-map compositions.
    """

    kernel = _mapped_solid_kernel(query)
    mapping = kernel.atlas.mapping
    if not isinstance(mapping, MappedCellBoundaryMap):
        raise TypeError(
            "_mapped_wall_bounds.mapping must be MappedCellBoundaryMap for the full source facets; got another mapping type."
        )
    cell_map = mapping.cell_map
    topology = reference_cell_topology(cell_map.coordinate_element.cell_kind)
    coordinates = np.asarray(mapping.coordinates)
    routes = np.asarray(cell_map.coordinate_dofs)
    lower, upper = [], []
    sources = {}
    for cell, facet in zip(
        np.asarray(mapping.chart_cells), np.asarray(mapping.chart_facets), strict=True
    ):
        vertices = np.asarray(topology.vertices)[
            list(topology.entities[topology.dimension - 1][facet])
        ]
        origin = vertices[0]
        axes = (
            (vertices[1] - origin)[:, None]
            if topology.dimension == 2
            else np.stack(
                (vertices[1] - origin, vertices[2 if len(vertices) == 3 else 3] - origin),
                axis=1,
            )
        )
        if int(cell) not in sources:
            sources[int(cell)] = coordinate_polynomials(
                cell_map.coordinate_element, coordinates[routes[cell]], budget=budget
            )
        source = sources[int(cell)]
        expressions = (
            None
            if source is None
            else restrict_coordinates(
                source, cell_map.coordinate_element.cell_kind, origin, axes, budget=budget
            )
        )
        if expressions is None:
            # Whole-cell bounds conservatively enclose every incident facet.
            lower.append(np.asarray(kernel.locator.cell_lower)[cell])
            upper.append(np.asarray(kernel.locator.cell_upper)[cell])
        else:
            domain = "simplex" if len(vertices) in (2, 3) else "box"
            boxes = [
                polynomial_bounds(value, domain, topology.dimension - 1, budget=budget)
                for value in expressions
            ]
            lower.append([box[0] for box in boxes])
            upper.append([box[1] for box in boxes])
    return np.asarray(lower), np.asarray(upper)


@final
class _PartCharts(StrictModule, NonTrainableState):
    """Prepared donor charts and host cell data in mesh block order."""

    discretization: FiniteElementDiscretization | None
    locators: tuple[PreparedMappedCellLocator | PreparedPolyhedralCellLocator, ...]
    measures: np.ndarray
    vertices: np.ndarray
    affine: np.ndarray


@final
class _PartWall(StrictModule, NonTrainableState):
    """Protected wall facets and its prepared or refit hierarchy."""

    facets: Array | None
    bvh: PackedBVH | None


def _prepare_polyhedral_charts(
    result: CellMeshingResult, policy: OversetPolicy, scope: NativeExecutionScope, /
) -> _PartCharts:
    mesh = result.mesh
    elements, _, _ = result.geometry.resolve(mesh)
    if any(not _represented_linear_element(element) for element in elements):
        raise ValueError(
            "_prepare_polyhedral_charts.elements must use represented planar vertex geometry; got at least one unrepresented or non-linear element."
        )
    connectivity = mesh.connectivity
    match connectivity:
        case PolyhedralConnectivity():
            pass
        case _:
            raise TypeError(
                "_prepare_polyhedral_charts.polyhedra.incidence must provide full polyhedral face incidence; got unavailable incidence."
            )
    face_offsets = np.asarray(connectivity.face_vertex_offsets)
    face_values = np.asarray(connectivity.face_vertex_values)
    cell_offsets = np.asarray(connectivity.cell_face_offsets)
    cell_faces = np.asarray(connectivity.cell_face_values)
    cell_signs = np.asarray(connectivity.cell_face_sign_values)
    charts, offset = [], 0
    maximum_face_arity = int(np.max(np.diff(face_offsets), initial=0))
    with coordinate_phase(
        scope=scope, cavity_upper=max(1, 2 * maximum_face_arity + 4)
    ) as budget:
        for block in mesh.blocks:
            loops = []
            for cell in range(offset, offset + block.cell_count):
                first, last = cell_offsets[cell : cell + 2]
                faces = []
                for face, sign in zip(
                    cell_faces[first:last], cell_signs[first:last], strict=True
                ):
                    loop = face_values[face_offsets[face] : face_offsets[face + 1]]
                    faces.append(loop if sign > 0 else loop[::-1])
                loops.append(tuple(faces))
            chart_mesh = cell_mesh_owner.CellMesh.from_mixed_3d(
                mesh.coordinates,
                (),
                polyhedra={block.name: tuple(loops)},
                polyhedral_cell_global_ids={block.name: block.global_ids},
                vertex_global_ids=mesh.vertex_global_ids,
            )
            charts.append(
                PreparedPolyhedralCellLocator(
                    chart_mesh, policy.location_policy, budget=budget
                )
            )
            offset += block.cell_count
    locators = tuple(charts)
    measures = np.concatenate(
        [np.asarray(locator.cell_map.cell_volumes) for locator in locators]
    )
    width = max(locator.cell_map.coordinate_dofs.shape[1] for locator in locators)
    vertices = np.concatenate(
        [
            np.pad(
                np.asarray(locator.cell_map.coordinate_dofs),
                ((0, 0), (0, width - locator.cell_map.coordinate_dofs.shape[1])),
                mode="edge",
            )
            for locator in locators
        ]
    )
    return _PartCharts(
        None, locators, measures, vertices, np.ones(vertices.shape[0], dtype=np.bool_)
    )


def _prepare_mapped_charts(
    result: CellMeshingResult, policy: OversetPolicy, scope: NativeExecutionScope, /
) -> _PartCharts:
    mesh = result.mesh
    elements = {block.name: lagrange_element(block.cell_kind, 1) for block in mesh.blocks}
    discretization = FiniteElementPlan(
        mesh,
        (FiniteElementFieldSpec("state", elements),),
        coordinate_spec=result.geometry,
    ).prepare()
    coordinates = discretization.default_runtime.coordinates
    charts, measures, affine = [], [], []
    queries_upper = cell_geometry_preparation_query_upper(result.geometry) * len(
        mesh.blocks
    )
    with coordinate_phase(scope=scope, queries_upper=queries_upper) as budget:
        for block_index, block in enumerate(mesh.blocks):
            cell_map = PreparedFiniteElementCellMap(discretization, block_index)
            location = (
                policy.location_policy
                or simplicial_locator_owner.SimplicialLocationPolicy(
                    min(block.cell_count, 64), 16, 1
                )
            )
            charts.append(
                PreparedMappedCellLocator(cell_map, coordinates, location, budget=budget)
            )
            measures.append(
                np.asarray(discretization.block_geometries[0][block_index].measure)
            )
            planar = np.full(
                block.cell_count,
                cell_map.coordinate_element.degree == 1
                and not isinstance(
                    cell_map.coordinate_element, RestrictedCellGeometryElement
                ),
                dtype=np.bool_,
            )
            if mesh.ambient_dimension == 3:
                physical = np.asarray(mesh.coordinates)[np.asarray(block.vertices)]
                for face in reference_cell_topology(block.cell_kind).entities[2]:
                    if len(face) > 3:
                        corners = physical[:, list(face)]
                        planar &= (
                            _signs(
                                orient3d(
                                    corners[:, 0],
                                    corners[:, 1],
                                    corners[:, 2],
                                    corners[:, 3],
                                    mode=_mode(),
                                    budget=budget,
                                )
                            )
                            == 0
                        )
            affine.append(planar)
    width = max(block.vertices.shape[1] for block in mesh.blocks)
    vertices = np.concatenate(
        [
            np.pad(
                np.asarray(block.vertices),
                ((0, 0), (0, width - block.vertices.shape[1])),
                mode="edge",
            )
            for block in mesh.blocks
        ]
    )
    return _PartCharts(
        discretization,
        tuple(charts),
        np.concatenate(measures),
        vertices,
        np.concatenate(affine),
    )


def _prepare_part_wall(
    part: MeshPart,
    mesh: cell_mesh_owner.CellMesh,
    spec: OversetPartSpec,
    charts: _PartCharts,
    previous: _PartGeometry | None,
    scope: NativeExecutionScope,
    /,
) -> _PartWall:
    if spec.wall is None:
        return _PartWall(None, None)
    rows = facet_rows(part, spec.wall)
    _require_closed(rows, part.name)
    facets = None
    match spec.solid_query:
        case PreparedBRepQuery() as query:
            bounds = np.asarray(query.bounds)
            wall_lower, wall_upper = bounds[0:1], bounds[1:2]
        case contracts_owner.CompiledGeometry() as query:
            with coordinate_phase(scope=scope) as budget:
                wall_lower, wall_upper = _mapped_wall_bounds(query, budget=budget)
        case None:
            if charts.discretization is not None and not np.all(charts.affine):
                raise ValueError(
                    "_prepare_part_wall.wall_authority must be native B-Rep or a closed mapped domain for non-affine discretized charts; got a non-affine wall with only discretized chart geometry."
                )
            corners = np.asarray(mesh.coordinates, dtype=np.float64)[
                _triangulate_loops(rows, mesh.ambient_dimension)
            ]
            facets = jnp.asarray(corners)
            wall_lower, wall_upper = _box_bounds(corners)
        case _ as unknown:
            assert_never(unknown)
    previous_bvh = None if previous is None else previous.wall_bvh
    bvh = (
        refit_packed_bvh_bounds(previous_bvh, wall_lower, wall_upper)
        if previous_bvh is not None
        and previous_bvh.item_bbox_min.shape == wall_lower.shape
        else prepare_bvh(wall_lower, wall_upper, dtype=jnp.float64)
    )
    return _PartWall(facets, bvh)


@checked
def part_geometry(
    part: MeshPart,
    spec: OversetPartSpec,
    policy: OversetPolicy,
    previous: _PartGeometry | None,
    /,
    *,
    scope: NativeExecutionScope,
) -> _PartGeometry:
    result = _cell_result(part)
    mesh = result.mesh
    if spec.image_rotation is not None and spec.image_rotation.shape != (
        mesh.ambient_dimension,
        mesh.ambient_dimension,
    ):
        raise ValueError(
            "part_geometry.spec.image_rotation must have one row and column per mesh ambient dimension; got a shape that changes the source carrier coordinate axes."
        )
    polyhedral = any(block.cell_kind == "polyhedron" for block in mesh.blocks)
    charts = (
        _prepare_polyhedral_charts(result, policy, scope)
        if polyhedral
        else _prepare_mapped_charts(result, policy, scope)
    )
    measures = charts.measures
    if spec.image_rotation is not None and not spec.image_isometry_exact:
        matrix = tuple(
            tuple(Fraction(float(value)) for value in row)
            for row in np.asarray(spec.image_rotation)
        )
        with coordinate_phase(scope=scope) as budget:
            metric_action = prepare_exact_small_linear_actions(
                matrix, matrix, budget=budget
            )
        if not metric_action.successful:
            raise ValueError(
                "part_geometry.registered_support_image.coordinate_map must be invertible; got an unsuccessful metric action."
            )
        scale = float(abs(metric_action.determinant))
        if scale != 1.0:
            measures = measures * scale
    lower = np.concatenate(
        [np.asarray(locator.cell_lower) for locator in charts.locators]
    )
    upper = np.concatenate(
        [np.asarray(locator.cell_upper) for locator in charts.locators]
    )
    wall = _prepare_part_wall(part, mesh, spec, charts, previous, scope)
    offsets = tuple(
        int(value)
        for value in np.concatenate(
            ([0], np.cumsum([block.cell_count for block in mesh.blocks]))
        )
    )
    return _PartGeometry(
        discretization=charts.discretization,
        locators=charts.locators,
        block_names=tuple(block.name for block in mesh.blocks),
        block_offsets=offsets,
        cell_measures=jnp.asarray(measures),
        cell_vertices=jnp.asarray(charts.vertices),
        cell_lower=jnp.asarray(lower),
        cell_upper=jnp.asarray(upper),
        affine_cells=jnp.asarray(charts.affine),
        cell_ids=jnp.concatenate([block.global_ids for block in mesh.blocks]),
        wall_facets=wall.facets,
        wall_bvh=wall.bvh,
        part_id=part.part_id,
        spec_id=spec.spec_id,
    )


# ---------------------------------------------------------------- hole cutting


def _segment_crossings(
    points: np.ndarray, corners: np.ndarray, mode: PredicateMode, /
) -> tuple[np.ndarray, np.ndarray]:
    """+x ray crossings and closed contacts of points with wall segments.

    The query is perturbed to `p + (0, eps)`: an endpoint at the query height
    lies below it, so a ray through a wall vertex counts exactly one edge.
    """
    first, second = corners[:, 0], corners[:, 1]
    orientation = _signs(orient2d(first, second, points, mode=mode))
    first_above = first[:, 1] > points[:, 1]
    second_above = second[:, 1] > points[:, 1]
    crossing = (first_above != second_above) & (
        (second_above & (orientation > 0)) | (first_above & (orientation < 0))
    )
    within = np.all(
        (points >= np.minimum(first, second)) & (points <= np.maximum(first, second)),
        axis=1,
    )
    return crossing, (orientation == 0) & within


def _perturbed_side(
    first: np.ndarray, second: np.ndarray, point: np.ndarray, mode: PredicateMode, /
) -> np.ndarray:
    """Sign of `orient2d(first, second, point + (eps, eps^2))` in the (y, z) plane."""
    exact = _signs(orient2d(first, second, point, mode=mode))
    tie = np.where(
        second[:, 1] != first[:, 1],
        np.sign(first[:, 1] - second[:, 1]),
        np.sign(second[:, 0] - first[:, 0]),
    ).astype(np.int8)
    return np.where(exact != 0, exact, tie)


def _closed_triangle_contact(
    corners: np.ndarray, points: np.ndarray, mode: PredicateMode, /
) -> np.ndarray:
    """Whether coplanar points lie in the closed triangle (any faithful projection)."""
    contact = np.zeros(points.shape[0], dtype=np.bool_)
    for axes in ((0, 1), (1, 2), (2, 0)):
        a, b, c = (corners[:, index][:, list(axes)] for index in range(3))
        projected = points[:, list(axes)]
        area = _signs(orient2d(a, b, c, mode=mode))
        sides = [
            _signs(orient2d(start, end, projected, mode=mode)) * area
            for start, end in ((a, b), (b, c), (c, a))
        ]
        contact |= (area != 0) & np.all(np.stack(sides) >= 0, axis=0)
    return contact


def _triangle_crossings(
    points: np.ndarray, corners: np.ndarray, mode: PredicateMode, /
) -> tuple[np.ndarray, np.ndarray]:
    """+x ray crossings and closed contacts of points with wall triangles.

    The query is perturbed to `p + (0, eps, eps^2)` in the (y, z) projection, so
    the perturbed ray meets no wall edge or vertex; a projected-degenerate
    (ray-parallel) triangle can then never contain it.
    """
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    yz = [1, 2]
    projected = points[:, yz]
    sides = [
        _perturbed_side(start[:, yz], end[:, yz], projected, mode)
        for start, end in ((a, b), (b, c), (c, a))
    ]
    inside = (sides[0] == sides[1]) & (sides[1] == sides[2]) & (sides[0] != 0)
    normal = _signs(orient2d(a[:, yz], b[:, yz], c[:, yz], mode=mode))
    volume = _signs(orient3d(a, b, c, points, mode=mode))
    # The hit parameter is -volume / normal: positive exactly for opposite signs.
    crossing = inside & (volume != 0) & (volume == -normal)
    touching = np.zeros_like(crossing)
    coplanar = np.flatnonzero(volume == 0)
    if coplanar.size:
        touching[coplanar] = _closed_triangle_contact(
            corners[coplanar], points[coplanar], mode
        )
    return crossing, touching


def _wall_classification(
    points: np.ndarray,
    facets: np.ndarray,
    bvh: PackedBVH,
    budget: int,
    /,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return inside (odd crossing parity) and on-wall flags against one wall using certain
    geometric predicates.
    """
    count, dimension = points.shape
    inside = np.zeros(count, dtype=np.bool_)
    contact = np.zeros(count, dtype=np.bool_)
    lower = np.min(facets, axis=(0, 1))
    upper = np.max(facets, axis=(0, 1))
    candidates = np.flatnonzero(np.all((points >= lower) & (points <= upper), axis=1))
    if candidates.size == 0:
        return inside, contact, 0
    query = points[candidates]
    ray_end = query.copy()
    ray_end[:, 0] = upper[0]
    rays = prepare_bvh(query, ray_end, dtype=jnp.float64)
    rows_parts, facet_parts, total = [], [], 0
    for rows, items in bvh_overlap_pair_blocks(rays, bvh, include_touching=True):
        total += rows.size
        if total > budget:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "OversetPolicy.maximum_wall_candidate_pairs must admit every hole-cut candidate pair; got a candidate count above the request bound.",
                stage="hole-cut",
                requested=(("maximum_wall_candidate_pairs", float(budget)),),
                achieved=(("wall_candidate_pairs", float(total)),),
            )
        rows_parts.append(rows)
        facet_parts.append(items)
    if not rows_parts:
        return inside, contact, 0
    rows = np.concatenate(rows_parts)
    items = np.concatenate(facet_parts)
    classify = _segment_crossings if dimension == 2 else _triangle_crossings
    crossing, touching = classify(query[rows], facets[items], _mode())
    crossings = np.bincount(rows[crossing], minlength=query.shape[0])
    inside[candidates] = crossings % 2 == 1
    contact[candidates] = np.bincount(rows[touching], minlength=query.shape[0]) > 0
    return inside, contact, total


@final
class _Blanking(StrictModule, NonTrainableState):
    vertex_status: np.ndarray
    fringe_layer: np.ndarray
    wall_contact: np.ndarray
    protected_conflict: np.ndarray
    cell_status: np.ndarray
    ambiguous_cells: np.ndarray


def _fringe_layers(
    cells: np.ndarray,
    hole: np.ndarray,
    cell_hole: np.ndarray,
    boundary: np.ndarray,
    layers: int,
    /,
) -> np.ndarray:
    touched = np.zeros(hole.shape, dtype=np.bool_)
    touched[cells[cell_hole].reshape((-1,))] = True
    frontier = (touched | boundary) & ~hole
    layer = np.where(frontier, 1, 0).astype(np.int32)
    for depth in range(2, layers + 1):
        reached = np.any(frontier[cells], axis=1) & ~cell_hole
        grown = np.zeros(hole.shape, dtype=np.bool_)
        grown[cells[reached].reshape((-1,))] = True
        frontier = grown & ~hole & (layer == 0)
        layer[frontier] = depth
    return layer


def _cell_boundary(part: MeshPart, /) -> tuple[np.ndarray, np.ndarray]:
    """Return represented planar boundary triangles/segments with their source cell rows."""

    mesh = _cell_result(part).mesh
    connectivity = mesh.connectivity
    rows, owners = [], []
    if isinstance(connectivity, PolyhedralConnectivity):
        face_offsets = np.asarray(connectivity.face_vertex_offsets)
        face_values = np.asarray(connectivity.face_vertex_values)
        cell_offsets = np.asarray(connectivity.cell_face_offsets)
        cell_faces = np.asarray(connectivity.cell_face_values)
        for cell in range(connectivity.cell_count):
            for face in cell_faces[cell_offsets[cell] : cell_offsets[cell + 1]]:
                loop = face_values[face_offsets[face] : face_offsets[face + 1]]
                for local in range(1, len(loop) - 1):
                    rows.append(loop[[0, local, local + 1]])
                    owners.append(cell)
    else:
        offset = 0
        for block in mesh.blocks:
            topology = reference_cell_topology(block.cell_kind)
            for cell, vertices in enumerate(np.asarray(block.vertices)):
                for facet in topology.entities[topology.dimension - 1]:
                    loop = vertices[list(facet)]
                    pieces = (
                        [loop]
                        if mesh.ambient_dimension == 2
                        else [
                            loop[[0, local, local + 1]]
                            for local in range(1, len(loop) - 1)
                        ]
                    )
                    rows.extend(pieces)
                    owners.extend([offset + cell] * len(pieces))
            offset += block.cell_count
    return np.asarray(mesh.coordinates)[np.asarray(rows)], np.asarray(
        owners, dtype=np.int64
    )


def _cut_cells(
    part: MeshPart,
    facets: np.ndarray,
    wall_bvh: PackedBVH,
    budget: list[int],
    /,
) -> np.ndarray:
    """Prove represented planar wall/cell contacts, never bbox overlap alone."""
    boundary, owners = _cell_boundary(part)
    dimension = boundary.shape[-1]
    lower, upper = _box_bounds(boundary)
    bvh = prepare_bvh(lower, upper, dtype=jnp.float64)
    cut = np.zeros(
        sum(block.cell_count for block in _cell_result(part).mesh.blocks), dtype=np.bool_
    )
    for rows, items in bvh_overlap_pair_blocks(bvh, wall_bvh, include_touching=True):
        if rows.size > budget[0]:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "OversetPolicy.maximum_wall_candidate_pairs must admit every wall-cell contact pair; got a contact count above the remaining bound.",
                stage="hole-cut",
            )
        budget[0] -= rows.size
        first, second = boundary[rows], facets[items]
        if dimension == 2:
            result = segment_intersections_2d(
                first[:, 0], first[:, 1], second[:, 0], second[:, 1], mode=_mode()
            )
            if not np.all(result.certain):
                raise RuntimeError(
                    "_cut_cells.result.certain must mark every wall/cell contact predicate certain; got at least one uncertain decision."
                )
            contact = np.asarray(result.status) != 0
        else:
            classes, status = triangle_intersection_classes(first, second)
            if np.any(status != int(MeshcoreStatus.OK)):
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "overset_hole_cut.wall_contact must classify represented facets with successful native predicates; got a failed contact classification.",
                    stage="hole-cut",
                )
            contact = classes != 0
        cut[owners[rows[contact]]] = True
    return cut


def _possible_cuts(
    geometry: _PartGeometry, wall_bvh: PackedBVH, budget: list[int], /
) -> np.ndarray:
    """Conservative source-box overlap: explicitly ambiguous, not proven contact."""
    bvh = prepare_bvh(geometry.cell_lower, geometry.cell_upper, dtype=jnp.float64)
    possible = np.zeros(geometry.cell_ids.shape[0], dtype=np.bool_)
    for rows, _ in bvh_overlap_pair_blocks(bvh, wall_bvh, include_touching=True):
        if rows.size > budget[0]:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "OversetPolicy.maximum_wall_candidate_pairs must admit every possible wall-cut candidate; got a candidate count above the remaining bound.",
                stage="hole-cut",
            )
        budget[0] -= rows.size
        possible[rows] = True
    return possible


def _bounded_brep_wall_membership(
    query: PreparedBRepQuery,
    points: np.ndarray,
    solid: int,
    policy: OversetPolicy,
    remaining: list[int],
    /,
    *,
    scope: NativeExecutionScope,
) -> BRepContainmentResult:
    """Share measured host wall work with the explicit native request allowance."""
    allowance = scope.remaining()
    operations = min(remaining[0], allowance.remaining_work_units)
    point_limit = min(query.policy.maximum_points, allowance.remaining_geometry_queries)
    try:
        if allowance.status != MeshcoreStatus.OK:
            raise MeshcoreError(allowance.status, "execution_remaining")
        with scope.batch(
            allowance.remaining_work_units,
            queries_upper=min(points.shape[0], point_limit),
            bytes_upper=min(
                query.policy.maximum_scratch_bytes,
                allowance.remaining_managed_bytes,
            ),
        ) as ticket:
            coordinate_budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=_UNBOUNDED_EXECUTION_COUNTER,
                maximum_memory_bytes=_UNBOUNDED_EXECUTION_COUNTER,
            )
            query_budget = BRepQueryBudget(
                operations,
                maximum_points=point_limit,
                maximum_scratch_bytes=query.policy.maximum_scratch_bytes,
                ticket=ticket,
                coordinate_budget=coordinate_budget,
            )
            with coordinate_budget.activate(), ticket.native_target():
                try:
                    result = query.contains(
                        points,
                        solid=solid,
                        budget=query_budget,
                        coordinate_budget=coordinate_budget,
                    )
                finally:
                    remaining[0] -= query_budget.operations
    except BRepQueryResourceError as failure:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            str(failure),
            stage="hole-cut",
            requested=(
                (f"brep_{failure.resource}", float(failure.requested)),
                (
                    "maximum_wall_candidate_pairs",
                    float(policy.maximum_wall_candidate_pairs),
                ),
            ),
            achieved=(
                (f"brep_remaining_{failure.resource}", float(failure.remaining)),
                ("wall_work", float(policy.maximum_wall_candidate_pairs - remaining[0])),
            ),
        ) from failure
    except MeshcoreError as failure:
        if failure.status not in (
            MeshcoreStatus.CAPACITY_EXCEEDED,
            MeshcoreStatus.REFINEMENT_LIMIT,
            MeshcoreStatus.TIMEOUT,
        ):
            raise
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            str(failure),
            stage="hole-cut",
            achieved=(
                ("native_status", float(failure.status)),
                ("wall_work", float(policy.maximum_wall_candidate_pairs - remaining[0])),
            ),
        ) from failure
    return result


@final
class _ForeignWall(StrictModule, NonTrainableState):
    """Wall geometry in the receptor chart and query points in the wall chart."""

    facets: np.ndarray | None
    hierarchy: PackedBVH
    query_points: np.ndarray


@final
class _WallMembership(StrictModule, NonTrainableState):
    """Resolved wall membership and contact masks in receptor vertex order."""

    inside: np.ndarray
    touching: np.ndarray


@final
class _WallCuts(StrictModule, NonTrainableState):
    """Accumulated foreign-wall cut mask in receptor cell order."""

    cells: np.ndarray


def _prepare_foreign_wall(
    source: OversetPartSpec,
    target: OversetPartSpec,
    geometry: _PartGeometry,
    hierarchy: PackedBVH,
    points: np.ndarray,
    dimension: int,
    scope: NativeExecutionScope,
    /,
) -> _ForeignWall:
    facets = (
        None
        if geometry.wall_facets is None
        else np.asarray(geometry.wall_facets, dtype=np.float64)
    )
    if source.image_rotation is not None or target.image_rotation is not None:
        transformed_bounds: Interval | None = None
        with coordinate_phase(scope=scope) as coordinate_budget:
            rotation, translation = _relative_image(
                source, target, dimension, budget=coordinate_budget
            )
            if rotation is not None and translation is not None:
                if facets is not None:
                    facets = _relative_points(source, target, facets)
                transformed_bounds = _image_bounds(
                    np.asarray(hierarchy.item_bbox_min),
                    np.asarray(hierarchy.item_bbox_max),
                    source,
                    target,
                    budget=coordinate_budget,
                )
        if transformed_bounds is not None:
            hierarchy = prepare_bvh(*transformed_bounds, dtype=jnp.float64)
    return _ForeignWall(facets, hierarchy, _relative_points(target, source, points))


def _classify_discretized_wall(
    geometry: _PartGeometry,
    hierarchy: PackedBVH,
    query_points: np.ndarray,
    budget: list[int],
    /,
) -> _WallMembership:
    if geometry.wall_facets is None:
        raise ValueError(
            "_blank_part.geometry.wall_facets must accompany an explicit PLC wall hierarchy; got no wall facets."
        )
    inside, touching, used = _wall_classification(
        query_points, np.asarray(geometry.wall_facets), hierarchy, budget[0]
    )
    budget[0] -= used
    return _WallMembership(inside, touching)


def _classify_mapped_wall(
    query: contracts_owner.CompiledGeometry,
    query_points: np.ndarray,
    vertex_ids: Array,
    /,
) -> _WallMembership:
    classified = mapped_mesh_support_query(query, jnp.asarray(query_points))
    resolved = np.asarray(classified.successful) & np.asarray(
        classified.location.candidates_complete
    )
    if not np.all(resolved):
        raise MeshingFailure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "overset_hole_cut.mapped_membership must resolve every queried point; got unresolved wall membership.",
            stage="hole-cut",
            entity_ids=tuple(int(value) for value in np.asarray(vertex_ids)[~resolved]),
        )
    inside = np.asarray(classified.inside)
    # This margin describes the exterior boundary, not internal cell faces or
    # Euclidean distance to the wall.
    kernel = _mapped_solid_kernel(query)
    margin = np.asarray(kernel.boundary_field(query.state, jnp.asarray(query_points)))
    touching = inside & (np.abs(margin) <= kernel.locator.policy.reference_tolerance)
    return _WallMembership(inside, touching)


def _classify_brep_wall(
    query: PreparedBRepQuery,
    query_points: np.ndarray,
    point_dimension: int,
    vertex_ids: Array,
    solid_index: int,
    policy: OversetPolicy,
    budget: list[int],
    scope: NativeExecutionScope,
    /,
) -> _WallMembership:
    if point_dimension != 3:
        raise ValueError(
            "_blank_part.mesh.ambient_dimension must be three for native B-Rep solid classification; got a non-3D part."
        )
    classified = _bounded_brep_wall_membership(
        query, query_points, solid_index, policy, budget, scope=scope
    )
    exhausted = np.asarray(classified.resource_exhausted)
    if np.any(exhausted):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "overset_hole_cut.brep_membership must fit its source work bound; got exhausted query work.",
            stage="hole-cut",
            entity_ids=tuple(int(value) for value in np.asarray(vertex_ids)[exhausted]),
            achieved=(
                ("wall_work", float(policy.maximum_wall_candidate_pairs - budget[0])),
            ),
        )
    touching = np.asarray(classified.distances) <= query.tolerances.classifier
    resolved = np.asarray(classified.status) == int(BRepProjectionStatus.UNIQUE)
    if np.any(~resolved & ~touching):
        raise MeshingFailure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "overset_hole_cut.brep_membership must resolve every non-contact point; got unresolved hole classification.",
            stage="hole-cut",
            entity_ids=tuple(
                int(value) for value in np.asarray(vertex_ids)[~resolved & ~touching]
            ),
        )
    return _WallMembership(np.asarray(classified.inside), touching)


def _accumulate_wall_cuts(
    part: MeshPart,
    geometry: _PartGeometry,
    wall: _ForeignWall,
    cut: np.ndarray,
    budget: list[int],
    scope: NativeExecutionScope,
    /,
) -> _WallCuts:
    facets = wall.facets
    if facets is not None:
        corners = np.unique(facets.reshape((-1, facets.shape[-1])), axis=0)
        with coordinate_phase(scope=scope) as coordinate_budget:
            for block, locator in enumerate(geometry.locators):
                located = _locate_part_points(
                    locator, jnp.asarray(corners), budget=coordinate_budget
                )
                containing = np.asarray(located.candidate_cells)
                cut[containing[containing >= 0] + geometry.block_offsets[block]] = True
        if np.all(np.asarray(geometry.affine_cells)):
            cut |= _cut_cells(part, facets, wall.hierarchy, budget)
        else:
            cut |= _possible_cuts(geometry, wall.hierarchy, budget)
    else:
        cut |= _possible_cuts(geometry, wall.hierarchy, budget)
    return _WallCuts(cut)


def _finalize_blanking(
    part: MeshPart,
    spec: OversetPartSpec,
    geometry: _PartGeometry,
    policy: OversetPolicy,
    cells: np.ndarray,
    hole: np.ndarray,
    contact: np.ndarray,
    cut: np.ndarray,
    /,
) -> _Blanking:
    enclosing = np.any(hole[cells], axis=1)
    cell_hole = enclosing | cut
    if spec.excluded is not None:
        identifiers = np.asarray(geometry.cell_ids)
        cell_hole |= np.isin(identifiers, np.asarray(spec.excluded.entity_ids))
    boundary = np.zeros_like(hole)
    if spec.boundary is not None:
        boundary[facet_rows(part, spec.boundary).reshape((-1,))] = True
    protected = np.zeros_like(hole)
    if spec.wall is not None:
        protected[facet_rows(part, spec.wall).reshape((-1,))] = True
    layer = _fringe_layers(cells, hole, cell_hole, boundary, policy.fringe_layers)
    layer[protected & ~hole] = 0
    vertex_status = np.select(
        (hole, layer > 0),
        (int(OversetVertexStatus.HOLE), int(OversetVertexStatus.RECEPTOR)),
        int(OversetVertexStatus.ACTIVE),
    ).astype(np.int8)
    cell_status = np.select(
        (cell_hole, np.any(layer[cells] > 0, axis=1)),
        (int(OversetCellStatus.HOLE), int(OversetCellStatus.FRINGE)),
        int(OversetCellStatus.ACTIVE),
    ).astype(np.int8)
    return _Blanking(
        vertex_status,
        layer,
        contact,
        protected & (vertex_status != int(OversetVertexStatus.ACTIVE)),
        cell_status,
        cut & ~enclosing,
    )


def _blank_part(
    index: int,
    parts: tuple[MeshPart, ...],
    specs: tuple[OversetPartSpec, ...],
    geometries: tuple[_PartGeometry, ...],
    policy: OversetPolicy,
    budget: list[int],
    /,
    *,
    scope: NativeExecutionScope,
) -> _Blanking:
    part, spec = parts[index], specs[index]
    mesh = _cell_result(part).mesh
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.asarray(geometries[index].cell_vertices, dtype=np.int64)
    hole = np.zeros(points.shape[0], dtype=np.bool_)
    contact = np.zeros_like(hole)
    cut = np.zeros(cells.shape[0], dtype=np.bool_)
    for other, geometry in enumerate(geometries):
        if other == index or geometry.wall_bvh is None:
            continue
        source_hierarchy = geometry.wall_bvh
        wall = _prepare_foreign_wall(
            specs[other],
            spec,
            geometry,
            source_hierarchy,
            points,
            mesh.ambient_dimension,
            scope,
        )
        match specs[other].solid_query:
            case None:
                membership = _classify_discretized_wall(
                    geometry, source_hierarchy, wall.query_points, budget
                )
            case contracts_owner.CompiledGeometry() as query:
                membership = _classify_mapped_wall(
                    query, wall.query_points, mesh.vertex_global_ids
                )
            case PreparedBRepQuery() as query:
                membership = _classify_brep_wall(
                    query,
                    wall.query_points,
                    points.shape[1],
                    mesh.vertex_global_ids,
                    specs[other].solid_index,
                    policy,
                    budget,
                    scope,
                )
            case _ as unknown:
                assert_never(unknown)
        hole |= membership.inside | membership.touching
        contact |= membership.touching
        cut = _accumulate_wall_cuts(
            part, geometries[index], wall, cut, budget, scope
        ).cells
    return _finalize_blanking(
        part, spec, geometries[index], policy, cells, hole, contact, cut
    )


# ------------------------------------------------------------ donor resolution


@final
class _Donors(StrictModule, NonTrainableState):
    status: np.ndarray
    parts: np.ndarray
    cells: np.ndarray
    candidates: np.ndarray
    residuals: np.ndarray
    measures: np.ndarray
    barycentric: np.ndarray


def _admissible(blanking: OversetPartBlanking, /) -> Array:
    return jnp.asarray(blanking.cell_status) == int(OversetCellStatus.ACTIVE)


def _locate_part_points(
    locator: PreparedMappedCellLocator | PreparedPolyhedralCellLocator,
    points: Array,
    /,
    *,
    cell_mask: Array | None = None,
    budget: CoordinateEnclosureBudget,
) -> simplicial_locator_owner.CellLocationResult:
    """Share native polyhedral query work; standalone FE queries keep their owner."""
    match locator:
        case PreparedPolyhedralCellLocator():
            return locator.locate(
                points,
                cell_mask=cell_mask,
                budget=budget if budget.ticket.native_accounting else None,
            )
        case PreparedMappedCellLocator():
            return locator.locate(points, cell_mask=cell_mask)
        case unknown:
            assert_never(unknown)


@checked
def resolve_donors(
    points: np.ndarray,
    receptor_index: int,
    specs: tuple[OversetPartSpec, ...],
    geometries: tuple[_PartGeometry, ...],
    admissible: tuple[Array, ...],
    policy: OversetPolicy,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _Donors:
    """Deterministic admissible donor of every point among the other parts."""
    count, dimension = points.shape
    parts = len(specs)
    capacity = count * sum(
        min(locator.policy.maximum_candidates, locator.cell_map.cell_count)
        for index, geometry in enumerate(geometries)
        if index != receptor_index
        for locator in geometry.locators
    )
    if capacity > policy.maximum_donor_candidate_pairs:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "OversetPolicy.maximum_donor_candidate_pairs must admit the prepared point-cell route capacity; got a donor capacity above the request bound.",
            stage="donor-location",
            requested=(
                (
                    "maximum_donor_candidate_pairs",
                    float(policy.maximum_donor_candidate_pairs),
                ),
            ),
            achieved=(("donor_candidate_capacity", float(capacity)),),
        )
    missing = np.ones((count, parts), dtype=np.bool_)
    unresolved = np.zeros(count, dtype=np.bool_)
    resource_exceeded = np.zeros(count, dtype=np.bool_)
    scale = np.full((count, parts), np.inf)
    cells = np.full((count, parts), -1, dtype=np.int64)
    residual = np.full((count, parts), np.inf)
    barycentric = np.zeros((count, parts, dimension + 1))
    for index, geometry in enumerate(geometries):
        if index == receptor_index or count == 0:
            continue
        donor_points = _relative_points(specs[receptor_index], specs[index], points)
        query = jnp.asarray(donor_points)
        for block, locator in enumerate(geometry.locators):
            start, stop = geometry.block_offsets[block : block + 2]
            located = _locate_part_points(
                locator, query, cell_mask=admissible[index][start:stop], budget=budget
            )
            status = np.asarray(located.status)
            complete = np.asarray(located.candidates_complete)
            found = (
                status == int(simplicial_locator_owner.CellLocationStatus.LOCATED)
            ) & complete
            unresolved |= ~complete | (
                ~found
                & (status != int(simplicial_locator_owner.CellLocationStatus.OUTSIDE))
            )
            resource_exceeded |= status == int(
                simplicial_locator_owner.CellLocationStatus.RESOURCE_EXCEEDED
            )
            candidates = np.asarray(located.candidate_cells)
            valid = candidates >= 0
            safe = np.maximum(candidates, 0) + start
            measures = np.asarray(geometry.cell_measures)[safe]
            identifiers = np.asarray(geometry.cell_ids)[safe]
            order = np.lexsort((identifiers, measures, ~valid), axis=1)
            slot = order[:, 0]
            rows = candidates[np.arange(count), slot]
            flat = np.maximum(rows, 0) + start
            candidate_scale = np.asarray(geometry.cell_measures)[flat]
            old_safe = np.maximum(cells[:, index], 0)
            better = found & (
                missing[:, index]
                | (candidate_scale < scale[:, index])
                | (
                    (candidate_scale == scale[:, index])
                    & (
                        np.asarray(geometry.cell_ids)[flat]
                        < np.asarray(geometry.cell_ids)[old_safe]
                    )
                )
            )
            missing[found, index] = False
            cells[better, index] = flat[better]
            scale[better, index] = candidate_scale[better]
            reference = np.asarray(located.candidate_reference)[np.arange(count), slot]
            match locator:
                case PreparedPolyhedralCellLocator():
                    selected = locator.cell_map.evaluate(
                        locator.coordinates,
                        jnp.asarray(np.maximum(rows, 0)),
                        jnp.asarray(reference),
                        budget=budget if budget.ticket.native_accounting else None,
                    )
                case PreparedMappedCellLocator():
                    selected = locator.cell_map.evaluate(
                        locator.coordinates,
                        jnp.asarray(np.maximum(rows, 0)),
                        jnp.asarray(reference),
                    )
                case unknown:
                    assert_never(unknown)
            residual[better, index] = np.linalg.norm(
                np.asarray(selected.physical_points)[better] - donor_points[better],
                axis=1,
            )
            if (
                isinstance(locator, PreparedMappedCellLocator)
                and locator.cell_map.coordinate_element.cell_kind in _SIMPLICES.values()
            ):
                barycentric[better, index] = np.concatenate(
                    (
                        1.0 - np.sum(reference[better], axis=1, keepdims=True),
                        reference[better],
                    ),
                    axis=1,
                )
    priority = np.broadcast_to(
        -np.asarray([spec.priority for spec in specs], dtype=np.int64), (count, parts)
    )
    name_ranks = np.argsort(
        np.argsort(np.asarray([spec.part_name for spec in specs]), kind="stable"),
        kind="stable",
    )
    part_order = np.broadcast_to(name_ranks, (count, parts))
    match policy.donor_priority:
        case "finest-donor":
            first, second = scale, priority
        case "part-priority":
            first, second = priority, scale
        case unknown:
            assert_never(unknown)
    point_order = np.broadcast_to(np.arange(count)[:, None], (count, parts))
    order = np.lexsort(
        (
            np.where(missing, -1, cells).ravel(),
            part_order.ravel(),
            second.ravel(),
            first.ravel(),
            missing.ravel(),
            point_order.ravel(),
        )
    )
    chosen = order[::parts] % parts if parts else np.zeros(0, dtype=np.int64)
    rows = np.arange(count)
    # An unresolved competing part might contain a higher-priority donor;
    # another located part is not proof that the declared selection is exhaustive.
    found = ~missing[rows, chosen] & ~unresolved
    status = np.full(count, int(CouplingSearchStatus.OUTSIDE), dtype=np.int32)
    status[found] = int(CouplingSearchStatus.FOUND)
    orphan = np.flatnonzero(~found)
    if orphan.size:
        blocked = np.zeros(orphan.size, dtype=np.bool_)
        for index, geometry in enumerate(geometries):
            if index == receptor_index:
                continue
            donor_points = _relative_points(
                specs[receptor_index], specs[index], points[orphan]
            )
            for locator in geometry.locators:
                located = _locate_part_points(
                    locator, jnp.asarray(donor_points), budget=budget
                )
                complete = np.asarray(located.candidates_complete)
                location_status = np.asarray(located.status)
                blocked |= (
                    location_status
                    == int(simplicial_locator_owner.CellLocationStatus.LOCATED)
                ) & complete
                unresolved[orphan] |= ~complete | ~np.isin(
                    location_status,
                    (
                        int(simplicial_locator_owner.CellLocationStatus.LOCATED),
                        int(simplicial_locator_owner.CellLocationStatus.OUTSIDE),
                    ),
                )
                resource_exceeded[orphan] |= location_status == int(
                    simplicial_locator_owner.CellLocationStatus.RESOURCE_EXCEEDED
                )
        status[orphan] = np.select(
            (resource_exceeded[orphan], unresolved[orphan], blocked),
            (
                int(CouplingSearchStatus.RESOURCE_EXCEEDED),
                int(CouplingSearchStatus.UNRESOLVED),
                int(CouplingSearchStatus.EXCLUDED_DONOR),
            ),
            int(CouplingSearchStatus.OUTSIDE),
        )
    return _Donors(
        status,
        np.where(found, chosen, -1).astype(np.int32),
        np.where(found, cells[rows, chosen], -1),
        np.sum(~missing, axis=1).astype(np.int32),
        np.where(found, residual[rows, chosen], np.inf),
        np.where(found, scale[rows, chosen], np.inf),
        np.where(found[:, None], barycentric[rows, chosen], 0.0),
    )


def _receptor_evidence(
    index: int,
    parts: tuple[MeshPart, ...],
    specs: tuple[OversetPartSpec, ...],
    geometries: tuple[_PartGeometry, ...],
    blanking: tuple[OversetPartBlanking, ...],
    policy: OversetPolicy,
    /,
    previous: tuple[OversetPartBlanking, ...] | None = None,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[OversetReceptorEvidence, _Donors]:
    mesh = _cell_result(parts[index]).mesh
    ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    receptor = np.asarray(blanking[index].vertex_status) == int(
        OversetVertexStatus.RECEPTOR
    )
    admissible = tuple(_admissible(item) for item in blanking)
    if previous is not None:
        receptor |= (
            np.asarray(previous[index].vertex_status) == int(OversetVertexStatus.HOLE)
        ) & (np.asarray(blanking[index].vertex_status) != int(OversetVertexStatus.HOLE))
        admissible = tuple(
            current & (old.cell_status != int(OversetCellStatus.HOLE))
            for current, old in zip(admissible, previous, strict=True)
        )
    rows = np.flatnonzero(receptor)
    rows = rows[np.argsort(ids[rows], kind="stable")]
    points = np.asarray(mesh.coordinates, dtype=np.float64)[rows]
    donors = resolve_donors(
        points,
        index,
        specs,
        geometries,
        admissible,
        policy,
        budget=budget,
    )
    cell_ids = [np.asarray(geometry.cell_ids, dtype=np.int64) for geometry in geometries]
    donor_cells = np.asarray(
        [
            cell_ids[part][cell] if part >= 0 else -1
            for part, cell in zip(donors.parts, donors.cells, strict=True)
        ],
        dtype=np.int64,
    )
    evidence = OversetReceptorEvidence(
        parts[index].name,
        ids[rows],
        rows,
        donors.status,
        donors.parts,
        donor_cells,
        donors.candidates,
        donors.residuals,
        donors.measures,
    )
    return evidence, donors


def _donor_packets(
    index: int,
    parts: tuple[MeshPart, ...],
    specs: tuple[OversetPartSpec, ...],
    geometries: tuple[_PartGeometry, ...],
    evidence: OversetReceptorEvidence,
    donors: _Donors,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[OversetDonorPacket, ...]:
    """Return one source- and revision-bound packet per served donor block."""
    target_sites = np.asarray(_cell_result(parts[index]).mesh.coordinates)[
        np.asarray(evidence.receptor_rows)
    ]
    sites = _image_points(specs[index], target_sites)
    if sites.shape != (
        evidence.receptor_ids.shape[0],
        _cell_result(parts[index]).mesh.ambient_dimension,
    ):
        raise ValueError(
            "_donor_packets.sites must provide one ambient-coordinate point per receptor evidence ID; got a row-count or coordinate-dimension mismatch."
        )
    packets = []
    for donor, geometry in enumerate(geometries):
        for block, name in enumerate(geometry.block_names):
            start, stop = geometry.block_offsets[block : block + 2]
            slots = np.flatnonzero(
                (donors.parts == donor) & (donors.cells >= start) & (donors.cells < stop)
            )
            if slots.size:
                packets.append(
                    OversetDonorPacket(
                        specs[index],
                        specs[donor],
                        np.asarray(evidence.receptor_ids)[slots],
                        slots,
                        _relative_points(specs[index], specs[donor], target_sites[slots]),
                        np.asarray(geometry.cell_ids)[donors.cells[slots]],
                        donors.cells[slots] - start,
                        donor_block=name,
                        budget=budget,
                        receptor_revision=parts[index].part_id,
                        donor_revision=parts[donor].part_id,
                        receptor_points=sites[slots],
                        target_points=target_sites[slots],
                    )
                )
    return tuple(packets)


def _packets_and_overlays(
    index: int,
    parts: tuple[MeshPart, ...],
    specs: tuple[OversetPartSpec, ...],
    blanking: tuple[OversetPartBlanking, ...],
    geometries: tuple[_PartGeometry, ...],
    evidence: OversetReceptorEvidence,
    donors: _Donors,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[list[OversetDonorPacket], list[OversetCoupling]]:
    """Donor packets and the published P1 vertex overlays of one receptor part."""
    receptor = parts[index]
    hole_ids = blanking[index].vertex_ids_with(OversetVertexStatus.HOLE)
    hole_scope = receptor.scope(0, hole_ids) if hole_ids.size else None
    packets = list(
        _donor_packets(index, parts, specs, geometries, evidence, donors, budget=budget)
    )
    receptor_ids = np.asarray(evidence.receptor_ids)
    overlays = []
    for packet in packets:
        donor = next(
            row for row, part in enumerate(parts) if part.name == packet.donor_part
        )
        block_index = geometries[donor].block_names.index(packet.donor_block)
        block = _cell_result(parts[donor]).mesh.blocks[block_index]
        slots = np.asarray(packet.receptor_slots)
        donor_rows = np.asarray(packet.donor_cell_rows)
        flat = donor_rows + geometries[donor].block_offsets[block_index]
        # Registration publishes vertex-P1 overlays for linear simplex donors;
        # field bindings publish mapped and polyhedral overlays.
        if packet.rotation is not None:
            continue
        if block.cell_kind not in _SIMPLICES.values() or not np.all(
            np.asarray(geometries[donor].affine_cells)[flat]
        ):
            continue
        donor_mesh = _cell_result(parts[donor]).mesh
        # The published stencil is the clipped, renormalized barycentric
        # partition of unity of the selected admissible donor cell.
        weights = np.maximum(donors.barycentric[slots], 0.0)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        vertex_ids = np.asarray(donor_mesh.vertex_global_ids, dtype=np.int64)
        corners = vertex_ids[np.asarray(block.vertices)[donor_rows]]
        stencil = np.where(weights > 0.0, corners, -1)
        target = receptor.scope(0, receptor_ids[slots])
        source = parts[donor].scope(0, np.unique(stencil[stencil >= 0]))
        search = CouplingSearchEvidence(
            MeshCouplingKind.OVERSET,
            _SEARCH_METHOD,
            source,
            target,
            np.full(slots.size, int(CouplingSearchStatus.FOUND), dtype=np.int32),
            stencil,
            flat,
            donors.residuals[slots],
        )
        overlays.append(
            OversetCoupling(
                parts[donor],
                receptor,
                source,
                target,
                stencil,
                np.where(weights > 0.0, weights, 0.0),
                hole_scope=hole_scope,
                search_evidence=search,
            )
        )
    return packets, overlays


# ------------------------------------------------------------ connectivity


type _OrganizationKey = tuple[
    str,
    str | None,
    str | None,
    str | None,
    tuple[int, ...] | None,
    int,
    str,
    tuple[int, ...],
]


def _organization_key(
    item: MeshPatch | MeshZone | MeshLabel | MeshAttribute, /
) -> _OrganizationKey:
    match item:
        case MeshPatch() | MeshLabel():
            role, material, region, shape = None, None, None, None
        case MeshZone():
            role, material, region, shape = (
                item.role,
                item.material_id,
                item.region_role,
                None,
            )
        case MeshAttribute():
            role, material, region, shape = item.role, None, None, item.component_shape
        case _ as unknown:
            assert_never(unknown)
    return (
        item.name,
        role,
        material,
        region,
        shape,
        item.scope.entity_dimension,
        item.scope.entity_set_id,
        tuple(np.asarray(item.scope.entity_ids).tolist()),
    )


def _association_keys(
    result: CellMeshingResult, /
) -> set[tuple[str, str, str, tuple[int, ...]]]:
    return {
        (
            item.association_kind,
            item.source_id,
            item.target_entity_set_id,
            tuple(np.asarray(item.target_global_ids).tolist()),
        )
        for item in result.associations
    }


def _require_motion_carrier(
    before: CellMeshingResult, after: CellMeshingResult, scope: NativeExecutionScope, /
) -> None:
    if before.mesh.topology_id != after.mesh.topology_id:
        raise ValueError(
            "_require_motion_carrier.after.mesh.topology_id must equal the predecessor topology_id and preserve every scientific entity ID; got a topology change."
        )
    if before.geometry.geometry_layout_id != after.geometry.geometry_layout_id:
        with coordinate_phase(
            scope=scope,
            queries_upper=cell_geometry_preparation_query_upper(after.geometry),
        ) as budget:
            layout_matches = translated_plc_layout_matches(
                before.geometry, after.geometry, budget=budget
            )
        if not layout_matches:
            raise ValueError(
                "_require_motion_carrier.after.geometry must retain the predecessor coordinate-element layout, permitting proved translated PLC layouts; got a layout mismatch."
            )
    if before.boundary is not None and after.boundary is None:
        raise ValueError(
            "_require_motion_carrier.after.boundary must retain the predecessor boundary model; got no boundary model."
        )
    if before.certification is not None and after.certification is None:
        raise ValueError(
            "_require_motion_carrier.after.certification must provide successor source, embedding, and coverage certificates; got no successor certification."
        )
    if before.region_evidence is not None and (
        after.region_evidence is None
        or before.region_evidence.cell_region_ids != after.region_evidence.cell_region_ids
    ):
        raise ValueError(
            "_require_motion_carrier.after.region_evidence must preserve all predecessor cell_region_ids; got missing evidence or changed material regions."
        )
    if _association_keys(before) != _association_keys(after):
        raise ValueError(
            "_require_motion_carrier.after.associations must preserve predecessor association keys; got dropped or replaced associations."
        )
    for field, old_records, new_records in (
        ("patches", before.patches, after.patches),
        ("zones", before.zones, after.zones),
        ("labels", before.labels, after.labels),
        ("attributes", before.attributes, after.attributes),
    ):
        if {_organization_key(item) for item in old_records} != {
            _organization_key(item) for item in new_records
        }:
            raise ValueError(
                f"{'overset_motion.' + field} must preserve predecessor scope and semantic keys; got dropped or relabeled obligations."
            )


@final
class _RegisteredSuccessor(StrictModule, NonTrainableState):
    """Validated successor carrier and its revision-bound overset specification."""

    part: MeshPart
    spec: OversetPartSpec


def _register_successor(
    old: MeshPart,
    part: MeshPart,
    spec: OversetPartSpec,
    query: PreparedBRepQuery | contracts_owner.CompiledGeometry | None,
    scope: NativeExecutionScope,
    /,
) -> _RegisteredSuccessor:
    if part.name != old.name:
        raise ValueError(
            "_register_successor.part.name must equal the registered predecessor part name; got a different part identity."
        )
    before, after = _cell_result(old), _cell_result(part)
    _require_motion_carrier(before, after, scope)
    if spec.solid_query is not None and part.part_id != old.part_id:
        if spec.wall is None:
            raise ValueError(
                "_register_successor.spec.wall must retain an explicit protected wall scope for the registered solid; got no wall scope."
            )
        wall_rows = np.unique(facet_rows(old, spec.wall))
        if not np.array_equal(
            np.asarray(before.mesh.coordinates)[wall_rows],
            np.asarray(after.mesh.coordinates)[wall_rows],
        ) and (
            query is None or _solid_query_id(query) == _solid_query_id(spec.solid_query)
        ):
            raise ValueError(
                "_register_successor.query must change to the successor closed-solid query when protected-wall coordinates move; got no query or the predecessor query."
            )
    rebound = (
        spec
        if part.part_id == old.part_id and query is None
        else spec.rebound(part, solid_query=query)
    )
    return _RegisteredSuccessor(part, rebound)


@final
class OversetConnectivity(StrictModule, NonTrainableState):
    """Represent one prepared registration epoch of native overset connectivity.

    `assembly` is the input assembly with one published `OversetCoupling` overlay per
    (receptor part, donor part) pair, so generic assembly consumers see the same
    receptor/hole/donor contract as any overset provider.
    `OversetConnectivity.complete` holds when every receptor has a donor and every protected wall vertex is
    active, neither blanked nor a receptor; ambiguous contacts and cut cells are
    reported and blanked conservatively.
    """

    assembly: MeshAssembly
    specs: tuple[OversetPartSpec, ...]
    policy: OversetPolicy
    blanking: tuple[OversetPartBlanking, ...]
    receptors: tuple[OversetReceptorEvidence, ...]
    packets: tuple[OversetDonorPacket, ...]
    geometries: tuple[_PartGeometry, ...]
    epoch: int = eqx.field(static=True)
    predecessor_id: str | None = eqx.field(static=True)
    wall_candidate_pairs: int = eqx.field(static=True)
    orphan_count: int = eqx.field(static=True)
    conflict_count: int = eqx.field(static=True)
    ambiguous_count: int = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    connectivity_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        assembly: MeshAssembly,
        specs: tuple[OversetPartSpec, ...],
        policy: OversetPolicy,
        blanking: tuple[OversetPartBlanking, ...],
        receptors: tuple[OversetReceptorEvidence, ...],
        packets: tuple[OversetDonorPacket, ...],
        geometries: tuple[_PartGeometry, ...],
        /,
        *,
        epoch: int,
        wall_candidate_pairs: int,
        predecessor_id: str | None = None,
    ) -> None:
        names = tuple(part.name for part in assembly.parts)
        if (
            tuple(spec.part_name for spec in specs) != names
            or tuple(item.part_name for item in blanking) != names
            or tuple(item.part_name for item in receptors) != names
            or len(geometries) != len(names)
        ):
            raise ValueError(
                "OversetConnectivity.records must align specs, blanking, receptors, and geometries with assembly part order; got a name-order or record-count mismatch."
            )
        for part, mask, geometry in zip(
            assembly.parts, blanking, geometries, strict=True
        ):
            if mask.part_id != part.part_id or geometry.part_id != part.part_id:
                raise ValueError(
                    "OversetConnectivity.blanking must bind corresponding assembly part IDs together with geometries; got records bound to another part revision."
                )
        expected = {
            (item.part_name, int(identifier))
            for item in receptors
            for identifier in np.asarray(item.receptor_ids)[np.asarray(item.found)]
        }
        served = set()
        for packet in packets:
            source, target = (
                assembly.part(packet.donor_part),
                assembly.part(packet.receptor_part),
            )
            if (
                source.part_id != packet.donor_revision
                or target.part_id != packet.receptor_revision
            ):
                raise ValueError(
                    "OversetConnectivity.packets must bind donor and receptor part revisions; got stale geometry or part revisions."
                )
            keys = {
                (packet.receptor_part, int(identifier))
                for identifier in np.asarray(packet.receptor_ids)
            }
            if served.intersection(keys):
                raise ValueError(
                    "OversetConnectivity.packets must assign each receptor identity once; got a duplicate receptor across packets."
                )
            served.update(keys)
        if served != expected:
            raise ValueError(
                "OversetConnectivity.donor_packets must cover every found receptor identity and no other identity; got missing or extra served identities."
            )
        self.assembly = assembly
        self.specs = specs
        self.policy = policy
        self.blanking = blanking
        self.receptors = receptors
        self.packets = packets
        self.geometries = geometries
        self.epoch = epoch
        self.predecessor_id = predecessor_id
        self.wall_candidate_pairs = wall_candidate_pairs
        self.orphan_count = sum(item.orphan_ids.size for item in receptors)
        self.conflict_count = sum(
            int(np.count_nonzero(np.asarray(item.protected_conflict)))
            for item in blanking
        )
        self.ambiguous_count = sum(
            int(np.count_nonzero(np.asarray(item.wall_contact)))
            + int(np.count_nonzero(np.asarray(item.ambiguous_cells)))
            for item in blanking
        )
        self.complete = self.orphan_count == 0 and self.conflict_count == 0
        self.connectivity_id = canonical_fingerprint(
            {
                "kind": "overset-connectivity",
                "assembly": assembly.assembly_id,
                "specs": [spec.spec_id for spec in specs],
                "policy": policy.policy_id,
                "epoch": epoch,
                "predecessor": predecessor_id,
                "blanking": [item.blanking_id for item in blanking],
                "receptors": [item.evidence_id for item in receptors],
                "packets": [item.packet_id for item in packets],
            }
        )

    def part_index(self, name: str, /) -> int:
        names = tuple(spec.part_name for spec in self.specs)
        if name not in names:
            raise KeyError(
                f"OversetConnectivity.part_index.name must identify a registered overset part; got unknown part {name!r}."
            )
        return names.index(name)

    def blanking_of(self, name: str, /) -> OversetPartBlanking:
        return self.blanking[self.part_index(name)]

    def receptors_of(self, name: str, /) -> OversetReceptorEvidence:
        return self.receptors[self.part_index(name)]

    def require_complete(self) -> None:
        """Refuse a registration with orphan receptors or blanked protected walls."""
        if not self.complete:
            raise OversetConnectivityError(
                f"OversetConnectivity.status must have zero orphan receptors and protected-wall conflicts; got {self.orphan_count} orphan receptors and {self.conflict_count} protected wall conflicts.",
                self,
            )

    def moved(
        self,
        coordinates: Mapping[str, ArrayLike],
        /,
        *,
        geometry: Mapping[str, cell_geometry_owner.CellGeometrySpec] | None = None,
        solid_queries: Mapping[str, PreparedBRepQuery | contracts_owner.CompiledGeometry]
        | None = None,
        record_phase: NativeMeshingPhaseRecorder | None = None,
    ) -> OversetConnectivity:
        """Prepare the next registration epoch after moving named parts.

        `coordinates` maps part names to new vertex coordinates in mesh row
        order. Moved parts are recertified with unchanged topology and IDs, their
        wall hierarchies are refit, and hole cutting and donor resolution are
        repeated for every part: donor evidence bound to the previous placement
        is never reused.
        """
        if not isinstance(coordinates, Mapping) or not coordinates:
            raise TypeError(
                "OversetConnectivity.moved.coordinates must be a non-empty mapping from moved part names to coordinates; got an empty or non-mapping value."
            )
        names = tuple(spec.part_name for spec in self.specs)
        if any(name not in names for name in coordinates):
            raise ValueError(
                "OversetConnectivity.moved.coordinates.keys must be assembly part names; got at least one name outside the assembly."
            )
        geometry_ = {} if geometry is None else dict(geometry)
        if not set(geometry_).issubset(coordinates):
            raise ValueError(
                "OversetConnectivity.moved.geometry must name only moved parts; got a successor coordinate map for an unmoved part."
            )
        solid_queries_ = {} if solid_queries is None else dict(solid_queries)
        if not set(solid_queries_).issubset(coordinates):
            raise ValueError(
                "OversetConnectivity.moved.solid_queries must name only moved parts; got a successor solid query for an unmoved part."
            )
        for spec in self.specs:
            if (
                spec.solid_query is not None
                and spec.part_name in coordinates
                and not np.array_equal(
                    np.asarray(coordinates[spec.part_name]),
                    np.asarray(
                        _cell_result(self.assembly.part(spec.part_name)).mesh.coordinates
                    ),
                )
                and spec.part_name not in solid_queries_
            ):
                raise ValueError(
                    "OversetConnectivity.moved.solid_queries must provide a successor B-Rep query for every moved registered solid; got moved solid coordinates without a successor query."
                )
        motion = canonical_fingerprint(
            {"kind": "overset-motion", "from": self.connectivity_id}
        )
        with measure_phase(record_phase, "geometry_transition"):
            parts = tuple(
                part.with_coordinates(
                    coordinates[part.name],
                    motion_id=motion,
                    geometry=geometry_.get(part.name),
                )
                if part.name in coordinates
                else part
                for part in self.assembly.parts
            )
            if record_phase is not None:
                jax.block_until_ready(parts)
        return self.reregister(
            {part.name: part for part in parts if part.name in coordinates},
            solid_queries=solid_queries_,
            record_phase=record_phase,
        )

    @checked
    def reregister(
        self,
        replacements: Mapping[str, MeshPart],
        /,
        *,
        solid_queries: Mapping[str, PreparedBRepQuery | contracts_owner.CompiledGeometry]
        | None = None,
        couplings: tuple[MeshCoupling, ...] | None = None,
        record_phase: NativeMeshingPhaseRecorder | None = None,
    ) -> OversetConnectivity:
        """Register recertified successors, including geometry-associated parts.

        This is the owner handoff for motions realized/certified by a geometry
        owner: no boundary, association, coordinate layout or entity identity
        may disappear. Non-overset links that touch moved revisions must be
        explicitly reprepared by their owner and passed through ``couplings``.
        """

        if not isinstance(replacements, Mapping) or not replacements:
            raise TypeError(
                "OversetConnectivity.reregister.replacements must be a non-empty mapping from moved names to certified MeshPart revisions; got an empty or non-mapping value."
            )
        names = tuple(part.name for part in self.assembly.parts)
        if not set(replacements).issubset(names):
            raise ValueError(
                "OversetConnectivity.reregister.replacements.keys must be assembly part names; got at least one name outside the assembly."
            )
        queries = {} if solid_queries is None else dict(solid_queries)
        if not set(queries).issubset(replacements):
            raise ValueError(
                "OversetConnectivity.reregister.queries.keys must be a subset of replacement names; got a successor solid query for a part without a replacement."
            )
        with NativeExecutionScope(self.policy.native_limits()) as scope:
            parts, specs = [], []
            for old, spec in zip(self.assembly.parts, self.specs, strict=True):
                part = replacements.get(old.name, old)
                match part:
                    case MeshPart():
                        pass
                    case _:
                        raise ValueError(
                            "_register_successor.part.name must equal the registered predecessor part name; got a different part identity."
                        )
                successor = _register_successor(
                    old, part, spec, queries.get(old.name), scope
                )
                parts.append(successor.part)
                specs.append(successor.spec)
            retained = (
                tuple(
                    link
                    for link in self.assembly.couplings
                    if not isinstance(link, OversetCoupling)
                )
                if couplings is None
                else tuple(couplings)
            )
            if any(isinstance(link, OversetCoupling) for link in retained):
                raise ValueError(
                    "OversetConnectivity.reregister.couplings must exclude OversetCoupling overlays; got a retained overset overlay."
                )
            assembly = MeshAssembly(tuple(parts), couplings=retained)
            return _prepare(
                assembly.parts,
                assembly.couplings,
                tuple(specs),
                self.policy,
                self.geometries,
                scope=scope,
                epoch=self.epoch + 1,
                predecessor_id=self.connectivity_id,
                record_phase=record_phase,
            )

    def composition_entries(self) -> tuple[CompositionEntry, ...]:
        """Geometry/revision-bound entries to register in a running composition."""
        parts = tuple(
            CompositionEntry(
                part,
                entry_id=_part_entry(part.name),
                role="topology",
                owner_id=_OWNER,
                structure_id=_cell_result(part).mesh.topology_id,
                revision_id=part.part_id,
                semantics_id=canonical_fingerprint(
                    {"kind": "overset-part-meaning", "part": part.name}
                ),
            )
            for part in self.assembly.parts
        )
        route = CompositionEntry(
            self,
            entry_id=_CONNECTIVITY_ENTRY,
            role="interface-route",
            owner_id=_OWNER,
            structure_id=self.connectivity_id,
            revision_id=self.connectivity_id,
            semantics_id=_SEARCH_METHOD,
            dependencies=tuple(entry.binding("revision") for entry in parts),
        )
        return (*parts, route)

    def registration(self) -> OversetRegistration:
        """Immutable restart inputs; no wall BVH, inverse map or live session."""
        return OversetRegistration(
            MeshAssembly(
                self.assembly.parts,
                couplings=tuple(
                    link
                    for link in self.assembly.couplings
                    if not isinstance(link, OversetCoupling)
                ),
            ),
            self.specs,
            self.policy,
            self.epoch,
            self.predecessor_id,
            self.connectivity_id,
        )


def _prepare(
    parts: tuple[MeshPart, ...],
    retained: tuple,
    specs: tuple[OversetPartSpec, ...],
    policy: OversetPolicy,
    previous: tuple[_PartGeometry, ...] | None,
    /,
    *,
    scope: NativeExecutionScope,
    epoch: int,
    predecessor_id: str | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> OversetConnectivity:
    with measure_phase(record_phase, "native_preparation"):
        geometries = tuple(
            previous[index]
            if (
                previous is not None
                and previous[index].part_id == part.part_id
                and previous[index].spec_id == spec.spec_id
            )
            else part_geometry(
                part,
                spec,
                policy,
                None if previous is None else previous[index],
                scope=scope,
            )
            for index, (part, spec) in enumerate(zip(parts, specs, strict=True))
        )
        if record_phase is not None:
            jax.block_until_ready(geometries)
    with measure_phase(record_phase, "region_classification"):
        budget = [policy.maximum_wall_candidate_pairs]
        blanking_records = []
        for index, part in enumerate(parts):
            values = _blank_part(
                index, parts, specs, geometries, policy, budget, scope=scope
            )
            blanking_records.append(
                OversetPartBlanking(
                    part,
                    values.vertex_status,
                    values.fringe_layer,
                    values.wall_contact,
                    values.protected_conflict,
                    values.cell_status,
                    values.ambiguous_cells,
                )
            )
        blanking = tuple(blanking_records)
        if record_phase is not None:
            jax.block_until_ready(blanking)
    with (
        measure_phase(record_phase, "donor_location"),
        coordinate_phase(scope=scope) as coordinate_budget,
    ):
        receptors, packets, overlays = [], [], []
        for index in range(len(parts)):
            evidence, donors = _receptor_evidence(
                index,
                parts,
                specs,
                geometries,
                blanking,
                policy,
                budget=coordinate_budget,
            )
            receptors.append(evidence)
            part_packets, part_overlays = _packets_and_overlays(
                index,
                parts,
                specs,
                blanking,
                geometries,
                evidence,
                donors,
                budget=coordinate_budget,
            )
            packets.extend(part_packets)
            overlays.extend(part_overlays)
        if record_phase is not None:
            jax.block_until_ready((receptors, packets, overlays))
    return OversetConnectivity(
        MeshAssembly(parts, couplings=(*retained, *overlays)),
        specs,
        policy,
        blanking,
        tuple(receptors),
        tuple(packets),
        geometries,
        epoch=epoch,
        wall_candidate_pairs=policy.maximum_wall_candidate_pairs - budget[0],
        predecessor_id=predecessor_id,
    )


@checked
def prepare_overset_connectivity(
    assembly: MeshAssembly,
    parts: tuple[OversetPartSpec, ...],
    /,
    *,
    policy: OversetPolicy | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> OversetConnectivity:
    """Cut holes, classify fringes and resolve donors of one assembly revision.

    `parts` declares the overset role of every assembly part exactly once. The
    assembly must not already carry overset overlays: the returned connectivity
    publishes its own. Incomplete coverage is reported, not raised; call
    `require_complete` where the consumer's law needs every receptor served.
    """
    policy_ = OversetPolicy() if policy is None else policy
    declared = tuple(parts)
    if not all(isinstance(spec, OversetPartSpec) for spec in declared):
        raise TypeError(
            "prepare_overset_connectivity.parts must contain only OversetPartSpec values; got at least one other value type."
        )
    names = tuple(part.name for part in assembly.parts)
    if len(names) < 2:
        raise ValueError(
            "prepare_overset_connectivity.parts must contain at least two parts; got fewer than two declared parts."
        )
    by_name = {spec.part_name: spec for spec in declared}
    if len(by_name) != len(declared) or set(by_name) != set(names):
        raise ValueError(
            "prepare_overset_connectivity.parts must declare the overset role of every assembly part once; got missing, duplicate, or foreign part names."
        )
    specs = tuple(by_name[name] for name in names)
    for spec, part in zip(specs, assembly.parts, strict=True):
        _cell_result(part)
        for scope in (spec.wall, spec.boundary, spec.excluded):
            if scope is not None:
                part.require_scope(scope)
        if (
            spec.excluded is not None
            and spec.excluded.entity_dimension != part.intrinsic_dimension
        ):
            raise ValueError(
                "prepare_overset_connectivity.spec.excluded.entity_dimension must equal part.intrinsic_dimension; got a non-cell excluded scope."
            )
        if (
            spec.wall is not None
            and spec.boundary is not None
            and np.intersect1d(
                np.asarray(spec.wall.entity_ids), np.asarray(spec.boundary.entity_ids)
            ).size
        ):
            raise ValueError(
                "prepare_overset_connectivity.spec.wall must have facets disjoint from spec.boundary; got shared facet IDs."
            )
    if any(isinstance(link, OversetCoupling) for link in assembly.couplings):
        raise ValueError(
            "prepare_overset_connectivity.assembly.couplings must be free of OversetCoupling overlays before preparation; got an existing overset overlay."
        )
    with NativeExecutionScope(policy_.native_limits()) as scope:
        return _prepare(
            assembly.parts,
            assembly.couplings,
            specs,
            policy_,
            None,
            scope=scope,
            epoch=0,
            record_phase=record_phase,
        )


@final
class _PacketLocator(simplicial_locator_owner.AbstractCellLocator, NonTrainableState):
    """Retain the chosen donor side at the packet's fixed receptor points."""

    base: (
        PreparedMappedCellLocator
        | simplicial_locator_owner.PreparedSimplicialCellLocator
        | PreparedPolyhedralCellLocator
    )
    points: Array
    selected_cells: Array
    cell_map: LocatedCellMap
    coordinates: Array
    policy: simplicial_locator_owner.SimplicialLocationPolicy
    locator_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: simplicial_locator_owner.AbstractCellLocator,
        packet: OversetDonorPacket,
        /,
        *,
        cell_offset: int = 0,
    ) -> None:
        match base:
            case (
                PreparedMappedCellLocator()
                | simplicial_locator_owner.PreparedSimplicialCellLocator()
                | PreparedPolyhedralCellLocator()
            ):
                pass
            case _:
                raise TypeError(
                    "_PacketLocator.source_locator must be bound to the packet source cells; got an unsupported or unbound cell locator."
                )
        self.base = base
        self.policy = base.policy
        self.points = packet.points
        self.selected_cells = packet.donor_cell_rows + cell_offset
        self.cell_map = base.cell_map
        self.coordinates = base.coordinates
        self.locator_id = canonical_fingerprint(
            {
                "kind": "overset-packet-locator",
                "base": base.locator_id,
                "packet": packet.packet_id,
                "cell_offset": cell_offset,
            }
        )

    def support_cells(self, points: Array, /) -> Array:
        """Return the packet's selected donor cells without repeating location."""
        return self.selected_cells

    @property
    def canonical_locator(
        self,
    ) -> (
        PreparedMappedCellLocator
        | simplicial_locator_owner.PreparedSimplicialCellLocator
        | PreparedPolyhedralCellLocator
    ):
        return self.base

    def locate(
        self, points: ArrayLike, /, *, cell_mask: ArrayLike | None = None
    ) -> simplicial_locator_owner.CellLocationResult:
        result = self.base.locate(points, cell_mask=cell_mask)
        # Support preparation queries other points. Only this packet's sites
        # carry a donor-side selection; those sites must retain the selected side.
        if not np.array_equal(np.asarray(points), np.asarray(self.points)):
            return result
        return self._select_packet_location(result)

    @eqx.filter_jit
    def _select_packet_location(
        self, result: simplicial_locator_owner.CellLocationResult, /
    ) -> simplicial_locator_owner.CellLocationResult:
        """Apply this packet's already selected donor side as one numeric action."""
        accepted = result.candidate_cells == self.selected_cells[:, None]
        found = jnp.any(accepted, axis=1) & result.successful & result.candidates_complete
        slot = jnp.argmax(accepted, axis=1)
        reference = result.candidate_reference[jnp.arange(slot.size), slot]
        evaluation = self.cell_map.evaluate(
            self.coordinates, self.selected_cells, reference
        )
        barycentric = (
            self.base.reference_weights(reference)
            if isinstance(
                self.base,
                (
                    PreparedMappedCellLocator,
                    simplicial_locator_owner.PreparedSimplicialCellLocator,
                ),
            )
            else result.barycentric
        )
        return eqx.tree_at(
            lambda item: (
                item.cell_ids,
                item.reference_coordinates,
                item.barycentric,
                item.geometry_residual,
                item.jacobian_condition,
                item.inside,
                item.candidate_count,
                item.status,
                item.successful,
                item.candidate_cells,
                item.candidate_reference,
            ),
            result,
            (
                jnp.where(found, self.selected_cells, -1),
                reference,
                jnp.where(found[:, None], barycentric, 0),
                jnp.linalg.norm(evaluation.physical_points - self.points, axis=1),
                jnp.linalg.cond(evaluation.jacobian),
                found,
                found.astype(jnp.int32),
                jnp.where(
                    found,
                    int(simplicial_locator_owner.CellLocationStatus.LOCATED),
                    jnp.where(
                        result.successful,
                        int(simplicial_locator_owner.CellLocationStatus.OUTSIDE),
                        result.status,
                    ),
                ),
                found,
                jnp.where(accepted & found[:, None], result.candidate_cells, -1),
                jnp.where(accepted[:, :, None], result.candidate_reference, 0),
            ),
        )


@final
class _FieldTargets(StrictModule, NonTrainableState):
    """Receptor evidence and donor packets admitted for the field epoch."""

    receptors: tuple[OversetReceptorEvidence, ...]
    packets: tuple[OversetDonorPacket, ...]


def _prepare_field_targets(
    connectivity: OversetConnectivity, previous: OversetConnectivity | None, /
) -> _FieldTargets:
    receptors, packets = connectivity.receptors, connectivity.packets
    if previous is None:
        return _FieldTargets(receptors, packets)
    if connectivity.predecessor_id != previous.connectivity_id:
        raise ValueError(
            "_prepare_field_targets.connectivity.predecessor_id must equal previous.connectivity_id; got a different predecessor."
        )
    records, donor_packets = [], []
    parts = connectivity.assembly.parts
    with _field_packet_coordinate_phase(connectivity) as budget:
        for index in range(len(parts)):
            evidence, donors = _receptor_evidence(
                index,
                parts,
                connectivity.specs,
                connectivity.geometries,
                connectivity.blanking,
                connectivity.policy,
                previous.blanking,
                budget=budget,
            )
            records.append(evidence)
            donor_packets.extend(
                _donor_packets(
                    index,
                    parts,
                    connectivity.specs,
                    connectivity.geometries,
                    evidence,
                    donors,
                    budget=budget,
                )
            )
    receptors, packets = tuple(records), tuple(donor_packets)
    if any(item.orphan_ids.size for item in receptors):
        raise OversetConnectivityError(
            "_prepare_field_targets.receptors must give every requested motion-transfer point a valid old-state donor; got one or more orphan points.",
            connectivity,
            field_evidence=receptors,
        )
    return _FieldTargets(receptors, packets)


@final
class _DonorFieldBindings(StrictModule, NonTrainableState):
    """Validated FE/FV reconstructions and their full coefficient layouts."""

    reconstructions: dict[tuple[str, str], views_owner.PreparedFieldReconstruction]
    layouts: dict[str, views_owner.PreparedFieldReconstruction]
    names: tuple[str, ...] = eqx.field(static=True)
    coefficient_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    value_shape: tuple[int, ...] = eqx.field(static=True)


def _prepare_fe_donor(
    connectivity: OversetConnectivity,
    result: CellMeshingResult,
    name: str,
    binding: FiniteElementDiscretization,
    field_name: str,
    packets: tuple[OversetDonorPacket, ...],
    /,
) -> dict[tuple[str, str], views_owner.PreparedFieldReconstruction]:
    if binding.mesh.mesh_id != result.mesh.mesh_id:
        raise ValueError(
            "overset_field_transfer.discretizations must bind the donor mesh revision; got a different FE mesh."
        )
    if (
        binding.default_runtime.geometry_layout_id != result.geometry.geometry_layout_id
        or not np.array_equal(
            np.asarray(binding.default_runtime.coordinates),
            np.asarray(result.geometry.coordinates),
        )
    ):
        raise ValueError(
            "overset_field_transfer.discretizations must bind the donor coordinate map; got a different FE geometry."
        )
    served_blocks = {
        packet.donor_block for packet in packets if packet.donor_part == name
    }
    # Unused supplied fields still retain and validate their full layout.
    if not served_blocks:
        served_blocks = {binding.dof_maps[binding.field_index(field_name)].block_names[0]}
    reconstructions = {}
    for block in sorted(served_blocks):
        block_index = binding.dof_maps[binding.field_index(field_name)].block_names.index(
            block
        )
        cell_map = PreparedFiniteElementCellMap(binding, block_index)
        locator = None
        if isinstance(cell_map.coordinate_element, RestrictedCellGeometryElement):
            geometry = connectivity.geometries[connectivity.part_index(name)]
            policy = geometry.locators[geometry.block_names.index(block)].policy
            locator = PreparedMappedCellLocator(
                cell_map, binding.default_runtime.coordinates, policy
            )
        reconstructions[name, block] = prepare_finite_element_field_reconstruction(
            binding,
            field_name,
            block_name=block,
            locator=locator,
            location_policy=connectivity.policy.location_policy,
        )
    return reconstructions


def _require_fv_donor(
    binding: views_owner.PreparedFieldReconstruction, result: CellMeshingResult, /
) -> None:
    match binding.kernel:
        case UnstructuredFiniteVolumeFieldReconstructionKernel() as kernel:
            pass
        case _:
            raise TypeError(
                "overset_field_transfer.discretizations must retain an unstructured FV reconstruction kernel; got another kernel kind."
            )
    kernel.require_source_geometry(result.geometry)
    match kernel.discretization:
        case (
            unstructured_owner.UnstructuredFiniteVolumeDiscretization() as discretization
        ):
            geometry = discretization.cell_geometry
            coordinates = (
                result.mesh.coordinates if geometry is None else geometry.coordinates
            )
        case TriangleFiniteVolumeDiscretization():
            coordinates = result.mesh.coordinates
        case _ as unknown:
            assert_never(unknown)
    if (
        kernel.discretization.mesh.mesh_id != result.mesh.mesh_id
        or kernel.locator.cell_map.cell_count
        != sum(block.cell_count for block in result.mesh.blocks)
        or not np.array_equal(
            np.asarray(kernel.locator.coordinates), np.asarray(coordinates)
        )
    ):
        raise ValueError(
            "overset_field_transfer.discretizations must bind the donor mesh and coordinate revision; got a different FV geometry."
        )


def _prepare_donor_fields(
    connectivity: OversetConnectivity,
    discretizations: Mapping[
        str, FiniteElementDiscretization | views_owner.PreparedFieldReconstruction
    ],
    field_name: str,
    packets: tuple[OversetDonorPacket, ...],
    /,
) -> _DonorFieldBindings:
    if not isinstance(discretizations, Mapping):
        raise TypeError(
            "_prepare_donor_fields.discretizations must map donor names to FE or FV field owners; got a non-mapping value."
        )
    names = tuple(sorted(discretizations))
    required = {packet.donor_part for packet in packets}
    if not required.issubset(names):
        raise ValueError(
            "overset_field_transfer.discretizations must bind every served donor; got missing field owners."
        )
    if not names:
        raise ValueError(
            "overset_field_transfer.discretizations must contain at least one donor field; got an empty mapping."
        )
    reconstructions, layouts = {}, {}
    for name in names:
        result = _cell_result(connectivity.assembly.part(name))
        match discretizations[name]:
            case FiniteElementDiscretization() as binding:
                fields = _prepare_fe_donor(
                    connectivity, result, name, binding, field_name, packets
                )
                reconstructions.update(fields)
                layouts[name] = next(reversed(fields.values()))
            case views_owner.PreparedFieldReconstruction() as binding:
                _require_fv_donor(binding, result)
                for block in result.mesh.blocks:
                    reconstructions[name, block.name] = binding
                layouts[name] = binding
            case _:
                raise TypeError(
                    "overset_field_transfer.discretizations must contain FE discretizations or FV field reconstructions; got another donor owner kind."
                )
    shapes = tuple(layouts[name].coefficient_shape for name in names)
    value_shapes = {owner.value_shape for owner in reconstructions.values()}
    if len(value_shapes) != 1:
        raise ValueError(
            "overset_field_transfer.discretizations must share one component shape; got different donor value shapes."
        )
    return _DonorFieldBindings(
        reconstructions, layouts, names, shapes, next(iter(value_shapes))
    )


@final
class _PacketFieldRoute(StrictModule, NonTrainableState):
    """One field query and the overlay carrying its source support evidence."""

    query: field_query_owner.PreparedFieldQuery
    coupling: OversetCoupling


def _prepare_packet_field_route(
    connectivity: OversetConnectivity,
    previous: OversetConnectivity | None,
    packet: OversetDonorPacket,
    owner: views_owner.PreparedFieldReconstruction,
    binding: FiniteElementDiscretization | views_owner.PreparedFieldReconstruction,
    field_name: str,
    action: OversetValueAction | None,
    /,
) -> _PacketFieldRoute:
    kernel = owner.kernel
    match kernel:
        case (
            FiniteElementFieldReconstructionKernel()
            | UnstructuredFiniteVolumeFieldReconstructionKernel()
        ):
            pass
        case _:
            raise TypeError(
                "overset_field_transfer.reconstruction must retain an FE or FV kernel; got another reconstruction kind."
            )
    donor = connectivity.assembly.part(packet.donor_part)
    target = connectivity.assembly.part(packet.receptor_part)
    donor_index = connectivity.part_index(packet.donor_part)
    target_index = connectivity.part_index(packet.receptor_part)
    geometry = connectivity.geometries[donor_index]
    offset = (
        geometry.block_offsets[geometry.block_names.index(packet.donor_block)]
        if isinstance(kernel, UnstructuredFiniteVolumeFieldReconstructionKernel)
        else 0
    )
    locator = _PacketLocator(kernel.locator, packet, cell_offset=offset)
    admissible = np.array(_admissible(connectivity.blanking[donor_index]), copy=True)
    if previous is not None:
        admissible &= np.asarray(previous.blanking[donor_index].cell_status) != int(
            OversetCellStatus.HOLE
        )
    match kernel:
        case FiniteElementFieldReconstructionKernel():
            match binding:
                case FiniteElementDiscretization():
                    pass
                case _:
                    raise TypeError(
                        "overset_field_transfer.packet must retain its FE discretization; got another donor binding kind."
                    )
            reconstruction = prepare_finite_element_field_reconstruction(
                binding,
                field_name,
                block_name=packet.donor_block,
                locator=locator,
                support_geometry=owner.support_geometry,
            )
            query = field_query_owner.PreparedFieldQuery(reconstruction, packet.points)
            support_ids = np.unique(np.asarray(packet.donor_cells))
        case UnstructuredFiniteVolumeFieldReconstructionKernel():
            query = kernel.prepare_packet_query(
                owner, locator, packet.points, jnp.asarray(admissible)
            )
            support_rows = kernel.query_support_rows(query)
            support_ids = np.asarray(geometry.cell_ids)[support_rows]
        case _ as unknown:
            assert_never(unknown)
    holes = connectivity.blanking[target_index].vertex_ids_with(OversetVertexStatus.HOLE)
    coupling = OversetCoupling.from_field_query(
        donor,
        target,
        donor.scope(donor.intrinsic_dimension, support_ids),
        target.scope(0, packet.receptor_ids),
        query,
        hole_scope=target.scope(0, holes) if holes.size else None,
        rotation=packet.rotation,
        translation=packet.translation,
        value_action="invariant" if action is None else action,
        source_image_rotation=connectivity.specs[donor_index].image_rotation,
        target_image_rotation=connectivity.specs[target_index].image_rotation,
    )
    return _PacketFieldRoute(query, coupling)


@final
class PreparedOversetFieldTransfer(StrictModule, NonTrainableState):
    """Prepare FE/FV field queries at sorted receptor IDs.

    Coefficients follow each donor owner's full FE DOF or FV cell-average order, not
    sorted support-scope IDs. Outputs follow this route's ``receptors``. Signed
    high-order/Piola weights remain defined by the source-bound field query.
    ``transpose`` is available only for coefficient-linear owners and is not a
    conservation proof. FV requires an explicit reconstruction policy from its owner;
    ``field_name`` selects FE fields, never an implicit replacement FV policy.
    Supplying ``previous`` also fills newly uncovered vertices and excludes predecessor
    holes. Coefficients must be the accepted predecessor or material-attached state,
    including accepted field-filled fringe values; ``CompositionTransport`` carries that
    fill obligation. Those coefficients are interpreted on the candidate coordinate map.
    Every current FV coefficient-read stencil must remain active, not hole/fringe state.
    Image-bound fields require an explicit `value_action`: invariant components, polar
    vectors under source isometries satisfying the Euclidean Gram identity `R^T R = I`
    in encoded-coefficient arithmetic, or explicitly contravariant components under
    numeric affine maps. Vector outputs are in each target's stored source chart;
    applying that target's image rotation yields registered-world components. The donor
    field basis and source coordinate certificate are never replaced by an image mesh.
    """

    connectivity: OversetConnectivity
    assembly: MeshAssembly
    receptors: tuple[OversetReceptorEvidence, ...]
    packets: tuple[OversetDonorPacket, ...]
    previous_id: str | None = eqx.field(static=True)
    queries: tuple[field_query_owner.PreparedFieldQuery, ...]
    field_couplings: tuple[OversetCoupling, ...]
    donor_names: tuple[str, ...] = eqx.field(static=True)
    coefficient_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    value_shape: tuple[int, ...] = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    value_action: OversetValueAction | None = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        connectivity: OversetConnectivity,
        discretizations: Mapping[
            str, FiniteElementDiscretization | views_owner.PreparedFieldReconstruction
        ],
        field_name: str,
        /,
        *,
        previous: OversetConnectivity | None = None,
        record_phase: NativeMeshingPhaseRecorder | None = None,
        value_action: OversetValueAction | None = None,
    ) -> None:

        action = (
            None
            if value_action is None
            else parse(value_action, OversetValueAction, "value_action")
        )
        if action is None and any(
            spec.image_rotation is not None for spec in connectivity.specs
        ):
            raise ValueError(
                "PreparedOversetFieldTransfer.value_action must be explicitly invariant, polar-vector, or contravariant-vector for image-bound fields; got no value_action."
            )
        if action == "polar-vector" and any(
            not spec.image_isometry_exact for spec in connectivity.specs
        ):
            raise ValueError(
                "PreparedOversetFieldTransfer.value_action must use polar-vector only when every image's encoded coefficients satisfy the Euclidean Gram identity; got a non-isometric affine image."
            )
        interpolation_started = phase_started(record_phase)
        connectivity.require_complete()
        targets = _prepare_field_targets(connectivity, previous)
        receptors, packets = targets.receptors, targets.packets
        fields = _prepare_donor_fields(connectivity, discretizations, field_name, packets)
        names, shapes = fields.names, fields.coefficient_shapes
        queries, overlays = [], []
        for packet in packets:
            key = packet.donor_part, packet.donor_block
            if key not in fields.reconstructions:
                raise ValueError(
                    "overset_field_transfer.donor_block must have a field binding; got an unbound selected block."
                )
            route = _prepare_packet_field_route(
                connectivity,
                previous,
                packet,
                fields.reconstructions[key],
                discretizations[packet.donor_part],
                field_name,
                action,
            )
            queries.append(route.query)
            overlays.append(route.coupling)
        self.assembly = MeshAssembly(
            connectivity.assembly.parts,
            couplings=(
                *tuple(
                    link
                    for link in connectivity.assembly.couplings
                    if not isinstance(link, OversetCoupling)
                ),
                *overlays,
            ),
        )
        self.connectivity = connectivity
        self.receptors, self.packets = receptors, packets
        self.previous_id = None if previous is None else previous.connectivity_id
        self.queries = tuple(queries)
        self.field_couplings = tuple(overlays)
        self.donor_names = names
        self.coefficient_shapes = shapes
        self.value_shape = fields.value_shape
        self.field_name = str(field_name)
        self.conservative = False
        self.value_action = action
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "overset-field-transfer",
                "connectivity": connectivity.connectivity_id,
                "previous": self.previous_id,
                "targets": [item.evidence_id for item in receptors],
                "field": self.field_name,
                "queries": [query.query_id for query in queries],
                "actions": [coupling.coupling_id for coupling in overlays],
                "spaces": [fields.layouts[name].reconstruction_id for name in names],
            }
        )
        if record_phase is not None:
            # Synchronize route/evidence outputs before recording preparation time.
            jax.block_until_ready(
                tuple(
                    (query.route, query.evidence, query.admitted, query.points)
                    for query in queries
                )
            )
            record_elapsed(
                record_phase, "interpolation_preparation", interpolation_started
            )

    @property
    def coefficient_linear(self) -> bool:
        """Return whether every bound query admits an algebraic transpose."""
        return all(query.coefficient_linear for query in self.queries)

    def apply(self, coefficients: Mapping[str, ArrayLike], /) -> dict[str, Array]:
        """Evaluate receptors using the FE/FV owner's full coefficient order."""
        if set(coefficients) != set(self.donor_names):
            raise ValueError(
                "PreparedOversetFieldTransfer.apply.coefficients.keys must equal the prepared donor_names; got missing or extra donor fields."
            )
        arrays = {name: jnp.asarray(coefficients[name]) for name in self.donor_names}
        for name, shape in zip(self.donor_names, self.coefficient_shapes, strict=True):
            if arrays[name].shape != shape:
                raise ValueError(
                    f"PreparedOversetFieldTransfer.coefficients must give field {name!r} shape {shape}; got a mismatched coefficient shape."
                )
        dtype = jnp.result_type(*(array.dtype for array in arrays.values()))
        outputs = {
            evidence.part_name: jnp.zeros(
                (evidence.receptor_ids.size, *self.value_shape), dtype=dtype
            )
            for evidence in self.receptors
        }
        for packet, coupling in zip(self.packets, self.field_couplings, strict=True):
            outputs[packet.receptor_part] = (
                outputs[packet.receptor_part]
                .at[packet.receptor_slots]
                .set(coupling.transfer(arrays[packet.donor_part]))
            )
        return outputs

    def transpose(self, cotangents: Mapping[str, ArrayLike], /) -> dict[str, Array]:
        """Apply the algebraic transpose of the bound query operator for coefficient-linear
        FE/FV owners only.
        """
        if not self.coefficient_linear:
            raise ValueError(
                "PreparedOversetFieldTransfer.transpose must use coefficient-linear reconstruction; got nonlinear FV reconstruction requiring PreparedFieldQuery linearization at a coefficient state"
            )
        if set(cotangents) != {item.part_name for item in self.receptors}:
            raise ValueError(
                "PreparedOversetFieldTransfer.transpose.cotangents.keys must name every receptor part and no other part; got missing or extra receptor names."
            )
        arrays = {name: jnp.asarray(value) for name, value in cotangents.items()}
        for item in self.receptors:
            if arrays[item.part_name].shape != (
                item.receptor_ids.size,
                *self.value_shape,
            ):
                raise ValueError(
                    "PreparedOversetFieldTransfer.transpose.cotangents must have one value_shape record per sorted receptor ID; got a shape or receptor-order mismatch."
                )
        dtype = jnp.result_type(*(array.dtype for array in arrays.values()))
        outputs = {
            name: jnp.zeros(shape, dtype=dtype)
            for name, shape in zip(self.donor_names, self.coefficient_shapes, strict=True)
        }
        for packet, coupling in zip(self.packets, self.field_couplings, strict=True):
            outputs[packet.donor_part] = outputs[packet.donor_part] + coupling.transpose(
                arrays[packet.receptor_part][packet.receptor_slots]
            )
        return outputs

    def prepare_owner_local_exchange(
        self,
        coefficient_ids: Mapping[str, ArrayLike],
        execution_group: ExecutionGroup,
        /,
        *,
        axis_name: str = "parts",
        message_capacity: int,
    ) -> overset_exchange_owner.PreparedOwnerLocalOversetExchange:
        """Bind sparse donor traffic to this fixed native query epoch.

        Explicit stable scalar coefficient IDs follow the flattened FE/FV coefficient
        layout. The exchange accepts only process-local owned fields and returns only
        process-local receptors; it does not gather entire donor fields or claim
        distributed geometric search.
        """

        # Lazy: _overset <-> _overset_exchange import cycle.
        from . import _overset_exchange as overset_exchange_owner

        return overset_exchange_owner.PreparedOwnerLocalOversetExchange(
            self,
            coefficient_ids,
            execution_group,
            axis_name=axis_name,
            message_capacity=message_capacity,
        )

    def duality_evidence(
        self,
        coefficients: Mapping[str, ArrayLike],
        cotangents: Mapping[str, ArrayLike],
        /,
        *,
        tolerance: float = 1e-10,
    ) -> views_owner.InterpolationTransposeEvidence:
        values, scattered = self.apply(coefficients), self.transpose(cotangents)
        return transpose_duality_evidence(
            jnp.concatenate([values[name].ravel() for name in sorted(values)]),
            jnp.concatenate(
                [jnp.asarray(cotangents[name]).ravel() for name in sorted(values)]
            ),
            jnp.concatenate(
                [jnp.asarray(coefficients[name]).ravel() for name in self.donor_names]
            ),
            jnp.concatenate([scattered[name].ravel() for name in self.donor_names]),
            tolerance=tolerance,
        )


@checked
def prepare_overset_field_transfer(
    connectivity: OversetConnectivity,
    discretizations: Mapping[
        str, FiniteElementDiscretization | views_owner.PreparedFieldReconstruction
    ],
    field_name: str = "state",
    /,
    *,
    previous: OversetConnectivity | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    value_action: OversetValueAction | None = None,
) -> PreparedOversetFieldTransfer:
    """Bind FE fields or explicit FV reconstruction owners and query overlays."""
    return PreparedOversetFieldTransfer(
        connectivity,
        discretizations,
        field_name,
        previous=previous,
        record_phase=record_phase,
        value_action=value_action,
    )


@final
class _OverlapDomain(StrictModule, NonTrainableState):
    """Represented planar cell domain admitted for common refinement."""

    mesh: cell_mesh_owner.CellMesh


def _selected_polyhedral_loops(
    connectivity: PolyhedralConnectivity, identifiers: np.ndarray, /
) -> tuple[tuple[np.ndarray, ...], ...]:
    cell_lookup = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(connectivity.cell_global_ids))
    }
    offsets, faces, signs = (
        np.asarray(connectivity.cell_face_offsets),
        np.asarray(connectivity.cell_face_values),
        np.asarray(connectivity.cell_face_sign_values),
    )
    face_offsets, vertices = (
        np.asarray(connectivity.face_vertex_offsets),
        np.asarray(connectivity.face_vertex_values),
    )
    selected = []
    for identifier in identifiers:
        row = cell_lookup[int(identifier)]
        loops = []
        for face, sign in zip(
            faces[offsets[row] : offsets[row + 1]],
            signs[offsets[row] : offsets[row + 1]],
            strict=True,
        ):
            loop = vertices[face_offsets[face] : face_offsets[face + 1]]
            loops.append(loop if sign > 0 else loop[::-1])
        selected.append(tuple(loops))
    return tuple(selected)


def _select_overlap_cells(
    mesh: cell_mesh_owner.CellMesh, scope: MeshingScope, /
) -> _OverlapDomain:
    if scope.entity_dimension != mesh.topological_dimension:
        raise ValueError(
            "_select_overlap_cells.scope.entity_dimension must equal mesh.topological_dimension; got a non-cell scope."
        )
    chosen = set(np.asarray(scope.entity_ids).tolist())
    fixed, polyhedra, poly_ids = [], {}, {}
    for block in mesh.blocks:
        mask = np.isin(np.asarray(block.global_ids), tuple(chosen))
        if not np.any(mask):
            continue
        identifiers = np.asarray(block.global_ids)[mask]
        if block.cell_kind != "polyhedron":
            fixed.append(
                reference_cell_types_owner.CellBlock(
                    block.name,
                    block.cell_kind,
                    np.asarray(block.vertices)[mask],
                    global_ids=identifiers,
                )
            )
        else:
            match mesh.connectivity:
                case PolyhedralConnectivity() as connectivity:
                    polyhedra[block.name] = _selected_polyhedral_loops(
                        connectivity, identifiers
                    )
                case _:
                    raise TypeError(
                        "_select_overlap_cells.mesh.connectivity must provide PolyhedralConnectivity for selected polyhedral cells; got another incidence kind."
                    )
            poly_ids[block.name] = identifiers
    if not fixed and not polyhedra:
        raise ValueError(
            "_select_overlap_cells.selection must contain at least one fixed or polyhedral cell; got no selected cells."
        )
    selected = (
        cell_mesh_owner.CellMesh.from_mixed_3d(
            mesh.coordinates,
            tuple(fixed),
            polyhedra=polyhedra,
            vertex_global_ids=mesh.vertex_global_ids,
            polyhedral_cell_global_ids=poly_ids,
            numeric_version=mesh.numeric_version,
        )
        if mesh.ambient_dimension == 3
        else cell_mesh_owner.CellMesh(
            mesh.coordinates,
            tuple(fixed),
            vertex_global_ids=mesh.vertex_global_ids,
            numeric_version=mesh.numeric_version,
        )
    )
    return _OverlapDomain(selected)


def _prepare_overlap_domain(
    part: MeshPart, scope: MeshingScope | None, /
) -> _OverlapDomain:
    result = _cell_result(part)
    mesh = result.mesh
    elements, routes, coordinates = result.geometry.resolve(mesh)
    if (
        not np.array_equal(np.asarray(coordinates), np.asarray(mesh.coordinates))
        or any(
            not np.array_equal(np.asarray(route), np.asarray(block.vertices))
            for route, block in zip(routes, mesh.blocks, strict=True)
        )
        or any(not _represented_linear_element(element) for element in elements)
    ):
        raise ValueError(
            "_prepare_overlap_domain.domain must retain mesh coordinates, block vertex routes, and represented planar-face linear elements; got changed coordinates or incidence, or erased mapped geometry."
        )
    if scope is None:
        return _OverlapDomain(mesh)
    part.require_scope(scope)
    return _select_overlap_cells(mesh, scope)


@checked
def prepare_overset_conservative_remap(
    source: MeshPart,
    target: MeshPart,
    /,
    *,
    source_cells: MeshingScope | None = None,
    target_cells: MeshingScope | None = None,
    policy: supermesh_owner.CommonRefinementPolicy | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> remap_evidence_owner.PreparedUnstructuredConservativeRemap:
    """Prepare a separate certified overlap remap of explicitly selected cell averages.

    Coverage of every requested cell is the default. Partial overlap requires an
    explicit common-refinement policy and retains that owner's uncovered-content ledger;
    neither this route nor interpolation invents values in uncovered volume.
    """

    with measure_phase(record_phase, "common_refinement"):
        remap = automatic_remap_owner.prepare_unstructured_conservative_remap(
            unstructured_owner.UnstructuredFiniteVolumePlan.from_cell_mesh(
                _prepare_overlap_domain(source, source_cells).mesh
            ).prepare(),
            unstructured_owner.UnstructuredFiniteVolumePlan.from_cell_mesh(
                _prepare_overlap_domain(target, target_cells).mesh
            ).prepare(),
            provenance="native-overset-certified-cell-overlap",
            policy=policy,
        )
        if record_phase is not None:
            jax.block_until_ready(remap)
    return remap


@checked
def prepare_overset_conservative_state_transport(
    source: CompositionEntry,
    target: CompositionEntry,
    states: tuple[CompositionEntry, ...],
    /,
    *,
    content_tolerance: float,
    policy: supermesh_owner.CommonRefinementPolicy | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> tuple[
    remap_evidence_owner.PreparedUnstructuredConservativeRemap,
    tuple[CompositionTransport, ...],
]:
    """Execute cell-average state transport with overlap covering every requested cell of a
    moving part.

    Physical fields, material densities and spatial history registers use the same
    certified physical intersections, with separate inventory evidence for every
    component. Categorical material labels are not cell-average fields. The caller
    supplies topology entries and every state disposition explicitly. Motion changing
    the represented domain needs an exterior-content ledger; this full-overlap route
    never invents incoming values or discards outflow.
    """
    if source.role != "topology" or target.role != "topology":
        raise ValueError(
            "prepare_overset_conservative_state_transport.source.role must be topology together with target.role; got a non-topology endpoint role."
        )
    if not isinstance(source.value, MeshPart) or not isinstance(target.value, MeshPart):
        raise TypeError(
            "prepare_overset_conservative_state_transport.source.value must hold MeshPart together with target.value; got a non-MeshPart endpoint value."
        )
    old, new = source.value, target.value
    if old.name != new.name or source.entry_id != target.entry_id:
        raise ValueError(
            "prepare_overset_conservative_state_transport.target must preserve the source part name and entry_id; got a changed registered part identity."
        )
    if (
        source.structure_id != _cell_result(old).mesh.topology_id
        or target.structure_id != _cell_result(new).mesh.topology_id
        or source.revision_id != old.part_id
        or target.revision_id != new.part_id
    ):
        raise ValueError(
            "prepare_overset_conservative_state_transport.source must bind the source mesh topology and part revision, and target must bind its successor; got stale endpoint mesh identities."
        )
    tolerance = float(content_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            "prepare_overset_conservative_state_transport.content_tolerance must be finite and non-negative; got a non-finite or negative value."
        )
    if not states or any(not isinstance(state, CompositionEntry) for state in states):
        raise TypeError(
            "prepare_overset_conservative_state_transport.states must be a non-empty collection of CompositionEntry values; got no states or a non-CompositionEntry state."
        )
    for state in states:
        if state.role not in ("physical-state", "model-state", "history"):
            raise ValueError(
                "prepare_overset_conservative_state_transport.state.role must be 'physical-state', 'model-state', or 'history'; got another state role."
            )
        if state.structure_id != source.structure_id:
            raise ValueError(
                "prepare_overset_conservative_state_transport.state.structure_id must equal source.structure_id; got a state bound to different cell topology."
            )
        bindings = [
            binding
            for binding in state.dependencies
            if binding.entry_id == source.entry_id and binding.facet == "revision"
        ]
        if len(bindings) != 1 or bindings[0].bound_id != source.revision_id:
            raise ValueError(
                "prepare_overset_conservative_state_transport.state.bindings must contain one binding whose bound_id equals source.revision_id; got zero, multiple, or stale revision bindings."
            )
    prepared = prepare_overset_conservative_remap(
        old,
        new,
        policy=policy,
        record_phase=record_phase,
    )
    remap = prepared.plan
    if not prepared.succeeded or remap is None:
        raise ValueError(
            f"prepare_overset_conservative_state_transport.overlap must succeed with a remap; got overlap refused: {prepared.reason}"
        )
    if not remap.require_complete:
        raise ValueError(
            "prepare_overset_conservative_state_transport.remap must cover every requested cell or bind an exterior-content owner; got partial overlap without exterior-content ownership."
        )
    transports = []
    for state in states:
        values = jnp.asarray(state.value)
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise ValueError(
                "prepare_overset_conservative_state_transport.state.values.dtype must be floating for real cell-average densities; got a non-floating or categorical dtype."
            )
        output = remap.apply(values)
        successful = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(output))
        trailing = (1,) * (values.ndim - 1)
        source_content = jnp.sum(
            values * remap.source_volumes.reshape((-1,) + trailing), axis=0
        ).reshape(-1)
        target_content = jnp.sum(
            output * remap.target_volumes.reshape((-1,) + trailing), axis=0
        ).reshape(-1)
        dependencies = tuple(
            target.binding(binding.facet)
            if binding.entry_id == source.entry_id
            else binding
            for binding in state.dependencies
        )
        proposed = CompositionEntry(
            output,
            entry_id=state.entry_id,
            role=state.role,
            owner_id=state.owner_id,
            structure_id=target.structure_id,
            revision_id=canonical_fingerprint(
                {
                    "kind": "moving-overset-conservative-state",
                    "source": state.record_id,
                    "target": target.record_id,
                    "remap": prepared.remap_id,
                    "values": array_tree_fingerprint(output),
                }
            ),
            semantics_id=state.semantics_id,
            dependencies=dependencies,
        )
        transports.append(
            CompositionTransport(
                "physical-remap",
                (state.entry_id,),
                (proposed,),
                source_structure_ids=(state.structure_id,),
                route_id=prepared.remap_id,
                successful=successful,
                source_content=source_content,
                target_content=target_content,
                content_tolerance=jnp.full(
                    source_content.shape, tolerance, dtype=jnp.float64
                ),
            )
        )
    return prepared, tuple(transports)


@final
class PreparedOversetMotion(StrictModule, NonTrainableState):
    """Unpublished moving registration with all owner/state obligations staged."""

    previous: OversetConnectivity
    candidate: OversetConnectivity
    rebind: CompositionRebind

    def commit(self, /, *, accepted_boundary: bool) -> CompositionRebindReceipt:
        """Publish only when every requested entity satisfies the coverage contract and
        owner state transports succeed.
        """
        if not isinstance(accepted_boundary, bool):
            raise TypeError(
                "PreparedOversetMotion.commit.accepted_boundary must be a host bool; got a non-bool decision."
            )
        return commit_composition_rebind(
            self.rebind,
            accepted_boundary=accepted_boundary and self.candidate.complete,
        )


@checked
def prepare_overset_motion_rebind(
    composition: Composition,
    previous: OversetConnectivity,
    candidate: OversetConnectivity,
    /,
    *,
    transports: tuple[CompositionTransport, ...] = (),
    reprepare: tuple[CompositionEntry, ...] = (),
    retain: tuple[str, ...] = (),
    invalidate: tuple[str, ...] = (),
) -> PreparedOversetMotion:
    """Stage motion through CompositionRebind without inferring, dropping or resetting any
    state.

    Physical, history, RNG, field, geometry-associated and solver-derived entries
    outside the overset owner require their explicit caller dispositions. Stale bindings
    or missing obligations fail at staging. Failed coverage or transport returns the
    previous composition object on commit.
    """
    if (
        candidate.epoch != previous.epoch + 1
        or candidate.predecessor_id != previous.connectivity_id
    ):
        raise ValueError(
            "prepare_overset_motion_rebind.candidate must advance the previous epoch by one and name previous.connectivity_id as predecessor; got a skipped epoch or different predecessor."
        )
    if tuple(spec.part_name for spec in candidate.specs) != tuple(
        spec.part_name for spec in previous.specs
    ):
        raise ValueError(
            "prepare_overset_motion_rebind.candidate.specs must preserve every previous part_name in order; got changed or reordered registered part identities."
        )
    for old, new in zip(previous.assembly.parts, candidate.assembly.parts, strict=True):
        if _cell_result(old).mesh.topology_id != _cell_result(new).mesh.topology_id:
            raise ValueError(
                "prepare_overset_motion_rebind.new.mesh.topology_id must equal old.mesh.topology_id and preserve all entity identities; got changed topology."
            )
    old_entries, new_entries = (
        previous.composition_entries(),
        candidate.composition_entries(),
    )
    own_retain, own_reprepare = [], []
    for old, new in zip(old_entries, new_entries, strict=True):
        accepted = composition.entry(old.entry_id)
        if accepted.record_id != old.record_id:
            raise ValueError(
                "prepare_overset_motion_rebind.accepted.record_id must equal old.record_id; got a composition entry for another registration revision."
            )
        if old.record_id == new.record_id:
            own_retain.append(old.entry_id)
        else:
            own_reprepare.append(new)
    rebind = CompositionRebind(
        composition,
        retain=(*own_retain, *retain),
        reprepare=(*own_reprepare, *reprepare),
        transports=transports,
        invalidate=invalidate,
    )
    return PreparedOversetMotion(previous, candidate, rebind)


@final
class OversetRegistration(StrictModule, NonTrainableState):
    """Describe an immutable restart registration, excluding prepared spatial state."""

    assembly: MeshAssembly
    specs: tuple[OversetPartSpec, ...]
    policy: OversetPolicy
    epoch: int = eqx.field(static=True)
    predecessor_id: str | None = eqx.field(static=True)
    connectivity_id: str = eqx.field(static=True)

    def prepare(
        self,
        /,
        *,
        record_phase: NativeMeshingPhaseRecorder | None = None,
    ) -> OversetConnectivity:
        """Rebuild and verify all blanking/donors against the persisted identity."""
        with NativeExecutionScope(self.policy.native_limits()) as scope:
            result = _prepare(
                self.assembly.parts,
                self.assembly.couplings,
                self.specs,
                self.policy,
                None,
                scope=scope,
                epoch=self.epoch,
                predecessor_id=self.predecessor_id,
                record_phase=record_phase,
            )
        if result.connectivity_id != self.connectivity_id:
            raise ValueError(
                "OversetRegistration.prepare.result.connectivity_id must equal OversetRegistration.connectivity_id; got a restart connectivity mismatch."
            )
        return result
