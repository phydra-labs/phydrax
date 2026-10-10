#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native overset connectivity of a `MeshAssembly` and its moving registrations.

A prepared overset connectivity binds the certified cell parts of one assembly
revision, their explicitly declared solid walls and overset (artificial) outer
boundaries, and one donor-resolution policy. Preparation is host-side and runs
in fixed phases:

1. *Hole cutting.* Declared PLC walls use bounded BVH ray candidates and exact
   crossing parity with symbolic perturbation, counting edge/vertex hits once.
   When a native B-Rep query is registered, its authoritative membership and
   explicit unresolved statuses own vertex classification instead. Contacts
   are blanked and reported. Exact represented facet/cell crossings and enclosed
   foreign wall vertices detect cut cells even without any enclosed mesh vertex;
   these cells are blanked and reported as under-resolved cuts.
2. *Fringe layers.* Unblanked vertices of blanked cells and the vertices of a
   part's overset boundary seed fringe layer one; each further layer adds the
   unclassified vertices of cells touching the previous layer. Fringe vertices
   are receptors, cells with a receptor vertex are fringe cells and only cells
   whose every vertex is active are donor-admissible, so donors never depend on
   other receptors. A part's own wall vertices are protected: a wall vertex that
   is not active is a reported conflict.
3. *Donor resolution.* Receptors are located by the donor part's inverse cell
   map restricted to admissible cells. Among containing donor parts the policy
   decides deterministically (finest donor cell, then declared part priority,
   then part name, then donor cell ID). A receptor without an admissible donor
   is an orphan with its reason.

Interpolative transfer is built as `PreparedFieldQuery` routes on the donor
field reconstruction (its pointwise evidence and exact transpose) and is never
conservative. The conservative mode is a separate common-refinement remap of
explicitly selected cell averages. Motion refits the wall hierarchies, repeats hole cutting
and donor resolution, and commits a new registration epoch through
`CompositionRebind` only when coverage and the state transfer of uncovered
vertices pass; otherwise the previous composition is returned.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from typing import assert_never, final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

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
    current_native_execution_budget,
    MeshcoreError,
    MeshcoreStatus,
    triangle_intersection_classes,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    AbstractCellLocator,
    CellGeometrySpec,
    CellLocationResult,
    CellLocationStatus,
    InterpolationTransposeEvidence,
    prepare_unstructured_conservative_remap,
    PreparedFieldQuery,
    PreparedFieldReconstruction,
    PreparedSimplicialCellLocator,
    PreparedUnstructuredConservativeRemap,
    SimplicialLocationPolicy,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from ..discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_geometry import (
    CellGeometryElement,
    CellVertexGeometryElement,
    RestrictedCellGeometryElement,
)
from ..discretization._coordinate_enclosure import outward
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._mapped_locator import PreparedMappedCellLocator
from ..discretization._polyhedral_locator import PreparedPolyhedralCellLocator
from ..discretization._simplicial_locator import LocatedCellMap
from ..discretization._view_support import (
    _MappedCellBoundaryMap,
    _MappedMeshSupportKernel,
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
from ..geometry import CommonRefinementPolicy, CompiledGeometry
from ..geometry._interval_enclosure import interval_add, interval_multiply
from ..geometry.brep._projection_contracts import BRepProjectionStatus
from ..geometry.brep._query import (
    BRepContainmentResult,
    BRepQueryBudget,
    BRepQueryResourceError,
    PreparedBRepQuery,
)
from ..lifecycle import (
    commit_composition_rebind,
    Composition,
    CompositionEntry,
    CompositionRebind,
    CompositionRebindReceipt,
    CompositionTransport,
)
from ..linalg import SmallLinearSolvePlan, solve_small_linear
from ..linalg._small_batched import prepare_exact_small_linear_actions
from ..typing import parse
from ._assembly import MeshAssembly, MeshPart
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._coupling import (
    _encoded_image_isometry,
    CouplingSearchEvidence,
    CouplingSearchStatus,
    MeshCoupling,
    MeshCouplingKind,
    OversetCoupling,
    OversetValueAction,
)
from ._measurements import measure_phase, NativeMeshingPhaseRecorder
from ._result import CellMeshingResult
from ._scope import MeshingScope


if TYPE_CHECKING:
    from ._overset_exchange import PreparedOwnerLocalOversetExchange


OversetDonorPriority: TypeAlias = Literal["finest-donor", "part-priority"]

_OWNER = "phydrax.meshing.overset"
_CONNECTIVITY_ENTRY = "overset:connectivity"
_SIMPLICES = {2: "triangle", 3: "tetrahedron"}
_SEARCH_METHOD = "native-overset-authoritative-hole-cut-admissible-donor"


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
    """Explicit overset role of one assembly part.

    `wall` names the facets (dimension `d - 1`) of the part's closed solid wall:
    it cuts holes in every other part and its vertices are protected in this
    part. `boundary` names the facets of the part's overset (artificial) outer
    boundary, whose vertices are receptors. Neither is inferred from names or
    coordinates. `priority` breaks donor ties under the policy (larger wins) and
    `owner_rank` is the execution rank that owns the part's donor packets.
    `excluded` is a full-dimensional cell scope never admitted as a donor.
    `solid_query` optionally owns authoritative native B-Rep or closed mapped
    domain membership. Possible wall/cell contacts lacking an intersection
    certificate are conservatively blanked and exposed as ambiguous cells.
    Paired `image_rotation` and `image_translation` author one registered
    support image, `x_world = R x_source + t`, for donors, receptors and walls.
    The original certified source carrier, coordinate axes and stable entity
    identities remain authoritative; this pose is not a periodic quotient.
    """

    part_name: str = eqx.field(static=True)
    wall: MeshingScope | None
    boundary: MeshingScope | None
    excluded: MeshingScope | None
    solid_query: PreparedBRepQuery | CompiledGeometry | None
    image_rotation: Array | None
    image_translation: Array | None
    solid_index: int = eqx.field(static=True)
    priority: int = eqx.field(static=True)
    owner_rank: int = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        part_name: str,
        /,
        *,
        wall: MeshingScope | None = None,
        boundary: MeshingScope | None = None,
        excluded: MeshingScope | None = None,
        solid_query: PreparedBRepQuery | CompiledGeometry | None = None,
        solid_index: int = 0,
        priority: int = 0,
        owner_rank: int = 0,
        image_rotation: ArrayLike | None = None,
        image_translation: ArrayLike | None = None,
    ) -> None:
        name = str(part_name).strip()
        if not name:
            raise ValueError("Overset part names must be non-empty.")
        for scope, label in (
            (wall, "wall"),
            (boundary, "boundary"),
            (excluded, "excluded"),
        ):
            if scope is not None and not isinstance(scope, MeshingScope):
                raise TypeError(f"{label} must be a MeshingScope or None.")
            if scope is not None and scope.source_id != name:
                raise ValueError(f"The {label} scope belongs to another part.")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise TypeError("priority must be an integer.")
        if isinstance(owner_rank, bool) or not isinstance(owner_rank, int):
            raise TypeError("owner_rank must be an integer.")
        if owner_rank < 0:
            raise ValueError("owner_rank must be non-negative.")
        if solid_query is not None and not isinstance(
            solid_query, (PreparedBRepQuery, CompiledGeometry)
        ):
            raise TypeError(
                "solid_query must be native B-Rep or whole mapped-cell support."
            )
        if isinstance(solid_query, CompiledGeometry):
            from ..discretization._view_support import _MappedMeshSupportKernel

            if not isinstance(solid_query.kernel, _MappedMeshSupportKernel):
                raise TypeError(
                    "Compiled wall authority must retain whole mapped-cell support."
                )
        if (
            isinstance(solid_index, bool)
            or not isinstance(solid_index, int)
            or solid_index < 0
        ):
            raise ValueError("solid_index must be a nonnegative integer.")
        if solid_query is not None and (
            wall is None
            or (
                isinstance(solid_query, PreparedBRepQuery)
                and solid_index >= len(solid_query.solid_faces)
            )
            or (isinstance(solid_query, CompiledGeometry) and solid_index != 0)
        ):
            raise ValueError(
                "A solid query needs its explicit protected wall and solid index."
            )
        if (image_rotation is None) != (image_translation is None):
            raise ValueError("A registered image requires both rotation and translation.")
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
                    "Registered images require finite planar or spatial isometries."
                )
            if not np.allclose(
                rotation.T @ rotation,
                np.eye(rotation.shape[0], dtype=np.float64),
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError("A registered image must be an isometry.")
        self.part_name = name
        self.wall = wall
        self.boundary = boundary
        self.excluded = excluded
        self.solid_query, self.solid_index = solid_query, solid_index
        self.image_rotation = None if rotation is None else jnp.asarray(rotation)
        self.image_translation = None if translation is None else jnp.asarray(translation)
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
        return self.image_rotation is None or _encoded_image_isometry(
            np.asarray(self.image_rotation)
        )

    def rebound(
        self,
        part: MeshPart,
        /,
        *,
        solid_query: PreparedBRepQuery | CompiledGeometry | None = None,
    ) -> OversetPartSpec:
        """Rebind the wall and boundary scopes to a moved revision of the part."""
        if not isinstance(part, MeshPart) or part.name != self.part_name:
            raise ValueError("A spec rebinds only to a revision of its own part.")
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
            "The authored image pullback failed its native linear solve.",
            stage="image-query",
            achieved=(
                ("image_solve_status", float(np.asarray(solved.status))),
                ("image_solve_condition", float(np.asarray(solved.condition_estimate))),
            ),
        )
    return np.asarray(solved.value).T.reshape(points.shape)


def _relative_image_actions(
    source: OversetPartSpec,
    target: OversetPartSpec,
    dimension: int,
    /,
) -> tuple[tuple[Fraction, ...], ...] | None:
    """Exact requested source-to-target actions of the authored binary64 maps."""
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
    solved = prepare_exact_small_linear_actions(matrix, right)
    if not solved.successful or solved.actions is None:
        raise ValueError(
            "A relative registered image must have an invertible authored coordinate map."
        )
    return solved.actions


def _relative_image(
    source: OversetPartSpec,
    target: OversetPartSpec,
    dimension: int,
    /,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    actions = _relative_image_actions(source, target, dimension)
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
) -> tuple[np.ndarray, np.ndarray]:
    """Outward full-source bounds using exact encoded-map solve actions."""
    dimension = lower.shape[-1]
    actions = _relative_image_actions(source, target, dimension)
    if actions is None:
        return lower, upper
    low, high = np.zeros_like(lower), np.zeros_like(upper)
    for source_axis in range(dimension):
        coefficient_low = np.asarray(
            [outward(row[source_axis], -np.inf) for row in actions], dtype=np.float64
        )
        coefficient_high = np.asarray(
            [outward(row[source_axis], np.inf) for row in actions], dtype=np.float64
        )
        term = interval_multiply(
            (coefficient_low, coefficient_high),
            (lower[..., source_axis, None], upper[..., source_axis, None]),
        )
        low, high = interval_add((low, high), term)
    translation_low = np.asarray(
        [outward(row[dimension], -np.inf) for row in actions], dtype=np.float64
    )
    translation_high = np.asarray(
        [outward(row[dimension], np.inf) for row in actions], dtype=np.float64
    )
    return interval_add((low, high), (translation_low, translation_high))


@final
class OversetPolicy(StrictModule, NonTrainableState):
    """Fringe depth, donor resolution and resource bounds of an overset assembly.

    `donor_priority="finest-donor"` prefers the containing donor cell of smallest
    measure, then the larger part priority; `"part-priority"` reverses the first
    two keys. Remaining ties are broken by part name and donor cell ID.
    `maximum_wall_candidate_pairs` is one shared wall allowance for ray/facet,
    cell/wall broad-phase pairs and authoritative B-Rep owner operations;
    `maximum_donor_candidate_pairs` bounds worst-case prepared point/cell
    routes across competing parts before allocation. Exhaustion never publishes
    a partial classification or silently ignores an unresolved competitor.
    """

    fringe_layers: int = eqx.field(static=True)
    donor_priority: OversetDonorPriority = eqx.field(static=True)
    location_policy: SimplicialLocationPolicy | None
    maximum_wall_candidate_pairs: int = eqx.field(static=True)
    maximum_donor_candidate_pairs: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        fringe_layers: int = 2,
        donor_priority: OversetDonorPriority = "finest-donor",
        location_policy: SimplicialLocationPolicy | None = None,
        maximum_wall_candidate_pairs: int = 50_000_000,
        maximum_donor_candidate_pairs: int = 50_000_000,
    ) -> None:
        for value, label in (
            (fringe_layers, "fringe_layers"),
            (maximum_wall_candidate_pairs, "maximum_wall_candidate_pairs"),
            (maximum_donor_candidate_pairs, "maximum_donor_candidate_pairs"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{label} must be an integer.")
            if value < 1:
                raise ValueError(f"{label} must be positive.")
        priority = parse(donor_priority, OversetDonorPriority, "donor_priority")
        if location_policy is not None and not isinstance(
            location_policy, SimplicialLocationPolicy
        ):
            raise TypeError("location_policy must be SimplicialLocationPolicy or None.")
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


# -------------------------------------------------------------------- evidence


@final
class OversetPartBlanking(StrictModule, NonTrainableState):
    """Hole/fringe classification of one part in mesh row order.

    `vertex_status`/`cell_status` hold `OversetVertexStatus`/`OversetCellStatus`
    codes and `fringe_layer` the receptor layer (0 for non-receptors).
    `wall_contact` marks authoritative boundary vertices (exact for PLC walls),
    `ambiguous_cells` blanked cuts without an enclosed vertex, including possible
    whole-source-box contacts lacking an intersection proof, and
    `protected_conflict` own-wall vertices that are not active.
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
            raise ValueError("Overset blanking must hold one record per vertex and cell.")
        self.vertex_ids = jnp.asarray(vertex_ids)
        self.vertex_status = jnp.asarray(arrays[0])
        self.fringe_layer = jnp.asarray(arrays[1])
        self.wall_contact = jnp.asarray(arrays[2])
        self.protected_conflict = jnp.asarray(arrays[3])
        self.cell_ids = jnp.asarray(cell_ids)
        self.cell_status = jnp.asarray(arrays[4])
        self.ambiguous_cells = jnp.asarray(arrays[5])
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
            raise TypeError("status must be OversetVertexStatus.")
        mask = np.asarray(self.vertex_status) == int(status)
        return np.sort(np.asarray(self.vertex_ids)[mask])

    def cell_ids_with(self, status: OversetCellStatus, /) -> np.ndarray:
        """Stable IDs of the cells with one status, in ascending order."""
        if not isinstance(status, OversetCellStatus):
            raise TypeError("status must be OversetCellStatus.")
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
            raise ValueError("Receptor evidence must hold one record per receptor.")
        if np.any(np.diff(arrays[0]) <= 0):
            raise ValueError("Receptor evidence follows ascending stable IDs.")
        (
            self.receptor_ids,
            self.receptor_rows,
            self.status,
            self.donor_parts,
            self.donor_cells,
            self.candidate_parts,
            self.residuals,
            self.donor_measures,
        ) = (jnp.asarray(value) for value in arrays)
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
    """Receptors served by one actual donor block, keyed by scientific IDs.

    `receptor_slots` index the receptor evidence in ascending stable-ID order.
    `donor_block` names the owning inverse chart and `donor_cell_rows` are local
    to that block; `donor_cells` remain original scientific cell IDs. Actual
    points, both part revisions, block identity and owner ranks bind the packet.
    Geometry-only packets do not declare a synthetic interpolation family.
    `points` are donor-source query sites, `target_points` are receptor sites in
    the target's stored source chart, and `receptor_points` are registered-world
    sites. `rotation`/`translation` map donor-source into target-source axes.
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
            raise ValueError("A donor packet holds one record per served receptor.")
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
                "Donor packets require sorted unique targets and exact finite revisions."
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
                "Donor packets must bind donor, target and registered-world image points."
            )
        rotation, translation = _relative_image(
            donor_part, receptor_part, arrays[2].shape[1]
        )
        self.receptor_points, self.target_points = jnp.asarray(world), jnp.asarray(target)
        self.rotation = None if rotation is None else jnp.asarray(rotation)
        self.translation = None if translation is None else jnp.asarray(translation)
        (
            self.receptor_ids,
            self.receptor_slots,
            self.points,
            self.donor_cells,
            self.donor_cell_rows,
        ) = (jnp.asarray(value) for value in arrays)
        self.donor_block = str(donor_block)
        if not self.donor_block:
            raise ValueError("A packet names its owning donor block.")
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
    """Owning charts and complete coordinate-source boxes in original cell order."""

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
    """An overset connectivity lacks complete coverage; see ``connectivity``."""

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
def _mapped_solid_kernel(query: CompiledGeometry, /) -> _MappedMeshSupportKernel:
    kernel = query.kernel
    if not isinstance(kernel, _MappedMeshSupportKernel):
        raise TypeError(
            "Overset mapped solids require their whole source-bound cell support."
        )
    return kernel


def _represented_linear_element(element: CellGeometryElement, /) -> bool:
    if isinstance(element, CellVertexGeometryElement):
        return True
    if isinstance(element, (FiniteElementSpec, RestrictedCellGeometryElement)):
        return element.degree == 1
    raise TypeError(
        "Represented overset cell domains require canonical vertex or finite-element coordinate sources."
    )


def _solid_query_id(query: PreparedBRepQuery | CompiledGeometry | None, /) -> str | None:
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
        raise TypeError("Native overset requires certified cell mesh parts.")
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
            "Native overset requires regular full-dimensional mapped or planar polyhedral cells."
        )
    return carrier


def _signs(result: PredicateResult, /) -> np.ndarray:
    if not bool(np.all(np.asarray(result.certain))):
        raise RuntimeError("Exact orientation predicates left an uncertain sign.")
    return np.asarray(result.signs, dtype=np.int8)


def _mode() -> PredicateMode:
    return resolve_host_predicate_mode(PredicateMode.EXACT)


def _facet_rows(part: MeshPart, scope: MeshingScope, /) -> np.ndarray:
    """Vertex rows of the declared boundary facets of one part."""
    part.require_scope(scope)
    mesh = _cell_result(part).mesh
    dimension = mesh.ambient_dimension
    if scope.entity_dimension != dimension - 1:
        raise ValueError("Overset walls and boundaries are facet scopes.")
    identifiers = np.asarray(mesh.entity_set(dimension - 1).entity_ids)
    order = np.argsort(identifiers, kind="stable")
    rows = order[np.searchsorted(identifiers[order], np.asarray(scope.entity_ids))]
    connectivity = mesh.connectivity
    if isinstance(connectivity, PolygonalConnectivity):
        facets = np.asarray(connectivity.edges, dtype=np.int64)
        boundary = np.asarray(connectivity.boundary_edges)
    elif isinstance(connectivity, PolyhedralConnectivity):
        offsets = np.asarray(connectivity.face_vertex_offsets)
        values = np.asarray(connectivity.face_vertex_values)
        loops = [values[offsets[row] : offsets[row + 1]] for row in rows]
        boundary = np.asarray(connectivity.face_neighbor) < 0
        if not np.all(boundary[rows]):
            raise ValueError("Overset scopes must contain exterior mesh facets.")
        width = max((len(loop) for loop in loops), default=0)
        return np.asarray(
            [np.pad(loop, (0, width - len(loop)), mode="edge") for loop in loops],
            dtype=np.int64,
        ).reshape((len(loops), width))
    elif isinstance(connectivity, (TetrahedralConnectivity, HexahedralConnectivity)):
        facets = np.asarray(connectivity.faces, dtype=np.int64)
        boundary = np.asarray(connectivity.boundary_faces)
    else:
        raise TypeError(
            "Overset facets require the owning polygonal or spatial cell incidence."
        )
    if not np.all(boundary[rows]):
        raise ValueError(
            f"Overset wall and boundary facets of {part.name!r} must be mesh "
            "boundary facets."
        )
    return facets[rows]


def _facet_loops(facets: np.ndarray, /) -> tuple[np.ndarray, ...]:
    return tuple(row[np.concatenate(([True], row[1:] != row[:-1]))] for row in facets)


def _require_closed(facets: np.ndarray, name: str, /) -> None:
    """A closed wall has two incident facets at every actual ridge."""
    loops = _facet_loops(facets)
    if not loops:
        raise ValueError(f"The overset wall of {name!r} is empty.")
    if len(loops[0]) == 2:
        ridges = np.concatenate(loops).reshape((-1, 1))
    else:
        ridges = np.sort(
            np.concatenate([np.stack((row, np.roll(row, -1)), axis=1) for row in loops]),
            axis=1,
        )
    _, counts = np.unique(ridges, axis=0, return_counts=True)
    if np.any(counts != 2):
        raise ValueError(f"The overset wall of {name!r} is not a closed surface.")


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


def _mapped_wall_bounds(query: CompiledGeometry, /) -> tuple[np.ndarray, np.ndarray]:
    """Enclose complete actual atlas facets by exact source compositions."""
    from ..discretization._coordinate_enclosure import (
        coordinate_polynomials,
        polynomial_bounds,
        restrict_coordinates,
    )
    from ..discretization._reference_cell import reference_cell_topology

    kernel = _mapped_solid_kernel(query)
    mapping = kernel.atlas.mapping
    if not isinstance(mapping, _MappedCellBoundaryMap):
        raise TypeError(
            "Mapped solid bounds require their canonical full-source facet map."
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
                cell_map.coordinate_element, coordinates[routes[cell]]
            )
        source = sources[int(cell)]
        expressions = (
            None
            if source is None
            else restrict_coordinates(
                source, cell_map.coordinate_element.cell_kind, origin, axes
            )
        )
        if expressions is None:
            # Complete source cell bounds remain conservative for every facet.
            lower.append(np.asarray(kernel.locator.cell_lower)[cell])
            upper.append(np.asarray(kernel.locator.cell_upper)[cell])
        else:
            domain = "simplex" if len(vertices) in (2, 3) else "box"
            boxes = [
                polynomial_bounds(value, domain, topology.dimension - 1)
                for value in expressions
            ]
            lower.append([box[0] for box in boxes])
            upper.append([box[1] for box in boxes])
    return np.asarray(lower), np.asarray(upper)


def _part_geometry(
    part: MeshPart,
    spec: OversetPartSpec,
    policy: OversetPolicy,
    previous: _PartGeometry | None,
    /,
) -> _PartGeometry:
    result = _cell_result(part)
    mesh = result.mesh
    if spec.image_rotation is not None and spec.image_rotation.shape != (
        mesh.ambient_dimension,
        mesh.ambient_dimension,
    ):
        raise ValueError(
            "The registered image must retain its source carrier's ambient coordinate axes."
        )
    polyhedral = any(block.cell_kind == "polyhedron" for block in mesh.blocks)
    discretization = None
    if polyhedral:
        from ..discretization import CellMesh

        elements, _, _ = result.geometry.resolve(mesh)
        if any(not _represented_linear_element(element) for element in elements):
            raise ValueError(
                "Face-defined polyhedral charts require actual planar vertex geometry."
            )
        connectivity = mesh.connectivity
        if not isinstance(connectivity, PolyhedralConnectivity):
            raise TypeError(
                "Polyhedral donor charts require canonical polyhedral incidence."
            )
        face_offsets = np.asarray(connectivity.face_vertex_offsets)
        face_values = np.asarray(connectivity.face_vertex_values)
        cell_offsets = np.asarray(connectivity.cell_face_offsets)
        cell_faces = np.asarray(connectivity.cell_face_values)
        cell_signs = np.asarray(connectivity.cell_face_sign_values)
        charts, offset = [], 0
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
            chart_mesh = CellMesh.from_mixed_3d(
                mesh.coordinates,
                (),
                polyhedra={block.name: tuple(loops)},
                polyhedral_cell_global_ids={block.name: block.global_ids},
                vertex_global_ids=mesh.vertex_global_ids,
            )
            charts.append(
                PreparedPolyhedralCellLocator(chart_mesh, policy.location_policy)
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
        affine = np.ones(vertices.shape[0], dtype=np.bool_)
    else:
        elements = {
            block.name: lagrange_element(block.cell_kind, 1) for block in mesh.blocks
        }
        discretization = FiniteElementPlan(
            mesh,
            (FiniteElementFieldSpec("state", elements),),
            coordinate_spec=result.geometry,
        ).prepare()
        coordinates = discretization.default_runtime.coordinates
        charts, measures_, affine_ = [], [], []
        for block_index, block in enumerate(mesh.blocks):
            cell_map = PreparedFiniteElementCellMap(discretization, block_index)
            location = policy.location_policy or SimplicialLocationPolicy(
                min(block.cell_count, 64), 16, 1
            )
            charts.append(PreparedMappedCellLocator(cell_map, coordinates, location))
            measures_.append(
                np.asarray(discretization.block_geometries[0][block_index].measure)
            )
            from ..discretization._reference_cell import reference_cell_topology

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
                                )
                            )
                            == 0
                        )
            affine_.append(planar)
        locators = tuple(charts)
        measures = np.concatenate(measures_)
        affine = np.concatenate(affine_)
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
    if spec.image_rotation is not None and not spec.image_isometry_exact:
        matrix = tuple(
            tuple(Fraction(float(value)) for value in row)
            for row in np.asarray(spec.image_rotation)
        )
        metric_action = prepare_exact_small_linear_actions(matrix, matrix)
        if not metric_action.successful:
            raise ValueError(
                "A registered support image must retain an invertible authored map."
            )
        scale = float(abs(metric_action.determinant))
        if scale != 1.0:
            measures = measures * scale
    lower = np.concatenate([np.asarray(locator.cell_lower) for locator in locators])
    upper = np.concatenate([np.asarray(locator.cell_upper) for locator in locators])
    facets, bvh = None, None
    if spec.wall is not None:
        rows = _facet_rows(part, spec.wall)
        _require_closed(rows, part.name)
        if isinstance(spec.solid_query, PreparedBRepQuery):
            bounds = np.asarray(spec.solid_query.bounds)
            wall_lower, wall_upper = bounds[0:1], bounds[1:2]
        elif isinstance(spec.solid_query, CompiledGeometry):
            wall_lower, wall_upper = _mapped_wall_bounds(spec.solid_query)
        else:
            if not polyhedral and not np.all(affine):
                raise ValueError(
                    "A non-affine protected wall needs native B-Rep or closed mapped-domain authority."
                )
            corners = np.asarray(mesh.coordinates, dtype=np.float64)[
                _triangulate_loops(rows, mesh.ambient_dimension)
            ]
            facets = jnp.asarray(corners)
            wall_lower, wall_upper = _box_bounds(corners)
        reuse = (
            previous is not None
            and previous.wall_bvh is not None
            and previous.wall_bvh.item_bbox_min.shape == wall_lower.shape
        )
        bvh = (
            refit_packed_bvh_bounds(previous.wall_bvh, wall_lower, wall_upper)
            if reuse
            else prepare_bvh(wall_lower, wall_upper, dtype=jnp.float64)
        )
    offsets = tuple(
        int(value)
        for value in np.concatenate(
            ([0], np.cumsum([block.cell_count for block in mesh.blocks]))
        )
    )
    return _PartGeometry(
        discretization=discretization,
        locators=locators,
        block_names=tuple(block.name for block in mesh.blocks),
        block_offsets=offsets,
        cell_measures=jnp.asarray(measures),
        cell_vertices=jnp.asarray(vertices),
        cell_lower=jnp.asarray(lower),
        cell_upper=jnp.asarray(upper),
        affine_cells=jnp.asarray(affine),
        cell_ids=jnp.concatenate([block.global_ids for block in mesh.blocks]),
        wall_facets=facets,
        wall_bvh=bvh,
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
    """Exact inside (odd crossing parity) and on-wall flags against one wall."""
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
                "Overset hole cutting exceeds maximum_wall_candidate_pairs.",
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


@dataclass(frozen=True, slots=True)
class _Blanking:
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
    """Actual planar boundary triangles/segments with original owning cell rows."""
    from ..discretization._reference_cell import reference_cell_topology

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
                "Overset wall/cell contacts exceed maximum_wall_candidate_pairs.",
                stage="hole-cut",
            )
        budget[0] -= rows.size
        first, second = boundary[rows], facets[items]
        if dimension == 2:
            result = segment_intersections_2d(
                first[:, 0], first[:, 1], second[:, 0], second[:, 1], mode=_mode()
            )
            if not np.all(result.certain):
                raise RuntimeError("Exact wall/cell contact left an uncertain decision.")
            contact = np.asarray(result.status) != 0
        else:
            classes, status = triangle_intersection_classes(first, second)
            if np.any(status != int(MeshcoreStatus.OK)):
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "Exact wall/cell contact could not classify its represented facets.",
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
                "Overset possible wall cuts exceed maximum_wall_candidate_pairs.",
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
) -> BRepContainmentResult:
    """Share actual host wall work and any original active native allowance."""
    execution = current_native_execution_budget()
    allowance = None if execution is None else execution.remaining()
    operations = (
        remaining[0]
        if allowance is None
        else min(remaining[0], allowance.remaining_work_units)
    )
    point_limit = (
        query.policy.maximum_points
        if allowance is None
        else min(
            query.policy.maximum_points,
            allowance.remaining_geometry_queries,
        )
    )
    query_budget = BRepQueryBudget(
        operations,
        maximum_points=point_limit,
        maximum_scratch_bytes=query.policy.maximum_scratch_bytes,
    )
    try:
        try:
            if execution is not None and allowance is not None:
                if allowance.status != MeshcoreStatus.OK:
                    raise MeshcoreError(allowance.status, "execution_remaining")
                execution.admit_work_bound(query_budget.maximum_operations)
            result = query.contains(points, solid=solid, budget=query_budget)
        finally:
            remaining[0] -= query_budget.operations
            if execution is not None and (query_budget.operations or query_budget.points):
                execution.charge(
                    work=query_budget.operations, geometry_queries=query_budget.points
                )
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


def _blank_part(
    index: int,
    parts: tuple[MeshPart, ...],
    specs: tuple[OversetPartSpec, ...],
    geometries: tuple[_PartGeometry, ...],
    policy: OversetPolicy,
    budget: list[int],
    /,
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
        facets = (
            None
            if geometry.wall_facets is None
            else np.asarray(geometry.wall_facets, dtype=np.float64)
        )
        rotation, translation = _relative_image(
            specs[other], spec, mesh.ambient_dimension
        )
        wall_bvh = geometry.wall_bvh
        if rotation is not None and translation is not None:
            if facets is not None:
                facets = _relative_points(specs[other], spec, facets)
            lower, upper = _image_bounds(
                np.asarray(wall_bvh.item_bbox_min),
                np.asarray(wall_bvh.item_bbox_max),
                specs[other],
                spec,
            )
            wall_bvh = prepare_bvh(lower, upper, dtype=jnp.float64)
        query_points = _relative_points(spec, specs[other], points)
        query = specs[other].solid_query
        if query is None:
            if geometry.wall_facets is None:
                raise ValueError(
                    "An explicit PLC wall hierarchy requires its owning wall facets."
                )
            inside, touching, used = _wall_classification(
                query_points,
                np.asarray(geometry.wall_facets),
                geometry.wall_bvh,
                budget[0],
            )
            budget[0] -= used
        elif isinstance(query, CompiledGeometry):
            from ..discretization._view_support import mapped_mesh_support_query

            classified = mapped_mesh_support_query(query, jnp.asarray(query_points))
            resolved = np.asarray(classified.successful) & np.asarray(
                classified.location.candidates_complete
            )
            if not np.all(resolved):
                raise MeshingFailure(
                    MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                    "Whole mapped-domain wall membership is unresolved.",
                    stage="hole-cut",
                    entity_ids=tuple(
                        int(value)
                        for value in np.asarray(mesh.vertex_global_ids)[~resolved]
                    ),
                )
            inside = np.asarray(classified.inside)
            # The owner supplies an exterior-only reference margin, not a
            # Euclidean distance and not an internal-cell zero set.
            kernel = _mapped_solid_kernel(query)
            margin = np.asarray(
                kernel.boundary_field(query.state, jnp.asarray(query_points))
            )
            touching = inside & (
                np.abs(margin) <= kernel.locator.policy.reference_tolerance
            )
        else:
            if points.shape[1] != 3:
                raise ValueError("Native B-Rep solid classification requires 3D parts.")
            classified = _bounded_brep_wall_membership(
                query,
                query_points,
                specs[other].solid_index,
                policy,
                budget,
            )
            exhausted = np.asarray(classified.resource_exhausted)
            if np.any(exhausted):
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "Native B-Rep membership exhausted its bounded source work.",
                    stage="hole-cut",
                    entity_ids=tuple(
                        int(value)
                        for value in np.asarray(mesh.vertex_global_ids)[exhausted]
                    ),
                    achieved=(
                        (
                            "wall_work",
                            float(policy.maximum_wall_candidate_pairs - budget[0]),
                        ),
                    ),
                )
            touching = np.asarray(classified.distances) <= query.tolerances.classifier
            resolved = np.asarray(classified.status) == int(BRepProjectionStatus.UNIQUE)
            if np.any(~resolved & ~touching):
                raise MeshingFailure(
                    MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                    "Authoritative B-Rep hole classification is unresolved.",
                    stage="hole-cut",
                    entity_ids=tuple(
                        int(value)
                        for value in np.asarray(mesh.vertex_global_ids)[
                            ~resolved & ~touching
                        ]
                    ),
                )
            inside = np.asarray(classified.inside)
        hole |= inside | touching
        contact |= touching
        if facets is not None:
            corners = np.unique(facets.reshape((-1, facets.shape[-1])), axis=0)
            for block, locator in enumerate(geometries[index].locators):
                located = locator.locate(jnp.asarray(corners))
                containing = np.asarray(located.candidate_cells)
                cut[
                    containing[containing >= 0] + geometries[index].block_offsets[block]
                ] = True
            if np.all(np.asarray(geometries[index].affine_cells)):
                cut |= _cut_cells(part, facets, wall_bvh, budget)
            else:
                cut |= _possible_cuts(geometries[index], wall_bvh, budget)
        else:
            cut |= _possible_cuts(geometries[index], wall_bvh, budget)
    enclosing = np.any(hole[cells], axis=1)
    cell_hole = enclosing | cut
    if spec.excluded is not None:
        identifiers = np.asarray(geometries[index].cell_ids)
        cell_hole |= np.isin(identifiers, np.asarray(spec.excluded.entity_ids))
    boundary = np.zeros_like(hole)
    if spec.boundary is not None:
        boundary[_facet_rows(part, spec.boundary).reshape((-1,))] = True
    protected = np.zeros_like(hole)
    if spec.wall is not None:
        protected[_facet_rows(part, spec.wall).reshape((-1,))] = True
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


# ------------------------------------------------------------ donor resolution


@dataclass(frozen=True, slots=True)
class _Donors:
    status: np.ndarray
    parts: np.ndarray
    cells: np.ndarray
    candidates: np.ndarray
    residuals: np.ndarray
    measures: np.ndarray
    barycentric: np.ndarray


def _admissible(blanking: OversetPartBlanking, /) -> Array:
    return jnp.asarray(blanking.cell_status) == int(OversetCellStatus.ACTIVE)


def _resolve_donors(
    points: np.ndarray,
    receptor_index: int,
    specs: tuple[OversetPartSpec, ...],
    geometries: tuple[_PartGeometry, ...],
    admissible: tuple[Array, ...],
    policy: OversetPolicy,
    /,
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
            "Overset donor preparation exceeds maximum_donor_candidate_pairs.",
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
            located = locator.locate(query, cell_mask=admissible[index][start:stop])
            status = np.asarray(located.status)
            complete = np.asarray(located.candidates_complete)
            found = (status == int(CellLocationStatus.LOCATED)) & complete
            unresolved |= ~complete | (
                ~found & (status != int(CellLocationStatus.OUTSIDE))
            )
            resource_exceeded |= status == int(CellLocationStatus.RESOURCE_EXCEEDED)
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
            selected = locator.cell_map.evaluate(
                locator.coordinates,
                jnp.asarray(np.maximum(rows, 0)),
                jnp.asarray(reference),
            )
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
                located = locator.locate(jnp.asarray(donor_points))
                complete = np.asarray(located.candidates_complete)
                location_status = np.asarray(located.status)
                blocked |= (location_status == int(CellLocationStatus.LOCATED)) & complete
                unresolved[orphan] |= ~complete | ~np.isin(
                    location_status,
                    (int(CellLocationStatus.LOCATED), int(CellLocationStatus.OUTSIDE)),
                )
                resource_exceeded[orphan] |= location_status == int(
                    CellLocationStatus.RESOURCE_EXCEEDED
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
    donors = _resolve_donors(
        points,
        index,
        specs,
        geometries,
        admissible,
        policy,
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
) -> tuple[OversetDonorPacket, ...]:
    """One exact revision-bound packet per served donor block."""
    target_sites = np.asarray(_cell_result(parts[index]).mesh.coordinates)[
        np.asarray(evidence.receptor_rows)
    ]
    sites = _image_points(specs[index], target_sites)
    if sites.shape != (
        evidence.receptor_ids.shape[0],
        _cell_result(parts[index]).mesh.ambient_dimension,
    ):
        raise ValueError("Packet points follow receptor evidence order.")
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
) -> tuple[list[OversetDonorPacket], list[OversetCoupling]]:
    """Donor packets and the published P1 vertex overlays of one receptor part."""
    receptor = parts[index]
    hole_ids = blanking[index].vertex_ids_with(OversetVertexStatus.HOLE)
    hole_scope = receptor.scope(0, hole_ids) if hole_ids.size else None
    packets = list(_donor_packets(index, parts, specs, geometries, evidence, donors))
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
        # Geometry registration never fabricates a vertex-P1 field for a mapped
        # or polyhedral donor. Actual field binding owns those overlays.
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


@final
class OversetConnectivity(StrictModule, NonTrainableState):
    """One prepared registration epoch of native overset connectivity.

    `assembly` is the input assembly with one published `OversetCoupling`
    overlay per (receptor part, donor part) pair, so generic assembly consumers
    see the same receptor/hole/donor contract as any overset provider.
    `complete` holds when no receptor is an orphan and no protected wall vertex
    is blanked or a receptor; ambiguous contacts and cut cells are reported and
    blanked conservatively.
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
            raise ValueError("Overset records must follow the assembly part order.")
        for part, mask, geometry in zip(
            assembly.parts, blanking, geometries, strict=True
        ):
            if mask.part_id != part.part_id or geometry.part_id != part.part_id:
                raise ValueError(
                    "Overset blanking and spatial evidence bind another part revision."
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
                raise ValueError("Donor packets bind stale geometry/part revisions.")
            keys = {
                (packet.receptor_part, int(identifier))
                for identifier in np.asarray(packet.receptor_ids)
            }
            if served.intersection(keys):
                raise ValueError("A receptor must occur in exactly one donor packet.")
            served.update(keys)
        if served != expected:
            raise ValueError(
                "Donor packets must cover exactly the found receptor identities."
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
            raise KeyError(f"Unknown overset part {name!r}.")
        return names.index(name)

    def blanking_of(self, name: str, /) -> OversetPartBlanking:
        return self.blanking[self.part_index(name)]

    def receptors_of(self, name: str, /) -> OversetReceptorEvidence:
        return self.receptors[self.part_index(name)]

    def require_complete(self) -> None:
        """Refuse a registration with orphan receptors or blanked protected walls."""
        if not self.complete:
            raise OversetConnectivityError(
                f"Overset connectivity is incomplete: {self.orphan_count} orphan "
                f"receptors, {self.conflict_count} protected wall conflicts.",
                self,
            )

    def moved(
        self,
        coordinates: Mapping[str, ArrayLike],
        /,
        *,
        geometry: Mapping[str, CellGeometrySpec] | None = None,
        solid_queries: Mapping[str, PreparedBRepQuery | CompiledGeometry] | None = None,
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
            raise TypeError("coordinates must map moved part names to coordinates.")
        names = tuple(spec.part_name for spec in self.specs)
        if any(name not in names for name in coordinates):
            raise ValueError("Overset motion names parts outside the assembly.")
        geometry_ = {} if geometry is None else dict(geometry)
        if not set(geometry_).issubset(coordinates):
            raise ValueError("Successor coordinate maps must belong to moved parts.")
        solid_queries_ = {} if solid_queries is None else dict(solid_queries)
        if not set(solid_queries_).issubset(coordinates):
            raise ValueError("Successor solid queries must belong to moved parts.")
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
                    "Moving an authoritative solid needs its successor B-Rep query."
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

    def reregister(
        self,
        replacements: Mapping[str, MeshPart],
        /,
        *,
        solid_queries: Mapping[str, PreparedBRepQuery | CompiledGeometry] | None = None,
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
            raise TypeError("replacements must map moved names to certified MeshParts.")
        names = tuple(part.name for part in self.assembly.parts)
        if not set(replacements).issubset(names):
            raise ValueError("Moving registration names parts outside the assembly.")
        queries = {} if solid_queries is None else dict(solid_queries)
        if not set(queries).issubset(replacements):
            raise ValueError("Successor solid queries belong to explicitly moved parts.")
        parts, specs = [], []
        for old, spec in zip(self.assembly.parts, self.specs, strict=True):
            part = replacements.get(old.name, old)
            if not isinstance(part, MeshPart) or part.name != old.name:
                raise ValueError("A successor must keep its registered part identity.")
            before, after = _cell_result(old), _cell_result(part)
            if before.mesh.topology_id != after.mesh.topology_id:
                raise ValueError(
                    "Motion preserves topology and every scientific entity ID."
                )
            if before.geometry.geometry_layout_id != after.geometry.geometry_layout_id:
                raise ValueError(
                    "Motion preserves the complete coordinate-element layout."
                )
            if before.boundary is not None and after.boundary is None:
                raise ValueError("Motion cannot drop the authoritative boundary model.")
            if before.certification is not None and after.certification is None:
                raise ValueError(
                    "Motion needs successor source/embedding/coverage certificates."
                )
            if before.region_evidence is not None and (
                after.region_evidence is None
                or before.region_evidence.cell_region_ids
                != after.region_evidence.cell_region_ids
            ):
                raise ValueError(
                    "Motion needs successor evidence preserving all material regions."
                )
            association_keys = lambda result: {
                (
                    item.association_kind,
                    item.source_id,
                    item.target_entity_set_id,
                    tuple(np.asarray(item.target_global_ids).tolist()),
                )
                for item in result.associations
            }
            if association_keys(before) != association_keys(after):
                raise ValueError(
                    "Motion cannot drop or replace scientific geometry associations."
                )
            for field in ("patches", "zones", "labels", "attributes"):

                def keys(result: CellMeshingResult) -> set[tuple[object, ...]]:
                    return {
                        (
                            item.name,
                            getattr(item, "role", None),
                            getattr(item, "material_id", None),
                            getattr(item, "region_role", None),
                            getattr(item, "component_shape", None),
                            item.scope.entity_dimension,
                            item.scope.entity_set_id,
                            tuple(np.asarray(item.scope.entity_ids).tolist()),
                        )
                        for item in getattr(result, field)
                    }

                if keys(before) != keys(after):
                    raise ValueError(
                        f"Motion cannot drop or relabel {field} obligations."
                    )
            query = queries.get(old.name)
            if spec.solid_query is not None and part.part_id != old.part_id:
                if spec.wall is None:
                    raise ValueError(
                        "An authoritative solid must retain its explicit protected wall scope."
                    )
                wall_rows = np.unique(_facet_rows(old, spec.wall))
                if not np.array_equal(
                    np.asarray(before.mesh.coordinates)[wall_rows],
                    np.asarray(after.mesh.coordinates)[wall_rows],
                ) and (
                    query is None
                    or _solid_query_id(query) == _solid_query_id(spec.solid_query)
                ):
                    raise ValueError(
                        "Moving an authoritative wall needs its successor closed-solid query."
                    )
            parts.append(part)
            specs.append(
                spec
                if part.part_id == old.part_id and query is None
                else spec.rebound(part, solid_query=query)
            )
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
                "The moving registration publishes its own overset overlays."
            )
        # Validate all external geometry/coordinate contracts before native work.
        assembly = MeshAssembly(tuple(parts), couplings=retained)
        return _prepare(
            assembly.parts,
            assembly.couplings,
            tuple(specs),
            self.policy,
            self.geometries,
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
            else _part_geometry(
                part, spec, policy, None if previous is None else previous[index]
            )
            for index, (part, spec) in enumerate(zip(parts, specs, strict=True))
        )
        if record_phase is not None:
            jax.block_until_ready(geometries)
    with measure_phase(record_phase, "region_classification"):
        budget = [policy.maximum_wall_candidate_pairs]
        blanking_records = []
        for index, part in enumerate(parts):
            values = _blank_part(index, parts, specs, geometries, policy, budget)
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
    with measure_phase(record_phase, "donor_location"):
        receptors, packets, overlays = [], [], []
        for index in range(len(parts)):
            evidence, donors = _receptor_evidence(
                index, parts, specs, geometries, blanking, policy
            )
            receptors.append(evidence)
            part_packets, part_overlays = _packets_and_overlays(
                index, parts, specs, blanking, geometries, evidence, donors
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
    if not isinstance(assembly, MeshAssembly):
        raise TypeError("assembly must be MeshAssembly.")
    policy_ = OversetPolicy() if policy is None else policy
    if not isinstance(policy_, OversetPolicy):
        raise TypeError("policy must be OversetPolicy or None.")
    declared = tuple(parts)
    if not all(isinstance(spec, OversetPartSpec) for spec in declared):
        raise TypeError("parts must contain OversetPartSpec values.")
    names = tuple(part.name for part in assembly.parts)
    if len(names) < 2:
        raise ValueError("Overset connectivity requires at least two parts.")
    by_name = {spec.part_name: spec for spec in declared}
    if len(by_name) != len(declared) or set(by_name) != set(names):
        raise ValueError("Declare the overset role of every assembly part exactly once.")
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
                "Excluded donor regions must be full-dimensional cell scopes."
            )
        if (
            spec.wall is not None
            and spec.boundary is not None
            and np.intersect1d(
                np.asarray(spec.wall.entity_ids), np.asarray(spec.boundary.entity_ids)
            ).size
        ):
            raise ValueError("Wall and overset boundary facets must be disjoint.")
    if any(isinstance(link, OversetCoupling) for link in assembly.couplings):
        raise ValueError("The assembly already carries overset overlays.")
    return _prepare(
        assembly.parts,
        assembly.couplings,
        specs,
        policy_,
        None,
        epoch=0,
        record_phase=record_phase,
    )


@final
class _PacketLocator(AbstractCellLocator, NonTrainableState):
    """Retain the chosen donor side at the packet's fixed receptor points."""

    base: (
        PreparedMappedCellLocator
        | PreparedSimplicialCellLocator
        | PreparedPolyhedralCellLocator
    )
    points: Array
    selected_cells: Array
    cell_map: LocatedCellMap
    coordinates: Array
    policy: SimplicialLocationPolicy
    locator_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: AbstractCellLocator,
        packet: OversetDonorPacket,
        /,
        *,
        cell_offset: int = 0,
    ) -> None:
        if not isinstance(
            base,
            (
                PreparedMappedCellLocator,
                PreparedSimplicialCellLocator,
                PreparedPolyhedralCellLocator,
            ),
        ):
            raise TypeError(
                "Overset packets require their canonical source-bound cell locator."
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

    @property
    def canonical_locator(
        self,
    ) -> (
        PreparedMappedCellLocator
        | PreparedSimplicialCellLocator
        | PreparedPolyhedralCellLocator
    ):
        return self.base

    def locate(
        self, points: ArrayLike, /, *, cell_mask: ArrayLike | None = None
    ) -> CellLocationResult:
        result = self.base.locate(points, cell_mask=cell_mask)
        # Support preparation queries other points. Only this packet's sites
        # carry a donor-side selection; those sites must retain that exact side.
        if not np.array_equal(np.asarray(points), np.asarray(self.points)):
            return result
        return self._select_packet_location(result)

    @eqx.filter_jit
    def _select_packet_location(
        self, result: CellLocationResult, /
    ) -> CellLocationResult:
        """Apply this packet's already selected donor side as one numeric action."""
        accepted = result.candidate_cells == self.selected_cells[:, None]
        found = jnp.any(accepted, axis=1) & result.successful & result.candidates_complete
        slot = jnp.argmax(accepted, axis=1)
        reference = result.candidate_reference[jnp.arange(slot.size), slot]
        evaluation = self.cell_map.evaluate(
            self.coordinates, self.selected_cells, reference
        )
        barycentric = (
            self.base._reference_weights(reference)
            if isinstance(
                self.base, (PreparedMappedCellLocator, PreparedSimplicialCellLocator)
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
                    int(CellLocationStatus.LOCATED),
                    jnp.where(
                        result.successful, int(CellLocationStatus.OUTSIDE), result.status
                    ),
                ),
                found,
                jnp.where(accepted & found[:, None], result.candidate_cells, -1),
                jnp.where(accepted[:, :, None], result.candidate_reference, 0),
            ),
        )


@final
class PreparedOversetFieldTransfer(StrictModule, NonTrainableState):
    """Actual FE/FV field queries at sorted receptor IDs, never a vertex downgrade.

    Coefficients follow each donor owner's full FE DOF or FV cell-average order,
    not sorted support-scope IDs. Outputs follow this route's ``receptors``.
    Signed high-order/Piola weights remain owned by the canonical field query.
    ``transpose`` is available only for coefficient-linear owners, and is not
    a conservation proof. FV requires an explicit owning reconstruction policy;
    ``field_name`` selects FE fields, never an implicit replacement FV policy.
    Supplying ``previous`` also fills newly uncovered vertices and excludes
    predecessor holes. Coefficients must be the accepted predecessor or
    material-attached state, including accepted field-filled fringe values;
    ``CompositionTransport`` owns that fill obligation. Those coefficients
    are interpreted on the candidate coordinate map. Every current FV
    coefficient-read stencil must remain active, not hole/fringe state.
    Image-bound fields require an explicit `value_action`: invariant components,
    polar vectors under exact source isometries, or explicitly contravariant
    components under numeric affine maps. Vector outputs are in each target's
    original stored source chart; applying that target's image rotation yields
    registered-world components. The donor field basis and source coordinate
    certificate are never replaced by an image mesh.
    """

    connectivity: OversetConnectivity
    assembly: MeshAssembly
    receptors: tuple[OversetReceptorEvidence, ...]
    packets: tuple[OversetDonorPacket, ...]
    previous_id: str | None = eqx.field(static=True)
    queries: tuple[PreparedFieldQuery, ...]
    field_couplings: tuple[OversetCoupling, ...]
    donor_names: tuple[str, ...] = eqx.field(static=True)
    coefficient_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    value_shape: tuple[int, ...] = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    value_action: OversetValueAction | None = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        connectivity: OversetConnectivity,
        discretizations: Mapping[
            str, FiniteElementDiscretization | PreparedFieldReconstruction
        ],
        field_name: str,
        /,
        *,
        previous: OversetConnectivity | None = None,
        record_phase: NativeMeshingPhaseRecorder | None = None,
        value_action: OversetValueAction | None = None,
    ) -> None:
        from ._measurements import phase_started, record_elapsed

        if record_phase is not None and not callable(record_phase):
            raise TypeError("record_phase must be a phase recorder callable or None.")
        action = (
            None
            if value_action is None
            else parse(value_action, OversetValueAction, "value_action")
        )
        if action is None and any(
            spec.image_rotation is not None for spec in connectivity.specs
        ):
            raise ValueError(
                "Image-bound fields require an explicit invariant, polar-vector or contravariant-vector value_action."
            )
        if action == "polar-vector" and any(
            not spec.image_isometry_exact for spec in connectivity.specs
        ):
            raise ValueError(
                "Polar-vector fields require exact encoded source-image isometries; declare contravariant-vector for an affine map."
            )
        interpolation_started = phase_started(record_phase)
        connectivity.require_complete()
        if previous is not None and not isinstance(previous, OversetConnectivity):
            raise TypeError("previous must be an accepted OversetConnectivity or None.")
        receptors, packets = connectivity.receptors, connectivity.packets
        if previous is not None:
            if connectivity.predecessor_id != previous.connectivity_id:
                raise ValueError(
                    "Motion interpolation must bind this registration's predecessor."
                )
            records, donor_packets = [], []
            parts = connectivity.assembly.parts
            for index, part in enumerate(parts):
                evidence, donors = _receptor_evidence(
                    index,
                    parts,
                    connectivity.specs,
                    connectivity.geometries,
                    connectivity.blanking,
                    connectivity.policy,
                    previous.blanking,
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
                    )
                )
            receptors, packets = tuple(records), tuple(donor_packets)
            if any(item.orphan_ids.size for item in receptors):
                raise OversetConnectivityError(
                    "Motion field transfer has uncovered or receptor points without valid old-state donors.",
                    connectivity,
                    field_evidence=receptors,
                )
        from ..discretization.finite_volume._field_view import (
            UnstructuredFiniteVolumeFieldReconstructionKernel,
        )

        if not isinstance(discretizations, Mapping):
            raise TypeError(
                "discretizations must map donors to actual FE or FV field owners."
            )
        names = tuple(sorted(discretizations))
        required = {packet.donor_part for packet in packets}
        if not required.issubset(names):
            raise ValueError("Every served donor needs its actual field binding.")
        if not names:
            raise ValueError("At least one actual donor field must be supplied.")
        reconstructions: dict[tuple[str, str], PreparedFieldReconstruction] = {}
        layouts: dict[str, PreparedFieldReconstruction] = {}
        for name in names:
            result = _cell_result(connectivity.assembly.part(name))
            binding = discretizations[name]
            if isinstance(binding, FiniteElementDiscretization):
                if binding.mesh.mesh_id != result.mesh.mesh_id:
                    raise ValueError(
                        "Donor FE fields are bound to another mesh revision."
                    )
                if (
                    binding.default_runtime.geometry_layout_id
                    != result.geometry.geometry_layout_id
                    or not np.array_equal(
                        np.asarray(binding.default_runtime.coordinates),
                        np.asarray(result.geometry.coordinates),
                    )
                ):
                    raise ValueError(
                        "Donor FE fields are bound to another coordinate map."
                    )
                served_blocks = {
                    packet.donor_block for packet in packets if packet.donor_part == name
                }
                # Unused supplied fields still retain and validate their real layout.
                if not served_blocks:
                    served_blocks = {
                        binding.dof_maps[binding._field_index(field_name)].block_names[0]
                    }

                for block in sorted(served_blocks):
                    block_index = binding.dof_maps[
                        binding._field_index(field_name)
                    ].block_names.index(block)
                    cell_map = PreparedFiniteElementCellMap(binding, block_index)
                    locator = None
                    if isinstance(
                        cell_map.coordinate_element, RestrictedCellGeometryElement
                    ):
                        geometry = connectivity.geometries[connectivity.part_index(name)]
                        policy = geometry.locators[
                            geometry.block_names.index(block)
                        ].policy
                        locator = PreparedMappedCellLocator(
                            cell_map, binding.default_runtime.coordinates, policy
                        )
                    reconstruction = prepare_finite_element_field_reconstruction(
                        binding,
                        field_name,
                        block_name=block,
                        locator=locator,
                        location_policy=connectivity.policy.location_policy,
                    )
                    reconstructions[name, block] = reconstruction
                    layouts[name] = reconstruction
            elif isinstance(binding, PreparedFieldReconstruction):
                kernel = binding.kernel
                if not isinstance(
                    kernel, UnstructuredFiniteVolumeFieldReconstructionKernel
                ):
                    raise TypeError(
                        "Explicit reconstruction must retain its owning unstructured FV operator."
                    )
                kernel.require_source_geometry(result.geometry)
                owning_geometry = getattr(kernel.discretization, "cell_geometry", None)
                coordinates = (
                    result.mesh.coordinates
                    if owning_geometry is None
                    else owning_geometry.coordinates
                )
                if (
                    kernel.discretization.mesh.mesh_id != result.mesh.mesh_id
                    or kernel.locator.cell_map.cell_count
                    != sum(block.cell_count for block in result.mesh.blocks)
                    or not np.array_equal(
                        np.asarray(kernel.locator.coordinates), np.asarray(coordinates)
                    )
                ):
                    raise ValueError(
                        "Donor FV fields are bound to another mesh/coordinate revision."
                    )
                for block in result.mesh.blocks:
                    reconstructions[name, block.name] = binding
                layouts[name] = binding
            else:
                raise TypeError(
                    "Each donor needs a finite-element discretization or owning PreparedFieldReconstruction."
                )
        shapes = tuple(layouts[name].coefficient_shape for name in names)
        value_shapes = {owner.value_shape for owner in reconstructions.values()}
        if len(value_shapes) != 1:
            raise ValueError("Overset donor fields must have one shared component shape.")
        queries, overlays = [], []
        for packet in packets:
            key = packet.donor_part, packet.donor_block
            if key not in reconstructions:
                raise ValueError("The selected donor block has no actual field binding.")
            owner = reconstructions[key]
            kernel = owner.kernel
            if not isinstance(
                kernel,
                (
                    FiniteElementFieldReconstructionKernel,
                    UnstructuredFiniteVolumeFieldReconstructionKernel,
                ),
            ):
                raise TypeError(
                    "Overset queries require the validated owning FE or FV reconstruction."
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
            admissible = np.array(
                _admissible(connectivity.blanking[donor_index]), copy=True
            )
            if previous is not None:
                admissible &= np.asarray(
                    previous.blanking[donor_index].cell_status
                ) != int(OversetCellStatus.HOLE)
            if isinstance(kernel, FiniteElementFieldReconstructionKernel):
                binding = discretizations[packet.donor_part]
                if not isinstance(binding, FiniteElementDiscretization):
                    raise TypeError(
                        "An FE packet must retain its actual finite-element discretization."
                    )
                reconstruction = prepare_finite_element_field_reconstruction(
                    binding,
                    field_name,
                    block_name=packet.donor_block,
                    locator=locator,
                    support_geometry=owner.support_geometry,
                )
                query = PreparedFieldQuery(reconstruction, packet.points)
                support_ids = np.unique(np.asarray(packet.donor_cells))
            elif isinstance(kernel, UnstructuredFiniteVolumeFieldReconstructionKernel):
                query = kernel.prepare_packet_query(
                    owner, locator, packet.points, jnp.asarray(admissible)
                )
                support_rows = kernel.query_support_rows(query)
                support_ids = np.asarray(connectivity.geometries[donor_index].cell_ids)[
                    support_rows
                ]
            holes = connectivity.blanking[target_index].vertex_ids_with(
                OversetVertexStatus.HOLE
            )
            overlays.append(
                OversetCoupling.from_field_query(
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
            )
            queries.append(query)
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
        self.value_shape = next(iter(value_shapes))
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
                "spaces": [layouts[name].reconstruction_id for name in names],
            }
        )
        if record_phase is not None:
            import jax

            # Synchronize the actual prepared route/evidence outputs, not an
            # asynchronous launch or unrelated source geometry/state arrays.
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
        """Whether every owning query admits an algebraic transpose."""
        return all(query.coefficient_linear for query in self.queries)

    def apply(self, coefficients: Mapping[str, ArrayLike], /) -> dict[str, Array]:
        """Evaluate receptors in the actual FE/FV owner's full coefficient order."""
        if set(coefficients) != set(self.donor_names):
            raise ValueError("Coefficients must name exactly the prepared donor fields.")
        arrays = {name: jnp.asarray(coefficients[name]) for name in self.donor_names}
        for name, shape in zip(self.donor_names, self.coefficient_shapes, strict=True):
            if arrays[name].shape != shape:
                raise ValueError(f"Field {name!r} must have coefficient shape {shape}.")
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
        """Exact algebraic transpose for coefficient-linear FE/FV owners only."""
        if not self.coefficient_linear:
            raise ValueError(
                "Nonlinear FV reconstruction has no global algebraic transpose; "
                "linearize the owning PreparedFieldQuery at a supplied coefficient state."
            )
        if set(cotangents) != {item.part_name for item in self.receptors}:
            raise ValueError("Cotangents must name every receptor part.")
        arrays = {name: jnp.asarray(value) for name, value in cotangents.items()}
        for item in self.receptors:
            if arrays[item.part_name].shape != (
                item.receptor_ids.size,
                *self.value_shape,
            ):
                raise ValueError("Receptor cotangents must follow sorted receptor IDs.")
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
    ) -> PreparedOwnerLocalOversetExchange:
        """Bind real sparse donor traffic to this fixed native query epoch.

        Explicit stable scalar coefficient IDs follow the actual flattened
        FE/FV coefficient layout. The exchange accepts only process-local
        owned fields and returns only process-local receptors; it does not
        gather complete donor fields or claim distributed geometric search.
        """
        from ._overset_exchange import PreparedOwnerLocalOversetExchange

        return PreparedOwnerLocalOversetExchange(
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
    ) -> InterpolationTransposeEvidence:
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


def prepare_overset_field_transfer(
    connectivity: OversetConnectivity,
    discretizations: Mapping[
        str, FiniteElementDiscretization | PreparedFieldReconstruction
    ],
    field_name: str = "state",
    /,
    *,
    previous: OversetConnectivity | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    value_action: OversetValueAction | None = None,
) -> PreparedOversetFieldTransfer:
    """Bind actual FE fields or explicit FV reconstruction owners and query overlays."""
    if not isinstance(connectivity, OversetConnectivity):
        raise TypeError("connectivity must be OversetConnectivity.")
    return PreparedOversetFieldTransfer(
        connectivity,
        discretizations,
        field_name,
        previous=previous,
        record_phase=record_phase,
        value_action=value_action,
    )


def prepare_overset_conservative_remap(
    source: MeshPart,
    target: MeshPart,
    /,
    *,
    source_cells: MeshingScope | None = None,
    target_cells: MeshingScope | None = None,
    policy: CommonRefinementPolicy | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> PreparedUnstructuredConservativeRemap:
    """Separate certified overlap remap of explicitly selected cell averages.

    Complete coverage is the default. Partial overlap requires an explicit
    common-refinement policy and retains that owner's uncovered-content ledger;
    neither this route nor interpolation invents values in uncovered volume.
    """

    def prepare(
        part: MeshPart, scope: MeshingScope | None
    ) -> UnstructuredFiniteVolumeDiscretization:
        from ..discretization import CellBlock, CellMesh

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
                "Certified overlap requires the actual represented planar-face cell domain, not erased mapped geometry."
            )
        if scope is not None:
            part.require_scope(scope)
            if scope.entity_dimension != mesh.topological_dimension:
                raise ValueError("Conservative overlap selections must be cell scopes.")
            chosen = set(np.asarray(scope.entity_ids).tolist())
            fixed, polyhedra, poly_ids = [], {}, {}
            for block in mesh.blocks:
                mask = np.isin(np.asarray(block.global_ids), tuple(chosen))
                if not np.any(mask):
                    continue
                identifiers = np.asarray(block.global_ids)[mask]
                if block.cell_kind != "polyhedron":
                    fixed.append(
                        CellBlock(
                            block.name,
                            block.cell_kind,
                            np.asarray(block.vertices)[mask],
                            global_ids=identifiers,
                        )
                    )
                else:
                    connectivity = mesh.connectivity
                    if not isinstance(connectivity, PolyhedralConnectivity):
                        raise TypeError(
                            "Selected polyhedral cells require their canonical full incidence."
                        )
                    cell_lookup = {
                        int(identifier): row
                        for row, identifier in enumerate(
                            np.asarray(connectivity.cell_global_ids)
                        )
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
                        selected.append(loops)
                    polyhedra[block.name], poly_ids[block.name] = selected, identifiers
            if not fixed and not polyhedra:
                raise ValueError("Conservative overlap selections must contain cells.")
            if mesh.ambient_dimension == 3:
                mesh = CellMesh.from_mixed_3d(
                    mesh.coordinates,
                    tuple(fixed),
                    polyhedra=polyhedra,
                    vertex_global_ids=mesh.vertex_global_ids,
                    polyhedral_cell_global_ids=poly_ids,
                    numeric_version=mesh.numeric_version,
                )
            else:
                mesh = CellMesh(
                    mesh.coordinates,
                    tuple(fixed),
                    vertex_global_ids=mesh.vertex_global_ids,
                    numeric_version=mesh.numeric_version,
                )
        # The FV owner validates planarity/closure/star geometry; common
        # refinement owns intersection and complete/partial coverage evidence.
        return UnstructuredFiniteVolumePlan.from_cell_mesh(mesh).prepare()

    with measure_phase(record_phase, "common_refinement"):
        remap = prepare_unstructured_conservative_remap(
            prepare(source, source_cells),
            prepare(target, target_cells),
            provenance="native-overset-certified-cell-overlap",
            policy=policy,
        )
        if record_phase is not None:
            jax.block_until_ready(remap)
    return remap


def prepare_overset_conservative_state_transport(
    source: CompositionEntry,
    target: CompositionEntry,
    states: tuple[CompositionEntry, ...],
    /,
    *,
    content_tolerance: float,
    policy: CommonRefinementPolicy | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> tuple[PreparedUnstructuredConservativeRemap, tuple[CompositionTransport, ...]]:
    """Execute complete-overlap cell-average state transport for a moving part.

    Physical fields, material densities and spatial history registers use the
    same certified physical intersections, with separate inventory evidence for
    every component. Categorical material labels are not cell-average fields.
    The caller supplies topology entries and every state disposition explicitly.
    Motion changing the represented domain needs an exterior-content ledger;
    this complete-overlap route never invents incoming values or discards outflow.
    """
    if not isinstance(source, CompositionEntry) or not isinstance(
        target, CompositionEntry
    ):
        raise TypeError("Conservative motion requires explicit topology entries.")
    if source.role != "topology" or target.role != "topology":
        raise ValueError("Conservative motion binds topology owner entries.")
    if not isinstance(source.value, MeshPart) or not isinstance(target.value, MeshPart):
        raise TypeError("Conservative motion topology entries must hold MeshParts.")
    old, new = source.value, target.value
    if old.name != new.name or source.entry_id != target.entry_id:
        raise ValueError("Conservative motion preserves the registered part identity.")
    if (
        source.structure_id != _cell_result(old).mesh.topology_id
        or target.structure_id != _cell_result(new).mesh.topology_id
        or source.revision_id != old.part_id
        or target.revision_id != new.part_id
    ):
        raise ValueError(
            "Conservative motion topology entries bind stale mesh identities."
        )
    tolerance = float(content_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Conserved-content tolerance must be finite and nonnegative.")
    if not states or any(not isinstance(state, CompositionEntry) for state in states):
        raise TypeError(
            "Conservative motion requires explicit cell-average state entries."
        )
    for state in states:
        if state.role not in ("physical-state", "model-state", "history"):
            raise ValueError(
                "Conservative cell averages must declare physical, material-model, or history state."
            )
        if state.structure_id != source.structure_id:
            raise ValueError("Cell-average state must bind the source cell topology.")
        bindings = [
            binding
            for binding in state.dependencies
            if binding.entry_id == source.entry_id and binding.facet == "revision"
        ]
        if len(bindings) != 1 or bindings[0].bound_id != source.revision_id:
            raise ValueError(
                "Cell-average state must bind the exact source part revision."
            )
    prepared = prepare_overset_conservative_remap(
        old,
        new,
        policy=policy,
        record_phase=record_phase,
    )
    remap = prepared.plan
    if not prepared.succeeded or remap is None:
        raise ValueError(f"Moving conservative state overlap refused: {prepared.reason}")
    if not remap.require_complete:
        raise ValueError(
            "Moving conservative state requires complete overlap or an explicit exterior-content owner."
        )
    transports = []
    for state in states:
        values = jnp.asarray(state.value)
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise ValueError(
                "Conservative state values must be real cell-average densities, not categorical labels."
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
        """Publish only complete coverage and successful owner state transports."""
        if not isinstance(accepted_boundary, bool):
            raise TypeError("accepted_boundary must be an explicit host bool decision.")
        return commit_composition_rebind(
            self.rebind,
            accepted_boundary=accepted_boundary and self.candidate.complete,
        )


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
    """Stage motion through CompositionRebind; never infer/drop/reset any state.

    Physical, history, RNG, field, geometry-associated and solver-derived entries
    outside the overset owner require their explicit caller dispositions. Stale
    bindings or missing obligations fail at staging. Failed coverage or transport
    returns the original composition object on commit.
    """
    if not isinstance(composition, Composition):
        raise TypeError("composition must be Composition.")
    if not isinstance(previous, OversetConnectivity) or not isinstance(
        candidate, OversetConnectivity
    ):
        raise TypeError("Moving registrations must be OversetConnectivity values.")
    if (
        candidate.epoch != previous.epoch + 1
        or candidate.predecessor_id != previous.connectivity_id
    ):
        raise ValueError(
            "Motion must advance exactly one epoch of this accepted registration."
        )
    if tuple(spec.part_name for spec in candidate.specs) != tuple(
        spec.part_name for spec in previous.specs
    ):
        raise ValueError("Motion preserves every registered part identity.")
    for old, new in zip(previous.assembly.parts, candidate.assembly.parts, strict=True):
        if _cell_result(old).mesh.topology_id != _cell_result(new).mesh.topology_id:
            raise ValueError(
                "Fixed-topology overset motion preserves all entity identities."
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
                "The composition does not hold this accepted registration revision."
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
    """Canonical immutable restart descriptor, excluding prepared spatial state."""

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
        result = _prepare(
            self.assembly.parts,
            self.assembly.couplings,
            self.specs,
            self.policy,
            None,
            epoch=self.epoch,
            predecessor_id=self.predecessor_id,
            record_phase=record_phase,
        )
        if result.connectivity_id != self.connectivity_id:
            raise ValueError(
                "Restart inputs do not reproduce the registered connectivity."
            )
        return result
