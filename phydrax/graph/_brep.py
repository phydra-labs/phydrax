"""Identity-keyed sparse B-Rep incidence and differentiable sample preparation."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._strict import StrictModule
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..geometry.brep._differentiable import FixedTopologyBRepRealization
from ..geometry.brep._edge_classification import BRepEdgeClassification
from ..geometry.brep._linked_parameters import BRepParameterGeometry, LinkedBRepQuadrature
from ..geometry.brep._measure import BRepMassProperties
from ..geometry.brep._model import BRepEntityId, BRepGeometry, BRepModel
from ..geometry.brep._patches import _surface_value_and_first_jets, SpherePatch
from ..geometry.brep._query import BRepQueryPolicy
from ..geometry.design._naming import RecipeEntityNames
from ..nn.operator.data import FunctionSamples
from ..typing import parse
from ._ir import GraphIR


# Type IDs are positions in these immutable vocabularies.
NODE_KINDS = ("face", "coedge", "edge", "vertex")
RELATION_KINDS = (
    "next",
    "previous",
    "mate",
    "seam_mate",
    "coedge_face",
    "face_coedge",
    "coedge_edge",
    "edge_coedge",
    "edge_vertex",
    "vertex_edge",
    "face_adjacent",
)
SurfaceFamily: TypeAlias = Literal[
    "plane",
    "cylinder",
    "cone",
    "sphere",
    "torus",
    "bspline",
    "extrusion",
    "revolution",
    "unsupported",
]
CurveFamily: TypeAlias = Literal[
    "line", "circle", "ellipse", "bspline", "degenerate", "unsupported"
]


def _surface_family(name: str) -> SurfaceFamily:
    family = {
        "PlanePatch": "plane",
        "CylinderPatch": "cylinder",
        "ConePatch": "cone",
        "SpherePatch": "sphere",
        "TorusPatch": "torus",
        "BSplineSurfacePatch": "bspline",
        "ExtrusionSurface": "extrusion",
        "RevolutionSurface": "revolution",
    }.get(name, "unsupported")
    return parse(family, SurfaceFamily, "BRepGraph.surface_family")


def _curve_family(name: str) -> CurveFamily:
    family = {
        "LineCurve": "line",
        "CircleCurve": "circle",
        "EllipseCurve": "ellipse",
        "BSplineCurve": "bspline",
        "degenerate": "degenerate",
    }.get(name, "unsupported")
    return parse(family, CurveFamily, "BRepGraph.curve_family")


@final
class BRepIncidenceGraph(StrictModule):
    """Sparse definition incidence with explicit source identity.

    ``nodes['coedge_type']`` is 0 for regular uses, 1 for seam uses, and 2
    for collapsed pole uses; it is meaningful only on coedge rows.
    Type IDs index ``node_kinds`` and ``relation_kinds``.
    ``recipe_names`` owns geometry-independent face/edge identity, where supported.
    """

    graph: GraphIR
    entity_ids: tuple[BRepEntityId, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    node_kinds: tuple[str, ...] = eqx.field(static=True)
    relation_kinds: tuple[str, ...] = eqx.field(static=True)
    surface_families: tuple[SurfaceFamily, ...] = eqx.field(static=True)
    curve_families: tuple[CurveFamily, ...] = eqx.field(static=True)
    loop_counts: tuple[int, ...] = eqx.field(static=True)
    mass_properties: BRepMassProperties | None
    edge_classifications: tuple[BRepEdgeClassification, ...]
    recipe_names: RecipeEntityNames | None = eqx.field(static=True)


def _loop_routes(
    geometry: BRepGeometry,
    offset: int,
    faces: list[int],
    route: Callable[[int, int, str, int], None],
) -> None:
    """Publish cyclic coedge uses and their face owners."""
    for face, loops in enumerate(geometry.face_loops):
        for loop in loops:
            for position, coedge in enumerate(loop):
                faces[coedge] = face
                node = offset + coedge
                route(node, offset + loop[(position + 1) % len(loop)], "next", -1)
                route(node, offset + loop[position - 1], "previous", -1)
                route(node, face, "coedge_face", -1)
                route(face, node, "face_coedge", -1)


def _edge_routes(
    geometry: BRepGeometry,
    offsets: list[int],
    faces: list[int],
    edge_count: int,
    route: Callable[[int, int, str, int], None],
) -> list[list[int]]:
    """Publish carrier incidence and paired uses without dense face features."""
    uses: list[list[int]] = [[] for _ in range(edge_count)]
    for coedge, edge in enumerate(geometry.coedge_edges):
        uses[edge].append(coedge)
        route(offsets[1] + coedge, offsets[2] + edge, "coedge_edge", edge)
        route(offsets[2] + edge, offsets[1] + coedge, "edge_coedge", edge)
    for edge, coedges in enumerate(uses):
        for first, second in combinations(coedges, 2):
            kind = "seam_mate" if faces[first] == faces[second] else "mate"
            route(offsets[1] + first, offsets[1] + second, kind, edge)
            route(offsets[1] + second, offsets[1] + first, kind, edge)
            if faces[first] != faces[second]:
                route(faces[first], faces[second], "face_adjacent", edge)
                route(faces[second], faces[first], "face_adjacent", edge)
        for vertex in geometry.edge_vertices[edge]:
            route(offsets[2] + edge, offsets[3] + vertex, "edge_vertex", edge)
            route(offsets[3] + vertex, offsets[2] + edge, "vertex_edge", edge)
    return uses


def brep_incidence_graph(
    model: BRepModel,
    /,
    *,
    mass_properties: BRepMassProperties | None = None,
    edge_classifications: tuple[BRepEdgeClassification, ...] = (),
    recipe_names: RecipeEntityNames | None = None,
) -> BRepIncidenceGraph:
    """Build definition incidence in model face/coedge/edge/vertex row order."""
    geometry = model.require_geometry()
    if recipe_names is not None and recipe_names.model_id != model.model_id:
        raise ValueError(
            f"BRepGraph.recipe_names must belong to model_id={model.model_id}; got {recipe_names.model_id}"
        )
    if mass_properties is not None and mass_properties.model_id != model.model_id:
        raise ValueError(
            f"BRepGraph.mass_properties must belong to model_id={model.model_id}; got {mass_properties.model_id}"
        )
    if edge_classifications and (
        len(edge_classifications) != model.topology.num_edges
        or any(
            record.model_id != model.model_id or record.edge_id.index != index
            for index, record in enumerate(edge_classifications)
        )
    ):
        raise ValueError(
            f"BRepGraph.edge_classifications must follow model edge order; got count={len(edge_classifications)}"
        )
    counts = (
        model.topology.num_faces,
        len(geometry.coedge_edges),
        model.topology.num_edges,
        geometry.vertex_points.shape[0],
    )
    offsets = np.cumsum((0, *counts[:-1])).tolist()
    faces = [-1] * counts[1]
    rows: list[tuple[int, int, int, int]] = []

    def route(source: int, target: int, kind: str, edge: int = -1) -> None:
        rows.append((source, target, RELATION_KINDS.index(kind), edge))

    _loop_routes(geometry, offsets[1], faces, route)
    uses = _edge_routes(geometry, offsets, faces, counts[2], route)
    data = np.asarray(rows, dtype=np.int32).reshape((-1, 4))
    coedge_type = [
        2
        if geometry.edge_curves[e] < 0
        else (1 if len(uses[e]) == 2 and faces[uses[e][0]] == faces[uses[e][1]] else 0)
        for e in geometry.coedge_edges
    ]
    graph = GraphIR(
        nodes={
            "type": jnp.asarray(np.repeat(np.arange(4, dtype=np.int32), counts)),
            "coedge_type": jnp.asarray(
                [0] * counts[0] + coedge_type + [0] * (counts[2] + counts[3]),
                dtype=jnp.int32,
            ),
        },
        edges={"type": jnp.asarray(data[:, 2]), "brep_edge": jnp.asarray(data[:, 3])},
        senders=data[:, 0],
        receivers=data[:, 1],
        n_node=[sum(counts)],
        n_edge=[len(rows)],
    )
    ids = tuple(
        BRepEntityId(model.report.source_revision, kind, index)
        for kind, count in zip(NODE_KINDS, counts, strict=True)
        for index in range(count)
    )
    return BRepIncidenceGraph(
        graph=graph,
        entity_ids=ids,
        model_id=model.model_id,
        node_kinds=NODE_KINDS,
        relation_kinds=RELATION_KINDS,
        surface_families=tuple(
            _surface_family(type(patch).__name__) for patch in model.patches
        ),
        curve_families=tuple(
            _curve_family(
                "degenerate" if index < 0 else type(geometry.curves[index]).__name__
            )
            for index in geometry.edge_curves
        ),
        loop_counts=tuple(len(loops) for loops in geometry.face_loops),
        mass_properties=mass_properties,
        edge_classifications=edge_classifications,
        recipe_names=recipe_names,
    )


@final
class BRepGraphSamples(StrictModule):
    face_grids: tuple[Array, ...]
    trim_status: tuple[Array, ...]
    normal_status: tuple[Array, ...]
    face_quadrature: tuple[FunctionSamples, ...]
    coedge_points: tuple[Array, ...]


@final
class BRepGraphSamplePlan(StrictModule):
    parameters: tuple[Array, ...]
    trim_status: tuple[Array, ...]
    quadrature_rules: tuple[tuple[Array, Array], ...]
    linked_quadrature: LinkedBRepQuadrature
    reference_bounds: Array
    coedge_parameters: tuple[Array, ...]
    coedge_faces: tuple[int, ...] = eqx.field(static=True)
    coedge_edges: tuple[int, ...] = eqx.field(static=True)
    coedge_senses: tuple[int, ...] = eqx.field(static=True)
    coedge_samples: int = eqx.field(static=True)
    orientation: Array
    model_id: str = eqx.field(static=True)
    grid_size: int = eqx.field(static=True)

    def evaluate(self, realization: FixedTopologyBRepRealization, /) -> BRepGraphSamples:
        """Evaluate linked JAX samples; trim bands retain unresolved status."""
        if len(realization.patches) != len(self.parameters):
            raise ValueError(
                f"BRepGraph.realization.patches must match prepared face count; got {len(realization.patches)}"
            )
        bounds = (
            self.reference_bounds
            if realization.parameter_bounds is None
            else realization.parameter_bounds
        )
        rules = self.quadrature_rules
        if realization.geometry is not None:
            rules = self.linked_quadrature.realize(
                BRepParameterGeometry(
                    patches=realization.patches,
                    geometry=realization.geometry,
                    parameter_bounds=bounds,
                )
            )
        grids, statuses, quadratures, trim_status = [], [], [], []
        for face, patch in enumerate(realization.patches):
            original = self.reference_bounds[face]
            reference = (self.parameters[face] - original[0]) / (
                original[1] - original[0]
            )
            uv = bounds[face, 0] + reference * (bounds[face, 1] - bounds[face, 0])
            domain = realization.atlas.trim_domains[face]
            if domain is None:
                trim_status.append(
                    jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
                )
            elif realization.geometry is None:
                trim_status.append(self.trim_status[face])
            else:
                # Chord winding equals source winding outside the certified band;
                # boundary/band points are unresolved, never tolerance-masked.
                status = jnp.where(
                    domain.in_band(reference),
                    3,
                    jnp.where(domain.contains(reference), 0, 1),
                )
                trim_status.append(
                    status.astype(jnp.int32).reshape((self.grid_size, self.grid_size))
                )
            points, differential = jax.vmap(
                lambda p: _surface_value_and_first_jets(patch, p)
            )(uv)
            area_vectors = jnp.cross(differential[:, :, 0], differential[:, :, 1])
            squared_norm = jnp.sum(area_vectors**2, axis=-1)
            singular = squared_norm == 0
            norm = jnp.sqrt(jnp.where(singular, 1.0, squared_norm))
            if isinstance(patch, SpherePatch):
                singular = singular | (jnp.abs(uv[:, 1]) == jnp.pi / 2)
            normals = (
                area_vectors
                * self.orientation[face]
                / jnp.where(singular, 1.0, norm)[:, None]
            )
            normals = jnp.where(singular[:, None], 0.0, normals)
            grids.append(
                jnp.concatenate((points, normals), axis=-1).T.reshape(
                    (6, self.grid_size, self.grid_size)
                )
            )
            statuses.append(
                singular.astype(jnp.int32).reshape((self.grid_size, self.grid_size))
            )
            q_uv, weights = rules[face]
            q_points, q_diff = jax.vmap(
                lambda p: _surface_value_and_first_jets(patch, p)
            )(q_uv)
            squared_area = jnp.sum(
                jnp.cross(q_diff[:, :, 0], q_diff[:, :, 1]) ** 2, axis=-1
            )
            q_area = jnp.where(
                squared_area > 0,
                jnp.sqrt(jnp.where(squared_area > 0, squared_area, 1.0)),
                0.0,
            )
            quadratures.append(
                FunctionSamples(
                    values=None,
                    coordinates=q_points,
                    quadrature_weights=weights * q_area,
                    support_id=f"{self.model_id}:face:{face}:quadrature",
                    measure_id=f"{self.model_id}:face:{face}:area",
                )
            )
        coedges = []
        fractions = jnp.linspace(0.0, 1.0, self.coedge_samples, dtype=jnp.float64)
        for index, face in enumerate(self.coedge_faces):
            if realization.geometry is None:
                uv = self.coedge_parameters[index]
            else:
                geometry = realization.geometry
                limits = geometry.edge_ranges[self.coedge_edges[index]]
                oriented = fractions if self.coedge_senses[index] > 0 else 1.0 - fractions
                parameters = limits[0] + oriented * (limits[1] - limits[0])
                uv = geometry.pcurves[index].evaluate(parameters)
            coedges.append(realization.patches[face].evaluate(uv))
        return BRepGraphSamples(
            face_grids=tuple(grids),
            trim_status=tuple(trim_status),
            normal_status=tuple(statuses),
            face_quadrature=tuple(quadratures),
            coedge_points=tuple(coedges),
        )


def prepare_brep_graph_samples(
    model: BRepModel,
    /,
    *,
    grid_size: int,
    coedge_samples: int,
    policy: BRepQueryPolicy,
    budget: CoordinateEnclosureBudget,
) -> BRepGraphSamplePlan:
    """Reserve sample work through the caller's ticket before host preparation."""
    if grid_size < 2 or coedge_samples < 2:
        raise ValueError(
            f"BRepGraph.sample_counts must be at least two; got grid={grid_size}, coedge={coedge_samples}"
        )
    geometry = model.require_geometry()
    count = (
        len(model.patches) * grid_size**2 + len(geometry.coedge_edges) * coedge_samples
    )
    band_rows = max(
        (
            sum(loop.chords.shape[0] for loop in domain.loops)
            for domain in model.trim_domains
            if domain is not None
        ),
        default=0,
    )
    # The quadrature owner enforces its own maxima; reserve those declared
    # upper bounds on the request ticket before it allocates any nodes.
    budget.reserve(
        count + policy.maximum_operations,
        count * (128 + band_rows * 64)
        + policy.maximum_scratch_bytes
        + policy.maximum_points * 64,
    )
    bounds = np.asarray(model.parameter_bounds)
    parameters, statuses = [], []
    for face, domain in enumerate(model.trim_domains):
        first = np.linspace(bounds[face, 0, 0], bounds[face, 1, 0], grid_size)
        second = np.linspace(bounds[face, 0, 1], bounds[face, 1, 1], grid_size)
        uv = np.stack(np.meshgrid(first, second, indexing="ij"), axis=-1).reshape((-1, 2))
        parameters.append(jnp.asarray(uv))
        if domain is None:
            status = np.zeros(grid_size**2, dtype=np.int32)
        else:
            result = domain.classify(uv, budget=budget)
            status = np.where(
                result.resolved,
                np.where(result.boundary, 2, np.where(result.inside, 0, 1)),
                3,
            ).astype(np.int32)
        statuses.append(jnp.asarray(status.reshape((grid_size, grid_size))))
    faces = [-1] * len(geometry.coedge_edges)
    for face, loops in enumerate(geometry.face_loops):
        for loop in loops:
            for coedge in loop:
                faces[coedge] = face
    coedges = []
    for coedge, edge in enumerate(geometry.coedge_edges):
        limits = np.asarray(geometry.edge_ranges[edge])
        if geometry.coedge_senses[coedge] < 0:
            limits = limits[::-1]
        t = jnp.asarray(np.linspace(limits[0], limits[1], coedge_samples))
        coedges.append(geometry.pcurves[coedge].evaluate(t))
    linked = LinkedBRepQuadrature(model, coordinate_budget=budget, policy=policy)
    rules = linked.realize(
        BRepParameterGeometry(
            patches=model.patches,
            geometry=geometry,
            parameter_bounds=model.parameter_bounds,
        )
    )
    return BRepGraphSamplePlan(
        parameters=tuple(parameters),
        trim_status=tuple(statuses),
        quadrature_rules=rules,
        linked_quadrature=linked,
        reference_bounds=model.parameter_bounds,
        coedge_parameters=tuple(coedges),
        coedge_faces=tuple(faces),
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        coedge_samples=coedge_samples,
        orientation=model.orientation,
        model_id=model.model_id,
        grid_size=grid_size,
    )
