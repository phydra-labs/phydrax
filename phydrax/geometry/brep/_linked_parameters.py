#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Linked native numerical leaves within an immutable B-Rep design epoch."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._numerics._quadrature_rules import gauss_legendre_data
from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked
from .._atlas import CurveTrimLoop, DynamicCurveTrimLoop, TrimDomain
from .._interval_enclosure import (
    _evaluate_device_jaxpr,
    device_interval_evaluation_error,
)
from ..design._schema import DesignState, ParameterCollector
from ._measure import _coedge_quadrature_breaks, surface_quadrature_u_breaks
from ._model import BRepGeometry, BRepModel, BRepPCurve
from ._patches import AbstractSurfacePatch
from ._query import _curve_value, BRepQueryPolicy
from ._query_types import BRepQueryBudget


class BRepParameterGeometry(StrictModule):
    """One linked realization of native carriers, trims, bounds, and mesh UVs.

    Incidence, native entity identity, orientation, and source-root bindings are
    fixed by the model passed to the map's compile boundary. Numerical leaves
    are JAX values; constructors do not transfer them to the host. Omitted mesh
    UVs use affine transport between the source and realized parameter boxes.
    Non-affine trim changes must supply their linked triangle parameters.
    """

    patches: tuple[AbstractSurfacePatch, ...]
    geometry: BRepGeometry
    parameter_bounds: Array
    triangle_parameters: Array | None

    @checked
    def __init__(
        self,
        *,
        patches: Iterable[AbstractSurfacePatch],
        geometry: BRepGeometry,
        parameter_bounds: ArrayLike,
        triangle_parameters: ArrayLike | None = None,
    ) -> None:
        patches_ = tuple(patches)
        bounds = jnp.asarray(parameter_bounds, dtype=jnp.float64)
        if bounds.shape != (len(patches_), 2, 2):
            raise ValueError(
                f"BRepParameterGeometry.parameter_bounds must align with native faces; got shape={bounds.shape}."
            )
        self.patches = patches_
        self.geometry = geometry
        self.parameter_bounds = bounds
        self.triangle_parameters = (
            None
            if triangle_parameters is None
            else jnp.asarray(triangle_parameters, dtype=jnp.float64)
        )


class AbstractBoundBRepParameterMap(StrictModule):
    """Compiled map reading one global design state without host conversion."""

    @abstractmethod
    def realize(self, state: DesignState, /) -> BRepParameterGeometry:
        raise NotImplementedError


class AbstractBRepParameterMap(StrictModule):
    """Declare linked parameters through the shared global parameter collector."""

    @abstractmethod
    def compile(
        self, context: ParameterCollector, model: BRepModel, /
    ) -> AbstractBoundBRepParameterMap:
        raise NotImplementedError


class _LinkedGreenCoedge(StrictModule):
    coedge: int = eqx.field(static=True)
    fractions: Array


class LinkedBRepQuadrature(StrictModule):
    """Green-reduced quadrature with fixed layout and dynamic source trims.

    Unlike the host rule, zero-weight nodes are retained so changing trim
    coefficients cannot change the numerical tree or quadrature layout.
    """

    nodes: Array
    weights: Array
    face_coedges: tuple[tuple[_LinkedGreenCoedge, ...], ...]
    u_fractions: tuple[Array, ...]

    def __init__(
        self,
        model: BRepModel,
        /,
        *,
        coordinate_budget: CoordinateEnclosureBudget,
        policy: BRepQueryPolicy | None = None,
    ) -> None:
        policy = BRepQueryPolicy() if policy is None else policy
        rule = gauss_legendre_data(policy.quadrature_order)
        geometry = model.require_geometry()
        ranges, bounds = jax.device_get((geometry.edge_ranges, model.parameter_bounds))
        ranges, bounds = np.asarray(ranges), np.asarray(bounds)
        budget = BRepQueryBudget(
            policy.maximum_operations, policy.maximum_points, policy.maximum_scratch_bytes
        )
        face_coedges, u_fractions = [], []
        for face, loops in enumerate(geometry.face_loops):
            cuts = surface_quadrature_u_breaks(
                model.patches[face],
                bounds[face],
                budget,
                coordinate_budget=coordinate_budget,
            )
            u_fractions.append(
                jnp.asarray(
                    (cuts - bounds[face, 0, 0])
                    / (bounds[face, 1, 0] - bounds[face, 0, 0]),
                    dtype=jnp.float64,
                )
            )
            entries = []
            for loop in loops:
                for coedge in loop:
                    first, last = ranges[geometry.coedge_edges[coedge]]
                    cuts = _coedge_quadrature_breaks(
                        geometry.pcurves[coedge],
                        float(first),
                        float(last),
                        policy.quadrature_subdivisions,
                        budget,
                    )
                    count = (
                        (cuts.size - 1)
                        * policy.quadrature_order**2
                        * (u_fractions[-1].size - 1)
                    )
                    budget.admit(count, count, 96 * (budget.points + count))
                    budget.consume(count, points=count)
                    entries.append(
                        _LinkedGreenCoedge(
                            coedge=coedge,
                            fractions=jnp.asarray(
                                (cuts - first) / (last - first), dtype=jnp.float64
                            ),
                        )
                    )
            face_coedges.append(tuple(entries))
        self.nodes = jnp.asarray(rule.nodes, dtype=jnp.float64)
        self.weights = jnp.asarray(rule.weights, dtype=jnp.float64)
        self.face_coedges = tuple(face_coedges)
        self.u_fractions = tuple(u_fractions)

    def realize(self, value: BRepParameterGeometry, /) -> tuple[tuple[Array, Array], ...]:
        rules = []
        geometry = value.geometry
        for face, entries in enumerate(self.face_coedges):
            box = value.parameter_bounds[face]
            cuts = box[0, 0] + self.u_fractions[face] * (box[1, 0] - box[0, 0])
            parameters, weights = [], []
            for entry in entries:
                coedge = entry.coedge
                first, last = geometry.edge_ranges[geometry.coedge_edges[coedge]]
                breaks = first + entry.fractions * (last - first)
                half = 0.5 * (breaks[1:] - breaks[:-1])
                t = (0.5 * (breaks[1:] + breaks[:-1]))[:, None] + half[
                    :, None
                ] * self.nodes
                pcurve = geometry.pcurves[coedge]
                uv = pcurve.evaluate(t.reshape(-1))
                velocity = jax.vmap(jax.jacfwd(pcurve.evaluate))(t.reshape(-1))
                outer = (
                    geometry.coedge_senses[coedge]
                    * (half[:, None] * self.weights).reshape(-1)
                    * velocity[:, 1]
                )
                for lower, upper in zip(cuts[:-1], cuts[1:], strict=True):
                    span = 0.5 * jnp.maximum(jnp.minimum(uv[:, 0], upper) - lower, 0.0)
                    u = lower + span[:, None] * (1.0 + self.nodes)
                    parameters.append(
                        jnp.stack(
                            (u, jnp.broadcast_to(uv[:, 1, None], u.shape)), axis=-1
                        ).reshape(-1, 2)
                    )
                    weights.append(
                        (outer[:, None] * span[:, None] * self.weights).reshape(-1)
                    )
            rules.append((jnp.concatenate(parameters), jnp.concatenate(weights)))
        return tuple(rules)


def validate_linked_geometry(
    value: BRepParameterGeometry, model: BRepModel, /
) -> BRepParameterGeometry:
    """Keep scientific incidence immutable while validating numerical intervals."""
    original = model.require_geometry()
    geometry = value.geometry
    fields = (
        "edge_curves",
        "edge_vertices",
        "coedge_edges",
        "coedge_senses",
        "face_loops",
        "shell_faces",
        "shell_orientations",
        "solid_shells",
        "occurrences",
        "assembly_containers",
        "geometry_id",
    )
    if any(getattr(geometry, field) != getattr(original, field) for field in fields):
        raise ValueError(
            "BRepParameterGeometry.geometry must preserve fixed incidence, orientation, placements, and native identity; got a changed epoch."
        )
    if len(value.patches) != len(model.patches) or len(geometry.pcurves) != len(
        original.pcurves
    ):
        raise ValueError(
            "BRepParameterGeometry.patches must preserve native face and coedge counts; got a changed count."
        )
    if geometry.edge_ranges.shape != original.edge_ranges.shape:
        raise ValueError(
            f"BRepParameterGeometry.edge_ranges must preserve native edge count; got shape={geometry.edge_ranges.shape}."
        )
    if geometry.vertex_points.shape != original.vertex_points.shape or len(
        geometry.curves
    ) != len(original.curves):
        raise ValueError(
            "BRepParameterGeometry.geometry must preserve native vertex and curve counts; got a changed count."
        )
    roots = (
        geometry.vertex_roots,
        geometry.edge_endpoint_roots,
        geometry.coedge_endpoint_roots,
    )
    original_roots = (
        original.vertex_roots,
        original.edge_endpoint_roots,
        original.coedge_endpoint_roots,
    )
    if jax.tree_util.tree_structure(roots) != jax.tree_util.tree_structure(
        original_roots
    ):
        raise ValueError(
            "BRepParameterGeometry.geometry must preserve native root identity and structure; got changed source roots."
        )
    bounds = eqx.error_if(
        value.parameter_bounds,
        jnp.any(~jnp.isfinite(value.parameter_bounds))
        | jnp.any(value.parameter_bounds[:, 1] <= value.parameter_bounds[:, 0]),
        "BRepParameterGeometry.parameter_bounds must contain finite nonempty intervals; got invalid numerical bounds.",
    )
    ranges = eqx.error_if(
        geometry.edge_ranges,
        jnp.any(~jnp.isfinite(geometry.edge_ranges))
        | jnp.any(geometry.edge_ranges[:, 1] <= geometry.edge_ranges[:, 0]),
        "BRepParameterGeometry.edge_ranges must contain finite nonempty intervals; got invalid numerical ranges.",
    )
    geometry = eqx.tree_at(lambda item: item.edge_ranges, geometry, ranges)
    return eqx.tree_at(
        lambda item: (item.parameter_bounds, item.geometry), value, (bounds, geometry)
    )


def linked_triangle_parameters(
    value: BRepParameterGeometry, model: BRepModel, /
) -> Array:
    if value.triangle_parameters is not None:
        if value.triangle_parameters.shape != model.triangle_parameters.shape:
            raise ValueError(
                f"BRepParameterGeometry.triangle_parameters must preserve mesh corner layout; got shape={value.triangle_parameters.shape}."
            )
        return value.triangle_parameters
    faces = model.triangle_face_ids
    old, new = model.parameter_bounds[faces], value.parameter_bounds[faces]
    reference = (model.triangle_parameters - old[:, None, 0]) / (
        old[:, None, 1] - old[:, None, 0]
    )
    return new[:, None, 0] + reference * (new[:, None, 1] - new[:, None, 0])


def linked_curve_interval_box(curve: BRepPCurve, interval: Array, /) -> Array:
    """Enclose a native trim's whole parameter interval on device."""
    limits = jax.lax.stop_gradient(interval)
    program = jax.make_jaxpr(curve.evaluate)(limits[0])
    outputs = _evaluate_device_jaxpr(program, ((limits[0], limits[1]),))
    if len(outputs) != 1:
        raise ValueError(
            "BRepParameterGeometry.pcurve must return one native point array; got multiple outputs."
        )
    lower, upper = outputs[0]
    return jax.lax.stop_gradient(jnp.stack((lower, upper)))


def linked_trim_seam_evidence(
    value: BRepParameterGeometry,
    interval_box: Callable[[BRepPCurve, Array], Array],
    /,
) -> tuple[Array, Array]:
    """Evaluate native coedge joins with directed arithmetic-error allowances."""
    geometry = value.geometry
    samples = jnp.linspace(0.0, 1.0, 9, dtype=jnp.float64)
    uses: dict[int, list[tuple[Array, Array]]] = {}
    residuals, bounds = [], []

    def compare(first: Array, second: Array, error: Array) -> None:
        difference = first - second
        magnitude = jnp.max(jnp.abs(difference), axis=-1)
        normalized = difference / jnp.where(magnitude > 0.0, magnitude, 1.0)[..., None]
        residuals.append(jnp.max(magnitude * jnp.sqrt(jnp.sum(normalized**2, axis=-1))))
        frozen = jax.lax.stop_gradient(
            error + 16.0 * jnp.finfo(jnp.float64).eps * (jnp.abs(first) + jnp.abs(second))
        )
        allowance = jnp.nextafter(frozen, jnp.inf)
        bounds.append(jnp.nextafter(jnp.max(jnp.sum(allowance, axis=-1)), jnp.inf))

    for face, loops in enumerate(geometry.face_loops):
        patch = value.patches[face]
        for loop in loops:
            endpoints = []
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                first, last = geometry.edge_ranges[edge]
                parameters = first + samples * (last - first)
                pcurve = geometry.pcurves[coedge]

                def evaluate(parameter: Array) -> Array:
                    return patch.evaluate(pcurve.evaluate(parameter))

                points = evaluate(parameters)
                errors = jax.vmap(
                    lambda parameter: device_interval_evaluation_error(
                        evaluate, parameter
                    )
                )(jax.lax.stop_gradient(parameters))
                uses.setdefault(edge, []).append((points, errors))
                sense = geometry.coedge_senses[coedge]
                start, end = (0, -1) if sense > 0 else (-1, 0)
                endpoints.append((points[start], points[end], errors[start], errors[end]))
                uv = pcurve.evaluate(parameters)
                box = value.parameter_bounds[face]
                uv_error = jax.vmap(
                    lambda parameter: device_interval_evaluation_error(
                        pcurve.evaluate, parameter
                    )
                )(jax.lax.stop_gradient(parameters))
                escaped = jnp.maximum(box[0] - uv, uv - box[1])
                # Parameter-domain violations are refusal evidence, not geometry.
                invalid = jnp.any(escaped > uv_error)
                enclosure = interval_box(pcurve, geometry.edge_ranges[edge])
                interval_error = (
                    512.0
                    * jnp.finfo(jnp.float64).eps
                    * (1.0 + jnp.max(jnp.abs(enclosure)) + jnp.max(jnp.abs(box)))
                )
                invalid |= jnp.any(enclosure[0] < box[0] - interval_error) | jnp.any(
                    enclosure[1] > box[1] + interval_error
                )
                residuals.append(jnp.where(invalid, jnp.inf, 0.0))
            for current, following in zip(
                endpoints, (*endpoints[1:], endpoints[0]), strict=True
            ):
                compare(current[1], following[0], current[3] + following[2])
    for edge, edge_uses in uses.items():
        first, first_error = edge_uses[0]
        for points, errors in edge_uses[1:]:
            compare(first, points, first_error + errors)
        vertices = jnp.asarray(geometry.edge_vertices[edge], dtype=jnp.int32)
        compare(
            first[jnp.asarray((0, -1), dtype=jnp.int32)],
            geometry.vertex_points[vertices],
            first_error[jnp.asarray((0, -1), dtype=jnp.int32)],
        )
        curve_index = geometry.edge_curves[edge]
        if curve_index >= 0:
            curve = geometry.curves[curve_index]
            limits = geometry.edge_ranges[edge]
            parameters = limits[0] + samples * (limits[1] - limits[0])

            def evaluate_curve(parameter: Array) -> Array:
                return _curve_value(curve, parameter)

            curve_error = jax.vmap(
                lambda parameter: device_interval_evaluation_error(
                    evaluate_curve, parameter
                )
            )(jax.lax.stop_gradient(parameters))
            compare(first, evaluate_curve(parameters), first_error + curve_error)
    return jnp.max(jnp.stack(residuals)), jnp.max(jnp.stack(bounds))


class _LinkedTrimLoopPlan(StrictModule):
    coedges: tuple[int, ...] = eqx.field(static=True)
    arc_groups: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    first: Array
    last: Array


class LinkedTrimDomains(StrictModule):
    """Transport native source-arc covers without recreating host trim objects."""

    templates: tuple[TrimDomain, ...]
    plans: tuple[tuple[_LinkedTrimLoopPlan, ...], ...]

    def __init__(
        self, domains: tuple[TrimDomain, ...], geometry: BRepGeometry, /
    ) -> None:
        plans = []
        for domain, loops in zip(domains, geometry.face_loops, strict=True):
            face_plans = []
            for template, coedges in zip(domain.loops, loops, strict=True):
                if not isinstance(template, CurveTrimLoop):
                    raise ValueError(
                        "LinkedTrimDomains.templates must contain native source-curve covers; got another trim representation."
                    )
                indices, first, last = jax.device_get(
                    (template.arc_curves, template.arc_first, template.arc_last)
                )
                face_plans.append(
                    _LinkedTrimLoopPlan(
                        coedges=coedges,
                        arc_groups=tuple(
                            tuple(np.flatnonzero(np.asarray(indices) == index).tolist())
                            for index in range(len(coedges))
                        ),
                        first=jnp.asarray(first, dtype=jnp.float64),
                        last=jnp.asarray(last, dtype=jnp.float64),
                    )
                )
            plans.append(tuple(face_plans))
        self.templates = domains
        self.plans = tuple(plans)

    def realize(
        self,
        value: BRepParameterGeometry,
        interval_box: Callable[[BRepPCurve, Array], Array],
        /,
        *,
        normalized: bool = False,
    ) -> tuple[TrimDomain, ...]:
        domains = []
        geometry = value.geometry
        for face, (template, plans) in enumerate(
            zip(self.templates, self.plans, strict=True)
        ):
            box = value.parameter_bounds[face]
            loops = []
            for plan in plans:
                chord_array = jnp.zeros((plan.first.size, 2), dtype=jnp.float64)
                lower_array, upper_array = (
                    jnp.zeros_like(chord_array),
                    jnp.zeros_like(chord_array),
                )
                junctions = []
                for coedge, arc_indices in zip(
                    plan.coedges, plan.arc_groups, strict=True
                ):
                    indices = jnp.asarray(arc_indices, dtype=jnp.int32)
                    limits = geometry.edge_ranges[geometry.coedge_edges[coedge]]
                    fractions = jnp.stack(
                        (plan.first[indices], plan.last[indices]), axis=-1
                    )
                    if geometry.coedge_senses[coedge] < 0:
                        fractions = 1.0 - fractions
                    parameters = limits[0] + fractions * (limits[1] - limits[0])
                    pcurve = geometry.pcurves[coedge]
                    chord_array = chord_array.at[indices].set(
                        pcurve.evaluate(parameters[:, 0])
                    )
                    enclosures = jax.vmap(
                        lambda interval: interval_box(pcurve, interval)
                    )(jnp.sort(parameters, axis=-1))
                    lower_array = lower_array.at[indices].set(enclosures[:, 0])
                    upper_array = upper_array.at[indices].set(enclosures[:, 1])
                for coedge in plan.coedges:
                    edge = geometry.coedge_edges[coedge]
                    endpoint = 0 if geometry.coedge_senses[coedge] > 0 else 1
                    point = geometry.pcurves[coedge].evaluate(
                        geometry.edge_ranges[edge, endpoint]
                    )
                    junctions.append(point)
                junction_array = jnp.stack(junctions)
                if normalized:
                    width = box[1] - box[0]
                    chord_array = (chord_array - box[0]) / width
                    lower_array = (lower_array - box[0]) / width
                    upper_array = (upper_array - box[0]) / width
                    junction_array = (junction_array - box[0]) / width
                margin = (
                    256.0 * jnp.finfo(jnp.float64).eps * (1.0 + jnp.abs(junction_array))
                )
                loops.append(
                    DynamicCurveTrimLoop(
                        chords=chord_array,
                        arc_lower=lower_array,
                        arc_upper=upper_array,
                        junction_lower=junction_array - margin,
                        junction_upper=junction_array + margin,
                        source_id=template.trim_id,
                    )
                )
            domains.append(
                eqx.tree_at(
                    lambda domain: (domain.outer, domain.holes),
                    template,
                    (loops[0], tuple(loops[1:])),
                )
            )
        return tuple(domains)
