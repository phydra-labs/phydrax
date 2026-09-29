#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded-face traces of global tensor spectral fields.

A global tensor spectral field is one smooth synthesis over its axis box, so
its trace on a bounded face `x_a = a` (or `b`) is the synthesis evaluated on the
face: exact and single valued. The exterior facets are the faces of the bounded
axes, one entity per `(axis, lower/upper)` with local face `2 * axis + side`
(`side` 0 lower, 1 upper), in ascending axis order with the lower face first.
Periodic (Fourier) axes have no faces and no outward normal; unbounded axes
have none either; there are no interior facets.

Sites are either the native tangential nodes with their prepared quadrature
weights (Clenshaw--Curtis on Chebyshev--Lobatto axes, Gauss, Radau, or Lobatto
on Legendre axes, uniform on periodic axes) or the points of a `FacetTraceRule`
mapped onto the face. The route is sum factorized: every face contracts the
modal coefficients with the endpoint basis row of its normal axis and the
tangential basis rows at its sites; no coefficient-by-site matrix is formed.
"""

from __future__ import annotations

from typing import assert_never, final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._trainable import NonTrainableState
from ..._validation import positive_integer
from ...linalg import ArraySpace
from ...typing import parse
from .._integration_domain import IntegrationDomain
from .._reference_cell import FacetShape
from .._side_actions import (
    AbstractSideRoute,
    FacetTraceRule,
    PreparedTraceAction,
    SideActionDescriptor,
    SideOrientation,
    SideTraceQuantity,
)
from .._topology import EntitySelection
from .._views import FieldTraceSide
from ._basis import (
    ChebyshevBasisPlan,
    LegendreBasisPlan,
    PreparedSpectralAxis,
    SineBasisPlan,
)
from ._precision import SpectralPrecisionPolicy


# Mode labels and face-site labels of the sum-factorized face synthesis.
_MODE_LABELS = "abcdefgh"
_SITE_LABELS = "ijklmnop"
# A mode belongs to the support rows when its trace exceeds this fraction of
# the largest basis value on the selected faces (rounding level otherwise).
_SUPPORT_TOLERANCE = 64.0 * float(np.finfo(np.float64).eps)


def _bounded(axis: PreparedSpectralAxis, /) -> bool:
    return not axis.periodic and axis.bounds is not None


def spectral_face_domain(
    axes: tuple[PreparedSpectralAxis, ...],
    axis_names: tuple[str, ...],
    support_id: str,
    /,
) -> IntegrationDomain:
    """Exterior faces of the bounded axes in canonical order."""
    faces = np.asarray(
        [
            2 * index + side
            for index, axis in enumerate(axes)
            if _bounded(axis)
            for side in (0, 1)
        ],
        dtype=np.int32,
    )
    if faces.size == 0:
        raise ValueError(
            "Every axis of this spectral space is periodic or unbounded, so it "
            "has no boundary faces."
        )
    return IntegrationDomain(
        "exterior_facet",
        faces,
        support_id,
        canonical_fingerprint(
            {
                "kind": "tensor-spectral-boundary-faces",
                "support": support_id,
                "axis_names": list(axis_names),
                "bounded": [_bounded(axis) for axis in axes],
            }
        ),
        owner_cells=np.zeros(faces.shape, dtype=np.int32),
        owner_local_entities=faces,
    )


def _face_active_mask(axes: tuple[PreparedSpectralAxis, ...], /) -> np.ndarray:
    return np.repeat(np.asarray([_bounded(axis) for axis in axes], np.bool_), 2)


def select_spectral_faces(
    axes: tuple[PreparedSpectralAxis, ...],
    base: IntegrationDomain,
    selection: EntitySelection,
    /,
) -> IntegrationDomain:
    """Restrict the boundary faces to one entity selection."""
    if not isinstance(selection, EntitySelection):
        raise TypeError("selection must be EntitySelection or None.")
    if selection.entity_set_id != base.entity_set_id:
        raise ValueError("Entity selection does not match the spectral boundary faces.")
    mask = np.asarray(selection.mask, dtype=np.bool_)
    if mask.shape != (2 * len(axes),):
        raise ValueError("The face selection must cover the 2 * dimension local faces.")
    faces = np.asarray(base.entity_indices, dtype=np.int32)
    selected = faces[mask[faces]]
    return IntegrationDomain(
        base.kind,
        selected,
        base.support_id,
        base.entity_set_id,
        owner_cells=np.zeros(selected.shape, dtype=np.int32),
        owner_local_entities=selected,
        selection_id=selection.selection_id,
    )


def spectral_face_selection(
    axes: tuple[PreparedSpectralAxis, ...],
    axis_names: tuple[str, ...],
    base: IntegrationDomain,
    axis: str,
    side: Literal["lower", "upper"],
    /,
) -> EntitySelection:
    """Select one bounded face; periodic and unbounded axes are refused."""
    name = str(axis)
    if name not in axis_names:
        raise KeyError(f"Unknown spectral axis {name!r}.")
    index = axis_names.index(name)
    prepared = axes[index]
    if prepared.periodic:
        raise ValueError(
            f"Axis {name!r} is periodic ({prepared.family}): it has no boundary "
            "face and no outward normal. Periodic coupling uses the periodic "
            "image, not a boundary trace."
        )
    if prepared.bounds is None:
        raise ValueError(f"Axis {name!r} is unbounded and has no boundary face.")
    match side:
        case "lower":
            offset = 0
        case "upper":
            offset = 1
        case _:
            raise ValueError("side must be 'lower' or 'upper'.")
    mask = np.zeros((2 * len(axes),), dtype=np.bool_)
    mask[2 * index + offset] = True
    return EntitySelection(base.entity_set_id, mask, active_mask=_face_active_mask(axes))


def _selected_faces(
    axes: tuple[PreparedSpectralAxis, ...],
    axis_names: tuple[str, ...],
    base: IntegrationDomain,
    domain: IntegrationDomain,
    side: FieldTraceSide,
    /,
) -> tuple[tuple[int, int], ...]:
    """Verified `(axis, side)` of every facet of `domain`, in facet order."""
    if not isinstance(domain, IntegrationDomain):
        raise TypeError("domain must be an IntegrationDomain.")
    match domain.kind:
        case "exterior_facet":
            pass
        case "interior_facet":
            raise ValueError(
                "A global spectral field is one smooth synthesis over its box; it "
                "has no interior facets."
            )
        case _:
            raise ValueError("Spectral side traces act on exterior boundary faces.")
    if domain.support_id != base.support_id or domain.entity_set_id != base.entity_set_id:
        raise ValueError(
            "The facet domain belongs to another owner's support; prepare it from "
            "this spectral discretization."
        )
    match side:
        case "owner":
            pass
        case "neighbor":
            raise ValueError(
                "Exterior faces have no neighbor side; prepare the owner trace."
            )
        case "average":
            raise ValueError(
                "Side actions are one-sided; compose the owner and neighbor actions "
                "instead of side='average'."
            )
        case _:
            assert_never(side)
    faces = np.asarray(domain.entity_indices, dtype=np.int32)
    if faces.size == 0 or np.any(faces >= 2 * len(axes)):
        raise ValueError("The facet domain names undeclared spectral faces.")
    if np.any(np.asarray(domain.owner_local_entities, dtype=np.int32) != faces):
        raise ValueError("The facet domain routes do not match the spectral faces.")
    selected: list[tuple[int, int]] = []
    for face in faces.tolist():
        index, offset = divmod(int(face), 2)
        axis = axes[index]
        name = axis_names[index]
        if axis.periodic:
            raise ValueError(
                f"Axis {name!r} is periodic ({axis.family}): it has no boundary "
                "face and no outward normal."
            )
        if axis.bounds is None:
            raise ValueError(f"Axis {name!r} is unbounded and has no boundary face.")
        match axis.plan:
            case SineBasisPlan():
                raise ValueError(
                    f"Axis {name!r} is a sine axis: every sine mode vanishes on its "
                    "faces (built-in homogeneous Dirichlet values), so there is no "
                    "boundary value trace to prepare."
                )
            case _:
                pass
        selected.append((index, offset))
    return tuple(selected)


def _facet_shape(dimension: int, /) -> FacetShape:
    match dimension:
        case 1:
            return "point"
        case 2:
            return "edge"
        case 3:
            return "quadrilateral"
        case _:
            raise ValueError(
                "Facet trace rules exist for point, edge, and quadrilateral faces; "
                "use rule=None for the native tangential quadrature of "
                f"{dimension}-dimensional boxes."
            )


def _native_exact_degree(axis: PreparedSpectralAxis, /) -> int | None:
    """Polynomial exactness of the prepared axis quadrature (`None` if trigonometric)."""
    count = axis.physical_count
    match axis.plan:
        case ChebyshevBasisPlan():
            return count - 1
        case LegendreBasisPlan() as plan:
            match plan.node_rule:
                case "gauss":
                    return 2 * count - 1
                case "radau":
                    return 2 * count - 2
                case "lobatto":
                    return 2 * count - 3
                case _:
                    assert_never(plan.node_rule)
        case _:
            return None


def _polynomial(axis: PreparedSpectralAxis, /) -> bool:
    match axis.plan:
        case ChebyshevBasisPlan() | LegendreBasisPlan():
            return True
        case _:
            return False


def _axis_sites(
    axis: PreparedSpectralAxis, rule: FacetTraceRule | None, /
) -> tuple[np.ndarray, np.ndarray]:
    """Physical tangential coordinates and weights along one axis."""
    if rule is None:
        return (
            np.asarray(axis.nodes, dtype=np.float64),
            np.asarray(axis.quadrature_weights, dtype=np.float64),
        )
    bounds = np.asarray(axis.bounds, dtype=np.float64)
    parameters, weights = rule.reference("edge")
    length = float(bounds[1] - bounds[0])
    return bounds[0] + length * parameters[:, 0], length * weights


def _contraction_weights(
    normals: np.ndarray, quantity: SideTraceQuantity, components: tuple[int, ...], /
) -> tuple[np.ndarray | None, tuple[int, ...]]:
    """Per-face component contraction `(faces, *value_shape, *components)`."""
    dimension = normals.shape[-1]
    match quantity:
        case "value":
            return None, components
        case "normal" | "tangential":
            if components != (dimension,):
                raise ValueError(
                    f"{quantity!r} traces need a vector field with one component "
                    f"per spectral axis (component_shape=({dimension},))."
                )
            if quantity == "normal":
                return normals, ()
            if dimension == 2:
                return np.stack((-normals[:, 1], normals[:, 0]), axis=-1), ()
            if dimension == 3:
                projector = np.eye(3) - normals[:, :, None] * normals[:, None, :]
                return projector, (3,)
            raise ValueError("One-dimensional boundary points have no tangential trace.")
        case "conormal-flux":
            raise ValueError(
                "Conormal fluxes are published by compiled physics owners, not by "
                "discretization traces."
            )
        case _:
            assert_never(quantity)


@final
class SpectralFaceRoute(AbstractSideRoute, NonTrainableState):
    """Sum-factorized spectral synthesis on selected bounded faces.

    Face `f` with normal axis `a` contracts the modal coefficients with the
    endpoint basis row `phi^a(x_a = a or b)` and the tangential basis rows at
    its sites, in the canonical per-axis ordering of
    `PreparedSpectralAxis.evaluate_basis`. `transpose` is the exact bilinear
    transpose of that contraction. For a real physical dtype the trace is the
    real part of the complex synthesis (the `reconstruct` semantics) and
    `<T c, w> = Re(sum(c * T^T w))`. Normal and tangential traces contract the
    trace components with the per-face outward normal or tangent.
    """

    endpoint_rows: tuple[Array, ...]
    tangential_rows: tuple[tuple[Array, ...], ...]
    contraction: Array | None
    faces: tuple[tuple[int, int], ...] = eqx.field(static=True)
    global_shape: tuple[int, ...] = eqx.field(static=True)
    trace_value_shape: tuple[int, ...] = eqx.field(static=True)
    site_count: int = eqx.field(static=True)
    real_output: bool = eqx.field(static=True)
    output_dtype: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        axes: tuple[PreparedSpectralAxis, ...],
        faces: tuple[tuple[int, int], ...],
        axis_sites: tuple[tuple[np.ndarray, ...], ...],
        precision: SpectralPrecisionPolicy,
        /,
        *,
        component_shape: tuple[int, ...],
        contraction: np.ndarray | None,
        value_shape: tuple[int, ...],
    ) -> None:
        if not faces or len(axis_sites) != len(faces):
            raise ValueError("A spectral face route needs one site set per face.")
        if len(axes) > len(_MODE_LABELS):
            raise ValueError(
                f"Spectral face routes support at most {len(_MODE_LABELS)} axes."
            )
        counts = {
            int(np.prod([coordinates.size for coordinates in sites], dtype=np.int64))
            for sites in axis_sites
        }
        if len(counts) != 1:
            raise ValueError(
                "The selected faces have different site counts; prepare one trace "
                "per normal axis or use a FacetTraceRule."
            )
        endpoint_rows: list[Array] = []
        tangential_rows: list[tuple[Array, ...]] = []
        for (index, offset), sites in zip(faces, axis_sites, strict=True):
            axis = axes[index]
            bounds = np.asarray(axis.bounds, dtype=np.float64)
            endpoint_rows.append(
                axis.evaluate_basis(np.asarray([bounds[offset]]), order=0)[0]
            )
            tangential = tuple(other for other in range(len(axes)) if other != index)
            if len(sites) != len(tangential):
                raise ValueError("Face sites must give one coordinate set per tangent.")
            tangential_rows.append(
                tuple(
                    axes[other].evaluate_basis(coordinates, order=0)
                    for other, coordinates in zip(tangential, sites, strict=True)
                )
            )
        physical = jnp.dtype(precision.physical_dtype)
        real = not jnp.issubdtype(physical, jnp.complexfloating)
        self.endpoint_rows = tuple(endpoint_rows)
        self.tangential_rows = tuple(tangential_rows)
        self.contraction = (
            None
            if contraction is None
            else jnp.asarray(contraction, dtype=jnp.finfo(physical).dtype)
        )
        self.faces = faces
        self.global_shape = (
            *(axis.mode_count for axis in axes),
            *component_shape,
        )
        self.trace_value_shape = value_shape
        self.site_count = counts.pop()
        self.real_output = real
        self.output_dtype = str(physical)
        self.route_id = canonical_fingerprint(
            {
                "kind": "spectral-face-route",
                "axes": [axis.axis_id for axis in axes],
                "faces": [list(face) for face in faces],
                "sites": [
                    [array_tree_fingerprint(coordinates) for coordinates in sites]
                    for sites in axis_sites
                ],
                "components": list(component_shape),
                "value_shape": list(value_shape),
                "contraction": (
                    None if contraction is None else array_tree_fingerprint(contraction)
                ),
                "precision": precision.policy_id,
            }
        )

    @property
    def coefficient_shape(self) -> tuple[int, ...]:
        return self.global_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (len(self.faces), self.site_count, *self.trace_value_shape)

    def _subscripts(self, index: int, /) -> tuple[str, str, str]:
        """Mode, tangential-row, and site subscripts of a face normal to `index`."""
        modes = _MODE_LABELS[: len(self.tangential_rows[0]) + 1]
        tangents = "".join(
            f",{_SITE_LABELS[other]}{modes[other]}"
            for other in range(len(modes))
            if other != index
        )
        sites = "".join(
            _SITE_LABELS[other] for other in range(len(modes)) if other != index
        )
        return modes, tangents, sites

    def _value_subscripts(self) -> str:
        return "v" * len(self.trace_value_shape)

    def apply(self, coefficients: Array, /) -> Array:
        faces = []
        for (index, _), endpoint, tangential in zip(
            self.faces, self.endpoint_rows, self.tangential_rows, strict=True
        ):
            modes, tangents, sites = self._subscripts(index)
            values = contract(
                f"{modes}...,{modes[index]}{tangents}->{sites}...",
                coefficients,
                endpoint,
                *tangential,
            )
            faces.append(values.reshape((self.site_count, *values.shape[len(sites) :])))
        traces = jnp.stack(faces)
        if self.real_output:
            traces = jnp.real(traces)
        traces = traces.astype(jnp.dtype(self.output_dtype))
        if self.contraction is None:
            return traces
        values = self._value_subscripts()
        return contract(
            f"fqc,f{values}c->fq{values}",
            traces,
            self.contraction.astype(traces.dtype),
        )

    def transpose(self, values: Array, /) -> Array:
        cotangent = values
        if self.contraction is not None:
            value_axes = self._value_subscripts()
            cotangent = contract(
                f"fq{value_axes},f{value_axes}c->fqc",
                cotangent,
                self.contraction.astype(cotangent.dtype),
            )
        coefficient_dtype = self.endpoint_rows[0].dtype
        cotangent = cotangent.astype(coefficient_dtype)
        total = jnp.zeros(self.global_shape, dtype=coefficient_dtype)
        for face, ((index, _), endpoint, tangential) in enumerate(
            zip(self.faces, self.endpoint_rows, self.tangential_rows, strict=True)
        ):
            modes, tangents, sites = self._subscripts(index)
            site_shape = tuple(rows.shape[0] for rows in tangential)
            local = cotangent[face].reshape((*site_shape, *cotangent.shape[2:]))
            total = total + contract(
                f"{sites}...,{modes[index]}{tangents}->{modes}...",
                local,
                endpoint,
                *tangential,
            )
        return total


def _support_rows(route: SpectralFaceRoute, /) -> np.ndarray:
    """Modes of coefficient axis 0 with a nonzero trace on the selected faces."""
    rows = []
    for (index, _), endpoint, tangential in zip(
        route.faces, route.endpoint_rows, route.tangential_rows, strict=True
    ):
        values = (
            np.abs(np.asarray(endpoint))[None, :]
            if index == 0
            else np.abs(np.asarray(tangential[0]))
        )
        largest = float(np.max(values))
        rows.append(np.any(values > _SUPPORT_TOLERANCE * largest, axis=0))
    return np.flatnonzero(np.any(np.stack(rows), axis=0)).astype(np.int32)


def prepare_spectral_face_trace(
    axes: tuple[PreparedSpectralAxis, ...],
    axis_names: tuple[str, ...],
    precision: SpectralPrecisionPolicy,
    base: IntegrationDomain,
    domain: IntegrationDomain,
    /,
    *,
    owner_id: str,
    field_space_id: str,
    revision_id: str,
    rule: FacetTraceRule | None,
    quantity: SideTraceQuantity,
    side: FieldTraceSide,
    component_shape: tuple[int, ...],
) -> PreparedTraceAction:
    """Prepare the exact trace of a tensor spectral field on bounded faces."""
    if rule is not None and not isinstance(rule, FacetTraceRule):
        raise TypeError("rule must be a FacetTraceRule or None.")
    quantity = parse(quantity, SideTraceQuantity, "quantity")
    side = parse(side, FieldTraceSide, "side")
    components = tuple(
        positive_integer(size, "component_shape") for size in component_shape
    )
    dimension = len(axes)
    faces = _selected_faces(axes, axis_names, base, domain, side)
    facet_shape = None if rule is None else _facet_shape(dimension)
    axis_sites = tuple(
        tuple(
            _axis_sites(axes[other], rule) for other in range(dimension) if other != index
        )
        for index, _ in faces
    )
    normals = np.zeros((len(faces), dimension), dtype=np.float64)
    for row, (index, offset) in enumerate(faces):
        normals[row, index] = -1.0 if offset == 0 else 1.0
    contraction, value_shape = _contraction_weights(normals, quantity, components)
    route = SpectralFaceRoute(
        axes,
        faces,
        tuple(tuple(coordinates for coordinates, _ in sites) for sites in axis_sites),
        precision,
        component_shape=components,
        contraction=contraction,
        value_shape=value_shape,
    )
    sites = np.empty((len(faces), route.site_count, dimension), dtype=np.float64)
    weights = np.empty((len(faces), route.site_count), dtype=np.float64)
    for row, ((index, offset), tangential) in enumerate(
        zip(faces, axis_sites, strict=True)
    ):
        # Row-major tensor order over the tangential axes (ascending axis index).
        others = [other for other in range(dimension) if other != index]
        grids = np.meshgrid(
            *(coordinates for coordinates, _ in tangential), indexing="ij"
        )
        measure = np.ones((), dtype=np.float64)
        for other, grid, (_, factor) in zip(others, grids, tangential, strict=True):
            sites[row, :, other] = grid.reshape((-1,))
            measure = np.multiply.outer(measure, factor)
        sites[row, :, index] = np.asarray(axes[index].bounds, dtype=np.float64)[offset]
        weights[row] = measure.reshape((-1,))
    tangent_axes = {
        other for index, _ in faces for other in range(dimension) if other != index
    }
    polynomial = all(_polynomial(axes[other]) for other in tangent_axes)
    if dimension == 1:
        trace_degree: int | None = 0
        exact_degree: int | None = None
    elif not polynomial:
        trace_degree, exact_degree = None, None
    else:
        trace_degree = max(axes[other].mode_count - 1 for other in tangent_axes)
        if rule is None or facet_shape is None:
            native = [_native_exact_degree(axes[other]) for other in tangent_axes]
            exact_degree = min(degree for degree in native if degree is not None)
        else:
            exact_degree = rule.exact_degree(facet_shape)
    coordinate_dtype = jnp.finfo(jnp.dtype(precision.physical_dtype)).dtype
    orientation: SideOrientation = "unoriented" if quantity == "value" else "outward"
    descriptor = SideActionDescriptor(
        owner_id=owner_id,
        field_space_id=canonical_fingerprint(
            {
                "kind": "tensor-spectral-modal-trace-space",
                "space": field_space_id,
                "components": list(components),
            }
        ),
        quantity=quantity,
        representation="quadrature-values",
        orientation=orientation,
        approximation="exact",
        side=side,
        domain=domain,
        revision_id=revision_id,
        rule=rule,
        trace_degree=trace_degree,
        quadrature_exact_degree=exact_degree,
    )
    rows = _support_rows(route)
    if rows.size == 0:
        raise ValueError("No spectral mode has a nonzero trace on the selected faces.")
    return PreparedTraceAction(
        descriptor,
        route,
        ArraySpace(route.coefficient_shape, dtype=jnp.dtype(precision.coefficient_dtype)),
        sites=sites.astype(coordinate_dtype),
        weights=weights.astype(coordinate_dtype),
        normals=np.broadcast_to(
            normals[:, None, :], (len(faces), route.site_count, dimension)
        ).astype(coordinate_dtype),
        support_rows=rows,
    )


__all__ = [
    "prepare_spectral_face_trace",
    "select_spectral_faces",
    "spectral_face_domain",
    "spectral_face_selection",
    "SpectralFaceRoute",
]
