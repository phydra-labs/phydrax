#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Mapping
from typing import Any, cast, overload

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._sampling import (
    derive_key,
    DESIGN_ALGORITHM_VERSION,
    design_capabilities,
    design_name,
    materialize_design,
    SampleAddress,
)
from .._strict import StrictModule
from ..discretization._axis import (
    AbstractAxisSpec,
    AxisDiscretization,
    broadcasted_grid,
    cut_cell_geometry_weight_from_adf,
    sdf_mask_from_adf,
    TensorGridPlan,
)
from ..geometry import BoundaryAtlasProvider, sample_boundary_atlas
from ._base import AbstractGeometry, EnforcementGateMethod
from ._dataset import DatasetDomain
from ._domain import Domain
from ._factor_component import FactorComponent
from ._function import DomainFunction
from ._measure import (
    BaseMeasure,
    ExactMass,
    Mass,
    product_mass,
    sum_mass,
    UnknownMass,
)
from ._ragged_series_dataset import RaggedSeriesDatasetDomain
from ._scalar import AbstractScalarDomain
from ._selection import (
    Boundary,
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
    Selection,
    SelectionSpec,
)
from ._structure import (
    _axis_name_for_coord,
    GridBatch,
    GridSampling,
    NumPoints,
    PointBatch,
    Points,
    PointSampling,
    SampleLayout,
)


def _as_field(x: Array, *, dims: tuple[str | None, ...]) -> cx.Field:
    return cx.Field(x, dims=dims)


class _NormalCallable(StrictModule):
    geom: AbstractGeometry

    def __init__(self, geom: AbstractGeometry):
        self.geom = geom

    def __call__(self, x: Array, /, *, key=None, **kwargs: Any) -> Array:
        del key, kwargs
        pts_in = jnp.asarray(x, dtype=float)
        d = int(self.geom.spatial_dim)
        if pts_in.ndim == 0:
            if d != 1:
                raise ValueError("Expected a geometry point with shape (..., dim).")
            pts = pts_in.reshape((1, 1))
        elif pts_in.ndim == 1:
            if d == 1:
                pts = pts_in.reshape((-1, 1))
            else:
                if pts_in.shape[0] != d:
                    raise ValueError("Expected a geometry point with shape (..., dim).")
                pts = pts_in.reshape((1, d))
        else:
            if pts_in.shape[-1] != d:
                raise ValueError("Expected a geometry point with shape (..., dim).")
            pts = pts_in.reshape((-1, d))

        n = jnp.asarray(self.geom._boundary_normals(pts), dtype=float)
        eps = jnp.finfo(float).eps
        nrm = jnp.linalg.norm(n, axis=-1, keepdims=True) + eps
        n = n / nrm
        if pts_in.ndim == 0:
            return n.reshape(())
        return n.reshape(pts_in.shape)


class _SdfCallable(StrictModule):
    geom: AbstractGeometry

    def __init__(self, geom: AbstractGeometry):
        self.geom = geom

    def __call__(self, x: Any, /, *, key=None, **kwargs: Any) -> Array:
        del key, kwargs
        d = int(self.geom.spatial_dim)

        if isinstance(x, tuple):
            coords = tuple(jnp.asarray(c, dtype=float).reshape((-1,)) for c in x)
            if len(coords) != d:
                raise ValueError(
                    f"coord-separable sdf expects {d} coordinate arrays, got {len(coords)}."
                )
            grid = broadcasted_grid(coords)
            pts = grid.reshape((-1, d))
            sdf = self.geom.adf(pts)
            return jnp.asarray(sdf, dtype=float).reshape(grid.shape[:-1])

        pts_in = jnp.asarray(x, dtype=float)
        if pts_in.ndim == 0:
            if d != 1:
                raise ValueError("Expected a geometry point with shape (..., dim).")
            return jnp.asarray(self.geom.adf(pts_in.reshape(())), dtype=float).reshape(())
        if pts_in.ndim == 1:
            return jnp.asarray(self.geom.adf(pts_in), dtype=float)

        if pts_in.shape[-1] != d:
            raise ValueError("Expected a geometry point with shape (..., dim).")
        pts = pts_in.reshape((-1, d))
        sdf = self.geom.adf(pts)
        return jnp.asarray(sdf, dtype=float).reshape(pts_in.shape[:-1])


class _EnforcementGateCallable(StrictModule):
    gate: Callable[[Array], Array]
    dim: int

    def __init__(
        self,
        geom: AbstractGeometry,
        *,
        method: EnforcementGateMethod,
        saturation_fraction: float,
        linear_fraction: float,
    ):
        self.gate = geom.make_enforcement_gate(
            method=method,
            saturation_fraction=saturation_fraction,
            linear_fraction=linear_fraction,
        )
        self.dim = int(geom.spatial_dim)

    def __call__(self, x: Any, /, *, key=None, **kwargs: Any) -> Array:
        del key, kwargs
        if isinstance(x, tuple):
            coords = tuple(jnp.asarray(c, dtype=float).reshape((-1,)) for c in x)
            if len(coords) != self.dim:
                raise ValueError(
                    "coord-separable enforcement gate expects "
                    f"{self.dim} coordinate arrays, got {len(coords)}."
                )
            grid = broadcasted_grid(coords)
            points = grid.reshape((-1, self.dim))
            values = self.gate(points)
            return jnp.asarray(values, dtype=float).reshape(grid.shape[:-1])

        points_in = jnp.asarray(x, dtype=float)
        if points_in.ndim == 0:
            if self.dim != 1:
                raise ValueError("Expected a geometry point with shape (..., dim).")
            return jnp.asarray(self.gate(points_in.reshape(())), dtype=float).reshape(())
        if points_in.ndim == 1:
            return jnp.asarray(self.gate(points_in), dtype=float)
        if points_in.shape[-1] != self.dim:
            raise ValueError("Expected a geometry point with shape (..., dim).")
        points = points_in.reshape((-1, self.dim))
        values = self.gate(points)
        return jnp.asarray(values, dtype=float).reshape(points_in.shape[:-1])


def _sample_geometry(
    geom: AbstractGeometry,
    component: Selection,
    num_points: int,
    *,
    sampler: str,
    key: Key[Array, ""],
) -> Array:
    if isinstance(component, Interior):
        return jnp.asarray(
            geom.sample_interior(num_points, sampler=sampler, key=key), dtype=float
        )
    if isinstance(component, Boundary):
        if component.tags is not None or component.entity_ids is not None:
            if not isinstance(geom, BoundaryAtlasProvider):
                raise ValueError(
                    f"{type(geom).__name__} does not expose boundary entities."
                )
            atlas = geom.boundary_atlas.select(
                tags=component.tags,
                entity_ids=component.entity_ids,
            )
            return sample_boundary_atlas(atlas, num_points, key=key).points
        return jnp.asarray(
            geom.sample_boundary(num_points, sampler=sampler, key=key), dtype=float
        )
    if isinstance(component, Fixed):
        raise ValueError(
            "Fixed(x) is not supported for geometries in sampling; "
            "use a unary DomainFunction mask instead."
        )
    raise TypeError(f"Unsupported geometry component {type(component).__name__}.")


def _sample_scalar(
    dom: AbstractScalarDomain,
    component: Selection,
    num_points: int,
    *,
    sampler: str,
    key: Key[Array, ""],
) -> Array:
    if isinstance(component, Interior):
        return jnp.asarray(dom.sample(num_points, sampler=sampler, key=key), dtype=float)
    if isinstance(component, FixedStart):
        return jnp.asarray(dom.fixed("start"), dtype=float)
    if isinstance(component, FixedEnd):
        return jnp.asarray(dom.fixed("end"), dtype=float)
    if isinstance(component, Fixed):
        return jnp.asarray(component.value, dtype=float).reshape(())
    if isinstance(component, Boundary):
        # Boundary on scalar domains is a discrete set of two endpoints. We sample
        # from this set; measure semantics treat this as counting measure with mass 2.
        choices = jnp.stack(
            [jnp.asarray(dom.fixed("start")), jnp.asarray(dom.fixed("end"))], axis=0
        )
        idx = jr.randint(key, shape=(int(num_points),), minval=0, maxval=2)
        return choices[idx]
    raise TypeError(f"Unsupported scalar component {type(component).__name__}.")


def _explicit_point_array(domain: Domain, label: str, value: ArrayLike, /) -> Array:
    factor = domain.factor(label)
    array = jnp.asarray(value, dtype=float)
    if isinstance(factor, AbstractGeometry):
        if array.ndim == 1:
            array = array.reshape((-1, 1) if int(factor.spatial_dim) == 1 else (1, -1))
        if array.ndim != 2 or int(array.shape[1]) != int(factor.spatial_dim):
            raise ValueError(
                f"Geometry coordinates for {label!r} must have shape "
                f"(num_points, {factor.spatial_dim}), got {array.shape}."
            )
        return array
    if isinstance(factor, AbstractScalarDomain):
        if array.ndim == 0:
            return array.reshape((1,))
        if array.ndim == 1:
            return array
        if array.ndim == 2 and int(array.shape[1]) == 1:
            return array[:, 0]
        raise ValueError(
            f"Scalar coordinates for {label!r} must have shape (num_points,), "
            f"got {array.shape}."
        )
    raise TypeError(
        f"Explicit points do not support factor {type(factor).__name__} "
        f"for label {label!r}."
    )


def _split_explicit_points(
    domain: Domain,
    labels: tuple[str, ...],
    coordinates: ArrayLike,
    /,
) -> frozendict[str, Array]:
    stacked = jnp.asarray(coordinates, dtype=float)
    if stacked.ndim == 1:
        stacked = stacked.reshape((1, -1))
    if stacked.ndim != 2:
        raise ValueError(
            "Stacked coordinates must have shape (num_points, coordinate_dim), "
            f"got {stacked.shape}."
        )

    widths: list[int] = []
    for label in labels:
        factor = domain.factor(label)
        if isinstance(factor, AbstractGeometry):
            widths.append(int(factor.spatial_dim))
        elif isinstance(factor, AbstractScalarDomain):
            widths.append(1)
        else:
            raise TypeError(
                f"Explicit points do not support factor {type(factor).__name__} "
                f"for label {label!r}."
            )
    total = sum(widths)
    if int(stacked.shape[1]) != total:
        raise ValueError(
            f"Stacked coordinates require coordinate_dim={total}, got {stacked.shape[1]}."
        )

    result: dict[str, Array] = {}
    offset = 0
    for label, width in zip(labels, widths, strict=True):
        result[label] = (
            stacked[:, offset] if width == 1 else stacked[:, offset : offset + width]
        )
        offset += width
    return frozendict(result)


def _fixed_component_labels(component: "DomainComponent", /) -> frozenset[str]:
    return frozenset(
        label
        for label in component.domain.labels
        if isinstance(
            component.spec.selection_for(label),
            (FixedStart, FixedEnd, Fixed),
        )
    )


class DomainComponent(StrictModule):
    r"""A domain equipped with component selection, filters, and weights.

    A `DomainComponent` represents a product component of a labeled domain.
    Given a labeled domain $\Omega = \prod_{\ell\in\mathcal{L}} \Omega_\ell$, and a
    `ComponentSpec` selecting a subset/type for each label, the component corresponds
    to a set (schematically)

    $$
    \Omega_{\text{comp}} = \prod_{\ell\in\mathcal{L}} \Omega_\ell^{(\text{spec})},
    $$

    together with its associated product measure. For example:

    - geometry interior $\Omega_\ell$ uses volume/area/length measure;
    - geometry boundary $\partial\Omega_\ell$ uses surface measure;
    - scalar interior uses Lebesgue measure on $[a,b]$;
    - scalar boundary uses counting measure on $\{a,b\}$ (total mass $2$);
    - fixed scalar slices use a unit-mass Dirac measure.

    Additional selection and weighting can be applied via:
    - `where`: per-label indicator functions;
    - `where_all`: a global indicator `DomainFunction`;
    - `weight_all`: a global weight `DomainFunction`.

    These are incorporated downstream in integral/mean estimators and constraint
    losses.
    """

    domain: Domain
    spec: SelectionSpec
    factor_components: tuple[FactorComponent, ...]
    where: frozendict[str, Callable]
    where_all: DomainFunction | None
    weight_all: DomainFunction | None
    density_normalized: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        domain: Domain,
        spec: SelectionSpec | None = None,
        where: Mapping[str, Callable] | None = None,
        where_all: DomainFunction | Callable | None = None,
        weight_all: DomainFunction | Callable | None = None,
        density_normalized: bool = False,
    ):
        self.domain = domain
        self.spec = spec or SelectionSpec()
        unknown = tuple(
            label for label in self.spec.by_label if label not in self.domain.labels
        )
        if unknown:
            raise KeyError(
                f"SelectionSpec contains labels {unknown} outside domain "
                f"{self.domain.labels}."
            )
        self.factor_components = tuple(
            factor.bind_component(
                {label: self.spec.selection_for(label) for label in factor.labels}
            )
            for factor in self.domain.joint_factors
        )
        where_all_ = where_all
        if where_all_ is not None and not isinstance(where_all_, DomainFunction):
            where_all_ = self.domain.Function(*self.domain.labels)(where_all_)
        weight_all_ = weight_all
        if weight_all_ is not None and not isinstance(weight_all_, DomainFunction):
            weight_all_ = self.domain.Function(*self.domain.labels)(weight_all_)

        self.where = frozendict(where or {})
        self.where_all = where_all_
        self.weight_all = weight_all_
        self.density_normalized = bool(density_normalized)

    def factor_component(self, label: str, /) -> FactorComponent:
        """Return the bound joint-factor component that owns ``label``."""
        for component in self.factor_components:
            if label in component.factor.labels:
                return component
        raise KeyError(
            f"Label {label!r} is outside component domain {self.domain.labels}."
        )

    @property
    def base_measure(self) -> BaseMeasure:
        """Product measure before restriction or density transformations."""
        return BaseMeasure(
            "external",
            product_mass(
                tuple(component.measure.mass for component in self.factor_components)
            ),
        )

    @property
    def mass(self) -> Mass:
        """Typed component mass after explicit restrictions and density."""
        if self.where or self.where_all is not None:
            return UnknownMass("restricted component mass requires numerical estimation")
        if self.weight_all is not None:
            if self.density_normalized:
                return ExactMass(1.0)
            return UnknownMass(
                "density-weighted component mass requires numerical estimation"
            )
        return self.base_measure.mass

    def restrict(
        self,
        *,
        per_coordinate: Mapping[str, Callable] | None = None,
        predicate: DomainFunction | Callable | None = None,
    ) -> "DomainComponent":
        """Return this component with an explicit predicate restriction."""
        merged = dict(self.where)
        for label, condition in (per_coordinate or {}).items():
            if label not in self.domain.labels:
                raise KeyError(f"Restriction label {label!r} is outside the domain.")
            if label in merged:
                raise ValueError(
                    f"Coordinate {label!r} already has a restriction; compose it explicitly."
                )
            merged[label] = condition
        if predicate is not None and self.where_all is not None:
            raise ValueError(
                "A global restriction already exists; compose predicates explicitly."
            )
        return DomainComponent(
            domain=self.domain,
            spec=self.spec,
            where=merged,
            where_all=self.where_all if predicate is None else predicate,
            weight_all=self.weight_all,
            density_normalized=self.density_normalized,
        )

    def with_density(
        self,
        density: DomainFunction | Callable,
        /,
        *,
        normalized: bool = False,
    ) -> "DomainComponent":
        """Return this component weighted by a declared non-negative density."""
        density_fn = (
            density
            if isinstance(density, DomainFunction)
            else DomainFunction(
                domain=self.domain,
                deps=self.domain.labels,
                func=density,
                metadata={},
            )
        )
        combined = density_fn if self.weight_all is None else self.weight_all * density_fn
        return DomainComponent(
            domain=self.domain,
            spec=self.spec,
            where=self.where,
            where_all=self.where_all,
            weight_all=combined,
            density_normalized=normalized if self.weight_all is None else False,
        )

    def _sample_graph_batch(
        self,
        num_points: NumPoints,
        *,
        structure: SampleLayout,
        sampler: str,
        key: Key[Array, ""],
    ):
        from .graph._batch import GraphBatch
        from .graph._dataset import GraphDatasetDomain
        from .graph._domain import GraphDomain

        graph_labels: list[str] = []
        for lbl in self.domain.labels:
            factor = self.domain.factor(lbl)

            if isinstance(factor, (GraphDomain, GraphDatasetDomain)):
                graph_labels.append(lbl)

        if not graph_labels:
            return None
        if len(graph_labels) > 1:
            raise ValueError(
                "Sampling multiple GraphDomain factors is not supported yet."
            )

        graph_label = graph_labels[0]
        graph_factor = self.domain.factor(graph_label)

        assert isinstance(graph_factor, (GraphDomain, GraphDatasetDomain))

        fixed_labels = frozenset(
            lbl
            for lbl in self.domain.labels
            if isinstance(self.spec.selection_for(lbl), (FixedStart, FixedEnd, Fixed))
        )
        structure_out = structure.canonicalize(
            self.domain.labels, fixed_labels=fixed_labels
        )
        graph_axis = structure_out.axis_for(graph_label)
        if graph_axis is None:
            raise ValueError(
                f"GraphDomain label {graph_label!r} must be sampled on an axis."
            )
        for block in structure_out.blocks:
            if graph_label in block and len(block) != 1:
                raise ValueError(
                    "GraphDomain labels must be sampled in singleton SampleLayout "
                    f"blocks; got {block!r}."
                )

        if isinstance(num_points, int):
            if len(structure_out.blocks) != 1:
                raise ValueError(
                    "num_points=int is only valid for exactly one sampling block."
                )
            num_points_by_block = (int(num_points),)
        else:
            if len(num_points) != len(structure_out.blocks):
                raise ValueError(
                    f"num_points must have length {len(structure_out.blocks)} to match blocks."
                )
            num_points_by_block = tuple(int(n) for n in num_points)

        label_to_block_index = {
            lbl: i for i, block in enumerate(structure_out.blocks) for lbl in block
        }
        label_to_idx = {lbl: i for i, lbl in enumerate(self.domain.labels)}
        block_keys = jr.split(key, len(structure_out.blocks) + 1)[1:]
        graph_n = num_points_by_block[label_to_block_index[graph_label]]
        if isinstance(graph_factor, GraphDatasetDomain):
            graph_batch = graph_factor.sample_component(
                self.spec.selection_for(graph_label),
                graph_n,
                structure=SampleLayout(((graph_label,),), axis_names=(graph_axis,)),
                label=graph_label,
                sampler=sampler,
                key=jr.fold_in(
                    block_keys[label_to_block_index[graph_label]],
                    label_to_idx[graph_label],
                ),
            )
        else:
            graph_batch = graph_factor.sample_component(
                self.spec.selection_for(graph_label),
                graph_n,
                structure=SampleLayout(((graph_label,),), axis_names=(graph_axis,)),
                label=graph_label,
            )

        points: dict[str, Any] = dict(graph_batch.points)

        for lbl in self.domain.labels:
            if lbl == graph_label:
                continue
            comp = self.spec.selection_for(lbl)
            factor = self.domain.factor(lbl)

            if lbl in fixed_labels:
                if isinstance(factor, AbstractScalarDomain):
                    if isinstance(comp, FixedStart):
                        val = factor.fixed("start")
                    elif isinstance(comp, FixedEnd):
                        val = factor.fixed("end")
                    else:
                        assert isinstance(comp, Fixed)
                        val = jnp.asarray(comp.value, dtype=float).reshape(())
                    points[lbl] = _as_field(
                        jnp.asarray(val, dtype=float).reshape(()), dims=()
                    )
                    continue

                if isinstance(factor, AbstractGeometry):
                    assert isinstance(comp, Fixed)
                    val = jnp.asarray(comp.value, dtype=float).reshape(
                        (factor.spatial_dim,)
                    )
                    points[lbl] = _as_field(val, dims=(None,))
                    continue

                raise TypeError(
                    f"Unsupported domain factor type {type(factor).__name__}."
                )

            axis = structure_out.axis_for(lbl)
            if axis is None:
                raise ValueError(f"Missing sampling axis for non-fixed label {lbl!r}.")
            bi = label_to_block_index[lbl]
            n = num_points_by_block[bi]
            k = jr.fold_in(block_keys[bi], label_to_idx[lbl])

            if isinstance(factor, AbstractGeometry):
                arr = _sample_geometry(factor, comp, n, sampler=sampler, key=k)
                if arr.ndim == 1:
                    arr = arr.reshape((-1, 1))
                points[lbl] = _as_field(arr, dims=(axis, None))
                continue

            if isinstance(factor, AbstractScalarDomain):
                arr = _sample_scalar(factor, comp, n, sampler=sampler, key=k).reshape(
                    (-1,)
                )
                points[lbl] = _as_field(arr, dims=(axis,))
                continue

            if isinstance(factor, (DatasetDomain, RaggedSeriesDatasetDomain)):
                samples = factor.sample(n, sampler=sampler, key=k)

                def _to_field(v):
                    arr = jnp.asarray(v)
                    if arr.ndim == 0:
                        raise ValueError(
                            "Dataset samples must have a leading sample axis."
                        )
                    return _as_field(arr, dims=(axis,) + (None,) * (arr.ndim - 1))

                points[lbl] = jax.tree_util.tree_map(_to_field, samples)
                continue

            raise TypeError(f"Unsupported domain factor type {type(factor).__name__}.")

        return GraphBatch(
            points=frozendict(points),
            structure=structure_out,
            graph=graph_batch.graph,
            graph_label=graph_label,
            component_kind=graph_batch.component_kind,
        )

    def points(
        self,
        coordinates: Mapping[str, ArrayLike] | ArrayLike,
        /,
    ) -> PointBatch:
        """Bind explicit coordinates to this component as one paired point batch."""
        fixed_labels = _fixed_component_labels(self)
        free_labels = tuple(
            label for label in self.domain.labels if label not in fixed_labels
        )
        layout = SampleLayout((free_labels,) if free_labels else ()).canonicalize(
            self.domain.labels,
            fixed_labels=fixed_labels,
        )
        axis_names = layout.axis_names
        if axis_names is None:
            raise RuntimeError("Explicit-point layout was not canonicalized.")
        axis = axis_names[0] if axis_names else None

        if isinstance(coordinates, Mapping):
            unknown = tuple(
                label for label in coordinates if label not in self.domain.labels
            )
            if unknown:
                raise KeyError(f"Unknown explicit coordinate labels {unknown!r}.")
            raw = frozendict(
                {
                    label: jnp.asarray(value, dtype=float)
                    for label, value in coordinates.items()
                }
            )
        else:
            if not free_labels:
                raise ValueError(
                    "A fully fixed component requires an empty coordinate mapping."
                )
            raw = _split_explicit_points(self.domain, free_labels, coordinates)

        points: dict[str, Any] = {}
        point_count: int | None = None
        for label in self.domain.labels:
            selection = self.spec.selection_for(label)
            factor = self.domain.factor(label)
            if label in fixed_labels:
                if isinstance(factor, AbstractScalarDomain):
                    if isinstance(selection, FixedStart):
                        value = factor.fixed("start")
                    elif isinstance(selection, FixedEnd):
                        value = factor.fixed("end")
                    else:
                        assert isinstance(selection, Fixed)
                        value = jnp.asarray(selection.value, dtype=float).reshape(())
                    points[label] = cx.Field(
                        jnp.asarray(value, dtype=float).reshape(()),
                        dims=(),
                    )
                    continue
                if isinstance(factor, AbstractGeometry):
                    if not isinstance(selection, Fixed):
                        raise TypeError(
                            f"Fixed geometry label {label!r} requires Fixed(...)."
                        )
                    value = jnp.asarray(selection.value, dtype=float).reshape(
                        (int(factor.spatial_dim),)
                    )
                    points[label] = cx.Field(value, dims=(None,))
                    continue
                raise TypeError(
                    f"Explicit points do not support fixed factor "
                    f"{type(factor).__name__} for label {label!r}."
                )

            if label not in raw:
                raise KeyError(
                    f"Missing explicit coordinates for free label {label!r}; "
                    f"expected {free_labels!r}."
                )
            if axis is None:
                raise RuntimeError("Free explicit coordinates require a sampling axis.")
            value = _explicit_point_array(self.domain, label, raw[label])
            count = int(value.shape[0])
            if point_count is None:
                point_count = count
            elif count != point_count:
                raise ValueError(
                    "Explicit coordinate labels must have the same leading "
                    f"point count; expected {point_count}, got {count} for {label!r}."
                )
            points[label] = (
                cx.Field(value, dims=(axis, None))
                if isinstance(factor, AbstractGeometry)
                else cx.Field(value, dims=(axis,))
            )

        return PointBatch(frozendict(points), layout)

    @overload
    def sample(
        self,
        sampling: PointSampling,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PointBatch: ...

    @overload
    def sample(
        self,
        sampling: GridSampling,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> GridBatch: ...

    def sample(
        self,
        sampling: PointSampling | GridSampling,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PointBatch | GridBatch:
        """Materialize a typed sampling request."""
        if isinstance(sampling, PointSampling):
            return self._sample_points(sampling, key=key)
        if isinstance(sampling, GridSampling):
            return self._sample_grid(sampling, key=key)
        raise TypeError("sampling must be a PointSampling or GridSampling.")

    def _sample_points(
        self,
        sampling: PointSampling,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PointBatch:
        from ._irregular_trajectory_dataset import (
            IrregularTrajectoryDatasetDomain,
            sample_irregular_trajectory_component,
        )
        from ._reference import reference_transport
        from ._trajectory_dataset import (
            sample_trajectory_component,
            TrajectoryDatasetDomain,
        )
        from .graph._trajectory import (
            GraphTrajectoryDatasetDomain,
            sample_graph_trajectory_component,
        )

        num_points = sampling.count
        structure = sampling.layout
        if structure is None:
            coupled = isinstance(
                self.domain,
                (
                    GraphTrajectoryDatasetDomain,
                    IrregularTrajectoryDatasetDomain,
                    TrajectoryDatasetDomain,
                ),
            )
            if coupled:
                active_labels = self.domain.labels
            else:
                active_labels = tuple(
                    label
                    for label in self.domain.labels
                    if not isinstance(
                        self.spec.selection_for(label),
                        (FixedStart, FixedEnd, Fixed),
                    )
                )
            structure = (
                SampleLayout((active_labels,)) if active_labels else SampleLayout(())
            )
        design = sampling.design
        sampler_name = design_name(design)

        if isinstance(self.domain, GraphTrajectoryDatasetDomain):
            return cast(
                PointBatch,
                sample_graph_trajectory_component(
                    self,
                    num_points,
                    structure=structure,
                    sampler=sampler_name,
                    key=key,
                ),
            )

        if isinstance(self.domain, TrajectoryDatasetDomain):
            return sample_trajectory_component(
                self,
                num_points,
                structure=structure,
                sampler=sampler_name,
                key=key,
            )

        if isinstance(self.domain, IrregularTrajectoryDatasetDomain):
            return sample_irregular_trajectory_component(
                self,
                num_points,
                structure=structure,
                sampler=sampler_name,
                key=key,
            )

        graph_batch = self._sample_graph_batch(
            num_points,
            structure=structure,
            sampler=sampler_name,
            key=key,
        )
        if graph_batch is not None:
            return graph_batch

        fixed_labels = frozenset(
            lbl
            for lbl in self.domain.labels
            if isinstance(self.spec.selection_for(lbl), (FixedStart, FixedEnd, Fixed))
        )
        structure = structure.canonicalize(self.domain.labels, fixed_labels=fixed_labels)

        if isinstance(num_points, int):
            if len(structure.blocks) != 1:
                raise ValueError(
                    "num_points=int is only valid for paired sampling (exactly one block)."
                )
            num_points_by_block = (int(num_points),)
        else:
            if len(num_points) != len(structure.blocks):
                raise ValueError(
                    f"num_points must have length {len(structure.blocks)} to match blocks."
                )
            num_points_by_block = tuple(int(n) for n in num_points)

        label_to_block_index: dict[str, int] = {}
        for i, block in enumerate(structure.blocks):
            for lbl in block:
                label_to_block_index[lbl] = i
        label_to_idx = {lbl: i for i, lbl in enumerate(self.domain.labels)}

        block_keys = jr.split(key, len(structure.blocks) + 1)
        keys_for_blocks = block_keys[1:]

        transported: dict[str, Any] = {}
        capabilities = design_capabilities(design)
        for block_index, block in enumerate(structure.blocks):
            if len(block) == 1:
                continue
            block_transports = []
            unsupported = []
            for label in block:
                factor = self.domain.factor(label)

                transport = reference_transport(
                    factor,
                    self.spec.selection_for(label),
                )
                if transport is None:
                    unsupported.append(label)
                else:
                    block_transports.append((label, transport))

            if unsupported:
                if not capabilities.factorwise_composable:
                    raise ValueError(
                        f"{sampler_name!r} requires one exact joint reference "
                        f"transport for paired block {block!r}; unsupported labels="
                        f"{tuple(unsupported)!r}. Use a factorwise-composable "
                        "design, split the labels into separate blocks, or provide "
                        "exact transports."
                    )
                continue

            reference_dimension = sum(
                transport.reference_dimension for _, transport in block_transports
            )
            address = SampleAddress(
                "domain",
                "paired-block",
                algorithm_version=DESIGN_ALGORITHM_VERSION,
                target=block,
                role=sampler_name,
            )
            design_key = derive_key(key, address)
            unit = materialize_design(
                design,
                count=num_points_by_block[block_index],
                dimension=reference_dimension,
                key=design_key,
            )
            offset = 0
            for label, transport in block_transports:
                next_offset = offset + transport.reference_dimension
                transported[label] = transport.map(unit[:, offset:next_offset])
                offset = next_offset

        points: dict[str, Any] = {}
        for lbl in self.domain.labels:
            comp = self.spec.selection_for(lbl)
            factor = self.domain.factor(lbl)

            if lbl in fixed_labels:
                if isinstance(factor, AbstractScalarDomain):
                    if isinstance(comp, FixedStart):
                        val = factor.fixed("start")
                    elif isinstance(comp, FixedEnd):
                        val = factor.fixed("end")
                    else:
                        assert isinstance(comp, Fixed)
                        val = jnp.asarray(comp.value, dtype=float).reshape(())
                    points[lbl] = _as_field(
                        jnp.asarray(val, dtype=float).reshape(()), dims=()
                    )
                    continue

                if isinstance(factor, AbstractGeometry):
                    assert isinstance(comp, Fixed)
                    val = jnp.asarray(comp.value, dtype=float).reshape(
                        (factor.spatial_dim,)
                    )
                    points[lbl] = _as_field(val, dims=(None,))
                    continue

                raise TypeError(
                    f"Unsupported domain factor type {type(factor).__name__}."
                )

            axis = structure.axis_for(lbl)
            if axis is None:
                raise ValueError(f"Missing sampling axis for non-fixed label {lbl!r}.")
            bi = label_to_block_index[lbl]
            n = num_points_by_block[bi]

            if lbl in transported:
                samples = transported[lbl]
                if isinstance(factor, AbstractGeometry):
                    arr = jnp.asarray(samples, dtype=float)
                    if arr.ndim == 1:
                        arr = arr.reshape((-1, 1))
                    points[lbl] = _as_field(arr, dims=(axis, None))
                    continue
                if isinstance(factor, AbstractScalarDomain):
                    arr = jnp.asarray(samples, dtype=float).reshape((-1,))
                    points[lbl] = _as_field(arr, dims=(axis,))
                    continue
                if isinstance(factor, DatasetDomain):

                    def _transported_field(value):
                        arr = jnp.asarray(value)
                        return _as_field(
                            arr,
                            dims=(axis,) + (None,) * (arr.ndim - 1),
                        )

                    points[lbl] = jax.tree_util.tree_map(
                        _transported_field,
                        samples,
                    )
                    continue

            if isinstance(factor, AbstractGeometry):
                k = jr.fold_in(keys_for_blocks[bi], label_to_idx[lbl])
                arr = _sample_geometry(factor, comp, n, sampler=sampler_name, key=k)
                if arr.ndim == 1:
                    arr = arr.reshape((-1, 1))
                points[lbl] = _as_field(arr, dims=(axis, None))
                continue

            if isinstance(factor, AbstractScalarDomain):
                k = jr.fold_in(keys_for_blocks[bi], label_to_idx[lbl])
                arr = _sample_scalar(
                    factor, comp, n, sampler=sampler_name, key=k
                ).reshape((-1,))
                points[lbl] = _as_field(arr, dims=(axis,))
                continue

            if isinstance(factor, (DatasetDomain, RaggedSeriesDatasetDomain)):
                k = jr.fold_in(keys_for_blocks[bi], label_to_idx[lbl])
                samples = factor.sample(n, sampler=sampler_name, key=k)

                def _to_field(v):
                    arr = jnp.asarray(v)
                    if arr.ndim == 0:
                        raise ValueError(
                            "Dataset samples must have a leading sample axis."
                        )
                    return _as_field(arr, dims=(axis,) + (None,) * (arr.ndim - 1))

                points[lbl] = jax.tree_util.tree_map(_to_field, samples)
                continue

            raise TypeError(f"Unsupported domain factor type {type(factor).__name__}.")

        return PointBatch(points=frozendict(points), structure=structure)

    def _sample_grid(
        self,
        sampling: GridSampling,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> GridBatch:
        r"""Materialize a coordinate grid with optional dense point blocks.

        Coordinate requests may be counts, axis specifications, or `TensorGridPlan`
        values. Remaining non-fixed labels are sampled through `sampling.dense`.
        """
        from ._trajectory_dataset import TrajectoryDatasetDomain

        coord_separable = sampling.axes
        sampler = design_name(sampling.design)
        unknown = tuple(
            label for label in coord_separable if label not in self.domain.labels
        )
        if unknown:
            raise KeyError(
                f"GridSampling contains labels {unknown!r} outside domain "
                f"{self.domain.labels!r}."
            )

        if isinstance(self.domain, TrajectoryDatasetDomain):
            raise ValueError(
                "TrajectoryDatasetDomain requires paired data-time sampling; "
                "coord-separable trajectory sampling is not supported."
            )

        coord_labels = tuple(lbl for lbl in self.domain.labels if lbl in coord_separable)

        fixed_labels = frozenset(
            lbl
            for lbl in self.domain.labels
            if isinstance(self.spec.selection_for(lbl), (FixedStart, FixedEnd, Fixed))
        )
        coord_label_set = frozenset(coord_labels)
        if fixed_labels & coord_label_set:
            raise ValueError(
                "GridSampling.axes must not include fixed labels; got "
                f"{tuple(sorted(fixed_labels & coord_label_set))!r}."
            )

        dense_labels = tuple(
            lbl
            for lbl in self.domain.labels
            if (lbl not in fixed_labels) and (lbl not in coord_label_set)
        )
        dense_sampling = sampling.dense
        if dense_sampling is None:
            if dense_labels:
                raise ValueError(
                    "GridSampling.dense is required for non-grid labels "
                    f"{dense_labels!r}."
                )
            num_points: NumPoints = ()
            dense_structure_in = SampleLayout(())
            dense_sampler = sampler
        else:
            num_points = dense_sampling.count
            dense_structure_in = dense_sampling.layout or (
                SampleLayout((dense_labels,)) if dense_labels else SampleLayout(())
            )
            dense_sampler = design_name(dense_sampling.design)
        dense_structure_out = dense_structure_in.canonicalize(
            dense_labels, fixed_labels=frozenset()
        )

        if isinstance(num_points, int):
            if len(dense_structure_out.blocks) == 0:
                num_points_by_block = ()
            else:
                if len(dense_structure_out.blocks) != 1:
                    raise ValueError(
                        "PointSampling.count=int requires exactly one dense layout block."
                    )
                num_points_by_block = (int(num_points),)
        else:
            num_points_by_block = tuple(int(n) for n in num_points)
            if len(num_points_by_block) != len(dense_structure_out.blocks):
                raise ValueError(
                    f"PointSampling.count must have length {len(dense_structure_out.blocks)} "
                    "to match dense layout blocks."
                )

        label_to_block_index: dict[str, int] = {}
        for i, block in enumerate(dense_structure_out.blocks):
            for lbl in block:
                label_to_block_index[lbl] = i

        label_to_idx = {lbl: i for i, lbl in enumerate(self.domain.labels)}

        num_dense_blocks = len(dense_structure_out.blocks)
        coord_keys = jr.split(key, len(coord_labels) + 1)
        coord_keys_for_labels = coord_keys[1:]
        dense_keys = jr.split(coord_keys[0], num_dense_blocks + 1)
        dense_keys_for_blocks = dense_keys[1:]

        coord_axes_by_label: dict[str, tuple[str, ...]] = {}
        coord_mask_by_label: dict[str, cx.Field] = {}
        coord_geometry_weight_by_label: dict[str, cx.Field] = {}
        coord_geometry_order_by_label: dict[str, int] = {}
        axis_discretization_by_axis: dict[str, AxisDiscretization] = {}
        points: dict[str, Any] = {}

        coord_key_by_label = {
            lbl: jr.fold_in(coord_keys_for_labels[i], label_to_idx[lbl])
            for i, lbl in enumerate(coord_labels)
        }

        for lbl in self.domain.labels:
            comp = self.spec.selection_for(lbl)
            factor = self.domain.factor(lbl)

            if lbl in fixed_labels:
                if isinstance(factor, AbstractScalarDomain):
                    if isinstance(comp, FixedStart):
                        val = factor.fixed("start")
                    elif isinstance(comp, FixedEnd):
                        val = factor.fixed("end")
                    else:
                        assert isinstance(comp, Fixed)
                        val = jnp.asarray(comp.value, dtype=float).reshape(())
                    points[lbl] = _as_field(
                        jnp.asarray(val, dtype=float).reshape(()), dims=()
                    )
                    continue

                if isinstance(factor, AbstractGeometry):
                    assert isinstance(comp, Fixed)
                    val = jnp.asarray(comp.value, dtype=float).reshape(
                        (factor.spatial_dim,)
                    )
                    points[lbl] = _as_field(val, dims=(None,))
                    continue

                raise TypeError(
                    f"Unsupported domain factor type {type(factor).__name__}."
                )

            if lbl in coord_label_set:
                if isinstance(factor, AbstractGeometry):
                    var_dim = int(factor.spatial_dim)
                elif isinstance(factor, AbstractScalarDomain):
                    var_dim = 1
                else:
                    raise TypeError(
                        "coord_separable requires a geometry/scalar label; got "
                        f"{lbl!r} with factor {type(factor).__name__}."
                    )
                if not isinstance(comp, Interior):
                    raise ValueError(
                        "coord_separable currently supports only Interior() components; "
                        f"got {type(comp).__name__} for {lbl!r}."
                    )

                n_spec = coord_separable[lbl]
                where_fn = self.where.get(lbl)

                axis_specs: tuple[AbstractAxisSpec, ...] | None = None
                counts: tuple[int, ...] | None = None

                if isinstance(n_spec, TensorGridPlan):
                    axis_specs = n_spec.axes
                elif isinstance(n_spec, AbstractAxisSpec):
                    axis_specs = (n_spec,) * var_dim
                elif isinstance(n_spec, int):
                    counts = (int(n_spec),) * var_dim
                else:
                    seq = tuple(n_spec)
                    if not seq:
                        raise ValueError(f"coord_separable[{lbl!r}] must be non-empty.")
                    axis_specs_candidate = tuple(
                        s for s in seq if isinstance(s, AbstractAxisSpec)
                    )
                    if len(axis_specs_candidate) == len(seq):
                        axis_specs = axis_specs_candidate
                    else:
                        counts_candidate = tuple(
                            int(n) for n in seq if isinstance(n, int)
                        )
                        if len(counts_candidate) == len(seq):
                            counts = counts_candidate
                        else:
                            raise TypeError(
                                f"coord_separable[{lbl!r}] must be int, Sequence[int], "
                                "AbstractAxisSpec, Sequence[AbstractAxisSpec], or TensorGridPlan."
                            )

                geometry_weight_arr: Array | None = None
                geometry_order = 0
                if isinstance(factor, AbstractGeometry):
                    if axis_specs is not None:
                        if len(axis_specs) != var_dim:
                            raise ValueError(
                                f"coord_separable[{lbl!r}] must have length {var_dim}."
                            )
                        bounds = jnp.asarray(factor.mesh_bounds, dtype=float)
                        coords = []
                        for i, spec in enumerate(axis_specs):
                            disc = spec.materialize(bounds[0, i], bounds[1, i])
                            coords.append(disc.nodes)
                            axis_name = _axis_name_for_coord(lbl, i)
                            axis_discretization_by_axis[axis_name] = disc

                        coords_tuple = tuple(coords)
                        mask_arr = sdf_mask_from_adf(factor.adf, coords_tuple)
                        if (
                            isinstance(n_spec, TensorGridPlan)
                            and n_spec.cut_cell_order > 0
                        ):
                            base_weights: list[Array] = []
                            for i, coord in enumerate(coords_tuple):
                                axis_name = _axis_name_for_coord(lbl, i)
                                disc = axis_discretization_by_axis[axis_name]
                                if disc.quad_weights is not None:
                                    base_weights.append(
                                        jnp.asarray(disc.quad_weights, dtype=float)
                                    )
                                else:
                                    length = bounds[1, i] - bounds[0, i]
                                    base_weights.append(
                                        jnp.full(
                                            coord.shape,
                                            length / float(coord.shape[0]),
                                            dtype=float,
                                        )
                                    )
                            geometry_order = n_spec.cut_cell_order
                            geometry_weight_arr = cut_cell_geometry_weight_from_adf(
                                factor.adf,
                                coords_tuple,
                                bounds,
                                tuple(base_weights),
                                mask_arr,
                                factor.volume,
                                order=geometry_order,
                            )
                        if where_fn is not None:
                            grid = broadcasted_grid(coords_tuple)
                            pts = grid.reshape((-1, var_dim))
                            where_mask = jax.vmap(where_fn)(pts).reshape(grid.shape[:-1])
                            mask_arr = mask_arr & jnp.asarray(where_mask, dtype=bool)

                        coords_out = coords_tuple
                        mask = mask_arr
                    else:
                        assert counts is not None
                        if len(counts) != var_dim:
                            raise ValueError(
                                f"coord_separable[{lbl!r}] must have length {var_dim}."
                            )
                        coords_out, mask = factor._sample_interior_separable(
                            counts,
                            sampler=sampler,
                            where=where_fn,
                            key=coord_key_by_label[lbl],
                        )
                else:
                    assert isinstance(factor, AbstractScalarDomain)
                    if axis_specs is not None:
                        if len(axis_specs) != 1:
                            raise ValueError(
                                f"coord_separable[{lbl!r}] must have length 1."
                            )
                        start = jnp.asarray(factor.fixed("start"), dtype=float).reshape(
                            ()
                        )
                        end = jnp.asarray(factor.fixed("end"), dtype=float).reshape(())
                        disc = axis_specs[0].materialize(start, end)
                        coord = jnp.asarray(disc.nodes, dtype=float).reshape((-1,))
                        axis_name = _axis_name_for_coord(lbl, 0)
                        axis_discretization_by_axis[axis_name] = disc
                    else:
                        assert counts is not None
                        if len(counts) != 1:
                            raise ValueError(
                                f"coord_separable[{lbl!r}] must have length 1."
                            )
                        coord = jnp.asarray(
                            _sample_scalar(
                                factor,
                                comp,
                                int(counts[0]),
                                sampler=sampler,
                                key=coord_key_by_label[lbl],
                            ),
                            dtype=float,
                        ).reshape((-1,))
                    if where_fn is not None:
                        mask = jnp.asarray(jax.vmap(where_fn)(coord), dtype=bool).reshape(
                            (-1,)
                        )
                    else:
                        mask = jnp.ones((coord.shape[0],), dtype=bool)
                    coords_out = (coord,)

                if len(coords_out) != var_dim:
                    raise ValueError(
                        f"{type(factor).__name__}._sample_interior_separable returned "
                        f"{len(coords_out)} coordinate arrays; expected {var_dim}."
                    )

                coord_axes: list[jax.Array] = []
                for c in coords_out:
                    arr = jnp.asarray(c, dtype=float)
                    if arr.ndim == 2 and arr.shape[1] == 1:
                        arr = arr.reshape((-1,))
                    if arr.ndim != 1:
                        raise ValueError(
                            "coord-separable coordinate arrays must be 1D; got shape "
                            f"{arr.shape} for label {lbl!r}."
                        )
                    coord_axes.append(arr)

                axis_names = tuple(
                    _axis_name_for_coord(lbl, i) for i in range(len(coord_axes))
                )
                points[lbl] = tuple(
                    cx.Field(arr, dims=(ax,))
                    for arr, ax in zip(coord_axes, axis_names, strict=True)
                )
                coord_axes_by_label[lbl] = axis_names

                mask_arr = jnp.asarray(mask, dtype=bool)
                coord_mask_by_label[lbl] = cx.Field(mask_arr, dims=axis_names)
                if geometry_weight_arr is not None:
                    coord_geometry_weight_by_label[lbl] = cx.Field(
                        geometry_weight_arr,
                        dims=axis_names,
                    )
                    coord_geometry_order_by_label[lbl] = geometry_order
                continue

            axis = dense_structure_out.axis_for(lbl)
            if axis is None:
                raise ValueError(f"Missing sampling axis for non-fixed label {lbl!r}.")
            bi = label_to_block_index[lbl]
            n = num_points_by_block[bi]

            if isinstance(factor, AbstractGeometry):
                k = jr.fold_in(dense_keys_for_blocks[bi], label_to_idx[lbl])
                arr = _sample_geometry(factor, comp, n, sampler=dense_sampler, key=k)
                if arr.ndim == 1:
                    arr = arr.reshape((-1, 1))
                points[lbl] = _as_field(arr, dims=(axis, None))
                continue

            if isinstance(factor, AbstractScalarDomain):
                k = jr.fold_in(dense_keys_for_blocks[bi], label_to_idx[lbl])
                arr = _sample_scalar(
                    factor, comp, n, sampler=dense_sampler, key=k
                ).reshape((-1,))
                points[lbl] = _as_field(arr, dims=(axis,))
                continue

            if isinstance(factor, (DatasetDomain, RaggedSeriesDatasetDomain)):
                k = jr.fold_in(dense_keys_for_blocks[bi], label_to_idx[lbl])
                samples = factor.sample(n, sampler=dense_sampler, key=k)

                def _to_field(v):
                    arr = jnp.asarray(v)
                    if arr.ndim == 0:
                        raise ValueError(
                            "Dataset samples must have a leading sample axis."
                        )
                    return _as_field(arr, dims=(axis,) + (None,) * (arr.ndim - 1))

                points[lbl] = jax.tree_util.tree_map(_to_field, samples)
                continue

            raise TypeError(f"Unsupported domain factor type {type(factor).__name__}.")

        return GridBatch(
            points=frozendict(points),
            dense_structure=dense_structure_out,
            coord_axes_by_label=frozendict(coord_axes_by_label),
            coord_mask_by_label=frozendict(coord_mask_by_label),
            coord_geometry_weight_by_label=frozendict(coord_geometry_weight_by_label),
            coord_geometry_order_by_label=frozendict(coord_geometry_order_by_label),
            axis_discretization_by_axis=frozendict(axis_discretization_by_axis),
        )

    def normals(
        self,
        points: PointBatch | Points,
        /,
        *,
        var: str,
    ) -> cx.Field:
        r"""Compute outward unit normals on a geometry boundary.

        For a geometry label `var` with boundary component, this returns the unit normal
        field $n(x)$ on $\partial\Omega$.

        The returned `coordax.Field` has the same named axes as the provided boundary
        points.
        """
        if isinstance(points, PointBatch):
            points_map = points.points
        else:
            points_map = points

        if var not in self.domain.labels:
            raise KeyError(f"Label {var!r} not in domain {self.domain.labels}.")

        comp = self.spec.selection_for(var)
        if not isinstance(comp, Boundary):
            raise ValueError(
                "DomainComponent.normals is only defined for Boundary() components."
            )

        factor = self.domain.factor(var)

        if not isinstance(factor, AbstractGeometry):
            raise TypeError(
                f"normals(var=...) requires a geometry label, got {type(factor).__name__}."
            )

        x = points_map[var]
        if not isinstance(x, cx.Field):
            raise TypeError(
                "normals(var=...) requires points[var] to be a coordax.Field of geometry coordinates."
            )

        pts = jnp.asarray(x.data, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape((1, -1))
        if pts.ndim != 2:
            raise ValueError(
                f"Expected geometry points to be rank-2 array, got shape {pts.shape}."
            )

        n = jnp.asarray(factor._boundary_normals(pts), dtype=float)
        eps = jnp.finfo(float).eps
        nrm = jnp.linalg.norm(n, axis=-1, keepdims=True) + eps
        n_unit = n / nrm
        return cx.Field(n_unit, dims=x.dims)

    def normal(self, /, *, var: str) -> DomainFunction:
        r"""Return a `DomainFunction` representing the outward unit normal $n(x)$.

        This is a convenience wrapper that returns a `DomainFunction` with
        `deps=(var,)`. For geometry labels, it is typically used in Neumann-type
        conditions involving $\partial u/\partial n$.
        """
        if var not in self.domain.labels:
            raise KeyError(f"Label {var!r} not in domain {self.domain.labels}.")

        comp = self.spec.selection_for(var)
        if not isinstance(comp, Boundary):
            raise ValueError(
                "DomainComponent.normal is only defined for Boundary() components."
            )

        factor = self.domain.factor(var)

        if not isinstance(factor, AbstractGeometry):
            raise TypeError(
                f"normal(var=...) requires a geometry label, got {type(factor).__name__}."
            )

        return DomainFunction(
            domain=self.domain, deps=(var,), func=_NormalCallable(factor)
        )

    def sdf(self, /, *, var: str) -> DomainFunction:
        r"""Return the geometry's signed boundary-defining field $\phi(x)$.

        The field preserves the geometry zero set and uses the convention

        - $\phi(x) < 0$ inside $\Omega$,
        - $\phi(x) = 0$ on $\partial\Omega$,
        - $\phi(x) > 0$ outside $\Omega$.

        Its magnitude is not guaranteed to equal metric distance away from the
        boundary. CAD geometries compactly saturate this geometry field; derivative
        hard constraints use their separately conditioned unit-jet ansatz factor.
        """
        if var not in self.domain.labels:
            raise KeyError(f"Label {var!r} not in domain {self.domain.labels}.")

        factor = self.domain.factor(var)

        if not isinstance(factor, AbstractGeometry):
            raise TypeError(
                f"sdf(var=...) requires a geometry label, got {type(factor).__name__}."
            )

        return DomainFunction(domain=self.domain, deps=(var,), func=_SdfCallable(factor))

    def enforcement_gate(
        self,
        /,
        *,
        method: EnforcementGateMethod = "auto",
        var: str,
        saturation_fraction: float = 0.5,
        linear_fraction: float = 0.5,
    ) -> DomainFunction:
        r"""Return a dimensionless gate for optimization-conditioned hard constraints.

        The gate is zero on the selected geometry boundary and positive, typically
        order one, in the interior. Unlike :meth:`sdf`, it is not used to define
        normals or derivative boundary conditions.
        """
        if var not in self.domain.labels:
            raise KeyError(f"Label {var!r} not in domain {self.domain.labels}.")

        component = self.spec.selection_for(var)
        if not isinstance(component, Boundary):
            raise ValueError(
                "DomainComponent.enforcement_gate is only defined for Boundary() "
                "components."
            )

        factor = self.domain.factor(var)

        if not isinstance(factor, AbstractGeometry):
            raise TypeError(
                "enforcement_gate(var=...) requires a geometry label, "
                f"got {type(factor).__name__}."
            )

        return DomainFunction(
            domain=self.domain,
            deps=(var,),
            func=_EnforcementGateCallable(
                factor,
                method=method,
                saturation_fraction=saturation_fraction,
                linear_fraction=linear_fraction,
            ),
        )


class ComponentSum(StrictModule):
    r"""An additive collection of measure-disjoint domain components.

    This represents components that decompose naturally into disjoint terms, such
    as the two endpoints of an interval or the codimension-one faces of a product
    domain. All terms must use the same compatible labeled domain.

    Measures and sampling allocations are additive. Arbitrary filter overlap
    cannot be detected, so callers must ensure distinct filtered terms are
    measure-disjoint; intersections are otherwise counted once per term.
    """

    terms: tuple[DomainComponent, ...]
    assume_disjoint: bool = eqx.field(static=True)

    def __init__(
        self,
        terms: tuple[DomainComponent, ...],
        /,
        *,
        assume_disjoint: bool = False,
    ):
        """Create an additive collection from non-empty compatible terms."""
        resolved_terms = tuple(terms)
        if not resolved_terms:
            raise ValueError("ComponentSum.terms must be non-empty.")
        if not all(isinstance(term, DomainComponent) for term in resolved_terms):
            raise TypeError("ComponentSum terms must be DomainComponent values.")

        domain = resolved_terms[0].domain
        for index, term in enumerate(resolved_terms):
            if not domain.same_support(term.domain):
                raise ValueError(
                    "ComponentSum terms must share the same compatible labeled domain."
                )
            for previous in resolved_terms[:index]:
                if bool(eqx.tree_equal(term, previous)):
                    raise ValueError("ComponentSum terms must not contain duplicates.")
        has_unresolved_overlap = any(
            term.where or term.where_all is not None or term.weight_all is not None
            for term in resolved_terms
        )
        if has_unresolved_overlap and not assume_disjoint:
            raise ValueError(
                "Predicate- or density-transformed ComponentSum terms require "
                "assume_disjoint=True; overlap cannot be certified structurally."
            )
        self.terms = resolved_terms
        self.assume_disjoint = bool(assume_disjoint)

    @property
    def domain(self) -> Domain:
        return self.terms[0].domain

    @property
    def labels(self) -> tuple[str, ...]:
        return self.domain.labels

    @property
    def mass(self) -> Mass:
        """Return the typed additive mass of all disjoint terms."""
        return sum_mass(tuple(term.mass for term in self.terms))

    @property
    def base_measure(self) -> BaseMeasure:
        """Additive reference measure before term restrictions and densities."""
        return BaseMeasure(
            "external",
            sum_mass(tuple(term.base_measure.mass for term in self.terms)),
        )

    def sample(
        self,
        sampling: PointSampling | tuple[PointSampling, ...],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        min_points_per_term: int = 1,
    ) -> tuple[PointBatch, ...]:
        """Sample every additive term with explicit per-term point requests."""
        num_terms = len(self.terms)
        if min_points_per_term < 1:
            raise ValueError("min_points_per_term must be >= 1.")
        if isinstance(sampling, GridSampling):
            raise TypeError("ComponentSum does not support GridSampling.")

        if isinstance(sampling, PointSampling):
            if not isinstance(sampling.count, int):
                raise ValueError(
                    "ComponentSum requires an integer total count or one "
                    "PointSampling per term."
                )
            total = sampling.count
            if total < num_terms * min_points_per_term:
                raise ValueError(
                    "PointSampling.count is too small to allocate at least "
                    f"{min_points_per_term} point(s) per term."
                )
            counts = [min_points_per_term] * num_terms
            for index in range(total - num_terms * min_points_per_term):
                counts[index % num_terms] += 1
            plans = tuple(
                PointSampling(
                    count,
                    layout=sampling.layout,
                    design=sampling.design,
                )
                for count in counts
            )
        else:
            plans = tuple(sampling)
            if len(plans) != num_terms or not all(
                isinstance(plan, PointSampling) for plan in plans
            ):
                raise ValueError(
                    f"ComponentSum requires {num_terms} PointSampling requests."
                )

        keys = jr.split(key, num_terms)
        return tuple(
            term.sample(plan, key=term_key)
            for term, plan, term_key in zip(self.terms, plans, keys, strict=True)
        )
