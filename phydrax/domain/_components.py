#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

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
    DesignLike,
    materialize_design,
    resolve_design,
    SampleAddress,
)
from .._strict import StrictModule
from ._base import _AbstractGeometry, EnforcementGateMethod
from ._dataset import DatasetDomain
from ._domain import _AbstractDomain, RelabeledDomain
from ._function import DomainFunction
from ._grid import (
    AbstractAxisSpec,
    AxisDiscretization,
    broadcasted_grid,
    cut_cell_geometry_weight_from_adf,
    GridSpec,
    sdf_mask_from_adf,
)
from ._ragged_series_dataset import RaggedSeriesDatasetDomain
from ._scalar import _AbstractScalarDomain
from ._structure import (
    _axis_name_for_coord,
    CoordSeparableBatch,
    NumPoints,
    Points,
    PointsBatch,
    ProductStructure,
)


class _AbstractVarComponent(StrictModule):
    pass


class Interior(_AbstractVarComponent):
    r"""Marker selecting the interior of a domain factor.

    For a geometry factor $\Omega\subset\mathbb{R}^d$, this corresponds to sampling
    from $\Omega$ (volume/area/length measure). For a scalar factor like a time
    interval $[t_0,t_1]$, this corresponds to sampling from the interval interior.
    """

    def __init__(self):
        """Create an interior component marker."""


class Boundary(_AbstractVarComponent):
    r"""Marker selecting the boundary of a domain factor.

    For a geometry factor $\Omega\subset\mathbb{R}^d$, this corresponds to sampling
    from $\partial\Omega$ (surface measure). For scalar factors (e.g. time intervals),
    the "boundary" is the discrete set of endpoints (counting measure).
    """

    def __init__(self):
        """Create a boundary component marker."""


class Fixed(_AbstractVarComponent):
    r"""Fix a scalar coordinate to a specific value.

    Interpreted as a Dirac measure of unit mass at the fixed value. This is supported
    for scalar domains (like `TimeInterval`) and is used for slices such as $t=t_0$.

    Note: Fixing geometry coordinates is not supported by the sampler; use a `where`
    mask or construct a lower-dimensional geometry instead.
    """

    value: Array

    def __init__(self, value: ArrayLike):
        """Create a fixed scalar component at the given value."""
        self.value = jnp.asarray(value, dtype=float)


class FixedStart(_AbstractVarComponent):
    r"""Fix a scalar domain to its start endpoint (e.g. $t=t_0$)."""

    def __init__(self):
        """Create a fixed-start component marker."""


class FixedEnd(_AbstractVarComponent):
    r"""Fix a scalar domain to its end endpoint (e.g. $t=t_1$)."""

    def __init__(self):
        """Create a fixed-end component marker."""


class ComponentSpec(StrictModule):
    r"""Mapping from domain labels to component selectors.

    A `ComponentSpec` assigns each label $\ell$ in a domain to one of:

    - `Interior()` : use the interior $\Omega_\ell$;
    - `Boundary()` : use the boundary $\partial\Omega_\ell$ (or endpoints for scalars);
    - `FixedStart()` / `FixedEnd()` : fix a scalar domain to its endpoints;
    - `Fixed(value)` : fix a scalar domain to an arbitrary value.

    Any label not explicitly specified defaults to `Interior()`.
    """

    by_label: frozendict[str, _AbstractVarComponent]

    def __init__(self, by_label: Mapping[str, _AbstractVarComponent] | None = None):
        """Create a component specification from a `{label: component}` mapping."""
        self.by_label = frozendict(by_label or {})

    def component_for(self, label: str, /) -> _AbstractVarComponent:
        """Return the component selector for `label` (defaults to `Interior()`)."""
        return self.by_label.get(label, Interior())


def _as_field(x: Array, *, dims: tuple[str | None, ...]) -> cx.Field:
    return cx.Field(x, dims=dims)


class _NormalCallable(StrictModule):
    geom: _AbstractGeometry

    def __init__(self, geom: _AbstractGeometry):
        self.geom = geom

    def __call__(self, x: Array, /, *, key=None, **kwargs: Any) -> Array:
        del key, kwargs
        pts_in = jnp.asarray(x, dtype=float)
        d = int(self.geom.var_dim)
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
    geom: _AbstractGeometry

    def __init__(self, geom: _AbstractGeometry):
        self.geom = geom

    def __call__(self, x: Any, /, *, key=None, **kwargs: Any) -> Array:
        del key, kwargs
        d = int(self.geom.var_dim)

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
        geom: _AbstractGeometry,
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
        self.dim = int(geom.var_dim)

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
    geom: _AbstractGeometry,
    component: _AbstractVarComponent,
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
        return jnp.asarray(
            geom.sample_boundary(num_points, sampler=sampler, key=key), dtype=float
        )
    if isinstance(component, Fixed):
        raise ValueError(
            "Fixed(x) is not supported for geometries in sampling; use a unary DomainFunction mask instead."
        )
    raise TypeError(f"Unsupported geometry component {type(component).__name__}.")


def _sample_scalar(
    dom: _AbstractScalarDomain,
    component: _AbstractVarComponent,
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

    domain: _AbstractDomain
    spec: ComponentSpec
    where: frozendict[str, Callable]
    where_all: DomainFunction | None
    weight_all: DomainFunction | None

    def __init__(
        self,
        *,
        domain: _AbstractDomain,
        spec: ComponentSpec | None = None,
        where: Mapping[str, Callable] | None = None,
        where_all: DomainFunction | None = None,
        weight_all: DomainFunction | None = None,
    ):
        self.domain = domain
        self.spec = spec or ComponentSpec()
        self.where = frozendict(where or {})
        self.where_all = where_all
        self.weight_all = weight_all

        if self.where_all is not None and not isinstance(self.where_all, DomainFunction):
            self.where_all = DomainFunction(
                domain=self.domain,
                deps=self.domain.labels,
                func=self.where_all,
                metadata={},
            )
        if self.weight_all is not None and not isinstance(
            self.weight_all, DomainFunction
        ):
            self.weight_all = DomainFunction(
                domain=self.domain,
                deps=self.domain.labels,
                func=self.weight_all,
                metadata={},
            )

    def measure(self) -> Array:
        r"""Return the total measure of this component.

        This computes the product of the per-label component measures, e.g.

        $$
        \mu(\Omega_{\text{comp}})=\prod_{\ell\in\mathcal{L}} \mu_\ell\left(\Omega_\ell^{(\text{spec})}\right).
        $$

        For scalar boundaries $\{a,b\}$ this uses counting measure (mass $2$), and for
        fixed slices uses unit mass.
        """
        from ._irregular_trajectory_dataset import irregular_trajectory_component_measure
        from ._trajectory_dataset import trajectory_component_measure
        from .graph._trajectory import graph_trajectory_component_measure

        graph_custom = graph_trajectory_component_measure(self)
        if graph_custom is not None:
            return graph_custom

        custom = trajectory_component_measure(self)
        if custom is not None:
            return custom

        irregular_custom = irregular_trajectory_component_measure(self)
        if irregular_custom is not None:
            return irregular_custom

        m = jnp.array(1.0, dtype=float)
        for lbl in self.domain.labels:
            comp = self.spec.component_for(lbl)
            factor = self.domain.factor(lbl)
            if isinstance(factor, RelabeledDomain):
                factor = factor.base

            if isinstance(factor, _AbstractGeometry):
                if isinstance(comp, Interior):
                    mi = factor.volume
                elif isinstance(comp, Boundary):
                    mi = factor.boundary_measure_value
                elif isinstance(comp, Fixed):
                    mi = jnp.array(1.0, dtype=float)
                else:
                    raise TypeError(
                        f"Unsupported geometry component {type(comp).__name__}."
                    )

            elif isinstance(factor, _AbstractScalarDomain):
                if isinstance(comp, Interior):
                    mi = factor.measure
                elif isinstance(comp, Boundary):
                    mi = jnp.array(2.0, dtype=float)
                elif isinstance(comp, (FixedStart, FixedEnd, Fixed)):
                    mi = jnp.array(1.0, dtype=float)
                else:
                    raise TypeError(
                        f"Unsupported scalar component {type(comp).__name__}."
                    )

            elif isinstance(factor, (DatasetDomain, RaggedSeriesDatasetDomain)):
                if isinstance(comp, Interior):
                    mi = factor.measure
                else:
                    raise TypeError(
                        f"Unsupported dataset component {type(comp).__name__}."
                    )

            else:
                from .graph._dataset import GraphDatasetDomain
                from .graph._domain import GraphDomain

                if isinstance(factor, (GraphDomain, GraphDatasetDomain)):
                    mi = factor.component_measure(comp)
                else:
                    raise TypeError(
                        f"Unsupported unary domain type {type(factor).__name__}."
                    )

            m = m * jnp.asarray(mi, dtype=float)
        return m

    def _sample_graph_batch(
        self,
        num_points: NumPoints,
        *,
        structure: ProductStructure,
        sampler: str,
        key: Key[Array, ""],
    ):
        from .graph._batch import GraphBatch
        from .graph._dataset import GraphDatasetDomain
        from .graph._domain import GraphDomain

        graph_labels: list[str] = []
        for lbl in self.domain.labels:
            factor = self.domain.factor(lbl)
            if isinstance(factor, RelabeledDomain):
                factor = factor.base
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
        if isinstance(graph_factor, RelabeledDomain):
            graph_factor = graph_factor.base
        assert isinstance(graph_factor, (GraphDomain, GraphDatasetDomain))

        fixed_labels = frozenset(
            lbl
            for lbl in self.domain.labels
            if isinstance(self.spec.component_for(lbl), (FixedStart, FixedEnd, Fixed))
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
                    "GraphDomain labels must be sampled in singleton ProductStructure "
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
                self.spec.component_for(graph_label),
                graph_n,
                structure=ProductStructure(((graph_label,),), axis_names=(graph_axis,)),
                label=graph_label,
                sampler=sampler,
                key=jr.fold_in(
                    block_keys[label_to_block_index[graph_label]],
                    label_to_idx[graph_label],
                ),
            )
        else:
            graph_batch = graph_factor.sample_component(
                self.spec.component_for(graph_label),
                graph_n,
                structure=ProductStructure(((graph_label,),), axis_names=(graph_axis,)),
                label=graph_label,
            )

        points: dict[str, Any] = dict(graph_batch.points)

        for lbl in self.domain.labels:
            if lbl == graph_label:
                continue
            comp = self.spec.component_for(lbl)
            factor = self.domain.factor(lbl)
            if isinstance(factor, RelabeledDomain):
                factor = factor.base

            if lbl in fixed_labels:
                if isinstance(factor, _AbstractScalarDomain):
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

                if isinstance(factor, _AbstractGeometry):
                    assert isinstance(comp, Fixed)
                    val = jnp.asarray(comp.value, dtype=float).reshape((factor.var_dim,))
                    points[lbl] = _as_field(val, dims=(None,))
                    continue

                raise TypeError(f"Unsupported unary domain type {type(factor).__name__}.")

            axis = structure_out.axis_for(lbl)
            if axis is None:
                raise ValueError(f"Missing sampling axis for non-fixed label {lbl!r}.")
            bi = label_to_block_index[lbl]
            n = num_points_by_block[bi]
            k = jr.fold_in(block_keys[bi], label_to_idx[lbl])

            if isinstance(factor, _AbstractGeometry):
                arr = _sample_geometry(factor, comp, n, sampler=sampler, key=k)
                if arr.ndim == 1:
                    arr = arr.reshape((-1, 1))
                points[lbl] = _as_field(arr, dims=(axis, None))
                continue

            if isinstance(factor, _AbstractScalarDomain):
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

            raise TypeError(f"Unsupported unary domain type {type(factor).__name__}.")

        return GraphBatch(
            points=frozendict(points),
            structure=structure_out,
            graph=graph_batch.graph,
            graph_label=graph_label,
            component_kind=graph_batch.component_kind,
        )

    def sample(
        self,
        num_points: NumPoints,
        *,
        structure: ProductStructure,
        sampler: DesignLike = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PointsBatch:
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

        design = resolve_design(sampler)
        sampler_name = design_name(design)

        if isinstance(self.domain, GraphTrajectoryDatasetDomain):
            return cast(
                PointsBatch,
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
            if isinstance(self.spec.component_for(lbl), (FixedStart, FixedEnd, Fixed))
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
                if isinstance(factor, RelabeledDomain):
                    factor = factor.base
                transport = reference_transport(
                    factor,
                    self.spec.component_for(label),
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
            comp = self.spec.component_for(lbl)
            factor = self.domain.factor(lbl)
            if isinstance(factor, RelabeledDomain):
                factor = factor.base

            if lbl in fixed_labels:
                if isinstance(factor, _AbstractScalarDomain):
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

                if isinstance(factor, _AbstractGeometry):
                    assert isinstance(comp, Fixed)
                    val = jnp.asarray(comp.value, dtype=float).reshape((factor.var_dim,))
                    points[lbl] = _as_field(val, dims=(None,))
                    continue

                raise TypeError(f"Unsupported unary domain type {type(factor).__name__}.")

            axis = structure.axis_for(lbl)
            if axis is None:
                raise ValueError(f"Missing sampling axis for non-fixed label {lbl!r}.")
            bi = label_to_block_index[lbl]
            n = num_points_by_block[bi]

            if lbl in transported:
                samples = transported[lbl]
                if isinstance(factor, _AbstractGeometry):
                    arr = jnp.asarray(samples, dtype=float)
                    if arr.ndim == 1:
                        arr = arr.reshape((-1, 1))
                    points[lbl] = _as_field(arr, dims=(axis, None))
                    continue
                if isinstance(factor, _AbstractScalarDomain):
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

            if isinstance(factor, _AbstractGeometry):
                k = jr.fold_in(keys_for_blocks[bi], label_to_idx[lbl])
                arr = _sample_geometry(factor, comp, n, sampler=sampler_name, key=k)
                if arr.ndim == 1:
                    arr = arr.reshape((-1, 1))
                points[lbl] = _as_field(arr, dims=(axis, None))
                continue

            if isinstance(factor, _AbstractScalarDomain):
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

            raise TypeError(f"Unsupported unary domain type {type(factor).__name__}.")

        return PointsBatch(points=frozendict(points), structure=structure)

    def sample_coord_separable(
        self,
        coord_separable: Mapping[
            str,
            int
            | Sequence[int]
            | AbstractAxisSpec
            | Sequence[AbstractAxisSpec]
            | GridSpec,
        ],
        /,
        *,
        num_points: NumPoints = (),
        dense_structure: ProductStructure | None = None,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> CoordSeparableBatch:
        r"""Sample a coordinate-separable batch.

        For selected unary labels, this samples each coordinate axis independently,
        producing coordinate arrays (and an associated boolean mask) suitable for
        grid-like evaluation. Any remaining (non-fixed, non-separable) labels are
        sampled using `dense_structure`.

        `coord_separable` values may be:
        - counts (`int` or `Sequence[int]`), using the configured random `sampler`;
        - axis specs (`AbstractAxisSpec`, `Sequence[AbstractAxisSpec]`, or `GridSpec`),
          producing deterministic grid nodes and attaching per-axis discretization
          metadata for quadrature/operators.

        This is useful when an operator factorizes across coordinate axes, or when a
        Cartesian grid is desired for quadrature-like reductions.
        """
        from ._trajectory_dataset import TrajectoryDatasetDomain

        if isinstance(self.domain, TrajectoryDatasetDomain):
            raise ValueError(
                "TrajectoryDatasetDomain requires paired data-time sampling; "
                "coord-separable trajectory sampling is not supported."
            )

        coord_labels = tuple(lbl for lbl in self.domain.labels if lbl in coord_separable)
        for lbl in coord_labels:
            if lbl not in self.domain.labels:
                raise KeyError(f"Label {lbl!r} not in domain {self.domain.labels}.")

        fixed_labels = frozenset(
            lbl
            for lbl in self.domain.labels
            if isinstance(self.spec.component_for(lbl), (FixedStart, FixedEnd, Fixed))
        )
        coord_label_set = frozenset(coord_labels)
        if fixed_labels & coord_label_set:
            raise ValueError(
                "coord_separable must not include fixed labels; got "
                f"{tuple(sorted(fixed_labels & coord_label_set))!r}."
            )

        dense_labels = tuple(
            lbl
            for lbl in self.domain.labels
            if (lbl not in fixed_labels) and (lbl not in coord_label_set)
        )
        dense_structure_in = dense_structure or ProductStructure(blocks=())
        dense_structure_out = dense_structure_in.canonicalize(
            dense_labels, fixed_labels=frozenset()
        )

        if isinstance(num_points, int):
            if len(dense_structure_out.blocks) == 0:
                num_points_by_block = ()
            else:
                if len(dense_structure_out.blocks) != 1:
                    raise ValueError(
                        "num_points=int is only valid when dense_structure has exactly one block."
                    )
                num_points_by_block = (int(num_points),)
        else:
            num_points_by_block = tuple(int(n) for n in num_points)
            if len(num_points_by_block) != len(dense_structure_out.blocks):
                raise ValueError(
                    f"num_points must have length {len(dense_structure_out.blocks)} to match dense_structure blocks."
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
            comp = self.spec.component_for(lbl)
            factor = self.domain.factor(lbl)
            if isinstance(factor, RelabeledDomain):
                factor = factor.base

            if lbl in fixed_labels:
                if isinstance(factor, _AbstractScalarDomain):
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

                if isinstance(factor, _AbstractGeometry):
                    assert isinstance(comp, Fixed)
                    val = jnp.asarray(comp.value, dtype=float).reshape((factor.var_dim,))
                    points[lbl] = _as_field(val, dims=(None,))
                    continue

                raise TypeError(f"Unsupported unary domain type {type(factor).__name__}.")

            if lbl in coord_label_set:
                if isinstance(factor, _AbstractGeometry):
                    var_dim = int(factor.var_dim)
                elif isinstance(factor, _AbstractScalarDomain):
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

                if isinstance(n_spec, GridSpec):
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
                                f"coord_separable[{lbl!r}] must be int, Sequence[int], AxisSpec, "
                                "Sequence[AxisSpec], or GridSpec."
                            )

                geometry_weight_arr: Array | None = None
                geometry_order = 0
                if isinstance(factor, _AbstractGeometry):
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
                        if isinstance(n_spec, GridSpec) and n_spec.cut_cell_order > 0:
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
                    assert isinstance(factor, _AbstractScalarDomain)
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

            if isinstance(factor, _AbstractGeometry):
                k = jr.fold_in(dense_keys_for_blocks[bi], label_to_idx[lbl])
                arr = _sample_geometry(factor, comp, n, sampler=sampler, key=k)
                if arr.ndim == 1:
                    arr = arr.reshape((-1, 1))
                points[lbl] = _as_field(arr, dims=(axis, None))
                continue

            if isinstance(factor, _AbstractScalarDomain):
                k = jr.fold_in(dense_keys_for_blocks[bi], label_to_idx[lbl])
                arr = _sample_scalar(factor, comp, n, sampler=sampler, key=k).reshape(
                    (-1,)
                )
                points[lbl] = _as_field(arr, dims=(axis,))
                continue

            if isinstance(factor, (DatasetDomain, RaggedSeriesDatasetDomain)):
                k = jr.fold_in(dense_keys_for_blocks[bi], label_to_idx[lbl])
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

            raise TypeError(f"Unsupported unary domain type {type(factor).__name__}.")

        return CoordSeparableBatch(
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
        points: PointsBatch | Points,
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
        if isinstance(points, PointsBatch):
            points_map = points.points
        else:
            points_map = points

        if var not in self.domain.labels:
            raise KeyError(f"Label {var!r} not in domain {self.domain.labels}.")

        comp = self.spec.component_for(var)
        if not isinstance(comp, Boundary):
            raise ValueError(
                "DomainComponent.normals is only defined for Boundary() components."
            )

        factor = self.domain.factor(var)
        if isinstance(factor, RelabeledDomain):
            factor = factor.base
        if not isinstance(factor, _AbstractGeometry):
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

        comp = self.spec.component_for(var)
        if not isinstance(comp, Boundary):
            raise ValueError(
                "DomainComponent.normal is only defined for Boundary() components."
            )

        factor = self.domain.factor(var)
        if isinstance(factor, RelabeledDomain):
            factor = factor.base
        if not isinstance(factor, _AbstractGeometry):
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
        if isinstance(factor, RelabeledDomain):
            factor = factor.base
        if not isinstance(factor, _AbstractGeometry):
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

        component = self.spec.component_for(var)
        if not isinstance(component, Boundary):
            raise ValueError(
                "DomainComponent.enforcement_gate is only defined for Boundary() "
                "components."
            )

        factor = self.domain.factor(var)
        if isinstance(factor, RelabeledDomain):
            factor = factor.base
        if not isinstance(factor, _AbstractGeometry):
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


class DomainComponentUnion(StrictModule):
    r"""An additive collection of measure-disjoint domain components.

    This represents components that decompose naturally into disjoint terms, such
    as the two endpoints of an interval or the codimension-one faces of a product
    domain. All terms must use the same compatible labeled domain.

    Measures and sampling allocations are additive. Arbitrary filter overlap
    cannot be detected, so callers must ensure distinct filtered terms are
    measure-disjoint; intersections are otherwise counted once per term.
    """

    terms: tuple[DomainComponent, ...]

    def __init__(self, terms: tuple[DomainComponent, ...]):
        """Create an additive collection from non-empty compatible terms."""
        resolved_terms = tuple(terms)
        if not resolved_terms:
            raise ValueError("DomainComponentUnion.terms must be non-empty.")
        if not all(isinstance(term, DomainComponent) for term in resolved_terms):
            raise TypeError("DomainComponentUnion terms must be DomainComponent values.")

        domain = resolved_terms[0].domain
        for index, term in enumerate(resolved_terms):
            if term.domain.labels != domain.labels or not domain.equivalent(term.domain):
                raise ValueError(
                    "DomainComponentUnion terms must share the same compatible "
                    "labeled domain."
                )
            for previous in resolved_terms[:index]:
                if bool(eqx.tree_equal(term, previous)):
                    raise ValueError(
                        "DomainComponentUnion terms must not contain duplicates."
                    )
        self.terms = resolved_terms

    @property
    def domain(self) -> _AbstractDomain:
        return self.terms[0].domain

    @property
    def labels(self) -> tuple[str, ...]:
        return self.domain.labels

    def measure(self) -> Array:
        r"""Return the sum of term measures $\sum_i \mu(\Omega_i)$."""
        m = jnp.array(0.0, dtype=float)
        for term in self.terms:
            m = m + term.measure()
        return m

    def sample(
        self,
        num_points: int | tuple[Any, ...],
        *,
        structure: ProductStructure,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
        min_points_per_term: int = 1,
    ) -> tuple[PointsBatch, ...]:
        """Sample each union term and return a tuple of `PointsBatch` values."""
        num_terms = len(self.terms)
        if min_points_per_term < 1:
            raise ValueError("min_points_per_term must be >= 1.")

        if isinstance(num_points, int):
            total = int(num_points)
            if total < num_terms * min_points_per_term:
                raise ValueError(
                    "num_points is too small to allocate at least "
                    f"{min_points_per_term} point(s) per term."
                )
            counts = [min_points_per_term] * num_terms
            remaining = total - num_terms * min_points_per_term
            for i in range(remaining):
                counts[i % num_terms] += 1
            per_term = tuple(int(c) for c in counts)
        else:
            if len(num_points) != num_terms:
                raise ValueError(
                    f"num_points must have length {num_terms} (one entry per union term)."
                )
            per_term = tuple(num_points)

        keys = jr.split(key, num_terms)
        batches: list[PointsBatch] = []
        for term, n, k in zip(self.terms, per_term, keys, strict=True):
            batches.append(
                term.sample(
                    n,
                    structure=structure,
                    sampler=sampler,
                    key=k,
                )
            )
        return tuple(batches)


VarComponent = _AbstractVarComponent
