#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key, PyTree

from ..._callable import _ensure_special_kwonly_args
from ..._doc import DOC_KEY0
from ...domain._base import _AbstractGeometry
from ...domain._components import DomainComponent, DomainComponentUnion
from ...domain._domain import RelabeledDomain
from ...domain._function import DomainFunction
from ...domain._scalar import _AbstractScalarDomain
from ...domain._structure import CoordSeparableBatch, PointsBatch, QuadratureBatch


def _first_field_leaf(tree: PyTree[Any]) -> cx.Field:
    leaves = jax.tree_util.tree_leaves(tree, is_leaf=lambda x: isinstance(x, cx.Field))
    for leaf in leaves:
        if isinstance(leaf, cx.Field):
            return leaf
    raise ValueError("Expected at least one coordax.Field leaf.")


def _as_float_mask(mask: cx.Field, /) -> cx.Field:
    data = jnp.asarray(mask.data)
    if data.dtype == jnp.bool_:
        return cx.Field(data.astype(float), dims=mask.dims)
    return cx.Field(data, dims=mask.dims)


def _sum_over(field: cx.Field, axis: str, /) -> cx.Field:
    if axis not in field.named_dims:
        raise ValueError(
            f"Cannot sum over missing axis {axis!r} in Field.dims={field.dims}."
        )
    pos = field.dims.index(axis)
    data = jnp.sum(jnp.asarray(field.data), axis=pos)
    dims = field.dims[:pos] + field.dims[pos + 1 :]
    return cx.Field(data, dims=dims)


def _axes_for_over(structure, over: str | tuple[str, ...] | None) -> tuple[str, ...]:
    axis_names = structure.axis_names
    if axis_names is None:
        raise ValueError("PointsBatch.structure must be canonicalized (axis_names set).")

    if over is None:
        return tuple(axis_names)

    if isinstance(over, str):
        axis = structure.axis_for(over)
        if axis is None:
            raise ValueError(f"Cannot integrate over fixed label {over!r}.")
        block = next((b for b in structure.blocks if over in b), None)
        if block is None:
            raise ValueError(f"Label {over!r} not found in ProductStructure.blocks.")
        if len(block) != 1:
            raise ValueError(
                f"over={over!r} does not align with a singleton block; use over={tuple(block)!r}."
            )
        return (axis,)

    req = frozenset(over)
    for block, axis in zip(structure.blocks, axis_names, strict=True):
        if frozenset(block) == req:
            return (axis,)
    raise ValueError(f"over={tuple(over)!r} does not match any ProductStructure block.")


def _label_measure(component: DomainComponent, label: str, /) -> Array:
    comp = component.spec.component_for(label)
    factor = component.domain.factor(label)
    if isinstance(factor, RelabeledDomain):
        factor = factor.base

    if isinstance(factor, _AbstractGeometry):
        from ...domain._components import Boundary, Fixed, Interior

        if isinstance(comp, Interior):
            return jnp.asarray(factor.volume, dtype=float)
        if isinstance(comp, Boundary):
            return jnp.asarray(factor.boundary_measure_value, dtype=float)
        if isinstance(comp, Fixed):
            return jnp.array(1.0, dtype=float)
        raise TypeError(f"Unsupported geometry component {type(comp).__name__}.")

    if isinstance(factor, _AbstractScalarDomain):
        from ...domain._components import Boundary, Fixed, FixedEnd, FixedStart, Interior

        if isinstance(comp, Interior):
            return jnp.asarray(factor.measure, dtype=float)
        if isinstance(comp, Boundary):
            return jnp.array(2.0, dtype=float)
        if isinstance(comp, (FixedStart, FixedEnd, Fixed)):
            return jnp.array(1.0, dtype=float)
        raise TypeError(f"Unsupported scalar component {type(comp).__name__}.")

    from ...domain._dataset import DatasetDomain

    if isinstance(factor, DatasetDomain):
        from ...domain._components import Interior

        if isinstance(comp, Interior):
            return jnp.asarray(factor.measure, dtype=float)
        raise TypeError(f"Unsupported dataset component {type(comp).__name__}.")

    raise TypeError(f"Unsupported unary domain type {type(factor).__name__}.")


def _default_quadrature_total_weight(
    component: DomainComponent, batch: PointsBatch, /
) -> cx.Field:
    if batch.structure.axis_names is None:
        raise ValueError("PointsBatch.structure must be canonicalized (axis_names set).")

    w_total = cx.Field(jnp.array(1.0, dtype=float), dims=())
    for block, axis in zip(
        batch.structure.blocks, batch.structure.axis_names, strict=True
    ):
        block_measure = jnp.array(1.0, dtype=float)
        for lbl in block:
            block_measure = block_measure * _label_measure(component, lbl)

        ref_label = block[0]
        field = _first_field_leaf(batch[ref_label])
        if axis not in field.named_shape:
            raise ValueError(
                f"Cannot infer size for sampling axis {axis!r} from points[{ref_label!r}]."
            )
        n = int(field.named_shape[axis])
        w_axis = cx.Field(
            jnp.full((n,), block_measure / float(n), dtype=float), dims=(axis,)
        )
        w_total = w_total * w_axis
    return w_total


def build_quadrature(
    component: DomainComponent, batch: PointsBatch, /
) -> QuadratureBatch:
    r"""Build a uniform tensor-product quadrature for a `PointsBatch`.

    For each sampling axis $a$ with $n_a$ points, this constructs weights

    $$
    w_a = \frac{\mu_a}{n_a},
    $$

    where $\mu_a$ is the measure associated with the labels in the corresponding
    sampling block (as determined by `component`). The full weight is the product
    $w=\prod_a w_a$.

    **Arguments:**

    - `component`: The `DomainComponent` whose measure is being integrated.
    - `batch`: A `PointsBatch` sampled with a canonicalized `ProductStructure`.

    **Returns:**

    - A `QuadratureBatch` with per-axis weights compatible with `batch`.
    """
    if batch.structure.axis_names is None:
        raise ValueError("PointsBatch.structure must be canonicalized (axis_names set).")

    weights_by_axis: dict[str, cx.Field] = {}
    for block, axis in zip(
        batch.structure.blocks, batch.structure.axis_names, strict=True
    ):
        block_measure = jnp.array(1.0, dtype=float)
        for lbl in block:
            block_measure = block_measure * _label_measure(component, lbl)

        ref_label = block[0]
        field = _first_field_leaf(batch[ref_label])
        if axis not in field.named_shape:
            raise ValueError(
                f"Cannot infer size for sampling axis {axis!r} from points[{ref_label!r}]."
            )
        n = int(field.named_shape[axis])
        weights_by_axis[axis] = cx.Field(
            jnp.full((n,), block_measure / float(n), dtype=float), dims=(axis,)
        )
    return QuadratureBatch(batch, weights_by_axis=weights_by_axis)


def _default_quadrature_total_weight_coord_separable(
    component: DomainComponent, batch: CoordSeparableBatch, /
) -> cx.Field:
    if batch.dense_structure.axis_names is None:
        raise ValueError(
            "CoordSeparableBatch.dense_structure must be canonicalized (axis_names set)."
        )

    w_total = cx.Field(jnp.array(1.0, dtype=float), dims=())

    # Coord-separable geometry labels: per-axis AABB lengths.
    for lbl, axes in batch.coord_axes_by_label.items():
        factor = component.domain.factor(lbl)
        if isinstance(factor, RelabeledDomain):
            factor = factor.base
        if not isinstance(factor, _AbstractGeometry):
            raise TypeError(
                "coord-separable quadrature requires a geometry label; "
                f"got {lbl!r} with factor {type(factor).__name__}."
            )
        bounds = jnp.asarray(factor.mesh_bounds, dtype=float)
        lengths = bounds[1] - bounds[0]
        if len(axes) != int(factor.var_dim):
            raise ValueError(
                f"coord-separable axis count mismatch for {lbl!r}: got {len(axes)} axes "
                f"but geometry var_dim={int(factor.var_dim)}."
            )
        for i, axis in enumerate(axes):
            field = _first_field_leaf(batch.points[lbl][i])
            n = int(field.data.shape[0])
            disc = batch.axis_discretization_by_axis.get(axis)
            if disc is not None and disc.quad_weights is not None:
                w_axis = cx.Field(jnp.asarray(disc.quad_weights), dims=(axis,))
            else:
                w_axis = cx.Field(
                    jnp.full((n,), lengths[i] / float(n), dtype=float), dims=(axis,)
                )
            w_total = w_total * w_axis

    # Dense blocks: same as paired sampling weights.
    for block, axis in zip(
        batch.dense_structure.blocks, batch.dense_structure.axis_names, strict=True
    ):
        block_measure = jnp.array(1.0, dtype=float)
        for lbl in block:
            block_measure = block_measure * _label_measure(component, lbl)

        ref_label = block[0]
        field = _first_field_leaf(batch[ref_label])
        if axis not in field.named_shape:
            raise ValueError(
                f"Cannot infer size for sampling axis {axis!r} from points[{ref_label!r}]."
            )
        n = int(field.named_shape[axis])
        w_axis = cx.Field(
            jnp.full((n,), block_measure / float(n), dtype=float), dims=(axis,)
        )
        w_total = w_total * w_axis

    return w_total


def _where_product(
    component: DomainComponent,
    batch: PointsBatch | CoordSeparableBatch,
    /,
    *,
    key: Key[Array, ""],
    **kwargs: Any,
) -> cx.Field:
    m = cx.Field(jnp.array(1.0, dtype=float), dims=())

    if isinstance(batch, CoordSeparableBatch):
        for mask in batch.coord_mask_by_label.values():
            m = m * _as_float_mask(mask)

    for lbl, where_fn in component.where.items():
        if isinstance(batch, CoordSeparableBatch) and (lbl in batch.coord_axes_by_label):
            continue
        wrapped = _ensure_special_kwonly_args(where_fn)
        out = cx.cmap(wrapped, out_axes="leading")(batch[lbl], key=key)
        if not isinstance(out, cx.Field):
            raise TypeError("Per-label where must return a coordax.Field.")
        m = m * _as_float_mask(out)

    if component.where_all is not None:
        out = component.where_all(batch, key=key, **kwargs)
        m = m * _as_float_mask(out)

    return m


def _weight_product(
    component: DomainComponent,
    batch: PointsBatch | CoordSeparableBatch,
    /,
    *,
    key: Key[Array, ""],
    **kwargs: Any,
) -> cx.Field:
    w = cx.Field(jnp.array(1.0, dtype=float), dims=())
    if component.weight_all is None:
        return w
    out = component.weight_all(batch, key=key, **kwargs)
    if not isinstance(out, cx.Field):
        raise TypeError("weight_all must return a coordax.Field.")
    return w * out


def integral(
    f: DomainFunction | ArrayLike | None,
    batch: PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...],
    /,
    *,
    component: DomainComponent | DomainComponentUnion,
    quadrature: QuadratureBatch | tuple[QuadratureBatch | None, ...] | None = None,
    over: str | tuple[str, ...] | None = None,
    key: Key[Array, ""] = DOC_KEY0,
    **kwargs: Any,
) -> cx.Field:
    r"""Estimate an integral over a `DomainComponent`.

    Given an integrand $f$ and a component $\Omega_{\text{comp}}$ with measure
    $\mu$, this computes a Monte Carlo / quadrature estimate of

    $$
    \int_{\Omega_{\text{comp}}} f(z)\,d\mu(z).
    $$

    Sampling structure is provided by `batch`:
    - `PointsBatch`: paired sampling according to a `ProductStructure`;
    - `CoordSeparableBatch`: coordinate-separable sampling for some geometry labels.

    Filtering and weighting:
    - `component.where` / `component.where_all` act as indicator functions;
    - `component.weight_all` multiplies the integrand.

    **Arguments:**

    - `f`: Integrand as a `DomainFunction` or array-like constant.
    - `batch`: Sampled points (or a tuple of batches for `DomainComponentUnion`).
    - `component`: Component (or union of components) to integrate over.
    - `quadrature`: Optional explicit per-axis weights (`QuadratureBatch` for `PointsBatch` only).
    - `over`: Which label/block to integrate over; `None` integrates over all axes implied by `batch`.
    - `key`: PRNG key forwarded to `where`/`weight` callables and `f` (when needed).
    - `kwargs`: Extra keyword arguments forwarded to `f` and component callables.

    **Returns:**

    - A `coordax.Field` containing the reduced integral value, with remaining named axes
      corresponding to any non-integrated sampling axes.
    """
    if isinstance(component, DomainComponentUnion):
        if not isinstance(batch, tuple):
            raise TypeError(
                "For DomainComponentUnion, batch must be a tuple[PointsBatch, ...]."
            )
        if len(batch) != len(component.terms):
            raise ValueError("batch must align with component.terms.")

        if quadrature is None or isinstance(quadrature, QuadratureBatch):
            quad_terms = (quadrature,) * len(component.terms)
        else:
            quad_terms = tuple(quadrature)
            if len(quad_terms) != len(component.terms):
                raise ValueError("quadrature must align with component.terms.")

        out = cx.Field(jnp.array(0.0, dtype=float), dims=())
        keys = jr.split(key, len(component.terms))
        for term, b, q, k in zip(component.terms, batch, quad_terms, keys, strict=True):
            out = out + integral(
                f, b, component=term, quadrature=q, over=over, key=k, **kwargs
            )
        return out

    if isinstance(batch, tuple):
        raise TypeError(
            "For DomainComponent, batch must be a PointsBatch or CoordSeparableBatch."
        )
    if not isinstance(batch, (PointsBatch, CoordSeparableBatch)):
        raise TypeError(
            "For DomainComponent, batch must be a PointsBatch or CoordSeparableBatch."
        )
    batch_single = batch

    fn = (
        f
        if isinstance(f, DomainFunction)
        else DomainFunction(domain=component.domain, deps=(), func=f)
    )
    for lbl in fn.domain.labels:
        if lbl not in batch_single:
            raise KeyError(
                f"Cannot evaluate integrand: missing label {lbl!r} in sampled points."
            )

    y = fn(batch_single, key=key, **kwargs)
    if not isinstance(y, cx.Field):
        raise TypeError("integrand must evaluate to a coordax.Field.")

    if isinstance(batch_single, PointsBatch):
        axes = _axes_for_over(batch_single.structure, over)
    else:
        if over is None:
            dense_axis_names = batch_single.dense_structure.axis_names
            if dense_axis_names is None:
                raise ValueError(
                    "CoordSeparableBatch.dense_structure must be canonicalized (axis_names set)."
                )
            axes = tuple(
                ax
                for lbl in component.domain.labels
                for ax in batch_single.coord_axes_by_label.get(lbl, ())
            ) + tuple(dense_axis_names)
        elif isinstance(over, str):
            if over in batch_single.coord_axes_by_label:
                axes = batch_single.coord_axes_by_label[over]
            else:
                axes = _axes_for_over(batch_single.dense_structure, over)
        else:
            axes = _axes_for_over(batch_single.dense_structure, over)

    if quadrature is not None:
        if isinstance(quadrature, QuadratureBatch):
            if isinstance(batch_single, CoordSeparableBatch):
                raise ValueError(
                    "QuadratureBatch is not yet supported for CoordSeparableBatch."
                )
            q = quadrature
        else:
            raise TypeError("quadrature must be a QuadratureBatch or None.")
        if q.batch.structure.axis_names != batch_single.structure.axis_names:
            raise ValueError("quadrature.batch must match batch structure.")
        wq = q.total_weight()
        axis_names = batch_single.structure.axis_names
        if axis_names is None:
            raise ValueError(
                "PointsBatch.structure must be canonicalized (axis_names set)."
            )
        missing = [ax for ax in axis_names if ax not in wq.named_dims]
        if missing:
            raise ValueError(
                f"QuadratureBatch missing weights for axes {tuple(missing)!r}."
            )
    else:
        if isinstance(batch_single, PointsBatch):
            wq = _default_quadrature_total_weight(component, batch_single)
        else:
            wq = _default_quadrature_total_weight_coord_separable(component, batch_single)

    m = _where_product(component, batch_single, key=key, **kwargs)
    w_sel = _weight_product(component, batch_single, key=key, **kwargs)

    acc = wq * m * w_sel * y
    for axis in axes:
        acc = _sum_over(acc, axis)
    return acc


def mean(
    f: DomainFunction | ArrayLike | None,
    batch: PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...],
    /,
    *,
    component: DomainComponent | DomainComponentUnion,
    quadrature: QuadratureBatch | tuple[QuadratureBatch | None, ...] | None = None,
    over: str | tuple[str, ...] | None = None,
    key: Key[Array, ""] = DOC_KEY0,
    **kwargs: Any,
) -> cx.Field:
    r"""Estimate the mean value of an integrand over a component.

    Computes

    $$
    \frac{\int_{\Omega_{\text{comp}}} f(z)\,d\mu(z)}
         {\int_{\Omega_{\text{comp}}} 1\,d\mu(z)}.
    $$

    **Arguments:**

    - `f`, `batch`, `component`, `quadrature`, `over`, `key`, `kwargs`: As in `integral`.

    **Returns:**

    - A `coordax.Field` containing the mean value.
    """
    num = integral(
        f, batch, component=component, quadrature=quadrature, over=over, key=key, **kwargs
    )

    if isinstance(component, DomainComponentUnion):
        if not isinstance(batch, tuple):
            raise TypeError(
                "For DomainComponentUnion, batch must be a tuple[PointsBatch, ...]."
            )
        if quadrature is None or isinstance(quadrature, QuadratureBatch):
            quad_terms = (quadrature,) * len(component.terms)
        else:
            quad_terms = tuple(quadrature)
            if len(quad_terms) != len(component.terms):
                raise ValueError("quadrature must align with component.terms.")
        keys = jr.split(key, len(component.terms))
        den = cx.Field(jnp.array(0.0, dtype=float), dims=())
        for term, b, q, k in zip(component.terms, batch, quad_terms, keys, strict=True):
            den = den + integral(
                1.0, b, component=term, quadrature=q, over=over, key=k, **kwargs
            )
        den_checked = eqx.error_if(den.data, den.data == 0, "mean denominator is zero.")
        return num / cx.Field(den_checked, dims=den.dims)

    den = integral(
        1.0,
        batch,
        component=component,
        quadrature=quadrature,
        over=over,
        key=key,
        **kwargs,
    )
    den_checked = eqx.error_if(den.data, den.data == 0, "mean denominator is zero.")
    return num / cx.Field(den_checked, dims=den.dims)


def integrate_interior(
    f: DomainFunction | ArrayLike | None,
    batch: PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...],
    /,
    *,
    component: DomainComponent | DomainComponentUnion,
    quadrature: QuadratureBatch | tuple[QuadratureBatch | None, ...] | None = None,
    over: str | tuple[str, ...] | None = None,
    key: Key[Array, ""] = DOC_KEY0,
    **kwargs: Any,
) -> cx.Field:
    r"""Alias for `integral`.

    Interior/boundary semantics are encoded by `component` (e.g. its `ComponentSpec`).
    """
    return integral(
        f, batch, component=component, quadrature=quadrature, over=over, key=key, **kwargs
    )


def integrate_boundary(
    f: DomainFunction | ArrayLike | None,
    batch: PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...],
    /,
    *,
    component: DomainComponent | DomainComponentUnion,
    quadrature: QuadratureBatch | tuple[QuadratureBatch | None, ...] | None = None,
    over: str | tuple[str, ...] | None = None,
    key: Key[Array, ""] = DOC_KEY0,
    **kwargs: Any,
) -> cx.Field:
    r"""Alias for `integral`.

    Interior/boundary semantics are encoded by `component` (e.g. its `ComponentSpec`).
    """
    return integral(
        f, batch, component=component, quadrature=quadrature, over=over, key=key, **kwargs
    )
