#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import re
from collections.abc import Mapping
from typing import TypeAlias

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from .._frozendict import frozendict
from .._strict import StrictModule
from ._grid import AxisDiscretization


_AXIS_PREFIX = "__phydra_blk__"
_SEP_AXIS_PREFIX = "__phydra_sep__"
_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


Points: TypeAlias = frozendict[str, PyTree[cx.Field]]
NumPoints: TypeAlias = int | tuple[int, ...]


def _validate_label(label: str) -> None:
    if not _LABEL_RE.fullmatch(label):
        raise ValueError(
            f"Invalid domain label {label!r}. Labels must match {_LABEL_RE.pattern!r}."
        )


def _axis_name_for_block(block: tuple[str, ...]) -> str:
    return _AXIS_PREFIX + "__".join(block)


def _axis_name_for_coord(label: str, axis: int, /) -> str:
    _validate_label(label)
    return f"{_SEP_AXIS_PREFIX}{label}__{int(axis)}"


def _validate_reserved_axes(points: Points, *, allowed_axes: frozenset[str]) -> None:
    leaves = jax.tree_util.tree_leaves(points, is_leaf=lambda x: isinstance(x, cx.Field))
    for leaf in leaves:
        if not isinstance(leaf, cx.Field):
            continue
        for dim in leaf.named_dims:
            if dim is None:
                continue
            if dim.startswith(_AXIS_PREFIX):
                if dim not in allowed_axes:
                    raise ValueError(
                        "Found reserved sampling-axis dim not declared in ProductStructure: "
                        f"{dim!r}. Allowed: {tuple(sorted(allowed_axes))!r}."
                    )


def _validate_reserved_sep_axes(points: Points, *, allowed_axes: frozenset[str]) -> None:
    leaves = jax.tree_util.tree_leaves(points, is_leaf=lambda x: isinstance(x, cx.Field))
    for leaf in leaves:
        if not isinstance(leaf, cx.Field):
            continue
        for dim in leaf.named_dims:
            if dim is None:
                continue
            if dim.startswith(_SEP_AXIS_PREFIX):
                if dim not in allowed_axes:
                    raise ValueError(
                        "Found reserved coord-separable dim not declared in CoordSeparableBatch: "
                        f"{dim!r}. Allowed: {tuple(sorted(allowed_axes))!r}."
                    )


class ProductStructure(StrictModule):
    r"""Describes how a product domain is sampled into named axes.

    Phydrax uses *labeled* product domains, e.g. $\Omega = \Omega_x \times \Omega_t$
    with labels `"x"` and `"t"`. When sampling points from a `DomainComponent`,
    we often want to control which variables are sampled *jointly* and which are
    sampled in separate blocks.

    A `ProductStructure` defines a partition of the (non-fixed) labels into blocks
    `(B_1, \dots, B_k)`. For each block $B_j$ we sample $n_j$ joint points in
    $\prod_{\ell\in B_j}\Omega_\ell$. Each block corresponds to one named sampling
    axis (e.g. `__phydra_blk__x__t`), and values evaluated on the resulting
    `PointsBatch` are `coordax.Field`s carrying those axis names.

    For example:

    - `ProductStructure((("x", "t"),))` samples paired space-time points.
    - `ProductStructure((("x",), ("t",)))` samples space and time independently,
      producing a Cartesian product grid in evaluation.
    """

    blocks: tuple[tuple[str, ...], ...]
    axis_names: tuple[str, ...] | None = None

    def __init__(
        self,
        blocks: tuple[tuple[str, ...], ...],
        *,
        axis_names: tuple[str, ...] | None = None,
    ):
        for block in blocks:
            for label in block:
                _validate_label(label)
        if axis_names is not None:
            if len(axis_names) != len(blocks):
                raise ValueError("axis_names must have the same length as blocks.")
            for name in axis_names:
                if not name.startswith(_AXIS_PREFIX):
                    raise ValueError(
                        f"Invalid axis name {name!r}. Must start with {_AXIS_PREFIX!r}."
                    )
        self.blocks = blocks
        self.axis_names = axis_names

    def canonicalize(
        self,
        domain_labels: tuple[str, ...],
        *,
        fixed_labels: frozenset[str] = frozenset(),
    ) -> "ProductStructure":
        r"""Return a structure with blocks ordered canonically for a domain.

        Canonicalization enforces:

        - Every non-fixed domain label appears in exactly one block.
        - Each block is sorted in the order it appears in `domain_labels`.
        - `axis_names` is set (generated if missing).

        This is required before constructing a `PointsBatch`, since named axes are
        derived from the canonical blocks.
        """
        for label in domain_labels:
            _validate_label(label)

        domain_pos = {lbl: i for i, lbl in enumerate(domain_labels)}
        seen: set[str] = set()
        canon_blocks: list[tuple[str, ...]] = []
        for block in self.blocks:
            block_set = set(block)
            if block_set & fixed_labels:
                fixed = sorted(block_set & fixed_labels)
                raise ValueError(
                    f"ProductStructure includes fixed labels {fixed}; fixed labels must not appear in blocks."
                )
            if len(block_set) != len(block):
                raise ValueError(f"Duplicate labels in block {block}.")
            if seen & block_set:
                dup = sorted(seen & block_set)
                raise ValueError(f"Label(s) {dup} appear in more than one block.")
            seen |= block_set
            try:
                canon = tuple(sorted(block_set, key=lambda s: domain_pos[s]))
            except KeyError as e:
                raise ValueError(
                    f"Unknown label {e.args[0]!r} in ProductStructure; expected subset of {domain_labels}."
                ) from None
            canon_blocks.append(canon)

        remaining = [
            lbl for lbl in domain_labels if (lbl not in seen and lbl not in fixed_labels)
        ]
        if remaining:
            raise ValueError(
                "ProductStructure does not cover all non-fixed labels; missing "
                f"{tuple(remaining)!r}."
            )

        if self.axis_names is None:
            axis_names = tuple(_axis_name_for_block(b) for b in canon_blocks)
        else:
            axis_names = self.axis_names
        return ProductStructure(tuple(canon_blocks), axis_names=axis_names)

    def axis_for(self, label: str) -> str | None:
        """Return the sampling axis name corresponding to `label` (or `None` if fixed)."""
        if self.axis_names is None:
            raise ValueError(
                "ProductStructure.axis_for requires axis_names to be set (call canonicalize first)."
            )
        for block, name in zip(self.blocks, self.axis_names, strict=True):
            if label in block:
                return name
        return None


class PointsBatch(StrictModule, Mapping[str, PyTree[cx.Field]]):
    r"""A labeled batch of sampled points with explicit axis semantics.

    A `PointsBatch` is a mapping `{label: coordax.Field}` paired with a canonicalized
    `ProductStructure`. Each label's point array is stored as a `coordax.Field`
    whose named axes correspond to the sampling block(s) that include that label.

    If a label is in a block with axis name `a`, then its sampled points carry a named
    dimension `a` of length equal to the number of samples in that block.
    """

    points: Points
    structure: ProductStructure

    def __init__(self, points: Points, structure: ProductStructure):
        """Construct a `PointsBatch` from sampled points and a canonical structure."""
        if structure.axis_names is None:
            raise ValueError(
                "PointsBatch requires a canonicalized ProductStructure (axis_names set)."
            )
        _validate_reserved_axes(points, allowed_axes=frozenset(structure.axis_names))
        self.points = points
        self.structure = structure

    def __getitem__(self, key: str) -> PyTree[cx.Field]:
        return self.points[key]

    def __iter__(self):
        return iter(self.points)

    def __len__(self) -> int:
        return len(self.points)


class QuadratureBatch(StrictModule):
    r"""Quadrature weights associated with a `PointsBatch`.

    A `QuadratureBatch` stores one 1D weight field per sampling axis. The total
    tensor-product weight is

    $$
    w(z) = \prod_{a \in \mathcal{A}} w_a,
    $$

    where each $w_a$ is a `coordax.Field` with dims `(a,)`. These weights are used by
    integral estimators such as `phydrax.operators.integral`.
    """

    batch: PointsBatch
    weights_by_axis: frozendict[str, cx.Field]

    def __init__(
        self,
        batch: PointsBatch,
        *,
        weights_by_axis: frozendict[str, cx.Field] | Mapping[str, cx.Field],
    ):
        weights = frozendict(weights_by_axis)
        axis_names = batch.structure.axis_names
        if axis_names is None:
            raise ValueError(
                "QuadratureBatch requires a canonicalized ProductStructure (axis_names set)."
            )
        for axis, w in weights.items():
            if axis not in axis_names:
                raise ValueError(
                    "Quadrature weights provided for unknown axis "
                    f"{axis!r}; expected one of {axis_names}."
                )
            if not isinstance(w, cx.Field):
                raise TypeError(
                    f"Quadrature weight for axis {axis!r} must be a coordax.Field."
                )
            if w.dims != (axis,):
                raise ValueError(
                    f"Quadrature weight for axis {axis!r} must have dims ({axis!r},), got {w.dims}."
                )
        self.batch = batch
        self.weights_by_axis = weights

    def total_weight(self) -> cx.Field:
        r"""Return the product weight field.

        If `weights_by_axis = {a: w_a}`, returns $w=\prod_a w_a$ as a `coordax.Field`.
        """
        w_total = cx.Field(jnp.array(1.0, dtype=float), dims=())
        for w in self.weights_by_axis.values():
            w_total = w_total * w
        return w_total


class CoordSeparableBatch(StrictModule, Mapping[str, PyTree[cx.Field]]):
    r"""A batch that separates coordinate axes for selected geometry labels.

    For some geometries it is efficient to sample each coordinate axis independently,
    e.g. draw $(x_1,\dots,x_{n_1})$ and $(y_1,\dots,y_{n_2})$ separately and evaluate
    on the implied grid.

    A `CoordSeparableBatch` stores:
    - `coord_axes_by_label`: which named axes correspond to each coordinate component,
      e.g. `("x0", "x1")` for a 2D geometry label.
    - `coord_mask_by_label`: a mask `coordax.Field` that can be used to exclude
      coordinate combinations outside an irregular geometry (e.g. AABB grid masking).
    - `dense_structure`: a normal `ProductStructure` for any remaining (non-separable)
      labels sampled in paired blocks.
    """

    points: Points
    dense_structure: ProductStructure
    coord_axes_by_label: frozendict[str, tuple[str, ...]]
    coord_mask_by_label: frozendict[str, cx.Field]
    axis_discretization_by_axis: frozendict[str, AxisDiscretization]

    def __init__(
        self,
        points: Points,
        *,
        dense_structure: ProductStructure,
        coord_axes_by_label: frozendict[str, tuple[str, ...]]
        | Mapping[str, tuple[str, ...]],
        coord_mask_by_label: frozendict[str, cx.Field] | Mapping[str, cx.Field],
        axis_discretization_by_axis: frozendict[str, AxisDiscretization]
        | Mapping[str, AxisDiscretization]
        | None = None,
    ):
        """Construct a coordinate-separable batch of points."""
        if dense_structure.axis_names is None:
            raise ValueError(
                "CoordSeparableBatch requires a canonicalized dense_structure (axis_names set)."
            )

        axes_by_label = frozendict(coord_axes_by_label)
        mask_by_label = frozendict(coord_mask_by_label)
        disc_by_axis = frozendict(axis_discretization_by_axis or {})

        allowed_sep: set[str] = set()
        for lbl, axes in axes_by_label.items():
            _validate_label(lbl)
            for ax in axes:
                if not ax.startswith(_SEP_AXIS_PREFIX):
                    raise ValueError(
                        f"Invalid coord-separable axis name {ax!r}. Must start with {_SEP_AXIS_PREFIX!r}."
                    )
                allowed_sep.add(ax)

            if lbl not in points:
                raise KeyError(f"Missing coord-separable label {lbl!r} in points.")
            if lbl not in mask_by_label:
                raise KeyError(f"Missing coord-separable mask for label {lbl!r}.")

            x = points[lbl]
            if not isinstance(x, tuple):
                raise TypeError(
                    f"CoordSeparableBatch expects points[{lbl!r}] to be a tuple of coordax.Field axes."
                )
            if len(x) != len(axes):
                raise ValueError(
                    f"CoordSeparableBatch points[{lbl!r}] has {len(x)} axis arrays "
                    f"but coord_axes_by_label declares {len(axes)}."
                )
            for field, ax_name in zip(x, axes, strict=True):
                if not isinstance(field, cx.Field):
                    raise TypeError(
                        f"CoordSeparableBatch expects points[{lbl!r}] entries to be coordax.Field."
                    )
                if field.dims != (ax_name,):
                    raise ValueError(
                        f"CoordSeparableBatch expects points[{lbl!r}] axis to have dims "
                        f"({ax_name!r},), got {field.dims}."
                    )

            mask = mask_by_label[lbl]
            if not isinstance(mask, cx.Field):
                raise TypeError(
                    f"CoordSeparableBatch mask for {lbl!r} must be a coordax.Field."
                )
            if mask.dims != axes:
                raise ValueError(
                    f"CoordSeparableBatch mask for {lbl!r} must have dims {axes}, got {mask.dims}."
                )

        for axis, disc in disc_by_axis.items():
            if axis not in allowed_sep:
                raise ValueError(
                    "Found discretization for unknown coord-separable axis "
                    f"{axis!r}. Allowed: {tuple(sorted(allowed_sep))!r}."
                )
            if not isinstance(disc, AxisDiscretization):
                raise TypeError(
                    f"axis_discretization_by_axis[{axis!r}] must be an AxisDiscretization."
                )

        _validate_reserved_axes(
            points, allowed_axes=frozenset(dense_structure.axis_names)
        )
        _validate_reserved_sep_axes(points, allowed_axes=frozenset(allowed_sep))

        self.points = points
        self.dense_structure = dense_structure
        self.coord_axes_by_label = axes_by_label
        self.coord_mask_by_label = mask_by_label
        self.axis_discretization_by_axis = disc_by_axis

    def __getitem__(self, key: str) -> PyTree[cx.Field]:
        return self.points[key]

    def __iter__(self):
        return iter(self.points)

    def __len__(self) -> int:
        return len(self.points)


__all__ = [
    "Points",
    "NumPoints",
    "ProductStructure",
    "PointsBatch",
    "QuadratureBatch",
    "CoordSeparableBatch",
    "_axis_name_for_coord",
]
