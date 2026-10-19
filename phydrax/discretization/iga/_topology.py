#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import checked
from .._topology import TensorTopology
from . import _basis as basis_owner
from ._identity import BaseSpanId, InterfaceId


class SplineSpanTopology(StrictModule, NonTrainableState):
    """Positive-span topology with immutable host routes and no coefficient sites."""

    tensor_topology: TensorTopology
    axis_names: tuple[str, ...] = eqx.field(static=True)
    axis_sizes: tuple[int, ...] = eqx.field(static=True)
    active_mask: np.ndarray
    topology_id: str = eqx.field(static=True)
    span_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    span_ids: tuple[BaseSpanId, ...]
    patch_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_names: Sequence[str] | basis_owner.TensorSplineBasisSpec,
        span_indices: Sequence[ArrayLike] | None = None,
        /,
        *,
        patch_id: str = "patch",
    ) -> None:

        match axis_names:
            case basis_owner.TensorSplineBasisSpec() as basis:
                names = tuple(str(name) for name in basis.axis_names)
                indices = tuple(np.asarray(axis.span_indices) for axis in basis.axes)
            case _:
                if span_indices is None:
                    raise TypeError(
                        "SplineSpanTopology.span_indices must accompany axis names; got None"
                    )
                names = tuple(str(name) for name in axis_names)
                indices = tuple(np.asarray(value) for value in span_indices)
        patch = str(patch_id)
        if not patch:
            raise ValueError(
                f"SplineSpanTopology.patch_id must be non-empty; got {patch!r}"
            )
        if len(names) != len(indices) or not names:
            raise ValueError(
                f"SplineSpanTopology.span_indices must have one array per nonempty axis set; got names={len(names)}, indices={len(indices)}"
            )
        normalized = []
        for value in indices:
            if (
                value.ndim != 1
                or not np.issubdtype(value.dtype, np.integer)
                or value.size == 0
            ):
                raise ValueError(
                    f"SplineSpanTopology.span_indices must be nonempty rank-1 integer arrays; got shape={value.shape}, dtype={value.dtype}"
                )
            value = value.astype(np.int32, copy=False)
            if np.any(value < 0) or np.any(np.diff(value) <= 0):
                raise ValueError(
                    f"SplineSpanTopology.span_indices must be strictly increasing and nonnegative; got {value}"
                )
            normalized.append(value)
        shape = tuple(value.size for value in normalized)
        topology_id = canonical_fingerprint(
            {
                "kind": "iga-spline-span-topology",
                "patch": patch,
                "axis_names": list(names),
                "positive_spans": [array_tree_fingerprint(value) for value in normalized],
            }
        )
        self.axis_names = names
        self.axis_sizes = shape
        self.active_mask = np.ones(shape, dtype=np.bool_)
        self.topology_id = topology_id
        self.tensor_topology = TensorTopology(
            names,
            shape,
            active_mask=self.active_mask,
            topology_id=topology_id,
        )
        routes = tuple(np.ndindex(shape))
        self.span_indices = tuple(
            tuple(int(index) for index in value) for value in normalized
        )
        self.patch_id = patch
        self.span_ids = tuple(BaseSpanId(patch, route) for route in routes)

    @property
    def span_shape(self) -> tuple[int, ...]:
        return self.axis_sizes

    @property
    def cell_count(self) -> int:
        return prod(self.axis_sizes)

    def span_id(self, row: int, /) -> BaseSpanId:
        row_ = int(row)
        if not 0 <= row_ < self.cell_count:
            raise IndexError(
                f"SplineSpanTopology.row must be within the cell range; got row={row_}, cells={self.cell_count}"
            )
        return self.span_ids[row_]

    def span_route(self, row: int, /) -> tuple[int, ...]:
        self.span_id(row)
        return tuple(int(index) for index in np.unravel_index(int(row), self.axis_sizes))


class PatchAtlas(StrictModule, NonTrainableState):
    """Named fixed positive-span patch topologies with content identity."""

    patch_ids: tuple[str, ...] = eqx.field(static=True)
    topologies: tuple[SplineSpanTopology, ...]
    atlas_id: str = eqx.field(static=True)

    def __init__(
        self, patches: Mapping[str, SplineSpanTopology] | Sequence[SplineSpanTopology], /
    ) -> None:
        match patches:
            case Mapping():
                items = tuple((str(name), topology) for name, topology in patches.items())
            case _:
                values = tuple(patches)
                items = tuple((topology.patch_id, topology) for topology in values)
        if not items or any(
            not isinstance(topology, SplineSpanTopology) for _, topology in items
        ):
            raise TypeError(
                "PatchAtlas.patches must contain one or more SplineSpanTopology values; got empty or untyped topologies"
            )
        names = tuple(name for name, _ in items)
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError(
                f"PatchAtlas.patch_ids must be nonempty and unique; got {names!r}"
            )
        if any(name != topology.patch_id for name, topology in items):
            raise ValueError(
                f"PatchAtlas.patches must use keys matching topology patch IDs; got {names!r}"
            )
        self.patch_ids = names
        self.topologies = tuple(topology for _, topology in items)
        self.atlas_id = canonical_fingerprint(
            {
                "kind": "iga-patch-atlas",
                "patches": [
                    {"id": name, "topology": topology.topology_id}
                    for name, topology in items
                ],
            }
        )

    def topology(self, patch_id: str, /) -> SplineSpanTopology:
        patch = str(patch_id)
        for name, topology in zip(self.patch_ids, self.topologies, strict=True):
            if name == patch:
                return topology
        raise KeyError(f"PatchAtlas.patch_id must name an atlas patch; got {patch!r}")


class CellComplex(StrictModule, NonTrainableState):
    """Closed collection of span patches and explicitly declared interfaces."""

    atlas: PatchAtlas
    interfaces: tuple[InterfaceId, ...]
    complex_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, atlas: PatchAtlas, interfaces: Sequence[InterfaceId] = (), /
    ) -> None:
        interface_values = tuple(interfaces)
        if not all(isinstance(interface, InterfaceId) for interface in interface_values):
            raise TypeError(
                "CellComplex.interfaces must contain InterfaceId values; got untyped interface"
            )
        values = tuple(interface.value for interface in interface_values)
        if len(set(values)) != len(values):
            raise ValueError(
                f"CellComplex.interfaces must have unique identities; got {values!r}"
            )
        patches = set(atlas.patch_ids)
        if any(
            interface.left_patch_id not in patches
            or interface.right_patch_id not in patches
            for interface in interface_values
        ):
            raise ValueError(
                "CellComplex.interfaces must refer to atlas patches; got unknown patch endpoint"
            )
        self.atlas = atlas
        self.interfaces = interface_values
        self.complex_id = canonical_fingerprint(
            {
                "kind": "iga-cell-complex",
                "atlas": atlas.atlas_id,
                "interfaces": list(values),
            }
        )


__all__ = ["CellComplex", "PatchAtlas", "SplineSpanTopology"]
