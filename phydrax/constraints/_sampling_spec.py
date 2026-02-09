#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast, TypeAlias, TypeGuard

from ..domain._base import _AbstractGeometry
from ..domain._components import (
    DomainComponent,
    DomainComponentUnion,
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
)
from ..domain._domain import RelabeledDomain
from ..domain._grid import AbstractAxisSpec, GridSpec
from ..domain._scalar import _AbstractScalarDomain
from ..domain._structure import NumPoints, ProductStructure


CoordSamplingValue: TypeAlias = (
    int | Sequence[int] | AbstractAxisSpec | Sequence[AbstractAxisSpec] | GridSpec
)

CoordSamplingMap: TypeAlias = Mapping[str, CoordSamplingValue]
SamplingNumPoints: TypeAlias = (
    NumPoints | CoordSamplingMap | tuple[NumPoints, CoordSamplingMap]
)


def _is_coord_sampling_map(value: object, /) -> TypeGuard[CoordSamplingMap]:
    if not isinstance(value, Mapping):
        return False
    return all(isinstance(k, str) for k in value)


def _coerce_dense_num_points(value: Any, /) -> NumPoints:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, tuple):
        if all(isinstance(n, int) for n in value):
            return tuple(int(n) for n in value)
    raise TypeError(
        "Dense num_points must be int or tuple[int, ...]. "
        "For coord-separable sampling, pass num_points as a mapping or (dense_num_points, coord_map)."
    )


def _fixed_labels(component: DomainComponent, /) -> frozenset[str]:
    return frozenset(
        lbl
        for lbl in component.domain.labels
        if isinstance(component.spec.component_for(lbl), (FixedStart, FixedEnd, Fixed))
    )


def _unwrap_factor(factor: object, /) -> object:
    return factor.base if isinstance(factor, RelabeledDomain) else factor


def _validate_coord_sampling_labels(
    component: DomainComponent | DomainComponentUnion,
    coord_map: CoordSamplingMap,
    /,
) -> None:
    if isinstance(component, DomainComponentUnion):
        raise ValueError(
            "coord-separable sampling is not supported for DomainComponentUnion."
        )
    coord_labels = tuple(coord_map)
    for lbl in coord_labels:
        if lbl not in component.domain.labels:
            raise KeyError(f"Label {lbl!r} not in domain {component.domain.labels}.")
    fixed = _fixed_labels(component)
    bad_fixed = tuple(lbl for lbl in coord_labels if lbl in fixed)
    if bad_fixed:
        raise ValueError(
            f"coord-separable labels must not include fixed labels; got {bad_fixed!r}."
        )
    bad_components = tuple(
        lbl
        for lbl in coord_labels
        if not isinstance(component.spec.component_for(lbl), Interior)
    )
    if bad_components:
        raise ValueError(
            "coord-separable labels must use Interior() components; got "
            f"{bad_components!r}."
        )
    bad_factors = tuple(
        lbl
        for lbl in coord_labels
        if not isinstance(
            _unwrap_factor(component.domain.factor(lbl)),
            (_AbstractGeometry, _AbstractScalarDomain),
        )
    )
    if bad_factors:
        raise TypeError(
            "coord-separable sampling for constraints requires geometry/scalar labels; got "
            f"{bad_factors!r}."
        )


def _resolve_dense_structure_for_coord_sampling(
    component: DomainComponent,
    /,
    *,
    structure: ProductStructure,
    coord_labels: frozenset[str],
    dense_structure: ProductStructure | None,
) -> ProductStructure:
    if dense_structure is not None:
        return dense_structure

    fixed = _fixed_labels(component)
    dense_labels = tuple(
        lbl
        for lbl in component.domain.labels
        if lbl not in coord_labels and lbl not in fixed
    )
    if not dense_labels:
        return ProductStructure(blocks=())

    blocks: list[tuple[str, ...]] = []
    for block in structure.blocks:
        filtered = tuple(
            lbl for lbl in block if lbl not in coord_labels and lbl not in fixed
        )
        if filtered:
            blocks.append(filtered)
    covered = set(lbl for block in blocks for lbl in block)
    missing = tuple(lbl for lbl in dense_labels if lbl not in covered)
    if missing:
        raise ValueError(
            "coord-separable sampling requires dense_structure to cover non-separable labels; "
            f"missing {missing!r}."
        )
    return ProductStructure(tuple(blocks))


def parse_sampling_num_points(
    component: DomainComponent | DomainComponentUnion,
    /,
    *,
    num_points: SamplingNumPoints,
    structure: ProductStructure,
    dense_structure: ProductStructure | None,
) -> tuple[NumPoints, CoordSamplingMap | None, ProductStructure | None]:
    if _is_coord_sampling_map(num_points):
        coord_map = num_points
        _validate_coord_sampling_labels(component, coord_map)
        assert isinstance(component, DomainComponent)
        dense_structure_out = _resolve_dense_structure_for_coord_sampling(
            component,
            structure=structure,
            coord_labels=frozenset(coord_map),
            dense_structure=dense_structure,
        )
        empty_dense_num_points = cast(NumPoints, ())
        return empty_dense_num_points, coord_map, dense_structure_out

    if (
        isinstance(num_points, tuple)
        and len(num_points) == 2
        and _is_coord_sampling_map(num_points[1])
    ):
        dense_num_points = _coerce_dense_num_points(num_points[0])
        coord_map = num_points[1]
        _validate_coord_sampling_labels(component, coord_map)
        assert isinstance(component, DomainComponent)
        dense_structure_out = _resolve_dense_structure_for_coord_sampling(
            component,
            structure=structure,
            coord_labels=frozenset(coord_map),
            dense_structure=dense_structure,
        )
        return dense_num_points, coord_map, dense_structure_out

    if dense_structure is not None:
        raise ValueError(
            "dense_structure is only valid when num_points requests coord-separable sampling."
        )

    dense_num_points = _coerce_dense_num_points(num_points)
    return dense_num_points, None, None
