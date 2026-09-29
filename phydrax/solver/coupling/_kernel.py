#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Kernel detection of affine spatial coupled operators.

The kernel of a coupled operator is sought in a declared, bounded span: the
kernels the components publish (candidates), completed by every coordinate of
the law-owned blocks and of the interface-only components (components that
publish no side traces, such as boundary-integral owners). A coupled kernel
may run through unknowns that no owner declares a kernel for: the constant
mode ``u = 1, phi = 1, q = 0, c = 1`` of a pure-Neumann interior next to a
bounded exterior carries the law-owned trace projection ``phi`` and the
boundary owner's far-field constant ``c``. Volume (trace) owners enter only
through their published kernels.

Within that span the detection is exact: the operator's images of an
orthonormal basis of the span are materialized once (under the declared
materialization budget) and their null directions are the kernel. Images are
read with every output block scaled to unit gain on a fixed probe, so owners
of very different operator scales (a high-degree virtual element next to
boundary-integral blocks) do not set each other's null threshold. The left
kernel is sought the same way in the row space, from the component kernels
read as row covectors (exact for the symmetric owner operators that publish
them) and the rows of the same owners; a kernel pair of unequal dimension is
refused rather than gauged with the wrong basis.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax.numpy as jnp
import numpy as np
from jax import Array

from ...linalg import BlockSpace, MaterializationPolicy
from ._assembly import coupled_weak_operator, CoupledChart, OwnerPath
from ._components import AbstractTraceComponent


type _Kernels = dict[str, tuple[Array, ...]]


def _block_ranges(space: BlockSpace, /) -> dict[OwnerPath, tuple[int, int]]:
    """Canonical flat coordinate range of every ``(owner, block)`` of a solve space."""
    ranges: dict[OwnerPath, tuple[int, int]] = {}
    offset = 0
    for owner, member in zip(space.names, space.spaces, strict=True):
        if not isinstance(member, BlockSpace):
            raise TypeError("Solve owners are block spaces.")
        for name, block in zip(member.names, member.spaces, strict=True):
            ranges[(owner, name)] = (offset, offset + block.size)
            offset += block.size
    return ranges


def _published_kernels(chart: CoupledChart, args: Mapping[str, object], /) -> _Kernels:
    kernels: _Kernels = {}
    for component in chart.components:
        kernel = component.nullspace(args[component.name])
        if kernel is not None:
            kernels[component.name] = kernel
    return kernels


def _candidates(chart: CoupledChart, kernels: _Kernels, /, *, rows: bool) -> np.ndarray:
    """Solve-coordinate columns of the published kernels (states or row covectors).

    Row covectors pair each owner's row blocks one-to-one with its state blocks
    (the square-owner invariant of the chart).
    """
    space = chart.row_space if rows else chart.state_space
    ranges = _block_ranges(space)
    columns: list[np.ndarray] = []
    for component in chart.components:
        kernel = kernels.get(component.name)
        if kernel is None:
            continue
        blocks = component.row_blocks if rows else component.state_blocks
        for block, basis in zip(blocks, kernel, strict=True):
            path = (component.name, block.name)
            positions = chart.kept_positions(path, rows=rows)
            local = np.asarray(basis).reshape((block.space.size, -1))
            restricted = local if positions is None else local[np.asarray(positions)]
            start, stop = ranges[path]
            for column in restricted.T:
                vector = np.zeros((space.size,), dtype=np.float64)
                vector[start:stop] = column
                columns.append(vector)
    return np.stack(columns, axis=1)


def _completion(chart: CoupledChart, kernels: _Kernels, /, *, rows: bool) -> np.ndarray:
    """Flat coordinates of the law-owned and interface-only owner blocks."""
    completing = {law.law_id for law in chart.laws} | {
        component.name
        for component in chart.components
        if not isinstance(component, AbstractTraceComponent)
        and component.name not in kernels
    }
    ranges = _block_ranges(chart.row_space if rows else chart.state_space)
    indices = [
        np.arange(start, stop)
        for (owner, _), (start, stop) in ranges.items()
        if owner in completing
    ]
    return np.concatenate(indices) if indices else np.zeros((0,), dtype=np.int64)


def _orthonormal_span(
    candidates: np.ndarray, completion: np.ndarray, size: int, /
) -> np.ndarray:
    """Orthonormal basis of the candidate span plus the completion coordinates."""
    left, singular, _ = np.linalg.svd(candidates, full_matrices=False)
    rank = np.count_nonzero(singular > singular[0] * size * np.finfo(np.float64).eps)
    units = np.zeros((size, completion.size), dtype=np.float64)
    units[completion, np.arange(completion.size)] = 1.0
    return np.concatenate((left[:, :rank], units), axis=1)


def _require_budget(
    rows: int, columns: int, materialization: MaterializationPolicy, /
) -> None:
    entries = rows * columns
    if entries > materialization.max_entries or 8 * entries > materialization.max_bytes:
        raise ValueError(
            f"Kernel detection materializes {entries} operator entries ({rows} rows x "
            f"{columns} span columns), beyond the declared materialization budget; "
            "raise CoupledResourcePolicy(materialization=...)."
        )


def _null_directions(
    apply: Callable[[Array], Array],
    basis: np.ndarray,
    gains: np.ndarray,
    tolerance: float,
    /,
) -> np.ndarray:
    """Orthonormal combinations of ``basis`` columns that ``apply`` annihilates.

    ``gains`` divides each output coordinate by its block's probe gain, so a
    direction is null when its gain-scaled image is at most ``tolerance``.
    """
    images = np.asarray(apply(jnp.asarray(basis))) / gains[:, None]
    _, singular, right = np.linalg.svd(images, full_matrices=True)
    padded = np.concatenate((singular, np.zeros((basis.shape[1] - singular.size,))))
    return basis @ right[padded <= tolerance].T


def _block_gains(
    apply: Callable[[Array], Array], inputs: BlockSpace, outputs: BlockSpace, /
) -> np.ndarray:
    """Per-coordinate probe gain of the output blocks of ``apply``.

    The gain of one output block is the RMS of its image of a fixed
    nonsymmetric probe per unit RMS probe entry; after dividing by it every
    block of the image has unit gain. A block with a vanishing image (zero
    rows) keeps unit gain.
    """
    probe = np.linspace(1.0, 2.0, inputs.size) * np.cos(
        np.arange(inputs.size, dtype=np.float64)
    )
    image = np.asarray(apply(jnp.asarray(probe[:, None])))[:, 0]
    scale = float(np.sqrt(np.mean(probe**2)))
    gains = np.ones((outputs.size,), dtype=np.float64)
    for start, stop in _block_ranges(outputs).values():
        gain = float(np.sqrt(np.mean(image[start:stop] ** 2))) / scale
        if gain > 0.0:
            gains[start:stop] = gain
    return gains


def detect_coupled_kernel(
    chart: CoupledChart,
    args: Mapping[str, object],
    tolerance: float,
    materialization: MaterializationPolicy,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Orthonormal right (state) and left (row-covector) kernels in the declared span.

    ``None`` when no component publishes a kernel or when neither a right nor a
    left kernel survives coupling.
    A direction is null when its image, with every output block divided by its
    probe gain, is at most ``tolerance`` times its norm. Raises when the
    materialized span exceeds ``materialization`` or when the left and right
    kernels differ in dimension.
    """
    kernels = _published_kernels(chart, args)
    if not kernels:
        return None
    operator = coupled_weak_operator(chart, args)
    rows, states = chart.row_space, chart.state_space
    state = _orthonormal_span(
        _candidates(chart, kernels, rows=False),
        _completion(chart, kernels, rows=False),
        chart.state_space.size,
    )
    _require_budget(chart.row_space.size, state.shape[1], materialization)
    right = _null_directions(
        operator.mv_block,
        state,
        _block_gains(operator.mv_block, states, rows),
        tolerance,
    )
    covectors = _orthonormal_span(
        _candidates(chart, kernels, rows=True),
        _completion(chart, kernels, rows=True),
        chart.row_space.size,
    )
    _require_budget(chart.state_space.size, covectors.shape[1], materialization)
    left = _null_directions(
        operator.transpose_mv_block,
        covectors,
        _block_gains(operator.transpose_mv_block, rows, states),
        tolerance,
    )
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            f"The coupled operator has a {right.shape[1]}-dimensional right kernel "
            f"but a {left.shape[1]}-dimensional left kernel within the component "
            "kernels completed by law-owned and interface-only blocks; no gauge "
            "can be declared for this kernel pair."
        )
    if right.shape[1] == 0:
        return None
    return right, left


__all__ = ["detect_coupled_kernel"]
