#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import heapq
import math

import numpy as np

from ._spectrum import DiscreteLaplacianEigenbasis, LaplacianEigenbasisReport


def _first_missing_product_eigenvalue(
    factors: tuple[DiscreteLaplacianEigenbasis, ...],
    /,
) -> float:
    candidates = []
    minimum_values = [float(np.asarray(factor.eigenvalues)[0]) for factor in factors]
    for index, factor in enumerate(factors):
        next_value = factor.report.next_eigenvalue
        if np.isfinite(next_value):
            candidates.append(
                float(next_value)
                + sum(
                    value
                    for position, value in enumerate(minimum_values)
                    if position != index
                )
            )
    return min(candidates, default=float("inf"))


def _lowest_product_modes(
    eigenvalue_arrays: tuple[np.ndarray, ...],
    count: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate the lowest Cartesian sums without materializing every mode."""
    shape = tuple(int(values.size) for values in eigenvalue_arrays)
    start = tuple(0 for _ in shape)
    heap = [(float(sum(values[0] for values in eigenvalue_arrays)), start)]
    visited = {start}
    selected_values = []
    selected_indices = []
    while len(selected_values) < count:
        value, index = heapq.heappop(heap)
        selected_values.append(value)
        selected_indices.append(index)
        for axis, size in enumerate(shape):
            if index[axis] + 1 >= size:
                continue
            neighbor_list = list(index)
            neighbor_list[axis] += 1
            neighbor = tuple(neighbor_list)
            if neighbor in visited:
                continue
            visited.add(neighbor)
            neighbor_value = float(
                sum(
                    eigenvalue_arrays[factor][neighbor[factor]]
                    for factor in range(len(shape))
                )
            )
            heapq.heappush(heap, (neighbor_value, neighbor))
    return np.asarray(selected_values), np.asarray(selected_indices, dtype=np.int64)


def product_laplacian_eigenbasis(
    factors: tuple[DiscreteLaplacianEigenbasis, ...],
    /,
    *,
    num_modes: int | None,
    degeneracy_tolerance: float = 1e-8,
    orthonormality_tolerance: float = 1e-8,
) -> DiscreteLaplacianEigenbasis:
    """Materialize certified low eigenpairs of a summed product Laplacian."""
    resolved = tuple(factors)
    if len(resolved) < 2:
        raise ValueError("A product Laplacian requires at least two factors.")
    if not all(isinstance(factor, DiscreteLaplacianEigenbasis) for factor in resolved):
        raise TypeError("Product factors must be DiscreteLaplacianEigenbasis objects.")
    if any(
        not factor.report.exact
        and (
            not factor.report.tail_certified
            or not np.isfinite(factor.report.next_eigenvalue)
        )
        for factor in resolved
    ):
        raise ValueError("Every truncated factor requires a certified spectral tail.")
    if float(degeneracy_tolerance) < 0.0:
        raise ValueError("degeneracy_tolerance must be nonnegative.")
    if float(orthonormality_tolerance) <= 0.0:
        raise ValueError("orthonormality_tolerance must be positive.")

    mode_shapes = tuple(factor.mode_count for factor in resolved)
    entity_shapes = tuple(factor.entity_count for factor in resolved)
    available_modes = math.prod(mode_shapes)
    requested = available_modes if num_modes is None else int(num_modes)
    if requested <= 0 or requested > available_modes:
        raise ValueError("num_modes must lie within the materialized product rank.")
    eigenvalue_arrays = tuple(np.asarray(factor.eigenvalues) for factor in resolved)
    summed_values, mode_indices = _lowest_product_modes(
        eigenvalue_arrays,
        min(available_modes, requested + 1),
    )

    omitted_factor_value = _first_missing_product_eigenvalue(resolved)
    next_enumerated = (
        float(summed_values[requested]) if requested < available_modes else float("inf")
    )
    next_eigenvalue = min(next_enumerated, omitted_factor_value)
    retained_boundary = float(summed_values[requested - 1])
    if np.isfinite(next_eigenvalue):
        boundary_gap = next_eigenvalue - retained_boundary
        scale = max(1.0, abs(next_eigenvalue), abs(retained_boundary))
        if boundary_gap <= float(degeneracy_tolerance) * scale:
            raise ValueError("num_modes splits a degenerate product eigenspace.")
    else:
        boundary_gap = float("inf")

    entity_count = math.prod(entity_shapes)
    flat_entities = np.arange(entity_count)
    selected_modes = mode_indices[:requested]
    functions = np.ones((entity_count, requested), dtype=float)
    measure = np.ones((entity_count,), dtype=float)
    active = np.ones((entity_count,), dtype=bool)
    entity_stride = entity_count
    for factor_index, factor in enumerate(resolved):
        entity_stride //= factor.entity_count
        factor_entities = (flat_entities // entity_stride) % factor.entity_count
        factor_functions = np.asarray(factor.eigenfunctions)
        functions *= factor_functions[
            factor_entities[:, None],
            selected_modes[None, :, factor_index],
        ]
        measure *= np.asarray(factor.probability_measure)[factor_entities]
        active &= np.asarray(factor.active_mask)[factor_entities]
    functions[~active] = 0.0
    exact = requested == available_modes and all(
        factor.report.exact for factor in resolved
    )
    source_id = "product:" + "|".join(factor.basis_id for factor in resolved)
    gram = functions.T @ (measure[:, None] * functions)
    residual = float(np.max(np.abs(gram - np.eye(requested))))
    report = LaplacianEigenbasisReport(
        method_id="best-first-product-sum",
        source_id=source_id,
        requested_modes=num_modes,
        retained_modes=requested,
        active_dimension=int(np.count_nonzero(active)),
        zero_mode_count=int(np.count_nonzero(summed_values[:requested] == 0.0)),
        canonicalized_zero_count=0,
        exact=exact,
        tail_certified=True,
        next_eigenvalue=next_eigenvalue,
        boundary_gap=boundary_gap,
        orthonormality_residual=residual,
    )
    return DiscreteLaplacianEigenbasis(
        summed_values[:requested],
        functions,
        measure,
        spectral_dimension=sum(factor.spectral_dimension for factor in resolved),
        basis_id=f"{source_id}:rank={requested}:exact={int(exact)}",
        active_mask=active,
        report=report,
        negative_eigenvalue_tolerance=0.0,
        orthonormality_tolerance=float(orthonormality_tolerance),
    )


__all__ = ["product_laplacian_eigenbasis"]
