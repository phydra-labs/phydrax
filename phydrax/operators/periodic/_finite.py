#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact finite orbital realizations with explicit boundary and disorder routes."""

from __future__ import annotations

from itertools import product
from math import isfinite, prod
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ._family import (
    PeriodicFiniteRealization,
    PeriodicResourceError,
    realize_periodic_translation_family,
)
from ._orbital import PreparedPeriodicOrbitalPencil


BoundaryKind = Literal["open", "periodic", "twisted", "slab"]


class PeriodicFiniteBoundaryPlan(StrictModule, NonTrainableState):
    supercell_shape: tuple[int, ...] = eqx.field(static=True)
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    twists: tuple[float, ...] = eqx.field(static=True)
    kind: BoundaryKind = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        supercell_shape: tuple[int, ...],
        periodic_axes: tuple[bool, ...],
        twists: tuple[float, ...],
        kind: BoundaryKind,
        /,
    ):
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in supercell_shape
        ):
            raise TypeError("Finite supercell dimensions must be exact integers.")
        shape = tuple(int(value) for value in supercell_shape)
        axes = tuple(bool(value) for value in periodic_axes)
        twist = tuple(float(value) for value in twists)
        if (
            not shape
            or len(shape) not in (1, 2, 3)
            or any(value <= 0 for value in shape)
            or len(axes) != len(shape)
            or len(twist) != len(shape)
            or any(not isfinite(value) for value in twist)
            or any(
                value != 0.0 and not axis for value, axis in zip(twist, axes, strict=True)
            )
            or kind not in ("open", "periodic", "twisted", "slab")
        ):
            raise ValueError("Finite boundary shape, axes, twists, or kind are invalid.")
        expected_kind = (
            "open"
            if not any(axes)
            else "twisted"
            if any(value != 0.0 for value in twist)
            else "periodic"
            if all(axes)
            else "slab"
        )
        if kind != expected_kind:
            raise ValueError("Finite boundary kind does not match its axes and twists.")
        self.supercell_shape = shape
        self.periodic_axes = axes
        self.twists = twist
        self.kind = kind
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "periodic-finite-boundary-plan",
                "route": kind,
                "shape": list(shape),
                "periodic_axes": list(axes),
                "twists": list(twist),
            }
        )

    @classmethod
    def open(cls, supercell_shape: tuple[int, ...], /) -> "PeriodicFiniteBoundaryPlan":
        rank = len(supercell_shape)
        return cls(supercell_shape, (False,) * rank, (0.0,) * rank, "open")

    @classmethod
    def periodic(
        cls, supercell_shape: tuple[int, ...], /
    ) -> "PeriodicFiniteBoundaryPlan":
        rank = len(supercell_shape)
        return cls(supercell_shape, (True,) * rank, (0.0,) * rank, "periodic")

    @classmethod
    def twisted(
        cls, supercell_shape: tuple[int, ...], twists: tuple[float, ...], /
    ) -> "PeriodicFiniteBoundaryPlan":
        rank = len(supercell_shape)
        return cls(supercell_shape, (True,) * rank, twists, "twisted")

    @classmethod
    def slab(
        cls, supercell_shape: tuple[int, ...], open_axis: int, /
    ) -> "PeriodicFiniteBoundaryPlan":
        rank = len(supercell_shape)
        axis = int(open_axis)
        if rank < 2 or axis < 0 or axis >= rank:
            raise ValueError(
                "Slab open_axis must select one axis of a rank-2/3 realization."
            )
        periodic = tuple(index != axis for index in range(rank))
        return cls(supercell_shape, periodic, (0.0,) * rank, "slab")


class PrescribedPeriodicDisorder(StrictModule, NonTrainableState):
    onsite_shifts: Array
    basis_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    disorder_id: str = eqx.field(static=True)

    def __init__(
        self,
        onsite_shifts: ArrayLike,
        basis_id: str,
        source_id: str,
        /,
    ):
        shifts = np.asarray(onsite_shifts)
        basis = str(basis_id).strip()
        source = str(source_id).strip()
        if shifts.ndim != 2 or np.any(~np.isfinite(shifts)) or not basis or not source:
            raise ValueError(
                "Prescribed disorder requires finite (cell, orbital) shifts and identities."
            )
        self.onsite_shifts = jnp.asarray(shifts)
        self.basis_id = basis
        self.source_id = source
        self.disorder_id = canonical_fingerprint(
            {
                "kind": "prescribed-periodic-disorder",
                "basis": basis,
                "source": source,
                "onsite_shifts": array_tree_fingerprint(shifts),
            }
        )


class PeriodicFiniteOrbitalRealization(StrictModule, NonTrainableState):
    hamiltonian: PeriodicFiniteRealization
    overlap: PeriodicFiniteRealization
    cell_coordinates: Array
    boundary: PeriodicFiniteBoundaryPlan
    disorder: PrescribedPeriodicDisorder | None
    basis_id: str = eqx.field(static=True)
    orbital_labels: tuple[str, ...] = eqx.field(static=True)
    order: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: PeriodicFiniteRealization,
        overlap: PeriodicFiniteRealization,
        cell_coordinates: ArrayLike,
        boundary: PeriodicFiniteBoundaryPlan,
        disorder: PrescribedPeriodicDisorder | None,
        basis_id: str,
        orbital_labels: tuple[str, ...],
        /,
    ):
        if not isinstance(hamiltonian, PeriodicFiniteRealization) or not isinstance(
            overlap, PeriodicFiniteRealization
        ):
            raise TypeError(
                "Finite orbital realization requires typed H and S realizations."
            )
        if (
            hamiltonian.input_size != overlap.input_size
            or hamiltonian.output_size != overlap.output_size
        ):
            raise ValueError("Finite Hamiltonian and overlap dimensions do not align.")
        self.hamiltonian = hamiltonian
        self.overlap = overlap
        self.cell_coordinates = jnp.asarray(cell_coordinates, dtype=jnp.int32)
        self.boundary = boundary
        self.disorder = disorder
        self.basis_id = str(basis_id)
        self.orbital_labels = tuple(orbital_labels)
        self.order = "cell-c-order_then-orbital-order"
        self.realization_id = canonical_fingerprint(
            {
                "kind": "periodic-finite-orbital-realization",
                "hamiltonian": hamiltonian.realization_id,
                "overlap": overlap.realization_id,
                "boundary": boundary.boundary_id,
                "disorder": None if disorder is None else disorder.disorder_id,
                "basis": self.basis_id,
                "order": self.order,
            }
        )


class PeriodicFiniteOrbitalPlan(StrictModule, NonTrainableState):
    pencil: PreparedPeriodicOrbitalPencil
    boundary: PeriodicFiniteBoundaryPlan
    disorder: PrescribedPeriodicDisorder | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pencil: PreparedPeriodicOrbitalPencil,
        boundary: PeriodicFiniteBoundaryPlan,
        /,
        *,
        disorder: PrescribedPeriodicDisorder | None = None,
    ):
        if not isinstance(pencil, PreparedPeriodicOrbitalPencil) or not isinstance(
            boundary, PeriodicFiniteBoundaryPlan
        ):
            raise TypeError(
                "Finite orbital plan requires a prepared pencil and boundary plan."
            )
        if len(boundary.supercell_shape) != pencil.plan.basis.cell.rank:
            raise ValueError("Finite boundary rank does not match orbital pencil rank.")
        if disorder is not None:
            if not isinstance(disorder, PrescribedPeriodicDisorder):
                raise TypeError("disorder must be PrescribedPeriodicDisorder or None.")
            expected = (prod(boundary.supercell_shape), pencil.plan.basis.orbital_count)
            if (
                disorder.basis_id != pencil.plan.basis.basis_id
                or disorder.onsite_shifts.shape != expected
            ):
                raise ValueError(
                    "Prescribed disorder support does not match finite cell/orbital order."
                )
        self.pencil = pencil
        self.boundary = boundary
        self.disorder = disorder
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-finite-orbital-plan",
                "pencil": pencil.prepared_id,
                "boundary": boundary.boundary_id,
                "disorder": None if disorder is None else disorder.disorder_id,
            }
        )

    def realize(self, /) -> PeriodicFiniteOrbitalRealization:
        hamiltonian = realize_periodic_translation_family(
            self.pencil.hamiltonian,
            self.boundary.supercell_shape,
            periodic_axes=self.boundary.periodic_axes,
            twists=self.boundary.twists,
        )
        overlap = realize_periodic_translation_family(
            self.pencil.overlap,
            self.boundary.supercell_shape,
            periodic_axes=self.boundary.periodic_axes,
            twists=self.boundary.twists,
        )
        if self.disorder is not None:
            diagonal = np.arange(hamiltonian.input_size, dtype=np.int32)
            shifts = np.asarray(self.disorder.onsite_shifts).reshape((-1,))
            if (
                diagonal.size + hamiltonian.entry_count
                > self.pencil.hamiltonian.plan.maximum_finite_entries
            ):
                raise PeriodicResourceError(
                    "Disordered finite realization exceeds maximum_finite_entries."
                )
            hamiltonian = PeriodicFiniteRealization(
                np.concatenate((np.asarray(hamiltonian.row_indices), diagonal)),
                np.concatenate((np.asarray(hamiltonian.column_indices), diagonal)),
                np.concatenate((np.asarray(hamiltonian.values), shifts)),
                supercell_shape=hamiltonian.supercell_shape,
                periodic_axes=hamiltonian.periodic_axes,
                twists=hamiltonian.twists,
                output_size=hamiltonian.output_size,
                input_size=hamiltonian.input_size,
                maximum_dense_entries=hamiltonian.maximum_dense_entries,
                source_prepared_id=f"{hamiltonian.source_prepared_id}:{self.disorder.disorder_id}",
            )
        coordinates = np.asarray(
            tuple(product(*(range(value) for value in self.boundary.supercell_shape))),
            dtype=np.int32,
        )
        return PeriodicFiniteOrbitalRealization(
            hamiltonian,
            overlap,
            coordinates,
            self.boundary,
            self.disorder,
            self.pencil.plan.basis.basis_id,
            self.pencil.plan.basis.labels,
        )


class FiniteLayerObservableResult(StrictModule, NonTrainableState):
    layer_populations: Array
    total_population: Array
    partition_residual: Array
    axis: int = eqx.field(static=True)
    successful: Array
    realization_id: str = eqx.field(static=True)


def finite_layer_populations(
    realization: PeriodicFiniteOrbitalRealization,
    coefficients: ArrayLike,
    occupations: ArrayLike,
    axis: int,
    /,
) -> FiniteLayerObservableResult:
    """Generalized-Mulliken populations partitioned by one finite cell axis."""

    if not isinstance(realization, PeriodicFiniteOrbitalRealization):
        raise TypeError("realization must be PeriodicFiniteOrbitalRealization.")
    coefficient = jnp.asarray(coefficients)
    occupation = jnp.asarray(occupations, dtype=coefficient.real.dtype)
    if (
        coefficient.ndim != 2
        or coefficient.shape[0] != realization.overlap.input_size
        or occupation.shape != (coefficient.shape[1],)
    ):
        raise ValueError(
            "Finite coefficients and occupations do not align with realization size."
        )
    axis_ = int(axis)
    if axis_ < 0 or axis_ >= len(realization.boundary.supercell_shape):
        raise ValueError("Layer axis exceeds finite realization rank.")
    metric_vectors = jnp.swapaxes(
        realization.overlap.apply(jnp.swapaxes(coefficient, 0, 1)), 0, 1
    )
    orbital_population = jnp.real(
        contract(
            "an,n,an->a", jnp.conj(coefficient), occupation, metric_vectors, backend="jax"
        )
    )
    orbital_count = len(realization.orbital_labels)
    cell_population = orbital_population.reshape((-1, orbital_count)).sum(axis=1)
    layer_index = realization.cell_coordinates[:, axis_]
    layer = jnp.zeros(
        (realization.boundary.supercell_shape[axis_],), dtype=cell_population.dtype
    )
    layer = layer.at[layer_index].add(cell_population)
    total = jnp.sum(occupation)
    residual = jnp.abs(jnp.sum(layer) - total)
    successful = jnp.all(jnp.isfinite(layer)) & (residual <= 1.0e-8)
    return FiniteLayerObservableResult(
        layer,
        total,
        residual,
        axis_,
        successful,
        realization.realization_id,
    )


__all__ = [
    "BoundaryKind",
    "FiniteLayerObservableResult",
    "PeriodicFiniteBoundaryPlan",
    "PeriodicFiniteOrbitalPlan",
    "PeriodicFiniteOrbitalRealization",
    "PrescribedPeriodicDisorder",
    "finite_layer_populations",
]
