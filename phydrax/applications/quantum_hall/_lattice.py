#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Magnetic honeycomb and square-lattice quantum Hall models."""

from __future__ import annotations

from math import gcd, isfinite, pi, sqrt

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import PeriodicCell
from ...operators.periodic import (
    periodic_translation_family_from_dense_blocks,
    PeriodicBlochGauge,
    PeriodicOrbitalBasisPlan,
    PeriodicOrbitalPencilPlan,
    PreparedPeriodicOrbitalPencil,
)
from ...units import ENERGY, LENGTH, UnitDefinition


def _units(energy_unit: UnitDefinition, length_unit: UnitDefinition) -> None:
    if not isinstance(energy_unit, UnitDefinition) or not isinstance(
        length_unit, UnitDefinition
    ):
        raise TypeError("Hall lattice models require typed energy and length units.")
    if energy_unit.dimension != ENERGY or length_unit.dimension != LENGTH:
        raise ValueError("Hall lattice units must have energy and length dimensions.")


def _prepared_pencil(
    cell: PeriodicCell,
    labels: tuple[str, ...],
    centers: np.ndarray,
    translations: tuple[tuple[int, int], ...],
    matrices: tuple[np.ndarray, ...],
    energy_unit: UnitDefinition,
    length_unit: UnitDefinition,
    /,
) -> PreparedPeriodicOrbitalPencil:
    blocks = np.stack(matrices)[:, :, None, :, None]
    family = periodic_translation_family_from_dense_blocks(
        np.asarray(translations, dtype=np.int32),
        blocks,
        maximum_dense_entries=16_000_000,
        maximum_finite_entries=32_000_000,
    )
    basis = PeriodicOrbitalBasisPlan(
        cell,
        labels,
        centers,
        length_unit,
        PeriodicBlochGauge("lattice"),
        spin_order="spinless"
        if len(labels) == len({label.split("-")[0] for label in labels})
        else "interleaved-alpha-beta",
    )
    return PeriodicOrbitalPencilPlan.orthonormal(
        basis,
        family.plan,
        family.state,
        energy_unit,
    ).prepare()


def _honeycomb_cell(length: float) -> PeriodicCell:
    vectors = length * np.asarray(
        ((sqrt(3.0) / 2.0, 1.5), (-sqrt(3.0) / 2.0, 1.5)),
        dtype=np.float64,
    )
    return PeriodicCell(vectors)


class HaldaneModelPlan(StrictModule, NonTrainableState):
    """Spinless honeycomb Haldane model in an explicit lattice Bloch gauge."""

    nearest_hopping: float = eqx.field(static=True)
    sublattice_mass: float = eqx.field(static=True)
    next_nearest_hopping: float = eqx.field(static=True)
    next_nearest_phase: float = eqx.field(static=True)
    lattice_spacing: float = eqx.field(static=True)
    energy_unit: UnitDefinition
    length_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        nearest_hopping: float,
        sublattice_mass: float,
        next_nearest_hopping: float,
        next_nearest_phase: float,
        lattice_spacing: float,
        energy_unit: UnitDefinition,
        length_unit: UnitDefinition,
        /,
    ):
        values = tuple(
            float(value)
            for value in (
                nearest_hopping,
                sublattice_mass,
                next_nearest_hopping,
                next_nearest_phase,
                lattice_spacing,
            )
        )
        if any(not isfinite(value) for value in values) or values[-1] <= 0.0:
            raise ValueError("Haldane couplings and lattice spacing must be finite.")
        _units(energy_unit, length_unit)
        (
            self.nearest_hopping,
            self.sublattice_mass,
            self.next_nearest_hopping,
            self.next_nearest_phase,
            self.lattice_spacing,
        ) = values
        self.energy_unit = energy_unit
        self.length_unit = length_unit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "haldane-model-plan",
                "values": values,
                "energy_unit": energy_unit.unit_id,
                "length_unit": length_unit.unit_id,
            }
        )

    def prepare(self, /) -> PreparedPeriodicOrbitalPencil:
        t1 = self.nearest_hopping
        t2 = self.next_nearest_hopping
        phase = self.next_nearest_phase
        mass = self.sublattice_mass
        translations: list[tuple[int, int]] = [(0, 0)]
        onsite = np.asarray(((mass, -t1), (-t1, -mass)), dtype=np.complex128)
        matrices: list[np.ndarray] = [onsite]
        for translation in ((-1, 0), (0, -1)):
            forward = np.zeros((2, 2), dtype=np.complex128)
            forward[0, 1] = -t1
            reverse = np.conj(forward.T)
            translations.extend((translation, tuple(-value for value in translation)))
            matrices.extend((forward, reverse))
        amplitude = t2 * np.exp(1.0j * phase)
        for translation in ((1, 0), (0, -1), (-1, 1)):
            forward = np.diag((amplitude, np.conj(amplitude))).astype(np.complex128)
            reverse = np.conj(forward.T)
            translations.extend((translation, tuple(-value for value in translation)))
            matrices.extend((forward, reverse))
        return _prepared_pencil(
            _honeycomb_cell(self.lattice_spacing),
            ("A", "B"),
            np.asarray(((0.0, 0.0), (1.0 / 3.0, 1.0 / 3.0))),
            tuple(translations),
            tuple(matrices),
            self.energy_unit,
            self.length_unit,
        )


class KaneMeleModelPlan(StrictModule, NonTrainableState):
    """Spinful Kane--Mele model with intrinsic and Rashba spin-orbit terms."""

    nearest_hopping: float = eqx.field(static=True)
    sublattice_mass: float = eqx.field(static=True)
    intrinsic_spin_orbit: float = eqx.field(static=True)
    rashba_spin_orbit: float = eqx.field(static=True)
    lattice_spacing: float = eqx.field(static=True)
    energy_unit: UnitDefinition
    length_unit: UnitDefinition
    time_reversal_unitary: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        nearest_hopping: float,
        sublattice_mass: float,
        intrinsic_spin_orbit: float,
        rashba_spin_orbit: float,
        lattice_spacing: float,
        energy_unit: UnitDefinition,
        length_unit: UnitDefinition,
        /,
    ):
        values = tuple(
            float(value)
            for value in (
                nearest_hopping,
                sublattice_mass,
                intrinsic_spin_orbit,
                rashba_spin_orbit,
                lattice_spacing,
            )
        )
        if any(not isfinite(value) for value in values) or values[-1] <= 0.0:
            raise ValueError("Kane--Mele couplings and lattice spacing must be finite.")
        _units(energy_unit, length_unit)
        time_reversal = np.kron(np.eye(2), np.asarray(((0.0, 1.0), (-1.0, 0.0))))
        (
            self.nearest_hopping,
            self.sublattice_mass,
            self.intrinsic_spin_orbit,
            self.rashba_spin_orbit,
            self.lattice_spacing,
        ) = values
        self.energy_unit = energy_unit
        self.length_unit = length_unit
        self.time_reversal_unitary = jnp.asarray(time_reversal)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kane-mele-model-plan",
                "values": values,
                "energy_unit": energy_unit.unit_id,
                "length_unit": length_unit.unit_id,
            }
        )

    def prepare(self, /) -> PreparedPeriodicOrbitalPencil:
        t1 = self.nearest_hopping
        mass = self.sublattice_mass
        intrinsic = self.intrinsic_spin_orbit
        rashba = self.rashba_spin_orbit
        sigma_x = np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=np.complex128)
        sigma_y = np.asarray(((0.0, -1.0j), (1.0j, 0.0)), dtype=np.complex128)
        sigma_z = np.asarray(((1.0, 0.0), (0.0, -1.0)), dtype=np.complex128)
        identity = np.eye(2, dtype=np.complex128)
        onsite = np.zeros((4, 4), dtype=np.complex128)
        onsite[:2, :2] = mass * identity
        onsite[2:, 2:] = -mass * identity
        translations: list[tuple[int, int]] = [(0, 0)]
        matrices: list[np.ndarray] = [onsite]
        nearest = (
            ((0, 0), np.asarray((0.0, 1.0))),
            ((-1, 0), np.asarray((-sqrt(3.0) / 2.0, -0.5))),
            ((0, -1), np.asarray((sqrt(3.0) / 2.0, -0.5))),
        )
        for translation, direction in nearest:
            spin_block = -t1 * identity + 1.0j * rashba * (
                direction[1] * sigma_x - direction[0] * sigma_y
            )
            if translation == (0, 0):
                matrices[0][:2, 2:] += spin_block
                matrices[0][2:, :2] += np.conj(spin_block.T)
            else:
                forward = np.zeros((4, 4), dtype=np.complex128)
                forward[:2, 2:] = spin_block
                translations.extend((translation, tuple(-value for value in translation)))
                matrices.extend((forward, np.conj(forward.T)))
        for translation in ((1, 0), (0, -1), (-1, 1)):
            forward = np.zeros((4, 4), dtype=np.complex128)
            forward[:2, :2] = 1.0j * intrinsic * sigma_z
            forward[2:, 2:] = -1.0j * intrinsic * sigma_z
            translations.extend((translation, tuple(-value for value in translation)))
            matrices.extend((forward, np.conj(forward.T)))
        prepared = _prepared_pencil(
            _honeycomb_cell(self.lattice_spacing),
            ("A-up", "A-down", "B-up", "B-down"),
            np.asarray(
                (
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (1.0 / 3.0, 1.0 / 3.0),
                    (1.0 / 3.0, 1.0 / 3.0),
                )
            ),
            tuple(translations),
            tuple(matrices),
            self.energy_unit,
            self.length_unit,
        )
        return prepared


class HofstadterModelPlan(StrictModule, NonTrainableState):
    """Square-lattice Hofstadter model in Landau gauge."""

    flux_numerator: int = eqx.field(static=True)
    flux_denominator: int = eqx.field(static=True)
    hopping: float = eqx.field(static=True)
    lattice_spacing: float = eqx.field(static=True)
    energy_unit: UnitDefinition
    length_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        flux_numerator: int,
        flux_denominator: int,
        hopping: float,
        lattice_spacing: float,
        energy_unit: UnitDefinition,
        length_unit: UnitDefinition,
        /,
    ):
        numerator = int(flux_numerator)
        denominator = int(flux_denominator)
        hopping_ = float(hopping)
        spacing = float(lattice_spacing)
        if (
            denominator < 1
            or not 0 <= numerator < denominator
            or gcd(numerator, denominator) != 1
        ):
            raise ValueError("Hofstadter flux must be a reduced fraction in [0, 1).")
        if not isfinite(hopping_) or not isfinite(spacing) or spacing <= 0.0:
            raise ValueError("Hofstadter hopping and lattice spacing must be finite.")
        _units(energy_unit, length_unit)
        self.flux_numerator = numerator
        self.flux_denominator = denominator
        self.hopping = hopping_
        self.lattice_spacing = spacing
        self.energy_unit = energy_unit
        self.length_unit = length_unit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hofstadter-model-plan",
                "flux": (numerator, denominator),
                "hopping": hopping_,
                "lattice_spacing": spacing,
                "energy_unit": energy_unit.unit_id,
                "length_unit": length_unit.unit_id,
            }
        )

    def prepare(self, /) -> PreparedPeriodicOrbitalPencil:
        q = self.flux_denominator
        t = self.hopping
        onsite = np.zeros((q, q), dtype=np.complex128)
        for orbital in range(q - 1):
            onsite[orbital + 1, orbital] = -t
            onsite[orbital, orbital + 1] = -t
        x_forward = np.zeros((q, q), dtype=np.complex128)
        x_forward[0, q - 1] = -t
        x_reverse = np.conj(x_forward.T)
        y_forward = np.diag(
            tuple(
                -t * np.exp(2.0j * pi * self.flux_numerator * orbital / q)
                for orbital in range(q)
            )
        )
        y_reverse = np.conj(y_forward.T)
        cell = PeriodicCell(
            self.lattice_spacing
            * np.asarray(((float(q), 0.0), (0.0, 1.0)), dtype=np.float64)
        )
        return _prepared_pencil(
            cell,
            tuple(f"x-{index}" for index in range(q)),
            np.asarray(tuple((index / q, 0.0) for index in range(q))),
            ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)),
            (onsite, x_forward, x_reverse, y_forward, y_reverse),
            self.energy_unit,
            self.length_unit,
        )


__all__ = ["HaldaneModelPlan", "HofstadterModelPlan", "KaneMeleModelPlan"]
