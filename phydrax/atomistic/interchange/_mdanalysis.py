#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import numpy as np

from ...units import (
    ANGSTROM,
    conversion_factor,
    convert_value,
    derived_unit,
    KILOJOULE_PER_MOLE,
    PICOSECOND,
)
from .._frame import AtomisticFrame, AtomisticMetadata, AtomisticSelectionPlan
from .._sites import AtomisticSiteDomain
from .._units import AtomisticUnitSystem, molar_energy_to_single_system_factor


_ANGSTROM_PER_PICOSECOND = derived_unit("angstrom/ps", ((ANGSTROM, 1), (PICOSECOND, -1)))


def _mdanalysis():
    if find_spec("MDAnalysis") is None:
        raise ImportError(
            "MDAnalysis interoperability requires the optional MDAnalysis package."
        )
    return import_module("MDAnalysis")


def atomistic_frame_from_mdanalysis(
    universe,
    /,
    *,
    system_id: str,
    topology_id: str,
    units: AtomisticUnitSystem,
    source_id: str,
):
    if not isinstance(units, AtomisticUnitSystem):
        raise TypeError("units must be an AtomisticUnitSystem.")
    atoms = universe.atoms
    timestep = universe.trajectory.ts
    dimensions = (
        None
        if timestep.dimensions is None
        else convert_value(
            timestep.triclinic_dimensions,
            source=ANGSTROM,
            target=units.scale.length_unit,
        )
    )
    velocity = (
        None
        if not timestep.has_velocities
        else convert_value(
            np.asarray(atoms.velocities),
            source=_ANGSTROM_PER_PICOSECOND,
            target=units.velocity_unit,
        )
    )
    if not timestep.has_forces:
        force = None
    else:
        energy_factor = molar_energy_to_single_system_factor(
            KILOJOULE_PER_MOLE,
            units.scale.energy_unit,
            constant_set_id=units.constant_set_id,
        )
        length_factor = float(conversion_factor(ANGSTROM, units.scale.length_unit))
        force = np.asarray(atoms.forces) * energy_factor / length_factor
    return AtomisticFrame(
        convert_value(
            timestep.time,
            source=PICOSECOND,
            target=units.time_unit,
        ),
        timestep.frame,
        convert_value(
            np.asarray(atoms.positions),
            source=ANGSTROM,
            target=units.scale.length_unit,
        ),
        np.asarray(atoms.indices),
        velocities=velocity,
        forces=force,
        cell_vectors=dimensions,
        coordinate_domain=AtomisticSiteDomain.PHYSICAL_ATOMS,
        system_id=system_id,
        topology_id=topology_id,
        units=units,
        source_id=source_id,
    )


def atomistic_metadata_from_mdanalysis(universe, /):
    atoms = universe.atoms
    return AtomisticMetadata(
        tuple(atoms.names),
        tuple(atoms.elements),
        np.asarray(atoms.resids),
        tuple(atoms.resnames),
        tuple(atoms.chainIDs),
        tuple(atoms.segids),
    )


def mdanalysis_universe_from_frames(frames, /):
    mda = _mdanalysis()
    values = tuple(frames)
    if not values:
        raise ValueError("At least one frame is required.")
    units = values[0].units
    if any(value.units.unit_system_id != units.unit_system_id for value in values[1:]):
        raise ValueError("MDAnalysis export requires one complete unit system.")
    universe = mda.Universe.empty(values[0].positions.shape[0], trajectory=True)
    coordinates = np.stack(
        tuple(
            np.asarray(
                convert_value(
                    value.positions,
                    source=value.units.scale.length_unit,
                    target=ANGSTROM,
                )
            )
            for value in values
        )
    )
    universe.load_new(coordinates, format=mda.coordinates.memory.MemoryReader)
    return universe


def mdanalysis_selection(universe, selection: str, /, *, stable_ids=None):
    group = universe.select_atoms(str(selection))
    ids = (
        np.asarray(universe.atoms.indices, dtype=np.int64)
        if stable_ids is None
        else np.asarray(stable_ids, dtype=np.int64)
    )
    if ids.shape != (len(universe.atoms),):
        raise ValueError("MDAnalysis stable IDs must align with universe atoms.")
    mask = np.zeros(ids.shape, dtype=bool)
    mask[np.asarray(group.indices, dtype=np.int64)] = True
    return AtomisticSelectionPlan(ids, mask)


__all__ = [
    "atomistic_frame_from_mdanalysis",
    "atomistic_metadata_from_mdanalysis",
    "mdanalysis_selection",
    "mdanalysis_universe_from_frames",
]
