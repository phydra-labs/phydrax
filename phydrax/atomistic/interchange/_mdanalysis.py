#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import numpy as np

from .._frame import AtomisticFrame, AtomisticMetadata, AtomisticSelectionPlan
from .._sites import AtomisticSiteDomain


def _mdanalysis():
    if find_spec("MDAnalysis") is None:
        raise ImportError(
            "MDAnalysis interoperability requires the optional MDAnalysis package."
        )
    return import_module("MDAnalysis")


def atomistic_frame_from_mdanalysis(
    universe, /, *, system_id: str, topology_id: str, unit_system_id: str, source_id: str
):
    atoms = universe.atoms
    timestep = universe.trajectory.ts
    dimensions = None if timestep.dimensions is None else timestep.triclinic_dimensions
    velocity = None if not timestep.has_velocities else np.asarray(atoms.velocities)
    force = None if not timestep.has_forces else np.asarray(atoms.forces)
    return AtomisticFrame(
        timestep.time,
        timestep.frame,
        np.asarray(atoms.positions),
        np.asarray(atoms.indices),
        velocities=velocity,
        forces=force,
        cell_vectors=dimensions,
        coordinate_domain=AtomisticSiteDomain.PHYSICAL_ATOMS,
        system_id=system_id,
        topology_id=topology_id,
        unit_system_id=unit_system_id,
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
    universe = mda.Universe.empty(values[0].positions.shape[0], trajectory=True)
    coordinates = np.stack(tuple(np.asarray(value.positions) for value in values))
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
