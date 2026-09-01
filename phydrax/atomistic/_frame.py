#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ._sites import AtomisticSiteDomain


class AtomisticFrameFields(IntFlag):
    POSITIONS = 1 << 0
    VELOCITIES = 1 << 1
    MOMENTA = 1 << 2
    FORCES = 1 << 3
    CELL = 1 << 4
    IMAGES = 1 << 5
    ENERGY = 1 << 6
    AUXILIARY = 1 << 7


class AtomisticMetadata(StrictModule, NonTrainableState):
    atom_names: tuple[str, ...] = eqx.field(static=True)
    element_labels: tuple[str, ...] = eqx.field(static=True)
    residue_ids: Array
    residue_names: tuple[str, ...] = eqx.field(static=True)
    chain_ids: tuple[str, ...] = eqx.field(static=True)
    segment_ids: tuple[str, ...] = eqx.field(static=True)
    metadata_id: str = eqx.field(static=True)

    def __init__(
        self,
        atom_names,
        element_labels,
        residue_ids,
        residue_names,
        chain_ids,
        segment_ids,
        /,
    ):
        names = tuple(str(value) for value in atom_names)
        count = len(names)
        elements = tuple(str(value) for value in element_labels)
        residues = np.asarray(residue_ids)
        residue_names_ = tuple(str(value) for value in residue_names)
        chains = tuple(str(value) for value in chain_ids)
        segments = tuple(str(value) for value in segment_ids)
        if any(
            len(value) != count for value in (elements, residue_names_, chains, segments)
        ) or residues.shape != (count,):
            raise ValueError("Atomistic metadata fields must have one entry per atom.")
        if not np.issubdtype(residues.dtype, np.integer):
            raise TypeError("residue_ids must be integers.")
        self.atom_names = names
        self.element_labels = elements
        self.residue_ids = jnp.asarray(residues, dtype=jnp.int32)
        self.residue_names = residue_names_
        self.chain_ids = chains
        self.segment_ids = segments
        self.metadata_id = canonical_fingerprint(
            {
                "kind": "atomistic-metadata",
                "atom_names": list(names),
                "elements": list(elements),
                "residue_ids": array_tree_fingerprint(residues),
                "residue_names": list(residue_names_),
                "chains": list(chains),
                "segments": list(segments),
            }
        )

    @classmethod
    def minimal(cls, atomic_numbers: ArrayLike, /) -> "AtomisticMetadata":
        numbers = np.asarray(atomic_numbers)
        count = numbers.size
        return cls(
            tuple(f"A{index}" for index in range(count)),
            tuple(str(int(value)) for value in numbers),
            np.arange(count),
            ("MOL",) * count,
            ("A",) * count,
            ("SYSTEM",) * count,
        )


class AtomisticSelectionPlan(StrictModule, NonTrainableState):
    mask: Array
    stable_ids: Array
    selection_id: str = eqx.field(static=True)

    def __init__(self, stable_ids: ArrayLike, mask: ArrayLike, /):
        ids = np.asarray(stable_ids)
        selected = np.asarray(mask, dtype=bool)
        if ids.ndim != 1 or selected.shape != ids.shape:
            raise ValueError("Selection mask must align with stable IDs.")
        self.mask = jnp.asarray(selected)
        self.stable_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.selection_id = canonical_fingerprint(
            {
                "kind": "atomistic-selection",
                "ids": array_tree_fingerprint(ids),
                "mask": array_tree_fingerprint(selected),
            }
        )

    def __and__(self, other: "AtomisticSelectionPlan") -> "AtomisticSelectionPlan":
        self._require_same(other)
        return AtomisticSelectionPlan(self.stable_ids, self.mask & other.mask)

    def __or__(self, other: "AtomisticSelectionPlan") -> "AtomisticSelectionPlan":
        self._require_same(other)
        return AtomisticSelectionPlan(self.stable_ids, self.mask | other.mask)

    def __invert__(self) -> "AtomisticSelectionPlan":
        return AtomisticSelectionPlan(self.stable_ids, ~self.mask)

    def _require_same(self, other, /) -> None:
        if not isinstance(other, AtomisticSelectionPlan) or not np.array_equal(
            np.asarray(self.stable_ids), np.asarray(other.stable_ids)
        ):
            raise ValueError("Selections belong to different stable-ID supports.")


class AtomisticFrame(StrictModule):
    time: Array
    step: Array
    positions: Array
    stable_ids: Array
    velocities: Array | None
    momenta: Array | None
    forces: Array | None
    cell_vectors: Array | None
    image_counts: Array | None
    energy: Array | None
    auxiliary: dict[str, Array]
    valid: Array
    coordinate_domain: AtomisticSiteDomain = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        time: ArrayLike,
        step: ArrayLike,
        positions: ArrayLike,
        stable_ids: ArrayLike,
        /,
        *,
        velocities: ArrayLike | None = None,
        momenta: ArrayLike | None = None,
        forces: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
        image_counts: ArrayLike | None = None,
        energy: ArrayLike | None = None,
        auxiliary: dict[str, ArrayLike] | None = None,
        valid: ArrayLike = True,
        coordinate_domain: AtomisticSiteDomain = AtomisticSiteDomain.DOF_ATOMS,
        system_id: str,
        topology_id: str,
        unit_system_id: str,
        source_id: str,
    ):
        position = jnp.asarray(positions)
        ids = jnp.asarray(stable_ids, dtype=jnp.int64)
        if (
            position.ndim != 2
            or position.shape[-1] != 3
            or ids.shape != position.shape[:1]
        ):
            raise ValueError("Frame positions and stable IDs have incompatible shapes.")

        def optional(value, dtype=None):
            return None if value is None else jnp.asarray(value, dtype=dtype)

        velocity = optional(velocities, position.dtype)
        momentum = optional(momenta, position.dtype)
        force = optional(forces, position.dtype)
        images = optional(image_counts, jnp.int32)
        for name, value in (
            ("velocities", velocity),
            ("momenta", momentum),
            ("forces", force),
            ("image_counts", images),
        ):
            if value is not None and value.shape != position.shape:
                raise ValueError(f"Frame {name} must match positions.")
        cell = optional(cell_vectors, position.dtype)
        if cell is not None and cell.shape != (3, 3):
            raise ValueError("Frame cell_vectors must have shape (3,3).")
        self.time = jnp.asarray(time, dtype=position.dtype).reshape(())
        self.step = jnp.asarray(step, dtype=jnp.int64).reshape(())
        self.positions = position
        self.stable_ids = ids
        self.velocities = velocity
        self.momenta = momentum
        self.forces = force
        self.cell_vectors = cell
        self.image_counts = images
        self.energy = optional(energy, position.dtype)
        auxiliary_values = (
            {}
            if auxiliary is None
            else {str(key): jnp.asarray(value) for key, value in auxiliary.items()}
        )
        if any(not key or "/" in key for key in auxiliary_values):
            raise ValueError("Frame auxiliary names must be non-empty path-free strings.")
        if not isinstance(coordinate_domain, AtomisticSiteDomain):
            raise TypeError("coordinate_domain must be AtomisticSiteDomain.")
        self.auxiliary = auxiliary_values
        self.valid = jnp.asarray(valid, dtype=bool).reshape(())
        self.coordinate_domain = coordinate_domain
        self.system_id = str(system_id)
        self.topology_id = str(topology_id)
        self.unit_system_id = str(unit_system_id)
        self.source_id = str(source_id)
        if any(
            not value
            for value in (
                self.system_id,
                self.topology_id,
                self.unit_system_id,
                self.source_id,
            )
        ):
            raise ValueError("Frame identities must be non-empty.")


class AbstractAtomisticTrajectorySourcePlan(StrictModule, NonTrainableState):
    source_id: AbstractAttribute[str]

    @abc.abstractmethod
    def open(self) -> "AtomisticTrajectoryReader":
        raise NotImplementedError


class AbstractAtomisticTrajectorySinkPlan(StrictModule, NonTrainableState):
    sink_id: AbstractAttribute[str]

    @abc.abstractmethod
    def open(self, *, append: bool = False) -> "AtomisticTrajectoryWriter":
        raise NotImplementedError


class AtomisticTrajectoryReader(abc.ABC):
    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class AtomisticTrajectoryWriter(abc.ABC):
    @abc.abstractmethod
    def write(self, frame: AtomisticFrame, /) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class InMemoryTrajectorySourcePlan(AbstractAtomisticTrajectorySourcePlan):
    frames: tuple[AtomisticFrame, ...]
    source_id: str = eqx.field(static=True)

    def __init__(self, frames, /):
        values = tuple(frames)
        if any(not isinstance(value, AtomisticFrame) for value in values):
            raise TypeError("frames must contain AtomisticFrame values.")
        self.frames = values
        self.source_id = canonical_fingerprint(
            {
                "kind": "in-memory-trajectory",
                "frames": [value.source_id for value in values],
            }
        )

    def open(self):
        return _InMemoryReader(self.frames)


class _InMemoryReader(AtomisticTrajectoryReader):
    def __init__(self, frames):
        self.frames = frames

    def __iter__(self):
        return iter(self.frames)

    def close(self):
        return None


__all__ = [
    "AbstractAtomisticTrajectorySinkPlan",
    "AbstractAtomisticTrajectorySourcePlan",
    "AtomisticFrame",
    "AtomisticFrameFields",
    "AtomisticMetadata",
    "AtomisticSelectionPlan",
    "AtomisticTrajectoryReader",
    "AtomisticTrajectoryWriter",
    "InMemoryTrajectorySourcePlan",
]
