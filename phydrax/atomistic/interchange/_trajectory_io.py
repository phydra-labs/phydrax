#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import base64
import json
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from .._frame import (
    AbstractAtomisticTrajectorySinkPlan,
    AbstractAtomisticTrajectorySourcePlan,
    AtomisticFrame,
    AtomisticTrajectoryReader,
    AtomisticTrajectoryWriter,
)
from .._sites import AtomisticSiteDomain


def _h5py():
    if find_spec("h5py") is None:
        raise ImportError("H5MD trajectory I/O requires the optional 'h5py' package.")
    return import_module("h5py")


_H5MD_FIELD_PATHS = {
    "time": "position/time",
    "step": "position/step",
    "positions": "position/value",
    "velocities": "velocity/value",
    "momenta": "momentum/value",
    "forces": "force/value",
    "cell_vectors": "box/edges/value",
    "image_counts": "image/value",
    "energy": "observables/energy/value",
    "valid": "observables/valid/value",
    "source_id": "observables/source_id/value",
}


def _h5md_field_path(name: str, /) -> str:
    if name.startswith("auxiliary/"):
        return f"observables/auxiliary/{name.removeprefix('auxiliary/')}/value"
    return _H5MD_FIELD_PATHS[name]


def _decoded_text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class H5MDTrajectoryPlan(
    AbstractAtomisticTrajectorySourcePlan, AbstractAtomisticTrajectorySinkPlan
):
    path: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    sink_id: str = eqx.field(static=True)

    def __init__(self, path: str | Path, /):
        target = str(Path(path))
        self.path = target
        identifier = canonical_fingerprint({"kind": "atomistic-h5md", "path": target})
        self.source_id = identifier
        self.sink_id = identifier

    def open(self, *, append: bool | None = None):
        if append is None:
            return H5MDTrajectoryReader(self.path, self.source_id)
        return H5MDTrajectoryWriter(self.path, self.sink_id, append=append)


class H5MDTrajectoryWriter(AtomisticTrajectoryWriter):
    def __init__(self, path: str, sink_id: str, /, *, append: bool):
        self.h5py = _h5py()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.h5py.File(target, "a" if append else "w")
        self.sink_id = sink_id
        self.group = self.handle.require_group("particles").require_group("phydrax")
        h5md = self.handle.require_group("h5md")
        h5md.attrs["version"] = np.asarray((1, 1), dtype=np.int32)
        creator = h5md.require_group("creator")
        creator.attrs["name"] = "phydrax"
        self.count = int(self.group.attrs.get("committed_frames", 0))
        if self.count == 0:
            for name in tuple(self.group.keys()):
                del self.group[name]
            for name in (
                "system_id",
                "topology_id",
                "unit_system_id",
                "coordinate_domain",
                "frame_fields",
            ):
                if name in self.group.attrs:
                    del self.group.attrs[name]
            self.expected_fields = None
        else:
            datasets = self._frame_datasets()
            if "frame_fields" not in self.group.attrs:
                raise ValueError("H5MD stream lacks its canonical frame-field manifest.")
            self.expected_fields = tuple(json.loads(self.group.attrs["frame_fields"]))
            for name, dataset in datasets:
                if dataset.shape[0] < self.count:
                    raise ValueError(
                        f"H5MD dataset {name!r} ends before the commit boundary."
                    )
                if dataset.shape[0] > self.count:
                    dataset.resize((self.count,) + dataset.shape[1:])

    def _frame_datasets(self):
        result = []

        def collect(name, value):
            if isinstance(value, self.h5py.Dataset) and name != "id":
                result.append((name, value))

        self.group.visititems(collect)
        return tuple(result)

    def _dataset(self, name, shape, dtype):
        fields = _h5md_field_path(name).split("/")
        parent = self.group
        for field in fields[:-1]:
            parent = parent.require_group(field)
        leaf = fields[-1]
        if leaf in parent:
            return parent[leaf]
        return parent.create_dataset(
            leaf, shape=(0,) + shape, maxshape=(None,) + shape, dtype=dtype, chunks=True
        )

    def write(self, frame: AtomisticFrame, /) -> None:
        if not isinstance(frame, AtomisticFrame):
            raise TypeError("frame must be AtomisticFrame.")
        if self.count:
            identities = (
                ("system_id", frame.system_id),
                ("topology_id", frame.topology_id),
                ("unit_system_id", frame.unit_system_id),
                ("coordinate_domain", frame.coordinate_domain.value),
            )
            if any(self.group.attrs[name] != value for name, value in identities):
                raise ValueError("Cannot append a frame with incompatible identities.")
            if not np.array_equal(
                np.asarray(self.group["id"]), np.asarray(frame.stable_ids)
            ):
                raise ValueError("Cannot append a frame with different stable IDs.")
        else:
            self.group.attrs["system_id"] = frame.system_id
            self.group.attrs["topology_id"] = frame.topology_id
            self.group.attrs["unit_system_id"] = frame.unit_system_id
            self.group.attrs["coordinate_domain"] = frame.coordinate_domain.value
            self.group.create_dataset("id", data=np.asarray(frame.stable_ids))
            position_group = self.group.require_group("position")
            position_group.attrs["dimension"] = 3
            if frame.cell_vectors is not None:
                box = self.group.require_group("box")
                box.attrs["dimension"] = 3
                box.attrs["boundary"] = np.asarray(
                    ("periodic",) * 3,
                    dtype=self.h5py.string_dtype(encoding="utf-8"),
                )
        values = {
            "time": np.asarray(frame.time),
            "step": np.asarray(frame.step),
            "positions": np.asarray(frame.positions),
            "valid": np.asarray(frame.valid),
            "source_id": np.asarray(
                frame.source_id, dtype=self.h5py.string_dtype(encoding="utf-8")
            ),
        }
        for name, value in (
            ("velocities", frame.velocities),
            ("momenta", frame.momenta),
            ("forces", frame.forces),
            ("cell_vectors", frame.cell_vectors),
            ("image_counts", frame.image_counts),
            ("energy", frame.energy),
        ):
            if value is not None:
                values[name] = np.asarray(value)
        for name, value in frame.auxiliary.items():
            values[f"auxiliary/{name}"] = np.asarray(value)
        fields = tuple(sorted(values))
        if self.expected_fields is None:
            self.expected_fields = fields
            self.group.attrs["frame_fields"] = json.dumps(fields, separators=(",", ":"))
        elif fields != self.expected_fields:
            raise ValueError("H5MD frame fields changed during append.")
        for name, value in values.items():
            dataset = self._dataset(name, value.shape, value.dtype)
            if dataset.shape[0] != self.count:
                raise ValueError("H5MD frame fields changed during append.")
            dataset.resize((self.count + 1,) + dataset.shape[1:])
            dataset[self.count] = value
        self.handle.flush()
        self.count += 1
        self.group.attrs["committed_frames"] = self.count
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class H5MDTrajectoryReader(AtomisticTrajectoryReader):
    def __init__(self, path: str, source_id: str, /):
        self.handle = _h5py().File(path, "r")
        if not np.array_equal(
            np.asarray(self.handle["h5md"].attrs["version"]), np.asarray((1, 1))
        ):
            self.handle.close()
            raise ValueError("Unsupported H5MD version.")
        self.group = self.handle["particles/phydrax"]
        self.count = int(self.group.attrs["committed_frames"])
        self.source_id = source_id

    def __iter__(self):
        group = self.group
        ids = np.asarray(group["id"])
        auxiliary_names = tuple(name for name in group.get("observables/auxiliary", {}))
        for index in range(self.count):
            optional = lambda name: (
                None
                if _h5md_field_path(name) not in group
                else np.asarray(group[_h5md_field_path(name)][index])
            )
            auxiliary = {
                name: np.asarray(group[f"observables/auxiliary/{name}/value"][index])
                for name in auxiliary_names
            }
            yield AtomisticFrame(
                group[_h5md_field_path("time")][index],
                group[_h5md_field_path("step")][index],
                group[_h5md_field_path("positions")][index],
                ids,
                velocities=optional("velocities"),
                momenta=optional("momenta"),
                forces=optional("forces"),
                cell_vectors=optional("cell_vectors"),
                image_counts=optional("image_counts"),
                energy=optional("energy"),
                auxiliary=auxiliary,
                valid=group[_h5md_field_path("valid")][index],
                coordinate_domain=AtomisticSiteDomain(group.attrs["coordinate_domain"]),
                system_id=group.attrs["system_id"],
                topology_id=group.attrs["topology_id"],
                unit_system_id=group.attrs["unit_system_id"],
                source_id=_decoded_text(group[_h5md_field_path("source_id")][index]),
            )

    def close(self) -> None:
        self.handle.close()


class ExtendedXYZTrajectoryPlan(
    AbstractAtomisticTrajectorySourcePlan, AbstractAtomisticTrajectorySinkPlan
):
    path: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    sink_id: str = eqx.field(static=True)

    def __init__(self, path: str | Path, /):
        target = str(Path(path))
        self.path = target
        identifier = canonical_fingerprint(
            {"kind": "atomistic-extended-xyz", "path": target}
        )
        self.source_id = identifier
        self.sink_id = identifier

    def open(self, *, append: bool | None = None):
        if append is None:
            return ExtendedXYZTrajectoryReader(self.path, self.source_id)
        return ExtendedXYZTrajectoryWriter(self.path, self.sink_id, append=append)


class ExtendedXYZTrajectoryWriter(AtomisticTrajectoryWriter):
    def __init__(self, path: str, sink_id: str, /, *, append: bool):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(target, "a" if append else "w", encoding="utf-8")
        self.sink_id = sink_id

    def write(self, frame: AtomisticFrame, /) -> None:
        if not isinstance(frame, AtomisticFrame):
            raise TypeError("frame must be AtomisticFrame.")
        position = np.asarray(frame.positions)
        metadata = {
            "time": float(frame.time),
            "step": int(frame.step),
            "system_id": frame.system_id,
            "topology_id": frame.topology_id,
            "unit_system_id": frame.unit_system_id,
            "coordinate_domain": frame.coordinate_domain.value,
            "source_id": frame.source_id,
            "stable_ids": np.asarray(frame.stable_ids).tolist(),
            "velocities": None
            if frame.velocities is None
            else np.asarray(frame.velocities).tolist(),
            "momenta": None
            if frame.momenta is None
            else np.asarray(frame.momenta).tolist(),
            "forces": None if frame.forces is None else np.asarray(frame.forces).tolist(),
            "cell": None
            if frame.cell_vectors is None
            else np.asarray(frame.cell_vectors).tolist(),
            "image_counts": None
            if frame.image_counts is None
            else np.asarray(frame.image_counts).tolist(),
            "energy": None if frame.energy is None else np.asarray(frame.energy).tolist(),
            "auxiliary": {
                name: np.asarray(value).tolist()
                for name, value in frame.auxiliary.items()
            },
            "valid": bool(frame.valid),
        }
        encoded = base64.urlsafe_b64encode(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        self.handle.write(f"{position.shape[0]}\n")
        self.handle.write(f"Properties=species:S:1:pos:R:3 phydrax_json={encoded}\n")
        for coordinate in position:
            self.handle.write(
                f"X {coordinate[0]:.17g} {coordinate[1]:.17g} {coordinate[2]:.17g}\n"
            )
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class ExtendedXYZTrajectoryReader(AtomisticTrajectoryReader):
    def __init__(self, path: str, source_id: str, /):
        self.handle = open(path, encoding="utf-8")
        self.source_id = source_id

    def __iter__(self):
        while True:
            count_line = self.handle.readline()
            if not count_line:
                break
            count = int(count_line)
            header = self.handle.readline().split()
            fields_by_name = {
                key: value
                for token in header
                if "=" in token
                for key, value in (token.split("=", maxsplit=1),)
            }
            if (
                fields_by_name.get("Properties") != "species:S:1:pos:R:3"
                or "phydrax_json" not in fields_by_name
            ):
                raise ValueError("Extended XYZ header is missing PhydraX metadata.")
            metadata = json.loads(
                base64.urlsafe_b64decode(fields_by_name["phydrax_json"].encode("ascii"))
            )
            position = np.zeros((count, 3), dtype=float)
            for index in range(count):
                fields = self.handle.readline().split()
                if len(fields) != 4:
                    raise ValueError("Extended XYZ atom row has the wrong arity.")
                position[index] = tuple(float(value) for value in fields[1:4])
            yield AtomisticFrame(
                metadata["time"],
                metadata["step"],
                position,
                metadata["stable_ids"],
                velocities=metadata["velocities"],
                momenta=metadata["momenta"],
                forces=metadata["forces"],
                cell_vectors=metadata["cell"],
                image_counts=metadata["image_counts"],
                energy=metadata["energy"],
                auxiliary=metadata["auxiliary"],
                valid=metadata["valid"],
                coordinate_domain=AtomisticSiteDomain(metadata["coordinate_domain"]),
                system_id=metadata["system_id"],
                topology_id=metadata["topology_id"],
                unit_system_id=metadata["unit_system_id"],
                source_id=metadata["source_id"],
            )

    def close(self) -> None:
        self.handle.close()


__all__ = ["ExtendedXYZTrajectoryPlan", "H5MDTrajectoryPlan"]
