#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._frame import AtomisticFrame


class PackmolRegionConstraint(StrictModule, NonTrainableState):
    kind: str = eqx.field(static=True)
    parameters: tuple[float, ...] = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    def __init__(self, kind: str, parameters, /):
        if kind not in (
            "inside-box",
            "inside-sphere",
            "outside-sphere",
            "inside-cylinder",
        ):
            raise ValueError("Unsupported PACKMOL region constraint.")
        values = tuple(float(value) for value in parameters)
        expected = {
            "inside-box": 6,
            "inside-sphere": 4,
            "outside-sphere": 4,
            "inside-cylinder": 7,
        }
        if (
            len(values) != expected[kind]
            or not np.isfinite(values).all()
            or (
                kind in ("inside-sphere", "outside-sphere", "inside-cylinder")
                and values[-1] <= 0.0
            )
        ):
            raise ValueError("PACKMOL region parameters are invalid.")
        self.kind = kind
        self.parameters = values
        self.constraint_id = canonical_fingerprint(
            {"kind": "packmol-region", "region_kind": kind, "parameters": list(values)}
        )

    def render(self) -> str:
        keyword = self.kind.replace("-", " ")
        return keyword + " " + " ".join(f"{value:.17g}" for value in self.parameters)


class PackmolComponentPlan(StrictModule, NonTrainableState):
    template: AtomisticFrame
    count: int = eqx.field(static=True)
    constraints: tuple[PackmolRegionConstraint, ...]
    component_id: str = eqx.field(static=True)

    def __init__(self, template: AtomisticFrame, count: int, /, *, constraints=()):
        if not isinstance(template, AtomisticFrame):
            raise TypeError("template must be AtomisticFrame.")
        count_ = int(count)
        values = tuple(constraints)
        if count_ <= 0 or any(
            not isinstance(value, PackmolRegionConstraint) for value in values
        ):
            raise ValueError("PACKMOL component count or constraints are invalid.")
        self.template = template
        self.count = count_
        self.constraints = values
        self.component_id = canonical_fingerprint(
            {
                "kind": "packmol-component",
                "template": template.source_id,
                "count": count_,
                "constraints": [value.constraint_id for value in values],
            }
        )


class PackmolAssemblyPlan(StrictModule, NonTrainableState):
    components: tuple[PackmolComponentPlan, ...]
    tolerance: float = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    executable: str = eqx.field(static=True)
    timeout: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        components,
        /,
        *,
        tolerance: float = 2.0,
        seed: int = 0,
        executable: str = "packmol",
        timeout: float = 300.0,
    ):
        values = tuple(components)
        if not values or any(
            not isinstance(value, PackmolComponentPlan) for value in values
        ):
            raise TypeError("components must contain PackmolComponentPlan values.")
        tolerance_ = float(tolerance)
        timeout_ = float(timeout)
        seed_ = int(seed)
        executable_ = str(executable).strip()
        if tolerance_ <= 0.0 or timeout_ <= 0.0 or seed_ < 0 or not executable_:
            raise ValueError(
                "PACKMOL tolerance, timeout, seed, or executable is invalid."
            )
        self.components = values
        self.tolerance = tolerance_
        self.seed = seed_
        self.executable = executable_
        self.timeout = timeout_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "packmol-assembly",
                "components": [value.component_id for value in values],
                "tolerance": tolerance_,
                "seed": self.seed,
                "executable": self.executable,
                "timeout": timeout_,
            }
        )

    def run(self, /) -> "PackmolAssemblyResult":
        with tempfile.TemporaryDirectory(prefix="phydrax-packmol-") as directory:
            root = Path(directory)
            lines = [
                f"tolerance {self.tolerance:.17g}",
                "filetype xyz",
                "output output.xyz",
                f"seed {self.seed}",
            ]
            stable_ids = []
            molecule_ids = []
            component_slices = []
            atom_offset = 0
            component_start = 0
            molecule_offset = 0
            for component_index, component in enumerate(self.components):
                template_path = root / f"component-{component_index}.xyz"
                positions = np.asarray(component.template.positions)
                with template_path.open("w", encoding="utf-8") as handle:
                    handle.write(f"{positions.shape[0]}\ncomponent\n")
                    for point in positions:
                        handle.write(
                            f"X {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}\n"
                        )
                lines.extend(
                    (f"structure {template_path.name}", f"  number {component.count}")
                )
                lines.extend(
                    f"  {constraint.render()}" for constraint in component.constraints
                )
                lines.append("end structure")
                for molecule in range(component.count):
                    stable_ids.extend(
                        range(atom_offset, atom_offset + positions.shape[0])
                    )
                    molecule_ids.extend([molecule_offset + molecule] * positions.shape[0])
                    atom_offset += positions.shape[0]
                molecule_offset += component.count
                component_stop = component_start + component.count * positions.shape[0]
                component_slices.append((component_start, component_stop))
                component_start = component_stop
            payload = "\n".join(lines) + "\n"
            input_digest = canonical_fingerprint(
                {"kind": "packmol-input", "payload": payload}
            )
            completed = subprocess.run(
                [self.executable],
                cwd=root,
                input=payload,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"PACKMOL failed with exit code {completed.returncode}: {completed.stderr}"
                )
            output = root / "output.xyz"
            with output.open(encoding="utf-8") as handle:
                count = int(handle.readline())
                handle.readline()
                positions = np.zeros((count, 3), dtype=float)
                for index in range(count):
                    fields = handle.readline().split()
                    positions[index] = tuple(float(value) for value in fields[1:4])
        minimum = np.inf
        if positions.shape[0] > 1:
            displacement = positions[:, None, :] - positions[None, :, :]
            distance = np.sqrt(np.sum(displacement * displacement, axis=-1))
            distance = np.where(np.eye(positions.shape[0], dtype=bool), np.inf, distance)
            minimum = float(np.min(distance))
        successful = (
            completed.returncode == 0
            and positions.shape[0] == len(stable_ids)
            and minimum > 0.0
        )
        return PackmolAssemblyResult(
            positions,
            np.asarray(stable_ids),
            np.asarray(molecule_ids),
            minimum,
            completed.stdout,
            completed.stderr,
            successful,
            input_digest,
            self.executable,
            tuple(component_slices),
            self.plan_id,
        )


class PackmolAssemblyResult(StrictModule, NonTrainableState):
    positions: np.ndarray
    stable_ids: np.ndarray
    molecule_ids: np.ndarray
    minimum_distance: float = eqx.field(static=True)
    stdout: str = eqx.field(static=True)
    stderr: str = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    input_digest: str = eqx.field(static=True)
    executable: str = eqx.field(static=True)
    component_slices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


__all__ = [
    "PackmolAssemblyPlan",
    "PackmolAssemblyResult",
    "PackmolComponentPlan",
    "PackmolRegionConstraint",
]
