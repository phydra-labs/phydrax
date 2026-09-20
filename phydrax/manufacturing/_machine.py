#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MachineFrame:
    frame_id: str
    dimension: int

    def __post_init__(self):
        if not self.frame_id or self.dimension not in (2, 3):
            raise ValueError("Machine frame requires ID and 2D/3D dimension.")


@dataclass(frozen=True, slots=True)
class MachineConfiguration:
    machine_id: str
    frame: MachineFrame
    head_count: int = 1

    def __post_init__(self):
        if not self.machine_id or self.head_count <= 0:
            raise ValueError("Machine configuration is invalid.")


__all__ = ["MachineConfiguration", "MachineFrame"]
