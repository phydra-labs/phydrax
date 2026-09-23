#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import math
from dataclasses import dataclass

from ._path import ToolpathEvent


@dataclass(frozen=True, slots=True)
class GCodeProgram:
    events: tuple[ToolpathEvent, ...]


def parse_linear_gcode(
    text: str, /, *, frame_id="machine", default_feed_m_s=0.01
) -> GCodeProgram:
    if not isinstance(text, str):
        raise TypeError("G-code source must be text.")
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("G-code frame identifier must be nonempty.")
    if not math.isfinite(default_feed_m_s) or default_feed_m_s <= 0:
        raise ValueError("Default G-code feed must be finite and positive.")
    position = [0.0, 0.0, 0.0]
    time = 0.0
    events = []
    for index, raw in enumerate(text.splitlines()):
        line = raw.split(";", 1)[0].strip().upper()
        if not line:
            continue
        fields = line.split()
        command = fields[0]
        if command not in ("G0", "G00", "G1", "G01", "G4", "G04"):
            raise ValueError(f"Unsupported G-code command {command!r}.")
        values = {field[0]: float(field[1:]) for field in fields[1:]}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("G-code numeric fields must be finite.")
        if command in ("G4", "G04"):
            duration = values.get("P", 0.0) / 1000.0
            if duration < 0:
                raise ValueError("G-code dwell duration must be nonnegative.")
            events.append(
                ToolpathEvent(
                    f"gcode-{index}",
                    "dwell",
                    time,
                    time + duration,
                    frame_id,
                    tuple(position),
                    tuple(position),
                )
            )
            time += duration
            continue
        target = [
            values.get("X", position[0] * 1000) / 1000,
            values.get("Y", position[1] * 1000) / 1000,
            values.get("Z", position[2] * 1000) / 1000,
        ]
        speed = values.get("F", default_feed_m_s * 60000) / 60000
        distance = sum((a - b) ** 2 for a, b in zip(target, position, strict=True)) ** 0.5
        if speed <= 0:
            raise ValueError("G-code feed speed must be positive.")
        duration = distance / speed
        events.append(
            ToolpathEvent(
                f"gcode-{index}",
                "move",
                time,
                time + duration,
                frame_id,
                tuple(position),
                tuple(target),
            )
        )
        position = target
        time += duration
    return GCodeProgram(tuple(events))


__all__ = ["GCodeProgram", "parse_linear_gcode"]
