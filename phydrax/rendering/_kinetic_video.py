#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

from .._external_runtime import (
    EnergyRunResult,
    ExternalExecutionPolicy,
    PinnedExecutable,
    run_energy_command,
)


@dataclass(frozen=True, slots=True)
class KineticVideoPlan:
    executable: PinnedExecutable
    frames_per_second: int = 30
    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    timeout_seconds: float = 300.0
    maximum_output_bytes: int = 1 << 30
    execution_policy: ExternalExecutionPolicy | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.executable, PinnedExecutable):
            raise TypeError("executable must be a PinnedExecutable.")
        if self.frames_per_second < 1:
            raise ValueError("frames_per_second must be positive.")
        if not self.codec or not self.pixel_format:
            raise ValueError("codec and pixel_format must be non-empty.")
        if self.timeout_seconds <= 0.0 or self.maximum_output_bytes < 1:
            raise ValueError("Video execution limits must be positive.")

    def encode_png_frames(self, frames: tuple[bytes, ...], /) -> EnergyRunResult:
        if not frames or any(
            not isinstance(frame, bytes) or not frame for frame in frames
        ):
            raise ValueError("frames must contain non-empty PNG byte strings.")
        inputs = {f"frames/{index:08d}.png": frame for index, frame in enumerate(frames)}
        return run_energy_command(
            self.executable,
            (
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(self.frames_per_second),
                "-i",
                "frames/%08d.png",
                "-c:v",
                self.codec,
                "-pix_fmt",
                self.pixel_format,
                "output.mp4",
            ),
            inputs=inputs,
            outputs=("output.mp4",),
            timeout=self.timeout_seconds,
            max_output_bytes=self.maximum_output_bytes,
            execution_policy=self.execution_policy,
        )


__all__ = ["KineticVideoPlan"]
