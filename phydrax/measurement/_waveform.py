#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Time-resolved waveform supports and normalized pulse responses."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._support import RaySampleSupport
from ._time import SampleTimeAxis


@dataclass(frozen=True, slots=True)
class WaveformSupport:
    rays: RaySampleSupport
    delay_axis: SampleTimeAxis
    receiver_ids: tuple[str, ...] = ("receiver-0",)
    bin_widths: np.ndarray | None = None
    sample_shape: tuple[int, ...] = field(init=False)
    support_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.rays, RaySampleSupport) or not isinstance(
            self.delay_axis, SampleTimeAxis
        ):
            raise TypeError("Waveform support requires ray and delay-axis contracts.")
        receivers = tuple(str(value) for value in self.receiver_ids)
        if (
            not receivers
            or any(not value for value in receivers)
            or len(receivers) != len(set(receivers))
        ):
            raise ValueError("receiver_ids must be nonempty and unique.")
        if self.bin_widths is None:
            times = self.delay_axis.sample_times
            if times.size == 1:
                raise ValueError("A one-bin waveform requires explicit bin_widths.")
            edges = np.empty(times.size + 1)
            edges[1:-1] = 0.5 * (times[:-1] + times[1:])
            edges[0] = times[0] - 0.5 * (times[1] - times[0])
            edges[-1] = times[-1] + 0.5 * (times[-1] - times[-2])
            widths = np.diff(edges)
        else:
            widths = np.array(self.bin_widths, dtype=float, copy=True)
        if (
            widths.shape != (self.delay_axis.sample_count,)
            or not np.all(np.isfinite(widths))
            or np.any(widths <= 0.0)
        ):
            raise ValueError(
                "bin_widths must be finite and positive for every delay bin."
            )
        widths.setflags(write=False)
        shape = (self.rays.sample_shape[0], self.delay_axis.sample_count, len(receivers))
        object.__setattr__(self, "receiver_ids", receivers)
        object.__setattr__(self, "bin_widths", widths)
        object.__setattr__(self, "sample_shape", shape)
        object.__setattr__(
            self,
            "support_id",
            canonical_fingerprint(
                {
                    "kind": "waveform-support",
                    "rays": self.rays.support_id,
                    "delay_axis": self.delay_axis.time_axis_id,
                    "receivers": list(receivers),
                    "bin_widths": array_tree_fingerprint(widths),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class PulseResponse:
    times: np.ndarray
    amplitudes: np.ndarray
    time_unit_id: str
    response_id: str = field(init=False)

    def __post_init__(self) -> None:
        times = np.array(self.times, dtype=float, copy=True)
        amplitudes = np.array(self.amplitudes, dtype=float, copy=True)
        if (
            times.ndim != 1
            or times.shape != amplitudes.shape
            or times.size < 2
            or not np.all(np.diff(times) > 0.0)
        ):
            raise ValueError(
                "Pulse times/amplitudes must be equally sized with increasing times."
            )
        if (
            not np.all(np.isfinite(times))
            or not np.all(np.isfinite(amplitudes))
            or np.any(amplitudes < 0.0)
        ):
            raise ValueError("Pulse response values must be finite and nonnegative.")
        area = float(np.trapezoid(amplitudes, times))
        if area <= 0.0:
            raise ValueError("Pulse response requires positive integrated amplitude.")
        amplitudes /= area
        times.setflags(write=False)
        amplitudes.setflags(write=False)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "amplitudes", amplitudes)
        object.__setattr__(
            self,
            "response_id",
            canonical_fingerprint(
                {
                    "kind": "pulse-response",
                    "times": array_tree_fingerprint(times),
                    "amplitudes": array_tree_fingerprint(amplitudes),
                    "time_unit": self.time_unit_id,
                }
            ),
        )


__all__ = ["PulseResponse", "WaveformSupport"]
