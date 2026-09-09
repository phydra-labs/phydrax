#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Pinned, host-only XFOIL 6.9/6.99 viscous operating points.

The actual XFOIL command language and PACC files are used, not a hypothetical
engine JSON protocol. Each point starts a new process and boundary layer. PACC
writes a viscous point only when LVCONV is true (upstream src/xoper.f). Missing
points remain missing physics, never NaNs or optimization penalties.

References: https://web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt and
https://github.com/RobotLocomotion/xfoil/blob/master/src/xoper.f .
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .._fingerprint import canonical_fingerprint
from ..artifacts import ScientificArtifactEnvelope
from .energy_runtime import (
    _artifact,
    _host_only,
    EnergyRunResult,
    PinnedExecutable,
    run_energy_command,
)


@dataclass(frozen=True, slots=True)
class XFOILOperatingPoint:
    """Fixed-Reynolds, fixed-Mach viscous analysis; alpha is in degrees."""

    alpha: float
    reynolds: float
    mach: float = 0.0
    ncrit: float = 9.0
    transition_top: float = 1.0
    transition_bottom: float = 1.0

    def __post_init__(self) -> None:
        _host_only(
            self.alpha,
            self.reynolds,
            self.mach,
            self.ncrit,
            self.transition_top,
            self.transition_bottom,
        )
        if not all(
            math.isfinite(v)
            for v in (
                self.alpha,
                self.reynolds,
                self.mach,
                self.ncrit,
                self.transition_top,
                self.transition_bottom,
            )
        ):
            raise ValueError("Operating-point values must be finite.")
        if self.reynolds <= 0 or not 0 <= self.mach < 1 or self.ncrit <= 0:
            raise ValueError("Require Re > 0, 0 <= Mach < 1, and ncrit > 0.")
        if not -180 < self.alpha < 180:
            raise ValueError("alpha must lie strictly between -180 and 180 degrees.")
        if not all(0 < x <= 1 for x in (self.transition_top, self.transition_bottom)):
            raise ValueError("Transition locations must lie in (0, 1].")


@dataclass(frozen=True, slots=True)
class XFOILCoefficients:
    alpha: float
    lift: float
    drag: float
    pressure_drag: float
    moment: float
    transition_top: float
    transition_bottom: float


@dataclass(frozen=True, slots=True)
class XFOILPointResult:
    operating_point: XFOILOperatingPoint
    coefficients: XFOILCoefficients | None
    geometry_sha256: str
    converged: bool
    failure_reason: str
    run: EnergyRunResult
    artifact: ScientificArtifactEnvelope

    def require_convergence(self) -> XFOILPointResult:
        if not self.converged:
            raise XFOILConvergenceError(self)
        return self


class XFOILConvergenceError(RuntimeError):
    def __init__(self, result: XFOILPointResult):
        self.result = result
        super().__init__(result.failure_reason)


@dataclass(frozen=True, slots=True)
class XFOILPolarResult:
    """Independently initialized points, in request order, including failures."""

    points: tuple[XFOILPointResult, ...]
    artifact: ScientificArtifactEnvelope

    @property
    def converged(self) -> bool:
        return all(point.converged for point in self.points)


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
_COLUMNS = ("alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr")
_ITERATION_COLUMNS = ("Top_Itr", "Bot_Itr")


def _number(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def _parse_polar(
    data: bytes,
    point: XFOILOperatingPoint,
    *,
    version: str | None = None,
) -> XFOILCoefficients | None:
    """Check native, rounded header values before accepting a converged PACC row."""
    text = data.decode("ascii")
    release = re.search(r"XFOIL\s+Version\s+(6\.(?:90?|99))\b", text)
    if release is None:
        raise ValueError("Expected an XFOIL 6.9/6.99 polar header.")
    if version is not None and float(release[1]) != float(version):
        raise ValueError("XFOIL reported a different version than the executable pin.")
    if "Reynolds number fixed" not in text or "Mach number fixed" not in text:
        raise ValueError("Only type-1, fixed-Re/Mach polars are supported.")
    header = re.search(
        rf"Mach\s*=\s*({_NUMBER})\s+Re\s*=\s*({_NUMBER})\s+e\s*6\s+Ncrit\s*=\s*({_NUMBER})",
        text,
    )
    trip = re.search(
        rf"xtrf\s*=\s*({_NUMBER})\s*\(top\)\s*({_NUMBER})\s*\(bottom\)", text
    )
    if header is None or trip is None:
        raise ValueError("XFOIL polar is missing operating-condition evidence.")
    observed = tuple(_number(v) for v in (*header.groups(), *trip.groups()))
    expected = (
        point.mach,
        point.reynolds / 1e6,
        point.ncrit,
        point.transition_top,
        point.transition_bottom,
    )
    # POLWRIT prints these header fields to three decimals. Exact requested
    # values are retained separately in the artifact and the command transcript.
    if any(
        not math.isfinite(v) or abs(v - e) > 0.000501 for v, e in zip(observed, expected)
    ):
        raise ValueError("XFOIL polar operating conditions do not match the request.")
    lines = text.splitlines()
    headers = []
    for index, line in enumerate(lines):
        columns = tuple(line.split())
        if columns in (_COLUMNS, _COLUMNS + _ITERATION_COLUMNS):
            headers.append((index, columns))
    if len(headers) != 1:
        raise ValueError("Expected the standard XFOIL polar columns.")
    header_index, columns = headers[0]
    rows = []
    for line in lines[header_index + 1 :]:
        fields = line.split()
        if not fields or all(set(field) == {"-"} for field in fields):
            continue
        if len(fields) != len(columns):
            raise ValueError("Malformed XFOIL polar row.")
        values = tuple(_number(field) for field in fields)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite XFOIL coefficients are not physical results.")
        base = values[: len(_COLUMNS)]
        if abs(base[0] - point.alpha) > 0.000501:
            raise ValueError("XFOIL returned a different angle of attack.")
        if base[2] < 0 or not all(0 <= v <= 1 for v in base[5:]):
            raise ValueError("Invalid drag or transition positions in XFOIL polar.")
        rows.append(XFOILCoefficients(*base))
    if len(rows) > 1:
        raise ValueError("An independent operating-point run produced multiple rows.")
    return rows[0] if rows else None


def _geometry_bytes(coordinates: Sequence[Sequence[float]]) -> bytes:
    _host_only(coordinates)
    points = tuple(tuple(float(v) for v in point) for point in coordinates)
    if not 20 <= len(points) <= 240 or any(len(p) != 2 for p in points):
        raise ValueError("Geometry requires 20–240 ordered (x, y) surface nodes.")
    if not all(math.isfinite(v) for p in points for v in p):
        raise ValueError("Airfoil geometry must be finite.")
    if any(p == q for p, q in zip(points, points[1:])):
        raise ValueError("Consecutive duplicate surface nodes are invalid.")
    xs = [p[0] for p in points]
    if abs(min(xs)) > 1e-8 or abs(max(xs) - 1) > 1e-8:
        raise ValueError(
            "Supply unit-chord geometry with leading edge x=0 and trailing edge x=1."
        )
    if abs(points[0][0] - 1) > 1e-8 or abs(points[-1][0] - 1) > 1e-8:
        raise ValueError(
            "Order nodes from trailing edge around leading edge to trailing edge."
        )
    return (
        "phydrax-airfoil\n" + "".join(f"{x:.17g} {y:.17g}\n" for x, y in points)
    ).encode("ascii")


def run_xfoil_point(
    executable: PinnedExecutable,
    coordinates: Sequence[Sequence[float]],
    operating_point: XFOILOperatingPoint,
    *,
    iterations: int = 200,
    timeout: float = 60,
    max_output_bytes: int = 8 * 1024 * 1024,
) -> XFOILPointResult:
    """Launch real XFOIL; do not re-panel, normalize, warm-start, or retry.

    The pin covers executable bytes, not its dynamic libraries. The inherited
    process environment and engine remain trusted, not sandboxed. The existing
    bounded runtime provides private paths, process-group timeout and log/output
    limits. Failure to launch/collect raises EnergyRuntimeError with evidence.
    A clean process exit alone is never treated as viscous convergence.
    """
    _host_only(coordinates, iterations, timeout)
    if executable.version not in ("6.9", "6.99"):
        raise ValueError("This command profile supports XFOIL 6.9 and 6.99 only.")
    if type(iterations) is not int or not 1 <= iterations <= 10000:
        raise ValueError("iterations must be an integer in [1, 10000].")
    geometry = _geometry_bytes(coordinates)
    p = operating_point
    commands = (
        "PLOP",
        "G",
        "",
        "LOAD airfoil.dat",
        "PCOP",
        "PSAV realized.dat",
        "OPER",
        "TYPE 1",
        f"VISC {p.reynolds:.17g}",
        f"MACH {p.mach:.17g}",
        "VPAR",
        f"N {p.ncrit:.17g}",
        f"XTR {p.transition_top:.17g} {p.transition_bottom:.17g}",
        "",
        "ITER",
        str(iterations),
        "PACC",
        "",
        "",
        f"ALFA {p.alpha:.17g}",
        "PWRT",
        "polar.dat",
        "PACC",
        "",
        "QUIT",
        "",
    )
    run = run_energy_command(
        executable,
        (),
        inputs={"airfoil.dat": geometry},
        outputs=("polar.dat", "realized.dat"),
        stdin="\n".join(commands).encode("ascii"),
        timeout=timeout,
        max_output_bytes=max_output_bytes,
        environment={"LC_ALL": "C"},
    )
    realized = tuple(
        tuple(_number(v) for v in line.split())
        for line in run.output("realized.dat").decode("ascii").splitlines()
        if line.strip()
    )
    declared = tuple(tuple(float(v) for v in point) for point in coordinates)
    if len(realized) != len(declared) or any(
        len(actual) != 2
        or any(not math.isfinite(a) or abs(a - b) > 5e-7 for a, b in zip(actual, wanted))
        for actual, wanted in zip(realized, declared)
    ):
        raise ValueError("XFOIL changed the declared surface nodes.")
    coefficients = _parse_polar(run.output("polar.dat"), p, version=executable.version)
    failure = (
        ""
        if coefficients is not None
        else "XFOIL did not converge this viscous operating point; PACC contains no row."
    )
    geometry_sha256 = hashlib.sha256(geometry).hexdigest()
    artifact = _artifact(
        "xfoil-operating-point",
        {
            "run": run.artifact.artifact_id,
            "geometry_sha256": geometry_sha256,
            "converged": not failure,
            "alpha": p.alpha,
            "reynolds": p.reynolds,
            "mach": p.mach,
            "ncrit": p.ncrit,
            "transition_top": p.transition_top,
            "transition_bottom": p.transition_bottom,
            "iterations": iterations,
            "geometry_mode": "PCOP; no re-paneling",
        },
        producer="XFOIL",
        version=executable.version,
        build_id=executable.sha256,
        license_id=executable.license_id,
        resource_id=run.artifact.resource_id,
        error=failure,
        parents=(run.artifact.artifact_id,),
    )
    return XFOILPointResult(
        p, coefficients, geometry_sha256, not failure, failure, run, artifact
    )


def run_xfoil_polar(
    executable: PinnedExecutable,
    coordinates: Sequence[Sequence[float]],
    operating_points: Sequence[XFOILOperatingPoint],
    *,
    iterations: int = 200,
    timeout_per_point: float = 60,
    max_output_bytes: int = 8 * 1024 * 1024,
) -> XFOILPolarResult:
    """Run at most 256 independent points; no continuation across failed states."""
    _host_only(coordinates, iterations, timeout_per_point)
    if not 1 <= len(operating_points) <= 256:
        raise ValueError("A polar requires between 1 and 256 operating points.")
    points = tuple(
        run_xfoil_point(
            executable,
            coordinates,
            point,
            iterations=iterations,
            timeout=timeout_per_point,
            max_output_bytes=max_output_bytes,
        )
        for point in operating_points
    )
    parents = tuple(point.artifact.artifact_id for point in points)
    artifact = _artifact(
        "xfoil-independent-polar",
        parents,
        producer="XFOIL",
        version=executable.version,
        build_id=executable.sha256,
        license_id=executable.license_id,
        resource_id=canonical_fingerprint(parents),
        parents=parents,
        error=""
        if all(point.converged for point in points)
        else "One or more operating points did not converge.",
    )
    return XFOILPolarResult(points, artifact)


__all__ = [
    "XFOILCoefficients",
    "XFOILConvergenceError",
    "XFOILOperatingPoint",
    "XFOILPointResult",
    "XFOILPolarResult",
    "run_xfoil_point",
    "run_xfoil_polar",
]
