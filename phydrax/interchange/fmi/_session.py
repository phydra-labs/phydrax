#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Bounded FMI 2.0 synchronous Co-Simulation through optional FMPy."""

from __future__ import annotations

import io
import math
import re
import uuid
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..._fingerprint import canonical_fingerprint
from ...artifacts import ScientificArtifactEnvelope
from .._energy_worker import _archive_members, _DEFAULT_BYTES
from .._resource import read_bounded_resource
from ..energy_runtime import (
    _artifact,
    _host_only,
    _HostWorker,
    _limits,
    _require_optional,
)


@dataclass(frozen=True, slots=True)
class FMIVariable:
    name: str
    value_reference: int
    type: str
    causality: str
    variability: str
    unit: str
    initial: str


@dataclass(frozen=True, slots=True)
class FMIModelDescription:
    model_name: str
    guid: str
    model_identifier: str
    variables: tuple[FMIVariable, ...]
    can_get_set_state: bool
    can_serialize_state: bool
    variable_step: bool
    archive_sha256: str

    def variable(self, name: str) -> FMIVariable:
        for variable in self.variables:
            if variable.name == name:
                return variable
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class FMIStepResult:
    requested_time: float
    reached_time: float
    status: str
    early_return: bool
    terminated: bool
    event_handling: str
    artifact: ScientificArtifactEnvelope


@dataclass(frozen=True, slots=True)
class FMIState:
    """Owned native state token, valid only in its creating live session."""

    session_id: str
    token: int
    time: float


def _xml_description(data: bytes, digest: str) -> FMIModelDescription:
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("modelDescription.xml exceeds 4 MiB.")
    text = data.decode("utf-8-sig")
    if "<!DOCTYPE" in text or "<!ENTITY" in text:
        raise ValueError("FMU XML document types and entity declarations are forbidden.")
    # UTF-8 is the intentionally closed XML encoding subset; accepting UTF-16
    # here would bypass the declaration checks above.
    declaration = re.match(r"\s*<\?xml[^>]*encoding\s*=\s*['\"]([^'\"]+)", text)
    if declaration and declaration.group(1).lower() not in ("utf-8", "utf8"):
        raise ValueError("FMU model descriptions must use UTF-8.")
    depth = nodes = attributes = 0
    parser = ET.iterparse(io.StringIO(text), events=("start", "end"))
    for event, element in parser:
        if event == "start":
            depth += 1
            nodes += 1
            attributes += len(element.attrib)
            if depth > 32 or nodes > 100000 or attributes > 200000:
                raise ValueError("FMU model description exceeds XML structural bounds.")
        else:
            depth -= 1
    root = parser.root
    if root.tag != "fmiModelDescription" or root.get("fmiVersion") != "2.0":
        raise ValueError("Only FMI 2.0 synchronous Co-Simulation is supported.")
    co_simulation = root.find("CoSimulation")
    if co_simulation is None:
        raise ValueError(
            "A CoSimulation declaration is required; Model Exchange is unsupported."
        )

    def flag(name: str) -> bool:
        value = co_simulation.get(name, "false")
        if value not in ("true", "false", "1", "0"):
            raise ValueError(f"Invalid FMI Boolean capability {name!r}.")
        return value in ("true", "1")

    if flag("canRunAsynchronuously"):
        raise ValueError("Asynchronous FMI 2.0 doStep is outside the supported subset.")
    identifier = co_simulation.get("modelIdentifier", "")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError("Invalid FMI modelIdentifier.")
    model_name, guid = root.get("modelName", ""), root.get("guid", "")
    if not model_name or not guid:
        raise ValueError("FMU modelName and guid must be declared.")
    declared_units = {}
    for simple in root.findall("./TypeDefinitions/SimpleType"):
        real = simple.find("Real")
        if real is not None:
            declared_units[simple.get("name")] = real.get("unit", "")
    variables = []
    names = set()
    container = root.find("ModelVariables")
    if container is None:
        raise ValueError("FMU model description lacks ModelVariables.")
    for scalar in container:
        if scalar.tag != "ScalarVariable" or len(scalar) != 1:
            raise ValueError(
                "Each FMI2 ScalarVariable must declare exactly one scalar type."
            )
        name = scalar.get("name", "")
        if not name or name in names:
            raise ValueError("FMU variable names must be nonempty and unique.")
        names.add(name)
        typed = scalar[0]
        if typed.tag not in ("Real", "Integer", "Boolean", "String", "Enumeration"):
            raise ValueError(f"Unsupported FMI2 scalar type {typed.tag!r}.")
        reference = int(scalar.attrib["valueReference"])
        if reference < 0 or reference >= 2**32:
            raise ValueError("FMI value references must be unsigned 32-bit integers.")
        causality = scalar.get("causality", "local")
        variability = scalar.get("variability", "continuous")
        if causality not in (
            "input",
            "output",
            "parameter",
            "calculatedParameter",
            "local",
            "independent",
        ):
            raise ValueError(f"Invalid FMI causality {causality!r}.")
        if variability not in ("constant", "fixed", "tunable", "discrete", "continuous"):
            raise ValueError(f"Invalid FMI variability {variability!r}.")
        variables.append(
            FMIVariable(
                name,
                reference,
                typed.tag,
                causality,
                variability,
                typed.get("unit", declared_units.get(typed.get("declaredType"), "")),
                scalar.get("initial", ""),
            )
        )
    return FMIModelDescription(
        model_name,
        guid,
        identifier,
        tuple(variables),
        flag("canGetAndSetFMUstate"),
        flag("canSerializeFMUstate"),
        flag("canHandleVariableCommunicationStepSize"),
        digest,
    )


def inspect_fmu(
    path: str | Path,
    *,
    sha256: str,
    trusted_root: str | Path,
    max_archive_bytes: int = 64 * 1024 * 1024,
    max_unpacked_bytes: int = 256 * 1024 * 1024,
    max_files: int = 4096,
) -> FMIModelDescription:
    """Inspect bounded exact FMU bytes without importing a runtime or loading code."""
    _host_only()
    resource = read_bounded_resource(
        path, trusted_root=trusted_root, limits=_limits(max_archive_bytes)
    )
    if resource.manifest.content_sha256 != sha256:
        raise ValueError("FMU archive SHA-256 does not match its required pin.")
    with zipfile.ZipFile(io.BytesIO(resource.data)) as archive:
        _archive_members(archive, max_unpacked_bytes, max_files)
        if archive.getinfo("modelDescription.xml").file_size > 4 * 1024 * 1024:
            raise ValueError("modelDescription.xml exceeds 4 MiB.")
        return _xml_description(archive.read("modelDescription.xml"), sha256)


def _scalar_value(variable: FMIVariable, value: Any) -> Any:
    kind = variable.type
    if kind == "Real":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"FMI Real {variable.name!r} requires a finite number.")
        return float(value)
    if kind in ("Integer", "Enumeration"):
        if type(value) is not int or not -(2**31) <= value < 2**31:
            raise ValueError(f"FMI Integer {variable.name!r} requires an int32 value.")
    elif kind == "Boolean" and type(value) is not bool:
        raise TypeError(
            f"FMI Boolean {variable.name!r} requires bool, not truthiness conversion."
        )
    elif kind == "String" and (not isinstance(value, str) or "\x00" in value):
        raise TypeError(f"FMI String {variable.name!r} requires a NUL-free string.")
    return value


class FMICoSimulationSession:
    """Owned, timeout-bounded FMI2 session; use as a context manager.

    Supports scalar get/set and synchronous steps. FMI2 internal events remain
    the FMU's responsibility. Model Exchange, FMI3 clocks/event mode, pending
    asynchronous steps, serialization and derivatives are not exposed.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        sha256: str,
        trusted_root: str | Path,
        license_id: str,
        start_time: float = 0,
        stop_time: float | None = None,
        start_values: Mapping[str, Any] | None = None,
        timeout: float = 30,
        expected_fmpy_version: str | None = None,
        source_url: str = "",
        max_archive_bytes: int = 64 * 1024 * 1024,
        max_unpacked_bytes: int = 256 * 1024 * 1024,
        max_files: int = 4096,
    ):
        _host_only(start_time, stop_time, start_values)
        if not license_id.strip():
            raise ValueError("An explicit FMU license_id is required.")
        start_time = float(start_time)
        if not math.isfinite(start_time) or (
            stop_time is not None
            and (not math.isfinite(stop_time) or stop_time <= start_time)
        ):
            raise ValueError(
                "FMI experiment start/stop times must be finite and ordered."
            )
        self.model = inspect_fmu(
            path,
            sha256=sha256,
            trusted_root=trusted_root,
            max_archive_bytes=max_archive_bytes,
            max_unpacked_bytes=max_unpacked_bytes,
            max_files=max_files,
        )
        _require_optional(
            "fmpy",
            "install FMPy and supply an FMI 2.0 Co-Simulation FMU with a host-compatible binary",
        )
        values = self._validated_values(start_values or {}, initializing=True)
        resource = read_bounded_resource(
            path, trusted_root=trusted_root, limits=_limits(max_archive_bytes)
        )
        if resource.manifest.content_sha256 != sha256:
            raise ValueError("FMU changed between inspection and session preparation.")
        self.session_id = uuid.uuid4().hex
        self.time = start_time
        self.terminated = False
        self._license = license_id
        self._resource_id = canonical_fingerprint(
            {
                "archive": sha256,
                "source_url": source_url,
                "start_time": start_time,
                "stop_time": stop_time,
                "start_values": values,
            }
        )
        self._worker = _HostWorker(
            "fmi",
            {
                "description": asdict(self.model),
                "start_time": start_time,
                "stop_time": stop_time,
                "start_values": values,
                "max_unpacked_bytes": max_unpacked_bytes,
                "max_files": max_files,
                "expected_version": expected_fmpy_version,
            },
            inputs={"model.fmu": resource.data},
            timeout=timeout,
            max_bytes=max(_DEFAULT_BYTES, max_archive_bytes),
        )

    @property
    def closed(self) -> bool:
        return self._worker.closed

    @property
    def artifact(self) -> ScientificArtifactEnvelope:
        failed = next(
            (
                call["error"]
                for call in reversed(self._worker.calls)
                if call["status"] == "failed"
            ),
            "",
        )
        return _artifact(
            "fmi2-co-simulation-session",
            {"calls": self._worker.calls, "runtime": self._worker.info},
            producer="FMPy/FMI2",
            version=self._worker.info["version"],
            build_id=self._worker.info["build_id"],
            license_id=self._license,
            resource_id=self._resource_id,
            error=failed,
        )

    def _validated_values(
        self, values: Mapping[str, Any], *, initializing: bool
    ) -> dict[str, Any]:
        result = {}
        for name, value in values.items():
            variable = self.model.variable(name)
            if variable.causality != "input" and not (
                initializing and variable.causality == "parameter"
            ):
                raise ValueError(
                    f"FMI variable {name!r} is not settable at this lifecycle phase."
                )
            result[name] = _scalar_value(variable, value)
        return result

    def set_values(self, values: Mapping[str, Any]) -> None:
        if self.terminated:
            raise RuntimeError("The FMU requested termination.")
        self._worker.call(
            "set", {"values": self._validated_values(values, initializing=False)}
        )

    def get_values(self, names: Sequence[str]) -> dict[str, Any]:
        for name in names:
            self.model.variable(name)
        return self._worker.call("get", {"names": list(names)})

    def advance(self, target_time: float) -> FMIStepResult:
        _host_only(target_time)
        target_time = float(target_time)
        if self.terminated or not math.isfinite(target_time) or target_time <= self.time:
            raise ValueError(
                "FMI advance requires an active session and a finite later time."
            )
        data = self._worker.call("advance", {"target_time": target_time})
        self.time, self.terminated = data["reached_time"], data["terminated"]
        return FMIStepResult(
            target_time,
            self.time,
            data["status"],
            data["early_return"],
            self.terminated,
            "internal-to-fmi2-co-simulation",
            self.artifact,
        )

    def save_state(self) -> FMIState:
        if not self.model.can_get_set_state:
            raise ValueError("The FMU does not advertise get/set FMU state.")
        if self.terminated:
            raise RuntimeError("Cannot capture state after FMU termination.")
        token = self._worker.call("save_state")
        return FMIState(self.session_id, token, self.time)

    def _check_state(self, state: FMIState) -> None:
        if not isinstance(state, FMIState) or state.session_id != self.session_id:
            raise ValueError("FMI state belongs to a different session.")

    def restore_state(self, state: FMIState) -> None:
        self._check_state(state)
        reached = self._worker.call("restore_state", {"token": state.token})
        self.time, self.terminated = reached, False

    def free_state(self, state: FMIState) -> None:
        self._check_state(state)
        self._worker.call("free_state", {"token": state.token})

    def close(self) -> None:
        self._worker.close()

    def __enter__(self) -> FMICoSimulationSession:
        if self.closed:
            raise RuntimeError("FMI session is closed.")
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> bool:
        try:
            self.close()
        except Exception as cleanup_error:
            if exception is None:
                raise
            exception.add_note(f"FMI cleanup also failed: {cleanup_error}")
        return False


__all__ = [
    "FMIVariable",
    "FMIModelDescription",
    "FMIStepResult",
    "FMIState",
    "FMICoSimulationSession",
    "inspect_fmu",
]
