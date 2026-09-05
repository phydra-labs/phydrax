#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Standalone native worker: standard library until the selected engine opens.

Executed by filename, deliberately without importing phydrax or initializing JAX.
The parent owns typed records, JAX host guards and canonical artifact identities.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import socket
import stat
import struct
import sys
import time
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


_DEFAULT_BYTES = 64 * 1024 * 1024


def _relative_path(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or "\x00" in name
        or ":" in name
        or path.is_absolute()
        or any(p in ("", ".", "..") for p in name.split("/"))
        or len(path.parts) > 32
    ):
        raise ValueError(f"Expected a bounded relative POSIX file path, got {name!r}.")
    return str(path)


def _digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _package_identity(distribution: str, expected_version: str | None) -> dict[str, Any]:
    package = importlib.metadata.distribution(distribution)
    version = package.version
    if expected_version is not None and expected_version != version:
        raise ValueError(
            f"Expected {distribution} {expected_version}, installed {version}."
        )
    # Identify actual Python/native implementation bytes, not just the declared
    # release or an unchanged RECORD beside a locally modified installation.
    implementation = {}
    for file in package.files or ():
        name = str(file)
        if name.endswith((".py", ".so", ".dylib", ".dll", ".pyd")) or ".so." in name:
            implementation[name] = _digest_file(Path(package.locate_file(file)))
    record = package.read_text("RECORD")
    return {
        "package": distribution,
        "version": version,
        "build_evidence": {
            "version": version,
            "implementation": implementation,
            "record": record or "distribution has no RECORD",
        },
    }


def _send_packet(connection: socket.socket, value: Any) -> None:
    data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    connection.sendall(struct.pack("!Q", len(data)) + data)


def _receive_packet(connection: socket.socket, maximum: int) -> Any:
    timeout = connection.gettimeout()
    deadline = None if timeout is None else time.monotonic() + timeout

    def receive(count: int) -> bytes:
        parts = bytearray()
        while len(parts) < count:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("External runtime response exceeded its deadline.")
                connection.settimeout(remaining)
            block = connection.recv(min(count - len(parts), 65536))
            if not block:
                raise EOFError("External runtime closed its control connection.")
            parts.extend(block)
        return bytes(parts)

    length = struct.unpack("!Q", receive(8))[0]
    if length > maximum:
        raise ValueError("External runtime response exceeds the message limit.")
    return json.loads(receive(length))


def _archive_members(
    archive: zipfile.ZipFile, max_unpacked_bytes: int, max_files: int
) -> tuple[zipfile.ZipInfo, ...]:
    members = tuple(archive.infolist())
    if not members or len(members) > max_files:
        raise ValueError("FMU archive member count is outside the configured bound.")
    total = 0
    seen = set()
    for item in members:
        name = item.filename[:-1] if item.is_dir() else item.filename
        name = _relative_path(name)
        if name in seen:
            raise ValueError(f"Duplicate FMU archive member {name!r}.")
        seen.add(name)
        mode = item.external_attr >> 16
        if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError("FMU archive links and special files are forbidden.")
        if item.flag_bits & 1:
            raise ValueError("Encrypted FMU archives are unsupported.")
        if item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise ValueError("FMU archives must use stored or deflated members.")
        total += item.file_size
        if (
            total > max_unpacked_bytes
            or item.file_size > max(1, item.compress_size) * 1000
        ):
            raise ValueError(
                "FMU archive exceeds the unpacked-byte/compression-ratio limit."
            )
    if "modelDescription.xml" not in seen:
        raise ValueError("FMU archive lacks modelDescription.xml.")
    return members


def _worker_main(descriptor: int) -> None:
    """Child-only dispatch. Native imports and mutable vendor handles stay here."""
    connection = socket.socket(fileno=descriptor)
    handler = None
    try:
        request = _receive_packet(connection, _DEFAULT_BYTES)
        payload = request["payload"]
        kind = payload["kind"]
        if kind == "fmi":
            handler = _FMIWorker()
        elif kind == "helics":
            handler = _HELICSWorker()
        elif kind == "opendss":
            handler = _OpenDSSWorker()
        else:
            raise ValueError(f"Unknown runtime kind {kind!r}.")
        _send_packet(connection, {"ok": True, "value": handler.open(payload["config"])})
        while True:
            request = _receive_packet(connection, _DEFAULT_BYTES)
            operation = request["operation"]
            if operation == "close":
                handler.close()
                handler = None
                _send_packet(connection, {"ok": True, "value": None})
                break
            result = handler.call(operation, request["payload"])
            _send_packet(connection, {"ok": True, "value": result})
    except BaseException as failure:
        # Exceptions are detached as text, never pickled vendor objects.
        _send_packet(
            connection, {"ok": False, "error": f"{type(failure).__name__}: {failure}"}
        )
    finally:
        if handler is not None:
            handler.close()
        connection.close()


class _OpenDSSWorker:
    def __init__(self):
        self.engine = None

    def open(self, config: Mapping[str, Any]) -> dict[str, Any]:
        module = importlib.import_module("opendssdirect")
        identity = _package_identity("opendssdirect.py", config["expected_version"])
        native = _package_identity("dss-python-backend", None)
        binding = _package_identity("dss-python", None)
        identity["build_evidence"] = {
            "direct": identity["build_evidence"],
            "binding": binding["build_evidence"],
            "native": native["build_evidence"],
        }
        identity["native_package_version"] = native["version"]
        self.engine = module.dss.NewContext()
        self.engine.Basic.AllowChangeDir(False)
        self.engine.Basic.DataPath(str(Path.cwd()))
        identity["engine_version"] = self.engine.Basic.Version()
        return identity

    def call(self, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if operation != "run":
            raise ValueError(f"Unsupported OpenDSS operation {operation!r}.")
        dss = self.engine
        for command in payload["commands"]:
            dss.Text.Command(command)
            number = dss.Error.Number()
            if number:
                raise RuntimeError(f"OpenDSS error {number}: {dss.Error.Description()}")
        if dss.Basic.NumCircuits() != 1:
            raise ValueError("The command sequence must create exactly one circuit.")
        elements = []
        for name in dss.Circuit.AllElementNames():
            dss.Circuit.SetActiveElement(name)
            powers = dss.CktElement.Powers()
            elements.append(
                [
                    name,
                    dss.CktElement.NumTerminals(),
                    dss.CktElement.NumConductors(),
                    list(zip(powers[::2], powers[1::2])),
                ]
            )
        volts = dss.Circuit.AllBusVolts()
        return {
            "converged": bool(dss.Solution.Converged()),
            "bus_names": dss.Circuit.AllBusNames(),
            "node_names": dss.Circuit.AllNodeNames(),
            "node_voltages": list(zip(volts[::2], volts[1::2])),
            "node_voltages_pu": dss.Circuit.AllBusMagPu(),
            "total_power": dss.Circuit.TotalPower(),
            "losses": dss.Circuit.Losses(),
            "element_powers": elements,
        }

    def close(self) -> None:
        if self.engine is not None:
            self.engine.Basic.ClearAll()
            self.engine = None


class _FMIWorker:
    def __init__(self):
        self.fmu = None
        self.instantiated = False
        self.initialized = False
        self.terminated = False
        self.states = {}
        self.next_state = 0
        self.step_size = None

    def open(self, config: Mapping[str, Any]) -> dict[str, Any]:
        module = importlib.import_module("fmpy.fmi2")
        self.exception_type = importlib.import_module("fmpy.fmi1").FMICallException
        identity = _package_identity("FMPy", config["expected_version"])
        self.model = config["description"]
        self.variables = {
            variable["name"]: variable for variable in self.model["variables"]
        }
        destination = Path("unpacked")
        destination.mkdir()
        with zipfile.ZipFile("model.fmu") as archive:
            members = _archive_members(
                archive, config["max_unpacked_bytes"], config["max_files"]
            )
            remaining = config["max_unpacked_bytes"]
            for member in members:
                path = destination / member.filename
                if member.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, path.open("xb") as target:
                    while block := source.read(min(65536, remaining + 1)):
                        remaining -= len(block)
                        if remaining < 0:
                            raise ValueError(
                                "FMU extraction exceeded unpacked-byte limit."
                            )
                        target.write(block)
        self.fmu = module.FMU2Slave(
            guid=self.model["guid"],
            modelIdentifier=self.model["model_identifier"],
            unzipDirectory=str(destination.resolve()),
            instanceName="phydrax",
        )
        self.fmu.instantiate()
        self.instantiated = True
        self.time = config["start_time"]
        self.stop_time = config["stop_time"]
        self.fmu.setupExperiment(startTime=self.time, stopTime=self.stop_time)
        self._set(config["start_values"])
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
        self.initialized = True
        return identity

    def _set(self, values: Mapping[str, Any]) -> None:
        for name, value in values.items():
            variable = self.variables[name]
            reference = [variable["value_reference"]]
            if variable["type"] == "Real":
                self.fmu.setReal(reference, [value])
            elif variable["type"] in ("Integer", "Enumeration"):
                self.fmu.setInteger(reference, [value])
            elif variable["type"] == "Boolean":
                self.fmu.setBoolean(reference, [value])
            else:
                self.fmu.setString(reference, [value])

    def _get(self, names: Sequence[str]) -> dict[str, Any]:
        values = {}
        for name in names:
            variable = self.variables[name]
            reference = [variable["value_reference"]]
            if variable["type"] == "Real":
                value = self.fmu.getReal(reference)[0]
            elif variable["type"] in ("Integer", "Enumeration"):
                value = self.fmu.getInteger(reference)[0]
            elif variable["type"] == "Boolean":
                value = bool(self.fmu.getBoolean(reference)[0])
            else:
                value = self.fmu.getString(reference)[0].decode("utf-8")
            values[name] = value
        return values

    def call(self, operation: str, payload: Mapping[str, Any]) -> Any:
        if operation == "set":
            self._set(payload["values"])
            return None
        if operation == "get":
            return self._get(payload["names"])
        if operation == "advance":
            return self._advance(payload["target_time"])
        if operation in ("save_state", "restore_state", "free_state"):
            if not self.model["can_get_set_state"]:
                raise ValueError("FMU state operations were not advertised.")
            if operation == "save_state":
                if len(self.states) >= 64:
                    raise ValueError("At most 64 live native FMU states are permitted.")
                token = self.next_state
                self.next_state += 1
                self.states[token] = (self.fmu.getFMUstate(), self.time, self.step_size)
                return token
            token = payload["token"]
            state, reached, step_size = self.states[token]
            if operation == "restore_state":
                self.fmu.setFMUstate(state)
                self.time, self.step_size, self.terminated = reached, step_size, False
                return reached
            self.fmu.freeFMUstate(state)
            del self.states[token]
            return None
        raise ValueError(f"Unsupported FMI operation {operation!r}.")

    def _advance(self, target: float) -> dict[str, Any]:
        step_size = target - self.time
        if (
            self.terminated
            or step_size <= 0
            or (self.stop_time is not None and target > self.stop_time)
        ):
            raise ValueError("FMI step is outside the active experiment interval.")
        if (
            not self.model["variable_step"]
            and self.step_size is not None
            and not math.isclose(step_size, self.step_size, rel_tol=1e-12, abs_tol=0)
        ):
            raise ValueError(
                "The FMU does not advertise variable communication step sizes."
            )
        self.step_size = step_size
        try:
            status = self.fmu.fmi2DoStep(
                self.fmu.component, self.time, step_size, not bool(self.states)
            )
        except self.exception_type as failure:
            status = failure.status
            if status != 2:
                if status == 5:
                    self.fmu.cancelStep()
                raise
        if status in (0, 1):
            reached, terminated = target, False
        elif status == 2:
            reached = self.fmu.getRealStatus(2)  # fmi2LastSuccessfulTime
            terminated = self.fmu.getBooleanStatus(3)  # fmi2Terminated
            if not math.isfinite(reached) or not self.time <= reached <= target:
                raise ValueError(
                    "FMU returned an invalid lastSuccessfulTime after discard."
                )
        else:
            raise RuntimeError(f"Unsupported FMI doStep status {status}.")
        self.time, self.terminated = reached, terminated
        return {
            "reached_time": reached,
            "status": ("ok", "warning", "discard")[status],
            "early_return": reached < target,
            "terminated": terminated,
        }

    def close(self) -> None:
        if self.fmu is None:
            return
        fmu, self.fmu = self.fmu, None
        try:
            for state, _, _ in self.states.values():
                fmu.freeFMUstate(state)
            self.states.clear()
        finally:
            try:
                if self.initialized:
                    fmu.terminate()
            finally:
                if self.instantiated:
                    fmu.freeInstance()
                else:
                    fmu.freeLibrary()


class _HELICSWorker:
    def __init__(self):
        self.library = None
        self.broker = None
        self.federate = None
        self.info = None
        self.mode = "created"
        self.received = set()
        self.time = 0.0

    def open(self, config: Mapping[str, Any]) -> dict[str, str]:
        h = importlib.import_module("helics")
        self.library = h
        identity = _package_identity("helics", config["expected_version"])
        if identity["version"].split(".")[0] != "3":
            raise ValueError("This adapter declares only the HELICS3 value-federate API.")
        broker_address = config["broker"]
        port_option = "" if config["core_type"] == "ipc" else " --useosport"
        if broker_address is None:
            try:
                self.broker = h.helicsCreateBroker(
                    config["core_type"],
                    "phydrax-" + uuid.uuid4().hex,
                    f"--federates={config['federate_count']}" + port_option,
                )
            except h.HelicsException as failure:
                raise RuntimeError(
                    f"HELICS broker creation failed: {failure}"
                ) from failure
            broker_address = h.helicsBrokerGetAddress(self.broker)
        self.info = h.helicsCreateFederateInfo()
        h.helicsFederateInfoSetCoreTypeFromString(self.info, config["core_type"])
        h.helicsFederateInfoSetCoreInitString(self.info, "--federates=1" + port_option)
        h.helicsFederateInfoSetBroker(self.info, broker_address)
        h.helicsFederateInfoSetTimeProperty(
            self.info, h.HELICS_PROPERTY_TIME_DELTA, config["time_delta"]
        )
        try:
            self.federate = h.helicsCreateValueFederate(config["name"], self.info)
        except h.HelicsException as failure:
            raise RuntimeError(
                f"HELICS federate creation failed for broker {broker_address!r}: {failure}"
            ) from failure
        h.helicsFederateInfoFree(self.info)
        self.info = None
        types = {
            "double": h.HELICS_DATA_TYPE_DOUBLE,
            "integer": h.HELICS_DATA_TYPE_INT,
            "boolean": h.HELICS_DATA_TYPE_BOOLEAN,
            "string": h.HELICS_DATA_TYPE_STRING,
            "complex": h.HELICS_DATA_TYPE_COMPLEX,
            "vector": h.HELICS_DATA_TYPE_VECTOR,
        }
        self.types = types
        self.publications = {}
        self.inputs = {}
        for channel in config["publications"]:
            handle = h.helicsFederateRegisterGlobalPublication(
                self.federate, channel["name"], types[channel["type"]], channel["unit"]
            )
            self.publications[channel["name"]] = (channel, handle)
        for channel in config["subscriptions"]:
            handle = h.helicsFederateRegisterInput(
                self.federate, channel["name"], types[channel["type"]], channel["unit"]
            )
            h.helicsInputAddTarget(handle, channel["target"])
            h.helicsInputSetOption(handle, h.HELICS_HANDLE_OPTION_CONNECTION_REQUIRED, 1)
            h.helicsInputSetOption(
                handle, h.HELICS_HANDLE_OPTION_SINGLE_CONNECTION_ONLY, 1
            )
            h.helicsInputSetOption(handle, h.HELICS_HANDLE_OPTION_STRICT_TYPE_CHECKING, 1)
            self.inputs[channel["name"]] = (channel, handle)
        identity["broker_address"] = broker_address
        identity["native_version"] = h.helicsGetVersion()
        return identity

    def call(self, operation: str, payload: Mapping[str, Any]) -> Any:
        h, federate = self.library, self.federate
        if operation == "enter_async":
            h.helicsFederateEnterExecutingModeAsync(federate)
            self.mode = "entering"
        elif operation == "enter_complete":
            h.helicsFederateEnterExecutingModeComplete(federate)
            self.mode = "executing"
            for channel, handle in self.inputs.values():
                if (
                    h.helicsInputGetOption(handle, h.HELICS_HANDLE_OPTION_CONNECTIONS)
                    != 1
                ):
                    raise ValueError(
                        f"HELICS input {channel['name']!r} requires exactly one publisher."
                    )
                if (
                    h.helicsInputGetPublicationDataType(handle)
                    != self.types[channel["type"]]
                ):
                    raise ValueError(
                        f"HELICS input {channel['name']!r} publication type differs."
                    )
                if h.helicsInputGetInjectionUnits(handle) != channel["unit"]:
                    raise ValueError(
                        f"HELICS input {channel['name']!r} requires exact units; implicit conversion is disabled."
                    )
            self.time = h.helicsFederateGetCurrentTime(federate)
            return self.time
        elif operation == "publish":
            for name, value in payload["values"].items():
                channel, handle = self.publications[name]
                kind = channel["type"]
                if kind == "double":
                    h.helicsPublicationPublishDouble(handle, value)
                elif kind == "integer":
                    h.helicsPublicationPublishInteger(handle, value)
                elif kind == "boolean":
                    h.helicsPublicationPublishBoolean(handle, value)
                elif kind == "string":
                    h.helicsPublicationPublishString(handle, value)
                elif kind == "complex":
                    h.helicsPublicationPublishComplex(handle, *value)
                else:
                    h.helicsPublicationPublishVector(handle, value)
        elif operation == "time_async":
            self.requested_time = payload["target_time"]
            h.helicsFederateRequestTimeAsync(federate, self.requested_time)
            self.mode = "advancing"
        elif operation == "time_complete":
            granted = h.helicsFederateRequestTimeComplete(federate)
            terminated = granted == h.HELICS_TIME_MAXTIME
            if (
                not math.isfinite(granted)
                or granted < self.time
                or (not terminated and granted > self.requested_time)
            ):
                raise ValueError(
                    "HELICS returned a grant outside the requested monotone interval."
                )
            self.time = granted
            self.mode = "terminated" if terminated else "executing"
            return {"time": granted, "terminated": terminated}
        elif operation == "read":
            result = []
            for name in payload["names"]:
                channel, handle = self.inputs[name]
                updated = h.helicsInputIsUpdated(handle)
                if updated:
                    self.received.add(name)
                received = name in self.received
                value, update_time = None, None
                if received:
                    update_time = h.helicsInputLastUpdateTime(handle)
                    kind = channel["type"]
                    if kind == "double":
                        value = h.helicsInputGetDouble(handle)
                    elif kind == "integer":
                        value = h.helicsInputGetInteger(handle)
                    elif kind == "boolean":
                        value = h.helicsInputGetBoolean(handle)
                    elif kind == "string":
                        if h.helicsInputGetStringSize(handle) > 1024 * 1024:
                            raise ValueError("HELICS string exceeds 1 MiB.")
                        value = h.helicsInputGetString(handle)
                    elif kind == "complex":
                        value = h.helicsInputGetComplex(handle)
                        value = [value.real, value.imag]
                    else:
                        if h.helicsInputGetVectorSize(handle) > 65536:
                            raise ValueError("HELICS vector exceeds 65536 entries.")
                        value = h.helicsInputGetVector(handle)
                result.append(
                    {
                        "channel": name,
                        "value": value,
                        "last_update_time": update_time,
                        "updated": updated,
                        "has_value": received,
                    }
                )
            return result
        else:
            raise ValueError(f"Unsupported HELICS operation {operation!r}.")
        return None

    def close(self) -> None:
        if self.library is None:
            return
        h = self.library
        try:
            if self.federate is not None:
                federate, self.federate = self.federate, None
                try:
                    h.helicsFederateFinalize(federate)
                finally:
                    h.helicsFederateFree(federate)
        finally:
            try:
                if self.info is not None:
                    h.helicsFederateInfoFree(self.info)
                    self.info = None
            finally:
                if self.broker is not None:
                    broker, self.broker = self.broker, None
                    try:
                        h.helicsBrokerDisconnect(broker)
                    finally:
                        h.helicsBrokerFree(broker)


if __name__ == "__main__":
    _worker_main(int(sys.argv[1]))
