#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded, nonexecuting imports for explicitly closed balanced power subsets.

MATPOWER: version-2 numeric function cases, polynomial P costs of degree <= 2.
PSS/E: revision 33 RAW, constant-power loads, fixed shunts, branches, fixed
CW=CZ=CM=1 two-winding transformers, paired with positive-inertia GENCLS DYR.
CGMES: 2.4.15 / CIM16 EQ-Core + TP + SSH (optional SV), bus-branch RDF/XML;
lines, constant-power consumers, synchronous generators and fixed linear shunts.
CGMES transformers, switches, connectivity-node topology and controls other than
local generator voltage regulation are deliberately unsupported, not approximated.

All inputs are resident UTF-8 text, never filenames, Python, MATLAB, or URLs.
Failures carry the same canonical AdapterReport type used by successful imports.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from math import inf, isfinite, pi
from typing import NoReturn

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...interchange import (
    AdapterError,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
    bounded_resource_from_bytes,
    ResourceLimits,
    ResourceReadError,
)
from ._dynamics import ClassicalMachine
from ._network import (
    Branch,
    Bus,
    BusControl,
    Generator,
    Load,
    PowerNetwork,
    PowerStudy,
    Shunt,
)


@dataclass(frozen=True)
class PowerParserLimits:
    """Host parsing bounds; limits apply to the complete import, not each row."""

    max_bytes: int = 4 * 1024 * 1024
    max_rows: int = 100_000
    max_tokens: int = 1_000_000
    max_token_chars: int = 512
    max_xml_depth: int = 16

    def __post_init__(self) -> None:
        for value in (
            self.max_bytes,
            self.max_rows,
            self.max_tokens,
            self.max_token_chars,
            self.max_xml_depth,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("Power parser limits must be positive integers.")


class PowerCaseAdaptation(StrictModule, NonTrainableState):
    """Physical network, explicit study controls, semantic report and machine specs."""

    network: PowerNetwork
    study: PowerStudy
    report: AdapterReport
    dynamics: tuple[ClassicalMachine, ...] = ()


class PowerImportError(AdapterError):
    """Fail-closed import error; ``report.valid`` is always false."""

    report: AdapterReport

    def __init__(self, report: AdapterReport, message: str):
        self.report = report
        super().__init__(report.status, message)


_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eEdD][+-]?[0-9]+)?")
_DEFAULT_LIMITS = PowerParserLimits()


class _Import:
    def __init__(self, format: str, texts: Sequence[str], limits: PowerParserLimits):
        self.format = format
        self.limits = limits
        self.losses: list[AdapterLoss] = []
        self.generator_voltages: dict[str, float] = {}
        self.rows = 0
        self.tokens = 0
        self.source_id = "unread-power-source"
        self.target_id = "no-power-network-produced"
        if len(texts) > limits.max_rows:
            self.fail("/source", "Document count exceeds row/resource limit.")
        total = 0
        identities = []
        for text in texts:
            if not isinstance(text, str):
                self.fail("/source", "Inputs must be text, not paths or bytes.")
            if len(text) > limits.max_bytes - total:
                self.fail("/source", "Source exceeds byte limit.")
            try:
                data = text.encode("utf-8")
                total += len(data)
                if total > limits.max_bytes:
                    self.fail("/source", "Combined source exceeds byte limit.")
                resource = bounded_resource_from_bytes(
                    data,
                    limits=ResourceLimits(
                        limits.max_bytes,
                        limits.max_xml_depth,
                        limits.max_rows,
                        limits.max_tokens,
                        limits.max_rows,
                    ),
                )
            except (UnicodeError, ResourceReadError) as exc:
                self.fail("/source", str(exc))
            identities.append(resource.manifest.content_sha256)
            if "\x00" in text:
                self.fail("/source", "NUL characters are not allowed.")
        self.source_id = canonical_fingerprint({"format": format, "sources": identities})

    def report(
        self, status: AdapterStatus, losses: Sequence[AdapterLoss]
    ) -> AdapterReport:
        return AdapterReport(
            status,
            self.format,
            "phydrax-balanced-positive-sequence-rms",
            source_id=self.source_id,
            target_id=self.target_id,
            coordinate_mapping=(
                "MW/Mvar -> total-three-phase per-unit on network base_mva",
                "bus injections generation-positive; component terminal currents inward",
                "positive-sequence RMS voltage on line-line kV base; angles radians",
            ),
            preserved_fields=(
                "supported electrical parameters",
                "supported service states",
            ),
            assumptions=("balanced positive sequence; exp(+i omega t)",),
            losses=losses,
        )

    def fail(self, path: str, message: str, *, unsupported: bool = False) -> NoReturn:
        status = (
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
            if unsupported
            else AdapterStatus.MALFORMED_SOURCE
        )
        issue = AdapterLoss(
            path, "import", "unsupported", message, changes_interpretation=True
        )
        raise PowerImportError(
            self.report(status, (*self.losses, issue)), f"{path}: {message}"
        )

    def drop(self, path: str, message: str, *, synthesized: bool = False) -> None:
        if len(self.losses) >= self.limits.max_rows:
            self.fail(path, "Semantic issue count exceeds limit.")
        self.losses.append(
            AdapterLoss(
                path,
                "import",
                "synthesized" if synthesized else "dropped",
                message,
                changes_interpretation=synthesized,
            )
        )

    def row(self, path: str) -> None:
        self.rows += 1
        if self.rows > self.limits.max_rows:
            self.fail(path, "Row/resource count exceeds limit.")

    def token(self, token: str, path: str) -> str:
        self.tokens += 1
        if (
            self.tokens > self.limits.max_tokens
            or len(token) > self.limits.max_token_chars
        ):
            self.fail(path, "Token count or token length exceeds limit.")
        return token

    def number(self, token: str, path: str) -> float:
        if not _NUMBER.fullmatch(token):
            self.fail(path, "Expected a finite numeric literal, not an expression.")
        value = float(token.replace("D", "e").replace("d", "e"))
        if not isfinite(value):
            self.fail(path, "Numeric literal must be finite.")
        return value

    def integer(self, value: float, path: str) -> int:
        if not isfinite(value) or value != int(value):
            self.fail(path, "Expected an integer.")
        return int(value)

    def flag(self, value: float, path: str) -> bool:
        if value not in (0, 1):
            self.fail(path, "Status must be 0 or 1.")
        return bool(value)

    def generator_voltage(
        self,
        controls: dict[str, BusControl],
        bus: str,
        voltage: float,
        active: bool,
        path: str,
    ) -> None:
        if bus not in controls:
            self.fail(path, "Generator references a missing bus control.")
        control = controls[bus]
        if voltage <= 0 or not isfinite(voltage) or control.voltage <= 0:
            self.fail(
                path,
                "Generator setpoint and initial bus voltage must be positive and finite.",
            )
        if not active or control.kind == "pq":
            self.drop(
                f"{path}/voltage",
                "Inactive/PQ generator voltage is non-operative; study retains the bus initial voltage.",
            )
            return
        if (
            bus in self.generator_voltages
            and abs(self.generator_voltages[bus] - voltage) > 1e-12
        ):
            self.fail(
                path,
                "Active generators controlling one bus have conflicting voltage setpoints.",
            )
        self.generator_voltages[bus] = voltage
        if abs(control.voltage - voltage) > 1e-12:
            self.drop(
                f"/study/{bus}/initial_voltage",
                "The active generator voltage setpoint supersedes the source bus initial magnitude.",
            )
            controls[bus] = BusControl(
                bus=bus, kind=control.kind, voltage=voltage, angle=control.angle
            )

    def finish(
        self,
        buses: Sequence[Bus],
        controls: Sequence[BusControl],
        branches: Sequence[Branch],
        generators: Sequence[Generator],
        loads: Sequence[Load],
        shunts: Sequence[Shunt],
        base: float,
        frequency: float,
        dynamics: tuple[ClassicalMachine, ...] = (),
    ) -> PowerCaseAdaptation:
        if (
            not buses
            or not isfinite(base)
            or base <= 0
            or not isfinite(frequency)
            or frequency <= 0
        ):
            self.fail(
                "/network",
                "Nonempty buses and positive finite base/frequency are required.",
            )
        ids = {bus.id for bus in buses}
        if len(ids) != len(buses):
            self.fail("/buses", "Duplicate bus identifiers.")
        if len(controls) != len(ids) or {control.bus for control in controls} != ids:
            self.fail("/study", "Exactly one study control per physical bus is required.")
        for control in controls:
            if control.kind not in ("pq", "pv", "reference"):
                self.fail(
                    f"/study/{control.bus}",
                    "Unsupported study bus control kind.",
                    unsupported=True,
                )
            if (
                not isfinite(control.voltage)
                or not isfinite(control.angle)
                or control.voltage <= 0
            ):
                self.fail(
                    f"/study/{control.bus}",
                    "Study voltage must be positive; voltage and angle must be finite.",
                )
        for name, records in (
            ("branches", branches),
            ("generators", generators),
            ("loads", loads),
        ):
            if len({item.id for item in records}) != len(records):
                self.fail(f"/{name}", "Duplicate identifiers.")
            for item in records:
                ends = (
                    (item.from_bus, item.to_bus)
                    if isinstance(item, Branch)
                    else (item.bus,)
                )
                if not all(end in ids for end in ends):
                    self.fail(f"/{name}/{item.id}", "Reference to missing bus.")
        if any(shunt.bus not in ids for shunt in shunts):
            self.fail("/shunts", "Reference to missing bus.")
        for bus in buses:
            if not all(isfinite(value) for value in (bus.base_kv, bus.v_min, bus.v_max)):
                self.fail(f"/buses/{bus.id}", "Converted bus data must remain finite.")
            if bus.base_kv <= 0 or not 0 < bus.v_min <= bus.v_max:
                self.fail(
                    f"/buses/{bus.id}",
                    "Voltage base and ordered limits must be positive.",
                )
        for branch in branches:
            if not all(
                isfinite(value)
                for value in (branch.r, branch.x, branch.b, branch.tap, branch.phase)
            ):
                self.fail(
                    f"/branches/{branch.id}", "Converted branch data must remain finite."
                )
            if (
                branch.from_bus == branch.to_bus
                or branch.r < 0
                or branch.tap <= 0
                or branch.rate <= 0
            ):
                self.fail(
                    f"/branches/{branch.id}",
                    "Invalid terminals, resistance, ratio or rating.",
                )
            if branch.in_service and branch.r == 0 and branch.x == 0:
                self.fail(
                    f"/branches/{branch.id}",
                    "Zero-impedance active branches require bus reduction.",
                    unsupported=True,
                )
        for generator in generators:
            if not all(
                isfinite(value) for value in (generator.p, generator.q, *generator.cost)
            ):
                self.fail(
                    f"/generators/{generator.id}",
                    "Converted generation and costs must remain finite.",
                )
            if generator.p_min > generator.p_max or generator.q_min > generator.q_max:
                self.fail(f"/generators/{generator.id}", "Reversed generation limits.")
        for load in loads:
            if not all(isfinite(value) for value in (load.p, load.q)):
                self.fail(f"/loads/{load.id}", "Converted load power must remain finite.")
        for shunt in shunts:
            if not isfinite(shunt.g) or not isfinite(shunt.b) or shunt.g < 0:
                self.fail(
                    f"/shunts/{shunt.bus}",
                    "Passive shunt admittance must be finite with nonnegative conductance.",
                )
        try:
            network = PowerNetwork(
                buses=tuple(buses),
                branches=tuple(branches),
                generators=tuple(generators),
                loads=tuple(loads),
                shunts=tuple(shunts),
                base_mva=base,
                frequency=frequency,
            )
            study = PowerStudy(controls=tuple(controls))
        except ValueError as exc:
            self.fail("/network", str(exc))
        self.target_id = canonical_fingerprint(
            {
                "source": self.source_id,
                "adapter": "balanced-power-subset-study-2",
                "base_mva": base,
                "frequency": frequency,
            }
        )
        status = AdapterStatus.DECLARED_LOSS if self.losses else AdapterStatus.LOSSLESS
        return PowerCaseAdaptation(
            network, study, self.report(status, self.losses), dynamics
        )


def _numeric_rows(ctx: _Import, body: str, path: str) -> list[list[float]]:
    rows = []
    for raw in re.split(r"[;\n]", body):
        raw = raw.strip()
        if not raw:
            continue
        ctx.row(path)
        fields = re.split(r"[\s,]+", raw)
        rows.append([ctx.number(ctx.token(field, path), path) for field in fields])
    return rows


def parse_matpower(
    text: str,
    *,
    limits: PowerParserLimits = _DEFAULT_LIMITS,
) -> PowerCaseAdaptation:
    """Parse a numeric MATPOWER v2 function-case without executing any source.

    Only version/baseMVA/bus/gen/branch/gencost assignments are accepted. Active
    PQ capability curves, ramp/participation controls, angle-difference limits,
    piecewise/reactive costs, startup/shutdown costs and isolated buses fail.
    RATE_A is the continuous MVA limit; B/C ratings and area/zone labels are
    explicitly reported as discarded non-operating metadata. Frequency is 60 Hz
    because MATPOWER v2 carries no frequency (also recorded in the report).
    """
    ctx = _Import("MATPOWER-v2", (text,), limits)
    if "%{" in text or "%}" in text:
        ctx.fail(
            "/source",
            "MATLAB block comments are outside the numeric grammar.",
            unsupported=True,
        )
    source = "\n".join(line.split("%", 1)[0] for line in text.splitlines())
    source = re.sub(r"\.\.\.[ \t]*\n", " ", source)
    header = re.match(
        r"\s*function[ \t]+mpc[ \t]*=[ \t]*([A-Za-z][A-Za-z0-9_]*)(?:[ \t]*\([ \t]*\))?[ \t]*;?[ \t]*(?:\n|$)",
        source,
    )
    if header is None:
        ctx.fail("/function", "Expected a numeric function mpc = case_name declaration.")
    ctx.token(header.group(1), "/function")
    cursor = header.end()
    fields = {}
    assignment = re.compile(r"\s*mpc\.([A-Za-z][A-Za-z0-9_]*)\s*=\s*")
    while source[cursor:].strip():
        if re.fullmatch(r"\s*end\s*;?\s*", source[cursor:]):
            break
        match = assignment.match(source, cursor)
        if match is None:
            ctx.fail(
                "/source",
                "Only literal mpc assignments are permitted; executable MATLAB is rejected.",
                unsupported=True,
            )
        name = ctx.token(match.group(1), "/source")
        path = f"/mpc/{name}"
        if name not in {"version", "baseMVA", "bus", "gen", "branch", "gencost"}:
            ctx.fail(path, "Unknown MATPOWER field semantics.", unsupported=True)
        if name in fields:
            ctx.fail(path, "Duplicate assignment.")
        cursor = match.end()
        if name in {"bus", "gen", "branch", "gencost"}:
            if source[cursor : cursor + 1] != "[":
                ctx.fail(path, "Expected a literal numeric matrix.")
            end = source.find("]", cursor + 1)
            if end < 0:
                ctx.fail(path, "Unterminated matrix.")
            fields[name] = _numeric_rows(ctx, source[cursor + 1 : end], path)
            cursor = end + 1
        else:
            end = source.find(";", cursor)
            if end < 0:
                ctx.fail(path, "Scalar assignment requires semicolon.")
            token = ctx.token(source[cursor:end].strip(), path)
            if name == "version":
                if token != "'2'":
                    ctx.fail(
                        path, "Only MATPOWER version '2' is supported.", unsupported=True
                    )
                fields[name] = token
            else:
                fields[name] = ctx.number(token, path)
            cursor = end
        terminator = re.match(r"[ \t\r\n]*;", source[cursor:])
        if terminator is None:
            ctx.fail(path, "Assignment requires a semicolon.")
        cursor += terminator.end()
    if not {"version", "baseMVA", "bus", "gen", "branch"}.issubset(fields):
        ctx.fail("/mpc", "Missing required case fields.")
    base = fields["baseMVA"]
    if base <= 0:
        ctx.fail("/mpc/baseMVA", "Base MVA must be positive.")
    buses, branches, generators, loads, shunts = [], [], [], [], []
    controls: dict[str, BusControl] = {}
    for row in fields["bus"]:
        if len(row) != 13:
            ctx.fail("/mpc/bus", "Exactly 13 unsolved v2 bus columns are supported.")
        bid = str(ctx.integer(row[0], "/mpc/bus/id"))
        kind = ctx.integer(row[1], f"/bus/{bid}/type")
        if kind not in (1, 2, 3):
            ctx.fail(
                f"/bus/{bid}/type",
                "Only active PQ/PV/reference buses are supported.",
                unsupported=True,
            )
        buses.append(Bus(id=bid, base_kv=row[9], v_min=row[12], v_max=row[11]))
        controls[bid] = BusControl(
            bus=bid,
            kind={1: "pq", 2: "pv", 3: "reference"}[kind],
            voltage=row[7],
            angle=row[8] * pi / 180,
        )
        if row[2] or row[3]:
            loads.append(
                Load(id=f"load:{bid}", bus=bid, p=row[2] / base, q=row[3] / base)
            )
        if row[4] or row[5]:
            shunts.append(Shunt(bus=bid, g=row[4] / base, b=row[5] / base))
    gen_rows = fields["gen"]
    costs = [(0.0, 0.0, 0.0)] * len(gen_rows)
    if "gencost" in fields:
        if len(fields["gencost"]) != len(gen_rows):
            ctx.fail(
                "/mpc/gencost",
                "Exactly one real-power cost row per generator is supported.",
                unsupported=True,
            )
        for index, row in enumerate(fields["gencost"]):
            path = f"/mpc/gencost/{index}"
            if len(row) < 5:
                ctx.fail(path, "Incomplete polynomial cost.")
            n = ctx.integer(row[3], path)
            if row[0] != 2 or n not in (1, 2, 3):
                ctx.fail(
                    path,
                    "Only degree <= 2 polynomial P costs are supported.",
                    unsupported=True,
                )
            if len(row) != 4 + n:
                ctx.fail(path, "Coefficient count does not match NCOST.")
            if row[1] or row[2]:
                ctx.fail(
                    path,
                    "Startup/shutdown costs need commitment semantics.",
                    unsupported=True,
                )
            coefficients = [0.0] * (3 - n) + row[4:]
            costs[index] = (
                coefficients[0] * base * base,
                coefficients[1] * base,
                coefficients[2],
            )
    for index, row in enumerate(gen_rows):
        path = f"/mpc/gen/{index}"
        if len(row) not in (10, 21):
            ctx.fail(path, "Expected 10 or 21 unsolved v2 generator columns.")
        active = ctx.flag(row[7], path)
        if active and any(row[10:]):
            ctx.fail(
                path,
                "Active capability/ramp/participation fields are unsupported.",
                unsupported=True,
            )
        if not active and any(row[10:]):
            ctx.drop(
                path, "Inactive generator capability/ramp/participation fields omitted."
            )
        generator_bus = str(ctx.integer(row[0], path))
        ctx.generator_voltage(controls, generator_bus, row[5], active, path)
        generators.append(
            Generator(
                id=f"gen:{index + 1}",
                bus=generator_bus,
                p=row[1] / base,
                q=row[2] / base,
                p_min=row[9] / base,
                p_max=row[8] / base,
                q_min=row[4] / base,
                q_max=row[3] / base,
                cost=costs[index],
                in_service=active,
            )
        )
    for index, row in enumerate(fields["branch"]):
        path = f"/mpc/branch/{index}"
        if len(row) != 13:
            ctx.fail(path, "Exactly 13 unsolved v2 branch columns are supported.")
        active = ctx.flag(row[10], path)
        if active and not (
            (row[11] == 0 and row[12] == 0) or (row[11] <= -360 and row[12] >= 360)
        ):
            ctx.fail(
                path,
                "Active angle-difference constraints are unsupported.",
                unsupported=True,
            )
        if row[5] < 0 or row[8] < 0:
            ctx.fail(path, "Rating and tap must be nonnegative.")
        branches.append(
            Branch(
                id=f"branch:{index + 1}",
                from_bus=str(ctx.integer(row[0], path)),
                to_bus=str(ctx.integer(row[1], path)),
                r=row[2],
                x=row[3],
                b=row[4],
                rate=row[5] / base if row[5] else inf,
                tap=row[8] or 1.0,
                phase=row[9] * pi / 180,
                in_service=active,
            )
        )
    ctx.drop(
        "/mpc/metadata",
        "Area/zone labels, generator machine base and emergency B/C ratings "
        "are not operating RMS/continuous-limit fields.",
    )
    ctx.drop(
        "/frequency",
        "MATPOWER has no frequency; the imported network explicitly uses 60 Hz.",
        synthesized=True,
    )
    return ctx.finish(
        buses, tuple(controls.values()), branches, generators, loads, shunts, base, 60.0
    )


def _psse_fields(ctx: _Import, text: str, path: str) -> list[str]:
    """PSS/E comma/space literals, single-quoted strings and slash comments."""
    fields = []
    cursor = 0
    after_comma = False
    while cursor < len(text):
        if text[cursor].isspace():
            cursor += 1
            continue
        if text[cursor] == "/":
            break
        if text[cursor] == ",":
            if after_comma or not fields:
                ctx.fail(
                    path, "Blank/defaulted RAW fields are outside this explicit subset."
                )
            after_comma = True
            cursor += 1
            continue
        if text[cursor] == "'":
            match = re.match(r"'((?:[^']|'')*)'", text[cursor:])
            if match is None:
                ctx.fail(path, "Unterminated quoted string.")
            token = match.group(1).replace("''", "'").strip()
            cursor += match.end()
        else:
            match = re.match(r"[^\s,/'\"]+", text[cursor:])
            if match is None:
                ctx.fail(path, "Invalid record token.")
            token = match.group()
            cursor += match.end()
        fields.append(ctx.token(token, path))
        after_comma = False
    return fields


def parse_psse(
    raw_text: str,
    dyr_text: str,
    *,
    limits: PowerParserLimits = _DEFAULT_LIMITS,
) -> PowerCaseAdaptation:
    """Import a paired revision-33 RAW and GENCLS-only DYR dataset.

    Every in-service generator must have exactly one GENCLS H,D record. H, D,
    RAW MBASE, ZR and ZX become ClassicalMachine parameters on machine base.
    No missing inertia/reactance is fabricated. Inactive models are audited and
    skipped. Remote regulation, ZIP loads, switched shunts, DC/FACTS, corrections,
    three-winding and automatic-tap transformer semantics are rejected.
    """
    ctx = _Import("PSS/E-RAW33+DYR-GENCLS", (raw_text, dyr_text), limits)
    lines = raw_text.splitlines()
    if len(lines) < 4:
        ctx.fail("/RAW", "Missing header/title records.")
    header = _psse_fields(ctx, lines[0], "/RAW/header")
    if len(header) != 6:
        ctx.fail("/RAW/header", "Expected IC,SBASE,REV,XFRRAT,NXFRAT,BASFRQ.")
    h = [ctx.number(value, "/RAW/header") for value in header]
    if h[0] != 0 or h[2] != 33:
        ctx.fail(
            "/RAW/header",
            "Only complete IC=0 revision-33 RAW is supported.",
            unsupported=True,
        )
    if h[3] != 0 or h[4] != 0:
        ctx.fail(
            "/RAW/header",
            "Only MVA transformer/branch rating units are supported.",
            unsupported=True,
        )
    base, frequency = h[1], h[5]
    if base <= 0 or frequency <= 0:
        ctx.fail("/RAW/header", "Base MVA and frequency must be positive.")
    sections: list[list[list[str]]] = [[]]
    terminated = False
    for index, line in enumerate(lines[3:], 4):
        path = f"/RAW/line/{index}"
        if terminated:
            if line.strip():
                ctx.fail(path, "Data follows RAW terminator.")
            continue
        fields = _psse_fields(ctx, line, path)
        if not fields:
            continue
        if fields == ["Q"] or fields == ["q"]:
            terminated = True
            continue
        ctx.row(path)
        if fields == ["0"]:
            sections.append([])
        else:
            sections[-1].append(fields)
    if not terminated or len(sections) < 7:
        ctx.fail(
            "/RAW",
            "RAW requires six electrical sections with zero delimiters and Q terminator.",
        )
    if len(sections) > 20:
        ctx.fail(
            "/RAW",
            "Sections beyond the revision-33 record set are unsupported.",
            unsupported=True,
        )
    # Revision-33 tail order: AREA, 2TDC, VSC, impedance corrections, MTDC,
    # multi-section lines, ZONE, inter-area transfers, OWNER, FACTS, switched
    # shunts, GNE, induction machines. Administrative AREA/ZONE/OWNER rows alone
    # can be discarded; all other nonempty sections are unsupported.
    for index, rows in enumerate(sections[6:], 6):
        if rows and index not in (6, 12, 14):
            ctx.fail(
                f"/RAW/section/{index}",
                "Unsupported active RAW section.",
                unsupported=True,
            )
        for row in rows:
            path = f"/RAW/section/{index}"
            if len(row) != (5 if index == 6 else 2):
                ctx.fail(path, "Malformed administrative record.")
            if ctx.integer(ctx.number(row[0], path), path) <= 0:
                ctx.fail(path, "Administrative identifier must be positive.")
            if index == 6:
                if ctx.number(row[1], path) != 0 or ctx.number(row[2], path) != 0:
                    ctx.fail(
                        path,
                        "Area interchange/slack control is unsupported.",
                        unsupported=True,
                    )
                if ctx.number(row[3], path) < 0:
                    ctx.fail(path, "Negative area interchange tolerance.")
        if rows:
            ctx.drop(
                f"/RAW/section/{index}",
                "Administrative area/zone/owner records omitted; no interchange dispatch is performed.",
            )
    buses, branches, generators, loads, shunts = [], [], [], [], []
    bus_ids: set[str] = set()
    controls: dict[str, BusControl] = {}

    def num(row: list[str], index: int, path: str) -> float:
        return ctx.number(row[index], path)

    def bid(token: str, path: str) -> str:
        value = ctx.integer(ctx.number(token, path), path)
        if value <= 0:
            ctx.fail(path, "Bus identifier must be positive.")
        return str(value)

    for row in sections[0]:
        if len(row) != 13:
            ctx.fail("/RAW/bus", "Expected 13 revision-33 bus fields.")
        identifier = bid(row[0], "/RAW/bus")
        kind = ctx.integer(num(row, 3, "/RAW/bus"), "/RAW/bus")
        if kind not in (1, 2, 3):
            ctx.fail(
                "/RAW/bus",
                "Only active PQ/PV/reference buses are supported.",
                unsupported=True,
            )
        bus_ids.add(identifier)
        buses.append(
            Bus(
                id=identifier,
                base_kv=num(row, 2, "/RAW/bus"),
                v_min=num(row, 10, "/RAW/bus"),
                v_max=num(row, 9, "/RAW/bus"),
            )
        )
        controls[identifier] = BusControl(
            bus=identifier,
            kind={1: "pq", 2: "pv", 3: "reference"}[kind],
            voltage=num(row, 7, "/RAW/bus"),
            angle=num(row, 8, "/RAW/bus") * pi / 180,
        )
    for row in sections[1]:
        path = "/RAW/load"
        if len(row) not in (13, 14):
            ctx.fail(path, "Expected 13 or 14 RAW33 load fields.")
        identifier = bid(row[0], path)
        active = ctx.flag(num(row, 2, path), path)
        if active and any(num(row, i, path) for i in (7, 8, 9, 10)):
            ctx.fail(
                path,
                "Active constant-current/admittance loads require ZIP semantics.",
                unsupported=True,
            )
        loads.append(
            Load(
                id=f"load:{identifier}:{row[1]}",
                bus=identifier,
                p=num(row, 5, path) / base,
                q=num(row, 6, path) / base,
                in_service=active,
            )
        )
        if not active and any(num(row, i, path) for i in (7, 8, 9, 10)):
            ctx.drop(
                f"{path}/{identifier}:{row[1]}", "Inactive ZIP coefficients omitted."
            )
    for row in sections[2]:
        path = "/RAW/fixed_shunt"
        if len(row) != 5:
            ctx.fail(path, "Expected five fixed-shunt fields.")
        identifier = bid(row[0], path)
        if identifier not in bus_ids:
            ctx.fail(path, "Missing shunt bus.")
        if ctx.flag(num(row, 2, path), path):
            shunts.append(
                Shunt(
                    bus=identifier, g=num(row, 3, path) / base, b=num(row, 4, path) / base
                )
            )
        else:
            ctx.drop(
                f"{path}/{identifier}:{row[1]}",
                "Out-of-service fixed shunt omitted from admittance.",
            )
    machine_data: dict[tuple[str, str], tuple[str, bool, float, float, float]] = {}
    active_generator_buses: set[str] = set()
    for row in sections[3]:
        path = "/RAW/generator"
        if len(row) not in (20, 22, 24, 26, 28):
            ctx.fail(
                path, "Expected RAW33 generator with complete owner pairs and WMOD/WPF."
            )
        identifier = bid(row[0], path)
        active = ctx.flag(num(row, 14, path), path)
        if active:
            if identifier in active_generator_buses:
                ctx.fail(
                    path,
                    "Multiple active RAW generators at one bus require RMPCT sharing semantics.",
                    unsupported=True,
                )
            active_generator_buses.add(identifier)
        remote = num(row, 7, path)
        if active and remote not in (0, float(identifier)):
            ctx.fail(
                path,
                "Remote generator voltage regulation is unsupported.",
                unsupported=True,
            )
        if active and (
            num(row, len(row) - 2, path) != 0
            or any(num(row, i, path) for i in (11, 12))
            or num(row, 13, path) != 1
        ):
            ctx.fail(
                path,
                "Wind control and generator step-up transformer fields are unsupported.",
                unsupported=True,
            )
        gid = f"gen:{identifier}:{row[1]}"
        key = (identifier, row[1])
        if key in machine_data:
            ctx.fail(path, "Duplicate generator bus/ID.")
        machine_data[key] = (
            gid,
            active,
            num(row, 8, path),
            num(row, 9, path),
            num(row, 10, path),
        )
        ctx.generator_voltage(controls, identifier, num(row, 6, path), active, path)
        generators.append(
            Generator(
                id=gid,
                bus=identifier,
                p=num(row, 2, path) / base,
                q=num(row, 3, path) / base,
                q_max=num(row, 4, path) / base,
                q_min=num(row, 5, path) / base,
                p_max=num(row, 16, path) / base,
                p_min=num(row, 17, path) / base,
                in_service=active,
            )
        )
    for row in sections[4]:
        path = "/RAW/branch"
        if len(row) not in (16, 18, 20, 22, 24):
            ctx.fail(path, "Expected RAW33 branch fields with complete owner pairs.")
        first = bid(row[0], path)
        second = str(abs(ctx.integer(num(row, 1, path), path)))
        active = ctx.flag(num(row, 13, path), path)
        rate = num(row, 6, path)
        if rate < 0:
            ctx.fail(path, "Negative branch rating.")
        branches.append(
            Branch(
                id=f"branch:{first}:{second}:{row[2]}",
                from_bus=first,
                to_bus=second,
                r=num(row, 3, path),
                x=num(row, 4, path),
                b=num(row, 5, path),
                rate=rate / base if rate else inf,
                in_service=active,
            )
        )
        if active:
            for bus, gi, bi in ((first, 9, 10), (second, 11, 12)):
                g, b = num(row, gi, path), num(row, bi, path)
                if g or b:
                    shunts.append(Shunt(bus=bus, g=g, b=b))
    transformer_rows = sections[5]
    cursor = 0
    while cursor < len(transformer_rows):
        path = f"/RAW/transformer/{cursor}"
        row = transformer_rows[cursor]
        if len(row) < 12:
            ctx.fail(path, "Incomplete transformer header.")
        if num(row, 2, path) != 0:
            ctx.fail(
                path, "Three-winding transformers are unsupported.", unsupported=True
            )
        if cursor + 4 > len(transformer_rows):
            ctx.fail(path, "Incomplete two-winding transformer record.")
        impedance, first_winding, second_winding = transformer_rows[
            cursor + 1 : cursor + 4
        ]
        cursor += 4
        if len(impedance) != 3 or len(first_winding) != 17 or len(second_winding) != 2:
            ctx.fail(path, "Expected 3/17/2 two-winding continuation fields.")
        active = ctx.flag(num(row, 11, path), path)
        if any(num(row, i, path) != 1 for i in (4, 5, 6)):
            ctx.fail(
                path, "Only CW=CZ=CM=1 transformer coding is supported.", unsupported=True
            )
        if any(num(row, i, path) for i in (7, 8)) or any(
            num(first_winding, i, path) for i in (6, 13, 14, 15, 16)
        ):
            ctx.fail(
                path,
                "Magnetizing branches, automatic control and corrections are unsupported.",
                unsupported=True,
            )
        first, second = bid(row[0], path), bid(row[1], path)
        tap1, tap2 = num(first_winding, 0, path), num(second_winding, 0, path)
        if tap1 <= 0 or tap2 <= 0:
            ctx.fail(path, "Transformer winding ratios must be positive.")
        rate = num(first_winding, 3, path)
        if rate < 0:
            ctx.fail(path, "Negative transformer rating.")
        # CZ=1 impedance is on system MVA base. Moving the secondary ideal
        # winding ratio to the primary multiplies series impedance by tap2².
        branches.append(
            Branch(
                id=f"transformer:{first}:{second}:{row[3]}",
                from_bus=first,
                to_bus=second,
                r=num(impedance, 0, path) * tap2**2,
                x=num(impedance, 1, path) * tap2**2,
                tap=tap1 / tap2,
                phase=num(first_winding, 2, path) * pi / 180,
                rate=rate / base if rate else inf,
                in_service=active,
            )
        )
    dynamics = []
    seen: set[tuple[str, str]] = set()
    # Slash terminates DYR records, including multiline records. Quoted strings
    # may not contain slash in this closed grammar (model and machine IDs only).
    records = dyr_text.split("/")
    if records[-1].strip():
        ctx.fail("/DYR", "Every DYR record requires a slash terminator.")
    for index, record in enumerate(records[:-1]):
        if not record.strip():
            continue
        path = f"/DYR/{index}"
        ctx.row(path)
        row = _psse_fields(ctx, record, path)
        if len(row) < 3:
            ctx.fail(path, "Expected bus, quoted model, machine ID and parameters.")
        key = (bid(row[0], path), row[2])
        if key not in machine_data:
            ctx.fail(path, "DYR references a missing generator.")
        gid, active, mbase, resistance, reactance = machine_data[key]
        if not active:
            ctx.drop(path, f"Inactive generator model {row[1]} omitted.")
            continue
        if row[1].upper() != "GENCLS":
            ctx.fail(
                path,
                f"Unknown active dynamic model {row[1]!r}; only GENCLS is supported.",
                unsupported=True,
            )
        if len(row) != 5 or key in seen:
            ctx.fail(
                path, "GENCLS requires exactly H,D and one record per active generator."
            )
        inertia, damping = num(row, 3, path), num(row, 4, path)
        if inertia <= 0 or damping < 0 or mbase <= 0 or reactance <= 0 or resistance < 0:
            ctx.fail(path, "GENCLS requires H>0,D>=0,MBASE>0,ZX>0,ZR>=0.")
        seen.add(key)
        dynamics.append(
            ClassicalMachine(
                generator=gid,
                inertia=inertia,
                damping=damping,
                base_mva=mbase,
                xd_prime=reactance,
                stator_resistance=resistance,
            )
        )
    if seen != {key for key, value in machine_data.items() if value[1]}:
        ctx.fail("/DYR", "Every active RAW generator requires an explicit GENCLS record.")
    ctx.drop(
        "/RAW/metadata",
        "Names, area/zone/owner labels, emergency limits, ratings B/C, dispatch "
        "participation and load-scaling flags are not used in the fixed-dispatch "
        "RMS network.",
    )
    return ctx.finish(
        buses,
        tuple(controls.values()),
        branches,
        generators,
        loads,
        shunts,
        base,
        frequency,
        tuple(dynamics),
    )


_CIM = "http://iec.ch/TC57/2013/CIM-schema-cim16#"
_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_MD = "http://iec.ch/TC57/61970-552/ModelDescription/1#"
_PROFILES = {
    "http://entsoe.eu/CIM/EquipmentCore/3/1": "EQ",
    "http://entsoe.eu/CIM/Topology/4/1": "TP",
    "http://entsoe.eu/CIM/SteadyStateHypothesis/1/1": "SSH",
    "http://entsoe.eu/CIM/StateVariables/4/1": "SV",
}
_COMMON_PROPERTIES = {
    "IdentifiedObject.name",
    "IdentifiedObject.description",
    "IdentifiedObject.mRID",
}
_CIM_PROPERTIES = {
    "BaseVoltage": {"BaseVoltage.nominalVoltage"},
    "TopologicalNode": {"TopologicalNode.BaseVoltage"},
    "Terminal": {
        "Terminal.ConductingEquipment",
        "Terminal.TopologicalNode",
        "ACDCTerminal.connected",
        "ACDCTerminal.sequenceNumber",
        "Terminal.phases",
    },
    "ACLineSegment": {
        "ConductingEquipment.BaseVoltage",
        "ACLineSegment.r",
        "ACLineSegment.x",
        "ACLineSegment.bch",
        "ACLineSegment.gch",
        "Equipment.aggregate",
    },
    "EnergyConsumer": {"EnergyConsumer.p", "EnergyConsumer.q", "Equipment.aggregate"},
    "SynchronousMachine": {
        "RotatingMachine.p",
        "RotatingMachine.q",
        "RotatingMachine.ratedS",
        "RotatingMachine.ratedU",
        "RotatingMachine.GeneratingUnit",
        "SynchronousMachine.minQ",
        "SynchronousMachine.maxQ",
        "SynchronousMachine.referencePriority",
        "SynchronousMachine.operatingMode",
        "SynchronousMachine.type",
        "RegulatingCondEq.RegulatingControl",
        "RegulatingCondEq.controlEnabled",
        "Equipment.aggregate",
    },
    "GeneratingUnit": {
        "GeneratingUnit.minOperatingP",
        "GeneratingUnit.maxOperatingP",
        "GeneratingUnit.normalPF",
    },
    "RegulatingControl": {
        "RegulatingControl.Terminal",
        "RegulatingControl.mode",
        "RegulatingControl.targetValue",
        "RegulatingControl.enabled",
        "RegulatingControl.discrete",
        "RegulatingControl.targetDeadband",
    },
    "LinearShuntCompensator": {
        "LinearShuntCompensator.gPerSection",
        "LinearShuntCompensator.bPerSection",
        "ShuntCompensator.nomU",
        "ShuntCompensator.sections",
        "ShuntCompensator.normalSections",
        "ShuntCompensator.maximumSections",
        "RegulatingCondEq.controlEnabled",
        "Equipment.aggregate",
    },
    "SvVoltage": {"SvVoltage.TopologicalNode", "SvVoltage.v", "SvVoltage.angle"},
}
_REF_PROPERTIES = {
    "TopologicalNode.BaseVoltage",
    "Terminal.ConductingEquipment",
    "Terminal.TopologicalNode",
    "ConductingEquipment.BaseVoltage",
    "RotatingMachine.GeneratingUnit",
    "RegulatingCondEq.RegulatingControl",
    "RegulatingControl.Terminal",
    "SvVoltage.TopologicalNode",
}
_ENUM_PROPERTIES = {
    "Terminal.phases",
    "SynchronousMachine.operatingMode",
    "SynchronousMachine.type",
    "RegulatingControl.mode",
}


def _rdf_id(value: str) -> str:
    return value[1:] if value.startswith("#") else value


def _cgmes_resources(
    ctx: _Import, texts: Sequence[str]
) -> dict[str, tuple[str, dict[str, str]]]:
    resources: dict[str, tuple[str, dict[str, str]]] = {}
    profiles: set[str] = set()
    for document, text in enumerate(texts):
        path = f"/CGMES/document/{document}"
        if re.search(
            r"<!\s*(?:DOCTYPE|ENTITY|ELEMENT|ATTLIST|NOTATION)|<!\[CDATA\[|<\?(?!xml(?:\s|\?>))",
            text,
            re.IGNORECASE,
        ):
            ctx.fail(
                path,
                "DTD, entities, CDATA and processing instructions are forbidden.",
                unsupported=True,
            )
        parser = ET.XMLPullParser(events=("start", "end"))
        depth = 0
        root = None
        try:
            for start in range(0, len(text), 4096):
                parser.feed(text[start : start + 4096])
                for event, element in parser.read_events():
                    if event == "start":
                        depth += 1
                        ctx.row(path)
                        if depth > ctx.limits.max_xml_depth:
                            ctx.fail(path, "XML nesting exceeds depth limit.")
                        if root is None:
                            root = element
                        ctx.token(element.tag, path)
                        for key, value in element.attrib.items():
                            ctx.token(key, path)
                            ctx.token(value, path)
                    else:
                        if element.text and element.text.strip():
                            ctx.token(element.text.strip(), path)
                        depth -= 1
            parser.close()
        except ET.ParseError as exc:
            ctx.fail(path, f"Malformed RDF/XML: {exc}")
        if root is None or root.tag != f"{{{_RDF}}}RDF" or root.attrib:
            ctx.fail(path, "Expected an unadorned rdf:RDF root.")
        if root.text and root.text.strip():
            ctx.fail(path, "Unexpected RDF root text.")
        for element in root:
            if element.tail and element.tail.strip():
                ctx.fail(path, "Mixed RDF content is unsupported.")
            allowed_attributes = {f"{{{_RDF}}}ID", f"{{{_RDF}}}about"}
            if (
                not set(element.attrib).issubset(allowed_attributes)
                or len(element.attrib) != 1
            ):
                ctx.fail(path, "Resources require exactly one rdf:ID or rdf:about.")
            identifier = _rdf_id(next(iter(element.attrib.values())))
            if not identifier:
                ctx.fail(path, "Empty RDF resource identity.")
            if element.text and element.text.strip():
                ctx.fail(path, "Resource text is unsupported.")
            kind = element.tag
            children = list(element)
            if kind == f"{{{_RDF}}}Description":
                types = [child for child in children if child.tag == f"{{{_RDF}}}type"]
                if len(types) != 1 or set(types[0].attrib) != {f"{{{_RDF}}}resource"}:
                    ctx.fail(
                        path, "rdf:Description requires exactly one explicit rdf:type."
                    )
                if (
                    len(types[0])
                    or (types[0].text and types[0].text.strip())
                    or (types[0].tail and types[0].tail.strip())
                ):
                    ctx.fail(path, "rdf:type must be a flat resource reference.")
                uri = types[0].attrib[f"{{{_RDF}}}resource"]
                if not uri.startswith(_CIM):
                    ctx.fail(path, "Unsupported RDF type namespace.", unsupported=True)
                kind = f"{{{_CIM}}}{uri[len(_CIM) :]}"
                children.remove(types[0])
            if kind == f"{{{_MD}}}FullModel":
                for prop in children:
                    name = prop.tag
                    if name == f"{{{_MD}}}Model.profile":
                        profile = (prop.text or "").strip()
                        if profile not in _PROFILES:
                            ctx.fail(
                                path,
                                f"Unsupported CGMES profile {profile!r}.",
                                unsupported=True,
                            )
                        profiles.add(_PROFILES[profile])
                    elif name not in {
                        f"{{{_MD}}}Model.{part}"
                        for part in (
                            "version",
                            "created",
                            "scenarioTime",
                            "description",
                            "modelingAuthoritySet",
                            "DependentOn",
                        )
                    }:
                        ctx.fail(
                            path,
                            "Unsupported FullModel metadata property.",
                            unsupported=True,
                        )
                    if len(prop) or (prop.tail and prop.tail.strip()):
                        ctx.fail(path, "Nested or mixed model metadata is unsupported.")
                    if name == f"{{{_MD}}}Model.DependentOn":
                        if set(prop.attrib) != {f"{{{_RDF}}}resource"} or (
                            prop.text and prop.text.strip()
                        ):
                            ctx.fail(
                                path, "Model.DependentOn must be a resource reference."
                            )
                    elif prop.attrib:
                        ctx.fail(path, "Model metadata must be literal text.")
                continue
            if not kind.startswith(f"{{{_CIM}}}"):
                ctx.fail(
                    path, "Only CIM16 resource classes are supported.", unsupported=True
                )
            kind = kind[len(_CIM) + 2 :]
            if kind not in _CIM_PROPERTIES:
                ctx.fail(
                    f"/CGMES/{identifier}",
                    f"Unsupported CIM class {kind!r}.",
                    unsupported=True,
                )
            previous = resources.get(identifier)
            if previous is not None and previous[0] != kind:
                ctx.fail(path, "Conflicting resource classes across profiles.")
            properties = {} if previous is None else previous[1]
            for prop in children:
                if (
                    len(prop)
                    or not prop.tag.startswith(f"{{{_CIM}}}")
                    or (prop.tail and prop.tail.strip())
                ):
                    ctx.fail(
                        path,
                        "Only flat CIM property literals/resource links are supported.",
                    )
                name = prop.tag[len(_CIM) + 2 :]
                if name not in _CIM_PROPERTIES[kind] | _COMMON_PROPERTIES:
                    ctx.fail(
                        f"/CGMES/{identifier}/{name}",
                        "Unsupported CIM property semantics.",
                        unsupported=True,
                    )
                if name in _REF_PROPERTIES | _ENUM_PROPERTIES:
                    if set(prop.attrib) != {f"{{{_RDF}}}resource"} or (
                        prop.text and prop.text.strip()
                    ):
                        ctx.fail(path, "Reference property requires rdf:resource only.")
                    value = prop.attrib[f"{{{_RDF}}}resource"]
                    if name in _REF_PROPERTIES:
                        value = _rdf_id(value)
                else:
                    if prop.attrib:
                        ctx.fail(path, "Literal property cannot have RDF attributes.")
                    value = (prop.text or "").strip()
                if name in properties and properties[name] != value:
                    ctx.fail(path, "Conflicting property values across profiles.")
                properties[name] = value
            resources[identifier] = (kind, properties)
    if not {"EQ", "TP", "SSH"}.issubset(profiles):
        ctx.fail(
            "/CGMES/profiles",
            "Explicit CGMES 2.4.15 EQ-Core, TP and SSH profiles are required.",
        )
    for identifier, (_, properties) in resources.items():
        for name in _REF_PROPERTIES & properties.keys():
            if properties[name] not in resources:
                ctx.fail(
                    f"/CGMES/{identifier}/{name}", "Dangling RDF resource reference."
                )
    return resources


def parse_cgmes(
    text: str | Sequence[str],
    *,
    base_mva: float = 100.0,
    frequency: float = 50.0,
    limits: PowerParserLimits = _DEFAULT_LIMITS,
) -> PowerCaseAdaptation:
    """Import the explicit CGMES 2.4.15/CIM16 balanced bus-branch subset.

    Accept one merged RDF/XML document or an EQ/TP/SSH document sequence. Exact
    namespace URIs/profile URIs are required. rdf:ID/about/resource and explicitly
    typed rdf:Description are resolved locally; no URL/entity is fetched. Optional
    SvVoltage supplies kV/degrees. CIM equipment p/q is inward (load-positive),
    hence generator p/q is negated. AC-line ohms/siemens use each node's line-line
    kV base; unequal-base lines fail. Shunts must be fixed, linear and unregulated.
    Unknown classes/properties fail even if apparently inactive: their activation
    semantics cannot be inferred safely. No full-CGMES conformance is claimed.
    """
    if not isinstance(text, (str, Sequence)):
        raise TypeError("CGMES input must be text or a finite sequence of documents.")
    texts = (text,) if isinstance(text, str) else text
    ctx = _Import("CGMES-2.4.15-CIM16-balanced", texts, limits)
    if (
        not isfinite(base_mva)
        or base_mva <= 0
        or not isfinite(frequency)
        or frequency <= 0
    ):
        ctx.fail("/CGMES/base", "Base MVA and frequency must be finite and positive.")
    resources = _cgmes_resources(ctx, texts)

    def props(identifier: str, kind: str | None = None) -> dict[str, str]:
        if identifier not in resources or (
            kind is not None and resources[identifier][0] != kind
        ):
            ctx.fail(f"/CGMES/{identifier}", f"Expected existing {kind or 'resource'}.")
        return resources[identifier][1]

    def value(identifier: str, name: str, default: str | None = None) -> str:
        result = props(identifier).get(name, default)
        if result is None:
            ctx.fail(f"/CGMES/{identifier}/{name}", "Required property is missing.")
        return result

    def number(identifier: str, name: str, default: str | None = None) -> float:
        return ctx.number(value(identifier, name, default), f"/CGMES/{identifier}/{name}")

    def flag(identifier: str, name: str, default: str | None = None) -> bool:
        token = value(identifier, name, default)
        if token not in ("true", "false", "1", "0"):
            ctx.fail(f"/CGMES/{identifier}/{name}", "Expected an XML boolean.")
        return token in ("true", "1")

    terminals: dict[str, list[tuple[int, str, str, bool]]] = {}
    bus_base: dict[str, float] = {}
    voltages: dict[str, tuple[float, float]] = {}
    for identifier, (kind, properties) in resources.items():
        if "Equipment.aggregate" in properties and flag(
            identifier, "Equipment.aggregate"
        ):
            ctx.fail(
                f"/CGMES/{identifier}",
                "Aggregated equipment semantics are unsupported.",
                unsupported=True,
            )
        if kind == "TopologicalNode":
            base_id = value(identifier, "TopologicalNode.BaseVoltage")
            props(base_id, "BaseVoltage")
            kv = number(base_id, "BaseVoltage.nominalVoltage")
            if kv <= 0:
                ctx.fail(f"/CGMES/{identifier}", "Voltage base must be positive.")
            bus_base[identifier] = kv
        elif kind == "Terminal":
            if (
                value(identifier, "Terminal.phases", _CIM + "PhaseCode.ABC")
                != _CIM + "PhaseCode.ABC"
            ):
                ctx.fail(
                    f"/CGMES/{identifier}",
                    "Only balanced ABC terminals are supported.",
                    unsupported=True,
                )
            equipment = value(identifier, "Terminal.ConductingEquipment")
            node = value(identifier, "Terminal.TopologicalNode")
            props(node, "TopologicalNode")
            sequence = ctx.integer(
                number(identifier, "ACDCTerminal.sequenceNumber"), f"/CGMES/{identifier}"
            )
            connected = flag(identifier, "ACDCTerminal.connected")
            terminals.setdefault(equipment, []).append(
                (sequence, identifier, node, connected)
            )
        elif kind == "SvVoltage":
            node = value(identifier, "SvVoltage.TopologicalNode")
            props(node, "TopologicalNode")
            if node in voltages:
                ctx.fail(f"/CGMES/{identifier}", "Duplicate node SvVoltage.")
            voltages[node] = (
                number(identifier, "SvVoltage.v"),
                number(identifier, "SvVoltage.angle") * pi / 180,
            )

    def ends(identifier: str, count: int) -> list[tuple[int, str, str, bool]]:
        result = sorted(terminals.get(identifier, ()))
        if len(result) != count or [item[0] for item in result] != list(
            range(1, count + 1)
        ):
            ctx.fail(
                f"/CGMES/{identifier}",
                f"Expected exactly {count} consecutively numbered terminals.",
            )
        return result

    branches, generators, loads, shunts = [], [], [], []
    modes = {identifier: "pq" for identifier in bus_base}
    setpoints: dict[str, float] = {}
    used_controls: set[str] = set()
    used_units: set[str] = set()
    equipment_classes = {
        "ACLineSegment",
        "EnergyConsumer",
        "SynchronousMachine",
        "LinearShuntCompensator",
    }
    for equipment in terminals:
        if resources[equipment][0] not in equipment_classes:
            ctx.fail(
                f"/CGMES/{equipment}",
                "Terminal references unsupported conducting-equipment class.",
            )
    for identifier, (kind, properties) in resources.items():
        path = f"/CGMES/{identifier}"
        if kind == "ACLineSegment":
            first, second = ends(identifier, 2)
            bus1, bus2 = first[2], second[2]
            if bus_base[bus1] != bus_base[bus2]:
                ctx.fail(
                    path,
                    "AC lines cannot connect unequal voltage bases.",
                    unsupported=True,
                )
            if "ConductingEquipment.BaseVoltage" in properties:
                base_id = value(identifier, "ConductingEquipment.BaseVoltage")
                props(base_id, "BaseVoltage")
                if number(base_id, "BaseVoltage.nominalVoltage") != bus_base[bus1]:
                    ctx.fail(path, "Equipment and terminal voltage bases disagree.")
            zbase = bus_base[bus1] ** 2 / base_mva
            active = first[3] and second[3]
            if first[3] != second[3]:
                ctx.fail(
                    path,
                    "One-ended open lines require disconnected-end charging semantics.",
                    unsupported=True,
                )
            branches.append(
                Branch(
                    id=identifier,
                    from_bus=bus1,
                    to_bus=bus2,
                    r=number(identifier, "ACLineSegment.r") / zbase,
                    x=number(identifier, "ACLineSegment.x") / zbase,
                    b=number(identifier, "ACLineSegment.bch", "0") * zbase,
                    in_service=active,
                )
            )
            g = number(identifier, "ACLineSegment.gch", "0") * zbase / 2
            if active and g:
                shunts.extend((Shunt(bus=bus1, g=g), Shunt(bus=bus2, g=g)))
        elif kind == "EnergyConsumer":
            terminal = ends(identifier, 1)[0]
            loads.append(
                Load(
                    id=identifier,
                    bus=terminal[2],
                    p=number(identifier, "EnergyConsumer.p") / base_mva,
                    q=number(identifier, "EnergyConsumer.q") / base_mva,
                    in_service=terminal[3],
                )
            )
        elif kind == "SynchronousMachine":
            terminal = ends(identifier, 1)[0]
            bus, active = terminal[2], terminal[3]
            if (
                value(identifier, "SynchronousMachine.operatingMode")
                != _CIM + "SynchronousMachineOperatingMode.generator"
            ):
                ctx.fail(
                    path, "Only generator operating mode is supported.", unsupported=True
                )
            if (
                value(
                    identifier,
                    "SynchronousMachine.type",
                    _CIM + "SynchronousMachineKind.generator",
                )
                != _CIM + "SynchronousMachineKind.generator"
            ):
                ctx.fail(
                    path, "Only generator machine kind is supported.", unsupported=True
                )
            unit = value(identifier, "RotatingMachine.GeneratingUnit")
            props(unit, "GeneratingUnit")
            if unit in used_units:
                ctx.fail(
                    path,
                    "Shared generating-unit limits require allocation semantics.",
                    unsupported=True,
                )
            used_units.add(unit)
            priority = ctx.integer(
                number(identifier, "SynchronousMachine.referencePriority", "0"), path
            )
            if priority not in (0, 1):
                ctx.fail(
                    path,
                    "Only explicit referencePriority 0 or 1 is supported.",
                    unsupported=True,
                )
            enabled = flag(identifier, "RegulatingCondEq.controlEnabled", "false")
            if enabled:
                control = value(identifier, "RegulatingCondEq.RegulatingControl")
                props(control, "RegulatingControl")
                if control in used_controls:
                    ctx.fail(
                        path, "Shared voltage controls are unsupported.", unsupported=True
                    )
                used_controls.add(control)
                if (
                    value(control, "RegulatingControl.mode")
                    != _CIM + "RegulatingControlModeKind.voltage"
                    or value(control, "RegulatingControl.Terminal") != terminal[1]
                ):
                    ctx.fail(
                        path,
                        "Only local terminal voltage regulation is supported.",
                        unsupported=True,
                    )
                if (
                    not flag(control, "RegulatingControl.enabled")
                    or flag(control, "RegulatingControl.discrete", "false")
                    or number(control, "RegulatingControl.targetDeadband", "0") != 0
                ):
                    ctx.fail(
                        path,
                        "Disabled, discrete or deadband voltage controls are unsupported.",
                        unsupported=True,
                    )
                voltage = number(control, "RegulatingControl.targetValue") / bus_base[bus]
                if active:
                    if bus in setpoints and setpoints[bus] != voltage:
                        ctx.fail(path, "Conflicting generator voltage setpoints.")
                    setpoints[bus] = voltage
                    modes[bus] = (
                        "reference" if priority or modes[bus] == "reference" else "pv"
                    )
            elif active and priority:
                ctx.fail(
                    path, "Reference generator requires enabled local voltage control."
                )
            generators.append(
                Generator(
                    id=identifier,
                    bus=bus,
                    p=-number(identifier, "RotatingMachine.p") / base_mva,
                    q=-number(identifier, "RotatingMachine.q") / base_mva,
                    p_min=number(unit, "GeneratingUnit.minOperatingP") / base_mva,
                    p_max=number(unit, "GeneratingUnit.maxOperatingP") / base_mva,
                    q_min=number(identifier, "SynchronousMachine.minQ") / base_mva,
                    q_max=number(identifier, "SynchronousMachine.maxQ") / base_mva,
                    in_service=active,
                )
            )
        elif kind == "LinearShuntCompensator":
            terminal = ends(identifier, 1)[0]
            if flag(identifier, "RegulatingCondEq.controlEnabled", "false"):
                ctx.fail(
                    path, "Active shunt regulation is unsupported.", unsupported=True
                )
            sections = number(identifier, "ShuntCompensator.sections")
            maximum = number(identifier, "ShuntCompensator.maximumSections")
            nominal = number(identifier, "ShuntCompensator.nomU")
            if (
                sections < 0
                or sections > maximum
                or sections != int(sections)
                or nominal <= 0
            ):
                ctx.fail(path, "Invalid fixed shunt section count or nominal voltage.")
            zbase = bus_base[terminal[2]] ** 2 / base_mva
            if terminal[3]:
                shunts.append(
                    Shunt(
                        bus=terminal[2],
                        g=number(identifier, "LinearShuntCompensator.gPerSection", "0")
                        * sections
                        * zbase,
                        b=number(identifier, "LinearShuntCompensator.bPerSection")
                        * sections
                        * zbase,
                    )
                )
            else:
                ctx.drop(path, "Disconnected shunt omitted from admittance.")
    for identifier, (kind, _) in resources.items():
        if kind == "RegulatingControl" and identifier not in used_controls:
            ctx.fail(
                f"/CGMES/{identifier}",
                "Unconsumed regulating-control semantics.",
                unsupported=True,
            )
        if kind == "GeneratingUnit" and identifier not in used_units:
            ctx.fail(
                f"/CGMES/{identifier}",
                "GeneratingUnit has no supported synchronous machine.",
            )
    buses = [Bus(id=identifier, base_kv=kv) for identifier, kv in bus_base.items()]
    controls = [
        BusControl(
            bus=identifier,
            kind=modes[identifier],
            voltage=setpoints.get(
                identifier, voltages.get(identifier, (kv, 0.0))[0] / kv
            ),
            angle=voltages.get(identifier, (kv, 0.0))[1],
        )
        for identifier, kv in bus_base.items()
    ]
    ctx.drop(
        "/CGMES/metadata",
        "Names, model headers, rated machine data, normal power factors and normal "
        "shunt section settings are omitted; SSH operating values are authoritative.",
    )
    ctx.drop(
        "/CGMES/limits",
        "This closed profile carries no accepted operational-limit resources; "
        "native voltage bounds are 0.9–1.1 pu and line ratings are unconstrained.",
        synthesized=True,
    )
    ctx.drop(
        "/CGMES/base",
        f"Caller declares base_mva={base_mva} and frequency={frequency}; neither is inferred from model headers.",
        synthesized=True,
    )
    if len(voltages) != len(bus_base):
        ctx.drop(
            "/CGMES/initial_voltage",
            "Nodes without SvVoltage use zero angle and 1 pu magnitude unless a local generator setpoint is present.",
            synthesized=True,
        )
    return ctx.finish(
        buses, controls, branches, generators, loads, shunts, base_mva, frequency
    )
