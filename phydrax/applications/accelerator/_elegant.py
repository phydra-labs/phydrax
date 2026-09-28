#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned elegant oracle for one-dimensional CSR tracking on a :class:`CSRLattice`.

elegant (M. Borland, Argonne Advanced Photon Source; EPICS Open License, SPDX
``EPICS``; https://www.aps.anl.gov/Accelerator-Operations-Physics/Software) is
used only as a caller-pinned external executable; no source is copied. Validated
live: elegant 2026.3.0 (conda-forge, osx-arm64). References: M. Borland,
"elegant: A Flexible SDDS-Compliant Code for Accelerator Simulation", APS
LS-287 (2000); M. Borland, "Simple method for particle tracking with coherent
synchrotron radiation", PRST-AB 4, 070701 (2001).

:func:`elegant_csr_input` translates a :class:`CSRTrackingPlan` and an
:class:`AcceleratorBunch` into an elegant run file ``csr.ele``, a lattice
``csr.lte``, and a binary SDDS particle file ``beam.sdds``;
:func:`run_elegant_csr` runs the pinned binary and publishes the final particle
SDDS file ``csr.out`` as a bounded artifact; :func:`elegant_csr_bunch` reads it
back into accelerator coordinates.

Supported subset (everything else is refused before running): the
``"1d-steady"`` model and the unshielded ``"1d-transient-shielded"`` model
(``CSRCSBEND`` has no parallel-plate image model, so ``plate_gap`` is refused);
an electron bunch (``reference_charge = −1`` elementary charge and the
electron rest energy of the plan's scale, which must be referenced to SI);
active particles that are all valid, uniquely identified, and of one weight
(elegant macroparticles carry equal charge); the canonical momentum
normalization with either longitudinal sign.

Element mapping: a bend becomes ``CSRCSBEND`` (``L``, ``ANGLE = hL``, hard-edge
``E1``/``E2`` with ``HGAP = FINT = 0`` and ``EDGE_ORDER = 1``, ``N_SLICES``
from the plan substeps, ``STEADY_STATE`` from the model); a drift becomes
``CSRDRIFT`` (``N_KICKS`` from the substeps, Stupakov's exit transient
``USE_STUPAKOV = 1`` for the transient model and ``CSR = 0`` for the steady
model, whose wake vanishes in drifts). Both keep elegant's symplectic
integration: its ``LINEARIZE = 1`` matrices carry a spurious zeroth-order
time-of-flight term (−1 µm over the test chicane in 2026.3.0), while the exact
maps differ from the first-order maps of :func:`track_csr` only at second
order. A leading ``CHARGE`` element
carries the bunch charge. The histogram uses ``BIN_RANGE_FACTOR = 1.2`` and
enough ``BINS`` that the initial bin equals the plan's longitudinal cell; the
plan's Gaussian smoothing width ``w`` becomes the order-1 Savitzky–Golay
(moving-average) half-width ``n`` of equal variance, ``n(n+1)Δ²/3 = w²``.

Coordinates (lengths in meters through ``unit_si_map``): ``x`` and ``y``
unchanged; slopes ``x′ = (px/p₀)/p_z`` with ``p_z = √((1+δ)² − (px/p₀)² −
(py/p₀)²)`` and the exact inverse ``px/p₀ = (1+δ)x′/√(1+x′²+y′²)``; arrival
time ``t = σζ/(β₀c)`` on entry and ``ζ = σ(β₀ct − L)`` on exit with the
reference time of flight over the lattice length ``L`` (``σ = +1`` for
``"positive-late"``); momentum ``p = (1+δ)β₀γ₀`` in units of ``m_ec``.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never, NamedTuple

import equinox as eqx
import numpy as np

from ..._external_runtime import (
    PinnedExecutable,
    PinnedFileOutputs,
    PinnedFileRequest,
    run_pinned_command,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...interchange import AdapterLoss, AdapterReport, AdapterStatus
from ._beam import _late_sign, AcceleratorBunch
from ._csr import CSRTrackingPlan


_RUN = "csr.ele"
_LATTICE = "csr.lte"
_BEAM = "beam.sdds"
_OUTPUT = "csr.out"
# elegant requires an RPN definitions file; the translated decks use no RPN.
_DEFINITIONS = "defns.rpn"
_BIN_RANGE_FACTOR = 1.2
_COLUMNS = ("x", "xp", "y", "yp", "t", "p")
_SDDS_CODES = {
    "double": "f8",
    "float": "f4",
    "long": "i4",
    "ulong": "u4",
    "long64": "i8",
    "ulong64": "u8",
    "short": "i2",
    "ushort": "u2",
    "character": "S1",
}
_NAMELIST_FIELD = re.compile(r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|[^,\s]*)')


@dataclass(frozen=True, slots=True)
class ElegantCSRProvider:
    """A pinned ``elegant`` binary (external oracle only)."""

    executable: PinnedExecutable

    def __post_init__(self) -> None:
        if not isinstance(self.executable, PinnedExecutable):
            raise TypeError("executable must be a PinnedExecutable elegant binary.")


class ElegantCSRResult(StrictModule):
    """elegant-tracked bunch with its pinned-run identity and adapter report.

    ``bunch`` is the input bunch after the lattice in accelerator coordinates;
    particles elegant lost are inactive. ``output_sha256`` is the digest of the
    published ``csr.out`` artifact and the report's ``source_id``.
    """

    bunch: AcceleratorBunch
    provider_version: str = eqx.field(static=True)
    executable_sha256: str = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    output_sha256: str = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)


class _Units(NamedTuple):
    length: float
    time: float
    speed_of_light: float
    beta_gamma: float
    beta: float
    charge: float
    sign: float


class _SDDSPage(NamedTuple):
    parameters: dict[str, float | int | str]
    columns: dict[str, np.ndarray]


class _SDDSField(NamedTuple):
    name: str
    kind: str
    fixed_value: str | None


def _number(value: float, /) -> str:
    return repr(float(value))


def _checked_units(plan: CSRTrackingPlan, bunch: AcceleratorBunch, /) -> _Units:
    """Validate the supported subset and return the SI conversion of the pair."""
    if not isinstance(plan, CSRTrackingPlan):
        raise TypeError("plan must be a CSRTrackingPlan.")
    if not isinstance(bunch, AcceleratorBunch):
        raise TypeError("bunch must be an AcceleratorBunch.")
    csr = plan.plan
    match csr.model:
        case "1d-steady" | "1d-transient-shielded":
            pass
        case "3d-steady-igf" | "3d-retarded-mesh":
            raise ValueError("elegant models one-dimensional CSR only.")
        case _:
            assert_never(csr.model)
    if csr.plate_gap is not None:
        raise ValueError("elegant's CSRCSBEND has no parallel-plate shielding.")
    sign = _late_sign(bunch.convention)
    scale = csr.scale
    units = scale.unit_si_map()
    light = float(scale.speed_of_light)
    electron = float(scale.electron_mass * scale.speed_of_light**2)
    rest = float(bunch.reference_rest_energy)
    momentum = float(bunch.reference_momentum)
    if float(bunch.reference_charge) != -1.0 or not math.isclose(
        rest, electron, rel_tol=1.0e-9
    ):
        raise ValueError("The elegant oracle tracks electron bunches only.")
    if not math.isclose(rest, csr.rest_energy, rel_tol=1.0e-12) or not math.isclose(
        momentum, csr.momentum, rel_tol=1.0e-12
    ):
        raise ValueError("The bunch reference does not match the CSR plan.")
    active = np.asarray(bunch.active)
    if not np.all(np.asarray(bunch.valid)[active]):
        raise ValueError("The elegant oracle needs every active particle valid.")
    weights = np.asarray(bunch.weights, dtype=np.float64)[active]
    identifiers = np.asarray(bunch.particle_ids)[active]
    if weights.size < 2 or not np.all(weights == weights[0]):
        raise ValueError(
            "elegant macroparticles carry equal charge; the oracle needs at least "
            "two active particles of one weight."
        )
    if np.unique(identifiers).size != identifiers.size:
        raise ValueError("Active particle_ids must be unique.")
    beta_gamma = momentum / rest
    return _Units(
        length=units["length"][0],
        time=units["time"][0],
        speed_of_light=light,
        beta_gamma=beta_gamma,
        beta=beta_gamma / math.hypot(1.0, beta_gamma),
        charge=float(np.sum(weights))
        * float(scale.elementary_charge)
        * units["charge"][0],
        sign=sign,
    )


def _smoothing_halfwidth(width: float, spacing: float, /) -> int:
    """Moving-average half-width ``n`` whose variance ``n(n+1)Δ²/3`` is ``w²``."""
    ratio = width / spacing
    return round(0.5 * (math.sqrt(1.0 + 12.0 * ratio * ratio) - 1.0))


def _lattice_text(plan: CSRTrackingPlan, bins: int, halfwidth: int, /) -> str:
    csr = plan.plan
    steady = csr.model == "1d-steady"
    length_unit = csr.scale.unit_si_map()["length"][0]
    lattice = csr.lattice
    lengths = np.asarray(lattice.lengths)
    curvatures = np.asarray(lattice.curvatures)
    entrance = np.asarray(lattice.entrance_edges)
    exit_ = np.asarray(lattice.exit_edges)
    lines: list[str] = []
    names = ["Q"]
    for index, substeps in enumerate(plan.substeps):
        name = f"E{index}"
        length = _number(lengths[index] * length_unit)
        if curvatures[index] == 0.0:
            csr_flags = "CSR=0" if steady else "USE_STUPAKOV=1"
            lines.append(f"{name}: CSRDRIFT, L={length}, N_KICKS={substeps}, {csr_flags}")
        else:
            lines.append(
                f"{name}: CSRCSBEND, L={length}, "
                f"ANGLE={_number(curvatures[index] * lengths[index])}, "
                f"E1={_number(entrance[index])}, E2={_number(exit_[index])}, "
                f"HGAP=0, FINT=0, EDGE_ORDER=1, N_SLICES={substeps}, "
                f"BINS={bins}, BIN_RANGE_FACTOR={_number(_BIN_RANGE_FACTOR)}, "
                f"SG_HALFWIDTH={halfwidth}, SG_ORDER=1, "
                f"STEADY_STATE={1 if steady else 0}"
            )
        names.append(name)
    return "\n".join(lines) + f"\nCSR: LINE=({', '.join(names)})\n"


def _sdds_binary(columns: tuple[tuple[str, str, str, np.ndarray], ...], /) -> bytes:
    """One-page little-endian binary SDDS file of ``(name, units, type, values)``."""
    header = ["SDDS1"]
    fields: list[tuple[str, str]] = []
    for name, units, kind, _ in columns:
        header.append(f'&column name={name}, units="{units}", type={kind}, &end')
        fields.append((name, "<" + _SDDS_CODES[kind]))
    header.append("&data mode=binary, endian=little, &end")
    count = columns[0][3].shape[0]
    records = np.empty((count,), dtype=np.dtype(fields))
    for name, _, _, values in columns:
        records[name] = values
    return (
        ("\n".join(header) + "\n").encode() + struct.pack("<i", count) + records.tobytes()
    )


def elegant_csr_input(
    plan: CSRTrackingPlan, bunch: AcceleratorBunch, /
) -> dict[str, bytes]:
    """Translate the supported subset into ``csr.ele``, ``csr.lte``, and ``beam.sdds``."""
    units = _checked_units(plan, bunch)
    csr = plan.plan
    active = np.asarray(bunch.active)
    coordinates = np.asarray(bunch.coordinates, dtype=np.float64)[active]
    zeta = coordinates[:, 4]
    extent = float(np.max(zeta) - np.min(zeta))
    if extent <= 0.0:
        raise ValueError("The elegant CSR histogram needs a bunch of nonzero length.")
    spacing = csr.spacing[-1]
    bins = max(2, math.ceil(_BIN_RANGE_FACTOR * extent / spacing))
    halfwidth = _smoothing_halfwidth(csr.smoothing[-1], spacing)
    px, py, delta = coordinates[:, 1], coordinates[:, 3], coordinates[:, 5]
    longitudinal = (1.0 + delta) ** 2 - px * px - py * py
    if np.any(longitudinal <= 0.0):
        raise ValueError("Every active particle needs a forward momentum p_z > 0.")
    forward = np.sqrt(longitudinal)
    run = "\n".join(
        [
            "&run_setup",
            f"lattice = {_LATTICE},",
            "use_beamline = CSR,",
            f"p_central = {_number(units.beta_gamma)},",
            f"output = {_OUTPUT},",
            "default_order = 1,",
            "&end",
            "&run_control n_steps = 1, &end",
            f"&sdds_beam input = {_BEAM}, &end",
            "&track &end",
            "",
        ]
    )
    lattice = f"Q: CHARGE, TOTAL={_number(units.charge)}\n" + _lattice_text(
        plan, bins, halfwidth
    )
    beam = _sdds_binary(
        (
            ("x", "m", "double", coordinates[:, 0] * units.length),
            ("xp", "", "double", px / forward),
            ("y", "m", "double", coordinates[:, 2] * units.length),
            ("yp", "", "double", py / forward),
            (
                "t",
                "s",
                "double",
                units.sign * zeta / (units.beta * units.speed_of_light) * units.time,
            ),
            ("p", "m$be$nc", "double", (1.0 + delta) * units.beta_gamma),
            ("particleID", "", "long", np.asarray(bunch.particle_ids)[active]),
        )
    )
    return {
        _RUN: run.encode(),
        _LATTICE: lattice.encode(),
        _BEAM: beam,
        _DEFINITIONS: b"",
    }


def _namelist(text: str, /) -> tuple[str, dict[str, str]]:
    body = text.strip()
    command, _, rest = body[1:].partition(" ")
    rest = rest.rsplit("&end", 1)[0]
    values: dict[str, str] = {}
    for key, value in _NAMELIST_FIELD.findall(rest):
        if value.startswith('"'):
            value = value[1:-1].replace('\\"', '"')
        values[key] = value
    return command.strip().rstrip(","), values


def _parameter_value(text: str, kind: str, /) -> float | int | str:
    match kind:
        case "double" | "float":
            return float(text)
        case "long" | "ulong" | "long64" | "ulong64" | "short" | "ushort":
            return int(text)
        case "string" | "character":
            return text
        case _:
            raise ValueError(f"Unsupported SDDS type {kind!r}.")


def _read_sdds(data: bytes, /) -> tuple[_SDDSPage, ...]:
    """Read binary SDDS pages of numeric columns (SDDS protocol versions 1–5).

    Parameters may be numeric or strings; arrays, string columns, ASCII data,
    and undeclared byte order are refused. Every page must fit ``data``.
    """
    end = data.find(b"\n")
    if (
        not data.startswith(b"SDDS")
        or end < 0
        or data[4:end]
        not in (
            b"1",
            b"2",
            b"3",
            b"4",
            b"5",
        )
    ):
        raise ValueError("Not an SDDS file (protocol versions 1–5).")
    offset = end + 1
    order = ""
    parameters: list[_SDDSField] = []
    columns: list[_SDDSField] = []
    options: dict[str, str] | None = None
    pending = ""
    while options is None:
        end = data.find(b"\n", offset)
        if end < 0:
            raise ValueError("The SDDS header has no &data command.")
        line = data[offset:end].decode("ascii")
        offset = end + 1
        if not pending and line.startswith("!#"):
            order = {"little-endian": "<", "big-endian": ">"}.get(line[2:].strip(), order)
            continue
        pending = f"{pending} {line}" if pending else line
        if "&end" not in pending:
            continue
        command, values = _namelist(pending)
        pending = ""
        match command:
            case "description" | "associate":
                pass
            case "parameter" | "column":
                kind = values.get("type", "")
                if kind not in _SDDS_CODES and kind != "string":
                    raise ValueError(f"Unsupported SDDS type {kind!r}.")
                target = parameters if command == "parameter" else columns
                target.append(_SDDSField(values["name"], kind, values.get("fixed_value")))
            case "data":
                options = values
            case _:
                raise ValueError(f"Unsupported SDDS command &{command}.")
    if options.get("mode") != "binary":
        raise ValueError("Only binary SDDS data is supported.")
    if options.get("no_row_counts", "0") != "0":
        raise ValueError("SDDS pages without row counts are unsupported.")
    order = {"little": "<", "big": ">"}.get(options.get("endian", ""), order)
    if not order:
        raise ValueError("The SDDS byte order is undeclared.")
    if any(column.kind == "string" for column in columns):
        raise ValueError("String SDDS columns are unsupported.")
    column_major = options.get("column_major_order", "0") != "0"
    record = np.dtype(
        [(field.name, order + _SDDS_CODES[field.kind]) for field in columns]
    )
    pages: list[_SDDSPage] = []
    while offset < len(data):
        (rows,) = struct.unpack_from(order + "i", data, offset)
        offset += 4
        if rows == -(2**31):
            (rows,) = struct.unpack_from(order + "q", data, offset)
            offset += 8
        if rows < 0:
            raise ValueError("Negative SDDS row count.")
        values: dict[str, float | int | str] = {}
        for field in parameters:
            if field.fixed_value is not None:
                values[field.name] = _parameter_value(field.fixed_value, field.kind)
            elif field.kind == "string":
                (size,) = struct.unpack_from(order + "i", data, offset)
                if size < 0 or offset + 4 + size > len(data):
                    raise ValueError("Truncated SDDS string parameter.")
                values[field.name] = data[offset + 4 : offset + 4 + size].decode()
                offset += 4 + size
            else:
                code = np.dtype(order + _SDDS_CODES[field.kind])
                if offset + code.itemsize > len(data):
                    raise ValueError("Truncated SDDS parameter.")
                values[field.name] = np.frombuffer(data, code, 1, offset)[0].item()
                offset += code.itemsize
        if offset + rows * record.itemsize > len(data):
            raise ValueError("Truncated SDDS page.")
        arrays: dict[str, np.ndarray] = {}
        if column_major:
            for field in columns:
                code = np.dtype(order + _SDDS_CODES[field.kind])
                arrays[field.name] = np.frombuffer(data, code, rows, offset).copy()
                offset += rows * code.itemsize
        else:
            table = np.frombuffer(data, record, rows, offset)
            arrays = {field.name: table[field.name].copy() for field in columns}
            offset += rows * record.itemsize
        pages.append(_SDDSPage(values, arrays))
    return tuple(pages)


def elegant_csr_bunch(
    data: bytes, plan: CSRTrackingPlan, bunch: AcceleratorBunch, /
) -> AcceleratorBunch:
    """Read elegant's final particle SDDS ``data`` for ``bunch`` tracked by ``plan``.

    Particles are matched by ``particleID``; particles absent from the output
    (lost in elegant) become inactive with their input coordinates.
    """
    if not isinstance(data, bytes):
        raise TypeError("data must be the bytes of an SDDS file.")
    units = _checked_units(plan, bunch)
    pages = _read_sdds(data)
    if len(pages) != 1:
        raise ValueError("The elegant output must hold exactly one page.")
    page = pages[0]
    if any(name not in page.columns for name in (*_COLUMNS, "particleID")):
        raise ValueError("The elegant output lacks particle coordinate columns.")
    central = page.parameters.get("pCentral")
    if not isinstance(central, float) or not math.isclose(
        central, units.beta_gamma, rel_tol=1.0e-12
    ):
        raise ValueError("The elegant output reference momentum differs from the plan.")
    identifiers = np.asarray(bunch.particle_ids)
    active = np.asarray(bunch.active)
    rows = {int(identifier): row for row, identifier in enumerate(identifiers)}
    returned = page.columns["particleID"].astype(np.int64)
    if np.unique(returned).size != returned.size or not np.all(
        np.isin(returned, identifiers[active])
    ):
        raise ValueError("The elegant output particle identities are inconsistent.")
    xp = page.columns["xp"].astype(np.float64)
    yp = page.columns["yp"].astype(np.float64)
    delta = page.columns["p"].astype(np.float64) / units.beta_gamma - 1.0
    slope = (1.0 + delta) / np.sqrt(1.0 + xp * xp + yp * yp)
    lattice_length = plan.plan.lattice.total_length
    zeta = units.sign * (
        units.beta
        * units.speed_of_light
        * page.columns["t"].astype(np.float64)
        / units.time
        - lattice_length
    )
    final = np.stack(
        (
            page.columns["x"].astype(np.float64) / units.length,
            slope * xp,
            page.columns["y"].astype(np.float64) / units.length,
            slope * yp,
            zeta,
            delta,
        ),
        axis=-1,
    )
    if not np.all(np.isfinite(final)):
        raise ValueError("elegant returned nonfinite coordinates.")
    coordinates = np.asarray(bunch.coordinates, dtype=np.float64).copy()
    indices = np.asarray(
        [rows[int(identifier)] for identifier in returned], dtype=np.int64
    )
    coordinates[indices] = final
    tracked = np.zeros(active.shape, dtype=np.bool_)
    tracked[indices] = True
    return AcceleratorBunch(
        coordinates.astype(np.asarray(bunch.coordinates).dtype),
        bunch.weights,
        bunch.particle_ids,
        active=active & tracked,
        reference_rest_energy=float(bunch.reference_rest_energy),
        reference_momentum=float(bunch.reference_momentum),
        reference_charge=float(bunch.reference_charge),
        convention=bunch.convention,
        bunch_id=f"{bunch.bunch_id}:elegant:{plan.plan_id}",
    )


def _losses(
    plan: CSRTrackingPlan, lost: int, halfwidth: int, /
) -> tuple[AdapterLoss, ...]:
    csr = plan.plan
    losses = [
        AdapterLoss(
            "csr.grid",
            "import",
            "transformed",
            "elegant histograms the bunch over 1.2 times its current extent at "
            "every kick (bin width equal to the plan cell only initially) and "
            "differentiates it by Savitzky-Golay filtering, instead of the fixed "
            "plan grid with cell-integrated kernels.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "transport.order",
            "import",
            "transformed",
            "elegant integrates the exact bend and drift maps with its "
            "fourth-order symplectic integrator over N_SLICES = substeps (a "
            "constant time-of-flight offset of order step⁴), whereas track_csr "
            "applies first-order maps in px/p0, py/p0; they differ at second "
            "order in the transverse slopes and δ.",
            changes_interpretation=False,
        ),
    ]
    width = csr.smoothing[-1]
    if width > 0.0:
        losses.append(
            AdapterLoss(
                "csr.smoothing",
                "import",
                "transformed" if halfwidth > 0 else "dropped",
                f"The Gaussian density smoothing of width {width!r} is realized as "
                f"an order-1 Savitzky-Golay moving average of half-width "
                f"{halfwidth} bins with matched variance.",
                changes_interpretation=False,
            )
        )
    if csr.model == "1d-transient-shielded":
        losses.append(
            AdapterLoss(
                "csr.model",
                "import",
                "transformed",
                "elegant's transient CSR uses the ultrarelativistic entrance "
                "transient of Saldin et al. (1997) in CSRCSBEND and Stupakov's exit "
                "transient in CSRDRIFT, not the exact retarded line-charge integral "
                "over the full lattice and density history of 1d-transient-shielded.",
                changes_interpretation=True,
            )
        )
    if lost:
        losses.append(
            AdapterLoss(
                "particles",
                "import",
                "dropped",
                f"elegant lost {lost} particles; they are inactive in the result.",
                changes_interpretation=True,
            )
        )
    return tuple(losses)


def run_elegant_csr(
    provider: ElegantCSRProvider,
    plan: CSRTrackingPlan,
    bunch: AcceleratorBunch,
    destination: str | Path,
    /,
    *,
    timeout: float = 600.0,
    maximum_output_bytes: int = 1 << 30,
) -> ElegantCSRResult:
    """Track ``bunch`` through ``plan`` with the pinned elegant and read it back."""
    if not isinstance(provider, ElegantCSRProvider):
        raise TypeError("provider must be an ElegantCSRProvider.")
    inputs = elegant_csr_input(plan, bunch)
    artifacts = PinnedFileOutputs(
        str(destination),
        (PinnedFileRequest(_OUTPUT, maximum_output_bytes),),
        maximum_output_bytes,
    )
    run = run_pinned_command(
        provider.executable,
        (_RUN, f"-rpnDefns={_DEFINITIONS}"),
        inputs=inputs,
        timeout=timeout,
        artifacts=artifacts,
    )
    artifact = run.file_artifact(_OUTPUT)
    final = elegant_csr_bunch(Path(artifact.location).read_bytes(), plan, bunch)
    lost = int(np.sum(np.asarray(bunch.active)) - np.sum(np.asarray(final.active)))
    losses = _losses(
        plan, lost, _smoothing_halfwidth(plan.plan.smoothing[-1], plan.plan.spacing[-1])
    )
    target_id = canonical_fingerprint(
        {
            "kind": "elegant-csr-bunch",
            "arrays": array_tree_fingerprint(
                (final.coordinates, final.weights, final.particle_ids, final.active)
            ),
            "bunch_id": final.bunch_id,
        }
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        "elegant-sdds-particles",
        "phydrax-accelerator-bunch",
        source_id=artifact.sha256,
        target_id=target_id,
        coordinate_mapping=(
            "x -> x",
            "y -> y",
            "xp, yp -> px/p0, py/p0 = (1+δ)(x', y')/√(1+x'²+y'²)",
            "t -> zeta = σ(β0 c t − L)",
            "p -> delta = p/(β0γ0) − 1",
        ),
        preserved_fields=("particleID", "x", "y"),
        assumptions=(
            "elegant evaluates CSR with its own compiled physical constants.",
            "The reference time of flight is the CSRLattice length over β0c.",
            "CSRCSBEND N_SLICES and CSRDRIFT N_KICKS follow the plan substeps.",
        ),
        losses=losses,
    )
    return ElegantCSRResult(
        final,
        provider.executable.version,
        provider.executable.sha256,
        provider.executable.license_id,
        artifact.sha256,
        report,
    )


__all__ = [
    "ElegantCSRProvider",
    "ElegantCSRResult",
    "elegant_csr_bunch",
    "elegant_csr_input",
    "run_elegant_csr",
]
