#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Extract ENDF/B-VIII.0 fission-gas yields and fission energy release.

Regenerates the packaged ``phydrax/nuclear/data/fission_gas_yields.json`` from
the pinned ENDF/B-VIII.0 distribution:

- cumulative fission-product yields (MF8/MT459) from the neutron-induced
  fission-yield sublibrary distributed by NNDC;
- fission energy-release components (MF1/MT458) from the neutron sublibrary
  distributed by the IAEA Nuclear Data Section;
- ground-state half-lives from NUBASE2020 (IAEA AMDC distribution), used only to
  verify that the declared gas-nuclide selection is complete and long-lived.

Every download is verified against its pinned SHA-256 digest and byte size
before parsing. Regeneration refuses (non-zero exit) on any digest mismatch,
missing record, or failed selection check. After regeneration, pin the printed
digest and size in ``phydrax/nuclear/_fission_products.py``.

Usage::

    python tools/extract_fission_gas_yields.py [--cache-dir DIR] [--output PATH]
    python tools/extract_fission_gas_yields.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from phydrax.nuclear import FISSIONING_NUCLIDES, FissioningNuclide


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = _ROOT / "phydrax" / "nuclear" / "data" / "fission_gas_yields.json"
_DEFAULT_CACHE = _ROOT / ".tmp" / "endf-b-viii.0"

_TOOL = "tools/extract_fission_gas_yields.py"
# NUBASE2020 (Kondev et al., Chin. Phys. C 45, 030001 (2021)) year: 365.2422 d.
_SECONDS_PER_YEAR = 365.2422 * 86400.0
_HALF_LIFE_THRESHOLD_YEARS = 1.0
_KRYPTON = 36
_XENON = 54
# Stable or long-lived (T1/2 > 1 y) krypton and xenon ground states, verified
# against NUBASE2020 at extraction time. Xe-129 is excluded below.
_GAS_MASS_NUMBERS: Mapping[int, tuple[int, ...]] = {
    _KRYPTON: (78, 80, 81, 82, 83, 84, 85, 86),
    _XENON: (124, 126, 128, 130, 131, 132, 134, 136),
}
# Stable gas nuclides whose ENDF cumulative yield is carried by a precursor
# that does not decay on reactor time scales (I-129, T1/2 = 16.14 My).
_EXCLUDED_BY_PRECURSOR: Mapping[tuple[int, int], tuple[int, int]] = {
    (_XENON, 129): (53, 129),
}
_ELEMENT_SYMBOLS: Mapping[int, str] = {
    34: "Se",
    35: "Br",
    36: "Kr",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    92: "U",
    94: "Pu",
}
_ENERGY_COMPONENTS = ("EFR", "ENP", "END", "EGP", "EGD", "EB", "ENU", "ER", "ET")


@dataclass(frozen=True, slots=True)
class _Source:
    role: str
    url: str
    sha256: str
    size_bytes: int

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


_THERMAL_ENERGY_EV = 0.0253


@dataclass(frozen=True, slots=True)
class _System:
    """One fissioning nuclide and the incident energy its data are taken at.

    Fissile targets use the thermal point (0.0253 eV); fertile targets use the
    lowest incident energy tabulated in the NFY sublibrary for that target.
    """

    fissioning_nuclide: FissioningNuclide
    proton_number: int
    mass_number: int
    mat: int
    thermal: bool
    neutron: _Source

    @property
    def label(self) -> str:
        return _label(self.proton_number, self.mass_number)

    @property
    def yield_member(self) -> str:
        symbol = _ELEMENT_SYMBOLS[self.proton_number]
        return (
            f"ENDF-B-VIII.0_nfy/nfy-{self.proton_number:03d}_{symbol}_"
            f"{self.mass_number}.endf"
        )

    @property
    def neutron_member(self) -> str:
        return self.neutron.filename.removesuffix(".zip") + ".dat"


_NFY = _Source(
    "fission-product-yield-sublibrary",
    "https://www.nndc.bnl.gov/endf-b8.0/zips/ENDF-B-VIII.0_nfy.zip",
    "92c5371fdb21eecf4989f48828671b904186abc6386b3d7510c8fcee2ee5ffcf",
    2922601,
)
_NUBASE = _Source(
    "nubase2020-half-lives",
    "https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt",
    "1585a5eea86c5e17e90307c7e6e786d060049c4039e392a261ff6db977df9859",
    761906,
)
_NEUTRON_ROOT = "https://www-nds.iaea.org/public/download-endf/ENDF-B-VIII.0/n/"


def _neutron(filename: str, sha256: str, size_bytes: int, /) -> _Source:
    return _Source("neutron-sublibrary", _NEUTRON_ROOT + filename, sha256, size_bytes)


def _system(nuclide: FissioningNuclide, /) -> _System:
    match nuclide:
        case "U235":
            archive = _neutron(
                "n_9228_92-U-235.zip",
                "6b06f87d01736faa31ee582504c8615cb904745c09b46adceae9168285e8ef28",
                13590400,
            )
            return _System(nuclide, 92, 235, 9228, True, archive)
        case "U238":
            archive = _neutron(
                "n_9237_92-U-238.zip",
                "5bbb4b6c1a46c9634c452eed028d70cf3226b2d5d5e893d34bae163b4d66afef",
                4576541,
            )
            return _System(nuclide, 92, 238, 9237, False, archive)
        case "Pu239":
            archive = _neutron(
                "n_9437_94-Pu-239.zip",
                "00dc580cae355607d16b275f2ab17233772054ef9a7c4abc23386caa6087d011",
                5391913,
            )
            return _System(nuclide, 94, 239, 9437, True, archive)
        case "Pu240":
            archive = _neutron(
                "n_9440_94-Pu-240.zip",
                "f85604f48a6f581671ee1d9660f87f4242b8e8bad61a322115b1ead2da97e07c",
                13099106,
            )
            return _System(nuclide, 94, 240, 9440, False, archive)
        case "Pu241":
            archive = _neutron(
                "n_9443_94-Pu-241.zip",
                "27387f9f8bde5a551cd1eb253c8df31107521b6cbdaa1ea0c00c4a1316e0e5ff",
                81869,
            )
            return _System(nuclide, 94, 241, 9443, True, archive)
        case "Pu242":
            archive = _neutron(
                "n_9446_94-Pu-242.zip",
                "a7ab770daf48b8c979675d27f8bf959cd20f712a55ab7a9104f63b7f235d9943",
                685196,
            )
            return _System(nuclide, 94, 242, 9446, False, archive)
        case _:
            assert_never(nuclide)


def _label(proton_number: int, mass_number: int, /) -> str:
    return f"{_ELEMENT_SYMBOLS[proton_number]}-{mass_number}"


# --------------------------------------------------------------------------- fetch


def _fetch(source: _Source, cache: Path, /) -> bytes:
    """Return verified source bytes, downloading into ``cache`` when absent."""

    path = cache / source.filename
    if not path.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(source.url, timeout=600) as response:
            payload = response.read()
        path.write_bytes(payload)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != source.size_bytes or digest != source.sha256:
        raise ValueError(
            f"{source.url} does not match its pinned identity "
            f"(sha256 {digest}, {len(data)} bytes); refusing to extract."
        )
    return data


def _member(archive: bytes, name: str, /) -> bytes:
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        return handle.read(name)


# ------------------------------------------------------------------ ENDF-6 records

_ENDF_FLOAT = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))([eE]?)([+-]\d+)?$")


def _endf_float(field: str, /) -> float:
    """Parse one 11-column ENDF-6 real (``1.234567+8`` or Fortran E format)."""

    text = field.strip()
    if not text:
        return 0.0
    match = _ENDF_FLOAT.match(text)
    if match is None:
        raise ValueError(f"Malformed ENDF-6 real {field!r}.")
    mantissa, _, exponent = match.groups()
    return float(mantissa if exponent is None else f"{mantissa}e{exponent}")


def _endf_integer(field: str, /) -> int:
    text = field.strip()
    return 0 if not text else int(text)


def _section(text: str, mat: int, mf: int, mt: int, /) -> list[str]:
    lines = [
        line[:66]
        for line in text.splitlines()
        if len(line) >= 75
        and line[66:70].strip() == str(mat)
        and line[70:72].strip() == str(mf)
        and line[72:75].strip() == str(mt)
    ]
    if not lines:
        raise ValueError(f"MAT={mat} MF={mf} MT={mt} is absent.")
    return lines


class _Reader:
    """Sequential CONT/LIST/TAB1 reader over one ENDF-6 section."""

    def __init__(self, lines: Sequence[str], /) -> None:
        self._lines = lines
        self._position = 0

    def cont(self) -> tuple[float, float, int, int, int, int]:
        line = self._lines[self._position]
        self._position += 1
        return (
            _endf_float(line[0:11]),
            _endf_float(line[11:22]),
            _endf_integer(line[22:33]),
            _endf_integer(line[33:44]),
            _endf_integer(line[44:55]),
            _endf_integer(line[55:66]),
        )

    def reals(self, count: int, /) -> list[float]:
        values: list[float] = []
        while len(values) < count:
            line = self._lines[self._position]
            self._position += 1
            values.extend(_endf_float(line[11 * k : 11 * (k + 1)]) for k in range(6))
        return values[:count]

    def integers(self, count: int, /) -> list[int]:
        values: list[int] = []
        while len(values) < count:
            line = self._lines[self._position]
            self._position += 1
            values.extend(_endf_integer(line[11 * k : 11 * (k + 1)]) for k in range(6))
        return values[:count]


def _evaluation_identity(text: str, mat: int, sublibrary: str, /) -> str:
    """Verify the MT451 library banner and return the evaluation line."""

    lines = _section(text, mat, 1, 451)
    banner = lines[6] if len(lines) > 6 else ""
    kind = lines[7] if len(lines) > 7 else ""
    distribution = lines[5] if len(lines) > 5 else ""
    if (
        not banner.startswith("----ENDF/B-VIII.0")
        or sublibrary not in kind
        or "DIST-FEB18" not in distribution
    ):
        raise ValueError(f"MAT={mat} is not an ENDF/B-VIII.0 {sublibrary} evaluation.")
    return " ".join(lines[4].split())


type _YieldTable = dict[tuple[int, int], tuple[float, float]]


def _fission_yields(text: str, system: _System, mt: int, /) -> dict[float, _YieldTable]:
    """MF8 yields ``{E: {(ZA, FPS): (Y, DY)}}`` at every tabulated incident energy."""

    reader = _Reader(_section(text, system.mat, 8, mt))
    za, _, energies, _, _, _ = reader.cont()
    if round(za) != 1000 * system.proton_number + system.mass_number:
        raise ValueError(f"MF8/MT{mt} target does not match {system.label}.")
    tables: dict[float, _YieldTable] = {}
    for _ in range(energies):
        energy, _, _, _, count, products = reader.cont()
        values = reader.reals(count)
        if count != 4 * products or energy in tables:
            raise ValueError(f"MF8/MT{mt} of {system.label} is malformed.")
        tables[energy] = {
            (round(values[4 * index]), round(values[4 * index + 1])): (
                values[4 * index + 2],
                values[4 * index + 3],
            )
            for index in range(products)
        }
    return tables


def _incident_energy(system: _System, tabulated: Sequence[float], /) -> float:
    """Thermal point for fissile targets; lowest tabulated energy otherwise."""

    if not tabulated:
        raise ValueError(f"{system.label} has no tabulated fission yields.")
    if not system.thermal:
        return min(tabulated)
    if _THERMAL_ENERGY_EV not in tabulated:
        raise ValueError(f"{system.label} has no thermal (0.0253 eV) fission yields.")
    return _THERMAL_ENERGY_EV


def _energy_release(text: str, system: _System, /) -> dict[str, list[float]]:
    """MF1/MT458 thermal (fissile) or threshold (fertile) point components [eV].

    ENDF-102 section 1.5.1 defines the NPLY=0 LIST record as the thermal-point
    evaluation for fissile nuclides and the near-threshold value for fertile
    nuclides; section 1.5.2 recommends ER at that point as the energy-independent
    fission Q-value.
    """

    reader = _Reader(_section(text, system.mat, 1, 458))
    reader.cont()
    _, _, _, order, count, pairs = reader.cont()
    if order != 0 or count != 18 or pairs != 9:
        raise ValueError(f"MF1/MT458 of MAT={system.mat} is not an NPLY=0 point record.")
    values = reader.reals(18)
    components = {
        name: [values[2 * index], values[2 * index + 1]]
        for index, name in enumerate(_ENERGY_COMPONENTS)
    }
    partial = sum(components[name][0] for name in _ENERGY_COMPONENTS[:7])
    total = components["ET"][0]
    recoverable = components["ER"][0]
    # ER = ET - ENU must hold to the seven significant digits of ENDF-6 reals.
    # The partial sum is a parse-integrity check only: Pu-240 (MAT 9440) lists
    # an ET that omits its 4.4 keV END component (relative 2e-5).
    if not math.isclose(
        total - components["ENU"][0], recoverable, rel_tol=1.0e-6
    ) or not math.isclose(partial, total, rel_tol=1.0e-4):
        raise ValueError(f"MF1/MT458 of MAT={system.mat} is internally inconsistent.")
    return components


# ------------------------------------------------------------------- half-lives

_NUBASE_UNITS_S: Mapping[str, float] = {
    "ys": 1.0e-24,
    "zs": 1.0e-21,
    "as": 1.0e-18,
    "fs": 1.0e-15,
    "ps": 1.0e-12,
    "ns": 1.0e-9,
    "us": 1.0e-6,
    "ms": 1.0e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "y": _SECONDS_PER_YEAR,
    "ky": 1.0e3 * _SECONDS_PER_YEAR,
    "My": 1.0e6 * _SECONDS_PER_YEAR,
    "Gy": 1.0e9 * _SECONDS_PER_YEAR,
    "Ty": 1.0e12 * _SECONDS_PER_YEAR,
    "Py": 1.0e15 * _SECONDS_PER_YEAR,
    "Ey": 1.0e18 * _SECONDS_PER_YEAR,
    "Zy": 1.0e21 * _SECONDS_PER_YEAR,
    "Yy": 1.0e24 * _SECONDS_PER_YEAR,
}

type _HalfLives = dict[tuple[int, int, int], float]


def _half_lives(text: str, /) -> _HalfLives:
    """NUBASE2020 half-lives in years keyed by ``(Z, A, state)``; stable is ``inf``.

    States without a half-life entry (levels, particle-unbound states) are
    recorded as zero.
    """

    result: _HalfLives = {}
    for raw in text.splitlines():
        if raw.startswith("#"):
            continue
        line = raw.ljust(80)
        mass = int(line[0:3])
        proton = int(line[4:7])
        state_code = line[7]
        if not state_code.isdigit():
            continue
        value = line[69:78].strip().rstrip("#").lstrip("<>~")
        unit = line[78:80].strip()
        if value == "stbl":
            years = math.inf
        elif value in ("", "p-unst"):
            years = 0.0
        else:
            years = float(value) * _NUBASE_UNITS_S[unit] / _SECONDS_PER_YEAR
        result[(proton, mass, int(state_code))] = years
    return result


def _is_long_lived(half_lives: _HalfLives, proton: int, mass: int, state: int, /) -> bool:
    # NUBASE2020 lists every observed nuclide; an absent one lies beyond the
    # known neutron-rich edge of the chart and is not long-lived.
    return half_lives.get((proton, mass, state), 0.0) > _HALF_LIFE_THRESHOLD_YEARS


# -------------------------------------------------------------------- selection


def _verify_selection(
    system: _System,
    cumulative: _YieldTable,
    independent: _YieldTable,
    half_lives: _HalfLives,
    /,
) -> None:
    """Refuse unless the declared gas set is exactly the in-reactor chain ends."""

    for proton, masses in _GAS_MASS_NUMBERS.items():
        present = {za % 1000 for za, state in cumulative if za // 1000 == proton}
        for mass, state in (
            (za % 1000, state) for za, state in cumulative if za // 1000 == proton
        ):
            if state > 0 and any(
                _is_long_lived(half_lives, proton, mass, level)
                for level in range(1, 7)
                if (proton, mass, level) in half_lives
            ):
                raise ValueError(
                    f"Long-lived isomer of {_label(proton, mass)} is unhandled."
                )
        for mass in sorted(present):
            long_lived = _is_long_lived(half_lives, proton, mass, 0)
            declared = mass in masses
            excluded = (proton, mass) in _EXCLUDED_BY_PRECURSOR
            if declared and not long_lived:
                raise ValueError(f"{_label(proton, mass)} is not stable or long-lived.")
            if long_lived and not (declared or excluded):
                raise ValueError(
                    f"{_label(proton, mass)} is long-lived but not declared."
                )
        for mass in masses:
            if (1000 * proton + mass, 0) not in cumulative:
                raise ValueError(
                    f"{_label(proton, mass)} is absent from the "
                    f"{system.fissioning_nuclide} cumulative yields."
                )
            _verify_precursors(system, cumulative, half_lives, proton, mass)
    for (proton, mass), _ in _EXCLUDED_BY_PRECURSOR.items():
        direct = sum(
            value
            for (za, _state), (value, _) in independent.items()
            if za == 1000 * proton + mass
        )
        if direct != 0.0:
            raise ValueError(
                f"{_label(proton, mass)} has a non-zero independent yield for "
                f"{system.fissioning_nuclide}; the exclusion would drop production."
            )


def _verify_precursors(
    system: _System,
    cumulative: _YieldTable,
    half_lives: _HalfLives,
    proton: int,
    mass: int,
    /,
) -> None:
    """Refuse if a long-lived beta-minus precursor is carried into the yield.

    The isobaric chain is walked toward lower Z until a stable nuclide or the end
    of NUBASE2020. A radioactive precursor with T1/2 > 1 y is admissible only when
    the evaluation treats it as a chain end, which appears as a precursor
    cumulative yield exceeding the gas nuclide's cumulative yield.
    """

    gas = cumulative[(1000 * proton + mass, 0)][0]
    for precursor in range(proton - 1, 0, -1):
        key = (precursor, mass, 0)
        if key not in half_lives or math.isinf(half_lives[key]):
            return
        if half_lives[key] > _HALF_LIFE_THRESHOLD_YEARS:
            carried = cumulative.get((1000 * precursor + mass, 0), (0.0, 0.0))[0]
            if not carried > gas:
                raise ValueError(
                    f"{_label(proton, mass)} cumulative yield may include the "
                    f"long-lived precursor Z={precursor} for {system.fissioning_nuclide}."
                )


# ---------------------------------------------------------------------- payload


def _gas_records(proton: int, cumulative: _YieldTable, /) -> list[list[float | int]]:
    return [
        [mass, *cumulative[(1000 * proton + mass, 0)]]
        for mass in _GAS_MASS_NUMBERS[proton]
    ]


def _system_record(
    system: _System,
    nfy: bytes,
    cache: Path,
    half_lives: _HalfLives,
    /,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    yield_bytes = _member(nfy, system.yield_member)
    yield_text = yield_bytes.decode("ascii")
    archive = _fetch(system.neutron, cache)
    neutron_bytes = _member(archive, system.neutron_member)
    neutron_text = neutron_bytes.decode("ascii")
    cumulative_tables = _fission_yields(yield_text, system, 459)
    independent_tables = _fission_yields(yield_text, system, 454)
    energy = _incident_energy(system, sorted(cumulative_tables))
    cumulative = cumulative_tables[energy]
    independent = independent_tables[energy]
    _verify_selection(system, cumulative, independent, half_lives)
    record: dict[str, object] = {
        "fissioning_nuclide": system.fissioning_nuclide,
        "target": system.label,
        "mat": system.mat,
        "incident_energy_ev": energy,
        "incident_energy_selection": (
            "thermal point (0.0253 eV)"
            if system.thermal
            else "lowest incident energy tabulated in the NFY sublibrary"
        ),
        "tabulated_incident_energies_ev": sorted(cumulative_tables),
        "yield_evaluation": _evaluation_identity(
            yield_text, system.mat, "NEUTRON-INDUCED FISSION PRODUCT YIELDS"
        ),
        "energy_release_evaluation": _evaluation_identity(
            neutron_text, system.mat, "INCIDENT NEUTRON DATA"
        ),
        "krypton": _gas_records(_KRYPTON, cumulative),
        "xenon": _gas_records(_XENON, cumulative),
        "energy_release_ev": _energy_release(neutron_text, system),
    }
    sources: list[dict[str, object]] = [
        {
            "role": _NFY.role,
            "url": _NFY.url,
            "sha256": _NFY.sha256,
            "size_bytes": _NFY.size_bytes,
            "member": system.yield_member,
            "member_sha256": hashlib.sha256(yield_bytes).hexdigest(),
        },
        {
            "role": system.neutron.role,
            "url": system.neutron.url,
            "sha256": system.neutron.sha256,
            "size_bytes": system.neutron.size_bytes,
            "member": system.neutron_member,
            "member_sha256": hashlib.sha256(neutron_bytes).hexdigest(),
        },
    ]
    return record, sources


def _extraction_record() -> dict[str, object]:
    return {
        "cumulative_yield_section": "MF8/MT459 ground states (FPS=0)",
        "independent_yield_section": "MF8/MT454 (exclusion verification only)",
        "energy_release_section": (
            "MF1/MT458 NPLY=0 LIST: thermal point (fissile) or threshold (fertile)"
        ),
        "recoverable_energy_component": "ER = ET - ENU (includes END, EGD, EB)",
        "half_life_threshold_years": _HALF_LIFE_THRESHOLD_YEARS,
        "gas_nuclides": [
            _label(proton, mass)
            for proton, masses in _GAS_MASS_NUMBERS.items()
            for mass in masses
        ],
        "excluded_nuclides": [
            {
                "nuclide": _label(*gas),
                "long_lived_precursor": _label(*precursor),
                "in_reactor_yield": "independent yield (verified zero)",
            }
            for gas, precursor in _EXCLUDED_BY_PRECURSOR.items()
        ],
    }


def extract(cache: Path, /) -> bytes:
    """Return the canonical packaged JSON bytes."""

    nfy = _fetch(_NFY, cache)
    half_lives = _half_lives(_fetch(_NUBASE, cache).decode("ascii"))
    systems: list[dict[str, object]] = []
    sources: list[dict[str, object]] = [
        {
            "role": _NUBASE.role,
            "url": _NUBASE.url,
            "sha256": _NUBASE.sha256,
            "size_bytes": _NUBASE.size_bytes,
            "member": None,
            "member_sha256": None,
        }
    ]
    for nuclide in FISSIONING_NUCLIDES:
        record, used = _system_record(_system(nuclide), nfy, cache, half_lives)
        systems.append(record)
        sources.extend(value for value in used if value not in sources)
    payload = {
        "library": "ENDF/B-VIII.0",
        "release": "DIST-FEB18",
        "extraction": _extraction_record(),
        "extractor": _TOOL,
        "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": sources,
        "fission_systems": systems,
    }
    return (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("ascii")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the packaged file equals a fresh extraction",
    )
    arguments = parser.parse_args(argv)
    payload = extract(arguments.cache_dir)
    digest = hashlib.sha256(payload).hexdigest()
    if arguments.check:
        current = arguments.output.read_bytes()
        if current != payload:
            print(f"{arguments.output} differs from a fresh extraction.", file=sys.stderr)
            return 1
        print(f"{arguments.output} matches (sha256 {digest}, {len(payload)} bytes).")
        return 0
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(payload)
    print(f"wrote {arguments.output}: sha256 {digest}, {len(payload)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
