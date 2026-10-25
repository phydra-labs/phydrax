#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Authenticated DFT-D3 reference tables, damping/coordination policies and signed C6 factors.

Scientific sources (external provenance, never schema generations):

* Grimme, Antony, Ehrlich, Krieg, J. Chem. Phys. 132, 154104 (2010): reference
  ``C6(Z_a, CN^p; Z_b, CN^q)``, covalent radii (Pyykko-Atsumi, metals -10 %,
  folded with ``k2 = 4/3``), the ``r4r2`` factors and the Gaussian coordination
  interpolation ``L = exp(-k3 (CN - CN^ref)^2)`` with ``k3 = 4``.
* Grimme, Ehrlich, Goerigk, J. Comput. Chem. 32, 1456 (2011): Becke-Johnson
  damping ``R0 = a1 sqrt(3 r4r2_a r4r2_b) + a2``.
* NVIDIA nvalchemi-toolkit-ops ``10e12340c39fd12a2375d6301768e72d2cd12868``
  (Apache-2.0) ``interactions/dispersion``: the inspected conventional-CN
  interpolation threshold, the FourierD3 modified counting function and the
  signed low-rank reference decomposition.
* Valeeva et al., arXiv:2607.15103 (FourierD3).

No table is bundled and nothing is downloaded. The reference data in Grimme's
``dftd3.tgz`` carry the GPL-1.0-or-later notice of ``dftd3.f``; a caller who holds
the files and the right to use them builds :class:`D3ReferenceTables` through
:meth:`D3ReferenceTables.from_grimme_fortran` (digest-verified host parsing) or
from explicit arrays, always with an explicit :class:`D3TableProvenance`.
Synthetic fixtures are admitted as such and never qualify element fidelity.

Units are explicit: covalent radii and ``r4r2`` are lengths in ``length_unit``;
``C6`` is ``energy_unit * length_unit**6``. ``k1``, ``k3`` and the coordination
numbers are dimensionless. The FourierD3 counting regularizer ``epsilon`` is an
area (``epsilon_length_unit**2``) and is converted, never silently reused in
another unit.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import parse
from ..units import BOHR, conversion_factor, ENERGY, LENGTH, UnitDefinition


D3TableKind: TypeAlias = Literal[
    "grimme-dftd3-reference", "caller-derived", "synthetic-fixture"
]
D3InterpolationRule: TypeAlias = Literal[
    "normalized-gaussian", "nvalchemi-relative-threshold"
]
C6DecompositionCriterion: TypeAlias = Literal["maximum-relative-entry-error"]

#: Grimme's conventional counting steepness ``k1`` (dimensionless).
D3_COUNTING_STEEPNESS = 16.0
#: Grimme's Gaussian interpolation width ``k3`` with ``L = exp(-k3 dCN^2)``.
D3_INTERPOLATION_WIDTH = 4.0
#: Relative log-weight below which the inspected NVIDIA conventional kernel
#: skips a reference pair (``exp_arg < -12`` after subtracting the maximum).
NVALCHEMI_INTERPOLATION_THRESHOLD = 12.0
#: FourierD3 counting regularizer of the inspected NVIDIA source, an area in the
#: source length unit (the source is unit-agnostic; its D3 tables and documented
#: cutoffs are atomic units, so the declared unit is bohr squared).
NVALCHEMI_FOURIER_EPSILON = 1.0e-6

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ATOMIC_NUMBER = 118


def _finite(value: float, name: str, /) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive(value: float, name: str, /) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _text(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty stripped string.")
    return value


def _species(values: Sequence[int], name: str, /) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of atomic numbers.")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise TypeError(f"{name} must contain integers.")
    species = tuple(int(value) for value in values)
    if not species or any(not 0 < value <= _MAX_ATOMIC_NUMBER for value in species):
        raise ValueError(f"{name} must name atomic numbers 1..{_MAX_ATOMIC_NUMBER}.")
    if tuple(sorted(set(species))) != species:
        raise ValueError(f"{name} must be strictly increasing.")
    return species


def _unit(value: UnitDefinition, dimension: object, name: str, /) -> UnitDefinition:
    if not isinstance(value, UnitDefinition):
        raise TypeError(f"{name} must be a UnitDefinition.")
    if value.dimension != dimension:
        raise ValueError(f"{name} has the wrong physical dimension.")
    return value


def length_factor(source: UnitDefinition, target: UnitDefinition, /) -> float:
    """Exact host factor converting a length from ``source`` to ``target``."""
    _unit(source, LENGTH, "source length unit")
    _unit(target, LENGTH, "target length unit")
    return float(conversion_factor(source, target))


def energy_factor(source: UnitDefinition, target: UnitDefinition, /) -> float:
    """Exact host factor converting an energy from ``source`` to ``target``."""
    _unit(source, ENERGY, "source energy unit")
    _unit(target, ENERGY, "target energy unit")
    return float(conversion_factor(source, target))


@final
class D3TableProvenance(StrictModule, NonTrainableState):
    """Source, content digests and rights of one reference-table admission.

    ``digests`` pairs each source artifact name with its SHA-256 hex digest.
    ``use_admitted`` records that the caller holds the right to use the data
    numerically; preparation refuses tables without it. Redistribution is a
    separate right and is never implied by a digest or by use admission.
    ``synthetic-fixture`` tables need no source digest and qualify nothing
    beyond their own invented species.

    PyTree layout: the declared source/kind/license/admission labels, rights
    flags and artifact names are static; the content-derived artifact digests
    (``digest_identities``, one row per artifact) and ``provenance_identity``
    are dynamic :class:`IdentityWords` holding the exact former digests, which
    the host properties ``digests``/``digest``/``provenance_id`` decode.
    """

    source: str = eqx.field(static=True)
    kind: D3TableKind = eqx.field(static=True)
    digest_artifacts: tuple[str, ...] = eqx.field(static=True)
    digest_identities: IdentityWords
    license_id: str = eqx.field(static=True)
    use_admitted: bool = eqx.field(static=True)
    redistribution_admitted: bool = eqx.field(static=True)
    admission: str = eqx.field(static=True)
    provenance_identity: IdentityWords

    def __init__(
        self,
        source: str,
        /,
        *,
        kind: D3TableKind,
        license_id: str,
        use_admitted: bool,
        redistribution_admitted: bool,
        admission: str,
        digests: Sequence[tuple[str, str]] = (),
    ) -> None:
        name = _text(source, "source")
        table_kind = parse(kind, D3TableKind, "kind")
        records: list[tuple[str, str]] = []
        for item in digests:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("digests must contain (artifact, sha256) pairs.")
            artifact, digest = _text(item[0], "artifact"), str(item[1]).lower()
            if not _SHA256.match(digest):
                raise ValueError(f"Digest of {artifact!r} is not a SHA-256 hex string.")
            records.append((artifact, digest))
        if len({artifact for artifact, _ in records}) != len(records):
            raise ValueError("Each source artifact has one digest.")
        if table_kind != "synthetic-fixture" and not records:
            raise ValueError("Non-synthetic D3 tables require source digests.")
        for flag, label in (
            (use_admitted, "use_admitted"),
            (redistribution_admitted, "redistribution_admitted"),
        ):
            if not isinstance(flag, bool):
                raise TypeError(f"{label} must be a boolean.")
        ordered = tuple(sorted(records))
        license_ = _text(license_id, "license_id")
        admission_ = _text(admission, "admission")
        words = np.zeros((len(ordered), 8), dtype=np.uint32)
        for row, (_, digest) in enumerate(ordered):
            words[row] = np.asarray(IdentityWords.from_fingerprint(digest).words)
        self.source = name
        self.kind = table_kind
        self.digest_artifacts = tuple(artifact for artifact, _ in ordered)
        self.digest_identities = IdentityWords(jnp.asarray(words))
        self.license_id = license_
        self.use_admitted = use_admitted
        self.redistribution_admitted = redistribution_admitted
        self.admission = admission_
        self.provenance_identity = IdentityWords.of(
            {
                "kind": "d3-table-provenance",
                "source": name,
                "table_kind": table_kind,
                "digests": [list(item) for item in ordered],
                "license_id": license_,
                "use_admitted": use_admitted,
                "redistribution_admitted": redistribution_admitted,
                "admission": admission_,
            }
        )

    @property
    def provenance_id(self) -> str:
        """Host decoding of the canonical provenance digest."""
        return self.provenance_identity.fingerprint()

    @property
    def digests(self) -> tuple[tuple[str, str], ...]:
        """Host decoding of the sorted ``(artifact, sha256)`` pairs."""
        return tuple(
            (artifact, IdentityWords(self.digest_identities.words[row]).fingerprint())
            for row, artifact in enumerate(self.digest_artifacts)
        )

    def require_use(self) -> None:
        """Refuse numerical use of tables whose use right was not admitted."""
        if not self.use_admitted:
            raise ValueError(
                f"D3 reference tables from {self.source!r} ({self.license_id}) have no "
                "admitted use right; refusing to prepare them."
            )

    def digest(self, artifact: str, /) -> str:
        for name, value in self.digests:
            if name == artifact:
                return value
        raise ValueError(f"No declared digest for source artifact {artifact!r}.")


def _table_identity(
    species: tuple[int, ...],
    arrays: dict[str, np.ndarray],
    length_unit: UnitDefinition,
    energy_unit: UnitDefinition,
    covalent_radius_scale: float,
    provenance: D3TableProvenance,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "d3-reference-tables",
            "species": list(species),
            "arrays": array_tree_fingerprint(arrays),
            "length_unit": length_unit.unit_id,
            "energy_unit": energy_unit.unit_id,
            "covalent_radius_scale": float(covalent_radius_scale).hex(),
            "provenance": provenance.provenance_id,
        }
    )


def _species_lookup(species: tuple[int, ...], /) -> np.ndarray:
    """Fixed-length ``int32[119]`` row of every atomic number, ``-1`` if uncovered."""
    lookup = np.full((_MAX_ATOMIC_NUMBER + 1,), -1, dtype=np.int32)
    lookup[np.asarray(species)] = np.arange(len(species), dtype=np.int32)
    return lookup


@final
class D3ReferenceTables(StrictModule, NonTrainableState):
    """Species-compact, unit-explicit and authenticated D3 reference tables.

    Rows follow ``species`` (strictly increasing atomic numbers). For ``S``
    species and ``R`` reference slots:

    * ``covalent_radii[S]`` and ``r4r2[S]`` are positive lengths;
      ``covalent_radius_scale`` is the factor already folded into the radii
      (Grimme's tables fold ``k2 = 4/3``).
    * ``reference_valid[S, R]`` marks populated slots and
      ``reference_coordination[S, R]`` their finite non-negative coordination
      numbers (zero on unpopulated slots, never a sentinel value).
    * ``c6_reference[S, S, R, R]`` is exactly symmetric under
      ``(a, p) <-> (b, q)``, positive on every populated slot pair and zero
      elsewhere: a reference block is complete or the tables are refused.

    ``species_lookup[Z]`` (``Z = 0 .. 118``) is the row of atomic number ``Z``
    or ``-1``; an uncovered element is a refusal, never padding. All arrays
    are dynamic leaves (coefficient derivatives are available) but fixed by
    default.

    PyTree layout: ``S`` and ``R`` (the table shapes) and the declared units
    are the layout. The row labels ``atomic_numbers`` (``int32[S]``, decoded
    by the host property ``species``), the fixed-length ``species_lookup``,
    ``covalent_radius_scale`` (float64 scalar) and ``tables_identity`` (exact
    former ``tables_id`` digest, decoded by the host property) are dynamic, so
    tables of equal shape with other elements or values share one executable.
    """

    covalent_radii: Array
    r4r2: Array
    c6_reference: Array
    reference_coordination: Array
    reference_valid: Array
    atomic_numbers: Array
    species_lookup: Array
    covalent_radius_scale: Array
    length_unit: UnitDefinition
    energy_unit: UnitDefinition
    provenance: D3TableProvenance
    tables_identity: IdentityWords

    def __init__(
        self,
        species: Sequence[int],
        /,
        *,
        covalent_radii: ArrayLike,
        r4r2: ArrayLike,
        c6_reference: ArrayLike,
        reference_coordination: ArrayLike,
        reference_valid: ArrayLike,
        length_unit: UnitDefinition,
        energy_unit: UnitDefinition,
        covalent_radius_scale: float,
        provenance: D3TableProvenance,
    ) -> None:
        covered = _species(species, "species")
        count = len(covered)
        length = _unit(length_unit, LENGTH, "length_unit")
        energy = _unit(energy_unit, ENERGY, "energy_unit")
        scale = _positive(covalent_radius_scale, "covalent_radius_scale")
        if not isinstance(provenance, D3TableProvenance):
            raise TypeError("provenance must be a D3TableProvenance.")
        arrays = _validated_tables(
            count,
            np.asarray(covalent_radii),
            np.asarray(r4r2),
            np.asarray(c6_reference),
            np.asarray(reference_coordination),
            np.asarray(reference_valid),
        )
        self.covalent_radii = jnp.asarray(arrays["covalent_radii"])
        self.r4r2 = jnp.asarray(arrays["r4r2"])
        self.c6_reference = jnp.asarray(arrays["c6_reference"])
        self.reference_coordination = jnp.asarray(arrays["reference_coordination"])
        self.reference_valid = jnp.asarray(arrays["reference_valid"])
        self.atomic_numbers = jnp.asarray(np.asarray(covered, dtype=np.int32))
        self.species_lookup = jnp.asarray(_species_lookup(covered))
        self.covalent_radius_scale = jnp.asarray(scale, dtype=jnp.float64)
        self.length_unit = length
        self.energy_unit = energy
        self.provenance = provenance
        self.tables_identity = IdentityWords.from_fingerprint(
            _table_identity(covered, arrays, length, energy, scale, provenance)
        )

    @property
    def species(self) -> tuple[int, ...]:
        """Host decoding of the covered atomic numbers (row order)."""
        return tuple(int(value) for value in np.asarray(self.atomic_numbers))

    @property
    def tables_id(self) -> str:
        """Host decoding of the canonical reference-table digest."""
        return self.tables_identity.fingerprint()

    @property
    def species_count(self) -> int:
        return int(self.atomic_numbers.shape[0])

    @property
    def reference_slots(self) -> int:
        return int(self.reference_valid.shape[1])

    def uncovered(self, atomic_numbers: ArrayLike, /) -> tuple[int, ...]:
        """Distinct atomic numbers of ``atomic_numbers`` without a table row (host)."""
        numbers = np.unique(np.asarray(atomic_numbers, dtype=np.int64))
        covered = set(self.species)
        return tuple(int(value) for value in numbers if int(value) not in covered)

    def rows(self, atomic_numbers: ArrayLike, /) -> np.ndarray:
        """Host table rows of ``atomic_numbers``; refuses any uncovered element."""
        numbers = np.asarray(atomic_numbers, dtype=np.int64)
        missing = self.uncovered(numbers)
        if missing:
            raise ValueError(
                f"D3 reference tables do not cover atomic numbers {list(missing)}."
            )
        lookup = {value: index for index, value in enumerate(self.species)}
        return np.asarray(
            [lookup[int(value)] for value in numbers.reshape(-1)], dtype=np.int32
        ).reshape(numbers.shape)

    def require_use(self) -> None:
        self.provenance.require_use()

    def validate_restored(self) -> None:
        """Re-validate content and identity of a restored (constructor-bypassed) table."""
        species = _species(self.species, "species")
        arrays = _validated_tables(
            len(species),
            np.asarray(self.covalent_radii),
            np.asarray(self.r4r2),
            np.asarray(self.c6_reference),
            np.asarray(self.reference_coordination),
            np.asarray(self.reference_valid),
        )
        if not np.array_equal(np.asarray(self.species_lookup), _species_lookup(species)):
            raise ValueError("Restored D3 species lookup does not match its species.")
        identity = _table_identity(
            species,
            arrays,
            self.length_unit,
            self.energy_unit,
            _positive(float(self.covalent_radius_scale), "covalent_radius_scale"),
            self.provenance,
        )
        if identity != self.tables_id:
            raise ValueError("Restored D3 reference tables do not match their identity.")

    @classmethod
    def from_grimme_fortran(
        cls,
        dftd3_source: str | Path,
        pars_source: str | Path,
        /,
        *,
        species: Sequence[int],
        length_unit: UnitDefinition,
        energy_unit: UnitDefinition,
        provenance: D3TableProvenance,
    ) -> "D3ReferenceTables":
        """Parse caller-held Grimme ``dftd3.f``/``pars.f`` into compact tables (host).

        The SHA-256 of each file must equal the digest declared in
        ``provenance`` under the artifact names ``"dftd3.f"`` and ``"pars.f"``.
        ``rcov`` (already folded with ``k2 = 4/3``) and ``r2r4`` are read from
        the active ``data`` statements of ``dftd3.f``; the ``pars`` records
        ``(C6, Z_i + 100 (p-1), Z_j + 100 (q-1), CN_i^p, CN_j^q)`` from
        ``pars.f``. A reference block that is incomplete, asymmetric or whose
        reference coordination differs between partners is refused. The units
        of the archive (bohr, hartree bohr^6) must be declared by the caller.
        Nothing is downloaded.
        """
        if provenance.kind != "grimme-dftd3-reference":
            raise ValueError(
                "Grimme Fortran sources require a grimme-dftd3-reference provenance."
            )
        texts = {}
        for artifact, location in (("dftd3.f", dftd3_source), ("pars.f", pars_source)):
            content = Path(location).read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != provenance.digest(artifact):
                raise ValueError(
                    f"{artifact} SHA-256 {digest} does not match the declared digest."
                )
            texts[artifact] = content.decode("ascii")
        covered = _species(species, "species")
        rcov = _fortran_data(texts["dftd3.f"], "rcov")
        r2r4 = _fortran_data(texts["dftd3.f"], "r2r4")
        if rcov.size != 94 or r2r4.size != 94:
            raise ValueError("dftd3.f must define 94 rcov and r2r4 values.")
        if covered[-1] > 94:
            raise ValueError("Grimme D3 reference data cover atomic numbers 1..94.")
        records = _pars_records(texts["pars.f"])
        tables = _grimme_blocks(records, covered)
        index = np.asarray(covered) - 1
        return cls(
            covered,
            covalent_radii=rcov[index],
            r4r2=r2r4[index],
            c6_reference=tables[0],
            reference_coordination=tables[1],
            reference_valid=tables[2],
            length_unit=length_unit,
            energy_unit=energy_unit,
            covalent_radius_scale=4.0 / 3.0,
            provenance=provenance,
        )


def _validated_tables(
    count: int,
    covalent_radii: np.ndarray,
    r4r2: np.ndarray,
    c6_reference: np.ndarray,
    reference_coordination: np.ndarray,
    reference_valid: np.ndarray,
    /,
) -> dict[str, np.ndarray]:
    for name, value in (
        ("covalent_radii", covalent_radii),
        ("r4r2", r4r2),
        ("c6_reference", c6_reference),
        ("reference_coordination", reference_coordination),
    ):
        if value.dtype.kind != "f":
            raise TypeError(f"{name} must be floating point.")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite.")
    if reference_valid.dtype != np.bool_:
        raise TypeError("reference_valid must be boolean.")
    if covalent_radii.shape != (count,) or r4r2.shape != (count,):
        raise ValueError("covalent_radii and r4r2 need one value per species.")
    if np.any(covalent_radii <= 0.0) or np.any(r4r2 <= 0.0):
        raise ValueError("Covered species need positive covalent radii and r4r2.")
    if reference_valid.ndim != 2 or reference_valid.shape[0] != count:
        raise ValueError("reference_valid must have shape (S, R).")
    slots = reference_valid.shape[1]
    if slots < 1:
        raise ValueError("At least one reference slot is required.")
    if reference_coordination.shape != (count, slots):
        raise ValueError("reference_coordination must have shape (S, R).")
    if c6_reference.shape != (count, count, slots, slots):
        raise ValueError("c6_reference must have shape (S, S, R, R).")
    if not np.all(np.any(reference_valid, axis=1)):
        raise ValueError("Every covered species needs at least one reference slot.")
    if np.any(reference_coordination[reference_valid] < 0.0):
        raise ValueError("Reference coordination numbers must be non-negative.")
    if np.any(reference_coordination[~reference_valid] != 0.0):
        raise ValueError("Unpopulated reference slots must carry zero coordination.")
    if not np.array_equal(c6_reference, np.transpose(c6_reference, (1, 0, 3, 2))):
        raise ValueError("c6_reference must be exactly symmetric under (a,p)<->(b,q).")
    populated = reference_valid[:, None, :, None] & reference_valid[None, :, None, :]
    if np.any(c6_reference[populated] <= 0.0):
        raise ValueError("Every populated reference slot pair needs a positive C6.")
    if np.any(c6_reference[~populated] != 0.0):
        raise ValueError("Unpopulated reference slot pairs must carry zero C6.")
    return {
        "covalent_radii": covalent_radii.astype(np.float64),
        "r4r2": r4r2.astype(np.float64),
        "c6_reference": c6_reference.astype(np.float64),
        "reference_coordination": reference_coordination.astype(np.float64),
        "reference_valid": reference_valid.copy(),
    }


def _fortran_lines(content: str, /) -> list[str]:
    """Fixed-form Fortran lines without comment lines or trailing ``!`` comments."""
    lines = []
    for line in content.splitlines():
        if not line or line[0] in "cC*!" or line.lstrip().startswith("!"):
            continue
        lines.append(line.split("!", 1)[0])
    return lines


_FLOAT = re.compile(r"[-+]?\d+\.\d*(?:[dDeE][-+]?\d+)?")


def _fortran_data(content: str, name: str, /) -> np.ndarray:
    """Values of the single active ``data name / ... /`` statement."""
    start = re.compile(rf"^\s+data\s+{name}\s*/", re.IGNORECASE)
    blocks: list[str] = []
    collecting = False
    text = ""
    for line in _fortran_lines(content):
        if collecting:
            body = line[6:]
        else:
            match = start.match(line)
            if match is None:
                continue
            collecting = True
            body = line[match.end() :]
        if "/" in body:
            blocks.append(text + " " + body.split("/", 1)[0])
            collecting = False
            text = ""
        else:
            text += " " + body
    if len(blocks) != 1:
        raise ValueError(f"Expected one active data statement for {name!r}.")
    values = [
        float(token.replace("d", "e").replace("D", "e"))
        for token in _FLOAT.findall(blocks[0])
    ]
    return np.asarray(values, dtype=np.float64)


def _pars_records(content: str, /) -> np.ndarray:
    declared = re.search(r"nlines\s*=\s*(\d+)", content)
    if declared is None:
        raise ValueError("pars.f must declare nlines.")
    values: list[float] = []
    record = re.compile(r"[-+]?\d+\.\d+[dDeE][-+]?\d+")
    for line in _fortran_lines(content):
        if "pars(" in line.lower():
            line = line.split("=", 1)[1] if "=" in line else ""
        values.extend(
            float(token.replace("D", "e").replace("d", "e"))
            for token in record.findall(line)
        )
    count = int(declared.group(1))
    if len(values) != 5 * count:
        raise ValueError(
            f"pars.f declares {count} records but holds {len(values)} values."
        )
    return np.asarray(values, dtype=np.float64).reshape(count, 5)


def _decode(encoded: float, /) -> tuple[int, int]:
    value = int(round(encoded))
    if value != encoded or value < 1:
        raise ValueError("pars.f contains a non-integral element code.")
    slot = (value - 1) // 100
    return value - 100 * slot, slot


def _grimme_blocks(
    records: np.ndarray, species: tuple[int, ...], /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slots = 5
    row = {value: index for index, value in enumerate(species)}
    count = len(species)
    c6 = np.zeros((count, count, slots, slots))
    coordination = np.zeros((count, slots))
    valid = np.zeros((count, slots), dtype=np.bool_)
    seen: dict[tuple[int, int], float] = {}
    for c6_value, code_i, code_j, cn_i, cn_j in records:
        z_i, p = _decode(code_i)
        z_j, q = _decode(code_j)
        if not (0 < z_i <= 94 and 0 < z_j <= 94 and p < slots and q < slots):
            raise ValueError("pars.f contains an out-of-range record.")
        for key, value in (((z_i, p), cn_i), ((z_j, q), cn_j)):
            if seen.setdefault(key, value) != value:
                raise ValueError(
                    f"Reference coordination of Z={key[0]} slot {key[1]} differs "
                    "between partner records."
                )
        if z_i in row and z_j in row:
            a, b = row[z_i], row[z_j]
            if c6[a, b, p, q] != 0.0 and c6[a, b, p, q] != c6_value:
                raise ValueError("pars.f contains conflicting C6 records.")
            c6[a, b, p, q] = c6_value
            c6[b, a, q, p] = c6_value
    for (atomic_number, slot), value in seen.items():
        if atomic_number in row:
            coordination[row[atomic_number], slot] = value
            valid[row[atomic_number], slot] = True
    return c6, coordination, valid


@final
class D3BJDamping(StrictModule, NonTrainableState):
    """Becke-Johnson rational damping ``s6``, ``s8``, ``a1`` and ``a2``.

    ``a2`` is a length in ``a2_unit`` (Grimme's published values are bohr).
    The four parameters are dynamic scalar leaves so their derivatives are
    available, fixed by default. ``functional`` names the published
    parametrization (external provenance only).

    PyTree layout: ``functional`` is a static declared provenance label (like a
    term ``name``); ``damping_identity`` is the dynamic exact former
    ``damping_id`` digest (decoded by the host property).
    """

    s6: Array
    s8: Array
    a1: Array
    a2: Array
    a2_unit: UnitDefinition
    functional: str = eqx.field(static=True)
    damping_identity: IdentityWords

    def __init__(
        self,
        *,
        a1: float,
        a2: float,
        s8: float,
        s6: float = 1.0,
        a2_unit: UnitDefinition = BOHR,
        functional: str,
    ) -> None:
        values = {
            "s6": _finite(s6, "s6"),
            "s8": _finite(s8, "s8"),
            "a1": _finite(a1, "a1"),
            "a2": _finite(a2, "a2"),
        }
        if values["a1"] < 0.0 or values["a2"] < 0.0:
            raise ValueError("BJ damping a1 and a2 must be non-negative.")
        if values["a1"] == 0.0 and values["a2"] == 0.0:
            raise ValueError("BJ damping needs a positive damping radius.")
        unit = _unit(a2_unit, LENGTH, "a2_unit")
        for name, value in values.items():
            setattr(self, name, jnp.asarray(value, dtype=jnp.float64))
        self.a2_unit = unit
        self.functional = _text(functional, "functional")
        self.damping_identity = IdentityWords.of(
            {
                "kind": "d3-bj-damping",
                **{name: float(value).hex() for name, value in values.items()},
                "a2_unit": unit.unit_id,
                "functional": self.functional,
            }
        )

    @property
    def damping_id(self) -> str:
        """Host decoding of the canonical damping digest."""
        return self.damping_identity.fingerprint()


@final
class D3CoordinationPolicy(StrictModule, NonTrainableState):
    """Conventional truncated D3 coordination and its C6 interpolation rule.

    ``CN_i = sum_routes 1 / (1 + exp(-k1 (R_cov,ab / r - 1)))`` over every full
    directed image route with ``0 < r < coordination cutoff`` (the cutoff
    leaves a value step; it is never a smooth switch). The C6 interpolation
    weights ``exp(-k3 [(CN_i - CN_a^p)^2 + (CN_j - CN_b^q)^2])`` over populated
    slot pairs are normalized with a stop-gradient log-sum-exp shift (an exact
    algebraic identity). ``interpolation``:

    * ``"normalized-gaussian"``: every populated slot pair contributes; smooth
      in both coordination numbers.
    * ``"nvalchemi-relative-threshold"``: the inspected NVIDIA kernel's rule,
      skipping pairs whose shifted log weight is below ``-threshold``. It is
      discontinuous where a pair crosses the threshold (relative weight
      ``exp(-threshold)``); derivatives hold only away from those sets.

    Grimme's Fortran ``getc6`` sums unshifted weights and falls back to the
    nearest reference when they underflow (``rsum <= 1e-99``); neither rule
    reproduces that underflow branch.

    PyTree layout: ``interpolation`` (and with it whether ``threshold`` is
    present) is the static method selector. ``steepness``, ``width`` and
    ``threshold`` are dynamic float64 scalars and ``policy_identity`` the
    exact former ``policy_id`` digest (decoded by the host property).
    """

    interpolation: D3InterpolationRule = eqx.field(static=True)
    steepness: Array
    width: Array
    threshold: Array | None
    policy_identity: IdentityWords

    def __init__(
        self,
        interpolation: D3InterpolationRule,
        /,
        *,
        steepness: float = D3_COUNTING_STEEPNESS,
        width: float = D3_INTERPOLATION_WIDTH,
        threshold: float | None = None,
    ) -> None:
        rule = parse(interpolation, D3InterpolationRule, "interpolation")
        if rule == "nvalchemi-relative-threshold":
            cut = _positive(
                NVALCHEMI_INTERPOLATION_THRESHOLD if threshold is None else threshold,
                "threshold",
            )
        elif threshold is not None:
            raise ValueError("Only the thresholded interpolation rule has a threshold.")
        else:
            cut = None
        steep = _positive(steepness, "steepness")
        spread = _positive(width, "width")
        self.interpolation = rule
        self.steepness = jnp.asarray(steep, dtype=jnp.float64)
        self.width = jnp.asarray(spread, dtype=jnp.float64)
        self.threshold = None if cut is None else jnp.asarray(cut, dtype=jnp.float64)
        self.policy_identity = IdentityWords.of(
            {
                "kind": "d3-conventional-coordination",
                "interpolation": rule,
                "steepness": float(steep).hex(),
                "width": float(spread).hex(),
                "threshold": None if cut is None else float(cut).hex(),
                "counting": "logistic-truncated-at-coordination-cutoff",
            }
        )

    @property
    def policy_id(self) -> str:
        """Host decoding of the canonical coordination-policy digest."""
        return self.policy_identity.fingerprint()


@final
class FourierD3CoordinationDefinition(StrictModule, NonTrainableState):
    """The FourierD3 modified coordination counting function (source definition).

    For a route of length ``r`` and folded covalent distance ``R_cov`` (the sum
    of the two table radii), with ``R_t = (R_cov / covalent_radius_scale + R_c) / 2``,
    ``s = max(r, R_t)`` and ``g = R_c - s``::

        k(r) = steepness + (s - R_t)^2 / (g^2 + epsilon)
        f(r) = 1 / (1 + exp(-k(r) (R_cov / r - 1)))      for 0 < r < R_c

    and zero otherwise; ``R_c`` must equal the coordination neighborhood
    cutoff. ``epsilon`` is an area in ``epsilon_length_unit**2`` and is
    converted to the evaluation length unit. Regularity follows from the
    formula: ``k`` is C1 at ``R_t`` (second derivative jumps by
    ``2 / ((R_c - R_t)^2 + epsilon)``), and at ``R_c`` the value steps from
    ``1 / (1 + exp(k(R_c) (1 - R_cov / R_c)))`` (positive whenever
    ``epsilon > 0``) to zero, so neither C1 nor C0 holds exactly there.

    PyTree layout: ``steepness``, ``epsilon`` and ``width`` are dynamic float64
    scalars and ``definition_identity`` the exact former ``definition_id``
    digest (decoded by the host property); the unit is static.
    """

    steepness: Array
    epsilon: Array
    epsilon_length_unit: UnitDefinition
    width: Array
    definition_identity: IdentityWords

    def __init__(
        self,
        *,
        steepness: float = D3_COUNTING_STEEPNESS,
        epsilon: float = NVALCHEMI_FOURIER_EPSILON,
        epsilon_length_unit: UnitDefinition = BOHR,
        width: float = D3_INTERPOLATION_WIDTH,
    ) -> None:
        steep = _positive(steepness, "steepness")
        area = _positive(epsilon, "epsilon")
        unit = _unit(epsilon_length_unit, LENGTH, "epsilon_length_unit")
        spread = _positive(width, "width")
        self.steepness = jnp.asarray(steep, dtype=jnp.float64)
        self.epsilon = jnp.asarray(area, dtype=jnp.float64)
        self.epsilon_length_unit = unit
        self.width = jnp.asarray(spread, dtype=jnp.float64)
        self.definition_identity = IdentityWords.of(
            {
                "kind": "fourier-d3-modified-coordination",
                "steepness": float(steep).hex(),
                "epsilon": float(area).hex(),
                "epsilon_length_unit": unit.unit_id,
                "width": float(spread).hex(),
                "interpolation": "normalized-gaussian-unthresholded",
            }
        )

    @property
    def definition_id(self) -> str:
        """Host decoding of the canonical counting-definition digest."""
        return self.definition_identity.fingerprint()

    def epsilon_in(self, unit: UnitDefinition, /) -> float:
        """``epsilon`` converted to ``unit**2`` (host)."""
        return float(self.epsilon) * length_factor(self.epsilon_length_unit, unit) ** 2

    def log_cutoff_step(
        self,
        covalent_distance: float,
        cutoff: float,
        covalent_radius_scale: float,
        epsilon: float,
        /,
    ) -> float:
        """Natural log of the counting value just inside ``cutoff`` (host, float64).

        All lengths share one unit; ``epsilon`` is already in that unit squared.
        """
        transition = 0.5 * (covalent_distance / covalent_radius_scale + cutoff)
        if not transition < cutoff:
            raise ValueError(
                "The modified counting transition must lie inside the cutoff."
            )
        steep = float(self.steepness) + (cutoff - transition) ** 2 / epsilon
        argument = steep * (covalent_distance / cutoff - 1.0)
        return -math.log1p(math.exp(-argument)) if argument > -700.0 else argument


def _scan_errors(
    block: np.ndarray,
    values: np.ndarray,
    vectors: np.ndarray,
    populated: np.ndarray,
    cap: int,
    /,
) -> tuple[float, ...]:
    """Maximum relative populated-entry error after each rank-one signed update.

    The truncations accumulate ``lambda_r v_r v_r^T`` in magnitude order; every
    consumer (selection and re-measurement of stored factors) replays exactly
    this arithmetic, so a stored error is reproduced bit for bit.
    """
    reference = block[populated]
    scale = np.abs(reference)
    accumulated = np.zeros_like(block)
    errors = []
    for rank in range(cap):
        accumulated += values[rank] * np.outer(vectors[:, rank], vectors[:, rank])
        errors.append(float(np.max(np.abs(accumulated[populated] - reference) / scale)))
    return tuple(errors)


@final
class FourierC6Decomposition(StrictModule, NonTrainableState):
    """Signed low-rank factors of the reference species/slot C6 block matrix.

    ``M[(a,p),(b,q)] = c6_reference[a,b,p,q]`` over the decomposed ``species``
    is eigendecomposed on the host (bounded by ``maximum_block_size``) and
    ordered by eigenvalue magnitude. Negative eigenvalues are retained, never
    clipped. Unpopulated slots are set exactly to zero in the stored factors
    before any error is measured, so every reported error is that of the stored
    factors. The rank is the smallest ``r <= max_rank`` whose maximum relative
    entry error over populated slot pairs is ``<= tolerance``, found by scanning
    every rank (signed errors are not monotone). ``rank_errors[r-1]`` is the
    achieved error at rank ``r``. When no admitted rank meets the tolerance,
    ``meets_tolerance`` is false, the scanned maximum rank is stored and every
    consumer refuses it.

    This is a decomposition of the reference tables, never of an instantaneous
    atom-by-atom coefficient matrix; coefficients follow from the per-atom
    normalized interpolation of the factors.

    PyTree layout: the selected ``rank`` (eigenvalue/factor shapes), the scan
    length ``max_rank`` (``rank_errors`` shape) and ``criterion`` are static
    layout. The scanned ``spectrum``/``rank_errors`` (float64 arrays), row
    labels ``atomic_numbers`` (``int32[S]``, host property ``species``),
    ``tolerance``, ``achieved_error`` (float64), ``meets_tolerance`` (bool),
    ``negative_eigenvalues`` (int32) and the exact former ``tables_id``/
    ``decomposition_id`` digests (``tables_identity``/``decomposition_identity``,
    host properties) are dynamic.
    """

    eigenvalues: Array
    factors: Array
    spectrum: Array
    rank_errors: Array
    atomic_numbers: Array
    tables_identity: IdentityWords
    tolerance: Array
    max_rank: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    achieved_error: Array
    meets_tolerance: Array
    negative_eigenvalues: Array
    criterion: C6DecompositionCriterion = eqx.field(static=True)
    decomposition_identity: IdentityWords

    def __init__(
        self,
        tables: D3ReferenceTables,
        species: Sequence[int],
        /,
        *,
        tolerance: float,
        max_rank: int | None = None,
        maximum_block_size: int = 1024,
    ) -> None:
        if not isinstance(tables, D3ReferenceTables):
            raise TypeError("tables must be D3ReferenceTables.")
        covered = _species(species, "species")
        tol = _positive(tolerance, "tolerance")
        rows = tables.rows(np.asarray(covered))
        valid = np.asarray(tables.reference_valid)[rows]
        slots = valid.shape[1]
        size = len(covered) * slots
        if isinstance(maximum_block_size, bool) or int(maximum_block_size) < 1:
            raise ValueError("maximum_block_size must be a positive integer.")
        if size > int(maximum_block_size):
            raise ValueError(
                f"The reference block has {size} rows, exceeding maximum_block_size "
                f"{int(maximum_block_size)}."
            )
        populated_rows = int(np.count_nonzero(valid))
        if max_rank is None:
            cap = populated_rows
        else:
            if isinstance(max_rank, bool) or not isinstance(max_rank, (int, np.integer)):
                raise TypeError("max_rank must be an integer.")
            if int(max_rank) < 1:
                raise ValueError("max_rank must be positive.")
            cap = min(int(max_rank), populated_rows)
        c6 = np.asarray(tables.c6_reference)[np.ix_(rows, rows)]
        block = np.transpose(c6, (0, 2, 1, 3)).reshape(size, size)
        values, vectors = np.linalg.eigh(block)
        order = np.argsort(-np.abs(values), kind="stable")
        values, vectors = values[order], vectors[:, order]
        flat_valid = valid.reshape(size)
        vectors = np.where(flat_valid[:, None], vectors, 0.0)
        populated = flat_valid[:, None] & flat_valid[None, :]
        errors = _scan_errors(block, values, vectors, populated, cap)
        meeting = [index for index, error in enumerate(errors) if error <= tol]
        rank = meeting[0] + 1 if meeting else cap
        factors = vectors[:, :rank].reshape(len(covered), slots, rank)
        kept = values[:rank]
        criterion: C6DecompositionCriterion = "maximum-relative-entry-error"
        self.eigenvalues = jnp.asarray(kept)
        self.factors = jnp.asarray(factors)
        self.spectrum = jnp.asarray(np.asarray(values, dtype=np.float64))
        self.rank_errors = jnp.asarray(np.asarray(errors, dtype=np.float64))
        self.atomic_numbers = jnp.asarray(np.asarray(covered, dtype=np.int32))
        self.tables_identity = tables.tables_identity
        self.tolerance = jnp.asarray(tol, dtype=jnp.float64)
        self.max_rank = cap
        self.rank = rank
        self.achieved_error = jnp.asarray(errors[rank - 1], dtype=jnp.float64)
        self.meets_tolerance = jnp.asarray(bool(meeting), dtype=jnp.bool_)
        self.negative_eigenvalues = jnp.asarray(
            int(np.count_nonzero(kept < 0.0)), dtype=jnp.int32
        )
        self.criterion = criterion
        self.decomposition_identity = IdentityWords.of(
            {
                "kind": "fourier-c6-decomposition",
                "tables": tables.tables_id,
                "species": list(covered),
                "tolerance": float(tol).hex(),
                "max_rank": cap,
                "rank": rank,
                "criterion": criterion,
                "factors": array_tree_fingerprint(
                    {"eigenvalues": kept, "factors": factors}
                ),
            }
        )

    @property
    def species(self) -> tuple[int, ...]:
        """Host decoding of the decomposed atomic numbers (channel order)."""
        return tuple(int(value) for value in np.asarray(self.atomic_numbers))

    @property
    def tables_id(self) -> str:
        """Host decoding of the decomposed reference-table digest."""
        return self.tables_identity.fingerprint()

    @property
    def decomposition_id(self) -> str:
        """Host decoding of the canonical decomposition digest."""
        return self.decomposition_identity.fingerprint()

    @property
    def species_count(self) -> int:
        return int(self.atomic_numbers.shape[0])

    def reconstruction_error(self, tables: D3ReferenceTables, /) -> float:
        """Maximum relative populated-entry error of the stored factors (host)."""
        if tables.tables_id != self.tables_id:
            raise ValueError("The decomposition belongs to other reference tables.")
        rows = tables.rows(np.asarray(self.species))
        valid = np.asarray(tables.reference_valid)[rows]
        size = valid.size
        c6 = np.asarray(tables.c6_reference)[np.ix_(rows, rows)]
        block = np.transpose(c6, (0, 2, 1, 3)).reshape(size, size)
        flat = np.asarray(self.factors).reshape(size, self.rank)
        populated = valid.reshape(size)[:, None] & valid.reshape(size)[None, :]
        values = np.asarray(self.eigenvalues)
        return _scan_errors(block, values, flat, populated, self.rank)[-1]

    def require_admitted(self) -> None:
        """Refuse a decomposition whose capped rank misses its tolerance (host)."""
        if not bool(self.meets_tolerance):
            raise ValueError(
                f"C6 decomposition misses tolerance {float(self.tolerance):g}: best "
                f"scanned rank {self.rank} (cap {self.max_rank}) achieves "
                f"{float(self.achieved_error):.3e}."
            )

    def validate_restored(self, tables: D3ReferenceTables, /) -> None:
        """Re-measure the stored factors and identity after restoration."""
        if self.eigenvalues.shape != (self.rank,) or self.factors.shape[-1] != self.rank:
            raise ValueError("Restored C6 factors do not match their rank.")
        error = self.reconstruction_error(tables)
        if error != float(self.achieved_error):
            raise ValueError("Restored C6 factors do not reproduce their achieved error.")
        identity = canonical_fingerprint(
            {
                "kind": "fourier-c6-decomposition",
                "tables": tables.tables_id,
                "species": list(self.species),
                "tolerance": float(self.tolerance).hex(),
                "max_rank": self.max_rank,
                "rank": self.rank,
                "criterion": self.criterion,
                "factors": array_tree_fingerprint(
                    {
                        "eigenvalues": np.asarray(self.eigenvalues),
                        "factors": np.asarray(self.factors),
                    }
                ),
            }
        )
        if identity != self.decomposition_id:
            raise ValueError("Restored C6 decomposition does not match its identity.")


__all__ = [
    "C6DecompositionCriterion",
    "D3_COUNTING_STEEPNESS",
    "D3_INTERPOLATION_WIDTH",
    "D3BJDamping",
    "D3CoordinationPolicy",
    "D3InterpolationRule",
    "D3ReferenceTables",
    "D3TableKind",
    "D3TableProvenance",
    "energy_factor",
    "FourierC6Decomposition",
    "FourierD3CoordinationDefinition",
    "length_factor",
    "NVALCHEMI_FOURIER_EPSILON",
    "NVALCHEMI_INTERPOLATION_THRESHOLD",
]
