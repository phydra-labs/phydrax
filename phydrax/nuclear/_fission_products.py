#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Chain-end fission-gas yields and recoverable fission energy (ENDF/B-VIII.0).

The packaged table is produced by ``tools/extract_fission_gas_yields.py`` from
the pinned ENDF/B-VIII.0 distribution and verified against its pinned SHA-256
digest and byte size on every load.

Yields are taken at the thermal point (0.0253 eV) for U-235, Pu-239, and
Pu-241 and at the lowest incident energy tabulated in the ENDF/B-VIII.0
fission-yield sublibrary for U-238 (500 keV), Pu-240, and Pu-242 (both
0.0253 eV); the packaged data record each energy and its selection rule.

Xenon and krypton yields are ENDF MF8/MT459 cumulative ground-state yields
[atoms/fission] summed over the stable and long-lived (T1/2 > 1 y) isotopes
Kr-78, 80, 81, 82, 83, 84, 85, 86 and Xe-124, 126, 128, 130, 131, 132, 134, 136.
Xe-129 is excluded: its ENDF cumulative yield is carried by I-129
(T1/2 = 16.14 My), which does not decay on reactor time scales, and its
independent yield is zero for every tabulated system. Energy per fission is the
MF1/MT458 ``ER`` component (total energy less neutrino energy; includes the
delayed-neutron, delayed-gamma, and delayed-beta components) of the NPLY=0
point record: the thermal value for fissile targets and the near-threshold
value for fertile targets (ENDF-102 section 1.5).
"""

from __future__ import annotations

import math
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._document_resource import decode_json_resource
from .._external_resource import bounded_resource_from_bytes, ResourceLimits
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import read_reference_artifact, ReferenceArtifactManifest
from ..typing import parse
from ..units import conversion_factor, ELECTRONVOLT, JOULE
from ._provenance import NuclearDataProvenance


FissioningNuclide: TypeAlias = Literal["U235", "U238", "Pu239", "Pu240", "Pu241", "Pu242"]
# Canonical order of every fission-fraction axis.
FISSIONING_NUCLIDES: tuple[FissioningNuclide, ...] = typing.get_args(FissioningNuclide)

_DATA_DIRECTORY = Path(__file__).with_name("data")
_ARTIFACT_NAME = "fission_gas_yields.json"
_ARTIFACT_SHA256 = "4cccee6930ac8f6ffdae085041e9c0c17782503211eaff96d3cbc127eb849448"
_ARTIFACT_SIZE_BYTES = 16333


def _packaged_manifest() -> ReferenceArtifactManifest:
    # ENDF/B-VIII.0 and NUBASE2020 are released publicly by NNDC (BNL, US DOE)
    # and the IAEA Nuclear Data Section without a restrictive license.
    return ReferenceArtifactManifest(
        f"phydrax/nuclear/data/{_ARTIFACT_NAME}",
        checksum_algorithm="sha256",
        checksum=_ARTIFACT_SHA256,
        size_bytes=_ARTIFACT_SIZE_BYTES,
        license_id="LicenseRef-ENDF-B-VIII.0-public-release",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="publicly-available-scientific-data",
        nondimensionalization={"atoms_per_fission": 1.0, "joules_per_fission": 1.0},
        uncertainty=None,
        lineage_ids=(
            "endf-b-viii.0-fission-product-yield-sublibrary",
            "endf-b-viii.0-neutron-sublibrary",
            "nubase2020",
        ),
    )


def _mapping(value: object, name: str, /) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object.")
    return value


def _sequence(value: object, name: str, /) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"{name} must be a JSON array.")
    return value


def _real(value: object, name: str, /) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _gas_yield(records: object, name: str, /) -> float:
    """Sum of ``[mass_number, cumulative_yield, uncertainty]`` rows [atoms/fission]."""

    rows = _sequence(records, name)
    if not rows:
        raise ValueError(f"{name} must list at least one isotope.")
    return math.fsum(_real(_sequence(row, name)[1], name) for row in rows)


def _table_arrays(
    systems: Sequence[Any], electronvolt: float, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = FISSIONING_NUCLIDES
    if len(systems) != len(expected):
        raise ValueError("Packaged fission systems must match FissioningNuclide.")
    xenon: list[float] = []
    krypton: list[float] = []
    energy: list[float] = []
    for declared, value in zip(expected, systems, strict=True):
        record = _mapping(value, "fission system")
        nuclide = parse(record["fissioning_nuclide"], FissioningNuclide, "nuclide")
        if nuclide != declared:
            raise ValueError("Packaged fission systems must follow FissioningNuclide.")
        xenon.append(_gas_yield(record["xenon"], f"{nuclide} xenon"))
        krypton.append(_gas_yield(record["krypton"], f"{nuclide} krypton"))
        release = _mapping(record["energy_release_ev"], f"{nuclide} energy release")
        recoverable = _sequence(release["ER"], f"{nuclide} ER")
        energy.append(_real(recoverable[0], f"{nuclide} ER") * electronvolt)
    return (
        np.asarray(xenon, dtype=np.float64),
        np.asarray(krypton, dtype=np.float64),
        np.asarray(energy, dtype=np.float64),
    )


def _provenance(
    payload: Mapping[str, Any], reference: ReferenceArtifactManifest, /
) -> NuclearDataProvenance:
    sources = [
        _mapping(value, "source") for value in _sequence(payload["sources"], "sources")
    ]
    yield_sources = [
        value for value in sources if value["role"] == "fission-product-yield-sublibrary"
    ]
    if not yield_sources:
        raise ValueError("Packaged data must name its fission-yield source.")
    systems = [
        _mapping(value, "fission system")
        for value in _sequence(payload["fission_systems"], "fission_systems")
    ]
    evaluation = "; ".join(
        f"{value['fissioning_nuclide']}: yields [{value['yield_evaluation']}], "
        f"energy release [{value['energy_release_evaluation']}]"
        for value in systems
    )
    return NuclearDataProvenance(
        reference,
        str(yield_sources[0]["url"]),
        str(payload["library"]),
        str(payload["release"]),
        evaluation,
        processing_tool=str(payload["extractor"]),
        processing_release=f"sha256:{payload['extractor_sha256']}",
        processing_parameters=_mapping(payload["extraction"], "extraction"),
        parent_data_ids=tuple(
            dict.fromkeys(
                str(value[key])
                for value in sources
                for key in ("sha256", "member_sha256")
                if value[key] is not None
            )
        ),
    )


class FissionGasYieldTable(StrictModule, NonTrainableState):
    """Chain-end Xe and Kr yields and recoverable energy per fissioning system.

    Arrays follow ``FISSIONING_NUCLIDES`` order:
    ``xenon_yields`` and ``krypton_yields`` [atoms/fission] and
    ``energy_per_fission`` [J/fission].
    """

    xenon_yields: Array
    krypton_yields: Array
    energy_per_fission: Array
    nuclides: tuple[FissioningNuclide, ...] = eqx.field(static=True)
    provenance: NuclearDataProvenance = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        xenon_yields: ArrayLike,
        krypton_yields: ArrayLike,
        energy_per_fission: ArrayLike,
        provenance: NuclearDataProvenance,
    ) -> None:
        nuclides = FISSIONING_NUCLIDES
        xenon = np.asarray(xenon_yields, dtype=np.float64)
        krypton = np.asarray(krypton_yields, dtype=np.float64)
        energy = np.asarray(energy_per_fission, dtype=np.float64)
        for name, values in (
            ("xenon_yields", xenon),
            ("krypton_yields", krypton),
            ("energy_per_fission", energy),
        ):
            if values.shape != (len(nuclides),):
                raise ValueError(f"{name} must have one value per FissioningNuclide.")
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} must be finite and nonnegative.")
        if np.any(energy <= 0.0):
            raise ValueError("energy_per_fission must be positive.")
        if not isinstance(provenance, NuclearDataProvenance):
            raise TypeError("provenance must be NuclearDataProvenance.")
        self.xenon_yields = jnp.asarray(xenon)
        self.krypton_yields = jnp.asarray(krypton)
        self.energy_per_fission = jnp.asarray(energy)
        self.nuclides = nuclides
        self.provenance = provenance
        self.table_id = canonical_fingerprint(
            {
                "kind": "fission-gas-yield-table",
                "nuclides": list(nuclides),
                "arrays": array_tree_fingerprint((xenon, krypton, energy)),
                "provenance": provenance.provenance_id,
            }
        )

    @classmethod
    def endf_b_viii_0(cls) -> FissionGasYieldTable:
        """Load the packaged ENDF/B-VIII.0 table after verifying its digest."""

        reference = _packaged_manifest()
        verified = read_reference_artifact(
            _DATA_DIRECTORY / _ARTIFACT_NAME, reference, trusted_root=_DATA_DIRECTORY
        )
        resource = bounded_resource_from_bytes(
            verified.data,
            limits=ResourceLimits(reference.size_bytes, 16, 100_000, 100_000, 0),
            source_path=verified.manifest.source_path,
        )
        payload = _mapping(decode_json_resource(resource).value, "fission-gas yields")
        xenon, krypton, energy = _table_arrays(
            _sequence(payload["fission_systems"], "fission_systems"),
            float(conversion_factor(ELECTRONVOLT, JOULE)),
        )
        return cls(
            xenon_yields=xenon,
            krypton_yields=krypton,
            energy_per_fission=energy,
            provenance=_provenance(payload, reference),
        )

    def _fractions(self, fission_fractions: ArrayLike, /) -> Array:
        fractions = jnp.asarray(fission_fractions, dtype=self.xenon_yields.dtype)
        if fractions.ndim < 1 or fractions.shape[-1] != len(self.nuclides):
            raise ValueError(
                "fission_fractions must have a trailing axis of one fraction per "
                "FissioningNuclide."
            )
        tolerance = 64.0 * jnp.finfo(fractions.dtype).eps
        return eqx.error_if(
            fractions,
            jnp.any(fractions < 0.0)
            | jnp.any(jnp.abs(jnp.sum(fractions, axis=-1) - 1.0) > tolerance),
            "fission_fractions must be nonnegative and sum to one.",
        )

    def mixed_yields(self, fission_fractions: ArrayLike, /) -> tuple[Array, Array]:
        """Fission-fraction-weighted ``(xenon, krypton)`` yields [atoms/fission]."""

        fractions = self._fractions(fission_fractions)
        return fractions @ self.xenon_yields, fractions @ self.krypton_yields

    def mixed_energy_per_fission(self, fission_fractions: ArrayLike, /) -> Array:
        """Fission-fraction-weighted recoverable energy per fission [J]."""

        return self._fractions(fission_fractions) @ self.energy_per_fission


__all__ = ["FISSIONING_NUCLIDES", "FissionGasYieldTable", "FissioningNuclide"]
