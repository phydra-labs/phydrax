#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Trusted conversion of admitted mace-torch checkpoints into native MACE models.

The converter is a host boundary. Source bytes are admitted through the
canonical external-artifact policy (exact SHA-256, size and license), executed
only by a caller-pinned provider interpreter through ``run_pinned_command``,
and returned as a bounded pickle-free array archive. The provider program
(``_mace_worker.py``) verifies the installed mace-torch/e3nn releases, refuses
non-standard module trees, and proves that the extracted architecture
declaration rebuilds a provider model that reproduces the source exactly.

Full-object torch checkpoints are executable pickles. They are deserialized
only when the caller constructs ``TrustedTorchPickleSource`` for the exact
admitted digest; there is no fallback from a refused or failed safe load. The
pinned subprocess bounds wall time, log bytes and output files; it is neither a
security nor a memory sandbox. The provider therefore receives the explicit
``MACEConversionLimits`` and admits the source container, and for a safe state
dict every storage and tensor view from its pickle metadata, before anything
is deserialized; it pins every size-determining dimension of a safe source's
declared architecture to those admitted tensors and plans every provider model
construction (parameters, buffers and construction workspace) before building;
and it admits every result array from its metadata before converting it and
streams the result archive. A trusted full-object pickle executes, and
allocates, with the caller's privileges. Conversion outputs contain no provider
payload: the native model keeps the original source parameterization (e3nn
weights, symmetric-contraction W and the provider U basis as fixed leaves).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

import numpy as np

from ..._array_archive import ArrayArchiveLimits, read_array_archive
from ..._artifact_security import (
    AdmittedExternalArtifact,
    ExternalArtifactPolicy,
    read_admitted_artifact,
)
from ..._external_runtime import (
    ExternalExecutionPolicy,
    ExternalRuntimeError,
    PinnedExecutable,
    PinnedFileOutputs,
    PinnedFileRequest,
    run_pinned_command,
)
from ..._fingerprint import canonical_fingerprint
from ...artifacts import ArtifactManifest
from ...typing import parse
from .._types import AtomisticPrecisionPolicy, AtomisticScaleContract


if TYPE_CHECKING:
    from ...nn.atomistic._mace import MACEArchitecture, MACEPotential
    from ...nn.atomistic._mace_interaction import MACEInteractionKind
    from ...sparse import StreamedRelationPlan


MACESourceKind: TypeAlias = Literal["torch-full-model", "torch-state-dict"]
MACEEvaluationDtype: TypeAlias = Literal["float32", "float64"]

# Provider releases whose source was read to define this conversion contract.
MACE_PROVIDER_RELEASES: Mapping[str, str] = MappingProxyType(
    {"e3nn": "0.4.4", "mace-torch": "0.3.16"}
)

_WORKER_NAME = "_mace_worker.py"
_REFUSED_EXIT = 3
_RESULT_FORMAT = "phydrax-mace-provider-result"
_BASIS_TOLERANCE = 1.0e-12
_NORMALIZATION_TOLERANCE = 1.0e-12
# Fixed isolated bootstrap: add only the declared provider site-packages, then
# run the staged worker as ``__main__`` with its explicit arguments.
_BOOTSTRAP = (
    "import runpy, site, sys; site.addsitedir(sys.argv[1]); "
    "sys.argv = [sys.argv[2], sys.argv[1], *sys.argv[3:]]; "
    "runpy.run_path(sys.argv[0], run_name='__main__')"
)
_IRREP = re.compile(r"(\d+)x(\d+)([eo])")
_SHARED_DECLARATION_FIELDS = frozenset(
    {
        "MLP_irreps",
        "apply_cutoff",
        "atomic_energies",
        "atomic_numbers",
        "avg_num_neighbors",
        "correlation",
        "distance_transform",
        "gate",
        "heads",
        "hidden_irreps",
        "interaction_classes",
        "max_ell",
        "model_class",
        "num_bessel",
        "num_interactions",
        "num_polynomial_cutoff",
        "pair_repulsion",
        "r_max",
        "radial_MLP",
        "radial_type",
        "use_agnostic_product",
        "use_last_readout_only",
        "use_reduced_cg",
    }
)
_SCALE_SHIFT_FIELDS = frozenset({"atomic_inter_scale", "atomic_inter_shift"})
_RESULT_FIELDS = {
    "extract": frozenset(
        {"arrays", "declaration", "format", "modules", "operation", "provider", "source"}
    ),
    "evaluate": frozenset(
        {
            "arrays",
            "declaration",
            "evaluation",
            "format",
            "operation",
            "provider",
            "source",
        }
    ),
    "gradients": frozenset(
        {
            "arrays",
            "declaration",
            "evaluation",
            "format",
            "operation",
            "provider",
            "source",
        }
    ),
    "fixture": frozenset({"arrays", "fixture", "format", "operation", "provider"}),
}


class MACESourceRefusedError(ValueError):
    """The provider refused a source outside the admitted standard MACE contract."""


@dataclass(frozen=True, slots=True)
class MACEProviderRuntime:
    """Caller-pinned provider interpreter and its exact provider site-packages.

    ``interpreter`` pins the interpreter bytes. mace-torch and e3nn must be the
    admitted ``MACE_PROVIDER_RELEASES``; the provider verifies their installed
    files against their RECORD digests on every run. ``torch_version`` is the
    caller-declared torch release, verified by version and recorded with its
    RECORD digest. ``cuequivariance_version`` declares the optional
    cuequivariance and cuequivariance-torch release (one shared version) that
    mace-torch uses to construct reduced generalized-CG U bases (RECORD
    verified); it must be declared exactly when installed, and reduced-CG
    sources can be identified and rebuilt only with it.
    """

    interpreter: PinnedExecutable
    site_packages: str
    torch_version: str
    cuequivariance_version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.interpreter, PinnedExecutable):
            raise TypeError("interpreter must be a PinnedExecutable.")
        location = Path(self.site_packages).expanduser().resolve(strict=True)
        if not location.is_dir():
            raise ValueError("site_packages must be an existing directory.")
        for name in ("torch_version", "cuequivariance_version"):
            value = getattr(self, name)
            if (name == "torch_version" or value is not None) and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be an explicit release.")
        object.__setattr__(self, "site_packages", str(location))

    @property
    def distributions(self) -> dict[str, str]:
        optional = (
            {}
            if self.cuequivariance_version is None
            else {
                "cuequivariance": self.cuequivariance_version,
                "cuequivariance-torch": self.cuequivariance_version,
            }
        )
        return {**MACE_PROVIDER_RELEASES, "torch": self.torch_version, **optional}


@dataclass(frozen=True, slots=True)
class TrustedTorchPickleSource:
    """Explicit capability to execute the pickle of exactly one admitted digest.

    Constructing this record is the caller's trust decision for code execution
    by ``torch.load(weights_only=False)``. It names the exact SHA-256 it
    authorizes and a nonempty statement of why the source is trusted.
    """

    sha256: str
    statement: str

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("Trusted source sha256 must be lowercase hexadecimal.")
        if not isinstance(self.statement, str) or not self.statement.strip():
            raise ValueError("A trusted pickle source requires a trust statement.")


@dataclass(frozen=True, slots=True)
class MACESource:
    """One admitted MACE source and its explicit loading contract.

    ``"torch-full-model"`` requires a ``TrustedTorchPickleSource`` for the
    admitted digest. ``"torch-state-dict"`` is still a pickle (``torch.save``'s
    ``data.pkl``) restricted to a dict of plain tensors: the provider admits its
    metadata without unpickling, then loads it with
    ``torch.load(weights_only=True)``. It requires the exact declared provider
    architecture (a declaration previously returned by extraction).
    """

    artifact: AdmittedExternalArtifact
    manifest: ArtifactManifest
    policy: ExternalArtifactPolicy
    kind: MACESourceKind
    trust: TrustedTorchPickleSource | None = None
    architecture: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, AdmittedExternalArtifact):
            raise TypeError("artifact must be an AdmittedExternalArtifact.")
        if not isinstance(self.manifest, ArtifactManifest):
            raise TypeError("manifest must be an ArtifactManifest.")
        if not isinstance(self.policy, ExternalArtifactPolicy):
            raise TypeError("policy must be an ExternalArtifactPolicy.")
        kind = parse(self.kind, MACESourceKind, "kind")
        match kind:
            case "torch-full-model":
                if not isinstance(self.trust, TrustedTorchPickleSource):
                    raise PermissionError(
                        "Full-object torch checkpoints require an explicit "
                        "TrustedTorchPickleSource before deserialization."
                    )
                if self.trust.sha256 != self.artifact.sha256:
                    raise PermissionError(
                        "The trusted-source capability names a different digest."
                    )
            case "torch-state-dict":
                if self.trust is not None:
                    raise ValueError("Safe state-dict sources are never pickle-trusted.")
                if self.architecture is None:
                    raise ValueError(
                        "A state-dict source requires its exact declared architecture."
                    )
        architecture = (
            None
            if self.architecture is None
            else MappingProxyType(_validated_declaration(self.architecture))
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "architecture", architecture)


@dataclass(frozen=True, slots=True)
class MACEConversionLimits:
    """Explicit byte, member, element and time bounds for one provider run.

    The provider enforces them before allocation. Source bytes must be the
    stored ZIP container ``torch.save`` writes: at most ``max_members``
    records, uncompressed, expanding to at most ``max_source_bytes``; each
    tensor storage within ``max_tensor_bytes``; ``data.pkl``, the metadata
    records and the ZIP directory within ``max_manifest_bytes``. For a safe
    state dict every tensor view is admitted from the pickle metadata before
    ``torch.load``: rank within the result archive rank limit, at most
    ``max_tensor_elements`` elements and ``max_tensor_bytes`` logical bytes,
    inside its storage, with at most ``max_source_bytes`` in total. The
    declared architecture's elements, interactions, radial widths,
    spherical-harmonic degree, hidden irreps, correlation orders, heads and
    readout widths must then equal those of the admitted tensors before the
    provider builds it. Every provider construction (source, rebuild or
    fixture) first bounds the declared extents its planner expands (the
    ``(max_ell + 1)**2`` edge-harmonic components within the tensor limits,
    at most ``max_members`` hidden irreps entries, one nonlinear-readout MLP
    entry) and is then planned from irreps metadata: each parameter, buffer,
    coupling-basis recursion depth, traced example input and einsum
    intermediate must fit ``max_tensor_elements``/``max_tensor_bytes``, the
    model's parameters and buffers together ``max_source_bytes``, and its
    correlation orders ``max_members``. Before the provider discovers any
    neighbor list (evaluation, gradients and its reproduction probes) the
    edges are counted without materializing them: the ``[E, 3]`` float64
    shifts and periodic ``[T, 3]`` image-shift candidates must fit the tensor
    limits, and an evaluation result (energies, forces, node energies,
    ``[2, E]`` edge indices, ``[E, 3]`` unit shifts, stress) is admitted from
    those counts before any model runs. Result arrays obey the same
    per-array bounds and are streamed into an archive of at most
    ``max_members`` members and ``max_result_bytes`` bytes, which the host
    re-admits under ``archive_limits``. ``timeout_seconds`` and
    ``max_log_bytes`` bound the process itself, which is not a memory sandbox: a
    trusted full-object pickle allocates as it executes.
    """

    max_source_bytes: int = 1_073_741_824
    max_result_bytes: int = 2_147_483_648
    max_tensor_bytes: int = 536_870_912
    max_tensor_elements: int = 134_217_728
    max_members: int = 4_097
    max_manifest_bytes: int = 16_777_216
    max_log_bytes: int = 16_777_216
    timeout_seconds: float = 1_800.0

    def __post_init__(self) -> None:
        integers = (
            self.max_source_bytes,
            self.max_result_bytes,
            self.max_tensor_bytes,
            self.max_tensor_elements,
            self.max_members,
            self.max_manifest_bytes,
            self.max_log_bytes,
        )
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError("MACE conversion limits must be positive integers.")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive and finite.")

    def archive_limits(self) -> ArrayArchiveLimits:
        return ArrayArchiveLimits(
            max_container_bytes=self.max_result_bytes,
            max_aggregate_bytes=self.max_result_bytes,
            max_member_bytes=self.max_tensor_bytes,
            max_manifest_bytes=self.max_manifest_bytes,
            max_members=self.max_members,
            max_central_directory_bytes=self.max_manifest_bytes,
            max_axis_length=self.max_tensor_elements,
            max_array_elements=self.max_tensor_elements,
            max_total_array_elements=max(self.max_tensor_elements, self.max_result_bytes),
        )


def _worker_limits(limits: MACEConversionLimits, /) -> dict[str, int]:
    """The bounds the provider enforces before it allocates source or result arrays."""

    archive = limits.archive_limits()
    return {
        "max_source_bytes": limits.max_source_bytes,
        "max_result_bytes": archive.max_container_bytes,
        "max_tensor_bytes": archive.max_member_bytes,
        "max_tensor_elements": archive.max_array_elements,
        "max_members": archive.max_members,
        "max_manifest_bytes": archive.max_manifest_bytes,
        "max_array_rank": archive.max_array_rank,
        "max_total_elements": archive.max_total_array_elements,
    }


@dataclass(frozen=True, slots=True)
class MACESourceProvenance:
    """Source, provider, trust and transform identity of one conversion.

    This record is serialized into native model artifacts; it is evidence of
    origin and conversion, not a native schema generation.
    ``layout_normalization`` records how the provider expressed a source saved
    by an earlier mace-torch release in the admitted release's exact layout
    (``legacy_head``, ``relabelled_reduced_cg`` when the stored U basis
    contradicts the source's construction attribute, ``dropped_inert_state``,
    ``integral_buffers``, ``derived_zero_flags``, and ``precision_buffers`` cast
    exactly to the parameter precision); every entry is false or empty for a
    source already in the admitted layout, including every safe state-dict
    source. ``declaration["use_reduced_cg"]`` names the identified stored basis
    and ``reparameterized_u_buffers`` the stored U buffers that span it in
    another exact path parameterization (a source fact, executed as stored).
    """

    source_sha256: str
    source_byte_size: int
    source_license_id: str
    source_manifest_id: str
    source_admission_id: str
    source_kind: MACESourceKind
    trusted_full_object: bool
    source_dtype: str
    provider: Mapping[str, Any]
    interpreter_sha256: str
    interpreter_version: str
    declaration: Mapping[str, Any]
    head: str
    basis_residual: float
    layout_normalization: Mapping[str, Any]
    conversion_id: str

    def to_record(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "source_byte_size": self.source_byte_size,
            "source_license_id": self.source_license_id,
            "source_manifest_id": self.source_manifest_id,
            "source_admission_id": self.source_admission_id,
            "source_kind": self.source_kind,
            "trusted_full_object": self.trusted_full_object,
            "source_dtype": self.source_dtype,
            "provider": _plain(self.provider),
            "interpreter_sha256": self.interpreter_sha256,
            "interpreter_version": self.interpreter_version,
            "declaration": _plain(self.declaration),
            "head": self.head,
            "basis_residual": self.basis_residual,
            "layout_normalization": _plain(self.layout_normalization),
            "conversion_id": self.conversion_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any], /) -> MACESourceProvenance:
        """Parse one serialized record and verify its conversion identity."""

        fields = {field.name for field in dataclasses.fields(cls)}
        if not isinstance(record, Mapping) or set(record) != fields:
            raise ValueError("MACE source provenance fields are not canonical.")
        provenance = cls(
            source_sha256=_digest(record["source_sha256"], "source_sha256"),
            source_byte_size=_count(record["source_byte_size"], "source_byte_size"),
            source_license_id=_text(record["source_license_id"], "source_license_id"),
            source_manifest_id=_text(record["source_manifest_id"], "source_manifest_id"),
            source_admission_id=_text(
                record["source_admission_id"], "source_admission_id"
            ),
            source_kind=parse(record["source_kind"], MACESourceKind, "source_kind"),
            trusted_full_object=_boolean(
                record["trusted_full_object"], "trusted_full_object"
            ),
            source_dtype=_text(record["source_dtype"], "source_dtype"),
            provider=MappingProxyType(_provider_record(record["provider"])),
            interpreter_sha256=_digest(
                record["interpreter_sha256"], "interpreter_sha256"
            ),
            interpreter_version=_text(
                record["interpreter_version"], "interpreter_version"
            ),
            declaration=MappingProxyType(_validated_declaration(record["declaration"])),
            head=_text(record["head"], "head"),
            basis_residual=_finite(record["basis_residual"], "basis_residual"),
            layout_normalization=MappingProxyType(
                _layout_record(record["layout_normalization"])
            ),
            conversion_id=_text(record["conversion_id"], "conversion_id"),
        )
        if provenance.conversion_id != _conversion_id(provenance):
            raise ValueError("MACE source provenance identity is not canonical.")
        return provenance


@dataclass(frozen=True, slots=True)
class MACECheckpointConversion:
    """A native MACE model, its provenance and its exact source parameterization.

    ``source_tensors`` are the admitted provider tensors (read-only, source
    dtype) passed to the native factory together with the audited degree
    transforms and couplings. ``reconstruct`` rebuilds the native model from
    replaced source tensors through the same factory, which is how source
    parameter directions map into native parameter space.
    """

    potential: MACEPotential
    provenance: MACESourceProvenance
    architecture: MACEArchitecture
    source_tensors: Mapping[str, np.ndarray]
    degree_transforms: tuple[np.ndarray, ...]
    couplings: Mapping[tuple[int, int, int], np.ndarray]
    precision: AtomisticPrecisionPolicy | None
    streaming: StreamedRelationPlan | None
    maximum_source_entries: int

    def reconstruct(self, replacements: Mapping[str, np.ndarray], /) -> MACEPotential:
        """Rebuild the native model with some source tensors replaced exactly.

        Replacements must name existing source tensors and keep their shapes;
        the factory's own validation applies unchanged.
        """

        from ...nn.atomistic._mace_source import mace_potential_from_source
        from ...units import ANGSTROM, ELECTRONVOLT

        tensors = dict(self.source_tensors)
        for name, value in replacements.items():
            original = tensors.get(name)
            replacement = np.asarray(value)
            if original is None or replacement.shape != original.shape:
                raise ValueError(
                    f"Replacement {name!r} is not an existing source tensor."
                )
            tensors[name] = replacement
        return mace_potential_from_source(
            AtomisticScaleContract(ANGSTROM, ELECTRONVOLT),
            self.architecture,
            tensors,
            degree_transforms=self.degree_transforms,
            couplings=self.couplings,
            source_id=self.provenance.conversion_id,
            precision=self.precision,
            streaming=self.streaming,
            maximum_source_entries=self.maximum_source_entries,
        )


@dataclass(frozen=True, slots=True)
class MACEProviderConfiguration:
    """One finite or fully periodic configuration in Angstrom.

    ``cell`` rows are lattice vectors; it must be zero for finite systems.
    """

    numbers: tuple[int, ...]
    positions: np.ndarray
    cell: np.ndarray
    periodic: bool

    def __post_init__(self) -> None:
        numbers = tuple(int(value) for value in self.numbers)
        positions = np.array(self.positions, dtype=np.float64, copy=True)
        cell = np.array(self.cell, dtype=np.float64, copy=True)
        if not numbers or any(value <= 0 for value in numbers):
            raise ValueError("Configurations require positive atomic numbers.")
        if positions.shape != (len(numbers), 3) or cell.shape != (3, 3):
            raise ValueError("Configuration positions or cell have the wrong shape.")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(cell)):
            raise ValueError("Configuration geometry must be finite.")
        if not isinstance(self.periodic, bool):
            raise TypeError("periodic must be a bool.")
        if not self.periodic and np.any(cell != 0.0):
            raise ValueError("Finite configurations carry a zero cell.")
        positions.setflags(write=False)
        cell.setflags(write=False)
        object.__setattr__(self, "numbers", numbers)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "cell", cell)

    def to_request(self) -> dict[str, Any]:
        return {
            "numbers": list(self.numbers),
            "positions": self.positions.tolist(),
            "cell": self.cell.tolist(),
            "pbc": [self.periodic] * 3,
        }


@dataclass(frozen=True, slots=True)
class MACEProviderCase:
    """Provider outputs for one configuration (eV, eV/Angstrom, eV/Angstrom^3).

    ``stress`` is the provider's ``(1/V) dE/d(symmetric strain)`` for periodic
    cases and ``None`` for finite ones. ``edge_index`` rows are provider sender
    and receiver indices with integer ``unit_shifts``.
    """

    energy: float
    forces: np.ndarray
    stress: np.ndarray | None
    node_energy: np.ndarray
    edge_index: np.ndarray
    unit_shifts: np.ndarray


@dataclass(frozen=True, slots=True)
class MACEProviderEvaluation:
    head: str
    evaluation_dtype: MACEEvaluationDtype
    cases: tuple[MACEProviderCase, ...]
    provider: Mapping[str, Any]
    declaration: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MACEProviderGradients:
    """Provider parameter gradients in the original source parameterization.

    For each case, ``energy_gradients[name]`` is dE/d(theta_name) and
    ``force_gradients[name]`` is d(sum F * W)/d(theta_name) for the supplied
    force weights W, keyed by provider state-dict parameter names.
    """

    head: str
    evaluation_dtype: MACEEvaluationDtype
    parameters: tuple[str, ...]
    energies: tuple[float, ...]
    forces: tuple[np.ndarray, ...]
    energy_gradients: tuple[Mapping[str, np.ndarray], ...]
    force_gradients: tuple[Mapping[str, np.ndarray], ...]
    provider: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MACEProviderFixture:
    """A deterministic provider-built source fixture on disk.

    ``kind`` is ``"torch-state-dict"`` (tensors only) or ``"torch-full-model"``
    (a pickled provider module, which conversion executes only under explicit
    trust like any full-object source). ``dtype`` is the provider's parameter
    and buffer precision, including its stored coupling buffers.
    """

    path: Path
    kind: MACESourceKind
    dtype: MACEEvaluationDtype
    sha256: str
    byte_size: int
    seed: int
    declaration: Mapping[str, Any]
    provider: Mapping[str, Any]


def _plain(value: Any, /) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _text(value: Any, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be canonical nonempty text.")
    return value


def _digest(value: Any, name: str, /) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return text


def _count(value: Any, name: str, /) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _positive(value: Any, name: str, /) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _boolean(value: Any, name: str, /) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean.")
    return value


def _finite(value: Any, name: str, /) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def _irreps(text: Any, name: str, /) -> tuple[tuple[int, int, int], ...]:
    """Parse e3nn's canonical ``str(Irreps)`` into (multiplicity, l, parity)."""

    if not isinstance(text, str) or not text:
        raise ValueError(f"{name} must be e3nn irreps text.")
    entries = []
    for item in text.split("+"):
        match = _IRREP.fullmatch(item)
        if match is None:
            raise ValueError(f"{name} contains invalid irreps text {item!r}.")
        entries.append(
            (int(match.group(1)), int(match.group(2)), 1 if match.group(3) == "e" else -1)
        )
    return tuple(entries)


def _numbers(value: Any, name: str, /) -> list[float]:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(flat)):
        raise ValueError(f"{name} must be finite.")
    return flat.tolist()


def _validated_declaration(value: Any, /) -> dict[str, Any]:
    """Validate one provider architecture declaration without interpreting it."""

    if not isinstance(value, Mapping):
        raise TypeError("A MACE architecture declaration must be a mapping.")
    declaration = dict(value)
    match declaration.get("model_class"):
        case "MACE":
            expected = _SHARED_DECLARATION_FIELDS
        case "ScaleShiftMACE":
            expected = _SHARED_DECLARATION_FIELDS | _SCALE_SHIFT_FIELDS
        case other:
            raise ValueError(f"Unsupported source model class {other!r}.")
    if set(declaration) != expected:
        raise ValueError("MACE architecture declaration fields are not exact.")
    count = _positive(declaration["num_interactions"], "num_interactions")
    for name in ("num_bessel", "num_polynomial_cutoff"):
        _positive(declaration[name], name)
    _count(declaration["max_ell"], "max_ell")
    for name in ("r_max", "avg_num_neighbors"):
        if _finite(declaration[name], name) <= 0.0:
            raise ValueError(f"{name} must be positive.")
    classes = declaration["interaction_classes"]
    correlation = declaration["correlation"]
    if (
        not isinstance(classes, list)
        or len(classes) != count
        or not isinstance(correlation, list)
        or len(correlation) != count
    ):
        raise ValueError("Per-interaction declaration lists have the wrong length.")
    for item in correlation:
        _positive(item, "correlation")
    numbers = declaration["atomic_numbers"]
    heads = declaration["heads"]
    if (
        not isinstance(numbers, list)
        or not numbers
        or len(set(numbers)) != len(numbers)
        or any(type(item) is not int or item <= 0 for item in numbers)
    ):
        raise ValueError("atomic_numbers must be unique positive integers.")
    if (
        not isinstance(heads, list)
        or not heads
        or len(set(heads)) != len(heads)
        or any(not isinstance(item, str) or not item for item in heads)
    ):
        raise ValueError("heads must be unique nonempty names.")
    _irreps(declaration["hidden_irreps"], "hidden_irreps")
    if declaration["MLP_irreps"] is not None:
        _irreps(declaration["MLP_irreps"], "MLP_irreps")
    _numbers(declaration["atomic_energies"], "atomic_energies")
    for name in _SCALE_SHIFT_FIELDS & set(declaration):
        _numbers(declaration[name], name)
    return declaration


def _provider_record(value: Any, /) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "distributions",
        "python",
        "site_packages",
    }:
        raise ValueError("Provider identity record is not canonical.")
    distributions = value["distributions"]
    if not isinstance(distributions, Mapping):
        raise ValueError("Provider distributions are not a mapping.")
    for name, record in distributions.items():
        if not isinstance(record, Mapping) or set(record) != {
            "record_sha256",
            "verified_files",
            "version",
        }:
            raise ValueError(f"Provider distribution {name!r} record is invalid.")
        _digest(record["record_sha256"], "record_sha256")
        _count(record["verified_files"], "verified_files")
        _text(record["version"], "version")
    for name, version in MACE_PROVIDER_RELEASES.items():
        record = distributions.get(name)
        if record is None or record["version"] != version or record["verified_files"] < 1:
            raise ValueError(f"Provider {name} is not the verified admitted release.")
    _text(value["python"], "python")
    _text(value["site_packages"], "site_packages")
    return _plain(value)


_LAYOUT_FLAGS = ("legacy_head", "relabelled_reduced_cg")
_LAYOUT_NORMALIZATIONS = (
    "derived_zero_flags",
    "dropped_inert_state",
    "integral_buffers",
    "precision_buffers",
)
_LAYOUT_LISTS = (*_LAYOUT_NORMALIZATIONS, "reparameterized_u_buffers")


def _layout_record(value: Any, /) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {*_LAYOUT_FLAGS, *_LAYOUT_LISTS}:
        raise ValueError("Source layout normalization record is not canonical.")
    for name in _LAYOUT_FLAGS:
        _boolean(value[name], name)
    for name in _LAYOUT_LISTS:
        names = value[name]
        if (
            not isinstance(names, list)
            or names != sorted(set(names))
            or any(not isinstance(item, str) or not item for item in names)
        ):
            raise ValueError(f"Source layout normalization {name} is not canonical.")
    return _plain(value)


def _normalized_layout(record: Mapping[str, Any], /) -> bool:
    return any(record[name] for name in (*_LAYOUT_FLAGS, *_LAYOUT_NORMALIZATIONS))


def _worker_bytes() -> bytes:
    return resources.files(__package__).joinpath(_WORKER_NAME).read_bytes()


def _source_payload(source: MACESource, limits: MACEConversionLimits, /) -> bytes:
    if source.manifest.byte_size > limits.max_source_bytes:
        raise ValueError("MACE source exceeds the declared source byte limit.")
    return read_admitted_artifact(source.artifact, source.manifest, policy=source.policy)


def _source_request(source: MACESource, /) -> dict[str, Any]:
    return {
        "kind": source.kind,
        "sha256": source.artifact.sha256,
        "byte_size": source.artifact.byte_size,
        "trusted_full_object": source.kind == "torch-full-model",
    }


def _run_provider(
    provider: MACEProviderRuntime,
    request: Mapping[str, Any],
    source_payload: bytes | None,
    limits: MACEConversionLimits,
    /,
    *,
    extra_outputs: Sequence[tuple[str, int]] = (),
) -> tuple[dict[str, Any], dict[str, np.ndarray], Path, tempfile.TemporaryDirectory[str]]:
    if not isinstance(provider, MACEProviderRuntime):
        raise TypeError("provider must be a MACEProviderRuntime.")
    if not isinstance(limits, MACEConversionLimits):
        raise TypeError("limits must be MACEConversionLimits.")
    complete = {
        **request,
        "provider": {"distributions": provider.distributions},
        "limits": _worker_limits(limits),
    }
    encoded = json.dumps(complete, allow_nan=False, sort_keys=True).encode("utf-8")
    worker = _worker_bytes()
    inputs = {"worker.py": worker, "request.json": encoded}
    if source_payload is not None:
        inputs["source.bin"] = source_payload
    staged = sum(len(payload) for payload in inputs.values())
    destination = tempfile.TemporaryDirectory(prefix="phydrax-mace-provider-")
    requests = (
        PinnedFileRequest("result.zip", limits.max_result_bytes),
        *(PinnedFileRequest(name, maximum) for name, maximum in extra_outputs),
    )
    try:
        result = run_pinned_command(
            provider.interpreter,
            [
                "-I",
                "-S",
                "-c",
                _BOOTSTRAP,
                provider.site_packages,
                "worker.py",
                "request.json",
                "result.zip",
            ],
            inputs=inputs,
            timeout=limits.timeout_seconds,
            max_output_bytes=staged + limits.max_log_bytes,
            execution_policy=ExternalExecutionPolicy(inherit_environment=False),
            artifacts=PinnedFileOutputs(
                destination.name,
                requests,
                limits.max_result_bytes + sum(maximum for _, maximum in extra_outputs),
            ),
        )
    except ExternalRuntimeError as error:
        destination.cleanup()
        failed = error.result
        if failed is not None and failed.returncode == _REFUSED_EXIT:
            message = failed.stderr.decode("utf-8", "replace").strip().splitlines()
            refusal = next(
                (line for line in reversed(message) if line.startswith("REFUSED: ")),
                "REFUSED: provider refusal without a message",
            )
            raise MACESourceRefusedError(refusal.removeprefix("REFUSED: ")) from error
        raise
    location = Path(result.file_artifact("result.zip").location)
    manifest, arrays = read_array_archive(location, limits=limits.archive_limits())
    operation = request["operation"]
    if (
        manifest.get("format") != _RESULT_FORMAT
        or manifest.get("operation") != operation
        or set(manifest) != _RESULT_FIELDS[operation]
    ):
        destination.cleanup()
        raise ValueError("Provider result manifest is not the canonical result.")
    manifest["provider"] = _provider_record(manifest["provider"])
    for name, version in provider.distributions.items():
        if manifest["provider"]["distributions"][name]["version"] != version:
            destination.cleanup()
            raise ValueError("Provider result reports a different release.")
    return manifest, arrays, location, destination


def _checked_source_result(
    manifest: Mapping[str, Any], source: MACESource, /
) -> dict[str, Any]:
    recorded = manifest["source"]
    if (
        not isinstance(recorded, Mapping)
        or set(recorded)
        != {
            "kind",
            "sha256",
            "byte_size",
            "dtype",
            "reproduction_deviation",
            "layout_normalization",
        }
        or recorded["kind"] != source.kind
        or recorded["sha256"] != source.artifact.sha256
        or recorded["byte_size"] != source.artifact.byte_size
        or type(recorded["reproduction_deviation"]) not in (int, float)
    ):
        raise ValueError("Provider result does not describe the admitted source.")
    parse(recorded["dtype"], MACEEvaluationDtype, "provider source dtype")
    layout = _layout_record(recorded["layout_normalization"])
    if source.kind == "torch-state-dict" and _normalized_layout(layout):
        raise ValueError("A safe state-dict source cannot require layout normalization.")
    declaration = _validated_declaration(manifest["declaration"])
    if source.architecture is not None and dict(source.architecture) != declaration:
        raise ValueError("Provider declaration differs from the declared architecture.")
    return declaration


def _selected_head(declaration: Mapping[str, Any], head: str | None, /) -> str:
    heads = declaration["heads"]
    if head is None:
        if len(heads) != 1:
            raise ValueError("Multihead MACE sources require an explicit head.")
        return heads[0]
    if head not in heads:
        raise ValueError(f"Head {head!r} is not one of the source heads {heads}.")
    return head


def _linear_normalization(record: Mapping[str, Any], name: str, /) -> None:
    irreps_in = _irreps(record["irreps_in"], f"{name}.irreps_in")
    instructions = record["instructions"]
    for instruction in instructions:
        fan_in = sum(
            irreps_in[other["i_in"]][0]
            for other in instructions
            if other["i_out"] == instruction["i_out"]
        )
        expected = 1.0 / math.sqrt(fan_in)
        if not math.isclose(
            instruction["path_weight"], expected, rel_tol=_NORMALIZATION_TOLERANCE
        ):
            raise MACESourceRefusedError(
                f"{name} uses a non-standard e3nn linear normalization."
            )


def _tensor_product_normalization(
    record: Mapping[str, Any], mode: str, name: str, /
) -> None:
    irreps_in1 = _irreps(record["irreps_in1"], f"{name}.irreps_in1")
    irreps_in2 = _irreps(record["irreps_in2"], f"{name}.irreps_in2")
    irreps_out = _irreps(record["irreps_out"], f"{name}.irreps_out")
    instructions = record["instructions"]

    def elements(instruction: Mapping[str, Any], /) -> int:
        multiplicity_in1 = irreps_in1[instruction["i_in1"]][0]
        multiplicity_in2 = irreps_in2[instruction["i_in2"]][0]
        return multiplicity_in1 * multiplicity_in2 if mode == "uvw" else multiplicity_in2

    for instruction in instructions:
        if instruction["connection_mode"] != mode or not instruction["has_weight"]:
            raise MACESourceRefusedError(
                f"{name} uses an unadmitted tensor-product path."
            )
        # e3nn "component" irrep and "element" path normalization defaults.
        fan_in = sum(
            elements(other)
            for other in instructions
            if other["i_out"] == instruction["i_out"]
        )
        degree = irreps_out[instruction["i_out"]][1]
        expected = math.sqrt((2 * degree + 1) / fan_in)
        if not math.isclose(
            instruction["path_weight"], expected, rel_tol=_NORMALIZATION_TOLERANCE
        ):
            raise MACESourceRefusedError(
                f"{name} uses a non-standard tensor-product normalization."
            )
        if mode == "uvu" and instruction["path_shape"] != [
            irreps_in1[instruction["i_in1"]][0],
            irreps_in2[instruction["i_in2"]][0],
        ]:
            raise MACESourceRefusedError(f"{name} has a non-channelwise path shape.")


def _silu_scale(activation: Any, name: str, /) -> float:
    if (
        not isinstance(activation, Mapping)
        or activation.get("function") != "silu"
        or activation.get("identity") is not False
    ):
        raise MACESourceRefusedError(f"{name} must be the normalized SiLU activation.")
    return _finite(activation["constant"], f"{name}.constant")


def _radial_scale(networks: Sequence[Mapping[str, Any]], /) -> float:
    scales = set()
    for index, network in enumerate(networks):
        layers = network["layers"]
        for position, layer in enumerate(layers):
            if layer["var_in"] != 1.0 or layer["var_out"] != 1.0:
                raise MACESourceRefusedError("Radial networks must use unit variances.")
            if position == len(layers) - 1:
                if layer["activation"] is not None:
                    raise MACESourceRefusedError("Radial outputs carry no activation.")
            else:
                scales.add(_silu_scale(layer["activation"], f"radial network {index}"))
    if len(scales) != 1:
        raise MACESourceRefusedError("Radial networks disagree on their activation.")
    return scales.pop()


def _check_density(record: Mapping[str, Any] | None, kind: str, /) -> None:
    expects_density = kind in (
        "RealAgnosticDensityInteractionBlock",
        "RealAgnosticDensityResidualInteractionBlock",
    )
    if expects_density != (record is not None):
        raise MACESourceRefusedError("Density network presence contradicts its class.")
    if record is not None:
        layers = record["layers"]
        if (
            len(layers) != 1
            or layers[0]["activation"] is not None
            or layers[0]["h_out"] != 1
            or layers[0]["var_in"] != 1.0
            or layers[0]["var_out"] != 1.0
        ):
            raise MACESourceRefusedError("Density networks must be one linear layer.")


def _admitted_modules(
    modules: Mapping[str, Any], declaration: Mapping[str, Any], /
) -> tuple[float, float | None]:
    """Refuse non-standard normalizations; return radial/readout SiLU scales."""

    _linear_normalization(modules["node_embedding"], "node_embedding")
    interactions = modules["interactions"]
    for index, interaction in enumerate(interactions):
        prefix = f"interactions.{index}"
        _linear_normalization(interaction["linear_up"], f"{prefix}.linear_up")
        _linear_normalization(interaction["linear"], f"{prefix}.linear")
        _tensor_product_normalization(interaction["conv_tp"], "uvu", f"{prefix}.conv_tp")
        _tensor_product_normalization(interaction["skip_tp"], "uvw", f"{prefix}.skip_tp")
        _check_density(interaction["density_fn"], interaction["class"])
    for index, product in enumerate(modules["products"]):
        _linear_normalization(product["linear"], f"products.{index}.linear")
    readout_scale = None
    for index, readout in enumerate(modules["readouts"]):
        match readout["class"]:
            case "LinearReadoutBlock":
                _linear_normalization(readout["linear"], f"readouts.{index}.linear")
            case "NonLinearReadoutBlock":
                _linear_normalization(readout["linear_1"], f"readouts.{index}.linear_1")
                _linear_normalization(readout["linear_2"], f"readouts.{index}.linear_2")
                readout_scale = _silu_scale(readout["activation"], "readout activation")
            case other:
                raise MACESourceRefusedError(f"Readout {other!r} is not admitted.")
    radial_scale = _radial_scale(
        [interaction["conv_tp_weights"] for interaction in interactions]
    )
    if declaration["gate"] not in (None, "silu"):
        raise MACESourceRefusedError("Only SiLU readout gates are admitted.")
    return radial_scale, readout_scale


def _hidden_layout(declaration: Mapping[str, Any], /) -> tuple[int, int]:
    hidden = _irreps(declaration["hidden_irreps"], "hidden_irreps")
    channels = hidden[0][0]
    expected = tuple((channels, degree, (-1) ** degree) for degree in range(len(hidden)))
    if hidden != expected:
        raise MACESourceRefusedError(
            "Hidden irreps must be equal-channel C x (l, (-1)^l) blocks for l <= L."
        )
    return channels, len(hidden) - 1


def _interaction_kind(name: str, /) -> MACEInteractionKind:
    match name:
        case "RealAgnosticInteractionBlock":
            return "real-agnostic"
        case "RealAgnosticResidualInteractionBlock":
            return "real-agnostic-residual"
        case "RealAgnosticDensityInteractionBlock":
            return "real-agnostic-density"
        case "RealAgnosticDensityResidualInteractionBlock":
            return "real-agnostic-density-residual"
        case _:
            raise MACESourceRefusedError(f"Interaction {name!r} is not admitted.")


def _scalar(arrays: Mapping[str, np.ndarray], name: str, /) -> float:
    value = arrays[f"state/{name}"]
    if value.shape != () or not np.isfinite(value):
        raise ValueError(f"Source scalar {name} is not one finite value.")
    return float(value)


def _architecture(
    declaration: Mapping[str, Any],
    modules: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    head: str,
    /,
) -> MACEArchitecture:
    from ...nn.atomistic._mace import MACEArchitecture

    radial_scale, readout_scale = _admitted_modules(modules, declaration)
    channels, hidden_degree = _hidden_layout(declaration)
    count = declaration["num_interactions"]
    if declaration["use_agnostic_product"] and any(
        product["use_agnostic_product"] is not True for product in modules["products"]
    ):
        raise MACESourceRefusedError("Agnostic-product flags are inconsistent.")
    agnesi = (
        (
            _scalar(arrays, "radial_embedding.distance_transform.a"),
            _scalar(arrays, "radial_embedding.distance_transform.q"),
            _scalar(arrays, "radial_embedding.distance_transform.p"),
        )
        if declaration["distance_transform"] == "Agnesi"
        else None
    )
    readout_width = None
    if declaration["MLP_irreps"] is not None:
        mlp = _irreps(declaration["MLP_irreps"], "MLP_irreps")
        if len(mlp) != 1 or mlp[0][1:] != (0, 1):
            raise MACESourceRefusedError("Readout MLP irreps must be scalars.")
        readout_width = mlp[0][0]
    return MACEArchitecture(
        species=tuple(declaration["atomic_numbers"]),
        cutoff=float(declaration["r_max"]),
        radial_basis_count=declaration["num_bessel"],
        cutoff_power=declaration["num_polynomial_cutoff"],
        distance_transform="agnesi" if agnesi is not None else "none",
        agnesi_parameters=agnesi,
        cutoff_placement="embedding" if declaration["apply_cutoff"] else "weights",
        channel_count=channels,
        hidden_degree=hidden_degree,
        edge_degree=declaration["max_ell"],
        interactions=tuple(
            _interaction_kind(name) for name in declaration["interaction_classes"]
        ),
        correlations=tuple(declaration["correlation"]),
        radial_widths=tuple(declaration["radial_MLP"]),
        readout_width=readout_width,
        radial_activation_scale=radial_scale,
        readout_activation_scale=readout_scale,
        average_neighbor_count=float(declaration["avg_num_neighbors"]),
        heads=tuple(declaration["heads"]),
        head=head,
        energy_scaling="scale-shift"
        if declaration["model_class"] == "ScaleShiftMACE"
        else "unscaled",
        pair_repulsion=declaration["pair_repulsion"],
        readout_correlation=None,
        last_readout_only=declaration["use_last_readout_only"] and count > 1,
        agnostic_product=declaration["use_agnostic_product"],
    )


def _degree_transforms(
    arrays: Mapping[str, np.ndarray], /
) -> tuple[tuple[np.ndarray, ...], float]:
    """Fit ``T_l`` with ``x_native = T_l @ x_source`` from provider evidence."""

    from ...special import RealCartesianHarmonics

    directions = arrays["basis/directions"]
    source = arrays["basis/spherical_harmonics"].astype(np.float64)
    components = source.shape[1]
    degree = math.isqrt(components) - 1
    if (degree + 1) ** 2 != components or directions.shape != (source.shape[0], 3):
        raise ValueError("Provider harmonic evidence has an inconsistent layout.")
    native = np.asarray(
        RealCartesianHarmonics(degree, normalization="fully_normalized")(directions),
        dtype=np.float64,
    )
    transforms = []
    residual = 0.0
    for order in range(degree + 1):
        span = slice(order * order, (order + 1) * (order + 1))
        # native_l = source_l @ T_l.T, solved over all probe directions.
        transposed, *_ = np.linalg.lstsq(source[:, span], native[:, span], rcond=None)
        transform = np.ascontiguousarray(transposed.T)
        fit = float(np.max(np.abs(source[:, span] @ transform.T - native[:, span])))
        orthogonality = float(
            np.max(np.abs(transform @ transform.T - np.eye(2 * order + 1)))
        )
        residual = max(residual, fit, orthogonality)
        transforms.append(transform)
    if residual > _BASIS_TOLERANCE:
        raise MACESourceRefusedError(
            "Provider harmonics are not an orthogonal transform of native harmonics."
        )
    return tuple(transforms), residual


def _couplings(
    arrays: Mapping[str, np.ndarray], /
) -> dict[tuple[int, int, int], np.ndarray]:
    """Return the coupling coefficients the provider actually executes.

    Paths whose e3nn code reads a stored ``_w3j`` buffer use that buffer's
    source-precision values (widened once); a float32 checkpoint evaluated in
    float64 by its provider uses exactly these widened values. Paths e3nn
    evaluates with analytic scalar forms carry no buffer and use the provider's
    float64 Wigner 3j table. Buffers must agree across interactions and with
    the provider table at their own precision.
    """

    couplings: dict[tuple[int, int, int], np.ndarray] = {}
    for name, value in arrays.items():
        if not name.startswith("basis/wigner_3j/"):
            continue
        degrees = tuple(
            int(part) for part in name.removeprefix("basis/wigner_3j/").split("_")
        )
        if len(degrees) != 3 or value.shape != tuple(2 * item + 1 for item in degrees):
            raise ValueError("Provider coupling evidence has an inconsistent layout.")
        couplings[(degrees[0], degrees[1], degrees[2])] = value.astype(np.float64)
    stored: dict[tuple[int, int, int], np.ndarray] = {}
    for name, value in arrays.items():
        leaf = name.rsplit(".", 1)[-1]
        if not name.startswith("state/") or not leaf.startswith("_w3j_"):
            continue
        parts = tuple(int(part) for part in leaf.removeprefix("_w3j_").split("_"))
        path = (parts[0], parts[1], parts[2])
        reference = couplings.get(path)
        if reference is None or value.shape != reference.shape:
            raise ValueError(f"Source coupling buffer {name} has no provider path.")
        tolerance = 8.0 * float(np.finfo(value.dtype).eps)
        if np.max(np.abs(value.astype(np.float64) - reference)) > tolerance:
            raise ValueError(f"Source coupling buffer {name} is not its Wigner 3j table.")
        previous = stored.get(path)
        if previous is not None and not np.array_equal(previous, value):
            raise ValueError(f"Source coupling buffers for {path} disagree.")
        stored[path] = value
    for path, value in stored.items():
        couplings[path] = value.astype(np.float64)
    return couplings


def _zeroed_target(name: str, /) -> str:
    owner, flag = name.rsplit(".", 1)
    if flag == "weights_max_zeroed":
        return f"{owner}.weights_max"
    index = flag.removeprefix("weights_").removesuffix("_zeroed")
    return f"{owner}.weights.{index}"


def _native_tensors(
    arrays: Mapping[str, np.ndarray],
    declaration: Mapping[str, Any],
    /,
) -> dict[str, np.ndarray]:
    """Verify and drop provider-derived buffers; return the native inventory."""

    state = {
        name.removeprefix("state/"): value
        for name, value in arrays.items()
        if name.startswith("state/")
    }
    tensors: dict[str, np.ndarray] = {}
    for name, value in state.items():
        leaf = name.rsplit(".", 1)[-1]
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise ValueError(f"Source tensor {name} is not finite.")
        if leaf == "output_mask":
            continue
        if leaf == "bias" or name.endswith("conv_tp.weight"):
            if value.size:
                raise MACESourceRefusedError(f"Source tensor {name} must be empty.")
            continue
        if name == "num_interactions":
            if value.shape != () or int(value) != declaration["num_interactions"]:
                raise ValueError("Source interaction count buffer is inconsistent.")
            continue
        if leaf.startswith("_w3j_"):
            # Verified and carried by ``_couplings``.
            continue
        if leaf.endswith("_zeroed"):
            target = state.get(_zeroed_target(name))
            if target is None or value.dtype != np.bool_ or value.shape != ():
                raise ValueError(f"Source zero flag {name} is inconsistent.")
            if bool(value) and np.any(target != 0):
                raise ValueError(f"Source zero flag {name} contradicts its weights.")
            continue
        tensors[name] = value
    return tensors


def _conversion_id(provenance: MACESourceProvenance, /) -> str:
    record = provenance.to_record()
    record.pop("conversion_id")
    return canonical_fingerprint({"kind": "mace-source-conversion", **record})


def convert_mace_checkpoint(
    source: MACESource,
    /,
    *,
    provider: MACEProviderRuntime,
    head: str | None = None,
    precision: AtomisticPrecisionPolicy | None = None,
    streaming: StreamedRelationPlan | None = None,
    limits: MACEConversionLimits = MACEConversionLimits(),
) -> MACECheckpointConversion:
    """Convert one admitted standard mace-torch source into a native model.

    Source energies are eV and lengths Angstrom. ``head`` selects the active
    readout head and is required for multihead sources. Non-standard module
    trees, normalizations, layouts or inconsistent buffers refuse with
    ``MACESourceRefusedError`` before any native model exists.
    """

    from ...nn.atomistic._mace_source import mace_potential_from_source
    from ...units import ANGSTROM, ELECTRONVOLT

    if not isinstance(source, MACESource):
        raise TypeError("source must be a MACESource.")
    payload = _source_payload(source, limits)
    request = {"operation": "extract", "source": _source_request(source)}
    if source.architecture is not None:
        request["architecture"] = _plain(source.architecture)
    manifest, arrays, _, directory = _run_provider(provider, request, payload, limits)
    try:
        declaration = _checked_source_result(manifest, source)
        active_head = _selected_head(declaration, head)
        architecture = _architecture(
            declaration, manifest["modules"], arrays, active_head
        )
        transforms, residual = _degree_transforms(arrays)
        couplings = _couplings(arrays)
        tensors = _native_tensors(arrays, declaration)
    finally:
        directory.cleanup()
    provenance_values = {
        "source_sha256": source.artifact.sha256,
        "source_byte_size": source.artifact.byte_size,
        "source_license_id": source.artifact.license_id,
        "source_manifest_id": source.artifact.manifest_id,
        "source_admission_id": source.artifact.admission_id,
        "source_kind": source.kind,
        "trusted_full_object": source.kind == "torch-full-model",
        "source_dtype": manifest["source"]["dtype"],
        "provider": MappingProxyType(manifest["provider"]),
        "interpreter_sha256": provider.interpreter.sha256,
        "interpreter_version": provider.interpreter.version,
        "declaration": MappingProxyType(declaration),
        "head": active_head,
        "basis_residual": residual,
        "layout_normalization": MappingProxyType(
            _layout_record(manifest["source"]["layout_normalization"])
        ),
    }
    unsigned = MACESourceProvenance(**provenance_values, conversion_id="pending")
    provenance = dataclasses.replace(unsigned, conversion_id=_conversion_id(unsigned))
    potential = mace_potential_from_source(
        AtomisticScaleContract(ANGSTROM, ELECTRONVOLT),
        architecture,
        tensors,
        degree_transforms=transforms,
        couplings=couplings,
        source_id=provenance.conversion_id,
        precision=precision,
        streaming=streaming,
        maximum_source_entries=limits.max_tensor_elements,
    )
    return MACECheckpointConversion(
        potential=potential,
        provenance=provenance,
        architecture=architecture,
        source_tensors=MappingProxyType(tensors),
        degree_transforms=transforms,
        couplings=MappingProxyType(couplings),
        precision=precision,
        streaming=streaming,
        maximum_source_entries=limits.max_tensor_elements,
    )


def _cases(
    configurations: Sequence[MACEProviderConfiguration], /
) -> list[dict[str, Any]]:
    if isinstance(configurations, MACEProviderConfiguration) or not configurations:
        raise ValueError("Provide a nonempty sequence of configurations.")
    if any(not isinstance(item, MACEProviderConfiguration) for item in configurations):
        raise TypeError("configurations must be MACEProviderConfiguration values.")
    return [item.to_request() for item in configurations]


def evaluate_mace_source(
    source: MACESource,
    configurations: Sequence[MACEProviderConfiguration],
    /,
    *,
    provider: MACEProviderRuntime,
    head: str | None = None,
    evaluation_dtype: MACEEvaluationDtype = "float64",
    limits: MACEConversionLimits = MACEConversionLimits(),
) -> MACEProviderEvaluation:
    """Evaluate the admitted source with its own provider and neighbor list.

    This is the independent source-side oracle of fidelity campaigns; it never
    touches native execution.
    """

    dtype = parse(evaluation_dtype, MACEEvaluationDtype, "evaluation_dtype")
    payload = _source_payload(source, limits)
    probe = {
        "operation": "evaluate",
        "source": _source_request(source),
        "configurations": _cases(configurations),
        "evaluation_dtype": dtype,
    }
    if source.architecture is not None:
        probe["architecture"] = _plain(source.architecture)
    declaration = dict(source.architecture) if source.architecture is not None else None
    if declaration is not None:
        probe["head"] = _selected_head(declaration, head)
    elif head is not None:
        probe["head"] = head
    else:
        raise ValueError("Provider evaluation of a full-object source names its head.")
    manifest, arrays, _, directory = _run_provider(provider, probe, payload, limits)
    try:
        recorded = _checked_source_result(manifest, source)
        _selected_head(recorded, probe["head"])
        cases = []
        for index, configuration in enumerate(configurations):
            prefix = f"cases/{index:04d}/"
            cases.append(
                MACEProviderCase(
                    energy=float(arrays[prefix + "energy"]),
                    forces=arrays[prefix + "forces"],
                    stress=arrays[prefix + "stress"] if configuration.periodic else None,
                    node_energy=arrays[prefix + "node_energy"],
                    edge_index=arrays[prefix + "edge_index"],
                    unit_shifts=arrays[prefix + "unit_shifts"],
                )
            )
    finally:
        directory.cleanup()
    evaluation = manifest["evaluation"]
    if evaluation != {"head": probe["head"], "evaluation_dtype": dtype}:
        raise ValueError("Provider evaluation metadata differs from the request.")
    return MACEProviderEvaluation(
        head=probe["head"],
        evaluation_dtype=dtype,
        cases=tuple(cases),
        provider=MappingProxyType(manifest["provider"]),
        declaration=MappingProxyType(recorded),
    )


def mace_source_gradients(
    source: MACESource,
    configurations: Sequence[MACEProviderConfiguration],
    force_weights: Sequence[np.ndarray],
    /,
    *,
    provider: MACEProviderRuntime,
    head: str,
    evaluation_dtype: MACEEvaluationDtype = "float64",
    limits: MACEConversionLimits = MACEConversionLimits(),
) -> MACEProviderGradients:
    """Return provider gradients in the original source parameter space.

    Native parameter-direction parity contracts these with source-space
    directions; a merged-polynomial artifact cannot satisfy it by construction.
    """

    dtype = parse(evaluation_dtype, MACEEvaluationDtype, "evaluation_dtype")
    weights = [np.asarray(item, dtype=np.float64) for item in force_weights]
    if len(weights) != len(configurations) or any(
        weight.shape != (len(item.numbers), 3)
        for weight, item in zip(weights, configurations, strict=True)
    ):
        raise ValueError("Every configuration requires [atoms, 3] force weights.")
    payload = _source_payload(source, limits)
    request = {
        "operation": "gradients",
        "source": _source_request(source),
        "configurations": _cases(configurations),
        "force_weights": [weight.tolist() for weight in weights],
        "evaluation_dtype": dtype,
        "head": _text(head, "head"),
    }
    if source.architecture is not None:
        request["architecture"] = _plain(source.architecture)
    manifest, arrays, _, directory = _run_provider(provider, request, payload, limits)
    try:
        _checked_source_result(manifest, source)
        evaluation = manifest["evaluation"]
        parameters = tuple(evaluation["parameters"])
        if evaluation["head"] != head or evaluation["evaluation_dtype"] != dtype:
            raise ValueError("Provider gradient metadata differs from the request.")
        energies, forces, energy_gradients, force_gradients = [], [], [], []
        for index in range(len(configurations)):
            prefix = f"cases/{index:04d}/"
            energies.append(float(arrays[prefix + "energy"]))
            forces.append(arrays[prefix + "forces"])
            energy_gradients.append(
                MappingProxyType(
                    {
                        name: arrays[f"{prefix}energy_gradient/{name}"]
                        for name in parameters
                    }
                )
            )
            force_gradients.append(
                MappingProxyType(
                    {
                        name: arrays[f"{prefix}force_gradient/{name}"]
                        for name in parameters
                    }
                )
            )
    finally:
        directory.cleanup()
    return MACEProviderGradients(
        head=head,
        evaluation_dtype=dtype,
        parameters=parameters,
        energies=tuple(energies),
        forces=tuple(forces),
        energy_gradients=tuple(energy_gradients),
        force_gradients=tuple(force_gradients),
        provider=MappingProxyType(manifest["provider"]),
    )


def _same_at_precision(realized: Any, requested: Any, dtype: np.dtype, /) -> bool:
    """Equality up to rounding of requested real values to the source dtype."""

    if isinstance(requested, Mapping):
        return (
            isinstance(realized, Mapping)
            and set(realized) == set(requested)
            and all(
                _same_at_precision(realized[key], requested[key], dtype)
                for key in requested
            )
        )
    if isinstance(requested, list):
        return (
            isinstance(realized, list)
            and len(realized) == len(requested)
            and all(
                _same_at_precision(left, right, dtype)
                for left, right in zip(realized, requested, strict=True)
            )
        )
    if type(requested) is float and type(realized) is float:
        return realized == float(np.asarray(requested, dtype=dtype))
    return type(realized) is type(requested) and realized == requested


def create_mace_provider_fixture(
    declaration: Mapping[str, Any],
    destination: str | Path,
    /,
    *,
    provider: MACEProviderRuntime,
    seed: int,
    kind: MACESourceKind = "torch-state-dict",
    dtype: MACEEvaluationDtype = "float64",
    limits: MACEConversionLimits = MACEConversionLimits(),
) -> MACEProviderFixture:
    """Build a deterministic provider-initialized source fixture.

    The provider constructs the declared standard model under ``seed`` in
    ``dtype`` and saves it with ``torch.save`` as a state dict or a full
    module. The fixture is a lawful test oracle; it does not qualify any named
    external checkpoint.
    """

    checked = _validated_declaration(declaration)
    source_kind = parse(kind, MACESourceKind, "kind")
    source_dtype = parse(dtype, MACEEvaluationDtype, "dtype")
    if type(seed) is not int:
        raise TypeError("seed must be an int.")
    target = Path(destination).expanduser().resolve(strict=True)
    if not target.is_dir():
        raise ValueError("Fixture destination must be an existing directory.")
    manifest, _, location, directory = _run_provider(
        provider,
        {
            "operation": "fixture",
            "architecture": checked,
            "seed": seed,
            "source_kind": source_kind,
            "source_dtype": source_dtype,
        },
        None,
        limits,
        extra_outputs=(("fixture.pt", limits.max_source_bytes),),
    )
    try:
        record = manifest["fixture"]
        payload = location.with_name("fixture.pt").read_bytes()
        if (
            record["seed"] != seed
            or record["source_kind"] != source_kind
            or record["source_dtype"] != source_dtype
            or record["fixture_byte_size"] != len(payload)
            or record["fixture_sha256"] != hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError("Provider fixture record differs from its bytes.")
        declared = _validated_declaration(record["declaration"])
        if not _same_at_precision(declared, checked, np.dtype(source_dtype)):
            raise ValueError(
                "Provider fixture does not realize the requested declaration."
            )
        published = target / f"mace-fixture-{record['fixture_sha256'][:16]}.pt"
        with published.open("xb") as stream:
            stream.write(payload)
    finally:
        directory.cleanup()
    return MACEProviderFixture(
        path=published,
        kind=source_kind,
        dtype=source_dtype,
        sha256=record["fixture_sha256"],
        byte_size=len(payload),
        seed=seed,
        declaration=MappingProxyType(declared),
        provider=MappingProxyType(manifest["provider"]),
    )


__all__ = [
    "convert_mace_checkpoint",
    "create_mace_provider_fixture",
    "evaluate_mace_source",
    "MACE_PROVIDER_RELEASES",
    "mace_source_gradients",
    "MACECheckpointConversion",
    "MACEConversionLimits",
    "MACEEvaluationDtype",
    "MACEProviderCase",
    "MACEProviderConfiguration",
    "MACEProviderEvaluation",
    "MACEProviderFixture",
    "MACEProviderGradients",
    "MACEProviderRuntime",
    "MACESource",
    "MACESourceKind",
    "MACESourceProvenance",
    "MACESourceRefusedError",
    "TrustedTorchPickleSource",
]
