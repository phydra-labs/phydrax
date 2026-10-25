#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Admitted AIMNetcentral 0.2.0 sources converted into native AIMNet2 models.

Three source formats are admitted, each through the canonical external
artifact policy (exact SHA-256, size and caller-declared license):

* ``"hf-safetensors"``: a Hugging Face ``config.json`` (with ``model_yaml``)
  plus one ``ensemble_<k>.safetensors``. Both are parsed here, pickle-free,
  with explicit byte, member and element bounds. A ``config.json`` without
  ``model_yaml`` refuses: the source's fallback that downloads and unpickles a
  registry ``.pt`` with ``weights_only=False`` is never reproduced.
* ``"torch-v2-state"``: the source's ``torch.save`` dict (``model_yaml``,
  metadata and ``state_dict``). The ZIP container and every record are
  admitted first; ``data.pkl`` is decoded with ``pickletools`` and interpreted
  symbolically (only plain containers, scalars, ``OrderedDict`` and
  ``_rebuild_tensor_v2`` over typed storages); nothing is unpickled and every
  tensor is a bounded strided view of its admitted storage record.
* ``"torch-legacy-jit"``: a TorchScript ``.jpt`` is executable. It is loaded
  only by the isolated provider (``_aimnet2_worker.py``) under an explicit
  ``TrustedTorchPickleSource`` naming its digest, with the original model YAML
  supplied as an admitted artifact; the source's documented ``load_v1_model``
  rule then externalizes ``LRCoulomb``/``DFTD3`` and embeds ``SRCoulomb``.
  There is no fallback from a refused safe format to executable loading.

Conversion never claims weight rights or availability: provenance records the
caller's admitted license identifiers and digests only. Product inference and
training are native; the provider is used only for legacy extraction and for
optional isolated source-oracle evaluation.
"""

from __future__ import annotations

import dataclasses
import io
import json
import math
import pickletools
import struct
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

import numpy as np
import yaml

from ..._array_archive import ArrayArchiveLimits, read_array_archive, write_array_archive
from ..._artifact_security import (
    AdmittedExternalArtifact,
    ExternalArtifactPolicy,
    read_admitted_artifact,
)
from ..._external_resource import ResourceLimits
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
from ...nn.atomistic._aimnet2 import AIMNet2OutputKind
from ...nn.atomistic._aimnet2_source import AIMNet2SourceRefusedError
from ...typing import parse
from .._types import AtomisticPrecisionPolicy, AtomisticScaleContract
from ._source_admission import TrustedTorchPickleSource


if TYPE_CHECKING:
    from ...nn.atomistic._aimnet2 import AIMNet2Potential
    from ...nn.atomistic._aimnet2_source import (
        AIMNet2CalculatorComposition,
        AIMNet2SourceReconstruction,
    )


AIMNet2SourceFormat: TypeAlias = Literal[
    "hf-safetensors", "torch-v2-state", "torch-legacy-jit"
]
# Source release whose code defines this conversion contract.
AIMNET2_SOURCE_PIN: Mapping[str, str] = MappingProxyType(
    {
        "repository": "https://github.com/isayevlab/aimnetcentral",
        "commit": "b1b0fd8c8b8d8acb8811eb646f9ba7b15d72da55",
        "release": "0.2.0",
    }
)
# Distributions the optional source oracle verifies in its provider.
AIMNET2_PROVIDER_RELEASES: Mapping[str, str] = MappingProxyType({"aimnet": "0.2.0"})

_WORKER_NAME = "_aimnet2_worker.py"
_REFUSED_EXIT = 3
_RESULT_FORMAT = "phydrax-aimnet2-provider-result"
_BOOTSTRAP = (
    "import runpy, site, sys; site.addsitedir(sys.argv[1]); "
    "sys.argv = [sys.argv[2], sys.argv[1], *sys.argv[3:]]; "
    "runpy.run_path(sys.argv[0], run_name='__main__')"
)
_ZIP_END = struct.Struct("<4s4H2LH")
_ZIP64_LOCATOR = struct.Struct("<4sLQL")
_ZIP64_END = struct.Struct("<4sQ2H2L4Q")
_TORCH_METADATA_RECORDS = frozenset(
    {
        "byteorder",
        "version",
        ".data/version",
        ".format_version",
        ".storage_alignment",
        ".data/serialization_id",
    }
)
_STORAGES = {
    "DoubleStorage": "<f8",
    "FloatStorage": "<f4",
    "LongStorage": "<i8",
    "IntStorage": "<i4",
    "ShortStorage": "<i2",
    "CharStorage": "<i1",
    "ByteStorage": "<u1",
    "BoolStorage": "|b1",
}
_SAFETENSORS_DTYPES = {
    "F64": "<f8",
    "F32": "<f4",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "<i1",
    "U8": "<u1",
    "BOOL": "|b1",
}
# Source metadata fields (``ModelMetadata`` and the HF family config).
_METADATA_FIELDS = frozenset(
    {
        "format_version",
        "cutoff",
        "needs_coulomb",
        "needs_dispersion",
        "coulomb_mode",
        "coulomb_sr_rc",
        "coulomb_sr_envelope",
        "d3_params",
        "has_embedded_lr",
        "implemented_species",
        "family",
        "supports_charged_systems",
        "has_embedded_d3ts",
    }
)
_HF_FIELDS = frozenset(
    {
        "config_schema_version",
        "family_name",
        "ensemble_size",
        "member_names",
        "library_name",
        "tags",
        "model_yaml",
    }
)


def _refuse(message: str, /) -> AIMNet2SourceRefusedError:
    return AIMNet2SourceRefusedError(message)


# Default structured-metadata budget: expanded scalar text bytes, nesting
# depth, expanded container/scalar nodes and mapping entries (losses unused).
AIMNET2_METADATA_LIMITS = ResourceLimits(
    max_bytes=16_777_216,
    max_depth=64,
    max_nodes=1_048_576,
    max_attributes=1_048_576,
    max_losses=0,
)


@dataclass(frozen=True, slots=True)
class AIMNet2ExtractionLimits:
    """Byte, member, element, metadata and time bounds of one extraction run.

    Containers are admitted from their directory/header before any record is
    read: at most ``max_members`` records or tensors expanding to at most
    ``max_source_bytes``; each tensor within ``max_tensor_elements`` and
    ``max_tensor_bytes``; ``data.pkl``, JSON/YAML declarations, safetensors
    headers and ZIP directories within ``max_manifest_bytes``. Every tensor
    declaration of a source, and their aggregate logical bytes, are admitted
    before the first tensor is copied. Decoded metadata (JSON, YAML, pickled
    plain values) is expanded under ``metadata``: cycles and YAML aliases
    refuse, and depth, expanded nodes, mapping entries and scalar text bytes
    are charged before anything is constructed. Provider runs are additionally
    bounded by ``timeout_seconds``, ``max_log_bytes`` and ``max_result_bytes``;
    the provider is not a memory sandbox.
    """

    max_source_bytes: int = 1_073_741_824
    max_result_bytes: int = 2_147_483_648
    max_tensor_bytes: int = 536_870_912
    max_tensor_elements: int = 134_217_728
    max_members: int = 4_097
    max_manifest_bytes: int = 16_777_216
    max_log_bytes: int = 16_777_216
    timeout_seconds: float = 1_800.0
    metadata: ResourceLimits = AIMNET2_METADATA_LIMITS

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
            raise ValueError("AIMNet2 extraction limits must be positive integers.")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive and finite.")
        if not isinstance(self.metadata, ResourceLimits):
            raise TypeError("metadata must be ResourceLimits.")

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

    def worker_record(self) -> dict[str, int]:
        archive = self.archive_limits()
        return {
            "max_source_bytes": self.max_source_bytes,
            "max_result_bytes": self.max_result_bytes,
            "max_tensor_bytes": self.max_tensor_bytes,
            "max_tensor_elements": self.max_tensor_elements,
            "max_members": self.max_members,
            "max_manifest_bytes": self.max_manifest_bytes,
            "max_array_rank": archive.max_array_rank,
            "max_total_elements": archive.max_total_array_elements,
        }


@dataclass(frozen=True, slots=True)
class AIMNet2ProviderRuntime:
    """Caller-pinned provider interpreter and its exact ``site-packages``.

    ``torch_version`` is verified by the provider for legacy TorchScript
    extraction. Oracle evaluation additionally verifies every installed
    ``aimnet`` file against its RECORD digests at ``AIMNET2_PROVIDER_RELEASES``.
    """

    interpreter: PinnedExecutable
    site_packages: str
    torch_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.interpreter, PinnedExecutable):
            raise TypeError("interpreter must be a PinnedExecutable.")
        location = Path(self.site_packages).expanduser().resolve(strict=True)
        if not location.is_dir():
            raise ValueError("site_packages must be an existing directory.")
        if not isinstance(self.torch_version, str) or not self.torch_version.strip():
            raise ValueError("torch_version must be an explicit release.")
        object.__setattr__(self, "site_packages", str(location))


@dataclass(frozen=True, slots=True)
class AIMNet2LegacyOverrides:
    """The source ``load_v1_model`` keyword declarations for a legacy model."""

    implemented_species: tuple[int, ...] | None = None
    family: str | None = None
    supports_charged_systems: bool | None = None

    def __post_init__(self) -> None:
        if self.implemented_species is not None:
            species = tuple(sorted({int(value) for value in self.implemented_species}))
            if not species or species[0] <= 0:
                raise ValueError("implemented_species must be positive atomic numbers.")
            object.__setattr__(self, "implemented_species", species)
        if self.family is not None and (
            not isinstance(self.family, str) or not self.family.strip()
        ):
            raise ValueError("family must be nonempty text or None.")
        if self.supports_charged_systems is not None and not isinstance(
            self.supports_charged_systems, bool
        ):
            raise TypeError("supports_charged_systems must be a boolean or None.")


@dataclass(frozen=True, slots=True)
class AIMNet2Source:
    """One admitted AIMNet2 source: weights plus its declaration artifact.

    ``declaration`` is the HF ``config.json`` (``"hf-safetensors"``) or the
    original model YAML (``"torch-legacy-jit"``); the self-describing
    ``"torch-v2-state"`` takes none. Executable ``"torch-legacy-jit"`` requires
    ``trust`` naming the exact admitted weights digest.
    """

    artifact: AdmittedExternalArtifact
    manifest: ArtifactManifest
    policy: ExternalArtifactPolicy
    format: AIMNet2SourceFormat
    declaration: AdmittedExternalArtifact | None = None
    declaration_manifest: ArtifactManifest | None = None
    trust: TrustedTorchPickleSource | None = None
    legacy: AIMNet2LegacyOverrides | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, AdmittedExternalArtifact):
            raise TypeError("artifact must be an AdmittedExternalArtifact.")
        if not isinstance(self.manifest, ArtifactManifest):
            raise TypeError("manifest must be an ArtifactManifest.")
        if not isinstance(self.policy, ExternalArtifactPolicy):
            raise TypeError("policy must be an ExternalArtifactPolicy.")
        source_format = parse(self.format, AIMNet2SourceFormat, "format")
        needs_declaration = source_format != "torch-v2-state"
        if needs_declaration != (self.declaration is not None) or (
            (self.declaration is None) != (self.declaration_manifest is None)
        ):
            raise ValueError(
                f"{source_format} sources "
                + ("require" if needs_declaration else "take no")
                + " declaration artifact and manifest."
            )
        if self.declaration is not None and not isinstance(
            self.declaration, AdmittedExternalArtifact
        ):
            raise TypeError("declaration must be an AdmittedExternalArtifact.")
        if source_format == "torch-legacy-jit":
            if not isinstance(self.trust, TrustedTorchPickleSource):
                raise PermissionError(
                    "Legacy TorchScript sources are executable; they require an "
                    "explicit TrustedTorchPickleSource before loading."
                )
            if self.trust.sha256 != self.artifact.sha256:
                raise PermissionError("The trust capability names a different digest.")
        elif self.trust is not None:
            raise ValueError("Safe AIMNet2 formats are never execution-trusted.")
        if self.legacy is not None and (
            source_format != "torch-legacy-jit"
            or not isinstance(self.legacy, AIMNet2LegacyOverrides)
        ):
            raise ValueError("Legacy overrides apply only to legacy TorchScript sources.")
        object.__setattr__(self, "format", source_format)


@dataclass(frozen=True, slots=True)
class AIMNet2SourceRecord:
    """Pickle-free source content: exact YAML, metadata and state tensors.

    ``declaration`` is the source model exactly as declared (embedded
    long-range modules included); ``legacy_attributes`` are the TorchScript
    attributes the source ``load_v1_model`` reads, for legacy sources only.
    """

    format: AIMNet2SourceFormat
    model_yaml: str
    declaration: Mapping[str, Any]
    metadata: Mapping[str, Any]
    tensors: Mapping[str, np.ndarray]
    legacy_attributes: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class AIMNet2SourceProvenance:
    """Origin, trust and transform identity of one conversion (not a schema)."""

    source_sha256: str
    source_byte_size: int
    source_license_id: str
    source_manifest_id: str
    source_admission_id: str
    declaration_sha256: str | None
    declaration_license_id: str | None
    source_format: AIMNet2SourceFormat
    trusted_executable: bool
    source_pin: Mapping[str, str]
    provider: Mapping[str, Any] | None
    metadata: Mapping[str, Any]
    model_yaml: str
    legacy_attributes: Mapping[str, Any] | None
    externalized: Mapping[str, Any]
    composition: Mapping[str, Any]
    reconstruction: Mapping[str, Any]
    conversion_id: str

    def to_record(self) -> dict[str, Any]:
        return {
            field.name: _plain(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any], /) -> AIMNet2SourceProvenance:
        """Parse one serialized record exactly and verify its conversion identity.

        Every field must be present and no other; digests, sizes, texts, the
        source format and its trust flag, the pinned source release, the
        provider and legacy records (present exactly for executable legacy
        sources) and the plain mapping sections are checked, and the canonical
        ``conversion_id`` is recomputed from the parsed values.
        """
        names = [field.name for field in dataclasses.fields(cls)]
        if not isinstance(record, Mapping) or set(record) != set(names):
            raise ValueError("AIMNet2 source provenance fields are not canonical.")
        source_format = parse(
            record["source_format"], AIMNet2SourceFormat, "source_format"
        )
        legacy = source_format == "torch-legacy-jit"
        declaration_sha256 = record["declaration_sha256"]
        declaration_license = record["declaration_license_id"]
        if (declaration_sha256 is None) != (declaration_license is None) or (
            (declaration_sha256 is None) != (source_format == "torch-v2-state")
        ):
            raise ValueError(
                "Declaration digest and license are present exactly for sources with "
                "a declaration artifact."
            )
        if record["trusted_executable"] is not legacy:
            raise ValueError("trusted_executable must mark exactly legacy TorchScript.")
        if (record["provider"] is None) == legacy or (
            record["legacy_attributes"] is None
        ) == legacy:
            raise ValueError(
                "Provider and legacy attribute records exist exactly for legacy "
                "TorchScript sources."
            )
        pin = _record_mapping(record["source_pin"], "source_pin")
        if pin != dict(AIMNET2_SOURCE_PIN):
            raise ValueError("The provenance names another AIMNet2 source release.")
        size = record["source_byte_size"]
        if type(size) is not int or size < 0:
            raise ValueError("source_byte_size must be a nonnegative integer.")
        parsed = cls(
            source_sha256=_record_digest(record["source_sha256"], "source_sha256"),
            source_byte_size=size,
            source_license_id=_record_text(
                record["source_license_id"], "source_license_id"
            ),
            source_manifest_id=_record_text(
                record["source_manifest_id"], "source_manifest_id"
            ),
            source_admission_id=_record_text(
                record["source_admission_id"], "source_admission_id"
            ),
            declaration_sha256=None
            if declaration_sha256 is None
            else _record_digest(declaration_sha256, "declaration_sha256"),
            declaration_license_id=None
            if declaration_license is None
            else _record_text(declaration_license, "declaration_license_id"),
            source_format=source_format,
            trusted_executable=legacy,
            source_pin=MappingProxyType(pin),
            provider=None
            if record["provider"] is None
            else MappingProxyType(_record_mapping(record["provider"], "provider")),
            metadata=MappingProxyType(_record_mapping(record["metadata"], "metadata")),
            model_yaml=_record_text(record["model_yaml"], "model_yaml"),
            legacy_attributes=None
            if record["legacy_attributes"] is None
            else MappingProxyType(
                _record_mapping(record["legacy_attributes"], "legacy_attributes")
            ),
            externalized=MappingProxyType(
                _record_mapping(record["externalized"], "externalized")
            ),
            composition=MappingProxyType(
                _record_mapping(record["composition"], "composition")
            ),
            reconstruction=MappingProxyType(
                _record_mapping(record["reconstruction"], "reconstruction")
            ),
            conversion_id=_record_text(record["conversion_id"], "conversion_id"),
        )
        if parsed.conversion_id != _conversion_id(parsed):
            raise ValueError("AIMNet2 source provenance identity is not canonical.")
        return parsed


def _record_text(value: Any, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be canonical nonempty text.")
    return value


def _record_digest(value: Any, name: str, /) -> str:
    text = _record_text(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return text


def _record_mapping(value: Any, name: str, /) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with text keys.")
    return _plain(value)


@dataclass(frozen=True, slots=True)
class AIMNet2CheckpointConversion:
    """A native model, its calculator composition and its exact provenance.

    ``potential`` evaluates the source model energy only; ``composition``
    names the separately composed calculator terms the source would add.
    """

    potential: AIMNet2Potential
    composition: AIMNet2CalculatorComposition
    reconstruction: AIMNet2SourceReconstruction
    provenance: AIMNet2SourceProvenance


def _plain(value: Any, /) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _artifact_bytes(
    artifact: AdmittedExternalArtifact,
    manifest: ArtifactManifest,
    policy: ExternalArtifactPolicy,
    maximum: int,
    /,
) -> bytes:
    if manifest.byte_size > maximum:
        raise _refuse("The AIMNet2 source exceeds its declared byte limit.")
    return read_admitted_artifact(artifact, manifest, policy=policy)


def _unique_object(pairs: list[tuple[str, Any]], /) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _refuse(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


# ``int.__repr__`` (what JSON encoding uses) is admitted below CPython's
# default 4300-digit conversion limit; 14000 bits is at most 4215 digits.
_JSON_INTEGER_BITS = 14_000
_JSON_TEXT_CHUNK = 1 << 20


def _json_text_bytes(text: str, remaining: int, /) -> int:
    """Exact ``canonical_json`` byte length of ``text`` (with quotes).

    ``canonical_json`` uses ``ensure_ascii=True``: printable ASCII other than
    ``"`` and ``\\`` is one byte; ``"``, ``\\``, ``\\b\\f\\n\\r\\t`` are two;
    every other code point below U+10000 (C0 controls, DEL, non-ASCII and lone
    surrogates) is a six-byte ``\\uXXXX`` escape and supplementary code points
    are a twelve-byte surrogate-pair escape. Counted in bounded chunks without
    building the encoding; once the count exceeds ``remaining`` it returns
    early (the caller refuses).
    """
    if len(text) + 2 > remaining:
        return len(text) + 2
    total = 2
    for start in range(0, len(text), _JSON_TEXT_CHUNK):
        codes = np.frombuffer(
            text[start : start + _JSON_TEXT_CHUNK].encode("utf-32-le", "surrogatepass"),
            dtype="<u4",
        )
        cost = np.where(codes >= 0x10000, 12, 6)
        cost = np.where((codes >= 0x20) & (codes <= 0x7E), 1, cost)
        two = np.isin(codes, (0x22, 0x5C, 0x08, 0x09, 0x0A, 0x0C, 0x0D))
        cost = np.where(two, 2, cost)
        total += int(np.sum(cost, dtype=np.int64))
        if total > remaining:
            return total
    return total


# Keep expansion, cycle detection and all four resource charges in one traversal:
# separating the nested container cases can allocate before charging a hostile node.
def _bounded_plain(value: Any, name: str, budget: ResourceLimits, /) -> Any:
    """Expand decoded metadata into plain data under ``budget``, iteratively.

    Shared references (pickle memo, DAGs) are expanded and charged on every
    occurrence, because the result is an expanded tree; a reference to a
    container on the current path is a cycle and refuses. Only exact
    ``dict``/``list`` containers, mappings whose keys are all text or all
    integers (what ``canonical_json`` can serialize), and ``None``/``bool``/
    ``int``/finite ``float``/``str`` scalars are admitted.

    Before a node is constructed it is charged against ``budget``: one node
    (``max_nodes``), its mapping entries (``max_attributes``), its nesting
    (``max_depth``) and its exact ``canonical_json`` encoded length
    (``max_bytes``): brackets, commas and colons, keys (integer keys as quoted
    JSON keys), ``null``/``true``/``false``, ``repr`` of floats and integers,
    and quoted, escaped text (``_json_text_bytes``). The charged total is the
    length of ``canonical_json`` of the expanded value.
    """
    charged = [0, 0, 0]  # nodes, mapping entries, encoded bytes
    text_sizes: dict[int, int] = {}

    def charge(nodes: int, entries: int, size: int) -> None:
        charged[0] += nodes
        charged[1] += entries
        charged[2] += size
        if charged[0] > budget.max_nodes:
            raise _refuse(f"{name} expands beyond the metadata node budget.")
        if charged[1] > budget.max_attributes:
            raise _refuse(f"{name} expands beyond the metadata entry budget.")
        if charged[2] > budget.max_bytes:
            raise _refuse(f"{name} expands beyond the metadata byte budget.")

    def integer_size(item: int) -> int:
        if item.bit_length() > _JSON_INTEGER_BITS:
            raise _refuse(f"{name} contains an integer too large to serialize.")
        return len(repr(item))

    def text_size(item: str) -> int:
        # Shared text objects are measured once; every occurrence is charged.
        known = text_sizes.get(id(item))
        if known is not None:
            return known
        size = _json_text_bytes(item, budget.max_bytes - charged[2])
        text_sizes[id(item)] = size
        return size

    def scalar_size(item: Any) -> int:
        if item is None or item is True:
            return 4
        if item is False:
            return 5
        if type(item) is int:
            return integer_size(item)
        if type(item) is float:
            if not math.isfinite(item):
                raise _refuse(f"{name} contains a nonfinite number.")
            return len(repr(item))
        if type(item) is str:
            return text_size(item)
        raise _refuse(f"{name} is not plain data ({type(item).__name__}).")

    active: set[int] = set()
    stack: list[tuple[Any, Any, Any, int]] = []
    root: list[Any] = [None]

    def open_(item: Any, depth: int, assign: Any) -> None:
        if type(item) is dict or type(item) is list:
            if depth > budget.max_depth:
                raise _refuse(f"{name} nests beyond the metadata depth budget.")
            if id(item) in active:
                raise _refuse(f"{name} contains a reference cycle.")
            mapping = type(item) is dict
            count = len(item)
            # Brackets, commas between entries and one colon per entry.
            charge(
                1,
                count if mapping else 0,
                2 + max(count - 1, 0) + (count if mapping else 0),
            )
            if mapping and len({type(key) for key in item}) > 1:
                raise _refuse(f"{name} mixes key types; it has no canonical encoding.")
            out: Any = {} if mapping else []
            assign(out)
            active.add(id(item))
            stack.append((item, out, iter(item.items() if mapping else item), depth))
            return
        charge(1, 0, scalar_size(item))
        assign(item)

    open_(value, 1, lambda item: root.__setitem__(0, item))
    while stack:
        source, out, items, depth = stack[-1]
        try:
            entry = next(items)
        except StopIteration:
            stack.pop()
            active.discard(id(source))
            continue
        if type(out) is dict:
            key, item = entry
            if type(key) is str:
                charge(0, 0, text_size(key))
            elif type(key) is int:
                charge(0, 0, integer_size(key) + 2)
            else:
                raise _refuse(f"{name} has a non-text, non-integer key.")
            open_(
                item,
                depth + 1,
                lambda child, out=out, key=key: out.__setitem__(key, child),
            )
        else:
            open_(entry, depth + 1, out.append)
    return root[0]


def _json(data: bytes, limits: AIMNet2ExtractionLimits, name: str, /) -> dict[str, Any]:
    if len(data) > limits.max_manifest_bytes:
        raise _refuse(f"{name} exceeds the manifest byte limit.")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _refuse(f"{name} is not UTF-8 JSON.") from error
    except RecursionError as error:
        raise _refuse(f"{name} nests beyond the decoder's depth.") from error
    value = _bounded_plain(value, name, limits.metadata)
    if not isinstance(value, dict):
        raise _refuse(f"{name} must be a JSON object.")
    return value


def _model_declaration(text: Any, limits: AIMNet2ExtractionLimits, /) -> dict[str, Any]:
    """Safe YAML without anchors/aliases, depth-checked before composition."""
    if not isinstance(text, str) or not text.strip():
        raise _refuse("The source model_yaml must be nonempty text.")
    if len(text.encode("utf-8", "surrogatepass")) > limits.max_manifest_bytes:
        raise _refuse("model_yaml exceeds the manifest byte limit.")
    budget = limits.metadata
    try:
        depth = 0
        for event in yaml.parse(text, Loader=yaml.SafeLoader):
            if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None):
                raise _refuse("model_yaml anchors and aliases are not admitted.")
            if isinstance(event, yaml.CollectionStartEvent):
                depth += 1
                if depth > budget.max_depth:
                    raise _refuse("model_yaml nests beyond the metadata depth budget.")
            elif isinstance(event, yaml.CollectionEndEvent):
                depth -= 1
        value = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise _refuse("model_yaml is not safe YAML.") from error
    value = _bounded_plain(value, "model_yaml", budget)
    if not isinstance(value, dict):
        raise _refuse("model_yaml must declare one model mapping.")
    return value


def _source_metadata(values: Mapping[str, Any], /) -> dict[str, Any]:
    metadata = {name: values[name] for name in _METADATA_FIELDS if name in values}
    if "cutoff" not in metadata:
        raise _refuse("Source metadata must declare the model cutoff.")
    return metadata


def read_safetensors_tensors(
    data: bytes, limits: AIMNet2ExtractionLimits, /
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Admit and read one safetensors buffer without any provider.

    Two phases: the header must be a unique-key JSON object within
    ``max_manifest_bytes`` and the metadata budget; every declaration must
    have an admitted full-precision dtype, a nonnegative shape within the
    element and byte limits and an exact byte range, and all ranges must tile
    the payload without overlap. Only then is any tensor copied.
    """
    if len(data) < 8:
        raise _refuse("The safetensors buffer has no header.")
    if len(data) > limits.max_source_bytes:
        raise _refuse("The safetensors buffer exceeds the source byte limit.")
    (length,) = struct.unpack_from("<Q", data, 0)
    if length > limits.max_manifest_bytes or 8 + length > len(data):
        raise _refuse("The safetensors header exceeds its bounds.")
    header = _json(data[8 : 8 + length], limits, "safetensors header")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise _refuse("safetensors __metadata__ must map text to text.")
    if len(header) > limits.max_members:
        raise _refuse("The safetensors buffer declares too many tensors.")
    payload = memoryview(data)[8 + length :]
    rank = limits.archive_limits().max_array_rank
    declarations: list[tuple[str, np.dtype[Any], tuple[int, ...], int, int]] = []
    for name, entry in header.items():
        if not isinstance(entry, dict) or set(entry) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            raise _refuse(f"safetensors entry {name!r} is malformed.")
        code = entry["dtype"]
        if code not in _SAFETENSORS_DTYPES:
            raise _refuse(f"safetensors dtype {code!r} of {name!r} is not admitted.")
        dtype = np.dtype(_SAFETENSORS_DTYPES[code])
        shape = entry["shape"]
        offsets = entry["data_offsets"]
        if (
            not isinstance(shape, list)
            or len(shape) > rank
            or any(type(extent) is not int or extent < 0 for extent in shape)
            or any(extent > limits.max_tensor_elements for extent in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
        ):
            raise _refuse(f"safetensors entry {name!r} has malformed extents.")
        elements = math.prod(shape)
        size = elements * dtype.itemsize
        if elements > limits.max_tensor_elements or size > limits.max_tensor_bytes:
            raise _refuse(f"safetensors tensor {name!r} exceeds the tensor limits.")
        begin, end = offsets
        if end - begin != size or end > len(payload):
            raise _refuse(f"safetensors tensor {name!r} has an inexact byte range.")
        declarations.append((name, dtype, tuple(shape), begin, end))
    position = 0
    for _, _, _, begin, end in sorted(declarations, key=lambda item: (item[3], item[4])):
        if begin != position:
            raise _refuse("safetensors byte ranges overlap or leave holes.")
        position = end
    if position != len(payload):
        raise _refuse("safetensors payload has trailing bytes.")
    # Admitted: the ranges tile the payload, so copies total at most its size.
    tensors = {
        name: (
            np.frombuffer(payload, dtype=dtype, count=math.prod(shape), offset=begin)
            .reshape(shape)
            .astype(dtype.newbyteorder("="), copy=True)
            if math.prod(shape)
            else np.empty(shape, dtype=dtype.newbyteorder("="))
        )
        for name, dtype, shape, begin, _ in declarations
    }
    return tensors, dict(metadata)


# The ZIP/ZIP64 grammar stays explicit in read order so no member is opened before
# directory-wide resource and canonical-layout admission has completed.
def _torch_container(
    data: bytes, limits: AIMNet2ExtractionLimits, /
) -> tuple[zipfile.ZipFile, str]:
    """Admit the ``torch.save`` ZIP directory and records before reading any."""
    end = len(data) - _ZIP_END.size
    if (
        not data.startswith(b"PK\x03\x04")
        or end < 0
        or data[end : end + 4] != b"PK\x05\x06"
    ):
        raise _refuse("The source is not the ZIP container torch.save writes.")
    if len(data) > limits.max_source_bytes:
        raise _refuse("The torch source exceeds the source byte limit.")
    _, disk, directory_disk, disk_entries, entries, size, offset, comment = (
        _ZIP_END.unpack_from(data, end)
    )
    if disk or directory_disk or disk_entries != entries or comment:
        raise _refuse("The torch source ZIP directory is not canonical.")
    directory_end = end
    locator = end - _ZIP64_LOCATOR.size
    if locator >= _ZIP64_END.size and data[locator : locator + 4] == b"PK\x06\x07":
        _, locator_disk, record, disks = _ZIP64_LOCATOR.unpack_from(data, locator)
        if locator_disk or disks != 1 or record != locator - _ZIP64_END.size:
            raise _refuse("The torch source ZIP64 locator is not canonical.")
        fields = _ZIP64_END.unpack_from(data, record)
        signature, _, _, _, disk64, directory64, disk_entries64, entries64 = fields[:8]
        size64, offset64 = fields[8], fields[9]
        if (
            signature != b"PK\x06\x06"
            or disk64
            or directory64
            or disk_entries64 != entries64
            or entries not in (entries64, 0xFFFF)
            or size not in (size64, 0xFFFFFFFF)
            or offset not in (offset64, 0xFFFFFFFF)
        ):
            raise _refuse("The torch source ZIP64 directory is inconsistent.")
        entries, size, offset, directory_end = entries64, size64, offset64, record
    elif entries == 0xFFFF or 0xFFFFFFFF in (size, offset):
        raise _refuse("The torch source ZIP64 directory is missing.")
    if entries > limits.max_members or size > limits.max_manifest_bytes:
        raise _refuse("The torch source directory exceeds its member or byte limits.")
    if offset + size != directory_end:
        raise _refuse("The torch source ZIP directory is not where its end record says.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise _refuse("The torch source ZIP directory is malformed.") from error
    members = archive.infolist()
    names = [member.filename for member in members]
    prefix = names[0].split("/", 1)[0] + "/" if names and "/" in names[0] else ""
    try:
        if (
            len(members) != entries
            or not prefix
            or len(set(names)) != len(names)
            or any(not name.startswith(prefix) for name in names)
            or prefix + "data.pkl" not in names
        ):
            raise _refuse("The torch source records are not one torch.save archive.")
        expanded = 0
        for member in members:
            record = member.filename.removeprefix(prefix)
            if (
                member.compress_type != zipfile.ZIP_STORED
                or member.compress_size != member.file_size
                or member.flag_bits & 0x1
            ):
                raise _refuse("Torch source records must be stored and unencrypted.")
            key = record.removeprefix("data/")
            if key != record and key and "/" not in key:
                maximum = limits.max_tensor_bytes
            elif record == "data.pkl" or record in _TORCH_METADATA_RECORDS:
                maximum = limits.max_manifest_bytes
            else:
                raise _refuse(
                    f"Torch record {record!r} is outside the torch.save layout."
                )
            if member.file_size > maximum:
                raise _refuse(f"Torch record {record!r} exceeds its byte limit.")
            expanded += member.file_size
        if expanded > limits.max_source_bytes:
            raise _refuse("Torch records expand beyond the source byte limit.")
    except AIMNet2SourceRefusedError:
        archive.close()
        raise
    return archive, prefix


@dataclass(frozen=True, slots=True)
class _Global:
    module: str
    name: str


class _OrderedMapping(dict):
    """A pickled ``collections.OrderedDict``, reconstructed only as data."""


@dataclass(frozen=True, slots=True)
class _Storage:
    key: str
    dtype: str
    numel: int


@dataclass(frozen=True, slots=True)
class _Tensor:
    storage: _Storage
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def _global(module: Any, name: Any, /) -> _Global:
    admitted = (module, name) in (
        ("collections", "OrderedDict"),
        ("torch._utils", "_rebuild_tensor_v2"),
    ) or (module == "torch" and name in _STORAGES)
    if not admitted:
        raise _refuse(f"The source pickle names {module}.{name}; refused.")
    return _Global(module, name)


def _extents(value: Any, /) -> bool:
    return isinstance(value, tuple) and all(
        type(extent) is int and extent >= 0 for extent in value
    )


def _call(function: Any, arguments: Any, /) -> Any:
    if not isinstance(function, _Global) or not isinstance(arguments, tuple):
        raise _refuse("The source pickle reduces a non-global.")
    if (function.module, function.name) == (
        "collections",
        "OrderedDict",
    ) and not arguments:
        return _OrderedMapping()
    if (function.module, function.name) == ("torch._utils", "_rebuild_tensor_v2") and (
        len(arguments) == 6
    ):
        storage, offset, shape, stride, requires_grad, hooks = arguments
        if (
            not isinstance(storage, _Storage)
            or type(offset) is not int
            or offset < 0
            or not _extents(shape)
            or not _extents(stride)
            or len(shape) != len(stride)
            or type(requires_grad) is not bool
            or not isinstance(hooks, _OrderedMapping)
            or hooks
        ):
            raise _refuse("A pickled tensor declaration is malformed.")
        return _Tensor(storage, offset, shape, stride)
    raise _refuse(f"The source pickle calls {function.module}.{function.name}.")


# Deliberately one fail-closed opcode dispatch: stack, memo and storage effects
# must remain auditable together; a handler registry would obscure the allowlist.
def _symbolic_pickle(payload: bytes, /) -> tuple[Any, dict[str, _Storage]]:
    """Interpret ``data.pkl`` symbolically; nothing is unpickled or called."""
    stack: list[Any] = []
    frames: list[list[Any]] = []
    memo: dict[int, Any] = {}
    storages: dict[str, _Storage] = {}
    result: Any = None
    try:
        for opcode, argument, _ in pickletools.genops(payload):
            match opcode.name:
                case "PROTO" | "FRAME":
                    pass
                case "MARK":
                    frames.append(stack)
                    stack = []
                case "BINPUT" | "LONG_BINPUT":
                    if not isinstance(argument, int):
                        raise ValueError(opcode.name)
                    memo[argument] = stack[-1]
                case "MEMOIZE":
                    memo[len(memo)] = stack[-1]
                case "BINGET" | "LONG_BINGET":
                    if not isinstance(argument, int):
                        raise ValueError(opcode.name)
                    stack.append(memo[argument])
                case "GLOBAL":
                    module, name = str(argument).split(" ")
                    stack.append(_global(module, name))
                case "STACK_GLOBAL":
                    name = stack.pop()
                    stack.append(_global(stack.pop(), name))
                case (
                    "BINUNICODE"
                    | "SHORT_BINUNICODE"
                    | "BINUNICODE8"
                    | "BININT"
                    | "BININT1"
                    | "BININT2"
                    | "LONG1"
                    | "BINFLOAT"
                ):
                    stack.append(argument)
                case "NONE":
                    stack.append(None)
                case "NEWTRUE" | "NEWFALSE":
                    stack.append(opcode.name == "NEWTRUE")
                case "EMPTY_TUPLE":
                    stack.append(())
                case "TUPLE":
                    items = tuple(stack)
                    stack = frames.pop()
                    stack.append(items)
                case "TUPLE1" | "TUPLE2" | "TUPLE3":
                    count = int(opcode.name[-1])
                    if len(stack) < count:
                        raise IndexError(opcode.name)
                    items = tuple(stack[-count:])
                    del stack[-count:]
                    stack.append(items)
                case "EMPTY_DICT":
                    stack.append({})
                case "EMPTY_LIST":
                    stack.append([])
                case "APPEND":
                    item = stack.pop()
                    if not isinstance(stack[-1], list):
                        raise ValueError(opcode.name)
                    stack[-1].append(item)
                case "APPENDS":
                    items = stack
                    stack = frames.pop()
                    if not isinstance(stack[-1], list):
                        raise ValueError(opcode.name)
                    stack[-1].extend(items)
                case "SETITEM":
                    value = stack.pop()
                    key = stack.pop()
                    if not isinstance(stack[-1], dict):
                        raise ValueError(opcode.name)
                    stack[-1][key] = value
                case "SETITEMS":
                    items = stack
                    stack = frames.pop()
                    if len(items) % 2 or not isinstance(stack[-1], dict):
                        raise ValueError(opcode.name)
                    for key, value in zip(items[::2], items[1::2], strict=True):
                        stack[-1][key] = value
                case "BINPERSID":
                    identifier = stack.pop()
                    if not (
                        isinstance(identifier, tuple)
                        and len(identifier) == 5
                        and identifier[0] == "storage"
                        and isinstance(identifier[1], _Global)
                        and identifier[1].module == "torch"
                        and identifier[1].name in _STORAGES
                        and isinstance(identifier[2], str)
                        and identifier[2]
                        and "/" not in identifier[2]
                        and type(identifier[4]) is int
                        and identifier[4] >= 0
                    ):
                        raise _refuse("A pickled storage declaration is malformed.")
                    storage = _Storage(
                        identifier[2], _STORAGES[identifier[1].name], identifier[4]
                    )
                    if storages.setdefault(storage.key, storage) != storage:
                        raise _refuse("A storage is declared inconsistently.")
                    stack.append(storage)
                case "REDUCE":
                    arguments = stack.pop()
                    stack.append(_call(stack.pop(), arguments))
                case "BUILD":
                    state = stack.pop()
                    if (
                        not isinstance(stack[-1], _OrderedMapping)
                        or not isinstance(state, dict)
                        or set(state) - {"_metadata"}
                    ):
                        raise ValueError(opcode.name)
                case "STOP":
                    result = stack.pop()
                    if stack or frames:
                        raise ValueError(opcode.name)
                case other:
                    raise _refuse(
                        f"Pickle opcode {other} is outside the admitted format."
                    )
    except AIMNet2SourceRefusedError:
        raise
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise _refuse("The source pickle metadata is malformed.") from error
    return result, storages


def read_torch_v2_source(
    data: bytes, limits: AIMNet2ExtractionLimits, /
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read the source's ``torch.save`` v2 dict pickle-free.

    Returns its plain top-level fields (including ``model_yaml``), expanded
    under the metadata budget, and the ``state_dict`` tensors. Every tensor
    declaration (rank, extents, elements, bytes, storage span) and the
    aggregate logical bytes of all views, including views sharing a storage or
    with zero strides, are admitted before any storage is read or copied.
    """
    archive, prefix = _torch_container(data, limits)
    with archive:
        if prefix + "byteorder" in archive.namelist() and (
            archive.read(prefix + "byteorder") != b"little"
        ):
            raise _refuse("Only little-endian torch storages are admitted.")
        result, storages = _symbolic_pickle(archive.read(prefix + "data.pkl"))
        if not isinstance(result, dict) or "state_dict" not in result:
            raise _refuse("The torch source is not an AIMNet2 v2 dict with a state_dict.")
        if any(not isinstance(key, str) for key in result):
            raise _refuse("The v2 source dict has non-text keys.")
        state = result["state_dict"]
        if not isinstance(state, dict) or len(state) > limits.max_members:
            raise _refuse("The v2 state_dict must map at most max_members tensors.")
        if any(
            not isinstance(key, str) or not isinstance(value, _Tensor)
            for key, value in state.items()
        ):
            raise _refuse("The v2 state_dict must map names to tensors.")
        fields = _bounded_plain(
            {key: value for key, value in result.items() if key != "state_dict"},
            "v2 source fields",
            limits.metadata,
        )
        records = {
            member.filename.removeprefix(prefix + "data/"): member.file_size
            for member in archive.infolist()
            if member.filename.startswith(prefix + "data/")
        }
        if set(records) != set(storages):
            raise _refuse("Storage records differ from the declared storages.")
        for key, storage in storages.items():
            if records[key] != storage.numel * np.dtype(storage.dtype).itemsize:
                raise _refuse(f"Storage {key!r} size differs from its declaration.")
        # Phase 1: admit every declaration and the aggregate before any copy.
        rank = limits.archive_limits().max_array_rank
        admitted: list[tuple[str, _Tensor, tuple[int, ...], int]] = []
        logical = 0
        for name, tensor in state.items():
            itemsize = np.dtype(tensor.storage.dtype).itemsize
            if len(tensor.shape) > rank or any(
                extent > limits.max_tensor_elements for extent in tensor.shape
            ):
                raise _refuse(
                    f"Source tensor {name!r} exceeds the rank or extent limits."
                )
            elements = math.prod(tensor.shape)
            if (
                elements > limits.max_tensor_elements
                or elements * itemsize > limits.max_tensor_bytes
            ):
                raise _refuse(f"Source tensor {name!r} exceeds the tensor limits.")
            # Unit or empty axes never move; their strides are normalized to 0.
            stride = tuple(
                step if extent > 1 else 0
                for extent, step in zip(tensor.shape, tensor.stride, strict=True)
            )
            span = tensor.offset + (
                sum((extent - 1) * step for extent, step in zip(tensor.shape, stride)) + 1
                if elements
                else 0
            )
            if span > tensor.storage.numel:
                raise _refuse(f"Source tensor {name!r} views beyond its storage.")
            logical += elements * itemsize
            if logical > limits.max_source_bytes:
                raise _refuse(
                    "Source tensors view more logical bytes than the source byte limit."
                )
            admitted.append((name, tensor, stride, itemsize))
        # Phase 2: storages within the admitted container, then admitted copies.
        buffers = {
            key: np.frombuffer(
                archive.read(prefix + "data/" + key), dtype=np.dtype(storage.dtype)
            )
            for key, storage in storages.items()
        }
        tensors: dict[str, np.ndarray] = {}
        for name, tensor, stride, itemsize in admitted:
            base = buffers[tensor.storage.key]
            native = base.dtype.newbyteorder("=")
            if math.prod(tensor.shape) == 0:
                tensors[name] = np.empty(tensor.shape, dtype=native)
                continue
            view = np.lib.stride_tricks.as_strided(
                base[tensor.offset :],
                shape=tensor.shape,
                strides=tuple(step * itemsize for step in stride),
                writeable=False,
            )
            tensors[name] = np.array(view, dtype=native, copy=True)
    return fields, tensors


def _declaration_bytes(
    source: AIMNet2Source, limits: AIMNet2ExtractionLimits, /
) -> bytes:
    """Admitted bytes of the source's declaration artifact (config or YAML)."""
    declaration, manifest = source.declaration, source.declaration_manifest
    if declaration is None or manifest is None:
        raise ValueError(f"{source.format} sources require a declaration artifact.")
    return _artifact_bytes(
        declaration, manifest, source.policy, limits.max_manifest_bytes
    )


def _hf_record(
    source: AIMNet2Source, limits: AIMNet2ExtractionLimits, /
) -> AIMNet2SourceRecord:
    config = _json(
        _declaration_bytes(source, limits),
        limits,
        "config.json",
    )
    unknown = sorted(set(config) - _METADATA_FIELDS - _HF_FIELDS)
    if unknown:
        raise _refuse(f"config.json declares unadmitted fields {unknown}.")
    if "model_yaml" not in config:
        raise _refuse(
            "config.json has no model_yaml; the source's executable registry "
            "fallback is not reproduced."
        )
    model_yaml = config["model_yaml"]
    declaration = _model_declaration(model_yaml, limits)
    tensors, header_metadata = read_safetensors_tensors(
        _artifact_bytes(
            source.artifact, source.manifest, source.policy, limits.max_source_bytes
        ),
        limits,
    )
    metadata = _source_metadata(config)
    metadata["hf"] = {
        name: _plain(config[name])
        for name in sorted(_HF_FIELDS - {"model_yaml"})
        if name in config
    }
    metadata["safetensors_metadata"] = header_metadata
    return AIMNet2SourceRecord(
        "hf-safetensors", model_yaml, declaration, metadata, tensors, None
    )


def _v2_record(
    source: AIMNet2Source, limits: AIMNet2ExtractionLimits, /
) -> AIMNet2SourceRecord:
    fields, tensors = read_torch_v2_source(
        _artifact_bytes(
            source.artifact, source.manifest, source.policy, limits.max_source_bytes
        ),
        limits,
    )
    unknown = sorted(set(fields) - _METADATA_FIELDS - {"model_yaml"})
    if unknown:
        raise _refuse(f"The v2 source declares unadmitted fields {unknown}.")
    model_yaml = fields.get("model_yaml")
    if not isinstance(model_yaml, str):
        raise _refuse("The v2 source must carry its model_yaml text.")
    declaration = _model_declaration(model_yaml, limits)
    return AIMNet2SourceRecord(
        "torch-v2-state",
        model_yaml,
        declaration,
        _source_metadata(fields),
        tensors,
        None,
    )


def _legacy_record(
    source: AIMNet2Source,
    provider: AIMNet2ProviderRuntime | None,
    limits: AIMNet2ExtractionLimits,
    /,
) -> tuple[AIMNet2SourceRecord, dict[str, Any]]:
    if provider is None:
        raise ValueError("Legacy TorchScript extraction requires a provider runtime.")
    text = _declaration_bytes(source, limits)
    try:
        legacy_yaml = text.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _refuse("The legacy model YAML is not UTF-8.") from error
    original = _model_declaration(legacy_yaml, limits)
    payload = _artifact_bytes(
        source.artifact, source.manifest, source.policy, limits.max_source_bytes
    )
    request = {
        "operation": "extract-jit",
        "source": {
            "sha256": source.artifact.sha256,
            "byte_size": source.artifact.byte_size,
            "trusted_executable": True,
        },
    }
    manifest, arrays, directory = _run_provider(
        provider, request, {"source.bin": payload}, limits
    )
    try:
        attributes = manifest["attributes"]
        if not isinstance(attributes, Mapping) or not isinstance(
            attributes.get("children"), list
        ):
            raise ValueError("Provider result lacks JIT attributes.")
        tensors = {
            name.removeprefix("state/"): value
            for name, value in arrays.items()
            if name.startswith("state/")
        }
        if len(tensors) != len(arrays):
            raise ValueError("Provider result contains undeclared arrays.")
    finally:
        directory.cleanup()
    overrides = source.legacy or AIMNet2LegacyOverrides()
    # ``load_v1_model``: cutoff from the JIT model; Coulomb and DFT-D3 become
    # calculator terms by the shared externalization rule at conversion.
    outputs = (original.get("kwargs") or {}).get("outputs") or {}
    entries = outputs.values() if isinstance(outputs, Mapping) else outputs
    metadata: dict[str, Any] = {
        "format_version": 2,
        "cutoff": attributes["cutoff"],
        "has_embedded_d3ts": any(
            isinstance(entry, Mapping) and "D3TS" in str(entry.get("class", ""))
            for entry in entries
        ),
    }
    if overrides.implemented_species is not None:
        metadata["implemented_species"] = list(overrides.implemented_species)
    if overrides.family is not None:
        metadata["family"] = overrides.family
    if overrides.supports_charged_systems is not None:
        metadata["supports_charged_systems"] = overrides.supports_charged_systems
    record = AIMNet2SourceRecord(
        "torch-legacy-jit",
        legacy_yaml,
        original,
        metadata,
        tensors,
        MappingProxyType(dict(attributes)),
    )
    return record, manifest["provider"]


def extract_aimnet2_source(
    source: AIMNet2Source,
    /,
    *,
    provider: AIMNet2ProviderRuntime | None = None,
    limits: AIMNet2ExtractionLimits = AIMNet2ExtractionLimits(),
) -> tuple[AIMNet2SourceRecord, Mapping[str, Any] | None]:
    """Admit one source and return its pickle-free content and provider record."""
    if not isinstance(source, AIMNet2Source):
        raise TypeError("source must be an AIMNet2Source.")
    if not isinstance(limits, AIMNet2ExtractionLimits):
        raise TypeError("limits must be AIMNet2ExtractionLimits.")
    match source.format:
        case "hf-safetensors":
            return _hf_record(source, limits), None
        case "torch-v2-state":
            return _v2_record(source, limits), None
        case "torch-legacy-jit":
            record, provider_record = _legacy_record(source, provider, limits)
            return record, MappingProxyType(provider_record)
    raise AssertionError(source.format)


def _conversion_id(provenance: AIMNet2SourceProvenance, /) -> str:
    record = provenance.to_record()
    record.pop("conversion_id")
    return canonical_fingerprint({"kind": "aimnet2-source-conversion", **record})


def convert_aimnet2_checkpoint(
    source: AIMNet2Source,
    /,
    *,
    provider: AIMNet2ProviderRuntime | None = None,
    ts_pair_cutoff: float | None = None,
    precision: AtomisticPrecisionPolicy | None = None,
    limits: AIMNet2ExtractionLimits = AIMNet2ExtractionLimits(),
) -> AIMNet2CheckpointConversion:
    """Convert one admitted AIMNet2 source into a native model and its records.

    Source energies are eV, lengths angstrom and charges elementary charge.
    ``ts_pair_cutoff`` must declare the D3TS route radius when the model embeds
    D3TS. The metadata ``cutoff`` must cover the AEV radius (a smaller source
    neighbor radius would truncate the AEV, a different model definition).
    """
    from ...nn.atomistic._aimnet2_source import (
        aimnet2_calculator_composition,
        aimnet2_potential_from_source,
        externalize_source_long_range,
    )
    from ...units import ANGSTROM, ELECTRONVOLT

    record, provider_record = extract_aimnet2_source(
        source, provider=provider, limits=limits
    )
    metadata = record.metadata
    cutoff = metadata["cutoff"]
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
        raise _refuse("Source metadata cutoff must be a number.")
    core, state, externalized = externalize_source_long_range(
        record.declaration,
        record.tensors,
        legacy_attributes=record.legacy_attributes,
    )
    composition = aimnet2_calculator_composition(metadata, externalized=externalized)
    declared = metadata.get("implemented_species")
    if declared is not None and (
        not isinstance(declared, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in declared
        )
    ):
        raise _refuse("implemented_species must be a list of atomic numbers.")
    reconstruction = aimnet2_potential_from_source(
        AtomisticScaleContract(ANGSTROM, ELECTRONVOLT),
        core,
        state,
        declared_species=declared or None,
        charged_systems=composition.supports_charged_systems is not False,
        ts_pair_cutoff=ts_pair_cutoff,
        neighbor_cutoff=float(cutoff),
        precision=precision,
    )
    if float(cutoff) < reconstruction.architecture.aev_cutoff:
        raise _refuse(
            "The source neighbor cutoff is smaller than rc_s; the source would "
            "truncate its AEV."
        )
    if "has_embedded_d3ts" in metadata and composition.embedded_ts_dispersion != (
        reconstruction.architecture.stage(AIMNet2OutputKind.TS_DISPERSION) is not None
    ):
        raise _refuse("Metadata has_embedded_d3ts differs from the model declaration.")
    if (composition.coulomb_mode == "sr_embedded") != (
        reconstruction.architecture.stage(AIMNet2OutputKind.SHORT_RANGE_COULOMB)
        is not None
    ):
        raise _refuse("Metadata coulomb_mode differs from the model declaration.")
    values = {
        "source_sha256": source.artifact.sha256,
        "source_byte_size": source.artifact.byte_size,
        "source_license_id": source.artifact.license_id,
        "source_manifest_id": source.artifact.manifest_id,
        "source_admission_id": source.artifact.admission_id,
        "declaration_sha256": None
        if source.declaration is None
        else source.declaration.sha256,
        "declaration_license_id": None
        if source.declaration is None
        else source.declaration.license_id,
        "source_format": source.format,
        "trusted_executable": source.format == "torch-legacy-jit",
        "source_pin": MappingProxyType(dict(AIMNET2_SOURCE_PIN)),
        "provider": provider_record,
        "metadata": MappingProxyType(_plain(metadata)),
        "model_yaml": record.model_yaml,
        "legacy_attributes": record.legacy_attributes,
        "externalized": MappingProxyType(externalized.to_record()),
        "composition": MappingProxyType(composition.to_record()),
        "reconstruction": MappingProxyType(reconstruction.to_record()),
    }
    unsigned = AIMNet2SourceProvenance(**values, conversion_id="pending")
    provenance = dataclasses.replace(unsigned, conversion_id=_conversion_id(unsigned))
    return AIMNet2CheckpointConversion(
        potential=reconstruction.potential,
        composition=composition,
        reconstruction=reconstruction,
        provenance=provenance,
    )


def _worker_bytes() -> bytes:
    return resources.files(__package__).joinpath(_WORKER_NAME).read_bytes()


def _run_provider(
    provider: AIMNet2ProviderRuntime,
    request: Mapping[str, Any],
    staged: Mapping[str, bytes],
    limits: AIMNet2ExtractionLimits,
    /,
) -> tuple[dict[str, Any], dict[str, np.ndarray], tempfile.TemporaryDirectory[str]]:
    if not isinstance(provider, AIMNet2ProviderRuntime):
        raise TypeError("provider must be an AIMNet2ProviderRuntime.")
    complete = {
        **request,
        "provider": {"torch": provider.torch_version, **AIMNET2_PROVIDER_RELEASES},
        "limits": limits.worker_record(),
    }
    encoded = json.dumps(complete, allow_nan=False, sort_keys=True).encode("utf-8")
    inputs = {"worker.py": _worker_bytes(), "request.json": encoded, **staged}
    total = sum(len(payload) for payload in inputs.values())
    destination = tempfile.TemporaryDirectory(prefix="phydrax-aimnet2-provider-")
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
            max_output_bytes=total + limits.max_log_bytes,
            execution_policy=ExternalExecutionPolicy(inherit_environment=False),
            artifacts=PinnedFileOutputs(
                destination.name,
                (PinnedFileRequest("result.zip", limits.max_result_bytes),),
                limits.max_result_bytes,
            ),
        )
    except ExternalRuntimeError as error:
        destination.cleanup()
        failed = error.result
        if failed is not None and failed.returncode == _REFUSED_EXIT:
            lines = failed.stderr.decode("utf-8", "replace").strip().splitlines()
            refusal = next(
                (line for line in reversed(lines) if line.startswith("REFUSED: ")),
                "REFUSED: provider refusal without a message",
            )
            raise _refuse(refusal.removeprefix("REFUSED: ")) from error
        raise
    location = Path(result.file_artifact("result.zip").location)
    try:
        manifest, arrays = read_array_archive(location, limits=limits.archive_limits())
    except Exception:
        destination.cleanup()
        raise
    if (
        manifest.get("format") != _RESULT_FORMAT
        or manifest.get("operation") != request["operation"]
    ):
        destination.cleanup()
        raise ValueError("Provider result manifest is not the canonical result.")
    provider_record = manifest.get("provider")
    if (
        not isinstance(provider_record, Mapping)
        or provider_record.get("torch", {}).get("version") != provider.torch_version
    ):
        destination.cleanup()
        raise ValueError("Provider result reports a different torch release.")
    return manifest, arrays, destination


@dataclass(frozen=True, slots=True)
class AIMNet2OracleConfiguration:
    """One finite configuration for the isolated source oracle (angstrom, e).

    ``multiplicity`` is required exactly for two-channel (NSE) sources.
    """

    atomic_numbers: tuple[int, ...]
    positions: np.ndarray
    charge: float
    multiplicity: float | None = None

    def __post_init__(self) -> None:
        numbers = tuple(int(value) for value in self.atomic_numbers)
        positions = np.asarray(self.positions, dtype=np.float64)
        if not numbers or positions.shape != (len(numbers), 3):
            raise ValueError("positions must have shape (atoms, 3).")
        if not np.all(np.isfinite(positions)) or not math.isfinite(self.charge):
            raise ValueError("Oracle inputs must be finite.")
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "charge", float(self.charge))

    def to_request(self) -> dict[str, Any]:
        return {
            "numbers": list(self.atomic_numbers),
            "positions": self.positions.tolist(),
            "charge": self.charge,
            "multiplicity": self.multiplicity,
        }


@dataclass(frozen=True, slots=True)
class AIMNet2OracleEvaluation:
    """Source-module outputs (model energy only, no calculator terms)."""

    energies: np.ndarray
    forces: tuple[np.ndarray, ...]
    charges: tuple[np.ndarray, ...]
    spin_charges: tuple[np.ndarray, ...] | None
    provider: Mapping[str, Any]
    evaluation_dtype: str


def evaluate_aimnet2_source(
    record: AIMNet2SourceRecord,
    configurations: Sequence[AIMNet2OracleConfiguration],
    /,
    *,
    provider: AIMNet2ProviderRuntime,
    evaluation_dtype: Literal["float32", "float64"] = "float64",
    limits: AIMNet2ExtractionLimits = AIMNet2ExtractionLimits(),
) -> AIMNet2OracleEvaluation:
    """Evaluate the source model with the RECORD-verified source package.

    The provider rebuilds the model from ``record.model_yaml`` (``aimnet.``
    classes only), loads the pickle-free state with the source key rules and
    runs each configuration in the source's dense single-molecule mode. It is
    an isolated reference oracle, never a product route.
    """
    if not isinstance(record, AIMNet2SourceRecord):
        raise TypeError("record must be an AIMNet2SourceRecord.")
    cases = tuple(configurations)
    if not cases or any(
        not isinstance(case, AIMNet2OracleConfiguration) for case in cases
    ):
        raise TypeError("configurations must be AIMNet2OracleConfiguration values.")
    if evaluation_dtype not in ("float32", "float64"):
        raise ValueError("evaluation_dtype must be float32 or float64.")
    with tempfile.TemporaryDirectory(prefix="phydrax-aimnet2-state-") as staging:
        location = Path(staging) / "state.zip"
        write_array_archive(
            location,
            manifest={"format": "phydrax-aimnet2-state"},
            limits=limits.archive_limits(),
            arrays={f"state/{name}": value for name, value in record.tensors.items()},
        )
        state = location.read_bytes()
    request = {
        "operation": "evaluate",
        "model_yaml": record.model_yaml,
        "evaluation_dtype": evaluation_dtype,
        "configurations": [case.to_request() for case in cases],
    }
    manifest, arrays, directory = _run_provider(
        provider, request, {"state.zip": state}, limits
    )
    try:
        count = len(cases)
        energies = np.asarray(arrays["energy"], dtype=np.float64)
        if energies.shape != (count,):
            raise ValueError("Provider energies do not match the configurations.")
        forces = tuple(np.asarray(arrays[f"forces/{index}"]) for index in range(count))
        charges = tuple(np.asarray(arrays[f"charges/{index}"]) for index in range(count))
        spin = (
            tuple(np.asarray(arrays[f"spin_charges/{index}"]) for index in range(count))
            if "spin_charges/0" in arrays
            else None
        )
    finally:
        directory.cleanup()
    return AIMNet2OracleEvaluation(
        energies=energies,
        forces=forces,
        charges=charges,
        spin_charges=spin,
        provider=MappingProxyType(manifest["provider"]),
        evaluation_dtype=evaluation_dtype,
    )


__all__ = [
    "AIMNET2_PROVIDER_RELEASES",
    "AIMNET2_SOURCE_PIN",
    "AIMNet2CheckpointConversion",
    "AIMNet2ExtractionLimits",
    "AIMNet2LegacyOverrides",
    "AIMNet2OracleConfiguration",
    "AIMNet2OracleEvaluation",
    "AIMNet2ProviderRuntime",
    "AIMNet2Source",
    "AIMNet2SourceFormat",
    "AIMNet2SourceProvenance",
    "AIMNet2SourceRecord",
    "AIMNet2SourceRefusedError",
    "convert_aimnet2_checkpoint",
    "evaluate_aimnet2_source",
    "extract_aimnet2_source",
    "read_safetensors_tensors",
    "read_torch_v2_source",
]
