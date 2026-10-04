#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pickle-free durable artifacts of native atomistic models.

An artifact is the canonical array archive holding the model's registered
structure recipe (types, static fields and exact array specifications, without
Python module paths) and its dynamic leaves. Reconstruction is bounded and
fail-closed: exact member/byte/shape/dtype inventory before allocation,
registered type and field guard, one ``phydrax.typing`` validation of the
complete restored value, then the model class's own scientific/domain validator
(the constructor's owning checks), and finally recomputation of every recorded
identity. Serialized identities are never trusted as evidence of themselves.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .._array_archive import (
    array_collection_digest,
    ArrayArchiveCorruptionError,
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    read_array_archive,
    write_array_archive,
)
from .._fingerprint import canonical_fingerprint
from .._identity import NumericRevision
from .._model import (
    artifact_value,
    artifact_value_id,
    model_structure_recipe,
    register_artifact_value,
)
from .._model._structure import (
    model_from_array_recipe,
    model_recipe_array_inventory,
    model_recipe_template,
    pack_model_array_tree,
)
from ..units._dimension import DimensionSignature
from ..units._unit import UnitDefinition
from ._potential import (
    atomistic_potential_revision,
    AtomisticPotentialCapabilities,
    AtomisticPotentialRequirements,
    AtomisticSpeciesKind,
)
from ._types import AtomisticPrecisionPolicy, AtomisticScaleContract
from .interchange._mace_checkpoint import MACESourceProvenance


# Atomistic and unit contract types embedded in every native atomistic model.
# Model-family and substrate types are registered by their owning modules.
for _artifact_id, _artifact_type in (
    ("phydrax.atomistic:AtomisticScaleContract", AtomisticScaleContract),
    ("phydrax.atomistic:AtomisticPrecisionPolicy", AtomisticPrecisionPolicy),
    ("phydrax.atomistic:AtomisticPotentialCapabilities", AtomisticPotentialCapabilities),
    ("phydrax.atomistic:AtomisticPotentialRequirements", AtomisticPotentialRequirements),
    ("phydrax.atomistic:AtomisticSpeciesKind", AtomisticSpeciesKind),
    ("phydrax.units:UnitDefinition", UnitDefinition),
    ("phydrax.units:DimensionSignature", DimensionSignature),
):
    register_artifact_value(_artifact_id, _artifact_type)


_ARTIFACT_FORMAT = "phydrax-atomistic-model-artifact"
_LEAF_PREFIX = "model/leaves"
# Composed native models nest registered dataclasses deeply (layers, linear
# maps, coupling plans). Keep every byte and rank bound; admit that bounded
# tree depth and the leaf count of multilayer equivariant models.
ATOMISTIC_MODEL_ARTIFACT_LIMITS = dataclasses.replace(
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    max_members=4_097,
    max_manifest_bytes=16_777_216,
    max_central_directory_bytes=4_194_304,
    max_manifest_nesting=128,
)
_SECTION_FIELDS = frozenset(
    {"identity", "licenses", "model_recipe", "model_type", "source", "versions"}
)
_MANIFEST_FIELDS = _SECTION_FIELDS | {"arrays", "format"}
_IDENTITY_FIELDS = frozenset(
    {
        "architecture_id",
        "artifact_id",
        "content_id",
        "method_id",
        "numeric_revision_id",
        "semantic_id",
        "structure_id",
    }
)


class AtomisticModelArtifactError(ValueError):
    """A structurally readable artifact that fails scientific or identity checks."""


@dataclass(frozen=True, slots=True)
class _RegisteredModel:
    validate: Callable[[Any], None]
    revision: Callable[[Any], NumericRevision]
    exact: Callable[[Any], Any]


_REGISTERED_MODELS: dict[type, _RegisteredModel] = {}


def register_atomistic_model_artifact(
    model_type: type,
    /,
    *,
    validate: Callable[[Any], None],
    revision: Callable[[Any], NumericRevision],
    exact: Callable[[Any], Any],
) -> None:
    """Admit one model class with its owning domain validator and revision.

    ``validate`` must rerun every scientific check the constructor and
    preparation own (finite positive scales/cutoffs, species/head maps,
    selectors, coefficient/path/basis and parameter-space consistency, table
    bindings) on an already structurally validated restored value, raising
    ``ValueError``/``TypeError`` on refusal. ``revision`` returns the value's
    canonical (for prepared forms: bound) numeric revision and ``exact`` its
    exact trainable potential, whose architecture and method identify it. The
    class and every type reachable from its instances must also be registered
    with ``register_artifact_value``.
    """

    if not isinstance(model_type, type):
        raise TypeError("model_type must be a class.")
    if not callable(validate) or not callable(revision) or not callable(exact):
        raise TypeError("validate, revision and exact must be callables.")
    artifact_value_id(model_type)
    existing = _REGISTERED_MODELS.get(model_type)
    entry = _RegisteredModel(validate, revision, exact)
    if existing is not None and existing != entry:
        raise ValueError(f"{model_type.__qualname__} is already registered differently.")
    _REGISTERED_MODELS[model_type] = entry


def _exact_potential(model: Any, /) -> Any:
    return model


def _prepared_potential(prepared: Any, /) -> Any:
    return prepared.model


def _register_native_models() -> None:
    # Native model owners import atomistic; resolve them lazily to avoid a cycle.
    from ..nn.atomistic._mace import MACEPotential
    from ..nn.atomistic._mace_prepare import PreparedMACEPotential

    register_atomistic_model_artifact(
        MACEPotential,
        validate=MACEPotential.validate,
        revision=atomistic_potential_revision,
        exact=_exact_potential,
    )
    register_atomistic_model_artifact(
        PreparedMACEPotential,
        validate=PreparedMACEPotential.validate,
        revision=PreparedMACEPotential.revision,
        exact=_prepared_potential,
    )


def _registered(model_type: type, /) -> _RegisteredModel:
    _register_native_models()
    entry = _REGISTERED_MODELS.get(model_type)
    if entry is None:
        raise TypeError(
            f"{model_type.__qualname__} has no registered atomistic artifact validator."
        )
    return entry


@dataclass(frozen=True, slots=True)
class AtomisticModelArtifactManifest:
    """Verified identities of one native atomistic model artifact.

    ``structure_id`` content-addresses the static architecture/preparation
    recipe; ``numeric_revision`` is the model's canonical parameter revision;
    ``content_id`` digests every dynamic leaf, including fixed coefficient,
    basis and table leaves excluded from the parameter revision. ``source`` is
    the conversion provenance of an imported model, if any.
    """

    model_type: str
    model_recipe: Mapping[str, Any]
    architecture_id: str
    method_id: str
    structure_id: str
    semantic_id: str
    numeric_revision: NumericRevision
    content_id: str
    source: MACESourceProvenance | None
    licenses: tuple[str, ...]
    versions: Mapping[str, str]
    artifact_id: str


@dataclass(frozen=True, slots=True)
class AtomisticModelArtifact:
    """A restored, fully validated model with its verified manifest."""

    model: Any
    manifest: AtomisticModelArtifactManifest


def _runtime_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name) for name in ("phydrax", "equinox", "jax")
    }


def _identity(
    model: Any,
    recipe: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    source: MACESourceProvenance | None,
    licenses: Sequence[str],
    entry: _RegisteredModel,
    /,
) -> tuple[dict[str, str], NumericRevision]:
    revision = entry.revision(model)
    if not isinstance(revision, NumericRevision):
        raise TypeError("Registered revision functions must return NumericRevision.")
    exact = entry.exact(model)
    record = {
        "architecture_id": exact.architecture_id,
        "method_id": exact.method_id,
        "structure_id": canonical_fingerprint(
            {"kind": "atomistic-model-structure", "recipe": recipe}
        ),
        "semantic_id": revision.semantic_id,
        "numeric_revision_id": revision.revision_id,
        "content_id": array_collection_digest(arrays),
    }
    record["artifact_id"] = canonical_fingerprint(
        {
            "kind": "atomistic-model-artifact",
            **record,
            "source": None if source is None else source.conversion_id,
            "licenses": list(licenses),
        }
    )
    return record, revision


def _licenses(licenses: Sequence[str], /) -> tuple[str, ...]:
    if isinstance(licenses, str):
        raise TypeError("licenses must be a sequence of license identifiers.")
    values = tuple(licenses)
    if any(
        not isinstance(item, str) or not item or item != item.strip() for item in values
    ):
        raise ValueError("License identifiers must be canonical nonempty text.")
    if len(set(values)) != len(values):
        raise ValueError("License identifiers must be unique.")
    return tuple(sorted(values))


def _source_binding(model: Any, source: MACESourceProvenance, /) -> None:
    declaration = source.declaration
    configuration = model.configuration
    if (
        tuple(configuration.species) != tuple(declaration["atomic_numbers"])
        or configuration.head != source.head
        or float(configuration.cutoff) != float(declaration["r_max"])
    ):
        raise AtomisticModelArtifactError(
            "Model species, head or cutoff differ from its recorded source."
        )


def write_atomistic_model_artifact(
    path: str | Path,
    model: Any,
    /,
    *,
    source: MACESourceProvenance | None = None,
    licenses: Sequence[str] = (),
) -> AtomisticModelArtifactManifest:
    """Validate and atomically write one pickle-free native model artifact.

    The model is validated by its registered domain validator before anything
    is written; ``source`` binds conversion provenance (species, head and cutoff
    must agree with it). ``licenses`` records the model's rights identifiers;
    redistribution permission is never inferred.
    """

    section, arrays, manifest = model_artifact_section(
        model, source=source, licenses=licenses
    )
    write_array_archive(
        path,
        manifest={"format": _ARTIFACT_FORMAT, **section},
        limits=ATOMISTIC_MODEL_ARTIFACT_LIMITS,
        arrays=arrays,
    )
    return manifest


def model_artifact_section(
    model: Any,
    /,
    *,
    source: MACESourceProvenance | None = None,
    licenses: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, np.ndarray], AtomisticModelArtifactManifest]:
    """Validate a model and return its manifest section, leaves and manifest.

    Shared by standalone artifacts and restart bundles so both persist exactly
    one model representation under the ``model/leaves`` array namespace.
    """

    entry = _registered(type(model))
    entry.validate(model)
    if source is not None:
        if not isinstance(source, MACESourceProvenance):
            raise TypeError("source must be MACESourceProvenance or None.")
        _source_binding(entry.exact(model), source)
    license_ids = _licenses(licenses)
    recipe = model_structure_recipe(model)
    if recipe.get("kind") != "dataclass":
        raise TypeError("Atomistic model artifacts require a registered dataclass.")
    arrays = pack_model_array_tree(
        model, recipe, prefix=_LEAF_PREFIX, limits=ATOMISTIC_MODEL_ARTIFACT_LIMITS
    )
    identity, revision = _identity(model, recipe, arrays, source, license_ids, entry)
    versions = _runtime_versions()
    section = {
        "model_type": recipe["type"],
        "model_recipe": recipe,
        "identity": identity,
        "source": None if source is None else source.to_record(),
        "licenses": list(license_ids),
        "versions": versions,
    }
    return (
        section,
        arrays,
        _manifest(recipe, identity, revision, source, license_ids, versions),
    )


def _manifest(
    recipe: Mapping[str, Any],
    identity: Mapping[str, str],
    revision: NumericRevision,
    source: MACESourceProvenance | None,
    licenses: tuple[str, ...],
    versions: Mapping[str, str],
    /,
) -> AtomisticModelArtifactManifest:
    return AtomisticModelArtifactManifest(
        model_type=recipe["type"],
        model_recipe=MappingProxyType(dict(recipe)),
        architecture_id=identity["architecture_id"],
        method_id=identity["method_id"],
        structure_id=identity["structure_id"],
        semantic_id=identity["semantic_id"],
        numeric_revision=revision,
        content_id=identity["content_id"],
        source=source,
        licenses=licenses,
        versions=MappingProxyType(dict(versions)),
        artifact_id=identity["artifact_id"],
    )


def _validated_section(section: Mapping[str, Any], /) -> None:
    identity = section["identity"]
    versions = section["versions"]
    if (
        not isinstance(section["model_type"], str)
        or not isinstance(section["model_recipe"], dict)
        or not isinstance(identity, dict)
        or set(identity) != _IDENTITY_FIELDS
        or any(not isinstance(value, str) or not value for value in identity.values())
        or not isinstance(section["licenses"], list)
        or not isinstance(versions, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str) or not value
            for key, value in versions.items()
        )
        or (section["source"] is not None and not isinstance(section["source"], dict))
    ):
        raise ArrayArchiveCorruptionError("Atomistic model artifact metadata is invalid.")


def _restore_model(
    recipe: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    model_type: str,
    /,
    *,
    prefix: str = _LEAF_PREFIX,
    limits: ArrayArchiveLimits = ATOMISTIC_MODEL_ARTIFACT_LIMITS,
) -> tuple[Any, _RegisteredModel]:
    """Reconstruct and fully validate one registered model from host arrays."""

    if recipe.get("kind") != "dataclass" or recipe.get("type") != model_type:
        raise ArrayArchiveCorruptionError("Atomistic model type is inconsistent.")
    _register_native_models()
    try:
        declared = artifact_value(model_type)
    except ValueError as error:
        raise ArrayArchiveCorruptionError("Atomistic model type is unknown.") from error
    entry = _REGISTERED_MODELS.get(declared) if isinstance(declared, type) else None
    if entry is None:
        raise ArrayArchiveCorruptionError(
            "Archive type is not a registered atomistic model artifact class."
        )
    try:
        template = model_recipe_template(recipe, limits=limits)
        inventory = model_recipe_array_inventory(recipe, prefix=prefix, limits=limits)
    except (KeyError, TypeError, ValueError) as error:
        raise ArrayArchiveCorruptionError("Atomistic model recipe is invalid.") from error
    if type(template) is not declared or set(arrays) != {
        entry_.name for entry_ in inventory
    }:
        raise ArrayArchiveCorruptionError(
            "Atomistic model payload does not match its exact recipe inventory."
        )
    try:
        model = model_from_array_recipe(recipe, arrays, prefix=prefix, limits=limits)
    except (KeyError, TypeError, ValueError) as error:
        raise ArrayArchiveCorruptionError(
            "Atomistic model payload is incompatible with its recipe."
        ) from error
    if type(model) is not declared or model_structure_recipe(model) != recipe:
        raise ArrayArchiveCorruptionError(
            "Atomistic model structure changed during restoration."
        )
    try:
        entry.validate(model)
    except (TypeError, ValueError) as error:
        raise AtomisticModelArtifactError(
            "Restored atomistic model fails its scientific validation."
        ) from error
    return model, entry


def read_atomistic_model_artifact(
    path: str | Path,
    /,
    *,
    numeric_revision_id: str | None = None,
) -> AtomisticModelArtifact:
    """Restore, validate and identity-check one native model artifact.

    ``numeric_revision_id`` pins the expected parameter revision; a different
    restored revision refuses. No optional provider package is imported.
    """

    manifest, arrays = read_array_archive(path, limits=ATOMISTIC_MODEL_ARTIFACT_LIMITS)
    if set(manifest) != _MANIFEST_FIELDS:
        raise ArrayArchiveCorruptionError("Atomistic model artifact fields are invalid.")
    if manifest["format"] != _ARTIFACT_FORMAT:
        raise ArrayArchiveCorruptionError("Archive is not an atomistic model artifact.")
    section = {
        name: value
        for name, value in manifest.items()
        if name not in ("format", "arrays")
    }
    return model_artifact_from_section(
        section, arrays, numeric_revision_id=numeric_revision_id
    )


def model_artifact_from_section(
    section: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    /,
    *,
    numeric_revision_id: str | None = None,
) -> AtomisticModelArtifact:
    """Restore one model section from exactly its ``model/leaves`` arrays."""

    if not isinstance(section, Mapping) or set(section) != _SECTION_FIELDS:
        raise ArrayArchiveCorruptionError("Atomistic model section fields are invalid.")
    _validated_section(section)
    recipe = section["model_recipe"]
    model, entry = _restore_model(recipe, arrays, section["model_type"])
    try:
        source = (
            None
            if section["source"] is None
            else MACESourceProvenance.from_record(section["source"])
        )
        license_ids = _licenses(section["licenses"])
    except (TypeError, ValueError) as error:
        raise AtomisticModelArtifactError(
            "Atomistic model provenance or licenses are invalid."
        ) from error
    if source is not None:
        _source_binding(entry.exact(model), source)
    if license_ids != tuple(section["licenses"]):
        raise AtomisticModelArtifactError("Artifact licenses are not canonical.")
    identity, revision = _identity(model, recipe, arrays, source, license_ids, entry)
    if section["identity"] != identity:
        raise AtomisticModelArtifactError(
            "Atomistic model artifact identity does not match the restored model."
        )
    if numeric_revision_id is not None and revision.revision_id != numeric_revision_id:
        raise AtomisticModelArtifactError(
            "Restored model numeric revision differs from the pinned revision."
        )
    return AtomisticModelArtifact(
        model=model,
        manifest=_manifest(
            recipe, identity, revision, source, license_ids, section["versions"]
        ),
    )


def atomistic_model_identity(model: Any, /) -> dict[str, str]:
    """Return the intrinsic identity record of an in-memory registered model.

    Architecture, method, structure, semantic, numeric-revision and full-leaf
    content identities, independent of provenance and licenses. Runtime state
    bundles bind to this record; it equals the corresponding fields of a
    restored artifact manifest for the same model.
    """

    entry = _registered(type(model))
    recipe = model_structure_recipe(model)
    arrays = pack_model_array_tree(
        model, recipe, prefix=_LEAF_PREFIX, limits=ATOMISTIC_MODEL_ARTIFACT_LIMITS
    )
    identity = _identity(model, recipe, arrays, None, (), entry)[0]
    identity.pop("artifact_id")
    return identity


__all__ = [
    "ATOMISTIC_MODEL_ARTIFACT_LIMITS",
    "atomistic_model_identity",
    "AtomisticModelArtifact",
    "AtomisticModelArtifactError",
    "AtomisticModelArtifactManifest",
    "model_artifact_from_section",
    "model_artifact_section",
    "read_atomistic_model_artifact",
    "register_atomistic_model_artifact",
    "write_atomistic_model_artifact",
]
