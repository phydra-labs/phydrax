#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider-independent molecular model-chemistry identities."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import ReferenceArtifactManifest


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    result = tuple(sorted(_identifier(value, name) for value in values))
    if len(result) != len(set(result)):
        raise ValueError(f"{name} values must be unique.")
    return result


class ElectronicReferenceKind(StrEnum):
    RESTRICTED = "restricted"
    UNRESTRICTED = "unrestricted"
    RESTRICTED_OPEN_SHELL = "restricted-open-shell"


class ElectronicMethodFamily(StrEnum):
    HARTREE_FOCK = "hartree-fock"
    KOHN_SHAM_DFT = "kohn-sham-dft"
    SEMIEMPIRICAL = "semiempirical"
    TIGHT_BINDING = "tight-binding"
    WAVEFUNCTION = "wavefunction"
    LEARNED = "learned"
    COMPOSITE = "composite"
    CUSTOM_EXTERNAL = "custom-external"


class ElectronicMethodPlan(StrictModule, NonTrainableState):
    """Physical electronic approximation, independent of its implementation."""

    family: ElectronicMethodFamily = eqx.field(static=True)
    method: str = eqx.field(static=True)
    reference: ElectronicReferenceKind = eqx.field(static=True)
    definition_ids: tuple[str, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        family: ElectronicMethodFamily,
        method: str,
        reference: ElectronicReferenceKind,
        /,
        *,
        definition_ids: Sequence[str] = (),
    ):
        if not isinstance(family, ElectronicMethodFamily):
            raise TypeError("family must be ElectronicMethodFamily.")
        if not isinstance(reference, ElectronicReferenceKind):
            raise TypeError("reference must be ElectronicReferenceKind.")
        method_ = _identifier(method, "method")
        definitions = _identifiers(definition_ids, "definition_id")
        self.family = family
        self.method = method_
        self.reference = reference
        self.definition_ids = definitions
        self.method_id = canonical_fingerprint(
            {
                "kind": "electronic-method",
                "family": family.value,
                "method": method_,
                "reference": reference.value,
                "definitions": list(definitions),
            }
        )


class BasisSetReference(StrictModule, NonTrainableState):
    """Exact basis label, convention, source, and optional governed payload."""

    name: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    artifact: ReferenceArtifactManifest | None = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        source_id: str,
        /,
        *,
        convention: str = "provider-native",
        artifact: ReferenceArtifactManifest | None = None,
    ):
        name_ = _identifier(name, "basis name")
        source = _identifier(source_id, "basis source_id")
        convention_ = _identifier(convention, "basis convention")
        if convention_ not in ("provider-native", "cartesian", "spherical"):
            raise ValueError(
                "basis convention must be provider-native, cartesian, or spherical."
            )
        if artifact is not None and not isinstance(artifact, ReferenceArtifactManifest):
            raise TypeError("artifact must be ReferenceArtifactManifest or None.")
        self.name = name_
        self.source_id = source
        self.convention = convention_
        self.artifact = artifact
        self.basis_id = canonical_fingerprint(
            {
                "kind": "basis-set-reference",
                "name": name_,
                "source": source,
                "convention": convention_,
                "artifact": None if artifact is None else artifact.manifest_id,
            }
        )


class ElectronicEnvironmentPlan(StrictModule, NonTrainableState):
    """Environment semantics that change an electronic Hamiltonian."""

    kind: str = eqx.field(static=True)
    parameter_ids: tuple[str, ...] = eqx.field(static=True)
    environment_id: str = eqx.field(static=True)

    def __init__(self, kind: str = "vacuum", /, *, parameter_ids: Sequence[str] = ()):
        kind_ = _identifier(kind, "environment kind")
        parameters = _identifiers(parameter_ids, "environment parameter_id")
        self.kind = kind_
        self.parameter_ids = parameters
        self.environment_id = canonical_fingerprint(
            {
                "kind": "electronic-environment",
                "environment_kind": kind_,
                "parameters": list(parameters),
            }
        )


class ElectronicModelChemistryPlan(StrictModule, NonTrainableState):
    """Method, basis, corrections, environment, and model-artifact identity."""

    method: ElectronicMethodPlan
    basis: BasisSetReference | None
    environment: ElectronicEnvironmentPlan
    correction_ids: tuple[str, ...] = eqx.field(static=True)
    relativistic_id: str | None = eqx.field(static=True)
    numerical_definition_ids: tuple[str, ...] = eqx.field(static=True)
    model_artifact: ReferenceArtifactManifest | None = eqx.field(static=True)
    model_chemistry_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: ElectronicMethodPlan,
        /,
        *,
        basis: BasisSetReference | None = None,
        environment: ElectronicEnvironmentPlan | None = None,
        correction_ids: Sequence[str] = (),
        relativistic_id: str | None = None,
        numerical_definition_ids: Sequence[str] = (),
        model_artifact: ReferenceArtifactManifest | None = None,
    ):
        if not isinstance(method, ElectronicMethodPlan):
            raise TypeError("method must be ElectronicMethodPlan.")
        if basis is not None and not isinstance(basis, BasisSetReference):
            raise TypeError("basis must be BasisSetReference or None.")
        environment_ = ElectronicEnvironmentPlan() if environment is None else environment
        if not isinstance(environment_, ElectronicEnvironmentPlan):
            raise TypeError("environment must be ElectronicEnvironmentPlan or None.")
        corrections = _identifiers(correction_ids, "correction_id")
        relativistic = (
            None if relativistic_id is None else _identifier(relativistic_id, "relativistic_id")
        )
        numerical = _identifiers(numerical_definition_ids, "numerical_definition_id")
        if model_artifact is not None and not isinstance(
            model_artifact, ReferenceArtifactManifest
        ):
            raise TypeError("model_artifact must be ReferenceArtifactManifest or None.")
        if method.family is ElectronicMethodFamily.LEARNED and model_artifact is None:
            raise ValueError("Learned model chemistry requires a governed model artifact.")
        self.method = method
        self.basis = basis
        self.environment = environment_
        self.correction_ids = corrections
        self.relativistic_id = relativistic
        self.numerical_definition_ids = numerical
        self.model_artifact = model_artifact
        self.model_chemistry_id = canonical_fingerprint(
            {
                "kind": "electronic-model-chemistry",
                "method": method.method_id,
                "basis": None if basis is None else basis.basis_id,
                "environment": environment_.environment_id,
                "corrections": list(corrections),
                "relativistic": relativistic,
                "numerical_definitions": list(numerical),
                "model_artifact": (
                    None if model_artifact is None else model_artifact.manifest_id
                ),
            }
        )


__all__ = [
    "BasisSetReference",
    "ElectronicEnvironmentPlan",
    "ElectronicMethodFamily",
    "ElectronicMethodPlan",
    "ElectronicModelChemistryPlan",
    "ElectronicReferenceKind",
]
