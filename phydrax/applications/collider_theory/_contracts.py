#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...particle_physics import HEPProviderBinding, SystematicConfiguration


class PerturbativeOrder(StrEnum):
    LO = "LO"
    NLO = "NLO"
    NNLO = "NNLO"
    N3LO = "N3LO"
    RESUMMED = "resummed"
    NONPERTURBATIVE = "nonperturbative"


class TheoryModelKind(StrEnum):
    STANDARD_MODEL = "standard-model"
    EFT = "effective-field-theory"
    BSM = "beyond-standard-model"


class TheoryModel(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    kind: TheoryModelKind = eqx.field(static=True)
    parameter_names: tuple[str, ...] = eqx.field(static=True)
    scheme_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        kind: TheoryModelKind,
        parameter_names: Sequence[str],
        /,
        *,
        scheme_id: str,
        source_id: str,
    ):
        name_ = str(name).strip()
        parameters = tuple(sorted(str(value).strip() for value in parameter_names))
        scheme = str(scheme_id).strip()
        source = str(source_id).strip()
        if (
            not name_
            or not isinstance(kind, TheoryModelKind)
            or any(not value for value in parameters)
            or len(set(parameters)) != len(parameters)
            or not scheme
            or not source
        ):
            raise ValueError(
                "Theory model identity, parameters, scheme, and source are invalid."
            )
        self.name = name_
        self.kind = kind
        self.parameter_names = parameters
        self.scheme_id = scheme
        self.source_id = source
        self.model_id = canonical_fingerprint(
            {
                "kind": "collider-theory-model",
                "name": name_,
                "model_kind": kind.value,
                "parameters": list(parameters),
                "scheme": scheme,
                "source": source,
            }
        )


class TheoryProcessPlan(StrictModule, NonTrainableState):
    model: TheoryModel
    provider: HEPProviderBinding
    systematics: SystematicConfiguration
    process_expression: str = eqx.field(static=True)
    perturbative_order: PerturbativeOrder = eqx.field(static=True)
    subtraction_or_slicing_id: str = eqx.field(static=True)
    matching_merging_id: str = eqx.field(static=True)
    pdf_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    shower_hadronization_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: TheoryModel,
        provider: HEPProviderBinding,
        systematics: SystematicConfiguration,
        /,
        *,
        process_expression: str,
        perturbative_order: PerturbativeOrder,
        subtraction_or_slicing_id: str = "none",
        matching_merging_id: str = "none",
        pdf_id: str = "none",
        scale_id: str,
        shower_hadronization_id: str = "none",
    ):
        if (
            not isinstance(model, TheoryModel)
            or not isinstance(provider, HEPProviderBinding)
            or not isinstance(systematics, SystematicConfiguration)
        ):
            raise TypeError(
                "model, provider, and systematics must use typed HEP contracts."
            )
        if not isinstance(perturbative_order, PerturbativeOrder):
            raise TypeError("perturbative_order must be PerturbativeOrder.")
        values = tuple(
            str(value).strip()
            for value in (
                process_expression,
                subtraction_or_slicing_id,
                matching_merging_id,
                pdf_id,
                scale_id,
                shower_hadronization_id,
            )
        )
        if any(not value for value in values):
            raise ValueError("Theory process identities must be non-empty.")
        capability = "hep.hard-process-generation"
        if not provider.supports(capability):
            raise ValueError(f"Provider profile does not support {capability}.")
        (
            self.process_expression,
            self.subtraction_or_slicing_id,
            self.matching_merging_id,
            self.pdf_id,
            self.scale_id,
            self.shower_hadronization_id,
        ) = values
        self.model = model
        self.provider = provider
        self.systematics = systematics
        self.perturbative_order = perturbative_order
        self.plan_id = canonical_fingerprint(
            {
                "kind": "collider-theory-process",
                "model": model.model_id,
                "provider": provider.binding_id,
                "systematics": systematics.configuration_id,
                "values": list(values),
                "order": perturbative_order.value,
            }
        )


__all__ = ["PerturbativeOrder", "TheoryModel", "TheoryModelKind", "TheoryProcessPlan"]
