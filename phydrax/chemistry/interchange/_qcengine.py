#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit QCEngine execution over the native QCSchema boundary."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import equinox as eqx
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint, canonical_mapping
from .._calculation import ElectronicCalculationPlan
from .._properties import ElectronicProperty
from .._provider import (
    AbstractElectronicProvider,
    AbstractPreparedElectronicCalculation,
    ElectronicProviderCapabilities,
)
from .._result import ElectronicEvaluation
from ._qcschema import (
    electronic_calculation_to_qcelemental,
    electronic_evaluation_from_qcschema,
)


def is_qcengine_available() -> bool:
    return (
        importlib.util.find_spec("qcengine") is not None
        and importlib.util.find_spec("qcelemental") is not None
    )


def require_qcengine():
    if not is_qcengine_available():
        raise ImportError(
            "QCEngine execution requires optional dependencies 'qcengine' and 'qcelemental'."
        )
    return importlib.import_module("qcengine")


class PreparedQCEngineCalculation(AbstractPreparedElectronicCalculation):
    calculation: ElectronicCalculationPlan
    capabilities: ElectronicProviderCapabilities
    program: str = eqx.field(static=True)
    keywords: Mapping[str, Any] = eqx.field(static=True)
    local_options: Mapping[str, Any] = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        calculation: ElectronicCalculationPlan,
        capabilities: ElectronicProviderCapabilities,
        program: str,
        keywords: Mapping[str, Any],
        local_options: Mapping[str, Any],
        provider_id: str,
        /,
    ):
        if not isinstance(capabilities, ElectronicProviderCapabilities):
            raise TypeError("capabilities must be ElectronicProviderCapabilities.")
        self.calculation = calculation
        self.capabilities = capabilities
        self.program = program
        self.keywords = keywords
        self.local_options = local_options
        self.provider_id = provider_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-qcengine-calculation",
                "calculation": calculation.calculation_id,
                "capabilities": capabilities.capabilities_id,
                "program": program,
                "keywords": dict(keywords),
                "local_options": dict(local_options),
                "provider": provider_id,
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None = None,
        /,
    ) -> ElectronicEvaluation:
        qcengine = require_qcengine()
        atomic_input, _ = electronic_calculation_to_qcelemental(
            self.calculation,
            positions,
            cell_vectors=cell_vectors,
            keywords=self.keywords,
        )
        output = qcengine.compute(
            atomic_input,
            self.program,
            local_options=dict(self.local_options),
            raise_error=False,
        )
        record = output.model_dump(mode="json")
        result, _ = electronic_evaluation_from_qcschema(
            self.calculation,
            self.provider_id,
            positions,
            record,
            cell_vectors=cell_vectors,
        )
        return result


class QCEngineProvider(AbstractElectronicProvider):
    program: str = eqx.field(static=True)
    keywords: Mapping[str, Any] = eqx.field(static=True)
    local_options: Mapping[str, Any] = eqx.field(static=True)
    provider_version: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    model_chemistry_id: str = eqx.field(static=True)
    capabilities: ElectronicProviderCapabilities

    def __init__(
        self,
        program: str,
        capabilities: ElectronicProviderCapabilities,
        /,
        *,
        keywords: Mapping[str, Any] | None = None,
        model_chemistry_id: str,
        local_options: Mapping[str, Any] | None = None,
    ):
        program_ = str(program).strip()
        if not program_:
            raise ValueError("QCEngine program must be non-empty.")
        if not isinstance(capabilities, ElectronicProviderCapabilities):
            raise TypeError("capabilities must be ElectronicProviderCapabilities.")
        if ElectronicProperty.HESSIAN in capabilities.properties:
            raise ValueError(
                "QCEngine Hessians must use the native force-difference workflow."
            )
        if capabilities.periodic_geometry:
            raise ValueError(
                "The QCEngine QCSchema molecular provider is finite and nonperiodic."
            )
        model_id = str(model_chemistry_id).strip()
        if not model_id:
            raise ValueError("model_chemistry_id must be non-empty.")
        keywords_ = MappingProxyType(canonical_mapping({} if keywords is None else keywords))
        local_ = MappingProxyType(
            canonical_mapping({} if local_options is None else local_options)
        )
        available = is_qcengine_available()
        version = importlib.metadata.version("qcengine") if available else "unavailable"
        self.program = program_
        self.keywords = keywords_
        self.local_options = local_
        self.provider_version = version
        self.model_chemistry_id = model_id
        self.capabilities = capabilities
        self.provider_id = canonical_fingerprint(
            {
                "kind": "qcengine-provider",
                "program": program_,
                "version": version,
                "keywords": dict(keywords_),
                "local_options": dict(local_),
                "capabilities": capabilities.capabilities_id,
                "model_chemistry": model_id,
            }
        )

    def prepare(
        self, calculation: ElectronicCalculationPlan, /
    ) -> PreparedQCEngineCalculation:
        if not is_qcengine_available():
            raise ImportError(
                "QCEngine execution requires optional dependencies 'qcengine' and 'qcelemental'."
            )
        self.capabilities.require(calculation)
        if (
            calculation.model_chemistry.model_chemistry_id
            != self.model_chemistry_id
        ):
            raise ValueError("QCEngine provider is bound to another model chemistry.")
        return PreparedQCEngineCalculation(
            calculation,
            self.capabilities,
            self.program,
            self.keywords,
            self.local_options,
            self.provider_id,
        )


__all__ = [
    "PreparedQCEngineCalculation",
    "QCEngineProvider",
    "is_qcengine_available",
    "require_qcengine",
]
