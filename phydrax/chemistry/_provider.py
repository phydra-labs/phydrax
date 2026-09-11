#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit electronic-provider capability and preparation boundary."""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Literal, TYPE_CHECKING

import equinox as eqx
from jaxtyping import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ._model import ElectronicReferenceKind
from ._properties import ElectronicProperty
from ._result import (
    ElectronicEnergyEvaluation,
    ElectronicEnergyForceEvaluation,
    ElectronicEnergyForceHessianEvaluation,
    ElectronicEvaluation,
    ElectronicGroundStatePropertyEvaluation,
)


if TYPE_CHECKING:
    from ._calculation import ElectronicCalculationPlan


ElectronicExecutionKind = Literal["host", "device"]
ElectronicConcurrencyKind = Literal["serial", "thread-safe", "process-isolated"]


class ElectronicProviderUnavailableError(RuntimeError):
    """An explicitly selected provider is unavailable."""


class ElectronicCapabilityError(ValueError):
    """An available provider cannot satisfy the requested calculation."""


class ElectronicProviderCapabilities(StrictModule, NonTrainableState):
    """Immutable scientific and execution support of one provider."""

    properties: tuple[ElectronicProperty, ...] = eqx.field(static=True)
    references: tuple[ElectronicReferenceKind, ...] = eqx.field(static=True)
    finite_geometry: bool = eqx.field(static=True)
    periodic_geometry: bool = eqx.field(static=True)
    total_charge: bool = eqx.field(static=True)
    spin_multiplicity: bool = eqx.field(static=True)
    external_point_charges: bool = eqx.field(static=True)
    point_charge_forces: bool = eqx.field(static=True)
    conservative_forces: bool = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)
    batching: bool = eqx.field(static=True)
    execution: ElectronicExecutionKind = eqx.field(static=True)
    concurrency: ElectronicConcurrencyKind = eqx.field(static=True)
    capabilities_id: str = eqx.field(static=True)

    def __init__(
        self,
        properties: Sequence[ElectronicProperty],
        references: Sequence[ElectronicReferenceKind],
        /,
        *,
        finite_geometry: bool = True,
        periodic_geometry: bool = False,
        total_charge: bool = True,
        spin_multiplicity: bool = True,
        external_point_charges: bool = False,
        point_charge_forces: bool = False,
        conservative_forces: bool = True,
        differentiable: bool = False,
        batching: bool = False,
        execution: ElectronicExecutionKind = "host",
        concurrency: ElectronicConcurrencyKind = "serial",
    ):
        properties_ = tuple(sorted(set(properties), key=lambda value: value.value))
        references_ = tuple(sorted(set(references), key=lambda value: value.value))
        if not properties_ or any(
            not isinstance(value, ElectronicProperty) for value in properties_
        ):
            raise TypeError("properties must contain ElectronicProperty values.")
        if not references_ or any(
            not isinstance(value, ElectronicReferenceKind) for value in references_
        ):
            raise TypeError("references must contain ElectronicReferenceKind values.")
        if ElectronicProperty.ENERGY not in properties_:
            raise ValueError("Electronic providers must support energy.")
        if (
            ElectronicProperty.HESSIAN in properties_
            and ElectronicProperty.FORCES not in properties_
        ):
            raise ValueError("Hessian provider capability requires forces.")
        if execution not in ("host", "device"):
            raise ValueError("execution must be host or device.")
        if concurrency not in ("serial", "thread-safe", "process-isolated"):
            raise ValueError("Unknown electronic provider concurrency model.")
        if point_charge_forces and not external_point_charges:
            raise ValueError("Point-charge forces require external point-charge support.")
        values = {
            "finite_geometry": bool(finite_geometry),
            "periodic_geometry": bool(periodic_geometry),
            "total_charge": bool(total_charge),
            "spin_multiplicity": bool(spin_multiplicity),
            "external_point_charges": bool(external_point_charges),
            "point_charge_forces": bool(point_charge_forces),
            "conservative_forces": bool(conservative_forces),
            "differentiable": bool(differentiable),
            "batching": bool(batching),
        }
        self.properties = properties_
        self.references = references_
        for name, value in values.items():
            setattr(self, name, value)
        self.execution = execution
        self.concurrency = concurrency
        self.capabilities_id = canonical_fingerprint(
            {
                "kind": "electronic-provider-capabilities",
                "properties": [value.value for value in properties_],
                "references": [value.value for value in references_],
                **values,
                "execution": execution,
                "concurrency": concurrency,
            }
        )

    def require(self, calculation: ElectronicCalculationPlan, /) -> None:
        from ._calculation import ElectronicCalculationPlan

        if not isinstance(calculation, ElectronicCalculationPlan):
            raise TypeError("calculation must be ElectronicCalculationPlan.")
        periodic = calculation.system.cell is not None and any(
            calculation.system.cell.periodic_axes
        )
        if periodic and not self.periodic_geometry:
            raise ElectronicCapabilityError("Provider does not support periodic geometry.")
        if not periodic and not self.finite_geometry:
            raise ElectronicCapabilityError("Provider does not support finite geometry.")
        missing = tuple(
            property_
            for property_ in calculation.request.properties
            if property_ not in self.properties
        )
        if missing:
            names = ", ".join(value.value for value in missing)
            raise ElectronicCapabilityError(f"Provider lacks requested properties: {names}.")
        reference = calculation.model_chemistry.method.reference
        if reference not in self.references:
            raise ElectronicCapabilityError(
                f"Provider does not support reference kind {reference.value!r}."
            )
        state = calculation.state
        if state.total_charge != 0 and not self.total_charge:
            raise ElectronicCapabilityError("Provider does not support charged systems.")
        if state.spin_multiplicity != 1 and not self.spin_multiplicity:
            raise ElectronicCapabilityError("Provider does not support open-shell systems.")


class AbstractPreparedElectronicCalculation(StrictModule, NonTrainableState):
    calculation: AbstractAttribute[ElectronicCalculationPlan]
    capabilities: AbstractAttribute[ElectronicProviderCapabilities]
    provider_id: AbstractAttribute[str]
    prepared_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None = None,
        /,
    ) -> ElectronicEvaluation:
        raise NotImplementedError


class AbstractElectronicProvider(StrictModule, NonTrainableState):
    provider_id: AbstractAttribute[str]
    capabilities: AbstractAttribute[ElectronicProviderCapabilities]

    @abc.abstractmethod
    def prepare(
        self, calculation: ElectronicCalculationPlan, /
    ) -> AbstractPreparedElectronicCalculation:
        raise NotImplementedError


ElectronicEvaluator = Callable[
    ["ElectronicCalculationPlan", ArrayLike, ArrayLike | None], ElectronicEvaluation
]


class CallablePreparedElectronicCalculation(AbstractPreparedElectronicCalculation):
    calculation: ElectronicCalculationPlan
    capabilities: ElectronicProviderCapabilities
    evaluator: ElectronicEvaluator
    provider_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        calculation: ElectronicCalculationPlan,
        capabilities: ElectronicProviderCapabilities,
        evaluator: ElectronicEvaluator,
        provider_id: str,
        /,
    ):
        if not isinstance(capabilities, ElectronicProviderCapabilities):
            raise TypeError("capabilities must be ElectronicProviderCapabilities.")
        self.calculation = calculation
        self.capabilities = capabilities
        self.evaluator = evaluator
        self.provider_id = provider_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-callable-electronic-calculation",
                "calculation": calculation.calculation_id,
                "capabilities": capabilities.capabilities_id,
                "provider": provider_id,
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None = None,
        /,
    ) -> ElectronicEvaluation:
        result = self.evaluator(self.calculation, positions, cell_vectors)
        if not isinstance(
            result,
            (
                ElectronicEnergyEvaluation,
                ElectronicEnergyForceEvaluation,
                ElectronicEnergyForceHessianEvaluation,
                ElectronicGroundStatePropertyEvaluation,
            ),
        ):
            raise TypeError("Electronic evaluator returned an unsupported result type.")
        if result.header.provider_id != self.provider_id:
            raise ValueError("Electronic evaluator changed provider identity.")
        if result.header.request_id != self.calculation.request.request_id:
            raise ValueError("Electronic evaluator changed property-request identity.")
        return result


class CallableElectronicProvider(AbstractElectronicProvider):
    evaluator: ElectronicEvaluator
    provider_id: str = eqx.field(static=True)
    capabilities: ElectronicProviderCapabilities

    def __init__(
        self,
        evaluator: ElectronicEvaluator,
        provider_id: str,
        capabilities: ElectronicProviderCapabilities,
        /,
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        identifier = str(provider_id).strip()
        if not identifier:
            raise ValueError("provider_id must be non-empty.")
        if not isinstance(capabilities, ElectronicProviderCapabilities):
            raise TypeError("capabilities must be ElectronicProviderCapabilities.")
        self.evaluator = evaluator
        self.provider_id = identifier
        self.capabilities = capabilities

    def prepare(
        self, calculation: ElectronicCalculationPlan, /
    ) -> CallablePreparedElectronicCalculation:
        from ._calculation import ElectronicCalculationPlan

        if not isinstance(calculation, ElectronicCalculationPlan):
            raise TypeError("calculation must be ElectronicCalculationPlan.")
        self.capabilities.require(calculation)
        return CallablePreparedElectronicCalculation(
            calculation,
            self.capabilities,
            self.evaluator,
            self.provider_id,
        )


__all__ = [
    "AbstractElectronicProvider",
    "AbstractPreparedElectronicCalculation",
    "CallableElectronicProvider",
    "CallablePreparedElectronicCalculation",
    "ElectronicCapabilityError",
    "ElectronicConcurrencyKind",
    "ElectronicExecutionKind",
    "ElectronicProviderCapabilities",
    "ElectronicProviderUnavailableError",
]
