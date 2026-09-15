#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed physical electronic methods built on the canonical model identity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._model import (
    AbstractElectronicMethodPlan,
    ElectronicMethodFamily,
    ElectronicReferenceKind,
)


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _assign_method(
    instance: AbstractElectronicMethodPlan,
    family: ElectronicMethodFamily,
    method: str,
    reference: ElectronicReferenceKind,
    definition_ids: Sequence[str] = (),
    /,
) -> None:
    if not isinstance(family, ElectronicMethodFamily):
        raise TypeError("family must be ElectronicMethodFamily.")
    if not isinstance(reference, ElectronicReferenceKind):
        raise TypeError("reference must be ElectronicReferenceKind.")
    method_ = _identifier(method, "method")
    definitions = tuple(
        sorted(_identifier(value, "definition_id") for value in definition_ids)
    )
    if len(set(definitions)) != len(definitions):
        raise ValueError("definition_ids must be unique.")
    instance.family = family
    instance.method = method_
    instance.reference = reference
    instance.definition_ids = definitions
    instance.method_id = canonical_fingerprint(
        {
            "kind": "electronic-method",
            "family": family.value,
            "method": method_,
            "reference": reference.value,
            "definitions": list(definitions),
        }
    )


class DensityFunctionalPlan(StrictModule, NonTrainableState):
    """One exact exchange-correlation functional composition."""

    name: str = eqx.field(static=True)
    family: str = eqx.field(static=True)
    components: tuple[tuple[str, float], ...] = eqx.field(static=True)
    exact_exchange_fraction: float = eqx.field(static=True)
    long_range_exchange_fraction: float = eqx.field(static=True)
    range_separation: float = eqx.field(static=True)
    nonlocal_correlation_id: str | None = eqx.field(static=True)
    functional_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        family: str,
        components: Mapping[str, float],
        /,
        *,
        exact_exchange_fraction: float = 0.0,
        long_range_exchange_fraction: float = 0.0,
        range_separation: float = 0.0,
        nonlocal_correlation_id: str | None = None,
    ):
        name_ = _identifier(name, "functional name")
        family_ = _identifier(family, "functional family")
        normalized = tuple(
            sorted((str(key).strip(), float(value)) for key, value in components.items())
        )
        if not normalized or any(
            not key or not isfinite(weight) for key, weight in normalized
        ):
            raise ValueError("Functional components must be non-empty finite weights.")
        if len({key for key, _ in normalized}) != len(normalized):
            raise ValueError("Functional component names must be unique.")
        exchange = float(exact_exchange_fraction)
        long_range_exchange = float(long_range_exchange_fraction)
        separation = float(range_separation)
        if (
            not isfinite(exchange)
            or not 0.0 <= exchange <= 1.0
            or not isfinite(long_range_exchange)
            or not 0.0 <= long_range_exchange <= 1.0
            or exchange + long_range_exchange > 1.0
        ):
            raise ValueError(
                "Short- and long-range exact-exchange fractions must be physical."
            )
        if not isfinite(separation) or separation < 0.0:
            raise ValueError("range_separation must be finite and non-negative.")
        if (separation == 0.0) != (long_range_exchange == 0.0):
            raise ValueError(
                "Long-range exchange and range separation must be enabled together."
            )
        nonlocal_id = (
            None
            if nonlocal_correlation_id is None
            else _identifier(nonlocal_correlation_id, "nonlocal_correlation_id")
        )
        self.name = name_
        self.family = family_
        self.components = normalized
        self.exact_exchange_fraction = exchange
        self.long_range_exchange_fraction = long_range_exchange
        self.range_separation = separation
        self.nonlocal_correlation_id = nonlocal_id
        self.functional_id = canonical_fingerprint(
            {
                "kind": "density-functional-plan",
                "name": name_,
                "family": family_,
                "components": [list(value) for value in normalized],
                "exact_exchange_fraction": exchange,
                "long_range_exchange_fraction": long_range_exchange,
                "range_separation": separation,
                "nonlocal_correlation_id": nonlocal_id,
            }
        )

    @classmethod
    def slater_exchange(cls) -> DensityFunctionalPlan:
        return cls("slater-x", "lda", {"slater-x": 1.0})

    @classmethod
    def lda_vwn(cls) -> DensityFunctionalPlan:
        return cls("lda-vwn", "lda", {"slater-x": 1.0, "vwn-c": 1.0})

    @classmethod
    def lda_pw92(cls) -> DensityFunctionalPlan:
        return cls("lda-pw92", "lda", {"slater-x": 1.0, "pw92-c": 1.0})

    @classmethod
    def pbe(cls) -> DensityFunctionalPlan:
        return cls("pbe", "gga", {"pbe-x": 1.0, "pbe-c": 1.0})

    @classmethod
    def regularized_meta_pbe(cls, strength: float = 0.01) -> DensityFunctionalPlan:
        strength_ = float(strength)
        if not isfinite(strength_):
            raise ValueError("Meta-GGA correction strength must be finite.")
        return cls(
            "regularized-meta-pbe",
            "meta-gga",
            {
                "pbe-x": 1.0,
                "pbe-c": 1.0,
                "regularized-meta": strength_,
            },
        )

    @classmethod
    def pbe0(cls) -> DensityFunctionalPlan:
        return cls(
            "pbe0",
            "hybrid-gga",
            {"pbe-x": 0.75, "pbe-c": 1.0},
            exact_exchange_fraction=0.25,
        )

    @classmethod
    def range_separated_pbe_correlation(
        cls, range_separation: float
    ) -> DensityFunctionalPlan:
        return cls(
            "long-range-hf-pbe-c",
            "range-separated-hybrid-gga",
            {"pbe-c": 1.0},
            long_range_exchange_fraction=1.0,
            range_separation=range_separation,
        )

    @classmethod
    def b3lyp(cls) -> DensityFunctionalPlan:
        return cls(
            "b3lyp",
            "hybrid-gga",
            {"slater-x": 0.08, "b88-x": 0.72, "lyp-c": 0.81, "vwn-c": 0.19},
            exact_exchange_fraction=0.20,
        )


class HartreeFockMethodPlan(AbstractElectronicMethodPlan):
    family: ElectronicMethodFamily = eqx.field(static=True)
    method: str = eqx.field(static=True)
    reference: ElectronicReferenceKind = eqx.field(static=True)
    definition_ids: tuple[str, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference: ElectronicReferenceKind = ElectronicReferenceKind.RESTRICTED,
        /,
    ):
        _assign_method(self, ElectronicMethodFamily.HARTREE_FOCK, "hf", reference)


class KohnShamMethodPlan(AbstractElectronicMethodPlan):
    family: ElectronicMethodFamily = eqx.field(static=True)
    method: str = eqx.field(static=True)
    reference: ElectronicReferenceKind = eqx.field(static=True)
    definition_ids: tuple[str, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        functional: DensityFunctionalPlan,
        reference: ElectronicReferenceKind = ElectronicReferenceKind.RESTRICTED,
        /,
    ):
        if not isinstance(functional, DensityFunctionalPlan):
            raise TypeError("functional must be DensityFunctionalPlan.")
        _assign_method(
            self,
            ElectronicMethodFamily.KOHN_SHAM_DFT,
            functional.name,
            reference,
            (functional.functional_id,),
        )


class MP2MethodPlan(AbstractElectronicMethodPlan):
    family: ElectronicMethodFamily = eqx.field(static=True)
    method: str = eqx.field(static=True)
    reference: ElectronicReferenceKind = eqx.field(static=True)
    definition_ids: tuple[str, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference: ElectronicReferenceKind = ElectronicReferenceKind.RESTRICTED,
        /,
        *,
        spin_scaling_id: str | None = None,
    ):
        definitions = (
            ()
            if spin_scaling_id is None
            else (_identifier(spin_scaling_id, "spin_scaling_id"),)
        )
        _assign_method(self, ElectronicMethodFamily.MP2, "mp2", reference, definitions)


class CoupledClusterMethodPlan(AbstractElectronicMethodPlan):
    family: ElectronicMethodFamily = eqx.field(static=True)
    method: str = eqx.field(static=True)
    reference: ElectronicReferenceKind = eqx.field(static=True)
    definition_ids: tuple[str, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        level: str = "ccsd",
        reference: ElectronicReferenceKind = ElectronicReferenceKind.RESTRICTED,
        /,
    ):
        level_ = _identifier(level, "coupled-cluster level").lower()
        if level_ not in ("ccsd", "ccsd(t)"):
            raise ValueError("Coupled-cluster level must be ccsd or ccsd(t).")
        _assign_method(self, ElectronicMethodFamily.COUPLED_CLUSTER, level_, reference)


class ConfigurationInteractionMethodPlan(AbstractElectronicMethodPlan):
    family: ElectronicMethodFamily = eqx.field(static=True)
    method: str = eqx.field(static=True)
    reference: ElectronicReferenceKind = eqx.field(static=True)
    definition_ids: tuple[str, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        level: str = "fci",
        reference: ElectronicReferenceKind = ElectronicReferenceKind.RESTRICTED,
        /,
    ):
        level_ = _identifier(level, "configuration-interaction level").lower()
        if level_ not in ("cis", "cisd", "fci", "selected-ci"):
            raise ValueError("Unknown configuration-interaction level.")
        _assign_method(
            self,
            ElectronicMethodFamily.CONFIGURATION_INTERACTION,
            level_,
            reference,
        )


class MulticonfigurationMethodPlan(AbstractElectronicMethodPlan):
    family: ElectronicMethodFamily = eqx.field(static=True)
    method: str = eqx.field(static=True)
    reference: ElectronicReferenceKind = eqx.field(static=True)
    definition_ids: tuple[str, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: str = "casscf",
        reference: ElectronicReferenceKind = ElectronicReferenceKind.RESTRICTED,
        /,
        *,
        active_space_id: str,
    ):
        method_ = _identifier(method, "multiconfiguration method").lower()
        if method_ not in ("casci", "casscf", "dmrg-casci", "dmrg-casscf"):
            raise ValueError("Unknown multiconfiguration method.")
        _assign_method(
            self,
            ElectronicMethodFamily.MULTICONFIGURATION,
            method_,
            reference,
            (_identifier(active_space_id, "active_space_id"),),
        )


class ADCMethodPlan(AbstractElectronicMethodPlan):
    family: ElectronicMethodFamily = eqx.field(static=True)
    method: str = eqx.field(static=True)
    reference: ElectronicReferenceKind = eqx.field(static=True)
    definition_ids: tuple[str, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        order: str = "adc(2)",
        reference: ElectronicReferenceKind = ElectronicReferenceKind.RESTRICTED,
        /,
    ):
        order_ = _identifier(order, "ADC order").lower()
        if order_ not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise ValueError("ADC order must be adc(2), adc(2)-x, or adc(3).")
        _assign_method(self, ElectronicMethodFamily.ADC, order_, reference)


class ExternalElectronicMethodPlan(AbstractElectronicMethodPlan):
    family: ElectronicMethodFamily = eqx.field(static=True)
    method: str = eqx.field(static=True)
    reference: ElectronicReferenceKind = eqx.field(static=True)
    definition_ids: tuple[str, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: str,
        reference: ElectronicReferenceKind,
        /,
        *,
        definition_ids: Sequence[str] = (),
    ):
        _assign_method(
            self,
            ElectronicMethodFamily.CUSTOM_EXTERNAL,
            method,
            reference,
            definition_ids,
        )


__all__ = [
    "ADCMethodPlan",
    "ConfigurationInteractionMethodPlan",
    "CoupledClusterMethodPlan",
    "DensityFunctionalPlan",
    "ExternalElectronicMethodPlan",
    "HartreeFockMethodPlan",
    "KohnShamMethodPlan",
    "MP2MethodPlan",
    "MulticonfigurationMethodPlan",
]
