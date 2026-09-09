#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule
from ..domain import (
    DomainComponent,
    DomainFunction,
    LocalFieldRef,
    PairedSupport,
    SubdomainPatch,
)
from ._base import AbstractResidualCondition


FieldBinding = str | LocalFieldRef


def _field_name(value: FieldBinding, /) -> str:
    if isinstance(value, LocalFieldRef):
        return value.solver_name
    name = str(value)
    if not name:
        raise ValueError("Interface field names must be non-empty.")
    return name


def _field_names(
    left_field: FieldBinding,
    right_field: FieldBinding,
    /,
) -> tuple[str, str]:
    left = _field_name(left_field)
    right = _field_name(right_field)
    if left == right:
        raise ValueError("Left and right interface fields must be distinct bindings.")
    return left, right


def _target_field(
    domain,
    target: DomainFunction | ArrayLike,
    /,
) -> DomainFunction:
    if isinstance(target, DomainFunction):
        if not target.domain.same_support(domain):
            raise ValueError("Interface target must live on the paired-support domain.")
        return target
    return DomainFunction(domain=domain, deps=(), func=jnp.asarray(target))


def _dependency_union(
    left: DomainFunction,
    right: DomainFunction,
    /,
) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*left.deps, *right.deps)))


class _NormalContract(StrictModule):
    flux: DomainFunction
    normal: DomainFunction
    flux_positions: tuple[int, ...] = eqx.field(static=True)
    normal_positions: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        flux: DomainFunction,
        normal: DomainFunction,
        deps: tuple[str, ...],
    ):
        by_label = {label: index for index, label in enumerate(deps)}
        self.flux = flux
        self.normal = normal
        self.flux_positions = tuple(by_label[label] for label in flux.deps)
        self.normal_positions = tuple(by_label[label] for label in normal.deps)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        flux = self.flux.func(
            *(args[index] for index in self.flux_positions),
            key=key,
            **kwargs,
        )
        normal = self.normal.func(
            *(args[index] for index in self.normal_positions),
            key=key,
            **kwargs,
        )
        flux_array = jnp.asarray(flux)
        normal_array = jnp.asarray(normal)
        if flux_array.ndim == 0 and normal_array.ndim == 0:
            return flux_array * normal_array
        return ein.contract("...i,...i->...", flux_array, normal_array)


def _normal_flux(
    flux: DomainFunction,
    normal: DomainFunction,
    /,
) -> DomainFunction:
    if not flux.domain.same_support(normal.domain):
        raise ValueError("Flux trace and interface normal must have the same support.")
    deps = _dependency_union(flux, normal)
    return DomainFunction(
        domain=flux.domain,
        deps=deps,
        func=_NormalContract(flux, normal, deps),
    )


class SubdomainValueJump(AbstractResidualCondition):
    """Prescribed pointwise value jump across one paired support."""

    fields: tuple[str, ...] = eqx.field(static=True)
    on: DomainComponent
    pairing: PairedSupport
    target: DomainFunction
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        left_field: FieldBinding,
        right_field: FieldBinding,
        pairing: PairedSupport,
        /,
        *,
        target: DomainFunction | ArrayLike = 0.0,
        label: str | None = None,
    ):
        if not isinstance(pairing, PairedSupport):
            raise TypeError("pairing must be a PairedSupport.")
        self.fields = _field_names(left_field, right_field)
        self.on = pairing.component
        self.pairing = pairing
        self.target = _target_field(pairing.component.domain, target)
        self.label = None if label is None else str(label)

    def residual(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        left = self.pairing.trace(functions[self.fields[0]], side="left")
        right = self.pairing.trace(functions[self.fields[1]], side="right")
        return right - left - self.target


class SubdomainFluxJump(AbstractResidualCondition):
    """Prescribed oriented normal-flux jump across one paired support."""

    fields: tuple[str, ...] = eqx.field(static=True)
    on: DomainComponent
    pairing: PairedSupport
    left_flux: Callable[[DomainFunction], DomainFunction] = eqx.field(static=True)
    right_flux: Callable[[DomainFunction], DomainFunction] = eqx.field(static=True)
    target: DomainFunction
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        left_field: FieldBinding,
        right_field: FieldBinding,
        pairing: PairedSupport,
        left_flux: Callable[[DomainFunction], DomainFunction],
        right_flux: Callable[[DomainFunction], DomainFunction],
        /,
        *,
        target: DomainFunction | ArrayLike = 0.0,
        label: str | None = None,
    ):
        if not isinstance(pairing, PairedSupport):
            raise TypeError("pairing must be a PairedSupport.")
        if pairing.normal is None:
            raise ValueError("Flux-jump conditions require an oriented interface normal.")
        if not callable(left_flux) or not callable(right_flux):
            raise TypeError("left_flux and right_flux must be callable.")
        self.fields = _field_names(left_field, right_field)
        self.on = pairing.component
        self.pairing = pairing
        self.left_flux = left_flux
        self.right_flux = right_flux
        self.target = _target_field(pairing.component.domain, target)
        self.label = None if label is None else str(label)

    def residual(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        left_local = self.left_flux(functions[self.fields[0]])
        right_local = self.right_flux(functions[self.fields[1]])
        if not isinstance(left_local, DomainFunction) or not isinstance(
            right_local, DomainFunction
        ):
            raise TypeError("Flux callbacks must return DomainFunction objects.")
        left = self.pairing.trace(left_local, side="left")
        right = self.pairing.trace(right_local, side="right")
        normal = self.pairing.normal
        if normal is None:
            raise RuntimeError("Validated flux pairing lost its normal field.")
        return _normal_flux(right - left, normal) - self.target


class SubdomainTransmission(AbstractResidualCondition):
    """General paired-support transmission residual."""

    fields: tuple[str, ...] = eqx.field(static=True)
    on: DomainComponent
    pairing: PairedSupport
    operator: Callable[
        [DomainFunction, DomainFunction, PairedSupport], DomainFunction
    ] = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        left_field: FieldBinding,
        right_field: FieldBinding,
        pairing: PairedSupport,
        operator: Callable[
            [DomainFunction, DomainFunction, PairedSupport], DomainFunction
        ],
        /,
        *,
        label: str | None = None,
    ):
        if not isinstance(pairing, PairedSupport):
            raise TypeError("pairing must be a PairedSupport.")
        if not callable(operator):
            raise TypeError("operator must be callable.")
        self.fields = _field_names(left_field, right_field)
        self.on = pairing.component
        self.pairing = pairing
        self.operator = operator
        self.label = None if label is None else str(label)

    def residual(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        result = self.operator(
            functions[self.fields[0]],
            functions[self.fields[1]],
            self.pairing,
        )
        if not isinstance(result, DomainFunction):
            raise TypeError("Transmission operators must return a DomainFunction.")
        if not result.domain.same_support(self.on.domain):
            raise ValueError(
                "Transmission operator result must live on the paired-support domain."
            )
        return result


class LocalizedResidual(AbstractResidualCondition):
    """Pull one ambient residual onto a local patch integration support."""

    fields: tuple[str, ...] = eqx.field(static=True)
    on: DomainComponent
    condition: AbstractResidualCondition
    patch: SubdomainPatch
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        condition: AbstractResidualCondition,
        patch: SubdomainPatch,
        /,
        *,
        on: DomainComponent | None = None,
        label: str | None = None,
    ):
        if not isinstance(condition, AbstractResidualCondition):
            raise TypeError("condition must be an AbstractResidualCondition.")
        if not isinstance(patch, SubdomainPatch):
            raise TypeError("patch must be a SubdomainPatch.")
        component = patch.interior if on is None else on
        if not isinstance(
            component, DomainComponent
        ) or not component.domain.same_support(patch.domain):
            raise ValueError("Localized residual support must live on the patch domain.")
        self.fields = condition.fields
        self.on = component
        self.condition = condition
        self.patch = patch
        self.label = condition.label if label is None else str(label)

    def residual(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        return self.patch.restrict(self.condition.residual(functions))


def localize_residual(
    condition: AbstractResidualCondition,
    patch: SubdomainPatch,
    /,
    *,
    on: DomainComponent | None = None,
    label: str | None = None,
) -> LocalizedResidual:
    return LocalizedResidual(condition, patch, on=on, label=label)


def subdomain_overlap_consistency(
    left_field: FieldBinding,
    right_field: FieldBinding,
    pairing: PairedSupport,
    /,
    *,
    label: str | None = None,
) -> SubdomainValueJump:
    """Construct zero value mismatch over a paired overlap volume."""
    if pairing.topology != "overlap-volume":
        raise ValueError("Overlap consistency requires an overlap-volume pairing.")
    return SubdomainValueJump(
        left_field,
        right_field,
        pairing,
        label=label,
    )


__all__ = [
    "LocalizedResidual",
    "SubdomainFluxJump",
    "SubdomainTransmission",
    "SubdomainValueJump",
    "localize_residual",
    "subdomain_overlap_consistency",
]
