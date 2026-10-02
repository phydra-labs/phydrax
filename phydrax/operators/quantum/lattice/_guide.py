#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Frozen positive similarities and their original physical inverse metric."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any, final, Literal, Protocol, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._identity import strict_module_payload
from ...._strict import StrictModule
from ....typing import Identifier, parse, PRNGKey
from .._amplitude import LogAmplitude
from ._address import QuantumAddress, QuantumConfigurationDomain
from ._column import (
    QuantumColumn,
    QuantumColumnSource,
    QuantumLatticeColumnOperator,
    RawQuantumExcitation,
)


QuantumGuideNodePolicy: TypeAlias = Literal["reject", "positive-log-floor"]


class QuantumGuideProvider(Protocol):
    """Frozen StrictModule callable with an explicit address-to-model mapping."""

    def __call__(self, address: QuantumAddress, /) -> LogAmplitude: ...


def _validate_frozen_provider_state(value: Any, path: str, /) -> None:
    """Admit immutable state, including fields absent from PyTree flattening.

    Provider fields are a genuinely dynamic interchange boundary. Identity encoding
    remains with the native StrictModule identity owner; this walk owns only the
    stronger frozen-guide mutability restriction.
    """
    if isinstance(value, (np.ndarray, list, dict, set, bytearray)):
        raise TypeError(
            f"Guide snapshot field {path} contains mutable {type(value).__name__} state."
        )
    if isinstance(value, StrictModule):
        for field in fields(value):
            _validate_frozen_provider_state(
                object.__getattribute__(value, field.name), f"{path}.{field.name}"
            )
    elif isinstance(value, tuple):
        for index, child in enumerate(value):
            _validate_frozen_provider_state(child, f"{path}[{index}]")
    elif isinstance(value, Mapping):
        raise TypeError(
            f"Guide snapshot field {path} requires the native immutable mapping owner."
        )


@final
class QuantumGuide(StrictModule):
    """Magnitude snapshot, not a live optimizer or a global spectrum certificate.

    ``globally_positive`` records a provider's declared global assumption; encountered
    checks never establish that assumption. Mapping identity is explicit because
    local-state indices need not be the provider's physical input coordinates.
    """

    domain: QuantumConfigurationDomain
    provider: QuantumGuideProvider
    provider_id: str = eqx.field(static=True)
    mapping_id: str = eqx.field(static=True)
    policy: QuantumGuideNodePolicy = eqx.field(static=True)
    log_floor: float | None = eqx.field(static=True)
    globally_positive: bool = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    numeric_id: str = eqx.field(static=True)
    guide_id: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)

    def __init__(
        self,
        domain: QuantumConfigurationDomain,
        provider: QuantumGuideProvider,
        /,
        *,
        provider_id: str,
        mapping_id: str,
        policy: QuantumGuideNodePolicy = "reject",
        log_floor: float | None = None,
        globally_positive: bool = False,
    ) -> None:
        if not isinstance(domain, QuantumConfigurationDomain):
            raise TypeError("Guide requires a configuration domain.")
        if not isinstance(provider, StrictModule) or not callable(provider):
            raise TypeError(
                "Guide providers must be frozen callable StrictModule snapshots."
            )
        _validate_frozen_provider_state(provider, "provider")
        if not provider_id or not mapping_id:
            raise ValueError(
                "Provider and explicit input-mapping identities are required."
            )
        selected = parse(policy, QuantumGuideNodePolicy, "policy")
        match selected:
            case "reject":
                if log_floor is not None:
                    raise ValueError("Reject policy does not accept a log floor.")
            case "positive-log-floor":
                if log_floor is None or not np.isfinite(log_floor):
                    raise ValueError("Positive log floor must be explicitly finite.")
            case _:
                raise ValueError("Unknown guide node policy.")
        payload = strict_module_payload(provider)
        numeric = parse(payload["numeric_content_id"], Identifier, "provider_numeric_id")
        self.domain = domain
        self.provider = provider
        self.provider_id = provider_id
        self.mapping_id = mapping_id
        self.policy = selected
        self.log_floor = log_floor
        self.globally_positive = globally_positive
        self.domain_id = domain.domain_id
        self.numeric_id = numeric
        self.guide_id = canonical_fingerprint(
            {
                "kind": "positive-quantum-guide",
                "domain": domain.domain_id,
                "provider": provider_id,
                "mapping": mapping_id,
                "numeric": numeric,
                "policy": selected,
                "floor": log_floor,
                "global_assumption": globally_positive,
            }
        )
        self.metric_id = canonical_fingerprint(
            {"kind": "inverse-guide-physical-metric", "guide": self.guide_id}
        )

    def log_value(self, address: QuantumAddress, /) -> tuple[Array, Array]:
        self.domain.validate_address(address)
        amplitude = self.provider(address)
        if not isinstance(amplitude, LogAmplitude):
            raise TypeError("Guide provider must return LogAmplitude.")
        if amplitude.log_abs.shape != ():
            raise ValueError("A guide address evaluation must return a scalar amplitude.")
        logarithm = amplitude.log_abs.astype(jnp.float64)
        valid = amplitude.valid & self.domain.contains_keys(address.key_words)
        match self.policy:
            case "reject":
                valid = valid & amplitude.nonzero & jnp.isfinite(logarithm)
            case "positive-log-floor":
                floor = self.log_floor
                if floor is None:
                    raise ValueError("Admitted floor policy is missing its finite floor.")
                logarithm = jnp.maximum(logarithm, jnp.asarray(floor, dtype=jnp.float64))
                valid = valid & jnp.isfinite(logarithm)
            case _:
                raise ValueError("Unknown guide node policy.")
        return logarithm, valid

    def metric(self, address: QuantumAddress, /) -> tuple[Array, Array]:
        logarithm, valid = self.log_value(address)
        value = jnp.exp(-2.0 * logarithm)
        return value, valid & jnp.isfinite(value) & (value > 0)

    def physical_matrix_element(
        self, bra: QuantumAddress, ket: QuantumAddress, element: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Return Q_ij/(g_i*g_j); generally Q_ij/(conj(g_i)*g_j)."""
        scalar = jnp.asarray(element, dtype=jnp.complex128)
        if scalar.shape != ():
            raise ValueError("A physical matrix element must be one scalar.")
        left, left_valid = self.log_value(bra)
        right, right_valid = self.log_value(ket)
        factor = jnp.exp(-left - right)
        value = scalar * factor.astype(jnp.complex128)
        return value, left_valid & right_valid & jnp.isfinite(factor) & (
            factor > 0
        ) & jnp.isfinite(value) & ((scalar == 0) | (value != 0))


@final
class GuidedQuantumColumnOperator(StrictModule):
    """H_g = D H D^-1, retaining original-H evidence and metric identity."""

    original: QuantumLatticeColumnOperator
    guide: QuantumGuide
    domain: QuantumConfigurationDomain
    operator_id: str = eqx.field(static=True)
    self_adjoint: bool = eqx.field(static=True)
    raw_route_bound: int = eqx.field(static=True)

    def __init__(
        self, original: QuantumLatticeColumnOperator, guide: QuantumGuide, /
    ) -> None:
        if not isinstance(original, QuantumLatticeColumnOperator) or not isinstance(
            guide, QuantumGuide
        ):
            raise TypeError(
                "Guided action requires a native original column operator and guide."
            )
        if not original.self_adjoint:
            raise ValueError(
                "Projector guides require a certified self-adjoint original Hamiltonian."
            )
        if original.domain.domain_id != guide.domain_id:
            raise ValueError("Guide and original Hamiltonian have different domains.")
        self.original = original
        self.guide = guide
        self.domain = original.domain
        self.operator_id = canonical_fingerprint(
            {
                "kind": "guided-quantum-column-operator",
                "original": original.operator_id,
                "guide": guide.guide_id,
                "metric": guide.metric_id,
            }
        )
        self.self_adjoint = False
        self.raw_route_bound = original.raw_route_bound

    def prepare_source(self, address: QuantumAddress, /) -> QuantumColumnSource:
        source = self.original.prepare_source(address)
        logarithm, valid = self.guide.log_value(address)
        return QuantumColumnSource(
            address=source.address,
            coordinate=source.coordinate,
            valid=source.valid & valid,
            guide_log=logarithm,
            guide_id=self.guide.guide_id,
        )

    def _source(
        self, address: QuantumAddress | QuantumColumnSource, /
    ) -> QuantumColumnSource:
        if isinstance(address, QuantumColumnSource):
            self.domain.validate_address(address.address)
            if address.guide_id == self.guide.guide_id and address.guide_log is not None:
                return address
            if address.guide_id is not None:
                raise ValueError("Source cache belongs to a different frozen guide.")
            return self.prepare_source(address.address)
        return self.prepare_source(address)

    def _transform_route(
        self, source: QuantumColumnSource, route: RawQuantumExcitation, /
    ) -> RawQuantumExcitation:
        source_log = source.guide_log
        if source_log is None:
            raise ValueError("Guided source cache is missing its frozen log magnitude.")

        def transform(candidate: RawQuantumExcitation) -> RawQuantumExcitation:
            target = QuantumAddress(
                key_words=candidate.target_key,
                domain_id=self.domain.domain_id,
                codec_id=self.domain.codec.codec_id,
            )
            target_log, target_valid = self.guide.log_value(target)
            ratio = jnp.exp(target_log - source_log)
            value = candidate.matrix_element * ratio.astype(jnp.complex128)
            successful = (
                candidate.successful
                & source.valid
                & target_valid
                & jnp.isfinite(ratio)
                & (ratio > 0)
                & jnp.isfinite(value)
                & ((candidate.matrix_element == 0) | (value != 0))
            )
            return eqx.tree_at(
                lambda result: (result.matrix_element, result.successful),
                candidate,
                (value, successful),
            )

        return jax.lax.cond(route.valid, transform, lambda candidate: candidate, route)

    def raw_route(
        self, address: QuantumAddress | QuantumColumnSource, index: ArrayLike, /
    ) -> RawQuantumExcitation:
        source = self._source(address)
        return self._transform_route(source, self.original.raw_route(source, index))

    def sample_raw_excitation(
        self, key: PRNGKey, address: QuantumAddress | QuantumColumnSource, /
    ) -> RawQuantumExcitation:
        source = self._source(address)
        return self._transform_route(
            source, self.original.sample_raw_excitation(key, source)
        )

    def outgoing_column(
        self, address: QuantumAddress | QuantumColumnSource, /
    ) -> QuantumColumn:
        source = self._source(address)
        column = self.original.outgoing_column(source)
        source_log = source.guide_log
        if source_log is None:
            raise ValueError("Guided source cache is missing its frozen log magnitude.")

        def transform(key: Array, value: Array, valid: Array) -> tuple[Array, Array]:
            def active(unused: Array) -> tuple[Array, Array]:
                target = QuantumAddress(
                    key_words=key,
                    domain_id=self.domain.domain_id,
                    codec_id=self.domain.codec.codec_id,
                )
                target_log, target_valid = self.guide.log_value(target)
                ratio = jnp.exp(target_log - source_log)
                result = value * ratio.astype(jnp.complex128)
                return result, source.valid & target_valid & jnp.isfinite(ratio) & (
                    ratio > 0
                ) & jnp.isfinite(result) & ((value == 0) | (result != 0))

            return jax.lax.cond(
                valid,
                active,
                lambda unused: (jnp.complex128(0), jnp.asarray(True, dtype=jnp.bool_)),
                jnp.asarray(0, dtype=jnp.int32),
            )

        values, successful = jax.vmap(transform)(
            column.target_keys, column.matrix_elements, column.valid
        )
        return eqx.tree_at(
            lambda result: (result.matrix_elements, result.successful),
            column,
            (values, column.successful & source.valid & jnp.all(successful)),
        )

    def diagonal(self, address: QuantumAddress | QuantumColumnSource, /) -> Array:
        source = self._source(address)
        value = self.original.diagonal(source)
        return eqx.error_if(
            value, ~source.valid, "Guide source is invalid for diagonal action."
        )


__all__ = [
    "GuidedQuantumColumnOperator",
    "QuantumGuide",
    "QuantumGuideNodePolicy",
    "QuantumGuideProvider",
]
