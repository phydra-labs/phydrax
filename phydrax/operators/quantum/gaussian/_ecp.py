#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Effective-core-potential identities and provider-neutral integral boundary."""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence

import equinox as eqx
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import AbstractAttribute, StrictModule
from ...._trainable import NonTrainableState
from ....atomistic import AtomisticSystemPlan
from ._basis import PreparedGaussianBasis


class ECPGaussianTerm(StrictModule, NonTrainableState):
    radial_power: int = eqx.field(static=True)
    exponent: float = eqx.field(static=True)
    coefficient: float = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(self, radial_power: int, exponent: float, coefficient: float, /):
        power = int(radial_power)
        exponent_ = float(exponent)
        coefficient_ = float(coefficient)
        if (
            power < 0
            or not np.isfinite(exponent_)
            or exponent_ <= 0.0
            or not np.isfinite(coefficient_)
        ):
            raise ValueError("ECP Gaussian term parameters are invalid.")
        self.radial_power = power
        self.exponent = exponent_
        self.coefficient = coefficient_
        self.term_id = canonical_fingerprint(
            {
                "kind": "ecp-gaussian-term",
                "radial_power": power,
                "exponent": exponent_,
                "coefficient": coefficient_,
            }
        )


class ECPChannelPlan(StrictModule, NonTrainableState):
    angular_momentum: int = eqx.field(static=True)
    terms: tuple[ECPGaussianTerm, ...]
    spin_orbit: bool = eqx.field(static=True)
    channel_id: str = eqx.field(static=True)

    def __init__(
        self,
        angular_momentum: int,
        terms: Sequence[ECPGaussianTerm],
        /,
        *,
        spin_orbit: bool = False,
    ):
        angular = int(angular_momentum)
        terms_ = tuple(terms)
        if (
            angular < 0
            or not terms_
            or any(not isinstance(value, ECPGaussianTerm) for value in terms_)
        ):
            raise ValueError(
                "ECP channels require non-negative angular momentum and terms."
            )
        self.angular_momentum = angular
        self.terms = terms_
        self.spin_orbit = bool(spin_orbit)
        self.channel_id = canonical_fingerprint(
            {
                "kind": "ecp-channel",
                "angular_momentum": angular,
                "terms": [value.term_id for value in terms_],
                "spin_orbit": self.spin_orbit,
            }
        )


class EffectiveCorePotentialPlan(StrictModule, NonTrainableState):
    center_particle_id: int = eqx.field(static=True)
    core_electron_count: int = eqx.field(static=True)
    local_channel: int = eqx.field(static=True)
    channels: tuple[ECPChannelPlan, ...]
    source_artifact_id: str = eqx.field(static=True)
    ecp_id: str = eqx.field(static=True)

    def __init__(
        self,
        center_particle_id: int,
        core_electron_count: int,
        local_channel: int,
        channels: Sequence[ECPChannelPlan],
        source_artifact_id: str,
        /,
    ):
        center = int(center_particle_id)
        core = int(core_electron_count)
        local = int(local_channel)
        channels_ = tuple(channels)
        artifact = str(source_artifact_id).strip()
        momenta = tuple(value.angular_momentum for value in channels_)
        if (
            core <= 0
            or local not in momenta
            or len(set(momenta)) != len(momenta)
            or not artifact
        ):
            raise ValueError(
                "ECP core, local channel, channels, or artifact identity is invalid."
            )
        self.center_particle_id = center
        self.core_electron_count = core
        self.local_channel = local
        self.channels = channels_
        self.source_artifact_id = artifact
        self.ecp_id = canonical_fingerprint(
            {
                "kind": "effective-core-potential",
                "center_particle_id": center,
                "core_electron_count": core,
                "local_channel": local,
                "channels": [value.channel_id for value in channels_],
                "source_artifact": artifact,
            }
        )

    def prepare(self, system: AtomisticSystemPlan, /) -> PreparedECP:
        return PreparedECP(self, system)


class PreparedECP(StrictModule, NonTrainableState):
    plan: EffectiveCorePotentialPlan
    center_index: int = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: EffectiveCorePotentialPlan, system: AtomisticSystemPlan, /):
        ids = np.asarray(system.particle_ids)
        matches = np.flatnonzero(ids == plan.center_particle_id)
        if matches.size != 1 or not bool(np.asarray(system.active_mask)[matches[0]]):
            raise ValueError("ECP center must be one active stable system particle ID.")
        atomic_number = int(np.asarray(system.atomic_numbers)[matches[0]])
        if plan.core_electron_count >= atomic_number:
            raise ValueError(
                "ECP core electron count must be smaller than nuclear charge."
            )
        self.plan = plan
        self.center_index = int(matches[0])
        self.system_id = system.system_id
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-ecp", "plan": plan.ecp_id, "system": system.system_id}
        )


class ECPIntegralEvaluation(StrictModule):
    matrix: Array
    gradient: Array | None
    hessian: Array | None
    successful: Array
    provider_id: str = eqx.field(static=True)


class AbstractECPIntegralProvider(StrictModule, NonTrainableState):
    provider_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        basis: PreparedGaussianBasis,
        ecp: PreparedECP,
        positions: ArrayLike,
        /,
        *,
        derivative_order: int = 0,
    ) -> ECPIntegralEvaluation:
        raise NotImplementedError


ECPIntegralEvaluator = Callable[
    [PreparedGaussianBasis, PreparedECP, ArrayLike, int], ECPIntegralEvaluation
]


class CallableECPIntegralProvider(AbstractECPIntegralProvider):
    evaluator: ECPIntegralEvaluator = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(self, evaluator: ECPIntegralEvaluator, provider_id: str, /):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        provider = str(provider_id).strip()
        if not provider:
            raise ValueError("provider_id must be non-empty.")
        self.evaluator = evaluator
        self.provider_id = provider

    def evaluate(
        self,
        basis: PreparedGaussianBasis,
        ecp: PreparedECP,
        positions: ArrayLike,
        /,
        *,
        derivative_order: int = 0,
    ) -> ECPIntegralEvaluation:
        result = self.evaluator(basis, ecp, positions, int(derivative_order))
        if not isinstance(result, ECPIntegralEvaluation):
            raise TypeError("ECP evaluator must return ECPIntegralEvaluation.")
        return result


__all__ = [
    "AbstractECPIntegralProvider",
    "CallableECPIntegralProvider",
    "ECPChannelPlan",
    "ECPGaussianTerm",
    "ECPIntegralEvaluation",
    "EffectiveCorePotentialPlan",
    "PreparedECP",
]
