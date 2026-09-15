#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed scalar impurity environments and causal hybridization evaluation."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...discretization.dlr import matsubara_frequencies
from ._thermal_green import GreenFunctionStatus


class HybridizationMoments(StrictModule):
    """First two Anderson-bath moments ``sum |V|² epsilon**k``."""

    zeroth: Array
    first: Array

    def __init__(self, zeroth: ArrayLike, first: ArrayLike, /):
        zeroth_ = jnp.asarray(zeroth)
        first_ = jnp.asarray(first)
        if zeroth_.shape != () or first_.shape != ():
            raise ValueError("Hybridization moments must be scalar.")
        if not np.isfinite(np.asarray(zeroth_)) or not np.isfinite(np.asarray(first_)):
            raise ValueError("Hybridization moments must be finite.")
        if float(zeroth_) < 0.0:
            raise ValueError("The zeroth hybridization moment must be non-negative.")
        self.zeroth = zeroth_
        self.first = first_


class HybridizationEvidence(StrictModule):
    """Separate Matsubara causality and asymptotic-moment evidence."""

    causality_residual: Array
    moment_residual: Array
    finite: Array
    causal: Array
    moments_valid: Array
    valid: Array
    status: Array


class MatsubaraHybridization(StrictModule):
    """Scalar fermionic hybridization samples on integer Matsubara labels."""

    indices: Array
    values: Array
    sample_active: Array
    moments: HybridizationMoments | None
    evidence: HybridizationEvidence
    beta: float = eqx.field(static=True)
    frequency_unit: str = eqx.field(static=True)
    environment_id: str = eqx.field(static=True)

    def __init__(
        self,
        beta: float,
        indices: ArrayLike,
        values: ArrayLike,
        /,
        *,
        sample_active: ArrayLike | None = None,
        moments: HybridizationMoments | None = None,
        moment_residual: ArrayLike = 0.0,
        causality_tolerance: float = 1e-10,
        moment_tolerance: float = 1e-8,
        frequency_unit: str = "native-energy",
        environment_id: str | None = None,
    ):
        beta_ = float(beta)
        if not isfinite(beta_) or beta_ <= 0.0:
            raise ValueError("beta must be finite and positive.")
        indices_ = jnp.asarray(indices)
        values_ = jnp.asarray(values)
        if indices_.ndim != 1 or not jnp.issubdtype(indices_.dtype, jnp.integer):
            raise TypeError("Hybridization indices must be a rank-one integer array.")
        if values_.shape != indices_.shape:
            raise ValueError("Hybridization values must match the Matsubara indices.")
        if sample_active is None:
            active = jnp.ones(indices_.shape, dtype=bool)
        else:
            active = jnp.asarray(sample_active, dtype=bool)
            if active.shape != indices_.shape:
                raise ValueError("sample_active must match the Matsubara indices.")
        if moments is not None and not isinstance(moments, HybridizationMoments):
            raise TypeError("moments must be HybridizationMoments or None.")
        moment_residual_ = jnp.asarray(moment_residual)
        if moment_residual_.shape != ():
            raise ValueError("moment_residual must be scalar.")
        causal_tolerance_ = float(causality_tolerance)
        moment_tolerance_ = float(moment_tolerance)
        if (
            not isfinite(causal_tolerance_)
            or causal_tolerance_ < 0.0
            or not isfinite(moment_tolerance_)
            or moment_tolerance_ < 0.0
        ):
            raise ValueError("Hybridization tolerances must be finite and non-negative.")
        unit = str(frequency_unit)
        if not unit:
            raise ValueError("frequency_unit must be non-empty.")
        frequency = matsubara_frequencies(indices_, beta=beta_, statistics="fermionic")
        causality_residual = jnp.max(
            jnp.where(
                active,
                jnp.maximum(jnp.sign(frequency) * jnp.imag(values_), 0.0),
                0.0,
            ),
            initial=0.0,
        )
        finite = (
            jnp.all(jnp.isfinite(values_) | ~active)
            & jnp.isfinite(causality_residual)
            & jnp.isfinite(moment_residual_)
        )
        causal = causality_residual <= causal_tolerance_
        moments_valid = moment_residual_ <= moment_tolerance_
        valid = finite & causal & moments_valid
        status = jnp.where(
            ~finite,
            int(GreenFunctionStatus.NONFINITE),
            jnp.where(
                valid,
                int(GreenFunctionStatus.SUCCESS),
                int(GreenFunctionStatus.RESIDUAL_TOO_LARGE),
            ),
        ).astype(jnp.int32)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "matsubara-hybridization",
                    "beta": beta_,
                    "unit": unit,
                    "samples": array_tree_fingerprint(
                        {"indices": indices_, "values": values_, "active": active}
                    ),
                }
            )
            if environment_id is None
            else str(environment_id)
        )
        if not identifier:
            raise ValueError("environment_id must be non-empty.")
        self.beta = beta_
        self.indices = indices_.astype(jnp.int32)
        self.values = values_
        self.sample_active = active
        self.moments = moments
        self.evidence = HybridizationEvidence(
            causality_residual,
            moment_residual_,
            finite,
            causal,
            moments_valid,
            valid,
            status,
        )
        self.frequency_unit = unit
        self.environment_id = identifier

    @property
    def frequencies(self) -> Array:
        return matsubara_frequencies(self.indices, beta=self.beta, statistics="fermionic")

    @property
    def valid(self) -> Array:
        return self.evidence.valid


class AndersonBath(StrictModule):
    """Finite scalar Anderson bath with real levels and complex couplings."""

    site_energies: Array
    couplings: Array
    moments: HybridizationMoments
    frequency_unit: str = eqx.field(static=True)
    bath_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_energies: ArrayLike,
        couplings: ArrayLike,
        /,
        *,
        frequency_unit: str = "native-energy",
        bath_id: str | None = None,
    ):
        energies = jnp.asarray(site_energies)
        couplings_ = jnp.asarray(couplings)
        if energies.ndim != 1 or couplings_.shape != energies.shape:
            raise ValueError(
                "Anderson bath energies and couplings must be matching vectors."
            )
        if jnp.issubdtype(energies.dtype, jnp.complexfloating):
            raise TypeError("Anderson bath site energies must be real.")
        if not np.all(np.isfinite(np.asarray(energies))) or not np.all(
            np.isfinite(np.asarray(couplings_))
        ):
            raise ValueError("Anderson bath parameters must be finite.")
        unit = str(frequency_unit)
        if not unit:
            raise ValueError("frequency_unit must be non-empty.")
        strengths = jnp.abs(couplings_) ** 2
        moments = HybridizationMoments(jnp.sum(strengths), jnp.sum(strengths * energies))
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "anderson-bath",
                    "unit": unit,
                    "parameters": array_tree_fingerprint(
                        {"energies": energies, "couplings": couplings_}
                    ),
                }
            )
            if bath_id is None
            else str(bath_id)
        )
        if not identifier:
            raise ValueError("bath_id must be non-empty.")
        self.site_energies = energies
        self.couplings = couplings_
        self.moments = moments
        self.frequency_unit = unit
        self.bath_id = identifier

    @property
    def site_count(self) -> int:
        return int(self.site_energies.shape[0])


class ImpurityEnvironment(StrictModule):
    """Exclusive environment: sampled hybridization or finite Anderson bath."""

    hybridization: MatsubaraHybridization | None
    bath: AndersonBath | None
    environment_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        hybridization: MatsubaraHybridization | None = None,
        bath: AndersonBath | None = None,
    ):
        if (hybridization is None) == (bath is None):
            raise ValueError(
                "ImpurityEnvironment requires exactly one of hybridization or bath."
            )
        if hybridization is not None and not isinstance(
            hybridization, MatsubaraHybridization
        ):
            raise TypeError("hybridization must be MatsubaraHybridization or None.")
        if bath is not None and not isinstance(bath, AndersonBath):
            raise TypeError("bath must be AndersonBath or None.")
        self.hybridization = hybridization
        self.bath = bath
        self.environment_id = (
            hybridization.environment_id if hybridization is not None else bath.bath_id
        )


def evaluate_anderson_hybridization(bath: AndersonBath, frequency: ArrayLike, /) -> Array:
    """Evaluate ``Delta(z)=sum_l |V_l|²/(z-epsilon_l)``."""

    if not isinstance(bath, AndersonBath):
        raise TypeError("bath must be an AndersonBath.")
    z = jnp.asarray(frequency)
    denominator = z[..., None] - bath.site_energies
    return jnp.sum(jnp.abs(bath.couplings) ** 2 / denominator, axis=-1)


def anderson_bath_to_matsubara(
    bath: AndersonBath,
    beta: float,
    indices: ArrayLike,
    /,
    *,
    sample_active: ArrayLike | None = None,
) -> MatsubaraHybridization:
    """Materialize the exact causal bath hybridization on Matsubara labels."""

    labels = jnp.asarray(indices)
    frequency = matsubara_frequencies(labels, beta=beta, statistics="fermionic")
    values = evaluate_anderson_hybridization(bath, 1j * frequency)
    return MatsubaraHybridization(
        beta,
        labels,
        values,
        sample_active=sample_active,
        moments=bath.moments,
        frequency_unit=bath.frequency_unit,
        environment_id=canonical_fingerprint(
            {
                "kind": "anderson-bath-matsubara",
                "bath": bath.bath_id,
                "beta": float(beta),
                "indices": array_tree_fingerprint(labels),
            }
        ),
    )


__all__ = [
    "AndersonBath",
    "HybridizationEvidence",
    "HybridizationMoments",
    "ImpurityEnvironment",
    "MatsubaraHybridization",
    "anderson_bath_to_matsubara",
    "evaluate_anderson_hybridization",
]
