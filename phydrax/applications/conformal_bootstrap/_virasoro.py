#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite BPZ-hypergeometric and exact Ising sigma Virasoro blocks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule


IsingSigmaChannel: TypeAlias = Literal["identity", "energy"]


def _complex_payload(value: complex, /) -> tuple[float, float]:
    return float(value.real), float(value.imag)


def _hyp2f1_series(
    first: complex,
    second: complex,
    third: complex,
    points: Array,
    order: int,
    pole_tolerance: float,
    /,
) -> tuple[Array, Array, Array]:
    a = jnp.asarray(first, dtype=jnp.complex128)
    b = jnp.asarray(second, dtype=jnp.complex128)
    c = jnp.asarray(third, dtype=jnp.complex128)
    initial = (
        jnp.ones(points.shape, dtype=jnp.complex128),
        jnp.ones(points.shape, dtype=jnp.complex128),
    )

    def body(index, carry):
        term, value = carry
        denominator = (c + index) * (index + 1.0)
        denominator = eqx.error_if(
            denominator,
            jnp.abs(denominator) <= pole_tolerance,
            "BPZ hypergeometric denominator reaches a retained pole.",
        )
        next_term = term * (a + index) * (b + index) / denominator * points
        return next_term, value + next_term

    term, value = jax.lax.fori_loop(0, order - 1, body, initial)
    tail = jnp.abs(term) * points / jnp.maximum(1e-30, 1.0 - points)
    return value, term, tail


def _elliptic_k_agm(parameter: Array, iterations: int = 12, /) -> Array:
    first = jnp.ones_like(parameter)
    second = jnp.sqrt(1.0 - parameter)
    for _ in range(iterations):
        first, second = 0.5 * (first + second), jnp.sqrt(first * second)
    return jnp.pi / (2.0 * first)


def elliptic_nome(cross_ratio: ArrayLike, /) -> Array:
    points = jnp.asarray(cross_ratio, dtype=jnp.float64)
    if points.ndim == 0:
        points = points.reshape((1,))
    points = eqx.error_if(
        points,
        jnp.any((points <= 0.0) | (points >= 1.0)),
        "Elliptic nome requires real cross ratios in (0, 1).",
    )
    return jnp.exp(-jnp.pi * _elliptic_k_agm(1.0 - points) / _elliptic_k_agm(points))


class BPZVirasoroBlockPlan(StrictModule):
    """Caller-derived second-order BPZ block in one explicit branch convention."""

    cross_ratios: Array
    central_charge: float = eqx.field(static=True)
    external_weights: tuple[float, float, float, float] = eqx.field(static=True)
    internal_weight: float = eqx.field(static=True)
    z_exponent: complex = eqx.field(static=True)
    one_minus_z_exponent: complex = eqx.field(static=True)
    hypergeometric_parameters: tuple[complex, complex, complex] = eqx.field(static=True)
    series_order: int = eqx.field(static=True)
    pole_tolerance: float = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)
    derivation_source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cross_ratios: ArrayLike,
        /,
        *,
        central_charge: float,
        external_weights: Sequence[float],
        internal_weight: float,
        z_exponent: complex,
        one_minus_z_exponent: complex,
        hypergeometric_parameters: Sequence[complex],
        series_order: int = 256,
        pole_tolerance: float = 1e-12,
        branch_id: str,
        derivation_source_id: str,
    ):
        points = np.asarray(cross_ratios, dtype=float)
        charge = float(central_charge)
        weights = tuple(float(value) for value in external_weights)
        internal = float(internal_weight)
        parameters = tuple(complex(value) for value in hypergeometric_parameters)
        order = int(series_order)
        tolerance = float(pole_tolerance)
        branch = str(branch_id)
        source = str(derivation_source_id)
        if points.ndim != 1 or points.size == 0 or not np.all(np.isfinite(points)):
            raise ValueError("cross_ratios must be one nonempty finite vector.")
        if np.any((points <= 0.0) | (points >= 1.0)):
            raise ValueError("BPZ cross ratios must lie in (0, 1).")
        if not np.isfinite(charge) or charge <= 0.0:
            raise ValueError("central_charge must be finite and positive.")
        if len(weights) != 4 or any(
            not np.isfinite(value) or value < 0.0 for value in weights
        ):
            raise ValueError(
                "external_weights must contain four finite nonnegative values."
            )
        if not np.isfinite(internal) or internal < 0.0:
            raise ValueError("internal_weight must be finite and nonnegative.")
        if len(parameters) != 3 or not all(np.isfinite(value) for value in parameters):
            raise ValueError("Three finite hypergeometric parameters are required.")
        if not np.isfinite(z_exponent) or not np.isfinite(one_minus_z_exponent):
            raise ValueError("BPZ prefactor exponents must be finite.")
        if order < 2 or not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("BPZ series order or pole tolerance is invalid.")
        if not branch or not source:
            raise ValueError("BPZ branch and derivation source identities are required.")
        self.cross_ratios = jnp.asarray(points)
        self.central_charge = charge
        self.external_weights = weights
        self.internal_weight = internal
        self.z_exponent = complex(z_exponent)
        self.one_minus_z_exponent = complex(one_minus_z_exponent)
        self.hypergeometric_parameters = parameters
        self.series_order = order
        self.pole_tolerance = tolerance
        self.branch_id = branch
        self.derivation_source_id = source
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bpz-degenerate-virasoro-block-plan",
                "cross_ratios": array_tree_fingerprint(points),
                "central_charge": charge,
                "external_weights": weights,
                "internal_weight": internal,
                "z_exponent": _complex_payload(complex(z_exponent)),
                "one_minus_z_exponent": _complex_payload(complex(one_minus_z_exponent)),
                "hypergeometric_parameters": tuple(
                    _complex_payload(value) for value in parameters
                ),
                "series_order": order,
                "pole_tolerance": tolerance,
                "branch_id": branch,
                "derivation_source_id": source,
            }
        )


class VirasoroBlockEvidence(StrictModule):
    values: Array
    final_terms: Array
    tail_proxies: Array
    elliptic_nomes: Array
    finite: Array
    plan_id: str = eqx.field(static=True)
    block_family: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class PreparedBPZVirasoroBlocks(StrictModule):
    plan: BPZVirasoroBlockPlan
    prepared_id: str = eqx.field(static=True)

    def block(self, /) -> tuple[Array, Array, Array]:
        hypergeometric, final, tail = _hyp2f1_series(
            *self.plan.hypergeometric_parameters,
            self.plan.cross_ratios,
            self.plan.series_order,
            self.plan.pole_tolerance,
        )
        prefactor = (
            self.plan.cross_ratios**self.plan.z_exponent
            * (1.0 - self.plan.cross_ratios) ** self.plan.one_minus_z_exponent
        )
        return prefactor * hypergeometric, prefactor * final, jnp.abs(prefactor) * tail

    def evidence(self, /) -> VirasoroBlockEvidence:
        values, final, tails = self.block()
        finite = (
            jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(final))
            & jnp.all(jnp.isfinite(tails))
        )
        return VirasoroBlockEvidence(
            values=values,
            final_terms=final,
            tail_proxies=tails,
            elliptic_nomes=elliptic_nome(self.plan.cross_ratios),
            finite=finite,
            plan_id=self.plan.plan_id,
            block_family="second-order-bpz-hypergeometric",
            claim="finite-caller-derived-degenerate-virasoro-block-branch",
        )


def prepare_bpz_virasoro_blocks(
    plan: BPZVirasoroBlockPlan, /
) -> PreparedBPZVirasoroBlocks:
    if not isinstance(plan, BPZVirasoroBlockPlan):
        raise TypeError("plan must be BPZVirasoroBlockPlan.")
    return PreparedBPZVirasoroBlocks(
        plan=plan,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-bpz-virasoro-blocks",
                "plan": plan.plan_id,
                "method": "finite-hypergeometric-series",
            }
        ),
    )


class IsingSigmaVirasoroPlan(StrictModule):
    """Exact c=1/2 four-sigma holomorphic Virasoro block branch."""

    cross_ratios: Array
    channel: IsingSigmaChannel = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, cross_ratios: ArrayLike, channel: IsingSigmaChannel, /):
        points = np.asarray(cross_ratios, dtype=float)
        channel_value = str(channel)
        if points.ndim != 1 or points.size == 0 or not np.all(np.isfinite(points)):
            raise ValueError("cross_ratios must be one nonempty finite vector.")
        if np.any((points <= 0.0) | (points >= 1.0)):
            raise ValueError("Ising sigma cross ratios must lie in (0, 1).")
        if channel_value not in {"identity", "energy"}:
            raise ValueError("Ising sigma channel must be identity or energy.")
        self.cross_ratios = jnp.asarray(points)
        self.channel = channel_value
        self.plan_id = canonical_fingerprint(
            {
                "kind": "exact-ising-sigma-virasoro-block-plan",
                "cross_ratios": array_tree_fingerprint(points),
                "channel": channel_value,
                "central_charge": 0.5,
                "external_weight": 1.0 / 16.0,
                "internal_weight": 0.0 if channel_value == "identity" else 0.5,
                "branch": "principal-real-0-less-z-less-1",
            }
        )


class PreparedIsingSigmaVirasoroBlocks(StrictModule):
    plan: IsingSigmaVirasoroPlan
    prepared_id: str = eqx.field(static=True)

    def block(self, /) -> Array:
        points = self.plan.cross_ratios
        inner = 1.0 + jnp.sqrt(1.0 - points)
        if self.plan.channel == "energy":
            inner = 1.0 - jnp.sqrt(1.0 - points)
        return jnp.sqrt(inner) / jnp.sqrt(2.0) / (points * (1.0 - points)) ** (1.0 / 8.0)

    def evidence(self, /) -> VirasoroBlockEvidence:
        values = self.block()
        zeros = jnp.zeros_like(values)
        return VirasoroBlockEvidence(
            values=values,
            final_terms=zeros,
            tail_proxies=zeros,
            elliptic_nomes=elliptic_nome(self.plan.cross_ratios),
            finite=jnp.all(jnp.isfinite(values)),
            plan_id=self.plan.plan_id,
            block_family="exact-ising-c-half-four-sigma",
            claim="exact-degenerate-ising-virasoro-block-on-principal-real-branch",
        )


def prepare_ising_sigma_virasoro_blocks(
    plan: IsingSigmaVirasoroPlan, /
) -> PreparedIsingSigmaVirasoroBlocks:
    if not isinstance(plan, IsingSigmaVirasoroPlan):
        raise TypeError("plan must be IsingSigmaVirasoroPlan.")
    return PreparedIsingSigmaVirasoroBlocks(
        plan=plan,
        prepared_id=canonical_fingerprint(
            {"kind": "prepared-exact-ising-sigma-block", "plan": plan.plan_id}
        ),
    )


class VirasoroCrossingEvidence(StrictModule):
    direct_correlator: Array
    crossed_correlator: Array
    pointwise_residual: Array
    maximum_residual: Array
    finite: Array
    accepted: Array
    crossing_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def ising_sigma_crossing_evidence(
    cross_ratios: ArrayLike,
    /,
    *,
    tolerance: float = 1e-10,
) -> VirasoroCrossingEvidence:
    points = jnp.asarray(cross_ratios, dtype=jnp.float64)
    identity = prepare_ising_sigma_virasoro_blocks(
        IsingSigmaVirasoroPlan(points, "identity")
    ).block()
    energy = prepare_ising_sigma_virasoro_blocks(
        IsingSigmaVirasoroPlan(points, "energy")
    ).block()
    reflected_points = 1.0 - points
    reflected_identity = prepare_ising_sigma_virasoro_blocks(
        IsingSigmaVirasoroPlan(reflected_points, "identity")
    ).block()
    reflected_energy = prepare_ising_sigma_virasoro_blocks(
        IsingSigmaVirasoroPlan(reflected_points, "energy")
    ).block()
    direct = jnp.abs(identity) ** 2 + jnp.abs(energy) ** 2
    crossed = jnp.abs(reflected_identity) ** 2 + jnp.abs(reflected_energy) ** 2
    residual = jnp.abs(direct - crossed) / jnp.maximum(1.0, jnp.abs(direct))
    maximum = jnp.max(residual)
    finite = jnp.all(jnp.isfinite(direct)) & jnp.all(jnp.isfinite(crossed))
    crossing_id = canonical_fingerprint(
        {
            "kind": "ising-sigma-virasoro-crossing-evidence",
            "cross_ratios": array_tree_fingerprint(np.asarray(points)),
            "channels": ("identity", "energy"),
            "ope_coefficient_squares": (1.0, 1.0),
        }
    )
    return VirasoroCrossingEvidence(
        direct_correlator=direct,
        crossed_correlator=crossed,
        pointwise_residual=residual,
        maximum_residual=maximum,
        finite=finite,
        accepted=finite & (maximum <= float(tolerance)),
        crossing_id=crossing_id,
        claim="exact-finite-ising-four-sigma-crossing-reference-not-general-2d-bootstrap",
    )


__all__ = [
    "BPZVirasoroBlockPlan",
    "IsingSigmaChannel",
    "IsingSigmaVirasoroPlan",
    "PreparedBPZVirasoroBlocks",
    "PreparedIsingSigmaVirasoroBlocks",
    "VirasoroBlockEvidence",
    "VirasoroCrossingEvidence",
    "elliptic_nome",
    "ising_sigma_crossing_evidence",
    "prepare_bpz_virasoro_blocks",
    "prepare_ising_sigma_virasoro_blocks",
]
