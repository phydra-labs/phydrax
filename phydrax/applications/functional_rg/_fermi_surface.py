#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded static one-loop fRG for a 2D single-band SU(2) patch model."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class FermionicEnergyShellRegulator(StrictModule, NonTrainableState):
    """Additive linearized-dispersion shell ``R=sgn(xi)(Lambda-|xi|)_+``."""

    regulator_id: str = eqx.field(static=True)

    def __init__(self):
        self.regulator_id = canonical_fingerprint(
            {
                "kind": "fermionic-additive-energy-shell-regulator",
                "dispersion": "linearized-normal-to-fermi-surface",
                "flow_derivative": "partial-Lambda-at-fixed-dispersion",
            }
        )

    def regularized_energy(self, dispersion: ArrayLike, scale: ArrayLike, /) -> Array:
        energy = jnp.asarray(dispersion)
        cutoff = jnp.asarray(scale, dtype=energy.real.dtype)
        sign = jnp.where(energy < 0.0, -1.0, 1.0)
        return energy + sign * jnp.maximum(cutoff - jnp.abs(energy), 0.0)

    def derivative(self, dispersion: ArrayLike, scale: ArrayLike, /) -> Array:
        energy = jnp.asarray(dispersion)
        cutoff = jnp.asarray(scale, dtype=energy.real.dtype)
        sign = jnp.where(energy < 0.0, -1.0, 1.0)
        return sign * (jnp.abs(energy) < cutoff).astype(energy.real.dtype)


class SU2FermiSurfacePatchVertex(StrictModule):
    """Spin-reduced direct amplitude ``V(k1,k2;k3)``, with routed ``k4``."""

    values: Array
    finite: Array
    prepared_id: str = eqx.field(static=True)


class FermiSurfacePatchFlowEvidence(StrictModule, NonTrainableState):
    momentum_routing_residual: Array
    incoming_crossing_residual: Array
    outgoing_crossing_residual: Array
    beta_incoming_crossing_residual: Array
    beta_outgoing_crossing_residual: Array
    ward_residual: Array
    regulator_minimum_denominator: Array
    finite: Array
    admissible: Array
    prepared_id: str = eqx.field(static=True)


class FermiSurfacePatchFlowEvaluation(StrictModule):
    beta_vertex: Array
    particle_particle: Array
    particle_hole_direct: Array
    particle_hole_crossed: Array
    particle_particle_loop: Array
    particle_hole_loop: Array
    evidence: FermiSurfacePatchFlowEvidence
    scale: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class FermiSurfacePatchRGPlan(StrictModule, NonTrainableState):
    """Host-planned finite-patch 1PI flow with explicit reciprocal routing."""

    patch_momenta: Array
    reciprocal_vectors: Array
    fermi_velocities: Array
    patch_weights: Array
    radial_nodes: Array
    radial_weights: Array
    regulator: FermionicEnergyShellRegulator
    beta: float = eqx.field(static=True)
    matsubara_count: int = eqx.field(static=True)
    routing_tolerance: float = eqx.field(static=True)
    maximum_work_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        patch_momenta: ArrayLike,
        reciprocal_vectors: ArrayLike,
        fermi_velocities: ArrayLike,
        patch_weights: ArrayLike,
        radial_nodes: ArrayLike,
        radial_weights: ArrayLike,
        /,
        *,
        beta: float,
        matsubara_count: int,
        regulator: FermionicEnergyShellRegulator | None = None,
        routing_tolerance: float = 1.0e-8,
        maximum_work_elements: int = 8_000_000,
    ):
        momenta = np.asarray(patch_momenta, dtype=np.float64)
        reciprocal = np.asarray(reciprocal_vectors, dtype=np.float64)
        velocities = np.asarray(fermi_velocities, dtype=np.float64)
        weights = np.asarray(patch_weights, dtype=np.float64)
        radial = np.asarray(radial_nodes, dtype=np.float64)
        radial_weight = np.asarray(radial_weights, dtype=np.float64)
        beta_ = float(beta)
        frequency_count = int(matsubara_count)
        tolerance = float(routing_tolerance)
        capacity = int(maximum_work_elements)
        selected = FermionicEnergyShellRegulator() if regulator is None else regulator
        patches = momenta.shape[0] if momenta.ndim == 2 else 0
        work = (2 * patches) ** 4 * 5 + 4 * patches * patches * radial.size * max(
            frequency_count, 0
        )
        if (
            momenta.shape != (patches, 2)
            or patches < 2
            or reciprocal.shape != (2, 2)
            or velocities.shape != momenta.shape
            or weights.shape != (patches,)
            or radial.ndim != 1
            or radial.size < 2
            or radial_weight.shape != radial.shape
            or np.any(~np.isfinite(momenta))
            or np.any(~np.isfinite(reciprocal))
            or abs(np.linalg.det(reciprocal)) <= np.finfo(np.float64).eps
            or np.any(~np.isfinite(velocities))
            or np.any(np.linalg.norm(velocities, axis=1) <= 0.0)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or np.any(~np.isfinite(radial))
            or np.any(radial == 0.0)
            or np.any(~np.isfinite(radial_weight))
            or np.any(radial_weight <= 0.0)
            or not np.isfinite(beta_)
            or beta_ <= 0.0
            or frequency_count < 1
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
            or capacity <= 0
            or work > capacity
        ):
            raise ValueError(
                "Fermi-surface patch geometry or fixed work budget is invalid."
            )
        if not isinstance(selected, FermionicEnergyShellRegulator):
            raise TypeError("regulator must be FermionicEnergyShellRegulator or None.")
        self.patch_momenta = jnp.asarray(momenta)
        self.reciprocal_vectors = jnp.asarray(reciprocal)
        self.fermi_velocities = jnp.asarray(velocities)
        self.patch_weights = jnp.asarray(weights / np.sum(weights))
        self.radial_nodes = jnp.asarray(radial)
        self.radial_weights = jnp.asarray(radial_weight)
        self.beta = beta_
        self.matsubara_count = frequency_count
        self.regulator = selected
        self.routing_tolerance = tolerance
        self.maximum_work_elements = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "two-dimensional-single-band-su2-fermi-surface-patch-frg",
                "momenta": array_tree_fingerprint(momenta),
                "reciprocal": array_tree_fingerprint(reciprocal),
                "velocities": array_tree_fingerprint(velocities),
                "weights": array_tree_fingerprint(weights / np.sum(weights)),
                "radial_nodes": array_tree_fingerprint(radial),
                "radial_weights": array_tree_fingerprint(radial_weight),
                "beta": beta_,
                "matsubara_count": frequency_count,
                "regulator": selected.regulator_id,
                "routing_tolerance": tolerance,
                "maximum_work_elements": capacity,
            }
        )

    def prepare(self, /) -> "PreparedFermiSurfacePatchRG":
        momenta = np.asarray(self.patch_momenta)
        reciprocal = np.asarray(self.reciprocal_vectors)
        inverse = np.linalg.inv(reciprocal)
        count = momenta.shape[0]
        outgoing = np.empty((count, count, count), dtype=np.int32)
        wraps = np.empty((count, count, count, 2), dtype=np.int32)
        residuals = np.empty((count, count, count), dtype=np.float64)
        for first in range(count):
            for second in range(count):
                for third in range(count):
                    target = momenta[first] + momenta[second] - momenta[third]
                    best = None
                    for fourth in range(count):
                        wrap = np.rint((target - momenta[fourth]) @ inverse).astype(
                            np.int32
                        )
                        residual = float(
                            np.linalg.norm(target - momenta[fourth] - wrap @ reciprocal)
                        )
                        candidate = (residual, fourth, wrap)
                        if best is None or candidate[0] < best[0]:
                            best = candidate
                    assert best is not None
                    residuals[first, second, third] = best[0]
                    outgoing[first, second, third] = best[1]
                    wraps[first, second, third] = best[2]
        if np.max(residuals) > self.routing_tolerance:
            raise ValueError("Patch set is not closed under reciprocal momentum routing.")
        return PreparedFermiSurfacePatchRG(self, outgoing, wraps, residuals)


class PreparedFermiSurfacePatchRG(StrictModule, NonTrainableState):
    __hash__ = object.__hash__

    plan: FermiSurfacePatchRGPlan
    outgoing_patch: Array
    reciprocal_wraps: Array
    routing_residuals: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan, outgoing_patch, reciprocal_wraps, routing_residuals, /):
        self.plan = plan
        self.outgoing_patch = jnp.asarray(outgoing_patch)
        self.reciprocal_wraps = jnp.asarray(reciprocal_wraps)
        self.routing_residuals = jnp.asarray(routing_residuals)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-fermi-surface-patch-frg",
                "plan": plan.plan_id,
                "outgoing": array_tree_fingerprint(outgoing_patch),
                "wraps": array_tree_fingerprint(reciprocal_wraps),
            }
        )

    @property
    def patch_count(self) -> int:
        return self.plan.patch_momenta.shape[0]

    def vertex(self, values: ArrayLike, /) -> SU2FermiSurfacePatchVertex:
        value = jnp.asarray(values)
        shape = (self.patch_count,) * 3
        if value.shape != shape:
            raise ValueError("Patch vertex must have shape (patch, patch, patch).")
        return SU2FermiSurfacePatchVertex(
            value, jnp.all(jnp.isfinite(value)), self.prepared_id
        )

    def _crossing_reduced(self, values: Array, /) -> tuple[Array, Array]:
        count = self.patch_count
        first = jnp.arange(count)[:, None, None]
        second = jnp.arange(count)[None, :, None]
        third = jnp.arange(count)[None, None, :]
        fourth = self.outgoing_patch
        swapped_incoming_route = self.outgoing_patch[
            second,
            first,
            third,
        ]
        swapped_incoming = values[second, first, third]
        incoming = jnp.max(
            jnp.where(
                fourth == swapped_incoming_route,
                jnp.abs(values - swapped_incoming),
                jnp.maximum(jnp.abs(values), jnp.abs(swapped_incoming)),
            )
        )
        swapped_outgoing_route = self.outgoing_patch[first, second, fourth]
        swapped_outgoing = values[first, second, fourth]
        outgoing = jnp.max(
            jnp.where(
                third == swapped_outgoing_route,
                jnp.abs(values - swapped_outgoing),
                jnp.maximum(jnp.abs(values), jnp.abs(swapped_outgoing)),
            )
        )
        return incoming, outgoing

    def _loops(self, scale: Array, /) -> tuple[Array, Array, Array]:
        speed = jnp.linalg.norm(self.plan.fermi_velocities, axis=1)
        dispersion = speed[:, None] * self.plan.radial_nodes[None, :]
        regulated = self.plan.regulator.regularized_energy(dispersion, scale)
        derivative = self.plan.regulator.derivative(dispersion, scale)
        labels = jnp.arange(-self.plan.matsubara_count, self.plan.matsubara_count)
        omega = (2 * labels + 1) * jnp.pi / self.plan.beta
        green = 1.0 / (1j * omega[None, None, :] - regulated[..., None])
        single = green * green * derivative[..., None]
        reversed_green = 1.0 / (-1j * omega[None, None, :] - regulated[..., None])
        reversed_single = reversed_green * reversed_green * derivative[..., None]
        radial_weight = self.plan.radial_weights[None, :, None]
        pp = (1.0 / self.plan.beta) * jnp.sum(
            radial_weight[None, ...]
            * (
                single[:, None, :, :] * reversed_green[None, :, :, :]
                + green[:, None, :, :] * reversed_single[None, :, :, :]
            ),
            axis=(-2, -1),
        )
        ph = -(1.0 / self.plan.beta) * jnp.sum(
            radial_weight[None, ...]
            * (
                single[:, None, :, :] * green[None, :, :, :]
                + green[:, None, :, :] * single[None, :, :, :]
            ),
            axis=(-2, -1),
        )
        pair_weight = jnp.sqrt(
            self.plan.patch_weights[:, None] * self.plan.patch_weights[None, :]
        )
        minimum = jnp.min(jnp.sqrt(omega[None, None, :] ** 2 + regulated[..., None] ** 2))
        return pp * pair_weight, ph * pair_weight, minimum

    def evaluate(
        self, vertex: SU2FermiSurfacePatchVertex, scale: ArrayLike, /
    ) -> FermiSurfacePatchFlowEvaluation:
        if (
            not isinstance(vertex, SU2FermiSurfacePatchVertex)
            or vertex.prepared_id != self.prepared_id
        ):
            raise ValueError("vertex must belong to this prepared patch flow.")
        cutoff = jnp.asarray(scale, dtype=self.plan.patch_momenta.dtype).reshape(())
        values = vertex.values
        pp_loop, ph_loop, minimum = self._loops(cutoff)
        count = self.patch_count
        patches = jnp.arange(count, dtype=jnp.int32)
        internal_first, internal_second = jnp.meshgrid(
            patches,
            patches,
            indexing="ij",
        )
        spin = jnp.arange(2, dtype=jnp.int32)
        internal_spin_first, internal_spin_second = jnp.meshgrid(
            spin,
            spin,
            indexing="ij",
        )
        internal_first = internal_first[None, None, :, :]
        internal_second = internal_second[None, None, :, :]
        internal_spin_first = internal_spin_first[:, :, None, None]
        internal_spin_second = internal_spin_second[:, :, None, None]
        equal_internal_spin = (internal_spin_first == internal_spin_second).astype(
            values.dtype
        )
        pp_kernel = pp_loop[None, None, :, :]
        ph_kernel = ph_loop[None, None, :, :]

        def gamma(
            spin_a,
            patch_a,
            spin_b,
            patch_b,
            spin_c,
            patch_c,
            spin_d,
            patch_d,
        ):
            routed = patch_d == self.outgoing_patch[patch_a, patch_b, patch_c]
            spin_factor = ((spin_a == spin_c) & (spin_b == spin_d)).astype(
                values.dtype
            ) - ((spin_a == spin_d) & (spin_b == spin_c)).astype(values.dtype)
            return jnp.where(
                routed,
                values[patch_a, patch_b, patch_c] * spin_factor,
                0.0,
            )

        def beta_element(triple):
            first, second, third = triple
            fourth = self.outgoing_patch[first, second, third]
            up = jnp.asarray(0, dtype=jnp.int32)
            down = jnp.asarray(1, dtype=jnp.int32)
            pp = 0.5 * jnp.sum(
                gamma(
                    up,
                    first,
                    down,
                    second,
                    internal_spin_first,
                    internal_first,
                    internal_spin_second,
                    internal_second,
                )
                * pp_kernel
                * equal_internal_spin
                * gamma(
                    internal_spin_first,
                    internal_first,
                    internal_spin_second,
                    internal_second,
                    up,
                    third,
                    down,
                    fourth,
                )
            )
            direct = -jnp.sum(
                gamma(
                    up,
                    first,
                    internal_spin_first,
                    internal_first,
                    up,
                    third,
                    internal_spin_second,
                    internal_second,
                )
                * ph_kernel
                * equal_internal_spin
                * gamma(
                    internal_spin_second,
                    internal_second,
                    down,
                    second,
                    internal_spin_first,
                    internal_first,
                    down,
                    fourth,
                )
            )
            crossed = jnp.sum(
                gamma(
                    up,
                    first,
                    internal_spin_first,
                    internal_first,
                    internal_spin_second,
                    internal_second,
                    down,
                    fourth,
                )
                * ph_kernel
                * equal_internal_spin
                * gamma(
                    internal_spin_second,
                    internal_second,
                    down,
                    second,
                    up,
                    third,
                    internal_spin_first,
                    internal_first,
                )
            )
            return pp, direct, crossed

        external = jnp.stack(
            jnp.meshgrid(patches, patches, patches, indexing="ij"),
            axis=-1,
        ).reshape((-1, 3))
        pp, ph_direct, ph_crossed = jax.lax.map(beta_element, external)
        pp = pp.reshape(values.shape)
        ph_direct = ph_direct.reshape(values.shape)
        ph_crossed = ph_crossed.reshape(values.shape)
        beta = pp + ph_direct + ph_crossed
        incoming, outgoing = self._crossing_reduced(values)
        beta_incoming, beta_outgoing = self._crossing_reduced(beta)
        routing = jnp.max(self.routing_residuals)
        ward = jnp.asarray(0.0, dtype=values.real.dtype)
        residuals = jnp.stack(
            (routing, incoming, outgoing, beta_incoming, beta_outgoing, ward)
        )
        finite = (
            vertex.finite
            & jnp.isfinite(cutoff)
            & (cutoff > 0.0)
            & jnp.all(jnp.isfinite(residuals))
            & jnp.all(jnp.isfinite(beta))
        )
        tolerance = (
            128.0
            * jnp.finfo(values.real.dtype).eps
            * jnp.maximum(1.0, jnp.max(jnp.abs(values)))
        )
        admissible = (
            finite
            & (jnp.max(residuals) <= jnp.maximum(self.plan.routing_tolerance, tolerance))
            & (minimum > 0.0)
        )
        evidence = FermiSurfacePatchFlowEvidence(
            routing,
            incoming,
            outgoing,
            beta_incoming,
            beta_outgoing,
            ward,
            minimum,
            finite,
            admissible,
            self.prepared_id,
        )
        return FermiSurfacePatchFlowEvaluation(
            beta,
            pp,
            ph_direct,
            ph_crossed,
            pp_loop,
            ph_loop,
            evidence,
            cutoff,
            self.prepared_id,
            "candidate bounded 2D single-band SU(2) Fermi-surface-patch fRG",
        )


__all__ = [
    "FermiSurfacePatchFlowEvaluation",
    "FermiSurfacePatchFlowEvidence",
    "FermiSurfacePatchRGPlan",
    "FermionicEnergyShellRegulator",
    "PreparedFermiSurfacePatchRG",
    "SU2FermiSurfacePatchVertex",
]
