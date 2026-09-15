#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded static one-loop fRG for a 2D single-band SU(2) patch model."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract


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
        momenta = np.asarray(patch_momenta, dtype=float)
        reciprocal = np.asarray(reciprocal_vectors, dtype=float)
        velocities = np.asarray(fermi_velocities, dtype=float)
        weights = np.asarray(patch_weights, dtype=float)
        radial = np.asarray(radial_nodes, dtype=float)
        radial_weight = np.asarray(radial_weights, dtype=float)
        beta_ = float(beta)
        frequency_count = int(matsubara_count)
        tolerance = float(routing_tolerance)
        capacity = int(maximum_work_elements)
        selected = FermionicEnergyShellRegulator() if regulator is None else regulator
        patches = int(momenta.shape[0]) if momenta.ndim == 2 else 0
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
            or abs(np.linalg.det(reciprocal)) <= np.finfo(float).eps
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
        residuals = np.empty((count, count, count), dtype=float)
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
        return int(self.plan.patch_momenta.shape[0])

    def vertex(self, values: ArrayLike, /) -> SU2FermiSurfacePatchVertex:
        value = jnp.asarray(values)
        shape = (self.patch_count,) * 3
        if value.shape != shape:
            raise ValueError("Patch vertex must have shape (patch, patch, patch).")
        return SU2FermiSurfacePatchVertex(
            value, jnp.all(jnp.isfinite(value)), self.prepared_id
        )

    def _spin_vertex(self, values: Array, /) -> Array:
        count = self.patch_count
        direct = jnp.zeros((count, count, count, count), dtype=values.dtype)
        for first in range(count):
            for second in range(count):
                for third in range(count):
                    fourth = int(self.outgoing_patch[first, second, third])
                    direct = direct.at[first, second, third, fourth].set(
                        values[first, second, third]
                    )
        spin_identity = jnp.eye(2, dtype=values.dtype)
        direct_spin = (
            direct[..., None, None, None, None]
            * spin_identity[None, None, None, None, :, None, :, None]
            * spin_identity[None, None, None, None, None, :, None, :]
        )
        gamma = direct_spin - jnp.swapaxes(jnp.swapaxes(direct_spin, 2, 3), 6, 7)
        return jnp.transpose(gamma, (0, 4, 1, 5, 2, 6, 3, 7)).reshape((2 * count,) * 4)

    def _project(self, spin_vertex: Array, /) -> Array:
        count = self.patch_count
        gamma = spin_vertex.reshape((count, 2, count, 2, count, 2, count, 2))
        values = jnp.zeros((count, count, count), dtype=spin_vertex.dtype)
        for first in range(count):
            for second in range(count):
                for third in range(count):
                    fourth = int(self.outgoing_patch[first, second, third])
                    values = values.at[first, second, third].set(
                        gamma[first, 0, second, 1, third, 0, fourth, 1]
                    )
        return values

    @staticmethod
    def _crossing(gamma: Array, /) -> tuple[Array, Array]:
        incoming = jnp.max(jnp.abs(gamma + jnp.swapaxes(gamma, 0, 1)))
        outgoing = jnp.max(jnp.abs(gamma + jnp.swapaxes(gamma, 2, 3)))
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
        gamma = self._spin_vertex(vertex.values)
        pp_loop, ph_loop, minimum = self._loops(cutoff)
        spin_pp = jnp.repeat(jnp.repeat(pp_loop, 2, axis=0), 2, axis=1)
        spin_ph = jnp.repeat(jnp.repeat(ph_loop, 2, axis=0), 2, axis=1)
        pp = 0.5 * contract("ijmn,mn,mnkl->ijkl", gamma, spin_pp, gamma, backend="jax")
        ph_direct = -contract("imkn,mn,njml->ijkl", gamma, spin_ph, gamma, backend="jax")
        ph_crossed = contract("imln,mn,njmk->ijkl", gamma, spin_ph, gamma, backend="jax")
        beta_gamma = pp + ph_direct + ph_crossed
        incoming, outgoing = self._crossing(gamma)
        beta_incoming, beta_outgoing = self._crossing(beta_gamma)
        routing = jnp.max(self.routing_residuals)
        allowed_patch = (
            self.outgoing_patch[..., None]
            == jnp.arange(self.patch_count)[None, None, None, :]
        )
        allowed_spin = jnp.broadcast_to(
            allowed_patch[:, None, :, None, :, None, :, None],
            (self.patch_count, 2) * 4,
        ).reshape(gamma.shape)
        ward = jnp.max(jnp.abs(jnp.where(allowed_spin, 0.0, gamma)))
        residuals = jnp.stack(
            (routing, incoming, outgoing, beta_incoming, beta_outgoing, ward)
        )
        finite = (
            vertex.finite
            & jnp.isfinite(cutoff)
            & (cutoff > 0.0)
            & jnp.all(jnp.isfinite(residuals))
            & jnp.all(jnp.isfinite(beta_gamma))
        )
        tolerance = (
            128.0
            * jnp.finfo(gamma.real.dtype).eps
            * jnp.maximum(1.0, jnp.max(jnp.abs(gamma)))
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
            self._project(beta_gamma),
            self._project(pp),
            self._project(ph_direct),
            self._project(ph_crossed),
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
