#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative core-conductor observation of committed Shorten fiber voltage.

Pereira Botelho, Curran & Lowery (2019), Eq. 3, independently discretized
with nodal control volumes and sealed terminal axial fluxes. This is not an
ionic-current extractor or an additional source to add to ionic currents.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ..fibers import PreparedSkeletalFiberBundle, SkeletalFiberBundleState


PEREIRA_BOTELHO_2019_DOI = "10.1371/journal.pcbi.1007267"


class FiberCurrentState(StrictModule, NonTrainableState):
    time_ms: Array
    accepted_observations: Array
    membrane_voltage_V: Array
    transmembrane_current_A: Array
    transmembrane_line_current_A_per_m: Array
    prepared_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)


class FiberCurrentEvidence(StrictModule, NonTrainableState):
    finite: Array
    neutral: Array
    identity_matches: Array
    time_admissible: Array
    net_current_A: Array
    neutrality_bound_A: Array
    successful: Array
    source_owner_id: str = eqx.field(static=True)


class FiberCurrentCandidate(StrictModule, NonTrainableState):
    source_state: FiberCurrentState
    candidate_state: FiberCurrentState
    committed_voltage_mV: Array
    evidence: FiberCurrentEvidence

    def commit(
        self,
        current: FiberCurrentState,
        committed_fiber_state: SkeletalFiberBundleState,
        /,
    ) -> FiberCurrentState:
        """Commit only the observation, rejecting stale fiber/observer snapshots."""
        prior = self.source_state
        proposed = self.candidate_state
        if (
            current.prepared_id != prior.prepared_id
            or current.geometry_id != prior.geometry_id
        ):
            return current
        current_matches = (
            (current.accepted_observations == prior.accepted_observations)
            & (current.time_ms == prior.time_ms)
            & jnp.all(current.membrane_voltage_V == prior.membrane_voltage_V)
            & jnp.all(current.transmembrane_current_A == prior.transmembrane_current_A)
            & jnp.all(
                current.transmembrane_line_current_A_per_m
                == prior.transmembrane_line_current_A_per_m
            )
        )
        fiber_matches = (committed_fiber_state.time_ms == proposed.time_ms) & jnp.all(
            committed_fiber_state.values[..., 0] == self.committed_voltage_mV
        )
        accepted = self.evidence.successful & current_matches & fiber_matches
        return FiberCurrentState(
            jnp.where(accepted, proposed.time_ms, current.time_ms),
            jnp.where(
                accepted, proposed.accepted_observations, current.accepted_observations
            ),
            jnp.where(accepted, proposed.membrane_voltage_V, current.membrane_voltage_V),
            jnp.where(
                accepted,
                proposed.transmembrane_current_A,
                current.transmembrane_current_A,
            ),
            jnp.where(
                accepted,
                proposed.transmembrane_line_current_A_per_m,
                current.transmembrane_line_current_A_per_m,
            ),
            current.prepared_id,
            current.geometry_id,
        )


class PereiraBotelho2019FiberCurrentPlan(StrictModule):
    """One sealed-end cable source, with caller-provided fixed world geometry.

    Intracellular conductivity is derived, not independently defaulted:
    sigma_i = 2 C_m D / radius. This enforces the capacitance/diffusivity
    convention of the bound Shorten monodomain bundle. D is converted from
    mm²/ms to m²/s, and C_m from microfarad/cm² to F/m².
    """

    fiber_ids: tuple[str, ...] = eqx.field(static=True)
    positions_m: Array
    radius_m: Array
    geometry_id: str = eqx.field(static=True)
    geometry_source_id: str = eqx.field(static=True)
    geometry_license: str = eqx.field(static=True)
    absolute_neutrality_tolerance_A: float = eqx.field(static=True)
    relative_neutrality_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fiber_ids: tuple[str, ...],
        positions_m: ArrayLike,
        radius_m: ArrayLike,
        /,
        *,
        geometry_source_id: str,
        geometry_license: str,
        absolute_neutrality_tolerance_A: float = 1.0e-15,
        relative_neutrality_tolerance: float = 1.0e-6,
    ):
        ids = tuple(str(value).strip() for value in fiber_ids)
        positions = jnp.asarray(positions_m, dtype=float)
        radius = jnp.asarray(radius_m, dtype=positions.dtype)
        if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError(
                "fiber_ids must be nonempty and unique: exactly one owner per fiber."
            )
        if (
            positions.ndim != 3
            or positions.shape[0] != len(ids)
            or positions.shape[2] != 3
        ):
            raise ValueError("positions_m must have shape (fiber, node, 3).")
        if positions.shape[1] < 3 or radius.shape != (len(ids),):
            raise ValueError(
                "At least three nodes and one positive radius per fiber are required."
            )
        lengths = np.sqrt(np.sum(np.diff(np.asarray(positions), axis=1) ** 2, axis=-1))
        if not (
            np.all(np.isfinite(np.asarray(positions)))
            and np.all(np.isfinite(np.asarray(radius)))
            and np.all(np.asarray(radius) > 0.0)
            and np.all(lengths > 0.0)
        ):
            raise ValueError(
                "Fiber geometry must be finite with positive radii and edge lengths."
            )
        if not geometry_source_id.strip() or not geometry_license.strip():
            raise ValueError(
                "Explicit geometry provenance and reuse rights are required."
            )
        tolerances = (
            float(absolute_neutrality_tolerance_A),
            float(relative_neutrality_tolerance),
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError("Neutrality tolerances must be positive and finite.")
        self.fiber_ids = ids
        self.positions_m = positions
        self.radius_m = radius
        self.geometry_source_id = geometry_source_id.strip()
        self.geometry_license = geometry_license.strip()
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "fixed-world-fiber-geometry-metres",
                "fibers": ids,
                "positions": array_tree_fingerprint(positions),
                "radius": array_tree_fingerprint(radius),
                "source": self.geometry_source_id,
                "license": self.geometry_license,
            }
        )
        self.absolute_neutrality_tolerance_A, self.relative_neutrality_tolerance = (
            tolerances
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pereira-botelho-2019-core-conductor-sealed-nodal-fv",
                "doi": PEREIRA_BOTELHO_2019_DOI,
                "source_xml_sha256": "35fc3bf5a74321c8a38e052c16517b7b4177c60694398b68f887c5c25e25ca7a",
                "equation": "3; outward d_s(sigma_i*pi*r^2*d_s(V_m))",
                "geometry": self.geometry_id,
                "absolute_neutrality_A": tolerances[0].hex(),
                "relative_neutrality": tolerances[1].hex(),
            }
        )

    def prepare(self, fiber: PreparedSkeletalFiberBundle, /) -> PreparedFiberCurrent:
        return PreparedFiberCurrent(self, fiber)


class PreparedFiberCurrent(StrictModule):
    plan: PereiraBotelho2019FiberCurrentPlan
    intracellular_conductivity_S_per_m: Array
    edge_conductance_S: Array
    control_length_m: Array
    fiber_prepared_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PereiraBotelho2019FiberCurrentPlan,
        fiber: PreparedSkeletalFiberBundle,
        /,
    ):
        if not isinstance(fiber, PreparedSkeletalFiberBundle):
            raise TypeError(
                "Requires the fixed-geometry Shorten fiber family, not moving geometry."
            )
        if (
            plan.fiber_ids != fiber.plan.fiber_ids
            or plan.positions_m.shape[1] != fiber.plan.node_count
        ):
            raise ValueError(
                "Fiber identities and node topology must match the electrophysiology owner."
            )
        edge_length = jnp.sqrt(jnp.sum(jnp.diff(plan.positions_m, axis=1) ** 2, axis=-1))
        expected = (
            np.asarray(fiber.plan.fiber_length_mm)[:, None]
            * 1.0e-3
            / (fiber.plan.node_count - 1)
        )
        if not np.allclose(np.asarray(edge_length), expected, rtol=1.0e-6, atol=0.0):
            raise ValueError(
                "Every world edge must match the uniform committed monodomain metric."
            )
        capacitance = fiber.model.membrane_capacitance_uF_per_cm2 * 1.0e-2
        diffusion = fiber.plan.diffusivity_mm2_per_ms * 1.0e-3
        conductivity = 2.0 * capacitance * diffusion / plan.radius_m
        if not np.all(np.asarray(conductivity) > 0.0):
            raise ValueError(
                "Physical current observation requires positive intracellular conductivity."
            )
        self.plan = plan
        self.intracellular_conductivity_S_per_m = conductivity
        self.edge_conductance_S = (conductivity * jnp.pi * plan.radius_m**2)[
            :, None
        ] / edge_length
        self.control_length_m = 0.5 * (
            jnp.pad(edge_length, ((0, 0), (1, 0)))
            + jnp.pad(edge_length, ((0, 0), (0, 1)))
        )
        self.fiber_prepared_id = fiber.prepared_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-fiber-current",
                "plan": plan.plan_id,
                "fiber": fiber.prepared_id,
                "fiber_parameters": array_tree_fingerprint(fiber.model.parameters),
                "conductivity": array_tree_fingerprint(conductivity),
                "boundary": "zero axial flux at both terminals; half endpoint control volumes",
            }
        )

    def initialize(self, /) -> FiberCurrentState:
        zeros = jnp.zeros(self.control_length_m.shape, dtype=self.control_length_m.dtype)
        return FiberCurrentState(
            jnp.asarray(-jnp.inf, dtype=zeros.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            zeros,
            zeros,
            zeros,
            self.prepared_id,
            self.plan.geometry_id,
        )

    def propose(
        self,
        current: FiberCurrentState,
        committed_fiber_state: SkeletalFiberBundleState,
        /,
        *,
        fiber_prepared_id: str,
        geometry_id: str,
    ) -> FiberCurrentCandidate:
        """Observe an accepted snapshot; a fiber Candidate must be committed first."""
        voltage_mV = committed_fiber_state.values[..., 0]
        if voltage_mV.shape != self.control_length_m.shape:
            raise ValueError("Committed fiber voltage has foreign topology.")
        if current.transmembrane_current_A.shape != self.control_length_m.shape:
            raise ValueError("Current observer has foreign topology.")
        voltage = voltage_mV * 1.0e-3
        gradient_flux = self.edge_conductance_S * jnp.diff(voltage, axis=-1)
        # Divergence of +sigma_i A grad(V_m), not intracellular axial current.
        # Terminal zero flux supplies the end effects without mean subtraction.
        outward = jnp.diff(jnp.pad(gradient_flux, ((0, 0), (1, 1))), axis=-1)
        line = outward / self.control_length_m
        net = jnp.sum(outward, axis=-1)
        bound = self.plan.absolute_neutrality_tolerance_A + (
            self.plan.relative_neutrality_tolerance * jnp.sum(jnp.abs(outward), axis=-1)
        )
        neutral = jnp.all(jnp.abs(net) <= bound)
        finite = (
            jnp.all(jnp.isfinite(voltage))
            & jnp.all(jnp.isfinite(line))
            & jnp.isfinite(committed_fiber_state.time_ms)
        )
        identities = jnp.asarray(
            fiber_prepared_id == self.fiber_prepared_id
            and geometry_id == self.plan.geometry_id
            and current.prepared_id == self.prepared_id
            and current.geometry_id == self.plan.geometry_id
        )
        time_valid = committed_fiber_state.time_ms >= current.time_ms
        successful = finite & neutral & identities & time_valid
        proposed = FiberCurrentState(
            committed_fiber_state.time_ms,
            current.accepted_observations + 1,
            voltage,
            outward,
            line,
            self.prepared_id,
            self.plan.geometry_id,
        )
        return FiberCurrentCandidate(
            current,
            proposed,
            voltage_mV,
            FiberCurrentEvidence(
                finite,
                neutral,
                identities,
                time_valid,
                net,
                bound,
                successful,
                self.prepared_id,
            ),
        )


__all__ = [
    "PEREIRA_BOTELHO_2019_DOI",
    "FiberCurrentCandidate",
    "FiberCurrentEvidence",
    "FiberCurrentState",
    "PereiraBotelho2019FiberCurrentPlan",
    "PreparedFiberCurrent",
]
