#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellMesh
from ._geometry import build_sharp_crack_topology, CrackFrontGeometry, SharpCrackTopology
from ._observables import StressIntensityFactors
from ._quadrature import build_sharp_crack_quadrature, SharpCrackQuadrature


class CrackGrowthProposal(StrictModule, NonTrainableState):
    """Auditable frozen-tip proposal; it cannot mutate an accepted crack state."""

    direction: Array
    increment: Array
    driving_force: Array
    toughness: Array
    kink_angle: Array
    tip_id: int = eqx.field(static=True)
    criterion: str = eqx.field(static=True)
    admissible: bool = eqx.field(static=True)
    rejection_reasons: tuple[str, ...] = eqx.field(static=True)
    base_geometry_id: str = eqx.field(static=True)
    base_topology_id: str = eqx.field(static=True)
    base_state_version: int = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    def __init__(
        self,
        tip_id: int,
        direction: ArrayLike,
        increment: ArrayLike,
        driving_force: ArrayLike,
        toughness: ArrayLike,
        kink_angle: ArrayLike,
        /,
        *,
        criterion: str,
        admissible: bool,
        rejection_reasons: tuple[str, ...] = (),
        base_geometry_id: str,
        base_topology_id: str,
        base_state_version: int,
    ):
        tip_identifier = int(tip_id)
        direction_ = np.asarray(direction)
        increment_ = np.asarray(increment)
        force = np.asarray(driving_force)
        toughness_ = np.asarray(toughness)
        angle = np.asarray(kink_angle)
        criterion_ = str(criterion)
        reasons = tuple(str(value) for value in rejection_reasons)
        geometry_identifier = str(base_geometry_id)
        topology_identifier = str(base_topology_id)
        version = int(base_state_version)
        if (
            tip_identifier < 0
            or direction_.shape != (2,)
            or np.any(~np.isfinite(direction_))
            or not np.isclose(np.linalg.norm(direction_), 1.0, rtol=0.0, atol=1.0e-10)
            or any(
                value.shape != () or not np.isfinite(value)
                for value in (increment_, force, toughness_, angle)
            )
            or increment_ <= 0.0
            or toughness_ <= 0.0
            or force < 0.0
            or not criterion_
            or any(not value for value in reasons)
            or len(set(reasons)) != len(reasons)
            or bool(admissible) == bool(reasons)
            or not geometry_identifier
            or not topology_identifier
            or version < 0
        ):
            raise ValueError("Crack-growth proposal data or provenance are inconsistent.")
        self.direction = jnp.asarray(direction_)
        self.increment = jnp.asarray(increment_)
        self.driving_force = jnp.asarray(force)
        self.toughness = jnp.asarray(toughness_)
        self.kink_angle = jnp.asarray(angle)
        self.tip_id = tip_identifier
        self.criterion = criterion_
        self.admissible = bool(admissible)
        self.rejection_reasons = reasons
        self.base_geometry_id = geometry_identifier
        self.base_topology_id = topology_identifier
        self.base_state_version = version
        self.proposal_id = canonical_fingerprint(
            {
                "kind": "crack-growth-proposal",
                "tip_id": tip_identifier,
                "direction": direction_.tolist(),
                "increment": float(increment_),
                "driving_force": float(force),
                "toughness": float(toughness_),
                "kink_angle": float(angle),
                "criterion": criterion_,
                "admissible": bool(admissible),
                "reasons": list(reasons),
                "geometry": geometry_identifier,
                "topology": topology_identifier,
                "state_version": version,
            }
        )


class SharpFractureState(StrictModule, NonTrainableState):
    """Accepted sharp geometry, topology, and integration realization."""

    geometry: CrackFrontGeometry
    topology: SharpCrackTopology
    quadrature: SharpCrackQuadrature
    accepted_step: int = eqx.field(static=True)
    state_version: int = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: CrackFrontGeometry,
        topology: SharpCrackTopology,
        quadrature: SharpCrackQuadrature,
        /,
        *,
        accepted_step: int = 0,
        state_version: int = 0,
    ):
        if not isinstance(geometry, CrackFrontGeometry):
            raise TypeError("geometry must be CrackFrontGeometry.")
        if not isinstance(topology, SharpCrackTopology):
            raise TypeError("topology must be SharpCrackTopology.")
        if not isinstance(quadrature, SharpCrackQuadrature):
            raise TypeError("quadrature must be SharpCrackQuadrature.")
        step = int(accepted_step)
        version = int(state_version)
        if (
            topology.geometry_id != geometry.geometry_id
            or quadrature.geometry_id != geometry.geometry_id
            or quadrature.topology_id != topology.topology_id
            or step < 0
            or version < 0
        ):
            raise ValueError(
                "Sharp accepted state provenance or versions are inconsistent."
            )
        self.geometry = geometry
        self.topology = topology
        self.quadrature = quadrature
        self.accepted_step = step
        self.state_version = version
        self.state_id = canonical_fingerprint(
            {
                "kind": "accepted-sharp-fracture-state",
                "geometry": geometry.geometry_id,
                "topology": topology.topology_id,
                "quadrature": quadrature.quadrature_id,
                "accepted_step": step,
                "state_version": version,
            }
        )


class CrackGrowthTransaction(StrictModule, NonTrainableState):
    """Opaque candidate that atomically promotes or rolls back a sharp topology event."""

    proposal: CrackGrowthProposal
    candidate: SharpFractureState
    accepted: bool = eqx.field(static=True)
    base_state_id: str = eqx.field(static=True)
    base_state_version: int = eqx.field(static=True)
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        proposal: CrackGrowthProposal,
        candidate: SharpFractureState,
        /,
        *,
        accepted: bool,
        base_state_id: str,
        base_state_version: int,
    ):
        if not isinstance(proposal, CrackGrowthProposal):
            raise TypeError("proposal must be CrackGrowthProposal.")
        if not isinstance(candidate, SharpFractureState):
            raise TypeError("candidate must be SharpFractureState.")
        identifier = str(base_state_id)
        version = int(base_state_version)
        accepted_ = bool(accepted)
        if (
            not identifier
            or version < 0
            or proposal.base_state_version != version
            or (accepted_ and not proposal.admissible)
            or (accepted_ and candidate.state_version != version + 1)
            or (not accepted_ and candidate.state_id != identifier)
        ):
            raise ValueError(
                "Crack-growth transaction acceptance or base state is invalid."
            )
        self.proposal = proposal
        self.candidate = candidate
        self.accepted = accepted_
        self.base_state_id = identifier
        self.base_state_version = version
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "crack-growth-transaction",
                "proposal": proposal.proposal_id,
                "candidate": candidate.state_id,
                "accepted": accepted_,
                "base_state": identifier,
                "base_version": version,
            }
        )

    def _validate_current(self, current: SharpFractureState) -> None:
        if not isinstance(current, SharpFractureState):
            raise TypeError("current must be SharpFractureState.")
        if (
            current.state_id != self.base_state_id
            or current.state_version != self.base_state_version
        ):
            raise ValueError("Crack-growth transaction targets a stale accepted state.")

    def commit(self, current: SharpFractureState, /) -> SharpFractureState:
        self._validate_current(current)
        return self.candidate if self.accepted else current

    def rollback(self, current: SharpFractureState, /) -> SharpFractureState:
        self._validate_current(current)
        return current


def propose_mixed_mode_growth(
    state: SharpFractureState,
    evidence: StressIntensityFactors,
    tip_id: int,
    increment: ArrayLike,
    toughness: ArrayLike,
    /,
) -> CrackGrowthProposal:
    """Apply the maximum-hoop-stress direction and a J >= toughness gate."""

    if not isinstance(state, SharpFractureState):
        raise TypeError("state must be SharpFractureState.")
    if not isinstance(evidence, StressIntensityFactors):
        raise TypeError("evidence must be StressIntensityFactors.")
    if (
        evidence.topology_id != state.topology.topology_id
        or evidence.quadrature_id != state.quadrature.quadrature_id
        or evidence.state_version != state.state_version
    ):
        raise ValueError("SIF evidence is stale for the accepted sharp state.")
    identifier = int(tip_id)
    increment_ = float(np.asarray(increment))
    toughness_ = float(np.asarray(toughness))
    if (
        identifier not in set(np.asarray(state.geometry.tip_ids).tolist())
        or not math.isfinite(increment_)
        or increment_ <= 0.0
        or not math.isfinite(toughness_)
        or toughness_ <= 0.0
    ):
        raise ValueError("Crack-growth tip, increment, or toughness is invalid.")
    mode_i = float(evidence.mode_i)
    mode_ii = float(evidence.mode_ii)
    if abs(mode_ii) <= np.finfo(float).eps * max(1.0, abs(mode_i)):
        kink_angle = 0.0
    else:
        kink_angle = 2.0 * math.atan(
            (mode_i - math.sqrt(mode_i * mode_i + 8.0 * mode_ii * mode_ii))
            / (4.0 * mode_ii)
        )
    _, tangent, normal = state.geometry.tip_frame(identifier)
    direction = math.cos(kink_angle) * np.asarray(tangent) + math.sin(
        kink_angle
    ) * np.asarray(normal)
    driving_force = max(float(evidence.j_integral), 0.0)
    reasons: list[str] = []
    if not bool(evidence.qualified):
        reasons.append("unqualified-sif-evidence")
    if driving_force < toughness_:
        reasons.append("driving-force-below-toughness")
    admissible = not reasons
    return CrackGrowthProposal(
        identifier,
        direction,
        increment_,
        driving_force,
        toughness_,
        kink_angle,
        criterion="maximum-hoop-stress",
        admissible=admissible,
        rejection_reasons=tuple(reasons),
        base_geometry_id=state.geometry.geometry_id,
        base_topology_id=state.topology.topology_id,
        base_state_version=state.state_version,
    )


def prepare_crack_growth_transaction(
    mesh: CellMesh,
    current: SharpFractureState,
    proposal: CrackGrowthProposal,
    /,
    *,
    accepted: bool,
    quadrature_order: int = 2,
    conservation_tolerance: float = 1.0e-10,
) -> CrackGrowthTransaction:
    """Build and certify a candidate before exposing one atomic commit decision."""

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(current, SharpFractureState):
        raise TypeError("current must be SharpFractureState.")
    if not isinstance(proposal, CrackGrowthProposal):
        raise TypeError("proposal must be CrackGrowthProposal.")
    if (
        mesh.mesh_id != current.topology.mesh_id
        or proposal.base_geometry_id != current.geometry.geometry_id
        or proposal.base_topology_id != current.topology.topology_id
        or proposal.base_state_version != current.state_version
    ):
        raise ValueError("Crack-growth proposal is stale for the accepted state.")
    accept_candidate = bool(accepted) and proposal.admissible
    if not accept_candidate:
        return CrackGrowthTransaction(
            proposal,
            current,
            accepted=False,
            base_state_id=current.state_id,
            base_state_version=current.state_version,
        )

    tip_origin, _, _ = current.geometry.tip_frame(proposal.tip_id)
    endpoint = np.asarray(tip_origin) + float(proposal.increment) * np.asarray(
        proposal.direction
    )
    geometry = current.geometry.with_tip_extension(proposal.tip_id, endpoint)
    topology = build_sharp_crack_topology(
        mesh,
        geometry,
        topology_version=current.topology.topology_version + 1,
    )
    quadrature = build_sharp_crack_quadrature(mesh, topology, order=quadrature_order)
    tolerance = float(conservation_tolerance)
    if (
        not np.isfinite(tolerance)
        or tolerance <= 0.0
        or float(quadrature.evidence.relative_area_defect) > tolerance
        or float(quadrature.evidence.face_measure_defect) > tolerance
    ):
        raise ValueError("Candidate crack quadrature failed conservation certification.")
    candidate = SharpFractureState(
        geometry,
        topology,
        quadrature,
        accepted_step=current.accepted_step,
        state_version=current.state_version + 1,
    )
    return CrackGrowthTransaction(
        proposal,
        candidate,
        accepted=True,
        base_state_id=current.state_id,
        base_state_version=current.state_version,
    )


__all__ = [
    "CrackGrowthProposal",
    "CrackGrowthTransaction",
    "SharpFractureState",
    "prepare_crack_growth_transaction",
    "propose_mixed_mode_growth",
]
