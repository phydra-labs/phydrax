#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Convention-pinned finite-cutoff Lorentzian EPRL 4-simplex declarations."""

from __future__ import annotations

from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...tensor_network import su2_fusion


EPRL_TRIANGLES = tuple(
    (first, second) for first in range(5) for second in range(first + 1, 5)
)


def _triangle_key(first: int, second: int, /) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


class EPRLVertexPlan(StrictModule):
    """One regulated Lorentzian EPRL vertex in an ordered intertwiner basis."""

    boundary_twice_spins: tuple[int, ...] = eqx.field(static=True)
    boundary_twice_intertwiners: tuple[int, ...] = eqx.field(static=True)
    internal_twice_spin_support: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    immirzi_parameter: float = eqx.field(static=True)
    delta_l: int = eqx.field(static=True)
    face_amplitude: str = eqx.field(static=True)
    edge_amplitude: str = eqx.field(static=True)
    coherent_phase_convention: str = eqx.field(static=True)
    normal_frame_id: str = eqx.field(static=True)
    quadrature_id: str = eqx.field(static=True)
    precision_bits: int = eqx.field(static=True)
    maximum_support_tuples: int = eqx.field(static=True)
    support_tuple_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        boundary_twice_spins: tuple[int, ...],
        boundary_twice_intertwiners: tuple[int, ...],
        /,
        *,
        immirzi_parameter: float,
        delta_l: int,
        face_amplitude: str,
        edge_amplitude: str,
        coherent_phase_convention: str,
        normal_frame_id: str,
        quadrature_id: str,
        precision_bits: int,
        maximum_support_tuples: int,
    ):
        spins = tuple(boundary_twice_spins)
        intertwiners = tuple(boundary_twice_intertwiners)
        gamma = float(immirzi_parameter)
        cutoff = int(delta_l)
        strings = tuple(
            str(value).strip()
            for value in (
                face_amplitude,
                edge_amplitude,
                coherent_phase_convention,
                normal_frame_id,
                quadrature_id,
            )
        )
        precision = int(precision_bits)
        maximum = int(maximum_support_tuples)
        if len(spins) != 10 or any(value < 0 for value in spins):
            raise ValueError("An EPRL 4-simplex requires ten nonnegative doubled spins.")
        if len(intertwiners) != 5 or any(value < 0 for value in intertwiners):
            raise ValueError(
                "An EPRL 4-simplex requires five nonnegative doubled intertwiners."
            )
        if not np.isfinite(gamma) or gamma <= 0.0 or cutoff < 0:
            raise ValueError("EPRL Immirzi parameter/cutoff are invalid.")
        if any(not value for value in strings) or precision < 64 or maximum < 1:
            raise ValueError(
                "EPRL conventions, precision, and support capacity are required."
            )
        spin_lookup = {
            triangle: spins[index] for index, triangle in enumerate(EPRL_TRIANGLES)
        }
        for tetrahedron in range(5):
            incident = tuple(
                spin_lookup[_triangle_key(tetrahedron, other)]
                for other in range(5)
                if other != tetrahedron
            )
            intertwiner = intertwiners[tetrahedron]
            if intertwiner not in su2_fusion(
                incident[0], incident[1]
            ) or intertwiner not in su2_fusion(incident[2], incident[3]):
                raise ValueError(
                    f"Boundary intertwiner {tetrahedron} is inadmissible in the declared coupling tree."
                )
        support = tuple(
            tuple(spin + 2 * shell for shell in range(cutoff + 1)) for spin in spins
        )
        count = prod(len(value) for value in support)
        if count > maximum:
            raise ValueError("EPRL internal spin support exceeds maximum_support_tuples.")
        self.boundary_twice_spins = spins
        self.boundary_twice_intertwiners = intertwiners
        self.internal_twice_spin_support = support
        self.immirzi_parameter = gamma
        self.delta_l = cutoff
        self.face_amplitude = strings[0]
        self.edge_amplitude = strings[1]
        self.coherent_phase_convention = strings[2]
        self.normal_frame_id = strings[3]
        self.quadrature_id = strings[4]
        self.precision_bits = precision
        self.maximum_support_tuples = maximum
        self.support_tuple_count = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-lorentzian-eprl-vertex-plan",
                "triangles": EPRL_TRIANGLES,
                "boundary_twice_spins": spins,
                "boundary_twice_intertwiners": intertwiners,
                "immirzi_parameter": gamma,
                "delta_l": cutoff,
                "support": support,
                "face_amplitude": strings[0],
                "edge_amplitude": strings[1],
                "coherent_phase_convention": strings[2],
                "normal_frame_id": strings[3],
                "quadrature_id": strings[4],
                "precision_bits": precision,
                "maximum_support_tuples": maximum,
            }
        )


class EPRLSemanticEvidence(StrictModule):
    tetrahedron_admissible: Array
    support_tuple_count: Array
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def assess_eprl_semantics(plan: EPRLVertexPlan, /) -> EPRLSemanticEvidence:
    if not isinstance(plan, EPRLVertexPlan):
        raise TypeError("plan must be EPRLVertexPlan.")
    spin_lookup = {
        triangle: plan.boundary_twice_spins[index]
        for index, triangle in enumerate(EPRL_TRIANGLES)
    }
    admissible = []
    for tetrahedron in range(5):
        incident = tuple(
            spin_lookup[_triangle_key(tetrahedron, other)]
            for other in range(5)
            if other != tetrahedron
        )
        intertwiner = plan.boundary_twice_intertwiners[tetrahedron]
        admissible.append(
            intertwiner in su2_fusion(incident[0], incident[1])
            and intertwiner in su2_fusion(incident[2], incident[3])
        )
    values = jnp.asarray(admissible)
    finite = jnp.asarray(
        np.isfinite(plan.immirzi_parameter) and plan.precision_bits >= 64
    )
    accepted = (
        finite
        & jnp.all(values)
        & (plan.support_tuple_count <= plan.maximum_support_tuples)
    )
    return EPRLSemanticEvidence(
        tetrahedron_admissible=values,
        support_tuple_count=jnp.asarray(plan.support_tuple_count, dtype=jnp.int64),
        finite=finite,
        accepted=accepted,
        plan_id=plan.plan_id,
        claim="finite-cutoff-eprl-semantic-admission-not-an-amplitude-or-quantum-gravity-claim",
    )


__all__ = [
    "EPRL_TRIANGLES",
    "EPRLSemanticEvidence",
    "EPRLVertexPlan",
    "assess_eprl_semantics",
]
