#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._numerics._quadrature_rules import gauss_legendre_data
from ...._strict import StrictModule
from ....atomistic._system import PreparedAtomisticSystem
from ....atomistic.sampling._collective_variable import (
    CollectiveVariableKind,
    CollectiveVariablePlan,
    PreparedCollectiveVariable,
)


class NascentObservation(StrictModule):
    contact_similarity: Array
    contact_count: Array
    contact_available: Array
    gauss_entanglement: Array
    quadrature_difference: Array
    curve_separation: Array
    entanglement_available: Array
    successful: Array


class NascentChainObservations(StrictModule):
    """Native reference-contact CV plus oriented polygonal Gauss entanglement.

    Pair and curve inputs are stable particle IDs, never sequence/slot indices.
    Contacts involving future material are explicitly unavailable. Both curves
    must be fully active to evaluate entanglement. Open-curve Gauss integrals
    are geometric observations, not integer linking numbers or knot labels.
    Curves may close by repeating their first ID. Refined quadrature difference
    is convergence evidence, not a certified error bound. Intersecting/degenerate
    curves are refused by numeric status. Preparation is host-only; evaluation
    is JIT/gradient compatible on a fixed, nonsingular geometry branch.
    """

    contact: PreparedCollectiveVariable | None
    left_slots: Array
    right_slots: Array
    nodes: Array
    weights: Array
    fine_nodes: Array
    fine_weights: Array
    contact_count: int = eqx.field(static=True)
    curves_available: bool = eqx.field(static=True)
    minimum_separation: float = eqx.field(static=True)

    def __init__(
        self,
        system: PreparedAtomisticSystem,
        /,
        *,
        contact_particle_pairs: tuple[tuple[int, int], ...],
        reference_distances: tuple[float, ...],
        contact_width: float,
        left_curve_ids: tuple[int, ...] = (),
        right_curve_ids: tuple[int, ...] = (),
        quadrature_order: int = 4,
        minimum_separation: float = 1e-8,
    ):
        if system.cell is not None:
            raise ValueError(
                "Nascent observations require explicitly unwrapped nonperiodic coordinates."
            )
        if quadrature_order < 2 or int(quadrature_order) != quadrature_order:
            raise ValueError(
                "Entanglement quadrature order must be an integer at least two."
            )
        if not np.isfinite(minimum_separation) or minimum_separation <= 0:
            raise ValueError("Minimum curve separation must be finite and positive.")
        if not np.isfinite(contact_width) or contact_width <= 0:
            raise ValueError("Contact width must be finite and positive.")
        ids = np.asarray(system.plan.particle_ids)
        slot = {int(identifier): index for index, identifier in enumerate(ids)}
        active = np.asarray(system.active_mask)
        pairs = tuple(contact_particle_pairs)
        if len(reference_distances) != len(pairs) or any(
            not np.isfinite(d) or d <= 0 for d in reference_distances
        ):
            raise ValueError(
                "Every contact requires a finite positive reference distance."
            )
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs):
            raise ValueError("Contacts must join two distinct stable IDs.")
        if any(identifier not in slot for pair in pairs for identifier in pair):
            raise ValueError("Unknown contact particle ID.")
        keep = [i for i, pair in enumerate(pairs) if all(active[slot[x]] for x in pair)]
        self.contact_count = len(keep)
        self.contact = (
            CollectiveVariablePlan(
                CollectiveVariableKind.CONTACT_SIMILARITY,
                [[slot[x] for x in pairs[i]] for i in keep],
                reference=[reference_distances[i] for i in keep],
                parameters=[contact_width],
            ).prepare(system)
            if keep
            else None
        )
        left, right = tuple(left_curve_ids), tuple(right_curve_ids)
        if bool(left) != bool(right) or (left and (len(left) < 2 or len(right) < 2)):
            raise ValueError("Entanglement requires two nonempty polygonal curves.")
        if set(left) & set(right) or any(x not in slot for x in left + right):
            raise ValueError("Entanglement curves require disjoint known stable IDs.")
        if any(
            a == b
            for curve in (left, right)
            for a, b in zip(curve[:-1], curve[1:], strict=True)
        ):
            raise ValueError("A curve cannot contain a zero-identity segment.")
        self.curves_available = bool(left) and all(active[slot[x]] for x in left + right)
        self.left_slots = jnp.asarray([slot[x] for x in left], dtype=jnp.int32)
        self.right_slots = jnp.asarray([slot[x] for x in right], dtype=jnp.int32)
        rule, fine = (
            gauss_legendre_data(quadrature_order),
            gauss_legendre_data(2 * quadrature_order),
        )
        self.nodes, self.weights = (
            0.5 * (jnp.asarray(rule.nodes) + 1),
            0.5 * jnp.asarray(rule.weights),
        )
        self.fine_nodes, self.fine_weights = (
            0.5 * (jnp.asarray(fine.nodes) + 1),
            0.5 * jnp.asarray(fine.weights),
        )
        self.minimum_separation = float(minimum_separation)

    def evaluate(self, positions: Array, /) -> NascentObservation:
        zero = jnp.zeros((), dtype=positions.dtype)
        contact = None if self.contact is None else self.contact.evaluate(positions)
        q = zero if contact is None else contact.value
        success = jnp.asarray(True) if contact is None else contact.successful
        if not self.curves_available:
            return NascentObservation(
                q,
                jnp.asarray(self.contact_count),
                self.contact_count > 0,
                zero,
                zero,
                zero,
                False,
                success,
            )
        left, right = positions[self.left_slots], positions[self.right_slots]
        a, b = left[:-1, None, :], right[None, :-1, :]
        u, v = left[1:, None, :] - a, right[None, 1:, :] - b
        r = a - b
        uu, vv, uv = jnp.sum(u * u, -1), jnp.sum(v * v, -1), jnp.sum(u * v, -1)
        ru, rv = jnp.sum(r * u, -1), jnp.sum(r * v, -1)
        safe_uu, safe_vv = jnp.where(uu > 0, uu, 1), jnp.where(vv > 0, vv, 1)
        determinant = uu * vv - uv * uv
        safe_det = jnp.where(determinant > 0, determinant, 1)
        s = (uv * rv - vv * ru) / safe_det
        t = (uu * rv - uv * ru) / safe_det
        interior = (determinant > 0) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
        interior_sq = jnp.where(
            interior, jnp.sum((r + s[..., None] * u - t[..., None] * v) ** 2, -1), jnp.inf
        )
        endpoint_squares = []
        for end in (0.0, 1.0):
            edge_r = r + end * u
            projected = jnp.clip(jnp.sum(edge_r * v, -1) / safe_vv, 0, 1)
            endpoint_squares.append(jnp.sum((edge_r - projected[..., None] * v) ** 2, -1))
            edge_r = r - end * v
            projected = jnp.clip(-jnp.sum(edge_r * u, -1) / safe_uu, 0, 1)
            endpoint_squares.append(jnp.sum((edge_r + projected[..., None] * u) ** 2, -1))
        separation = jnp.sqrt(jnp.min(jnp.stack([interior_sq, *endpoint_squares])))

        def integral(nodes, weights):
            delta = (
                r[:, :, None, None, :]
                + nodes[None, None, :, None, None] * u[:, :, None, None, :]
                - nodes[None, None, None, :, None] * v[:, :, None, None, :]
            )
            squared = jnp.sum(delta**2, axis=-1)
            denominator = jnp.where(squared > 0, squared, 1) ** 1.5
            numerator = jnp.sum(jnp.cross(u, v)[:, :, None, None, :] * delta, axis=-1)
            return jnp.sum(
                numerator
                / denominator
                * weights[None, None, :, None]
                * weights[None, None, None, :]
            ) / (4 * jnp.pi)

        coarse, fine = (
            integral(self.nodes, self.weights),
            integral(self.fine_nodes, self.fine_weights),
        )
        valid = (
            jnp.all(uu > 0)
            & jnp.all(vv > 0)
            & (separation > self.minimum_separation)
            & jnp.isfinite(fine)
        )
        return NascentObservation(
            q,
            jnp.asarray(self.contact_count),
            self.contact_count > 0,
            jnp.where(valid, fine, zero),
            jnp.abs(fine - coarse),
            separation,
            True,
            success & valid,
        )


__all__ = ["NascentChainObservations", "NascentObservation"]
