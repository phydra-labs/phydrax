# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._strict import StrictModule
from ._frames import base_frames


class NucleotideStructureQualification(StrictModule):
    frame_valid: Array
    ring_covered: Array
    maximum_ring_deviation: Array
    ring_planar: Array
    backbone_covered: Array
    backbone_distance: Array
    backbone_in_interval: Array
    successful: Array


class NucleotideStructureQualifier(StrictModule):
    """Declared ring planarity/backbone continuity, not complete physical validity.

    Bounds are in the same native length unit as bound coordinates. No covalent
    bonds are guessed: O3′→P checks follow only the construct's directed edges.
    Missing six-ring atoms and missing phosphates remain explicit failures.
    """

    binding: object
    ring_indices: Array
    ring_mask: Array
    backbone_indices: Array
    backbone_mask: Array
    maximum_deviation: float = eqx.field(static=True)
    backbone_interval: tuple[float, float] = eqx.field(static=True)
    image_policy: str = eqx.field(static=True)

    def __init__(
        self, binding, *, maximum_ring_deviation, backbone_interval, image_policy
    ):
        if (
            image_policy not in ("nonperiodic", "unwrapped")
            or not np.isfinite(maximum_ring_deviation)
            or maximum_ring_deviation <= 0
            or len(backbone_interval) != 2
            or not np.all(np.isfinite(backbone_interval))
            or not 0 < backbone_interval[0] < backbone_interval[1]
        ):
            raise ValueError(
                "Qualification needs explicit images and finite geometry bounds."
            )
        mapping = binding.mapping
        lookup = {
            (key, name): int(row)
            for key, name, row, valid in zip(
                mapping.nucleotide_keys,
                mapping.atom_names,
                np.asarray(binding.atom_indices),
                np.asarray(binding.atom_mask),
                strict=True,
            )
            if valid
        }
        rings = [
            [lookup.get((key, name), -1) for name in ("N1", "C2", "N3", "C4", "C5", "C6")]
            for key in binding.construct.nucleotide_keys
        ]
        edges = [
            [lookup.get((a, "O3'"), -1), lookup.get((b, "P"), -1)]
            for a, b in binding.construct.directed_edges
        ]
        rings, edges = (
            np.asarray(rings),
            np.asarray(edges, dtype=np.int64).reshape((-1, 2)),
        )
        self.binding = binding
        self.ring_indices, self.ring_mask = (
            jnp.asarray(np.maximum(rings, 0)),
            jnp.asarray(rings >= 0),
        )
        self.backbone_indices, self.backbone_mask = (
            jnp.asarray(np.maximum(edges, 0)),
            jnp.asarray(edges >= 0),
        )
        self.maximum_deviation, self.backbone_interval, self.image_policy = (
            float(maximum_ring_deviation),
            tuple(backbone_interval),
            image_policy,
        )

    def evaluate(self, positions):
        coordinates = jnp.asarray(positions)
        frame = base_frames(coordinates, self.binding, image_policy=self.image_policy)
        ring_points = coordinates[self.ring_indices]
        covered = jnp.all(self.ring_mask, axis=-1) & jnp.all(
            jnp.isfinite(ring_points), axis=(-2, -1)
        )
        deviation = jnp.max(
            jnp.abs(
                jnp.sum(
                    (ring_points - frame.centers[:, None, :])
                    * frame.axes[:, :, 2][:, None, :],
                    axis=-1,
                )
            ),
            axis=-1,
        )
        planar = covered & frame.valid & (deviation <= self.maximum_deviation)
        edge_points = coordinates[self.backbone_indices]
        edge_covered = jnp.all(self.backbone_mask, axis=-1) & jnp.all(
            jnp.isfinite(edge_points), axis=(-2, -1)
        )
        distance = jnp.sqrt(
            jnp.sum((edge_points[:, 1] - edge_points[:, 0]) ** 2, axis=-1)
        )
        in_interval = (
            edge_covered
            & (distance >= self.backbone_interval[0])
            & (distance <= self.backbone_interval[1])
        )
        return NucleotideStructureQualification(
            frame.valid,
            covered,
            deviation,
            planar,
            edge_covered,
            distance,
            in_interval,
            jnp.all(planar) & jnp.all(in_interval),
        )


__all__ = ["NucleotideStructureQualification", "NucleotideStructureQualifier"]
