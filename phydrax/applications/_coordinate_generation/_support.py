# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Host-compiled fixed-chemistry coordinate ABI, not a physical coarse map."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomisticBatch, AtomisticCoordinateDiffusion
from ...stochastic import VariancePreservingDiffusion


@dataclass(frozen=True)
class CoordinateResourcePolicy:
    max_atoms: int = 256
    max_records: int = 10000
    max_pairs: int = 1024
    max_training_steps: int = 100000
    max_samples: int = 1024
    max_width: int = 512
    max_depth: int = 8
    max_solver_steps: int = 4096
    max_condition_features: int = 64

    def __post_init__(self):
        for value in (
            self.max_atoms,
            self.max_records,
            self.max_pairs,
            self.max_training_steps,
            self.max_samples,
            self.max_width,
            self.max_depth,
            self.max_solver_steps,
            self.max_condition_features,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Resource limits must be positive integers.")


@dataclass(frozen=True)
class CoordinateGeometryPolicy:
    """Caller-declared geometry bounds in the template length unit.

    Each chiral tuple (center, a, b, c) declares the sign of
    dot(a-center, cross(b-center, c-center)); it is not a CIP label.
    These sparse checks are proposal screening, not force-field qualification.
    """

    bond_atom_ids: tuple[tuple[int, int], ...]
    bond_bounds: tuple[tuple[float, float], ...]
    chiral_atom_ids: tuple[tuple[int, int, int, int], ...]
    chiral_signs: tuple[int, ...]
    minimum_chiral_volume: float
    policy_id: str
    achiral: bool = False

    def __post_init__(self):
        if any(
            not isinstance(column, tuple)
            for column in (
                self.bond_atom_ids,
                self.bond_bounds,
                self.chiral_atom_ids,
                self.chiral_signs,
            )
        ):
            raise TypeError("Geometry policy columns must be immutable tuples.")
        if any(
            not isinstance(row, tuple)
            for column in (self.bond_atom_ids, self.bond_bounds, self.chiral_atom_ids)
            for row in column
        ):
            raise TypeError("Geometry policy rows must be immutable tuples.")
        if not self.policy_id or not self.bond_atom_ids:
            raise ValueError(
                "Geometry requires a policy identity and explicit bond bounds."
            )
        if len(self.bond_atom_ids) != len(self.bond_bounds):
            raise ValueError("Every declared bond needs lower and upper bounds.")
        for pair, bounds in zip(self.bond_atom_ids, self.bond_bounds, strict=True):
            if len(pair) != 2 or len(set(pair)) != 2:
                raise ValueError("Bond endpoints must be distinct stable atom IDs.")
            if (
                len(bounds) != 2
                or not np.isfinite(bounds).all()
                or not 0 < bounds[0] < bounds[1]
            ):
                raise ValueError("Bond bounds must be finite, positive, and ordered.")
        if len(self.chiral_atom_ids) != len(self.chiral_signs):
            raise ValueError("Every chiral center requires an explicit orientation sign.")
        if self.achiral == bool(self.chiral_atom_ids):
            raise ValueError(
                "Declare either chiral centers or an explicitly achiral profile."
            )
        if not np.isfinite(self.minimum_chiral_volume) or self.minimum_chiral_volume < 0:
            raise ValueError("Minimum chiral volume must be finite and nonnegative.")
        for atoms, sign in zip(self.chiral_atom_ids, self.chiral_signs, strict=True):
            if len(atoms) != 4 or len(set(atoms)) != 4 or sign not in (-1, 1):
                raise ValueError(
                    "Chirality requires four distinct atoms and sign -1 or +1."
                )


class PreparedCoordinateSupport(StrictModule, NonTrainableState):
    template: AtomisticBatch
    diffusion: AtomisticCoordinateDiffusion
    source_law: object
    construct_id: str = eqx.field(static=True)
    token_labels: tuple[str, ...] = eqx.field(static=True)
    atom_token_indices: tuple[int, ...] = eqx.field(static=True)
    atom_names: tuple[str, ...] = eqx.field(static=True)
    token_features: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    gauge_indices: tuple[int, int, int] = eqx.field(static=True)
    bond_indices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    chiral_indices: tuple[tuple[int, int, int, int], ...] = eqx.field(static=True)
    geometry: CoordinateGeometryPolicy = eqx.field(static=True)
    resources: CoordinateResourcePolicy = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    @property
    def dimension(self):
        return self.template.atom_capacity * 3

    def center(self, positions):
        """Numeric fixed-support mass centering, including explicit padding."""
        values = jnp.asarray(positions)
        mask = self.template.atom_mask[0]
        weights = self.diffusion.center_weights[0]
        clean = jnp.where(mask[:, None], values, 0.0)
        center = jnp.sum(clean * weights[:, None], axis=-2, keepdims=True)
        return jnp.where(mask[:, None], clean - center, 0.0)

    def canonicalize(self, positions):
        """Proper anchor frame, never a reflection; returns geometry validity."""
        values = self.center(positions)
        a, b, c = self.gauge_indices
        first = values[..., b, :] - values[..., a, :]
        second = values[..., c, :] - values[..., a, :]
        norm1 = jnp.sqrt(jnp.sum(first**2, axis=-1, keepdims=True))
        e1 = first / jnp.maximum(norm1, jnp.finfo(values.dtype).tiny)
        normal = jnp.cross(e1, second)
        norm3 = jnp.sqrt(jnp.sum(normal**2, axis=-1, keepdims=True))
        e3 = normal / jnp.maximum(norm3, jnp.finfo(values.dtype).tiny)
        e2 = jnp.cross(e3, e1)
        canonical = jnp.stack(
            tuple(jnp.sum(values * axis[..., None, :], axis=-1) for axis in (e1, e2, e3)),
            axis=-1,
        )
        scale = jnp.maximum(jnp.sqrt(jnp.sum(second**2, axis=-1)), 1.0)
        tolerance = 100 * jnp.finfo(values.dtype).eps * scale
        valid = (norm1[..., 0] > tolerance) & (norm3[..., 0] > tolerance)
        return canonical, valid


def prepare_coordinate_support(
    template,
    *,
    construct_id,
    token_labels,
    atom_token_indices,
    atom_names,
    gauge_atom_ids,
    geometry,
    resources=CoordinateResourcePolicy(),
):
    """Internal shared fixed-support compiler; biological wrappers bind the tokens."""
    if not isinstance(template, AtomisticBatch) or template.case_count != 1:
        raise ValueError("Coordinate models require one fixed-chemistry template case.")
    if template.atom_capacity > resources.max_atoms:
        raise ValueError("Template exceeds the atom resource limit.")
    if len(geometry.bond_atom_ids) + len(geometry.chiral_atom_ids) > resources.max_pairs:
        raise ValueError("Geometry screening exceeds the sparse-pair resource limit.")
    if template.has_periodic_metadata:
        raise ValueError("Coordinate generation excludes periodic cells.")
    if not construct_id or not token_labels:
        raise ValueError("A fixed construct and its chemical token labels are required.")
    ids = tuple(int(i) for i in np.asarray(template.particle_ids[0]))
    active = tuple(bool(v) for v in np.asarray(template.atom_mask[0]))
    lookup = {
        atom_id: row
        for row, (atom_id, mask) in enumerate(zip(ids, active, strict=True))
        if mask
    }
    if len(lookup) != sum(active):
        raise ValueError("Active stable atom IDs must be unique.")
    tokens, names = tuple(atom_token_indices), tuple(atom_names)
    if len(tokens) != len(ids) or len(names) != len(ids):
        raise ValueError("Token and atom-name mapping must cover the complete support.")
    for index, name, mask in zip(tokens, names, active, strict=True):
        if mask and (
            not isinstance(index, int) or not 0 <= index < len(token_labels) or not name
        ):
            raise ValueError(
                "Every material atom needs an explicit chemical token and atom name."
            )
        if not mask and (index != -1 or name != ""):
            raise ValueError("Padding uses token -1 and an empty atom name.")
    if set(index for index, mask in zip(tokens, active, strict=True) if mask) != set(
        range(len(token_labels))
    ):
        raise ValueError("Every declared token requires material atom coverage.")
    if len(
        {
            (token, name)
            for token, name, mask in zip(tokens, names, active, strict=True)
            if mask
        }
    ) != sum(active):
        raise ValueError("Atom names must be unique within each biological token.")
    atom_groups = (
        tuple(gauge_atom_ids),
        *geometry.bond_atom_ids,
        *geometry.chiral_atom_ids,
    )
    if len(gauge_atom_ids) != 3 or len(set(gauge_atom_ids)) != 3:
        raise ValueError("Gauge requires three distinct stable atom IDs.")
    if any(atom_id not in lookup for group in atom_groups for atom_id in group):
        raise ValueError(
            "Gauge/geometry IDs must bind material atoms, never padding or sequence positions."
        )
    chemical_vocabulary = tuple(sorted(set(token_labels)))
    name_vocabulary = tuple(
        sorted(set(name for name, mask in zip(names, active, strict=True) if mask))
    )
    features = []
    numbers = np.asarray(template.atomic_numbers[0])
    for row, (token, name, mask) in enumerate(zip(tokens, names, active, strict=True)):
        chemical = token_labels[token] if mask else ""
        features.append(
            tuple(
                float(v)
                for v in (
                    *[mask and chemical == label for label in chemical_vocabulary],
                    *[mask and name == label for label in name_vocabulary],
                    numbers[row] / 118.0 if mask else 0.0,
                    (token + 1) / len(token_labels) if mask else 0.0,
                )
            )
        )
    diffusion = AtomisticCoordinateDiffusion(
        template, VariancePreservingDiffusion(template.positions.size)
    )
    support_id = canonical_fingerprint(
        {
            "kind": "fixed-chemical-coordinate-support",
            "construct": construct_id,
            "topology": template.atom_topology_id,
            "scale": template.scale.scale_id,
            "material": array_tree_fingerprint(
                (
                    template.atomic_numbers,
                    template.particle_ids,
                    template.atom_mask,
                    template.masses,
                )
            ),
            "tokens": token_labels,
            "atom_tokens": tokens,
            "names": names,
            "gauge_ids": gauge_atom_ids,
            "geometry": asdict(geometry),
            "resources": asdict(resources),
        }
    )
    support = PreparedCoordinateSupport(
        template,
        diffusion,
        diffusion.process.asymptotic_terminal_reference().law,
        construct_id,
        tuple(token_labels),
        tokens,
        names,
        tuple(features),
        tuple(lookup[i] for i in gauge_atom_ids),
        tuple(tuple(lookup[i] for i in pair) for pair in geometry.bond_atom_ids),
        tuple(tuple(lookup[i] for i in atoms) for atoms in geometry.chiral_atom_ids),
        geometry,
        resources,
        support_id,
    )
    _, gauge_valid = support.canonicalize(template.positions[0])
    if not bool(gauge_valid):
        raise ValueError("Template gauge anchors are degenerate.")
    return support


class CoordinateProposalQualification(eqx.Module):
    finite: object
    gauge_valid: object
    bond_valid: object
    chirality_valid: object
    accepted: object
    policy_id: str = eqx.field(static=True)
    scientific_claim: str = eqx.field(
        static=True,
        default="declared sparse geometry screening; not physical or predictive qualification",
    )


def qualify_coordinate_proposals(support, positions, *, solver_valid=None):
    """Retain one result per sample; no rejection sampling or hidden repairs."""
    values = jnp.asarray(positions)
    if values.ndim != 3 or values.shape[1:] != (support.template.atom_capacity, 3):
        raise ValueError("Proposals must have shape (sample, atom_capacity, 3).")
    finite = jnp.all(
        jnp.isfinite(jnp.where(support.template.atom_mask[0, :, None], values, 0.0)),
        axis=(1, 2),
    )
    _, gauge_valid = support.canonicalize(values)
    pairs = jnp.asarray(support.bond_indices)
    distances = jnp.sqrt(
        jnp.sum((values[:, pairs[:, 0]] - values[:, pairs[:, 1]]) ** 2, axis=-1)
    )
    bounds = jnp.asarray(support.geometry.bond_bounds)
    bond_valid = jnp.all(
        (distances >= bounds[:, 0]) & (distances <= bounds[:, 1]), axis=1
    )
    if support.chiral_indices:
        atoms = jnp.asarray(support.chiral_indices)
        origin = values[:, atoms[:, 0]]
        a, b, c = (values[:, atoms[:, i]] - origin for i in (1, 2, 3))
        volumes = jnp.sum(a * jnp.cross(b, c), axis=-1)
        chirality_valid = jnp.all(
            volumes * jnp.asarray(support.geometry.chiral_signs)
            > support.geometry.minimum_chiral_volume,
            axis=1,
        )
    else:
        chirality_valid = jnp.ones(values.shape[0], dtype=bool)
    accepted = finite & gauge_valid & bond_valid & chirality_valid
    if solver_valid is not None:
        solver_valid = jnp.asarray(solver_valid, dtype=bool)
        if solver_valid.shape != accepted.shape:
            raise ValueError("Solver validity must retain one entry per proposal.")
        accepted = accepted & solver_valid
    return CoordinateProposalQualification(
        finite,
        gauge_valid,
        bond_valid,
        chirality_valid,
        accepted,
        support.geometry.policy_id,
    )
