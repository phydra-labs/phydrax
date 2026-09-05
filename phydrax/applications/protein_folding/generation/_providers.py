# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Offline user-output admission. No provider runtime, network, or weight download."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ....units import conversion_factor
from ..._coordinate_generation._providers import CoordinateProviderProvenance
from ..._coordinate_generation._support import (
    CoordinateResourcePolicy,
    prepare_coordinate_support,
)
from .._construct import ProteinAtomKey, ProteinConstruct
from .._hypotheses import ProteinStructureHypothesis


@dataclass(frozen=True)
class ProteinProviderHypotheses:
    hypotheses: tuple[ProteinStructureHypothesis, ...]
    provenance: CoordinateProviderProvenance


def import_protein_hypotheses(
    construct,
    source_atoms,
    positions,
    length_unit,
    sources,
    *,
    provenance,
    confidence=None,
    resources=CoordinateResourcePolicy(),
    commercial_use=False,
    training_use=False,
    redistribution=False,
    export=False,
):
    """Import all explicitly mapped user-supplied outputs; confidence stays provider-specific."""
    rights = provenance.admit(
        commercial_use=commercial_use,
        training_use=training_use,
        redistribution=redistribution,
        export=export,
    )
    coordinates = np.asarray(positions)
    rows, envelopes = tuple(source_atoms), tuple(sources)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (len(rows), 3):
        raise ValueError(
            "Provider outputs need explicit shape (hypothesis, source_atom, 3)."
        )
    count = coordinates.shape[0]
    if (
        not 1 <= count <= resources.max_samples
        or len(rows) > resources.max_atoms
        or len(envelopes) != count
    ):
        raise ValueError(
            "Provider outputs exceed capacity or lack per-hypothesis raw source artifacts."
        )
    provenance.require_sources(envelopes)
    confidences = (
        tuple(() for _ in range(count)) if confidence is None else tuple(confidence)
    )
    if len(confidences) != count:
        raise ValueError("Retain one provider-specific confidence record per hypothesis.")
    return ProteinProviderHypotheses(
        tuple(
            ProteinStructureHypothesis(
                construct,
                rows,
                coordinates[i],
                length_unit,
                envelopes[i],
                rights,
                provider=provenance.provider_id,
                confidence=confidences[i],
            )
            for i in range(count)
        ),
        provenance,
    )


def prepare_protein_coordinate_support(
    construct,
    template,
    atom_ids,
    *,
    gauge_atom_ids,
    geometry,
    resources=CoordinateResourcePolicy(),
):
    """Bind real residue/atom tokens to existing stable atomistic IDs."""
    if not isinstance(construct, ProteinConstruct):
        raise TypeError("Protein generation requires an explicit ProteinConstruct.")
    mapping = dict(atom_ids)
    keys = construct.residue_keys
    if any(
        not isinstance(key, ProteinAtomKey) or key.residue not in keys for key in mapping
    ):
        raise ValueError("Protein atom mapping must use atom keys from this construct.")
    if any(
        isinstance(atom_id, bool)
        or not isinstance(atom_id, int)
        or not 0 <= atom_id < 2**63
        for atom_id in mapping.values()
    ):
        raise ValueError("Protein atom mapping requires explicit nonnegative int64 IDs.")
    reverse = {value: key for key, value in mapping.items()}
    ids, active = np.asarray(template.particle_ids[0]), np.asarray(template.atom_mask[0])
    if len(reverse) != len(mapping) or set(reverse) != set(int(i) for i in ids[active]):
        raise ValueError(
            "Protein mapping must cover material atoms exactly; missing atoms are not padding."
        )
    tokens, names = [], []
    lookup = {key: index for index, key in enumerate(keys)}
    for atom_id, mask in zip(ids, active, strict=True):
        key = reverse[int(atom_id)] if mask else None
        tokens.append(lookup[key.residue] if mask else -1)
        names.append(key.atom_name if mask else "")
    labels = tuple(
        "protein:" + amino for sequence in construct.sequences for amino in sequence
    )
    return prepare_coordinate_support(
        template,
        construct_id=construct.fingerprint(),
        token_labels=labels,
        atom_token_indices=tuple(tokens),
        atom_names=tuple(names),
        gauge_atom_ids=gauge_atom_ids,
        geometry=geometry,
        resources=resources,
    )


def map_protein_hypothesis(
    hypothesis, support, atom_ids, *, training_use=False, commercial_use=False
):
    """Map a full raw hypothesis into the model ABI without changing the raw object."""
    hypothesis.require_rights(training_use=training_use, commercial_use=commercial_use)
    if hypothesis.construct.fingerprint() != support.construct_id:
        raise ValueError("Hypothesis construct differs from the model construct.")
    rows = {row.atom_key: i for i, row in enumerate(hypothesis.source_atoms)}
    mapping = dict(atom_ids)
    if set(rows) != set(mapping):
        raise ValueError(
            "Hypothesis must cover the full declared chemical mapping; no implicit atom completion."
        )
    ids = np.asarray(support.template.particle_ids[0])
    mask = np.asarray(support.template.atom_mask[0])
    reverse = {value: key for key, value in mapping.items()}
    if len(reverse) != len(mapping) or set(reverse) != set(int(i) for i in ids[mask]):
        raise ValueError("Atom IDs must bijectively cover material model support.")
    values = np.zeros((len(ids), 3), dtype=np.asarray(hypothesis.positions).dtype)
    for index, (atom_id, active) in enumerate(zip(ids, mask, strict=True)):
        if active:
            key = reverse[int(atom_id)]
            if (
                key.atom_name != support.atom_names[index]
                or key.residue
                != hypothesis.construct.residue_keys[support.atom_token_indices[index]]
            ):
                raise ValueError(
                    "Source atom-to-token assignment disagrees with the trained model ABI."
                )
            if hypothesis.source_atoms[rows[key]].element != int(
                support.template.atomic_numbers[0, index]
            ):
                raise ValueError(
                    "Provider element identity disagrees with fixed model chemistry."
                )
            values[index] = np.asarray(hypothesis.positions)[rows[key]]
    return jnp.asarray(
        values
        * float(
            conversion_factor(hypothesis.length_unit, support.template.scale.length_unit)
        )
    )
