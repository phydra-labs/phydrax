# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Nucleotide identity adapters for the existing fixed-chemistry coordinate model.

The numeric implementation is shared acyclically with protein generation; DNA
and RNA identities stay owned here. No sequence conversion or provider execution.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ....units import conversion_factor
from ..._coordinate_generation._native import require_coordinate_rights
from ..._coordinate_generation._providers import CoordinateProviderProvenance
from ..._coordinate_generation._support import (
    CoordinateResourcePolicy,
    prepare_coordinate_support,
)
from .._binding import NucleotideAtomMapping
from .._hypotheses import NucleicStructureHypothesis


@dataclass(frozen=True)
class NucleicProviderHypotheses:
    hypotheses: tuple[NucleicStructureHypothesis, ...]
    provenance: CoordinateProviderProvenance
    confidence: tuple[tuple[tuple[str, float], ...], ...]


def import_nucleic_hypotheses(
    mapping,
    positions,
    length_unit,
    sources,
    *,
    provenance,
    coordinate_mask=None,
    confidence=None,
    resources=CoordinateResourcePolicy(),
    commercial_use=False,
    training_use=False,
    redistribution=False,
    export=False,
):
    """Admit all offline supplied conformers and retain provider-specific confidence."""
    rights = provenance.admit(
        commercial_use=commercial_use,
        training_use=training_use,
        redistribution=redistribution,
        export=export,
    )
    coordinates = np.asarray(positions)
    envelopes = tuple(sources)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (len(mapping.atom_ids), 3):
        raise ValueError("Provider output requires shape (hypothesis, mapped_atom, 3).")
    count = coordinates.shape[0]
    if (
        not 1 <= count <= resources.max_samples
        or len(mapping.atom_ids) > resources.max_atoms
        or len(envelopes) != count
    ):
        raise ValueError(
            "Provider outputs exceed capacity or lack per-hypothesis raw source artifacts."
        )
    provenance.require_sources(envelopes)
    confidences = (
        tuple(() for _ in range(count))
        if confidence is None
        else tuple(tuple(row) for row in confidence)
    )
    if len(confidences) != count or any(
        not name or not np.isfinite(value) for row in confidences for name, value in row
    ):
        raise ValueError(
            "Provider-specific confidence must be finite, named, and aligned to every hypothesis."
        )
    hypotheses = tuple(
        NucleicStructureHypothesis(
            mapping,
            coordinates[i],
            length_unit,
            envelopes[i],
            rights,
            coordinate_mask=coordinate_mask,
            image_policy="nonperiodic",
        )
        for i in range(count)
    )
    return NucleicProviderHypotheses(hypotheses, provenance, confidences)


def prepare_nucleic_coordinate_support(
    mapping, template, *, gauge_atom_ids, geometry, resources=CoordinateResourcePolicy()
):
    """Compile actual base/sugar-polymer atom tokens, retaining DNA/RNA distinction."""
    if not isinstance(mapping, NucleotideAtomMapping):
        raise TypeError("Nucleic generation requires explicit NucleotideAtomMapping.")
    ids = np.asarray(template.particle_ids[0])
    active = np.asarray(template.atom_mask[0])
    reverse = {
        atom_id: (key, name)
        for atom_id, key, name in zip(
            mapping.atom_ids, mapping.nucleotide_keys, mapping.atom_names, strict=True
        )
    }
    if set(reverse) != set(int(i) for i in ids[active]):
        raise ValueError(
            "Nucleotide mapping must cover material support exactly; missing atoms are not padding."
        )
    keys = mapping.construct.nucleotide_keys
    lookup = {key: index for index, key in enumerate(keys)}
    tokens, names = [], []
    for atom_id, mask in zip(ids, active, strict=True):
        key, name = reverse[int(atom_id)] if mask else (None, "")
        tokens.append(lookup[key] if mask else -1)
        names.append(name)
    labels = tuple(
        polymer + ":" + base
        for polymer, sequence in zip(
            mapping.construct.polymer_types, mapping.construct.sequences, strict=True
        )
        for base in sequence
    )
    return prepare_coordinate_support(
        template,
        construct_id=mapping.construct.fingerprint(),
        token_labels=labels,
        atom_token_indices=tuple(tokens),
        atom_names=tuple(names),
        gauge_atom_ids=gauge_atom_ids,
        geometry=geometry,
        resources=resources,
    )


def map_nucleic_hypothesis(
    hypothesis, support, *, training_use=False, commercial_use=False
):
    """Lossless stable-ID reorder and exact-unit conversion, not chemical completion."""
    require_coordinate_rights(
        hypothesis.rights, training_use=training_use, commercial_use=commercial_use
    )
    mapping = hypothesis.mapping
    if mapping.construct.fingerprint() != support.construct_id:
        raise ValueError("Nucleic hypothesis and model constructs differ.")
    if hypothesis.image_policy != "nonperiodic":
        raise ValueError(
            "Native coordinate generation initially requires nonperiodic hypotheses."
        )
    if not np.asarray(hypothesis.coordinate_mask).all():
        raise ValueError(
            "Training/inference mapping requires complete coordinate coverage, not padding of missing atoms."
        )
    reverse = {atom_id: row for row, atom_id in enumerate(mapping.atom_ids)}
    ids, active = (
        np.asarray(support.template.particle_ids[0]),
        np.asarray(support.template.atom_mask[0]),
    )
    if set(reverse) != set(int(i) for i in ids[active]):
        raise ValueError("Hypothesis IDs must cover material model support exactly.")
    values = np.zeros((len(ids), 3), dtype=np.asarray(hypothesis.positions).dtype)
    for index, (atom_id, mask) in enumerate(zip(ids, active, strict=True)):
        if mask:
            row = reverse[int(atom_id)]
            if (
                mapping.atom_names[row] != support.atom_names[index]
                or mapping.nucleotide_keys[row]
                != mapping.construct.nucleotide_keys[support.atom_token_indices[index]]
            ):
                raise ValueError(
                    "Nucleotide atom-to-token mapping differs from the trained model ABI."
                )
            values[index] = np.asarray(hypothesis.positions)[row]
    return jnp.asarray(
        values
        * float(
            conversion_factor(hypothesis.length_unit, support.template.scale.length_unit)
        )
    )
