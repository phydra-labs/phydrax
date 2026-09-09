# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Offline user-output admission. No provider runtime, network, or weight download."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ...._strict import StrictModule
from ....atomistic import AtomisticBatch
from ....qualification import ReferenceArtifactManifest
from ....units import conversion_factor
from ..._coordinate_generation._native import (
    CoordinateFitResult,
    sample_coordinate_proposals,
)
from ..._coordinate_generation._providers import CoordinateProviderProvenance
from ..._coordinate_generation._support import (
    CoordinateGeometryPolicy,
    CoordinateResourcePolicy,
    prepare_coordinate_support,
)
from .._construct import ProteinAtomKey, ProteinConstruct
from .._hypotheses import ProteinStructureHypothesis
from .._qualification import PreparedProteinQualification, ProteinGeometryEvidence
from ._internal_coordinates import (
    prepare_protein_coordinate_decoder,
    PreparedProteinCoordinateDecoder,
    ProteinDecodedCoordinates,
)


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


def prepare_bound_protein_coordinate_generation(
    binding,
    qualification,
    *,
    resources=CoordinateResourcePolicy(),
    **decoder_options,
):
    """Bind sparse support and internal decoding to one full protein qualification."""
    if not isinstance(qualification, PreparedProteinQualification):
        raise TypeError("Full PreparedProteinQualification evidence is required.")
    if qualification.binding_id != binding.binding_id:
        raise ValueError("Protein binding and geometry qualification differ.")
    system = binding.force_field.system
    ids = np.asarray(system.plan.particle_ids)
    active = np.asarray(system.active_mask)
    template = AtomisticBatch(
        np.asarray(system.plan.atomic_numbers)[None],
        np.asarray(binding.realized_positions)[None],
        np.asarray(system.plan.masses)[None],
        system.plan.units.scale,
        particle_ids=ids[None],
        atom_type_ids=np.asarray(system.plan.atom_type_ids)[None],
        element_mask=np.asarray(system.plan.element_mask)[None],
        atom_mask=active[None],
        structure_ids=(binding.hypothesis.hypothesis_id,),
    )
    bond_rows = np.asarray(qualification.bond_indices)
    chiral_rows = np.asarray(qualification.chirality_indices)
    geometry = CoordinateGeometryPolicy(
        tuple(tuple(int(ids[index]) for index in row) for row in bond_rows),
        tuple(
            (float(lower), float(upper))
            for lower, upper in zip(
                np.asarray(qualification.bond_lower),
                np.asarray(qualification.bond_upper),
                strict=True,
            )
        ),
        tuple(tuple(int(ids[index]) for index in row) for row in chiral_rows),
        (1,) * len(chiral_rows),
        qualification.minimum_chiral_volume,
        "full-protein-qualification-sparse-binding:" + qualification.qualification_id,
        achiral=not len(chiral_rows),
    )
    atom_ids = dict(zip(binding.atom_keys, binding.atom_ids, strict=True))
    first = binding.chemistry.construct.residue_keys[0]
    gauge = tuple(atom_ids[ProteinAtomKey(first, name)] for name in ("N", "CA", "C"))
    support = prepare_protein_coordinate_support(
        binding.chemistry.construct,
        template,
        atom_ids,
        gauge_atom_ids=gauge,
        geometry=geometry,
        resources=resources,
    )
    decoder = prepare_protein_coordinate_decoder(binding, support, **decoder_options)
    return support, decoder


class ProteinCoordinateFailureEvidence(StrictModule):
    """One explicit failure bit per requested sample and evaluated gate."""

    raw_nonfinite: object
    solver_failed: object
    periodic_representation_failed: object
    closure_failed: object
    decoded_nonfinite: object
    sparse_gauge_failed: object
    sparse_bond_failed: object
    sparse_chirality_failed: object
    full_bond_failed: object
    full_chirality_failed: object
    full_clash_failed: object
    full_peptide_failed: object
    full_torsion_failed: object
    successful: object
    evidence_scope: str = eqx.field(
        static=True,
        default="unfiltered requested samples; no rejection resampling or hidden minimization",
    )


@dataclass(frozen=True)
class ProteinCoordinateProposalBatch:
    """Fixed-construct internal proposals with sparse and full geometry evidence."""

    internal_coordinates: object
    raw_decoded_coordinates: object
    decoded_coordinates: object
    decoder_evidence: ProteinDecodedCoordinates
    sparse_geometry: object
    protein_geometry: ProteinGeometryEvidence
    failures: ProteinCoordinateFailureEvidence
    conditions: object
    solver_valid: object
    solver_status: object
    sample_ids: tuple[str, ...]
    parent_fit_id: str
    plan_id: str
    qualification_id: str
    rights: tuple[ReferenceArtifactManifest, ...]
    physical_energy_evidence: None = None
    scientific_scope: str = (
        "fixed-construct non-equilibrium proposals only; not sequence-general and "
        "generated frequency is not physical population"
    )


def sample_protein_coordinate_proposals(
    fit,
    key,
    conditions,
    qualification,
    *,
    commercial_use=False,
    export=False,
    rtol=1e-5,
    atol=1e-7,
    max_steps=1024,
):
    """Sample once per request and retain raw, decoded, and every failure gate."""
    if not isinstance(fit, CoordinateFitResult) or not isinstance(
        qualification, PreparedProteinQualification
    ):
        raise TypeError(
            "Protein proposal qualification requires a coordinate fit and full geometry plan."
        )
    decoder = fit.model.decoder
    if not isinstance(decoder, PreparedProteinCoordinateDecoder):
        raise ValueError("Protein proposals require a fitted protein internal decoder.")
    if decoder.plan.binding_id != qualification.binding_id:
        raise ValueError(
            "Decoder and full protein qualification bind different chemistry."
        )
    generic = sample_coordinate_proposals(
        fit,
        key,
        conditions,
        commercial_use=commercial_use,
        export=export,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    decoded = generic.decoder_evidence
    if not isinstance(decoded, ProteinDecodedCoordinates):
        raise TypeError(
            "Generic coordinate sampling did not retain protein decoder evidence."
        )
    full = jax.vmap(qualification.evaluate)(generic.canonical_positions)
    raw_nonfinite = ~jnp.all(jnp.isfinite(generic.raw_coordinates), axis=1)
    closure_failed = ~jnp.all(decoded.closure.valid, axis=1)
    sparse = generic.qualification
    failures = ProteinCoordinateFailureEvidence(
        raw_nonfinite,
        ~jnp.asarray(generic.solver_valid, dtype=bool),
        ~jnp.all(decoded.periodic_valid, axis=1),
        closure_failed,
        ~decoded.finite,
        ~sparse.gauge_valid,
        ~sparse.bond_valid,
        ~sparse.chirality_valid,
        ~jnp.all(full.covalent_valid, axis=1),
        ~jnp.all(full.chirality_valid, axis=1),
        ~jnp.all(full.clash_free, axis=1),
        ~jnp.all(full.peptide_planar, axis=1),
        ~jnp.all(full.torsion_valid, axis=1),
        (
            jnp.asarray(generic.solver_valid, dtype=bool)
            & decoded.valid
            & sparse.accepted
            & full.successful
        ),
    )
    return ProteinCoordinateProposalBatch(
        generic.raw_coordinates,
        generic.raw_positions,
        generic.canonical_positions,
        decoded,
        sparse,
        full,
        failures,
        generic.conditions,
        generic.solver_valid,
        generic.solver_status,
        generic.sample_ids,
        generic.parent_fit_id,
        decoder.plan.plan_id,
        qualification.qualification_id,
        generic.rights,
    )
