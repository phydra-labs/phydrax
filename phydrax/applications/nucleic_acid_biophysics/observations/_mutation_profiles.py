# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Externally mapped single-molecule mutation profiles with immutable lineage."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....qualification import ReferenceArtifactManifest


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{owner} must be a non-empty canonical identifier.")
    return value


def _identifiers(values, owner: str, /, *, nonempty: bool = True) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{owner} must be a tuple.")
    result = tuple(_identifier(value, owner) for value in values)
    if (nonempty and not result) or len(set(result)) != len(result):
        raise ValueError(f"{owner} must contain unique canonical identifiers.")
    return result


@dataclass(frozen=True, slots=True, init=False)
class MutationProfileCase:
    """One source-resolved assay case; independence is preparation-and-construct based."""

    case_id: str
    independent_unit_id: str
    construct_id: str
    condition_id: str
    preparation_id: str
    batch_id: str
    replicate_id: str
    reagent_id: str
    protocol_id: str
    source_manifest_ids: tuple[str, ...]
    parent_case_ids: tuple[str, ...]

    def __init__(
        self,
        *,
        construct_id: str,
        condition_id: str,
        preparation_id: str,
        batch_id: str,
        replicate_id: str,
        reagent_id: str,
        protocol_id: str,
        source_manifest_ids: tuple[str, ...],
        parent_case_ids: tuple[str, ...] = (),
    ):
        values = tuple(
            _identifier(value, name)
            for value, name in (
                (construct_id, "construct ID"),
                (condition_id, "condition ID"),
                (preparation_id, "preparation ID"),
                (batch_id, "batch ID"),
                (replicate_id, "replicate ID"),
                (reagent_id, "reagent ID"),
                (protocol_id, "protocol ID"),
            )
        )
        sources = _identifiers(source_manifest_ids, "source manifest IDs")
        parents = _identifiers(parent_case_ids, "parent case IDs", nonempty=False)
        independent = canonical_fingerprint(
            {
                "kind": "mutation-profile-independent-unit",
                "construct": values[0],
                "preparation": values[2],
            }
        )
        case = canonical_fingerprint(
            {
                "kind": "mutation-profile-case",
                "coordinates": values,
                "sources": sources,
                "parents": parents,
            }
        )
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "independent_unit_id", independent)
        for name, value in zip(
            (
                "construct_id",
                "condition_id",
                "preparation_id",
                "batch_id",
                "replicate_id",
                "reagent_id",
                "protocol_id",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_manifest_ids", sources)
        object.__setattr__(self, "parent_case_ids", parents)


class MutationProfileBatch(StrictModule, NonTrainableState):
    """Mapped-read rows with explicit coverage, missingness, and provenance.

    ``coverage`` is mapped depth and ``observed_mask`` is effective depth. They
    remain distinct because a covered base can be ineligible for the likelihood.
    Excluded mapping categories and zero-effective-depth rows are retained; the
    likelihood-ready rows are exposed by :attr:`analysis_mask`.
    """

    mutation: Array
    observed_mask: Array
    coverage: Array
    construct_index: Array
    condition_index: Array
    replicate_index: Array
    preparation_index: Array
    batch_index: Array
    reagent_index: Array
    protocol_index: Array
    source_index: Array
    case_index: Array
    source_row_index: Array
    mapping_category_index: Array
    mapping_included: Array
    nucleotide_ids: tuple[str, ...] = eqx.field(static=True)
    construct_ids: tuple[str, ...] = eqx.field(static=True)
    condition_ids: tuple[str, ...] = eqx.field(static=True)
    replicate_ids: tuple[str, ...] = eqx.field(static=True)
    preparation_ids: tuple[str, ...] = eqx.field(static=True)
    batch_ids: tuple[str, ...] = eqx.field(static=True)
    reagent_ids: tuple[str, ...] = eqx.field(static=True)
    protocol_ids: tuple[str, ...] = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    mapping_category_ids: tuple[str, ...] = eqx.field(static=True)
    cases: tuple[MutationProfileCase, ...] = eqx.field(static=True)
    sources: tuple[ReferenceArtifactManifest, ...]
    batch_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        mutation: ArrayLike,
        observed_mask: ArrayLike,
        coverage: ArrayLike,
        construct_index: ArrayLike,
        condition_index: ArrayLike,
        replicate_index: ArrayLike,
        source_ids: tuple[str, ...],
        /,
        *,
        nucleotide_ids: tuple[str, ...],
        preparation_index: ArrayLike,
        batch_index: ArrayLike,
        reagent_index: ArrayLike,
        protocol_index: ArrayLike,
        source_index: ArrayLike,
        case_index: ArrayLike,
        source_row_index: ArrayLike,
        mapping_category_index: ArrayLike,
        mapping_included: ArrayLike,
        construct_ids: tuple[str, ...],
        condition_ids: tuple[str, ...],
        replicate_ids: tuple[str, ...],
        preparation_ids: tuple[str, ...],
        batch_ids: tuple[str, ...],
        reagent_ids: tuple[str, ...],
        protocol_ids: tuple[str, ...],
        mapping_category_ids: tuple[str, ...],
        cases: tuple[MutationProfileCase, ...],
        sources: tuple[ReferenceArtifactManifest, ...],
    ):
        if (
            not isinstance(cases, tuple)
            or not cases
            or any(not isinstance(case, MutationProfileCase) for case in cases)
        ):
            raise TypeError(
                "cases must be a non-empty tuple of MutationProfileCase records."
            )
        if (
            not isinstance(sources, tuple)
            or not sources
            or any(
                not isinstance(source, ReferenceArtifactManifest) for source in sources
            )
        ):
            raise TypeError(
                "sources must contain admitted ReferenceArtifactManifest values."
            )
        mutation_ = np.asarray(mutation)
        observed_ = np.asarray(observed_mask)
        coverage_ = np.asarray(coverage)
        if (
            mutation_.ndim != 2
            or not mutation_.size
            or observed_.shape != mutation_.shape
            or coverage_.shape != mutation_.shape
        ):
            raise ValueError(
                "Mutation, effective-depth, and coverage arrays must share a non-empty "
                "(profile, nucleotide) shape."
            )
        for value, name in (
            (mutation_, "mutation"),
            (observed_, "observed mask"),
            (coverage_, "coverage"),
        ):
            if value.dtype != bool and not np.issubdtype(value.dtype, np.integer):
                raise TypeError(f"Mapped {name} must be a boolean or integer array.")
        mutation_ = mutation_.astype(np.int8)
        observed_ = observed_.astype(bool)
        coverage_ = coverage_.astype(np.int8)
        if (
            np.any((mutation_ != 0) & (mutation_ != 1))
            or np.any((coverage_ != 0) & (coverage_ != 1))
            or np.any(observed_ & (coverage_ == 0))
            or np.any((mutation_ == 1) & ~observed_)
        ):
            raise ValueError(
                "Coverage and mutation must be binary, effective depth must be covered, "
                "and every mutation must be effectively observed."
            )
        profiles, sites = mutation_.shape
        nucleotide_ids_ = _identifiers(nucleotide_ids, "nucleotide IDs")
        if len(nucleotide_ids_) != sites:
            raise ValueError("Nucleotide IDs must map the complete trailing site axis.")
        vocabularies = (
            _identifiers(construct_ids, "construct IDs"),
            _identifiers(condition_ids, "condition IDs"),
            _identifiers(replicate_ids, "replicate IDs"),
            _identifiers(preparation_ids, "preparation IDs"),
            _identifiers(batch_ids, "batch IDs"),
            _identifiers(reagent_ids, "reagent IDs"),
            _identifiers(protocol_ids, "protocol IDs"),
            _identifiers(source_ids, "source IDs"),
            _identifiers(tuple(case.case_id for case in cases), "case IDs"),
            _identifiers(mapping_category_ids, "mapping category IDs"),
        )
        if tuple(source.manifest_id for source in sources) != vocabularies[7]:
            raise ValueError(
                "source_ids must preserve the admitted source-manifest order."
            )
        if any(
            not set(case.source_manifest_ids).issubset(vocabularies[7]) for case in cases
        ):
            raise ValueError(
                "Every case source must belong to the admitted batch sources."
            )
        index_inputs = (
            construct_index,
            condition_index,
            replicate_index,
            preparation_index,
            batch_index,
            reagent_index,
            protocol_index,
            source_index,
            case_index,
            mapping_category_index,
        )
        indices: list[np.ndarray] = []
        for raw, vocabulary in zip(index_inputs, vocabularies, strict=True):
            value = np.asarray(raw)
            if value.shape != (profiles,) or not np.issubdtype(value.dtype, np.integer):
                raise TypeError(
                    "Every profile metadata index must be a one-dimensional integer array."
                )
            value = value.astype(np.int32)
            if np.any(value < 0) or np.any(value >= len(vocabulary)):
                raise ValueError("A profile metadata index is outside its vocabulary.")
            indices.append(value)
        source_rows = np.asarray(source_row_index)
        included = np.asarray(mapping_included)
        if (
            source_rows.shape != (profiles,)
            or not np.issubdtype(source_rows.dtype, np.integer)
            or np.any(source_rows < 1)
            or included.shape != (profiles,)
            or included.dtype != bool
        ):
            raise ValueError(
                "Source rows and boolean mapping inclusion must provide one valid value per profile."
            )
        included = included.astype(bool)
        source_coordinates = tuple(
            zip(indices[7].tolist(), source_rows.astype(np.int64).tolist(), strict=True)
        )
        if len(set(source_coordinates)) != profiles:
            raise ValueError(
                "Each source row may occur only once in a mutation-profile batch."
            )
        for row in range(profiles):
            case = cases[int(indices[8][row])]
            indexed = tuple(
                vocabulary[int(index[row])]
                for vocabulary, index in zip(vocabularies[:7], indices[:7], strict=True)
            )
            declared = (
                case.construct_id,
                case.condition_id,
                case.replicate_id,
                case.preparation_id,
                case.batch_id,
                case.reagent_id,
                case.protocol_id,
            )
            if (
                indexed != declared
                or vocabularies[7][int(indices[7][row])] not in case.source_manifest_ids
            ):
                raise ValueError(
                    "Profile indices disagree with immutable case provenance."
                )
        replicate_coordinates: dict[int, tuple[int, int, int, int, int, int]] = {}
        for row in range(profiles):
            replicate = int(indices[2][row])
            coordinates = tuple(int(indices[index][row]) for index in (0, 1, 3, 4, 5, 6))
            previous = replicate_coordinates.setdefault(replicate, coordinates)
            if previous != coordinates:
                raise ValueError(
                    "A replicate ID must identify one construct, condition, preparation, "
                    "batch, reagent, and protocol."
                )
        (
            self.construct_ids,
            self.condition_ids,
            self.replicate_ids,
            self.preparation_ids,
            self.batch_ids,
            self.reagent_ids,
            self.protocol_ids,
            self.source_ids,
            self.case_ids,
            self.mapping_category_ids,
        ) = vocabularies
        self.nucleotide_ids = nucleotide_ids_
        self.cases = cases
        self.sources = sources
        self.mutation = jnp.asarray(mutation_, dtype=jnp.int8)
        self.observed_mask = jnp.asarray(observed_)
        self.coverage = jnp.asarray(coverage_, dtype=jnp.int8)
        (
            self.construct_index,
            self.condition_index,
            self.replicate_index,
            self.preparation_index,
            self.batch_index,
            self.reagent_index,
            self.protocol_index,
            self.source_index,
            self.case_index,
            self.mapping_category_index,
        ) = tuple(jnp.asarray(value, dtype=jnp.int32) for value in indices)
        self.source_row_index = jnp.asarray(source_rows, dtype=jnp.int32)
        self.mapping_included = jnp.asarray(included)
        self.batch_fingerprint = canonical_fingerprint(
            {
                "kind": "mutation-profile-batch",
                "nucleotides": nucleotide_ids_,
                "vocabularies": vocabularies,
                "cases": [case.case_id for case in cases],
                "arrays": array_tree_fingerprint(
                    (mutation_, observed_, coverage_, *indices, source_rows, included)
                ),
            }
        )

    @property
    def profile_count(self) -> int:
        return int(self.mutation.shape[0])

    @property
    def nucleotide_count(self) -> int:
        return int(self.mutation.shape[1])

    @property
    def analysis_mask(self) -> Array:
        return self.mapping_included & jnp.any(self.observed_mask, axis=-1)

    @property
    def excluded_profile_count(self) -> Array:
        return jnp.sum((~self.analysis_mask).astype(jnp.int32))

    def profile_mask_for_cases(self, case_ids, /) -> Array:
        """Select complete source rows for exact admitted case identities."""
        identifiers = tuple(case_ids)
        if (
            not identifiers
            or len(set(identifiers)) != len(identifiers)
            or any(not isinstance(value, str) or not value for value in identifiers)
        ):
            raise ValueError(
                "case_ids must contain unique non-empty admitted case identities."
            )
        unknown = set(identifiers) - set(self.case_ids)
        if unknown:
            raise ValueError(
                f"Unknown mutation-profile case IDs: {tuple(sorted(unknown))!r}."
            )
        selected = jnp.zeros((self.profile_count,), dtype=bool)
        for case_id in identifiers:
            selected = selected | (self.case_index == self.case_ids.index(case_id))
        return selected

    def analysis_profile_mask(self, profile_mask: ArrayLike | None = None, /) -> Array:
        """Intersect a caller-frozen profile split with admitted analysis rows."""
        if profile_mask is None:
            return self.analysis_mask
        selected = np.asarray(profile_mask)
        if selected.shape != (self.profile_count,) or selected.dtype != bool:
            raise TypeError(
                "profile_mask must be a boolean value for every retained profile."
            )
        return self.analysis_mask & jnp.asarray(selected)

    def require_rights(
        self,
        requested_use=None,
        *,
        profile_mask: ArrayLike | None = None,
    ) -> tuple[str, ...]:
        use = {} if requested_use is None else dict(requested_use)
        if profile_mask is None:
            source_indices = range(len(self.sources))
        else:
            selected = np.asarray(profile_mask)
            if selected.shape != (self.profile_count,) or selected.dtype != bool:
                raise TypeError(
                    "profile_mask must be a boolean value for every retained profile."
                )
            source_indices = tuple(
                sorted(set(np.asarray(self.source_index)[selected].tolist()))
            )
            if not source_indices:
                raise ValueError(
                    "At least one profile source must be selected for requested use."
                )
        return tuple(
            self.sources[index].require_rights(**use) for index in source_indices
        )


__all__ = ["MutationProfileBatch", "MutationProfileCase"]
