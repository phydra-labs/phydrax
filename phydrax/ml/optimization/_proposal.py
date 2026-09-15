#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...optim._programming._mixed_integer import (
    MixedIntegerCandidate,
    MixedIntegerCandidateAudit,
    MixedIntegerProgram,
)
from ...optim._programming._mixed_integer_audit import (
    audit_mixed_integer_candidate,
)
from ...optim._programming._mixed_integer_policy import (
    MixedIntegerCertification,
)


class AdjacentLatticeMaterializer(StrictModule, NonTrainableState):
    """Deterministically select floor or ceil for each discrete coordinate."""

    tie_up: bool = eqx.field(static=True)

    def __init__(self, *, tie_up: bool = True):
        self.tie_up = bool(tie_up)

    def materialize(
        self,
        program: MixedIntegerProgram,
        latent: ArrayLike,
        thresholds: ArrayLike,
        /,
    ) -> Array:
        if not isinstance(program, MixedIntegerProgram):
            raise TypeError("program must be a MixedIntegerProgram.")
        value = jnp.asarray(latent, dtype=program.relaxation.linear.dtype)
        if value.shape != (program.relaxation.num_variables,):
            raise ValueError("latent proposal has the wrong shape.")
        threshold = jnp.asarray(thresholds, dtype=value.dtype)
        if threshold.shape != (len(program.discrete_indices),):
            raise ValueError("thresholds must match the discrete coordinate count.")
        threshold_host = np.asarray(threshold)
        if not np.all(np.isfinite(threshold_host)) or np.any(
            (threshold_host < 0.0) | (threshold_host > 1.0)
        ):
            raise ValueError("thresholds must be finite values in [0, 1].")
        indices = jnp.asarray(program.discrete_indices, dtype=jnp.int32)
        discrete = value[indices]
        lower_lattice = jnp.floor(discrete)
        fraction = discrete - lower_lattice
        select_upper = fraction >= threshold if self.tie_up else fraction > threshold
        rounded = lower_lattice + select_upper.astype(value.dtype)
        rounded = jnp.clip(
            rounded,
            program.relaxation.lower_bounds[indices],
            program.relaxation.upper_bounds[indices],
        )
        return value.at[indices].set(rounded)


class MixedIntegerProposalManifest(StrictModule, NonTrainableState):
    """Frozen learned proposal identity bound to one canonical family."""

    structure_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    def __init__(
        self,
        structure_id: str,
        model_id: str,
        /,
        *,
        support_id: str = "unqualified-support",
    ):
        structure = str(structure_id)
        model = str(model_id)
        support = str(support_id)
        if any(not value for value in (structure, model, support)):
            raise ValueError("Proposal manifest identifiers must be nonempty.")
        self.structure_id = structure
        self.model_id = model
        self.support_id = support
        self.proposal_id = canonical_fingerprint(
            {
                "kind": "mixed-integer-proposal-manifest",
                "structure": structure,
                "model": model,
                "support": support,
            }
        )


class MixedIntegerProposalResult(StrictModule, NonTrainableState):
    latent: Array
    materialized: Array
    candidate: MixedIntegerCandidate
    audit: MixedIntegerCandidateAudit
    support_accepted: Array
    accepted_as_start: Array
    manifest: MixedIntegerProposalManifest


class ParametricMixedIntegerProposal(StrictModule):
    """Callable latent/threshold model whose output remains solver input only."""

    model: Callable[[Any], tuple[Array, Array]] = eqx.field(static=True)
    materializer: AdjacentLatticeMaterializer
    manifest: MixedIntegerProposalManifest

    def __init__(
        self,
        model: Callable[[Any], tuple[Array, Array]],
        manifest: MixedIntegerProposalManifest,
        /,
        *,
        materializer: AdjacentLatticeMaterializer | None = None,
    ):
        if not callable(model):
            raise TypeError("model must be callable.")
        if not isinstance(manifest, MixedIntegerProposalManifest):
            raise TypeError("manifest must be MixedIntegerProposalManifest.")
        selected = AdjacentLatticeMaterializer() if materializer is None else materializer
        if not isinstance(selected, AdjacentLatticeMaterializer):
            raise TypeError("materializer must be AdjacentLatticeMaterializer.")
        self.model = model
        self.materializer = selected
        self.manifest = manifest

    def propose(
        self,
        program: MixedIntegerProgram,
        context: Any,
        /,
        *,
        certification: MixedIntegerCertification | None = None,
        support_accepted: bool = False,
    ) -> MixedIntegerProposalResult:
        latent, thresholds = self.model(context)
        return materialize_mixed_integer_proposal(
            program,
            latent,
            thresholds,
            self.manifest,
            materializer=self.materializer,
            certification=certification,
            support_accepted=support_accepted,
        )


def materialize_mixed_integer_proposal(
    program: MixedIntegerProgram,
    latent: ArrayLike,
    thresholds: ArrayLike,
    manifest: MixedIntegerProposalManifest,
    /,
    *,
    materializer: AdjacentLatticeMaterializer | None = None,
    certification: MixedIntegerCertification | None = None,
    support_accepted: bool = False,
) -> MixedIntegerProposalResult:
    if not isinstance(program, MixedIntegerProgram):
        raise TypeError("program must be a MixedIntegerProgram.")
    if not isinstance(manifest, MixedIntegerProposalManifest):
        raise TypeError("manifest must be MixedIntegerProposalManifest.")
    if manifest.structure_id != program.structure_id:
        raise ValueError("Proposal manifest does not match the mixed-integer structure.")
    selected = AdjacentLatticeMaterializer() if materializer is None else materializer
    certification_ = (
        MixedIntegerCertification() if certification is None else certification
    )
    if not isinstance(selected, AdjacentLatticeMaterializer):
        raise TypeError("materializer must be AdjacentLatticeMaterializer.")
    if not isinstance(certification_, MixedIntegerCertification):
        raise TypeError("certification must be MixedIntegerCertification.")
    latent_ = jnp.asarray(latent, dtype=program.relaxation.linear.dtype)
    materialized = selected.materialize(program, latent_, thresholds)
    candidate = MixedIntegerCandidate(
        materialized,
        source_kind="learned-proposal",
        source_id=manifest.proposal_id,
    )
    audit = audit_mixed_integer_candidate(program, candidate, certification_)
    support = jnp.asarray(bool(support_accepted))
    accepted = support & audit.valid
    return MixedIntegerProposalResult(
        latent_,
        materialized,
        candidate,
        audit,
        support,
        accepted,
        manifest,
    )


__all__ = [
    "AdjacentLatticeMaterializer",
    "MixedIntegerProposalManifest",
    "MixedIntegerProposalResult",
    "ParametricMixedIntegerProposal",
    "materialize_mixed_integer_proposal",
]
