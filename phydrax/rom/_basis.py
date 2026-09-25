#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._identity import NumericRevision, SemanticProvenance
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import ArraySpace, DiagonalPairing, LinearSubspace
from ..ml.decomposition import SubspaceModel


BasisRole: TypeAlias = Literal["state", "nonlinear-term", "residual", "roq"]


def _identifier(name: str, value: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be a non-empty string.")
    return identifier


class ReducedBasisArtifact(StrictModule, NonTrainableState):
    """Numerical linear basis bound to its physical support and provenance."""

    subspace: LinearSubspace
    numeric_revision: NumericRevision
    role: BasisRole = eqx.field(static=True)
    state_contract_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    source_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def __init__(
        self,
        subspace: LinearSubspace,
        /,
        *,
        role: BasisRole,
        state_contract_id: str,
        support_id: str,
        measure_id: str,
        geometry_id: str,
        source_artifact_ids: Sequence[str],
        evidence_ids: Sequence[str] = (),
    ):
        if not isinstance(subspace, LinearSubspace):
            raise TypeError("subspace must be a LinearSubspace.")
        if subspace.batch_shape:
            raise ValueError("Reduced basis artifacts require one unbatched subspace.")
        if role not in ("state", "nonlinear-term", "residual", "roq"):
            raise ValueError("role must identify one supported basis role.")
        rank = int(np.asarray(subspace.dimension))
        if rank <= 0:
            raise ValueError("Reduced basis artifacts require positive active rank.")
        sources = tuple(
            _identifier("source_artifact_id", item) for item in source_artifact_ids
        )
        if not sources or len(set(sources)) != len(sources):
            raise ValueError("source_artifact_ids must be unique and non-empty.")
        evidence = tuple(_identifier("evidence_id", item) for item in evidence_ids)
        if len(set(evidence)) != len(evidence):
            raise ValueError("evidence_ids must be unique.")
        revision = NumericRevision(
            SemanticProvenance(
                {
                    "kind": "reduced-basis",
                    "role": role,
                    "space": subspace.space.space_id,
                }
            ),
            {"basis": subspace.basis, "dimension": subspace.dimension},
        )
        state_contract = _identifier("state_contract_id", state_contract_id)
        support = _identifier("support_id", support_id)
        measure = _identifier("measure_id", measure_id)
        geometry = _identifier("geometry_id", geometry_id)
        self.subspace = subspace
        self.numeric_revision = revision
        self.role = role
        self.state_contract_id = state_contract
        self.support_id = support
        self.measure_id = measure
        self.geometry_id = geometry
        self.source_artifact_ids = sources
        self.evidence_ids = evidence
        self.rank = rank
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "reduced-basis-artifact",
                "role": role,
                "space": subspace.space.space_id,
                "state_contract": state_contract,
                "support": support,
                "measure": measure,
                "geometry": geometry,
                "sources": list(sources),
                "evidence": list(evidence),
                "revision": revision.revision_id,
            }
        )

    @property
    def basis_matrix(self) -> Array:
        return self.subspace.basis[:, : self.rank]


def reduced_basis_from_subspace_model(
    model: SubspaceModel,
    /,
    *,
    role: BasisRole,
    state_contract_id: str,
    support_id: str,
    measure_id: str,
    geometry_id: str,
    source_artifact_ids: Sequence[str],
    evidence_ids: Sequence[str] = (),
) -> tuple[ReducedBasisArtifact, Array]:
    """Bind one array POD model to a physical basis artifact and affine offset."""
    if not isinstance(model, SubspaceModel):
        raise TypeError("model must be a SubspaceModel.")
    if model.case_shape:
        raise ValueError("A reduced basis requires one global, unbatched subspace model.")
    support = np.asarray(model.feature_support)
    metric = np.asarray(model.feature_metric)
    if not np.all(support) or not np.all(np.isfinite(metric) & (metric > 0.0)):
        raise ValueError(
            "Intrusive reduced bases require every feature to have positive physical measure."
        )
    pairing = DiagonalPairing(jnp.asarray(model.feature_metric))
    space = ArraySpace(
        (model.in_size,),
        dtype=model.components.dtype,
        pairing=pairing,
        space_id=canonical_fingerprint(
            {
                "kind": "subspace-model-array-space",
                "state_contract": state_contract_id,
                "support": support_id,
                "measure": measure_id,
                "geometry": geometry_id,
                "metric": array_tree_fingerprint(model.feature_metric)["sha256"],
            }
        ),
    )
    basis = jnp.swapaxes(model.components, -1, -2)
    subspace = LinearSubspace(
        space,
        basis,
        orthonormal=True,
        subspace_id=canonical_fingerprint(
            {
                "kind": "subspace-model-linear-subspace",
                "basis": array_tree_fingerprint(basis)["sha256"],
                "space": space.space_id,
            }
        ),
    )
    artifact = ReducedBasisArtifact(
        subspace,
        role=role,
        state_contract_id=state_contract_id,
        support_id=support_id,
        measure_id=measure_id,
        geometry_id=geometry_id,
        source_artifact_ids=source_artifact_ids,
        evidence_ids=evidence_ids,
    )
    return artifact, jnp.asarray(model.offset)


__all__ = [
    "BasisRole",
    "ReducedBasisArtifact",
    "reduced_basis_from_subspace_model",
]
