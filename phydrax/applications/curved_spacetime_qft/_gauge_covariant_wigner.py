#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Wilson-line gauge-covariant Wigner transport on existing gauge links."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import OrientedEdgePathPlan
from ...graph._matrix_gauge import (
    gauge_transform_links,
    MatrixGaugeLinkSpace,
    path_holonomy,
)
from ...metrix import SpecialUnitaryGroup, UnitaryGroup
from ...metrix._gauge_representation import (
    AbstractGaugeRepresentation,
    AdjointGaugeRepresentation,
    U1ChargeRepresentation,
)
from ..cosmology._quantum_dark_kinetics import QuantumKineticState


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


class GaugeCovariantWignerProfile(StrictModule, NonTrainableState):
    support: tuple[str, ...] = eqx.field(static=True)
    refusals: tuple[str, ...] = eqx.field(static=True)
    differentiation: str = eqx.field(static=True)
    rights_id: str = eqx.field(static=True)
    provenance_ids: tuple[str, ...] = eqx.field(static=True)
    production_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    production_ready: bool = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(self, production_evidence_ids: Sequence[str] = (), /):
        evidence = tuple(
            _identifier(value, "production_evidence_id")
            for value in production_evidence_ids
        )
        if len(set(evidence)) != len(evidence):
            raise ValueError("Production evidence identities must be unique.")
        support = (
            "compact-u1-charge-representation",
            "existing-matrix-gauge-link-space",
            "fixed-capacity-oriented-wilson-lines",
            "nonabelian-real-adjoint-when-explicitly-owned",
        )
        refusals = (
            "implicit-continuum-gauge-potential",
            "nonabelian-fundamental-without-real-link-owner",
            "mixed-path-anchor-wigner-sum",
            "implicit-frame-or-unit-conversion",
        )
        differentiation = "algorithmic-through-fixed-wilson-products-and-fourier-sum"
        rights_id = "phydra-native-no-external-provider"
        provenance_ids = ("existing-link-wilson-line-wigner-transform",)
        self.support = support
        self.refusals = refusals
        self.differentiation = differentiation
        self.rights_id = rights_id
        self.provenance_ids = provenance_ids
        self.production_evidence_ids = evidence
        self.production_ready = bool(evidence)
        self.profile_id = canonical_fingerprint(
            {
                "kind": "gauge-covariant-wigner-profile",
                "support": support,
                "refusals": refusals,
                "differentiation": differentiation,
                "rights_id": rights_id,
                "provenance_ids": provenance_ids,
                "production_evidence_ids": evidence,
            }
        )


class GaugeCovariantWignerPlan(StrictModule, NonTrainableState):
    quantum_support: QuantumKineticState
    link_space: MatrixGaugeLinkSpace
    representation: AbstractGaugeRepresentation
    paths: OrientedEdgePathPlan
    profile: GaugeCovariantWignerProfile
    momentum_count: int = eqx.field(static=True)
    representation_dimension: int = eqx.field(static=True)
    anchor_vertex: int = eqx.field(static=True)
    nonabelian_link_owner_id: str | None = eqx.field(static=True)
    maximum_work_bytes: int = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    link_space_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quantum_support: QuantumKineticState,
        link_space: MatrixGaugeLinkSpace,
        representation: AbstractGaugeRepresentation,
        paths: OrientedEdgePathPlan,
        /,
        *,
        nonabelian_link_owner_id: str | None = None,
        maximum_work_bytes: int = 256_000_000,
        production_evidence_ids: Sequence[str] = (),
    ):
        if not isinstance(quantum_support, QuantumKineticState):
            raise TypeError("quantum_support must be a QuantumKineticState.")
        if not isinstance(link_space, MatrixGaugeLinkSpace):
            raise TypeError("link_space must be the existing MatrixGaugeLinkSpace owner.")
        if not isinstance(representation, AbstractGaugeRepresentation):
            raise TypeError("representation must implement AbstractGaugeRepresentation.")
        if not isinstance(paths, OrientedEdgePathPlan):
            raise TypeError("paths must be an OrientedEdgePathPlan.")
        if paths.topology_id != link_space.topology.topology_id:
            raise ValueError(
                "Wilson paths and gauge links must share exact topology identity."
            )
        if representation.group.group_id != link_space.group.group_id:
            raise ValueError(
                "Gauge representation and link space must share exact group identity."
            )
        starts = np.asarray(paths.start_vertices)
        if np.any(starts != starts[0]):
            raise ValueError(
                "A Wigner sum requires every Wilson path to share one anchor vertex."
            )
        owner = None
        if isinstance(representation, U1ChargeRepresentation):
            if (
                not isinstance(link_space.group, UnitaryGroup)
                or link_space.group.dimension != 1
            ):
                raise TypeError(
                    "U(1) Wigner transport requires one-dimensional unitary links."
                )
            if nonabelian_link_owner_id is not None:
                raise ValueError(
                    "U(1) transport must not declare a non-Abelian link owner."
                )
        else:
            if not isinstance(link_space.group, SpecialUnitaryGroup):
                raise TypeError(
                    "Non-Abelian Wigner transport requires special-unitary links."
                )
            if not isinstance(representation, AdjointGaugeRepresentation):
                raise TypeError(
                    "Non-Abelian Wigner transport fails closed without a real adjoint representation."
                )
            if nonabelian_link_owner_id is None:
                raise ValueError(
                    "Non-Abelian Wigner transport requires an explicit real-representation link owner."
                )
            owner = _identifier(nonabelian_link_owner_id, "nonabelian_link_owner_id")
        maximum = int(maximum_work_bytes)
        momentum_count = int(quantum_support.occupancy.shape[2])
        dimension = int(representation.dimension)
        complex_bytes = np.dtype(np.complex128).itemsize
        required = (
            (paths.num_paths + momentum_count) * dimension * dimension * complex_bytes
        )
        if maximum < 1 or required > maximum:
            raise ValueError(
                f"Gauge-covariant Wigner work requires {required} bytes; capacity is {maximum}."
            )
        profile = GaugeCovariantWignerProfile(production_evidence_ids)
        self.quantum_support = quantum_support
        self.link_space = link_space
        self.representation = representation
        self.paths = paths
        self.profile = profile
        self.momentum_count = momentum_count
        self.representation_dimension = dimension
        self.anchor_vertex = int(starts[0])
        self.nonabelian_link_owner_id = owner
        self.maximum_work_bytes = maximum
        self.support_id = quantum_support.support_id
        self.topology_id = paths.topology_id
        self.link_space_id = link_space.link_space_id
        self.representation_id = representation.representation_id
        self.frame_id = quantum_support.frame_id
        self.frame_realization_id = quantum_support.frame.realization_id()
        self.unit_contract_id = quantum_support.unit_contract_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-gauge-covariant-wigner-plan",
                "support": self.support_id,
                "link_space": self.link_space_id,
                "paths": paths.path_plan_id,
                "representation": self.representation_id,
                "anchor_vertex": self.anchor_vertex,
                "momentum_count": momentum_count,
                "nonabelian_link_owner": owner,
                "maximum_work_bytes": maximum,
                "frame": self.frame_id,
                "frame_realization": self.frame_realization_id,
                "units": self.unit_contract_id,
                "profile": profile.profile_id,
            }
        )


class GaugeCovariantWignerState(StrictModule):
    wigner_matrix: Array
    transported_bilocal: Array
    wilson_matrices: Array
    hermiticity_residual: Array
    finite: Array
    hermitian: Array
    frame_token: Array
    frame_time: Array
    frame_scale_factor: Array
    frame_realization_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    link_space_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)


class GaugeCovarianceEvidence(StrictModule):
    transformed: GaugeCovariantWignerState
    expected_wigner_matrix: Array
    covariance_residual: Array
    finite: Array
    covariant: Array
    plan_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)


def _bilocal(plan: GaugeCovariantWignerPlan, value: ArrayLike, /) -> Array:
    bilocal = jnp.asarray(value)
    expected = (
        plan.paths.num_paths,
        plan.representation_dimension,
        plan.representation_dimension,
    )
    if bilocal.shape != expected:
        raise ValueError(f"bilocal correlation must have shape {expected}.")
    return eqx.error_if(
        bilocal,
        ~jnp.all(jnp.isfinite(bilocal)),
        "Bilocal correlation must be finite.",
    )


def gauge_transform_bilocal(
    plan: GaugeCovariantWignerPlan,
    bilocal_correlation: ArrayLike,
    vertex_elements: ArrayLike,
    /,
) -> Array:
    """Transform ``C(end,start)`` so ``U(start,end) C(end,start)`` is covariant."""

    if not isinstance(plan, GaugeCovariantWignerPlan):
        raise TypeError("plan must be GaugeCovariantWignerPlan.")
    bilocal = _bilocal(plan, bilocal_correlation)
    vertices = jnp.asarray(vertex_elements)
    expected = (plan.link_space.num_vertices,) + plan.link_space.point_shape
    if vertices.shape != expected:
        raise ValueError(f"vertex_elements must have shape {expected}.")
    starts = vertices[plan.paths.start_vertices]
    ends = vertices[plan.paths.end_vertices]
    left = plan.representation.matrix(ends)
    right = plan.representation.matrix(plan.link_space.group.inverse(starts))
    return contract("pij,pjk,pkl->pil", left, bilocal, right)


def gauge_covariant_wigner_transform(
    plan: GaugeCovariantWignerPlan,
    links: ArrayLike,
    bilocal_correlation: ArrayLike,
    fourier_phases: ArrayLike,
    path_weights: ArrayLike,
    /,
    *,
    hermiticity_tolerance: float = 1.0e-10,
) -> GaugeCovariantWignerState:
    if not isinstance(plan, GaugeCovariantWignerPlan):
        raise TypeError("plan must be GaugeCovariantWignerPlan.")
    link_values = jnp.asarray(links)
    if link_values.shape != plan.link_space.configuration_shape:
        raise ValueError("Gauge links do not match the plan link space.")
    link_values = eqx.error_if(
        link_values,
        ~plan.link_space.contains(link_values),
        "Gauge links must be finite members of the declared link space.",
    )
    bilocal = _bilocal(plan, bilocal_correlation)
    phases = jnp.asarray(fourier_phases)
    weights = jnp.asarray(path_weights)
    if phases.shape != (plan.momentum_count, plan.paths.num_paths):
        raise ValueError("fourier_phases must have shape (momentum, path).")
    if weights.shape != (plan.paths.num_paths,):
        raise ValueError("path_weights must provide one finite real weight per path.")
    if jnp.issubdtype(weights.dtype, jnp.complexfloating):
        raise TypeError("path_weights must be real.")
    phases = eqx.error_if(
        phases,
        ~jnp.all(jnp.isfinite(phases)),
        "Wigner Fourier phases must be finite.",
    )
    weights = eqx.error_if(
        weights,
        ~jnp.all(jnp.isfinite(weights)),
        "Wigner path weights must be finite.",
    )
    holonomy = path_holonomy(plan.link_space, link_values, plan.paths)
    wilson = plan.representation.matrix(holonomy)
    transported = contract("pij,pjk->pik", wilson, bilocal)
    wigner = contract("qp,p,pij->qij", phases, weights, transported)
    residual = jnp.max(jnp.abs(wigner - jnp.conj(jnp.swapaxes(wigner, -1, -2))))
    finite = (
        jnp.all(jnp.isfinite(wilson))
        & jnp.all(jnp.isfinite(transported))
        & jnp.all(jnp.isfinite(wigner))
    )
    return GaugeCovariantWignerState(
        wigner,
        transported,
        wilson,
        residual,
        finite,
        residual <= hermiticity_tolerance,
        plan.quantum_support.frame.frame_token,
        plan.quantum_support.frame.time,
        plan.quantum_support.frame.scale_factor,
        plan.frame_realization_id,
        plan.plan_id,
        plan.support_id,
        plan.topology_id,
        plan.link_space_id,
        plan.representation_id,
        plan.frame_id,
        plan.unit_contract_id,
    )


def gauge_covariance_evidence(
    plan: GaugeCovariantWignerPlan,
    links: ArrayLike,
    bilocal_correlation: ArrayLike,
    fourier_phases: ArrayLike,
    path_weights: ArrayLike,
    vertex_elements: ArrayLike,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> GaugeCovarianceEvidence:
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Gauge covariance tolerance must be finite and nonnegative.")
    original = gauge_covariant_wigner_transform(
        plan,
        links,
        bilocal_correlation,
        fourier_phases,
        path_weights,
        hermiticity_tolerance=tolerance,
    )
    vertices = jnp.asarray(vertex_elements)
    transformed_links = gauge_transform_links(plan.link_space, links, vertices)
    transformed_bilocal = gauge_transform_bilocal(plan, bilocal_correlation, vertices)
    transformed = gauge_covariant_wigner_transform(
        plan,
        transformed_links,
        transformed_bilocal,
        fourier_phases,
        path_weights,
        hermiticity_tolerance=tolerance,
    )
    anchor = vertices[plan.anchor_vertex]
    left = plan.representation.matrix(anchor)
    right = plan.representation.matrix(plan.link_space.group.inverse(anchor))
    expected = contract("ij,qjk,kl->qil", left, original.wigner_matrix, right)
    residual = jnp.max(jnp.abs(transformed.wigner_matrix - expected))
    finite = original.finite & transformed.finite & jnp.isfinite(residual)
    return GaugeCovarianceEvidence(
        transformed,
        expected,
        residual,
        finite,
        finite & (residual <= tolerance),
        plan.plan_id,
        plan.topology_id,
        plan.representation_id,
    )


__all__ = [
    "GaugeCovarianceEvidence",
    "GaugeCovariantWignerPlan",
    "GaugeCovariantWignerProfile",
    "GaugeCovariantWignerState",
    "gauge_covariance_evidence",
    "gauge_covariant_wigner_transform",
    "gauge_transform_bilocal",
]
