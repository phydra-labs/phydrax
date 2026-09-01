#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._tree_math import tree_add_scaled
from ...continuation._bifurcation import (
    BifurcationCertificate,
    BifurcationStatus,
    correct_branch_seed,
    CorrectedBranchSeed,
)
from ...continuation._core import ContinuationBranch, ContinuationCurveProblem
from ...nonlinear import AbstractNonlinearMethod, NonlinearTermination
from ._equilibrium import MechanicsEquilibriumProblem
from ._stability import DynamicStabilityProblem, PhysicalStaticStabilityProblem


MechanicsBifurcationKind: TypeAlias = Literal[
    "limit-point",
    "branch-point",
    "pitchfork",
    "transcritical",
    "static-buckling",
    "loss-of-positive-definiteness",
    "hopf-flutter",
]
MechanicsClassificationStatus: TypeAlias = Literal[
    "certified",
    "candidate",
    "inconclusive",
]
MechanicsSelectionMode: TypeAlias = Literal[
    "stable-connected",
    "global-energy-minimum",
    "rate-independent-energetic",
    "dynamic-attractor",
    "user-declared",
]
MechanicsSelectionStatus: TypeAlias = Literal[
    "selected",
    "tie",
    "unavailable",
    "indeterminate",
]
MechanicsBranchRelation: TypeAlias = Literal[
    "primary",
    "branch-switch",
    "imperfection",
    "continued",
]


def _identifier(value: str | None, name: str, /) -> str:
    if value is None:
        raise ValueError(f"{name} must be supplied.")
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _finite_scalar(value: Any, name: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != () or not jnp.issubdtype(scalar.dtype, jnp.inexact):
        raise TypeError(f"{name} must be one real floating scalar.")
    if jnp.issubdtype(scalar.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    return scalar


def _classification_status(
    certificate: BifurcationCertificate,
    /,
) -> MechanicsClassificationStatus:
    if bool(certificate.certified):
        return "certified"
    status = int(certificate.status)
    if status in (
        int(BifurcationStatus.CANDIDATE_ONLY),
        int(BifurcationStatus.EXTENDED_SYSTEM_NOT_CONVERGED),
    ):
        return "candidate"
    return "inconclusive"


class MechanicsBifurcationRecord(StrictModule):
    """A local theorem interpreted only through physical mechanics evidence."""

    certificate: BifurcationCertificate
    static_stability: PhysicalStaticStabilityProblem | None
    dynamic_stability: DynamicStabilityProblem | None
    physical_mode: PyTree[Array] | None
    physical_mode_norm: Array
    root_mode: PyTree[Array] | None
    classification: MechanicsBifurcationKind = eqx.field(static=True)
    classification_status: MechanicsClassificationStatus = eqx.field(static=True)
    equilibrium_problem_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    equilibrium_provenance_id: str = eqx.field(static=True)
    physical_space_id: str = eqx.field(static=True)
    stability_problem_id: str = eqx.field(static=True)
    mode_provenance_id: str = eqx.field(static=True)
    eigenvalue_quantity: str = eqx.field(static=True)
    indicator_bracket_id: str | None = eqx.field(static=True)
    localization_id: str | None = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    @property
    def certified(self) -> bool:
        return self.classification_status == "certified"


class MechanicsBifurcationDetector(StrictModule):
    """Classify generic local certificates with physical static/dynamic contracts."""

    equilibrium: MechanicsEquilibriumProblem
    detector_id: str = eqx.field(static=True)

    def __init__(
        self,
        equilibrium: MechanicsEquilibriumProblem,
        /,
        *,
        detector_id: str | None = None,
    ):
        if not isinstance(equilibrium, MechanicsEquilibriumProblem):
            raise TypeError("equilibrium must be a MechanicsEquilibriumProblem.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "mechanics-bifurcation-detector",
                    "equilibrium": equilibrium.problem_id,
                    "realization": equilibrium.realization_id,
                    "provenance": equilibrium.provenance_id,
                }
            )
            if detector_id is None
            else _identifier(detector_id, "detector_id")
        )
        self.equilibrium = equilibrium
        self.detector_id = identifier

    def detect(
        self,
        certificate: BifurcationCertificate,
        /,
        *,
        static_stability: PhysicalStaticStabilityProblem | None = None,
        dynamic_stability: DynamicStabilityProblem | None = None,
        physical_mode: PyTree[Any] | None = None,
        mode_provenance_id: str | None = None,
        static_interpretation: (
            Literal["static-buckling", "loss-of-positive-definiteness"] | None
        ) = None,
        conservative_verified: bool = False,
        proportional_load_verified: bool = False,
        indicator_bracket_id: str | None = None,
        localization_id: str | None = None,
    ) -> MechanicsBifurcationRecord:
        """Interpret one theorem without promoting parameter Hessians or static Hopf."""
        if not isinstance(certificate, BifurcationCertificate):
            raise TypeError("certificate must be a BifurcationCertificate.")
        kind_map: dict[str, MechanicsBifurcationKind] = {
            "fold": "limit-point",
            "branch-point": "branch-point",
            "pitchfork": "pitchfork",
            "transcritical": "transcritical",
            "hopf": "hopf-flutter",
        }
        if static_interpretation is None:
            classification = kind_map[certificate.kind]
        else:
            if certificate.kind != "branch-point":
                raise ValueError(
                    "Static buckling interpretations require a branch-point certificate."
                )
            if not conservative_verified:
                raise ValueError(
                    "Static buckling interpretations require verified conservative "
                    "mechanics."
                )
            if (
                static_interpretation == "static-buckling"
                and not proportional_load_verified
            ):
                raise ValueError(
                    "Static-buckling load factors require verified proportional loading."
                )
            classification = static_interpretation
        if certificate.kind == "hopf":
            if static_interpretation is not None:
                raise ValueError(
                    "A Hopf certificate cannot have a static interpretation."
                )
            if static_stability is not None:
                raise ValueError(
                    "A static stability contract cannot support a Hopf claim."
                )
            if not isinstance(dynamic_stability, DynamicStabilityProblem):
                raise ValueError(
                    "Hopf classification requires a DynamicStabilityProblem."
                )
            stability = dynamic_stability
            space = stability.physical_space
            stability_provenance = (
                f"{stability.stiffness_provenance_id}/{stability.mass_provenance_id}"
            )
        else:
            if dynamic_stability is not None:
                raise ValueError(
                    "A dynamic stability contract cannot replace static bifurcation "
                    "evidence."
                )
            if not isinstance(static_stability, PhysicalStaticStabilityProblem):
                raise ValueError(
                    "Static bifurcation classification requires a "
                    "PhysicalStaticStabilityProblem."
                )
            stability = static_stability
            space = stability.physical_space
            stability_provenance = stability.tangent_provenance_id
        if stability.equilibrium_problem_id != self.equilibrium.problem_id:
            raise ValueError("Stability and detector equilibrium problems do not match.")
        if stability.realization_id != self.equilibrium.realization_id:
            raise ValueError("Stability and detector realizations do not match.")
        if stability.equilibrium_provenance_id != self.equilibrium.provenance_id:
            raise ValueError(
                "Stability and detector equilibrium provenance do not match."
            )
        resolved_mode = physical_mode
        if resolved_mode is None and certificate.kind != "hopf":
            if (
                self.equilibrium.root_coordinates == "physical-state"
                and certificate.right_nullvector is not None
                and certificate.geometry is not None
                and space.compatible(certificate.geometry.public_state_space)
            ):
                resolved_mode = certificate.right_nullvector
            else:
                raise ValueError(
                    "Static classification requires an explicitly lifted physical mode."
                )
        physical_mode_norm = jnp.asarray(jnp.nan)
        if resolved_mode is not None:
            resolved_mode = space.validate(resolved_mode)
            physical_mode_norm = jnp.sqrt(
                jnp.maximum(jnp.real(space.inner(resolved_mode, resolved_mode)), 0.0)
            )
            if not bool(jnp.isfinite(physical_mode_norm) & (physical_mode_norm > 0.0)):
                raise ValueError("The physical mode must have a finite, positive norm.")
            resolved_mode = jax.tree.map(
                lambda value: value / physical_mode_norm,
                resolved_mode,
            )
        mode_provenance = (
            stability_provenance
            if mode_provenance_id is None
            else _identifier(mode_provenance_id, "mode_provenance_id")
        )
        bracket_id = (
            None
            if indicator_bracket_id is None
            else _identifier(indicator_bracket_id, "indicator_bracket_id")
        )
        localization = (
            None
            if localization_id is None
            else _identifier(localization_id, "localization_id")
        )
        status = _classification_status(certificate)
        record_id = canonical_fingerprint(
            {
                "kind": "mechanics-bifurcation-record",
                "detector": self.detector_id,
                "certificate": certificate.certificate_id,
                "classification": classification,
                "stability": stability.problem_id,
                "mode_provenance": mode_provenance,
                "indicator_bracket": bracket_id,
                "localization": localization,
            }
        )
        return MechanicsBifurcationRecord(
            certificate=certificate,
            static_stability=static_stability,
            dynamic_stability=dynamic_stability,
            physical_mode=resolved_mode,
            root_mode=certificate.right_nullvector,
            classification=classification,
            classification_status=status,
            physical_mode_norm=physical_mode_norm,
            equilibrium_problem_id=self.equilibrium.problem_id,
            realization_id=self.equilibrium.realization_id,
            equilibrium_provenance_id=self.equilibrium.provenance_id,
            physical_space_id=space.space_id,
            stability_problem_id=stability.problem_id,
            mode_provenance_id=mode_provenance,
            eigenvalue_quantity=stability.eigenvalue_quantity,
            indicator_bracket_id=bracket_id,
            localization_id=localization,
            record_id=record_id,
        )


class MechanicsBranch(StrictModule):
    """One traced or corrected mechanics branch with complete parent provenance."""

    continuation: ContinuationBranch | None
    seed: CorrectedBranchSeed | None
    branch_id: str = eqx.field(static=True)
    parent_branch_id: str | None = eqx.field(static=True)
    parent_point_id: str | None = eqx.field(static=True)
    certificate_id: str | None = eqx.field(static=True)
    control_protocol: str = eqx.field(static=True)
    symmetry_orbit_id: str | None = eqx.field(static=True)
    imperfection_path_id: str | None = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        continuation: ContinuationBranch | None = None,
        seed: CorrectedBranchSeed | None = None,
        branch_id: str,
        parent_branch_id: str | None = None,
        parent_point_id: str | None = None,
        certificate_id: str | None = None,
        control_protocol: str,
        symmetry_orbit_id: str | None = None,
        imperfection_path_id: str | None = None,
        realization_id: str,
        provenance_id: str,
    ):
        if continuation is not None and not isinstance(continuation, ContinuationBranch):
            raise TypeError("continuation must be a ContinuationBranch or None.")
        if seed is not None and not isinstance(seed, CorrectedBranchSeed):
            raise TypeError("seed must be a CorrectedBranchSeed or None.")
        if continuation is None and seed is None:
            raise ValueError(
                "A mechanics branch requires a continuation or corrected seed."
            )
        branch = _identifier(branch_id, "branch_id")
        if continuation is not None and continuation.branch_id != branch:
            raise ValueError("Continuation and mechanics branch IDs must match.")
        if seed is not None and seed.seed.branch_id != branch:
            raise ValueError("Corrected seed and mechanics branch IDs must match.")
        parent_values = (parent_branch_id, parent_point_id, certificate_id)
        if any(value is None for value in parent_values) and any(
            value is not None for value in parent_values
        ):
            raise ValueError(
                "Parent branch, point, and certificate IDs must be supplied together."
            )
        self.continuation = continuation
        self.seed = seed
        self.branch_id = branch
        self.parent_branch_id = (
            None
            if parent_branch_id is None
            else _identifier(parent_branch_id, "parent_branch_id")
        )
        self.parent_point_id = (
            None
            if parent_point_id is None
            else _identifier(parent_point_id, "parent_point_id")
        )
        self.certificate_id = (
            None
            if certificate_id is None
            else _identifier(certificate_id, "certificate_id")
        )
        self.control_protocol = _identifier(control_protocol, "control_protocol")
        self.symmetry_orbit_id = (
            None
            if symmetry_orbit_id is None
            else _identifier(symmetry_orbit_id, "symmetry_orbit_id")
        )
        self.imperfection_path_id = (
            None
            if imperfection_path_id is None
            else _identifier(imperfection_path_id, "imperfection_path_id")
        )
        self.realization_id = _identifier(realization_id, "realization_id")
        self.provenance_id = _identifier(provenance_id, "provenance_id")


class MechanicsBranchEdge(StrictModule):
    """Directed lineage from one mechanics branch to one derived branch."""

    parent_branch_id: str = eqx.field(static=True)
    child_branch_id: str = eqx.field(static=True)
    source_point_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    construction_id: str = eqx.field(static=True)
    relation: MechanicsBranchRelation = eqx.field(static=True)
    symmetry_related: bool = eqx.field(static=True)
    edge_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        parent_branch_id: str,
        child_branch_id: str,
        source_point_id: str,
        certificate_id: str,
        construction_id: str,
        relation: MechanicsBranchRelation = "branch-switch",
        symmetry_related: bool = False,
        edge_id: str | None = None,
    ):
        if relation not in ("primary", "branch-switch", "imperfection", "continued"):
            raise ValueError("Unsupported mechanics branch relation.")
        parent = _identifier(parent_branch_id, "parent_branch_id")
        child = _identifier(child_branch_id, "child_branch_id")
        if parent == child:
            raise ValueError("A branch lineage edge cannot be a self-edge.")
        source = _identifier(source_point_id, "source_point_id")
        certificate = _identifier(certificate_id, "certificate_id")
        construction = _identifier(construction_id, "construction_id")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "mechanics-branch-edge",
                    "parent": parent,
                    "child": child,
                    "source": source,
                    "certificate": certificate,
                    "construction": construction,
                    "relation": relation,
                }
            )
            if edge_id is None
            else _identifier(edge_id, "edge_id")
        )
        self.parent_branch_id = parent
        self.child_branch_id = child
        self.source_point_id = source
        self.certificate_id = certificate
        self.construction_id = construction
        self.relation = relation
        self.symmetry_related = bool(symmetry_related)
        self.edge_id = identifier


class MechanicsBranchGraph(StrictModule):
    """Immutable acyclic graph of deterministic mechanics branch lineage."""

    branches: tuple[MechanicsBranch, ...]
    edges: tuple[MechanicsBranchEdge, ...]
    graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        branches: Sequence[MechanicsBranch],
        edges: Sequence[MechanicsBranchEdge] = (),
        /,
    ):
        branches_ = tuple(branches)
        edges_ = tuple(edges)
        if not branches_ or any(
            not isinstance(branch, MechanicsBranch) for branch in branches_
        ):
            raise ValueError("A mechanics branch graph requires MechanicsBranch values.")
        if any(not isinstance(edge, MechanicsBranchEdge) for edge in edges_):
            raise TypeError("edges must contain MechanicsBranchEdge values.")
        branch_ids = tuple(branch.branch_id for branch in branches_)
        if len(set(branch_ids)) != len(branch_ids):
            raise ValueError("Mechanics branch IDs must be unique.")
        edge_ids = tuple(edge.edge_id for edge in edges_)
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("Mechanics branch edge IDs must be unique.")
        known = set(branch_ids)
        if any(
            edge.parent_branch_id not in known or edge.child_branch_id not in known
            for edge in edges_
        ):
            raise ValueError("Every branch edge endpoint must belong to the graph.")
        parents = {edge.child_branch_id: edge.parent_branch_id for edge in edges_}
        if len(parents) != len(edges_):
            raise ValueError("A mechanics branch may have only one lineage parent.")
        edge_by_child = {edge.child_branch_id: edge for edge in edges_}
        for branch in branches_:
            incoming = edge_by_child.get(branch.branch_id)
            if branch.parent_branch_id is None:
                if incoming is not None:
                    raise ValueError(
                        "A root mechanics branch cannot have an incoming edge."
                    )
                continue
            if incoming is None:
                raise ValueError(
                    "Every derived mechanics branch requires a lineage edge."
                )
            if (
                incoming.parent_branch_id != branch.parent_branch_id
                or incoming.source_point_id != branch.parent_point_id
                or incoming.certificate_id != branch.certificate_id
            ):
                raise ValueError(
                    "Mechanics branch metadata and lineage edge do not match."
                )
        for branch_id in branch_ids:
            visited: set[str] = set()
            current = branch_id
            while current in parents:
                if current in visited:
                    raise ValueError("Mechanics branch lineage must be acyclic.")
                visited.add(current)
                current = parents[current]
        self.branches = branches_
        self.edges = edges_
        self.graph_id = canonical_fingerprint(
            {
                "kind": "mechanics-branch-graph",
                "branches": list(branch_ids),
                "edges": list(edge_ids),
            }
        )

    def branch(self, branch_id: str, /) -> MechanicsBranch:
        identifier = _identifier(branch_id, "branch_id")
        for branch in self.branches:
            if branch.branch_id == identifier:
                return branch
        raise KeyError(identifier)

    def add(
        self,
        branch: MechanicsBranch,
        edge: MechanicsBranchEdge,
        /,
    ) -> MechanicsBranchGraph:
        if not isinstance(branch, MechanicsBranch):
            raise TypeError("branch must be a MechanicsBranch.")
        if not isinstance(edge, MechanicsBranchEdge):
            raise TypeError("edge must be a MechanicsBranchEdge.")
        if edge.child_branch_id != branch.branch_id:
            raise ValueError("The edge child must be the added mechanics branch.")
        return MechanicsBranchGraph(self.branches + (branch,), self.edges + (edge,))


class BranchSwitchPolicy(StrictModule):
    """Augmented correction and duplicate/orbit tolerances for branch switching."""

    termination: NonlinearTermination | None
    amplitude: float = eqx.field(static=True)
    coordinate_offset: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    duplicate_state_tolerance: float = eqx.field(static=True)
    duplicate_coordinate_tolerance: float = eqx.field(static=True)
    symmetry_tolerance: float = eqx.field(static=True)
    quotient_symmetry: bool = eqx.field(static=True)
    control_protocol: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        amplitude: float,
        coordinate_offset: float,
        termination: NonlinearTermination | None = None,
        residual_tolerance: float = 1e-7,
        duplicate_state_tolerance: float = 1e-6,
        duplicate_coordinate_tolerance: float = 1e-7,
        symmetry_tolerance: float = 1e-6,
        quotient_symmetry: bool = False,
        control_protocol: str,
        policy_id: str | None = None,
    ):
        amplitude_ = float(amplitude)
        offset = float(coordinate_offset)
        tolerances = tuple(
            float(value)
            for value in (
                residual_tolerance,
                duplicate_state_tolerance,
                duplicate_coordinate_tolerance,
                symmetry_tolerance,
            )
        )
        if not isfinite(amplitude_) or amplitude_ <= 0.0:
            raise ValueError("amplitude must be finite and positive.")
        if not isfinite(offset) or offset == 0.0:
            raise ValueError("coordinate_offset must be finite and nonzero.")
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Branch-switch tolerances must be finite and non-negative.")
        if termination is not None and not isinstance(termination, NonlinearTermination):
            raise TypeError("termination must be a NonlinearTermination or None.")
        control = _identifier(control_protocol, "control_protocol")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "mechanics-branch-switch-policy",
                    "amplitude": amplitude_.hex(),
                    "coordinate_offset": offset.hex(),
                    "residual_tolerance": tolerances[0].hex(),
                    "duplicate_state_tolerance": tolerances[1].hex(),
                    "duplicate_coordinate_tolerance": tolerances[2].hex(),
                    "symmetry_tolerance": tolerances[3].hex(),
                    "quotient_symmetry": bool(quotient_symmetry),
                    "control_protocol": control,
                }
            )
            if policy_id is None
            else _identifier(policy_id, "policy_id")
        )
        self.termination = termination
        self.amplitude = amplitude_
        self.coordinate_offset = offset
        (
            self.residual_tolerance,
            self.duplicate_state_tolerance,
            self.duplicate_coordinate_tolerance,
            self.symmetry_tolerance,
        ) = tolerances
        self.quotient_symmetry = bool(quotient_symmetry)
        self.control_protocol = control
        self.policy_id = identifier


class MechanicsBranchSwitchResult(StrictModule):
    """Corrected proposals, accepted graph children, and explicit rejection IDs."""

    corrections: tuple[CorrectedBranchSeed, ...]
    graph: MechanicsBranchGraph
    accepted_branch_ids: tuple[str, ...] = eqx.field(static=True)
    failed_branch_ids: tuple[str, ...] = eqx.field(static=True)
    duplicate_branch_ids: tuple[str, ...] = eqx.field(static=True)
    symmetry_rejected_branch_ids: tuple[str, ...] = eqx.field(static=True)
    physical_mode_lift_id: str = eqx.field(static=True)


def _branch_anchor(branch: MechanicsBranch, /) -> tuple[PyTree[Array], Array]:
    if branch.seed is not None:
        return branch.seed.seed.state, branch.seed.seed.coordinate
    if branch.continuation is None:
        raise ValueError("A mechanics branch has no anchor state.")
    point = branch.continuation.points[0]
    return point.state, point.coordinate


def _matches_branch(
    graph: MechanicsBranchGraph,
    state: PyTree[Any],
    coordinate: Any,
    record: MechanicsBifurcationRecord,
    /,
    *,
    state_tolerance: float,
    coordinate_tolerance: float,
) -> MechanicsBranch | None:
    geometry = record.certificate.geometry
    if geometry is None:
        raise ValueError("Duplicate checking requires certificate geometry.")
    coordinates = geometry.state_to_execution(state)
    for branch in graph.branches:
        branch_state, branch_coordinate = _branch_anchor(branch)
        branch_coordinates = geometry.state_to_execution(branch_state)
        difference = tree_add_scaled(coordinates, branch_coordinates, -1.0)
        if bool(
            (geometry.state_norm(difference) <= state_tolerance)
            & (
                jnp.abs(jnp.asarray(coordinate) - branch_coordinate)
                <= coordinate_tolerance
            )
        ):
            return branch
    return None


def switch_mechanics_branch(
    record: MechanicsBifurcationRecord,
    problem: ContinuationCurveProblem,
    root_method: AbstractNonlinearMethod,
    graph: MechanicsBranchGraph,
    physical_mode_lift: Callable[[PyTree[Any], PyTree[Any], Any], PyTree[Any]],
    /,
    *,
    physical_mode_lift_id: str,
    source_branch_id: str,
    source_point_id: str,
    policy: BranchSwitchPolicy,
    symmetry: Callable[[PyTree[Any]], PyTree[Any]] | None = None,
    args: Any = None,
) -> MechanicsBranchSwitchResult:
    """Lift, correct, deduplicate, and register both local switched branches."""
    if not isinstance(record, MechanicsBifurcationRecord):
        raise TypeError("record must be a MechanicsBifurcationRecord.")
    if not record.certified or record.classification not in (
        "pitchfork",
        "transcritical",
    ):
        raise ValueError(
            "Mechanics branch switching requires certified pitchfork or "
            "transcritical evidence."
        )
    if record.physical_mode is None:
        raise ValueError("Mechanics branch switching requires a physical mode.")
    if not isinstance(problem, ContinuationCurveProblem):
        raise TypeError("problem must be a ContinuationCurveProblem.")
    if not isinstance(root_method, AbstractNonlinearMethod):
        raise TypeError("root_method must be an AbstractNonlinearMethod.")
    if not isinstance(graph, MechanicsBranchGraph):
        raise TypeError("graph must be a MechanicsBranchGraph.")
    if not isinstance(policy, BranchSwitchPolicy):
        raise TypeError("policy must be a BranchSwitchPolicy.")
    if not callable(physical_mode_lift):
        raise TypeError("physical_mode_lift must be callable.")
    if symmetry is not None and not callable(symmetry):
        raise TypeError("symmetry must be callable or None.")
    lift_id = _identifier(physical_mode_lift_id, "physical_mode_lift_id")
    source_branch = graph.branch(source_branch_id)
    source_point = _identifier(source_point_id, "source_point_id")
    if source_branch.continuation is not None and source_point not in {
        point.point_id for point in source_branch.continuation.points
    }:
        raise ValueError("source_point_id does not belong to source_branch_id.")
    lifted_mode = physical_mode_lift(
        record.physical_mode,
        record.certificate.state,
        args,
    )
    geometry = record.certificate.geometry
    if geometry is None:
        raise ValueError("Mechanics branch switching requires certificate geometry.")
    geometry.public_state_space.validate(lifted_mode)
    corrections: list[CorrectedBranchSeed] = []
    accepted: list[str] = []
    failed: list[str] = []
    duplicates: list[str] = []
    symmetry_rejected: list[str] = []
    updated_graph = graph
    for sign in (1, -1):
        branch_id = canonical_fingerprint(
            {
                "kind": "mechanics-switched-branch",
                "source_branch": source_branch.branch_id,
                "source_point": source_point,
                "certificate": record.certificate.certificate_id,
                "policy": policy.policy_id,
                "physical_mode": record.mode_provenance_id,
                "physical_mode_lift": lift_id,
                "sign": sign,
            }
        )
        correction = correct_branch_seed(
            problem,
            record.certificate,
            root_method,
            lifted_mode,
            signed_amplitude=sign * policy.amplitude,
            coordinate_offset=policy.coordinate_offset,
            branch_id=branch_id,
            source_point_id=source_point,
            termination=policy.termination,
            residual_tolerance=policy.residual_tolerance,
            args=args,
        )
        corrections.append(correction)
        if not bool(correction.successful):
            failed.append(branch_id)
            continue
        duplicate = _matches_branch(
            updated_graph,
            correction.seed.state,
            correction.seed.coordinate,
            record,
            state_tolerance=policy.duplicate_state_tolerance,
            coordinate_tolerance=policy.duplicate_coordinate_tolerance,
        )
        if duplicate is not None:
            duplicates.append(branch_id)
            continue
        symmetry_match = None
        if symmetry is not None:
            symmetric_state = geometry.public_state_space.validate(
                symmetry(correction.seed.state)
            )
            symmetry_match = _matches_branch(
                updated_graph,
                symmetric_state,
                correction.seed.coordinate,
                record,
                state_tolerance=policy.symmetry_tolerance,
                coordinate_tolerance=policy.duplicate_coordinate_tolerance,
            )
        if symmetry_match is not None and policy.quotient_symmetry:
            symmetry_rejected.append(branch_id)
            continue
        symmetry_orbit_id = (
            None
            if symmetry is None
            else canonical_fingerprint(
                {
                    "kind": "mechanics-symmetry-orbit",
                    "certificate": record.certificate.certificate_id,
                    "source_branch": source_branch.branch_id,
                    "source_point": source_point,
                }
            )
        )
        child = MechanicsBranch(
            seed=correction,
            branch_id=branch_id,
            parent_branch_id=source_branch.branch_id,
            parent_point_id=source_point,
            certificate_id=record.certificate.certificate_id,
            control_protocol=policy.control_protocol,
            symmetry_orbit_id=symmetry_orbit_id,
            imperfection_path_id=source_branch.imperfection_path_id,
            realization_id=record.realization_id,
            provenance_id=record.equilibrium_provenance_id,
        )
        edge = MechanicsBranchEdge(
            parent_branch_id=source_branch.branch_id,
            child_branch_id=branch_id,
            source_point_id=source_point,
            certificate_id=record.certificate.certificate_id,
            construction_id=correction.correction_id,
            symmetry_related=symmetry_match is not None,
        )
        updated_graph = updated_graph.add(child, edge)
        accepted.append(branch_id)
    return MechanicsBranchSwitchResult(
        corrections=tuple(corrections),
        graph=updated_graph,
        accepted_branch_ids=tuple(accepted),
        failed_branch_ids=tuple(failed),
        duplicate_branch_ids=tuple(duplicates),
        symmetry_rejected_branch_ids=tuple(symmetry_rejected),
        physical_mode_lift_id=lift_id,
    )


class ImperfectionFamily(StrictModule):
    """Named dimensional one-parameter physical imperfection field."""

    shape: PyTree[Array]
    units: str = eqx.field(static=True)
    orientation: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    fabrication_provenance_id: str = eqx.field(static=True)
    family_id: str = eqx.field(static=True)

    def __init__(
        self,
        shape: PyTree[Any],
        /,
        *,
        units: str,
        orientation: str,
        discretization_id: str,
        fabrication_provenance_id: str,
        family_id: str | None = None,
    ):
        shape_ = jax.tree.map(jnp.asarray, shape)
        leaves = jax.tree.leaves(shape_)
        if not leaves or any(
            not jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in leaves
        ):
            raise TypeError("imperfection shape must be a nonempty real floating PyTree.")
        if not bool(jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(x)) for x in leaves)))):
            raise ValueError("imperfection shape must be finite.")
        if not bool(jnp.any(jnp.stack(tuple(jnp.any(x != 0) for x in leaves)))):
            raise ValueError("imperfection shape must contain a nonzero direction.")
        units_ = _identifier(units, "units")
        orientation_ = _identifier(orientation, "orientation")
        discretization = _identifier(discretization_id, "discretization_id")
        provenance = _identifier(
            fabrication_provenance_id,
            "fabrication_provenance_id",
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "mechanics-imperfection-family",
                    "shape": array_tree_fingerprint(shape_),
                    "units": units_,
                    "orientation": orientation_,
                    "discretization": discretization,
                    "fabrication_provenance": provenance,
                }
            )
            if family_id is None
            else _identifier(family_id, "family_id")
        )
        self.shape = shape_
        self.units = units_
        self.orientation = orientation_
        self.discretization_id = discretization
        self.fabrication_provenance_id = provenance
        self.family_id = identifier

    def realize(self, amplitude: Any, /) -> PyTree[Array]:
        amplitude_ = _finite_scalar(amplitude, "imperfection amplitude")
        return jax.tree.map(lambda value: amplitude_ * value, self.shape)


class ImperfectionStudy(StrictModule):
    """Ordered imperfection amplitudes and their resolved mechanics records."""

    family: ImperfectionFamily
    amplitudes: Array
    records: tuple[MechanicsBifurcationRecord | None, ...]
    limit_resolved: Array
    study_id: str = eqx.field(static=True)

    def __init__(
        self,
        family: ImperfectionFamily,
        amplitudes: Any,
        records: Sequence[MechanicsBifurcationRecord | None],
        /,
        *,
        limit_resolved: Any,
        study_id: str | None = None,
    ):
        if not isinstance(family, ImperfectionFamily):
            raise TypeError("family must be an ImperfectionFamily.")
        amplitudes_ = jnp.asarray(amplitudes)
        if (
            amplitudes_.ndim != 1
            or not amplitudes_.size
            or not jnp.issubdtype(amplitudes_.dtype, jnp.floating)
        ):
            raise TypeError("imperfection amplitudes must be a nonempty real vector.")
        if not bool(jnp.all(jnp.isfinite(amplitudes_))):
            raise ValueError("imperfection amplitudes must be finite.")
        if not bool(jnp.any(amplitudes_ == 0.0)):
            raise ValueError(
                "An imperfection study must include the ideal zero baseline."
            )
        records_ = tuple(records)
        if len(records_) != amplitudes_.size or any(
            record is not None and not isinstance(record, MechanicsBifurcationRecord)
            for record in records_
        ):
            raise ValueError("records must align one-for-one with amplitudes.")
        resolved = jnp.asarray(limit_resolved, dtype=bool)
        if resolved.shape != ():
            raise ValueError("limit_resolved must be one scalar boolean.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "mechanics-imperfection-study",
                    "family": family.family_id,
                    "amplitudes": array_tree_fingerprint(amplitudes_),
                    "records": [
                        None if record is None else record.record_id
                        for record in records_
                    ],
                }
            )
            if study_id is None
            else _identifier(study_id, "study_id")
        )
        self.family = family
        self.amplitudes = amplitudes_
        self.records = records_
        self.limit_resolved = resolved
        self.study_id = identifier


class EnergyBarrierEvidence(StrictModule):
    """Conservative potential barrier certified by an admissible index-one saddle."""

    source_energy: Array
    target_energy: Array
    saddle_energy: Array
    stationary_residual: Array
    path_defect: Array
    refinement_defect: Array
    admissible: Array
    source_branch_id: str = eqx.field(static=True)
    target_branch_id: str = eqx.field(static=True)
    potential_provenance_id: str = eqx.field(static=True)
    source_morse_index: int = eqx.field(static=True)
    target_morse_index: int = eqx.field(static=True)
    saddle_morse_index: int = eqx.field(static=True)
    potential_verified: bool = eqx.field(static=True)
    conservative_verified: bool = eqx.field(static=True)
    stationary_tolerance: float = eqx.field(static=True)
    path_tolerance: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_branch_id: str,
        target_branch_id: str,
        source_energy: Any,
        target_energy: Any,
        saddle_energy: Any,
        source_morse_index: int,
        target_morse_index: int,
        saddle_morse_index: int,
        stationary_residual: Any,
        path_defect: Any,
        refinement_defect: Any,
        admissible: Any,
        potential_verified: bool,
        conservative_verified: bool,
        potential_provenance_id: str,
        stationary_tolerance: float = 1e-7,
        path_tolerance: float = 1e-5,
        evidence_id: str | None = None,
    ):
        if not potential_verified or not conservative_verified:
            raise ValueError(
                "Energy-barrier evidence requires a verified conservative potential."
            )
        if (
            int(source_morse_index) != 0
            or int(target_morse_index) != 0
            or int(saddle_morse_index) != 1
        ):
            raise ValueError(
                "Barrier evidence requires two minima and one index-one saddle."
            )
        source_branch = _identifier(source_branch_id, "source_branch_id")
        target_branch = _identifier(target_branch_id, "target_branch_id")
        if source_branch == target_branch:
            raise ValueError("Barrier endpoints must be distinct branches.")
        values = tuple(
            _finite_scalar(value, name)
            for value, name in (
                (source_energy, "source_energy"),
                (target_energy, "target_energy"),
                (saddle_energy, "saddle_energy"),
                (stationary_residual, "stationary_residual"),
                (path_defect, "path_defect"),
                (refinement_defect, "refinement_defect"),
            )
        )
        if not bool(jnp.all(jnp.isfinite(jnp.stack(values)))):
            raise ValueError("Energy-barrier values must be finite.")
        if bool(values[2] < jnp.maximum(values[0], values[1])):
            raise ValueError("The saddle energy must not lie below either minimum.")
        stationary_limit = float(stationary_tolerance)
        path_limit = float(path_tolerance)
        if (
            not isfinite(stationary_limit)
            or stationary_limit < 0.0
            or not isfinite(path_limit)
            or path_limit < 0.0
        ):
            raise ValueError("Barrier tolerances must be finite and non-negative.")
        admissible_ = jnp.asarray(admissible, dtype=bool)
        if admissible_.shape != ():
            raise ValueError("admissible must be one scalar boolean.")
        provenance = _identifier(
            potential_provenance_id,
            "potential_provenance_id",
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "mechanics-energy-barrier",
                    "source": source_branch,
                    "target": target_branch,
                    "potential": provenance,
                    "source_morse_index": int(source_morse_index),
                    "target_morse_index": int(target_morse_index),
                    "saddle_morse_index": int(saddle_morse_index),
                    "stationary_tolerance": stationary_limit.hex(),
                    "path_tolerance": path_limit.hex(),
                }
            )
            if evidence_id is None
            else _identifier(evidence_id, "evidence_id")
        )
        (
            self.source_energy,
            self.target_energy,
            self.saddle_energy,
            self.stationary_residual,
            self.path_defect,
            self.refinement_defect,
        ) = values
        self.admissible = admissible_
        self.source_branch_id = source_branch
        self.target_branch_id = target_branch
        self.source_morse_index = int(source_morse_index)
        self.target_morse_index = int(target_morse_index)
        self.saddle_morse_index = int(saddle_morse_index)
        self.potential_verified = True
        self.conservative_verified = True
        self.potential_provenance_id = provenance
        self.stationary_tolerance = stationary_limit
        self.path_tolerance = path_limit
        self.evidence_id = identifier

    @property
    def certified(self) -> Array:
        return (
            self.admissible
            & (self.stationary_residual <= self.stationary_tolerance)
            & (self.path_defect <= self.path_tolerance)
            & (self.refinement_defect <= self.path_tolerance)
        )


class PhysicalSelectionPolicy(StrictModule):
    """Explicit physical branch-selection protocol; no implicit fallback policy."""

    mode: MechanicsSelectionMode = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    user_branch_id: str | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: MechanicsSelectionMode,
        /,
        *,
        energy_tolerance: float = 1e-8,
        user_branch_id: str | None = None,
        policy_id: str | None = None,
    ):
        if mode not in (
            "stable-connected",
            "global-energy-minimum",
            "rate-independent-energetic",
            "dynamic-attractor",
            "user-declared",
        ):
            raise ValueError("Unsupported physical selection mode.")
        tolerance = float(energy_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("energy_tolerance must be finite and non-negative.")
        user = (
            None
            if user_branch_id is None
            else _identifier(user_branch_id, "user_branch_id")
        )
        if mode == "user-declared" and user is None:
            raise ValueError("user-declared selection requires user_branch_id.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "physical-selection-policy",
                    "mode": mode,
                    "energy_tolerance": tolerance.hex(),
                    "user_branch": user,
                }
            )
            if policy_id is None
            else _identifier(policy_id, "policy_id")
        )
        self.mode = mode
        self.energy_tolerance = tolerance
        self.user_branch_id = user
        self.policy_id = identifier


class MechanicsSelectionResult(StrictModule):
    """Selected mechanics branch or an explicit tie/unavailable/indeterminate result."""

    policy: PhysicalSelectionPolicy
    evidence: Any
    selected_branch_id: str | None = eqx.field(static=True)
    candidate_branch_ids: tuple[str, ...] = eqx.field(static=True)
    status: MechanicsSelectionStatus = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def disagrees_with(self, other: MechanicsSelectionResult, /) -> bool:
        if not isinstance(other, MechanicsSelectionResult):
            raise TypeError("other must be a MechanicsSelectionResult.")
        return (
            self.status == "selected"
            and other.status == "selected"
            and self.selected_branch_id != other.selected_branch_id
        )


def _selection_result(
    policy: PhysicalSelectionPolicy,
    candidates: tuple[str, ...],
    status: MechanicsSelectionStatus,
    selected: str | None,
    evidence: Any,
    /,
) -> MechanicsSelectionResult:
    return MechanicsSelectionResult(
        policy=policy,
        evidence=evidence,
        selected_branch_id=selected,
        candidate_branch_ids=candidates,
        status=status,
        result_id=canonical_fingerprint(
            {
                "kind": "mechanics-selection-result",
                "policy": policy.policy_id,
                "candidates": list(candidates),
                "status": status,
                "selected": selected,
            }
        ),
    )


def select_mechanics_branch(
    graph: MechanicsBranchGraph,
    policy: PhysicalSelectionPolicy,
    /,
    *,
    candidate_branch_ids: Sequence[str] | None = None,
    stable_branch_ids: Sequence[str] = (),
    connected_branch_id: str | None = None,
    branch_energies: Mapping[str, Any] | None = None,
    potential_verified: bool = False,
    dynamic_attractor_ids: Sequence[str] = (),
    barriers: Sequence[EnergyBarrierEvidence] = (),
) -> MechanicsSelectionResult:
    """Apply exactly one declared selection protocol with capability refusal."""
    if not isinstance(graph, MechanicsBranchGraph):
        raise TypeError("graph must be a MechanicsBranchGraph.")
    if not isinstance(policy, PhysicalSelectionPolicy):
        raise TypeError("policy must be a PhysicalSelectionPolicy.")
    candidates = (
        tuple(branch.branch_id for branch in graph.branches)
        if candidate_branch_ids is None
        else tuple(
            _identifier(value, "candidate_branch_id") for value in candidate_branch_ids
        )
    )
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Candidate branch IDs must be nonempty and unique.")
    for branch_id in candidates:
        graph.branch(branch_id)
    stable = {_identifier(value, "stable_branch_id") for value in stable_branch_ids}
    attractors = tuple(
        _identifier(value, "dynamic_attractor_id") for value in dynamic_attractor_ids
    )
    for branch_id in stable | set(attractors):
        graph.branch(branch_id)
    attractors = tuple(value for value in attractors if value in candidates)
    connected = (
        None
        if connected_branch_id is None
        else _identifier(connected_branch_id, "connected_branch_id")
    )
    if policy.mode == "stable-connected":
        if connected is None:
            return _selection_result(policy, candidates, "unavailable", None, None)
        graph.branch(connected)
        if connected not in candidates:
            return _selection_result(policy, candidates, "indeterminate", None, connected)
        if connected in stable:
            return _selection_result(
                policy, candidates, "selected", connected, tuple(stable)
            )
        return _selection_result(policy, candidates, "indeterminate", None, tuple(stable))
    if policy.mode == "dynamic-attractor":
        if not attractors:
            return _selection_result(policy, candidates, "unavailable", None, ())
        if len(attractors) != 1:
            return _selection_result(
                policy, candidates, "indeterminate", None, attractors
            )
        return _selection_result(
            policy, candidates, "selected", attractors[0], attractors
        )
    if policy.mode == "user-declared":
        selected = policy.user_branch_id
        if selected not in candidates:
            return _selection_result(policy, candidates, "indeterminate", None, selected)
        return _selection_result(policy, candidates, "selected", selected, selected)
    if not potential_verified:
        return _selection_result(
            policy,
            candidates,
            "unavailable",
            None,
            "verified conservative potential required",
        )
    if branch_energies is None or any(
        branch_id not in branch_energies for branch_id in candidates
    ):
        return _selection_result(policy, candidates, "unavailable", None, None)
    energies = {
        branch_id: _finite_scalar(branch_energies[branch_id], "branch energy")
        for branch_id in candidates
    }
    if any(not bool(jnp.isfinite(value)) for value in energies.values()):
        return _selection_result(policy, candidates, "indeterminate", None, energies)
    minimum = min(float(value) for value in energies.values())
    minimizers = tuple(
        branch_id
        for branch_id, energy in energies.items()
        if abs(float(energy) - minimum) <= policy.energy_tolerance
    )
    if policy.mode == "global-energy-minimum":
        if len(minimizers) != 1:
            return _selection_result(policy, candidates, "tie", None, energies)
        return _selection_result(policy, candidates, "selected", minimizers[0], energies)
    if connected is None or connected not in candidates:
        return _selection_result(policy, candidates, "unavailable", None, energies)
    if len(minimizers) != 1:
        return _selection_result(policy, candidates, "tie", None, energies)
    target = minimizers[0]
    if target == connected:
        return _selection_result(policy, candidates, "selected", connected, energies)
    barriers_ = tuple(barriers)
    if any(not isinstance(barrier, EnergyBarrierEvidence) for barrier in barriers_):
        raise TypeError("barriers must contain EnergyBarrierEvidence values.")
    barrier = next(
        (
            value
            for value in barriers_
            if value.source_branch_id == connected
            and value.target_branch_id == target
            and bool(value.certified)
        ),
        None,
    )
    if barrier is None:
        return _selection_result(policy, candidates, "unavailable", None, barriers_)
    energy_release = float(energies[connected] - energies[target])
    activation = float(barrier.saddle_energy - barrier.source_energy)
    selected = (
        target if energy_release > activation + policy.energy_tolerance else connected
    )
    return _selection_result(
        policy,
        candidates,
        "selected",
        selected,
        {"energies": energies, "barrier": barrier},
    )


__all__ = [
    "BranchSwitchPolicy",
    "EnergyBarrierEvidence",
    "ImperfectionFamily",
    "ImperfectionStudy",
    "MechanicsBifurcationDetector",
    "MechanicsBifurcationKind",
    "MechanicsBifurcationRecord",
    "MechanicsBranch",
    "MechanicsBranchEdge",
    "MechanicsBranchGraph",
    "MechanicsBranchSwitchResult",
    "MechanicsSelectionResult",
    "PhysicalSelectionPolicy",
    "select_mechanics_branch",
    "switch_mechanics_branch",
]
