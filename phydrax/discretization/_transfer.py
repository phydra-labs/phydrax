#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import assert_never, Literal, TypeAlias

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..linalg import AbstractLinearOperator
from ..typing import checked, parse
from ._core import PreparationReport, resolved_identifier
from ._spaces import DiscreteFieldSpace


TransferSemantics: TypeAlias = Literal[
    "unspecified",
    "interpolation",
    "nested-interpolation",
    "l2-projection",
    "conservative-remap",
    "restriction",
    "covariant-piola",
    "contravariant-piola",
    "material-pullback",
]
"""Checked meaning of a transfer's primal map.

``unspecified`` makes no semantic claim. ``interpolation`` applies target DOF
functionals to the source field; ``nested-interpolation`` does so on a target
mesh nested in the source through exact coordinate-map restriction, reproducing
every source field exactly. ``l2-projection`` is a Galerkin projection,
``conservative-remap`` an overlap-measure content remap, and ``restriction`` a
fine-to-coarse state restriction. ``covariant-piola`` and
``contravariant-piola`` are compatible H(curl) and H(div) transfers through the
oriented edge/face moment functionals of the target space.
``material-pullback`` transports a scalar under an explicitly certified material
reference correspondence between different physical coordinate realizations.
"""

TransferGeometryRelation: TypeAlias = Literal[
    "identity",
    "exact-restriction",
    "bounded-reconstruction",
    "topology-correspondence",
    "source-realization",
]
"""Relation between the source and target coordinate maps of a transfer.

``identity``: one geometry. ``exact-restriction``: every target coordinate map
is the exact restriction of one source map (nested affine or closed-family
refinement). ``bounded-reconstruction``: target maps approximate the source
geometry with a reported relative coverage defect.
``topology-correspondence`` binds only the actual scientific endpoints of a
topological content action. It makes no coordinate-map or physical-area claim.
``source-realization`` binds a declared common reference complex with independently
certified physical embeddings. It does not assert common physical coverage.
"""


def _compatible_conformity(
    source: DiscreteFieldSpace, target: DiscreteFieldSpace
) -> bool:
    return source.conformity == target.conformity


def _check_semantics(
    semantics: TransferSemantics,
    source: DiscreteFieldSpace,
    target: DiscreteFieldSpace,
    properties: TransferProperties,
    geometry: TransferGeometryBinding | None,
    /,
) -> None:
    """Refuse a declared transfer meaning the field spaces or evidence contradict."""

    match semantics:
        case "unspecified":
            return
        case "interpolation" | "l2-projection" | "restriction":
            if not _compatible_conformity(source, target):
                raise ValueError(
                    f"A {semantics} transfer needs one field conformity; got "
                    f"{source.conformity} -> {target.conformity}."
                )
            if semantics == "interpolation" and source.conformity in ("Hcurl", "Hdiv"):
                raise ValueError(
                    "Compatible H(curl)/H(div) fields transfer through Piola moment "
                    "functionals, not nodal interpolation."
                )
        case "nested-interpolation":
            if not _compatible_conformity(source, target) or source.conformity in (
                "Hcurl",
                "Hdiv",
            ):
                raise ValueError(
                    "Nested interpolation needs one non-compatible field conformity; "
                    f"got {source.conformity} -> {target.conformity}."
                )
            if not properties.nested:
                raise ValueError("Nested interpolation must claim a nested transfer.")
            if geometry is None or geometry.relation != "exact-restriction":
                raise ValueError(
                    "Nested interpolation requires an exact-restriction geometry binding."
                )
        case "conservative-remap":
            if not _compatible_conformity(source, target):
                raise ValueError("A conservative remap needs one field conformity.")
            if not properties.conservative:
                raise ValueError("A conservative remap must claim conservation.")
        case "material-pullback":
            if not _compatible_conformity(source, target) or source.conformity not in (
                "H1",
                "L2",
            ):
                raise ValueError(
                    "Material scalar pullback requires one scalar H1 or L2 conformity."
                )
            if geometry is None or geometry.relation != "source-realization":
                raise ValueError(
                    "Material pullback requires its actual source-realization correspondence."
                )
        case "covariant-piola" | "contravariant-piola":
            conformity = "Hcurl" if semantics == "covariant-piola" else "Hdiv"
            if source.conformity != conformity or target.conformity != conformity:
                raise ValueError(
                    f"A {semantics} transfer maps {conformity} fields; got "
                    f"{source.conformity} -> {target.conformity}."
                )
            if geometry is None:
                raise ValueError(f"A {semantics} transfer requires a geometry binding.")
        case unknown:
            assert_never(unknown)


class TransferProperties(StrictModule, NonTrainableState):
    """Explicitly claimed structural properties of one field transfer.

    ``semantics`` is checked by :class:`FieldTransfer` against its source and
    target field conformity and geometry binding.
    """

    constant_preserving: bool = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    positivity_preserving: bool = eqx.field(static=True)
    nested: bool = eqx.field(static=True)
    adjoint_paired: bool = eqx.field(static=True)
    differentiable_geometry: bool = eqx.field(static=True)
    exact_on: tuple[str, ...] = eqx.field(static=True)
    semantics: TransferSemantics = eqx.field(static=True)

    def __init__(
        self,
        *,
        constant_preserving: bool = False,
        conservative: bool = False,
        positivity_preserving: bool = False,
        nested: bool = False,
        adjoint_paired: bool = False,
        differentiable_geometry: bool = False,
        exact_on: Sequence[str] = (),
        semantics: TransferSemantics = "unspecified",
    ) -> None:
        exact = tuple(str(value) for value in exact_on)
        if any(not value for value in exact) or len(set(exact)) != len(exact):
            raise ValueError("exact_on entries must be unique non-empty strings.")
        self.constant_preserving = bool(constant_preserving)
        self.conservative = bool(conservative)
        self.positivity_preserving = bool(positivity_preserving)
        self.nested = bool(nested)
        self.adjoint_paired = bool(adjoint_paired)
        self.differentiable_geometry = bool(differentiable_geometry)
        self.exact_on = exact
        self.semantics = parse(semantics, TransferSemantics, "semantics")


class TransferGeometryBinding(StrictModule, NonTrainableState):
    """Actual source/target topology and coordinate-map identities of a transfer.

    ``coverage_defect`` is the largest accepted relative measure defect of the
    target geometry against the source geometry: zero for ``identity`` and
    ``exact-restriction`` bindings up to the reported roundoff certificate,
    positive for a ``bounded-reconstruction``.
    ``topology-correspondence`` instead requires ``None``: pure extensive
    correspondences do not supply a physical geometric coverage certificate.
    """

    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    relation: TransferGeometryRelation = eqx.field(static=True)
    coverage_defect: float | None = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_geometry_id: str,
        target_geometry_id: str,
        relation: TransferGeometryRelation,
        /,
        *,
        source_topology_id: str,
        target_topology_id: str,
        coverage_defect: float | None = 0.0,
    ) -> None:
        source = canonical_identifier(source_geometry_id, "source_geometry_id")
        target = canonical_identifier(target_geometry_id, "target_geometry_id")
        source_topology = canonical_identifier(source_topology_id, "source_topology_id")
        target_topology = canonical_identifier(target_topology_id, "target_topology_id")
        relation_ = parse(relation, TransferGeometryRelation, "relation")
        if relation_ in ("topology-correspondence", "source-realization"):
            if coverage_defect is not None:
                raise ValueError(
                    "A reference correspondence cannot claim common physical coverage."
                )
            defect = None
        else:
            if coverage_defect is None:
                raise ValueError(
                    "A geometric relation requires a finite coverage defect."
                )
            defect = float(coverage_defect)
            if not isfinite(defect) or defect < 0.0:
                raise ValueError("coverage_defect must be finite and nonnegative.")
        if relation_ == "identity" and source != target:
            raise ValueError("An identity geometry binding needs one geometry identity.")
        if relation_ == "source-realization" and source_topology != target_topology:
            raise ValueError(
                "Source-realization geometry requires one declared reference complex."
            )
        self.source_geometry_id = source
        self.target_geometry_id = target
        self.source_topology_id = source_topology
        self.target_topology_id = target_topology
        self.relation = relation_
        self.coverage_defect = defect
        self.binding_id = canonical_fingerprint(
            {
                "kind": "transfer-geometry-binding",
                "source": source,
                "target": target,
                "source_topology": source_topology,
                "target_topology": target_topology,
                "relation": relation_,
                "coverage_defect": defect,
            }
        )


class FieldTransfer(StrictModule, NonTrainableState):
    """Prepared primal, dual-pullback, and Hilbert-adjoint field-space map.

    The three maps are distinct contracts: ``primal_operator`` moves source
    coefficients to target coefficients, ``dual_pullback_operator`` is its
    algebraic transpose moving target duals (residuals, loads) back, and
    ``hilbert_adjoint_operator`` is the adjoint under the declared source and
    target inner products. All three are linear operators; nonlinear limiters,
    positivity repairs, and history updates belong to consumer transactions.
    """

    source: DiscreteFieldSpace
    target: DiscreteFieldSpace
    primal_operator: AbstractLinearOperator
    dual_pullback_operator: AbstractLinearOperator | None
    hilbert_adjoint_operator: AbstractLinearOperator | None
    properties: TransferProperties
    geometry: TransferGeometryBinding | None
    preparation: PreparationReport
    transfer_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: DiscreteFieldSpace,
        target: DiscreteFieldSpace,
        primal_operator: AbstractLinearOperator,
        /,
        *,
        dual_pullback_operator: AbstractLinearOperator | None = None,
        hilbert_adjoint_operator: AbstractLinearOperator | None = None,
        properties: TransferProperties | None = None,
        geometry: TransferGeometryBinding | None = None,
        preparation: PreparationReport | None = None,
        transfer_id: str | None = None,
    ) -> None:
        if not primal_operator.source.compatible(
            source.vector_space
        ) or not primal_operator.target.compatible(target.vector_space):
            raise ValueError(
                "Primal transfer spaces must match source and target fields."
            )
        reverse_operators = {
            "dual_pullback_operator": dual_pullback_operator,
            "hilbert_adjoint_operator": hilbert_adjoint_operator,
        }
        for name, operator in reverse_operators.items():
            if operator is None:
                continue
            if not operator.source.compatible(
                target.vector_space
            ) or not operator.target.compatible(source.vector_space):
                raise ValueError(f"{name} spaces must reverse source and target.")
        properties_ = TransferProperties() if properties is None else properties
        if properties_.conservative and dual_pullback_operator is None:
            raise ValueError("Conservative transfers require a dual_pullback_operator.")
        if properties_.adjoint_paired and hilbert_adjoint_operator is None:
            raise ValueError(
                "Adjoint-paired transfers require a hilbert_adjoint_operator."
            )
        _check_semantics(properties_.semantics, source, target, properties_, geometry)
        preparation_ = PreparationReport() if preparation is None else preparation
        self.source = source
        self.target = target
        self.primal_operator = primal_operator
        self.dual_pullback_operator = dual_pullback_operator
        self.hilbert_adjoint_operator = hilbert_adjoint_operator
        self.properties = properties_
        self.geometry = geometry
        self.preparation = preparation_
        self.transfer_id = resolved_identifier(
            "transfer_id",
            transfer_id,
            {
                "kind": "field-transfer",
                "source": source.field_space_id,
                "target": target.field_space_id,
                "primal": primal_operator.operator_id,
                "dual_pullback": (
                    None
                    if dual_pullback_operator is None
                    else dual_pullback_operator.operator_id
                ),
                "hilbert_adjoint": (
                    None
                    if hilbert_adjoint_operator is None
                    else hilbert_adjoint_operator.operator_id
                ),
                "properties": {
                    "constant_preserving": properties_.constant_preserving,
                    "conservative": properties_.conservative,
                    "positivity_preserving": properties_.positivity_preserving,
                    "nested": properties_.nested,
                    "adjoint_paired": properties_.adjoint_paired,
                    "differentiable_geometry": properties_.differentiable_geometry,
                    "exact_on": list(properties_.exact_on),
                    "semantics": properties_.semantics,
                },
                "geometry": None if geometry is None else geometry.binding_id,
                "preparation": preparation_.report_id,
            },
        )


__all__ = [
    "FieldTransfer",
    "TransferGeometryBinding",
    "TransferGeometryRelation",
    "TransferProperties",
    "TransferSemantics",
]
