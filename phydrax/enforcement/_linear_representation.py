#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._frozendict import frozendict
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ..conditions._evidence import AffineProjectionCertificate, ConditionRealizationStamp
from ..conditions._ir import ProductFieldSpec
from ..conditions._lowering import BoundCondition
from ..conditions._relations import Equality
from ..linalg._constraint_operators import (
    ConstraintOperatorPlan,
    PreparedConstraintOperator,
)
from ..linalg._constraints import ConstraintMap
from ..linalg._operators import AbstractLinearOperator
from ..linalg._real_coordinates import AbstractRealCoordinateMap, RealCoordinateEvidence
from ..linalg._spaces import AbstractVectorSpace, ArraySpace, BlockSpace
from ..linalg._structured_operators import StackedLinearOperator
from ._lifecycle import (
    commit_refresh,
    propose_refresh,
    RealizationLifecycleState,
    record_realization_stamp,
    RefreshValidation,
    validate_refresh,
)
from ._realization import (
    AbstractFieldRealization,
    ConditionEvaluationContext,
    FieldRealizationResult,
    RealizationStatus,
)


_MAX_NUMERIC_VERSION = (1 << 63) - 1
_EXACT_ASSEMBLY_KINDS = frozenset({"construction", "continuum", "exact"})


def _identifier(value: Any, name: str, /) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _identifiers(values: Sequence[Any], name: str, /) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if any(not value for value in result):
        raise ValueError(f"{name} entries must be non-empty.")
    return result


def _unique(values: Sequence[str], /) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _version(value: int, name: str = "numeric_version", /) -> int:
    result = int(value)
    if result < 0 or result > _MAX_NUMERIC_VERSION:
        raise ValueError(f"{name} must be a nonnegative 63-bit integer.")
    return result


def _fold_versions(values: Sequence[tuple[str, int]], /) -> int:
    digest = canonical_fingerprint(
        {
            "kind": "linear-representation-numeric-version-v1",
            "children": [
                {"representation": identifier, "numeric_version": version}
                for identifier, version in values
            ],
        }
    )
    return int(digest[:16], 16) & _MAX_NUMERIC_VERSION


def _scalar_nonnegative(value: Any, name: str, /) -> Array:
    result = jnp.asarray(value)
    if result.shape:
        raise ValueError(f"{name} must be scalar.")
    if not jnp.issubdtype(result.dtype, jnp.inexact):
        result = result.astype(float)
    if not bool(jnp.isfinite(result)) or bool(result < 0):
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


def _mapping(value: Mapping[str, Any], name: str, /) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"{name} keys must be non-empty strings.")
    return value


def _tree_add(left: PyTree[Any], right: PyTree[Any], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x + y, left, right)


def _tree_sub(left: PyTree[Any], right: PyTree[Any], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x - y, left, right)


def _tree_finite(value: PyTree[Any], /) -> Array:
    leaves = jax.tree.leaves(value)
    checks = tuple(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    if not checks:
        return jnp.asarray(False)
    return jnp.all(jnp.stack(checks))


class LinearRepresentationCertificate(StrictModule, NonTrainableState):
    """Construction contract for one finite linear field representation.

    The certificate describes stable structure only. Numerical basis, trunk, or
    geometry changes are represented by ``numeric_version`` and ``prepared_id``
    on the representation, never by changing this certificate in place.
    """

    field_spec_id: str = eqx.field(static=True)
    field_names: tuple[str, ...] = eqx.field(static=True)
    native_coefficient_space_id: str = eqx.field(static=True)
    coefficient_space_id: str = eqx.field(static=True)
    extraction_id: str = eqx.field(static=True)
    replacement_id: str = eqx.field(static=True)
    synthesis_id: str = eqx.field(static=True)
    coordinate_evidence_id: str | None = eqx.field(static=True)
    support_ids: tuple[str, ...] = eqx.field(static=True)
    layout_ids: tuple[str, ...] = eqx.field(static=True)
    topology_ids: tuple[str, ...] = eqx.field(static=True)
    maximum_derivative_orders: tuple[Any, ...] = eqx.field(static=True)
    construction_dependencies: tuple[str, ...] = eqx.field(static=True)
    source_certificate_ids: tuple[str, ...] = eqx.field(static=True)
    proof: str = eqx.field(static=True)
    zero_preserving: bool = eqx.field(static=True)
    round_trip_exact: bool = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        field_spec_id: str,
        field_names: Sequence[str],
        native_coefficient_space_id: str,
        coefficient_space_id: str,
        extraction_id: str,
        replacement_id: str,
        synthesis_id: str,
        coordinate_evidence_id: str | None = None,
        support_ids: Sequence[str] = (),
        layout_ids: Sequence[str] = (),
        topology_ids: Sequence[str] = (),
        maximum_derivative_orders: Sequence[Any] = (),
        construction_dependencies: Sequence[str] = (),
        source_certificate_ids: Sequence[str] = (),
        proof: str = "construction",
        zero_preserving: bool = True,
        round_trip_exact: bool = True,
    ):
        field_names_ = _identifiers(field_names, "field_names")
        if not field_names_ or len(set(field_names_)) != len(field_names_):
            raise ValueError("field_names must be nonempty and unique.")
        coordinate_id = (
            None
            if coordinate_evidence_id is None
            else _identifier(coordinate_evidence_id, "coordinate_evidence_id")
        )
        identifiers = {
            "field_spec_id": _identifier(field_spec_id, "field_spec_id"),
            "native_coefficient_space_id": _identifier(
                native_coefficient_space_id, "native_coefficient_space_id"
            ),
            "coefficient_space_id": _identifier(
                coefficient_space_id, "coefficient_space_id"
            ),
            "extraction_id": _identifier(extraction_id, "extraction_id"),
            "replacement_id": _identifier(replacement_id, "replacement_id"),
            "synthesis_id": _identifier(synthesis_id, "synthesis_id"),
        }
        support_ids_ = _identifiers(support_ids, "support_ids")
        layout_ids_ = _identifiers(layout_ids, "layout_ids")
        topology_ids_ = _identifiers(topology_ids, "topology_ids")
        dependencies_ = _identifiers(
            construction_dependencies, "construction_dependencies"
        )
        source_ids_ = _identifiers(source_certificate_ids, "source_certificate_ids")
        derivative_orders = tuple(maximum_derivative_orders)
        proof_ = _identifier(proof, "proof")
        payload = {
            "kind": "linear-representation-certificate-v1",
            **identifiers,
            "field_names": list(field_names_),
            "coordinate_evidence_id": coordinate_id,
            "support_ids": list(support_ids_),
            "layout_ids": list(layout_ids_),
            "topology_ids": list(topology_ids_),
            "maximum_derivative_orders": derivative_orders,
            "construction_dependencies": list(dependencies_),
            "source_certificate_ids": list(source_ids_),
            "proof": proof_,
            "zero_preserving": bool(zero_preserving),
            "round_trip_exact": bool(round_trip_exact),
        }
        self.field_spec_id = identifiers["field_spec_id"]
        self.field_names = field_names_
        self.native_coefficient_space_id = identifiers["native_coefficient_space_id"]
        self.coefficient_space_id = identifiers["coefficient_space_id"]
        self.extraction_id = identifiers["extraction_id"]
        self.replacement_id = identifiers["replacement_id"]
        self.synthesis_id = identifiers["synthesis_id"]
        self.coordinate_evidence_id = coordinate_id
        self.support_ids = support_ids_
        self.layout_ids = layout_ids_
        self.topology_ids = topology_ids_
        self.maximum_derivative_orders = derivative_orders
        self.construction_dependencies = dependencies_
        self.source_certificate_ids = source_ids_
        self.proof = proof_
        self.zero_preserving = bool(zero_preserving)
        self.round_trip_exact = bool(round_trip_exact)
        self.representation_id = canonical_fingerprint(payload)


class LinearAssemblyEvidence(StrictModule, NonTrainableState):
    """Identity, layout, and exactness evidence for an assembled linear action."""

    bound_condition_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    coefficient_space_id: str = eqx.field(static=True)
    codomain_id: str = eqx.field(static=True)
    quantifier_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    row_shape: tuple[int, ...] = eqx.field(static=True)
    row_dtype: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_revision: str = eqx.field(static=True)
    assembly_method: str = eqx.field(static=True)
    exactness: str = eqx.field(static=True)
    numeric_fingerprint: str = eqx.field(static=True)
    coordinate_evidence_id: str | None = eqx.field(static=True)
    derivative_orders: tuple[Any, ...] = eqx.field(static=True)
    integration_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    preserved_certificate_ids: tuple[str, ...] = eqx.field(static=True)
    error_bound: Array
    tolerance: Array
    zero_preserving: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        bound_condition_id: str,
        operator_id: str,
        coefficient_space_id: str,
        codomain_id: str,
        quantifier_id: str,
        representation_id: str,
        prepared_id: str,
        row_shape: Sequence[int],
        row_dtype: Any,
        support_id: str,
        geometry_revision: Any,
        assembly_method: str,
        exactness: str,
        numeric_fingerprint: str,
        coordinate_evidence_id: str | None = None,
        derivative_orders: Sequence[Any] = (),
        integration_evidence_ids: Sequence[str] = (),
        preserved_certificate_ids: Sequence[str] = (),
        error_bound: Any = 0.0,
        tolerance: Any = 0.0,
        zero_preserving: bool = True,
    ):
        identifiers = {
            "bound_condition_id": _identifier(bound_condition_id, "bound_condition_id"),
            "operator_id": _identifier(operator_id, "operator_id"),
            "coefficient_space_id": _identifier(
                coefficient_space_id, "coefficient_space_id"
            ),
            "codomain_id": _identifier(codomain_id, "codomain_id"),
            "quantifier_id": _identifier(quantifier_id, "quantifier_id"),
            "representation_id": _identifier(representation_id, "representation_id"),
            "prepared_id": _identifier(prepared_id, "prepared_id"),
            "support_id": _identifier(support_id, "support_id"),
            "geometry_revision": _identifier(geometry_revision, "geometry_revision"),
            "assembly_method": _identifier(assembly_method, "assembly_method"),
            "exactness": _identifier(exactness, "exactness"),
            "numeric_fingerprint": _identifier(
                numeric_fingerprint, "numeric_fingerprint"
            ),
        }
        shape = tuple(int(size) for size in row_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("row_shape dimensions must be positive.")
        dtype = np.dtype(row_dtype).name
        coordinate_id = (
            None
            if coordinate_evidence_id is None
            else _identifier(coordinate_evidence_id, "coordinate_evidence_id")
        )
        derivative_orders_ = tuple(derivative_orders)
        integration_ids = _identifiers(
            integration_evidence_ids, "integration_evidence_ids"
        )
        certificate_ids = _identifiers(
            preserved_certificate_ids, "preserved_certificate_ids"
        )
        error = _scalar_nonnegative(error_bound, "error_bound")
        tolerance_ = _scalar_nonnegative(tolerance, "tolerance")
        self.bound_condition_id = identifiers["bound_condition_id"]
        self.operator_id = identifiers["operator_id"]
        self.coefficient_space_id = identifiers["coefficient_space_id"]
        self.codomain_id = identifiers["codomain_id"]
        self.quantifier_id = identifiers["quantifier_id"]
        self.representation_id = identifiers["representation_id"]
        self.prepared_id = identifiers["prepared_id"]
        self.row_shape = shape
        self.row_dtype = dtype
        self.support_id = identifiers["support_id"]
        self.geometry_revision = identifiers["geometry_revision"]
        self.assembly_method = identifiers["assembly_method"]
        self.exactness = identifiers["exactness"]
        self.numeric_fingerprint = identifiers["numeric_fingerprint"]
        self.coordinate_evidence_id = coordinate_id
        self.derivative_orders = derivative_orders_
        self.integration_evidence_ids = integration_ids
        self.preserved_certificate_ids = certificate_ids
        self.error_bound = error
        self.tolerance = tolerance_
        self.zero_preserving = bool(zero_preserving)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "linear-assembly-evidence-v1",
                **identifiers,
                "row_shape": list(shape),
                "row_dtype": dtype,
                "coordinate_evidence_id": coordinate_id,
                "derivative_orders": derivative_orders_,
                "integration_evidence_ids": list(integration_ids),
                "preserved_certificate_ids": list(certificate_ids),
                "error_bound": float(np.asarray(error)),
                "tolerance": float(np.asarray(tolerance_)),
                "zero_preserving": bool(zero_preserving),
            }
        )

    @property
    def exact(self) -> bool:
        return self.exactness in _EXACT_ASSEMBLY_KINDS and bool(self.error_bound == 0)


class LinearConditionAssembly(StrictModule, NonTrainableState):
    """Canonical real-coordinate operator for one bound linear condition."""

    operator: AbstractLinearOperator
    evidence: LinearAssemblyEvidence
    codomain_coordinates: Callable[[Any], PyTree[Any]] | None
    numeric_version: int = eqx.field(static=True)
    assembly_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        evidence: LinearAssemblyEvidence,
        /,
        *,
        codomain_coordinates: Callable[[Any], PyTree[Any]] | None = None,
        numeric_version: int = 0,
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if not isinstance(evidence, LinearAssemblyEvidence):
            raise TypeError("evidence must be LinearAssemblyEvidence.")
        if operator.batch_shape:
            raise ValueError("A linear condition assembly must be unbatched.")
        if not isinstance(operator.target, ArraySpace):
            raise TypeError("A linear condition assembly target must be an ArraySpace.")
        if operator.operator_id != evidence.operator_id:
            raise ValueError("Assembly evidence names a different operator.")
        if operator.source.space_id != evidence.coefficient_space_id:
            raise ValueError("Assembly evidence names a different coefficient space.")
        if operator.target.shape != evidence.row_shape:
            raise ValueError("Assembly evidence row shape differs from the target space.")
        if operator.target.dtype != np.dtype(evidence.row_dtype):
            raise TypeError("Assembly evidence row dtype differs from the target space.")
        if codomain_coordinates is not None and not callable(codomain_coordinates):
            raise TypeError("codomain_coordinates must be callable or None.")
        version = _version(numeric_version)
        self.operator = operator
        self.evidence = evidence
        self.codomain_coordinates = codomain_coordinates
        self.numeric_version = version
        self.assembly_id = canonical_fingerprint(
            {
                "kind": "linear-condition-assembly-v1",
                "operator": operator.operator_id,
                "evidence": evidence.evidence_id,
                "numeric_version": version,
            }
        )

    def coordinates(self, value: Any, /) -> PyTree[Array]:
        coordinates = (
            value
            if self.codomain_coordinates is None
            else self.codomain_coordinates(value)
        )
        return self.operator.target.validate(coordinates)

    def relation_target(self, relation: Any, /) -> PyTree[Array]:
        if not isinstance(relation, Equality):
            raise TypeError("Coefficient elimination requires an Equality relation.")
        if not relation.has_target:
            return self.operator.target.zeros()
        return self.coordinates(relation.target)


class AbstractLinearRepresentation(StrictModule):
    """Explicit finite linear coordinates for one ordered field specification."""

    field_spec: AbstractAttribute[ProductFieldSpec]
    native_coefficient_space: AbstractAttribute[AbstractVectorSpace]
    coefficient_space: AbstractAttribute[AbstractVectorSpace]
    real_coordinates: AbstractAttribute[AbstractRealCoordinateMap | None]
    certificate: AbstractAttribute[LinearRepresentationCertificate]
    numeric_version: AbstractAttribute[int]
    prepared_id: AbstractAttribute[str]

    @property
    def representation_id(self) -> str:
        return self.certificate.representation_id

    @abstractmethod
    def extract(self, values: Mapping[str, Any], /) -> PyTree[Array]:
        """Extract one validated independent-real coefficient vector."""
        raise NotImplementedError

    @abstractmethod
    def replace(
        self,
        values: Mapping[str, Any],
        coefficients: PyTree[Any],
        /,
    ) -> frozendict[str, Any]:
        """Return ``values`` with represented fields replaced by ``coefficients``."""
        raise NotImplementedError

    @abstractmethod
    def synthesize(self, coefficients: PyTree[Any], /) -> frozendict[str, Any]:
        """Synthesize exactly the represented fields from real coefficients."""
        raise NotImplementedError

    @abstractmethod
    def assemble(self, bound: BoundCondition, /) -> LinearConditionAssembly:
        """Assemble the target-independent coefficient action for ``bound``."""
        raise NotImplementedError


class _ProductRealCoordinateMap(AbstractRealCoordinateMap, NonTrainableState):
    maps: tuple[AbstractRealCoordinateMap | None, ...]

    def __init__(
        self,
        source_space: BlockSpace,
        coordinate_space: BlockSpace,
        maps: Sequence[AbstractRealCoordinateMap | None],
        /,
    ):
        maps_ = tuple(maps)
        if len(maps_) != len(source_space.spaces):
            raise ValueError("A product coordinate map requires one map per block.")
        for native, coordinate, coordinate_map in zip(
            source_space.spaces,
            coordinate_space.spaces,
            maps_,
            strict=True,
        ):
            if coordinate_map is None:
                if not native.compatible(coordinate):
                    raise ValueError(
                        "An identity coordinate block must preserve its space."
                    )
                continue
            if not isinstance(coordinate_map, AbstractRealCoordinateMap):
                raise TypeError(
                    "Product coordinate entries must be coordinate maps or None."
                )
            if not coordinate_map.source_space.compatible(native) or not (
                coordinate_map.coordinate_space.compatible(coordinate)
            ):
                raise ValueError("A child coordinate map has incompatible spaces.")
        identifier = canonical_fingerprint(
            {
                "kind": "product-real-coordinate-map-v1",
                "source": source_space.space_id,
                "coordinate": coordinate_space.space_id,
                "maps": [
                    None if value is None else value.coordinate_id for value in maps_
                ],
            }
        )
        source_spec = jax.tree.leaves(source_space.structure())[0]
        coordinate_spec = jax.tree.leaves(coordinate_space.structure())[0]
        evidence = RealCoordinateEvidence(
            domain_kind=(
                "full"
                if all(
                    value is None or value.evidence.domain_kind == "full"
                    for value in maps_
                )
                else "constrained_subspace"
            ),
            source_space_id=source_space.space_id,
            coordinate_space_id=coordinate_space.space_id,
            source_dtype=np.dtype(source_spec.dtype).name,
            coordinate_dtype=np.dtype(coordinate_spec.dtype).name,
            source_shape=(source_space.size,),
            coordinate_shape=(coordinate_space.size,),
            norm_relation=(
                "isometry"
                if all(
                    value is None or value.evidence.norm_relation == "isometry"
                    for value in maps_
                )
                else "coordinate_equivalence"
            ),
            projection_kind="ordered-product",
            map_id=identifier,
        )
        self.source_space = source_space
        self.coordinate_space = coordinate_space
        self.evidence = evidence
        self.coordinate_id = identifier
        self.maps = maps_

    def validate_state(self, state: Any, /):
        return self.source_space.validate(state)

    def validate_coordinates(self, coordinates: Any, /):
        return self.coordinate_space.validate(coordinates)

    def to_real_coordinates(self, state: Any, /):
        values = self.validate_state(state)
        return self.coordinate_space.validate(
            tuple(
                coordinate.validate(value)
                if coordinate_map is None
                else coordinate_map.to_real_coordinates(value)
                for value, coordinate, coordinate_map in zip(
                    values,
                    self.coordinate_space.spaces,
                    self.maps,
                    strict=True,
                )
            )
        )

    def from_real_coordinates(self, coordinates: Any, /):
        values = self.validate_coordinates(coordinates)
        return self.source_space.validate(
            tuple(
                native.validate(value)
                if coordinate_map is None
                else coordinate_map.from_real_coordinates(value)
                for value, native, coordinate_map in zip(
                    values,
                    self.source_space.spaces,
                    self.maps,
                    strict=True,
                )
            )
        )

    def project(self, state: Any, /):
        values = self.validate_state(state)
        return self.source_space.validate(
            tuple(
                native.validate(value)
                if coordinate_map is None
                else coordinate_map.project(value)
                for value, native, coordinate_map in zip(
                    values,
                    self.source_space.spaces,
                    self.maps,
                    strict=True,
                )
            )
        )

    def defect(self, state: Any, /) -> Array:
        values = self.validate_state(state)
        defects = tuple(
            jnp.zeros(()) if coordinate_map is None else coordinate_map.defect(value)
            for value, coordinate_map in zip(values, self.maps, strict=True)
        )
        return jnp.max(jnp.stack(defects))


class CallableLinearRepresentation(AbstractLinearRepresentation, NonTrainableState):
    """Explicit linear representation backed by caller-supplied typed actions."""

    field_spec: ProductFieldSpec
    native_coefficient_space: AbstractVectorSpace
    coefficient_space: AbstractVectorSpace
    real_coordinates: AbstractRealCoordinateMap | None
    certificate: LinearRepresentationCertificate
    extraction: Callable[[Mapping[str, Any]], PyTree[Array]]
    replacement: Callable[[Mapping[str, Any], PyTree[Any]], Mapping[str, Any]]
    synthesis: Callable[[PyTree[Any]], Mapping[str, Any]]
    assembly: Callable[[BoundCondition], LinearConditionAssembly]
    numeric_version: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_spec: ProductFieldSpec,
        native_coefficient_space: AbstractVectorSpace,
        coefficient_space: AbstractVectorSpace,
        extraction: Callable[[Mapping[str, Any]], PyTree[Array]],
        replacement: Callable[[Mapping[str, Any], PyTree[Any]], Mapping[str, Any]],
        synthesis: Callable[[PyTree[Any]], Mapping[str, Any]],
        assembly: Callable[[BoundCondition], LinearConditionAssembly],
        /,
        *,
        certificate: LinearRepresentationCertificate,
        real_coordinates: AbstractRealCoordinateMap | None = None,
        numeric_version: int = 0,
        prepared_id: str | None = None,
    ):
        if not isinstance(field_spec, ProductFieldSpec):
            raise TypeError("field_spec must be ProductFieldSpec.")
        if not isinstance(
            native_coefficient_space, AbstractVectorSpace
        ) or not isinstance(coefficient_space, AbstractVectorSpace):
            raise TypeError("Coefficient spaces must be AbstractVectorSpace values.")
        if any(
            not callable(action)
            for action in (extraction, replacement, synthesis, assembly)
        ):
            raise TypeError("Linear representation actions must be callable.")
        if not isinstance(certificate, LinearRepresentationCertificate):
            raise TypeError("certificate must be LinearRepresentationCertificate.")
        if certificate.field_spec_id != field_spec.field_spec_id:
            raise ValueError("Representation certificate names a different field spec.")
        if certificate.field_names != field_spec.sources:
            raise ValueError("Representation certificate field order changed.")
        if (
            certificate.native_coefficient_space_id != native_coefficient_space.space_id
            or certificate.coefficient_space_id != coefficient_space.space_id
        ):
            raise ValueError("Representation certificate names incompatible spaces.")
        if real_coordinates is None:
            if not native_coefficient_space.compatible(coefficient_space):
                raise ValueError(
                    "Representations without a real-coordinate map require compatible spaces."
                )
        else:
            if not isinstance(real_coordinates, AbstractRealCoordinateMap):
                raise TypeError("real_coordinates must be an AbstractRealCoordinateMap.")
            if not real_coordinates.source_space.compatible(
                native_coefficient_space
            ) or not real_coordinates.coordinate_space.compatible(coefficient_space):
                raise ValueError("Real-coordinate map has incompatible spaces.")
        version = _version(numeric_version)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "callable-linear-representation",
                    "certificate": certificate.representation_id,
                    "numeric_version": version,
                }
            )
            if prepared_id is None
            else _identifier(prepared_id, "prepared_id")
        )
        self.field_spec = field_spec
        self.native_coefficient_space = native_coefficient_space
        self.coefficient_space = coefficient_space
        self.real_coordinates = real_coordinates
        self.certificate = certificate
        self.extraction = extraction
        self.replacement = replacement
        self.synthesis = synthesis
        self.assembly = assembly
        self.numeric_version = version
        self.prepared_id = identifier

    def extract(self, values: Mapping[str, Any], /) -> PyTree[Array]:
        native = self.native_coefficient_space.validate(self.extraction(values))
        return (
            self.coefficient_space.validate(native)
            if self.real_coordinates is None
            else self.real_coordinates.to_real_coordinates(native)
        )

    def replace(
        self,
        values: Mapping[str, Any],
        coefficients: PyTree[Any],
        /,
    ) -> frozendict[str, Any]:
        coordinate = self.coefficient_space.validate(coefficients)
        native = (
            self.native_coefficient_space.validate(coordinate)
            if self.real_coordinates is None
            else self.real_coordinates.from_real_coordinates(coordinate)
        )
        result = frozendict(self.replacement(values, native))
        if tuple(result.keys()) != self.field_spec.sources:
            raise ValueError("Replacement must return every represented field in order.")
        return result

    def synthesize(self, coefficients: PyTree[Any], /) -> frozendict[str, Any]:
        coordinate = self.coefficient_space.validate(coefficients)
        native = (
            self.native_coefficient_space.validate(coordinate)
            if self.real_coordinates is None
            else self.real_coordinates.from_real_coordinates(coordinate)
        )
        result = frozendict(self.synthesis(native))
        if tuple(result.keys()) != self.field_spec.sources:
            raise ValueError("Synthesis must return every represented field in order.")
        return result

    def assemble(self, bound: BoundCondition, /) -> LinearConditionAssembly:
        result = self.assembly(bound)
        if not isinstance(result, LinearConditionAssembly):
            raise TypeError("assembly must return LinearConditionAssembly.")
        if not result.operator.source.compatible(self.coefficient_space):
            raise ValueError("Assembly source differs from representation coordinates.")
        return result


class ProductLinearRepresentation(AbstractLinearRepresentation, NonTrainableState):
    """Ordered joint representation with one independent coefficient block per child."""

    representations: tuple[AbstractLinearRepresentation, ...]
    field_spec: ProductFieldSpec
    native_coefficient_space: BlockSpace
    coefficient_space: BlockSpace
    real_coordinates: AbstractRealCoordinateMap | None
    certificate: LinearRepresentationCertificate
    numeric_version: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, representations: Sequence[AbstractLinearRepresentation], /):
        children = tuple(representations)
        if not children or any(
            not isinstance(child, AbstractLinearRepresentation) for child in children
        ):
            raise TypeError(
                "representations must contain one or more AbstractLinearRepresentation values."
            )
        for child in children:
            if not isinstance(child.field_spec, ProductFieldSpec):
                raise TypeError("A child representation must expose ProductFieldSpec.")
            if child.certificate.field_spec_id != child.field_spec.field_spec_id:
                raise ValueError(
                    "A child certificate names a different field specification."
                )
            if child.certificate.field_names != child.field_spec.sources:
                raise ValueError(
                    "A child certificate field order differs from its sources."
                )
            if child.real_coordinates is not None:
                if not child.native_coefficient_space.compatible(
                    child.real_coordinates.source_space
                ):
                    raise ValueError(
                        "A child real-coordinate source space is incompatible."
                    )
                if not child.coefficient_space.compatible(
                    child.real_coordinates.coordinate_space
                ):
                    raise ValueError(
                        "A child real-coordinate target space is incompatible."
                    )
            _version(child.numeric_version)
            _identifier(child.prepared_id, "child prepared_id")
        field_spec = ProductFieldSpec(
            tuple(field for child in children for field in child.field_spec.fields)
        )
        native_space = BlockSpace(
            tuple(child.native_coefficient_space for child in children)
        )
        coefficient_space = BlockSpace(
            tuple(child.coefficient_space for child in children)
        )
        coordinate_maps = tuple(child.real_coordinates for child in children)
        real_coordinates = (
            None
            if all(value is None for value in coordinate_maps)
            else _ProductRealCoordinateMap(
                native_space, coefficient_space, coordinate_maps
            )
        )
        child_ids = tuple(child.representation_id for child in children)
        coordinate_evidence_id = (
            None if real_coordinates is None else real_coordinates.evidence.evidence_id
        )
        extraction_id = canonical_fingerprint(
            {"kind": "product-extraction-v1", "children": list(child_ids)}
        )
        replacement_id = canonical_fingerprint(
            {"kind": "product-replacement-v1", "children": list(child_ids)}
        )
        synthesis_id = canonical_fingerprint(
            {"kind": "product-synthesis-v1", "children": list(child_ids)}
        )
        certificate = LinearRepresentationCertificate(
            field_spec_id=field_spec.field_spec_id,
            field_names=field_spec.sources,
            native_coefficient_space_id=native_space.space_id,
            coefficient_space_id=coefficient_space.space_id,
            extraction_id=extraction_id,
            replacement_id=replacement_id,
            synthesis_id=synthesis_id,
            coordinate_evidence_id=coordinate_evidence_id,
            support_ids=_unique(
                tuple(
                    value for child in children for value in child.certificate.support_ids
                )
            ),
            layout_ids=_unique(
                tuple(
                    value for child in children for value in child.certificate.layout_ids
                )
            ),
            topology_ids=_unique(
                tuple(
                    value
                    for child in children
                    for value in child.certificate.topology_ids
                )
            ),
            maximum_derivative_orders=tuple(
                value
                for child in children
                for value in child.certificate.maximum_derivative_orders
            ),
            construction_dependencies=_unique(
                tuple(
                    value
                    for child in children
                    for value in child.certificate.construction_dependencies
                )
            ),
            source_certificate_ids=_unique(
                tuple(
                    value
                    for child in children
                    for value in (
                        child.representation_id,
                        *child.certificate.source_certificate_ids,
                    )
                )
            ),
            zero_preserving=all(child.certificate.zero_preserving for child in children),
            round_trip_exact=all(
                child.certificate.round_trip_exact for child in children
            ),
        )
        numeric_version = _fold_versions(
            tuple((child.representation_id, child.numeric_version) for child in children)
        )
        self.representations = children
        self.field_spec = field_spec
        self.native_coefficient_space = native_space
        self.coefficient_space = coefficient_space
        self.real_coordinates = real_coordinates
        self.certificate = certificate
        self.numeric_version = numeric_version
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "product-linear-representation-v1",
                "representation": certificate.representation_id,
                "numeric_version": numeric_version,
                "children": [child.prepared_id for child in children],
            }
        )

    def extract(self, values: Mapping[str, Any], /):
        checked = _mapping(values, "values")
        return self.coefficient_space.validate(
            tuple(child.extract(checked) for child in self.representations)
        )

    def replace(self, values: Mapping[str, Any], coefficients: Any, /):
        result: Mapping[str, Any] = _mapping(values, "values")
        blocks = self.coefficient_space.validate(coefficients)
        for child, block in zip(self.representations, blocks, strict=True):
            result = _mapping(child.replace(result, block), "replacement result")
        if tuple(result) != tuple(values):
            raise ValueError("Replacement must preserve the input mapping key order.")
        return frozendict(result)

    def synthesize(self, coefficients: Any, /):
        blocks = self.coefficient_space.validate(coefficients)
        result: dict[str, Any] = {}
        for child, block in zip(self.representations, blocks, strict=True):
            fields = _mapping(child.synthesize(block), "synthesis result")
            if tuple(fields) != child.field_spec.sources:
                raise ValueError(
                    "Child synthesis must return exactly its ordered field sources."
                )
            overlap = set(result).intersection(fields)
            if overlap:
                raise ValueError(
                    f"Product synthesis writes duplicate fields {overlap!r}."
                )
            result.update(fields)
        return frozendict(result)

    def assemble(self, bound: BoundCondition, /) -> LinearConditionAssembly:
        if not isinstance(bound, BoundCondition):
            raise TypeError("bound must be a BoundCondition.")
        if bound.condition.fields.sources != self.field_spec.sources:
            raise ValueError(
                "A product representation must cover all bound condition sources in order."
            )
        assemblies = tuple(child.assemble(bound) for child in self.representations)
        first = assemblies[0]
        for child, assembly in zip(self.representations, assemblies, strict=True):
            if not isinstance(assembly, LinearConditionAssembly):
                raise TypeError("Child assemble() must return LinearConditionAssembly.")
            if assembly.evidence.bound_condition_id != bound.bound_id:
                raise ValueError(
                    "Child assembly evidence names a different bound condition."
                )
            if assembly.evidence.representation_id != child.representation_id:
                raise ValueError(
                    "Child assembly evidence names a different representation."
                )
            if assembly.evidence.prepared_id != child.prepared_id:
                raise ValueError(
                    "Child assembly evidence names a different prepared representation."
                )
            if not assembly.operator.target.compatible(first.operator.target):
                raise ValueError("Product assembly children must share one target space.")
            if (
                assembly.evidence.codomain_id != first.evidence.codomain_id
                or assembly.evidence.quantifier_id != first.evidence.quantifier_id
                or assembly.evidence.support_id != first.evidence.support_id
                or assembly.evidence.coordinate_evidence_id
                != first.evidence.coordinate_evidence_id
            ):
                raise ValueError(
                    "Product assembly children disagree on condition semantics."
                )
            if assembly.numeric_version != child.numeric_version:
                raise ValueError(
                    "Child assembly and representation numeric versions differ."
                )
        operator = StackedLinearOperator(
            tuple(assembly.operator for assembly in assemblies),
            axis="horizontal",
            operator_id=canonical_fingerprint(
                {
                    "kind": "product-linear-condition-operator-v1",
                    "representation": self.representation_id,
                    "children": [
                        assembly.operator.operator_id for assembly in assemblies
                    ],
                }
            ),
        )
        if not operator.source.compatible(self.coefficient_space):
            raise ValueError(
                "Product assembly source differs from its coefficient space."
            )
        evidence = LinearAssemblyEvidence(
            bound_condition_id=bound.bound_id,
            operator_id=operator.operator_id,
            coefficient_space_id=self.coefficient_space.space_id,
            codomain_id=first.evidence.codomain_id,
            quantifier_id=first.evidence.quantifier_id,
            representation_id=self.representation_id,
            prepared_id=self.prepared_id,
            row_shape=first.evidence.row_shape,
            row_dtype=first.evidence.row_dtype,
            support_id=first.evidence.support_id,
            geometry_revision=canonical_fingerprint(
                {
                    "kind": "product-geometry-revision-v1",
                    "children": [
                        assembly.evidence.geometry_revision for assembly in assemblies
                    ],
                }
            ),
            assembly_method="product-horizontal-stack",
            exactness=(
                first.evidence.exactness
                if all(
                    assembly.evidence.exactness == first.evidence.exactness
                    for assembly in assemblies
                )
                else "mixed"
            ),
            numeric_fingerprint=canonical_fingerprint(
                {
                    "kind": "product-linear-assembly-numerics-v1",
                    "children": [
                        assembly.evidence.numeric_fingerprint for assembly in assemblies
                    ],
                }
            ),
            coordinate_evidence_id=first.evidence.coordinate_evidence_id,
            derivative_orders=tuple(
                value
                for assembly in assemblies
                for value in assembly.evidence.derivative_orders
            ),
            integration_evidence_ids=_unique(
                tuple(
                    value
                    for assembly in assemblies
                    for value in assembly.evidence.integration_evidence_ids
                )
            ),
            preserved_certificate_ids=_unique(
                tuple(
                    value
                    for child, assembly in zip(
                        self.representations, assemblies, strict=True
                    )
                    for value in (
                        child.representation_id,
                        *assembly.evidence.preserved_certificate_ids,
                    )
                )
            ),
            error_bound=jnp.max(
                jnp.stack(tuple(assembly.evidence.error_bound for assembly in assemblies))
            ),
            tolerance=jnp.max(
                jnp.stack(tuple(assembly.evidence.tolerance for assembly in assemblies))
            ),
            zero_preserving=all(
                assembly.evidence.zero_preserving for assembly in assemblies
            ),
        )
        return LinearConditionAssembly(
            operator,
            evidence,
            codomain_coordinates=first.codomain_coordinates,
            numeric_version=self.numeric_version,
        )


class CoefficientElimination(AbstractFieldRealization, NonTrainableState):
    """Exact equality realization by eliminating represented coefficients."""

    representation: AbstractLinearRepresentation
    assembly: LinearConditionAssembly
    prepared_operator: PreparedConstraintOperator
    constraint_map: ConstraintMap
    realization_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        representation: AbstractLinearRepresentation,
        assembly: LinearConditionAssembly,
        /,
        *,
        prepared_operator: PreparedConstraintOperator | None = None,
    ):
        if not isinstance(representation, AbstractLinearRepresentation):
            raise TypeError("representation must be AbstractLinearRepresentation.")
        if not isinstance(assembly, LinearConditionAssembly):
            raise TypeError("assembly must be LinearConditionAssembly.")
        if not assembly.operator.source.compatible(representation.coefficient_space):
            raise ValueError(
                "Assembly source must match the representation coefficient space."
            )
        if assembly.evidence.representation_id != representation.representation_id:
            raise ValueError("Assembly evidence names a different representation.")
        if assembly.evidence.prepared_id != representation.prepared_id:
            raise ValueError(
                "Assembly evidence names a different prepared representation."
            )
        if assembly.numeric_version != representation.numeric_version:
            raise ValueError("Assembly and representation numeric versions differ.")
        prepared = (
            ConstraintOperatorPlan(assembly.operator).prepare()
            if prepared_operator is None
            else prepared_operator
        )
        if not isinstance(prepared, PreparedConstraintOperator):
            raise TypeError(
                "prepared_operator must be PreparedConstraintOperator or None."
            )
        if prepared.operator.operator_id != assembly.operator.operator_id or not (
            prepared.source_space.compatible(representation.coefficient_space)
            and prepared.target_space.compatible(assembly.operator.target)
        ):
            raise ValueError("Prepared constraint operator does not match the assembly.")
        if not prepared.evidence.full_row_rank:
            raise ValueError(
                "Coefficient elimination requires full-row-rank constraints."
            )
        constraint_map = ConstraintMap(
            representation.coefficient_space,
            prepared.nullspace_operator.source,
            prepared.nullspace_operator,
            constraint_id=canonical_fingerprint(
                {
                    "kind": "coefficient-elimination-constraint-map-v1",
                    "assembly": assembly.assembly_id,
                    "prepared_constraint": prepared.prepared_id,
                }
            ),
        )
        self.representation = representation
        self.assembly = assembly
        self.prepared_operator = prepared
        self.constraint_map = constraint_map
        self.provider_id = "phydrax.enforcement.CoefficientElimination"
        self.realization_id = canonical_fingerprint(
            {
                "kind": "coefficient-elimination-v1",
                "representation": representation.prepared_id,
                "assembly": assembly.assembly_id,
                "constraint": prepared.prepared_id,
            }
        )

    def lift(self, target: Any, /):
        """Return the minimum-norm coefficient lift of a raw relation target."""
        return self.prepared_operator.strict_right_inverse(
            self.assembly.coordinates(target)
        )

    def _failed(
        self,
        state: RealizationLifecycleState,
        context: ConditionEvaluationContext,
        status: RealizationStatus,
        message: str,
        /,
        *,
        evidence: Any = None,
    ) -> FieldRealizationResult:
        proposal = propose_refresh((), state, context=context)
        validation = RefreshValidation.reject(
            status,
            message=message,
            evidence=evidence,
        )
        failed = commit_refresh(state, proposal, validation)
        return FieldRealizationResult.failure(
            status,
            state=failed,
            message=message,
            evidence=evidence,
        )

    def realize(
        self,
        fields: Mapping[str, Any],
        state: RealizationLifecycleState | None = None,
        *,
        context: ConditionEvaluationContext,
    ) -> FieldRealizationResult:
        if not isinstance(context, ConditionEvaluationContext):
            raise TypeError("context must be ConditionEvaluationContext.")
        current = RealizationLifecycleState.initial() if state is None else state
        if not isinstance(current, RealizationLifecycleState):
            raise TypeError("state must be RealizationLifecycleState or None.")
        proposal = propose_refresh((), current, context=context)
        validation = validate_refresh(proposal)
        committed = commit_refresh(current, proposal, validation)
        if not validation.accepted:
            return FieldRealizationResult.failure(
                validation.status,
                state=committed,
                message=validation.message,
                evidence=validation.evidence,
            )
        checked_fields = _mapping(fields, "fields")
        sources = context.condition.fields.sources
        if sources != self.representation.field_spec.sources:
            return self._failed(
                committed,
                context,
                RealizationStatus.INVALID_INPUT,
                "The condition sources do not match the representation field order.",
            )
        missing = tuple(name for name in sources if name not in checked_fields)
        if missing:
            return self._failed(
                committed,
                context,
                RealizationStatus.INVALID_INPUT,
                f"Represented fields are missing sources {missing!r}.",
            )
        bound = BoundCondition(
            context.condition,
            {name: checked_fields[name] for name in sources},
        )
        if bound.bound_id != self.assembly.evidence.bound_condition_id:
            return self._failed(
                committed,
                context,
                RealizationStatus.INVALID_INPUT,
                "The realization condition differs from the assembled condition.",
            )
        relation = context.condition.relation
        if not isinstance(relation, Equality):
            return self._failed(
                committed,
                context,
                RealizationStatus.UNSUPPORTED,
                "Coefficient elimination supports Equality relations only.",
            )
        if context.exact_required and not self.assembly.evidence.exact:
            return self._failed(
                committed,
                context,
                RealizationStatus.UNSUPPORTED,
                "The assembled coefficient action does not certify exact semantics.",
                evidence=self.assembly.evidence,
            )
        target = self.assembly.relation_target(relation)
        coefficients = self.representation.coefficient_space.validate(
            self.representation.extract(checked_fields)
        )
        action = self.prepared_operator.apply(coefficients)
        correction = self.prepared_operator.strict_right_inverse(
            _tree_sub(target, action)
        )
        realized_coefficients = self.representation.coefficient_space.validate(
            _tree_add(coefficients, correction)
        )
        residual = _tree_sub(self.prepared_operator.apply(realized_coefficients), target)
        residual_norm = jnp.sqrt(
            jnp.maximum(
                jnp.real(self.prepared_operator.target_space.inner(residual, residual)),
                0.0,
            )
        )
        tolerance = self.assembly.evidence.tolerance
        finite = _tree_finite(realized_coefficients) & jnp.isfinite(residual_norm)
        verified = finite & (residual_norm <= tolerance)
        stamp = ConditionRealizationStamp(
            context.condition_id,
            bound.bound_id,
            self.realization_id,
            self.provider_id,
            quantifier=context.quantifier,
            exact=self.assembly.evidence.exact,
        )
        certificate = AffineProjectionCertificate(
            stamp,
            residual_norm,
            tolerance,
            verified,
            certificate_id=canonical_fingerprint(
                {
                    "kind": "coefficient-elimination-certificate-v1",
                    "realization": self.realization_id,
                    "condition": context.condition_id,
                    "accepted_step": context.accepted_step,
                    "numeric_version": self.assembly.numeric_version,
                }
            ),
            rank=self.prepared_operator.rank,
            nullity=self.prepared_operator.nullity,
        )
        if not bool(np.asarray(verified)):
            status = (
                RealizationStatus.NONFINITE
                if not bool(np.asarray(finite))
                else RealizationStatus.VALIDATION_FAILED
            )
            return self._failed(
                committed,
                context,
                status,
                "Coefficient elimination failed its finite residual certificate.",
                evidence=certificate,
            )
        realized_fields = self.representation.replace(
            checked_fields, realized_coefficients
        )
        accepted_state = record_realization_stamp(committed, stamp)
        return FieldRealizationResult.success(
            realized_fields,
            state=accepted_state,
            stamp=stamp,
            evidence=certificate,
        )


__all__ = [
    "AbstractLinearRepresentation",
    "CallableLinearRepresentation",
    "CoefficientElimination",
    "LinearAssemblyEvidence",
    "LinearConditionAssembly",
    "LinearRepresentationCertificate",
    "ProductLinearRepresentation",
]
