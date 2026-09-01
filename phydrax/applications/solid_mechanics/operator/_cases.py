#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._frozendict import frozendict
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....nn.operator.data import (
    OperatorBatch,
    OperatorCaseProvenance,
    OperatorPrediction,
    stack_operator_batches,
)
from ....nn.operator.sampling import OperatorCase as CanonicalOperatorCase
from ....nn.operator.training._risk import MechanicsCaseReduction
from ._parameters import (
    MechanicsParameterDistribution,
    MechanicsParameterRealization,
)


class MechanicsGeometryMap(StrictModule, NonTrainableState):
    """Parameter-conditioned reference-to-physical geometry and measure map."""

    coordinate_map: Callable = eqx.field(static=True)
    jacobian_map: Callable = eqx.field(static=True)
    reference_domain_id: str = eqx.field(static=True)
    physical_domain_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    orientation: int = eqx.field(static=True)
    boundary_correspondence: frozendict[str, str] = eqx.field(static=True)
    coordinate_convention: Literal["reference", "physical"] = eqx.field(static=True)
    geometry_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        coordinate_map: Callable,
        jacobian_map: Callable,
        /,
        *,
        reference_domain_id: str,
        physical_domain_id: str,
        geometry_id: str,
        orientation: int = 1,
        coordinate_convention: Literal["reference", "physical"] = "reference",
        boundary_correspondence: Mapping[str, str] | None = None,
    ):
        if not callable(coordinate_map) or not callable(jacobian_map):
            raise TypeError(
                "Mechanics geometry coordinate and Jacobian maps must be callable."
            )
        reference_id = str(reference_domain_id)
        physical_id = str(physical_domain_id)
        identifier = str(geometry_id)
        if not reference_id or not physical_id or not identifier:
            raise ValueError("Mechanics geometry and domain IDs must be non-empty.")
        orientation_ = int(orientation)
        if orientation_ not in (-1, 1):
            raise ValueError("Mechanics geometry orientation must be -1 or 1.")
        if coordinate_convention not in ("reference", "physical"):
            raise ValueError(
                "Mechanics geometry coordinate convention must be 'reference' or 'physical'."
            )
        boundaries = frozendict(
            {
                str(reference): str(physical)
                for reference, physical in (
                    {} if boundary_correspondence is None else boundary_correspondence
                ).items()
            }
        )
        if any(
            not reference or not physical for reference, physical in boundaries.items()
        ):
            raise ValueError("Boundary correspondence IDs must be non-empty.")
        if len(set(boundaries.values())) != len(boundaries):
            raise ValueError("Physical boundary correspondence must be one-to-one.")
        fingerprint = canonical_fingerprint(
            {
                "kind": "mechanics-geometry-map",
                "geometry_id": identifier,
                "reference_domain": reference_id,
                "physical_domain": physical_id,
                "orientation": orientation_,
                "coordinate_convention": coordinate_convention,
                "boundaries": dict(boundaries),
            }
        )
        self.coordinate_map = coordinate_map
        self.jacobian_map = jacobian_map
        self.reference_domain_id = reference_id
        self.physical_domain_id = physical_id
        self.geometry_id = identifier
        self.orientation = orientation_
        self.boundary_correspondence = boundaries
        self.coordinate_convention = coordinate_convention
        self.geometry_fingerprint = fingerprint

    def map_coordinates(
        self,
        reference_coordinates: ArrayLike,
        realization: MechanicsParameterRealization,
        /,
    ) -> Array:
        coordinates = jnp.asarray(reference_coordinates)
        mapped = jnp.asarray(self.coordinate_map(coordinates, realization))
        if mapped.shape != coordinates.shape:
            raise ValueError("Mechanics geometry maps must preserve coordinate shape.")
        return eqx.error_if(
            mapped,
            jnp.any(~jnp.isfinite(mapped)),
            "Mechanics geometry produced nonfinite coordinates.",
        )

    def jacobian(
        self,
        reference_coordinates: ArrayLike,
        realization: MechanicsParameterRealization,
        /,
    ) -> Array:
        coordinates = jnp.asarray(reference_coordinates)
        if coordinates.ndim < 1:
            raise ValueError("Reference coordinates require a coordinate axis.")
        dimension = int(coordinates.shape[-1])
        jacobian = jnp.asarray(self.jacobian_map(coordinates, realization))
        expected = coordinates.shape[:-1] + (dimension, dimension)
        if jacobian.shape != expected:
            raise ValueError(
                f"Mechanics geometry Jacobian must have shape {expected}; got {jacobian.shape}."
            )
        determinant = jnp.linalg.det(jacobian)
        invalid = (
            jnp.any(~jnp.isfinite(jacobian))
            | jnp.any(~jnp.isfinite(determinant))
            | jnp.any(self.orientation * determinant <= 0.0)
        )
        return eqx.error_if(
            jacobian,
            invalid,
            "Mechanics geometry Jacobian is singular or violates declared orientation.",
        )

    def volume_transform(
        self,
        reference_coordinates: ArrayLike,
        realization: MechanicsParameterRealization,
        /,
    ) -> Array:
        return jnp.abs(jnp.linalg.det(self.jacobian(reference_coordinates, realization)))

    def surface_transform(
        self,
        reference_coordinates: ArrayLike,
        reference_normal: ArrayLike,
        realization: MechanicsParameterRealization,
        /,
    ) -> Array:
        """Apply Nanson's cofactor map to a reference oriented area vector."""
        jacobian = self.jacobian(reference_coordinates, realization)
        normal = jnp.asarray(reference_normal, dtype=jacobian.dtype)
        expected = jacobian.shape[:-2] + (jacobian.shape[-1],)
        normal = jnp.broadcast_to(normal, expected)
        determinant = jnp.linalg.det(jacobian)
        pulled = jnp.linalg.solve(jnp.swapaxes(jacobian, -1, -2), normal)
        transformed = determinant[..., None] * pulled
        return eqx.error_if(
            transformed,
            jnp.any(~jnp.isfinite(transformed)),
            "Mechanics surface transformation is nonfinite.",
        )

    def surface_measure_transform(
        self,
        reference_coordinates: ArrayLike,
        reference_normal: ArrayLike,
        realization: MechanicsParameterRealization,
        /,
    ) -> Array:
        transformed = self.surface_transform(
            reference_coordinates,
            reference_normal,
            realization,
        )
        return jnp.sqrt(jnp.sum(transformed * transformed, axis=-1))

    def volume_weights(
        self,
        reference_coordinates: ArrayLike,
        reference_weights: ArrayLike,
        realization: MechanicsParameterRealization,
        /,
    ) -> Array:
        """Push explicit reference-volume weights to the physical domain."""
        transform = self.volume_transform(reference_coordinates, realization)
        weights = jnp.asarray(reference_weights, dtype=transform.dtype)
        if weights.shape != transform.shape:
            raise ValueError(
                "Reference volume weights must match the geometry sample shape."
            )
        return eqx.error_if(
            weights * transform,
            jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0.0),
            "Reference volume weights must be finite and non-negative.",
        )

    def surface_weights(
        self,
        reference_coordinates: ArrayLike,
        reference_normal: ArrayLike,
        reference_weights: ArrayLike,
        realization: MechanicsParameterRealization,
        /,
    ) -> Array:
        """Push explicit reference-surface weights to the physical boundary."""
        transform = self.surface_measure_transform(
            reference_coordinates,
            reference_normal,
            realization,
        )
        weights = jnp.asarray(reference_weights, dtype=transform.dtype)
        if weights.shape != transform.shape:
            raise ValueError(
                "Reference surface weights must match the geometry sample shape."
            )
        return eqx.error_if(
            weights * transform,
            jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0.0),
            "Reference surface weights must be finite and non-negative.",
        )

    def physical_boundary(self, reference_boundary_id: str, /) -> str:
        if reference_boundary_id not in self.boundary_correspondence:
            raise KeyError(f"Unknown reference boundary {reference_boundary_id!r}.")
        return self.boundary_correspondence[reference_boundary_id]


class OperatorTrialFieldAdapter(StrictModule, NonTrainableState):
    """Convert named operator outputs into named, conditionally admissible fields."""

    field_names: tuple[str, ...] = eqx.field(static=True)
    field_factories: frozendict[str, Callable] = eqx.field(static=True)
    lifts: frozendict[str, Callable] = eqx.field(static=True)
    envelopes: frozendict[str, Callable] = eqx.field(static=True)
    constraint_validators: frozendict[str, Callable] = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)
    field_domain_ids: frozendict[str, str] = eqx.field(static=True)
    query_support_ids: frozendict[str, str] = eqx.field(static=True)
    adapter_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        field_names: Sequence[str],
        /,
        *,
        adapter_id: str,
        field_factories: Mapping[str, Callable] | None = None,
        lifts: Mapping[str, Callable] | None = None,
        envelopes: Mapping[str, Callable] | None = None,
        constraint_validators: Mapping[str, Callable] | None = None,
        field_domain_ids: Mapping[str, str] | None = None,
        query_support_ids: Mapping[str, str] | None = None,
    ):
        names = tuple(str(name) for name in field_names)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Operator trial field names must be non-empty and unique.")
        factories = frozendict({} if field_factories is None else field_factories)
        lift_map = frozendict({} if lifts is None else lifts)
        envelope_map = frozendict({} if envelopes is None else envelopes)
        validators = frozendict(
            {} if constraint_validators is None else constraint_validators
        )
        domains = frozendict(
            {
                str(name): str(domain)
                for name, domain in (
                    {} if field_domain_ids is None else field_domain_ids
                ).items()
            }
        )
        supports = frozendict(
            {
                str(name): str(support)
                for name, support in (
                    {} if query_support_ids is None else query_support_ids
                ).items()
            }
        )
        unknown = (
            set(factories)
            | set(lift_map)
            | set(envelope_map)
            | set(validators)
            | set(domains)
            | set(supports)
        ) - set(names)
        if unknown:
            raise ValueError(
                f"Trial field adapter references unknown fields {sorted(unknown)}."
            )
        if set(lift_map) != set(envelope_map):
            raise ValueError("Every hard-constraint lift requires exactly one envelope.")
        if set(validators) - set(lift_map):
            raise ValueError("Constraint validators require a lift/envelope pair.")
        for mapping in (factories, lift_map, envelope_map, validators):
            if any(not callable(function) for function in mapping.values()):
                raise TypeError("Trial field factories and constraints must be callable.")
        if any(not value for value in (*domains.values(), *supports.values())):
            raise ValueError("Trial field domain and support IDs must be non-empty.")
        identifier = str(adapter_id)
        if not identifier:
            raise ValueError("Operator trial field adapter IDs must be non-empty.")
        fingerprint = canonical_fingerprint(
            {
                "kind": "mechanics-operator-trial-field-adapter",
                "adapter_id": identifier,
                "fields": list(names),
                "factories": sorted(factories),
                "hard_constraints": sorted(lift_map),
                "validators": sorted(validators),
                "field_domains": dict(domains),
                "query_supports": dict(supports),
            }
        )
        self.field_names = names
        self.field_factories = factories
        self.lifts = lift_map
        self.envelopes = envelope_map
        self.constraint_validators = validators
        self.adapter_id = identifier
        self.field_domain_ids = domains
        self.query_support_ids = supports
        self.adapter_fingerprint = fingerprint

    def conditioned_values(
        self,
        prediction: OperatorPrediction,
        realization: MechanicsParameterRealization,
        /,
        *,
        geometry: MechanicsGeometryMap | None = None,
    ) -> frozendict[str, Array]:
        if not isinstance(prediction, OperatorPrediction):
            raise TypeError("prediction must be an OperatorPrediction.")
        if not isinstance(realization, MechanicsParameterRealization):
            raise TypeError("realization must be a MechanicsParameterRealization.")
        if set(prediction.fields) != set(self.field_names):
            raise ValueError(
                "Operator prediction fields must exactly match the trial field adapter."
            )
        if geometry is not None and not isinstance(geometry, MechanicsGeometryMap):
            raise TypeError("geometry must be a MechanicsGeometryMap or None.")
        values: dict[str, Array] = {}
        for name in self.field_names:
            field = prediction.field(name)
            query = prediction.query_geometry(field.query_name)
            if (
                name in self.query_support_ids
                and query.support_id != self.query_support_ids[name]
            ):
                raise ValueError(
                    f"Trial field {name!r} uses query support {query.support_id!r}; "
                    f"expected {self.query_support_ids[name]!r}."
                )
            if (
                geometry is not None
                and name in self.field_domain_ids
                and self.field_domain_ids[name] != geometry.physical_domain_id
            ):
                raise ValueError(
                    f"Trial field {name!r} domain does not match the case geometry."
                )
            raw = jnp.asarray(field.values)
            if name in self.lifts:
                coordinates = query.coordinates_array(
                    case_shape=prediction.case_shape,
                    flatten=False,
                )
                lift = jnp.asarray(self.lifts[name](coordinates, realization))
                envelope = jnp.asarray(self.envelopes[name](coordinates, realization))
                try:
                    lift = jnp.broadcast_to(lift, raw.shape)
                    envelope = jnp.broadcast_to(envelope, raw.shape)
                except ValueError as error:
                    raise ValueError(
                        f"Hard-constraint lift/envelope for field {name!r} cannot "
                        "broadcast to its prediction shape."
                    ) from error
                conditioned = lift + envelope * raw
                valid = jnp.all(jnp.isfinite(lift)) & jnp.all(jnp.isfinite(envelope))
                if name in self.constraint_validators:
                    valid = valid & jnp.all(
                        jnp.asarray(
                            self.constraint_validators[name](
                                conditioned,
                                query,
                                realization,
                            ),
                            dtype=bool,
                        )
                    )
                raw = eqx.error_if(
                    conditioned,
                    ~valid,
                    f"Hard constraint for trial field {name!r} is invalid.",
                )
            raw = eqx.error_if(
                raw,
                jnp.any(~jnp.isfinite(raw)),
                f"Trial field {name!r} is nonfinite.",
            )
            values[name] = raw
        return frozendict(values)

    def __call__(
        self,
        prediction: OperatorPrediction,
        realization: MechanicsParameterRealization,
        /,
        *,
        geometry: MechanicsGeometryMap | None = None,
    ) -> frozendict[str, Any]:
        conditioned = self.conditioned_values(
            prediction,
            realization,
            geometry=geometry,
        )
        fields: dict[str, Any] = {}
        for name, values in conditioned.items():
            if name not in self.field_factories:
                fields[name] = values
                continue
            predicted = prediction.field(name)
            fields[name] = self.field_factories[name](
                values,
                prediction.query_geometry(predicted.query_name),
                realization,
            )
        return frozendict(fields)


@dataclass(frozen=True)
class MechanicsOperatorCase:
    """Canonical operator case bound to one complete physical realization."""

    case: CanonicalOperatorCase
    realization: MechanicsParameterRealization
    geometry: MechanicsGeometryMap
    parameter_weight: float
    weight_kind: str
    identities: frozendict[str, str]
    case_fingerprint: str

    def __post_init__(self):
        if not isinstance(self.case, CanonicalOperatorCase):
            raise TypeError("case must be an OperatorCase.")
        if not isinstance(self.realization, MechanicsParameterRealization):
            raise TypeError("realization must be a MechanicsParameterRealization.")
        if not isinstance(self.geometry, MechanicsGeometryMap):
            raise TypeError("geometry must be a MechanicsGeometryMap.")
        weight = float(self.parameter_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                "Mechanics case parameter weights must be finite and positive."
            )
        if self.weight_kind not in ("equal", "probability", "importance"):
            raise ValueError("Unknown mechanics case parameter weight kind.")
        if self.case.provenance is None:
            raise ValueError("Mechanics operator cases require provenance.")
        if self.case.provenance.case_id != self.realization.case_id:
            raise ValueError("Mechanics case and realization IDs must match.")
        if not self.case_fingerprint:
            raise ValueError("Mechanics case fingerprints must be non-empty.")

    @property
    def batch(self) -> OperatorBatch:
        return self.case.batch

    @property
    def targets(self):
        return self.case.targets

    @property
    def provenance(self) -> OperatorCaseProvenance:
        provenance = self.case.provenance
        if provenance is None:
            raise RuntimeError("Mechanics operator case provenance is unavailable.")
        return provenance


class MechanicsCaseBuilder:
    """Build every declared physical case or fail without support renormalization."""

    def __init__(
        self,
        distribution: MechanicsParameterDistribution,
        geometry_factory: Callable,
        case_factory: Callable,
        /,
        *,
        reduction: MechanicsCaseReduction,
        mechanics_problem_id: str,
        material_id: str,
        load_id: str,
        boundary_condition_id: str,
        spatial_realization_id: str,
        validity: Callable | None = None,
        split_fingerprint: str | None = None,
    ):
        if not isinstance(distribution, MechanicsParameterDistribution):
            raise TypeError("distribution must be a MechanicsParameterDistribution.")
        if not callable(geometry_factory) or not callable(case_factory):
            raise TypeError("Mechanics geometry and case factories must be callable.")
        if not isinstance(reduction, MechanicsCaseReduction):
            raise TypeError("reduction must be a MechanicsCaseReduction.")
        if validity is not None and not callable(validity):
            raise TypeError("validity must be callable or None.")
        identities = tuple(
            str(value)
            for value in (
                mechanics_problem_id,
                material_id,
                load_id,
                boundary_condition_id,
                spatial_realization_id,
            )
        )
        if any(not value for value in identities):
            raise ValueError("Mechanics physical contract IDs must be non-empty.")
        if split_fingerprint is not None and not str(split_fingerprint):
            raise ValueError("split_fingerprint must be non-empty when provided.")
        self.distribution = distribution
        self.geometry_factory = geometry_factory
        self.case_factory = case_factory
        self.reduction = reduction
        self.mechanics_problem_id = identities[0]
        self.material_id = identities[1]
        self.load_id = identities[2]
        self.boundary_condition_id = identities[3]
        self.spatial_realization_id = identities[4]
        self.validity = validity
        self.split_fingerprint = (
            None if split_fingerprint is None else str(split_fingerprint)
        )
        if reduction.kind == "mean" and distribution.weight_kind != "equal":
            raise ValueError(
                "Plain mean mechanics reduction requires an equal-mass parameter design."
            )

    def _realization(
        self,
        case: int | str | MechanicsParameterRealization,
        /,
    ) -> MechanicsParameterRealization:
        if isinstance(case, MechanicsParameterRealization):
            if case.spec.spec_fingerprint != self.distribution.spec.spec_fingerprint:
                raise ValueError(
                    "Mechanics case realization uses a different parameter spec."
                )
            declared = self.distribution.realization(case.case_id)
            if case.realization_fingerprint != declared.realization_fingerprint:
                raise ValueError(
                    "Mechanics case realization differs from the declared distribution entry."
                )
            return declared
        if isinstance(case, str):
            return self.distribution.realization(case)
        index = int(case)
        if index < 0 or index >= len(self.distribution.realizations):
            raise IndexError("Mechanics parameter case index is out of range.")
        return self.distribution.realizations[index]

    def build(
        self,
        case: int | str | MechanicsParameterRealization,
        /,
    ) -> MechanicsOperatorCase:
        realization = self._realization(case)
        geometry = self.geometry_factory(realization)
        if not isinstance(geometry, MechanicsGeometryMap):
            raise TypeError("geometry_factory must return a MechanicsGeometryMap.")
        canonical = self.case_factory(realization, geometry)
        if not isinstance(canonical, CanonicalOperatorCase):
            raise TypeError("case_factory must return phydrax.nn.operator.OperatorCase.")
        if self.validity is not None:
            valid = jnp.asarray(
                self.validity(realization, geometry, canonical), dtype=bool
            )
            if valid.shape != () or not bool(valid):
                raise ValueError(
                    f"Mechanics physical case {realization.case_id!r} is invalid; "
                    "invalid cases cannot be omitted or renormalized."
                )
        _require_finite_case(canonical, realization.case_id)
        identities = {
            "mechanics_problem": self.mechanics_problem_id,
            "parameter_spec": self.distribution.spec.spec_id,
            "parameter_spec_fingerprint": self.distribution.spec.spec_fingerprint,
            "parameter_distribution": self.distribution.distribution_id,
            "parameter_distribution_fingerprint": (
                self.distribution.distribution_fingerprint
            ),
            "parameter_realization": realization.realization_id,
            "parameter_realization_fingerprint": realization.realization_fingerprint,
            "parameter_stratum": realization.stratum_id,
            "geometry": geometry.geometry_id,
            "geometry_fingerprint": geometry.geometry_fingerprint,
            "reference_domain": geometry.reference_domain_id,
            "physical_domain": geometry.physical_domain_id,
            "material": self.material_id,
            "load": self.load_id,
            "boundary_conditions": self.boundary_condition_id,
            "spatial_realization": self.spatial_realization_id,
            "risk_reduction": self.reduction.reduction_id,
        }
        if self.split_fingerprint is not None:
            identities["parameter_split"] = self.split_fingerprint
        order: Mapping[str, float] = {}
        if canonical.provenance is not None:
            for name, value in canonical.provenance.identities.items():
                if name in identities and identities[name] != value:
                    raise ValueError(
                        f"Operator case provenance conflicts on identity {name!r}."
                    )
                identities[name] = value
            order = canonical.provenance.order
            if canonical.provenance.case_id != realization.case_id:
                raise ValueError("Operator and mechanics parameter case IDs must match.")
        provenance = OperatorCaseProvenance(
            realization.case_id,
            identities=identities,
            order=order,
        )
        bound = CanonicalOperatorCase(
            canonical.batch,
            canonical.targets,
            provenance=provenance,
        )
        position = next(
            index
            for index, item in enumerate(self.distribution.realizations)
            if item.case_id == realization.case_id
        )
        case_fingerprint = canonical_fingerprint(
            {
                "kind": "mechanics-operator-case",
                "case_id": realization.case_id,
                "identities": identities,
                "inputs": {
                    name: {
                        "support": samples.support_id,
                        "measure": samples.measure_id,
                        "geometry": samples.geometry_fingerprint(),
                    }
                    for name, samples in bound.batch.inputs.items()
                },
                "queries": {
                    name: {
                        "support": samples.support_id,
                        "measure": samples.measure_id,
                        "geometry": samples.geometry_fingerprint(),
                    }
                    for name, samples in bound.batch.queries.items()
                },
            }
        )
        return MechanicsOperatorCase(
            case=bound,
            realization=realization,
            geometry=geometry,
            parameter_weight=float(self.distribution.normalized_weights[position]),
            weight_kind=self.distribution.weight_kind,
            identities=frozendict(identities),
            case_fingerprint=case_fingerprint,
        )

    def build_all(self, /) -> tuple[MechanicsOperatorCase, ...]:
        return tuple(
            self.build(realization) for realization in self.distribution.realizations
        )

    def stacked_batch(
        self,
        cases: Sequence[MechanicsOperatorCase] | None = None,
        /,
    ) -> OperatorBatch:
        """Stack the complete declared support on one explicit parameter axis."""
        resolved = self.build_all() if cases is None else tuple(cases)
        if not resolved:
            raise ValueError("Mechanics case batches cannot be empty.")
        expected = tuple(item.case_id for item in self.distribution.realizations)
        actual = tuple(item.realization.case_id for item in resolved)
        if actual != expected:
            raise ValueError(
                "Mechanics cases must retain complete declared distribution order."
            )
        return stack_operator_batches(
            tuple(item.batch for item in resolved),
            case_axis="parameter",
        )

    @property
    def builder_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "mechanics-operator-case-builder",
                "distribution": self.distribution.distribution_id,
                "distribution_fingerprint": (self.distribution.distribution_fingerprint),
                "risk": self.reduction.reduction_id,
                "problem": self.mechanics_problem_id,
                "material": self.material_id,
                "load": self.load_id,
                "boundary_conditions": self.boundary_condition_id,
                "spatial_realization": self.spatial_realization_id,
                "split": self.split_fingerprint,
            }
        )

    @property
    def metadata(self) -> frozendict[str, str]:
        """Fingerprint bindings for existing checkpoint/artifact metadata maps."""
        values = {
            "mechanics_case_builder_fingerprint": self.builder_id,
            "mechanics_parameter_spec_fingerprint": (
                self.distribution.spec.spec_fingerprint
            ),
            "mechanics_parameter_distribution_fingerprint": (
                self.distribution.distribution_fingerprint
            ),
            "mechanics_risk_fingerprint": self.reduction.reduction_id,
        }
        if self.split_fingerprint is not None:
            values["mechanics_parameter_split_fingerprint"] = self.split_fingerprint
        return frozendict(values)


def _require_finite_case(case: CanonicalOperatorCase, case_id: str, /) -> None:
    arrays: list[tuple[str, Any]] = []
    for category, mapping in (
        ("input", case.batch.inputs),
        ("query", case.batch.queries),
    ):
        for name, samples in mapping.items():
            arrays.extend(
                (
                    (f"{category}:{name}:values", samples.values),
                    (f"{category}:{name}:coordinates", samples.coordinates),
                    (f"{category}:{name}:quadrature", samples.quadrature_weights),
                )
            )
            arrays.extend(
                (f"{category}:{name}:axis:{axis.name}", axis.nodes)
                for axis in samples.axes
            )
    arrays.extend(
        (f"target:{name}", field.values) for name, field in case.targets.fields.items()
    )
    for label, value in arrays:
        if value is None:
            continue
        array = np.asarray(value)
        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"Mechanics case {case_id!r} has nonfinite {label}; invalid cases "
                "cannot be omitted or renormalized."
            )


__all__ = [
    "MechanicsCaseBuilder",
    "MechanicsGeometryMap",
    "MechanicsOperatorCase",
    "OperatorTrialFieldAdapter",
]
