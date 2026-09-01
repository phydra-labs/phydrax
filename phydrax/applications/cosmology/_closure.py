#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class CosmologyPhysicalState(StrictModule):
    """Canonical ordered physical parameters without outputs or numerical policy."""

    values: Array
    names: tuple[str, ...] = eqx.field(static=True)
    categorical_ids: tuple[str, ...] = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    state_form_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        names: tuple[str, ...],
        scale_id: str,
        /,
        *,
        categorical_ids: tuple[str, ...] = (),
    ):
        names_ = tuple(str(name).strip() for name in names)
        values_ = jnp.asarray(values).reshape((-1,))
        scale_id_ = str(scale_id).strip()
        categories = tuple(str(value).strip() for value in categorical_ids)
        if (
            not names_
            or len(names_) != values_.size
            or len(set(names_)) != len(names_)
            or any(not name for name in names_)
            or not scale_id_
            or any(not value for value in categories)
        ):
            raise ValueError(
                "Canonical cosmology physical-state coordinates are invalid."
            )
        values_ = eqx.error_if(
            values_,
            jnp.any(~jnp.isfinite(values_)),
            "Cosmology physical parameters must be finite.",
        )
        self.values = values_
        self.names = names_
        self.categorical_ids = categories
        self.scale_id = scale_id_
        self.state_form_id = canonical_fingerprint(
            {
                "kind": "cosmology-physical-state-form",
                "names": list(names_),
                "scale_id": scale_id_,
                "categorical_ids": list(categories),
            }
        )

    def content_id(self, /) -> str:
        return canonical_fingerprint(
            {
                "kind": "cosmology-physical-state",
                "form": self.state_form_id,
                "values": array_tree_fingerprint(self.values),
            }
        )


class PhysicalDependencyProjection(StrictModule, NonTrainableState):
    """Static named subset of a canonical physical state."""

    names: tuple[str, ...] = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)

    def __init__(self, names: tuple[str, ...], /):
        names_ = tuple(str(name).strip() for name in names)
        if (
            not names_
            or len(set(names_)) != len(names_)
            or any(not name for name in names_)
        ):
            raise ValueError("Physical dependency names must be non-empty and unique.")
        self.names = names_
        self.projection_id = canonical_fingerprint(
            {"kind": "physical-dependency-projection", "names": list(names_)}
        )

    def project(self, state: CosmologyPhysicalState, /) -> CosmologyRealizationSignature:
        if not isinstance(state, CosmologyPhysicalState):
            raise TypeError("state must be CosmologyPhysicalState.")
        missing = tuple(name for name in self.names if name not in state.names)
        if missing:
            raise ValueError(f"Physical state is missing projected parameters {missing}.")
        indices = tuple(state.names.index(name) for name in self.names)
        return CosmologyRealizationSignature(
            state.values[jnp.asarray(indices)],
            self.names,
            state.state_form_id,
            state.scale_id,
            self.projection_id,
        )


class CosmologyRealizationSignature(StrictModule):
    """Projected dynamic physical realization used by one product family."""

    parameter_values: Array
    parameter_names: tuple[str, ...] = eqx.field(static=True)
    physical_state_form_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameter_values: ArrayLike,
        parameter_names: tuple[str, ...],
        physical_state_form_id: str,
        scale_id: str,
        projection_id: str,
        /,
    ):
        names = tuple(str(name).strip() for name in parameter_names)
        values = jnp.asarray(parameter_values).reshape((-1,))
        identities = tuple(
            str(value).strip()
            for value in (physical_state_form_id, scale_id, projection_id)
        )
        if (
            not names
            or len(names) != values.size
            or len(set(names)) != len(names)
            or any(not name for name in names)
            or any(not value for value in identities)
        ):
            raise ValueError("Projected realization contract is invalid.")
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Projected physical parameters must be finite.",
        )
        self.parameter_values = values
        self.parameter_names = names
        self.physical_state_form_id = identities[0]
        self.scale_id = identities[1]
        self.projection_id = identities[2]

    def require_compatible(
        self, other: CosmologyRealizationSignature, token: ArrayLike, /
    ) -> Array:
        if not isinstance(other, CosmologyRealizationSignature):
            raise TypeError("other must be CosmologyRealizationSignature.")
        shared = tuple(
            name for name in self.parameter_names if name in other.parameter_names
        )
        if self.scale_id != other.scale_id or not shared:
            raise ValueError(
                "Projected realization contracts have no compatible overlap."
            )
        left = jnp.stack(
            tuple(
                self.parameter_values[self.parameter_names.index(name)] for name in shared
            )
        )
        right = jnp.stack(
            tuple(
                other.parameter_values[other.parameter_names.index(name)]
                for name in shared
            )
        )
        return eqx.error_if(
            jnp.asarray(token),
            jnp.any(left != right),
            "Cosmology products come from different physical realizations.",
        )

    def content_id(self, /) -> str:
        return canonical_fingerprint(
            {
                "kind": "projected-cosmology-realization",
                "physical_state_form_id": self.physical_state_form_id,
                "projection_id": self.projection_id,
                "scale_id": self.scale_id,
                "names": list(self.parameter_names),
                "values": array_tree_fingerprint(self.parameter_values),
            }
        )


class DifferentiationContract(StrictModule, NonTrainableState):
    """Independent derivative capabilities for one scientific product."""

    upstream_physical_parameters: bool = eqx.field(static=True)
    stored_values: bool = eqx.field(static=True)
    query_coordinates: bool = eqx.field(static=True)
    local_parameters: bool = eqx.field(static=True)
    stochastic_realization: bool = eqx.field(static=True)
    higher_order: bool = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        upstream_physical_parameters: bool,
        stored_values: bool,
        query_coordinates: bool,
        local_parameters: bool,
        stochastic_realization: bool = False,
        higher_order: bool = True,
    ):
        values = tuple(
            bool(value)
            for value in (
                upstream_physical_parameters,
                stored_values,
                query_coordinates,
                local_parameters,
                stochastic_realization,
                higher_order,
            )
        )
        (
            self.upstream_physical_parameters,
            self.stored_values,
            self.query_coordinates,
            self.local_parameters,
            self.stochastic_realization,
            self.higher_order,
        ) = values
        self.contract_id = canonical_fingerprint(
            {"kind": "differentiation-contract", "capabilities": list(values)}
        )

    @classmethod
    def native(cls) -> DifferentiationContract:
        return cls(
            upstream_physical_parameters=True,
            stored_values=True,
            query_coordinates=True,
            local_parameters=True,
        )

    @classmethod
    def coordinate_only(cls) -> DifferentiationContract:
        return cls(
            upstream_physical_parameters=False,
            stored_values=False,
            query_coordinates=True,
            local_parameters=False,
        )

    @classmethod
    def constant(cls) -> DifferentiationContract:
        return cls(
            upstream_physical_parameters=False,
            stored_values=False,
            query_coordinates=False,
            local_parameters=False,
            higher_order=False,
        )

    @classmethod
    def from_label(cls, label: str, /) -> DifferentiationContract:
        value = str(label).strip()
        if value == "native-parameter":
            return cls.native()
        if value == "coordinate-only":
            return cls.coordinate_only()
        if value == "constant":
            return cls.constant()
        raise ValueError("Unknown differentiation contract label.")

    def meet(self, *others: DifferentiationContract) -> DifferentiationContract:
        contracts = (self, *others)
        return DifferentiationContract(
            upstream_physical_parameters=all(
                value.upstream_physical_parameters for value in contracts
            ),
            stored_values=all(value.stored_values for value in contracts),
            query_coordinates=all(value.query_coordinates for value in contracts),
            local_parameters=any(value.local_parameters for value in contracts),
            stochastic_realization=any(
                value.stochastic_realization for value in contracts
            ),
            higher_order=all(value.higher_order for value in contracts),
        )


class ScientificArtifactEnvelope(StrictModule, NonTrainableState):
    """Content-addressed producer, license, lineage, and run-status evidence."""

    artifact_kind: str = eqx.field(static=True)
    content_digest: str = eqx.field(static=True)
    producer: str = eqx.field(static=True)
    producer_version: str = eqx.field(static=True)
    build_id: str = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    parent_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    resource_id: str = eqx.field(static=True)
    status: Literal["complete", "failed"] = eqx.field(static=True)
    failure_reason: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        artifact_kind: str,
        content_digest: str,
        producer: str,
        producer_version: str,
        build_id: str,
        license_id: str,
        resource_id: str,
        status: Literal["complete", "failed"],
        failure_reason: str = "none",
        parent_artifact_ids: tuple[str, ...] = (),
    ):
        values = tuple(
            str(value).strip()
            for value in (
                artifact_kind,
                content_digest,
                producer,
                producer_version,
                build_id,
                license_id,
                resource_id,
                failure_reason,
            )
        )
        parents = tuple(str(value).strip() for value in parent_artifact_ids)
        if any(not value for value in values) or any(not value for value in parents):
            raise ValueError("Scientific artifact fields must be non-empty.")
        if status not in ("complete", "failed"):
            raise ValueError("Scientific artifact status must be complete or failed.")
        if status == "complete" and values[7] != "none":
            raise ValueError("Complete artifacts cannot carry a failure reason.")
        (
            self.artifact_kind,
            self.content_digest,
            self.producer,
            self.producer_version,
            self.build_id,
            self.license_id,
            self.resource_id,
            self.failure_reason,
        ) = values
        self.parent_artifact_ids = parents
        self.status = status
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "scientific-artifact",
                "artifact_kind": values[0],
                "content_digest": values[1],
                "producer": values[2],
                "producer_version": values[3],
                "build_id": values[4],
                "license_id": values[5],
                "resource_id": values[6],
                "failure_reason": values[7],
                "parents": list(parents),
                "status": status,
            }
        )


class CoordinateLayout(StrictModule, NonTrainableState):
    labels: tuple[str, ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, labels: tuple[str, ...], /):
        labels_ = tuple(str(label).strip() for label in labels)
        if (
            not labels_
            or len(set(labels_)) != len(labels_)
            or any(not label for label in labels_)
        ):
            raise ValueError("Coordinate labels must be non-empty and unique.")
        self.labels = labels_
        self.layout_id = canonical_fingerprint(
            {"kind": "coordinate-layout", "labels": list(labels_)}
        )

    @property
    def size(self) -> int:
        return len(self.labels)


class TheoryVector(StrictModule):
    values: Array
    layout: CoordinateLayout
    product_id: str = eqx.field(static=True)

    def __init__(self, values: ArrayLike, layout: CoordinateLayout, product_id: str, /):
        value = jnp.asarray(values)
        if value.shape != (layout.size,):
            raise ValueError("Theory vector must match its coordinate layout.")
        product_id_ = str(product_id).strip()
        if not product_id_:
            raise ValueError("Theory vector product_id must be non-empty.")
        self.values = value
        self.layout = layout
        self.product_id = product_id_


class LinearObservationPlan(StrictModule, NonTrainableState):
    matrix: Array
    source: CoordinateLayout
    target: CoordinateLayout
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        source: CoordinateLayout,
        target: CoordinateLayout,
        /,
    ):
        values = jax.lax.stop_gradient(jnp.asarray(matrix))
        if values.shape != (target.size, source.size):
            raise ValueError("Observation matrix shape must match layouts.")
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Observation matrix must be finite.",
        )
        self.matrix = values
        self.source = source
        self.target = target
        self.plan_id = canonical_fingerprint(
            {
                "kind": "linear-observation-plan",
                "source": source.layout_id,
                "target": target.layout_id,
                "matrix": array_tree_fingerprint(values),
            }
        )

    def apply(self, theory: TheoryVector, /) -> TheoryVector:
        if theory.layout.layout_id != self.source.layout_id:
            raise ValueError("Theory layout does not match observation source.")
        values = contract("oi,i->o", self.matrix, theory.values)
        return TheoryVector(
            values,
            self.target,
            canonical_fingerprint(
                {
                    "kind": "observed-theory-vector",
                    "parent": theory.product_id,
                    "plan": self.plan_id,
                }
            ),
        )


class PrecisionCovarianceAction(StrictModule, NonTrainableState):
    precision: Array
    logdet_covariance: Array
    layout: CoordinateLayout
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        precision: ArrayLike,
        logdet_covariance: ArrayLike,
        layout: CoordinateLayout,
        /,
    ):
        matrix = jax.lax.stop_gradient(jnp.asarray(precision))
        logdet = jax.lax.stop_gradient(jnp.asarray(logdet_covariance, dtype=matrix.dtype))
        if matrix.shape != (layout.size, layout.size) or logdet.shape != ():
            raise ValueError("Precision/covariance determinant shapes are invalid.")
        matrix = eqx.error_if(
            matrix,
            jnp.any(~jnp.isfinite(matrix))
            | ~jnp.isfinite(logdet)
            | jnp.any(jnp.abs(matrix - matrix.T) > 1.0e-10)
            | jnp.any(jnp.diag(matrix) <= 0.0),
            "Precision action must be finite, symmetric, and positive on the diagonal.",
        )
        self.precision = matrix
        self.logdet_covariance = logdet
        self.layout = layout
        self.action_id = canonical_fingerprint(
            {
                "kind": "precision-covariance-action",
                "layout": layout.layout_id,
                "precision": array_tree_fingerprint(matrix),
                "logdet_covariance": array_tree_fingerprint(logdet),
            }
        )

    def quadratic(self, residual: ArrayLike, /) -> Array:
        value = jnp.asarray(residual, dtype=self.precision.dtype)
        if value.shape != (self.layout.size,):
            raise ValueError("Residual must match covariance layout.")
        return contract("i,ij,j->", value, self.precision, value)


class CorrelatedGaussianResult(StrictModule):
    residual: Array
    quadratic: Array
    log_probability: Array
    finite: Array
    successful: Array


class CorrelatedGaussianPlan(StrictModule, NonTrainableState):
    data: Array
    observation: LinearObservationPlan
    covariance: PrecisionCovarianceAction
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        data: ArrayLike,
        observation: LinearObservationPlan,
        covariance: PrecisionCovarianceAction,
        /,
    ):
        values = jax.lax.stop_gradient(jnp.asarray(data))
        if values.shape != (observation.target.size,):
            raise ValueError("Observed data must match observation target layout.")
        if covariance.layout.layout_id != observation.target.layout_id:
            raise ValueError("Covariance and observation target layouts disagree.")
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Observed data must be finite.",
        )
        self.data = values
        self.observation = observation
        self.covariance = covariance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "correlated-gaussian-plan",
                "observation": observation.plan_id,
                "covariance": covariance.action_id,
                "data": array_tree_fingerprint(values),
            }
        )

    def evaluate(self, theory: TheoryVector, /) -> CorrelatedGaussianResult:
        observed = self.observation.apply(theory)
        residual = self.data - observed.values
        quadratic = self.covariance.quadratic(residual)
        size = jnp.asarray(self.data.size, dtype=residual.dtype)
        log_probability = -0.5 * (
            quadratic
            + self.covariance.logdet_covariance
            + size * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=residual.dtype))
        )
        finite = jnp.all(jnp.isfinite(residual)) & jnp.isfinite(log_probability)
        return CorrelatedGaussianResult(
            residual, quadratic, log_probability, finite, finite
        )


__all__ = [
    "CoordinateLayout",
    "CorrelatedGaussianPlan",
    "CorrelatedGaussianResult",
    "CosmologyPhysicalState",
    "CosmologyRealizationSignature",
    "DifferentiationContract",
    "LinearObservationPlan",
    "PhysicalDependencyProjection",
    "PrecisionCovarianceAction",
    "ScientificArtifactEnvelope",
    "TheoryVector",
]
