#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.spectral._operators import (
    PreparedSpectralOperator,
    spectral_derivative_operator,
)
from ..discretization.spectral._space import TensorSpectralDiscretization
from ..discretization.spectral._transfer import (
    prepare_spectral_modal_transfer,
    PreparedSpectralModalTransfer,
)
from ..equations._les_closures import ResolvedLESFilter
from ._analysis import (
    ClosureAnalysisDAG,
    ClosureAnalysisNode,
    ClosureField,
    ClosureTarget,
)


LESStressConvention = Literal["full", "deviatoric"]


class LESAnalysisReference(StrictModule, NonTrainableState):
    """Content-addressed binding of one LES target to its complete offline lineage."""

    reference_manifest_id: str = eqx.field(static=True)
    source_discretization_id: str = eqx.field(static=True)
    resolved_discretization_id: str = eqx.field(static=True)
    filter_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    analysis_dag_id: str = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        reference_manifest_id: str,
        source_discretization_id: str,
        resolved_discretization_id: str,
        filter_id: str,
        target_id: str,
        analysis_dag_id: str,
    ):
        values = tuple(
            str(value).strip()
            for value in (
                reference_manifest_id,
                source_discretization_id,
                resolved_discretization_id,
                filter_id,
                target_id,
                analysis_dag_id,
            )
        )
        if any(not value for value in values):
            raise ValueError("LES analysis reference identities must be non-empty.")
        (
            manifest,
            source,
            resolved,
            filter_,
            target,
            dag,
        ) = values
        self.reference_manifest_id = manifest
        self.source_discretization_id = source
        self.resolved_discretization_id = resolved
        self.filter_id = filter_
        self.target_id = target
        self.analysis_dag_id = dag
        self.reference_id = canonical_fingerprint(
            {
                "kind": "les-analysis-reference",
                "reference_manifest": manifest,
                "source_discretization": source,
                "resolved_discretization": resolved,
                "resolved_filter": filter_,
                "target": target,
                "analysis_dag": dag,
            }
        )


class PeriodicLESAnalysisContext(StrictModule, NonTrainableState):
    """Immutable exact Fourier source-to-resolved analysis context.

    Physical source fields are projected into the source modal space, restricted by
    the prepared modal transfer, and reconstructed on the resolved grid. The
    attached ``ResolvedLESFilter`` is the runtime semantic identity; this context
    does not mint a second filter identity and never treats dealiasing or cell
    averaging as that projection.
    """

    source: TensorSpectralDiscretization
    resolved: TensorSpectralDiscretization
    resolved_filter: ResolvedLESFilter
    modal_transfer: PreparedSpectralModalTransfer
    reference_manifest_id: str = eqx.field(static=True)
    context_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: TensorSpectralDiscretization,
        resolved: TensorSpectralDiscretization,
        resolved_filter: ResolvedLESFilter,
        modal_transfer: PreparedSpectralModalTransfer,
        /,
        *,
        reference_manifest_id: str,
    ):
        if not isinstance(source, TensorSpectralDiscretization) or not isinstance(
            resolved, TensorSpectralDiscretization
        ):
            raise TypeError(
                "source and resolved must be TensorSpectralDiscretization values."
            )
        if not isinstance(resolved_filter, ResolvedLESFilter):
            raise TypeError("resolved_filter must be a ResolvedLESFilter.")
        if not isinstance(modal_transfer, PreparedSpectralModalTransfer):
            raise TypeError("modal_transfer must be a PreparedSpectralModalTransfer.")
        manifest = str(reference_manifest_id).strip()
        if not manifest:
            raise ValueError("reference_manifest_id must be non-empty.")
        if len(source.axes) != 3 or len(resolved.axes) != 3:
            raise ValueError("Periodic LES analysis requires three spectral axes.")
        if any(axis.family != "fourier" or not axis.periodic for axis in source.axes):
            raise ValueError("The LES source must be a three-dimensional Fourier space.")
        if any(axis.family != "fourier" or not axis.periodic for axis in resolved.axes):
            raise ValueError(
                "The LES resolved space must be a three-dimensional Fourier space."
            )
        source_axes = tuple(source.plan.axis_names)
        resolved_axes = tuple(resolved.plan.axis_names)
        if source_axes != resolved_axes or resolved_axes != resolved_filter.axis_names:
            raise ValueError(
                "Source, resolved, and resolved-filter axis names must match exactly."
            )
        if resolved_filter.family != "sharp-fourier-projection":
            raise ValueError(
                "Exact periodic LES analysis requires a sharp Fourier projection filter."
            )
        if any(
            source_count < resolved_count
            for source_count, resolved_count in zip(
                source.modal_shape, resolved.modal_shape, strict=True
            )
        ):
            raise ValueError(
                "The source resolution must retain every requested resolved mode."
            )
        transfer_source = modal_transfer.plan.source
        transfer_target = modal_transfer.plan.target
        if (
            not isinstance(transfer_source, TensorSpectralDiscretization)
            or not isinstance(transfer_target, TensorSpectralDiscretization)
            or transfer_source.prepared_id != source.prepared_id
            or transfer_target.prepared_id != resolved.prepared_id
            or modal_transfer.report.source_space_id != source.modal_space.field_space_id
            or modal_transfer.report.target_space_id
            != resolved.modal_space.field_space_id
            or modal_transfer.report.axis_actions
            != ("fourier-mode-map", "fourier-mode-map", "fourier-mode-map")
        ):
            raise ValueError(
                "modal_transfer must be the matching source-to-resolved Fourier transfer."
            )
        self.source = source
        self.resolved = resolved
        self.resolved_filter = resolved_filter
        self.modal_transfer = modal_transfer
        self.reference_manifest_id = manifest
        self.context_id = canonical_fingerprint(
            {
                "kind": "periodic-les-analysis-context",
                "reference_manifest": manifest,
                "source_discretization": source.prepared_id,
                "resolved_discretization": resolved.prepared_id,
                "resolved_filter": resolved_filter.filter_id,
                "modal_transfer": modal_transfer.prepared_id,
            }
        )

    @property
    def filter_id(self) -> str:
        """Return the attached core filter identity without introducing an alias."""

        return self.resolved_filter.filter_id

    def filter_modal(self, coefficients: ArrayLike, /) -> Array:
        """Restrict source modal coefficients to the exact resolved modal layout."""

        return self.modal_transfer(coefficients)

    def filter_field(self, values: ArrayLike, /) -> Array:
        """Apply the exact source-to-resolved projection to a physical field."""

        source_values = self._validate_source_values(values, "Filtered source field")
        source_coefficients = self.source.project(source_values)
        return self.resolved.reconstruct(self.filter_modal(source_coefficients))

    def filter_product(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Form a source-grid product and project that product to resolved modes."""

        left_ = self._validate_source_values(left, "Left product field")
        right_ = self._validate_source_values(right, "Right product field")
        product = left_ * right_
        return self.filter_field(product)

    def filter_closure_field(
        self,
        field: ClosureField,
        /,
        *,
        name: str | None = None,
    ) -> ClosureField:
        """Filter a lineage-bearing closure field onto the resolved physical grid."""

        _validate_source_field(field, self, "field")
        output_name = field.name if name is None else str(name).strip()
        if not output_name:
            raise ValueError("Filtered closure-field name must be non-empty.")
        return ClosureField(
            self.filter_field(field.values),
            name=output_name,
            units=field.units,
            schema_id=field.schema_id,
            lineage_ids=(*field.lineage_ids, field.field_id, self.context_id),
        )

    def analysis_dag(self, targets: Sequence[ClosureTarget], /) -> ClosureAnalysisDAG:
        """Build a provenance-rooted DAG from topologically ordered LES targets."""

        targets_ = tuple(targets)
        if not targets_:
            raise ValueError(
                "LES analysis DAG construction requires at least one target."
            )
        for target in targets_:
            _validate_context_target(target, self)
        nodes = tuple(target.node for target in targets_)
        node_ids = {node.node_id for node in nodes}
        external = list(
            dict.fromkeys(
                (
                    self.reference_manifest_id,
                    self.source.prepared_id,
                    self.resolved.prepared_id,
                    self.resolved_filter.filter_id,
                    self.modal_transfer.prepared_id,
                )
            )
        )
        for node in nodes:
            for input_id in node.input_ids:
                if input_id not in node_ids and input_id not in external:
                    external.append(input_id)
        return ClosureAnalysisDAG(tuple(external), nodes)

    def bind_target(
        self, target: ClosureTarget, dag: ClosureAnalysisDAG, /
    ) -> LESAnalysisReference:
        """Bind one context-owned target and containing DAG to reference provenance."""

        _validate_context_target(target, self)
        if not isinstance(dag, ClosureAnalysisDAG):
            raise TypeError("dag must be a ClosureAnalysisDAG.")
        required_external = {
            self.reference_manifest_id,
            self.source.prepared_id,
            self.resolved.prepared_id,
            self.resolved_filter.filter_id,
            self.modal_transfer.prepared_id,
        }
        if not required_external.issubset(
            dag.external_input_ids
        ) or target.node.node_id not in {node.node_id for node in dag.nodes}:
            raise ValueError(
                "The analysis DAG must contain the target and complete LES context roots."
            )
        return LESAnalysisReference(
            reference_manifest_id=self.reference_manifest_id,
            source_discretization_id=self.source.prepared_id,
            resolved_discretization_id=self.resolved.prepared_id,
            filter_id=self.resolved_filter.filter_id,
            target_id=target.target_id,
            analysis_dag_id=dag.dag_id,
        )

    def _validate_source_values(self, values: ArrayLike, owner: str, /) -> Array:
        array = jnp.asarray(values)
        rank = len(self.source.physical_shape)
        if array.ndim < rank or tuple(array.shape[:rank]) != self.source.physical_shape:
            raise ValueError(
                f"{owner} must begin with source physical shape "
                f"{self.source.physical_shape}; got {array.shape}."
            )
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            raise TypeError(f"{owner} must use an inexact dtype.")
        return array


def prepare_periodic_les_analysis(
    source: TensorSpectralDiscretization,
    resolved: TensorSpectralDiscretization,
    resolved_filter: ResolvedLESFilter,
    /,
    *,
    reference_manifest_id: str,
) -> PeriodicLESAnalysisContext:
    """Prepare the unique exact modal restriction used by periodic LES analysis."""

    transfer = prepare_spectral_modal_transfer(source, resolved)
    return PeriodicLESAnalysisContext(
        source,
        resolved,
        resolved_filter,
        transfer,
        reference_manifest_id=reference_manifest_id,
    )


def les_reynolds_stress_target(
    velocity: ClosureField,
    context: PeriodicLESAnalysisContext,
    /,
    *,
    convention: LESStressConvention = "full",
) -> ClosureTarget:
    """Return τᵢⱼ = F(uᵢuⱼ) − F(uᵢ)F(uⱼ) on the resolved grid."""

    _validate_velocity_field(velocity, context)
    convention_ = str(convention).strip()
    if convention_ not in ("full", "deviatoric"):
        raise ValueError("LES stress convention must be 'full' or 'deviatoric'.")
    values = velocity.values
    mean = context.filter_field(values)
    outer = values[..., :, None] * values[..., None, :]
    stress = context.filter_field(outer) - mean[..., :, None] * mean[..., None, :]
    if convention_ == "deviatoric":
        trace = jnp.trace(stress, axis1=-2, axis2=-1)
        identity = jnp.eye(3, dtype=stress.dtype)
        stress = stress - (trace / 3.0)[..., None, None] * identity
    node = ClosureAnalysisNode(
        "periodic_les_reynolds_stress",
        (velocity.field_id,),
        output_name="reynolds_sgs_stress",
        output_units=f"({velocity.units})^2",
        parameters=_context_parameters(
            context,
            (("stress_convention", convention_),),
        ),
    )
    return ClosureTarget(
        stress,
        node,
        target_kind="sgs_stress",
        schema_id=velocity.schema_id,
    )


def les_stress_divergence_target(
    stress: ClosureTarget,
    context: PeriodicLESAnalysisContext,
    derivatives: Sequence[PreparedSpectralOperator],
    /,
) -> ClosureTarget:
    """Return ∂ⱼτᵢⱼ using the supplied exact resolved-space derivatives."""

    _validate_stress_target(stress, context)
    derivatives_ = _validated_derivatives(derivatives, context)
    modal_stress = context.resolved.project(stress.values)
    divergence = jnp.zeros(
        context.resolved.physical_shape + (3,), dtype=stress.values.dtype
    )
    for axis, derivative in enumerate(derivatives_):
        component_modal = modal_stress[..., :, axis]
        divergence = divergence + context.resolved.reconstruct(
            derivative(component_modal)
        )
    node = ClosureAnalysisNode(
        "periodic_les_stress_divergence",
        (stress.node.node_id,),
        output_name="sgs_stress_divergence",
        output_units=f"({stress.units})/length",
        parameters=_context_parameters(
            context,
            tuple(
                (f"derivative_{axis}_id", derivative.operator_id)
                for axis, derivative in enumerate(derivatives_)
            ),
        ),
    )
    return ClosureTarget(
        divergence,
        node,
        target_kind="sgs_stress_divergence",
        schema_id=stress.schema_id,
    )


def les_energy_transfer_target(
    stress: ClosureTarget,
    velocity: ClosureField,
    context: PeriodicLESAnalysisContext,
    derivatives: Sequence[PreparedSpectralOperator],
    /,
) -> ClosureTarget:
    """Return Π = −τᵢⱼS̄ᵢⱼ, positive for forward SGS energy transfer.

    The sign convention is fixed, but exact stresses are not clipped: physically
    admissible local backscatter remains negative.
    """

    _validate_stress_target(stress, context)
    _validate_velocity_field(velocity, context)
    if stress.schema_id != velocity.schema_id:
        raise ValueError("Stress and velocity must share one schema identity.")
    derivatives_ = _validated_derivatives(derivatives, context)
    velocity_modal = context.filter_modal(context.source.project(velocity.values))
    gradient = _velocity_gradient(velocity_modal, context, derivatives_)
    strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
    transfer = -ein.contract("...ij,...ij->...", stress.values, strain, backend="jax")
    node = ClosureAnalysisNode(
        "periodic_les_energy_transfer",
        (stress.node.node_id, velocity.field_id),
        output_name="positive_sgs_energy_transfer",
        output_units=f"({stress.units})/time",
        parameters=_context_parameters(
            context,
            (
                ("transfer_sign", "positive-forward:-tau_ij*S_ij"),
                *tuple(
                    (f"derivative_{axis}_id", derivative.operator_id)
                    for axis, derivative in enumerate(derivatives_)
                ),
            ),
        ),
    )
    return ClosureTarget(
        transfer,
        node,
        target_kind="sgs_transfer",
        schema_id=velocity.schema_id,
    )


def les_scalar_flux_target(
    velocity: ClosureField,
    scalar: ClosureField,
    context: PeriodicLESAnalysisContext,
    /,
    *,
    name: str,
) -> ClosureTarget:
    """Return the named Reynolds SGS scalar flux F(uφ) − F(u)F(φ)."""

    _validate_velocity_field(velocity, context)
    _validate_source_field(scalar, context, "scalar")
    if velocity.schema_id != scalar.schema_id:
        raise ValueError("Velocity and scalar must share one schema identity.")
    scalar_values = _scalar_values(scalar, context)
    name_ = str(name).strip()
    if not name_:
        raise ValueError("Generic LES scalar-flux targets require a non-empty name.")
    mean_velocity = context.filter_field(velocity.values)
    mean_scalar = context.filter_field(scalar_values)
    product = velocity.values * scalar_values[..., None]
    flux = context.filter_field(product) - mean_velocity * mean_scalar[..., None]
    node = ClosureAnalysisNode(
        "periodic_les_scalar_flux",
        (velocity.field_id, scalar.field_id),
        output_name=f"{name_}_sgs_flux",
        output_units=f"({velocity.units})*({scalar.units})",
        parameters=_context_parameters(context, (("scalar_name", name_),)),
    )
    return ClosureTarget(
        flux,
        node,
        target_kind="scalar_flux",
        schema_id=velocity.schema_id,
    )


def _velocity_gradient(
    velocity_modal: Array,
    context: PeriodicLESAnalysisContext,
    derivatives: tuple[PreparedSpectralOperator, ...],
    /,
) -> Array:
    columns = tuple(
        context.resolved.reconstruct(derivative(velocity_modal))
        for derivative in derivatives
    )
    return jnp.stack(columns, axis=-1)


def _validated_derivatives(
    derivatives: Sequence[PreparedSpectralOperator],
    context: PeriodicLESAnalysisContext,
    /,
) -> tuple[PreparedSpectralOperator, ...]:
    values = tuple(derivatives)
    if len(values) != 3 or not all(
        isinstance(value, PreparedSpectralOperator) for value in values
    ):
        raise TypeError("derivatives must contain three PreparedSpectralOperator values.")
    modal_space_id = context.resolved.modal_space.field_space_id
    for axis, derivative in enumerate(values):
        expected_id = spectral_derivative_operator(context.resolved, axis).operator_id
        if (
            derivative.source_space.field_space_id != modal_space_id
            or derivative.target_space.field_space_id != modal_space_id
            or derivative.axes != (axis,)
            or derivative.axis_actions != ("derivative:1",)
            or derivative.classification != "spectral-derivative"
            or not derivative.exact
            or derivative.operator_id != expected_id
        ):
            raise ValueError(
                "Each derivative must be the matching exact first derivative for "
                "its resolved Fourier axis."
            )
    return values


def _validate_velocity_field(
    velocity: ClosureField, context: PeriodicLESAnalysisContext, /
) -> None:
    _validate_source_field(velocity, context, "velocity")
    if velocity.values.ndim != 4 or velocity.values.shape[-1] != 3:
        raise ValueError(
            "Periodic LES velocity must have source spatial axes followed by three components."
        )


def _validate_source_field(
    field: ClosureField,
    context: PeriodicLESAnalysisContext,
    owner: str,
    /,
) -> None:
    if not isinstance(field, ClosureField):
        raise TypeError(f"{owner} must be a ClosureField.")
    context._validate_source_values(field.values, f"LES {owner}")


def _scalar_values(scalar: ClosureField, context: PeriodicLESAnalysisContext, /) -> Array:
    rank = len(context.source.physical_shape)
    if scalar.values.ndim == rank:
        return scalar.values
    if scalar.values.ndim == rank + 1 and scalar.values.shape[-1] == 1:
        return scalar.values[..., 0]
    raise ValueError(
        "Periodic LES scalar fields must have no payload or one scalar payload."
    )


def _validate_stress_target(
    stress: ClosureTarget, context: PeriodicLESAnalysisContext, /
) -> None:
    if not isinstance(stress, ClosureTarget) or stress.target_kind != "sgs_stress":
        raise TypeError("stress must be an SGS stress ClosureTarget.")
    expected_shape = context.resolved.physical_shape + (3, 3)
    if stress.values.shape != expected_shape:
        raise ValueError(
            f"LES stress must have resolved physical shape {expected_shape}."
        )
    _validate_context_target(stress, context)


def _validate_context_target(
    target: ClosureTarget, context: PeriodicLESAnalysisContext, /
) -> None:
    if not isinstance(target, ClosureTarget):
        raise TypeError("target must be a ClosureTarget.")
    if dict(target.node.parameters).get("les_context_id") != context.context_id:
        raise ValueError("Closure target does not belong to this LES analysis context.")


def _context_parameters(
    context: PeriodicLESAnalysisContext,
    extra: tuple[tuple[str, str], ...] = (),
    /,
) -> tuple[tuple[str, str], ...]:
    return (
        ("les_context_id", context.context_id),
        ("reference_manifest_id", context.reference_manifest_id),
        ("source_discretization_id", context.source.prepared_id),
        ("resolved_discretization_id", context.resolved.prepared_id),
        ("filter_id", context.resolved_filter.filter_id),
        ("modal_transfer_id", context.modal_transfer.prepared_id),
        *extra,
    )


__all__ = [
    "LESAnalysisReference",
    "LESStressConvention",
    "PeriodicLESAnalysisContext",
    "les_energy_transfer_target",
    "les_reynolds_stress_target",
    "les_scalar_flux_target",
    "les_stress_divergence_target",
    "prepare_periodic_les_analysis",
]
