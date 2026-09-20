#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la
from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...metrix._metric import LorentzianMetric
from ...units import (
    derived_unit,
    KILOGRAM,
    METER,
    RADIAN,
    SECOND,
    UnitDefinition,
)
from ._gr_rays import gr_chart_identity, gr_metric_identity, GRRayResult


class InvariantTransferUnitContract(StrictModule, NonTrainableState):
    """Units for transfer of ``I_nu / nu**3`` along one path parameter.

    Emission values have invariant-Stokes units per path-parameter unit and every
    entry of a propagation matrix has inverse path-parameter units.  Unit
    conversion is deliberately outside a transfer plan, so mixed systems cannot
    enter a transformed solve unnoticed.
    """

    path_parameter_unit: UnitDefinition = eqx.field(static=True)
    invariant_stokes_unit: UnitDefinition = eqx.field(static=True)
    invariant_emission_unit: UnitDefinition = eqx.field(static=True)
    propagation_unit: UnitDefinition = eqx.field(static=True)
    units_id: str = eqx.field(static=True)

    def __init__(
        self,
        path_parameter_unit: UnitDefinition,
        invariant_stokes_unit: UnitDefinition,
        /,
    ):
        if not isinstance(path_parameter_unit, UnitDefinition) or not isinstance(
            invariant_stokes_unit, UnitDefinition
        ):
            raise TypeError("Transfer units must be UnitDefinition values.")
        if (
            path_parameter_unit.reference_system_id
            != invariant_stokes_unit.reference_system_id
        ):
            raise ValueError("Transfer units must share one reference system.")
        self.path_parameter_unit = path_parameter_unit
        self.invariant_stokes_unit = invariant_stokes_unit
        self.invariant_emission_unit = derived_unit(
            f"{invariant_stokes_unit.symbol}/{path_parameter_unit.symbol}",
            ((invariant_stokes_unit, 1), (path_parameter_unit, -1)),
        )
        self.propagation_unit = derived_unit(
            f"1/{path_parameter_unit.symbol}", ((path_parameter_unit, -1),)
        )
        self.units_id = canonical_fingerprint(
            {
                "kind": "invariant-transfer-units",
                "path_parameter": path_parameter_unit.unit_id,
                "invariant_stokes": invariant_stokes_unit.unit_id,
            }
        )

    @classmethod
    def si_affine_length(cls) -> InvariantTransferUnitContract:
        # I_nu has SI dimensions kg s^-1 rad^-2.  Division by nu^3 gives
        # kg s^2 rad^-2.  The label names the invariant rather than a display
        # convention for the underlying spectral intensity.
        invariant_stokes = derived_unit(
            "kg*s^2/rad^2", ((KILOGRAM, 1), (SECOND, 2), (RADIAN, -2))
        )
        return cls(METER, invariant_stokes)


class InvariantScalarTransferEvidence(StrictModule):
    finite: Array
    converged: Array
    physically_valid: Array
    in_support: Array
    minimum_intensity: Array
    qualified: Array
    derivative_valid: Array


class InvariantScalarTransferResult(StrictModule):
    invariant_intensity: Array
    history: Array
    optical_depth: Array
    evidence: InvariantScalarTransferEvidence
    plan_id: str = eqx.field(static=True)


class PolarizedTransferEvidence(StrictModule):
    finite: Array
    converged: Array
    physically_valid: Array
    in_support: Array
    basis_transported: Array
    stokes_cone_margin: Array
    maximum_basis_transport_error: Array
    qualified: Array
    derivative_valid: Array


class PolarizedInvariantTransferResult(StrictModule):
    invariant_stokes: Array
    history: Array
    optical_depth: Array
    evidence: PolarizedTransferEvidence
    plan_id: str = eqx.field(static=True)


def _prefix_mask(mask: np.ndarray, /) -> bool:
    return not bool(np.any(mask[1:] & ~mask[:-1]))


def _stable_slab_step(
    intensity: Array, emission: Array, extinction: Array, length: Array, /
) -> Array:
    optical_depth = extinction * length
    small = jnp.abs(optical_depth) <= jnp.sqrt(jnp.finfo(intensity.dtype).eps)
    denominator = jnp.where(small, jnp.ones_like(optical_depth), optical_depth)
    exact_phi = -jnp.expm1(-optical_depth) / denominator
    series_phi = 1.0 - 0.5 * optical_depth + optical_depth**2 / 6.0
    phi = jnp.where(small, series_phi, exact_phi)
    return intensity * jnp.exp(-optical_depth) + emission * length * phi


class InvariantScalarTransferPlan(StrictModule, NonTrainableState):
    """Piecewise-constant invariant scalar transfer in incident-to-observer order."""

    segment_lengths: Array
    active: Array
    units: InvariantTransferUnitContract
    capacity: int = eqx.field(static=True)
    active_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        segment_lengths: ArrayLike,
        units: InvariantTransferUnitContract,
        /,
        *,
        active: ArrayLike | None = None,
        path_id: str,
    ):
        lengths = np.asarray(segment_lengths, dtype=np.float64)
        active_host = (
            np.ones(lengths.shape, dtype=np.bool_)
            if active is None
            else np.asarray(active, dtype=np.bool_)
        )
        identifier = str(path_id).strip()
        if (
            lengths.ndim != 1
            or lengths.size == 0
            or active_host.shape != lengths.shape
            or np.any(~np.isfinite(lengths))
            or np.any(lengths < 0.0)
            or not np.any(active_host)
            or not _prefix_mask(active_host)
            or not identifier
        ):
            raise ValueError(
                "Transfer segments must be finite, nonnegative, nonempty, and use a nonempty active prefix."
            )
        if not isinstance(units, InvariantTransferUnitContract):
            raise TypeError("units must be an InvariantTransferUnitContract.")
        self.segment_lengths = jax.lax.stop_gradient(jnp.asarray(lengths))
        self.active = jax.lax.stop_gradient(jnp.asarray(active_host))
        self.units = units
        self.capacity = lengths.size
        self.active_count = int(np.count_nonzero(active_host))
        self.plan_id = canonical_fingerprint(
            {
                "kind": "invariant-scalar-transfer",
                "path_id": identifier,
                "units": units.units_id,
                "segments": array_tree_fingerprint((lengths, active_host)),
            }
        )

    def evaluate(
        self,
        invariant_emission: ArrayLike,
        invariant_extinction: ArrayLike,
        incident_invariant_intensity: ArrayLike = 0.0,
        /,
        *,
        support: ArrayLike | None = None,
    ) -> InvariantScalarTransferResult:
        emission = jnp.asarray(invariant_emission)
        if not jnp.issubdtype(emission.dtype, jnp.inexact):
            emission = emission.astype("float64")
        extinction = jnp.asarray(invariant_extinction, dtype=emission.dtype)
        incident = jnp.asarray(incident_invariant_intensity, dtype=emission.dtype)
        support_value = (
            jnp.ones((self.capacity,), dtype=jnp.bool_)
            if support is None
            else jnp.asarray(support, dtype=jnp.bool_)
        )
        if emission.shape != (self.capacity,) or extinction.shape != (self.capacity,):
            raise ValueError("Scalar transfer coefficients must match plan capacity.")
        if incident.shape != ():
            raise ValueError("Incident invariant intensity must be scalar.")
        if support_value.shape != (self.capacity,):
            raise ValueError("Scalar transfer support must match plan capacity.")

        lengths = jnp.where(self.active, self.segment_lengths, 0.0)
        emission_step = jnp.where(self.active, emission, 0.0)
        extinction_step = jnp.where(self.active, extinction, 0.0)

        def step(value, segment):
            source, opacity, length = segment
            result = _stable_slab_step(value, source, opacity, length)
            return result, result

        emergent, segment_history = jax.lax.scan(
            step,
            incident,
            (emission_step, extinction_step, lengths),
        )
        history = jnp.concatenate((incident[None], segment_history), axis=0)
        coefficient_finite = jnp.all(
            ~self.active | (jnp.isfinite(emission) & jnp.isfinite(extinction))
        )
        finite = (
            coefficient_finite & jnp.isfinite(incident) & jnp.all(jnp.isfinite(history))
        )
        physically_valid = (
            jnp.all(~self.active | ((emission >= 0.0) & (extinction >= 0.0)))
            & (incident >= 0.0)
            & jnp.all(history >= -32.0 * jnp.finfo(history.dtype).eps)
        )
        in_support = jnp.all(~self.active | support_value)
        converged = jnp.asarray(True)
        qualified = finite & converged & physically_valid & in_support
        minimum_intensity = jnp.min(history[: self.active_count + 1])
        evidence = InvariantScalarTransferEvidence(
            finite,
            converged,
            physically_valid,
            in_support,
            minimum_intensity,
            qualified,
            qualified,
        )
        optical_depth = jnp.sum(lengths * extinction_step)
        return InvariantScalarTransferResult(
            emergent, history, optical_depth, evidence, self.plan_id
        )


def stokes_basis_rotation(angle: ArrayLike, /) -> Array:
    """Mueller rotation from a local screen basis into a transported basis."""

    angle_value = jnp.asarray(angle)
    cosine = jnp.cos(2.0 * angle_value)
    sine = jnp.sin(2.0 * angle_value)
    zero = jnp.zeros_like(cosine)
    one = jnp.ones_like(cosine)
    rows = (
        jnp.stack((one, zero, zero, zero), axis=-1),
        jnp.stack((zero, cosine, sine, zero), axis=-1),
        jnp.stack((zero, -sine, cosine, zero), axis=-1),
        jnp.stack((zero, zero, zero, one), axis=-1),
    )
    return jnp.stack(rows, axis=-2)


def rotate_stokes_coefficients(
    emission: ArrayLike,
    propagation_matrix: ArrayLike,
    local_to_transported_angle: ArrayLike,
    /,
) -> tuple[Array, Array]:
    """Rotate local polarized coefficients without rotating the transported state."""

    source = jnp.asarray(emission)
    matrix = jnp.asarray(propagation_matrix, dtype=source.dtype)
    angles = jnp.asarray(local_to_transported_angle, dtype=source.dtype)
    if source.ndim < 1 or source.shape[-1] != 4:
        raise ValueError("Polarized emission must end in four Stokes components.")
    if matrix.shape != source.shape[:-1] + (4, 4):
        raise ValueError("Propagation matrices must match polarized emission shape.")
    if angles.shape != source.shape[:-1]:
        raise ValueError("Basis angles must match the coefficient sample shape.")
    rotation = stokes_basis_rotation(angles).astype(source.dtype)
    rotated_source = contract("...ij,...j->...i", rotation, source)
    rotated_matrix = contract("...ij,...jk,...lk->...il", rotation, matrix, rotation)
    return rotated_source, rotated_matrix


def _cone_margin(stokes: Array, /) -> Array:
    polarization_squared = contract("...i,...i->...", stokes[..., 1:], stokes[..., 1:])
    return stokes[..., 0] - jnp.sqrt(jnp.maximum(polarization_squared, 0.0))


def _standard_propagation_physical(matrix: Array, tolerance: float, /) -> Array:
    alpha_i = matrix[:, 0, 0]
    dichroism = matrix[:, 0, 1:]
    first_column_error = jnp.max(jnp.abs(matrix[:, 1:, 0] - dichroism), axis=-1)
    polarization_block = matrix[:, 1:, 1:]
    symmetric = 0.5 * (polarization_block + jnp.swapaxes(polarization_block, -1, -2))
    target = alpha_i[:, None, None] * jnp.eye(3, dtype=matrix.dtype)
    symmetric_error = jnp.max(jnp.abs(symmetric - target), axis=(-2, -1))
    dichroism_norm = jnp.sqrt(contract("ni,ni->n", dichroism, dichroism))
    scale = 1.0 + jnp.max(jnp.abs(matrix), axis=(-2, -1))
    return (
        (alpha_i >= -tolerance * scale)
        & (dichroism_norm <= alpha_i + tolerance * scale)
        & (first_column_error <= tolerance * scale)
        & (symmetric_error <= tolerance * scale)
    )


class PolarizedRayPathEvidence(StrictModule):
    gram_residual: Array
    tangent_residual: Array
    reported_transport_residual: Array
    evidence_binding_residual: Array
    node_valid: Array
    finite: Array
    ray_qualified: Array
    basis_transported: Array
    qualified: Array
    derivative_valid: Array

    @property
    def maximum_transport_residual(self) -> Array:
        return jnp.max(
            jnp.maximum(
                jnp.maximum(self.gram_residual, self.tangent_residual),
                self.reported_transport_residual,
            )
        )


class PolarizedRayPath(StrictModule, NonTrainableState):
    """One exact ray-result lane and its basis recomputation.

    A recorded backend event root is inserted immediately after its preceding
    history sample, retaining the final partial affine segment.  Gram and
    tangent-orthogonality residuals are recomputed from the exact metric, basis,
    and tangent samples and compared with the typed ray bundle's evidence.
    """

    coordinates: Array
    tangents: Array
    affine_parameter: Array
    active: Array
    valid: Array
    transported_screen_basis: Array
    evidence: PolarizedRayPathEvidence
    node_capacity: int = eqx.field(static=True)
    ray_index: int = eqx.field(static=True)
    basis_tolerance: float = eqx.field(static=True)
    ray_result_id: str = eqx.field(static=True)
    path_id: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    coordinate_unit_id: str = eqx.field(static=True)
    affine_parameter_unit_id: str = eqx.field(static=True)
    metric_semantic_id: str | None = eqx.field(static=True)
    metric_numeric_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        ray_result: GRRayResult,
        metric: LorentzianMetric,
        /,
        *,
        ray_index: int,
        metric_semantic_id: str | None = None,
        metric_numeric_id: str | None = None,
        basis_tolerance: float = 1.0e-8,
    ):
        if not isinstance(ray_result, GRRayResult):
            raise TypeError("ray_result must be a GRRayResult.")
        if not isinstance(metric, LorentzianMetric) or metric.chart.dimension != 4:
            raise TypeError("metric must be a four-dimensional LorentzianMetric.")
        if ray_result.ray_kind != "null":
            raise ValueError("Polarized transfer requires a null GR ray result.")
        terminal_basis = ray_result.terminal_transported_screen_basis
        if ray_result.transported_screen_basis is None or terminal_basis is None:
            raise ValueError("Polarized transfer requires transported screen bases.")
        expected_metric_id = gr_metric_identity(
            metric,
            semantic_id=metric_semantic_id,
            numeric_id=metric_numeric_id,
        )
        expected_chart_id = gr_chart_identity(metric)
        if (
            ray_result.metric_id != expected_metric_id
            or ray_result.chart_id != expected_chart_id
        ):
            raise ValueError(
                "Polarized transfer metric and chart must exactly match the ray result."
            )
        index = int(ray_index)
        tolerance = float(basis_tolerance)
        ray_count, history_count = ray_result.affine_parameter.shape
        if (
            index < 0
            or index >= ray_count
            or history_count < 2
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Polarized ray index, history, or tolerance is invalid.")
        ledger = ray_result.event_ledger

        points = ray_result.coordinates[index]
        tangents = ray_result.tangents[index]
        affine = ray_result.affine_parameter[index]
        active = ray_result.active[index]
        valid = ray_result.valid[index]
        basis = ray_result.transported_screen_basis[index]
        reported = ray_result.bundle_evidence.transport_residual[index]
        reported_valid = ray_result.bundle_evidence.transport_valid[index]
        positions = jnp.arange(history_count + 1, dtype=jnp.int32)
        history_positions = jnp.clip(positions, 0, history_count - 1)
        extended_points = points[history_positions]
        extended_tangents = tangents[history_positions]
        extended_affine = affine[history_positions]
        extended_active = jnp.where(
            positions < history_count, active[history_positions], False
        )
        extended_valid = jnp.where(
            positions < history_count, valid[history_positions], False
        )
        extended_basis = basis[history_positions]
        extended_reported = reported[history_positions]
        extended_reported_valid = jnp.where(
            positions < history_count, reported_valid[history_positions], False
        )

        recorded = ledger.recorded[index]
        insertion = jnp.clip(ledger.history_index[index] + 1, 1, history_count)
        is_root = positions == insertion
        retain_before_root = positions < insertion
        root_basis = terminal_basis[index]
        root_active = recorded & ledger.state_valid[index]
        extended_points = jnp.where(
            is_root[:, None], ledger.coordinates[index], extended_points
        )
        extended_tangents = jnp.where(
            is_root[:, None], ledger.tangent[index], extended_tangents
        )
        extended_affine = jnp.where(
            is_root, ledger.affine_parameter[index], extended_affine
        )
        extended_basis = jnp.where(is_root[:, None, None], root_basis, extended_basis)
        extended_active = jnp.where(
            recorded,
            (extended_active & retain_before_root) | (is_root & root_active),
            extended_active,
        )
        extended_valid = jnp.where(
            recorded,
            (extended_valid & retain_before_root) | (is_root & ledger.state_valid[index]),
            extended_valid,
        )

        identity = jnp.eye(4, dtype=extended_points.dtype)
        metrics = jax.vmap(
            lambda point, enabled: jax.lax.cond(
                enabled, lambda value: metric(value), lambda _: identity, point
            )
        )(extended_points, extended_active)
        spatial_sign = float(1 if metric.convention == "mostly_plus" else -1)
        gram = contract("hai,hij,hbj->hab", extended_basis, metrics, extended_basis)
        tangent_pairing = contract(
            "hai,hij,hj->ha", extended_basis, metrics, extended_tangents
        )
        target = spatial_sign * jnp.eye(2, dtype=extended_points.dtype)
        gram_residual = jnp.max(jnp.abs(gram - target), axis=(-2, -1))
        tangent_residual = jnp.max(jnp.abs(tangent_pairing), axis=-1)
        recomputed = jnp.maximum(gram_residual, tangent_residual)
        extended_reported = jnp.where(is_root & recorded, recomputed, extended_reported)
        extended_reported_valid = jnp.where(
            is_root & recorded, ledger.state_valid[index], extended_reported_valid
        )
        binding_residual = jnp.abs(extended_reported - recomputed)
        finite_nodes = (
            jnp.all(jnp.isfinite(extended_points), axis=-1)
            & jnp.all(jnp.isfinite(extended_tangents), axis=-1)
            & jnp.all(jnp.isfinite(extended_basis), axis=(-2, -1))
            & jnp.isfinite(extended_affine)
            & jnp.isfinite(recomputed)
            & jnp.isfinite(extended_reported)
        )
        residual_scale = 1.0 + jnp.maximum(
            jnp.abs(extended_reported), jnp.abs(recomputed)
        )
        node_valid = (
            extended_active
            & extended_valid
            & extended_reported_valid
            & finite_nodes
            & (recomputed <= tolerance)
            & (extended_reported <= tolerance)
            & (binding_residual <= tolerance * residual_scale)
        )
        basis_transported = jnp.any(extended_active) & jnp.all(
            jnp.where(extended_active, node_valid, True)
        )
        ray_qualified = ray_result.qualified[index]
        qualified = ray_qualified & basis_transported
        evidence = PolarizedRayPathEvidence(
            jnp.where(extended_active, gram_residual, 0.0),
            jnp.where(extended_active, tangent_residual, 0.0),
            jnp.where(extended_active, extended_reported, 0.0),
            jnp.where(extended_active, binding_residual, 0.0),
            node_valid,
            jnp.all(jnp.where(extended_active, finite_nodes, True)),
            ray_qualified,
            basis_transported,
            qualified,
            qualified & ray_result.derivative_valid[index],
        )
        self.coordinates = jax.lax.stop_gradient(extended_points)
        self.tangents = jax.lax.stop_gradient(extended_tangents)
        self.affine_parameter = jax.lax.stop_gradient(extended_affine)
        self.active = jax.lax.stop_gradient(extended_active)
        self.valid = jax.lax.stop_gradient(extended_valid)
        self.transported_screen_basis = jax.lax.stop_gradient(extended_basis)
        self.evidence = evidence
        self.node_capacity = history_count + 1
        self.ray_index = index
        self.metric_id = expected_metric_id
        self.chart_id = expected_chart_id
        self.basis_tolerance = tolerance
        self.convention_id = ray_result.convention_id
        self.scale_id = ray_result.scale_id
        self.coordinate_unit_id = ray_result.coordinate_unit_id
        self.affine_parameter_unit_id = ray_result.affine_parameter_unit_id
        self.metric_semantic_id = metric_semantic_id
        self.metric_numeric_id = metric_numeric_id
        self.ray_result_id = ray_result.result_id
        self.path_id = canonical_fingerprint(
            {
                "kind": "polarized-ray-path",
                "ray_result": ray_result.result_id,
                "event_ledger": ledger.ledger_id,
                "bundle": ray_result.bundle_evidence.bundle_id,
                "metric": expected_metric_id,
                "chart": expected_chart_id,
                "metric_semantic_id": metric_semantic_id,
                "metric_numeric_id": metric_numeric_id,
                "convention": ray_result.convention_id,
                "scale": ray_result.scale_id,
                "coordinate_unit": ray_result.coordinate_unit_id,
                "affine_parameter_unit": ray_result.affine_parameter_unit_id,
                "ray_index": index,
                "basis_tolerance": tolerance,
            }
        )


class PolarizedInvariantTransferPlan(StrictModule, NonTrainableState):
    """Invariant Stokes transfer on one exact, evidence-bound GR ray path."""

    path: PolarizedRayPath
    segment_lengths: Array
    active: Array
    node_active: Array
    units: InvariantTransferUnitContract
    capacity: int = eqx.field(static=True)
    active_count: int = eqx.field(static=True)
    cone_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        path: PolarizedRayPath,
        units: InvariantTransferUnitContract,
        /,
        *,
        cone_tolerance: float = 1.0e-10,
    ):
        if not isinstance(path, PolarizedRayPath):
            raise TypeError("path must be a PolarizedRayPath.")
        if not isinstance(units, InvariantTransferUnitContract):
            raise TypeError("units must be an InvariantTransferUnitContract.")
        if units.path_parameter_unit.unit_id != path.affine_parameter_unit_id:
            raise ValueError(
                "Transfer path-parameter unit must exactly match the GR ray path."
            )
        cone_tolerance_ = float(cone_tolerance)
        affine_host = np.asarray(path.affine_parameter, dtype=np.float64)
        node_active_host = np.asarray(path.active, dtype=np.bool_)
        active_host = node_active_host[:-1] & node_active_host[1:]
        lengths = np.diff(affine_host)
        if (
            lengths.ndim != 1
            or lengths.size == 0
            or node_active_host.shape != (lengths.size + 1,)
            or not np.any(active_host)
            or not _prefix_mask(active_host)
            or np.any(~np.isfinite(lengths[active_host]))
            or np.any(lengths[active_host] < 0.0)
            or not np.isfinite(cone_tolerance_)
            or cone_tolerance_ < 0.0
        ):
            raise ValueError(
                "Polarized ray path must provide a finite nonnegative active "
                "segment prefix and a finite cone tolerance."
            )
        active_count = int(np.count_nonzero(active_host))
        self.path = path
        self.segment_lengths = jax.lax.stop_gradient(jnp.asarray(lengths))
        self.active = jax.lax.stop_gradient(jnp.asarray(active_host))
        self.node_active = jax.lax.stop_gradient(jnp.asarray(node_active_host))
        self.units = units
        self.capacity = lengths.size
        self.active_count = active_count
        self.cone_tolerance = cone_tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polarized-invariant-transfer",
                "path": path.path_id,
                "ray_result": path.ray_result_id,
                "units": units.units_id,
                "segments": array_tree_fingerprint((lengths, active_host)),
                "cone_tolerance": cone_tolerance_,
            }
        )

    def evaluate(
        self,
        invariant_emission: ArrayLike,
        invariant_propagation_matrix: ArrayLike,
        incident_invariant_stokes: ArrayLike,
        local_to_transported_angle: ArrayLike,
        /,
        *,
        support: ArrayLike | None = None,
    ) -> PolarizedInvariantTransferResult:
        emission = jnp.asarray(invariant_emission)
        if not jnp.issubdtype(emission.dtype, jnp.inexact):
            emission = emission.astype("float64")
        matrix = jnp.asarray(invariant_propagation_matrix, dtype=emission.dtype)
        incident = jnp.asarray(incident_invariant_stokes, dtype=emission.dtype)
        angle = jnp.asarray(local_to_transported_angle, dtype=emission.dtype)
        support_value = (
            jnp.ones((self.capacity,), dtype=jnp.bool_)
            if support is None
            else jnp.asarray(support, dtype=jnp.bool_)
        )
        if emission.shape != (self.capacity, 4):
            raise ValueError("Polarized emission must have shape (capacity, 4).")
        if matrix.shape != (self.capacity, 4, 4):
            raise ValueError(
                "Polarized propagation matrices must have shape (capacity, 4, 4)."
            )
        if incident.shape != (4,) or angle.shape != (self.capacity,):
            raise ValueError("Incident Stokes vector or basis-angle shape is invalid.")
        if support_value.shape != (self.capacity,):
            raise ValueError("Polarized transfer support must match plan capacity.")

        safe_emission = jnp.where(self.active[:, None], emission, 0.0)
        safe_matrix = jnp.where(self.active[:, None, None], matrix, 0.0)
        safe_angle = jnp.where(self.active, angle, 0.0)
        rotated_emission, rotated_matrix = rotate_stokes_coefficients(
            safe_emission, safe_matrix, safe_angle
        )
        lengths = jnp.where(self.active, self.segment_lengths, 0.0)

        def step(carry, segment):
            stokes, prior_converged = carry
            source, operator, length = segment
            intensity_scale = jnp.maximum(
                jnp.maximum(
                    jnp.max(jnp.abs(stokes), initial=0.0),
                    jnp.max(jnp.abs(source), initial=0.0) * jnp.abs(length),
                ),
                jnp.finfo(stokes.dtype).tiny,
            )
            augmented = jnp.zeros((5, 5), dtype=stokes.dtype)
            augmented = augmented.at[:4, :4].set(-operator)
            augmented = augmented.at[:4, 4].set(source / intensity_scale)
            homogeneous = jnp.concatenate(
                (stokes / intensity_scale, jnp.ones((1,), dtype=stokes.dtype)),
                axis=0,
            )
            dense_operator = la.DenseLinearOperator(
                augmented, operator_id="gr-polarized-augmented-transfer"
            )
            action = la.matrix_exponential_action(dense_operator, homogeneous, length)
            updated = jnp.asarray(action.value)[:4] * intensity_scale
            return (updated, prior_converged & action.converged), updated

        (emergent, converged), segment_history = jax.lax.scan(
            step,
            (incident, jnp.asarray(True)),
            (rotated_emission, rotated_matrix, lengths),
        )
        history = jnp.concatenate((incident[None, :], segment_history), axis=0)
        active_coefficients_finite = jnp.all(
            ~self.active
            | (
                jnp.all(jnp.isfinite(emission), axis=-1)
                & jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
                & jnp.isfinite(angle)
            )
        )
        screen = self.path.transported_screen_basis.astype(emission.dtype)
        active_basis_finite = self.path.evidence.finite & jnp.all(
            ~self.node_active | jnp.all(jnp.isfinite(screen), axis=(-2, -1))
        )
        finite = (
            active_coefficients_finite
            & active_basis_finite
            & jnp.all(jnp.isfinite(incident))
            & jnp.all(jnp.isfinite(history))
        )
        maximum_basis_transport_error = self.path.evidence.maximum_transport_residual
        basis_transported = self.path.evidence.qualified
        source_margin = _cone_margin(rotated_emission)
        history_margin = _cone_margin(history)
        active_history_margin = jnp.min(
            jnp.where(self.node_active, history_margin, jnp.inf)
        )
        propagation_physical = _standard_propagation_physical(
            rotated_matrix, self.cone_tolerance
        )
        physically_valid = (
            jnp.all(~self.active | (source_margin >= -self.cone_tolerance))
            & jnp.all(~self.active | propagation_physical)
            & jnp.all(~self.node_active | (history_margin >= -self.cone_tolerance))
        )
        in_support = jnp.all(~self.active | support_value)
        qualified = finite & converged & physically_valid & in_support & basis_transported
        derivative_valid = qualified & self.path.evidence.derivative_valid
        evidence = PolarizedTransferEvidence(
            finite,
            converged,
            physically_valid,
            in_support,
            basis_transported,
            active_history_margin,
            maximum_basis_transport_error,
            qualified,
            derivative_valid,
        )
        optical_depth = jnp.sum(lengths * rotated_matrix[:, 0, 0])
        return PolarizedInvariantTransferResult(
            emergent, history, optical_depth, evidence, self.plan_id
        )
