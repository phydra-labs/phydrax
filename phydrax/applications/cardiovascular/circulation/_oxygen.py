#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Blood oxygen content, inversion, mixing, transport, and membrane exchange."""

from __future__ import annotations

from enum import IntFlag
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


_MM3_PER_DL = 100_000.0


class OxygenStatus(IntFlag):
    """Fail-closed status bits for oxygen evaluations and inversions."""

    SUCCESS = 0
    NONFINITE = 1
    OUT_OF_DOMAIN = 2
    INVERSION_RESIDUAL = 4
    INVALID_FLOW = 8
    TRANSPORT_CFL = 16


class BloodOxygenModel(StrictModule, NonTrainableState):
    """Hill-equilibrium oxygen content with dissolved and Hb-bound terms."""

    hemoglobin_g_per_dL: float = eqx.field(static=True)
    binding_capacity_mL_per_g: float = eqx.field(static=True)
    solubility_mL_per_dL_kPa: float = eqx.field(static=True)
    p50_kPa: float = eqx.field(static=True)
    hill_exponent: float = eqx.field(static=True)
    maximum_partial_pressure_kPa: float = eqx.field(static=True)
    inversion_steps: int = eqx.field(static=True)
    inversion_tolerance_mL_per_dL: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        hemoglobin_g_per_dL: float,
        /,
        *,
        binding_capacity_mL_per_g: float = 1.34,
        solubility_mL_per_dL_kPa: float = 0.0225,
        p50_kPa: float = 3.5,
        hill_exponent: float = 2.7,
        maximum_partial_pressure_kPa: float = 80.0,
        inversion_steps: int = 64,
        inversion_tolerance_mL_per_dL: float = 1.0e-8,
    ):
        hemoglobin = float(hemoglobin_g_per_dL)
        capacity = float(binding_capacity_mL_per_g)
        solubility = float(solubility_mL_per_dL_kPa)
        p50 = float(p50_kPa)
        exponent = float(hill_exponent)
        maximum_pressure = float(maximum_partial_pressure_kPa)
        steps = int(inversion_steps)
        tolerance = float(inversion_tolerance_mL_per_dL)
        values = (
            hemoglobin,
            capacity,
            solubility,
            p50,
            exponent,
            maximum_pressure,
            tolerance,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Oxygen model parameters must be finite.")
        if (
            hemoglobin < 0.0
            or capacity <= 0.0
            or solubility < 0.0
            or p50 <= 0.0
            or exponent <= 0.0
            or maximum_pressure <= 0.0
            or steps <= 0
            or tolerance <= 0.0
        ):
            raise ValueError("Oxygen model parameters are outside their fixed domains.")
        if hemoglobin == 0.0 and solubility == 0.0:
            raise ValueError("At least one oxygen storage mechanism must be present.")
        self.hemoglobin_g_per_dL = hemoglobin
        self.binding_capacity_mL_per_g = capacity
        self.solubility_mL_per_dL_kPa = solubility
        self.p50_kPa = p50
        self.hill_exponent = exponent
        self.maximum_partial_pressure_kPa = maximum_pressure
        self.inversion_steps = steps
        self.inversion_tolerance_mL_per_dL = tolerance
        self.model_id = canonical_fingerprint(
            {
                "kind": "blood-oxygen-content-v1",
                "hemoglobin_g_per_dL": hemoglobin,
                "binding_capacity_mL_per_g": capacity,
                "solubility_mL_per_dL_kPa": solubility,
                "p50_kPa": p50,
                "hill_exponent": exponent,
                "maximum_partial_pressure_kPa": maximum_pressure,
                "inversion_steps": steps,
                "inversion_tolerance_mL_per_dL": tolerance,
            }
        )

    @property
    def bound_capacity_mL_per_dL(self) -> float:
        return self.hemoglobin_g_per_dL * self.binding_capacity_mL_per_g


class OxygenContent(StrictModule):
    """Resolved dissolved, Hb-bound, and total oxygen content."""

    partial_pressure_kPa: Array
    saturation: Array
    dissolved_mL_per_dL: Array
    bound_mL_per_dL: Array
    total_mL_per_dL: Array
    status: Array
    successful: Array


def _hill_saturation(model: BloodOxygenModel, partial_pressure_kPa: Array) -> Array:
    pressure_power = partial_pressure_kPa**model.hill_exponent
    p50_power = model.p50_kPa**model.hill_exponent
    return pressure_power / (p50_power + pressure_power)


def evaluate_oxygen_content(
    model: BloodOxygenModel,
    partial_pressure_kPa: ArrayLike,
    /,
) -> OxygenContent:
    """Evaluate oxygen content on the model's declared pressure interval."""
    pressure = jnp.asarray(partial_pressure_kPa)
    valid = (
        jnp.isfinite(pressure)
        & (pressure >= 0.0)
        & (pressure <= model.maximum_partial_pressure_kPa)
    )
    safe_pressure = jnp.where(valid, pressure, 0.0)
    saturation = _hill_saturation(model, safe_pressure)
    dissolved = model.solubility_mL_per_dL_kPa * safe_pressure
    bound = model.bound_capacity_mL_per_dL * saturation
    total = dissolved + bound
    status = jnp.where(
        valid,
        jnp.asarray(int(OxygenStatus.SUCCESS), dtype=jnp.int32),
        jnp.asarray(int(OxygenStatus.OUT_OF_DOMAIN), dtype=jnp.int32),
    )
    status = jnp.where(
        jnp.isfinite(pressure),
        status,
        jnp.bitwise_or(status, int(OxygenStatus.NONFINITE)),
    )
    nan = jnp.asarray(jnp.nan, dtype=pressure.dtype)
    return OxygenContent(
        partial_pressure_kPa=jnp.where(valid, pressure, nan),
        saturation=jnp.where(valid, saturation, nan),
        dissolved_mL_per_dL=jnp.where(valid, dissolved, nan),
        bound_mL_per_dL=jnp.where(valid, bound, nan),
        total_mL_per_dL=jnp.where(valid, total, nan),
        status=status,
        successful=valid,
    )


class OxygenInversionResult(StrictModule):
    """Pressure/content result and inversion evidence."""

    content: OxygenContent
    target_total_mL_per_dL: Array
    residual_mL_per_dL: Array
    status: Array
    successful: Array


def invert_oxygen_saturation(
    model: BloodOxygenModel,
    saturation: ArrayLike,
    /,
) -> OxygenInversionResult:
    """Analytically invert Hill saturation within the pressure validity domain."""
    saturation_ = jnp.asarray(saturation)
    maximum_saturation = _hill_saturation(
        model,
        jnp.asarray(model.maximum_partial_pressure_kPa, dtype=saturation_.dtype),
    )
    valid = (
        jnp.isfinite(saturation_)
        & (saturation_ >= 0.0)
        & (saturation_ <= maximum_saturation)
    )
    safe_saturation = jnp.where(valid, saturation_, 0.0)
    pressure = model.p50_kPa * (safe_saturation / (1.0 - safe_saturation)) ** (
        1.0 / model.hill_exponent
    )
    content = evaluate_oxygen_content(model, pressure)
    residual = content.saturation - saturation_
    successful = valid & content.successful
    status = jnp.where(
        successful,
        jnp.asarray(int(OxygenStatus.SUCCESS), dtype=jnp.int32),
        jnp.asarray(int(OxygenStatus.OUT_OF_DOMAIN), dtype=jnp.int32),
    )
    status = jnp.where(
        jnp.isfinite(saturation_),
        status,
        jnp.bitwise_or(status, int(OxygenStatus.NONFINITE)),
    )
    nan = jnp.asarray(jnp.nan, dtype=saturation_.dtype)
    failed_content = OxygenContent(
        partial_pressure_kPa=jnp.where(successful, content.partial_pressure_kPa, nan),
        saturation=jnp.where(successful, content.saturation, nan),
        dissolved_mL_per_dL=jnp.where(successful, content.dissolved_mL_per_dL, nan),
        bound_mL_per_dL=jnp.where(successful, content.bound_mL_per_dL, nan),
        total_mL_per_dL=jnp.where(successful, content.total_mL_per_dL, nan),
        status=status,
        successful=successful,
    )
    return OxygenInversionResult(
        content=failed_content,
        target_total_mL_per_dL=jnp.where(successful, content.total_mL_per_dL, nan),
        residual_mL_per_dL=jnp.where(successful, residual, nan),
        status=status,
        successful=successful,
    )


def invert_oxygen_content(
    model: BloodOxygenModel,
    total_mL_per_dL: ArrayLike,
    /,
) -> OxygenInversionResult:
    """Invert total oxygen content by fixed-iteration monotone bisection."""
    target = jnp.asarray(total_mL_per_dL)
    maximum = evaluate_oxygen_content(
        model, jnp.asarray(model.maximum_partial_pressure_kPa, dtype=target.dtype)
    ).total_mL_per_dL
    valid = jnp.isfinite(target) & (target >= 0.0) & (target <= maximum)
    safe_target = jnp.where(valid, target, 0.0)

    def bisect(_, bracket):
        lower, upper = bracket
        middle = 0.5 * (lower + upper)
        middle_total = (
            model.solubility_mL_per_dL_kPa * middle
            + model.bound_capacity_mL_per_dL * _hill_saturation(model, middle)
        )
        lower = jnp.where(middle_total < safe_target, middle, lower)
        upper = jnp.where(middle_total < safe_target, upper, middle)
        return lower, upper

    lower, upper = jax.lax.fori_loop(
        0,
        model.inversion_steps,
        bisect,
        (
            jnp.zeros_like(safe_target),
            jnp.full_like(safe_target, model.maximum_partial_pressure_kPa),
        ),
    )
    pressure = 0.5 * (lower + upper)
    content = evaluate_oxygen_content(model, pressure)
    residual = content.total_mL_per_dL - target
    residual_valid = jnp.abs(residual) <= model.inversion_tolerance_mL_per_dL
    successful = valid & content.successful & residual_valid
    status = jnp.asarray(int(OxygenStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        valid,
        status,
        jnp.bitwise_or(status, int(OxygenStatus.OUT_OF_DOMAIN)),
    )
    status = jnp.where(
        jnp.isfinite(target),
        status,
        jnp.bitwise_or(status, int(OxygenStatus.NONFINITE)),
    )
    status = jnp.where(
        residual_valid,
        status,
        jnp.bitwise_or(status, int(OxygenStatus.INVERSION_RESIDUAL)),
    )
    nan = jnp.asarray(jnp.nan, dtype=target.dtype)
    failed_content = OxygenContent(
        partial_pressure_kPa=jnp.where(successful, content.partial_pressure_kPa, nan),
        saturation=jnp.where(successful, content.saturation, nan),
        dissolved_mL_per_dL=jnp.where(successful, content.dissolved_mL_per_dL, nan),
        bound_mL_per_dL=jnp.where(successful, content.bound_mL_per_dL, nan),
        total_mL_per_dL=jnp.where(successful, content.total_mL_per_dL, nan),
        status=status,
        successful=successful,
    )
    return OxygenInversionResult(
        content=failed_content,
        target_total_mL_per_dL=target,
        residual_mL_per_dL=jnp.where(successful, residual, nan),
        status=status,
        successful=successful,
    )


class OxygenMixingResult(StrictModule):
    """Flow-weighted content and exact oxygen-flux conservation evidence."""

    mixed_content_mL_per_dL: Array
    total_flow_mm3_per_ms: Array
    incoming_oxygen_flux_mL_per_ms: Array
    outgoing_oxygen_flux_mL_per_ms: Array
    conservation_residual_mL_per_ms: Array
    successful: Array


def mix_oxygen_content(
    flow_mm3_per_ms: ArrayLike,
    content_mL_per_dL: ArrayLike,
    /,
) -> OxygenMixingResult:
    """Conservatively mix nonnegative incoming streams without epsilon division."""
    flow = jnp.asarray(flow_mm3_per_ms)
    content = jnp.asarray(content_mL_per_dL, dtype=flow.dtype)
    if flow.ndim != 1 or content.shape != flow.shape:
        raise ValueError(
            "Flow and content must be one-dimensional arrays of equal shape."
        )
    total_flow = jnp.sum(flow)
    valid = (
        jnp.all(jnp.isfinite(flow))
        & jnp.all(jnp.isfinite(content))
        & jnp.all(flow >= 0.0)
        & jnp.all(content >= 0.0)
        & (total_flow > 0.0)
    )
    safe_flow = jnp.where(jnp.isfinite(flow) & (flow >= 0.0), flow, 0.0)
    safe_content = jnp.where(jnp.isfinite(content) & (content >= 0.0), content, 0.0)
    safe_total_flow = jnp.where(total_flow > 0.0, total_flow, 1.0)
    incoming_flux = jnp.sum(safe_flow * safe_content) / _MM3_PER_DL
    mixed = jnp.sum(safe_flow * safe_content) / safe_total_flow
    outgoing_flux = safe_total_flow * mixed / _MM3_PER_DL
    residual = outgoing_flux - incoming_flux
    nan = jnp.asarray(jnp.nan, dtype=flow.dtype)
    return OxygenMixingResult(
        mixed_content_mL_per_dL=jnp.where(valid, mixed, nan),
        total_flow_mm3_per_ms=jnp.where(valid, total_flow, nan),
        incoming_oxygen_flux_mL_per_ms=jnp.where(valid, incoming_flux, nan),
        outgoing_oxygen_flux_mL_per_ms=jnp.where(valid, outgoing_flux, nan),
        conservation_residual_mL_per_ms=jnp.where(valid, residual, nan),
        successful=valid,
    )


class OxygenTransportPlan(StrictModule, NonTrainableState):
    """Fixed-topology conservative content transport plan."""

    cell_volume_mm3: Array
    source_index: Array
    destination_index: Array
    inflow_cell_index: Array
    outflow_cell_index: Array
    step_size_ms: float = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    inflow_count: int = eqx.field(static=True)
    outflow_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_volume_mm3: ArrayLike,
        source_index: ArrayLike,
        destination_index: ArrayLike,
        step_size_ms: float,
        /,
        *,
        inflow_cell_index: ArrayLike = (),
        outflow_cell_index: ArrayLike = (),
    ):
        volumes_host = np.asarray(cell_volume_mm3, dtype=float)
        source_host = np.asarray(source_index, dtype=np.int32)
        destination_host = np.asarray(destination_index, dtype=np.int32)
        inflow_host = np.asarray(inflow_cell_index, dtype=np.int32)
        outflow_host = np.asarray(outflow_cell_index, dtype=np.int32)
        step = float(step_size_ms)
        if volumes_host.ndim != 1 or volumes_host.size == 0:
            raise ValueError("cell_volume_mm3 must be a non-empty vector.")
        if not np.all(np.isfinite(volumes_host)) or np.any(volumes_host <= 0.0):
            raise ValueError("Every transport cell volume must be finite and positive.")
        if source_host.ndim != 1 or destination_host.shape != source_host.shape:
            raise ValueError("Transport edge index vectors must have equal shape.")
        if inflow_host.ndim != 1 or outflow_host.ndim != 1:
            raise ValueError("Boundary index arrays must be one-dimensional.")
        count = int(volumes_host.size)
        for indices in (source_host, destination_host, inflow_host, outflow_host):
            if np.any(indices < 0) or np.any(indices >= count):
                raise ValueError("Transport topology contains an invalid cell index.")
        if np.any(source_host == destination_host):
            raise ValueError("Transport edges must connect distinct cells.")
        if not isfinite(step) or step <= 0.0:
            raise ValueError("step_size_ms must be finite and positive.")
        self.cell_volume_mm3 = jnp.asarray(volumes_host)
        self.source_index = jnp.asarray(source_host)
        self.destination_index = jnp.asarray(destination_host)
        self.inflow_cell_index = jnp.asarray(inflow_host)
        self.outflow_cell_index = jnp.asarray(outflow_host)
        self.step_size_ms = step
        self.cell_count = count
        self.edge_count = int(source_host.size)
        self.inflow_count = int(inflow_host.size)
        self.outflow_count = int(outflow_host.size)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "oxygen-content-transport-v1",
                "cell_volume_mm3": volumes_host.tolist(),
                "source_index": source_host.tolist(),
                "destination_index": destination_host.tolist(),
                "inflow_cell_index": inflow_host.tolist(),
                "outflow_cell_index": outflow_host.tolist(),
                "step_size_ms": step,
            }
        )


class OxygenTransportState(StrictModule):
    """Accepted cell oxygen contents and time cursor."""

    content_mL_per_dL: Array
    time_ms: Array
    step_index: Array


class OxygenTransportInputs(StrictModule):
    """Directed internal and boundary flows for one transport step."""

    edge_flow_mm3_per_ms: Array
    inflow_mm3_per_ms: Array
    inflow_content_mL_per_dL: Array
    outflow_mm3_per_ms: Array


class OxygenTransportEvidence(StrictModule):
    """Oxygen inventory conservation and advective-domain evidence."""

    previous_inventory_mL: Array
    candidate_inventory_mL: Array
    boundary_inventory_change_mL: Array
    conservation_residual_mL: Array
    maximum_outflow_fraction: Array
    status: Array
    successful: Array


class OxygenTransportResult(StrictModule):
    """Accepted and candidate transport states."""

    state: OxygenTransportState
    candidate: OxygenTransportState
    evidence: OxygenTransportEvidence


def initialize_oxygen_transport_state(
    plan: OxygenTransportPlan,
    content_mL_per_dL: ArrayLike,
    /,
) -> OxygenTransportState:
    content = jnp.asarray(content_mL_per_dL, dtype=plan.cell_volume_mm3.dtype)
    if content.shape != (plan.cell_count,):
        raise ValueError("Initial oxygen content shape does not match the plan.")
    content_host = np.asarray(content)
    if not np.all(np.isfinite(content_host)) or np.any(content_host < 0.0):
        raise ValueError("Initial oxygen content must be finite and nonnegative.")
    return OxygenTransportState(
        content_mL_per_dL=content,
        time_ms=jnp.asarray(0.0, dtype=content.dtype),
        step_index=jnp.asarray(0, dtype=jnp.int32),
    )


def step_oxygen_transport(
    plan: OxygenTransportPlan,
    state: OxygenTransportState,
    inputs: OxygenTransportInputs,
    /,
) -> OxygenTransportResult:
    """Advance directed upwind oxygen content and fail closed on invalid flows."""
    shapes_valid = (
        inputs.edge_flow_mm3_per_ms.shape == (plan.edge_count,)
        and inputs.inflow_mm3_per_ms.shape == (plan.inflow_count,)
        and inputs.inflow_content_mL_per_dL.shape == (plan.inflow_count,)
        and inputs.outflow_mm3_per_ms.shape == (plan.outflow_count,)
    )
    if not shapes_valid:
        raise ValueError("Oxygen transport input shapes do not match the plan.")
    edge_flow = jnp.asarray(inputs.edge_flow_mm3_per_ms, dtype=plan.cell_volume_mm3.dtype)
    inflow = jnp.asarray(inputs.inflow_mm3_per_ms, dtype=edge_flow.dtype)
    inflow_content = jnp.asarray(inputs.inflow_content_mL_per_dL, dtype=edge_flow.dtype)
    outflow = jnp.asarray(inputs.outflow_mm3_per_ms, dtype=edge_flow.dtype)
    finite = (
        jnp.all(jnp.isfinite(state.content_mL_per_dL))
        & jnp.all(jnp.isfinite(edge_flow))
        & jnp.all(jnp.isfinite(inflow))
        & jnp.all(jnp.isfinite(inflow_content))
        & jnp.all(jnp.isfinite(outflow))
    )
    nonnegative = (
        jnp.all(state.content_mL_per_dL >= 0.0)
        & jnp.all(edge_flow >= 0.0)
        & jnp.all(inflow >= 0.0)
        & jnp.all(inflow_content >= 0.0)
        & jnp.all(outflow >= 0.0)
    )
    safe_edge_flow = jnp.where(
        jnp.isfinite(edge_flow) & (edge_flow >= 0.0), edge_flow, 0.0
    )
    safe_inflow = jnp.where(jnp.isfinite(inflow) & (inflow >= 0.0), inflow, 0.0)
    safe_inflow_content = jnp.where(
        jnp.isfinite(inflow_content) & (inflow_content >= 0.0), inflow_content, 0.0
    )
    safe_outflow = jnp.where(jnp.isfinite(outflow) & (outflow >= 0.0), outflow, 0.0)
    content = jnp.where(
        jnp.isfinite(state.content_mL_per_dL) & (state.content_mL_per_dL >= 0.0),
        state.content_mL_per_dL,
        0.0,
    )
    dt = plan.step_size_ms
    inventory = plan.cell_volume_mm3 * content / _MM3_PER_DL
    edge_transfer = dt * safe_edge_flow * content[plan.source_index] / _MM3_PER_DL
    candidate_inventory = inventory
    candidate_inventory = candidate_inventory.at[plan.source_index].add(-edge_transfer)
    candidate_inventory = candidate_inventory.at[plan.destination_index].add(
        edge_transfer
    )
    boundary_in = dt * safe_inflow * safe_inflow_content / _MM3_PER_DL
    boundary_out = dt * safe_outflow * content[plan.outflow_cell_index] / _MM3_PER_DL
    candidate_inventory = candidate_inventory.at[plan.inflow_cell_index].add(boundary_in)
    candidate_inventory = candidate_inventory.at[plan.outflow_cell_index].add(
        -boundary_out
    )
    candidate_content = candidate_inventory * _MM3_PER_DL / plan.cell_volume_mm3
    outgoing_volume = jnp.zeros((plan.cell_count,), dtype=edge_flow.dtype)
    outgoing_volume = outgoing_volume.at[plan.source_index].add(dt * safe_edge_flow)
    outgoing_volume = outgoing_volume.at[plan.outflow_cell_index].add(dt * safe_outflow)
    outflow_fraction = outgoing_volume / plan.cell_volume_mm3
    maximum_outflow_fraction = jnp.max(outflow_fraction)
    cfl_valid = maximum_outflow_fraction <= 1.0
    candidate_valid = jnp.all(jnp.isfinite(candidate_content)) & jnp.all(
        candidate_content >= 0.0
    )
    successful = finite & nonnegative & cfl_valid & candidate_valid
    candidate = OxygenTransportState(
        content_mL_per_dL=candidate_content,
        time_ms=state.time_ms + dt,
        step_index=state.step_index + jnp.asarray(1, dtype=state.step_index.dtype),
    )
    accepted = jax.tree.map(
        lambda proposed, prior: jnp.where(successful, proposed, prior), candidate, state
    )
    boundary_change = jnp.sum(boundary_in) - jnp.sum(boundary_out)
    conservation_residual = (
        jnp.sum(candidate_inventory) - jnp.sum(inventory) - boundary_change
    )
    status = jnp.asarray(int(OxygenStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        finite,
        status,
        jnp.bitwise_or(status, int(OxygenStatus.NONFINITE)),
    )
    status = jnp.where(
        nonnegative & candidate_valid,
        status,
        jnp.bitwise_or(status, int(OxygenStatus.INVALID_FLOW)),
    )
    status = jnp.where(
        cfl_valid,
        status,
        jnp.bitwise_or(status, int(OxygenStatus.TRANSPORT_CFL)),
    )
    evidence = OxygenTransportEvidence(
        previous_inventory_mL=jnp.sum(inventory),
        candidate_inventory_mL=jnp.sum(candidate_inventory),
        boundary_inventory_change_mL=boundary_change,
        conservation_residual_mL=conservation_residual,
        maximum_outflow_fraction=maximum_outflow_fraction,
        status=status,
        successful=successful,
    )
    return OxygenTransportResult(state=accepted, candidate=candidate, evidence=evidence)


class MembraneOxygenatorModel(StrictModule, NonTrainableState):
    """Explicit gas-exchange model for an otherwise hydraulic oxygenator."""

    blood_model: BloodOxygenModel
    gas_partial_pressure_kPa: float = eqx.field(static=True)
    transfer_capacity_mm3_per_ms: float = eqx.field(static=True)
    minimum_flow_mm3_per_ms: float = eqx.field(static=True)
    maximum_flow_mm3_per_ms: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        blood_model: BloodOxygenModel,
        gas_partial_pressure_kPa: float,
        transfer_capacity_mm3_per_ms: float,
        /,
        *,
        minimum_flow_mm3_per_ms: float,
        maximum_flow_mm3_per_ms: float,
    ):
        if not isinstance(blood_model, BloodOxygenModel):
            raise TypeError("blood_model must be a BloodOxygenModel.")
        pressure = float(gas_partial_pressure_kPa)
        capacity = float(transfer_capacity_mm3_per_ms)
        minimum = float(minimum_flow_mm3_per_ms)
        maximum = float(maximum_flow_mm3_per_ms)
        if not all(isfinite(value) for value in (pressure, capacity, minimum, maximum)):
            raise ValueError("Membrane oxygenator parameters must be finite.")
        if (
            pressure < 0.0
            or pressure > blood_model.maximum_partial_pressure_kPa
            or capacity <= 0.0
            or minimum <= 0.0
            or maximum <= minimum
        ):
            raise ValueError("Membrane oxygenator parameters are outside their domains.")
        self.blood_model = blood_model
        self.gas_partial_pressure_kPa = pressure
        self.transfer_capacity_mm3_per_ms = capacity
        self.minimum_flow_mm3_per_ms = minimum
        self.maximum_flow_mm3_per_ms = maximum
        self.model_id = canonical_fingerprint(
            {
                "kind": "membrane-oxygenator-v1",
                "blood_model": blood_model.model_id,
                "gas_partial_pressure_kPa": pressure,
                "transfer_capacity_mm3_per_ms": capacity,
                "minimum_flow_mm3_per_ms": minimum,
                "maximum_flow_mm3_per_ms": maximum,
            }
        )


class OxygenatorExchangeResult(StrictModule):
    """Outlet oxygen state and oxygen-flux gain through a membrane model."""

    outlet: OxygenContent
    effectiveness: Array
    oxygen_transfer_mL_per_ms: Array
    status: Array
    successful: Array


def exchange_membrane_oxygen(
    model: MembraneOxygenatorModel,
    inlet_total_mL_per_dL: ArrayLike,
    flow_mm3_per_ms: ArrayLike,
    /,
) -> OxygenatorExchangeResult:
    """Apply finite-capacity membrane exchange on the explicit flow domain."""
    inlet = jnp.asarray(inlet_total_mL_per_dL)
    flow = jnp.asarray(flow_mm3_per_ms, dtype=inlet.dtype)
    inlet_resolved = invert_oxygen_content(model.blood_model, inlet)
    flow_valid = (
        jnp.isfinite(flow)
        & (flow >= model.minimum_flow_mm3_per_ms)
        & (flow <= model.maximum_flow_mm3_per_ms)
    )
    valid = inlet_resolved.successful & flow_valid
    safe_flow = jnp.where(flow_valid, flow, model.minimum_flow_mm3_per_ms)
    equilibrium = evaluate_oxygen_content(
        model.blood_model,
        jnp.asarray(model.gas_partial_pressure_kPa, dtype=inlet.dtype),
    )
    effectiveness = 1.0 - jnp.exp(-model.transfer_capacity_mm3_per_ms / safe_flow)
    outlet_total = inlet + effectiveness * (equilibrium.total_mL_per_dL - inlet)
    outlet = invert_oxygen_content(model.blood_model, outlet_total)
    successful = valid & outlet.successful
    oxygen_transfer = flow * (outlet_total - inlet) / _MM3_PER_DL
    status = jnp.asarray(int(OxygenStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        inlet_resolved.successful,
        status,
        jnp.bitwise_or(status, int(OxygenStatus.OUT_OF_DOMAIN)),
    )
    status = jnp.where(
        flow_valid,
        status,
        jnp.bitwise_or(status, int(OxygenStatus.INVALID_FLOW)),
    )
    status = jnp.where(
        jnp.isfinite(inlet) & jnp.isfinite(flow),
        status,
        jnp.bitwise_or(status, int(OxygenStatus.NONFINITE)),
    )
    nan = jnp.asarray(jnp.nan, dtype=inlet.dtype)
    failed_outlet = OxygenContent(
        partial_pressure_kPa=jnp.where(
            successful, outlet.content.partial_pressure_kPa, nan
        ),
        saturation=jnp.where(successful, outlet.content.saturation, nan),
        dissolved_mL_per_dL=jnp.where(
            successful, outlet.content.dissolved_mL_per_dL, nan
        ),
        bound_mL_per_dL=jnp.where(successful, outlet.content.bound_mL_per_dL, nan),
        total_mL_per_dL=jnp.where(successful, outlet.content.total_mL_per_dL, nan),
        status=status,
        successful=successful,
    )
    return OxygenatorExchangeResult(
        outlet=failed_outlet,
        effectiveness=jnp.where(successful, effectiveness, nan),
        oxygen_transfer_mL_per_ms=jnp.where(successful, oxygen_transfer, nan),
        status=status,
        successful=successful,
    )


__all__ = [
    "BloodOxygenModel",
    "MembraneOxygenatorModel",
    "OxygenContent",
    "OxygenInversionResult",
    "OxygenMixingResult",
    "OxygenStatus",
    "OxygenTransportEvidence",
    "OxygenTransportInputs",
    "OxygenTransportPlan",
    "OxygenTransportResult",
    "OxygenTransportState",
    "OxygenatorExchangeResult",
    "evaluate_oxygen_content",
    "exchange_membrane_oxygen",
    "initialize_oxygen_transport_state",
    "invert_oxygen_content",
    "invert_oxygen_saturation",
    "mix_oxygen_content",
    "step_oxygen_transport",
]
