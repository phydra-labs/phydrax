#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


RegionName = Literal["negative", "separator", "positive"]
_FARADAY_C_MOL = 96485.33212


class ThroughCellRegionPlan(StrictModule, NonTrainableState):
    """One fixed cell-centred region of a three-region through-cell mesh."""

    cell_count: int = eqx.field(static=True)
    reference_faces: Array
    region: RegionName = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_count: int,
        /,
        *,
        region: RegionName,
        reference_faces: ArrayLike | None = None,
    ):
        if isinstance(cell_count, bool) or not isinstance(cell_count, int):
            raise TypeError("cell_count must be an integer.")
        if cell_count < 1:
            raise ValueError("cell_count must be positive.")
        if region not in ("negative", "separator", "positive"):
            raise ValueError("region must be 'negative', 'separator', or 'positive'.")
        if reference_faces is None:
            faces_host = np.linspace(0.0, 1.0, cell_count + 1)
        else:
            faces_host = np.asarray(reference_faces, dtype=float)
        if (
            faces_host.shape != (cell_count + 1,)
            or np.any(~np.isfinite(faces_host))
            or np.any(np.diff(faces_host) <= 0.0)
            or not np.isclose(faces_host[0], 0.0)
            or not np.isclose(faces_host[-1], 1.0)
        ):
            raise ValueError(
                "reference_faces must be finite, strictly increasing, span [0, 1], "
                "and match cell_count."
            )
        self.cell_count = cell_count
        self.reference_faces = jnp.asarray(faces_host)
        self.region = region
        self.plan_id = canonical_fingerprint(
            {
                "kind": "battery-through-cell-region-plan",
                "region": region,
                "cell_count": cell_count,
                "reference_faces": faces_host.tolist(),
            }
        )


class _ThroughCellMetrics(StrictModule):
    cell_widths_m: Array
    cell_centers_m: Array
    face_positions_m: Array
    porosity: Array
    effective_diffusivity_m2_s: Array
    storage_volume_m3: Array
    domain_valid: Array


class _ThroughCellEvaluation(StrictModule):
    concentration_mol_m3: Array
    face_concentration_mol_m3: Array
    diffusive_molar_flux_mol_m2_s: Array
    total_molar_flux_mol_m2_s: Array
    electrolyte_current_density_a_m2: Array
    solid_current_density_a_m2: Array
    source_mol_m3_s: Array
    amount_rate_mol_s: Array
    total_amount_mol: Array
    explicit_dt_limit_s: Array
    collector_flux_residual_mol_m2_s: Array
    negative_separator_concentration_jump_mol_m3: Array
    separator_positive_concentration_jump_mol_m3: Array
    equation_mapping_residual_mol_s: Array
    conservation_residual_mol_s: Array
    current_split_residual_a_m2: Array
    domain_valid: Array

    @property
    def successful(self) -> Array:
        return self.domain_valid


class PreparedThroughCellMesh(StrictModule, NonTrainableState):
    """Prepared three-region FV topology with dynamic SI geometry and coefficients.

    The differential state is extensive electrolyte amount in each cell. Cellwise
    diffusivity uses half-cell resistances at faces; cellwise transference is
    arithmetically reconstructed at faces and retained in conservative molar flux.
    """

    negative: ThroughCellRegionPlan
    separator: ThroughCellRegionPlan
    positive: ThroughCellRegionPlan
    reference_cell_widths: Array
    reference_cell_centers: Array
    cell_region_indices: Array
    negative_mask: Array
    separator_mask: Array
    positive_mask: Array
    electrolyte_current_fraction_faces: Array
    negative_separator_face_index: int = eqx.field(static=True)
    separator_positive_face_index: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        negative: ThroughCellRegionPlan,
        separator: ThroughCellRegionPlan,
        positive: ThroughCellRegionPlan,
        /,
    ):
        for supplied, expected in (
            (negative, "negative"),
            (separator, "separator"),
            (positive, "positive"),
        ):
            if not isinstance(supplied, ThroughCellRegionPlan):
                raise TypeError(
                    "Through-cell regions must be ThroughCellRegionPlan values."
                )
            if supplied.region != expected:
                raise ValueError(f"Expected the {expected!r} through-cell region.")

        plans = (negative, separator, positive)
        widths = jnp.concatenate(tuple(jnp.diff(plan.reference_faces) for plan in plans))
        centers = jnp.concatenate(
            tuple(
                0.5 * (plan.reference_faces[:-1] + plan.reference_faces[1:])
                for plan in plans
            )
        )
        region_indices = jnp.concatenate(
            tuple(
                jnp.full((plan.cell_count,), index, dtype=jnp.int32)
                for index, plan in enumerate(plans)
            )
        )
        negative_fraction = negative.reference_faces
        separator_fraction = jnp.ones((separator.cell_count,), dtype=widths.dtype)
        positive_fraction = 1.0 - positive.reference_faces[1:]
        current_fraction = jnp.concatenate(
            (negative_fraction, separator_fraction, positive_fraction)
        )
        negative_separator_face = negative.cell_count
        separator_positive_face = negative.cell_count + separator.cell_count

        self.negative = negative
        self.separator = separator
        self.positive = positive
        self.reference_cell_widths = widths
        self.reference_cell_centers = centers
        self.cell_region_indices = region_indices
        self.negative_mask = region_indices == 0
        self.separator_mask = region_indices == 1
        self.positive_mask = region_indices == 2
        self.electrolyte_current_fraction_faces = current_fraction
        self.negative_separator_face_index = negative_separator_face
        self.separator_positive_face_index = separator_positive_face
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-battery-through-cell-mesh",
                "regions": [plan.plan_id for plan in plans],
            }
        )

    @property
    def cell_count(self) -> int:
        return (
            self.negative.cell_count
            + self.separator.cell_count
            + self.positive.cell_count
        )

    def metrics(
        self,
        /,
        *,
        negative_thickness_m: ArrayLike,
        separator_thickness_m: ArrayLike,
        positive_thickness_m: ArrayLike,
        negative_porosity: ArrayLike,
        separator_porosity: ArrayLike,
        positive_porosity: ArrayLike,
        bruggeman_coefficient: ArrayLike,
        electrolyte_diffusivity_m2_s: ArrayLike,
        electrode_area_m2: ArrayLike,
    ) -> _ThroughCellMetrics:
        lengths = jnp.stack(
            tuple(
                jnp.asarray(value)
                for value in (
                    negative_thickness_m,
                    separator_thickness_m,
                    positive_thickness_m,
                )
            )
        )
        porosities = jnp.stack(
            tuple(
                jnp.asarray(value)
                for value in (negative_porosity, separator_porosity, positive_porosity)
            )
        )
        bruggeman = jnp.asarray(bruggeman_coefficient)
        diffusivity = jnp.asarray(electrolyte_diffusivity_m2_s)
        area = jnp.asarray(electrode_area_m2)
        if lengths.shape != (3,) or porosities.shape != (3,):
            raise ValueError(
                "Through-cell lengths and porosities must be scalar by region."
            )
        if bruggeman.shape != () or area.shape != ():
            raise ValueError(
                "Through-cell Bruggeman coefficient and area must be scalar."
            )
        if diffusivity.shape != () and diffusivity.shape[-1:] != (self.cell_count,):
            raise ValueError(
                "Electrolyte diffusivity must be scalar or have one value per cell."
            )

        diffusivity_valid = jnp.all(
            jnp.isfinite(diffusivity) & (diffusivity > 0.0),
            axis=-1 if diffusivity.ndim else None,
        )
        domain_valid = (
            jnp.all(jnp.isfinite(lengths) & (lengths > 0.0))
            & jnp.all(jnp.isfinite(porosities) & (porosities > 0.0) & (porosities <= 1.0))
            & jnp.isfinite(bruggeman)
            & (bruggeman > 0.0)
            & diffusivity_valid
            & jnp.isfinite(area)
            & (area > 0.0)
        )
        safe_lengths = jnp.where(jnp.isfinite(lengths) & (lengths > 0.0), lengths, 1.0)
        safe_porosity = jnp.where(
            jnp.isfinite(porosities) & (porosities > 0.0) & (porosities <= 1.0),
            porosities,
            1.0,
        )
        safe_bruggeman = jnp.where(
            jnp.isfinite(bruggeman) & (bruggeman > 0.0), bruggeman, 1.0
        )
        safe_diffusivity = jnp.where(
            jnp.isfinite(diffusivity) & (diffusivity > 0.0), diffusivity, 1.0
        )
        safe_area = jnp.where(jnp.isfinite(area) & (area > 0.0), area, 1.0)

        cell_lengths = safe_lengths[self.cell_region_indices]
        cell_porosity = safe_porosity[self.cell_region_indices]
        widths = self.reference_cell_widths * cell_lengths
        local_centers = self.reference_cell_centers * cell_lengths
        starts = jnp.stack(
            (jnp.asarray(0.0), safe_lengths[0], safe_lengths[0] + safe_lengths[1])
        )
        centers = local_centers + starts[self.cell_region_indices]
        negative_faces = self.negative.reference_faces * safe_lengths[0]
        separator_faces = (
            safe_lengths[0] + self.separator.reference_faces[1:] * safe_lengths[1]
        )
        positive_faces = (
            safe_lengths[0]
            + safe_lengths[1]
            + self.positive.reference_faces[1:] * safe_lengths[2]
        )
        face_positions = jnp.concatenate(
            (negative_faces, separator_faces, positive_faces)
        )
        effective_diffusivity = cell_porosity**safe_bruggeman * safe_diffusivity
        storage_volume = safe_area * cell_porosity * widths
        return _ThroughCellMetrics(
            widths,
            centers,
            face_positions,
            cell_porosity,
            effective_diffusivity,
            storage_volume,
            domain_valid,
        )

    def initial_amounts(
        self,
        concentration_mol_m3: ArrayLike,
        /,
        **metric_inputs: ArrayLike,
    ) -> Array:
        concentration = jnp.asarray(concentration_mol_m3)
        if concentration.shape != ():
            raise ValueError("Initial electrolyte concentration must be scalar.")
        metrics = self.metrics(**metric_inputs)
        concentration = eqx.error_if(
            concentration,
            ~(jnp.isfinite(concentration) & (concentration > 0.0) & metrics.domain_valid),
            "Through-cell initialization requires positive finite concentration and geometry.",
        )
        return concentration * metrics.storage_volume_m3

    def evaluate(
        self,
        amounts_mol: ArrayLike,
        terminal_current_a: ArrayLike,
        /,
        *,
        negative_thickness_m: ArrayLike,
        separator_thickness_m: ArrayLike,
        positive_thickness_m: ArrayLike,
        negative_porosity: ArrayLike,
        separator_porosity: ArrayLike,
        positive_porosity: ArrayLike,
        bruggeman_coefficient: ArrayLike,
        electrolyte_diffusivity_m2_s: ArrayLike,
        electrode_area_m2: ArrayLike,
        transference_number: ArrayLike,
        maximum_absolute_current_a: ArrayLike,
    ) -> _ThroughCellEvaluation:
        amounts = jnp.asarray(amounts_mol)
        if amounts.ndim < 1 or amounts.shape[-1] != self.cell_count:
            raise ValueError(
                f"amounts_mol must have trailing shape ({self.cell_count},)."
            )
        leading_shape = amounts.shape[:-1]
        current = jnp.asarray(terminal_current_a)
        if current.shape == ():
            current = jnp.broadcast_to(current, leading_shape)
        elif current.shape != leading_shape:
            raise ValueError(
                "terminal_current_a must be scalar or match amount leading axes."
            )
        transference = jnp.asarray(transference_number)
        maximum_current = jnp.asarray(maximum_absolute_current_a)
        if maximum_current.shape != ():
            raise ValueError("The current envelope must be scalar.")
        if transference.shape == ():
            transference_cells = jnp.broadcast_to(
                transference, leading_shape + (self.cell_count,)
            )
        elif transference.shape == leading_shape:
            transference_cells = jnp.broadcast_to(
                transference[..., None], leading_shape + (self.cell_count,)
            )
        elif transference.shape == leading_shape + (self.cell_count,):
            transference_cells = transference
        else:
            raise ValueError(
                "Transference number must be scalar, match leading axes, or have "
                "one value per through-cell control volume."
            )

        metrics = self.metrics(
            negative_thickness_m=negative_thickness_m,
            separator_thickness_m=separator_thickness_m,
            positive_thickness_m=positive_thickness_m,
            negative_porosity=negative_porosity,
            separator_porosity=separator_porosity,
            positive_porosity=positive_porosity,
            bruggeman_coefficient=bruggeman_coefficient,
            electrolyte_diffusivity_m2_s=electrolyte_diffusivity_m2_s,
            electrode_area_m2=electrode_area_m2,
        )
        area = jnp.asarray(electrode_area_m2)
        lengths = jnp.stack(
            (
                jnp.asarray(negative_thickness_m),
                jnp.asarray(separator_thickness_m),
                jnp.asarray(positive_thickness_m),
            )
        )
        transference_valid = jnp.all(
            jnp.isfinite(transference_cells)
            & (transference_cells >= 0.0)
            & (transference_cells <= 1.0),
            axis=-1,
        )
        current_valid = (
            jnp.isfinite(maximum_current)
            & (maximum_current > 0.0)
            & jnp.isfinite(current)
            & (jnp.abs(current) <= maximum_current)
        )
        state_valid = jnp.all(jnp.isfinite(amounts) & (amounts >= 0.0), axis=-1)
        active = metrics.domain_valid & transference_valid & current_valid & state_valid
        safe_amounts = jnp.where(jnp.isfinite(amounts), amounts, 0.0)
        safe_transference_cells = jnp.where(
            transference_valid[..., None], transference_cells, 0.0
        )
        safe_current = jnp.where(current_valid & metrics.domain_valid, current, 0.0)
        concentration = safe_amounts / metrics.storage_volume_m3

        left_width = metrics.cell_widths_m[:-1]
        right_width = metrics.cell_widths_m[1:]
        left_diffusivity = metrics.effective_diffusivity_m2_s[..., :-1]
        right_diffusivity = metrics.effective_diffusivity_m2_s[..., 1:]
        resistance = (
            0.5 * left_width / left_diffusivity + 0.5 * right_width / right_diffusivity
        )
        conductance = 1.0 / resistance
        interior_diffusive_flux = -conductance * (
            concentration[..., 1:] - concentration[..., :-1]
        )
        zeros = jnp.zeros(leading_shape + (1,), dtype=concentration.dtype)
        diffusive_flux = jnp.concatenate((zeros, interior_diffusive_flux, zeros), axis=-1)

        left_trace = concentration[..., :-1] - (
            interior_diffusive_flux * 0.5 * left_width / left_diffusivity
        )
        right_trace = concentration[..., 1:] + (
            interior_diffusive_flux * 0.5 * right_width / right_diffusivity
        )
        interior_face_concentration = 0.5 * (left_trace + right_trace)
        face_concentration = jnp.concatenate(
            (
                concentration[..., :1],
                interior_face_concentration,
                concentration[..., -1:],
            ),
            axis=-1,
        )

        # The paper current is positive on discharge; the platform terminal current is
        # passive, hence the minus sign. Fractions are Eq. 48d's exact current split.
        paper_current_density = -safe_current / area
        electrolyte_current = (
            paper_current_density[..., None] * self.electrolyte_current_fraction_faces
        )
        solid_current = paper_current_density[..., None] - electrolyte_current
        transference_faces = jnp.concatenate(
            (
                safe_transference_cells[..., :1],
                0.5
                * (safe_transference_cells[..., :-1] + safe_transference_cells[..., 1:]),
                safe_transference_cells[..., -1:],
            ),
            axis=-1,
        )
        migration_flux = transference_faces * electrolyte_current / _FARADAY_C_MOL
        total_flux = diffusive_flux + migration_flux

        full_reaction_source = jnp.where(
            self.negative_mask,
            paper_current_density[..., None] / (_FARADAY_C_MOL * lengths[0]),
            jnp.where(
                self.positive_mask,
                -paper_current_density[..., None] / (_FARADAY_C_MOL * lengths[2]),
                0.0,
            ),
        )
        source = (
            full_reaction_source
            + (migration_flux[..., :-1] - migration_flux[..., 1:]) / metrics.cell_widths_m
        )
        amount_rate = area * (
            diffusive_flux[..., :-1]
            - diffusive_flux[..., 1:]
            + source * metrics.cell_widths_m
        )
        full_equation_rate = area * (
            total_flux[..., :-1]
            - total_flux[..., 1:]
            + full_reaction_source * metrics.cell_widths_m
        )
        equation_mapping_residual = jnp.max(
            jnp.abs(amount_rate - full_equation_rate), axis=-1
        )
        conservation_residual = jnp.sum(amount_rate, axis=-1) - area * (
            total_flux[..., 0]
            - total_flux[..., -1]
            + jnp.sum(full_reaction_source * metrics.cell_widths_m, axis=-1)
        )

        face_zeros = jnp.zeros(conductance.shape[:-1] + (1,), dtype=conductance.dtype)
        face_conductance = jnp.concatenate((face_zeros, conductance, face_zeros), axis=-1)
        loss_rate = (face_conductance[..., :-1] + face_conductance[..., 1:]) / (
            metrics.porosity * metrics.cell_widths_m
        )
        explicit_dt_limit = jnp.min(
            jnp.where(loss_rate > 0.0, 1.0 / loss_rate, jnp.inf),
            axis=-1,
        )

        negative_separator_face = self.negative_separator_face_index
        separator_positive_face = self.separator_positive_face_index
        negative_separator_jump = (
            left_trace[..., negative_separator_face - 1]
            - right_trace[..., negative_separator_face - 1]
        )
        separator_positive_jump = (
            left_trace[..., separator_positive_face - 1]
            - right_trace[..., separator_positive_face - 1]
        )
        collector_flux_residual = jnp.maximum(
            jnp.abs(total_flux[..., 0]), jnp.abs(total_flux[..., -1])
        )
        current_split_residual = jnp.max(
            jnp.abs(
                electrolyte_current + solid_current - paper_current_density[..., None]
            ),
            axis=-1,
        )
        finite_output = (
            jnp.all(jnp.isfinite(concentration), axis=-1)
            & jnp.all(jnp.isfinite(face_concentration), axis=-1)
            & jnp.all(jnp.isfinite(diffusive_flux), axis=-1)
            & jnp.all(jnp.isfinite(total_flux), axis=-1)
            & jnp.all(jnp.isfinite(amount_rate), axis=-1)
            & jnp.isfinite(explicit_dt_limit)
            & jnp.isfinite(equation_mapping_residual)
            & jnp.isfinite(conservation_residual)
        )
        reconstructed_positive = jnp.all(concentration > 0.0, axis=-1) & jnp.all(
            face_concentration > 0.0, axis=-1
        )
        domain_valid = active & finite_output & reconstructed_positive
        return _ThroughCellEvaluation(
            concentration,
            face_concentration,
            diffusive_flux,
            total_flux,
            electrolyte_current,
            solid_current,
            source,
            amount_rate,
            jnp.sum(safe_amounts, axis=-1),
            explicit_dt_limit,
            collector_flux_residual,
            negative_separator_jump,
            separator_positive_jump,
            equation_mapping_residual,
            conservation_residual,
            current_split_residual,
            domain_valid,
        )


__all__ = ["PreparedThroughCellMesh", "ThroughCellRegionPlan"]
