#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._coordinates import HarmonicCoordinateFields


def _nonempty(value: str, description: str, /) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{description} must be non-empty.")
    return result


def _normalized_or_invalid(
    vectors: Array, tolerance: float, /
) -> tuple[Array, Array, Array]:
    norms = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
    valid = (
        jnp.all(jnp.isfinite(vectors), axis=-1)
        & jnp.isfinite(norms)
        & (norms > tolerance)
    )
    # One is only an inactive arithmetic denominator; invalid rows are made NaN.
    denominator = jnp.where(valid, norms, jnp.ones_like(norms))
    normalized = vectors / denominator[..., None]
    normalized = jnp.where(valid[..., None], normalized, jnp.nan)
    return normalized, norms, valid


class VentricularLineField(StrictModule):
    """An unoriented ventricular line represented by direction and f⊗f tensor."""

    direction: Array
    structure_tensor: Array
    valid: Array
    line_id: str = eqx.field(static=True)

    def __init__(
        self,
        direction: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        line_id: str,
    ):
        direction_ = jnp.asarray(direction)
        valid_ = jnp.asarray(valid, dtype=bool)
        if direction_.ndim != 2 or direction_.shape[-1] != 3:
            raise ValueError("Line directions must have shape (cell_count, 3).")
        if valid_.shape != direction_.shape[:-1]:
            raise ValueError("Line validity must have shape (cell_count,).")
        self.direction = direction_
        self.structure_tensor = oe.contract("ci,cj->cij", direction_, direction_)
        self.valid = valid_
        self.line_id = _nonempty(line_id, "line_id")


class CardiacMaterialFrame(StrictModule):
    """Complete right-handed cellwise fiber/sheet/sheet-normal frame."""

    fiber: Array
    sheet: Array
    sheet_normal: Array
    valid: Array
    frame_id: str = eqx.field(static=True)

    def __init__(
        self,
        fiber: ArrayLike,
        sheet: ArrayLike,
        sheet_normal: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        frame_id: str,
    ):
        fiber_ = jnp.asarray(fiber)
        sheet_ = jnp.asarray(sheet, dtype=fiber_.dtype)
        normal_ = jnp.asarray(sheet_normal, dtype=fiber_.dtype)
        valid_ = jnp.asarray(valid, dtype=bool)
        if fiber_.ndim != 2 or fiber_.shape[-1] != 3:
            raise ValueError("Material-frame vectors must have shape (cell_count, 3).")
        if sheet_.shape != fiber_.shape or normal_.shape != fiber_.shape:
            raise ValueError("Fiber, sheet, and sheet-normal fields must share shape.")
        if valid_.shape != fiber_.shape[:-1]:
            raise ValueError("Material-frame validity must have shape (cell_count,).")
        self.fiber = fiber_
        self.sheet = sheet_
        self.sheet_normal = normal_
        self.valid = valid_
        self.frame_id = _nonempty(frame_id, "frame_id")

    @property
    def matrix(self) -> Array:
        """Return frames with ordered columns (fiber, sheet, sheet-normal)."""
        return jnp.stack((self.fiber, self.sheet, self.sheet_normal), axis=-1)


class VentricularMicrostructureEvidence(StrictModule, NonTrainableState):
    """Full degeneracy, orthonormality, and line-tensor qualification evidence."""

    transmural_gradient_norm: Array
    projected_longitudinal_gradient_norm: Array
    orthonormality_error: Array
    orientation_determinant: Array
    tensor_symmetry_error: Array
    transmural_fraction_in_range: Array
    finite: Array
    nondegenerate: Array
    successful: Array

    def __init__(self, **values):
        self.transmural_gradient_norm = jnp.asarray(values["transmural_gradient_norm"])
        self.projected_longitudinal_gradient_norm = jnp.asarray(
            values["projected_longitudinal_gradient_norm"]
        )
        self.orthonormality_error = jnp.asarray(values["orthonormality_error"])
        self.orientation_determinant = jnp.asarray(values["orientation_determinant"])
        self.tensor_symmetry_error = jnp.asarray(values["tensor_symmetry_error"])
        self.transmural_fraction_in_range = jnp.asarray(
            values["transmural_fraction_in_range"], dtype=bool
        )
        self.finite = jnp.asarray(values["finite"], dtype=bool)
        self.nondegenerate = jnp.asarray(values["nondegenerate"], dtype=bool)
        self.successful = jnp.asarray(values["successful"], dtype=bool)
        shape = self.transmural_gradient_norm.shape
        fields = (
            self.projected_longitudinal_gradient_norm,
            self.orthonormality_error,
            self.orientation_determinant,
            self.tensor_symmetry_error,
            self.transmural_fraction_in_range,
            self.finite,
            self.nondegenerate,
            self.successful,
        )
        if len(shape) != 1 or any(field.shape != shape for field in fields):
            raise ValueError(
                "Microstructure evidence must use one cellwise vector shape."
            )

    @property
    def all_successful(self) -> Array:
        return jnp.all(self.successful)


class VentricularMicrostructure(StrictModule):
    """Committed ventricular basis, helix rule, physical line, and material frame."""

    transmural_direction: Array
    longitudinal_direction: Array
    circumferential_direction: Array
    transmural_fraction: Array
    helix_angle_radians: Array
    fiber_line: VentricularLineField
    material_frame: CardiacMaterialFrame
    evidence: VentricularMicrostructureEvidence
    microstructure_id: str = eqx.field(static=True)

    @property
    def fiber(self) -> Array:
        return self.material_frame.fiber

    @property
    def sheet(self) -> Array:
        return self.material_frame.sheet

    @property
    def sheet_normal(self) -> Array:
        return self.material_frame.sheet_normal

    @property
    def fiber_structure_tensor(self) -> Array:
        return self.fiber_line.structure_tensor


class VentricularMicrostructureCandidate(StrictModule):
    """Uncommitted microstructure retaining invalid cells as explicit NaNs."""

    transmural_direction: Array
    longitudinal_direction: Array
    circumferential_direction: Array
    transmural_fraction: Array
    helix_angle_radians: Array
    fiber_line: VentricularLineField
    material_frame: CardiacMaterialFrame
    evidence: VentricularMicrostructureEvidence
    candidate_id: str = eqx.field(static=True)

    def commit(self, /) -> VentricularMicrostructure:
        checked_fiber = eqx.error_if(
            self.material_frame.fiber,
            ~self.evidence.all_successful,
            "Cannot commit degenerate ventricular microstructure.",
        )
        frame = CardiacMaterialFrame(
            checked_fiber,
            self.material_frame.sheet,
            self.material_frame.sheet_normal,
            self.material_frame.valid,
            frame_id=self.material_frame.frame_id,
        )
        line = VentricularLineField(
            checked_fiber,
            self.fiber_line.valid,
            line_id=self.fiber_line.line_id,
        )
        return VentricularMicrostructure(
            self.transmural_direction,
            self.longitudinal_direction,
            self.circumferential_direction,
            self.transmural_fraction,
            self.helix_angle_radians,
            line,
            frame,
            self.evidence,
            microstructure_id=canonical_fingerprint(
                {
                    "kind": "committed-ventricular-microstructure",
                    "candidate": self.candidate_id,
                }
            ),
        )


class VentricularMicrostructurePlan(StrictModule, NonTrainableState):
    """Exact gradient/projection/linear-helix ventricular construction plan."""

    transmural_coordinate: str = eqx.field(static=True)
    longitudinal_coordinate: str = eqx.field(static=True)
    transmural_endocardium: float = eqx.field(static=True)
    transmural_epicardium: float = eqx.field(static=True)
    helix_endocardium_radians: float = eqx.field(static=True)
    helix_epicardium_radians: float = eqx.field(static=True)
    gradient_tolerance: float = eqx.field(static=True)
    orthonormality_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transmural_coordinate: str,
        longitudinal_coordinate: str,
        /,
        *,
        transmural_endocardium: float = 0.0,
        transmural_epicardium: float = 1.0,
        helix_endocardium_degrees: float = 60.0,
        helix_epicardium_degrees: float = -60.0,
        gradient_tolerance: float = 0.0,
        orthonormality_tolerance: float = 1.0e-6,
    ):
        transmural = _nonempty(transmural_coordinate, "Transmural coordinate")
        longitudinal = _nonempty(longitudinal_coordinate, "Longitudinal coordinate")
        if transmural == longitudinal:
            raise ValueError("Transmural and longitudinal coordinates must be distinct.")
        endocardium = float(transmural_endocardium)
        epicardium = float(transmural_epicardium)
        angles = (
            math.radians(float(helix_endocardium_degrees)),
            math.radians(float(helix_epicardium_degrees)),
        )
        gradient_tolerance_ = float(gradient_tolerance)
        orthonormality_tolerance_ = float(orthonormality_tolerance)
        scalars = (
            endocardium,
            epicardium,
            *angles,
            gradient_tolerance_,
            orthonormality_tolerance_,
        )
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError("Microstructure plan scalars must be finite.")
        if endocardium == epicardium:
            raise ValueError("Transmural endpoint values must be distinct.")
        if gradient_tolerance_ < 0.0 or orthonormality_tolerance_ <= 0.0:
            raise ValueError("Microstructure tolerances must be non-negative/positive.")
        self.transmural_coordinate = transmural
        self.longitudinal_coordinate = longitudinal
        self.transmural_endocardium = endocardium
        self.transmural_epicardium = epicardium
        self.helix_endocardium_radians = angles[0]
        self.helix_epicardium_radians = angles[1]
        self.gradient_tolerance = gradient_tolerance_
        self.orthonormality_tolerance = orthonormality_tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ventricular-microstructure-plan",
                "transmural": transmural,
                "longitudinal": longitudinal,
                "transmural_endocardium": endocardium,
                "transmural_epicardium": epicardium,
                "helix_endocardium_radians": angles[0],
                "helix_epicardium_radians": angles[1],
                "gradient_tolerance": gradient_tolerance_,
                "orthonormality_tolerance": orthonormality_tolerance_,
            }
        )

    def prepare(
        self, fields: HarmonicCoordinateFields, /
    ) -> PreparedVentricularMicrostructure:
        return PreparedVentricularMicrostructure(self, fields)


class PreparedVentricularMicrostructure(StrictModule, NonTrainableState):
    """Coordinate-bound, fixed-cell-route ventricular microstructure builder."""

    plan: VentricularMicrostructurePlan
    fields: HarmonicCoordinateFields
    transmural_index: int = eqx.field(static=True)
    longitudinal_index: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: VentricularMicrostructurePlan,
        fields: HarmonicCoordinateFields,
        /,
    ):
        if not isinstance(plan, VentricularMicrostructurePlan):
            raise TypeError("plan must be VentricularMicrostructurePlan.")
        if not isinstance(fields, HarmonicCoordinateFields):
            raise TypeError("fields must be HarmonicCoordinateFields.")
        self.plan = plan
        self.fields = fields
        self.transmural_index = fields.coordinate_index(plan.transmural_coordinate)
        self.longitudinal_index = fields.coordinate_index(plan.longitudinal_coordinate)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-ventricular-microstructure",
                "plan": plan.plan_id,
                "coordinates": fields.fields_id,
            }
        )

    def build(self, /) -> VentricularMicrostructureCandidate:
        plan = self.plan
        transmural_gradient = self.fields.cell_gradients[self.transmural_index]
        longitudinal_gradient = self.fields.cell_gradients[self.longitudinal_index]
        transmural, transmural_norm, transmural_valid = _normalized_or_invalid(
            transmural_gradient, plan.gradient_tolerance
        )
        projection = (
            longitudinal_gradient
            - oe.contract("ci,ci->c", longitudinal_gradient, transmural)[:, None]
            * transmural
        )
        longitudinal, projected_norm, longitudinal_valid = _normalized_or_invalid(
            projection, plan.gradient_tolerance
        )
        circumferential = jnp.cross(longitudinal, transmural)
        circumferential, _, circumferential_valid = _normalized_or_invalid(
            circumferential, plan.gradient_tolerance
        )
        phi = self.fields.cell_values[self.transmural_index]
        fraction = (phi - plan.transmural_endocardium) / (
            plan.transmural_epicardium - plan.transmural_endocardium
        )
        helix = plan.helix_endocardium_radians + fraction * (
            plan.helix_epicardium_radians - plan.helix_endocardium_radians
        )
        fiber = (
            jnp.cos(helix)[:, None] * circumferential
            + jnp.sin(helix)[:, None] * longitudinal
        )
        sheet = transmural
        sheet_normal = jnp.cross(fiber, sheet)
        nondegenerate = transmural_valid & longitudinal_valid & circumferential_valid
        frame_matrix = jnp.stack((fiber, sheet, sheet_normal), axis=-1)
        gram = oe.contract("cji,cjk->cik", frame_matrix, frame_matrix)
        identity = jnp.eye(3, dtype=frame_matrix.dtype)
        orthonormality_error = jnp.max(jnp.abs(gram - identity), axis=(-2, -1))
        orientation = oe.contract("ci,ci->c", jnp.cross(fiber, sheet), sheet_normal)
        tensor = oe.contract("ci,cj->cij", fiber, fiber)
        tensor_symmetry_error = jnp.max(
            jnp.abs(tensor - jnp.swapaxes(tensor, -1, -2)), axis=(-2, -1)
        )
        range_tolerance = 128.0 * jnp.finfo(phi.dtype).eps
        in_range = (fraction >= -range_tolerance) & (fraction <= 1.0 + range_tolerance)
        finite = (
            jnp.all(jnp.isfinite(frame_matrix), axis=(-2, -1))
            & jnp.isfinite(helix)
            & jnp.isfinite(orthonormality_error)
            & jnp.isfinite(orientation)
            & jnp.isfinite(tensor_symmetry_error)
        )
        successful = (
            nondegenerate
            & finite
            & in_range
            & (orthonormality_error <= plan.orthonormality_tolerance)
            & (orientation > 0.0)
            & (tensor_symmetry_error <= plan.orthonormality_tolerance)
        )
        fiber = jnp.where(successful[:, None], fiber, jnp.nan)
        sheet = jnp.where(successful[:, None], sheet, jnp.nan)
        sheet_normal = jnp.where(successful[:, None], sheet_normal, jnp.nan)
        circumferential = jnp.where(successful[:, None], circumferential, jnp.nan)
        longitudinal = jnp.where(successful[:, None], longitudinal, jnp.nan)
        transmural = jnp.where(successful[:, None], transmural, jnp.nan)
        line_id = canonical_fingerprint(
            {"kind": "ventricular-fiber-line", "prepared": self.prepared_id}
        )
        frame_id = canonical_fingerprint(
            {"kind": "cardiac-material-frame", "prepared": self.prepared_id}
        )
        line = VentricularLineField(fiber, successful, line_id=line_id)
        frame = CardiacMaterialFrame(
            fiber, sheet, sheet_normal, successful, frame_id=frame_id
        )
        evidence = VentricularMicrostructureEvidence(
            transmural_gradient_norm=transmural_norm,
            projected_longitudinal_gradient_norm=projected_norm,
            orthonormality_error=orthonormality_error,
            orientation_determinant=orientation,
            tensor_symmetry_error=tensor_symmetry_error,
            transmural_fraction_in_range=in_range,
            finite=finite,
            nondegenerate=nondegenerate,
            successful=successful,
        )
        return VentricularMicrostructureCandidate(
            transmural,
            longitudinal,
            circumferential,
            fraction,
            helix,
            line,
            frame,
            evidence,
            candidate_id=canonical_fingerprint(
                {
                    "kind": "ventricular-microstructure-candidate",
                    "prepared": self.prepared_id,
                }
            ),
        )


def build_ventricular_microstructure(
    plan: VentricularMicrostructurePlan,
    fields: HarmonicCoordinateFields,
    /,
) -> VentricularMicrostructureCandidate:
    """Prepare and build a fail-closed ventricular microstructure candidate."""
    return plan.prepare(fields).build()


__all__ = [
    "CardiacMaterialFrame",
    "PreparedVentricularMicrostructure",
    "VentricularLineField",
    "VentricularMicrostructure",
    "VentricularMicrostructureCandidate",
    "VentricularMicrostructureEvidence",
    "VentricularMicrostructurePlan",
    "build_ventricular_microstructure",
]
