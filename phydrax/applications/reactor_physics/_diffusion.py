#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One-dimensional cell-centered multigroup neutron diffusion and criticality."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume import MetricLinePlan, PreparedMetricLine
from ...linalg import (
    ArraySpace,
    DifferentiationPolicy,
    FailurePolicy,
    GMRES,
    JacobiPreconditionerBuilder,
    LinearSolvePolicy,
    LinearSystem,
    PreconditioningPolicy,
    solve_checked,
    TolerancePolicy,
)
from ...nuclear import EnergyGroupStructure, NuclearDataProvenance
from ...sparse import EdgeRelation, SparseCoordinateOperator


@dataclass(frozen=True, slots=True)
class MultigroupMaterialData:
    energy_groups: EnergyGroupStructure
    diffusion_coefficient_m: np.ndarray
    removal_cross_section_m1: np.ndarray
    scattering_cross_section_m1: np.ndarray
    nu_fission_cross_section_m1: np.ndarray
    fission_spectrum: np.ndarray
    data: NuclearDataProvenance
    material_id: str
    data_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.energy_groups, EnergyGroupStructure):
            raise TypeError("energy_groups must be EnergyGroupStructure.")
        arrays = tuple(
            np.array(value, dtype=np.float64, copy=True)
            for value in (
                self.diffusion_coefficient_m,
                self.removal_cross_section_m1,
                self.scattering_cross_section_m1,
                self.nu_fission_cross_section_m1,
                self.fission_spectrum,
            )
        )
        diffusion, removal, scattering, fission, spectrum = arrays
        if diffusion.ndim != 2:
            raise ValueError("Multigroup diffusion data require shape (cell, group).")
        cells, groups = diffusion.shape
        if (
            groups != self.energy_groups.group_count
            or removal.shape != diffusion.shape
            or fission.shape != diffusion.shape
            or spectrum.shape != diffusion.shape
            or scattering.shape != (cells, groups, groups)
        ):
            raise ValueError(
                "Multigroup material arrays have inconsistent cell/group shapes."
            )
        if any(np.any(~np.isfinite(value)) for value in arrays):
            raise ValueError("Multigroup material data must be finite.")
        if (
            np.any(diffusion <= 0.0)
            or np.any(removal < 0.0)
            or np.any(scattering < 0.0)
            or np.any(fission < 0.0)
            or np.any(spectrum < 0.0)
        ):
            raise ValueError("Multigroup coefficients violate nonnegativity.")
        diagonal = np.arange(groups)
        if np.any(scattering[:, diagonal, diagonal] != 0.0):
            raise ValueError(
                "Within-group scattering belongs in removal, not transfer data."
            )
        fissioning = np.sum(fission, axis=1) > 0.0
        if np.any(
            fissioning
            & ~np.isclose(np.sum(spectrum, axis=1), 1.0, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError("Fission spectra must sum to one in fissioning cells.")
        if np.any(
            ~fissioning
            & ~np.isclose(np.sum(spectrum, axis=1), 0.0, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError("Nonfissioning cells require a zero fission spectrum.")
        if not isinstance(self.data, NuclearDataProvenance):
            raise TypeError("data must be NuclearDataProvenance.")
        material = str(self.material_id).strip()
        if not material or material != self.material_id:
            raise ValueError("material_id must be non-empty canonical text.")
        for name, value in (
            ("diffusion_coefficient_m", diffusion),
            ("removal_cross_section_m1", removal),
            ("scattering_cross_section_m1", scattering),
            ("nu_fission_cross_section_m1", fission),
            ("fission_spectrum", spectrum),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "material_id", material)
        object.__setattr__(
            self,
            "data_id",
            canonical_fingerprint(
                {
                    "kind": "multigroup-material-data",
                    "groups": self.energy_groups.group_id,
                    "arrays": [array_tree_fingerprint(value) for value in arrays],
                    "data": self.data.provenance_id,
                    "material": material,
                }
            ),
        )


class ReactorDiffusionSolveResult(StrictModule):
    scalar_flux_m2_s: Array
    residual_norm: Array
    finite: Array
    domain_valid: Array
    sensitivity_valid: Array
    successful: Array


class ReactorCriticalityResult(StrictModule):
    scalar_flux: Array
    k_effective: Array
    residual_norm: Array
    iteration_count: Array
    finite: Array
    converged: Array
    sensitivity_valid: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class MultigroupDiffusionPlan:
    geometry: MetricLinePlan
    material: MultigroupMaterialData
    relative_tolerance: float = 1.0e-9
    maximum_linear_steps: int = 512
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, MetricLinePlan):
            raise TypeError("geometry must be MetricLinePlan.")
        if not isinstance(self.material, MultigroupMaterialData):
            raise TypeError("material must be MultigroupMaterialData.")
        if (
            self.geometry.cell_measures.shape[0]
            != self.material.diffusion_coefficient_m.shape[0]
        ):
            raise ValueError("Reactor geometry and material cell counts disagree.")
        tolerance = float(self.relative_tolerance)
        steps = int(self.maximum_linear_steps)
        if not math.isfinite(tolerance) or tolerance <= 0.0 or steps < 1:
            raise ValueError("Reactor diffusion solve controls are invalid.")
        object.__setattr__(self, "relative_tolerance", tolerance)
        object.__setattr__(self, "maximum_linear_steps", steps)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "multigroup-diffusion-plan",
                    "geometry": self.geometry.plan_id,
                    "material": self.material.data_id,
                    "boundary": "zero-dirichlet-vacuum",
                    "relative_tolerance": tolerance,
                    "maximum_linear_steps": steps,
                }
            ),
        )

    def prepare(self) -> PreparedMultigroupDiffusion:
        geometry = self.geometry.prepare()
        cells, groups = self.material.diffusion_coefficient_m.shape
        size = cells * groups
        centers = np.asarray(geometry.coordinate_cells)
        faces = np.asarray(geometry.coordinate_faces)
        area = np.asarray(geometry.face_measures)
        volume = np.asarray(geometry.cell_measures)
        diffusion = self.material.diffusion_coefficient_m
        matrix = np.zeros((size, size), dtype=np.float64)

        def index(cell, group):
            return cell * groups + group

        for cell in range(cells):
            for group in range(groups):
                row = index(cell, group)
                left_distance = (
                    centers[cell] - faces[cell]
                    if cell == 0
                    else centers[cell] - centers[cell - 1]
                )
                right_distance = (
                    faces[cell + 1] - centers[cell]
                    if cell == cells - 1
                    else centers[cell + 1] - centers[cell]
                )
                left_diffusion = (
                    diffusion[cell, group]
                    if cell == 0
                    else 2.0
                    * diffusion[cell - 1, group]
                    * diffusion[cell, group]
                    / (diffusion[cell - 1, group] + diffusion[cell, group])
                )
                right_diffusion = (
                    diffusion[cell, group]
                    if cell == cells - 1
                    else 2.0
                    * diffusion[cell, group]
                    * diffusion[cell + 1, group]
                    / (diffusion[cell, group] + diffusion[cell + 1, group])
                )
                left_conductance = area[cell] * left_diffusion / left_distance
                right_conductance = area[cell + 1] * right_diffusion / right_distance
                matrix[row, row] += (
                    left_conductance
                    + right_conductance
                    + self.material.removal_cross_section_m1[cell, group] * volume[cell]
                )
                if cell > 0:
                    matrix[row, index(cell - 1, group)] -= left_conductance
                if cell < cells - 1:
                    matrix[row, index(cell + 1, group)] -= right_conductance
                for source_group in range(groups):
                    if source_group != group:
                        matrix[row, index(cell, source_group)] -= (
                            self.material.scattering_cross_section_m1[
                                cell, source_group, group
                            ]
                            * volume[cell]
                        )
        fission = np.zeros_like(matrix)
        for cell in range(cells):
            for target_group in range(groups):
                for source_group in range(groups):
                    fission[index(cell, target_group), index(cell, source_group)] = (
                        self.material.fission_spectrum[cell, target_group]
                        * self.material.nu_fission_cross_section_m1[cell, source_group]
                        * volume[cell]
                    )
        rows, columns = np.nonzero(matrix)
        relation = EdgeRelation(
            jnp.asarray(columns, dtype=jnp.int32),
            jnp.asarray(rows, dtype=jnp.int32),
            source_size=size,
            target_size=size,
        )
        space = ArraySpace((size,), dtype=np.float64)
        operator = SparseCoordinateOperator(
            relation,
            jnp.asarray(matrix[rows, columns]),
            source=space,
            target=space,
            operator_id=self.plan_id,
        )
        fission_operator = SparseCoordinateOperator(
            EdgeRelation(
                jnp.asarray(np.nonzero(fission)[1], dtype=jnp.int32),
                jnp.asarray(np.nonzero(fission)[0], dtype=jnp.int32),
                source_size=size,
                target_size=size,
            ),
            jnp.asarray(fission[np.nonzero(fission)]),
            source=space,
            target=space,
            operator_id=canonical_fingerprint(
                {"kind": "reactor-fission-operator", "plan": self.plan_id}
            ),
        )
        policy = LinearSolvePolicy(
            GMRES(restart=min(64, size)),
            tolerance=TolerancePolicy(
                relative=self.relative_tolerance,
                absolute=self.relative_tolerance,
                max_steps=self.maximum_linear_steps,
            ),
            preconditioning=PreconditioningPolicy(JacobiPreconditionerBuilder()),
            differentiation=DifferentiationPolicy("mathematical"),
            failure=FailurePolicy("status"),
        )
        return PreparedMultigroupDiffusion(
            geometry,
            operator,
            fission_operator,
            jnp.asarray(volume),
            groups,
            self.material.energy_groups.group_id,
            policy,
            self.plan_id,
        )


class PreparedMultigroupDiffusion(StrictModule, NonTrainableState):
    geometry: PreparedMetricLine
    loss_operator: SparseCoordinateOperator
    fission_operator: SparseCoordinateOperator
    cell_volume_m3: Array
    group_count: int = eqx.field(static=True)
    energy_group_id: str = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    @property
    def cell_count(self) -> int:
        return int(self.cell_volume_m3.size)

    def solve_fixed_source(
        self, source_m3_s: ArrayLike, /
    ) -> ReactorDiffusionSolveResult:
        source = jnp.asarray(source_m3_s, dtype=jnp.float64)
        if source.shape != (self.cell_count, self.group_count):
            raise ValueError("Fixed neutron source must have shape (cell, group).")
        domain_valid = jnp.all(jnp.isfinite(source)) & jnp.all(source >= 0.0)
        rhs = (source * self.cell_volume_m3[:, None]).reshape((-1,))
        linear, evidence = solve_checked(
            LinearSystem(self.loss_operator), rhs, policy=self.linear_policy
        )
        flux = linear.value.reshape(source.shape)
        residual = self.loss_operator.mv(linear.value) - rhs
        residual_norm = jnp.sqrt(jnp.vdot(residual, residual))
        finite = jnp.all(jnp.isfinite(flux)) & jnp.isfinite(residual_norm)
        successful = (
            domain_valid
            & linear.successful
            & evidence.valid
            & finite
            & jnp.all(flux >= 0.0)
        )
        return ReactorDiffusionSolveResult(
            flux, residual_norm, finite, domain_valid, successful, successful
        )

    def solve_criticality(
        self,
        initial_flux: ArrayLike,
        /,
        *,
        maximum_iterations: int = 64,
        residual_tolerance: float = 1.0e-8,
    ) -> ReactorCriticalityResult:
        flux = jnp.asarray(initial_flux, dtype=jnp.float64)
        if flux.shape != (self.cell_count, self.group_count):
            raise ValueError("Initial criticality flux must have shape (cell, group).")
        iterations = int(maximum_iterations)
        tolerance = float(residual_tolerance)
        if iterations < 1 or not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Criticality iteration controls are invalid.")
        domain_valid = jnp.all(jnp.isfinite(flux)) & jnp.all(flux > 0.0)
        vector = flux.reshape((-1,))
        initial_source = self.fission_operator.mv(vector)
        normalization = jnp.sum(initial_source)
        vector = vector / jnp.where(normalization > 0.0, normalization, 1.0)
        all_linear_successful = jnp.asarray(True)
        k_effective = jnp.asarray(1.0, dtype=vector.dtype)
        for _ in range(iterations):
            source = self.fission_operator.mv(vector)
            linear, evidence = solve_checked(
                LinearSystem(self.loss_operator), source, policy=self.linear_policy
            )
            new_source = self.fission_operator.mv(linear.value)
            k_effective = jnp.sum(new_source)
            vector = linear.value / jnp.where(k_effective > 0.0, k_effective, 1.0)
            all_linear_successful = (
                all_linear_successful & linear.successful & evidence.valid
            )
        source = self.fission_operator.mv(vector)
        residual = self.loss_operator.mv(vector) - source / k_effective
        residual_norm = jnp.sqrt(jnp.vdot(residual, residual))
        finite = (
            jnp.all(jnp.isfinite(vector))
            & jnp.isfinite(k_effective)
            & jnp.isfinite(residual_norm)
        )
        converged = residual_norm <= tolerance * jnp.maximum(
            1.0, jnp.sqrt(jnp.vdot(source, source))
        )
        successful = (
            domain_valid
            & all_linear_successful
            & finite
            & (k_effective > 0.0)
            & jnp.all(vector >= 0.0)
            & converged
        )
        return ReactorCriticalityResult(
            vector.reshape((self.cell_count, self.group_count)),
            k_effective,
            residual_norm,
            jnp.asarray(iterations, dtype=jnp.int32),
            finite,
            converged,
            successful,
            successful,
        )


__all__ = [
    "MultigroupDiffusionPlan",
    "MultigroupMaterialData",
    "PreparedMultigroupDiffusion",
    "ReactorCriticalityResult",
    "ReactorDiffusionSolveResult",
]
