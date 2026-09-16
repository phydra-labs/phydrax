#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.spectral import (
    project_tensor_spectral_symmetries,
    TensorSpectralDiscretization,
    TensorSpectralSymmetry,
)
from ...nonlinear import (
    implicit_root_result,
    NewtonKrylov,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._architecture import (
    IncompressibleGaussianMixturePlan,
    PolymerComponentPlan,
)


ContourIntegratorKind: TypeAlias = Literal["strang-2", "richardson-strang-4"]


class ContourIntegratorPlan(StrictModule, NonTrainableState):
    kind: ContourIntegratorKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, kind: ContourIntegratorKind = "richardson-strang-4", /):
        if kind not in ("strang-2", "richardson-strang-4"):
            raise ValueError("Unknown Gaussian-chain contour integrator.")
        self.kind = kind
        self.plan_id = canonical_fingerprint(
            {"kind": "contour-integrator-plan", "method": kind}
        )


class SCFTPlan(StrictModule, NonTrainableState):
    model: IncompressibleGaussianMixturePlan
    contour_integrator: ContourIntegratorPlan
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    minimum_partition: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: IncompressibleGaussianMixturePlan,
        /,
        *,
        contour_integrator: ContourIntegratorPlan | None = None,
        absolute_tolerance: float = 1.0e-8,
        relative_tolerance: float = 1.0e-8,
        maximum_iterations: int = 64,
        minimum_partition: float = 1.0e-14,
    ):
        if not isinstance(model, IncompressibleGaussianMixturePlan):
            raise TypeError("model must be IncompressibleGaussianMixturePlan.")
        integrator = (
            ContourIntegratorPlan() if contour_integrator is None else contour_integrator
        )
        if not isinstance(integrator, ContourIntegratorPlan):
            raise TypeError("contour_integrator must be ContourIntegratorPlan or None.")
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        iterations = int(maximum_iterations)
        minimum = float(minimum_partition)
        if (
            not math.isfinite(absolute)
            or absolute < 0.0
            or not math.isfinite(relative)
            or relative < 0.0
            or iterations <= 0
            or not math.isfinite(minimum)
            or minimum <= 0.0
        ):
            raise ValueError("SCFT solver controls are invalid.")
        self.model = model
        self.contour_integrator = integrator
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.maximum_iterations = iterations
        self.minimum_partition = minimum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "scft-plan",
                "model": model.plan_id,
                "contour_integrator": integrator.plan_id,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
                "maximum_iterations": iterations,
                "minimum_partition": minimum,
            }
        )

    def prepare(
        self,
        spectral: TensorSpectralDiscretization,
        /,
        *,
        symmetries: tuple[TensorSpectralSymmetry, ...] = (),
    ) -> "PreparedSCFT":
        return PreparedSCFT(self, spectral, symmetries=symmetries)


class SCFTComponentEvaluation(StrictModule):
    partition_function: Array
    density: Array
    minimum_propagator: Array
    successful: Array
    component_id: str = eqx.field(static=True)


class SCFTEvaluation(StrictModule):
    fields: Array
    densities: Array
    partition_functions: Array
    residual: Array
    incompressibility_residual: Array
    gauge_residual: Array
    free_energy: Array
    minimum_propagator: Array
    cell_scale: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class PreparedSCFT(StrictModule, NonTrainableState):
    plan: SCFTPlan
    spectral: TensorSpectralDiscretization
    symmetries: tuple[TensorSpectralSymmetry, ...]
    laplacian_eigenvalues: Array
    zero_mode_index: tuple[int, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SCFTPlan,
        spectral: TensorSpectralDiscretization,
        /,
        *,
        symmetries: tuple[TensorSpectralSymmetry, ...],
    ):
        if not isinstance(plan, SCFTPlan):
            raise TypeError("plan must be SCFTPlan.")
        if not isinstance(spectral, TensorSpectralDiscretization):
            raise TypeError("spectral must be TensorSpectralDiscretization.")
        if spectral.periodic_cell is None or any(
            axis.family != "fourier" for axis in spectral.axes
        ):
            raise ValueError("SCFT requires an all-Fourier periodic spectral grid.")
        symmetry_values = tuple(symmetries)
        if any(
            not isinstance(value, TensorSpectralSymmetry)
            or value.discretization.prepared_id != spectral.prepared_id
            or value.component_count != plan.model.species_count
            for value in symmetry_values
        ):
            raise ValueError("SCFT symmetries must match the grid and species count.")
        self.plan = plan
        self.spectral = spectral
        self.symmetries = symmetry_values
        self.laplacian_eigenvalues = spectral.laplacian_eigenvalues()
        self.zero_mode_index = (0,) * len(spectral.modal_shape)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-scft",
                "plan": plan.plan_id,
                "spectral": spectral.prepared_id,
                "symmetries": [value.symmetry_id for value in symmetry_values],
            }
        )

    @property
    def field_shape(self) -> tuple[int, ...]:
        return self.spectral.physical_shape + (self.plan.model.species_count,)

    def project_initial_fields(self, fields: ArrayLike, /) -> Array:
        value = jnp.asarray(fields, dtype=self.spectral._quadrature_weights.dtype)
        if value.shape != self.field_shape:
            raise ValueError(f"SCFT fields must have shape {self.field_shape}.")
        value = value - jnp.mean(value)
        if not self.symmetries:
            return value
        modal = self.spectral.project(value)
        projected = project_tensor_spectral_symmetries(modal, self.symmetries)
        return self.spectral.reconstruct(projected)

    def symmetry_defects(self, fields: ArrayLike, /) -> Array:
        value = jnp.asarray(fields, dtype=self.spectral._quadrature_weights.dtype)
        if value.shape != self.field_shape:
            raise ValueError(f"SCFT fields must have shape {self.field_shape}.")
        if not self.symmetries:
            return jnp.zeros((0,), dtype=value.dtype)
        modal = self.spectral.project(value)
        return jnp.stack(
            [symmetry.fixed_subspace_defect(modal) for symmetry in self.symmetries]
        )

    def _strang_step(
        self,
        propagator: Array,
        field: Array,
        contour_step: float,
        diffusion: float,
        cell_scale: Array,
        /,
    ) -> Array:
        half_potential = jnp.exp(-0.5 * contour_step * field)
        weighted = half_potential * propagator
        modal = self.spectral.project(weighted)
        diffusion_multiplier = jnp.exp(
            -contour_step
            * diffusion
            * self.laplacian_eigenvalues
            / (cell_scale * cell_scale)
        )
        diffused = self.spectral.reconstruct(
            modal * diffusion_multiplier,
            real_output=not jnp.issubdtype(propagator.dtype, jnp.complexfloating),
        )
        return half_potential * diffused

    def _contour_step(
        self,
        propagator: Array,
        field: Array,
        contour_step: float,
        diffusion: float,
        cell_scale: Array,
        /,
    ) -> Array:
        if self.plan.contour_integrator.kind == "strang-2":
            return self._strang_step(
                propagator, field, contour_step, diffusion, cell_scale
            )
        whole = self._strang_step(propagator, field, contour_step, diffusion, cell_scale)
        half = self._strang_step(
            propagator, field, 0.5 * contour_step, diffusion, cell_scale
        )
        refined = self._strang_step(
            half, field, 0.5 * contour_step, diffusion, cell_scale
        )
        return (4.0 * refined - whole) / 3.0

    def _propagate_block(
        self,
        initial: Array,
        field: Array,
        contour_fraction: float,
        contour_steps: int,
        diffusion: float,
        cell_scale: Array,
        /,
    ) -> Array:
        step = contour_fraction / contour_steps
        values = [initial]
        current = initial
        for _ in range(contour_steps):
            current = self._contour_step(current, field, step, diffusion, cell_scale)
            values.append(current)
        return jnp.stack(values, axis=0)

    def _component(
        self,
        component: PolymerComponentPlan,
        fields: Array,
        cell_scale: Array,
        /,
    ) -> SCFTComponentEvaluation:
        architecture = component.architecture
        cache: dict[tuple[int, str], Array] = {}

        def message(block_index: int, from_node: str) -> Array:
            key = (block_index, from_node)
            if key in cache:
                return cache[key]
            block = architecture.blocks[block_index]
            initial = jnp.ones(self.spectral.physical_shape, dtype=fields.dtype)
            for other_index in architecture.incident(from_node):
                if other_index == block_index:
                    continue
                other_node = architecture.other_node(other_index, from_node)
                initial = initial * message(other_index, other_node)[-1]
            diffusion = (
                component.polymerization_index
                * self.plan.model.statistical_segment_lengths[block.species_index] ** 2
                / 6.0
            )
            path = self._propagate_block(
                initial,
                fields[..., block.species_index],
                block.contour_fraction,
                block.contour_steps,
                diffusion,
                cell_scale,
            )
            cache[key] = path
            return path

        root_product = jnp.ones(self.spectral.physical_shape, dtype=fields.dtype)
        for block_index in architecture.incident(architecture.root_node):
            other = architecture.other_node(block_index, architecture.root_node)
            root_product = root_product * message(block_index, other)[-1]
        partition = jnp.mean(root_product)
        density = jnp.zeros(self.field_shape, dtype=fields.dtype)
        minimum_propagator = jnp.asarray(jnp.inf, dtype=fields.real.dtype)
        for block_index, block in enumerate(architecture.blocks):
            forward = message(block_index, block.source_node)
            backward = message(block_index, block.target_node)
            minimum_propagator = jnp.minimum(
                minimum_propagator,
                jnp.minimum(jnp.min(jnp.abs(forward)), jnp.min(jnp.abs(backward))),
            )
            product = forward * backward[::-1]
            weights = jnp.ones((block.contour_steps + 1,), dtype=fields.dtype)
            weights = weights.at[0].set(0.5).at[-1].set(0.5)
            integral = (
                block.contour_fraction
                / block.contour_steps
                * jnp.sum(
                    product
                    * weights.reshape(
                        (block.contour_steps + 1,)
                        + (1,) * len(self.spectral.physical_shape)
                    ),
                    axis=0,
                )
            )
            safe_partition = jnp.where(
                jnp.abs(partition) > self.plan.minimum_partition,
                partition,
                jnp.ones_like(partition),
            )
            contribution = component.volume_fraction * integral / safe_partition
            density = density.at[..., block.species_index].add(contribution)
        successful = (
            jnp.isfinite(partition)
            & (jnp.abs(partition) > self.plan.minimum_partition)
            & jnp.all(jnp.isfinite(density))
            & jnp.isfinite(minimum_propagator)
        )
        return SCFTComponentEvaluation(
            partition,
            density,
            minimum_propagator,
            successful,
            component.component_id,
        )

    def evaluate(
        self,
        fields: ArrayLike,
        /,
        *,
        chi_n: ArrayLike | None = None,
        cell_scale: ArrayLike = 1.0,
    ) -> SCFTEvaluation:
        raw_fields = jnp.asarray(fields)
        if not jnp.issubdtype(raw_fields.dtype, jnp.inexact):
            raise TypeError("SCFT fields must use a real or complex inexact dtype.")
        value = (
            raw_fields
            if jnp.issubdtype(raw_fields.dtype, jnp.complexfloating)
            else raw_fields.astype(self.spectral._quadrature_weights.dtype)
        )
        if value.shape != self.field_shape:
            raise ValueError(f"SCFT fields must have shape {self.field_shape}.")
        interactions = (
            self.plan.model.chi_n
            if chi_n is None
            else jnp.asarray(chi_n, dtype=value.dtype)
        )
        scale = jnp.asarray(cell_scale, dtype=value.real.dtype)
        if (
            interactions.shape
            != (
                self.plan.model.species_count,
                self.plan.model.species_count,
            )
            or scale.shape != ()
        ):
            raise ValueError(
                "SCFT interaction matrix or cell scale has an invalid shape."
            )
        components = tuple(
            self._component(component, value, scale)
            for component in self.plan.model.components
        )
        densities = sum(
            (component.density for component in components),
            jnp.zeros(self.field_shape, dtype=value.dtype),
        )
        partitions = jnp.stack([component.partition_function for component in components])
        minimum_propagator = jnp.min(
            jnp.stack([component.minimum_propagator for component in components])
        )
        interaction_fields = contract("...j,ij->...i", densities, interactions)
        exchange = (
            value[..., :-1]
            - interaction_fields[..., :-1]
            - (value[..., -1:] - interaction_fields[..., -1:])
        )
        incompressibility = jnp.sum(densities, axis=-1) - 1.0
        incompressibility_modal = self.spectral.project(incompressibility)
        gauge = jnp.mean(value)
        incompressibility_modal = incompressibility_modal.at[self.zero_mode_index].set(
            gauge.astype(incompressibility_modal.dtype)
        )
        gauge_fixed_incompressibility = self.spectral.reconstruct(
            incompressibility_modal,
            real_output=not jnp.issubdtype(value.dtype, jnp.complexfloating),
        )
        residual = jnp.concatenate(
            (exchange, gauge_fixed_incompressibility[..., None]), axis=-1
        )
        interaction_energy = 0.5 * contract(
            "...i,ij,...j->...", densities, interactions, densities
        )
        field_energy = jnp.sum(value * densities, axis=-1)
        chain_energy = -sum(
            component.volume_fraction
            / component.polymerization_index
            * jnp.log(partition)
            for component, partition in zip(
                self.plan.model.components, partitions, strict=True
            )
        )
        free_energy = chain_energy + jnp.mean(interaction_energy - field_energy)
        component_successful = jnp.all(
            jnp.stack([component.successful for component in components])
        )
        successful = (
            component_successful
            & jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(interactions))
            & jnp.isfinite(scale)
            & (scale > 0.0)
            & jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(free_energy)
        )
        return SCFTEvaluation(
            value,
            densities,
            partitions,
            residual,
            incompressibility,
            gauge,
            free_energy,
            minimum_propagator,
            scale,
            successful,
            self.prepared_id,
        )

    def root_problem(self, /) -> NonlinearSystemProblem:
        def residual(fields, _):
            evaluation = self.evaluate(fields)
            return jnp.where(evaluation.successful, evaluation.residual, jnp.nan)

        return NonlinearSystemProblem(residual, problem_id=f"{self.prepared_id}:root")

    def parameterized_root_problem(self, /) -> NonlinearSystemProblem:
        def residual(fields, args):
            interactions, scale = args
            evaluation = self.evaluate(fields, chi_n=interactions, cell_scale=scale)
            return jnp.where(evaluation.successful, evaluation.residual, jnp.nan)

        return NonlinearSystemProblem(
            residual, problem_id=f"{self.prepared_id}:parameterized-root"
        )


class SCFTResult(StrictModule):
    evaluation: SCFTEvaluation
    nonlinear: NonlinearResult
    symmetry_defects: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


def solve_scft(
    prepared: PreparedSCFT,
    initial_fields: ArrayLike,
    /,
) -> SCFTResult:
    if not isinstance(prepared, PreparedSCFT):
        raise TypeError("prepared must be PreparedSCFT.")
    initial = prepared.project_initial_fields(initial_fields)
    nonlinear = NewtonKrylov().solve(
        prepared.root_problem(),
        initial,
        termination=NonlinearTermination(
            absolute_residual=prepared.plan.absolute_tolerance,
            relative_residual=prepared.plan.relative_tolerance,
            maximum_steps=prepared.plan.maximum_iterations,
        ),
    )
    evaluation = prepared.evaluate(nonlinear.state)
    defects = prepared.symmetry_defects(nonlinear.state)
    successful = nonlinear.successful & evaluation.successful
    return SCFTResult(evaluation, nonlinear, defects, successful, prepared.prepared_id)


def solve_scft_implicit(
    prepared: PreparedSCFT,
    initial_fields: ArrayLike,
    /,
    *,
    chi_n: ArrayLike | None = None,
    cell_scale: ArrayLike = 1.0,
) -> SCFTResult:
    if not isinstance(prepared, PreparedSCFT):
        raise TypeError("prepared must be PreparedSCFT.")
    initial = prepared.project_initial_fields(initial_fields)
    interactions = (
        prepared.plan.model.chi_n
        if chi_n is None
        else jnp.asarray(chi_n, dtype=initial.dtype)
    )
    scale = jnp.asarray(cell_scale, dtype=initial.dtype)
    nonlinear = implicit_root_result(
        prepared.parameterized_root_problem(),
        initial,
        method=NewtonKrylov(),
        termination=NonlinearTermination(
            absolute_residual=prepared.plan.absolute_tolerance,
            relative_residual=prepared.plan.relative_tolerance,
            maximum_steps=prepared.plan.maximum_iterations,
        ),
        args=(interactions, scale),
    )
    evaluation = prepared.evaluate(nonlinear.state, chi_n=interactions, cell_scale=scale)
    defects = prepared.symmetry_defects(nonlinear.state)
    successful = nonlinear.successful & evaluation.successful
    return SCFTResult(evaluation, nonlinear, defects, successful, prepared.prepared_id)


__all__ = [
    "ContourIntegratorKind",
    "ContourIntegratorPlan",
    "PreparedSCFT",
    "SCFTComponentEvaluation",
    "SCFTEvaluation",
    "SCFTPlan",
    "SCFTResult",
    "solve_scft",
    "solve_scft_implicit",
]
