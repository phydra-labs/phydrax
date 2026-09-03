#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from operator import index
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum, compensated_sum_chunks
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..finite_volume._riemann import (
    AbstractSymmetricTwoPointFluxPlan,
    EntropyConservativeEulerFluxPlan,
)
from ._method import (
    PreparedPseudospectralMethod,
    PseudospectralMethodPlan,
    SpectralDifferentiabilityPolicy,
)
from ._space import TensorSpectralDiscretization


if TYPE_CHECKING:
    from ...equations import ConvexEntropyPair


class SpectralSplitFormPlan(StrictModule, NonTrainableState):
    """Certified periodic Fourier flux-differencing plan."""

    volume_flux: AbstractSymmetricTwoPointFluxPlan
    pair_chunk_size: int = eqx.field(static=True)
    maximum_pair_workspace_bytes: int = eqx.field(static=True)
    certification_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        volume_flux: AbstractSymmetricTwoPointFluxPlan,
        /,
        *,
        pair_chunk_size: int = 4096,
        maximum_pair_workspace_bytes: int = 512 * 1024**2,
        certification_tolerance: float = 1e-10,
    ):
        if not isinstance(volume_flux, AbstractSymmetricTwoPointFluxPlan):
            raise TypeError("volume_flux must be a symmetric two-point flux plan.")
        if not isinstance(volume_flux, EntropyConservativeEulerFluxPlan):
            raise TypeError(
                "Only the built-in analytic entropy-conservative Euler flux "
                "carries the spectral split-form certificate."
            )
        chunk = index(pair_chunk_size)
        workspace = index(maximum_pair_workspace_bytes)
        tolerance = float(certification_tolerance)
        if chunk <= 0 or workspace <= 0 or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Split-form chunk, workspace, and tolerance are invalid.")
        self.volume_flux = volume_flux
        self.pair_chunk_size = chunk
        self.maximum_pair_workspace_bytes = workspace
        self.certification_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-split-form-plan",
                "volume_flux": volume_flux.flux_id,
                "pair_chunk_size": chunk,
                "maximum_pair_workspace_bytes": workspace,
                "certification_tolerance": tolerance,
            }
        )


class SpectralSplitFormReport(StrictModule, NonTrainableState):
    pair_count: int = eqx.field(static=True)
    pair_chunk_size: int = eqx.field(static=True)
    pair_workspace_bytes: int = eqx.field(static=True)
    skew_sbp_defect: float = eqx.field(static=True)
    constant_state_defect: float = eqx.field(static=True)
    entropy_stable: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class PreparedSpectralSplitForm(StrictModule, NonTrainableState):
    plan: SpectralSplitFormPlan
    differentiation_matrices: tuple[Array, ...]
    quadrature_norms: tuple[Array, ...]
    pair_rows: tuple[Array, ...]
    pair_columns: tuple[Array, ...]
    report: SpectralSplitFormReport
    prepared_id: str = eqx.field(static=True)


class SpectralConservationMethodPlan(StrictModule, NonTrainableState):
    """Periodic conservative projected flux or certified Fourier split form."""

    pseudospectral: PseudospectralMethodPlan | None
    split_form: SpectralSplitFormPlan | None
    flux_polynomial_degree: int | None = eqx.field(static=True)
    entropy_diagnostics: bool = eqx.field(static=True)
    differentiability: SpectralDifferentiabilityPolicy = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        pseudospectral: PseudospectralMethodPlan | None = None,
        /,
        *,
        split_form: SpectralSplitFormPlan | None = None,
        flux_polynomial_degree: int | None = None,
        entropy_diagnostics: bool = False,
        differentiability: SpectralDifferentiabilityPolicy = "smooth_discrete",
    ):
        if (pseudospectral is None) == (split_form is None):
            raise ValueError(
                "Select exactly one of pseudospectral or split_form execution."
            )
        if pseudospectral is not None and not isinstance(
            pseudospectral, PseudospectralMethodPlan
        ):
            raise TypeError("pseudospectral must be PseudospectralMethodPlan or None.")
        if split_form is not None and not isinstance(split_form, SpectralSplitFormPlan):
            raise TypeError("split_form must be SpectralSplitFormPlan or None.")
        degree = None if flux_polynomial_degree is None else int(flux_polynomial_degree)
        if degree is not None and degree < 1:
            raise ValueError("flux_polynomial_degree must be positive or None.")
        if split_form is not None and degree is not None:
            raise ValueError("flux_polynomial_degree is invalid on the split-form route.")
        if differentiability not in (
            "smooth_discrete",
            "branchwise",
            "smooth_surrogate",
            "unsupported",
        ):
            raise ValueError("Unknown spectral differentiability policy.")
        self.pseudospectral = pseudospectral
        self.split_form = split_form
        self.flux_polynomial_degree = degree
        self.entropy_diagnostics = bool(entropy_diagnostics)
        self.differentiability = differentiability
        self.method_id = canonical_fingerprint(
            {
                "kind": "spectral-conservation-method",
                "pseudospectral": (
                    None if pseudospectral is None else pseudospectral.method_id
                ),
                "split_form": None if split_form is None else split_form.plan_id,
                "flux_polynomial_degree": degree,
                "entropy_diagnostics": bool(entropy_diagnostics),
                "differentiability": differentiability,
            }
        )

    def prepare(
        self,
        discretization: TensorSpectralDiscretization,
        /,
    ) -> "PreparedSpectralConservationMethod":
        if not all(axis.periodic for axis in discretization.axes):
            raise ValueError("Spectral conservation requires periodic axes.")
        if self.split_form is None:
            assert self.pseudospectral is not None
            nonlinear = (
                self.flux_polynomial_degree is None or self.flux_polynomial_degree > 1
            )
            pseudospectral = self.pseudospectral.prepare(
                discretization,
                required_polynomial_degree=self.flux_polynomial_degree,
                nonlinear=nonlinear,
            )
            return PreparedSpectralConservationMethod(
                self, discretization, pseudospectral, None
            )
        if any(axis.family != "fourier" for axis in discretization.axes):
            raise ValueError("Certified split forms require all-Fourier axes.")
        prepared_split = _prepare_split_form(self.split_form, discretization)
        return PreparedSpectralConservationMethod(
            self, discretization, None, prepared_split
        )


class PreparedSpectralConservationMethod(StrictModule, NonTrainableState):
    plan: SpectralConservationMethodPlan
    discretization: TensorSpectralDiscretization
    pseudospectral: PreparedPseudospectralMethod | None
    split_form: PreparedSpectralSplitForm | None
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SpectralConservationMethodPlan,
        discretization: TensorSpectralDiscretization,
        pseudospectral: PreparedPseudospectralMethod | None,
        split_form: PreparedSpectralSplitForm | None,
        /,
    ):
        self.plan = plan
        self.discretization = discretization
        self.pseudospectral = pseudospectral
        self.split_form = split_form
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-conservation-method",
                "plan": plan.method_id,
                "discretization": discretization.prepared_id,
                "pseudospectral": (
                    None if pseudospectral is None else pseudospectral.prepared_id
                ),
                "split_form": (None if split_form is None else split_form.prepared_id),
            }
        )


def _prepare_split_form(
    plan: SpectralSplitFormPlan,
    discretization: TensorSpectralDiscretization,
    /,
) -> PreparedSpectralSplitForm:
    if discretization.plan.precision.physical_dtype.startswith("complex"):
        raise ValueError("Certified split forms require a real physical state.")
    matrices = []
    cell = discretization.periodic_cell
    if cell is None or cell.rank != len(discretization.axes) or not cell.fully_periodic:
        raise ValueError(
            "Certified split forms require a full-rank all-periodic PeriodicCell."
        )
    reciprocal_defect = np.max(
        np.abs(
            np.asarray(cell.vectors) @ np.asarray(cell.reciprocal_vectors).T
            - 2.0 * np.pi * np.eye(cell.rank)
        ),
        initial=0.0,
    )
    if reciprocal_defect > plan.certification_tolerance:
        raise ValueError("PeriodicCell reciprocal geometry certification failed.")
    norms = []
    rows = []
    columns = []
    pair_count = 0
    maximum_defect = 0.0
    maximum_workspace = 0
    components = len(discretization.axes) + 2
    itemsize = np.dtype(discretization.plan.precision.physical_dtype).itemsize
    for axis_index, axis in enumerate(discretization.axes):
        count = axis.mode_count
        indices = np.arange(count)
        difference = indices[:, None] - indices[None, :]
        angle = np.pi * difference / count
        sign = (-1.0) ** difference
        if count % 2:
            denominator = np.sin(angle)
        else:
            denominator = np.tan(angle)
        derivative = np.where(
            difference == 0,
            0.0,
            0.5 * sign / np.where(difference == 0, 1.0, denominator),
        )
        derivative *= 2.0 * np.pi / float(axis.length)
        norm = np.full((count,), float(axis.length) / count)
        q_matrix = norm[:, None] * derivative
        skew = float(np.max(np.abs(q_matrix + q_matrix.T), initial=0.0))
        constant = float(np.max(np.abs(derivative @ np.ones(count)), initial=0.0))
        maximum_defect = max(maximum_defect, skew, constant)
        if maximum_defect > plan.certification_tolerance:
            raise ValueError("Fourier split-form SBP/skew certification failed.")
        pair_row, pair_column = np.triu_indices(count, k=1)
        nonzero = np.abs(q_matrix[pair_row, pair_column]) > 0.0
        pair_row = pair_row[nonzero]
        pair_column = pair_column[nonzero]
        transverse_line_count = int(
            np.prod(
                tuple(
                    other.mode_count
                    for other_index, other in enumerate(discretization.axes)
                    if other_index != axis_index
                )
            )
        )
        pair_count += int(pair_row.size) * transverse_line_count
        maximum_workspace = max(
            maximum_workspace,
            min(plan.pair_chunk_size, int(pair_row.size))
            * transverse_line_count
            * components
            * 3
            * itemsize,
        )
        matrices.append(jnp.asarray(q_matrix))
        norms.append(jnp.asarray(norm))
        rows.append(jnp.asarray(pair_row, dtype=jnp.int32))
        columns.append(jnp.asarray(pair_column, dtype=jnp.int32))
    workspace = maximum_workspace
    if workspace > plan.maximum_pair_workspace_bytes:
        raise ValueError("Split-form pair workspace exceeds the declared budget.")
    report = SpectralSplitFormReport(
        pair_count=pair_count,
        pair_chunk_size=plan.pair_chunk_size,
        pair_workspace_bytes=workspace,
        skew_sbp_defect=maximum_defect,
        constant_state_defect=maximum_defect,
        entropy_stable=True,
        report_id=canonical_fingerprint(
            {
                "kind": "spectral-split-form-report",
                "plan": plan.plan_id,
                "discretization": discretization.prepared_id,
                "pair_count": pair_count,
                "workspace": workspace,
                "skew_sbp_defect": maximum_defect,
            }
        ),
    )
    return PreparedSpectralSplitForm(
        plan=plan,
        differentiation_matrices=tuple(matrices),
        quadrature_norms=tuple(norms),
        pair_rows=tuple(rows),
        pair_columns=tuple(columns),
        report=report,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-spectral-split-form",
                "plan": plan.plan_id,
                "report": report.report_id,
            }
        ),
    )


class SpectralEntropyDiagnostics(StrictModule):
    pair_id: str = eqx.field(static=True)
    total_entropy: Array
    semidiscrete_entropy_rate: Array
    source_entropy_rate: Array
    convective_entropy_rate: Array
    convective_entropy_defect: Array
    skew_sbp_defect: Array
    pair_workspace_bytes: int = eqx.field(static=True)
    entropy_stable: Array
    admissible: Array
    precision_evidence: PrecisionEvidenceEnvelope


class SpectralConservationDiagnostics(StrictModule):
    total_integral: Array
    semidiscrete_integral_rate: Array
    source_integral: Array
    conservation_defect: Array
    entropy: SpectralEntropyDiagnostics | None
    precision_evidence: PrecisionEvidenceEnvelope
    method_id: str = eqx.field(static=True)


class PreparedSpectralConservationDynamics(StrictModule):
    """Pure periodic conservative pseudospectral semidiscretization."""

    system: Any
    discretization: TensorSpectralDiscretization
    method: PreparedSpectralConservationMethod
    entropy_pair: Any
    source: Any = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: Any,
        discretization: TensorSpectralDiscretization,
        method: SpectralConservationMethodPlan,
        /,
        *,
        source: Any = None,
        entropy_pair: "ConvexEntropyPair | None" = None,
    ):
        from ...equations import AbstractConservationSystem, ConvexEntropyPair

        if not isinstance(system, AbstractConservationSystem):
            raise TypeError("system must be an AbstractConservationSystem.")
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        if system.dimension != len(discretization.axes):
            raise ValueError("Conservation-system dimension must match spectral rank.")
        if source is not None and not callable(source):
            raise TypeError("source must be callable or None.")
        if entropy_pair is not None:
            if not isinstance(entropy_pair, ConvexEntropyPair):
                raise TypeError("entropy_pair must be a ConvexEntropyPair or None.")
            if entropy_pair.system.system_id != system.system_id:
                raise ValueError("entropy_pair must target the conservation system.")
        if method.split_form is not None:
            if source is not None:
                raise ValueError("Certified split forms do not accept source terms.")
            if entropy_pair is None:
                raise ValueError("Certified split forms require an entropy pair.")
            if system.component_count != system.dimension + 2:
                raise TypeError(
                    "The built-in split-form certificate requires Euler state layout."
                )
        prepared = method.prepare(discretization)
        self.system = system
        self.discretization = discretization
        self.method = prepared
        self.entropy_pair = entropy_pair
        self.source = source
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-conservation-dynamics",
                "system": system.system_id,
                "discretization": discretization.prepared_id,
                "method": prepared.prepared_id,
                "entropy_pair": None if entropy_pair is None else entropy_pair.pair_id,
                "source": None if source is None else repr(source),
            }
        )

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.discretization.modal_shape + (self.system.component_count,)

    def _validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError(
                f"Spectral conservation state must have shape {self.state_shape}; "
                f"got {value.shape}."
            )
        return value

    def _physical_source(
        self,
        time: Array,
        physical_state: Array,
        args: Any,
        /,
    ) -> Array:
        if self.source is None:
            return jnp.zeros_like(physical_state)
        evaluation = self.method.pseudospectral.dealiasing.evaluation
        points = evaluation.points.reshape(
            evaluation.physical_shape + (len(evaluation.axes),)
        )
        value = jnp.asarray(self.source(time, physical_state, points, args))
        if value.shape != physical_state.shape:
            raise ValueError("Spectral conservation source must match physical state.")
        return value

    def residual_parts(
        self,
        time: Array,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        coefficients = self._validate_state(state)
        if self.method.split_form is None:
            assert self.method.pseudospectral is not None
            dealiasing = self.method.pseudospectral.dealiasing
            physical = dealiasing.reconstruct(coefficients)
            convective = jnp.zeros_like(coefficients)
            for axis in range(len(self.discretization.axes)):
                flux = self.system.physical_flux(physical, axis, args)
                flux_coefficients = dealiasing.project(flux)
                convective = convective - self.discretization.modal_derivative(
                    flux_coefficients,
                    axis=axis,
                    order=1,
                )
            source = dealiasing.project(self._physical_source(time, physical, args))
            return convective, source
        split = self.method.split_form
        physical = self.discretization.reconstruct(coefficients)
        if jnp.iscomplexobj(physical):
            raise TypeError("Certified split-form physical state must be real.")
        physical_residual = jnp.zeros_like(physical)
        for axis, (q_matrix, norm, pair_rows, pair_columns) in enumerate(
            zip(
                split.differentiation_matrices,
                split.quadrature_norms,
                split.pair_rows,
                split.pair_columns,
                strict=True,
            )
        ):
            lines = jnp.moveaxis(physical, axis, 0)
            line_residual = jnp.zeros_like(lines)
            pair_count = int(pair_rows.size)
            for start in range(0, pair_count, split.plan.pair_chunk_size):
                stop = min(start + split.plan.pair_chunk_size, pair_count)
                rows = pair_rows[start:stop]
                columns = pair_columns[start:stop]
                pair_flux = split.plan.volume_flux.two_point_flux(
                    self.system,
                    lines[rows],
                    lines[columns],
                    axis,
                    args,
                )
                weights = q_matrix[rows, columns] / norm[rows]
                weight_shape = (weights.size,) + (1,) * (pair_flux.ndim - 1)
                contribution = 2.0 * weights.reshape(weight_shape) * pair_flux
                line_residual = line_residual.at[rows].add(-contribution)
                line_residual = line_residual.at[columns].add(contribution)
            physical_residual = physical_residual + jnp.moveaxis(line_residual, 0, axis)
        convective = self.discretization.project(physical_residual)
        return convective, jnp.zeros_like(convective)

    def __call__(self, time: Array, state: Array, args: Any = None) -> Array:
        convective, source = self.residual_parts(time, state, args)
        return convective + source

    def residual_with_diagnostics(
        self,
        time: Array,
        state: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, SpectralConservationDiagnostics]:
        coefficients = self._validate_state(state)
        convective, source = self.residual_parts(time, coefficients, args)
        residual = convective + source
        physical_state = self.discretization.reconstruct(coefficients)
        physical_residual = self.discretization.reconstruct(residual)
        physical_source = self.discretization.reconstruct(source)
        precision = self.discretization.plan.precision
        spatial_axes = tuple(range(len(self.discretization.axes)))
        weights = precision.reduction(self.discretization.quadrature_weights[..., None])
        total_terms = precision.reduction(weights * physical_state)
        residual_terms = precision.reduction(weights * physical_residual)
        source_terms = precision.reduction(weights * physical_source)
        total_integral = compensated_sum(total_terms, axis=spatial_axes)
        residual_integral = compensated_sum(residual_terms, axis=spatial_axes)
        source_integral = compensated_sum(source_terms, axis=spatial_axes)
        conservation_defect = compensated_sum_chunks(
            (residual_terms, -source_terms),
            output_ndim=1,
        )
        entropy = None
        if self.entropy_pair is not None:
            pair = self.entropy_pair
            entropy_variables = pair.entropy_variables(physical_state)
            convective_physical = self.discretization.reconstruct(convective)
            convective_density = ein.contract(
                "...i,...i->...",
                entropy_variables,
                convective_physical,
            )
            source_density = ein.contract(
                "...i,...i->...",
                entropy_variables,
                physical_source,
            )
            scalar_weights = self.discretization.quadrature_weights
            convective_rate = jnp.sum(scalar_weights * convective_density)
            source_rate = jnp.sum(scalar_weights * source_density)
            split_report = (
                None if self.method.split_form is None else self.method.split_form.report
            )
            entropy_tolerance = (
                0.0
                if self.method.split_form is None
                else self.method.split_form.plan.certification_tolerance
            )
            admissible = jnp.all(pair.admissible(physical_state))
            entropy = SpectralEntropyDiagnostics(
                pair_id=pair.pair_id,
                total_entropy=jnp.sum(scalar_weights * pair.entropy(physical_state)),
                semidiscrete_entropy_rate=convective_rate + source_rate,
                source_entropy_rate=source_rate,
                convective_entropy_rate=convective_rate,
                convective_entropy_defect=jnp.abs(convective_rate),
                skew_sbp_defect=jnp.asarray(
                    0.0 if split_report is None else split_report.skew_sbp_defect,
                    dtype=convective_rate.dtype,
                ),
                pair_workspace_bytes=(
                    0 if split_report is None else split_report.pair_workspace_bytes
                ),
                entropy_stable=(
                    jnp.asarray(False)
                    if split_report is None
                    else admissible & (jnp.abs(convective_rate) <= entropy_tolerance)
                ),
                admissible=admissible,
                precision_evidence=self.discretization.precision_evidence,
            )
        diagnostics = SpectralConservationDiagnostics(
            total_integral=total_integral,
            semidiscrete_integral_rate=residual_integral,
            source_integral=source_integral,
            conservation_defect=conservation_defect,
            entropy=entropy,
            precision_evidence=self.discretization.precision_evidence,
            method_id=self.method.plan.method_id,
        )
        return residual, diagnostics

    def linearize(self, time: Array, state: Array, args: Any = None, /):
        value = self._validate_state(state)
        return jax.linearize(lambda candidate: self(time, candidate, args), value)


__all__ = [
    "PreparedSpectralSplitForm",
    "PreparedSpectralConservationDynamics",
    "PreparedSpectralConservationMethod",
    "SpectralConservationDiagnostics",
    "SpectralConservationMethodPlan",
    "SpectralSplitFormPlan",
    "SpectralSplitFormReport",
    "SpectralEntropyDiagnostics",
]
