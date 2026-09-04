#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.spectral import TensorSpectralDiscretization
from ._channel_flow import (
    ChannelVelocityDiagnostics,
    CompiledChannelFlowDynamics,
)
from ._les_closures import (
    AlgebraicLESInputs,
    LESFilterScale,
    PreparedAlgebraicLESModel,
    ResolvedLESFilter,
)


class ChannelLESFilterGeometry(StrictModule, NonTrainableState):
    """Local resolved-grid widths for a Fourier–Chebyshev–Fourier channel.

    Fourier widths are the retained-grid physical spacings. The wall-normal
    widths are the positive Chebyshev nodal quadrature measures, interpolated
    from the retained collocation points to the nonlinear evaluation grid.
    Thus the widths represent the resolved grid rather than the padded grid.

    The wall-normal width varies in space, so differentiation and any implicit
    local grid filter do not commute. ``wall_normal_width_gradient`` and
    :meth:`wall_normal_scale_commutator` expose the product-rule evidence for
    that noncommutation. No commutator correction is modeled by this route.
    """

    directional_widths: Array
    wall_normal_widths: Array
    wall_normal_width_gradient: Array
    streamwise_width: Array
    spanwise_width: Array
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        retained: TensorSpectralDiscretization,
        evaluation: TensorSpectralDiscretization,
        /,
    ):
        if not isinstance(retained, TensorSpectralDiscretization) or not isinstance(
            evaluation, TensorSpectralDiscretization
        ):
            raise TypeError("Channel LES filter geometry requires tensor spectral grids.")
        retained_families = tuple(axis.family for axis in retained.axes)
        evaluation_families = tuple(axis.family for axis in evaluation.axes)
        if retained_families != ("fourier", "chebyshev", "fourier") or (
            evaluation_families != retained_families
        ):
            raise ValueError(
                "Channel LES filter geometry requires Fourier x Chebyshev x Fourier grids."
            )
        if tuple(retained.plan.axis_names) != tuple(evaluation.plan.axis_names):
            raise ValueError(
                "Retained and evaluation channel axes must have the same names."
            )

        x_axis, retained_wall_axis, z_axis = retained.axes
        evaluation_wall_axis = evaluation.axes[1]
        x_width = jnp.asarray(x_axis.length / x_axis.physical_count)
        z_width = jnp.asarray(z_axis.length / z_axis.physical_count)

        retained_nodes = np.asarray(retained_wall_axis.nodes)
        retained_measures = np.asarray(retained_wall_axis.quadrature_weights)
        evaluation_nodes = np.asarray(evaluation_wall_axis.nodes)
        order = np.argsort(retained_nodes)
        wall_widths = jnp.asarray(
            np.interp(
                evaluation_nodes,
                retained_nodes[order],
                retained_measures[order],
            ),
            dtype=evaluation_wall_axis.nodes.dtype,
        )
        wall_width_coefficients = evaluation_wall_axis.analyze(wall_widths)
        derivative = evaluation_wall_axis.derivative_matrix
        if derivative is None:
            raise ValueError("Prepared channel Chebyshev axis lacks modal derivatives.")
        width_gradient = jnp.real(
            evaluation_wall_axis.synthesize(derivative @ wall_width_coefficients)
        )
        ones = jnp.ones_like(wall_widths)
        directional = jnp.stack(
            (
                x_width * ones,
                wall_widths,
                z_width * ones,
            ),
            axis=-1,
        )[None, :, None, :]

        self.directional_widths = directional
        self.wall_normal_widths = wall_widths
        self.wall_normal_width_gradient = width_gradient
        self.streamwise_width = x_width
        self.spanwise_width = z_width
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "channel-les-filter-geometry",
                "retained_discretization": retained.prepared_id,
                "evaluation_discretization": evaluation.prepared_id,
                "fourier_width": "retained-physical-spacing",
                "wall_normal_width": "retained-nodal-measure-linear-evaluation",
                "commutation": "unmodeled-wall-normal",
            }
        )

    @property
    def filter_scale(self) -> LESFilterScale:
        """Return broadcast directional widths on the nonlinear evaluation grid."""
        return LESFilterScale(self.directional_widths)

    @property
    def noncommutation_evidence(self) -> Array:
        """Maximum local wall-normal scale commutator coefficient."""
        return jnp.max(jnp.abs(self.wall_normal_width_gradient))

    def wall_normal_scale_commutator(self, values: ArrayLike, /) -> Array:
        """Return ``(d Delta_y / dy) f``, evidencing local-scale noncommutation.

        This is diagnostic evidence, not a commutator correction to the LES
        equations. ``values`` must begin with the evaluation-grid physical shape.
        """
        value = jnp.asarray(values)
        if value.ndim < 3 or value.shape[1] != self.wall_normal_widths.size:
            raise ValueError(
                "Channel LES commutator values must carry the evaluation wall axis."
            )
        shape = (1, self.wall_normal_widths.size, 1) + (1,) * (value.ndim - 3)
        return self.wall_normal_width_gradient.reshape(shape) * value


class ChannelLESEvaluation(StrictModule):
    """Physical constitutive fields and retained modal SGS acceleration."""

    velocity: Array
    velocity_gradient: Array
    kinematic_viscosity: Array
    specific_deviatoric_stress: Array
    energy_transfer: Array
    subgrid_acceleration: Array
    finite: Array


class ChannelLESEnergyLedger(StrictModule):
    """Resolved kinetic-energy terms from molecular and modeled stresses.

    ``molecular_dissipation`` and ``subgrid_transfer`` are positive sinks.
    ``wall_power`` is positive when prescribed walls add resolved energy. The
    reported ``resolved_energy_rate`` is their exact signed combination; it
    excludes advection, pressure, body forcing, and mean-flow control.
    """

    molecular_dissipation: Array
    subgrid_transfer: Array
    wall_power: Array
    resolved_energy_rate: Array
    finite: Array


class ChannelLESExplicitRestriction(StrictModule):
    """Executable state-local explicit budget for channel SBDF2.

    The total rate combines a frozen-coefficient SGS diffusion bound with a
    resolved rotational-advection bound. The latter uses the retained Fourier
    wavenumbers and the induced infinity norm of the physical Chebyshev
    differentiation matrix, retaining its nonnormal amplification. The solver
    enforces the declared channel-SBDF2 radius before every accepted step.
    """

    maximum_resolved_speed: Array
    maximum_kinematic_viscosity: Array
    wall_normal_derivative_norm: Array
    horizontal_derivative_rate: Array
    advective_rate: Array
    diffusive_rate: Array
    total_explicit_rate: Array
    stability_radius: Array
    maximum_step: Array
    active: Array
    finite: Array
    temporal_method: str = eqx.field(static=True)

    def permits(self, step_size: ArrayLike, /) -> Array:
        """Return whether a positive finite step obeys the complete budget."""
        step = jnp.asarray(step_size, dtype=self.total_explicit_rate.dtype).reshape(())
        admissible = jnp.isfinite(step) & (step > 0.0)
        return self.finite & admissible & (~self.active | (step <= self.maximum_step))


class ChannelLESDiagnostics(StrictModule):
    """Channel constraints, LES energy evidence, and constitutive success."""

    kinetic_energy: Array
    divergence_norm: Array
    wall_residual: Array
    finite: Array
    valid: Array
    energy_ledger: ChannelLESEnergyLedger
    explicit_restriction: ChannelLESExplicitRestriction

    @property
    def successful(self) -> Array:
        return self.valid


class CompiledChannelLESDynamics(StrictModule):
    """Wall-resolved algebraic LES composed with compiled channel dynamics.

    The resolved velocity uses the retained Fourier–Chebyshev–Fourier grid. SGS
    stress is evaluated on the existing dealiased grid from all nine mixed
    derivatives ``gradient[i, j] = d velocity[i] / d x[j]``. Its negative
    divergence is restricted to retained modes before the channel Stokes solve
    enforces incompressibility and pressure. Velocity-owned walls retain full
    no-slip KKT constraints. Traction-owned walls retain only wall-normal
    velocity constraints, remove the SGS tangential wall flux, and leave total
    tangential traction to the channel boundary owner.

    The implicit variable wall-normal grid filter is explicitly marked
    noncommuting and no commutator model is supplied. Prepared coefficients must
    therefore have provenance for that exact unmodeled-noncommutation contract.
    Algebraic invariant and ratio laws are not generally finite polynomials, so
    the prepared padding grid controls aliasing without claiming exact
    constitutive dealiasing.
    """

    base: CompiledChannelFlowDynamics
    model: PreparedAlgebraicLESModel
    filter_geometry: ChannelLESFilterGeometry
    wall_normal_derivative_norm: Array
    horizontal_derivative_rate: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    base_compilation_id: str = eqx.field(static=True)
    les_prepared_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    source_hash: str = eqx.field(static=True)

    def __init__(
        self,
        base: CompiledChannelFlowDynamics,
        model: PreparedAlgebraicLESModel,
        /,
    ):
        if not isinstance(base, CompiledChannelFlowDynamics):
            raise TypeError("base must be CompiledChannelFlowDynamics.")
        if not isinstance(model, PreparedAlgebraicLESModel):
            raise TypeError("model must be PreparedAlgebraicLESModel.")
        discretization = base.discretization
        resolved_filter = model.provenance.resolved_filter
        expected_filter = channel_les_filter(discretization)
        if resolved_filter.filter_id != expected_filter.filter_id:
            raise ValueError(
                "Prepared LES model filter does not match the channel implicit-grid filter."
            )
        if model.provenance.discretization_id != discretization.prepared_id:
            raise ValueError(
                "Prepared LES model provenance does not match the retained channel grid."
            )
        geometry = ChannelLESFilterGeometry(
            discretization,
            base.spatial_method.dealiasing.evaluation,
        )
        evaluation_axis = base.spatial_method.dealiasing.evaluation.axes[1]
        if (
            evaluation_axis.modal_transform is None
            or evaluation_axis.derivative_matrix is None
        ):
            raise ValueError(
                "Channel LES stability requires explicit Chebyshev transform matrices."
            )
        physical_wall_derivative = (
            np.asarray(evaluation_axis.modal_transform.synthesis)
            @ np.asarray(evaluation_axis.derivative_matrix)
            @ np.asarray(evaluation_axis.modal_transform.analysis)
        )
        wall_derivative_norm = float(
            np.max(np.sum(np.abs(physical_wall_derivative), axis=1))
        )
        x_axis, _, z_axis = discretization.axes
        streamwise_rate = float(
            np.max(
                np.abs(
                    2.0
                    * np.pi
                    * np.asarray(x_axis.modes.mode_numbers)
                    / float(x_axis.length)
                )
            )
        )
        spanwise_rate = float(
            np.max(
                np.abs(
                    2.0
                    * np.pi
                    * np.asarray(z_axis.modes.mode_numbers)
                    / float(z_axis.length)
                )
            )
        )
        horizontal_rate = streamwise_rate + spanwise_rate
        self.base = base
        self.model = model
        self.filter_geometry = geometry
        self.state_shape = base.state_shape
        self.base_compilation_id = base.compilation_id
        self.wall_normal_derivative_norm = jnp.asarray(wall_derivative_norm)
        self.horizontal_derivative_rate = jnp.asarray(horizontal_rate)
        self.les_prepared_id = model.prepared_id
        self.compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-channel-les",
                "base": base.compilation_id,
                "prepared_les_model": model.prepared_id,
                "filter_geometry": geometry.geometry_id,
                "explicit_stability": {
                    "temporal_method": "channel-sbdf2",
                    "radius": float(0.25).hex(),
                    "advection": "rotational-induced-infinity-bound",
                    "wall_normal": "physical-chebyshev-induced-infinity-norm",
                    "sgs": "frozen-coefficient-directional-diffusion",
                },
            }
        )
        self.source_hash = base.source_hash

    @property
    def problem(self):
        return self.base.problem

    @property
    def stokes_plan(self):
        return self.base.stokes_plan

    @property
    def spatial_method(self):
        return self.base.spatial_method

    @property
    def discretization(self):
        return self.base.discretization

    @property
    def horizontal_admissibility(self) -> Array:
        return self.base.horizontal_admissibility

    @property
    def resolved_filter(self) -> ResolvedLESFilter:
        return self.model.provenance.resolved_filter

    def validate_state(
        self, state: ArrayLike, /, *, owner: str = "Channel LES state"
    ) -> Array:
        return self.base.validate_state(state, owner=owner)

    def admissible_modes(self, state: ArrayLike, /) -> Array:
        return self.base.admissible_modes(state)

    def project_state(self, values: ArrayLike, /) -> Array:
        return self.base.project_state(values)

    def reconstruct_state(self, state: ArrayLike, /) -> Array:
        return self.base.reconstruct_state(state)

    def prepare_stokes(self, shift: ArrayLike, /):
        return self.base.prepare_stokes(shift)

    def evaluate_subgrid(self, state: ArrayLike, /) -> ChannelLESEvaluation:
        """Evaluate dealiased stress and its retained negative divergence."""
        value = self.admissible_modes(state)
        dealiasing = self.spatial_method.dealiasing
        evaluation = dealiasing.evaluation
        padded = dealiasing.embed(value)
        velocity = evaluation.reconstruct(padded)
        derivative_modes = tuple(
            evaluation.modal_derivative(padded, axis=axis) for axis in range(3)
        )
        gradient_modes = jnp.stack(derivative_modes, axis=-1)
        gradient = evaluation.reconstruct(gradient_modes)
        constitutive = self.model.evaluate(
            AlgebraicLESInputs(gradient, self.filter_geometry.filter_scale)
        )
        stress = constitutive.specific_deviatoric_stress
        if self.stokes_plan.tangential_boundary == "traction":
            stress = stress.at[:, 0, :, 0, 1].set(0.0)
            stress = stress.at[:, 0, :, 2, 1].set(0.0)
            stress = stress.at[:, -1, :, 0, 1].set(0.0)
            stress = stress.at[:, -1, :, 2, 1].set(0.0)
        energy_transfer = -ein.contract(
            "...ij,...ij->...", stress, gradient, backend="jax"
        )
        stress_modes = evaluation.project(stress)
        stress_divergence = sum(
            (
                evaluation.modal_derivative(
                    stress_modes[..., :, axis],
                    axis=axis,
                )
                for axis in range(3)
            ),
            start=jnp.zeros(stress_modes.shape[:-1], dtype=stress_modes.dtype),
        )
        subgrid = self.admissible_modes(-dealiasing.restrict(stress_divergence))
        finite = (
            jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(constitutive.kinematic_viscosity))
            & jnp.all(jnp.isfinite(stress))
            & jnp.all(jnp.isfinite(energy_transfer))
            & jnp.all(jnp.isfinite(subgrid))
        )
        return ChannelLESEvaluation(
            velocity=velocity,
            velocity_gradient=gradient,
            kinematic_viscosity=constitutive.kinematic_viscosity,
            specific_deviatoric_stress=stress,
            energy_transfer=energy_transfer,
            subgrid_acceleration=subgrid,
            finite=finite,
        )

    def _energy_ledger_from_evaluation(
        self, evaluation: ChannelLESEvaluation, /
    ) -> ChannelLESEnergyLedger:
        grid = self.spatial_method.dealiasing.evaluation
        weights = grid.quadrature_weights
        gradient_squared = ein.contract(
            "...ij,...ij->...",
            evaluation.velocity_gradient,
            evaluation.velocity_gradient,
            backend="jax",
        )
        molecular = jnp.sum(weights * self.problem.viscosity * gradient_squared)
        subgrid = jnp.sum(weights * evaluation.energy_transfer)

        horizontal_weights = (
            grid.axes[0].quadrature_weights[:, None]
            * grid.axes[2].quadrature_weights[None, :]
        )
        velocity = evaluation.velocity
        gradient_y = evaluation.velocity_gradient[..., :, 1]
        stress_y = evaluation.specific_deviatoric_stress[..., :, 1]
        lower_traction = -self.problem.viscosity * gradient_y[:, 0, :] + stress_y[:, 0, :]
        upper_traction = (
            self.problem.viscosity * gradient_y[:, -1, :] - stress_y[:, -1, :]
        )
        lower_power_density = ein.contract(
            "...i,...i->...", velocity[:, 0, :], lower_traction, backend="jax"
        )
        upper_power_density = ein.contract(
            "...i,...i->...", velocity[:, -1, :], upper_traction, backend="jax"
        )
        wall_power = jnp.sum(
            horizontal_weights * (lower_power_density + upper_power_density)
        )
        resolved_rate = wall_power - molecular - subgrid
        finite = (
            evaluation.finite
            & jnp.isfinite(molecular)
            & jnp.isfinite(subgrid)
            & jnp.isfinite(wall_power)
            & jnp.isfinite(resolved_rate)
        )
        return ChannelLESEnergyLedger(
            molecular_dissipation=molecular,
            subgrid_transfer=subgrid,
            wall_power=wall_power,
            resolved_energy_rate=resolved_rate,
            finite=finite,
        )

    def energy_ledger(self, state: ArrayLike, /) -> ChannelLESEnergyLedger:
        """Return the molecular, SGS-transfer, and wall-work energy terms."""
        return self._energy_ledger_from_evaluation(self.evaluate_subgrid(state))

    def _restriction_from_evaluation(
        self, evaluation: ChannelLESEvaluation, /
    ) -> ChannelLESExplicitRestriction:
        inverse_width_squared = jnp.sum(
            1.0 / self.filter_geometry.directional_widths**2,
            axis=-1,
        )
        local_diffusive_rate = (
            2.0 * evaluation.kinematic_viscosity * inverse_width_squared
        )
        diffusive_rate = jnp.max(local_diffusive_rate)
        maximum_viscosity = jnp.max(evaluation.kinematic_viscosity)
        maximum_speed = jnp.max(
            jnp.sqrt(
                jnp.sum(
                    jnp.real(evaluation.velocity * jnp.conj(evaluation.velocity)),
                    axis=-1,
                )
            )
        )
        derivative_rate = self.horizontal_derivative_rate.astype(
            maximum_speed.dtype
        ) + self.wall_normal_derivative_norm.astype(maximum_speed.dtype)
        advective_rate = 4.0 * maximum_speed * derivative_rate
        total_rate = advective_rate + diffusive_rate
        active = total_rate > 0.0
        stability_radius = jnp.asarray(0.25, dtype=total_rate.dtype)
        maximum_step = jnp.where(
            active,
            stability_radius / total_rate,
            jnp.asarray(jnp.inf, dtype=total_rate.dtype),
        )
        finite = (
            evaluation.finite
            & jnp.isfinite(maximum_speed)
            & (maximum_speed >= 0.0)
            & jnp.isfinite(maximum_viscosity)
            & (maximum_viscosity >= 0.0)
            & jnp.isfinite(self.wall_normal_derivative_norm)
            & (self.wall_normal_derivative_norm >= 0.0)
            & jnp.isfinite(self.horizontal_derivative_rate)
            & (self.horizontal_derivative_rate >= 0.0)
            & jnp.isfinite(advective_rate)
            & (advective_rate >= 0.0)
            & jnp.isfinite(diffusive_rate)
            & (diffusive_rate >= 0.0)
            & jnp.isfinite(total_rate)
            & (total_rate >= 0.0)
        )
        return ChannelLESExplicitRestriction(
            maximum_resolved_speed=maximum_speed,
            maximum_kinematic_viscosity=maximum_viscosity,
            wall_normal_derivative_norm=self.wall_normal_derivative_norm,
            horizontal_derivative_rate=self.horizontal_derivative_rate,
            advective_rate=advective_rate,
            diffusive_rate=diffusive_rate,
            total_explicit_rate=total_rate,
            stability_radius=stability_radius,
            maximum_step=maximum_step,
            active=active,
            finite=finite,
            temporal_method="channel-sbdf2",
        )

    def explicit_restriction(self, state: ArrayLike, /) -> ChannelLESExplicitRestriction:
        """Return the enforced channel-SBDF2 explicit stability contract."""
        return self._restriction_from_evaluation(self.evaluate_subgrid(state))

    def state_diagnostics(self, state: ArrayLike, /) -> ChannelLESDiagnostics:
        velocity: ChannelVelocityDiagnostics = self.base.state_diagnostics(state)
        evaluation = self.evaluate_subgrid(state)
        ledger = self._energy_ledger_from_evaluation(evaluation)
        restriction = self._restriction_from_evaluation(evaluation)
        finite = velocity.finite & ledger.finite & restriction.finite
        valid = velocity.valid & finite
        return ChannelLESDiagnostics(
            kinetic_energy=velocity.kinetic_energy,
            divergence_norm=velocity.divergence_norm,
            wall_residual=velocity.wall_residual,
            finite=finite,
            valid=valid,
            energy_ledger=ledger,
            explicit_restriction=restriction,
        )

    def nonlinear(self, time: Array, state: ArrayLike, args: Any = None, /) -> Array:
        """Add retained SGS stress divergence before the channel Stokes solve."""
        resolved = self.base.nonlinear(time, state, args)
        subgrid = self.evaluate_subgrid(state).subgrid_acceleration
        return self.admissible_modes(resolved + subgrid)


def channel_les_filter(
    discretization: TensorSpectralDiscretization,
    /,
) -> ResolvedLESFilter:
    """Return the canonical wall-resolved implicit filter for one channel grid."""
    if not isinstance(discretization, TensorSpectralDiscretization):
        raise TypeError("discretization must be TensorSpectralDiscretization.")
    if tuple(axis.family for axis in discretization.axes) != (
        "fourier",
        "chebyshev",
        "fourier",
    ):
        raise ValueError("Channel LES requires a Fourier x Chebyshev x Fourier grid.")
    return ResolvedLESFilter(
        "fourier-chebyshev-fourier-implicit-grid",
        family="implicit-grid-volume",
        axis_names=tuple(discretization.plan.axis_names),
        topology="tensor-product",
        boundary_class="wall-bounded",
        scale_rule="volume-equivalent",
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )


def compile_channel_les(
    base: CompiledChannelFlowDynamics,
    model: PreparedAlgebraicLESModel,
    /,
) -> CompiledChannelLESDynamics:
    """Bind a prepared algebraic model to compiled wall-resolved channel flow."""
    return CompiledChannelLESDynamics(base, model)


__all__ = [
    "ChannelLESDiagnostics",
    "ChannelLESEnergyLedger",
    "ChannelLESEvaluation",
    "ChannelLESExplicitRestriction",
    "ChannelLESFilterGeometry",
    "CompiledChannelLESDynamics",
    "channel_les_filter",
    "compile_channel_les",
]
