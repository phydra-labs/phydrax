#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..discretization import CochainDiscretization
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    hermitian_inverse_sqrt,
    hermitian_sqrt,
    HermitianSpectrum,
    LinearSolvePolicy,
    LinearSystem,
    prepare,
    solve as solve_linear,
)
from ._maxwell import (
    _apply_hodge_metric,
    AbstractMaxwellConstitutivePlan,
    AbstractPreparedMaxwellConstitutive,
    MaxwellCapabilities,
)


class MaxwellConstitutiveEvidence(StrictModule):
    """Hermitian positivity and conditioning evidence for electric/magnetic maps."""

    electric_minimum_eigenvalue: Array
    magnetic_minimum_eigenvalue: Array
    electric_condition_number: Array
    magnetic_condition_number: Array
    evidence_id: str = eqx.field(static=True)


class MatrixMaxwellConstitutivePlan(AbstractMaxwellConstitutivePlan):
    """Budgeted dense constitutive maps for coupled anisotropic cochains."""

    electric_matrix: Array
    magnetic_matrix: Array
    maximum_dense_dofs: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        electric_matrix: ArrayLike,
        magnetic_matrix: ArrayLike,
        /,
        *,
        maximum_dense_dofs: int = 4096,
        plan_id: str | None = None,
    ):
        electric = jnp.asarray(electric_matrix)
        magnetic = jnp.asarray(magnetic_matrix)
        if electric.ndim != 2 or electric.shape[0] != electric.shape[1]:
            raise ValueError("electric_matrix must be square.")
        if magnetic.ndim != 2 or magnetic.shape[0] != magnetic.shape[1]:
            raise ValueError("magnetic_matrix must be square.")
        maximum = int(maximum_dense_dofs)
        if maximum <= 0:
            raise ValueError("maximum_dense_dofs must be positive.")
        if max(electric.shape[0], magnetic.shape[0]) > maximum:
            raise ValueError("Constitutive matrix exceeds maximum_dense_dofs.")
        if not jnp.issubdtype(electric.dtype, jnp.inexact):
            electric = electric.astype(float)
        if not jnp.issubdtype(magnetic.dtype, jnp.inexact):
            magnetic = magnetic.astype(float)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "matrix-maxwell-constitutive-plan",
                    "electric": array_tree_fingerprint(electric),
                    "magnetic": array_tree_fingerprint(magnetic),
                    "maximum_dense_dofs": maximum,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.electric_matrix = electric
        self.magnetic_matrix = magnetic
        self.maximum_dense_dofs = maximum
        self.plan_id = identifier

    def prepare(
        self,
        cochain: CochainDiscretization,
        /,
    ) -> PreparedMatrixMaxwellConstitutive:
        return PreparedMatrixMaxwellConstitutive(self, cochain)


def _metric_spectrum(
    name: str,
    matrix: Array,
    metric: Array,
    /,
) -> tuple[np.ndarray, float, float]:
    host = np.asarray(matrix)
    weight = np.asarray(metric)
    count = weight.shape[0]
    if host.shape != (count, count) or weight.ndim not in (1, 2):
        raise ValueError(f"{name} matrix shape does not match its cochain degree.")
    if weight.ndim == 2 and weight.shape != (count, count):
        raise ValueError(f"{name} Hodge metric must be square.")
    if np.any(~np.isfinite(host)) or np.any(~np.isfinite(weight)):
        raise ValueError(f"{name} matrix and Hodge metric must be finite.")
    metric_matrix = (
        jnp.diag(jnp.asarray(weight)) if weight.ndim == 1 else jnp.asarray(weight)
    )
    metric_tolerance = np.finfo(metric_matrix.real.dtype).eps * max(
        1.0,
        float(jnp.max(jnp.abs(metric_matrix))),
    )
    metric_spectrum = HermitianSpectrum(
        metric_matrix,
        tolerance=64.0 * metric_tolerance,
    )
    if (
        not bool(metric_spectrum.valid)
        or float(metric_spectrum.minimum_eigenvalue) <= 64.0 * metric_tolerance
    ):
        raise ValueError(f"{name} Hodge metric must be positive definite.")
    weighted = metric_matrix @ jnp.asarray(host)
    tolerance = np.finfo(weighted.real.dtype).eps * max(
        1.0,
        float(jnp.max(jnp.abs(weighted))),
    )
    weighted_residual = jnp.max(jnp.abs(weighted - jnp.conj(weighted.T)))
    if not bool(weighted_residual <= 64.0 * tolerance):
        raise ValueError(f"{name} constitutive map is not metric-Hermitian.")
    root_result = hermitian_sqrt(
        metric_matrix,
        tolerance=64.0 * metric_tolerance,
    )
    inverse_result = hermitian_inverse_sqrt(
        metric_matrix,
        tolerance=64.0 * metric_tolerance,
    )
    if not bool(root_result.valid & inverse_result.valid):
        raise ValueError(f"{name} Hodge metric square roots are invalid.")
    symmetric = root_result.value @ jnp.asarray(host) @ inverse_result.value
    constitutive_spectrum = HermitianSpectrum(
        symmetric,
        tolerance=64.0 * tolerance,
    )
    minimum = float(constitutive_spectrum.minimum_eigenvalue)
    maximum = float(jnp.max(constitutive_spectrum.eigenvalues))
    if not bool(constitutive_spectrum.valid) or minimum <= 64.0 * tolerance:
        raise ValueError(f"{name} constitutive map is not positive definite.")
    return np.asarray(symmetric), minimum, maximum / minimum


class PreparedMatrixMaxwellConstitutive(AbstractPreparedMaxwellConstitutive):
    """Metric-Hermitian positive constitutive maps with certified dense solves."""

    electric_matrix: Array
    magnetic_matrix: Array
    electric_solver: Any
    magnetic_solver: Any
    evidence: MaxwellConstitutiveEvidence
    capabilities: MaxwellCapabilities
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MatrixMaxwellConstitutivePlan,
        cochain: CochainDiscretization,
        /,
    ):
        if cochain.max_degree != 3:
            raise ValueError("Maxwell constitutive preparation requires dimension three.")
        _, electric_minimum, electric_condition = _metric_spectrum(
            "electric",
            plan.electric_matrix,
            cochain.hodge_metric(1),
        )
        _, magnetic_minimum, magnetic_condition = _metric_spectrum(
            "magnetic",
            plan.magnetic_matrix,
            cochain.hodge_metric(2),
        )
        evidence_id = canonical_fingerprint(
            {
                "kind": "maxwell-constitutive-evidence",
                "plan": plan.plan_id,
                "cochain": cochain.prepared_id,
                "electric_minimum": electric_minimum,
                "magnetic_minimum": magnetic_minimum,
                "electric_condition": electric_condition,
                "magnetic_condition": magnetic_condition,
            }
        )
        self.electric_matrix = plan.electric_matrix
        self.magnetic_matrix = plan.magnetic_matrix
        policy = LinearSolvePolicy(
            DenseLU(),
            failure=FailurePolicy("error"),
        )
        self.electric_solver = prepare(
            LinearSystem(
                DenseLinearOperator(self.electric_matrix),
                problem_id=f"{plan.plan_id}:electric-constitutive",
            ),
            policy,
        )
        self.magnetic_solver = prepare(
            LinearSystem(
                DenseLinearOperator(self.magnetic_matrix),
                problem_id=f"{plan.plan_id}:magnetic-constitutive",
            ),
            policy,
        )
        self.evidence = MaxwellConstitutiveEvidence(
            electric_minimum_eigenvalue=jnp.asarray(electric_minimum),
            magnetic_minimum_eigenvalue=jnp.asarray(magnetic_minimum),
            electric_condition_number=jnp.asarray(electric_condition),
            magnetic_condition_number=jnp.asarray(magnetic_condition),
            evidence_id=evidence_id,
        )
        self.capabilities = MaxwellCapabilities(
            lossless=True,
            passive=True,
            reversible=True,
            complex_required=(
                jnp.issubdtype(self.electric_matrix.dtype, jnp.complexfloating)
                or jnp.issubdtype(self.magnetic_matrix.dtype, jnp.complexfloating)
            ),
            structured_only=False,
            frequency_domain=True,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-matrix-maxwell-constitutive",
                "plan": plan.plan_id,
                "cochain": cochain.prepared_id,
                "evidence": evidence_id,
            }
        )

    def initialize_state(self, /) -> None:
        return None

    def validate_state(self, state: Any, /) -> None:
        if state is not None:
            raise ValueError("Instantaneous matrix material state must be None.")

    def electric_field(self, displacement: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return solve_linear(self.electric_solver, displacement).value

    def electric_displacement(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.electric_matrix @ electric

    def magnetic_field(self, flux: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return solve_linear(self.magnetic_solver, flux).value

    def magnetic_flux(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.magnetic_matrix @ magnetic

    def electric_conduction(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return jnp.zeros_like(electric)

    def magnetic_conduction(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return jnp.zeros_like(magnetic)

    def dissipated_power(
        self,
        electric: Array,
        magnetic: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        del electric, magnetic, electric_star, magnetic_star
        self.validate_state(state)
        return jnp.asarray(0.0)

    def advance_state(
        self,
        time: Array,
        state: Any,
        displacement: Array,
        magnetic_flux: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> None:
        del time, displacement, magnetic_flux, step_size, args
        self.validate_state(state)

    def energy(
        self,
        displacement: Array,
        magnetic_flux: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        electric = self.electric_field(displacement, state)
        magnetic = self.magnetic_field(magnetic_flux, state)
        return 0.5 * jnp.real(
            jnp.vdot(electric, _apply_hodge_metric(electric_star, displacement))
            + jnp.vdot(magnetic, _apply_hodge_metric(magnetic_star, magnetic_flux))
        )

    def energy_rate(
        self,
        displacement: Array,
        magnetic_flux: Array,
        displacement_rate: Array,
        magnetic_rate: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        electric = self.electric_field(displacement, state)
        magnetic = self.magnetic_field(magnetic_flux, state)
        return jnp.real(
            jnp.vdot(electric, _apply_hodge_metric(electric_star, displacement_rate))
            + jnp.vdot(magnetic, _apply_hodge_metric(magnetic_star, magnetic_rate))
        )

    def wave_speed_bound(self, /) -> Array:
        return 1.0 / jnp.sqrt(
            self.evidence.electric_minimum_eigenvalue
            * self.evidence.magnetic_minimum_eigenvalue
        )


class ConductiveMaxwellConstitutivePlan(AbstractMaxwellConstitutivePlan):
    """Passive diagonal material with electric and magnetic conductivity."""

    permittivity: Array
    permeability: Array
    electric_conductivity: Array
    magnetic_conductivity: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        permittivity: ArrayLike = 1.0,
        permeability: ArrayLike = 1.0,
        electric_conductivity: ArrayLike = 0.0,
        magnetic_conductivity: ArrayLike = 0.0,
    ):
        self.permittivity = jnp.asarray(permittivity)
        self.permeability = jnp.asarray(permeability)
        self.electric_conductivity = jnp.asarray(electric_conductivity)
        self.magnetic_conductivity = jnp.asarray(magnetic_conductivity)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conductive-maxwell-constitutive-plan",
                "permittivity": array_tree_fingerprint(self.permittivity),
                "permeability": array_tree_fingerprint(self.permeability),
                "electric_conductivity": array_tree_fingerprint(
                    self.electric_conductivity
                ),
                "magnetic_conductivity": array_tree_fingerprint(
                    self.magnetic_conductivity
                ),
            }
        )

    def prepare(
        self,
        cochain: CochainDiscretization,
        /,
    ) -> PreparedConductiveMaxwellConstitutive:
        return PreparedConductiveMaxwellConstitutive(self, cochain)


def _nonnegative_material(name: str, value: ArrayLike, count: int, /) -> Array:
    array = jnp.asarray(value)
    if jnp.iscomplexobj(array):
        raise TypeError(f"{name} must be real.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    if array.shape not in ((), (1,), (count,)):
        raise ValueError(f"{name} must be scalar or have shape ({count},).")
    array = jnp.broadcast_to(array, (count,))
    return eqx.error_if(
        array,
        jnp.any(~jnp.isfinite(array)) | jnp.any(array < 0.0),
        f"{name} must be finite and nonnegative.",
    )


class PreparedConductiveMaxwellConstitutive(AbstractPreparedMaxwellConstitutive):
    permittivity: Array
    permeability: Array
    electric_conductivity: Array
    magnetic_conductivity: Array
    capabilities: MaxwellCapabilities
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ConductiveMaxwellConstitutivePlan,
        cochain: CochainDiscretization,
        /,
    ):
        from ._maxwell import _positive_material

        self.permittivity = _positive_material(
            "permittivity", plan.permittivity, cochain.cell_counts[1]
        )
        self.permeability = _positive_material(
            "permeability", plan.permeability, cochain.cell_counts[2]
        )
        self.electric_conductivity = _nonnegative_material(
            "electric_conductivity",
            plan.electric_conductivity,
            cochain.cell_counts[1],
        )
        self.magnetic_conductivity = _nonnegative_material(
            "magnetic_conductivity",
            plan.magnetic_conductivity,
            cochain.cell_counts[2],
        )
        lossless = bool(
            jnp.all(self.electric_conductivity == 0.0)
            & jnp.all(self.magnetic_conductivity == 0.0)
        )
        self.capabilities = MaxwellCapabilities(
            lossless=lossless,
            passive=True,
            reversible=lossless,
            structured_only=False,
            frequency_domain=lossless,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-conductive-maxwell-constitutive",
                "plan": plan.plan_id,
                "cochain": cochain.prepared_id,
            }
        )

    def initialize_state(self, /) -> None:
        return None

    def validate_state(self, state: Any, /) -> None:
        if state is not None:
            raise ValueError("Conductive instantaneous material state must be None.")

    def electric_field(self, displacement: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return displacement / self.permittivity

    def electric_displacement(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.permittivity * electric

    def magnetic_field(self, flux: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return flux / self.permeability

    def magnetic_flux(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.permeability * magnetic

    def electric_conduction(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.electric_conductivity * electric

    def magnetic_conduction(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.magnetic_conductivity * magnetic

    def dissipated_power(
        self,
        electric: Array,
        magnetic: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        self.validate_state(state)
        return jnp.real(
            jnp.vdot(
                electric,
                _apply_hodge_metric(
                    electric_star,
                    self.electric_conductivity * electric,
                ),
            )
            + jnp.vdot(
                magnetic,
                _apply_hodge_metric(
                    magnetic_star,
                    self.magnetic_conductivity * magnetic,
                ),
            )
        )

    def advance_state(
        self,
        time: Array,
        state: Any,
        displacement: Array,
        magnetic_flux: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> None:
        del time, displacement, magnetic_flux, step_size, args
        self.validate_state(state)

    def energy(
        self,
        displacement: Array,
        magnetic_flux: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        electric = self.electric_field(displacement, state)
        magnetic = self.magnetic_field(magnetic_flux, state)
        return 0.5 * jnp.real(
            jnp.vdot(electric, _apply_hodge_metric(electric_star, displacement))
            + jnp.vdot(magnetic, _apply_hodge_metric(magnetic_star, magnetic_flux))
        )

    def energy_rate(
        self,
        displacement: Array,
        magnetic_flux: Array,
        displacement_rate: Array,
        magnetic_rate: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        return jnp.real(
            jnp.vdot(
                self.electric_field(displacement, state),
                _apply_hodge_metric(electric_star, displacement_rate),
            )
            + jnp.vdot(
                self.magnetic_field(magnetic_flux, state),
                _apply_hodge_metric(magnetic_star, magnetic_rate),
            )
        )

    def wave_speed_bound(self, /) -> Array:
        return jnp.sqrt(jnp.max(1.0 / self.permeability) / jnp.min(self.permittivity))


class DispersiveMaxwellState(StrictModule):
    polarization: Array
    velocity: Array


class LorentzDrudeMaxwellConstitutivePlan(AbstractMaxwellConstitutivePlan):
    """Passive Lorentz/Drude auxiliary differential-equation material."""

    permittivity_infinity: Array
    permeability: Array
    resonance_frequency: Array
    damping: Array
    oscillator_strength: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        resonance_frequency: ArrayLike,
        damping: ArrayLike,
        oscillator_strength: ArrayLike,
        /,
        *,
        permittivity_infinity: ArrayLike = 1.0,
        permeability: ArrayLike = 1.0,
    ):
        frequency = jnp.asarray(resonance_frequency, dtype=float)
        damping_ = jnp.asarray(damping, dtype=float)
        strength = jnp.asarray(oscillator_strength, dtype=float)
        if frequency.ndim != 1 or frequency.size == 0:
            raise ValueError("Lorentz/Drude poles require nonempty frequency vectors.")
        if damping_.shape != frequency.shape or strength.shape != frequency.shape:
            raise ValueError("Lorentz/Drude pole arrays must have matching shapes.")
        invalid = (
            jnp.any(~jnp.isfinite(frequency))
            | jnp.any(~jnp.isfinite(damping_))
            | jnp.any(~jnp.isfinite(strength))
            | jnp.any(frequency < 0.0)
            | jnp.any(damping_ < 0.0)
            | jnp.any(strength <= 0.0)
        )
        frequency = eqx.error_if(
            frequency,
            invalid,
            "Passive Lorentz/Drude poles require finite nonnegative frequency/damping and positive strength.",
        )
        self.permittivity_infinity = jnp.asarray(permittivity_infinity)
        self.permeability = jnp.asarray(permeability)
        self.resonance_frequency = frequency
        self.damping = damping_
        self.oscillator_strength = strength
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lorentz-drude-maxwell-plan",
                "frequency": array_tree_fingerprint(frequency),
                "damping": array_tree_fingerprint(damping_),
                "strength": array_tree_fingerprint(strength),
                "permittivity_infinity": array_tree_fingerprint(
                    self.permittivity_infinity
                ),
                "permeability": array_tree_fingerprint(self.permeability),
            }
        )

    def prepare(
        self,
        cochain: CochainDiscretization,
        /,
    ) -> PreparedLorentzDrudeMaxwellConstitutive:
        return PreparedLorentzDrudeMaxwellConstitutive(self, cochain)


class PreparedLorentzDrudeMaxwellConstitutive(AbstractPreparedMaxwellConstitutive):
    permittivity_infinity: Array
    permeability: Array
    resonance_frequency: Array
    damping: Array
    oscillator_strength: Array
    electric_count: int = eqx.field(static=True)
    capabilities: MaxwellCapabilities
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: LorentzDrudeMaxwellConstitutivePlan,
        cochain: CochainDiscretization,
        /,
    ):
        from ._maxwell import _positive_material

        self.electric_count = cochain.cell_counts[1]
        self.permittivity_infinity = _positive_material(
            "permittivity_infinity",
            plan.permittivity_infinity,
            self.electric_count,
        )
        self.permeability = _positive_material(
            "permeability",
            plan.permeability,
            cochain.cell_counts[2],
        )
        self.resonance_frequency = plan.resonance_frequency
        self.damping = plan.damping
        self.oscillator_strength = plan.oscillator_strength
        self.capabilities = MaxwellCapabilities(
            lossless=bool(jnp.all(self.damping == 0.0)),
            passive=True,
            dispersive=True,
            reversible=False,
            structured_only=False,
            frequency_domain=False,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-lorentz-drude-maxwell",
                "plan": plan.plan_id,
                "cochain": cochain.prepared_id,
            }
        )

    def initialize_state(self, /) -> DispersiveMaxwellState:
        shape = (self.resonance_frequency.size, self.electric_count)
        return DispersiveMaxwellState(jnp.zeros(shape), jnp.zeros(shape))

    def validate_state(self, state: Any, /) -> None:
        if not isinstance(state, DispersiveMaxwellState):
            raise TypeError("Dispersive material requires DispersiveMaxwellState.")
        shape = (self.resonance_frequency.size, self.electric_count)
        if state.polarization.shape != shape or state.velocity.shape != shape:
            raise ValueError("Dispersive Maxwell state has wrong shape.")

    def electric_field(self, displacement: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return (
            displacement - jnp.sum(state.polarization, axis=0)
        ) / self.permittivity_infinity

    def electric_displacement(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.permittivity_infinity * electric + jnp.sum(state.polarization, axis=0)

    def magnetic_field(self, flux: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return flux / self.permeability

    def magnetic_flux(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.permeability * magnetic

    def electric_conduction(self, electric: Array, state: Any, /) -> Array:
        del electric
        self.validate_state(state)
        return jnp.zeros((self.electric_count,))

    def magnetic_conduction(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return jnp.zeros_like(magnetic)

    def advance_state(
        self,
        time: Array,
        state: Any,
        displacement: Array,
        magnetic_flux: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> DispersiveMaxwellState:
        del time, magnetic_flux, args
        self.validate_state(state)
        electric = self.electric_field(displacement, state)
        frequency = self.resonance_frequency[:, None]
        damping = self.damping[:, None]
        strength = self.oscillator_strength[:, None]
        acceleration = (
            strength * electric[None, :]
            - frequency**2 * state.polarization
            - damping * state.velocity
        )
        velocity = (state.velocity + step_size * acceleration) / (
            1.0 + step_size * damping
        )
        polarization = state.polarization + step_size * velocity
        return DispersiveMaxwellState(polarization, velocity)

    def dissipated_power(
        self,
        electric: Array,
        magnetic: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        del electric, magnetic, magnetic_star
        self.validate_state(state)
        density = jnp.sum(
            self.damping[:, None] * state.velocity**2 / self.oscillator_strength[:, None],
            axis=0,
        )
        return jnp.sum(_apply_hodge_metric(electric_star, density))

    def energy(
        self,
        displacement: Array,
        magnetic_flux: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        self.validate_state(state)
        electric = self.electric_field(displacement, state)
        magnetic = self.magnetic_field(magnetic_flux, state)
        oscillator = jnp.sum(
            (
                state.velocity**2
                + self.resonance_frequency[:, None] ** 2 * state.polarization**2
            )
            / self.oscillator_strength[:, None],
            axis=0,
        )
        return 0.5 * jnp.real(
            jnp.vdot(
                electric,
                _apply_hodge_metric(
                    electric_star,
                    self.permittivity_infinity * electric,
                ),
            )
            + jnp.vdot(
                magnetic,
                _apply_hodge_metric(magnetic_star, magnetic_flux),
            )
            + jnp.sum(_apply_hodge_metric(electric_star, oscillator))
        )

    def energy_rate(
        self,
        displacement: Array,
        magnetic_flux: Array,
        displacement_rate: Array,
        magnetic_rate: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        return jnp.real(
            jnp.vdot(
                self.electric_field(displacement, state),
                _apply_hodge_metric(electric_star, displacement_rate),
            )
            + jnp.vdot(
                self.magnetic_field(magnetic_flux, state),
                _apply_hodge_metric(magnetic_star, magnetic_rate),
            )
        )

    def wave_speed_bound(self, /) -> Array:
        return jnp.sqrt(
            jnp.max(1.0 / self.permeability) / jnp.min(self.permittivity_infinity)
        )


def drude_maxwell_constitutive(
    plasma_frequency: ArrayLike,
    damping: ArrayLike,
    /,
    *,
    permittivity_infinity: ArrayLike = 1.0,
    permeability: ArrayLike = 1.0,
) -> LorentzDrudeMaxwellConstitutivePlan:
    strength = jnp.asarray(plasma_frequency, dtype=float) ** 2
    return LorentzDrudeMaxwellConstitutivePlan(
        jnp.zeros_like(strength),
        damping,
        strength,
        permittivity_infinity=permittivity_infinity,
        permeability=permeability,
    )


__all__ = [
    "ConductiveMaxwellConstitutivePlan",
    "DispersiveMaxwellState",
    "LorentzDrudeMaxwellConstitutivePlan",
    "MatrixMaxwellConstitutivePlan",
    "MaxwellConstitutiveEvidence",
    "PreparedConductiveMaxwellConstitutive",
    "PreparedLorentzDrudeMaxwellConstitutive",
    "PreparedMatrixMaxwellConstitutive",
    "drude_maxwell_constitutive",
]
