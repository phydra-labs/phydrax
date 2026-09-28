#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..discretization import CochainDiscretization
from ..ein import contract
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
    SmallLinearSolvePlan,
    solve as solve_linear,
    solve_small_linear,
)
from ..sparse import SparseLinearMap
from ..typing import ConvertibleToArray
from ._maxwell import (
    _apply_hodge_metric,
    _positive_angular_frequency,
    AbstractMaxwellConstitutivePlan,
    AbstractMaxwellFrequencyResponse,
    AbstractPreparedMaxwellConstitutive,
    DiagonalMaxwellFrequencyResponse,
    InstantaneousMaxwellFrequencyResponse,
    MaxwellCapabilities,
    MaxwellCochainLayout,
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
    ) -> None:
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
            electric = electric.astype("float64")
        if not jnp.issubdtype(magnetic.dtype, jnp.inexact):
            magnetic = magnetic.astype("float64")
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
        layout: MaxwellCochainLayout,
        /,
    ) -> PreparedMatrixMaxwellConstitutive:
        return PreparedMatrixMaxwellConstitutive(self, cochain, layout)


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
    metric_tolerance = float(
        np.finfo(metric_matrix.real.dtype).eps
        * max(
            1.0,
            float(jnp.max(jnp.abs(metric_matrix))),
        )
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
    tolerance = float(
        np.finfo(weighted.real.dtype).eps
        * max(
            1.0,
            float(jnp.max(jnp.abs(weighted))),
        )
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
    layout_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MatrixMaxwellConstitutivePlan,
        cochain: CochainDiscretization,
        layout: MaxwellCochainLayout,
        /,
    ) -> None:
        _, electric_minimum, electric_condition = _metric_spectrum(
            "electric",
            plan.electric_matrix,
            cochain.hodge_metric(layout.electric_degree),
        )
        _, magnetic_minimum, magnetic_condition = _metric_spectrum(
            "magnetic",
            plan.magnetic_matrix,
            cochain.hodge_metric(layout.magnetic_degree),
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
        self.layout_id = layout.layout_id
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
                "layout": layout.layout_id,
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

    @property
    def auxiliary_degrees(self, /) -> tuple[int, ...]:
        return ()

    def frequency_response(
        self, angular_frequency: ArrayLike, /
    ) -> InstantaneousMaxwellFrequencyResponse:
        return InstantaneousMaxwellFrequencyResponse(self, angular_frequency)


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
    ) -> None:
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
        layout: MaxwellCochainLayout,
        /,
    ) -> PreparedConductiveMaxwellConstitutive:
        return PreparedConductiveMaxwellConstitutive(self, cochain, layout)


def _nonnegative_material(name: str, value: ArrayLike, count: int, /) -> Array:
    array = jnp.asarray(value)
    if jnp.iscomplexobj(array):
        raise TypeError(f"{name} must be real.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype("float64")
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
    layout_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ConductiveMaxwellConstitutivePlan,
        cochain: CochainDiscretization,
        layout: MaxwellCochainLayout,
        /,
    ) -> None:
        from ._maxwell import _positive_material

        self.permittivity = _positive_material(
            "permittivity", plan.permittivity, layout.electric_count
        )
        self.permeability = _positive_material(
            "permeability", plan.permeability, layout.magnetic_count
        )
        self.electric_conductivity = _nonnegative_material(
            "electric_conductivity",
            plan.electric_conductivity,
            layout.electric_count,
        )
        self.magnetic_conductivity = _nonnegative_material(
            "magnetic_conductivity",
            plan.magnetic_conductivity,
            layout.magnetic_count,
        )
        self.layout_id = layout.layout_id
        lossless = bool(
            jnp.all(self.electric_conductivity == 0.0)
            & jnp.all(self.magnetic_conductivity == 0.0)
        )
        zero_magnetic_conductivity = bool(jnp.all(self.magnetic_conductivity == 0.0))
        self.capabilities = MaxwellCapabilities(
            lossless=lossless,
            passive=True,
            reversible=lossless,
            structured_only=False,
            frequency_domain=True,
            magnetic_closedness_preserving=zero_magnetic_conductivity,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-conductive-maxwell-constitutive",
                "plan": plan.plan_id,
                "cochain": cochain.prepared_id,
                "layout": layout.layout_id,
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

    @property
    def auxiliary_degrees(self, /) -> tuple[int, ...]:
        return ()

    def frequency_response(
        self, angular_frequency: ArrayLike, /
    ) -> DiagonalMaxwellFrequencyResponse:
        omega = _positive_angular_frequency(angular_frequency)
        return DiagonalMaxwellFrequencyResponse(
            omega,
            self.permittivity + 1j * self.electric_conductivity / omega,
            self.permeability + 1j * self.magnetic_conductivity / omega,
            lossless=self.capabilities.lossless,
            dispersive=not self.capabilities.lossless,
        )


class MaxwellLorentzPoles(StrictModule):
    """Passive Lorentz/Drude poles ``Ẍ + γẊ + ω₀²X = f F`` on one cochain degree.

    ``strength`` has shape ``(poles,)`` or spatial ``(poles, entities)``. A zero
    strength removes the pole from that cochain entity: its auxiliary state stays
    exactly zero and carries neither energy nor dissipation.
    """

    resonance_frequency: Array
    damping: Array
    strength: Array
    poles_id: str = eqx.field(static=True)

    def __init__(
        self,
        resonance_frequency: ConvertibleToArray,
        damping: ConvertibleToArray,
        strength: ConvertibleToArray,
        /,
    ) -> None:
        frequency = jnp.asarray(resonance_frequency, dtype=jnp.float64)
        damping_ = jnp.asarray(damping, dtype=jnp.float64)
        strength_ = jnp.asarray(strength, dtype=jnp.float64)
        if frequency.ndim != 1 or frequency.size == 0:
            raise ValueError(
                "Lorentz poles require a nonempty resonance_frequency vector."
            )
        if damping_.shape != frequency.shape:
            raise ValueError("Lorentz pole damping must match resonance_frequency.")
        if strength_.ndim not in (1, 2) or strength_.shape[0] != frequency.size:
            raise ValueError(
                "Lorentz pole strength must have shape (poles,) or (poles, entities)."
            )
        invalid = (
            jnp.any(~jnp.isfinite(frequency))
            | jnp.any(~jnp.isfinite(damping_))
            | jnp.any(~jnp.isfinite(strength_))
            | jnp.any(frequency < 0.0)
            | jnp.any(damping_ < 0.0)
            | jnp.any(strength_ < 0.0)
        )
        self.resonance_frequency = eqx.error_if(
            frequency,
            invalid,
            "Passive Lorentz/Drude poles require finite nonnegative frequency, damping, and strength.",
        )
        self.damping = damping_
        self.strength = strength_
        self.poles_id = canonical_fingerprint(
            {
                "kind": "maxwell-lorentz-poles",
                "frequency": array_tree_fingerprint(frequency),
                "damping": array_tree_fingerprint(damping_),
                "strength": array_tree_fingerprint(strength_),
            }
        )


class DispersiveMaxwellState(StrictModule):
    """Electric polarization and magnetization oscillator state ``(poles, entities)``."""

    polarization: Array
    velocity: Array
    magnetization: Array
    magnetization_velocity: Array


class LorentzDrudeMaxwellConstitutivePlan(AbstractMaxwellConstitutivePlan):
    """Passive Lorentz/Drude electric and magnetic auxiliary differential equations.

    ``D = ε∞E + ΣP`` with ``P̈ + γṖ + ω₀²P = f E`` on electric cochains and
    ``B = μ∞H + ΣM`` with ``M̈ + γṀ + ω₀²M = f H`` on magnetic cochains.
    """

    electric_poles: MaxwellLorentzPoles | None
    magnetic_poles: MaxwellLorentzPoles | None
    permittivity_infinity: Array
    permeability_infinity: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        electric_poles: MaxwellLorentzPoles | None = None,
        /,
        *,
        magnetic_poles: MaxwellLorentzPoles | None = None,
        permittivity_infinity: ArrayLike = 1.0,
        permeability_infinity: ArrayLike = 1.0,
    ) -> None:
        for name, poles in (
            ("electric_poles", electric_poles),
            ("magnetic_poles", magnetic_poles),
        ):
            if poles is not None and not isinstance(poles, MaxwellLorentzPoles):
                raise TypeError(f"{name} must be MaxwellLorentzPoles or None.")
        if electric_poles is None and magnetic_poles is None:
            raise ValueError(
                "Lorentz/Drude material requires electric or magnetic poles."
            )
        self.electric_poles = electric_poles
        self.magnetic_poles = magnetic_poles
        self.permittivity_infinity = jnp.asarray(permittivity_infinity)
        self.permeability_infinity = jnp.asarray(permeability_infinity)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lorentz-drude-maxwell-plan",
                "electric_poles": None
                if electric_poles is None
                else electric_poles.poles_id,
                "magnetic_poles": None
                if magnetic_poles is None
                else magnetic_poles.poles_id,
                "permittivity_infinity": array_tree_fingerprint(
                    self.permittivity_infinity
                ),
                "permeability_infinity": array_tree_fingerprint(
                    self.permeability_infinity
                ),
            }
        )

    def prepare(
        self,
        cochain: CochainDiscretization,
        layout: MaxwellCochainLayout,
        /,
    ) -> PreparedLorentzDrudeMaxwellConstitutive:
        return PreparedLorentzDrudeMaxwellConstitutive(self, cochain, layout)


def _pole_arrays(
    poles: MaxwellLorentzPoles | None, count: int, name: str, /
) -> tuple[Array, Array, Array]:
    if poles is None:
        empty = jnp.zeros((0,), dtype=jnp.float64)
        return empty, empty, jnp.zeros((0, count), dtype=jnp.float64)
    strength = poles.strength
    if strength.ndim == 1:
        strength = jnp.broadcast_to(strength[:, None], (strength.size, count))
    elif strength.shape[1] != count:
        raise ValueError(f"{name} strength must have shape (poles,) or (poles, {count}).")
    return poles.resonance_frequency, poles.damping, strength


def _oscillator_half_step(
    position: Array,
    velocity: Array,
    field: Callable[[Array], Array],
    frequency: Array,
    damping: Array,
    strength: Array,
    step_size: Array,
    /,
) -> tuple[Array, Array]:
    """Symmetric kick-drift-kick with Crank-Nicolson damping in each half kick.

    ``field`` maps the pole positions to the driving field at the held
    primary flux, so both kicks see the self-consistent field. Zero-strength
    entities keep exactly zero state.
    """
    half = 0.5 * step_size
    gamma = 0.5 * half * damping[:, None]
    squared = frequency[:, None] ** 2

    def kick(current: Array, rate: Array) -> Array:
        force = strength * field(current)[None, :] - squared * current
        return ((1.0 - gamma) * rate + half * force) / (1.0 + gamma)

    velocity = kick(position, velocity)
    position = position + step_size * velocity
    return position, kick(position, velocity)


def _masked_inverse_strength(strength: Array, /) -> Array:
    supported = strength > 0.0
    return jnp.where(supported, 1.0 / jnp.where(supported, strength, 1.0), 0.0)


def _oscillator_energy_density(
    position: Array, velocity: Array, frequency: Array, strength: Array, /
) -> Array:
    return jnp.sum(
        (velocity**2 + frequency[:, None] ** 2 * position**2)
        * _masked_inverse_strength(strength),
        axis=0,
    )


def _oscillator_dissipation_density(
    velocity: Array, damping: Array, strength: Array, /
) -> Array:
    return jnp.sum(
        damping[:, None] * velocity**2 * _masked_inverse_strength(strength), axis=0
    )


def _pole_susceptibility(
    omega: Array, frequency: Array, damping: Array, strength: Array, /
) -> Array:
    denominator = frequency[:, None] ** 2 - omega**2 - 1j * omega * damping[:, None]
    return jnp.sum(strength / denominator, axis=0)


class PreparedLorentzDrudeMaxwellConstitutive(AbstractPreparedMaxwellConstitutive):
    permittivity_infinity: Array
    permeability_infinity: Array
    electric_frequency: Array
    electric_damping: Array
    electric_strength: Array
    magnetic_frequency: Array
    magnetic_damping: Array
    magnetic_strength: Array
    electric_degree: int = eqx.field(static=True)
    magnetic_degree: int = eqx.field(static=True)
    capabilities: MaxwellCapabilities
    layout_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: LorentzDrudeMaxwellConstitutivePlan,
        cochain: CochainDiscretization,
        layout: MaxwellCochainLayout,
        /,
    ) -> None:
        from ._maxwell import _positive_material

        self.permittivity_infinity = _positive_material(
            "permittivity_infinity",
            plan.permittivity_infinity,
            layout.electric_count,
        )
        self.permeability_infinity = _positive_material(
            "permeability_infinity",
            plan.permeability_infinity,
            layout.magnetic_count,
        )
        (
            self.electric_frequency,
            self.electric_damping,
            self.electric_strength,
        ) = _pole_arrays(plan.electric_poles, layout.electric_count, "electric_poles")
        (
            self.magnetic_frequency,
            self.magnetic_damping,
            self.magnetic_strength,
        ) = _pole_arrays(plan.magnetic_poles, layout.magnetic_count, "magnetic_poles")
        self.electric_degree = layout.electric_degree
        self.magnetic_degree = layout.magnetic_degree
        self.layout_id = layout.layout_id
        self.capabilities = MaxwellCapabilities(
            lossless=bool(
                jnp.all(self.electric_damping == 0.0)
                & jnp.all(self.magnetic_damping == 0.0)
            ),
            passive=True,
            dispersive=True,
            reversible=False,
            structured_only=False,
            frequency_domain=True,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-lorentz-drude-maxwell",
                "plan": plan.plan_id,
                "cochain": cochain.prepared_id,
                "layout": layout.layout_id,
            }
        )

    def initialize_state(self, /) -> DispersiveMaxwellState:
        electric = jnp.zeros(self.electric_strength.shape)
        magnetic = jnp.zeros(self.magnetic_strength.shape)
        return DispersiveMaxwellState(electric, electric, magnetic, magnetic)

    def validate_state(self, state: Any, /) -> None:
        if not isinstance(state, DispersiveMaxwellState):
            raise TypeError("Dispersive material requires DispersiveMaxwellState.")
        electric, magnetic = self.electric_strength.shape, self.magnetic_strength.shape
        if (
            state.polarization.shape != electric
            or state.velocity.shape != electric
            or state.magnetization.shape != magnetic
            or state.magnetization_velocity.shape != magnetic
        ):
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
        return (flux - jnp.sum(state.magnetization, axis=0)) / self.permeability_infinity

    def magnetic_flux(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.permeability_infinity * magnetic + jnp.sum(
            state.magnetization, axis=0
        )

    def electric_conduction(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return jnp.zeros_like(electric)

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
        del time, args
        self.validate_state(state)
        polarization, velocity = _oscillator_half_step(
            state.polarization,
            state.velocity,
            lambda value: (
                (displacement - jnp.sum(value, axis=0)) / self.permittivity_infinity
            ),
            self.electric_frequency,
            self.electric_damping,
            self.electric_strength,
            step_size,
        )
        magnetization, magnetization_velocity = _oscillator_half_step(
            state.magnetization,
            state.magnetization_velocity,
            lambda value: (
                (magnetic_flux - jnp.sum(value, axis=0)) / self.permeability_infinity
            ),
            self.magnetic_frequency,
            self.magnetic_damping,
            self.magnetic_strength,
            step_size,
        )
        return DispersiveMaxwellState(
            polarization, velocity, magnetization, magnetization_velocity
        )

    def dissipated_power(
        self,
        electric: Array,
        magnetic: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        del electric, magnetic
        self.validate_state(state)
        electric_density = _oscillator_dissipation_density(
            state.velocity, self.electric_damping, self.electric_strength
        )
        magnetic_density = _oscillator_dissipation_density(
            state.magnetization_velocity, self.magnetic_damping, self.magnetic_strength
        )
        return jnp.sum(_apply_hodge_metric(electric_star, electric_density)) + jnp.sum(
            _apply_hodge_metric(magnetic_star, magnetic_density)
        )

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
        electric_oscillator = _oscillator_energy_density(
            state.polarization,
            state.velocity,
            self.electric_frequency,
            self.electric_strength,
        )
        magnetic_oscillator = _oscillator_energy_density(
            state.magnetization,
            state.magnetization_velocity,
            self.magnetic_frequency,
            self.magnetic_strength,
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
                _apply_hodge_metric(magnetic_star, self.permeability_infinity * magnetic),
            )
            + jnp.sum(_apply_hodge_metric(electric_star, electric_oscillator))
            + jnp.sum(_apply_hodge_metric(magnetic_star, magnetic_oscillator))
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
        # d/dt of field plus oscillator energy is E·Ḋ + H·Ḃ minus pole damping.
        electric = self.electric_field(displacement, state)
        magnetic = self.magnetic_field(magnetic_flux, state)
        return jnp.real(
            jnp.vdot(electric, _apply_hodge_metric(electric_star, displacement_rate))
            + jnp.vdot(magnetic, _apply_hodge_metric(magnetic_star, magnetic_rate))
        ) - self.dissipated_power(electric, magnetic, state, electric_star, magnetic_star)

    def wave_speed_bound(self, /) -> Array:
        return jnp.sqrt(
            jnp.max(1.0 / self.permeability_infinity)
            / jnp.min(self.permittivity_infinity)
        )

    @property
    def auxiliary_degrees(self, /) -> tuple[int, ...]:
        return (
            self.electric_degree,
            self.electric_degree,
            self.magnetic_degree,
            self.magnetic_degree,
        )

    def continuum_relative_permittivity(self, angular_frequency: ArrayLike, /) -> Array:
        """``ε∞ + Σ f/(ω₀² − ω² − iγω)`` per electric cochain entity."""
        omega = _positive_angular_frequency(angular_frequency)
        return self.permittivity_infinity + _pole_susceptibility(
            omega, self.electric_frequency, self.electric_damping, self.electric_strength
        )

    def continuum_relative_permeability(self, angular_frequency: ArrayLike, /) -> Array:
        """``μ∞ + Σ f/(ω₀² − ω² − iγω)`` per magnetic cochain entity."""
        omega = _positive_angular_frequency(angular_frequency)
        return self.permeability_infinity + _pole_susceptibility(
            omega, self.magnetic_frequency, self.magnetic_damping, self.magnetic_strength
        )

    def frequency_response(
        self, angular_frequency: ArrayLike, /
    ) -> DiagonalMaxwellFrequencyResponse:
        omega = _positive_angular_frequency(angular_frequency)
        return DiagonalMaxwellFrequencyResponse(
            omega,
            self.continuum_relative_permittivity(omega),
            self.continuum_relative_permeability(omega),
            lossless=self.capabilities.lossless,
            dispersive=True,
        )


def drude_maxwell_constitutive(
    plasma_frequency: ArrayLike,
    damping: ArrayLike,
    /,
    *,
    permittivity_infinity: ArrayLike = 1.0,
    permeability_infinity: ArrayLike = 1.0,
) -> LorentzDrudeMaxwellConstitutivePlan:
    """Drude poles ``ω₀ = 0`` with strength ``ωₚ²``.

    ``plasma_frequency`` has shape ``(poles,)`` or spatial ``(poles, electric
    entities)``; zero entries localize the plasma to its support.
    """
    frequency = jnp.asarray(plasma_frequency, dtype=jnp.float64)
    if frequency.ndim not in (1, 2):
        raise ValueError(
            "plasma_frequency must have shape (poles,) or (poles, entities)."
        )
    return LorentzDrudeMaxwellConstitutivePlan(
        MaxwellLorentzPoles(
            jnp.zeros((frequency.shape[0],), dtype=jnp.float64),
            damping,
            frequency**2,
        ),
        permittivity_infinity=permittivity_infinity,
        permeability_infinity=permeability_infinity,
    )


class _VertexEdgeCoupling(StrictModule):
    """Hodge-adjoint pair between vertex Cartesian vectors and edge circulations.

    ``edge_current`` integrates the endpoint-averaged vertex vector along each
    oriented edge; ``vertex_field`` is its adjoint for the electric Hodge pairing
    divided by the vertex dual volume, so ``⟨E, ⋆J_edge⟩ = Σ_v V_v E_v·J_v``.
    """

    components: tuple[SparseLinearMap, SparseLinearMap, SparseLinearMap]
    electric_star: Array
    vertex_volume: Array

    def __init__(self, cochain: CochainDiscretization, /) -> None:
        incidence = cochain.topology.incidences[0]
        vertices, edges = cochain.coordinates[0], cochain.coordinates[1]
        if vertices is None or edges is None:
            raise ValueError("Vertex-edge plasma coupling requires cochain coordinates.")
        vertex_points = np.asarray(vertices)
        edge_points = np.asarray(edges)
        if vertex_points.shape[1] != 3:
            raise ValueError("Vertex-edge plasma coupling requires three dimensions.")
        electric_star = cochain.hodge_metric(1)
        vertex_volume = cochain.hodge_metric(0)
        if electric_star.ndim != 1 or vertex_volume.ndim != 1:
            raise ValueError(
                "Magnetized plasma coupling requires diagonal Hodge metrics."
            )
        relation = incidence.relation
        source = np.asarray(relation.source_indices)
        target = np.asarray(relation.target_indices)
        valid = np.asarray(relation.valid, dtype=np.bool_)
        signs = np.asarray(incidence.signs)
        tails = valid & (signs < 0.0)
        if not np.array_equal(
            np.bincount(target[tails], minlength=edge_points.shape[0]),
            np.ones((edge_points.shape[0],), dtype=np.int64),
        ):
            raise ValueError("Every oriented edge must have exactly one tail vertex.")
        # Twice the tail-to-midpoint vector is the edge tangent, also across
        # periodic seams where the head vertex coordinate wraps.
        tangent = np.zeros(edge_points.shape, dtype=np.float64)
        tangent[target[tails]] = 2.0 * (
            edge_points[target[tails]] - vertex_points[source[tails]]
        )
        maps = tuple(
            SparseLinearMap(
                relation,
                jnp.asarray(np.where(valid, 0.5 * tangent[target, axis], 0.0)),
                operator_id=canonical_fingerprint(
                    {
                        "kind": "maxwell-vertex-edge-average",
                        "cochain": cochain.prepared_id,
                        "axis": axis,
                    }
                ),
            )
            for axis in range(3)
        )
        self.components = (maps[0], maps[1], maps[2])
        self.electric_star = electric_star
        self.vertex_volume = vertex_volume

    def vertex_field(self, electric: Array, /) -> Array:
        paired = self.electric_star * electric
        return (
            jnp.stack(
                tuple(component.transpose_mv(paired) for component in self.components),
                axis=0,
            )
            / self.vertex_volume[None, :]
        )

    def edge_current(self, current: Array, /) -> Array:
        return sum(
            (
                component.mv(current[axis])
                for axis, component in enumerate(self.components)
            ),
            start=jnp.zeros(self.electric_star.shape, dtype=current.dtype),
        )


def _phi1(value: Array, /) -> Array:
    """``(exp(z) − 1)/z`` with a series branch near zero (real or complex)."""
    small = jnp.abs(value) < 1e-3
    safe = jnp.where(small, jnp.ones_like(value), value)
    series = 1.0 + value / 2.0 + value * value / 6.0 + value * value * value / 24.0
    return jnp.where(small, series, jnp.expm1(safe) / safe)


def _axis_cross(axis: Array, vector: Array, /) -> Array:
    """``n̂ × u`` for species axes ``(S, 3)`` and species vectors ``(S, 3, V)``."""
    nx, ny, nz = axis[:, 0, None], axis[:, 1, None], axis[:, 2, None]
    ux, uy, uz = vector[:, 0], vector[:, 1], vector[:, 2]
    return jnp.stack((ny * uz - nz * uy, nz * ux - nx * uz, nx * uy - ny * ux), axis=1)


class MagnetizedColdPlasmaState(StrictModule):
    """Species current densities at vertices, shape ``(species, 3, vertices)``."""

    current: Array


class MagnetizedColdPlasmaMaxwellConstitutivePlan(AbstractMaxwellConstitutivePlan):
    """Cold multi-species magnetized plasma current ADE.

    Each species obeys ``J̇ = ε₀ωₚ²E + J × ω_c − νJ`` with the signed cyclotron
    vector ``ω_c = qB₀/m``. Currents live at vertices as Cartesian vectors so the
    gyration is an exact rotation; each half step integrates the linear ODE
    exactly for the field held at the half-step endpoint (exponential rotation,
    not Cayley), which makes the coupling symmetric about the step midpoint.
    Vertex fields and edge currents use a Hodge-adjoint averaging pair, so the
    plasma energy ``|J|²/(2ε₀ωₚ²)`` exchanges exactly with the field energy.
    ``plasma_frequency`` is ``(species,)`` or spatial ``(species, vertices)``.
    """

    plasma_frequency: Array
    cyclotron_frequency: Array
    collision_frequency: Array
    permittivity_infinity: Array
    permeability: Array
    vacuum_permittivity: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        plasma_frequency: ArrayLike,
        cyclotron_frequency: ArrayLike,
        /,
        *,
        collision_frequency: ArrayLike = 0.0,
        permittivity_infinity: ArrayLike = 1.0,
        permeability: ArrayLike = 1.0,
        vacuum_permittivity: float = 1.0,
    ) -> None:
        plasma = jnp.asarray(plasma_frequency, dtype=jnp.float64)
        if plasma.ndim not in (1, 2) or plasma.shape[0] == 0:
            raise ValueError(
                "plasma_frequency must have shape (species,) or (species, vertices)."
            )
        species = plasma.shape[0]
        cyclotron = jnp.asarray(cyclotron_frequency, dtype=jnp.float64)
        if cyclotron.shape != (species, 3):
            raise ValueError("cyclotron_frequency must have shape (species, 3).")
        collision = jnp.broadcast_to(
            jnp.asarray(collision_frequency, dtype=jnp.float64), (species,)
        )
        permittivity = float(vacuum_permittivity)
        if not np.isfinite(permittivity) or permittivity <= 0.0:
            raise ValueError("vacuum_permittivity must be finite and positive.")
        invalid = (
            jnp.any(~jnp.isfinite(plasma))
            | jnp.any(plasma < 0.0)
            | jnp.any(~jnp.isfinite(cyclotron))
            | jnp.any(~jnp.isfinite(collision))
            | jnp.any(collision < 0.0)
        )
        self.plasma_frequency = eqx.error_if(
            plasma,
            invalid,
            "Cold plasma requires finite nonnegative plasma and collision frequencies "
            "and a finite cyclotron vector.",
        )
        self.cyclotron_frequency = cyclotron
        self.collision_frequency = collision
        self.permittivity_infinity = jnp.asarray(permittivity_infinity)
        self.permeability = jnp.asarray(permeability)
        self.vacuum_permittivity = permittivity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "magnetized-cold-plasma-maxwell-plan",
                "plasma_frequency": array_tree_fingerprint(plasma),
                "cyclotron_frequency": array_tree_fingerprint(cyclotron),
                "collision_frequency": array_tree_fingerprint(collision),
                "permittivity_infinity": array_tree_fingerprint(
                    self.permittivity_infinity
                ),
                "permeability": array_tree_fingerprint(self.permeability),
                "vacuum_permittivity": permittivity,
            }
        )

    def prepare(
        self,
        cochain: CochainDiscretization,
        layout: MaxwellCochainLayout,
        /,
    ) -> PreparedMagnetizedColdPlasmaMaxwellConstitutive:
        return PreparedMagnetizedColdPlasmaMaxwellConstitutive(self, cochain, layout)


def _gyration_axes(cyclotron: Array, /) -> tuple[Array, Array]:
    magnitude = jnp.sqrt(jnp.sum(cyclotron**2, axis=1))
    axis = cyclotron / jnp.where(magnitude > 0.0, magnitude, 1.0)[:, None]
    return magnitude, axis


class MagnetizedColdPlasmaFrequencyResponse(AbstractMaxwellFrequencyResponse):
    """``ε(ω)E = ε∞E + (i/ω) J_edge(σ(ω) E_vertex)`` for the cold plasma ADE."""

    angular_frequency: Array
    permittivity_infinity: Array
    permeability: Array
    conductivity: Array
    coupling: _VertexEdgeCoupling
    lossless: bool = eqx.field(static=True)
    dispersive: bool = eqx.field(static=True)

    def __init__(
        self,
        constitutive: PreparedMagnetizedColdPlasmaMaxwellConstitutive,
        angular_frequency: ArrayLike,
        /,
    ) -> None:
        omega = _positive_angular_frequency(angular_frequency)
        magnitude, axis = _gyration_axes(constitutive.cyclotron_frequency)
        zero = jnp.zeros_like(axis[:, 0])
        generator = jnp.stack(
            (
                jnp.stack((zero, -axis[:, 2], axis[:, 1]), axis=-1),
                jnp.stack((axis[:, 2], zero, -axis[:, 0]), axis=-1),
                jnp.stack((-axis[:, 1], axis[:, 0], zero), axis=-1),
            ),
            axis=-2,
        )
        # (ν − iω) J + Ω n̂×J = ε₀ωₚ² E for the exp(−iωt) phasor.
        species = axis.shape[0]
        identity = jnp.broadcast_to(jnp.eye(3, dtype=jnp.complex128), (species, 3, 3))
        system = (constitutive.collision_frequency - 1j * omega)[
            :, None, None
        ] * identity + magnitude[:, None, None] * generator
        solved = solve_small_linear(SmallLinearSolvePlan(3), system, identity)
        resolvent = eqx.error_if(
            solved.value,
            ~jnp.all(solved.successful),
            "Magnetized plasma response is singular at a collisionless cyclotron resonance.",
        )
        self.angular_frequency = omega
        self.permittivity_infinity = constitutive.permittivity_infinity
        self.permeability = constitutive.permeability
        self.conductivity = contract(
            "sv,sab->vab",
            constitutive.plasma_weight.astype(jnp.complex128),
            resolvent,
        )
        self.coupling = constitutive.coupling
        self.lossless = constitutive.capabilities.lossless
        self.dispersive = True

    def electric_displacement(self, electric: Array, /) -> Array:
        vertex = self.coupling.vertex_field(electric)
        current = contract("vab,bv->av", self.conductivity, vertex)
        return self.permittivity_infinity * electric + (
            1j / self.angular_frequency
        ) * self.coupling.edge_current(current)

    def magnetic_field(self, flux: Array, /) -> Array:
        return flux / self.permeability


class PreparedMagnetizedColdPlasmaMaxwellConstitutive(
    AbstractPreparedMaxwellConstitutive
):
    permittivity_infinity: Array
    permeability: Array
    plasma_weight: Array
    cyclotron_frequency: Array
    collision_frequency: Array
    coupling: _VertexEdgeCoupling
    capabilities: MaxwellCapabilities
    layout_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MagnetizedColdPlasmaMaxwellConstitutivePlan,
        cochain: CochainDiscretization,
        layout: MaxwellCochainLayout,
        /,
    ) -> None:
        from ._maxwell import _positive_material

        if layout.polarization != "full_3d":
            raise ValueError("Magnetized cold plasma requires the full_3d layout.")
        vertex_count = cochain.cell_counts[0]
        plasma = plan.plasma_frequency
        if plasma.ndim == 1:
            plasma = jnp.broadcast_to(plasma[:, None], (plasma.size, vertex_count))
        elif plasma.shape[1] != vertex_count:
            raise ValueError(
                f"plasma_frequency must have shape (species,) or (species, {vertex_count})."
            )
        self.permittivity_infinity = _positive_material(
            "permittivity_infinity", plan.permittivity_infinity, layout.electric_count
        )
        self.permeability = _positive_material(
            "permeability", plan.permeability, layout.magnetic_count
        )
        self.plasma_weight = plan.vacuum_permittivity * plasma**2
        self.cyclotron_frequency = plan.cyclotron_frequency
        self.collision_frequency = plan.collision_frequency
        self.coupling = _VertexEdgeCoupling(cochain)
        self.layout_id = layout.layout_id
        self.capabilities = MaxwellCapabilities(
            lossless=bool(jnp.all(plan.collision_frequency == 0.0)),
            passive=True,
            dispersive=True,
            reversible=False,
            structured_only=False,
            frequency_domain=True,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-magnetized-cold-plasma-maxwell",
                "plan": plan.plan_id,
                "cochain": cochain.prepared_id,
                "layout": layout.layout_id,
            }
        )

    def initialize_state(self, /) -> MagnetizedColdPlasmaState:
        species, vertices = self.plasma_weight.shape
        return MagnetizedColdPlasmaState(jnp.zeros((species, 3, vertices)))

    def validate_state(self, state: Any, /) -> None:
        if not isinstance(state, MagnetizedColdPlasmaState):
            raise TypeError("Magnetized plasma requires MagnetizedColdPlasmaState.")
        species, vertices = self.plasma_weight.shape
        if state.current.shape != (species, 3, vertices):
            raise ValueError("Magnetized plasma current has wrong shape.")

    def electric_field(self, displacement: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return displacement / self.permittivity_infinity

    def electric_displacement(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.permittivity_infinity * electric

    def magnetic_field(self, flux: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return flux / self.permeability

    def magnetic_flux(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.permeability * magnetic

    def electric_conduction(self, electric: Array, state: Any, /) -> Array:
        del electric
        self.validate_state(state)
        return self.coupling.edge_current(jnp.sum(state.current, axis=0))

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
    ) -> MagnetizedColdPlasmaState:
        del time, magnetic_flux, args
        electric = self.electric_field(displacement, state)
        drive = (
            self.plasma_weight[:, None, :] * self.coupling.vertex_field(electric)[None]
        )
        magnitude, axis = _gyration_axes(self.cyclotron_frequency)
        collision = self.collision_frequency
        current = state.current
        once = _axis_cross(axis, current)
        twice = _axis_cross(axis, once)
        angle = magnitude * step_size
        decay = jnp.exp(-collision * step_size)[:, None, None]
        rotated = decay * (
            current
            - jnp.sin(angle)[:, None, None] * once
            + (1.0 - jnp.cos(angle))[:, None, None] * twice
        )
        # ∫₀ʰ exp(sA) ds for A = −ν − Ω n̂×: c₀ I − c₁ N + (c₀ − c₂) N².
        c0 = step_size * _phi1(-collision * step_size)
        rotating = step_size * _phi1((-collision + 1j * magnitude) * step_size)
        c1 = jnp.imag(rotating)[:, None, None]
        c2 = jnp.real(rotating)[:, None, None]
        c0 = c0[:, None, None]
        drive_once = _axis_cross(axis, drive)
        drive_twice = _axis_cross(axis, drive_once)
        forced = c0 * drive - c1 * drive_once + (c0 - c2) * drive_twice
        return MagnetizedColdPlasmaState(rotated + forced)

    def _inverse_weight(self, /) -> Array:
        return _masked_inverse_strength(self.plasma_weight)

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
        density = jnp.sum(
            self.collision_frequency[:, None]
            * jnp.sum(jnp.abs(state.current) ** 2, axis=1)
            * self._inverse_weight(),
            axis=0,
        )
        return jnp.sum(self.coupling.vertex_volume * density)

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
        plasma = jnp.sum(
            jnp.sum(jnp.abs(state.current) ** 2, axis=1) * self._inverse_weight(),
            axis=0,
        )
        return 0.5 * jnp.real(
            jnp.vdot(electric, _apply_hodge_metric(electric_star, displacement))
            + jnp.vdot(magnetic, _apply_hodge_metric(magnetic_star, magnetic_flux))
            + jnp.sum(self.coupling.vertex_volume * plasma)
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
        # d/dt |J|²/(2ε₀ωₚ²) = J·E_v − ν|J|²/(ε₀ωₚ²); the vertex/edge pair is
        # Hodge-adjoint, so Σ V_v J·E_v = ⟨E, ⋆J_edge⟩.
        electric = self.electric_field(displacement, state)
        magnetic = self.magnetic_field(magnetic_flux, state)
        exchange = self.electric_conduction(electric, state)
        return jnp.real(
            jnp.vdot(
                electric,
                _apply_hodge_metric(electric_star, displacement_rate + exchange),
            )
            + jnp.vdot(magnetic, _apply_hodge_metric(magnetic_star, magnetic_rate))
        ) - self.dissipated_power(electric, magnetic, state, electric_star, magnetic_star)

    def wave_speed_bound(self, /) -> Array:
        return jnp.sqrt(
            jnp.max(1.0 / self.permeability) / jnp.min(self.permittivity_infinity)
        )

    @property
    def auxiliary_degrees(self, /) -> tuple[int, ...]:
        return (0,)

    def frequency_response(
        self, angular_frequency: ArrayLike, /
    ) -> MagnetizedColdPlasmaFrequencyResponse:
        return MagnetizedColdPlasmaFrequencyResponse(self, angular_frequency)


__all__ = [
    "ConductiveMaxwellConstitutivePlan",
    "DispersiveMaxwellState",
    "LorentzDrudeMaxwellConstitutivePlan",
    "MagnetizedColdPlasmaFrequencyResponse",
    "MagnetizedColdPlasmaMaxwellConstitutivePlan",
    "MagnetizedColdPlasmaState",
    "MatrixMaxwellConstitutivePlan",
    "MaxwellConstitutiveEvidence",
    "MaxwellLorentzPoles",
    "PreparedConductiveMaxwellConstitutive",
    "PreparedLorentzDrudeMaxwellConstitutive",
    "PreparedMagnetizedColdPlasmaMaxwellConstitutive",
    "PreparedMatrixMaxwellConstitutive",
    "drude_maxwell_constitutive",
]
