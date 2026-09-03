#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.spectral._coordinates import HermitianSpectralCoordinates
from ..discretization.spectral._dealias import (
    PaddingDealiasingPlan,
    PreparedDealiasingPlan,
)
from ..discretization.spectral._space import TensorSpectralDiscretization
from ..linalg import ArraySpace, DiagonalLinearOperator
from ..solver._etdrk import ETDRKMethod, PreparedETDRKMethod
from ..solver._semilinear_drift import SemilinearDrift


DissipationOrder = Literal[1, 2, 3, 4]
BilinearSelector = Callable[[Callable[[Array, Array], Array], Array, ArrayLike], Array]


def _modal_geometry(
    discretization: TensorSpectralDiscretization, /
) -> tuple[tuple[Array, Array], Array, Array, Array, float]:
    if len(discretization.axes) != 2 or any(
        axis.family != "fourier" for axis in discretization.axes
    ):
        raise ValueError("The barotropic beta-plane requires two periodic Fourier axes.")
    coefficient_dtype = jnp.dtype(discretization.plan.precision.coefficient_dtype)
    if not jnp.issubdtype(coefficient_dtype, jnp.complexfloating):
        raise TypeError("The beta-plane requires full complex Fourier coefficients.")
    real_dtype = jnp.empty((), dtype=coefficient_dtype).real.dtype
    waves: list[Array] = []
    admissible = jnp.ones(discretization.modal_shape, dtype=bool)
    for axis_index, axis in enumerate(discretization.axes):
        values = (
            2.0
            * jnp.asarray(jnp.pi, dtype=real_dtype)
            * axis.modes.mode_numbers.astype(real_dtype)
            / axis.length.astype(real_dtype)
        )
        shape = [1, 1]
        shape[axis_index] = axis.mode_count
        waves.append(
            jnp.broadcast_to(values.reshape(tuple(shape)), discretization.modal_shape)
        )
        admissible = admissible & jnp.broadcast_to(
            (~axis.modes.nyquist_mask).reshape(tuple(shape)),
            discretization.modal_shape,
        )
    squared = waves[0] ** 2 + waves[1] ** 2
    safe = jnp.where(squared > 0.0, squared, jnp.ones_like(squared))
    inverse = jnp.where(squared > 0.0, 1.0 / safe, jnp.zeros_like(squared))
    admissible = admissible & (squared > 0.0)
    volume = float(np.prod([float(axis.length) for axis in discretization.axes]))
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("The periodic cell must have finite positive area.")
    return (waves[0], waves[1]), squared, inverse, admissible, volume


class BetaPlaneBudgets(StrictModule):
    kinetic_energy: Array
    enstrophy: Array
    energy_rate: Array
    enstrophy_rate: Array
    nonlinear_energy_rate: Array
    nonlinear_enstrophy_rate: Array
    reality_defect: Array
    finite: Array
    successful: Array
    problem_id: str = eqx.field(static=True)


class BarotropicBetaPlane(StrictModule, NonTrainableState):
    """Doubly-periodic barotropic vorticity dynamics on a beta plane.

    Vorticity and streamfunction obey ``zeta = Laplacian(psi)`` and velocity is
    ``(-psi_y, psi_x)``.  The modal linear rate is the exact Rossby-wave,
    drag, and hyperviscous rate.  Quadratic products use the prepared Phydrax
    spectral dealiasing path rather than an independent FFT convention.
    """

    discretization: TensorSpectralDiscretization
    dealiasing: PreparedDealiasingPlan
    coordinates: HermitianSpectralCoordinates
    wavenumbers: tuple[Array, Array]
    wavenumber_squared: Array
    inverse_wavenumber_squared: Array
    admissibility_mask: Array
    linear_diagonal: Array
    beta: float = eqx.field(static=True)
    linear_drag: float = eqx.field(static=True)
    viscosity: float = eqx.field(static=True)
    dissipation_order: int = eqx.field(static=True)
    volume: float = eqx.field(static=True)
    reality_tolerance: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        beta: float,
        linear_drag: float = 0.0,
        viscosity: float = 0.0,
        dissipation_order: DissipationOrder = 1,
        dealiasing: PreparedDealiasingPlan | None = None,
        reality_tolerance: float = 1.0e-10,
        maximum_coordinate_size: int = 10_000_000,
    ):
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        beta_ = float(beta)
        drag = float(linear_drag)
        viscosity_ = float(viscosity)
        order = int(dissipation_order)
        tolerance = float(reality_tolerance)
        if (
            not np.isfinite(beta_)
            or not np.isfinite(drag)
            or drag < 0.0
            or not np.isfinite(viscosity_)
            or viscosity_ < 0.0
            or order not in (1, 2, 3, 4)
            or not np.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("Beta-plane coefficients and tolerances are invalid.")
        waves, squared, inverse, admissible, volume = _modal_geometry(discretization)
        if dealiasing is None:
            prepared_dealiasing = PaddingDealiasingPlan(2).prepare(
                discretization,
                required_polynomial_degree=2,
            )
        else:
            prepared_dealiasing = dealiasing
        if not isinstance(prepared_dealiasing, PreparedDealiasingPlan):
            raise TypeError("dealiasing must be a PreparedDealiasingPlan or None.")
        if prepared_dealiasing.retained.prepared_id != discretization.prepared_id:
            raise ValueError("Dealiasing must retain the beta-plane discretization.")
        if not prepared_dealiasing.report.exact:
            raise ValueError("Beta-plane quadratic products require exact dealiasing.")
        coordinates = HermitianSpectralCoordinates(
            discretization,
            reality_tolerance=tolerance,
            maximum_coordinate_size=maximum_coordinate_size,
        )
        kx = waves[0]
        rossby = 1j * beta_ * kx * inverse
        decay = drag + viscosity_ * squared**order
        linear = (rossby - decay).astype(
            jnp.dtype(discretization.plan.precision.coefficient_dtype)
        )
        linear = jnp.where(admissible, linear, 0.0)
        identifier = canonical_fingerprint(
            {
                "kind": "barotropic-beta-plane",
                "discretization": discretization.prepared_id,
                "dealiasing": prepared_dealiasing.prepared_id,
                "beta": beta_,
                "linear_drag": drag,
                "viscosity": viscosity_,
                "dissipation_order": order,
                "reality_tolerance": tolerance,
                "zero_mode": "mean-vorticity-zero",
                "velocity": "minus-psi-y,psi-x",
            }
        )
        self.discretization = discretization
        self.dealiasing = prepared_dealiasing
        self.coordinates = coordinates
        self.wavenumbers = waves
        self.wavenumber_squared = squared
        self.inverse_wavenumber_squared = inverse
        self.admissibility_mask = admissible
        self.linear_diagonal = linear
        self.beta = beta_
        self.linear_drag = drag
        self.viscosity = viscosity_
        self.dissipation_order = order
        self.volume = volume
        self.reality_tolerance = tolerance
        self.problem_id = identifier

    @property
    def state_shape(self) -> tuple[int, int]:
        return self.discretization.modal_shape

    def validate_state(self, vorticity: ArrayLike, /) -> Array:
        value = jnp.asarray(vorticity)
        if value.shape != self.state_shape:
            raise ValueError(
                f"Vorticity must have modal shape {self.state_shape}; got {value.shape}."
            )
        if not jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("Modal vorticity must be complex-valued.")
        return value

    def project_state(self, vorticity: ArrayLike, /) -> Array:
        value = self.coordinates.project(self.validate_state(vorticity))
        return value * self.admissibility_mask

    def reality_defect(self, vorticity: ArrayLike, /) -> Array:
        return self.coordinates.reality_defect(self.validate_state(vorticity))

    def streamfunction(self, vorticity: ArrayLike, /) -> Array:
        value = self.validate_state(vorticity)
        return (
            -self.inverse_wavenumber_squared.astype(value.real.dtype)
            * value
            * self.admissibility_mask
        )

    def velocity(self, vorticity: ArrayLike, /, *, physical: bool = False) -> Array:
        psi = self.streamfunction(vorticity)
        kx, ky = self.wavenumbers
        modal = jnp.stack((-1j * ky * psi, 1j * kx * psi), axis=-1)
        modal = modal * self.admissibility_mask[..., None]
        if physical:
            return self.discretization.reconstruct(modal)
        return modal

    def bilinear_tendency(self, left: Array, right: Array, /) -> Array:
        """Return ``-J(psi(left), right)`` with exact quadratic dealiasing."""
        left_ = self.validate_state(left)
        right_ = self.validate_state(right)
        psi = self.streamfunction(left_)
        kx, ky = self.wavenumbers
        psi_x = self.dealiasing.reconstruct(1j * kx * psi)
        psi_y = self.dealiasing.reconstruct(1j * ky * psi)
        zeta_x = self.dealiasing.reconstruct(1j * kx * right_)
        zeta_y = self.dealiasing.reconstruct(1j * ky * right_)
        jacobian = psi_x * zeta_y - psi_y * zeta_x
        tendency = -self.dealiasing.project(jacobian)
        return self.coordinates.project(tendency) * self.admissibility_mask

    def nonlinear_tendency(
        self,
        vorticity: ArrayLike,
        /,
        *,
        selector: BilinearSelector | None = None,
        interaction_coordinate: ArrayLike = 0.0,
    ) -> Array:
        value = self.validate_state(vorticity)
        if selector is None:
            return self.bilinear_tendency(value, value)
        selected = jnp.asarray(
            selector(self.bilinear_tendency, value, interaction_coordinate)
        )
        if selected.shape != self.state_shape:
            raise ValueError("Interaction selector returned an incompatible state shape.")
        return self.coordinates.project(selected) * self.admissibility_mask

    def linear_tendency(self, vorticity: ArrayLike, /) -> Array:
        value = self.validate_state(vorticity)
        return self.linear_diagonal * value

    def tendency(
        self,
        time: ArrayLike,
        vorticity: ArrayLike,
        args: Any = None,
        /,
        *,
        selector: BilinearSelector | None = None,
        interaction_coordinate: ArrayLike = 0.0,
    ) -> Array:
        del time, args
        value = self.validate_state(vorticity)
        return self.linear_tendency(value) + self.nonlinear_tendency(
            value,
            selector=selector,
            interaction_coordinate=interaction_coordinate,
        )

    def __call__(
        self,
        time: ArrayLike,
        vorticity: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        return self.tendency(time, vorticity, args)

    @property
    def rossby_frequency(self) -> Array:
        return -self.beta * self.wavenumbers[0] * self.inverse_wavenumber_squared

    def kinetic_energy(self, vorticity: ArrayLike, /) -> Array:
        value = self.validate_state(vorticity)
        psi_values = self.discretization.reconstruct(self.streamfunction(value))
        zeta_values = self.discretization.reconstruct(value)
        return -0.5 * self.discretization.integral(psi_values * zeta_values) / self.volume

    def energy(self, vorticity: ArrayLike, /) -> Array:
        return self.kinetic_energy(vorticity)

    def enstrophy(self, vorticity: ArrayLike, /) -> Array:
        values = self.discretization.reconstruct(self.validate_state(vorticity))
        return 0.5 * self.discretization.integral(values**2) / self.volume

    def semilinear_drift(
        self,
        /,
        *,
        selector: BilinearSelector | None = None,
        interaction_id: str = "nl",
        interaction_coordinate: float = 0.0,
    ) -> SemilinearDrift:
        identifier = str(interaction_id)
        coordinate = float(interaction_coordinate)
        if (
            not identifier
            or not np.isfinite(coordinate)
            or coordinate < 0.0
            or coordinate > 1.0
        ):
            raise ValueError("Interaction identity and coordinate are invalid.")
        space = ArraySpace(
            self.state_shape,
            dtype=jnp.dtype(self.discretization.plan.precision.coefficient_dtype),
            space_id=f"beta-plane-state:{self.problem_id}",
        )
        operator_id = canonical_fingerprint(
            {"kind": "beta-plane-linear", "problem": self.problem_id}
        )
        operator = DiagonalLinearOperator(
            self.linear_diagonal.reshape((prod(self.state_shape),)),
            space=space,
            operator_id=operator_id,
        )

        def nonlinear(time: Array, state: Array, args: Any) -> Array:
            del time, args
            return self.nonlinear_tendency(
                state,
                selector=selector,
                interaction_coordinate=coordinate,
            )

        nonlinear_id = canonical_fingerprint(
            {
                "kind": "beta-plane-quadratic",
                "problem": self.problem_id,
                "interaction": identifier,
                "coordinate": coordinate,
            }
        )
        return SemilinearDrift(
            operator,
            nonlinear,
            state_shape=self.state_shape,
            operator_id=operator_id,
            nonlinear_id=nonlinear_id,
        )

    def prepare_etdrk(
        self,
        order: Literal[2, 4] = 4,
        /,
        *,
        selector: BilinearSelector | None = None,
        interaction_id: str = "nl",
        interaction_coordinate: float = 0.0,
    ) -> PreparedETDRKMethod:
        drift = self.semilinear_drift(
            selector=selector,
            interaction_id=interaction_id,
            interaction_coordinate=interaction_coordinate,
        )
        return ETDRKMethod(order).prepare(drift, coordinates=self.coordinates)

    def budgets(
        self,
        vorticity: ArrayLike,
        /,
        *,
        tendency: ArrayLike | None = None,
        nonlinear_tendency: ArrayLike | None = None,
    ) -> BetaPlaneBudgets:
        value = self.validate_state(vorticity)
        psi = self.streamfunction(value)
        zeta_values = self.discretization.reconstruct(value)
        psi_values = self.discretization.reconstruct(psi)
        rhs = (
            self.tendency(0.0, value)
            if tendency is None
            else self.validate_state(tendency)
        )
        nonlinear = (
            self.nonlinear_tendency(value)
            if nonlinear_tendency is None
            else self.validate_state(nonlinear_tendency)
        )
        rhs_values = self.discretization.reconstruct(rhs)
        nonlinear_values = self.discretization.reconstruct(nonlinear)
        inverse_rhs_values = self.discretization.reconstruct(
            -self.inverse_wavenumber_squared * rhs
        )
        inverse_nonlinear_values = self.discretization.reconstruct(
            -self.inverse_wavenumber_squared * nonlinear
        )
        energy = (
            -0.5 * self.discretization.integral(psi_values * zeta_values) / self.volume
        )
        enstrophy = 0.5 * self.discretization.integral(zeta_values**2) / self.volume
        energy_rate = (
            -0.5
            * self.discretization.integral(
                inverse_rhs_values * zeta_values + psi_values * rhs_values
            )
            / self.volume
        )
        enstrophy_rate = (
            self.discretization.integral(zeta_values * rhs_values) / self.volume
        )
        nonlinear_energy_rate = (
            -0.5
            * self.discretization.integral(
                inverse_nonlinear_values * zeta_values + psi_values * nonlinear_values
            )
            / self.volume
        )
        nonlinear_enstrophy_rate = (
            self.discretization.integral(zeta_values * nonlinear_values) / self.volume
        )
        values = jnp.stack(
            (
                energy,
                enstrophy,
                energy_rate,
                enstrophy_rate,
                nonlinear_energy_rate,
                nonlinear_enstrophy_rate,
            )
        )
        reality = self.reality_defect(value)
        finite = jnp.all(jnp.isfinite(values)) & jnp.isfinite(reality)
        successful = finite & (reality <= self.reality_tolerance)
        return BetaPlaneBudgets(
            kinetic_energy=energy,
            enstrophy=enstrophy,
            energy_rate=energy_rate,
            enstrophy_rate=enstrophy_rate,
            nonlinear_energy_rate=nonlinear_energy_rate,
            nonlinear_enstrophy_rate=nonlinear_enstrophy_rate,
            reality_defect=reality,
            finite=finite,
            successful=successful,
            problem_id=self.problem_id,
        )

    def modal_inner_product(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_ = self.validate_state(left)
        right_ = self.validate_state(right)
        return jnp.real(oe.contract("ij,ij->", jnp.conj(left_), right_))


__all__ = ["BarotropicBetaPlane", "BetaPlaneBudgets", "BilinearSelector"]
