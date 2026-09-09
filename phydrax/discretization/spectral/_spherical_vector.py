#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._spherical import SphericalSpectralDiscretization
from ._spherical_operators import PreparedSphericalSpinOperator, SphericalSpinOperatorPlan


class PreparedSphericalVectorOperators(StrictModule, NonTrainableState):
    """Intrinsic real tangent calculus using a single spin-one transform.

    East is ``e_phi``, north is ``-e_theta``, and positive curl points radially
    outward. The spin-one field is ``north - 1j * east``; thus ethbar gives
    ``divergence + 1j * curl``. Components at a pole use the longitude-labelled
    limiting tangent frame, not a globally continuous east/north chart.

    Modal and sampled arrays use the underlying transform's trailing spatial
    axes, with optional leading batch axes and one optional last channel axis.
    Real scalar coefficients must contain both conjugate orders (including a
    real m=0 column). Invalid padded modes are inert, as in the scalar space.

    ``mean_policy`` controls only the incompatible constant vorticity/divergence
    in :meth:`wind`: reject (also under JIT) or explicitly project it away.
    Constants in a scalar gradient are always valid and have zero gradient.
    """

    space: SphericalSpectralDiscretization
    raise_operator: PreparedSphericalSpinOperator
    inverse_laplacian: Array
    mean_policy: Literal["reject", "project"] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: SphericalSpectralDiscretization,
        /,
        *,
        mean_policy: Literal["reject", "project"] = "reject",
    ):
        if not isinstance(space, SphericalSpectralDiscretization):
            raise TypeError("space must be a prepared spherical spectral discretization.")
        if space.layout.spin != 0 or not space.layout.reality:
            raise ValueError("Tangent vector operators require a real spin-zero space.")
        if space.layout.bandlimit < 2:
            raise ValueError("Tangent vector operators require bandlimit >= 2.")
        if mean_policy not in ("reject", "project"):
            raise ValueError("mean_policy must be 'reject' or 'project'.")
        self.space = space
        self.raise_operator = SphericalSpinOperatorPlan("raise").prepare(space)
        laplacian = space.laplacian_multiplier()
        active = space.layout.valid_mask & (space.layout.degrees > 0)
        self.inverse_laplacian = jnp.where(
            active, 1.0 / jnp.where(active, laplacian, 1.0), 0.0
        )
        self.mean_policy = mean_policy
        self.operator_id = canonical_fingerprint(
            {
                "kind": "prepared-spherical-vector-operators",
                "space": space.prepared_id,
                "spin_operator": self.raise_operator.prepared_id,
                "frame": "east-e_phi:north-minus-e_theta:curl-outward",
                "mean_policy": mean_policy,
            }
        )

    def _broadcast(self, multiplier: Array, coefficients: Array, /) -> Array:
        layout = self.space.layout
        _, _, channel_last = layout._coefficient_axes(coefficients)
        return layout._broadcast_mode_array(
            multiplier, coefficients, channel_last=channel_last
        )

    def _conjugate_scalar(self, coefficients: Array, /) -> Array:
        """Coefficients of the complex conjugate spin-zero field."""
        layout = self.space.layout
        _, order_axis, _ = layout._coefficient_axes(coefficients)
        shape = [1] * coefficients.ndim
        shape[order_axis] = layout.coefficient_shape[1]
        return jnp.conj(
            jnp.take(coefficients, layout.conjugate_indices, axis=order_axis)
        ) * layout.conjugate_signs.reshape(tuple(shape))

    def _tolerance(self, coefficients: Array, /) -> Array:
        return (
            256.0
            * jnp.finfo(coefficients.real.dtype).eps
            * jnp.maximum(1.0, jnp.max(jnp.abs(coefficients), initial=0.0))
        )

    def _scalar(self, coefficients: ArrayLike, /) -> Array:
        modal = self.space.layout.mask_invalid(
            self.space.plan.precision.coefficients(coefficients)
        )
        conjugate = self._conjugate_scalar(modal)
        defect = jnp.max(jnp.abs(modal - conjugate), initial=0.0)
        modal = eqx.error_if(
            modal,
            ~jnp.all(jnp.isfinite(modal)) | (defect > self._tolerance(modal)),
            "Scalar coefficients must be finite and satisfy real-field conjugacy, "
            "including real m=0 coefficients.",
        )
        return modal

    def _analyze(self, east: ArrayLike, north: ArrayLike, /) -> Array:
        east_array, north_array = jnp.asarray(east), jnp.asarray(north)
        if east_array.shape != north_array.shape:
            raise ValueError("East and north components must have identical shapes.")
        if jnp.iscomplexobj(east_array) or jnp.iscomplexobj(north_array):
            raise TypeError("East and north tangent components must be real arrays.")
        field = north_array - 1j * east_array
        field = eqx.error_if(
            field,
            ~jnp.all(jnp.isfinite(field)),
            "East and north tangent components must be finite.",
        )
        return self.raise_operator.output_layout.mask_invalid(
            self.raise_operator.output_transform.analysis(field)
        )

    def _synthesize(self, spin_coefficients: Array, /) -> tuple[Array, Array]:
        field = self.raise_operator.output_transform.synthesis(spin_coefficients)
        return -jnp.imag(field), jnp.real(field)

    def gradient(self, scalar_coefficients: ArrayLike, /) -> tuple[Array, Array]:
        """Return physical east/north gradient, with one inverse-radius factor."""
        return self._synthesize(
            self.raise_operator.apply(self._scalar(scalar_coefficients))
        )

    def _differential(self, east: ArrayLike, north: ArrayLike, /) -> Array:
        spin_coefficients = self._analyze(east, north)
        # The spin-one lowering multiplier is minus the spin-zero raising one.
        return (
            -self._broadcast(self.raise_operator.multiplier, spin_coefficients)
            * spin_coefficients
        )

    def divergence(self, east: ArrayLike, north: ArrayLike, /) -> Array:
        """Return real-field scalar coefficients of the physical divergence."""
        differential = self._differential(east, north)
        return 0.5 * (differential + self._conjugate_scalar(differential))

    def curl(self, east: ArrayLike, north: ArrayLike, /) -> Array:
        """Return scalar coefficients of radially outward relative vorticity."""
        differential = self._differential(east, north)
        return -0.5j * (differential - self._conjugate_scalar(differential))

    def null_mode_defect(self, coefficients: ArrayLike, /) -> Array:
        """Maximum absolute ell=0 coefficient across batches/channels (JIT-safe)."""
        modal = self._scalar(coefficients)
        constant = self._broadcast(self.space.layout.degrees == 0, modal)
        return jnp.max(jnp.where(constant, jnp.abs(modal), 0.0), initial=0.0)

    def _potential(self, coefficients: ArrayLike, /) -> Array:
        modal = self._scalar(coefficients)
        if self.mean_policy == "reject":
            constant = self._broadcast(self.space.layout.degrees == 0, modal)
            defect = jnp.max(jnp.where(constant, jnp.abs(modal), 0.0), initial=0.0)
            modal = eqx.error_if(
                modal,
                defect > self._tolerance(modal),
                "Spherical wind requires zero-mean vorticity and divergence; "
                "use mean_policy='project' to remove incompatible constants.",
            )
        return modal * self._broadcast(self.inverse_laplacian, modal)

    def wind(
        self,
        vorticity_coefficients: ArrayLike,
        divergence_coefficients: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        """Helmholtz inversion: grad(Δ⁻¹ divergence) + k × grad(Δ⁻¹ vorticity).

        Potentials use zero constant gauge; source constants below 256 machine
        eps times max(1, coefficient magnitude) are treated as roundoff under
        ``reject``. :meth:`null_mode_defect` exposes the unprojected evidence.
        """
        streamfunction = self._potential(vorticity_coefficients)
        velocity_potential = self._potential(divergence_coefficients)
        if streamfunction.shape != velocity_potential.shape:
            raise ValueError("Vorticity and divergence must have identical modal shapes.")
        return self._synthesize(
            self.raise_operator.apply(velocity_potential + 1j * streamfunction)
        )


__all__ = ["PreparedSphericalVectorOperators"]
