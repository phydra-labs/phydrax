#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....operators.mechanics import VolumetricConstraint
from ._materials import (
    ExactIncompressibleCardiacMaterial,
    FiniteBulkCardiacMaterial,
    material_green_lagrange_strain,
    MaterialFrameInput,
    resolve_material_frame,
)


class Guccione1991Parameters(StrictModule, NonTrainableState):
    """Exact 1991 Guccione exponential orthotropic coefficients.

    ``energy_scale`` carries stress units (kPa in the cardiovascular kernel);
    ``fiber_exponent``, ``transverse_exponent``, and
    ``fiber_shear_exponent`` are dimensionless.
    """

    energy_scale: Array
    fiber_exponent: Array
    transverse_exponent: Array
    fiber_shear_exponent: Array

    def __init__(
        self,
        energy_scale: ArrayLike,
        fiber_exponent: ArrayLike,
        transverse_exponent: ArrayLike,
        fiber_shear_exponent: ArrayLike,
        /,
    ):
        values = tuple(
            jnp.asarray(value)
            for value in (
                energy_scale,
                fiber_exponent,
                transverse_exponent,
                fiber_shear_exponent,
            )
        )
        if any(
            value.shape != ()
            or jnp.issubdtype(value.dtype, jnp.complexfloating)
            or not bool(jnp.isfinite(value))
            or bool(value <= 0.0)
            for value in values
        ):
            raise ValueError(
                "Guccione coefficients must be positive finite real scalars."
            )
        (
            self.energy_scale,
            self.fiber_exponent,
            self.transverse_exponent,
            self.fiber_shear_exponent,
        ) = values


@dataclass(frozen=True, slots=True)
class _StaticGuccione1991Energy:
    """Array-free callable embedded as static data in the generic mixed law."""

    energy_scale: float
    fiber_exponent: float
    transverse_exponent: float
    fiber_shear_exponent: float
    material_frame: tuple[tuple[float, float, float], ...]

    def __call__(self, deformation_gradient: ArrayLike, /) -> Array:
        deformation = jnp.asarray(deformation_gradient)
        frame = jnp.asarray(self.material_frame, dtype=deformation.dtype)
        strain = material_green_lagrange_strain(deformation, frame)
        e_ff = strain[0, 0]
        e_ss = strain[1, 1]
        e_nn = strain[2, 2]
        e_fs = strain[0, 1]
        e_fn = strain[0, 2]
        e_sn = strain[1, 2]
        quadratic = (
            self.fiber_exponent * e_ff * e_ff
            + self.transverse_exponent * (e_ss * e_ss + e_nn * e_nn + 2.0 * e_sn * e_sn)
            + 2.0 * self.fiber_shear_exponent * (e_fs * e_fs + e_fn * e_fn)
        )
        return 0.5 * self.energy_scale * jnp.expm1(quadratic)


class Guccione1991Energy(StrictModule, NonTrainableState):
    r"""The Guccione et al. (1991) orthotropic myocardium energy.

    With material Green--Lagrange components ``E_ab`` in the reference
    fiber/sheet/sheet-normal frame, this implementation is exactly

    ``W = C/2 (exp(Q) - 1)``,

    ``Q = bf E_ff^2 + bt(E_ss^2 + E_nn^2 + 2 E_sn^2)``
    ``    + 2 bfs(E_fs^2 + E_fn^2)``.

    The explicit factors of two are part of this convention, not engineering
    shear substitutions. The law is objective because it depends on ``F.T F``
    and a fixed reference material frame.
    """

    parameters: Guccione1991Parameters
    material_frame: Array
    frame_cell_index: int | None = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    energy_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameters: Guccione1991Parameters,
        material_frame: MaterialFrameInput,
        /,
        *,
        frame_id: str | None = None,
        cell_index: int | None = None,
        frame_tolerance: float = 1.0e-8,
        energy_id: str | None = None,
    ):
        if not isinstance(parameters, Guccione1991Parameters):
            raise TypeError("parameters must be Guccione1991Parameters.")
        frame, identifier, selected_cell = resolve_material_frame(
            material_frame,
            frame_id=frame_id,
            cell_index=cell_index,
            tolerance=frame_tolerance,
        )
        generated = canonical_fingerprint(
            {
                "kind": "guccione-1991-exact-orthotropic-energy",
                "energy_scale": float(parameters.energy_scale).hex(),
                "fiber_exponent": float(parameters.fiber_exponent).hex(),
                "transverse_exponent": float(parameters.transverse_exponent).hex(),
                "fiber_shear_exponent": float(parameters.fiber_shear_exponent).hex(),
                "material_frame": array_tree_fingerprint(frame),
                "frame_id": identifier,
                "frame_cell_index": selected_cell,
            }
        )
        selected = generated if energy_id is None else str(energy_id)
        if not selected:
            raise ValueError("energy_id must be non-empty or None.")
        self.parameters = parameters
        self.material_frame = frame
        self.frame_cell_index = selected_cell
        self.frame_id = identifier
        self.energy_id = selected

    def __call__(self, deformation_gradient: ArrayLike, /) -> Array:
        strain = material_green_lagrange_strain(
            deformation_gradient,
            self.material_frame,
        )
        e_ff = strain[0, 0]
        e_ss = strain[1, 1]
        e_nn = strain[2, 2]
        e_fs = strain[0, 1]
        e_fn = strain[0, 2]
        e_sn = strain[1, 2]
        parameters = self.parameters
        quadratic = (
            parameters.fiber_exponent * e_ff * e_ff
            + parameters.transverse_exponent
            * (e_ss * e_ss + e_nn * e_nn + 2.0 * e_sn * e_sn)
            + 2.0 * parameters.fiber_shear_exponent * (e_fs * e_fs + e_fn * e_fn)
        )
        return 0.5 * parameters.energy_scale * jnp.expm1(quadratic)

    def finite_bulk(
        self,
        bulk_modulus: ArrayLike,
        /,
        *,
        volumetric_constraint: VolumetricConstraint | None = None,
        minimum_jacobian: float = 1.0e-8,
        material_id: str | None = None,
    ) -> FiniteBulkCardiacMaterial:
        """Create the distinct displacement-only finite-bulk route."""
        return FiniteBulkCardiacMaterial(
            self,
            bulk_modulus,
            energy_id=self.energy_id,
            volumetric_constraint=volumetric_constraint,
            minimum_jacobian=minimum_jacobian,
            material_id=material_id,
        )

    def exact_incompressible(
        self,
        /,
        *,
        volumetric_constraint: VolumetricConstraint | None = None,
        minimum_jacobian: float = 1.0e-8,
        material_id: str | None = None,
    ) -> ExactIncompressibleCardiacMaterial:
        """Create the distinct exact mixed u-p route; qualification occurs at prepare."""
        parameters = self.parameters
        static_energy = _StaticGuccione1991Energy(
            float(parameters.energy_scale),
            float(parameters.fiber_exponent),
            float(parameters.transverse_exponent),
            float(parameters.fiber_shear_exponent),
            tuple(
                tuple(float(component) for component in row)
                for row in self.material_frame.tolist()
            ),
        )
        return ExactIncompressibleCardiacMaterial(
            static_energy,
            energy_id=self.energy_id,
            volumetric_constraint=volumetric_constraint,
            minimum_jacobian=minimum_jacobian,
            material_id=material_id,
        )


def guccione_1991_reference_energy(
    deformation_gradient: ArrayLike,
    parameters: Guccione1991Parameters,
    material_frame: ArrayLike,
    /,
) -> Array:
    """Evaluate the exact Guccione convention without constructing a route."""
    if not isinstance(parameters, Guccione1991Parameters):
        raise TypeError("parameters must be Guccione1991Parameters.")
    strain = material_green_lagrange_strain(deformation_gradient, material_frame)
    e_ff, e_ss, e_nn = strain[0, 0], strain[1, 1], strain[2, 2]
    e_fs, e_fn, e_sn = strain[0, 1], strain[0, 2], strain[1, 2]
    quadratic = (
        parameters.fiber_exponent * e_ff**2
        + parameters.transverse_exponent * (e_ss**2 + e_nn**2 + 2.0 * e_sn**2)
        + 2.0 * parameters.fiber_shear_exponent * (e_fs**2 + e_fn**2)
    )
    return 0.5 * parameters.energy_scale * jnp.expm1(quadratic)


__all__ = [
    "Guccione1991Energy",
    "Guccione1991Parameters",
    "guccione_1991_reference_energy",
]
