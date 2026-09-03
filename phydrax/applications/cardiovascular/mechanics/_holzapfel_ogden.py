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
    material_invariants,
    MaterialFrameInput,
    resolve_material_frame,
)


class HolzapfelOgden2009Parameters(StrictModule, NonTrainableState):
    """Eight coefficients of the 2009 myocardium invariant convention.

    ``a``, ``a_f``, ``a_s``, and ``a_fs`` carry stress units (kPa in the
    cardiovascular kernel). Their paired ``b`` coefficients are dimensionless.
    """

    a: Array
    b: Array
    a_f: Array
    b_f: Array
    a_s: Array
    b_s: Array
    a_fs: Array
    b_fs: Array

    def __init__(
        self,
        a: ArrayLike,
        b: ArrayLike,
        a_f: ArrayLike,
        b_f: ArrayLike,
        a_s: ArrayLike,
        b_s: ArrayLike,
        a_fs: ArrayLike,
        b_fs: ArrayLike,
        /,
    ):
        values = tuple(
            jnp.asarray(value) for value in (a, b, a_f, b_f, a_s, b_s, a_fs, b_fs)
        )
        if any(
            value.shape != ()
            or jnp.issubdtype(value.dtype, jnp.complexfloating)
            or not bool(jnp.isfinite(value))
            for value in values
        ):
            raise ValueError("Holzapfel--Ogden coefficients must be finite real scalars.")
        amplitudes = values[0::2]
        exponents = values[1::2]
        if any(bool(value < 0.0) for value in amplitudes) or all(
            bool(value == 0.0) for value in amplitudes
        ):
            raise ValueError(
                "Holzapfel--Ogden stress amplitudes must be nonnegative with at "
                "least one positive amplitude."
            )
        if any(bool(value <= 0.0) for value in exponents):
            raise ValueError(
                "Holzapfel--Ogden exponential coefficients must be positive."
            )
        (
            self.a,
            self.b,
            self.a_f,
            self.b_f,
            self.a_s,
            self.b_s,
            self.a_fs,
            self.b_fs,
        ) = values


@dataclass(frozen=True, slots=True)
class _StaticHolzapfelOgden2009TensionOnlyEnergy:
    """Array-free callable embedded as static data in the generic mixed law."""

    a: float
    b: float
    a_f: float
    b_f: float
    a_s: float
    b_s: float
    a_fs: float
    b_fs: float
    material_frame: tuple[tuple[float, float, float], ...]

    def __call__(self, deformation_gradient: ArrayLike, /) -> Array:
        deformation = jnp.asarray(deformation_gradient)
        frame = jnp.asarray(self.material_frame, dtype=deformation.dtype)
        first, fiber, sheet, fiber_sheet = material_invariants(deformation, frame)
        fiber_extension = jnp.maximum(fiber - 1.0, 0.0)
        sheet_extension = jnp.maximum(sheet - 1.0, 0.0)
        return (
            self.a / (2.0 * self.b) * jnp.expm1(self.b * (first - 3.0))
            + self.a_f / (2.0 * self.b_f) * jnp.expm1(self.b_f * fiber_extension**2)
            + self.a_s / (2.0 * self.b_s) * jnp.expm1(self.b_s * sheet_extension**2)
            + self.a_fs / (2.0 * self.b_fs) * jnp.expm1(self.b_fs * fiber_sheet**2)
        )


class HolzapfelOgden2009TensionOnlyEnergy(StrictModule, NonTrainableState):
    r"""Precisely the 2009 eight-parameter, tension-only f/s convention.

    The isochoric invariants are ``I1``, ``I4f``, ``I4s``, and ``I8fs`` and

    ``W = a/(2b) expm1(b(I1-3))``
    ``  + af/(2bf) expm1(bf <I4f-1>_+^2)``
    ``  + as/(2bs) expm1(bs <I4s-1>_+^2)``
    ``  + afs/(2bfs) expm1(bfs I8fs^2)``.

    The fiber and sheet families are tension-only through the positive-part
    brackets; the fiber-sheet coupling is signed before squaring. This explicit
    class name prevents confusion with variants that omit tension gating, use
    engineering shear, or add a separate sheet-normal family.
    """

    parameters: HolzapfelOgden2009Parameters
    material_frame: Array
    frame_cell_index: int | None = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    energy_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameters: HolzapfelOgden2009Parameters,
        material_frame: MaterialFrameInput,
        /,
        *,
        frame_id: str | None = None,
        cell_index: int | None = None,
        frame_tolerance: float = 1.0e-8,
        energy_id: str | None = None,
    ):
        if not isinstance(parameters, HolzapfelOgden2009Parameters):
            raise TypeError("parameters must be HolzapfelOgden2009Parameters.")
        frame, identifier, selected_cell = resolve_material_frame(
            material_frame,
            frame_id=frame_id,
            cell_index=cell_index,
            tolerance=frame_tolerance,
        )
        parameters_payload = {
            name: float(value).hex()
            for name, value in (
                ("a", parameters.a),
                ("b", parameters.b),
                ("a_f", parameters.a_f),
                ("b_f", parameters.b_f),
                ("a_s", parameters.a_s),
                ("b_s", parameters.b_s),
                ("a_fs", parameters.a_fs),
                ("b_fs", parameters.b_fs),
            )
        }
        generated = canonical_fingerprint(
            {
                "kind": "holzapfel-ogden-2009-tension-only-fiber-sheet-energy",
                "parameters": parameters_payload,
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
        first, fiber, sheet, fiber_sheet = material_invariants(
            deformation_gradient,
            self.material_frame,
        )
        parameters = self.parameters
        fiber_extension = jnp.maximum(fiber - 1.0, 0.0)
        sheet_extension = jnp.maximum(sheet - 1.0, 0.0)
        isotropic = (
            parameters.a / (2.0 * parameters.b) * jnp.expm1(parameters.b * (first - 3.0))
        )
        fiber_energy = (
            parameters.a_f
            / (2.0 * parameters.b_f)
            * jnp.expm1(parameters.b_f * fiber_extension**2)
        )
        sheet_energy = (
            parameters.a_s
            / (2.0 * parameters.b_s)
            * jnp.expm1(parameters.b_s * sheet_extension**2)
        )
        shear_energy = (
            parameters.a_fs
            / (2.0 * parameters.b_fs)
            * jnp.expm1(parameters.b_fs * fiber_sheet**2)
        )
        return isotropic + fiber_energy + sheet_energy + shear_energy

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
        """Create the exact mixed u-p route; FE qualification occurs at prepare."""
        parameters = self.parameters
        static_energy = _StaticHolzapfelOgden2009TensionOnlyEnergy(
            float(parameters.a),
            float(parameters.b),
            float(parameters.a_f),
            float(parameters.b_f),
            float(parameters.a_s),
            float(parameters.b_s),
            float(parameters.a_fs),
            float(parameters.b_fs),
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


def holzapfel_ogden_2009_tension_only_reference_energy(
    deformation_gradient: ArrayLike,
    parameters: HolzapfelOgden2009Parameters,
    material_frame: ArrayLike,
    /,
) -> Array:
    """Evaluate the named 2009 convention without constructing a route."""
    if not isinstance(parameters, HolzapfelOgden2009Parameters):
        raise TypeError("parameters must be HolzapfelOgden2009Parameters.")
    first, fiber, sheet, fiber_sheet = material_invariants(
        deformation_gradient,
        material_frame,
    )
    fiber_extension = jnp.maximum(fiber - 1.0, 0.0)
    sheet_extension = jnp.maximum(sheet - 1.0, 0.0)
    return (
        parameters.a / (2.0 * parameters.b) * jnp.expm1(parameters.b * (first - 3.0))
        + parameters.a_f
        / (2.0 * parameters.b_f)
        * jnp.expm1(parameters.b_f * fiber_extension**2)
        + parameters.a_s
        / (2.0 * parameters.b_s)
        * jnp.expm1(parameters.b_s * sheet_extension**2)
        + parameters.a_fs
        / (2.0 * parameters.b_fs)
        * jnp.expm1(parameters.b_fs * fiber_sheet**2)
    )


__all__ = [
    "HolzapfelOgden2009Parameters",
    "HolzapfelOgden2009TensionOnlyEnergy",
    "holzapfel_ogden_2009_tension_only_reference_energy",
]
