#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from jaxtyping import ArrayLike

from ._complex_potential_2d import PlaneIsotropicMaterial
from ._holomorphic_constraints import (
    HolomorphicJetFunctionalTerm,
    HolomorphicPointFunctional,
)


def _complex_scalar(value: ArrayLike, name: str, /) -> complex:
    raw = np.asarray(value)
    if raw.shape != ():
        raise ValueError(f"{name} must be one complex scalar.")
    resolved = complex(raw)
    if not np.isfinite(abs(resolved)):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _normal_2d(value: ArrayLike, /) -> tuple[float, float]:
    raw = np.asarray(value)
    if raw.shape != (2,) or np.iscomplexobj(raw):
        raise TypeError("Plane boundary normal must be real shape (2,).")
    normal = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(normal)) or not np.any(normal != 0.0):
        raise ValueError("Plane boundary normal must be finite and nonzero.")
    return float(normal[0]), float(normal[1])


def _scaled_terms(
    terms: Sequence[HolomorphicJetFunctionalTerm],
    scale: float,
    /,
) -> tuple[HolomorphicJetFunctionalTerm, ...]:
    if scale == 0.0:
        return ()
    return tuple(
        HolomorphicJetFunctionalTerm(
            term.output_index,
            term.derivative_multi_index,
            scale * complex(np.asarray(term.weight)),
        )
        for term in terms
    )


def biharmonic_value_functional(
    coordinate: ArrayLike,
    /,
    *,
    phi_output: int = 0,
    psi_output: int = 1,
) -> HolomorphicPointFunctional:
    """Physical Goursat field value at one complex coordinate."""
    z = _complex_scalar(coordinate, "coordinate")
    return HolomorphicPointFunctional(
        z,
        (
            HolomorphicJetFunctionalTerm(phi_output, (0,), np.conj(z)),
            HolomorphicJetFunctionalTerm(psi_output, (0,), 1.0),
        ),
        construction="biharmonic-goursat-value-functional",
    )


def biharmonic_normal_derivative_functional(
    coordinate: ArrayLike,
    normal: ArrayLike,
    /,
    *,
    phi_output: int = 0,
    psi_output: int = 1,
) -> HolomorphicPointFunctional:
    """Physical Goursat supplied-normal derivative at one point."""
    z = _complex_scalar(coordinate, "coordinate")
    normal_x, normal_y = _normal_2d(normal)
    direction = complex(normal_x, normal_y)
    return HolomorphicPointFunctional(
        z,
        (
            HolomorphicJetFunctionalTerm(phi_output, (0,), np.conj(direction)),
            HolomorphicJetFunctionalTerm(
                phi_output,
                (1,),
                direction * np.conj(z),
            ),
            HolomorphicJetFunctionalTerm(psi_output, (1,), direction),
        ),
        construction="biharmonic-goursat-normal-functional",
    )


def biharmonic_robin_functional(
    coordinate: ArrayLike,
    normal: ArrayLike,
    /,
    *,
    value_weight: float,
    normal_weight: float,
    phi_output: int = 0,
    psi_output: int = 1,
) -> HolomorphicPointFunctional:
    """Weighted Goursat field value and supplied-normal derivative."""
    value_scale = float(value_weight)
    normal_scale = float(normal_weight)
    if not np.isfinite(value_scale) or not np.isfinite(normal_scale):
        raise ValueError("Biharmonic Robin weights must be finite.")
    if value_scale == 0.0 and normal_scale == 0.0:
        raise ValueError("Biharmonic Robin functional requires one nonzero weight.")
    value = biharmonic_value_functional(
        coordinate,
        phi_output=phi_output,
        psi_output=psi_output,
    )
    derivative = biharmonic_normal_derivative_functional(
        coordinate,
        normal,
        phi_output=phi_output,
        psi_output=psi_output,
    )
    return HolomorphicPointFunctional(
        coordinate,
        _scaled_terms(value.terms, value_scale)
        + _scaled_terms(derivative.terms, normal_scale),
        construction="biharmonic-goursat-robin-functional",
    )


def _plane_stress_terms(
    coordinate: complex,
    component: Literal["xx", "yy", "xy"],
    phi_output: int,
    psi_output: int,
    /,
) -> tuple[HolomorphicJetFunctionalTerm, ...]:
    if component == "xx":
        return (
            HolomorphicJetFunctionalTerm(phi_output, (1,), 2.0),
            HolomorphicJetFunctionalTerm(phi_output, (2,), -np.conj(coordinate)),
            HolomorphicJetFunctionalTerm(psi_output, (1,), -1.0),
        )
    if component == "yy":
        return (
            HolomorphicJetFunctionalTerm(phi_output, (1,), 2.0),
            HolomorphicJetFunctionalTerm(phi_output, (2,), np.conj(coordinate)),
            HolomorphicJetFunctionalTerm(psi_output, (1,), 1.0),
        )
    if component == "xy":
        return (
            HolomorphicJetFunctionalTerm(phi_output, (2,), -1j * np.conj(coordinate)),
            HolomorphicJetFunctionalTerm(psi_output, (1,), -1j),
        )
    raise ValueError("Unknown plane stress component.")


def plane_elasticity_stress_functional(
    coordinate: ArrayLike,
    component: Literal["xx", "yy", "xy"],
    /,
    *,
    phi_output: int = 0,
    psi_output: int = 1,
) -> HolomorphicPointFunctional:
    """One Kolosov–Muskhelishvili stress component."""
    z = _complex_scalar(coordinate, "coordinate")
    return HolomorphicPointFunctional(
        z,
        _plane_stress_terms(z, component, phi_output, psi_output),
        construction="plane-elasticity-stress-functional",
    )


def plane_elasticity_traction_functional(
    coordinate: ArrayLike,
    normal: ArrayLike,
    component: Literal["x", "y"],
    /,
    *,
    phi_output: int = 0,
    psi_output: int = 1,
) -> HolomorphicPointFunctional:
    """One physical traction component on a supplied normal."""
    z = _complex_scalar(coordinate, "coordinate")
    normal_x, normal_y = _normal_2d(normal)
    if component == "x":
        terms = _scaled_terms(
            _plane_stress_terms(z, "xx", phi_output, psi_output),
            normal_x,
        ) + _scaled_terms(
            _plane_stress_terms(z, "xy", phi_output, psi_output),
            normal_y,
        )
    elif component == "y":
        terms = _scaled_terms(
            _plane_stress_terms(z, "xy", phi_output, psi_output),
            normal_x,
        ) + _scaled_terms(
            _plane_stress_terms(z, "yy", phi_output, psi_output),
            normal_y,
        )
    else:
        raise ValueError("Plane traction component must be x or y.")
    return HolomorphicPointFunctional(
        z,
        terms,
        construction="plane-elasticity-traction-functional",
    )


def plane_elasticity_displacement_functional(
    coordinate: ArrayLike,
    material: PlaneIsotropicMaterial,
    component: Literal["x", "y"],
    /,
    *,
    phi_output: int = 0,
    psi_output: int = 1,
) -> HolomorphicPointFunctional:
    """One Kolosov–Muskhelishvili displacement component."""
    if not isinstance(material, PlaneIsotropicMaterial):
        raise TypeError("material must be PlaneIsotropicMaterial.")
    z = _complex_scalar(coordinate, "coordinate")
    scale = 1.0 / (2.0 * float(np.asarray(material.mu)))
    kappa = float(np.asarray(material.kappa))
    if component == "x":
        terms = (
            HolomorphicJetFunctionalTerm(phi_output, (0,), scale * kappa),
            HolomorphicJetFunctionalTerm(phi_output, (1,), -scale * np.conj(z)),
            HolomorphicJetFunctionalTerm(psi_output, (0,), -scale),
        )
    elif component == "y":
        terms = (
            HolomorphicJetFunctionalTerm(phi_output, (0,), -1j * scale * kappa),
            HolomorphicJetFunctionalTerm(
                phi_output,
                (1,),
                -1j * scale * np.conj(z),
            ),
            HolomorphicJetFunctionalTerm(psi_output, (0,), -1j * scale),
        )
    else:
        raise ValueError("Plane displacement component must be x or y.")
    return HolomorphicPointFunctional(
        z,
        terms,
        construction="plane-elasticity-displacement-functional",
        construction_dependencies=(material.material_id,),
    )


__all__ = [
    "biharmonic_normal_derivative_functional",
    "biharmonic_robin_functional",
    "biharmonic_value_functional",
    "plane_elasticity_displacement_functional",
    "plane_elasticity_stress_functional",
    "plane_elasticity_traction_functional",
]
