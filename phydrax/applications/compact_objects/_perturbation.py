#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from numbers import Integral
from typing import Literal, TypeAlias

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class PerturbationStatus(IntEnum):
    """Portable status for one separated black-hole perturbation component."""

    SUCCESS = 0
    INVALID_MODE = 1
    INVALID_DOMAIN = 2
    NONFINITE = 3
    ANGULAR_NONCONVERGENCE = 4
    RADIAL_NONCONVERGENCE = 5
    RESIDUAL_TOLERANCE_NOT_MET = 6
    ASYMPTOTIC_MISMATCH = 7
    MODE_NOT_ISOLATED = 8
    DERIVATIVE_INVALID = 9


class PerturbationConvention(StrictModule, NonTrainableState):
    """The fixed separation, Fourier, tetrad, and normalization convention.

    Every separated field has phase ``exp(-i omega t + i m phi)``.  Angular
    coefficients are expanded in orthonormal spin-weighted spherical harmonics,
    have unit Euclidean norm, and are phase-fixed by a real-positive target
    spherical coefficient.  Kerr radial coefficients use the Kinnersley tetrad.
    Radial profiles are reported with unit maximum modulus on their finite
    evidence domain; physical scattering amplitudes must be flux-normalized by
    the scattering layer rather than inferred from that display normalization.
    """

    fourier_phase: str = eqx.field(static=True)
    time_sign: int = eqx.field(static=True)
    azimuthal_sign: int = eqx.field(static=True)
    angular_basis: str = eqx.field(static=True)
    angular_normalization: str = eqx.field(static=True)
    radial_normalization: str = eqx.field(static=True)
    tetrad: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)

    def __init__(self):
        phase = "exp(-i*omega*t+i*m*phi)"
        angular_basis = "orthonormal spin-weighted spherical harmonics"
        angular_normalization = (
            "sum(abs(coefficients)**2)=1; target coefficient real-positive"
        )
        radial_normalization = "max finite-domain abs(radial solution)=1"
        tetrad = "Kinnersley"
        self.fourier_phase = phase
        self.time_sign = -1
        self.azimuthal_sign = 1
        self.angular_basis = angular_basis
        self.angular_normalization = angular_normalization
        self.radial_normalization = radial_normalization
        self.tetrad = tetrad
        self.convention_id = canonical_fingerprint(
            {
                "kind": "black-hole-perturbation-convention",
                "fourier_phase": phase,
                "time_sign": -1,
                "azimuthal_sign": 1,
                "angular_basis": angular_basis,
                "angular_normalization": angular_normalization,
                "radial_normalization": radial_normalization,
                "tetrad": tetrad,
            }
        )


_DEFAULT_CONVENTION_ID = PerturbationConvention().convention_id
_SUPPORTED_SPIN_WEIGHTS = frozenset((-2, -1, 0, 1, 2))


class SeparatedMode(StrictModule, NonTrainableState):
    """Static identity of one separated perturbation branch.

    ``family`` distinguishes, for example, a complex-frequency QNM branch from
    a real-frequency scattering channel.  ``sector`` distinguishes the
    Schwarzschild axial/polar master equations from the Teukolsky family; it is
    never inferred from eigenvalue ordering.
    """

    spin_weight: int = eqx.field(static=True)
    ell: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    overtone: int = eqx.field(static=True)
    sector: str = eqx.field(static=True)
    family: str = eqx.field(static=True)
    background_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    mode_id: str = eqx.field(static=True)

    def __init__(
        self,
        spin_weight: int,
        ell: int,
        m: int,
        /,
        *,
        overtone: int = 0,
        sector: str = "teukolsky",
        family: str = "qnm",
        background_id: str,
        convention_id: str | None = None,
    ):
        values = (spin_weight, ell, m, overtone)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) for value in values
        ):
            raise TypeError("Separated mode indices must be integers.")
        spin_, ell_, m_, overtone_ = (int(value) for value in values)
        if spin_ not in _SUPPORTED_SPIN_WEIGHTS:
            raise ValueError("spin_weight must be one of -2, -1, 0, 1, or 2.")
        if ell_ < max(abs(spin_), abs(m_)):
            raise ValueError("ell must be at least max(abs(spin_weight), abs(m)).")
        if overtone_ < 0:
            raise ValueError("overtone must be non-negative.")
        sector_, family_ = str(sector), str(family)
        background = str(background_id)
        convention = (
            _DEFAULT_CONVENTION_ID if convention_id is None else str(convention_id)
        )
        if not sector_ or not family_ or not background or not convention:
            raise ValueError(
                "Mode sector, family, background, and convention IDs must be non-empty."
            )
        self.spin_weight = spin_
        self.ell = ell_
        self.m = m_
        self.overtone = overtone_
        self.sector = sector_
        self.family = family_
        self.background_id = background
        self.convention_id = convention
        self.mode_id = canonical_fingerprint(
            {
                "kind": "separated-black-hole-mode",
                "spin_weight": spin_,
                "ell": ell_,
                "m": m_,
                "overtone": overtone_,
                "sector": sector_,
                "family": family_,
                "background": background,
                "convention": convention,
            }
        )


RadialWaveSense: TypeAlias = Literal["ingoing", "outgoing"]


class RadialBoundaryCondition(StrictModule, NonTrainableState):
    """Pure horizon and infinity phase choices under the fixed Fourier sign."""

    horizon: RadialWaveSense = eqx.field(static=True)
    infinity: RadialWaveSense = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        horizon: RadialWaveSense = "ingoing",
        infinity: RadialWaveSense = "outgoing",
        /,
        *,
        convention_id: str | None = None,
    ):
        if horizon not in ("ingoing", "outgoing") or infinity not in (
            "ingoing",
            "outgoing",
        ):
            raise ValueError("Radial boundary senses must be 'ingoing' or 'outgoing'.")
        convention = (
            _DEFAULT_CONVENTION_ID if convention_id is None else str(convention_id)
        )
        if not convention:
            raise ValueError("convention_id must be non-empty.")
        self.horizon = horizon
        self.infinity = infinity
        self.convention_id = convention
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "black-hole-radial-boundary-condition",
                "horizon": horizon,
                "infinity": infinity,
                "convention": convention,
            }
        )


__all__ = [
    "PerturbationConvention",
    "PerturbationStatus",
    "RadialBoundaryCondition",
    "RadialWaveSense",
    "SeparatedMode",
]
