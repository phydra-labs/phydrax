#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""CIE 1931 colorimetry of spectral reflectance factors and sRGB encoding.

No CIE data table is distributed with Phydrax: the CIE color-matching and
illuminant datasets are licensed CC BY-SA 4.0. The default observer is the
analytic Wyman-Sloan-Shirley (2013) multi-lobe fit; tabulated color-matching
functions and illuminants enter only as caller-supplied host resources admitted
with a SHA-256-pinned :class:`phydrax.artifacts.ArtifactManifest`.
"""

from __future__ import annotations

import abc
import math
from enum import IntFlag
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import normalized_identifier
from ..artifacts import (
    AdmittedExternalArtifact,
    ArtifactManifest,
    ExternalArtifactPolicy,
    read_admitted_artifact,
)
from ..ein import contract
from ..typing import (
    Bool,
    checked,
    Dim,
    Float,
    HostFloat64,
    Identifier,
    Int32,
    parse,
    VariadicDim,
)


ColorMatchingObserver: TypeAlias = Literal["cie-1931-2-degree"]
GamutMapping: TypeAlias = Literal["clip", "extended"]

# Declared ArtifactManifest.model of an admitted CIE 1931 2-degree CMF table.
COLOR_MATCHING_TABLE_MODEL = "cie-1931-2-degree-color-matching-functions"


class _WavelengthDim(Dim, minimum=2):
    """Vacuum-wavelength quadrature nodes of one colorimetry plan."""


class _TableDim(Dim, minimum=2):
    """Wavelength rows of one host spectral table."""


class _SampleDims(VariadicDim):
    """Broadcast spectrum-sample axes of one evaluation."""


# Wyman, Sloan and Shirley, "Simple Analytic Approximations to the CIE XYZ Color
# Matching Functions", JCGT 2(2), 2013, Eq. (4) and Table 1: channel, weight,
# center (nm), and inverse widths (1/nm) below and above the center.
_MULTILOBE_FIT = (
    (0, 0.362, 442.0, 0.0624, 0.0374),
    (0, 1.056, 599.8, 0.0264, 0.0323),
    (0, -0.065, 501.1, 0.0490, 0.0382),
    (1, 0.821, 568.8, 0.0213, 0.0247),
    (1, 0.286, 530.9, 0.0613, 0.0322),
    (2, 1.217, 437.0, 0.0845, 0.0278),
    (2, 0.681, 459.0, 0.0385, 0.0725),
)
# Table 2 of the same paper, relative to the 1 nm CIE 1931 curves on 360-830 nm.
_MULTILOBE_MAXIMUM_SQUARED_ERROR = (2.0e-4, 6.4e-5, 4.9e-4)
_MULTILOBE_MEAN_SQUARED_ERROR = (3.1e-5, 7.1e-6, 1.6e-5)
_CIE_1931_SUPPORT = (360.0e-9, 830.0e-9)

# IEC 61966-2-1 sRGB primaries and D65 white chromaticities.
_SRGB_PRIMARY_CHROMATICITIES = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))
_SRGB_WHITE_CHROMATICITY = (0.3127, 0.3290)
_SRGB_LINEAR_THRESHOLD = 0.0031308


def _chromaticity_xyz(x: float, y: float, /) -> np.ndarray:
    return np.asarray([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def _xyz_to_linear_srgb_matrix() -> np.ndarray:
    primaries = np.stack(
        [_chromaticity_xyz(x, y) for x, y in _SRGB_PRIMARY_CHROMATICITIES], axis=1
    )
    white = _chromaticity_xyz(*_SRGB_WHITE_CHROMATICITY)
    rgb_to_xyz = primaries * np.linalg.solve(primaries, white)[None, :]
    # The dense 3x3 XYZ-to-RGB conversion is itself the standardized object; it
    # is derived once on the host at full precision so that D65 white maps to 1.
    return np.linalg.solve(rgb_to_xyz, np.eye(3, dtype=np.float64))


_XYZ_TO_LINEAR_SRGB = _xyz_to_linear_srgb_matrix()


class ColorMatchingFitError(StrictModule, NonTrainableState):
    """Published per-channel error of an analytic color-matching approximation.

    Errors are ``(x, y, z)`` squared deviations from the reference curves on the
    stated reference grid.
    """

    maximum_squared_error: tuple[float, float, float] = eqx.field(static=True)
    mean_squared_error: tuple[float, float, float] = eqx.field(static=True)
    reference: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_squared_error: tuple[float, float, float],
        mean_squared_error: tuple[float, float, float],
        /,
        *,
        reference: str,
    ) -> None:
        maximum = tuple(float(value) for value in maximum_squared_error)
        mean = tuple(float(value) for value in mean_squared_error)
        if len(maximum) != 3 or len(mean) != 3:
            raise ValueError("Fit errors must have one value per tristimulus channel.")
        if not all(math.isfinite(value) and value >= 0.0 for value in maximum + mean):
            raise ValueError("Fit errors must be finite and nonnegative.")
        self.maximum_squared_error = (maximum[0], maximum[1], maximum[2])
        self.mean_squared_error = (mean[0], mean[1], mean[2])
        self.reference = normalized_identifier(reference, "reference")


class AbstractColorMatchingFunctions(StrictModule, NonTrainableState):
    """Host-evaluable CIE 1931 2-degree color-matching functions ``(x, y, z)``."""

    observer: eqx.AbstractVar[ColorMatchingObserver]
    fit_error: eqx.AbstractVar[ColorMatchingFitError | None]
    color_matching_id: eqx.AbstractVar[str]

    @property
    @abc.abstractmethod
    def support(self) -> tuple[float, float]:
        """Closed vacuum-wavelength interval (m) on which the functions are defined."""

    @abc.abstractmethod
    def sample(self, wavelengths: np.ndarray, /) -> np.ndarray:
        """Return host float64 values of shape ``(W, 3)`` at wavelengths in m."""


class AnalyticColorMatchingFunctions(AbstractColorMatchingFunctions):
    """Wyman-Sloan-Shirley (2013) multi-lobe piecewise-Gaussian CIE 1931 fit.

    Each lobe is ``a exp(-((lambda - mu) tau)^2 / 2)`` with ``tau`` switching at
    ``mu``. The published error (Table 2 of the paper, against the 1 nm CIE 1931
    curves) is carried in :attr:`fit_error`; it is below the within-subject
    variance of the color-matching experiments. The support is the CIE 1931
    table interval 360-830 nm.
    """

    observer: ColorMatchingObserver = eqx.field(static=True)
    fit_error: ColorMatchingFitError | None
    color_matching_id: str = eqx.field(static=True)

    def __init__(self) -> None:
        self.observer = "cie-1931-2-degree"
        self.fit_error = ColorMatchingFitError(
            _MULTILOBE_MAXIMUM_SQUARED_ERROR,
            _MULTILOBE_MEAN_SQUARED_ERROR,
            reference="Wyman, Sloan, Shirley, JCGT 2(2) 2013, Table 2",
        )
        self.color_matching_id = canonical_fingerprint(
            {
                "kind": "analytic-color-matching-functions",
                "observer": self.observer,
                "lobes": [list(lobe) for lobe in _MULTILOBE_FIT],
            }
        )

    @property
    def support(self) -> tuple[float, float]:
        return _CIE_1931_SUPPORT

    def sample(self, wavelengths: np.ndarray, /) -> np.ndarray:
        lobes = np.asarray(_MULTILOBE_FIT, dtype=np.float64)
        channels = lobes[:, 0].astype(np.int64)
        offset = np.asarray(wavelengths, dtype=np.float64)[:, None] * 1.0e9 - lobes[:, 2]
        scaled = offset * np.where(offset < 0.0, lobes[:, 3], lobes[:, 4])
        values = lobes[:, 1] * np.exp(-0.5 * scaled * scaled)
        return values @ np.eye(3, dtype=np.float64)[channels]


def _read_spectral_table(
    artifact: AdmittedExternalArtifact,
    manifest: ArtifactManifest,
    policy: ExternalArtifactPolicy,
    value_columns: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode headerless ``wavelength_nm,value...`` CSV rows of verified bytes."""
    payload = read_admitted_artifact(artifact, manifest, policy=policy)
    lines = payload.decode("ascii").splitlines()
    rows = [line.split(",") for line in lines if line.strip()]
    if len(rows) < 2 or any(len(row) != value_columns + 1 for row in rows):
        raise ValueError(
            f"Spectral table rows must be 'wavelength_nm' plus {value_columns} values."
        )
    table = np.asarray([[float(cell) for cell in row] for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(table)):
        raise ValueError("Spectral table values must be finite.")
    return table[:, 0] / 1.0e9, table[:, 1:]


def _host_table_grid(value: ArrayLike, name: str, /) -> np.ndarray:
    grid = np.asarray(value, dtype=np.float64)
    if grid.ndim != 1 or grid.shape[0] < 2:
        raise ValueError(f"{name} must be a rank-one array with at least two samples.")
    if not np.all(np.isfinite(grid)) or not np.all(grid > 0.0):
        raise ValueError(f"{name} must be finite and positive.")
    if not np.all(np.diff(grid) > 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return grid


class TabulatedColorMatchingFunctions(AbstractColorMatchingFunctions):
    """CIE 1931 2-degree functions decoded from one admitted host table.

    The table bytes are re-read through the admitted-artifact reader, so their
    size and SHA-256 must match the caller's trusted manifest, whose ``model``
    must be ``COLOR_MATCHING_TABLE_MODEL``. Rows are headerless
    ``wavelength_nm,x,y,z`` (the CIE ``CIE_xyz_1931_2deg.csv`` layout). Values
    between rows are linearly interpolated; the support is the table interval.
    """

    __strict_contract__ = True

    table_wavelengths: HostFloat64[_TableDim]
    table_values: HostFloat64[_TableDim, Literal[3]]
    observer: ColorMatchingObserver = eqx.field(static=True)
    fit_error: ColorMatchingFitError | None
    source_sha256: str = eqx.field(static=True)
    color_matching_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        artifact: AdmittedExternalArtifact,
        manifest: ArtifactManifest,
        /,
        *,
        policy: ExternalArtifactPolicy,
    ) -> None:
        if manifest.model != COLOR_MATCHING_TABLE_MODEL:
            raise ValueError(
                "The color-matching manifest model must be "
                f"{COLOR_MATCHING_TABLE_MODEL!r}."
            )
        wavelengths, values = _read_spectral_table(artifact, manifest, policy, 3)
        grid = _host_table_grid(wavelengths, "color-matching table wavelengths")
        if np.any(values < 0.0):
            raise ValueError("Color-matching table values must be nonnegative.")
        self.table_wavelengths = grid
        self.table_values = values
        self.observer = "cie-1931-2-degree"
        self.fit_error = None
        self.source_sha256 = manifest.sha256
        self.color_matching_id = canonical_fingerprint(
            {
                "kind": "tabulated-color-matching-functions",
                "observer": self.observer,
                "manifest": manifest.manifest_id,
                "sha256": manifest.sha256,
            }
        )

    @property
    def support(self) -> tuple[float, float]:
        return (float(self.table_wavelengths[0]), float(self.table_wavelengths[-1]))

    def sample(self, wavelengths: np.ndarray, /) -> np.ndarray:
        points = np.asarray(wavelengths, dtype=np.float64)
        return np.stack(
            [
                np.interp(points, self.table_wavelengths, self.table_values[:, channel])
                for channel in range(3)
            ],
            axis=-1,
        )


class SpectralIlluminant(StrictModule, NonTrainableState):
    """Relative spectral power of one named illuminant on a host wavelength table.

    Wavelengths are strictly increasing vacuum wavelengths in m; the power is
    nonnegative, finite and not identically zero. Only relative power matters:
    colorimetry normalizes a perfect reflector to ``Y = 1``. ``source_sha256``
    identifies the admitted host resource the samples were decoded from
    (see :func:`read_spectral_illuminant`) and is ``None`` for caller arrays.
    """

    __strict_contract__ = True

    wavelengths: HostFloat64[_TableDim]
    relative_power: HostFloat64[_TableDim]
    illuminant_id: Identifier = eqx.field(static=True)
    source_sha256: str | None = eqx.field(static=True)
    source_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        wavelengths: ArrayLike,
        relative_power: ArrayLike,
        /,
        *,
        illuminant_id: str,
        source_sha256: str | None = None,
    ) -> None:
        grid = _host_table_grid(wavelengths, "illuminant wavelengths")
        power = np.asarray(relative_power, dtype=np.float64)
        if power.shape != grid.shape:
            raise ValueError("relative_power must have one value per wavelength.")
        if (
            not np.all(np.isfinite(power))
            or np.any(power < 0.0)
            or not np.any(power > 0.0)
        ):
            raise ValueError(
                "relative_power must be finite, nonnegative and not identically zero."
            )
        identifier = normalized_identifier(illuminant_id, "illuminant_id")
        if source_sha256 is not None and (
            len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256)
        ):
            raise ValueError("source_sha256 must be 64 lowercase hexadecimal digits.")
        self.wavelengths = grid
        self.relative_power = power
        self.illuminant_id = identifier
        self.source_sha256 = source_sha256
        self.source_id = canonical_fingerprint(
            {
                "kind": "spectral-illuminant",
                "illuminant_id": identifier,
                "wavelengths": grid,
                "relative_power": power,
                "source_sha256": source_sha256,
            }
        )


def read_spectral_illuminant(
    artifact: AdmittedExternalArtifact,
    manifest: ArtifactManifest,
    /,
    *,
    policy: ExternalArtifactPolicy,
    illuminant_id: str,
) -> SpectralIlluminant:
    """Decode one illuminant from admitted, SHA-256-verified host bytes.

    Rows are headerless ``wavelength_nm,relative_power`` (the CIE
    ``CIE_std_illum_D65.csv`` layout). The bytes are re-read and re-verified
    against the trusted manifest before decoding.
    """
    if not isinstance(manifest, ArtifactManifest):
        raise TypeError("manifest must be an ArtifactManifest.")
    wavelengths, values = _read_spectral_table(artifact, manifest, policy, 1)
    return SpectralIlluminant(
        wavelengths,
        values[:, 0],
        illuminant_id=illuminant_id,
        source_sha256=manifest.sha256,
    )


class SpectralColorimetryStatus(IntFlag):
    """Fail-closed per-sample status bits of one colorimetry evaluation."""

    SUCCESS = 0
    NONFINITE_SPECTRUM = 1
    NEGATIVE_SPECTRUM = 2


class SpectralColorimetryEvidence(StrictModule):
    """Gamut, validity, and white-balance evidence of one colorimetry evaluation.

    ``gamut_excess`` is the largest distance of a linear-sRGB component outside
    ``[0, 1]`` after exposure; ``white_neutral_error`` is the plan's
    ``max |white_linear_srgb - 1|``.
    """

    __strict_contract__ = True

    out_of_gamut: Bool[_SampleDims]
    gamut_excess: Float[_SampleDims]
    finite: Bool[_SampleDims]
    accepted: Bool[_SampleDims]
    status: Int32[_SampleDims]
    white_neutral_error: float = eqx.field(static=True)
    gamut_mapping: GamutMapping = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)


class SpectralColorimetryResult(StrictModule):
    """Tristimulus, linear sRGB, and encoded sRGB colors of spectral samples.

    ``linear_srgb`` is the exposed, unmapped linear sRGB value;
    ``display_linear_srgb`` applies the plan's gamut mapping and
    ``encoded_srgb`` applies the IEC 61966-2-1 transfer function to it.
    Rejected samples carry NaN values.
    """

    __strict_contract__ = True

    xyz: Float[_SampleDims, Literal[3]]
    linear_srgb: Float[_SampleDims, Literal[3]]
    display_linear_srgb: Float[_SampleDims, Literal[3]]
    encoded_srgb: Float[_SampleDims, Literal[3]]
    evidence: SpectralColorimetryEvidence

    @property
    def status(self) -> Array:
        return self.evidence.status

    @property
    def accepted(self) -> Array:
        return self.evidence.accepted


def _plan_grid(value: ArrayLike, /) -> np.ndarray:
    grid = np.asarray(value)
    if not np.issubdtype(grid.dtype, np.floating):
        raise TypeError("wavelengths must be real floating vacuum wavelengths in m.")
    _host_table_grid(grid, "wavelengths")
    return grid


def _trapezoid_weights(grid: np.ndarray, /) -> np.ndarray:
    spacing = np.diff(grid)
    weights = np.zeros_like(grid)
    weights[:-1] += 0.5 * spacing
    weights[1:] += 0.5 * spacing
    return weights


def _coverage(
    color_matching: AbstractColorMatchingFunctions, lower: float, upper: float, /
) -> tuple[float, float, float]:
    support_lower, support_upper = color_matching.support
    fine = np.linspace(support_lower, support_upper, 4701, dtype=np.float64)
    values = color_matching.sample(fine) * _trapezoid_weights(fine)[:, None]
    inside = (fine >= lower) & (fine <= upper)
    fraction = values[inside].sum(axis=0) / values.sum(axis=0)
    return (float(fraction[0]), float(fraction[1]), float(fraction[2]))


class SpectralColorimetryPlan(StrictModule, NonTrainableState):
    """CIE 1931 tristimulus quadrature of reflectance factors under one illuminant.

    For a spectral reflectance (or transmittance) factor ``rho`` sampled on the
    plan wavelengths, ``XYZ = k sum_j w_j S_j rho_j cmf_j`` with trapezoid weights
    ``w``, illuminant power ``S`` linearly interpolated from its table and
    ``k = 1 / sum_j w_j S_j ybar_j``, so a perfect reflector has ``Y = 1``. The
    wavelengths must lie inside both the color-matching support and the
    illuminant table; nothing is extrapolated. Linear sRGB is
    ``exposure * M XYZ`` with the IEC 61966-2-1 matrix derived from the sRGB
    primaries and D65 white point.

    ``white_linear_srgb`` is the color of a perfect reflector without exposure.
    Under CIE D65 it is neutral up to the observer approximation: with the
    analytic default and a 1 nm grid, ``white_neutral_error`` is about 2.5e-3
    (see the thin-film optics guide for the qualification record).
    ``color_matching_coverage`` is the fraction of each color-matching integral
    inside the plan interval. ``gamut_mapping="clip"`` clips linear sRGB to
    ``[0, 1]`` before encoding; ``"extended"`` encodes out-of-range values with
    the sign-symmetric extension of the transfer function. Out-of-gamut samples
    are reported either way.
    """

    __strict_contract__ = True

    wavelengths: Float[_WavelengthDim]
    tristimulus_weights: Float[_WavelengthDim, Literal[3]]
    white_xyz: Float[Literal[3]]
    white_linear_srgb: Float[Literal[3]]
    fit_error: ColorMatchingFitError | None
    exposure: float = eqx.field(static=True)
    gamut_mapping: GamutMapping = eqx.field(static=True)
    observer: ColorMatchingObserver = eqx.field(static=True)
    illuminant_id: Identifier = eqx.field(static=True)
    illuminant_source_id: Identifier = eqx.field(static=True)
    color_matching_id: Identifier = eqx.field(static=True)
    white_neutral_error: float = eqx.field(static=True)
    color_matching_coverage: tuple[float, float, float] = eqx.field(static=True)
    maximum_wavelength_spacing: float = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        wavelengths: ArrayLike,
        illuminant: SpectralIlluminant,
        /,
        *,
        color_matching: AbstractColorMatchingFunctions | None = None,
        gamut_mapping: GamutMapping = "clip",
        exposure: float = 1.0,
    ) -> None:
        grid = _plan_grid(wavelengths)
        observer_functions = (
            AnalyticColorMatchingFunctions() if color_matching is None else color_matching
        )
        if not isinstance(observer_functions, AbstractColorMatchingFunctions):
            raise TypeError("color_matching must be AbstractColorMatchingFunctions.")
        mapping = parse(gamut_mapping, GamutMapping, "gamut_mapping")
        scale = float(exposure)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("exposure must be finite and positive.")
        nodes = grid.astype(np.float64)
        # Support bounds are compared in the grid precision so that a float32
        # grid node equal to a rounded table endpoint is inside the table.
        lower, upper = np.asarray(observer_functions.support, dtype=grid.dtype)
        if grid[0] < lower or grid[-1] > upper:
            raise ValueError("wavelengths must lie inside the color-matching support.")
        table = illuminant.wavelengths.astype(grid.dtype)
        if grid[0] < table[0] or grid[-1] > table[-1]:
            raise ValueError("wavelengths must lie inside the illuminant table.")
        power = np.interp(nodes, illuminant.wavelengths, illuminant.relative_power)
        weighted = (
            (_trapezoid_weights(nodes) * power)[:, None]
            * observer_functions.sample(nodes)
        )
        luminance = weighted[:, 1].sum()
        if not luminance > 0.0:
            raise ValueError("The illuminant has no luminance on the plan wavelengths.")
        weighted = weighted / luminance
        white = weighted.sum(axis=0)
        white_srgb = _XYZ_TO_LINEAR_SRGB @ white
        dtype = grid.dtype
        self.wavelengths = jnp.asarray(grid)
        self.tristimulus_weights = jnp.asarray(weighted, dtype=dtype)
        self.white_xyz = jnp.asarray(white, dtype=dtype)
        self.white_linear_srgb = jnp.asarray(white_srgb, dtype=dtype)
        self.fit_error = observer_functions.fit_error
        self.exposure = scale
        self.gamut_mapping = mapping
        self.observer = observer_functions.observer
        self.illuminant_id = illuminant.illuminant_id
        self.illuminant_source_id = illuminant.source_id
        self.color_matching_id = observer_functions.color_matching_id
        self.white_neutral_error = float(np.max(np.abs(white_srgb - 1.0)))
        self.color_matching_coverage = _coverage(observer_functions, nodes[0], nodes[-1])
        self.maximum_wavelength_spacing = float(np.max(np.diff(nodes)))
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-colorimetry-plan",
                "wavelengths": grid,
                "illuminant": illuminant.source_id,
                "color_matching": observer_functions.color_matching_id,
                "gamut_mapping": mapping,
                "exposure": scale.hex(),
            }
        )

    def evaluate(self, spectral_factor: ArrayLike, /) -> SpectralColorimetryResult:
        """Map reflectance factors of shape ``B + (W,)`` to colors of ``B + (3,)``."""
        factor = _spectral_samples(self, spectral_factor)
        finite = jnp.all(jnp.isfinite(factor), axis=-1)
        nonnegative = jnp.all(
            jnp.where(jnp.isfinite(factor), factor >= 0.0, True), axis=-1
        )
        valid = finite & nonnegative
        safe = jnp.where(valid[..., None], factor, 0.0)
        xyz = spectral_to_xyz(self, safe)
        linear = self.exposure * xyz_to_linear_srgb(xyz)
        excess = jnp.max(jnp.maximum(jnp.maximum(-linear, linear - 1.0), 0.0), axis=-1)
        match self.gamut_mapping:
            case "clip":
                display = jnp.clip(linear, 0.0, 1.0)
            case "extended":
                display = linear
            case _:
                assert_never(self.gamut_mapping)
        encoded = encode_srgb(display)
        status = jnp.where(
            ~finite, int(SpectralColorimetryStatus.NONFINITE_SPECTRUM), 0
        ).astype(jnp.int32) | jnp.where(
            finite & ~nonnegative, int(SpectralColorimetryStatus.NEGATIVE_SPECTRUM), 0
        ).astype(jnp.int32)
        mask = valid[..., None]
        evidence = SpectralColorimetryEvidence(
            out_of_gamut=(excess > 0.0) & valid,
            gamut_excess=jnp.where(valid, excess, jnp.nan),
            finite=finite,
            accepted=status == int(SpectralColorimetryStatus.SUCCESS),
            status=status,
            white_neutral_error=self.white_neutral_error,
            gamut_mapping=self.gamut_mapping,
            plan_id=self.plan_id,
        )
        return SpectralColorimetryResult(
            xyz=jnp.where(mask, xyz, jnp.nan),
            linear_srgb=jnp.where(mask, linear, jnp.nan),
            display_linear_srgb=jnp.where(mask, display, jnp.nan),
            encoded_srgb=jnp.where(mask, encoded, jnp.nan),
            evidence=evidence,
        )


def _spectral_samples(plan: SpectralColorimetryPlan, value: ArrayLike, /) -> Array:
    samples = jnp.asarray(value)
    if not jnp.issubdtype(samples.dtype, jnp.floating):
        raise TypeError("Spectral samples must be real floating data.")
    if samples.shape[-1:] != plan.wavelengths.shape:
        raise ValueError(
            "Spectral samples must end with one axis over the plan wavelengths."
        )
    return samples


def spectral_to_xyz(
    plan: SpectralColorimetryPlan, spectral_factor: ArrayLike, /
) -> Array:
    """Return CIE 1931 ``XYZ`` of spectral factors ``B + (W,)`` as ``B + (3,)``.

    A perfect reflector (all ones) maps to the plan's ``white_xyz`` with ``Y = 1``.
    """
    if not isinstance(plan, SpectralColorimetryPlan):
        raise TypeError("plan must be a SpectralColorimetryPlan.")
    factor = _spectral_samples(plan, spectral_factor)
    weights = plan.tristimulus_weights.astype(
        jnp.result_type(factor.dtype, plan.tristimulus_weights.dtype)
    )
    return contract("...w,wc->...c", factor, weights)


def xyz_to_linear_srgb(xyz: ArrayLike, /) -> Array:
    """Return linear sRGB ``(R, G, B)`` of CIE 1931 ``XYZ`` with D65 white ``Y = 1``."""
    values = jnp.asarray(xyz)
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError("xyz must be real floating data.")
    if values.shape[-1:] != (3,):
        raise ValueError("xyz must have a trailing tristimulus axis of length 3.")
    matrix = jnp.asarray(_XYZ_TO_LINEAR_SRGB, dtype=values.dtype)
    return contract("rc,...c->...r", matrix, values)


def encode_srgb(linear_srgb: ArrayLike, /) -> Array:
    """Apply the IEC 61966-2-1 sRGB transfer function to linear components.

    ``12.92 c`` for ``|c| <= 0.0031308`` and ``1.055 c^(1/2.4) - 0.055`` above,
    so 0 and 1 are fixed points. Values outside ``[0, 1]`` use the
    sign-symmetric extension; clip first for standard display values.
    """
    values = jnp.asarray(linear_srgb)
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError("linear_srgb must be real floating data.")
    magnitude = jnp.maximum(jnp.abs(values), _SRGB_LINEAR_THRESHOLD)
    curve = jnp.sign(values) * (1.055 * magnitude ** (1.0 / 2.4) - 0.055)
    return jnp.where(jnp.abs(values) <= _SRGB_LINEAR_THRESHOLD, 12.92 * values, curve)


__all__ = [
    "AbstractColorMatchingFunctions",
    "AnalyticColorMatchingFunctions",
    "COLOR_MATCHING_TABLE_MODEL",
    "ColorMatchingFitError",
    "ColorMatchingObserver",
    "GamutMapping",
    "SpectralColorimetryEvidence",
    "SpectralColorimetryPlan",
    "SpectralColorimetryResult",
    "SpectralColorimetryStatus",
    "SpectralIlluminant",
    "TabulatedColorMatchingFunctions",
    "encode_srgb",
    "read_spectral_illuminant",
    "spectral_to_xyz",
    "xyz_to_linear_srgb",
]
