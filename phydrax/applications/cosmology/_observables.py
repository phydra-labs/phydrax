#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._numerics import gauss_legendre_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._background import FLRWBackground
from ._distances import FLRWDistancePlan
from ._products import LagrangianGrowthHistory, MatterField, MatterPowerTable


class RadialGrid(StrictModule, NonTrainableState):
    """Fixed positive-redshift integration nodes and trapezoidal weights."""

    redshifts: Array
    weights: Array
    grid_id: str = eqx.field(static=True)

    def __init__(self, redshifts: ArrayLike, /):
        nodes = np.asarray(redshifts, dtype=float).reshape((-1,))
        if (
            nodes.size < 3
            or np.any(~np.isfinite(nodes))
            or np.any(nodes <= 0.0)
            or np.any(np.diff(nodes) <= 0.0)
        ):
            raise ValueError(
                "Radial redshift nodes must be finite, positive, and increasing."
            )
        differences = np.diff(nodes)
        weights = np.concatenate(
            (
                differences[:1] / 2.0,
                (differences[:-1] + differences[1:]) / 2.0,
                differences[-1:] / 2.0,
            )
        )
        self.redshifts = jnp.asarray(nodes)
        self.weights = jnp.asarray(weights)
        self.grid_id = canonical_fingerprint(
            {"kind": "survey-radial-grid", "redshifts": nodes.tolist()}
        )


class RedshiftDistribution(StrictModule):
    """Normalized non-negative n(z) on one radial grid."""

    grid: RadialGrid
    values: Array
    bin_id: str = eqx.field(static=True)

    def __init__(self, grid: RadialGrid, values: ArrayLike, bin_id: str, /):
        if not isinstance(grid, RadialGrid):
            raise TypeError("grid must be RadialGrid.")
        identifier = str(bin_id).strip()
        distribution = jnp.asarray(values, dtype=grid.redshifts.dtype)
        if distribution.shape != grid.redshifts.shape or not identifier:
            raise ValueError("Redshift distribution shape/bin identity is invalid.")
        distribution = eqx.error_if(
            distribution,
            jnp.any(~jnp.isfinite(distribution)) | jnp.any(distribution < 0.0),
            "Redshift distribution must be finite and non-negative.",
        )
        normalization = contract("z,z->", grid.weights, distribution)
        distribution = eqx.error_if(
            distribution,
            ~jnp.isfinite(normalization) | (normalization <= 0.0),
            "Redshift distribution is not normalizable.",
        )
        self.grid = grid
        self.values = distribution / normalization
        self.bin_id = identifier


class LinearDensityTracer(StrictModule):
    distribution: RedshiftDistribution
    bias: Array
    power_field: MatterField = eqx.field(static=True)
    tracer_id: str = eqx.field(static=True)

    def __init__(
        self,
        distribution: RedshiftDistribution,
        bias: ArrayLike,
        /,
        *,
        power_field: MatterField = "total_matter",
    ):
        if not isinstance(distribution, RedshiftDistribution):
            raise TypeError("distribution must be RedshiftDistribution.")
        bias_ = jnp.asarray(bias, dtype=distribution.values.dtype)
        if bias_.shape == ():
            bias_ = jnp.broadcast_to(bias_, distribution.values.shape)
        if bias_.shape != distribution.values.shape:
            raise ValueError("Density-tracer bias must be scalar or match radial nodes.")
        bias_ = eqx.error_if(
            bias_,
            jnp.any(~jnp.isfinite(bias_)),
            "Density-tracer bias must be finite.",
        )
        if power_field not in (
            "cold_baryon",
            "total_matter",
            "massive_neutrino_total",
        ):
            raise ValueError("Unknown density-tracer power field.")
        self.distribution = distribution
        self.bias = bias_
        self.power_field = power_field
        self.tracer_id = canonical_fingerprint(
            {
                "kind": "linear-density-tracer",
                "grid": distribution.grid.grid_id,
                "bin": distribution.bin_id,
                "power_field": power_field,
            }
        )


class LensingConvergenceTracer(StrictModule):
    distribution: RedshiftDistribution
    multiplicative_calibration: Array
    tracer_id: str = eqx.field(static=True)

    def __init__(
        self,
        distribution: RedshiftDistribution,
        /,
        *,
        multiplicative_calibration: ArrayLike = 0.0,
    ):
        if not isinstance(distribution, RedshiftDistribution):
            raise TypeError("distribution must be RedshiftDistribution.")
        calibration = jnp.asarray(
            multiplicative_calibration, dtype=distribution.values.dtype
        )
        if calibration.shape != ():
            raise ValueError("Lensing multiplicative calibration must be scalar.")
        calibration = eqx.error_if(
            calibration,
            ~jnp.isfinite(calibration),
            "Lensing calibration must be finite.",
        )
        self.distribution = distribution
        self.multiplicative_calibration = calibration
        self.tracer_id = canonical_fingerprint(
            {
                "kind": "lensing-convergence-tracer",
                "grid": distribution.grid.grid_id,
                "bin": distribution.bin_id,
            }
        )


class ObservablePrediction(StrictModule):
    values: Array
    coordinates: Array
    component_labels: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    successful: Array


class LimberAngularPowerPlan(StrictModule, NonTrainableState):
    """Flat-geometry Limber density/lensing angular spectra."""

    multipoles: Array
    pair_layout: tuple[tuple[int, int], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        multipoles: ArrayLike,
        tracer_count: int,
        /,
    ):
        ell = np.asarray(multipoles, dtype=int).reshape((-1,))
        count = int(tracer_count)
        if ell.size < 1 or np.any(ell < 2) or np.any(np.diff(ell) <= 0) or count < 1:
            raise ValueError("Limber multipoles/tracer count are invalid.")
        pairs = tuple(
            (left, right) for left in range(count) for right in range(left, count)
        )
        self.multipoles = jnp.asarray(ell)
        self.pair_layout = pairs
        self.plan_id = canonical_fingerprint(
            {
                "kind": "flat-limber-angular-power",
                "multipoles": ell.tolist(),
                "pair_layout": [list(pair) for pair in pairs],
            }
        )

    def predict(
        self,
        background: FLRWBackground,
        distance_plan: FLRWDistancePlan,
        power: MatterPowerTable,
        tracers: tuple[LinearDensityTracer | LensingConvergenceTracer, ...],
        /,
    ) -> ObservablePrediction:
        if len(tracers) != max(max(pair) for pair in self.pair_layout) + 1:
            raise ValueError("Tracer tuple does not match Limber plan layout.")
        grid = tracers[0].distribution.grid
        if any(tracer.distribution.grid.grid_id != grid.grid_id for tracer in tracers):
            raise ValueError("All Limber tracers must share one radial grid.")
        token = background.require_flat(grid.redshifts)
        token = background.realization.require_compatible(power.realization, token)
        if power.descriptor.spatial_dimension != 3 or not power.descriptor.is_auto:
            raise ValueError("Limber projection requires three-dimensional auto-power.")
        geometry = distance_plan.evaluate(background, token)
        z = grid.redshifts
        scale_factor = 1.0 / (1.0 + z)
        radial = geometry.radial_comoving_distance
        hubble = background.hubble(scale_factor)
        speed = distance_plan.light_speed
        kernels = []
        labels = []
        for tracer in tracers:
            if isinstance(tracer, LinearDensityTracer):
                if (
                    power.descriptor.left_field != tracer.power_field
                    or power.descriptor.right_field != tracer.power_field
                ):
                    raise ValueError("Density tracer and matter-power field disagree.")
                kernel = tracer.distribution.values * tracer.bias * hubble / speed
                labels.append(f"density:{tracer.distribution.bin_id}")
            else:
                separation = jnp.maximum(radial[None, :] - radial[:, None], 0.0)
                safe_radial = jnp.where(radial > 0.0, radial, 1.0)
                efficiency = contract(
                    "j,ij,j->i",
                    grid.weights,
                    separation / safe_radial[None, :],
                    tracer.distribution.values,
                )
                coefficient = (
                    1.5
                    * background.matter_density
                    * background.hubble_constant**2
                    / speed**2
                )
                kernel = (
                    (1.0 + tracer.multiplicative_calibration)
                    * coefficient
                    * radial
                    / scale_factor
                    * efficiency
                )
                labels.append(f"convergence:{tracer.distribution.bin_id}")
            kernels.append(kernel)
        kernel_array = jnp.stack(kernels)
        ell = self.multipoles.astype(radial.dtype)
        safe_radial = jnp.where(radial > 0.0, radial, jnp.inf)
        wavenumber = (ell[:, None] + 0.5) / safe_radial[None, :]
        power_values = jnp.stack(
            tuple(
                power.evaluate(wavenumber[:, index], scale_factor[index])
                for index in range(z.size)
            ),
            axis=1,
        )
        radial_measure = speed / hubble
        values = []
        for left, right in self.pair_layout:
            integrand = (
                radial_measure[None, :]
                * kernel_array[left][None, :]
                * kernel_array[right][None, :]
                * power_values
                / safe_radial[None, :] ** 2
            )
            values.append(contract("z,lz->l", grid.weights, integrand))
        result = jnp.stack(values)
        return ObservablePrediction(
            values=result,
            coordinates=self.multipoles,
            component_labels=tuple(
                f"{labels[left]}x{labels[right]}" for left, right in self.pair_layout
            ),
            plan_id=self.plan_id,
            successful=jnp.all(jnp.isfinite(result)),
        )


class RSDMultipoleResult(StrictModule):
    wavenumbers: Array
    monopole: Array
    quadrupole: Array
    hexadecapole: Array
    alpha_perpendicular: Array
    alpha_parallel: Array
    successful: Array


class LinearRSDMultipolePlan(StrictModule, NonTrainableState):
    """Linear Kaiser P0/P2/P4 with Alcock-Paczynski remapping."""

    wavenumbers: Array
    mu_nodes: Array
    mu_weights: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        wavenumbers: ArrayLike,
        /,
        *,
        mu_order: int = 32,
    ):
        k = np.asarray(wavenumbers, dtype=float).reshape((-1,))
        order = int(mu_order)
        if (
            k.size < 2
            or np.any(~np.isfinite(k))
            or np.any(k <= 0.0)
            or np.any(np.diff(k) <= 0.0)
            or order < 8
        ):
            raise ValueError("RSD k grid or mu quadrature is invalid.")
        rule = gauss_legendre_data(order)
        self.wavenumbers = jnp.asarray(k)
        self.mu_nodes = jnp.asarray(rule.nodes)
        self.mu_weights = jnp.asarray(rule.weights)
        self.plan_id = canonical_fingerprint(
            {"kind": "linear-rsd-ap-multipoles", "k": k.tolist(), "mu_order": order}
        )

    def predict(
        self,
        background: FLRWBackground,
        fiducial_background: FLRWBackground,
        distance_plan: FLRWDistancePlan,
        growth: LagrangianGrowthHistory,
        power: MatterPowerTable,
        bias: ArrayLike,
        redshift: ArrayLike,
        /,
    ) -> RSDMultipoleResult:
        if power.descriptor.stage != "linear" or not power.descriptor.is_auto:
            raise ValueError("Linear RSD requires linear auto-power.")
        z = jnp.asarray(redshift, dtype=self.wavenumbers.dtype)
        bias_ = jnp.asarray(bias, dtype=self.wavenumbers.dtype)
        if z.shape != () or bias_.shape != ():
            raise ValueError("RSD redshift and bias must be scalar.")
        scale_factor = 1.0 / (1.0 + z)
        scale_factor = background.realization.require_compatible(
            growth.realization, scale_factor
        )
        scale_factor = background.realization.require_compatible(
            power.realization, scale_factor
        )
        geometry = distance_plan.evaluate(background, z)
        fiducial = distance_plan.evaluate(fiducial_background, z)
        alpha_perpendicular = (
            geometry.transverse_comoving_distance / fiducial.transverse_comoving_distance
        )
        alpha_parallel = fiducial_background.hubble(scale_factor) / background.hubble(
            scale_factor
        )
        mu = self.mu_nodes[None, :]
        k = self.wavenumbers[:, None]
        transverse = k * jnp.sqrt(jnp.maximum(1.0 - mu**2, 0.0)) / alpha_perpendicular
        parallel = k * mu / alpha_parallel
        true_k = jnp.sqrt(transverse**2 + parallel**2)
        true_mu = parallel / jnp.where(true_k > 0.0, true_k, 1.0)
        _, growth_rate, _, _ = growth.evaluate(scale_factor)
        spectrum = (
            (bias_ + growth_rate * true_mu**2) ** 2
            * power.evaluate(true_k, scale_factor)
            / (alpha_perpendicular**2 * alpha_parallel)
        )
        legendre_0 = jnp.ones_like(mu)
        legendre_2 = 0.5 * (3.0 * mu**2 - 1.0)
        legendre_4 = (35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0
        monopole = 0.5 * contract("u,ku->k", self.mu_weights, spectrum * legendre_0)
        quadrupole = 2.5 * contract("u,ku->k", self.mu_weights, spectrum * legendre_2)
        hexadecapole = 4.5 * contract("u,ku->k", self.mu_weights, spectrum * legendre_4)
        successful = jnp.all(
            jnp.isfinite(jnp.stack((monopole, quadrupole, hexadecapole)))
        )
        return RSDMultipoleResult(
            self.wavenumbers,
            monopole,
            quadrupole,
            hexadecapole,
            alpha_perpendicular,
            alpha_parallel,
            successful,
        )


__all__ = [
    "LensingConvergenceTracer",
    "LimberAngularPowerPlan",
    "LinearDensityTracer",
    "LinearRSDMultipolePlan",
    "ObservablePrediction",
    "RSDMultipoleResult",
    "RadialGrid",
    "RedshiftDistribution",
]
