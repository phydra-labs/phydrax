#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


_TIME_CONVENTION = "exp(-i*omega*t)"
_PDE_ID = "three-dimensional-inviscid-incompressible-irrotational-laplace"
_FORMULATION_ID = "fourier-bessel-free-surface-green-radiation-pole-subtraction"
_PROVIDER_ID = "phydrax-fixed-gauss-legendre-trapezoidal-sommerfeld"
_PRECISION_ID = "float64-complex128"
_NON_GOALS = (
    "continuum-discretization certification",
    "forward speed",
    "viscosity",
    "nonlinear free surfaces",
    "irregular-frequency removal",
)


def _nonempty(value: str, name: str, /) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


class FreeSurfaceGreenPolicy3D(StrictModule, NonTrainableState):
    """Fixed quadrature, dispersion, tail, and memory policy for the 3D kernel."""

    radial_order_per_interval: int = eqx.field(static=True)
    angular_order: int = eqx.field(static=True)
    cutoff_clearance_factor: float = eqx.field(static=True)
    minimum_cutoff_root_ratio: float = eqx.field(static=True)
    maximum_wavenumber: float = eqx.field(static=True)
    maximum_spectral_tail_bound: float = eqx.field(static=True)
    root_tolerance: float = eqx.field(static=True)
    max_root_iterations: int = eqx.field(static=True)
    max_resident_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        radial_order_per_interval: int = 32,
        angular_order: int = 32,
        cutoff_clearance_factor: float = 12.0,
        minimum_cutoff_root_ratio: float = 2.0,
        maximum_wavenumber: float = 2_000.0,
        maximum_spectral_tail_bound: float = 1.0e-2,
        root_tolerance: float = 1.0e-13,
        max_root_iterations: int = 128,
        max_resident_bytes: int = 32 * 1024 * 1024,
    ):
        radial = int(radial_order_per_interval)
        angular = int(angular_order)
        cutoff_factor = float(cutoff_clearance_factor)
        root_ratio = float(minimum_cutoff_root_ratio)
        maximum = float(maximum_wavenumber)
        tail_limit = float(maximum_spectral_tail_bound)
        tolerance = float(root_tolerance)
        iterations = int(max_root_iterations)
        resident = int(max_resident_bytes)
        if radial < 8:
            raise ValueError("radial_order_per_interval must be at least eight.")
        if angular < 8 or angular % 2:
            raise ValueError("angular_order must be an even integer of at least eight.")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (cutoff_factor, maximum, tail_limit, tolerance)
        ):
            raise ValueError(
                "Green cutoff, tail, and root controls must be finite and positive."
            )
        if not math.isfinite(root_ratio) or root_ratio <= 1.0:
            raise ValueError("minimum_cutoff_root_ratio must exceed one.")
        if iterations < 1 or resident < 1:
            raise ValueError("Green iteration and memory limits must be positive.")
        self.radial_order_per_interval = radial
        self.angular_order = angular
        self.cutoff_clearance_factor = cutoff_factor
        self.minimum_cutoff_root_ratio = root_ratio
        self.maximum_wavenumber = maximum
        self.maximum_spectral_tail_bound = tail_limit
        self.root_tolerance = tolerance
        self.max_root_iterations = iterations
        self.max_resident_bytes = resident
        self.policy_id = canonical_fingerprint(
            {
                "kind": "free-surface-green-policy-3d",
                "radial_order_per_interval": radial,
                "angular_order": angular,
                "cutoff_clearance_factor": cutoff_factor,
                "minimum_cutoff_root_ratio": root_ratio,
                "maximum_wavenumber": maximum,
                "maximum_spectral_tail_bound": tail_limit,
                "root_tolerance": tolerance,
                "max_root_iterations": iterations,
                "max_resident_bytes": resident,
            }
        )


class FiniteDepthDispersionRoot3D(StrictModule, NonTrainableState):
    """Qualified positive root of k tanh(k h) = omega²/g for this 3D slice."""

    wavenumber: Array
    residual: Array
    bracket: Array
    converged: Array
    iterations: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)
    pde_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)


class FreeSurfaceGreenResourceEvidence3D(StrictModule, NonTrainableState):
    """Exact storage and conservative fixed-action scratch upper bound."""

    radial_node_count: int = eqx.field(static=True)
    angular_node_count: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    action_workspace_bytes: int = eqx.field(static=True)
    maximum_resident_bytes: int = eqx.field(static=True)
    within_policy: Array


class FreeSurfaceGreenErrorEvidence3D(StrictModule, NonTrainableState):
    """Dispersion, truncation, and stated non-certification evidence."""

    dispersion_residual: Array
    dispersion_tolerance: float = eqx.field(static=True)
    wavenumber_cutoff: float = eqx.field(static=True)
    minimum_clearance: float = eqx.field(static=True)
    spectral_tail_envelope_bound: Array
    maximum_spectral_tail_bound: float = eqx.field(static=True)
    radial_order_per_interval: int = eqx.field(static=True)
    angular_order: int = eqx.field(static=True)
    quadrature_convergence_estimated: bool = eqx.field(static=True)
    continuum_discretization_error_estimated: bool = eqx.field(static=True)
    supported: Array


def solve_finite_depth_dispersion_3d(
    angular_frequency: float,
    gravity: float,
    depth: float,
    /,
    *,
    tolerance: float = 1.0e-13,
    max_iterations: int = 128,
    frame_id: str = "z-up-cartesian",
    unit_system_id: str = "si",
) -> FiniteDepthDispersionRoot3D:
    """Bracket and bisect the unique positive finite-depth dispersion root."""
    omega = float(angular_frequency)
    gravity_ = float(gravity)
    depth_ = float(depth)
    tolerance_ = float(tolerance)
    iterations_limit = int(max_iterations)
    if any(
        not math.isfinite(value) or value <= 0.0 for value in (omega, gravity_, depth_)
    ):
        raise ValueError(
            "angular_frequency, gravity, and depth must be finite and positive."
        )
    if not math.isfinite(tolerance_) or tolerance_ <= 0.0 or iterations_limit < 1:
        raise ValueError("Dispersion root controls must be finite and positive.")
    frame = _nonempty(frame_id, "frame_id")
    units = _nonempty(unit_system_id, "unit_system_id")
    nu = omega * omega / gravity_

    def equation(wavenumber: float) -> float:
        return wavenumber * math.tanh(wavenumber * depth_) - nu

    lower = 0.0
    upper = max(2.0 * nu, 1.0 / depth_)
    while equation(upper) <= 0.0:
        upper *= 2.0
        if not math.isfinite(upper):
            raise ValueError("Could not bracket the finite-depth dispersion root.")
    for completed in range(1, iterations_limit + 1):
        midpoint = 0.5 * (lower + upper)
        value = equation(midpoint)
        if value > 0.0:
            upper = midpoint
        else:
            lower = midpoint
        if abs(value) <= tolerance_ * max(nu, 1.0) or (upper - lower) <= tolerance_ * max(
            midpoint, 1.0
        ):
            break
    root = midpoint
    residual = abs(equation(root))
    converged = residual <= 2.0 * tolerance_ * max(nu, 1.0)
    return FiniteDepthDispersionRoot3D(
        wavenumber=jnp.asarray(root, dtype=jnp.float64),
        residual=jnp.asarray(residual, dtype=jnp.float64),
        bracket=jnp.asarray((lower, upper), dtype=jnp.float64),
        converged=jnp.asarray(converged),
        iterations=completed,
        ambient_dimension=3,
        coordinate_convention="right-handed-cartesian-z-up",
        pde_id=_PDE_ID,
        geometry_id=f"horizontal-free-surface:z=0:bottom={-depth_:.17g}",
        formulation_id="monotone-finite-depth-dispersion-root",
        provider_id="phydrax-host-bisection",
        precision_id="float64",
        frame_id=frame,
        unit_system_id=units,
        non_goals=("capillary dispersion", "current-modified dispersion"),
    )


class FreeSurfaceGreenRepresentation3D(StrictModule, NonTrainableState):
    """Prepared outgoing free-surface Green representation for submerged points.

    The normalization is 1/(4πr), the time convention is exp(-iωt), z is up,
    and the radiation pole is interpreted as k-k₀-i0. The representation is
    an exact fixed-size JAX action for its declared radial/angular quadrature;
    it is not a continuum certificate.
    """

    radial_nodes: Array
    radial_weights: Array
    horizontal_directions: Array
    dispersion: FiniteDepthDispersionRoot3D
    resources: FreeSurfaceGreenResourceEvidence3D
    errors: FreeSurfaceGreenErrorEvidence3D
    angular_frequency: float = eqx.field(static=True)
    gravity: float = eqx.field(static=True)
    depth: float | None = eqx.field(static=True)
    free_surface_z: float = eqx.field(static=True)
    wavenumber: float = eqx.field(static=True)
    pole_log_integral: float = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)
    pde_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    time_convention: str = eqx.field(static=True)
    normal_convention: str = eqx.field(static=True)
    density_semantics: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)

    def _validate_point(self, value: ArrayLike, name: str, /) -> Array:
        point = jnp.asarray(value, dtype=jnp.float64)
        if point.shape != (3,):
            raise ValueError(f"{name} must have shape (3,); got {point.shape}.")
        relative_z = point[2] - self.free_surface_z
        point = eqx.error_if(
            point,
            ~jnp.all(jnp.isfinite(point)) | (relative_z >= 0.0),
            f"{name} must be finite and strictly below the free surface.",
        )
        if self.depth is not None:
            point = eqx.error_if(
                point,
                relative_z <= -self.depth,
                f"{name} must be strictly above the horizontal bottom.",
            )
        return point

    def _finite_vertical(
        self, k: Array, target_z: Array, source_z: Array, /
    ) -> tuple[Array, Array]:
        depth = self.depth
        assert depth is not None
        nu = self.angular_frequency * self.angular_frequency / self.gravity
        lower = jnp.minimum(target_z, source_z)
        upper = jnp.maximum(target_z, source_z)
        y1 = jnp.cosh(k * (lower + depth))
        y2 = jnp.cosh(k * upper) + (nu / k) * jnp.sinh(k * upper)
        denominator = k * jnp.sinh(k * depth) - nu * jnp.cosh(k * depth)
        vertical = 2.0 * k * y1 * y2 / denominator
        target_is_lower = target_z <= source_z
        dy1 = k * jnp.sinh(k * (target_z + depth))
        dy2 = k * jnp.sinh(k * target_z) + nu * jnp.cosh(k * target_z)
        vertical_gradient = jnp.where(
            target_is_lower,
            2.0
            * k
            * dy1
            * (jnp.cosh(k * source_z) + (nu / k) * jnp.sinh(k * source_z))
            / denominator,
            2.0 * k * jnp.cosh(k * (source_z + depth)) * dy2 / denominator,
        )
        direct = jnp.exp(-k * jnp.abs(target_z - source_z))
        direct_gradient = -k * jnp.sign(target_z - source_z) * direct
        return vertical - direct, vertical_gradient - direct_gradient

    def _pole_vertical(self, target_z: Array, source_z: Array, /) -> tuple[Array, Array]:
        root = self.wavenumber
        nu = self.angular_frequency * self.angular_frequency / self.gravity
        if self.depth is None:
            value = 2.0 * root * jnp.exp(root * (target_z + source_z))
            return value, root * value
        depth = self.depth
        lower = jnp.minimum(target_z, source_z)
        upper = jnp.maximum(target_z, source_z)
        y1 = jnp.cosh(root * (lower + depth))
        y2 = jnp.cosh(root * upper) + (nu / root) * jnp.sinh(root * upper)
        denominator_derivative = (
            jnp.sinh(root * depth)
            + root * depth * jnp.cosh(root * depth)
            - nu * depth * jnp.sinh(root * depth)
        )
        value = 2.0 * root * y1 * y2 / denominator_derivative
        target_is_lower = target_z <= source_z
        dy1 = root * jnp.sinh(root * (target_z + depth))
        dy2 = root * jnp.sinh(root * target_z) + nu * jnp.cosh(root * target_z)
        gradient = jnp.where(
            target_is_lower,
            2.0
            * root
            * dy1
            * (jnp.cosh(root * source_z) + (nu / root) * jnp.sinh(root * source_z))
            / denominator_derivative,
            2.0
            * root
            * jnp.cosh(root * (source_z + depth))
            * dy2
            / denominator_derivative,
        )
        return value, gradient

    def _wave_terms(self, target: Array, source: Array, /) -> tuple[Array, Array]:
        k = self.radial_nodes
        root = self.wavenumber
        target_z = target[2] - self.free_surface_z
        source_z = source[2] - self.free_surface_z
        horizontal = target[:2] - source[:2]
        projections = self.horizontal_directions @ horizontal
        phase = jnp.exp(1j * k[:, None] * projections[None, :])
        phase_root = jnp.exp(1j * root * projections)
        nu = self.angular_frequency * self.angular_frequency / self.gravity
        if self.depth is None:
            amplitude = (k + nu) * jnp.exp(k * (target_z + source_z)) / (k - root)
            amplitude_gradient = k * amplitude
        else:
            amplitude, amplitude_gradient = self._finite_vertical(k, target_z, source_z)
        pole, pole_gradient = self._pole_vertical(target_z, source_z)
        denominator = k - root
        regular = phase * amplitude[:, None] - (
            phase_root[None, :] * pole / denominator[:, None]
        )
        radial_value = jnp.sum(self.radial_weights * jnp.mean(regular, axis=1))
        pole_value = pole * jnp.mean(phase_root)
        value = (radial_value + pole_value * (self.pole_log_integral + 1j * jnp.pi)) / (
            4.0 * jnp.pi
        )

        phase_gradient = (
            1j
            * k[:, None, None]
            * self.horizontal_directions[None, :, :]
            * phase[:, :, None]
        )
        phase_root_gradient = 1j * root * self.horizontal_directions * phase_root[:, None]
        regular_horizontal = phase_gradient * amplitude[:, None, None] - (
            phase_root_gradient[None, :, :] * pole / denominator[:, None, None]
        )
        radial_horizontal = jnp.sum(
            self.radial_weights[:, None] * jnp.mean(regular_horizontal, axis=1),
            axis=0,
        )
        pole_horizontal = pole * jnp.mean(phase_root_gradient, axis=0)
        horizontal_gradient = (
            radial_horizontal + pole_horizontal * (self.pole_log_integral + 1j * jnp.pi)
        ) / (4.0 * jnp.pi)

        regular_vertical = phase * amplitude_gradient[:, None] - (
            phase_root[None, :] * pole_gradient / denominator[:, None]
        )
        radial_vertical = jnp.sum(
            self.radial_weights * jnp.mean(regular_vertical, axis=1)
        )
        pole_vertical = pole_gradient * jnp.mean(phase_root)
        vertical_gradient = (
            radial_vertical + pole_vertical * (self.pole_log_integral + 1j * jnp.pi)
        ) / (4.0 * jnp.pi)
        return value, jnp.concatenate(
            (horizontal_gradient, jnp.asarray((vertical_gradient,)))
        )

    def wave_correction(self, target: ArrayLike, source: ArrayLike, /) -> Array:
        """Evaluate the nonsingular free-surface/bottom correction only."""
        target_ = self._validate_point(target, "target")
        source_ = self._validate_point(source, "source")
        return self._wave_terms(target_, source_)[0]

    def wave_correction_target_gradient(
        self, target: ArrayLike, source: ArrayLike, /
    ) -> Array:
        """Differentiate the fixed correction with respect to target coordinates."""
        target_ = self._validate_point(target, "target")
        source_ = self._validate_point(source, "source")
        return self._wave_terms(target_, source_)[1]

    def value(self, target: ArrayLike, source: ArrayLike, /) -> Array:
        """Evaluate the complete normalized outgoing Green representation."""
        target_ = self._validate_point(target, "target")
        source_ = self._validate_point(source, "source")
        difference = target_ - source_
        radius = jnp.linalg.norm(difference)
        target_ = eqx.error_if(
            target_, radius == 0.0, "Green target and source must be distinct."
        )
        correction = self._wave_terms(target_, source_)[0]
        return 1.0 / (4.0 * jnp.pi * radius) + correction

    def target_gradient(self, target: ArrayLike, source: ArrayLike, /) -> Array:
        """Differentiate the complete representation with respect to the target."""
        target_ = self._validate_point(target, "target")
        source_ = self._validate_point(source, "source")
        difference = target_ - source_
        radius = jnp.linalg.norm(difference)
        target_ = eqx.error_if(
            target_, radius == 0.0, "Green target and source must be distinct."
        )
        direct = -difference / (4.0 * jnp.pi * radius**3)
        return direct + self._wave_terms(target_, source_)[1]


def prepare_free_surface_green_3d(
    angular_frequency: float,
    gravity: float,
    /,
    *,
    minimum_clearance: float,
    depth: float | None = None,
    free_surface_z: float = 0.0,
    frame_id: str = "z-up-cartesian",
    unit_system_id: str = "si",
    policy: FreeSurfaceGreenPolicy3D | None = None,
) -> FreeSurfaceGreenRepresentation3D:
    """Prepare a bounded infinite- or finite-depth outgoing Green representation."""
    selected = FreeSurfaceGreenPolicy3D() if policy is None else policy
    if not isinstance(selected, FreeSurfaceGreenPolicy3D):
        raise TypeError("policy must be FreeSurfaceGreenPolicy3D or None.")
    omega = float(angular_frequency)
    gravity_ = float(gravity)
    clearance = float(minimum_clearance)
    surface = float(free_surface_z)
    depth_ = None if depth is None else float(depth)
    if any(
        not math.isfinite(value) or value <= 0.0 for value in (omega, gravity_, clearance)
    ) or not math.isfinite(surface):
        raise ValueError(
            "angular_frequency, gravity, and minimum_clearance must be finite and positive."
        )
    if depth_ is not None and (not math.isfinite(depth_) or depth_ <= 0.0):
        raise ValueError("depth must be finite and positive when provided.")
    if depth_ is not None and clearance >= 0.5 * depth_:
        raise ValueError(
            "minimum_clearance must be less than half the finite water depth."
        )
    frame = _nonempty(frame_id, "frame_id")
    units = _nonempty(unit_system_id, "unit_system_id")
    if depth_ is None:
        root_value = omega * omega / gravity_
        dispersion = FiniteDepthDispersionRoot3D(
            wavenumber=jnp.asarray(root_value, dtype=jnp.float64),
            residual=jnp.asarray(0.0, dtype=jnp.float64),
            bracket=jnp.asarray((root_value, root_value), dtype=jnp.float64),
            converged=jnp.asarray(True),
            iterations=0,
            ambient_dimension=3,
            coordinate_convention="right-handed-cartesian-z-up",
            pde_id=_PDE_ID,
            geometry_id=f"horizontal-free-surface:z={surface:.17g}:infinite-depth",
            formulation_id="infinite-depth-dispersion-identity",
            provider_id="phydrax-analytic",
            precision_id="float64",
            frame_id=frame,
            unit_system_id=units,
            non_goals=("capillary dispersion", "current-modified dispersion"),
        )
    else:
        dispersion = solve_finite_depth_dispersion_3d(
            omega,
            gravity_,
            depth_,
            tolerance=selected.root_tolerance,
            max_iterations=selected.max_root_iterations,
            frame_id=frame,
            unit_system_id=units,
        )
        root_value = float(np.asarray(dispersion.wavenumber))
        if not bool(np.asarray(dispersion.converged)):
            raise ValueError("Finite-depth dispersion root did not converge.")

    cutoff = max(
        selected.cutoff_clearance_factor / clearance,
        selected.minimum_cutoff_root_ratio * root_value,
    )
    if cutoff > selected.maximum_wavenumber:
        raise ValueError(
            "Required Green wavenumber cutoff exceeds maximum_wavenumber; "
            "relax the declared clearance/error policy explicitly."
        )
    if depth_ is not None and cutoff * depth_ >= 300.0:
        raise ValueError(
            "Finite-depth modal factors exceed the qualified float64 hyperbolic range."
        )

    nodes, weights = np.polynomial.legendre.leggauss(selected.radial_order_per_interval)
    intervals = ((0.0, root_value), (root_value, cutoff))
    radial_nodes = []
    radial_weights = []
    for left, right in intervals:
        radial_nodes.append(0.5 * (right - left) * nodes + 0.5 * (right + left))
        radial_weights.append(0.5 * (right - left) * weights)
    radial_nodes_array = np.concatenate(radial_nodes)
    radial_weights_array = np.concatenate(radial_weights)
    angles = 2.0 * np.pi * np.arange(selected.angular_order) / selected.angular_order
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    resident_bytes = int(
        radial_nodes_array.nbytes + radial_weights_array.nbytes + directions.nbytes
    )
    workspace_bytes = int(
        radial_nodes_array.size
        * selected.angular_order
        * (np.dtype(np.complex128).itemsize * 10 + np.dtype(np.float64).itemsize * 4)
    )
    within_policy = resident_bytes <= selected.max_resident_bytes
    if not within_policy:
        raise ValueError("Prepared Green arrays exceed max_resident_bytes.")
    tail_bound = 8.0 * math.exp(-2.0 * clearance * cutoff) / (4.0 * math.pi * clearance)
    tail_supported = tail_bound <= selected.maximum_spectral_tail_bound
    if not tail_supported:
        raise ValueError(
            "Green spectral tail envelope exceeds maximum_spectral_tail_bound."
        )
    resources = FreeSurfaceGreenResourceEvidence3D(
        radial_node_count=int(radial_nodes_array.size),
        angular_node_count=selected.angular_order,
        resident_bytes=resident_bytes,
        action_workspace_bytes=workspace_bytes,
        maximum_resident_bytes=selected.max_resident_bytes,
        within_policy=jnp.asarray(within_policy),
    )
    errors = FreeSurfaceGreenErrorEvidence3D(
        dispersion_residual=dispersion.residual,
        dispersion_tolerance=selected.root_tolerance,
        wavenumber_cutoff=cutoff,
        minimum_clearance=clearance,
        spectral_tail_envelope_bound=jnp.asarray(tail_bound, dtype=jnp.float64),
        maximum_spectral_tail_bound=selected.maximum_spectral_tail_bound,
        radial_order_per_interval=selected.radial_order_per_interval,
        angular_order=selected.angular_order,
        quadrature_convergence_estimated=False,
        continuum_discretization_error_estimated=False,
        supported=(
            dispersion.converged
            & jnp.asarray(within_policy)
            & jnp.asarray(tail_supported)
        ),
    )
    geometry_id = (
        f"horizontal-free-surface:z={surface:.17g}:infinite-depth"
        if depth_ is None
        else f"horizontal-free-surface:z={surface:.17g}:bottom={surface - depth_:.17g}"
    )
    representation_id = canonical_fingerprint(
        {
            "kind": "free-surface-green-representation-3d",
            "omega": omega,
            "gravity": gravity_,
            "depth": depth_,
            "surface": surface,
            "frame": frame,
            "units": units,
            "policy": selected.policy_id,
            "radial_nodes": array_tree_fingerprint(radial_nodes_array),
            "directions": array_tree_fingerprint(directions),
        }
    )
    return FreeSurfaceGreenRepresentation3D(
        radial_nodes=jnp.asarray(radial_nodes_array, dtype=jnp.float64),
        radial_weights=jnp.asarray(radial_weights_array, dtype=jnp.float64),
        horizontal_directions=jnp.asarray(directions, dtype=jnp.float64),
        dispersion=dispersion,
        resources=resources,
        errors=errors,
        angular_frequency=omega,
        gravity=gravity_,
        depth=depth_,
        free_surface_z=surface,
        wavenumber=root_value,
        pole_log_integral=math.log((cutoff - root_value) / root_value),
        ambient_dimension=3,
        coordinate_convention="right-handed-cartesian-z-up",
        pde_id=_PDE_ID,
        geometry_id=geometry_id,
        formulation_id=_FORMULATION_ID,
        provider_id=_PROVIDER_ID,
        precision_id=_PRECISION_ID,
        frame_id=frame,
        unit_system_id=units,
        time_convention=_TIME_CONVENTION,
        normal_convention="body-to-fluid",
        density_semantics=(
            "piecewise-constant single-layer source strength; coefficients are "
            "not area weighted and are not the prescribed body-normal velocity"
        ),
        non_goals=_NON_GOALS,
        representation_id=representation_id,
    )


__all__ = [
    "FiniteDepthDispersionRoot3D",
    "FreeSurfaceGreenErrorEvidence3D",
    "FreeSurfaceGreenPolicy3D",
    "FreeSurfaceGreenRepresentation3D",
    "FreeSurfaceGreenResourceEvidence3D",
    "prepare_free_surface_green_3d",
    "solve_finite_depth_dispersion_3d",
]
