#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ..._model import AbstractArrayModel
from ..._sampling import materialize_design, SobolDesign
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry import regularized_delta_values, regularized_heaviside_values
from ...sampling.collocation import CausalTimeSlabSchedule


StefanRepresentation = Literal["explicit_front", "implicit_level_set", "reference_map"]


class OnePhaseStefanParameters(StrictModule, NonTrainableState):
    """Physical and ambient-domain parameters for a one-phase Stefan problem."""

    diffusivity: Array
    conductivity: Array
    volumetric_latent_heat: Array
    melting_temperature: Array
    initial_front: Array
    domain_length: Array
    final_time: Array
    jacobian_floor: Array

    def __init__(
        self,
        *,
        diffusivity: ArrayLike,
        conductivity: ArrayLike,
        volumetric_latent_heat: ArrayLike,
        melting_temperature: ArrayLike,
        initial_front: ArrayLike,
        domain_length: ArrayLike,
        final_time: ArrayLike,
        jacobian_floor: ArrayLike = 1.0e-6,
    ):
        self.diffusivity = _positive_scalar(diffusivity, "diffusivity")
        self.conductivity = _positive_scalar(conductivity, "conductivity")
        self.volumetric_latent_heat = _positive_scalar(
            volumetric_latent_heat,
            "volumetric_latent_heat",
        )
        self.melting_temperature = _finite_scalar(
            melting_temperature,
            "melting_temperature",
        )
        self.initial_front = _positive_scalar(initial_front, "initial_front")
        self.domain_length = _positive_scalar(domain_length, "domain_length")
        self.final_time = _positive_scalar(final_time, "final_time")
        self.jacobian_floor = _positive_scalar(jacobian_floor, "jacobian_floor")
        if float(self.initial_front) >= float(self.domain_length):
            raise ValueError("initial_front must lie strictly inside domain_length.")


class StefanBoundaryData(StrictModule, NonTrainableState):
    """Initial and fixed-boundary temperature functions."""

    initial_temperature: Callable[[Array], Array]
    boundary_temperature: Callable[[Array], Array]

    def __init__(
        self,
        initial_temperature: Callable[[Array], Array],
        boundary_temperature: Callable[[Array], Array],
        /,
    ):
        if not callable(initial_temperature) or not callable(boundary_temperature):
            raise TypeError("Stefan temperature data must be callable.")
        self.initial_temperature = initial_temperature
        self.boundary_temperature = boundary_temperature


class StefanCollocationBatch(StrictModule, NonTrainableState):
    """Shared unit-coordinate and ambient-coordinate collocation support."""

    interior_reference: Array
    ambient_points: Array
    boundary_times: Array
    interface_times: Array
    initial_reference: Array

    def __init__(
        self,
        *,
        interior_reference: ArrayLike,
        ambient_points: ArrayLike,
        boundary_times: ArrayLike,
        interface_times: ArrayLike,
        initial_reference: ArrayLike,
    ):
        interior = jnp.asarray(interior_reference, dtype=float)
        ambient = jnp.asarray(ambient_points, dtype=float)
        boundary = jnp.asarray(boundary_times, dtype=float).reshape((-1,))
        interface = jnp.asarray(interface_times, dtype=float).reshape((-1,))
        initial = jnp.asarray(initial_reference, dtype=float).reshape((-1,))
        if interior.ndim != 2 or interior.shape[-1] != 2:
            raise ValueError("interior_reference must have shape (point, 2).")
        if ambient.ndim != 2 or ambient.shape[-1] != 2:
            raise ValueError("ambient_points must have shape (point, 2).")
        if (
            min(
                interior.shape[0],
                ambient.shape[0],
                boundary.size,
                interface.size,
                initial.size,
            )
            <= 0
        ):
            raise ValueError("Every Stefan collocation block must be non-empty.")
        for values, name in (
            (interior, "interior_reference"),
            (ambient, "ambient_points"),
            (boundary, "boundary_times"),
            (interface, "interface_times"),
            (initial, "initial_reference"),
        ):
            if not bool(jnp.all(jnp.isfinite(values))):
                raise ValueError(f"{name} must be finite.")
        self.interior_reference = interior
        self.ambient_points = ambient
        self.boundary_times = boundary
        self.interface_times = interface
        self.initial_reference = initial


class ExplicitFrontStefanPINN(StrictModule):
    temperature: Callable[[Array], Array]
    front: Callable[[Array], Array]

    def __init__(
        self, temperature: Callable[[Array], Array], front: Callable[[Array], Array], /
    ):
        if not callable(temperature) or not callable(front):
            raise TypeError("Explicit Stefan fields must be callable.")
        self.temperature = temperature
        self.front = front


class ImplicitLevelSetStefanPINN(StrictModule):
    temperature: Callable[[Array], Array]
    level_set: Callable[[Array], Array]

    def __init__(
        self,
        temperature: Callable[[Array], Array],
        level_set: Callable[[Array], Array],
        /,
    ):
        if not callable(temperature) or not callable(level_set):
            raise TypeError("Implicit Stefan fields must be callable.")
        self.temperature = temperature
        self.level_set = level_set


class ReferenceMapStefanPINN(StrictModule):
    reference_temperature: Callable[[Array], Array]
    coordinate_map: Callable[[Array], Array]

    def __init__(
        self,
        reference_temperature: Callable[[Array], Array],
        coordinate_map: Callable[[Array], Array],
        /,
    ):
        if not callable(reference_temperature) or not callable(coordinate_map):
            raise TypeError("Reference-map Stefan fields must be callable.")
        self.reference_temperature = reference_temperature
        self.coordinate_map = coordinate_map


class StefanLoss(StrictModule):
    total: Array
    pde: Array
    initial: Array
    fixed_boundary: Array
    interface_temperature: Array
    stefan_balance: Array
    geometry: Array

    def __init__(
        self,
        *,
        pde: ArrayLike,
        initial: ArrayLike,
        fixed_boundary: ArrayLike,
        interface_temperature: ArrayLike,
        stefan_balance: ArrayLike,
        geometry: ArrayLike = 0.0,
    ):
        self.pde = jnp.asarray(pde).reshape(())
        self.initial = jnp.asarray(initial).reshape(())
        self.fixed_boundary = jnp.asarray(fixed_boundary).reshape(())
        self.interface_temperature = jnp.asarray(interface_temperature).reshape(())
        self.stefan_balance = jnp.asarray(stefan_balance).reshape(())
        self.geometry = jnp.asarray(geometry).reshape(())
        self.total = (
            self.pde
            + self.initial
            + self.fixed_boundary
            + self.interface_temperature
            + self.stefan_balance
            + self.geometry
        )


class StefanFitResult(StrictModule):
    model: Any
    loss_history: Array
    final_loss: StefanLoss


class StefanRepresentationComparison(StrictModule, NonTrainableState):
    explicit: StefanLoss
    implicit: StefanLoss
    reference: StefanLoss
    best_representation: str

    def __init__(
        self, explicit: StefanLoss, implicit: StefanLoss, reference: StefanLoss, /
    ):
        losses = jnp.asarray((explicit.total, implicit.total, reference.total))
        names = ("explicit_front", "implicit_level_set", "reference_map")
        self.explicit = explicit
        self.implicit = implicit
        self.reference = reference
        self.best_representation = names[int(jnp.argmin(losses))]


def stefan_collocation_batch(
    parameters: OnePhaseStefanParameters,
    /,
    *,
    interior_points: int,
    ambient_points: int,
    boundary_points: int,
    interface_points: int,
    initial_points: int,
    key: Key[Array, ""] = DOC_KEY0,
    final_time: float | None = None,
) -> StefanCollocationBatch:
    """Materialize reproducible scrambled-Sobol Stefan collocation blocks."""

    if not isinstance(parameters, OnePhaseStefanParameters):
        raise TypeError("parameters must be OnePhaseStefanParameters.")
    counts = tuple(
        int(value)
        for value in (
            interior_points,
            ambient_points,
            boundary_points,
            interface_points,
            initial_points,
        )
    )
    if any(value <= 0 for value in counts):
        raise ValueError("Stefan collocation counts must be positive.")
    terminal = float(parameters.final_time) if final_time is None else float(final_time)
    if not math.isfinite(terminal) or not 0.0 < terminal <= float(parameters.final_time):
        raise ValueError("final_time must lie in (0, parameters.final_time].")
    keys = jax.random.split(key, 5)
    design = SobolDesign(scrambled=True)
    interior = materialize_design(
        design,
        count=counts[0],
        dimension=2,
        key=keys[0],
    )
    ambient = materialize_design(
        design,
        count=counts[1],
        dimension=2,
        key=keys[1],
    )
    boundary = materialize_design(
        design,
        count=counts[2],
        dimension=1,
        key=keys[2],
    )[:, 0]
    interface = materialize_design(
        design,
        count=counts[3],
        dimension=1,
        key=keys[3],
    )[:, 0]
    initial = materialize_design(
        design,
        count=counts[4],
        dimension=1,
        key=keys[4],
    )[:, 0]
    return StefanCollocationBatch(
        interior_reference=interior.at[:, 1].multiply(terminal),
        ambient_points=ambient * jnp.asarray((float(parameters.domain_length), terminal)),
        boundary_times=terminal * boundary,
        interface_times=terminal * interface,
        initial_reference=initial,
    )


def explicit_front_stefan_loss(
    model: ExplicitFrontStefanPINN,
    batch: StefanCollocationBatch,
    parameters: OnePhaseStefanParameters,
    data: StefanBoundaryData,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> StefanLoss:
    """Evaluate a one-phase Stefan PINN with a separately learned front."""

    temperature = lambda point: _scalar_call(model.temperature, point, key)
    front = lambda time: _scalar_call(model.front, jnp.asarray((time,)), key)

    def pde_one(reference_point):
        coordinate, time = reference_point
        physical = jnp.asarray((coordinate * front(time), time))
        gradient = jax.grad(temperature)(physical)
        hessian = jax.hessian(temperature)(physical)
        return gradient[1] - parameters.diffusivity * hessian[0, 0]

    pde = jnp.mean(jax.vmap(lambda point: pde_one(point) ** 2)(batch.interior_reference))

    initial_x = batch.initial_reference * parameters.initial_front
    initial_points = jnp.stack((initial_x, jnp.zeros_like(initial_x)), axis=-1)
    initial_prediction = jax.vmap(temperature)(initial_points)
    initial_target = jax.vmap(data.initial_temperature)(initial_x)
    initial = jnp.mean((initial_prediction - initial_target) ** 2)

    boundary_points = jnp.stack(
        (jnp.zeros_like(batch.boundary_times), batch.boundary_times),
        axis=-1,
    )
    boundary_prediction = jax.vmap(temperature)(boundary_points)
    boundary_target = jax.vmap(data.boundary_temperature)(batch.boundary_times)
    fixed_boundary = jnp.mean((boundary_prediction - boundary_target) ** 2)

    def interface_one(time):
        position = front(time)
        point = jnp.asarray((position, time))
        value = temperature(point)
        gradient = jax.grad(temperature)(point)[0]
        speed = jax.grad(front)(time)
        return (
            value,
            parameters.volumetric_latent_heat * speed
            + parameters.conductivity * gradient,
        )

    interface_value, balance = jax.vmap(interface_one)(batch.interface_times)
    interface_temperature = jnp.mean(
        (interface_value - parameters.melting_temperature) ** 2
    )
    stefan_balance = jnp.mean(balance**2)
    front_initial = (front(jnp.asarray(0.0)) - parameters.initial_front) ** 2
    geometry = front_initial + jnp.mean(
        jax.nn.relu(-jax.vmap(front)(batch.interface_times)) ** 2
        + jax.nn.relu(jax.vmap(front)(batch.interface_times) - parameters.domain_length)
        ** 2
    )
    return StefanLoss(
        pde=pde,
        initial=initial,
        fixed_boundary=fixed_boundary,
        interface_temperature=interface_temperature,
        stefan_balance=stefan_balance,
        geometry=geometry,
    )


def implicit_level_set_stefan_loss(
    model: ImplicitLevelSetStefanPINN,
    batch: StefanCollocationBatch,
    parameters: OnePhaseStefanParameters,
    data: StefanBoundaryData,
    /,
    *,
    interface_width: float,
    gradient_floor: float = 1.0e-12,
    key: Key[Array, ""] = DOC_KEY0,
) -> StefanLoss:
    """Evaluate a fixed-ambient-domain Stefan PINN with a learned level set."""

    width = _positive_float(interface_width, "interface_width")
    floor = _positive_float(gradient_floor, "gradient_floor")
    temperature = lambda point: _scalar_call(model.temperature, point, key)
    level_set = lambda point: _scalar_call(model.level_set, point, key)

    def ambient_one(point):
        value = level_set(point)
        indicator = regularized_heaviside_values(-value, width=width)
        delta = regularized_delta_values(value, width=width)
        temperature_gradient = jax.grad(temperature)(point)
        temperature_hessian = jax.hessian(temperature)(point)
        pde_residual = (
            temperature_gradient[1] - parameters.diffusivity * temperature_hessian[0, 0]
        )
        phi_gradient = jax.grad(level_set)(point)
        phi_x = phi_gradient[0]
        magnitude = jnp.maximum(jnp.abs(phi_x), floor)
        normal = phi_x / magnitude
        speed = -phi_gradient[1] / magnitude
        interface_temperature = temperature(point) - parameters.melting_temperature
        balance = (
            parameters.volumetric_latent_heat * speed
            + parameters.conductivity * temperature_gradient[0] * normal
        )
        eikonal = magnitude - 1.0
        coarea = delta * magnitude
        return (
            indicator * pde_residual**2,
            coarea * interface_temperature**2,
            coarea * balance**2,
            width * delta * eikonal**2,
            indicator,
            coarea,
        )

    values = jax.vmap(ambient_one)(batch.ambient_points)
    pde = jnp.sum(values[0]) / jnp.maximum(jnp.sum(values[4]), 1.0e-12)
    interface_temperature = jnp.sum(values[1]) / jnp.maximum(jnp.sum(values[5]), 1.0e-12)
    stefan_balance = jnp.sum(values[2]) / jnp.maximum(jnp.sum(values[5]), 1.0e-12)
    eikonal = jnp.mean(values[3])

    initial_x = batch.initial_reference * parameters.initial_front
    initial_points = jnp.stack((initial_x, jnp.zeros_like(initial_x)), axis=-1)
    initial_temperature = jax.vmap(temperature)(initial_points)
    initial_target = jax.vmap(data.initial_temperature)(initial_x)
    initial_phi = jax.vmap(level_set)(initial_points)
    initial = jnp.mean((initial_temperature - initial_target) ** 2) + jnp.mean(
        (initial_phi - (initial_x - parameters.initial_front)) ** 2
    )

    boundary_points = jnp.stack(
        (jnp.zeros_like(batch.boundary_times), batch.boundary_times),
        axis=-1,
    )
    boundary_prediction = jax.vmap(temperature)(boundary_points)
    boundary_target = jax.vmap(data.boundary_temperature)(batch.boundary_times)
    fixed_boundary = jnp.mean((boundary_prediction - boundary_target) ** 2)
    return StefanLoss(
        pde=pde,
        initial=initial,
        fixed_boundary=fixed_boundary,
        interface_temperature=interface_temperature,
        stefan_balance=stefan_balance,
        geometry=eikonal,
    )


def reference_map_stefan_loss(
    model: ReferenceMapStefanPINN,
    batch: StefanCollocationBatch,
    parameters: OnePhaseStefanParameters,
    data: StefanBoundaryData,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> StefanLoss:
    """Evaluate a Stefan PINN pulled back to a learned reference-domain map."""

    temperature = lambda point: _scalar_call(model.reference_temperature, point, key)
    coordinate_map = lambda point: _scalar_call(model.coordinate_map, point, key)

    def pde_one(point):
        map_gradient = jax.grad(coordinate_map)(point)
        jacobian = map_gradient[0]
        safe_jacobian = jnp.maximum(jacobian, parameters.jacobian_floor)
        value_gradient = jax.grad(temperature)(point)
        physical_gradient_xi = jax.grad(
            lambda coordinate: (
                jax.grad(temperature)(coordinate)[0]
                / jnp.maximum(
                    jax.grad(coordinate_map)(coordinate)[0], parameters.jacobian_floor
                )
            )
        )(point)[0]
        physical_laplacian = physical_gradient_xi / safe_jacobian
        physical_time = (
            value_gradient[1] - (map_gradient[1] / safe_jacobian) * value_gradient[0]
        )
        residual = physical_time - parameters.diffusivity * physical_laplacian
        jacobian_defect = jax.nn.relu(parameters.jacobian_floor - jacobian)
        return residual**2, jacobian_defect**2

    pde_values, jacobian_defects = jax.vmap(pde_one)(batch.interior_reference)
    pde = jnp.mean(pde_values)

    initial_points = jnp.stack(
        (batch.initial_reference, jnp.zeros_like(batch.initial_reference)), axis=-1
    )
    initial_prediction = jax.vmap(temperature)(initial_points)
    initial_map = jax.vmap(coordinate_map)(initial_points)
    initial_target = jax.vmap(data.initial_temperature)(
        batch.initial_reference * parameters.initial_front
    )
    initial = jnp.mean((initial_prediction - initial_target) ** 2) + jnp.mean(
        (initial_map - parameters.initial_front * batch.initial_reference) ** 2
    )

    boundary_points = jnp.stack(
        (jnp.zeros_like(batch.boundary_times), batch.boundary_times), axis=-1
    )
    boundary_prediction = jax.vmap(temperature)(boundary_points)
    boundary_target = jax.vmap(data.boundary_temperature)(batch.boundary_times)
    boundary_map = jax.vmap(coordinate_map)(boundary_points)
    fixed_boundary = jnp.mean((boundary_prediction - boundary_target) ** 2) + jnp.mean(
        boundary_map**2
    )

    def interface_one(time):
        point = jnp.asarray((1.0, time))
        value = temperature(point)
        map_gradient = jax.grad(coordinate_map)(point)
        jacobian = jnp.maximum(map_gradient[0], parameters.jacobian_floor)
        physical_gradient = jax.grad(temperature)(point)[0] / jacobian
        speed = map_gradient[1]
        balance = (
            parameters.volumetric_latent_heat * speed
            + parameters.conductivity * physical_gradient
        )
        return value, balance

    interface_value, balance = jax.vmap(interface_one)(batch.interface_times)
    interface_temperature = jnp.mean(
        (interface_value - parameters.melting_temperature) ** 2
    )
    stefan_balance = jnp.mean(balance**2)
    geometry = (
        jnp.mean(jacobian_defects)
        + (coordinate_map(jnp.asarray((1.0, 0.0))) - parameters.initial_front) ** 2
    )
    return StefanLoss(
        pde=pde,
        initial=initial,
        fixed_boundary=fixed_boundary,
        interface_temperature=interface_temperature,
        stefan_balance=stefan_balance,
        geometry=geometry,
    )


def compare_stefan_representations(
    explicit: ExplicitFrontStefanPINN,
    implicit: ImplicitLevelSetStefanPINN,
    reference: ReferenceMapStefanPINN,
    batch: StefanCollocationBatch,
    parameters: OnePhaseStefanParameters,
    data: StefanBoundaryData,
    /,
    *,
    interface_width: float,
    key: Key[Array, ""] = DOC_KEY0,
) -> StefanRepresentationComparison:
    """Evaluate all three representations on identical physical collocation."""

    keys = jax.random.split(key, 3)
    return StefanRepresentationComparison(
        explicit_front_stefan_loss(explicit, batch, parameters, data, key=keys[0]),
        implicit_level_set_stefan_loss(
            implicit,
            batch,
            parameters,
            data,
            interface_width=interface_width,
            key=keys[1],
        ),
        reference_map_stefan_loss(reference, batch, parameters, data, key=keys[2]),
    )


def fit_stefan_pinn(
    model: Any,
    loss: Callable[[Any], StefanLoss],
    /,
    *,
    steps: int,
    optimizer: optax.GradientTransformation | None = None,
    jit: bool = True,
) -> StefanFitResult:
    """Optimize any Stefan representation against a typed ``StefanLoss``."""

    count = int(steps)
    if count < 0:
        raise ValueError("steps must be nonnegative.")
    transformation = optax.adam(1.0e-3) if optimizer is None else optimizer
    parameters, fixed = eqx.partition(model, eqx.is_inexact_array)
    state = transformation.init(parameters)

    def step(trainable, optimizer_state):
        def objective(current):
            evaluated = eqx.combine(current, fixed)
            result = loss(evaluated)
            if not isinstance(result, StefanLoss):
                raise TypeError("Stefan loss callback must return StefanLoss.")
            return result.total

        value, gradient = eqx.filter_value_and_grad(objective)(trainable)
        updates, next_state = transformation.update(gradient, optimizer_state, trainable)
        return optax.apply_updates(trainable, updates), next_state, value

    run_step = eqx.filter_jit(step) if jit else step
    history = []
    for _ in range(count):
        parameters, state, value = run_step(parameters, state)
        history.append(value)
    fitted = eqx.combine(parameters, fixed)
    final = loss(fitted)
    return StefanFitResult(
        model=fitted,
        loss_history=(jnp.stack(history) if history else jnp.zeros((0,))),
        final_loss=final,
    )


def fit_stefan_time_slabs(
    model: Any,
    schedule: CausalTimeSlabSchedule,
    build_loss: Callable[[Any, int, float], StefanLoss],
    /,
    *,
    steps_per_slab: int,
    optimizer: optax.GradientTransformation | None = None,
    jit: bool = True,
) -> StefanFitResult:
    """Train causally over increasing time slabs while carrying model parameters."""

    if not isinstance(schedule, CausalTimeSlabSchedule):
        raise TypeError("schedule must be a CausalTimeSlabSchedule.")
    current = model
    histories = []
    final = None
    for index in range(schedule.slab_count):
        _, upper = schedule.bounds(index)
        result = fit_stefan_pinn(
            current,
            lambda candidate, i=index, end=float(upper): build_loss(candidate, i, end),
            steps=steps_per_slab,
            optimizer=optimizer,
            jit=jit,
        )
        current = result.model
        histories.append(result.loss_history)
        final = result.final_loss
    assert final is not None
    return StefanFitResult(
        model=current,
        loss_history=jnp.concatenate(histories),
        final_loss=final,
    )


def _scalar_call(model: Callable[[Array], Array], point: Array, key, /) -> Array:
    value = (
        model(point, key=key) if isinstance(model, AbstractArrayModel) else model(point)
    )
    scalar = jnp.asarray(value)
    if scalar.shape != () or jnp.iscomplexobj(scalar):
        raise ValueError("Stefan fields must return one real scalar per coordinate.")
    return scalar


def _finite_scalar(value: ArrayLike, name: str, /) -> Array:
    scalar = jnp.asarray(value, dtype=float)
    if scalar.shape != () or not bool(jnp.isfinite(scalar)):
        raise ValueError(f"{name} must be one finite scalar.")
    return scalar


def _positive_scalar(value: ArrayLike, name: str, /) -> Array:
    scalar = _finite_scalar(value, name)
    if float(scalar) <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return scalar


def _positive_float(value: float, name: str, /) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return scalar


__all__ = [
    "ExplicitFrontStefanPINN",
    "ImplicitLevelSetStefanPINN",
    "OnePhaseStefanParameters",
    "ReferenceMapStefanPINN",
    "StefanBoundaryData",
    "StefanCollocationBatch",
    "StefanFitResult",
    "StefanLoss",
    "StefanRepresentation",
    "StefanRepresentationComparison",
    "compare_stefan_representations",
    "explicit_front_stefan_loss",
    "fit_stefan_pinn",
    "fit_stefan_time_slabs",
    "implicit_level_set_stefan_loss",
    "reference_map_stefan_loss",
    "stefan_collocation_batch",
]
