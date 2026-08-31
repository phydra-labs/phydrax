#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._strict import StrictModule


if TYPE_CHECKING:
    from ..optim import Bounds, NonlinearLeastSquaresProblem
    from ..uq import ParameterSpace, PosteriorProblem

from ._mna import full_mna_scattering_matrix, NodalCircuit, prepare_mna
from ._models import AbstractScatteringComponent
from ._network import (
    prepare_scattering_network,
    scattering_submatrix,
)
from ._ports import references_compatible, WaveReference
from ._topology import ScatteringNetwork


class ScatteringDataset(StrictModule):
    """Complex scattering observations with explicit selections and whitening."""

    angular_frequency: Array
    observed: Array
    references: tuple[WaveReference, ...]
    whitening: Array | None
    port_ids: tuple[str, ...] = eqx.field(static=True)
    input_ports: tuple[str, ...] = eqx.field(static=True)
    output_ports: tuple[str, ...] = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)

    def __init__(
        self,
        angular_frequency: ArrayLike,
        observed: ArrayLike,
        references: Sequence[WaveReference],
        /,
        *,
        port_ids: Sequence[str],
        input_ports: Sequence[str] | None = None,
        output_ports: Sequence[str] | None = None,
        whitening: ArrayLike | None = None,
        dataset_id: str = "scattering-dataset",
    ):
        omega = jnp.asarray(angular_frequency)
        values = jnp.asarray(observed).astype(jnp.result_type(observed, jnp.complex128))
        ports = tuple(str(value) for value in port_ids)
        refs = tuple(references)
        if (
            len(ports) != len(refs)
            or len(set(ports)) != len(ports)
            or any(not value for value in ports)
        ):
            raise ValueError(
                "port_ids and references must be unique, non-empty, and aligned."
            )
        inputs = (
            ports if input_ports is None else tuple(str(value) for value in input_ports)
        )
        outputs = (
            ports if output_ports is None else tuple(str(value) for value in output_ports)
        )
        if any(value not in ports for value in inputs + outputs):
            raise ValueError("Dataset selections must reference declared port_ids.")
        if values.shape != omega.shape + (len(outputs), len(inputs)):
            raise ValueError(
                "observed shape must be frequency_shape + (outputs, inputs)."
            )
        factor = None if whitening is None else jnp.asarray(whitening)
        residual_size = 2 * values.size
        if factor is not None and factor.shape != (residual_size, residual_size):
            raise ValueError(
                "whitening must be a real square factor over stacked real/imag residuals."
            )
        if factor is not None and not jnp.issubdtype(factor.dtype, jnp.floating):
            raise TypeError("whitening must be real-valued.")
        identifier = str(dataset_id)
        if not identifier:
            raise ValueError("dataset_id must be non-empty.")
        self.angular_frequency = omega
        self.observed = values
        self.references = refs
        self.whitening = factor
        self.port_ids = ports
        self.input_ports = inputs
        self.output_ports = outputs
        self.dataset_id = identifier


def complex_scattering_residual(
    predicted: ArrayLike,
    observed: ArrayLike,
    whitening: ArrayLike | None = None,
    /,
) -> Array:
    """Stack Cartesian real then imaginary residuals and apply an explicit factor."""
    predicted_ = jnp.asarray(predicted)
    observed_ = jnp.asarray(observed)
    if predicted_.shape != observed_.shape:
        raise ValueError("predicted and observed scattering shapes must match.")
    difference = (predicted_ - observed_).reshape((-1,))
    residual = jnp.concatenate((jnp.real(difference), jnp.imag(difference)))
    if whitening is None:
        return residual
    factor = jnp.asarray(whitening)
    if factor.shape != (residual.size, residual.size) or not jnp.issubdtype(
        factor.dtype, jnp.floating
    ):
        raise ValueError("whitening must be a real square factor matching the residual.")
    return factor @ residual


def _component_channels(component: AbstractScatteringComponent, /):
    identifiers: list[str] = []
    references: list[WaveReference] = []
    for port in component.ports:
        identifiers.extend(
            (port.port_id if port.size == 1 else f"{port.port_id}:{coordinate}")
            for coordinate in port.coordinate_ids
        )
        references.extend(port.references)
    return tuple(identifiers), tuple(references)


def _component_prediction(
    component: AbstractScatteringComponent, dataset: ScatteringDataset
) -> Array:
    response = component.evaluate(dataset.angular_frequency)
    available, model_references = _component_channels(component)
    if any(
        value not in available for value in dataset.input_ports + dataset.output_ports
    ):
        raise ValueError("Parameterized component does not expose the dataset ports.")
    for port_id, reference in zip(dataset.port_ids, dataset.references, strict=True):
        model_reference = model_references[available.index(port_id)]
        if not bool(references_compatible(model_reference, reference)):
            raise ValueError("Calibration requires explicit wave-reference alignment.")
    outputs = jnp.asarray(tuple(available.index(value) for value in dataset.output_ports))
    inputs = jnp.asarray(tuple(available.index(value) for value in dataset.input_ports))
    return response.matrix[..., outputs[:, None], inputs[None, :]]


def _prediction(model: Any, dataset: ScatteringDataset) -> Array:
    if isinstance(model, ScatteringNetwork):
        prepared = prepare_scattering_network(model, dataset.angular_frequency)
        for port_id, reference in zip(dataset.port_ids, dataset.references, strict=True):
            channel = prepared.plan.external_port_ids.index(port_id)
            model_reference = prepared.references[
                prepared.plan.external_channels[channel]
            ]
            if not bool(references_compatible(model_reference, reference)):
                raise ValueError(
                    "Calibration requires explicit wave-reference alignment."
                )
        return scattering_submatrix(prepared, dataset.input_ports, dataset.output_ports)
    if isinstance(model, NodalCircuit):
        prepared = prepare_mna(model, dataset.angular_frequency)
        available = tuple(port.port_id for port in model.ports)
        for port_id, reference in zip(dataset.port_ids, dataset.references, strict=True):
            model_reference = model.ports[available.index(port_id)].reference
            if not bool(references_compatible(model_reference, reference)):
                raise ValueError(
                    "Calibration requires explicit wave-reference alignment."
                )
        matrix = full_mna_scattering_matrix(prepared)
        outputs = jnp.asarray(
            tuple(available.index(value) for value in dataset.output_ports)
        )
        inputs = jnp.asarray(
            tuple(available.index(value) for value in dataset.input_ports)
        )
        return matrix[..., outputs[:, None], inputs[None, :]]
    if isinstance(model, AbstractScatteringComponent):
        return _component_prediction(model, dataset)
    raise TypeError(
        "parameterize must return a scattering component/network or NodalCircuit."
    )


class CalibrationResidualPlan(StrictModule):
    """Pure parameter binding into native circuit solves and Cartesian residuals."""

    datasets: tuple[ScatteringDataset, ...]
    parameterize: Callable[[PyTree[Any]], Any] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameterize: Callable[[PyTree[Any]], Any],
        datasets: Sequence[ScatteringDataset],
        /,
        *,
        problem_id: str = "scattering-calibration",
    ):
        values = tuple(datasets)
        if not callable(parameterize):
            raise TypeError("parameterize must be callable.")
        if not values or any(
            not isinstance(value, ScatteringDataset) for value in values
        ):
            raise ValueError(
                "datasets must be a non-empty sequence of ScatteringDataset values."
            )
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.datasets = values
        self.parameterize = parameterize
        self.problem_id = identifier

    def residual(self, parameters: PyTree[Any], /) -> Array:
        model = self.parameterize(parameters)
        return jnp.concatenate(
            tuple(
                complex_scattering_residual(
                    _prediction(model, dataset),
                    dataset.observed,
                    dataset.whitening,
                )
                for dataset in self.datasets
            )
        )


def scattering_least_squares_problem(
    parameterize: Callable[[PyTree[Any]], Any],
    datasets: Sequence[ScatteringDataset],
    /,
    *,
    bounds: Bounds | None = None,
    problem_id: str = "scattering-calibration",
) -> NonlinearLeastSquaresProblem:
    """Return the native Phydrax nonlinear least-squares problem."""
    from ..optim import NonlinearLeastSquaresProblem

    plan = CalibrationResidualPlan(parameterize, datasets, problem_id=problem_id)
    return NonlinearLeastSquaresProblem(
        lambda parameters, args: plan.residual(parameters),
        bounds=bounds,
        problem_id=problem_id,
    )


def scattering_posterior_problem(
    parameter_space: ParameterSpace,
    parameterize: Callable[[PyTree[Any]], Any],
    datasets: Sequence[ScatteringDataset],
    /,
) -> PosteriorProblem:
    """Return a native posterior with unit-normal Cartesian Gaussian likelihood."""
    from ..uq import ParameterSpace, PosteriorProblem

    if not isinstance(parameter_space, ParameterSpace):
        raise TypeError("parameter_space must be ParameterSpace.")
    plan = CalibrationResidualPlan(parameterize, datasets)

    def log_likelihood(parameters):
        residual = plan.residual(parameters)
        return -0.5 * jnp.vdot(residual, residual).real

    return PosteriorProblem(
        parameter_space,
        log_likelihood,
        predict=lambda parameters: tuple(
            _prediction(parameterize(parameters), dataset) for dataset in plan.datasets
        ),
        gauss_newton_residual=plan.residual,
    )


__all__ = [
    "CalibrationResidualPlan",
    "ScatteringDataset",
    "complex_scattering_residual",
    "scattering_least_squares_problem",
    "scattering_posterior_problem",
]
