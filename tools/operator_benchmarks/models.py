from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx

from .scenarios import augment_square_group_training, OperatorBenchmarkScenario


def _apply_query_mask(output, batch: phx.nn.operator.OperatorBatch):
    mask = batch.require_single_query().mask_array(case_shape=batch.case_shape)
    while mask.ndim < jnp.asarray(output).ndim:
        mask = mask[..., None]
    return output * mask


class IdentityBaseline(eqx.Module):
    """Parameter-free identity baseline for coincident source/query samples."""

    source_key: str = eqx.field(static=True)

    def __init__(self, source_key: str):
        self.source_key = str(source_key)

    def __call__(self, batch: phx.nn.operator.OperatorBatch):
        source = batch.input(self.source_key)
        if (
            source.values is None
            or source.sample_shape != batch.require_single_query().sample_shape
        ):
            raise ValueError("IdentityBaseline requires coincident source/query samples.")
        output = jnp.asarray(source.values)
        return _apply_query_mask(output, batch)


class WeightedMeanBaseline(eqx.Module):
    """Parameter-free continuum mean broadcast to every query point."""

    source_key: str = eqx.field(static=True)

    def __init__(self, source_key: str):
        self.source_key = str(source_key)

    def __call__(self, batch: phx.nn.operator.OperatorBatch):
        source = batch.input(self.source_key)
        if source.values is None or not source.sample_shape:
            raise ValueError(
                "WeightedMeanBaseline requires sampled scalar source values."
            )
        values = jnp.asarray(source.values)
        sample_ndim = len(source.sample_shape)
        case_ndim = len(batch.case_shape)
        if values.ndim != case_ndim + sample_ndim:
            raise ValueError("WeightedMeanBaseline currently supports scalar fields.")
        weights = source.weights(case_shape=batch.case_shape)
        axes = tuple(range(case_ndim, case_ndim + sample_ndim))
        mean = jnp.sum(values * weights, axis=axes)
        output = jnp.broadcast_to(
            mean.reshape(
                batch.case_shape + (1,) * len(batch.require_single_query().sample_shape)
            ),
            batch.case_shape + batch.require_single_query().sample_shape,
        )
        return _apply_query_mask(output, batch)


class NearestNeighborBaseline(eqx.Module):
    """Parameter-free nearest-source interpolation baseline."""

    source_key: str = eqx.field(static=True)

    def __init__(self, source_key: str):
        self.source_key = str(source_key)

    def __call__(self, batch: phx.nn.operator.OperatorBatch):
        source = batch.input(self.source_key)
        if source.values is None or not source.sample_shape:
            raise ValueError("NearestNeighborBaseline requires sampled source values.")
        source_dim = _coordinate_dimension(source)
        query_dim = _coordinate_dimension(batch.require_single_query())
        if source_dim != query_dim:
            raise ValueError(
                "NearestNeighborBaseline requires source and query coordinate "
                "dimensions to match."
            )
        case_count = prod(batch.case_shape) if batch.case_shape else 1
        source_coordinates = source.coordinates_array(
            case_shape=batch.case_shape,
            flatten=True,
        ).reshape((case_count, -1, source.coordinates_array().shape[-1]))
        query_coordinates = (
            batch.require_single_query()
            .coordinates_array(
                case_shape=batch.case_shape,
                flatten=True,
            )
            .reshape((case_count, -1, source_coordinates.shape[-1]))
        )
        values = jnp.asarray(source.values)
        if values.ndim != len(batch.case_shape) + len(source.sample_shape):
            raise ValueError("NearestNeighborBaseline currently supports scalar fields.")
        values = values.reshape((case_count, -1))
        distance = jnp.sum(
            (query_coordinates[:, :, None, :] - source_coordinates[:, None, :, :]) ** 2,
            axis=-1,
        )
        nearest = jnp.argmin(distance, axis=-1)
        output = jnp.take_along_axis(values, nearest, axis=1)
        output = output.reshape(
            batch.case_shape + batch.require_single_query().sample_shape
        )
        return _apply_query_mask(output, batch)


class ConstantOutputBaseline(eqx.Module):
    """Training-population target mean broadcast over any query geometry."""

    value: tuple[float, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, scenario: OperatorBenchmarkScenario):
        target = jnp.asarray(scenario.train_target)
        batch = scenario.train_batch
        mask = batch.require_single_query().mask_array(case_shape=batch.case_shape)
        while mask.ndim < target.ndim:
            mask = mask[..., None]
        reduced_axes = tuple(
            range(len(batch.case_shape) + len(batch.require_single_query().sample_shape))
        )
        denominator = jnp.maximum(
            jnp.sum(batch.require_single_query().mask_array(case_shape=batch.case_shape)),
            1,
        )
        mean = jnp.sum(jnp.where(mask, target, 0.0), axis=reduced_axes) / denominator
        self.output_shape = tuple(int(size) for size in mean.shape)
        self.value = tuple(float(value) for value in mean.reshape(-1))

    def __call__(self, batch: phx.nn.operator.OperatorBatch):
        value = jnp.asarray(self.value).reshape(self.output_shape)
        output = jnp.broadcast_to(
            value,
            batch.case_shape
            + batch.require_single_query().sample_shape
            + self.output_shape,
        )
        return _apply_query_mask(output, batch)


class LinearInterpolationBaseline(eqx.Module):
    """Piecewise-linear interpolation for scalar one-dimensional source fields."""

    source_key: str = eqx.field(static=True)

    def __init__(self, source_key: str):
        self.source_key = str(source_key)

    def __call__(self, batch: phx.nn.operator.OperatorBatch):
        source = batch.input(self.source_key)
        if source.values is None or _coordinate_dimension(source) != 1:
            raise ValueError(
                "LinearInterpolationBaseline requires one-dimensional source values."
            )
        if _coordinate_dimension(batch.require_single_query()) != 1:
            raise ValueError(
                "LinearInterpolationBaseline requires one-dimensional query points."
            )
        case_count = prod(batch.case_shape) if batch.case_shape else 1
        source_coordinates = source.coordinates_array(
            case_shape=batch.case_shape,
            flatten=True,
        ).reshape((case_count, -1))
        query_coordinates = (
            batch.require_single_query()
            .coordinates_array(
                case_shape=batch.case_shape,
                flatten=True,
            )
            .reshape((case_count, -1))
        )
        values = jnp.asarray(source.values)
        if values.ndim != len(batch.case_shape) + len(source.sample_shape):
            raise ValueError(
                "LinearInterpolationBaseline currently supports scalar fields."
            )
        values = values.reshape((case_count, -1))
        order = jnp.argsort(source_coordinates, axis=-1)
        sorted_coordinates = jnp.take_along_axis(source_coordinates, order, axis=-1)
        sorted_values = jnp.take_along_axis(values, order, axis=-1)
        output = jax.vmap(jnp.interp)(
            query_coordinates,
            sorted_coordinates,
            sorted_values,
        )
        output = output.reshape(
            batch.case_shape + batch.require_single_query().sample_shape
        )
        return _apply_query_mask(output, batch)


def _flatten_batch_inputs(batch: phx.nn.operator.OperatorBatch, names: tuple[str, ...]):
    case_count = prod(batch.case_shape) if batch.case_shape else 1
    features = []
    for name in names:
        values = batch.input(name).values
        if values is None:
            raise ValueError("Reduced-order baselines require valued operator inputs.")
        features.append(jnp.asarray(values).reshape((case_count, -1)))
    return jnp.concatenate(features, axis=-1)


class PODLinearROMBaseline(eqx.Module):
    """POD output basis with a fitted linear map from source observations."""

    output_mean: jax.Array
    basis: jax.Array
    coefficient_map: jax.Array
    input_names: tuple[str, ...] = eqx.field(static=True)
    query_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        scenario: OperatorBenchmarkScenario,
        *,
        rank: int,
    ):
        self.input_names = tuple(scenario.train_batch.inputs)
        self.query_shape = scenario.train_batch.require_single_query().sample_shape
        features = _flatten_batch_inputs(scenario.train_batch, self.input_names)
        targets = jnp.asarray(scenario.train_target).reshape(
            (prod(scenario.train_batch.case_shape), -1)
        )
        self.output_mean = jnp.mean(targets, axis=0)
        centered = targets - self.output_mean
        _, _, right = jnp.linalg.svd(centered, full_matrices=False)
        retained = max(1, min(int(rank), int(right.shape[0])))
        self.basis = right[:retained].T
        coefficients = centered @ self.basis
        design = jnp.concatenate(
            (features, jnp.ones((features.shape[0], 1), dtype=features.dtype)),
            axis=-1,
        )
        self.coefficient_map = jnp.linalg.pinv(design) @ coefficients

    def __call__(self, batch: phx.nn.operator.OperatorBatch):
        if batch.require_single_query().sample_shape != self.query_shape:
            raise ValueError("PODLinearROMBaseline requires its fitted query geometry.")
        features = _flatten_batch_inputs(batch, self.input_names)
        if int(features.shape[-1]) + 1 != int(self.coefficient_map.shape[0]):
            raise ValueError("PODLinearROMBaseline input shape differs from its fit.")
        design = jnp.concatenate(
            (features, jnp.ones((features.shape[0], 1), dtype=features.dtype)),
            axis=-1,
        )
        output = self.output_mean + (design @ self.coefficient_map) @ self.basis.T
        output = output.reshape(batch.case_shape + self.query_shape)
        return _apply_query_mask(output, batch)


class PointwiseAffineBaseline(eqx.Module):
    """Trainable scalar affine baseline on coincident samples."""

    weight: jax.Array
    bias: jax.Array
    source_key: str = eqx.field(static=True)

    def __init__(self, source_key: str, *, key: jax.Array):
        self.source_key = str(source_key)
        self.weight = 0.1 * jr.normal(key, ())
        self.bias = jnp.zeros(())

    def __call__(self, batch: phx.nn.operator.OperatorBatch):
        source = batch.input(self.source_key)
        if (
            source.values is None
            or source.sample_shape != batch.require_single_query().sample_shape
        ):
            raise ValueError(
                "PointwiseAffineBaseline requires coincident source/query samples."
            )
        output = self.weight * jnp.asarray(source.values) + self.bias
        return _apply_query_mask(output, batch)


_BENCHMARK_CAPABILITY_ARCHITECTURES = {
    "deeponet": "DeepONet",
    "function_frame_deeponet": "FunctionFrameReconstructor",
    "pod_deeponet": "PODDeepONet",
    "local_integral": "LocalIntegralOperator",
    "gino": "GINO",
    "geometry_informed_flower": "GeometryInformedFlower",
    "geometry_informed_flower_learned": "GeometryInformedFlower",
    "geometry_informed_flower_support": "GeometryInformedFlower",
    "geometry_informed_flower_support_conservative": "GeometryInformedFlower",
    "rigno": "RIGNO",
    "gaot": "GAOT",
    "gnot": "GNOT",
    "transolver": "Transolver",
    "upt": "UPT",
    "fno": "FNO",
    "fno_p4_augmented": "FNO",
    "tfno": "TFNO",
    "lattice_equivariant_cno": "LatticeEquivariantCNO",
    "cno": "CNO",
    "uno": "UNO",
    "sfno": "SFNO",
    "ifno": "IFNO",
    "axial_factorized_fno": "AxialFactorizedFNO",
    "poseidon": "Poseidon",
    "wavelet": "WaveletNeuralOperator",
    "multiwavelet": "MultiwaveletOperator",
    "flower_one_level": "Flower",
    "flower_multilevel": "Flower",
    "flower_resolution_consistent": "Flower",
    "laplace": "LaplaceTemporalOperator",
    "linear_recurrent": "LinearRecurrentOperator",
    "selective_state_space": "SelectiveStateSpaceMixer",
}


def _operator_problem_spec(
    scenario: OperatorBenchmarkScenario,
    /,
) -> phx.nn.operator.OperatorProblemSpec:
    if scenario.task is not None:
        return scenario.task.problem
    batches = (
        scenario.train_batch,
        *(() if scenario.validation is None else (scenario.validation.batch,)),
        *(evaluation.batch for evaluation in scenario.evaluations),
    )
    reference_query = scenario.train_batch.require_single_query()
    query_is_fixed = all(
        _same_query_geometry(reference_query, batch.require_single_query())
        for batch in batches[1:]
    )
    source_name, _ = _primary_source(scenario)

    def has_shared_topology(batch: phx.nn.operator.OperatorBatch, /) -> bool:
        source_topology = batch.input(source_name).topology
        query_topology = batch.require_single_query().topology
        if source_topology is None or query_topology is None:
            return False
        return phx.nn.operator.operator_topology_fingerprint(
            source_topology
        ) == phx.nn.operator.operator_topology_fingerprint(query_topology)

    shared_topology = all(has_shared_topology(batch) for batch in batches)
    coincident = all(
        _same_query_geometry(
            batch.input(source_name),
            batch.require_single_query(),
        )
        for batch in batches
    )
    source_query_relation = (
        "shared_topology"
        if shared_topology
        else "coincident"
        if coincident
        else "independent"
    )
    resolution_transfer = any(
        evaluation.batch.require_single_query().sample_shape
        != reference_query.sample_shape
        for evaluation in scenario.evaluations
    )
    rollout_steps = max(
        (evaluation.rollout_steps for evaluation in scenario.evaluations),
        default=1,
    )
    return phx.nn.operator.OperatorProblemSpec(
        source_query_relation=source_query_relation,
        query_is_fixed=query_is_fixed,
        requires_resolution_transfer=resolution_transfer,
        rollout_steps=rollout_steps,
    )


def _operator_field_specs(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    structured_tensors: bool = False,
) -> tuple[phx.nn.operator.OperatorFieldSpec, ...]:
    if scenario.task is not None:
        return scenario.task.fields
    symmetry = scenario.symmetry
    source_representations = (
        {} if symmetry is None else dict(symmetry.source_representations)
    )
    target_representation = (
        "scalar" if symmetry is None else symmetry.target_representation
    )

    def field_layout(name: str, representation: str):
        if not structured_tensors:
            return None
        tensor_type = phx.nn.operator.representations.TensorType(
            (),
            parity=-1 if representation == "pseudoscalar" else 1,
            dimension=2,
        )
        return phx.nn.operator.representations.TensorFieldLayout(
            (
                phx.nn.operator.representations.TensorFieldBlock(
                    name,
                    tensor_type,
                ),
            )
        )

    def field_representation(representation: str) -> str:
        return "tensor" if structured_tensors else representation

    return (
        *(
            phx.nn.operator.OperatorFieldSpec(
                name,
                role="source",
                source_name=name,
                representation=field_representation(
                    source_representations.get(name, "scalar")
                ),
                tensor_layout=field_layout(
                    name, source_representations.get(name, "scalar")
                ),
            )
            for name in scenario.train_batch.inputs
        ),
        phx.nn.operator.OperatorFieldSpec(
            "solution",
            role="target",
            query_name="query",
            representation=field_representation(target_representation),
            tensor_layout=field_layout("solution", target_representation),
        ),
    )


def _architecture_capability_reports(
    architecture: "OperatorArchitecture",
    scenario: OperatorBenchmarkScenario,
    /,
) -> tuple[phx.nn.operator.OperatorCompatibilityReport, ...]:
    capability_name = (
        architecture.capability_name
        or _BENCHMARK_CAPABILITY_ARCHITECTURES.get(architecture.name)
    )
    if capability_name is None:
        return ()
    configuration = (
        {}
        if architecture.capability_configuration_factory is None
        else dict(architecture.capability_configuration_factory(scenario))
    )
    problem = _operator_problem_spec(scenario)
    batches = (
        scenario.train_batch,
        *(() if scenario.validation is None else (scenario.validation.batch,)),
        *(evaluation.batch for evaluation in scenario.evaluations),
    )
    contract = phx.nn.operator.operator_architecture_contract(
        capability_name,
        configuration=configuration,
    )
    fields = _operator_field_specs(
        scenario,
        structured_tensors=contract.capabilities.requires_structured_tensors,
    )
    return tuple(
        phx.nn.operator.validate_operator_architecture(
            capability_name,
            batch,
            configuration=configuration,
            problem=problem,
            fields=fields,
            training_evidence=phx.nn.operator.OperatorTrainingEvidence(
                regime="task_specific"
            ),
        )
        for batch in batches
    )


@dataclass(frozen=True)
class OperatorArchitecture:
    name: str
    family: str
    factory: Callable[[OperatorBenchmarkScenario, int, float], Any]
    trainable: bool
    normalization: Literal["none", "sourcewise", "spectral"] = "sourcewise"
    promotion_scope: Literal["general", "specialized", "reference", "external"] = (
        "general"
    )
    configuration_factory: (
        Callable[
            [OperatorBenchmarkScenario, float],
            tuple[tuple[str, str], ...],
        ]
        | None
    ) = None
    training_scenario_factory: (
        Callable[[OperatorBenchmarkScenario], OperatorBenchmarkScenario] | None
    ) = None
    capability_name: str | None = None
    capability_configuration_factory: (
        Callable[[OperatorBenchmarkScenario], tuple[tuple[str, object], ...]] | None
    ) = None

    def build(
        self,
        scenario: OperatorBenchmarkScenario,
        seed: int = 0,
        *,
        size_scale: float = 1.0,
    ):
        if float(size_scale) <= 0.0:
            raise ValueError("size_scale must be positive.")
        return self.factory(scenario, int(seed), float(size_scale))

    def configuration(
        self,
        scenario: OperatorBenchmarkScenario,
        /,
        *,
        size_scale: float = 1.0,
    ) -> tuple[tuple[str, str], ...]:
        if self.configuration_factory is None:
            return ()
        return self.configuration_factory(scenario, float(size_scale))

    def training_scenario(
        self,
        scenario: OperatorBenchmarkScenario,
        /,
    ) -> OperatorBenchmarkScenario:
        if self.training_scenario_factory is None:
            return scenario
        return self.training_scenario_factory(scenario)

    def capability_reports(
        self,
        scenario: OperatorBenchmarkScenario,
        /,
    ) -> tuple[phx.nn.operator.OperatorCompatibilityReport, ...]:
        """Return the public configured-contract report for every scenario split."""
        return _architecture_capability_reports(self, scenario)

    def runtime_compatible(
        self,
        scenario: OperatorBenchmarkScenario,
        /,
    ) -> bool:
        return all(
            report.runtime_accepted for report in self.capability_reports(scenario)
        )


def _primary_source(scenario: OperatorBenchmarkScenario):
    name = next(iter(scenario.train_batch.inputs))
    return name, scenario.train_batch.input(name)


def _coincident(scenario: OperatorBenchmarkScenario) -> bool:
    source_name, _ = _primary_source(scenario)
    batches = (scenario.train_batch,) + tuple(
        evaluation.batch for evaluation in scenario.evaluations
    )
    return all(
        batch.input(source_name).sample_shape == batch.require_single_query().sample_shape
        for batch in batches
    )


def _same_query_geometry(
    left: phx.nn.operator.FunctionSamples,
    right: phx.nn.operator.FunctionSamples,
    /,
) -> bool:
    if left.sample_shape != right.sample_shape or len(left.axes) != len(right.axes):
        return False
    if left.axes:
        return all(
            first.name == second.name
            and first.basis == second.basis
            and first.periodic == second.periodic
            and bool(jnp.array_equal(first.nodes, second.nodes))
            for first, second in zip(left.axes, right.axes, strict=True)
        )
    if left.coordinates is None or right.coordinates is None:
        return left.coordinates is right.coordinates
    return bool(jnp.array_equal(left.coordinates, right.coordinates))


def _scalar_source(scenario: OperatorBenchmarkScenario) -> bool:
    _, source = _primary_source(scenario)
    if source.values is None:
        return False
    expected = len(scenario.train_batch.case_shape) + len(source.sample_shape)
    return jnp.asarray(source.values).ndim == expected


def _coordinate_dimension(samples: phx.nn.operator.FunctionSamples) -> int:
    if samples.axes:
        return len(samples.axes)
    if samples.coordinates is not None:
        return int(samples.coordinates.shape[-1])
    raise ValueError("Sample geometry has no coordinates.")


def _deeponet_factory(*, pod: bool, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        batch = scenario.train_batch
        latent = (
            min(
                max(1, round((4 if quick else 16) * size_scale)),
                int(batch.case_shape[0]),
            )
            if pod
            else max(1, round((6 if quick else 24) * size_scale))
        )
        keys = iter(jr.split(jr.key(seed), 2 * len(batch.inputs) + 2))
        branches = {}
        for name, samples in batch.inputs.items():
            if samples.values is None:
                raise ValueError("DeepONet benchmark inputs require values.")
            if samples.sample_shape:
                case_ndim = len(batch.case_shape)
                sample_ndim = len(samples.sample_shape)
                trailing = jnp.asarray(samples.values).shape[case_ndim + sample_ndim :]
                channels = 1 if not trailing else int(prod(trailing))
                coord_dim = _coordinate_dimension(samples)
                feature = phx.nn.models.MLP(
                    in_size=channels + coord_dim,
                    out_size=latent,
                    width_size=max(2, round((12 if quick else 48) * size_scale)),
                    depth=2,
                    key=next(keys),
                )
                branches[name] = phx.nn.operator.architectures.IntegralBranchEncoder(
                    feature_model=feature,
                    latent_size=latent,
                    value_channels="scalar" if channels == 1 else channels,
                    coord_dim=coord_dim,
                )
            else:
                in_size = prod(jnp.asarray(samples.values).shape[len(batch.case_shape) :])
                branch_model = phx.nn.models.MLP(
                    in_size=in_size,
                    out_size=latent,
                    width_size=max(2, round((12 if quick else 48) * size_scale)),
                    depth=2,
                    key=next(keys),
                )
                branches[name] = phx.nn.operator.architectures.FixedBranchEncoder(
                    branch_model, latent
                )
        query_dim = _coordinate_dimension(batch.require_single_query())
        if pod:
            target = jnp.asarray(scenario.train_target).reshape(
                (prod(batch.case_shape), -1)
            )
            _, _, right = jnp.linalg.svd(target, full_matrices=False)
            basis = right[:latent].T.reshape(
                batch.require_single_query().sample_shape + (latent,)
            )
            trunk = phx.nn.operator.architectures.PODBasis(basis, latent_size=latent)
        else:
            trunk = phx.nn.models.MLP(
                in_size=query_dim,
                out_size=latent,
                width_size=max(2, round((12 if quick else 48) * size_scale)),
                depth=2,
                key=next(keys),
            )
        return phx.nn.operator.architectures.DeepONet(
            branch=branches,
            trunk=trunk,
            coord_dim=query_dim,
            latent_size=latent,
            fusion="product",
        )

    return build


def _minimum_function_frame_equations(
    scenario: OperatorBenchmarkScenario,
    /,
) -> int:
    source_name, _ = _primary_source(scenario)
    batches = (
        scenario.train_batch,
        *(() if scenario.validation is None else (scenario.validation.batch,)),
        *(evaluation.batch for evaluation in scenario.evaluations),
    )
    minimum: int | None = None
    for batch in batches:
        if tuple(batch.inputs) != (source_name,):
            return 0
        source = batch.input(source_name)
        if (
            source.values is None
            or not source.sample_shape
            or (not source.axes and source.coordinates is None)
        ):
            return 0
        channels = _sample_channels(source, batch.case_shape)
        base_shape = batch.case_shape + source.sample_shape
        values = jnp.asarray(source.values)
        if tuple(values.shape) == base_shape:
            value_finite = jnp.isfinite(values)
        elif tuple(values.shape) == base_shape + (channels,):
            value_finite = jnp.all(jnp.isfinite(values), axis=-1)
        else:
            return 0
        coordinates = source.coordinates_array(case_shape=batch.case_shape)
        quadrature = source.quadrature(case_shape=batch.case_shape)
        requested = source.mask_array(case_shape=batch.case_shape)
        valid_measure = (
            jnp.all(jnp.isfinite(coordinates), axis=-1)
            & jnp.isfinite(quadrature)
            & (quadrature >= 0.0)
        )
        positive_measure = valid_measure & (quadrature > 0.0)
        if bool(jnp.any(requested & ~valid_measure)) or bool(
            jnp.any(requested & positive_measure & ~value_finite)
        ):
            return 0
        active = requested & positive_measure & value_finite
        sample_axes = tuple(
            range(
                len(batch.case_shape),
                len(batch.case_shape) + len(source.sample_shape),
            )
        )
        equations = int(jax.device_get(jnp.min(jnp.sum(active, axis=sample_axes))))
        current = equations * channels
        minimum = current if minimum is None else min(minimum, current)
    return 0 if minimum is None else minimum


def _function_frame_rank(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    quick: bool,
    size_scale: float,
) -> int:
    requested = max(1, round((4 if quick else 16) * size_scale))
    return min(requested, _minimum_function_frame_equations(scenario))


def _function_frame_compatible(scenario: OperatorBenchmarkScenario, /) -> bool:
    if len(scenario.train_batch.inputs) != 1 or _target_channels(scenario) != 1:
        return False
    source_name, source = _primary_source(scenario)
    query = scenario.train_batch.require_single_query()
    if (
        source.values is None
        or not source.sample_shape
        or (not source.axes and source.coordinates is None)
        or not query.sample_shape
        or (not query.axes and query.coordinates is None)
        or _sample_channels(source, scenario.train_batch.case_shape) != 1
    ):
        return False
    source_dim = _coordinate_dimension(source)
    target_dim = _coordinate_dimension(query)
    batches = (
        scenario.train_batch,
        *(() if scenario.validation is None else (scenario.validation.batch,)),
        *(evaluation.batch for evaluation in scenario.evaluations),
    )
    return _minimum_function_frame_equations(scenario) > 0 and all(
        tuple(batch.inputs) == (source_name,)
        and batch.input(source_name).values is not None
        and bool(batch.input(source_name).sample_shape)
        and (
            bool(batch.input(source_name).axes)
            or batch.input(source_name).coordinates is not None
        )
        and bool(batch.require_single_query().sample_shape)
        and (
            bool(batch.require_single_query().axes)
            or batch.require_single_query().coordinates is not None
        )
        and _sample_channels(batch.input(source_name), batch.case_shape) == 1
        and _coordinate_dimension(batch.input(source_name)) == source_dim
        and _coordinate_dimension(batch.require_single_query()) == target_dim
        for batch in batches
    )


def _function_frame_factory(*, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        source_name, source = _primary_source(scenario)
        query = scenario.train_batch.require_single_query()
        rank = _function_frame_rank(
            scenario,
            quick=quick,
            size_scale=size_scale,
        )
        width = max(2, round((12 if quick else 48) * size_scale))
        source_key, target_key, map_key = jr.split(jr.key(seed), 3)
        source_frame = phx.nn.operator.architectures.LearnedFunctionFrame(
            basis_model=phx.nn.models.MLP(
                in_size=_coordinate_dimension(source),
                out_size=rank,
                width_size=width,
                depth=2,
                key=source_key,
            ),
            rank=rank,
            coord_dim=_coordinate_dimension(source),
            frame_id=f"benchmark-{source_name}-source-frame",
        )
        target_frame = phx.nn.operator.architectures.LearnedFunctionFrame(
            basis_model=phx.nn.models.MLP(
                in_size=_coordinate_dimension(query),
                out_size=rank,
                width_size=width,
                depth=2,
                key=target_key,
            ),
            rank=rank,
            coord_dim=_coordinate_dimension(query),
            frame_id="benchmark-solution-target-frame",
        )
        coefficient_map = phx.nn.models.MLP(
            in_size=rank,
            out_size=rank,
            width_size=width,
            depth=2,
            key=map_key,
        )
        return phx.nn.operator.architectures.FunctionFrameReconstructor(
            source_frame=source_frame,
            target_frame=target_frame,
            coefficient_map=coefficient_map,
            source_name=source_name,
            policy=phx.nn.operator.architectures.FunctionProjectionPolicy(
                ridge=1e-5,
                min_samples=rank,
                rank_policy="regularized",
            ),
        )

    return build


def _function_frame_configuration(*, quick: bool):
    def configuration(
        scenario: OperatorBenchmarkScenario,
        size_scale: float,
    ) -> tuple[tuple[str, str], ...]:
        rank = _function_frame_rank(
            scenario,
            quick=quick,
            size_scale=size_scale,
        )
        width = max(2, round((12 if quick else 48) * size_scale))
        return (
            ("rank", str(rank)),
            ("frame_width", str(width)),
            ("coefficient_map", "nonlinear_mlp"),
            ("projection", "weighted_regularized"),
            ("ridge", "1e-5"),
        )

    return configuration


def _fno_factory(*, factorization: str, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        name, source = _primary_source(scenario)
        modes = tuple(
            max(1, min(4 if quick else 12, size // 2)) for size in source.sample_shape
        )
        channels = _sample_channels(source, scenario.train_batch.case_shape)
        output_channels = _target_channels(scenario)
        return phx.nn.operator.architectures.FNO(
            n_modes=modes,
            in_channels="scalar" if channels == 1 else channels,
            out_channels="scalar" if output_channels == 1 else output_channels,
            width=max(2, round((6 if quick else 32) * size_scale)),
            depth=1 if quick else 4,
            factorization=factorization,
            rank=0.5,
            source_key=name,
            key=jr.key(seed),
        )

    return build


def _fno_configuration(
    *,
    factorization: str,
    quick: bool,
    augmentation: str | None = None,
):
    def configuration(
        scenario: OperatorBenchmarkScenario,
        size_scale: float,
    ) -> tuple[tuple[str, str], ...]:
        _, source = _primary_source(scenario)
        modes = tuple(
            max(1, min(4 if quick else 12, size // 2)) for size in source.sample_shape
        )
        values = (
            ("n_modes", repr(modes)),
            ("width", str(max(2, round((6 if quick else 32) * size_scale)))),
            ("depth", str(1 if quick else 4)),
            ("factorization", str(factorization)),
            ("rank", "0.5"),
        )
        if augmentation is None:
            return values
        return values + (("augmentation", augmentation),)

    return configuration


def _hofno_factory(
    *,
    interaction_order: int,
    aliasing: str,
    quick: bool,
):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        name, source = _primary_source(scenario)
        modes = tuple(
            max(1, min(4 if quick else 12, size // 2)) for size in source.sample_shape
        )
        channels = _sample_channels(source, scenario.train_batch.case_shape)
        output_channels = _target_channels(scenario)
        return phx.nn.operator.architectures.HOFNO(
            n_modes=modes,
            in_channels="scalar" if channels == 1 else channels,
            out_channels="scalar" if output_channels == 1 else output_channels,
            width=max(2, round((6 if quick else 32) * size_scale)),
            depth=1 if quick else 4,
            interaction_order=interaction_order,
            factor_bias=False,
            spectral_channel_mixing="depthwise",
            aliasing=aliasing,
            ffn_expansion=2,
            coordinate_embedding=False,
            source_key=name,
            key=jr.key(seed),
        )

    return build


def _hofno_configuration(
    *,
    interaction_order: int,
    aliasing: str,
    quick: bool,
):
    def configuration(
        scenario: OperatorBenchmarkScenario,
        size_scale: float,
    ) -> tuple[tuple[str, str], ...]:
        _, source = _primary_source(scenario)
        modes = tuple(
            max(1, min(4 if quick else 12, size // 2)) for size in source.sample_shape
        )
        return (
            ("n_modes", repr(modes)),
            ("width", str(max(2, round((6 if quick else 32) * size_scale)))),
            ("depth", str(1 if quick else 4)),
            ("interaction_order", str(int(interaction_order))),
            ("factor_bias", "False"),
            ("spectral_channel_mixing", "depthwise"),
            ("aliasing", aliasing),
            ("ffn_expansion", "2"),
            ("coordinate_embedding", "False"),
        )

    return configuration


def _implicit_fno_factory(*, axial: bool, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        name, source = _primary_source(scenario)
        modes = tuple(
            max(1, min(4 if quick else 12, size // 2)) for size in source.sample_shape
        )
        channels = _sample_channels(source, scenario.train_batch.case_shape)
        output_channels = _target_channels(scenario)
        common = {
            "n_modes": modes,
            "in_channels": "scalar" if channels == 1 else channels,
            "out_channels": ("scalar" if output_channels == 1 else output_channels),
            "width": max(2, round((6 if quick else 32) * size_scale)),
            "source_key": name,
            "key": jr.key(seed),
        }
        if axial:
            return phx.nn.operator.architectures.AxialFactorizedFNO(
                **common,
                depth=1 if quick else 4,
            )
        return phx.nn.operator.architectures.IFNO(
            **common,
            iterations=1 if quick else 8,
        )

    return build


def _poseidon_factory(*, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        name, source = _primary_source(scenario)
        channels = _sample_channels(source, scenario.train_batch.case_shape)
        output_channels = _target_channels(scenario)
        return phx.nn.operator.architectures.Poseidon(
            image_shape=source.sample_shape,
            patch_size=1,
            in_channels="scalar" if channels == 1 else channels,
            out_channels="scalar" if output_channels == 1 else output_channels,
            embed_dim=max(2, round((4 if quick else 24) * size_scale)),
            depths=(1, 1),
            num_heads=(1, 1),
            window_size=2,
            mlp_ratio=2.0,
            skip_depths=(0,),
            source_key=name,
            key=jr.key(seed),
        )

    return build


def _cno_factory(*, uno: bool, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        name, source = _primary_source(scenario)
        channels = _sample_channels(source, scenario.train_batch.case_shape)
        output_channels = _target_channels(scenario)
        channel_settings = {
            "in_channels": "scalar" if channels == 1 else channels,
            "out_channels": "scalar" if output_channels == 1 else output_channels,
        }
        if uno:
            return phx.nn.operator.architectures.UNO(
                spatial_ndim=len(source.sample_shape),
                widths=tuple(
                    max(2, round(width * size_scale))
                    for width in ((4, 6) if quick else (24, 48, 64))
                ),
                **channel_settings,
                source_key=name,
                key=jr.key(seed),
            )
        return phx.nn.operator.architectures.CNO(
            spatial_ndim=len(source.sample_shape),
            width=max(2, round((4 if quick else 32) * size_scale)),
            depth=1 if quick else 4,
            **channel_settings,
            source_key=name,
            key=jr.key(seed),
        )

    return build


def _square_tensor_layout(
    name: str,
    representation: Literal["scalar", "pseudoscalar"],
    /,
    *,
    multiplicity: int = 1,
) -> phx.nn.operator.representations.TensorFieldLayout:
    tensor_type = phx.nn.operator.representations.TensorType(
        (),
        parity=-1 if representation == "pseudoscalar" else 1,
        dimension=2,
    )
    return phx.nn.operator.representations.TensorFieldLayout(
        (
            phx.nn.operator.representations.TensorFieldBlock(
                name,
                tensor_type,
                multiplicity=multiplicity,
            ),
        )
    )


def _lattice_equivariant_cno_factory(*, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        if scenario.symmetry is None or scenario.symmetry.group is None:
            raise ValueError(
                "LatticeEquivariantCNO requires declared exact square-group symmetry."
            )
        name, _ = _primary_source(scenario)
        source_representations = dict(scenario.symmetry.source_representations)
        source_representation = source_representations[name]
        target_representation = scenario.symmetry.target_representation
        width = max(2, min(16, round((4 if quick else 8) * size_scale)))
        group = (
            phx.nn.operator.representations.FiniteOrthogonalGroup.c4()
            if scenario.symmetry.group == "p4"
            else phx.nn.operator.representations.FiniteOrthogonalGroup.d4()
        )
        return phx.nn.operator.architectures.LatticeEquivariantCNO(
            group,
            _square_tensor_layout(name, source_representation),
            _square_tensor_layout("solution", target_representation),
            hidden_layout=_square_tensor_layout(
                "hidden",
                source_representation,
                multiplicity=width,
            ),
            width=width,
            depth=1 if quick else 4,
            kernel_size=3,
            activation="tanh",
            source_key=name,
            squeeze_scalar_output=True,
            key=jr.key(seed),
        )

    return build


def _lattice_equivariant_cno_configuration(*, quick: bool):
    def configuration(
        scenario: OperatorBenchmarkScenario,
        size_scale: float,
    ) -> tuple[tuple[str, str], ...]:
        if scenario.symmetry is None or scenario.symmetry.group is None:
            raise ValueError(
                "LatticeEquivariantCNO requires declared exact square-group symmetry."
            )
        return (
            (
                "symmetry_group",
                "C4" if scenario.symmetry.group == "p4" else "D4",
            ),
            (
                "width",
                str(max(2, min(16, round((4 if quick else 8) * size_scale)))),
            ),
            ("depth", str(1 if quick else 4)),
            ("kernel_size", "3"),
            ("activation", "tanh"),
            ("squeeze_scalar_output", "True"),
        )

    return configuration


def _lattice_equivariant_cno_compatible(
    scenario: OperatorBenchmarkScenario,
    /,
) -> bool:
    symmetry = scenario.symmetry
    if (
        symmetry is None
        or symmetry.group not in ("p4", "p4m")
        or not _periodic_tensor_grid_compatible(scenario)
        or len(scenario.train_batch.inputs) != 1
    ):
        return False
    name, source = _primary_source(scenario)
    source_representations = dict(symmetry.source_representations)
    return (
        len(source.sample_shape) == 2
        and _sample_channels(source, scenario.train_batch.case_shape) == 1
        and _target_channels(scenario) == 1
        and source_representations.get(name) in ("scalar", "pseudoscalar")
        and symmetry.target_representation in ("scalar", "pseudoscalar")
    )


def _wavelet_factory(*, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        name, source = _primary_source(scenario)
        channels = _sample_channels(source, scenario.train_batch.case_shape)
        output_channels = _target_channels(scenario)
        return phx.nn.operator.architectures.WaveletNeuralOperator(
            len(source.sample_shape),
            in_channels="scalar" if channels == 1 else channels,
            out_channels="scalar" if output_channels == 1 else output_channels,
            levels=2 if quick else 3,
            boundary="periodization",
            width=max(2, round((4 if quick else 32) * size_scale)),
            depth=1 if quick else 4,
            source_key=name,
            key=jr.key(seed),
        )

    return build


def _multiwavelet_factory(*, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        name, source = _primary_source(scenario)
        channels = _sample_channels(source, scenario.train_batch.case_shape)
        output_channels = _target_channels(scenario)
        return phx.nn.operator.architectures.MultiwaveletOperator(
            in_channels="scalar" if channels == 1 else channels,
            out_channels="scalar" if output_channels == 1 else output_channels,
            order=2 if quick else 3,
            levels=2 if quick else 3,
            boundary="periodization",
            width=max(2, round((4 if quick else 32) * size_scale)),
            depth=1 if quick else 4,
            source_key=name,
            key=jr.key(seed),
        )

    return build


def _flower_settings(
    scenario: OperatorBenchmarkScenario,
    size_scale: float,
    *,
    quick: bool,
    levels: int,
    transition_mode: Literal["learned", "resolution_consistent"],
):
    name, source = _primary_source(scenario)
    channels = _sample_channels(source, scenario.train_batch.case_shape)
    output_channels = _target_channels(scenario)
    groups = 2 if quick else 4
    base_width = 4 if quick else 32
    width = max(groups, round(base_width * size_scale / groups) * groups)
    return {
        "in_channels": "scalar" if channels == 1 else channels,
        "out_channels": "scalar" if output_channels == 1 else output_channels,
        "spatial_ndim": len(source.sample_shape),
        "boundary": "periodic",
        "width": width,
        "levels": int(levels),
        "num_heads": groups,
        "groups": groups,
        "source_key": name,
        "transition_mode": transition_mode,
        "query_mode": "coincident",
        "source_mask_mode": ("reject" if transition_mode == "learned" else "renormalize"),
        "probabilistic_routing": False,
        "conserve_mass": False,
    }


def _flower_factory(
    *,
    quick: bool,
    levels: int,
    transition_mode: Literal["learned", "resolution_consistent"],
):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        return phx.nn.operator.architectures.Flower(
            **_flower_settings(
                scenario,
                size_scale,
                quick=quick,
                levels=levels,
                transition_mode=transition_mode,
            ),
            key=jr.key(seed),
        )

    return build


def _flower_configuration(
    *,
    quick: bool,
    levels: int,
    transition_mode: Literal["learned", "resolution_consistent"],
):
    def configuration(
        scenario: OperatorBenchmarkScenario,
        size_scale: float,
    ) -> tuple[tuple[str, str], ...]:
        settings = _flower_settings(
            scenario,
            size_scale,
            quick=quick,
            levels=levels,
            transition_mode=transition_mode,
        )
        return tuple(
            (name, str(value)) for name, value in settings.items() if name != "source_key"
        )

    return configuration


def _local_factory(*, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        name, source = _primary_source(scenario)
        coord_dim = _coordinate_dimension(source)
        kernel = phx.nn.models.MLP(
            in_size=1 + 3 * coord_dim,
            out_size=1,
            width_size=max(2, round((8 if quick else 48) * size_scale)),
            depth=2,
            key=jr.key(seed),
        )
        return phx.nn.operator.architectures.LocalIntegralOperator(
            kernel_model=kernel,
            coord_dim=coord_dim,
            source_key=name,
            query_chunk_size=64 if quick else 256,
        )

    return build


def _sfno_factory(*, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        name, _ = _primary_source(scenario)
        metadata = dict(scenario.metadata)
        space = phx.discretization.SphericalSpectralPlan(
            int(metadata["bandlimit"]),
            sampling=metadata["sampling"],
        ).prepare()
        return phx.nn.operator.architectures.SFNO(
            space,
            in_channels="scalar",
            out_channels="scalar",
            width=max(2, round((4 if quick else 24) * size_scale)),
            depth=1 if quick else 4,
            source_key=name,
            key=jr.key(seed),
        )

    return build


def _laplace_factory(*, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        name, _ = _primary_source(scenario)
        return phx.nn.operator.architectures.LaplaceTemporalOperator(
            in_channels="scalar",
            out_channels="scalar",
            num_poles=max(1, round((4 if quick else 24) * size_scale)),
            source_key=name,
            key=jr.key(seed),
        )

    return build


def _linear_recurrent_factory(*, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        name, source = _primary_source(scenario)
        channels = _sample_channels(source, scenario.train_batch.case_shape)
        output_channels = _target_channels(scenario)
        time_axis = source.axes[0].name if source.axes else "time"
        return phx.nn.operator.architectures.LinearRecurrentOperator(
            in_channels="scalar" if channels == 1 else channels,
            out_channels="scalar" if output_channels == 1 else output_channels,
            state_size=max(4, round((8 if quick else 64) * size_scale)),
            execution="associative",
            time_axis=time_axis,
            source_key=name,
            key=jr.key(seed),
        )

    return build


def _linear_recurrent_configuration(*, quick: bool):
    def configuration(
        scenario: OperatorBenchmarkScenario,
        size_scale: float,
    ) -> tuple[tuple[str, str], ...]:
        return (
            ("state_size", str(max(4, round((8 if quick else 64) * size_scale)))),
            ("execution", "associative"),
            ("time_semantics", "ordered_samples"),
        )

    return configuration


def _training_delta_range(
    scenario: OperatorBenchmarkScenario,
) -> tuple[float, float]:
    metadata = dict(scenario.metadata)
    if "minimum_training_step" in metadata and "maximum_training_step" in metadata:
        return (
            float(metadata["minimum_training_step"]),
            float(metadata["maximum_training_step"]),
        )
    _, source = _primary_source(scenario)
    coordinates = source.coordinates_array(case_shape=scenario.train_batch.case_shape)
    if int(coordinates.shape[-1]) != 1:
        raise ValueError(
            "Temporal state-space benchmarks require scalar time coordinates."
        )
    times = coordinates[..., 0]
    mask = source.mask_array(case_shape=scenario.train_batch.case_shape)
    continuation = mask[..., :-1] & mask[..., 1:]
    delta = times[..., 1:] - times[..., :-1]
    lower = jnp.min(jnp.where(continuation, delta, jnp.inf))
    upper = jnp.max(jnp.where(continuation, delta, -jnp.inf))
    return float(lower), float(upper)


def _selective_state_space_factory(*, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        name, source = _primary_source(scenario)
        channels = _sample_channels(source, scenario.train_batch.case_shape)
        output_channels = _target_channels(scenario)
        time_axis = source.axes[0].name if source.axes else "time"
        return phx.nn.operator.architectures.SelectiveStateSpaceMixer(
            in_channels="scalar" if channels == 1 else channels,
            out_channels="scalar" if output_channels == 1 else output_channels,
            state_size=max(4, round((8 if quick else 64) * size_scale)),
            input_integration="linear",
            execution="associative",
            time_axis=time_axis,
            source_key=name,
            training_delta_range=_training_delta_range(scenario),
            key=jr.key(seed),
        )

    return build


def _selective_state_space_configuration(*, quick: bool):
    def configuration(
        scenario: OperatorBenchmarkScenario, size_scale: float
    ) -> tuple[tuple[str, str], ...]:
        lower, upper = _training_delta_range(scenario)
        return (
            ("state_size", str(max(4, round((8 if quick else 64) * size_scale)))),
            ("input_integration", "linear"),
            ("execution", "associative"),
            ("minimum_training_step", str(lower)),
            ("maximum_training_step", str(upper)),
        )

    return configuration


def _sample_channels(
    samples: phx.nn.operator.FunctionSamples,
    case_shape: tuple[int, ...],
) -> int:
    if samples.values is None:
        raise ValueError("Geometry architectures require valued operator inputs.")
    shape = jnp.asarray(samples.values).shape
    trailing = shape[len(case_shape) + len(samples.sample_shape) :]
    return 1 if not trailing else int(prod(trailing))


def _target_channels(scenario: OperatorBenchmarkScenario) -> int:
    batch = scenario.train_batch
    shape = jnp.asarray(scenario.train_target).shape
    trailing = shape[
        len(batch.case_shape) + len(batch.require_single_query().sample_shape) :
    ]
    return 1 if not trailing else int(prod(trailing))


def _minimum_valid_source_points(
    scenario: OperatorBenchmarkScenario,
    source_names: tuple[str, ...],
) -> int:
    batches = (scenario.train_batch,) + tuple(
        evaluation.batch for evaluation in scenario.evaluations
    )
    return min(
        int(
            jnp.min(
                jnp.sum(
                    batch.input(name)
                    .mask_array(case_shape=batch.case_shape)
                    .reshape((-1, prod(batch.input(name).sample_shape))),
                    axis=-1,
                )
            )
        )
        for batch in batches
        for name in source_names
    )


def _transolver_factory(*, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        name, source = _primary_source(scenario)
        heads = 2
        width = heads * max(
            1,
            round((4 if quick else 24) * size_scale / heads),
        )
        slices = min(
            max(1, round((4 if quick else 16) * size_scale)),
            _minimum_valid_source_points(scenario, (name,)),
        )
        channels = _sample_channels(source, scenario.train_batch.case_shape)
        output_channels = _target_channels(scenario)
        return phx.nn.operator.architectures.Transolver(
            in_channels="scalar" if channels == 1 else channels,
            out_channels="scalar" if output_channels == 1 else output_channels,
            coord_dim=_coordinate_dimension(source),
            num_slices=slices,
            width=width,
            depth=1 if quick else 4,
            num_heads=heads,
            slice_top_k=min(2, slices),
            source_key=name,
            key=jr.key(seed),
        )

    return build


def _gnot_factory(*, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        batch = scenario.train_batch
        heads = 2
        width = heads * max(
            1,
            round((4 if quick else 24) * size_scale / heads),
        )
        source_channels = {
            name: (
                "scalar"
                if _sample_channels(samples, batch.case_shape) == 1
                else _sample_channels(samples, batch.case_shape)
            )
            for name, samples in batch.inputs.items()
        }
        output_channels = _target_channels(scenario)
        query_channels = (
            0
            if batch.require_single_query().values is None
            else _sample_channels(batch.require_single_query(), batch.case_shape)
        )
        return phx.nn.operator.architectures.GNOT(
            in_channels=source_channels,
            out_channels="scalar" if output_channels == 1 else output_channels,
            coord_dim=_coordinate_dimension(batch.require_single_query()),
            query_channels=query_channels,
            hidden_channels=width,
            encoder_width=width,
            encoder_depth=1 if quick else 2,
            fusion_width=width,
            fusion_depth=1 if quick else 2,
            transformer_depth=1 if quick else 4,
            num_heads=heads,
            key=jr.key(seed),
        )

    return build


def _upt_factory(*, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        name, source = _primary_source(scenario)
        heads = 2
        width = heads * max(
            1,
            round((4 if quick else 24) * size_scale / heads),
        )
        channels = _sample_channels(source, scenario.train_batch.case_shape)
        output_channels = _target_channels(scenario)
        return phx.nn.operator.architectures.UPT(
            in_channels="scalar" if channels == 1 else channels,
            out_channels="scalar" if output_channels == 1 else output_channels,
            coord_dim=_coordinate_dimension(source),
            width=width,
            num_tokens=max(1, round((4 if quick else 16) * size_scale)),
            depth=1 if quick else 4,
            num_heads=heads,
            source_key=name,
            key=jr.key(seed),
        )

    return build


def _gino_settings(
    scenario: OperatorBenchmarkScenario,
    size_scale: float,
    *,
    quick: bool,
) -> dict[str, Any]:
    batch = scenario.train_batch
    coord_dim = _coordinate_dimension(batch.require_single_query())
    latent_resolution = 6 if quick else 16
    latent_shape = (latent_resolution,) * coord_dim
    base_width = 4 if quick else 24
    width = max(2, round(base_width * float(size_scale)))
    source_channels = {
        name: _sample_channels(samples, batch.case_shape)
        for name, samples in batch.inputs.items()
    }
    query_channels = (
        0
        if batch.require_single_query().values is None
        else _sample_channels(batch.require_single_query(), batch.case_shape)
    )
    target_shape = jnp.asarray(scenario.train_target).shape
    target_trailing = target_shape[
        len(batch.case_shape) + len(batch.require_single_query().sample_shape) :
    ]
    output_channels = 1 if not target_trailing else int(prod(target_trailing))
    all_coordinates = [
        samples.coordinates_array(case_shape=batch.case_shape, flatten=True)
        for samples in batch.inputs.values()
    ] + [
        batch.require_single_query().coordinates_array(
            case_shape=batch.case_shape, flatten=True
        )
    ]
    flattened = jnp.concatenate(
        tuple(coordinates.reshape((-1, coord_dim)) for coordinates in all_coordinates),
        axis=0,
    )
    lower = jnp.min(flattened, axis=0)
    upper = jnp.max(flattened, axis=0)
    latent_bounds = tuple(
        (float(lower[axis]), float(upper[axis])) for axis in range(coord_dim)
    )
    minimum_source_points = min(
        prod(samples.sample_shape) for samples in batch.inputs.values()
    )
    transfer_neighbors = min(8 if quick else 24, minimum_source_points)
    decoder_neighbors = min(8 if quick else 24, prod(latent_shape))
    return {
        "in_channels": source_channels,
        "out_channels": "scalar" if output_channels == 1 else output_channels,
        "coord_dim": coord_dim,
        "latent_shape": latent_shape,
        "latent_bounds": latent_bounds,
        "bounds_policy": "global",
        "latent_channels": width,
        "modes": tuple(
            max(1, min(3 if quick else 8, size // 2)) for size in latent_shape
        ),
        "fno_width": width,
        "fno_depth": 1 if quick else 4,
        "encoder_neighbors": transfer_neighbors,
        "decoder_neighbors": decoder_neighbors,
        "transfer_width": width,
        "transfer_depth": 1 if quick else 2,
        "query_channels": query_channels,
        "query_chunk_size": 64 if quick else 256,
    }


def _gino_factory(*, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        return phx.nn.operator.architectures.GINO(
            **_gino_settings(scenario, size_scale, quick=quick),
            key=jr.key(seed),
        )

    return build


def _gino_configuration(*, quick: bool):
    def configuration(
        scenario: OperatorBenchmarkScenario,
        size_scale: float,
    ) -> tuple[tuple[str, str], ...]:
        settings = _gino_settings(scenario, size_scale, quick=quick)
        return tuple((name, repr(value)) for name, value in sorted(settings.items()))

    return configuration


def _geometry_informed_flower_settings(
    scenario: OperatorBenchmarkScenario,
    size_scale: float,
    *,
    quick: bool,
    transition_mode: Literal["learned", "resolution_consistent"] = (
        "resolution_consistent"
    ),
    domain_support: bool = False,
    conserve_mass: bool = False,
) -> dict[str, Any]:
    settings = _gino_settings(scenario, size_scale, quick=quick)
    settings.pop("modes")
    settings.pop("fno_width")
    settings.pop("fno_depth")
    coord_dim = int(settings["coord_dim"])
    latent_resolution = 8 if quick else 16
    latent_shape = (latent_resolution,) * coord_dim
    width = int(settings["latent_channels"])
    head_groups = next(candidate for candidate in (4, 2, 1) if width % candidate == 0)
    settings.update(
        {
            "latent_shape": latent_shape,
            "decoder_neighbors": min(
                8 if quick else 24,
                prod(latent_shape),
            ),
            "boundary": "clamp",
            "flower_width": width,
            "flower_levels": 2,
            "flower_num_heads": head_groups,
            "flower_groups": head_groups,
            "transition_mode": transition_mode,
            "source_mask_mode": ("reject" if transition_mode == "learned" else "strict"),
        }
    )
    if domain_support:
        if scenario.domain_support_key is None or scenario.domain_support_kind is None:
            raise ValueError(
                "Domain-supported Flower requires an explicit scenario support contract."
            )
        settings.update(
            {
                "latent_support_key": scenario.domain_support_key,
                "latent_support_kind": scenario.domain_support_kind,
                "source_mask_mode": "renormalize",
            }
        )
        if scenario.domain_support_threshold is not None:
            settings["latent_support_threshold"] = scenario.domain_support_threshold
    if conserve_mass:
        if scenario.conservation_source_key is None:
            raise ValueError(
                "Conservative Flower requires an explicit scenario conservation source."
            )
        settings.update(
            {
                "conserve_mass": True,
                "conservation_source_key": scenario.conservation_source_key,
            }
        )
    return settings


def _geometry_informed_flower_factory(
    *,
    quick: bool,
    transition_mode: Literal["learned", "resolution_consistent"] = (
        "resolution_consistent"
    ),
    domain_support: bool = False,
    conserve_mass: bool = False,
):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        return phx.nn.operator.architectures.GeometryInformedFlower(
            **_geometry_informed_flower_settings(
                scenario,
                size_scale,
                quick=quick,
                transition_mode=transition_mode,
                domain_support=domain_support,
                conserve_mass=conserve_mass,
            ),
            key=jr.key(seed),
        )

    return build


def _geometry_informed_flower_configuration(
    *,
    quick: bool,
    transition_mode: Literal["learned", "resolution_consistent"] = (
        "resolution_consistent"
    ),
    domain_support: bool = False,
    conserve_mass: bool = False,
):
    def configuration(
        scenario: OperatorBenchmarkScenario,
        size_scale: float,
    ) -> tuple[tuple[str, str], ...]:
        settings = _geometry_informed_flower_settings(
            scenario,
            size_scale,
            quick=quick,
            transition_mode=transition_mode,
            domain_support=domain_support,
            conserve_mass=conserve_mass,
        )
        return tuple((name, repr(value)) for name, value in sorted(settings.items()))

    return configuration


def _rigno_settings(
    scenario: OperatorBenchmarkScenario,
    size_scale: float,
    *,
    quick: bool,
) -> dict[str, Any]:
    batch = scenario.train_batch
    coord_dim = _coordinate_dimension(batch.require_single_query())
    base_width = 4 if quick else 24
    width = max(2, round(base_width * float(size_scale)))
    source_channels = {
        name: _sample_channels(samples, batch.case_shape)
        for name, samples in batch.inputs.items()
    }
    query_channels = (
        0
        if batch.require_single_query().values is None
        else _sample_channels(batch.require_single_query(), batch.case_shape)
    )
    target_shape = jnp.asarray(scenario.train_target).shape
    target_trailing = target_shape[
        len(batch.case_shape) + len(batch.require_single_query().sample_shape) :
    ]
    output_channels = 1 if not target_trailing else int(prod(target_trailing))
    scenario_batches = (batch,) + tuple(
        evaluation.batch for evaluation in scenario.evaluations
    )
    minimum_source_points = min(
        prod(samples.sample_shape)
        for current in scenario_batches
        for samples in current.inputs.values()
    )
    minimum_valid_source_points = min(
        int(
            jnp.min(
                jnp.sum(
                    samples.mask_array(case_shape=current.case_shape).reshape(
                        (-1, prod(samples.sample_shape))
                    ),
                    axis=-1,
                )
            )
        )
        for current in scenario_batches
        for samples in current.inputs.values()
    )
    regional_count = min(
        8 if quick else 32,
        minimum_source_points,
        minimum_valid_source_points,
    )
    transfer_neighbors = min(8 if quick else 24, minimum_source_points)
    regional_neighbors = min(4 if quick else 12, regional_count)
    return {
        "in_channels": source_channels,
        "out_channels": "scalar" if output_channels == 1 else output_channels,
        "coord_dim": coord_dim,
        "regional_count": regional_count,
        "regional_mode": "farthest_point",
        "latent_channels": width,
        "processor_neighbors": regional_neighbors,
        "processor_depth": 1 if quick else 4,
        "processor_width": width,
        "processor_mlp_depth": 1 if quick else 2,
        "processor_shared": True,
        "processor_edge_dropout": 0.0,
        "encoder_neighbors": transfer_neighbors,
        "decoder_neighbors": regional_neighbors,
        "transfer_width": width,
        "transfer_depth": 1 if quick else 2,
        "query_channels": query_channels,
        "query_chunk_size": 64 if quick else 256,
    }


def _rigno_factory(*, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        return phx.nn.operator.architectures.RIGNO(
            **_rigno_settings(scenario, size_scale, quick=quick),
            key=jr.key(seed),
        )

    return build


def _rigno_configuration(*, quick: bool):
    def configuration(
        scenario: OperatorBenchmarkScenario,
        size_scale: float,
    ) -> tuple[tuple[str, str], ...]:
        settings = _rigno_settings(scenario, size_scale, quick=quick)
        return tuple((name, repr(value)) for name, value in sorted(settings.items()))

    return configuration


def _gaot_settings(
    scenario: OperatorBenchmarkScenario,
    size_scale: float,
    *,
    quick: bool,
) -> dict[str, Any]:
    batch = scenario.train_batch
    coord_dim = _coordinate_dimension(batch.require_single_query())
    latent_resolution = 4 if quick else 16
    latent_shape = (latent_resolution,) * coord_dim
    base_width = 4 if quick else 24
    width = max(2, round(base_width * float(size_scale)))
    transformer_heads = 2 if quick else 4
    transformer_width = transformer_heads * max(
        1,
        round(2 * width / transformer_heads),
    )
    source_channels = {
        name: _sample_channels(samples, batch.case_shape)
        for name, samples in batch.inputs.items()
    }
    query_channels = (
        0
        if batch.require_single_query().values is None
        else _sample_channels(batch.require_single_query(), batch.case_shape)
    )
    target_shape = jnp.asarray(scenario.train_target).shape
    target_trailing = target_shape[
        len(batch.case_shape) + len(batch.require_single_query().sample_shape) :
    ]
    output_channels = 1 if not target_trailing else int(prod(target_trailing))
    all_coordinates = [
        samples.coordinates_array(case_shape=batch.case_shape, flatten=True)
        for samples in batch.inputs.values()
    ] + [
        batch.require_single_query().coordinates_array(
            case_shape=batch.case_shape, flatten=True
        )
    ]
    flattened = jnp.concatenate(
        tuple(coordinates.reshape((-1, coord_dim)) for coordinates in all_coordinates),
        axis=0,
    )
    physical_scale = float(
        jnp.max(jnp.max(flattened, axis=0) - jnp.min(flattened, axis=0))
    )
    minimum_source_points = min(
        prod(samples.sample_shape) for samples in batch.inputs.values()
    )
    transfer_neighbors = min(
        8 if quick else 32,
        minimum_source_points,
        prod(latent_shape),
    )
    return {
        "in_channels": source_channels,
        "out_channels": "scalar" if output_channels == 1 else output_channels,
        "coord_dim": coord_dim,
        "latent_shape": latent_shape,
        "patch_shape": 2,
        "transfer_radius": 2.0 * physical_scale / float(latent_resolution - 1),
        "transfer_scales": (1.0, 2.0) if quick else (1.0, 2.0, 4.0),
        "bounds_policy": "case_bbox",
        "latent_channels": width,
        "transformer_width": transformer_width,
        "transformer_depth": 1 if quick else 3,
        "transformer_heads": transformer_heads,
        "attention_dropout": 0.0,
        "feed_forward_dropout": 0.0,
        "transfer_neighbors": transfer_neighbors,
        "transfer_width": width,
        "transfer_heads": 2 if quick else 4,
        "transfer_depth": 1 if quick else 2,
        "transfer_fusion": "gated",
        "coordinate_scale": physical_scale,
        "query_channels": query_channels,
        "query_chunk_size": 64 if quick else 256,
    }


def _gaot_factory(*, quick: bool):
    def build(scenario: OperatorBenchmarkScenario, seed: int, size_scale: float):
        return phx.nn.operator.architectures.GAOT(
            **_gaot_settings(scenario, size_scale, quick=quick),
            key=jr.key(seed),
        )

    return build


def _gaot_configuration(*, quick: bool):
    def configuration(
        scenario: OperatorBenchmarkScenario,
        size_scale: float,
    ) -> tuple[tuple[str, str], ...]:
        settings = _gaot_settings(scenario, size_scale, quick=quick)
        return tuple((name, repr(value)) for name, value in sorted(settings.items()))

    return configuration


def _geometry_compatible(scenario: OperatorBenchmarkScenario, /) -> bool:
    batches = (scenario.train_batch,) + tuple(
        evaluation.batch for evaluation in scenario.evaluations
    )
    for batch in batches:
        query_dim = _coordinate_dimension(batch.require_single_query())
        if query_dim not in (1, 2):
            return False
        for samples in batch.inputs.values():
            if (
                samples.values is None
                or not samples.sample_shape
                or _coordinate_dimension(samples) != query_dim
            ):
                return False
            has_quadrature = samples.quadrature_weights is not None or (
                bool(samples.axes)
                and all(axis.quadrature_weights is not None for axis in samples.axes)
            )
            if not has_quadrature:
                return False
    return True


def _domain_support_compatible(scenario: OperatorBenchmarkScenario, /) -> bool:
    support_key = scenario.domain_support_key
    if support_key is None or scenario.domain_support_kind is None:
        return False
    for batch in _all_source_batches(scenario):
        if support_key not in batch.inputs:
            return False
        support = batch.input(support_key)
        if (
            support.values is None
            or not support.sample_shape
            or _channel_layout(support, batch.case_shape) not in ((), (1,))
        ):
            return False
    return True


def _conservative_geometry_compatible(
    scenario: OperatorBenchmarkScenario,
    /,
) -> bool:
    source_key = scenario.conservation_source_key
    if source_key is None:
        return False
    target_channels = _target_channels(scenario)
    for batch in _all_source_batches(scenario):
        if source_key not in batch.inputs:
            return False
        source = batch.input(source_key)
        if _sample_channels(source, batch.case_shape) != target_channels:
            return False
        query_has_quadrature = (
            batch.require_single_query().quadrature_weights is not None
            or (
                bool(batch.require_single_query().axes)
                and all(
                    axis.quadrature_weights is not None
                    for axis in batch.require_single_query().axes
                )
            )
        )
        if not query_has_quadrature:
            return False
        source_measure = source.weights(case_shape=batch.case_shape)
        query_measure = batch.require_single_query().weights(case_shape=batch.case_shape)
        source_axes = tuple(range(len(batch.case_shape), source_measure.ndim))
        query_axes = tuple(range(len(batch.case_shape), query_measure.ndim))
        if bool(jnp.any(jnp.sum(source_measure, axis=source_axes) <= 0.0)) or bool(
            jnp.any(jnp.sum(query_measure, axis=query_axes) <= 0.0)
        ):
            return False
    return True


def _matching_coordinate_dimensions(scenario: OperatorBenchmarkScenario, /) -> bool:
    batches = (scenario.train_batch,) + tuple(
        evaluation.batch for evaluation in scenario.evaluations
    )
    return all(
        _coordinate_dimension(batch.input(next(iter(batch.inputs))))
        == _coordinate_dimension(batch.require_single_query())
        for batch in batches
    )


def _fixed_input_shapes(scenario: OperatorBenchmarkScenario, /) -> bool:
    expected_names = tuple(scenario.train_batch.inputs)

    def signature(batch):
        if tuple(batch.inputs) != expected_names:
            return None
        shapes = []
        for name in expected_names:
            values = batch.input(name).values
            if values is None:
                return None
            shapes.append(tuple(jnp.asarray(values).shape[len(batch.case_shape) :]))
        return tuple(shapes)

    expected = signature(scenario.train_batch)
    return expected is not None and all(
        signature(evaluation.batch) == expected for evaluation in scenario.evaluations
    )


def _channel_layout(
    samples: phx.nn.operator.FunctionSamples,
    case_shape: tuple[int, ...],
) -> tuple[int, ...] | None:
    if samples.values is None:
        return None
    trailing = tuple(
        jnp.asarray(samples.values).shape[len(case_shape) + len(samples.sample_shape) :]
    )
    return trailing


def _field_channels_compatible(scenario: OperatorBenchmarkScenario, /) -> bool:
    train = scenario.train_batch
    query = train.require_single_query()
    names = tuple(train.inputs)
    expected_sources = {
        name: _channel_layout(samples, train.case_shape)
        for name, samples in train.inputs.items()
    }
    expected_query = _channel_layout(query, train.case_shape)
    expected_target = tuple(
        jnp.asarray(scenario.train_target).shape[
            len(train.case_shape) + len(query.sample_shape) :
        ]
    )
    if (
        any(layout is None or len(layout) > 1 for layout in expected_sources.values())
        or (expected_query is not None and len(expected_query) > 1)
        or len(expected_target) > 1
    ):
        return False
    pairs = tuple(
        (evaluation.batch, evaluation.target) for evaluation in scenario.evaluations
    )
    for batch, target in ((train, scenario.train_target),) + pairs:
        if tuple(batch.inputs) != names:
            return False
        if any(
            _channel_layout(batch.input(name), batch.case_shape) != expected_sources[name]
            for name in names
        ):
            return False
        if (
            _channel_layout(batch.require_single_query(), batch.case_shape)
            != expected_query
        ):
            return False
        target_layout = tuple(
            jnp.asarray(target).shape[
                len(batch.case_shape) + len(batch.require_single_query().sample_shape) :
            ]
        )
        if target_layout != expected_target:
            return False
    return True


def _uniform_axis(axis: phx.nn.operator.OperatorAxis) -> bool:
    if int(axis.nodes.shape[0]) < 2:
        return False
    spacing = jnp.diff(axis.nodes)
    return bool(
        jnp.allclose(
            spacing,
            jnp.mean(spacing),
            rtol=1e-5,
            atol=1e-8,
        )
    )


def _coincident_tensor_grid_compatible(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    minimum_dimensions: int = 1,
) -> bool:
    source_name, _ = _primary_source(scenario)
    batches = (scenario.train_batch,) + tuple(
        evaluation.batch for evaluation in scenario.evaluations
    )
    return (
        _field_channels_compatible(scenario)
        and all(tuple(batch.inputs) == (source_name,) for batch in batches)
        and all(
            len(batch.input(source_name).axes) >= minimum_dimensions
            and _same_query_geometry(
                batch.input(source_name), batch.require_single_query()
            )
            and all(_uniform_axis(axis) for axis in batch.input(source_name).axes)
            for batch in batches
        )
    )


def _poseidon_compatible(scenario: OperatorBenchmarkScenario, /) -> bool:
    if not _coincident_tensor_grid_compatible(
        scenario,
        minimum_dimensions=2,
    ):
        return False
    _, source = _primary_source(scenario)
    image_shape = source.sample_shape
    if len(image_shape) != 2 or any(size % 2 for size in image_shape):
        return False
    source_name, _ = _primary_source(scenario)
    batches = (scenario.train_batch,) + tuple(
        evaluation.batch for evaluation in scenario.evaluations
    )
    return all(batch.input(source_name).sample_shape == image_shape for batch in batches)


def _all_source_batches(
    scenario: OperatorBenchmarkScenario,
) -> tuple[phx.nn.operator.OperatorBatch, ...]:
    validation = () if scenario.validation is None else (scenario.validation.batch,)
    return (
        scenario.train_batch,
        *validation,
        *(evaluation.batch for evaluation in scenario.evaluations),
    )


def _periodic_tensor_grid_compatible(scenario: OperatorBenchmarkScenario, /) -> bool:
    if not _coincident_tensor_grid_compatible(scenario):
        return False
    source_name, _ = _primary_source(scenario)
    return all(
        all(axis.periodic for axis in batch.input(source_name).axes)
        for batch in _all_source_batches(scenario)
    )


def _wavelet_tensor_grid_compatible(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    levels: int,
) -> bool:
    if not _periodic_tensor_grid_compatible(scenario):
        return False
    source_name, _ = _primary_source(scenario)
    minimum_size = 2 ** max(0, int(levels) - 1)
    return all(
        min(batch.input(source_name).sample_shape) > minimum_size
        for batch in _all_source_batches(scenario)
    )


def _flower_learned_compatible(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    levels: int,
) -> bool:
    if not _periodic_tensor_grid_compatible(scenario):
        return False
    source_name, _ = _primary_source(scenario)
    minimum_size = 2 ** int(levels)
    divisor = 2 ** max(0, int(levels) - 1)
    return all(
        min(batch.input(source_name).sample_shape) >= minimum_size
        and all(size % divisor == 0 for size in batch.input(source_name).sample_shape)
        and bool(
            jnp.all(batch.input(source_name).mask_array(case_shape=batch.case_shape))
        )
        for batch in _all_source_batches(scenario)
    )


def _flower_resolution_compatible(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    levels: int,
) -> bool:
    if not _periodic_tensor_grid_compatible(scenario):
        return False
    source_name, _ = _primary_source(scenario)
    minimum_size = 2 ** int(levels)
    divisor = 2 ** max(0, int(levels) - 1)
    return all(
        min(batch.input(source_name).sample_shape) >= minimum_size
        and all(size % divisor == 0 for size in batch.input(source_name).sample_shape)
        for batch in _all_source_batches(scenario)
    )


def _cochain_factory(
    *,
    quick: bool,
    routes: phx.nn.operator.architectures.TopologicalRouteConfig,
):
    def factory(
        scenario: OperatorBenchmarkScenario,
        seed: int,
        size_scale: float,
    ):
        if scenario.task is None:
            raise ValueError("Cochain architectures require a benchmark OperatorTask.")
        base_width = 8 if quick else 24
        width = max(2, round(base_width * float(size_scale) ** 0.5))
        depth = 2 if quick else 4
        boundary_policy = dict(scenario.metadata).get("boundary_policy", "absolute")
        return phx.nn.operator.architectures.CochainNeuralOperator(
            scenario.task.fields,
            width=width,
            depth=depth,
            routes=routes,
            boundary_policy=boundary_policy,
            key=jr.key(seed),
        )

    return factory


def _cochain_configuration(
    routes: phx.nn.operator.architectures.TopologicalRouteConfig,
    *,
    quick: bool,
):
    def configuration(
        scenario: OperatorBenchmarkScenario,
        size_scale: float,
    ) -> tuple[tuple[str, str], ...]:
        base_width = 8 if quick else 24
        return (
            ("width", str(max(2, round(base_width * float(size_scale) ** 0.5)))),
            ("depth", str(2 if quick else 4)),
            ("routes", ",".join(routes.enabled_routes)),
            (
                "boundary_policy",
                dict(scenario.metadata).get("boundary_policy", "absolute"),
            ),
        )

    return configuration


def _cochain_architectures(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    quick: bool,
) -> tuple[OperatorArchitecture, ...]:
    pointwise = phx.nn.operator.architectures.TopologicalRouteConfig(
        self_route=True,
        exterior_derivative=False,
        codifferential=False,
        lower_laplacian=False,
        upper_laplacian=False,
        harmonic=False,
    )
    local = phx.nn.operator.architectures.TopologicalRouteConfig(harmonic=False)
    harmonic_task = "harmonic_one_form" in scenario.regimes
    full = phx.nn.operator.architectures.TopologicalRouteConfig(harmonic=harmonic_task)
    candidates = [
        OperatorArchitecture(
            "cochain_pointwise",
            "cochain_ablation",
            _cochain_factory(quick=quick, routes=pointwise),
            True,
            normalization="none",
            promotion_scope="reference",
            configuration_factory=_cochain_configuration(pointwise, quick=quick),
            capability_name="CochainNeuralOperator",
        )
    ]
    if harmonic_task:
        candidates.append(
            OperatorArchitecture(
                "cochain_no_harmonic",
                "cochain_ablation",
                _cochain_factory(quick=quick, routes=local),
                True,
                normalization="none",
                promotion_scope="reference",
                configuration_factory=_cochain_configuration(local, quick=quick),
                capability_name="CochainNeuralOperator",
            )
        )
    candidates.append(
        OperatorArchitecture(
            "cochain_neural_operator",
            "metric_dec",
            _cochain_factory(quick=quick, routes=full),
            True,
            normalization="none",
            promotion_scope="specialized",
            configuration_factory=_cochain_configuration(full, quick=quick),
            capability_name="CochainNeuralOperator",
        )
    )
    return tuple(
        architecture
        for architecture in candidates
        if architecture.runtime_compatible(scenario)
    )


def compatible_architectures(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    quick: bool = False,
) -> tuple[OperatorArchitecture, ...]:
    """Return every architecture whose physical input contract matches ``scenario``."""
    if scenario.task is not None and any(
        field.cochain is not None for field in scenario.task.fields
    ):
        return _cochain_architectures(scenario, quick=quick)
    source_name, source = _primary_source(scenario)
    candidates = [
        OperatorArchitecture(
            "constant",
            "baseline",
            lambda current, seed, scale: ConstantOutputBaseline(current),
            False,
            normalization="none",
            promotion_scope="reference",
        ),
        OperatorArchitecture(
            "deeponet",
            "branch_trunk",
            _deeponet_factory(pod=False, quick=quick),
            True,
            normalization="sourcewise",
            promotion_scope="general",
        ),
    ]
    if _function_frame_compatible(scenario):
        candidates.append(
            OperatorArchitecture(
                "function_frame_deeponet",
                "branch_trunk",
                _function_frame_factory(quick=quick),
                True,
                normalization="sourcewise",
                promotion_scope="specialized",
                configuration_factory=_function_frame_configuration(quick=quick),
            )
        )
    scalar_source = _scalar_source(scenario)
    matching_coordinates = _matching_coordinate_dimensions(scenario)
    if scalar_source:
        candidates.append(
            OperatorArchitecture(
                "weighted_mean",
                "baseline",
                lambda current, seed, scale: WeightedMeanBaseline(source_name),
                False,
                normalization="none",
                promotion_scope="reference",
            )
        )
    if scalar_source and matching_coordinates:
        candidates.append(
            OperatorArchitecture(
                "nearest_neighbor",
                "baseline",
                lambda current, seed, scale: NearestNeighborBaseline(source_name),
                False,
                normalization="none",
                promotion_scope="reference",
            )
        )
        if _coordinate_dimension(source) == 1:
            candidates.append(
                OperatorArchitecture(
                    "linear_interpolation",
                    "baseline",
                    lambda current, seed, scale: LinearInterpolationBaseline(source_name),
                    False,
                    normalization="none",
                    promotion_scope="reference",
                )
            )
    fixed_query = all(
        _same_query_geometry(
            scenario.train_batch.require_single_query(),
            evaluation.batch.require_single_query(),
        )
        for evaluation in scenario.evaluations
    )
    if fixed_query:
        candidates.append(
            OperatorArchitecture(
                "pod_deeponet",
                "branch_trunk",
                _deeponet_factory(pod=True, quick=quick),
                True,
                normalization="sourcewise",
                promotion_scope="specialized",
            )
        )
        if _fixed_input_shapes(scenario):
            candidates.append(
                OperatorArchitecture(
                    "pod_linear_rom",
                    "reduced_order",
                    lambda current, seed, scale: PODLinearROMBaseline(
                        current,
                        rank=max(1, round((3 if quick else 12) * scale)),
                    ),
                    False,
                    normalization="none",
                    promotion_scope="reference",
                )
            )
    if len(scenario.train_batch.inputs) == 1 and scalar_source and matching_coordinates:
        candidates.append(
            OperatorArchitecture(
                "local_integral",
                "local",
                _local_factory(quick=quick),
                True,
                normalization="sourcewise",
                promotion_scope="specialized",
            )
        )
    if _geometry_compatible(scenario):
        candidates.append(
            OperatorArchitecture(
                "gino",
                "geometry_informed",
                _gino_factory(quick=quick),
                True,
                normalization="sourcewise",
                promotion_scope="general",
                configuration_factory=_gino_configuration(quick=quick),
            )
        )
        candidates.append(
            OperatorArchitecture(
                "geometry_informed_flower",
                "geometry_informed_warp",
                _geometry_informed_flower_factory(quick=quick),
                True,
                normalization="sourcewise",
                promotion_scope="general",
                configuration_factory=_geometry_informed_flower_configuration(
                    quick=quick
                ),
            )
        )
        candidates.append(
            OperatorArchitecture(
                "geometry_informed_flower_learned",
                "geometry_informed_warp",
                _geometry_informed_flower_factory(
                    quick=quick,
                    transition_mode="learned",
                ),
                True,
                normalization="sourcewise",
                promotion_scope="general",
                configuration_factory=_geometry_informed_flower_configuration(
                    quick=quick,
                    transition_mode="learned",
                ),
            )
        )
        if _domain_support_compatible(scenario):
            candidates.append(
                OperatorArchitecture(
                    "geometry_informed_flower_support",
                    "geometry_informed_warp",
                    _geometry_informed_flower_factory(
                        quick=quick,
                        domain_support=True,
                    ),
                    True,
                    normalization="sourcewise",
                    promotion_scope="specialized",
                    configuration_factory=(
                        _geometry_informed_flower_configuration(
                            quick=quick,
                            domain_support=True,
                        )
                    ),
                )
            )
            if _conservative_geometry_compatible(scenario):
                candidates.append(
                    OperatorArchitecture(
                        "geometry_informed_flower_support_conservative",
                        "geometry_informed_warp",
                        _geometry_informed_flower_factory(
                            quick=quick,
                            domain_support=True,
                            conserve_mass=True,
                        ),
                        True,
                        normalization="sourcewise",
                        promotion_scope="specialized",
                        configuration_factory=(
                            _geometry_informed_flower_configuration(
                                quick=quick,
                                domain_support=True,
                                conserve_mass=True,
                            )
                        ),
                    )
                )
        candidates.append(
            OperatorArchitecture(
                "rigno",
                "regional_graph",
                _rigno_factory(quick=quick),
                True,
                normalization="sourcewise",
                promotion_scope="general",
                configuration_factory=_rigno_configuration(quick=quick),
            )
        )
        if _coordinate_dimension(scenario.train_batch.require_single_query()) == 2:
            candidates.append(
                OperatorArchitecture(
                    "gaot",
                    "geometry_transformer",
                    _gaot_factory(quick=quick),
                    True,
                    normalization="sourcewise",
                    promotion_scope="general",
                    configuration_factory=_gaot_configuration(quick=quick),
                )
            )
        if _field_channels_compatible(scenario):
            candidates.append(
                OperatorArchitecture(
                    "gnot",
                    "heterogeneous_geometry_transformer",
                    _gnot_factory(quick=quick),
                    True,
                    normalization="sourcewise",
                    promotion_scope="general",
                )
            )
            if len(scenario.train_batch.inputs) == 1:
                candidates.extend(
                    (
                        OperatorArchitecture(
                            "transolver",
                            "physics_attention",
                            _transolver_factory(quick=quick),
                            True,
                            normalization="sourcewise",
                            promotion_scope="general",
                        ),
                        OperatorArchitecture(
                            "upt",
                            "latent_physics_transformer",
                            _upt_factory(quick=quick),
                            True,
                            normalization="sourcewise",
                            promotion_scope="general",
                        ),
                    )
                )
    if _coincident(scenario) and len(scenario.train_batch.inputs) == 1:
        candidates.extend(
            (
                OperatorArchitecture(
                    "identity",
                    "baseline",
                    lambda current, seed, scale: IdentityBaseline(source_name),
                    False,
                    normalization="none",
                    promotion_scope="reference",
                ),
                OperatorArchitecture(
                    "pointwise_affine",
                    "baseline",
                    lambda current, seed, scale: PointwiseAffineBaseline(
                        source_name,
                        key=jr.key(seed),
                    ),
                    True,
                    normalization="sourcewise",
                    promotion_scope="reference",
                ),
            )
        )
    if source.axes and _coincident(scenario):
        candidates.extend(
            (
                OperatorArchitecture(
                    "fno",
                    "spectral",
                    _fno_factory(factorization="dense", quick=quick),
                    True,
                    normalization="spectral",
                    promotion_scope="specialized",
                    configuration_factory=_fno_configuration(
                        factorization="dense",
                        quick=quick,
                    ),
                ),
                OperatorArchitecture(
                    "tfno",
                    "spectral",
                    _fno_factory(factorization="tucker", quick=quick),
                    True,
                    normalization="spectral",
                    promotion_scope="specialized",
                    configuration_factory=_fno_configuration(
                        factorization="tucker",
                        quick=quick,
                    ),
                ),
                OperatorArchitecture(
                    "cno",
                    "multiresolution",
                    _cno_factory(uno=False, quick=quick),
                    True,
                    normalization="sourcewise",
                    promotion_scope="specialized",
                ),
                OperatorArchitecture(
                    "uno",
                    "multiresolution",
                    _cno_factory(uno=True, quick=quick),
                    True,
                    normalization="sourcewise",
                    promotion_scope="specialized",
                ),
            )
        )
        if "spherical_field" in scenario.regimes:
            candidates.append(
                OperatorArchitecture(
                    "sfno",
                    "spherical_spectral",
                    _sfno_factory(quick=quick),
                    True,
                    normalization="spectral",
                    promotion_scope="specialized",
                )
            )
    if _coincident_tensor_grid_compatible(scenario):
        if _lattice_equivariant_cno_compatible(scenario):
            candidates.append(
                OperatorArchitecture(
                    "lattice_equivariant_cno",
                    "lattice_equivariant",
                    _lattice_equivariant_cno_factory(quick=quick),
                    True,
                    normalization="none",
                    promotion_scope="specialized",
                    configuration_factory=_lattice_equivariant_cno_configuration(
                        quick=quick
                    ),
                )
            )
        candidates.append(
            OperatorArchitecture(
                "ifno",
                "implicit_spectral",
                _implicit_fno_factory(axial=False, quick=quick),
                True,
                normalization="spectral",
                promotion_scope="specialized",
            )
        )
        if len(source.axes) >= 2:
            candidates.append(
                OperatorArchitecture(
                    "axial_factorized_fno",
                    "axial_spectral",
                    _implicit_fno_factory(axial=True, quick=quick),
                    True,
                    normalization="spectral",
                    promotion_scope="specialized",
                )
            )
        if "polynomial_nonlinearity" in scenario.regimes:
            candidates.extend(
                (
                    OperatorArchitecture(
                        "hofno_order1",
                        "higher_order_spectral",
                        _hofno_factory(
                            interaction_order=1,
                            aliasing="collocation",
                            quick=quick,
                        ),
                        True,
                        normalization="spectral",
                        promotion_scope="specialized",
                        configuration_factory=_hofno_configuration(
                            interaction_order=1,
                            aliasing="collocation",
                            quick=quick,
                        ),
                        capability_name="HOFNO",
                    ),
                    OperatorArchitecture(
                        "hofno_order2_collocation",
                        "higher_order_spectral",
                        _hofno_factory(
                            interaction_order=2,
                            aliasing="collocation",
                            quick=quick,
                        ),
                        True,
                        normalization="spectral",
                        promotion_scope="specialized",
                        configuration_factory=_hofno_configuration(
                            interaction_order=2,
                            aliasing="collocation",
                            quick=quick,
                        ),
                        capability_name="HOFNO",
                    ),
                    OperatorArchitecture(
                        "hofno_order2_dealiased",
                        "higher_order_spectral",
                        _hofno_factory(
                            interaction_order=2,
                            aliasing="dealiased",
                            quick=quick,
                        ),
                        True,
                        normalization="spectral",
                        promotion_scope="specialized",
                        configuration_factory=_hofno_configuration(
                            interaction_order=2,
                            aliasing="dealiased",
                            quick=quick,
                        ),
                        capability_name="HOFNO",
                    ),
                )
            )
        if (
            scenario.symmetry is not None
            and scenario.symmetry.group in ("p4", "p4m")
            and len(scenario.train_batch.inputs) == 1
            and len(source.sample_shape) == 2
        ):
            candidates.append(
                OperatorArchitecture(
                    "fno_p4_augmented",
                    "spectral_augmented",
                    _fno_factory(factorization="dense", quick=quick),
                    True,
                    normalization="spectral",
                    promotion_scope="specialized",
                    configuration_factory=_fno_configuration(
                        factorization="dense",
                        quick=quick,
                        augmentation="p4",
                    ),
                    training_scenario_factory=lambda current: (
                        augment_square_group_training(current, group="p4")
                    ),
                    capability_name="FNO",
                )
            )
        if _poseidon_compatible(scenario):
            candidates.append(
                OperatorArchitecture(
                    "poseidon",
                    "multiscale_transformer",
                    _poseidon_factory(quick=quick),
                    True,
                    normalization="sourcewise",
                    promotion_scope="specialized",
                )
            )
        wavelet_levels = 2 if quick else 3
        if _wavelet_tensor_grid_compatible(
            scenario,
            levels=wavelet_levels,
        ):
            candidates.append(
                OperatorArchitecture(
                    "wavelet",
                    "wavelet",
                    _wavelet_factory(quick=quick),
                    True,
                    normalization="spectral",
                    promotion_scope="specialized",
                )
            )
            if len(source.sample_shape) == 1:
                candidates.append(
                    OperatorArchitecture(
                        "multiwavelet",
                        "multiwavelet",
                        _multiwavelet_factory(quick=quick),
                        True,
                        normalization="spectral",
                        promotion_scope="specialized",
                    )
                )
        flower_levels = 2 if quick else 4
        if _flower_learned_compatible(scenario, levels=1):
            candidates.append(
                OperatorArchitecture(
                    "flower_one_level",
                    "learned_warp",
                    _flower_factory(
                        quick=quick,
                        levels=1,
                        transition_mode="learned",
                    ),
                    True,
                    normalization="sourcewise",
                    promotion_scope="specialized",
                    configuration_factory=_flower_configuration(
                        quick=quick,
                        levels=1,
                        transition_mode="learned",
                    ),
                )
            )
        if _flower_learned_compatible(scenario, levels=flower_levels):
            candidates.append(
                OperatorArchitecture(
                    "flower_multilevel",
                    "learned_warp",
                    _flower_factory(
                        quick=quick,
                        levels=flower_levels,
                        transition_mode="learned",
                    ),
                    True,
                    normalization="sourcewise",
                    promotion_scope="specialized",
                    configuration_factory=_flower_configuration(
                        quick=quick,
                        levels=flower_levels,
                        transition_mode="learned",
                    ),
                )
            )
        if _flower_resolution_compatible(scenario, levels=flower_levels):
            candidates.append(
                OperatorArchitecture(
                    "flower_resolution_consistent",
                    "learned_warp",
                    _flower_factory(
                        quick=quick,
                        levels=flower_levels,
                        transition_mode="resolution_consistent",
                    ),
                    True,
                    normalization="sourcewise",
                    promotion_scope="specialized",
                    configuration_factory=_flower_configuration(
                        quick=quick,
                        levels=flower_levels,
                        transition_mode="resolution_consistent",
                    ),
                )
            )
    if (
        "causal_transient" in scenario.regimes
        and len(scenario.train_batch.inputs) == 1
        and len(source.sample_shape) == 1
    ):
        candidates.append(
            OperatorArchitecture(
                "laplace",
                "laplace_temporal",
                _laplace_factory(quick=quick),
                True,
                normalization="sourcewise",
                promotion_scope="specialized",
            )
        )
        if _coincident(scenario):
            candidates.append(
                OperatorArchitecture(
                    "linear_recurrent",
                    "linear_recurrent",
                    _linear_recurrent_factory(quick=quick),
                    True,
                    normalization="sourcewise",
                    promotion_scope="specialized",
                    configuration_factory=_linear_recurrent_configuration(quick=quick),
                )
            )
            candidates.append(
                OperatorArchitecture(
                    "selective_state_space",
                    "selective_state_space",
                    _selective_state_space_factory(quick=quick),
                    True,
                    normalization="sourcewise",
                    promotion_scope="specialized",
                    configuration_factory=_selective_state_space_configuration(
                        quick=quick
                    ),
                )
            )
    return tuple(
        architecture
        for architecture in candidates
        if architecture.runtime_compatible(scenario)
    )


__all__ = [
    "ConstantOutputBaseline",
    "IdentityBaseline",
    "LinearInterpolationBaseline",
    "NearestNeighborBaseline",
    "OperatorArchitecture",
    "PODLinearROMBaseline",
    "PointwiseAffineBaseline",
    "WeightedMeanBaseline",
    "compatible_architectures",
]
