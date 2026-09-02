from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax.ein as ein
from phydrax.discretization import CochainFieldSpec, SphericalHarmonicPlan
from phydrax.graph import compute_harmonic_subspace, triangle_mesh_to_cochain_complex
from phydrax.nn.operator import (
    function_samples_from_cochain,
    FunctionSamples,
    OperatorAxis,
    OperatorBatch,
    OperatorFieldSpec,
    OperatorProblemSpec,
    OperatorQuerySpec,
    OperatorTargetBatch,
    OperatorTask,
)


SquareSymmetryGroup = Literal["p4", "p4m"]
SquareFieldRepresentation = Literal["scalar", "pseudoscalar"]


def _apply_square_group_action(
    values,
    element: int,
    /,
    *,
    group: SquareSymmetryGroup,
    representation: SquareFieldRepresentation = "scalar",
    spatial_axes: tuple[int, int] = (-2, -1),
):
    """Apply a discrete square-group action for benchmark augmentation and audits."""

    if group not in ("p4", "p4m"):
        raise ValueError("group must be 'p4' or 'p4m'.")
    element_count = 4 if group == "p4" else 8
    if not 0 <= int(element) < element_count:
        raise ValueError("Square-group element index is out of range.")
    if representation not in ("scalar", "pseudoscalar"):
        raise ValueError("representation must be 'scalar' or 'pseudoscalar'.")

    array = jnp.asarray(values)
    first = int(spatial_axes[0]) % array.ndim
    second = int(spatial_axes[1]) % array.ndim
    if first == second:
        raise ValueError("spatial_axes must identify two distinct axes.")
    reflection = int(element) >= 4
    transformed = jnp.flip(array, axis=first) if reflection else array
    transformed = jnp.rot90(
        transformed,
        k=int(element) % 4,
        axes=(first, second),
    )
    character = -1 if representation == "pseudoscalar" and reflection else 1
    return transformed * character


@dataclass(frozen=True)
class OperatorParameterRange:
    """Declared physical parameter range for a generated benchmark population."""

    name: str
    minimum: float
    maximum: float
    unit: str
    scale: Literal["linear", "log"] = "linear"

    def __post_init__(self):
        if not self.name:
            raise ValueError("Parameter names must be non-empty.")
        if float(self.minimum) > float(self.maximum):
            raise ValueError("Parameter range minimum cannot exceed its maximum.")
        if self.scale not in ("linear", "log"):
            raise ValueError("Parameter range scale must be 'linear' or 'log'.")
        if self.scale == "log" and float(self.minimum) <= 0.0:
            raise ValueError("Log-scaled parameter ranges must be strictly positive.")


@dataclass(frozen=True)
class OperatorDatasetProvenance:
    """Reproducibility record for a generated or externally sourced scenario."""

    source_uri: str
    generator: str
    generator_version: str
    license_id: str
    citation: str

    def __post_init__(self):
        values = (
            self.source_uri,
            self.generator,
            self.generator_version,
            self.license_id,
            self.citation,
        )
        if any(not value.strip() for value in values):
            raise ValueError("Dataset provenance fields must be non-empty.")


@dataclass(frozen=True)
class ReferenceSolverEvidence:
    """Numerical evidence that a scenario's target generator is converged."""

    method: str
    verification: Literal["analytic", "discrete_residual", "refinement"]
    resolutions: tuple[int, ...]
    relative_error: float
    tolerance: float

    def __post_init__(self):
        if not self.method:
            raise ValueError("Reference solver method must be non-empty.")
        if self.verification not in (
            "analytic",
            "discrete_residual",
            "refinement",
        ):
            raise ValueError("Unknown reference solver verification mode.")
        if not self.resolutions or any(int(size) <= 0 for size in self.resolutions):
            raise ValueError("Reference solver resolutions must be positive.")
        if float(self.relative_error) < 0.0 or float(self.tolerance) < 0.0:
            raise ValueError("Reference errors and tolerances must be non-negative.")

    @property
    def passed(self) -> bool:
        return float(self.relative_error) <= float(self.tolerance)


@dataclass(frozen=True)
class OperatorSymmetrySpec:
    """Exact and diagnostic square-group semantics for one physical scenario."""

    group: Literal["p4", "p4m"] | None
    audit_group: Literal["p4", "p4m"]
    source_representations: tuple[tuple[str, Literal["scalar", "pseudoscalar"]], ...]
    target_representation: Literal["scalar", "pseudoscalar"]
    spatial_axes: tuple[str, str] = ("x", "y")
    action_convention: str = "pullback by inverse spatial action"
    reference_tolerance: float = 1e-10
    reference_defects: tuple[tuple[int, float], ...] = ()
    intentionally_violated: bool = False

    def __post_init__(self):
        if self.group is not None and self.group not in ("p4", "p4m"):
            raise ValueError("Square symmetry group must be 'p4', 'p4m', or None.")
        if self.audit_group not in ("p4", "p4m"):
            raise ValueError("Square symmetry audit_group must be 'p4' or 'p4m'.")
        if self.group == "p4m" and self.audit_group != "p4m":
            raise ValueError("A p4m physical symmetry requires a p4m audit group.")
        names = tuple(name for name, _ in self.source_representations)
        if not names or len(set(names)) != len(names):
            raise ValueError(
                "Symmetry source representations must be non-empty and unique."
            )
        if len(self.spatial_axes) != 2 or len(set(self.spatial_axes)) != 2:
            raise ValueError("Square symmetry requires two distinct spatial axes.")
        if not self.action_convention.strip():
            raise ValueError("Symmetry action_convention must be non-empty.")
        if float(self.reference_tolerance) < 0.0:
            raise ValueError("Symmetry reference_tolerance must be non-negative.")
        expected = 4 if self.audit_group == "p4" else 8
        indices = tuple(index for index, _ in self.reference_defects)
        if self.reference_defects and indices != tuple(range(expected)):
            raise ValueError(
                "Symmetry reference_defects must cover every audit-group element in order."
            )
        if any(float(defect) < 0.0 for _, defect in self.reference_defects):
            raise ValueError("Symmetry reference defects must be non-negative.")

    @property
    def exact_element_count(self) -> int:
        if self.group is None:
            return 0
        return 4 if self.group == "p4" else 8

    @property
    def reference_mean_defect(self) -> float | None:
        if not self.reference_defects:
            return None
        return float(np.mean([defect for _, defect in self.reference_defects]))

    @property
    def reference_worst_defect(self) -> float | None:
        if not self.reference_defects:
            return None
        return float(max(defect for _, defect in self.reference_defects))


def _generated_provenance(generator: str) -> OperatorDatasetProvenance:
    return OperatorDatasetProvenance(
        source_uri="internal://tools.operator_benchmarks.scenarios",
        generator=generator,
        generator_version="2",
        license_id="PhydraX repository license",
        citation="PhydraX Operator Benchmark v2",
    )


def _case_ids(name: str, count: int) -> tuple[str, ...]:
    return tuple(f"{name}:realization:{index:06d}" for index in range(int(count)))


@dataclass(frozen=True)
class OperatorBenchmarkEvaluation:
    name: str
    batch: OperatorBatch
    target: jax.Array | OperatorTargetBatch
    split: str = "test"
    shift: str = "in_distribution"
    rollout_steps: int = 1
    rollout_source_key: str | None = None
    case_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperatorBenchmarkScenario:
    name: str
    train_batch: OperatorBatch
    train_target: jax.Array | OperatorTargetBatch
    evaluations: tuple[OperatorBenchmarkEvaluation, ...]
    validation: OperatorBenchmarkEvaluation | None = None
    seed: int = 0
    metadata: tuple[tuple[str, str], ...] = ()
    case_ids: tuple[str, ...] = ()
    provenance: OperatorDatasetProvenance | None = None
    dimensional_parameters: tuple[OperatorParameterRange, ...] = ()
    nondimensional_parameters: tuple[OperatorParameterRange, ...] = ()
    reference_evidence: ReferenceSolverEvidence | None = None
    regimes: tuple[str, ...] = ()
    ladder: str = ""
    difficulty: str = ""
    domain_support_key: str | None = None
    domain_support_kind: Literal["occupancy", "sdf"] | None = None
    domain_support_threshold: float | None = None
    conservation_source_key: str | None = None
    symmetry: OperatorSymmetrySpec | None = None
    task: OperatorTask | None = None

    def __post_init__(self):
        task = self.task
        if task is not None:
            expected_targets = {field.name for field in task.target_fields}

            def validate_task_data(
                batch: OperatorBatch,
                target: jax.Array | OperatorTargetBatch,
            ) -> None:
                task.validate_batch(batch)
                if not isinstance(target, OperatorTargetBatch):
                    raise TypeError(
                        "Task-backed benchmark scenarios require OperatorTargetBatch targets."
                    )
                target.validate(batch)
                if set(target.fields) != expected_targets:
                    raise ValueError(
                        "Benchmark target names must match the task target fields."
                    )

            validate_task_data(self.train_batch, self.train_target)
            for evaluation in self.evaluations:
                validate_task_data(evaluation.batch, evaluation.target)
            if self.validation is not None:
                validate_task_data(self.validation.batch, self.validation.target)
        if self.domain_support_key is None:
            if (
                self.domain_support_kind is not None
                or self.domain_support_threshold is not None
            ):
                raise ValueError(
                    "Domain support kind and threshold require domain_support_key."
                )
        else:
            if self.domain_support_kind not in ("occupancy", "sdf"):
                raise ValueError(
                    "A domain support source requires kind 'occupancy' or 'sdf'."
                )
            if self.domain_support_key not in self.train_batch.inputs:
                raise ValueError("domain_support_key must name a training-batch input.")
            if (
                self.domain_support_kind == "occupancy"
                and self.domain_support_threshold is not None
                and not 0.0 <= float(self.domain_support_threshold) <= 1.0
            ):
                raise ValueError("Occupancy support thresholds must lie in [0, 1].")
        if (
            self.conservation_source_key is not None
            and self.conservation_source_key not in self.train_batch.inputs
        ):
            raise ValueError("conservation_source_key must name a training-batch input.")
        if self.symmetry is not None:
            source_names = tuple(name for name, _ in self.symmetry.source_representations)
            if any(name not in self.train_batch.inputs for name in source_names):
                raise ValueError(
                    "Symmetry source representations must name training-batch inputs."
                )
            primary = self.train_batch.input(source_names[0])
            axis_names = tuple(axis.name for axis in primary.axes)
            if any(name not in axis_names for name in self.symmetry.spatial_axes):
                raise ValueError(
                    "Symmetry spatial axes must name axes on the primary source."
                )


def _periodic_axis(name: str, size: int) -> OperatorAxis:
    return OperatorAxis(
        name,
        jnp.linspace(0.0, 1.0, size, endpoint=False),
        quadrature_weights=jnp.full((size,), 1.0 / size),
        basis="fourier",
        periodic=True,
    )


def _interval_axis(name: str, size: int) -> OperatorAxis:
    nodes = jnp.linspace(0.0, 1.0, size)
    weights = jnp.ones((size,)).at[jnp.asarray([0, size - 1], dtype=jnp.int32)].set(
        0.5
    ) / (size - 1)
    return OperatorAxis(name, nodes, quadrature_weights=weights)


def _grid_batch(inputs, query_axes, *, case_axis="case") -> OperatorBatch:
    return OperatorBatch(
        inputs=inputs,
        queries={"query": FunctionSamples(values=None, axes=query_axes)},
        case_axes=(case_axis,),
    )


def _burgers_step(values, viscosity, dt):
    values_array = np.asarray(jax.device_get(values))
    size = int(values_array.shape[-1])
    frequencies = 2.0 * np.pi * np.fft.fftfreq(size, d=1.0 / size)
    transformed = np.fft.fft(values_array, axis=-1)
    gradient = np.fft.ifft(1j * frequencies * transformed, axis=-1).real
    nonlinear = -values_array * gradient
    return jnp.asarray(
        np.fft.ifft(
            (transformed + dt * np.fft.fft(nonlinear, axis=-1))
            / (1.0 + viscosity * dt * frequencies**2),
            axis=-1,
        ).real
    )


def _burgers_step_residual(values, target, viscosity, dt) -> float:
    values_array = np.asarray(jax.device_get(values))
    target_array = np.asarray(jax.device_get(target))
    size = int(values_array.shape[-1])
    frequencies = 2.0 * np.pi * np.fft.fftfreq(size, d=1.0 / size)
    transformed = np.fft.fft(values_array, axis=-1)
    gradient = np.fft.ifft(1j * frequencies * transformed, axis=-1).real
    right = transformed + dt * np.fft.fft(-values_array * gradient, axis=-1)
    left = (1.0 + viscosity * dt * frequencies**2) * np.fft.fft(
        target_array,
        axis=-1,
    )
    denominator = max(float(np.linalg.norm(right)), 1e-12)
    return float(np.linalg.norm(left - right) / denominator)


def _periodic_population_coefficients(key, num_cases, resolved_frequency):
    frequencies = jnp.arange(1, int(resolved_frequency) + 1, dtype=float)
    coefficients = jr.normal(
        key,
        (int(num_cases), int(resolved_frequency), 2),
    )
    coefficients = coefficients / (
        jnp.sqrt(frequencies)[None, :, None] * jnp.sqrt(float(resolved_frequency))
    )
    return frequencies, coefficients


def _evaluate_periodic_population(coefficients, frequencies, phase_coordinate):
    phase = frequencies[:, None] * phase_coordinate[None, :]
    basis = jnp.stack((jnp.sin(phase), jnp.cos(phase)), axis=-1)
    return ein.contract("cmk,mpk->cp", coefficients, basis)


def _planar_population_coefficients(key, num_cases, resolved_frequency):
    mode_x, mode_y = jnp.meshgrid(
        jnp.arange(int(resolved_frequency) + 1),
        jnp.arange(int(resolved_frequency) + 1),
        indexing="ij",
    )
    mode_x = mode_x.reshape(-1)[1:]
    mode_y = mode_y.reshape(-1)[1:]
    mode_scale = jnp.reciprocal(
        jnp.sqrt(mode_x.astype(float) ** 2 + mode_y.astype(float) ** 2)
    )
    coefficients = jr.normal(
        key,
        (int(num_cases), int(mode_x.shape[0]), 2),
    )
    coefficients = (
        coefficients * mode_scale[None, :, None] / jnp.sqrt(float(mode_x.shape[0]))
    )
    return mode_x, mode_y, coefficients


def _evaluate_planar_population(coefficients, mode_x, mode_y, coordinates):
    flattened = coordinates.reshape((-1, 2))
    phase = (
        2.0
        * jnp.pi
        * (
            mode_x[:, None] * flattened[None, :, 0]
            + mode_y[:, None] * flattened[None, :, 1]
        )
    )
    basis = jnp.stack((jnp.sin(phase), jnp.cos(phase)), axis=-1)
    values = ein.contract("cmk,mpk->cp", coefficients, basis)
    return values.reshape((int(coefficients.shape[0]), *coordinates.shape[:-1]))


def _square_symmetry_reference_defects(
    values,
    operator,
    /,
    *,
    group: Literal["p4", "p4m"],
    source_representation: Literal["scalar", "pseudoscalar"],
    target_representation: Literal["scalar", "pseudoscalar"],
) -> tuple[tuple[int, float], ...]:
    target = operator(values)
    count = 4 if group == "p4" else 8
    defects = []
    for element in range(count):
        transformed_values = _apply_square_group_action(
            values,
            element,
            group=group,
            representation=source_representation,
            spatial_axes=(-2, -1),
        )
        transformed_target = operator(transformed_values)
        expected_target = _apply_square_group_action(
            target,
            element,
            group=group,
            representation=target_representation,
            spatial_axes=(-2, -1),
        )
        denominator = jnp.maximum(jnp.linalg.norm(expected_target), 1e-12)
        defect = jnp.linalg.norm(transformed_target - expected_target) / denominator
        defects.append((element, float(defect)))
    return tuple(defects)


def augment_square_group_training(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    group: Literal["p4", "p4m"] = "p4",
) -> OperatorBenchmarkScenario:
    """Augment only the post-split training realizations by exact group actions."""
    symmetry = scenario.symmetry
    if symmetry is None:
        raise ValueError(
            "Square-group augmentation requires declared symmetry semantics."
        )
    if len(scenario.train_batch.case_shape) != 1:
        raise ValueError("Square-group augmentation requires one training case axis.")
    if scenario.train_batch.require_single_query().geometry_case_shape:
        raise ValueError(
            "Square-group augmentation requires case-independent query geometry."
        )
    case_count = int(scenario.train_batch.case_shape[0])
    element_count = 4 if group == "p4" else 8
    source_representations = dict(symmetry.source_representations)

    def augment_values(values, *, representation: SquareFieldRepresentation | None):
        if values is None:
            return None

        def augment_leaf(leaf):
            array = jnp.asarray(leaf)
            if int(array.shape[0]) != case_count:
                raise ValueError(
                    "Square-group augmentation requires case-leading source values."
                )
            if representation is None:
                return jnp.concatenate((array,) * element_count, axis=0)
            return jnp.concatenate(
                tuple(
                    _apply_square_group_action(
                        array,
                        element,
                        group=group,
                        representation=representation,
                        spatial_axes=(1, 2),
                    )
                    for element in range(element_count)
                ),
                axis=0,
            )

        return jax.tree_util.tree_map(augment_leaf, values)

    inputs = {}
    for name, samples in scenario.train_batch.inputs.items():
        if (
            samples.coordinates is not None
            or samples.topology is not None
            or samples.geometry_case_shape
        ):
            raise ValueError(
                "Square-group augmentation currently requires shared tensor-grid "
                "source geometry."
            )
        inputs[name] = FunctionSamples(
            values=augment_values(
                samples.values,
                representation=source_representations.get(name),
            ),
            axes=samples.axes,
            quadrature_weights=samples.quadrature_weights,
            mask=samples.mask,
        )
    train_batch = OperatorBatch(
        inputs=inputs,
        queries={"query": scenario.train_batch.require_single_query()},
        case_axes=scenario.train_batch.case_axes,
        case_shape=(case_count * element_count,),
    )
    train_target = augment_values(
        scenario.train_target,
        representation=symmetry.target_representation,
    )
    case_ids = tuple(
        f"{case_id}::{group}:{element}"
        for element in range(element_count)
        for case_id in scenario.case_ids
    )
    return replace(
        scenario,
        train_batch=train_batch,
        train_target=train_target,
        case_ids=case_ids,
        metadata=scenario.metadata
        + (
            ("training_augmentation", group),
            ("base_train_cases", str(case_count)),
            ("augmented_train_cases", str(case_count * element_count)),
        ),
    )


def _characteristic_translation_error(
    initial,
    target,
    *,
    phase_coordinate,
    density_jacobian,
    resolved_frequency: int,
    translation: float,
) -> float:
    """Recover the source modes and independently verify a characteristic shift."""
    initial_array = np.asarray(jax.device_get(initial), dtype=np.float64)
    target_array = np.asarray(jax.device_get(target), dtype=np.float64)
    phase = np.asarray(jax.device_get(phase_coordinate), dtype=np.float64)
    jacobian = np.asarray(jax.device_get(density_jacobian), dtype=np.float64)
    frequencies = np.arange(1, int(resolved_frequency) + 1, dtype=np.float64)

    def design(current_phase):
        mode_phase = current_phase[:, None] * frequencies[None, :]
        return np.concatenate(
            (
                np.ones((current_phase.shape[0], 1), dtype=np.float64),
                np.sin(mode_phase),
                np.cos(mode_phase),
            ),
            axis=1,
        )

    source_design = design(phase)
    shifted_design = design(phase - 2.0 * np.pi * float(translation))
    normalized_source = initial_array / jacobian[None, :]
    recovered, *_ = np.linalg.lstsq(
        source_design,
        normalized_source.T,
        rcond=None,
    )
    reconstructed = (shifted_design @ recovered).T * jacobian[None, :]
    denominator = max(float(np.linalg.norm(target_array)), 1e-12)
    return float(np.linalg.norm(reconstructed - target_array) / denominator)


def periodic_advection_scenario(
    *,
    train_resolution: int = 32,
    test_resolution: int = 48,
    num_cases: int = 8,
    speed_configuration: Literal["constant", "variable"] = "constant",
    speed: float = 1.0,
    speed_variation: float | None = None,
    dt: float = 0.05,
    target_steps: int = 2,
    rollout_steps: int = 1,
    maximum_frequency: int = 6,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    """Exact smooth periodic conservative advection in characteristic coordinates."""
    if min(int(train_resolution), int(test_resolution)) < 4:
        raise ValueError("Advection resolutions must be at least four.")
    if int(num_cases) <= 0:
        raise ValueError("num_cases must be positive.")
    if speed_configuration not in ("constant", "variable"):
        raise ValueError("speed_configuration must be 'constant' or 'variable'.")
    if float(speed) <= 0.0:
        raise ValueError("speed must be positive.")
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive.")
    if int(target_steps) <= 0 or int(rollout_steps) <= 0:
        raise ValueError("target_steps and rollout_steps must be positive.")
    if int(maximum_frequency) <= 0:
        raise ValueError("maximum_frequency must be positive.")

    if speed_variation is None:
        resolved_variation = 0.0 if speed_configuration == "constant" else 0.35
    else:
        resolved_variation = float(speed_variation)
    if speed_configuration == "constant" and resolved_variation != 0.0:
        raise ValueError("Constant-speed advection requires speed_variation=0.")
    if speed_configuration == "variable" and not 0.0 < resolved_variation < 1.0:
        raise ValueError("Variable-speed advection requires speed_variation in (0, 1).")

    resolved_frequency = min(
        int(maximum_frequency),
        max(1, (min(int(train_resolution), int(test_resolution)) - 1) // 2),
    )
    frequencies, coefficients = _periodic_population_coefficients(
        jr.key(seed),
        num_cases,
        resolved_frequency,
    )
    offsets = jnp.linspace(0.25, 0.75, int(num_cases))[:, None]
    scenario_name = (
        "periodic_advection_constant_speed_1d"
        if speed_configuration == "constant"
        else "periodic_advection_variable_speed_1d"
    )
    case_ids = _case_ids(scenario_name, num_cases)
    horizon = float(dt) * int(target_steps)
    translation = float(speed) * horizon

    def build(resolution, elapsed_time):
        axis = _periodic_axis("x", resolution)
        spatial_phase = 2.0 * jnp.pi * axis.nodes
        phase_coordinate = spatial_phase + resolved_variation * jnp.sin(spatial_phase)
        density_jacobian = 1.0 + resolved_variation * jnp.cos(spatial_phase)
        initial_profile = offsets + _evaluate_periodic_population(
            coefficients,
            frequencies,
            phase_coordinate,
        )
        shifted_profile = offsets + _evaluate_periodic_population(
            coefficients,
            frequencies,
            phase_coordinate - 2.0 * jnp.pi * float(speed) * float(elapsed_time),
        )
        initial = density_jacobian[None, :] * initial_profile
        target = density_jacobian[None, :] * shifted_profile
        batch = _grid_batch(
            {"state": FunctionSamples(values=initial, axes=(axis,))},
            (axis,),
        )
        error = _characteristic_translation_error(
            initial,
            target,
            phase_coordinate=phase_coordinate,
            density_jacobian=density_jacobian,
            resolved_frequency=resolved_frequency,
            translation=float(speed) * float(elapsed_time),
        )
        mass_scale = max(float(np.max(np.abs(np.asarray(offsets)))), 1e-12)
        mass_drift = float(
            np.max(
                np.abs(
                    np.mean(np.asarray(jax.device_get(target)), axis=-1)
                    - np.mean(np.asarray(jax.device_get(initial)), axis=-1)
                )
            )
            / mass_scale
        )
        return batch, target, error, mass_drift

    train_batch, train_target, train_error, train_mass_drift = build(
        train_resolution,
        horizon,
    )
    test_batch, test_target, test_error, test_mass_drift = build(
        test_resolution,
        horizon,
    )
    evaluations = [
        OperatorBenchmarkEvaluation(
            "train_resolution",
            train_batch,
            train_target,
            case_ids=case_ids,
        ),
        OperatorBenchmarkEvaluation(
            "higher_resolution",
            test_batch,
            test_target,
            shift="resolution",
            case_ids=case_ids,
        ),
    ]
    maximum_error = max(train_error, test_error)
    maximum_mass_drift = max(train_mass_drift, test_mass_drift)
    if int(rollout_steps) > 1:
        rollout_time = horizon * int(rollout_steps)
        _, rollout_target, rollout_error, rollout_mass_drift = build(
            test_resolution,
            rollout_time,
        )
        maximum_error = max(maximum_error, rollout_error)
        maximum_mass_drift = max(maximum_mass_drift, rollout_mass_drift)
        evaluations.append(
            OperatorBenchmarkEvaluation(
                "long_rollout",
                test_batch,
                rollout_target,
                shift="rollout",
                rollout_steps=int(rollout_steps),
                rollout_source_key="state",
                case_ids=case_ids,
            )
        )

    minimum_speed = float(speed) / (1.0 + resolved_variation)
    maximum_speed = float(speed) / (1.0 - resolved_variation)
    courant_numbers = (
        minimum_speed * float(dt) * min(int(train_resolution), int(test_resolution)),
        maximum_speed * float(dt) * max(int(train_resolution), int(test_resolution)),
    )
    return OperatorBenchmarkScenario(
        scenario_name,
        train_batch,
        train_target,
        tuple(evaluations),
        seed=int(seed),
        metadata=(
            ("equation", "u_t + d_x(a(x) u) = 0"),
            ("boundary", "periodic_unit_interval"),
            ("speed_configuration", speed_configuration),
            (
                "speed_law",
                "a(x)=speed/(1+variation*cos(2*pi*x))",
            ),
            (
                "characteristic_coordinate",
                "xi(x)=x+variation*sin(2*pi*x)/(2*pi)",
            ),
            ("speed", str(float(speed))),
            ("speed_variation", str(resolved_variation)),
            ("minimum_speed", str(minimum_speed)),
            ("maximum_speed", str(maximum_speed)),
            ("dt", str(float(dt))),
            ("target_steps", str(int(target_steps))),
            ("rollout_steps", str(int(rollout_steps))),
            ("time_horizon", str(horizon)),
            ("characteristic_translation", str(translation)),
            (
                "rollout_characteristic_translation",
                str(translation * int(rollout_steps)),
            ),
            ("train_resolution", str(int(train_resolution))),
            ("test_resolution", str(int(test_resolution))),
            ("maximum_frequency", str(int(maximum_frequency))),
            ("resolved_frequency", str(int(resolved_frequency))),
            ("maximum_relative_mass_drift", str(maximum_mass_drift)),
            ("population_seed", str(int(seed))),
            (
                "reference",
                "analytic characteristic translation with conservative density Jacobian",
            ),
        ),
        case_ids=case_ids,
        provenance=_generated_provenance("periodic_advection_scenario"),
        dimensional_parameters=(
            OperatorParameterRange(
                "advection_speed",
                minimum_speed,
                maximum_speed,
                "L T^-1",
            ),
            OperatorParameterRange("time_step", float(dt), float(dt), "T"),
            OperatorParameterRange(
                "time_horizon",
                horizon,
                horizon * int(rollout_steps),
                "T",
            ),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "courant_number",
                min(courant_numbers),
                max(courant_numbers),
                "1",
            ),
            OperatorParameterRange(
                "characteristic_translation",
                translation,
                translation * int(rollout_steps),
                "1",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="analytic conservative characteristic-coordinate translation",
            verification="analytic",
            resolutions=(int(train_resolution), int(test_resolution)),
            relative_error=maximum_error,
            tolerance=5e-6,
        ),
        regimes=(
            "smooth_periodic",
            f"{speed_configuration}_speed_advection",
            "conservative_transport",
            "resolution_transfer",
            "long_horizon" if int(rollout_steps) > 1 else "single_step",
        ),
    )


def periodic_acoustic_wave_scenario(
    *,
    train_resolution: int = 32,
    test_resolution: int = 48,
    num_cases: int = 8,
    sound_speed: float = 1.0,
    density: float = 1.0,
    dt: float = 0.05,
    target_steps: int = 2,
    rollout_steps: int = 1,
    maximum_wavenumber: int = 6,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    """Exact two-characteristic solutions of the periodic one-dimensional acoustics."""
    if min(int(train_resolution), int(test_resolution)) < 4:
        raise ValueError("Acoustic resolutions must be at least four.")
    if int(num_cases) <= 0:
        raise ValueError("num_cases must be positive.")
    if float(sound_speed) <= 0.0 or float(density) <= 0.0:
        raise ValueError("sound_speed and density must be positive.")
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive.")
    if int(target_steps) <= 0 or int(rollout_steps) <= 0:
        raise ValueError("target_steps and rollout_steps must be positive.")
    if int(maximum_wavenumber) <= 0:
        raise ValueError("maximum_wavenumber must be positive.")

    resolved_wavenumber = min(
        int(maximum_wavenumber),
        max(1, (min(int(train_resolution), int(test_resolution)) - 1) // 2),
    )
    wavenumbers, right_coefficients = _periodic_population_coefficients(
        jr.key(seed),
        num_cases,
        resolved_wavenumber,
    )
    _, left_coefficients = _periodic_population_coefficients(
        jr.fold_in(jr.key(seed), 1),
        num_cases,
        resolved_wavenumber,
    )
    scenario_name = "periodic_acoustic_wave_1d"
    case_ids = _case_ids(scenario_name, num_cases)
    horizon = float(dt) * int(target_steps)
    translation = float(sound_speed) * horizon
    impedance = float(density) * float(sound_speed)

    def build(resolution, elapsed_time):
        axis = _periodic_axis("x", resolution)
        phase_coordinate = 2.0 * jnp.pi * axis.nodes
        right_initial = _evaluate_periodic_population(
            right_coefficients,
            wavenumbers,
            phase_coordinate,
        )
        left_initial = _evaluate_periodic_population(
            left_coefficients,
            wavenumbers,
            phase_coordinate,
        )
        displacement = 2.0 * jnp.pi * float(sound_speed) * float(elapsed_time)
        right_target = _evaluate_periodic_population(
            right_coefficients,
            wavenumbers,
            phase_coordinate - displacement,
        )
        left_target = _evaluate_periodic_population(
            left_coefficients,
            wavenumbers,
            phase_coordinate + displacement,
        )

        def physical_state(right, left):
            pressure = 0.5 * (right + left)
            velocity = 0.5 * (right - left) / impedance
            return jnp.stack((pressure, velocity), axis=-1)

        initial = physical_state(right_initial, left_initial)
        target = physical_state(right_target, left_target)
        batch = _grid_batch(
            {"state": FunctionSamples(values=initial, axes=(axis,))},
            (axis,),
        )
        right_error = _characteristic_translation_error(
            right_initial,
            right_target,
            phase_coordinate=phase_coordinate,
            density_jacobian=jnp.ones_like(axis.nodes),
            resolved_frequency=resolved_wavenumber,
            translation=float(sound_speed) * float(elapsed_time),
        )
        left_error = _characteristic_translation_error(
            left_initial,
            left_target,
            phase_coordinate=phase_coordinate,
            density_jacobian=jnp.ones_like(axis.nodes),
            resolved_frequency=resolved_wavenumber,
            translation=-float(sound_speed) * float(elapsed_time),
        )
        initial_energy = jnp.mean(
            initial[..., 0] ** 2 / (2.0 * float(density) * float(sound_speed) ** 2)
            + 0.5 * float(density) * initial[..., 1] ** 2,
            axis=-1,
        )
        target_energy = jnp.mean(
            target[..., 0] ** 2 / (2.0 * float(density) * float(sound_speed) ** 2)
            + 0.5 * float(density) * target[..., 1] ** 2,
            axis=-1,
        )
        energy_error = float(
            jnp.max(
                jnp.abs(target_energy - initial_energy)
                / jnp.maximum(jnp.abs(initial_energy), 1e-12)
            )
        )
        return batch, target, max(right_error, left_error), energy_error

    train_batch, train_target, train_error, train_energy_error = build(
        train_resolution,
        horizon,
    )
    test_batch, test_target, test_error, test_energy_error = build(
        test_resolution,
        horizon,
    )
    evaluations = [
        OperatorBenchmarkEvaluation(
            "train_resolution",
            train_batch,
            train_target,
            case_ids=case_ids,
        ),
        OperatorBenchmarkEvaluation(
            "higher_resolution",
            test_batch,
            test_target,
            shift="resolution",
            case_ids=case_ids,
        ),
    ]
    maximum_error = max(
        train_error,
        test_error,
        train_energy_error,
        test_energy_error,
    )
    maximum_energy_error = max(train_energy_error, test_energy_error)
    if int(rollout_steps) > 1:
        rollout_time = horizon * int(rollout_steps)
        _, rollout_target, rollout_error, rollout_energy_error = build(
            test_resolution,
            rollout_time,
        )
        maximum_error = max(maximum_error, rollout_error, rollout_energy_error)
        maximum_energy_error = max(maximum_energy_error, rollout_energy_error)
        evaluations.append(
            OperatorBenchmarkEvaluation(
                "long_rollout",
                test_batch,
                rollout_target,
                shift="rollout",
                rollout_steps=int(rollout_steps),
                rollout_source_key="state",
                case_ids=case_ids,
            )
        )

    acoustic_courant_numbers = (
        float(sound_speed) * float(dt) * min(int(train_resolution), int(test_resolution)),
        float(sound_speed) * float(dt) * max(int(train_resolution), int(test_resolution)),
    )
    return OperatorBenchmarkScenario(
        scenario_name,
        train_batch,
        train_target,
        tuple(evaluations),
        seed=int(seed),
        metadata=(
            (
                "equation",
                "p_t + rho*c^2*v_x = 0; v_t + p_x/rho = 0",
            ),
            ("boundary", "periodic_unit_interval"),
            ("state_channels", "pressure,velocity"),
            ("characteristics", "w_plus=p+rho*c*v; w_minus=p-rho*c*v"),
            ("sound_speed", str(float(sound_speed))),
            ("density", str(float(density))),
            ("acoustic_impedance", str(impedance)),
            ("dt", str(float(dt))),
            ("target_steps", str(int(target_steps))),
            ("rollout_steps", str(int(rollout_steps))),
            ("time_horizon", str(horizon)),
            ("right_phase_translation", str(translation)),
            ("left_phase_translation", str(-translation)),
            (
                "rollout_right_phase_translation",
                str(translation * int(rollout_steps)),
            ),
            (
                "rollout_left_phase_translation",
                str(-translation * int(rollout_steps)),
            ),
            ("train_resolution", str(int(train_resolution))),
            ("test_resolution", str(int(test_resolution))),
            ("maximum_wavenumber", str(int(maximum_wavenumber))),
            ("resolved_wavenumber", str(int(resolved_wavenumber))),
            ("maximum_relative_energy_drift", str(maximum_energy_error)),
            ("population_seed", str(int(seed))),
            (
                "reference",
                "analytic independent left/right Riemann-invariant translation",
            ),
        ),
        case_ids=case_ids,
        provenance=_generated_provenance("periodic_acoustic_wave_scenario"),
        dimensional_parameters=(
            OperatorParameterRange(
                "sound_speed",
                float(sound_speed),
                float(sound_speed),
                "L T^-1",
            ),
            OperatorParameterRange(
                "reference_density",
                float(density),
                float(density),
                "M L^-3",
            ),
            OperatorParameterRange("time_step", float(dt), float(dt), "T"),
            OperatorParameterRange(
                "time_horizon",
                horizon,
                horizon * int(rollout_steps),
                "T",
            ),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "acoustic_courant_number",
                min(acoustic_courant_numbers),
                max(acoustic_courant_numbers),
                "1",
            ),
            OperatorParameterRange(
                "phase_translation_magnitude",
                translation,
                translation * int(rollout_steps),
                "1",
            ),
            OperatorParameterRange(
                "resolved_wavenumber",
                1.0,
                float(resolved_wavenumber),
                "1",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="analytic left/right acoustic Riemann-invariant translation",
            verification="analytic",
            resolutions=(int(train_resolution), int(test_resolution)),
            relative_error=maximum_error,
            tolerance=5e-6,
        ),
        regimes=(
            "periodic_acoustic_wave",
            "bidirectional_wave",
            "energy_conserving",
            "resolution_transfer",
            "long_horizon" if int(rollout_steps) > 1 else "single_step",
        ),
    )


def periodic_burgers_scenario(
    *,
    train_resolution: int = 32,
    test_resolution: int = 48,
    num_cases: int = 8,
    viscosity: float = 0.01,
    dt: float = 0.01,
    rollout_steps: int = 1,
    target_steps: int = 1,
    initial_condition: Literal["smooth", "shock"] = "smooth",
    maximum_frequency: int = 6,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    if min(int(train_resolution), int(test_resolution)) < 4:
        raise ValueError("Burgers resolutions must be at least four.")
    if int(num_cases) <= 0:
        raise ValueError("num_cases must be positive.")
    if float(viscosity) <= 0.0 or float(dt) <= 0.0:
        raise ValueError("viscosity and dt must be positive.")
    if int(target_steps) <= 0:
        raise ValueError("target_steps must be positive.")
    if int(rollout_steps) <= 0:
        raise ValueError("rollout_steps must be positive.")
    if initial_condition not in ("smooth", "shock"):
        raise ValueError("initial_condition must be 'smooth' or 'shock'.")
    if int(maximum_frequency) <= 0:
        raise ValueError("maximum_frequency must be positive.")
    resolved_frequency = min(
        int(maximum_frequency),
        max(1, (min(int(train_resolution), int(test_resolution)) - 1) // 2),
    )
    frequencies, coefficients = _periodic_population_coefficients(
        jr.key(seed),
        num_cases,
        resolved_frequency,
    )
    scenario_name = (
        "periodic_burgers_1d"
        if initial_condition == "smooth"
        else "periodic_burgers_shock_1d"
    )
    case_ids = _case_ids(scenario_name, num_cases)

    def build(resolution):
        axis = _periodic_axis("x", resolution)
        smooth_values = _evaluate_periodic_population(
            coefficients,
            frequencies,
            2.0 * jnp.pi * axis.nodes,
        )
        values = (
            smooth_values
            if initial_condition == "smooth"
            else jnp.tanh(smooth_values / 0.08)
        )
        target = values
        maximum_residual = 0.0
        for _ in range(int(target_steps)):
            next_target = _burgers_step(target, viscosity, dt)
            maximum_residual = max(
                maximum_residual,
                _burgers_step_residual(target, next_target, viscosity, dt),
            )
            target = next_target
        batch = _grid_batch(
            {"state": FunctionSamples(values=values, axes=(axis,))},
            (axis,),
        )
        return batch, target, maximum_residual

    train_batch, train_target, train_residual = build(train_resolution)
    test_batch, test_target, test_residual = build(test_resolution)
    evaluations = [
        OperatorBenchmarkEvaluation(
            "train_resolution",
            train_batch,
            train_target,
            case_ids=case_ids,
        ),
        OperatorBenchmarkEvaluation(
            "higher_resolution",
            test_batch,
            test_target,
            shift="resolution",
            case_ids=case_ids,
        ),
    ]
    if int(rollout_steps) > 1:
        rollout_target = jnp.asarray(test_batch.input("state").values)
        for _ in range(int(rollout_steps) * int(target_steps)):
            rollout_target = _burgers_step(rollout_target, viscosity, dt)
        evaluations.append(
            OperatorBenchmarkEvaluation(
                "long_rollout",
                test_batch,
                rollout_target,
                shift="rollout",
                rollout_steps=int(rollout_steps),
                rollout_source_key="state",
                case_ids=case_ids,
            )
        )
    mass_drifts = [
        float(
            jnp.max(
                jnp.abs(
                    jnp.mean(target, axis=-1)
                    - jnp.mean(batch.input("state").values, axis=-1)
                )
            )
        )
        for batch, target in (
            (train_batch, train_target),
            (test_batch, test_target),
        )
    ]
    if int(rollout_steps) > 1:
        mass_drifts.append(
            float(
                jnp.max(
                    jnp.abs(
                        jnp.mean(rollout_target, axis=-1)
                        - jnp.mean(test_batch.input("state").values, axis=-1)
                    )
                )
            )
        )
    time_horizon = float(dt) * int(target_steps)
    diffusive_steps = (
        float(viscosity * dt * train_resolution**2),
        float(viscosity * dt * test_resolution**2),
    )
    return OperatorBenchmarkScenario(
        scenario_name,
        train_batch,
        train_target,
        tuple(evaluations),
        seed=int(seed),
        metadata=(
            ("equation", "u_t + u*u_x = viscosity*u_xx"),
            ("boundary", "periodic_unit_interval"),
            ("viscosity", str(float(viscosity))),
            ("dt", str(float(dt))),
            ("target_steps", str(int(target_steps))),
            ("rollout_steps", str(int(rollout_steps))),
            ("time_horizon", str(time_horizon)),
            ("rollout_time_horizon", str(time_horizon * int(rollout_steps))),
            ("initial_condition", initial_condition),
            ("maximum_absolute_mass_drift", str(max(mass_drifts))),
            ("maximum_frequency", str(int(maximum_frequency))),
            ("resolved_frequency", str(int(resolved_frequency))),
            ("population_seed", str(int(seed))),
        ),
        case_ids=case_ids,
        provenance=_generated_provenance("periodic_burgers_scenario"),
        dimensional_parameters=(
            OperatorParameterRange(
                "viscosity",
                float(viscosity),
                float(viscosity),
                "L^2 T^-1",
                "log",
            ),
            OperatorParameterRange("time_step", float(dt), float(dt), "T"),
            OperatorParameterRange(
                "time_horizon",
                time_horizon,
                time_horizon * int(rollout_steps),
                "T",
            ),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "diffusive_step",
                min(diffusive_steps),
                max(diffusive_steps),
                "1",
                "log",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="semi-implicit Fourier pseudospectral update",
            verification="discrete_residual",
            resolutions=(int(train_resolution), int(test_resolution)),
            relative_error=max(train_residual, test_residual),
            tolerance=1e-10,
        ),
        regimes=(
            "shock_discontinuity" if initial_condition == "shock" else "smooth_periodic",
            "shock_formation"
            if initial_condition == "shock"
            else "smooth_viscous_evolution",
            "viscous_rollout"
            if initial_condition == "shock" and int(rollout_steps) > 1
            else "viscous_single_step",
            "resolution_transfer",
            "long_horizon" if int(rollout_steps) > 1 else "single_step",
        ),
    )


def _darcy_system(coefficient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = coefficient.shape
    interior = (nx - 2) * (ny - 2)
    matrix = np.zeros((interior, interior), dtype=np.float64)
    forcing = np.ones((interior,), dtype=np.float64)

    def index(i, j):
        return (i - 1) * (ny - 2) + (j - 1)

    hx2 = float((nx - 1) ** 2)
    hy2 = float((ny - 1) ** 2)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            row = index(i, j)
            east = 0.5 * (coefficient[i, j] + coefficient[i + 1, j]) * hx2
            west = 0.5 * (coefficient[i, j] + coefficient[i - 1, j]) * hx2
            north = 0.5 * (coefficient[i, j] + coefficient[i, j + 1]) * hy2
            south = 0.5 * (coefficient[i, j] + coefficient[i, j - 1]) * hy2
            matrix[row, row] = east + west + north + south
            if i + 1 < nx - 1:
                matrix[row, index(i + 1, j)] = -east
            if i - 1 > 0:
                matrix[row, index(i - 1, j)] = -west
            if j + 1 < ny - 1:
                matrix[row, index(i, j + 1)] = -north
            if j - 1 > 0:
                matrix[row, index(i, j - 1)] = -south
    return matrix, forcing


def _darcy_solution(coefficient: np.ndarray) -> tuple[np.ndarray, float]:
    matrix, forcing = _darcy_system(coefficient)
    interior_solution = np.linalg.solve(matrix, forcing)
    residual = matrix @ interior_solution - forcing
    relative_residual = float(
        np.linalg.norm(residual) / max(np.linalg.norm(forcing), 1e-12)
    )
    output = np.zeros_like(coefficient)
    output[1:-1, 1:-1] = interior_solution.reshape(
        (coefficient.shape[0] - 2, coefficient.shape[1] - 2)
    )
    return output, relative_residual


def darcy_scenario(
    *,
    resolution: int = 12,
    num_cases: int = 4,
    contrast: float = 0.35,
    maximum_frequency: int = 3,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    if not 0.0 <= float(contrast) < 1.0:
        raise ValueError("Darcy contrast must lie in [0, 1).")
    if int(maximum_frequency) <= 0:
        raise ValueError("maximum_frequency must be positive.")
    resolved_frequency = min(
        int(maximum_frequency),
        max(1, (int(resolution) - 2) // 2),
    )
    x_axis = _interval_axis("x", resolution)
    y_axis = _interval_axis("y", resolution)
    x, y = jnp.meshgrid(x_axis.nodes, y_axis.nodes, indexing="ij")
    coordinates = jnp.stack((x, y), axis=-1)
    mode_x, mode_y, population_coefficients = _planar_population_coefficients(
        jr.key(seed),
        num_cases,
        resolved_frequency,
    )
    logit = _evaluate_planar_population(
        population_coefficients,
        mode_x,
        mode_y,
        coordinates,
    )
    scale = jnp.max(jnp.abs(logit), axis=(-2, -1), keepdims=True)
    normalized_logit = jnp.clip(logit / jnp.maximum(scale, 1e-12), -1.0, 1.0)
    permeability = 1.0 + float(contrast) * normalized_logit

    targets = []
    residuals = []
    for coefficient in np.asarray(permeability):
        solution, residual = _darcy_solution(coefficient)
        targets.append(jnp.asarray(solution))
        residuals.append(residual)
    target = jnp.stack(targets)
    batch = _grid_batch(
        {
            "coefficient": FunctionSamples(
                values=permeability,
                axes=(x_axis, y_axis),
            )
        },
        (x_axis, y_axis),
    )
    case_ids = _case_ids("darcy_2d", num_cases)
    return OperatorBenchmarkScenario(
        "darcy_2d",
        batch,
        target,
        (
            OperatorBenchmarkEvaluation(
                "train_resolution",
                batch,
                target,
                case_ids=case_ids,
            ),
        ),
        seed=int(seed),
        case_ids=case_ids,
        provenance=_generated_provenance("darcy_scenario"),
        dimensional_parameters=(
            OperatorParameterRange(
                "permeability",
                1.0 - float(contrast),
                1.0 + float(contrast),
                "L^2",
            ),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "permeability_contrast_ratio",
                (1.0 + float(contrast)) / (1.0 - float(contrast)),
                (1.0 + float(contrast)) / (1.0 - float(contrast)),
                "1",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="direct finite-volume elliptic solve",
            verification="discrete_residual",
            resolutions=(int(resolution), int(resolution)),
            relative_error=max(residuals),
            tolerance=1e-10,
        ),
        regimes=("elliptic_contrast", "tensor_grid", "multi_mode_population"),
        metadata=(
            ("contrast", str(float(contrast))),
            ("maximum_frequency", str(int(maximum_frequency))),
            ("resolved_frequency", str(int(resolved_frequency))),
            ("population_seed", str(int(seed))),
        ),
    )


def square_diffusion_symmetry_scenario(
    *,
    resolution: int = 16,
    num_cases: int = 8,
    diffusivity: tuple[float, float] = (0.02, 0.02),
    dt: float = 0.05,
    chiral_strength: float = 0.0,
    maximum_frequency: int = 4,
    test_resolution: int | None = None,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    """Periodic square diffusion with exact D4, C4-only, or broken symmetry."""
    if int(resolution) < 5 or int(num_cases) <= 0:
        raise ValueError("Square diffusion requires resolution >= 5 and positive cases.")
    diffusivity_x, diffusivity_y = map(float, diffusivity)
    if min(diffusivity_x, diffusivity_y, float(dt)) <= 0.0:
        raise ValueError("Diffusivities and dt must be positive.")
    if int(maximum_frequency) <= 0:
        raise ValueError("maximum_frequency must be positive.")
    resolved_test_resolution = (
        int(resolution) + max(2, int(resolution) // 2)
        if test_resolution is None
        else int(test_resolution)
    )
    if resolved_test_resolution <= int(resolution):
        raise ValueError("test_resolution must exceed the training resolution.")

    resolved_frequency = min(
        int(maximum_frequency),
        max(1, (int(resolution) - 1) // 2),
    )
    x_axis = _periodic_axis("x", int(resolution))
    y_axis = _periodic_axis("y", int(resolution))
    x, y = jnp.meshgrid(x_axis.nodes, y_axis.nodes, indexing="ij")
    coordinates = jnp.stack((x, y), axis=-1)
    mode_x, mode_y, coefficients = _planar_population_coefficients(
        jr.key(seed),
        int(num_cases),
        resolved_frequency,
    )
    initial = _evaluate_planar_population(
        coefficients,
        mode_x,
        mode_y,
        coordinates,
    )

    def solution_operator(values):
        values_array = jnp.asarray(values)
        size_x, size_y = values_array.shape[-2:]
        wave_x = 2.0 * jnp.pi * jnp.fft.fftfreq(int(size_x), d=1.0 / int(size_x))
        wave_y = 2.0 * jnp.pi * jnp.fft.fftfreq(int(size_y), d=1.0 / int(size_y))
        wave_x, wave_y = jnp.meshgrid(wave_x, wave_y, indexing="ij")
        squared_wave_number = wave_x**2 + wave_y**2
        attenuation = jnp.exp(
            -float(dt) * (diffusivity_x * wave_x**2 + diffusivity_y * wave_y**2)
        )
        spectrum = jnp.fft.fftn(values_array, axes=(-2, -1))
        output = jnp.fft.ifftn(
            spectrum * attenuation,
            axes=(-2, -1),
        ).real
        if float(chiral_strength) != 0.0:
            gradient_x = jnp.fft.ifftn(
                1j * wave_x * spectrum,
                axes=(-2, -1),
            ).real
            gradient_y = jnp.fft.ifftn(
                1j * wave_y * spectrum,
                axes=(-2, -1),
            ).real
            laplacian_spectrum = -squared_wave_number * spectrum
            laplacian_gradient_x = jnp.fft.ifftn(
                1j * wave_x * laplacian_spectrum,
                axes=(-2, -1),
            ).real
            laplacian_gradient_y = jnp.fft.ifftn(
                1j * wave_y * laplacian_spectrum,
                axes=(-2, -1),
            ).real
            chirality = (
                gradient_x * laplacian_gradient_y - gradient_y * laplacian_gradient_x
            )
            output = output + float(chiral_strength) * chirality
        return output

    isotropic = bool(np.isclose(diffusivity_x, diffusivity_y))
    if isotropic and float(chiral_strength) == 0.0:
        name = "square_scalar_diffusion_d4"
        physical_group: Literal["p4", "p4m"] | None = "p4m"
    elif isotropic:
        name = "square_chiral_control_c4"
        physical_group = "p4"
    else:
        name = "square_anisotropic_control"
        physical_group = None
    target = solution_operator(initial)
    reference_defects = _square_symmetry_reference_defects(
        initial,
        solution_operator,
        group="p4m",
        source_representation="scalar",
        target_representation="scalar",
    )
    batch = _grid_batch(
        {"initial": FunctionSamples(values=initial, axes=(x_axis, y_axis))},
        (x_axis, y_axis),
    )
    case_ids = _case_ids(name, int(num_cases))
    test_x_axis = _periodic_axis("x", resolved_test_resolution)
    test_y_axis = _periodic_axis("y", resolved_test_resolution)
    test_x, test_y = jnp.meshgrid(
        test_x_axis.nodes,
        test_y_axis.nodes,
        indexing="ij",
    )
    test_coordinates = jnp.stack((test_x, test_y), axis=-1)
    resolution_initial = _evaluate_planar_population(
        coefficients,
        mode_x,
        mode_y,
        test_coordinates,
    )
    resolution_batch = _grid_batch(
        {
            "initial": FunctionSamples(
                values=resolution_initial,
                axes=(test_x_axis, test_y_axis),
            )
        },
        (test_x_axis, test_y_axis),
    )
    resolution_target = solution_operator(resolution_initial)
    _, _, shifted_coefficients = _planar_population_coefficients(
        jr.key(int(seed) + 104729),
        int(num_cases),
        resolved_frequency,
    )
    mode_radius = jnp.sqrt(mode_x.astype(float) ** 2 + mode_y.astype(float) ** 2)
    spectral_tilt = 2.0 * (mode_radius / jnp.maximum(jnp.max(mode_radius), 1.0)) ** 1.5
    shifted_coefficients = shifted_coefficients * spectral_tilt[None, :, None]
    shifted_initial = _evaluate_planar_population(
        shifted_coefficients,
        mode_x,
        mode_y,
        coordinates,
    )
    shifted_batch = _grid_batch(
        {
            "initial": FunctionSamples(
                values=shifted_initial,
                axes=(x_axis, y_axis),
            )
        },
        (x_axis, y_axis),
    )
    shifted_target = solution_operator(shifted_initial)
    shifted_case_ids = _case_ids(f"{name}_spectral_tilt", int(num_cases))
    symmetry = OperatorSymmetrySpec(
        group=physical_group,
        audit_group="p4m",
        source_representations=(("initial", "scalar"),),
        target_representation="scalar",
        reference_tolerance=1e-10,
        reference_defects=reference_defects,
        intentionally_violated=physical_group != "p4m",
    )
    return OperatorBenchmarkScenario(
        name,
        batch,
        target,
        (
            OperatorBenchmarkEvaluation(
                "nominal",
                batch,
                target,
                case_ids=case_ids,
            ),
            OperatorBenchmarkEvaluation(
                "resolution_extrapolation",
                resolution_batch,
                resolution_target,
                shift="resolution",
                case_ids=case_ids,
            ),
            OperatorBenchmarkEvaluation(
                "forcing_spectrum_shift",
                shifted_batch,
                shifted_target,
                shift="forcing_spectrum",
                case_ids=shifted_case_ids,
            ),
        ),
        seed=int(seed),
        case_ids=case_ids,
        provenance=_generated_provenance("square_diffusion_symmetry_scenario"),
        dimensional_parameters=(
            OperatorParameterRange(
                "diffusivity_x",
                diffusivity_x,
                diffusivity_x,
                "L^2 T^-1",
                "log",
            ),
            OperatorParameterRange(
                "diffusivity_y",
                diffusivity_y,
                diffusivity_y,
                "L^2 T^-1",
                "log",
            ),
            OperatorParameterRange("time_step", float(dt), float(dt), "T"),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "maximum_fourier_number",
                max(diffusivity_x, diffusivity_y) * float(dt),
                max(diffusivity_x, diffusivity_y) * float(dt),
                "1",
                "log",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="analytic periodic Fourier multiplier with spectral chiral control",
            verification="analytic",
            resolutions=(int(resolution), int(resolution)),
            relative_error=0.0,
            tolerance=0.0,
        ),
        regimes=(
            "smooth_periodic",
            "two_dimensional",
            "square_symmetry",
            "d4_exact"
            if physical_group == "p4m"
            else "c4_exact"
            if physical_group == "p4"
            else "symmetry_broken_control",
        ),
        metadata=(
            ("diffusivity_x", str(diffusivity_x)),
            ("diffusivity_y", str(diffusivity_y)),
            ("dt", str(float(dt))),
            ("chiral_strength", str(float(chiral_strength))),
            ("maximum_frequency", str(int(maximum_frequency))),
            ("resolved_frequency", str(int(resolved_frequency))),
            ("test_resolution", str(resolved_test_resolution)),
            ("forcing_shift", "high_frequency_spectral_tilt"),
            ("population_seed", str(int(seed))),
        ),
        symmetry=symmetry,
    )


def _vorticity_step(vorticity, viscosity, dt):
    vorticity_array = np.asarray(jax.device_get(vorticity))
    size_x, size_y = vorticity_array.shape[-2:]
    kx = 2.0 * np.pi * np.fft.fftfreq(size_x, d=1.0 / size_x)
    ky = 2.0 * np.pi * np.fft.fftfreq(size_y, d=1.0 / size_y)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx_grid**2 + ky_grid**2
    transformed = np.fft.fftn(vorticity_array, axes=(-2, -1))
    stream = np.zeros_like(transformed)
    np.divide(transformed, k2, out=stream, where=k2 > 0.0)
    velocity_x = np.fft.ifftn(
        1j * ky_grid * stream,
        axes=(-2, -1),
    ).real
    velocity_y = np.fft.ifftn(
        -1j * kx_grid * stream,
        axes=(-2, -1),
    ).real
    gradient_x = np.fft.ifftn(
        1j * kx_grid * transformed,
        axes=(-2, -1),
    ).real
    gradient_y = np.fft.ifftn(
        1j * ky_grid * transformed,
        axes=(-2, -1),
    ).real
    advection = velocity_x * gradient_x + velocity_y * gradient_y
    return jnp.asarray(
        np.fft.ifftn(
            (transformed - dt * np.fft.fftn(advection, axes=(-2, -1)))
            / (1.0 + viscosity * dt * k2),
            axes=(-2, -1),
        ).real
    )


def _vorticity_step_residual(vorticity, target, viscosity, dt) -> float:
    vorticity_array = np.asarray(jax.device_get(vorticity))
    target_array = np.asarray(jax.device_get(target))
    size_x, size_y = vorticity_array.shape[-2:]
    kx = 2.0 * np.pi * np.fft.fftfreq(size_x, d=1.0 / size_x)
    ky = 2.0 * np.pi * np.fft.fftfreq(size_y, d=1.0 / size_y)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx_grid**2 + ky_grid**2
    transformed = np.fft.fftn(vorticity_array, axes=(-2, -1))
    stream = np.zeros_like(transformed)
    np.divide(transformed, k2, out=stream, where=k2 > 0.0)
    velocity_x = np.fft.ifftn(
        1j * ky_grid * stream,
        axes=(-2, -1),
    ).real
    velocity_y = np.fft.ifftn(
        -1j * kx_grid * stream,
        axes=(-2, -1),
    ).real
    gradient_x = np.fft.ifftn(
        1j * kx_grid * transformed,
        axes=(-2, -1),
    ).real
    gradient_y = np.fft.ifftn(
        1j * ky_grid * transformed,
        axes=(-2, -1),
    ).real
    advection = velocity_x * gradient_x + velocity_y * gradient_y
    right = transformed - dt * np.fft.fftn(advection, axes=(-2, -1))
    left = (1.0 + viscosity * dt * k2) * np.fft.fftn(
        target_array,
        axes=(-2, -1),
    )
    denominator = max(float(np.linalg.norm(right)), 1e-12)
    return float(np.linalg.norm(left - right) / denominator)


def navier_stokes_scenario(
    *,
    resolution: int = 16,
    num_cases: int = 4,
    viscosity: float = 1e-3,
    dt: float = 0.01,
    target_steps: int = 1,
    maximum_frequency: int = 3,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    if int(target_steps) <= 0:
        raise ValueError("target_steps must be positive.")
    if int(maximum_frequency) <= 0:
        raise ValueError("maximum_frequency must be positive.")
    resolved_frequency = min(
        int(maximum_frequency),
        max(1, (int(resolution) - 1) // 2),
    )
    x_axis = _periodic_axis("x", resolution)
    y_axis = _periodic_axis("y", resolution)
    x, y = jnp.meshgrid(x_axis.nodes, y_axis.nodes, indexing="ij")
    mode_x, mode_y = jnp.meshgrid(
        jnp.arange(resolved_frequency + 1),
        jnp.arange(resolved_frequency + 1),
        indexing="ij",
    )
    mode_x = mode_x.reshape(-1)[1:]
    mode_y = mode_y.reshape(-1)[1:]
    mode_scale = jnp.reciprocal(
        jnp.sqrt(mode_x.astype(float) ** 2 + mode_y.astype(float) ** 2)
    )
    coefficient_key, phase_key = jr.split(jr.key(seed))
    coefficients = jr.normal(
        coefficient_key,
        (num_cases, int(mode_x.shape[0])),
    )
    coefficients = coefficients * mode_scale[None, :] / jnp.sqrt(mode_x.shape[0])
    phases = jr.uniform(
        phase_key,
        coefficients.shape,
        minval=0.0,
        maxval=2.0 * jnp.pi,
    )
    modes = jnp.sin(
        2.0
        * jnp.pi
        * (mode_x[:, None, None] * x[None, :, :] + mode_y[:, None, None] * y[None, :, :])
        + phases[:, :, None, None]
    )
    vorticity = ein.contract("cm,cmxy->cxy", coefficients, modes)
    target = vorticity
    maximum_residual = 0.0
    for _ in range(int(target_steps)):
        next_target = _vorticity_step(target, viscosity, dt)
        maximum_residual = max(
            maximum_residual,
            _vorticity_step_residual(target, next_target, viscosity, dt),
        )
        target = next_target

    def solution_operator(values):
        output = values
        for _ in range(int(target_steps)):
            output = _vorticity_step(output, viscosity, dt)
        return output

    symmetry = OperatorSymmetrySpec(
        group="p4m",
        audit_group="p4m",
        source_representations=(("vorticity", "pseudoscalar"),),
        target_representation="pseudoscalar",
        reference_tolerance=1e-10,
        reference_defects=_square_symmetry_reference_defects(
            vorticity,
            solution_operator,
            group="p4m",
            source_representation="pseudoscalar",
            target_representation="pseudoscalar",
        ),
    )
    batch = _grid_batch(
        {"vorticity": FunctionSamples(values=vorticity, axes=(x_axis, y_axis))},
        (x_axis, y_axis),
    )
    case_ids = _case_ids("navier_stokes_vorticity_2d", num_cases)
    return OperatorBenchmarkScenario(
        "navier_stokes_vorticity_2d",
        batch,
        target,
        (
            OperatorBenchmarkEvaluation(
                "short_rollout",
                batch,
                target,
                case_ids=case_ids,
            ),
        ),
        seed=int(seed),
        case_ids=case_ids,
        provenance=_generated_provenance("navier_stokes_scenario"),
        dimensional_parameters=(
            OperatorParameterRange(
                "viscosity",
                float(viscosity),
                float(viscosity),
                "L^2 T^-1",
                "log",
            ),
            OperatorParameterRange("time_step", float(dt), float(dt), "T"),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "diffusive_step",
                float(viscosity * dt * resolution**2),
                float(viscosity * dt * resolution**2),
                "1",
                "log",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="semi-implicit Fourier vorticity update",
            verification="discrete_residual",
            resolutions=(int(resolution), int(resolution)),
            relative_error=maximum_residual,
            tolerance=1e-10,
        ),
        regimes=(
            "smooth_periodic",
            "two_dimensional",
            "multi_step" if int(target_steps) > 1 else "single_step",
        ),
        metadata=(
            ("viscosity", str(float(viscosity))),
            ("dt", str(float(dt))),
            ("target_steps", str(int(target_steps))),
            ("maximum_frequency", str(int(maximum_frequency))),
            ("resolved_frequency", str(int(resolved_frequency))),
            ("population_seed", str(int(seed))),
        ),
        symmetry=symmetry,
    )


def green_function_scenario(
    *,
    source_points: int = 20,
    query_points: int = 31,
    num_cases: int = 6,
    kernel_length_scale: float = 0.15,
    maximum_frequency: int = 4,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    if float(kernel_length_scale) <= 0.0:
        raise ValueError("kernel_length_scale must be positive.")
    if int(maximum_frequency) <= 0:
        raise ValueError("maximum_frequency must be positive.")
    source_coordinate = jnp.linspace(0.0, 1.0, source_points)[:, None]
    query_coordinate = jnp.linspace(0.0, 1.0, query_points)[:, None]
    resolved_frequency = min(
        int(maximum_frequency),
        max(2, (int(source_points) - 1) // 2),
    )
    frequencies = jnp.arange(1, resolved_frequency + 1, dtype=float)
    coefficient_key, phase_key = jr.split(jr.key(seed))
    coefficients = (
        jr.normal(
            coefficient_key,
            (num_cases, resolved_frequency),
        )
        / jnp.sqrt(frequencies)[None, :]
    )
    phases = jr.uniform(
        phase_key,
        coefficients.shape,
        minval=0.0,
        maxval=2.0 * jnp.pi,
    )
    modes = jnp.sin(
        2.0 * jnp.pi * frequencies[None, :, None] * source_coordinate[:, 0][None, None, :]
        + phases[:, :, None]
    )
    forcing = ein.contract("cm,cms->cs", coefficients, modes) / jnp.sqrt(
        resolved_frequency
    )
    weights = jnp.ones((source_points,)) / source_points
    kernel = jnp.exp(
        -jnp.abs(query_coordinate[:, 0, None] - source_coordinate[:, 0][None, :])
        / float(kernel_length_scale)
    )
    target = ein.contract("qs,cs,s->cq", kernel, forcing, weights)
    source = FunctionSamples(
        values=forcing,
        coordinates=source_coordinate,
        quadrature_weights=weights,
    )
    query = FunctionSamples(values=None, coordinates=query_coordinate)
    batch = OperatorBatch(
        inputs={"forcing": source}, queries={"query": query}, case_axes=("case",)
    )
    changed_query_coordinate = (
        0.5 - 0.5 * jnp.cos(jnp.pi * jnp.linspace(0.0, 1.0, query_points))
    )[:, None]
    changed_kernel = jnp.exp(
        -jnp.abs(changed_query_coordinate[:, 0, None] - source_coordinate[:, 0][None, :])
        / float(kernel_length_scale)
    )
    changed_target = ein.contract(
        "qs,cs,s->cq",
        changed_kernel,
        forcing,
        weights,
    )
    changed_batch = OperatorBatch(
        inputs={"forcing": source},
        queries={
            "query": FunctionSamples(values=None, coordinates=changed_query_coordinate)
        },
        case_axes=("case",),
    )
    case_ids = _case_ids("analytic_green_query", num_cases)
    return OperatorBenchmarkScenario(
        "analytic_green_query",
        batch,
        target,
        (
            OperatorBenchmarkEvaluation(
                "changed_query_grid",
                changed_batch,
                changed_target,
                shift="query_geometry",
                case_ids=case_ids,
            ),
        ),
        seed=int(seed),
        case_ids=case_ids,
        provenance=_generated_provenance("green_function_scenario"),
        dimensional_parameters=(
            OperatorParameterRange(
                "kernel_length_scale",
                float(kernel_length_scale),
                float(kernel_length_scale),
                "L",
                "log",
            ),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "query_source_ratio",
                float(query_points / source_points),
                float(query_points / source_points),
                "1",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="direct analytic-kernel quadrature",
            verification="analytic",
            resolutions=(int(source_points), int(query_points)),
            relative_error=0.0,
            tolerance=0.0,
        ),
        regimes=("independent_query", "integral_operator", "point_cloud"),
        metadata=(
            ("kernel_length_scale", str(float(kernel_length_scale))),
            ("maximum_frequency", str(int(maximum_frequency))),
            ("resolved_frequency", str(int(resolved_frequency))),
            ("population_seed", str(int(seed))),
        ),
    )


def multi_input_diffusion_scenario(
    *,
    resolution: int = 24,
    num_cases: int = 6,
    dt: float = 0.05,
    diffusivity_range: tuple[float, float] = (0.01, 0.022),
    parameter_shift_factor: float | None = None,
    maximum_frequency: int = 5,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    minimum_diffusivity, maximum_diffusivity = map(float, diffusivity_range)
    if minimum_diffusivity <= 0.0 or maximum_diffusivity < minimum_diffusivity:
        raise ValueError("diffusivity_range must be positive and ordered.")
    if parameter_shift_factor is not None and float(parameter_shift_factor) <= 1.0:
        raise ValueError("parameter_shift_factor must exceed one.")
    if int(maximum_frequency) <= 0:
        raise ValueError("maximum_frequency must be positive.")
    resolved_frequency = min(
        int(maximum_frequency),
        max(1, (int(resolution) - 1) // 2),
    )
    axis = _periodic_axis("x", resolution)
    initial_key, forcing_key, diffusivity_key = jr.split(jr.key(seed), 3)
    frequencies, initial_coefficients = _periodic_population_coefficients(
        initial_key,
        num_cases,
        resolved_frequency,
    )
    _, forcing_coefficients = _periodic_population_coefficients(
        forcing_key,
        num_cases,
        resolved_frequency,
    )
    phase_coordinate = 2.0 * jnp.pi * axis.nodes
    initial = _evaluate_periodic_population(
        initial_coefficients,
        frequencies,
        phase_coordinate,
    )
    forcing = 0.25 * _evaluate_periodic_population(
        forcing_coefficients,
        frequencies,
        phase_coordinate,
    )
    diffusivity = jr.uniform(
        diffusivity_key,
        (int(num_cases), 1),
        minval=minimum_diffusivity,
        maxval=maximum_diffusivity,
    )
    squared_wave_number = (
        2.0 * jnp.pi * jnp.fft.rfftfreq(int(resolution), d=1.0 / int(resolution))
    ) ** 2

    def build(current_diffusivity):
        decay_rate = current_diffusivity * squared_wave_number[None, :]
        attenuation = jnp.exp(-decay_rate * float(dt))
        safe_rate = jnp.where(decay_rate > 0.0, decay_rate, 1.0)
        forcing_response = -jnp.expm1(-safe_rate * float(dt)) / safe_rate
        forcing_response = jnp.where(
            decay_rate > 0.0,
            forcing_response,
            float(dt),
        )
        target_spectrum = (
            jnp.fft.rfft(initial, axis=-1) * attenuation
            + jnp.fft.rfft(forcing, axis=-1) * forcing_response
        )
        target = jnp.fft.irfft(
            target_spectrum,
            n=int(resolution),
            axis=-1,
        )
        batch = _grid_batch(
            {
                "initial": FunctionSamples(values=initial, axes=(axis,)),
                "forcing": FunctionSamples(values=forcing, axes=(axis,)),
                "diffusivity": FunctionSamples(values=current_diffusivity),
            },
            (axis,),
        )
        return batch, target

    batch, target = build(diffusivity)
    case_ids = _case_ids("multi_input_diffusion", num_cases)
    evaluations = [
        OperatorBenchmarkEvaluation(
            "multi_input",
            batch,
            target,
            case_ids=case_ids,
        ),
    ]
    if parameter_shift_factor is not None:
        shifted_diffusivity = diffusivity * float(parameter_shift_factor)
        shifted_batch, shifted_target = build(shifted_diffusivity)
        evaluations.append(
            OperatorBenchmarkEvaluation(
                "diffusivity_extrapolation",
                shifted_batch,
                shifted_target,
                shift="parameter",
                case_ids=case_ids,
            )
        )
    maximum_parameter = maximum_diffusivity * (
        1.0 if parameter_shift_factor is None else float(parameter_shift_factor)
    )
    return OperatorBenchmarkScenario(
        "multi_input_diffusion",
        batch,
        target,
        tuple(evaluations),
        seed=int(seed),
        case_ids=case_ids,
        provenance=_generated_provenance("multi_input_diffusion_scenario"),
        dimensional_parameters=(
            OperatorParameterRange(
                "diffusivity",
                minimum_diffusivity,
                maximum_parameter,
                "L^2 T^-1",
                "log",
            ),
            OperatorParameterRange("time_step", float(dt), float(dt), "T"),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "fourier_number",
                minimum_diffusivity * dt,
                maximum_parameter * dt,
                "1",
                "log",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="closed-form forced Fourier-mode diffusion",
            verification="analytic",
            resolutions=(int(resolution),),
            relative_error=0.0,
            tolerance=0.0,
        ),
        regimes=(
            "multi_input",
            "parameter_extrapolation"
            if parameter_shift_factor is not None
            else "parameter_interpolation",
            "independent_multi_mode_population",
        ),
        metadata=(
            ("dt", str(float(dt))),
            ("diffusivity_min", str(minimum_diffusivity)),
            ("diffusivity_max", str(maximum_diffusivity)),
            ("maximum_frequency", str(int(maximum_frequency))),
            ("resolved_frequency", str(int(resolved_frequency))),
            ("population_seed", str(int(seed))),
        ),
    )


def causal_relaxation_scenario(
    *,
    source_points: int = 24,
    query_points: int | None = None,
    test_query_points: int | None = None,
    num_cases: int = 8,
    final_time: float = 1.0,
    decay_rate: float = 1.0,
    maximum_frequency: float = 2.0,
    modes: int = 8,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    """Analytic nonperiodic causal convolution with independent query times."""
    if int(source_points) < 2:
        raise ValueError("source_points must be at least two.")
    if query_points is None:
        query_points = int(source_points)
    if test_query_points is None:
        test_query_points = int(query_points) + max(2, int(query_points) // 2)
    if int(query_points) < 2 or int(test_query_points) < 2:
        raise ValueError("query point counts must be at least two.")
    if int(num_cases) <= 0:
        raise ValueError("num_cases must be positive.")
    if float(final_time) <= 0.0 or float(decay_rate) <= 0.0:
        raise ValueError("final_time and decay_rate must be positive.")
    if float(maximum_frequency) <= 0.0:
        raise ValueError("maximum_frequency must be positive.")
    if int(modes) <= 0:
        raise ValueError("modes must be positive.")

    maximum_resolved_modes = min(
        max(1, int(np.floor(float(maximum_frequency) * float(final_time)))),
        max(
            1,
            (
                min(
                    int(source_points),
                    int(query_points),
                    int(test_query_points),
                )
                - 1
            )
            // 2,
        ),
    )
    resolved_modes = min(int(modes), maximum_resolved_modes)
    frequencies = jnp.arange(1, resolved_modes + 1, dtype=float) / float(final_time)
    coefficient_key = jr.key(seed)
    coefficients = jr.normal(
        coefficient_key,
        (int(num_cases), resolved_modes, 2),
    ) / jnp.sqrt(float(resolved_modes))
    response_balancing = jnp.sqrt(
        float(decay_rate) ** 2 + (2.0 * jnp.pi * frequencies) ** 2
    )
    response_balancing = response_balancing / response_balancing[0]
    coefficients = coefficients * response_balancing[None, :, None]
    initial_values = jnp.zeros((int(num_cases),))

    def time_axis(size: int):
        nodes = jnp.linspace(0.0, float(final_time), int(size))
        weights = (
            jnp.ones((int(size),))
            .at[jnp.asarray([0, int(size) - 1], dtype=jnp.int32)]
            .set(0.5)
            * float(final_time)
            / (int(size) - 1)
        )
        return OperatorAxis("t", nodes, quadrature_weights=weights)

    def forcing(times):
        phase = 2.0 * jnp.pi * frequencies[:, None] * times[None, :]
        basis = jnp.stack((jnp.sin(phase), jnp.cos(phase)), axis=-1)
        return ein.contract("cmk,mtk->ct", coefficients, basis)

    def response(times):
        omega = 2.0 * jnp.pi * frequencies[:, None]
        time = times[None, :]
        transient = jnp.exp(-float(decay_rate) * time)
        kernel = (jnp.exp(1j * omega * time) - transient) / (
            float(decay_rate) + 1j * omega
        )
        forced = ein.contract(
            "cm,mt->ct",
            coefficients[..., 0],
            jnp.imag(kernel),
        ) + ein.contract(
            "cm,mt->ct",
            coefficients[..., 1],
            jnp.real(kernel),
        )
        return transient * initial_values[:, None] + forced

    source_axis = time_axis(int(source_points))
    query_axis = time_axis(int(query_points))
    test_axis = time_axis(int(test_query_points))
    source = FunctionSamples(values=forcing(source_axis.nodes), axes=(source_axis,))
    query = FunctionSamples(values=None, axes=(query_axis,))
    test_query = FunctionSamples(values=None, axes=(test_axis,))
    case_ids = _case_ids("causal_relaxation", int(num_cases))
    batch = OperatorBatch(
        inputs={"forcing": source},
        queries={"query": query},
        case_axes=("case",),
        case_shape=(int(num_cases),),
    )
    test_batch = OperatorBatch(
        inputs={"forcing": source},
        queries={"query": test_query},
        case_axes=("case",),
        case_shape=(int(num_cases),),
    )
    return OperatorBenchmarkScenario(
        "causal_relaxation_1d",
        batch,
        response(query_axis.nodes),
        (
            OperatorBenchmarkEvaluation(
                "temporal_resolution_transfer",
                test_batch,
                response(test_axis.nodes),
                shift="resolution",
                case_ids=case_ids,
            ),
        ),
        case_ids=case_ids,
        seed=int(seed),
        provenance=_generated_provenance("causal_relaxation_scenario"),
        dimensional_parameters=(
            OperatorParameterRange(
                "decay_rate",
                float(decay_rate),
                float(decay_rate),
                "T^-1",
                "log",
            ),
            OperatorParameterRange(
                "forcing_frequency",
                1.0 / float(final_time),
                float(maximum_frequency),
                "T^-1",
            ),
            OperatorParameterRange(
                "final_time",
                float(final_time),
                float(final_time),
                "T",
            ),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "decay_time",
                float(decay_rate) * float(final_time),
                float(decay_rate) * float(final_time),
                "1",
                "log",
            ),
            OperatorParameterRange(
                "frequency_time",
                0.25 * float(final_time),
                float(maximum_frequency) * float(final_time),
                "1",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="closed-form exponential convolution",
            verification="analytic",
            resolutions=(int(source_points), int(test_query_points)),
            relative_error=0.0,
            tolerance=0.0,
        ),
        regimes=("causal_transient", "nonperiodic_time", "independent_query"),
        metadata=(
            ("final_time", str(float(final_time))),
            ("decay_rate", str(float(decay_rate))),
            ("maximum_frequency", str(float(maximum_frequency))),
            ("modes", str(int(resolved_modes))),
            ("population_seed", str(int(seed))),
        ),
    )


def irregular_causal_relaxation_scenario(
    *,
    points: int = 32,
    num_cases: int = 6,
    final_time: float = 2.0,
    decay_rate: float = 0.7,
    maximum_frequency: float = 3.0,
    modes: int = 4,
    extrapolation_factor: float = 1.5,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    """Coincident irregular-time causal operator with ragged and step-shift probes."""
    point_count = int(points)
    case_count = int(num_cases)
    if point_count < 4 or case_count <= 0:
        raise ValueError("points must be at least four and num_cases must be positive.")
    if (
        float(final_time) <= 0.0
        or float(decay_rate) <= 0.0
        or float(maximum_frequency) <= 0.0
    ):
        raise ValueError(
            "final_time, decay_rate, and maximum_frequency must be positive."
        )
    if int(modes) <= 0 or float(extrapolation_factor) <= 1.0:
        raise ValueError(
            "modes must be positive and extrapolation_factor must exceed one."
        )

    resolved_modes = min(int(modes), (point_count - 1) // 2)
    frequencies = jnp.arange(1, resolved_modes + 1, dtype=float) / float(final_time)
    coefficient_key, train_key, shifted_key = jr.split(jr.key(seed), 3)
    coefficients = jr.normal(coefficient_key, (case_count, resolved_modes, 2)) / jnp.sqrt(
        float(resolved_modes)
    )
    response_balancing = jnp.sqrt(
        float(decay_rate) ** 2 + (2.0 * jnp.pi * frequencies) ** 2
    )
    coefficients = (
        coefficients * (response_balancing / response_balancing[0])[None, :, None]
    )

    def schedule(key, horizon, dispersion):
        gaps = jnp.exp(float(dispersion) * jr.normal(key, (case_count, point_count - 1)))
        cumulative = jnp.concatenate(
            (jnp.zeros((case_count, 1)), jnp.cumsum(gaps, axis=-1)), axis=-1
        )
        return float(horizon) * cumulative / cumulative[:, -1:]

    def forcing(times):
        phase = 2.0 * jnp.pi * frequencies[None, :, None] * times[:, None, :]
        basis = jnp.stack((jnp.sin(phase), jnp.cos(phase)), axis=-1)
        return ein.contract("cmk,cmtk->ct", coefficients, basis)

    def response(times):
        omega = 2.0 * jnp.pi * frequencies[None, :, None]
        time = times[:, None, :]
        transient = jnp.exp(-float(decay_rate) * time)
        kernel = (jnp.exp(1j * omega * time) - transient) / (
            float(decay_rate) + 1j * omega
        )
        return ein.contract(
            "cm,cmt->ct", coefficients[..., 0], jnp.imag(kernel)
        ) + ein.contract("cm,cmt->ct", coefficients[..., 1], jnp.real(kernel))

    def batch(times, *, mask=None):
        valid = (
            jnp.ones(times.shape, dtype=bool)
            if mask is None
            else jnp.asarray(mask, dtype=bool)
        )
        coordinates = jnp.where(valid, times, 0.0)[..., None]
        source = FunctionSamples(
            values=jnp.where(valid, forcing(times), 0.0),
            coordinates=coordinates,
            mask=valid,
        )
        query = FunctionSamples(values=None, coordinates=coordinates, mask=valid)
        return OperatorBatch(
            inputs={"forcing": source},
            queries={"query": query},
            case_axes=("case",),
            case_shape=(case_count,),
        )

    train_times = schedule(train_key, final_time, 0.45)
    shifted_times = schedule(shifted_key, final_time * float(extrapolation_factor), 1.0)
    minimum_length = max(2, point_count // 2)
    lengths = point_count - (
        jnp.arange(case_count) % max(1, point_count - minimum_length)
    )
    ragged_mask = jnp.arange(point_count)[None, :] < lengths[:, None]
    train_delta = train_times[:, 1:] - train_times[:, :-1]
    case_ids = _case_ids("irregular_causal_relaxation", case_count)
    return OperatorBenchmarkScenario(
        "irregular_causal_relaxation_1d",
        batch(train_times),
        response(train_times),
        (
            OperatorBenchmarkEvaluation(
                "irregular_step_extrapolation",
                batch(shifted_times),
                response(shifted_times),
                shift="temporal_step_extrapolation",
                case_ids=case_ids,
            ),
            OperatorBenchmarkEvaluation(
                "ragged_schedule",
                batch(train_times, mask=ragged_mask),
                jnp.where(ragged_mask, response(train_times), 0.0),
                shift="ragged_schedule",
                case_ids=case_ids,
            ),
        ),
        case_ids=case_ids,
        seed=int(seed),
        provenance=_generated_provenance("irregular_causal_relaxation_scenario"),
        dimensional_parameters=(
            OperatorParameterRange(
                "decay_rate",
                float(decay_rate),
                float(decay_rate),
                "T^-1",
                "log",
            ),
            OperatorParameterRange(
                "forcing_frequency",
                1.0 / float(final_time),
                float(maximum_frequency),
                "T^-1",
            ),
            OperatorParameterRange(
                "final_time",
                float(final_time),
                float(final_time) * float(extrapolation_factor),
                "T",
            ),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "decay_time",
                float(decay_rate) * float(final_time),
                float(decay_rate) * float(final_time) * float(extrapolation_factor),
                "1",
                "log",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="closed-form exponential convolution",
            verification="analytic",
            resolutions=(point_count,),
            relative_error=0.0,
            tolerance=0.0,
        ),
        regimes=(
            "causal_transient",
            "nonperiodic_time",
            "coincident_query",
            "irregular_time",
            "ragged_schedule",
            "temporal_step_extrapolation",
        ),
        ladder="temporal_irregularity",
        metadata=(
            ("final_time", str(float(final_time))),
            ("decay_rate", str(float(decay_rate))),
            ("minimum_training_step", str(float(jnp.min(train_delta)))),
            ("maximum_training_step", str(float(jnp.max(train_delta)))),
            ("extrapolation_factor", str(float(extrapolation_factor))),
            ("population_seed", str(int(seed))),
        ),
    )


def beam_transient_scenario(
    *,
    spatial_points: int = 20,
    time_points: int = 16,
    num_cases: int = 5,
    final_time: float = 1.0,
) -> OperatorBenchmarkScenario:
    if float(final_time) <= 0.0:
        raise ValueError("final_time must be positive.")
    x_axis = _interval_axis("x", spatial_points)
    time_axis = OperatorAxis(
        "t",
        jnp.linspace(0.0, float(final_time), time_points),
        quadrature_weights=jnp.ones((time_points,)) * float(final_time) / time_points,
    )
    case = jnp.arange(1, num_cases + 1, dtype=float)[:, None]
    load = jnp.sin(jnp.pi * x_axis.nodes)[None, :] * (1.0 + 0.2 * case)
    time = time_axis.nodes[:, None]
    space = x_axis.nodes[None, :]
    response = (1.0 - jnp.cos(jnp.pi**2 * time)) * jnp.sin(jnp.pi * space) / jnp.pi**4
    target = (
        load[:, None, :]
        / jnp.maximum(jnp.sin(jnp.pi * x_axis.nodes)[None, None, :], 1e-8)
        * response[None, :, :]
    )
    batch = _grid_batch(
        {"load": FunctionSamples(values=load, axes=(x_axis,))},
        (time_axis, x_axis),
    )
    case_ids = _case_ids("euler_bernoulli_transient", num_cases)
    return OperatorBenchmarkScenario(
        "euler_bernoulli_transient",
        batch,
        target,
        (
            OperatorBenchmarkEvaluation(
                "transient",
                batch,
                target,
                case_ids=case_ids,
            ),
        ),
        case_ids=case_ids,
        provenance=_generated_provenance("beam_transient_scenario"),
        dimensional_parameters=(
            OperatorParameterRange(
                "final_time", float(final_time), float(final_time), "T"
            ),
            OperatorParameterRange("beam_length", 1.0, 1.0, "L"),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "modal_time",
                float(jnp.pi**2 * final_time),
                float(jnp.pi**2 * final_time),
                "1",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="closed-form Euler-Bernoulli first-mode response",
            verification="analytic",
            resolutions=(int(time_points), int(spatial_points)),
            relative_error=0.0,
            tolerance=0.0,
        ),
        regimes=("long_horizon", "space_time", "independent_query"),
        metadata=(("final_time", str(float(final_time))),),
    )


def _deformed_elliptic_map(point, deformation):
    xi, eta = point
    displacement = jnp.stack(
        (
            jnp.sin(jnp.pi * xi) * jnp.sin(2.0 * jnp.pi * eta),
            jnp.sin(2.0 * jnp.pi * xi) * jnp.sin(jnp.pi * eta),
        )
    )
    return point + deformation * displacement


def _deformed_elliptic_solution(point, coefficients, boundary_value):
    xi, eta = point
    interior = (
        coefficients[0] * jnp.sin(jnp.pi * xi) * jnp.sin(jnp.pi * eta)
        + 0.5 * coefficients[1] * jnp.sin(2.0 * jnp.pi * xi) * jnp.sin(jnp.pi * eta)
        + 0.5 * coefficients[2] * jnp.sin(jnp.pi * xi) * jnp.sin(2.0 * jnp.pi * eta)
    )
    return interior + boundary_value * (1.0 - xi)


def _deformed_elliptic_diffusivity(point, material):
    xi, eta = point
    variation = 0.5 + 0.25 * jnp.sin(2.0 * jnp.pi * xi) * jnp.cos(2.0 * jnp.pi * eta)
    return 1.0 + material * variation


def _deformed_elliptic_forcing(
    point,
    deformation,
    coefficients,
    material,
    boundary_value,
):
    map_fn = lambda coordinate: _deformed_elliptic_map(coordinate, deformation)
    solution_fn = lambda coordinate: _deformed_elliptic_solution(
        coordinate, coefficients, boundary_value
    )

    def reference_flux(coordinate):
        jacobian = jax.jacfwd(map_fn)(coordinate)
        reference_gradient = jax.grad(solution_fn)(coordinate)
        physical_gradient = jnp.linalg.solve(jacobian.T, reference_gradient)
        physical_flux = (
            _deformed_elliptic_diffusivity(coordinate, material) * physical_gradient
        )
        return jnp.linalg.det(jacobian) * jnp.linalg.solve(jacobian, physical_flux)

    jacobian = jax.jacfwd(map_fn)(point)
    flux_jacobian = jax.jacfwd(reference_flux)(point)
    return -jnp.trace(flux_jacobian) / jnp.linalg.det(jacobian)


def _deformed_elliptic_point_cloud(
    cases: int,
    count: int,
    *,
    seed: int,
    boundary_fraction: float,
):
    if int(count) < 8:
        raise ValueError("Deformed elliptic point clouds require at least eight points.")
    boundary_count = max(
        4,
        min(int(count) - 1, round(float(boundary_fraction) * int(count))),
    )
    interior_count = int(count) - boundary_count
    interior_key, boundary_key, permutation_key = jr.split(jr.key(seed), 3)
    interior = jr.uniform(
        interior_key,
        (int(cases), interior_count, 2),
        minval=0.025,
        maxval=0.975,
    )
    boundary_parameter = jr.uniform(
        boundary_key,
        (int(cases), boundary_count),
        minval=0.025,
        maxval=0.975,
    )
    edge = jnp.arange(boundary_count, dtype=jnp.int32) % 4
    first = jnp.where(
        (edge == 0)[None, :],
        0.0,
        jnp.where(
            (edge == 1)[None, :],
            1.0,
            boundary_parameter,
        ),
    )
    second = jnp.where(
        (edge == 2)[None, :],
        0.0,
        jnp.where(
            (edge == 3)[None, :],
            1.0,
            boundary_parameter,
        ),
    )
    boundary = jnp.stack((first, second), axis=-1)
    points = jnp.concatenate((interior, boundary), axis=1)
    indicator = jnp.concatenate(
        (
            jnp.zeros((int(cases), interior_count)),
            jnp.ones((int(cases), boundary_count)),
        ),
        axis=1,
    )
    permutations = jax.vmap(lambda key: jr.permutation(key, int(count)))(
        jr.split(permutation_key, int(cases))
    )
    points = jnp.take_along_axis(points, permutations[..., None], axis=1)
    indicator = jnp.take_along_axis(indicator, permutations, axis=1)
    return points, indicator


def deformed_elliptic_scenario(
    *,
    points: int = 48,
    query_points: int | None = None,
    num_cases: int = 12,
    deformation_amplitude: float = 0.03,
    geometry_extrapolation_factor: float = 1.5,
    sensor_dropout_fraction: float = 0.2,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    """Manufactured elliptic operator on per-realization deformed point clouds."""

    if int(num_cases) < 3:
        raise ValueError("deformed_elliptic_scenario requires at least three cases.")
    if float(deformation_amplitude) <= 0.0:
        raise ValueError("deformation_amplitude must be positive.")
    if float(geometry_extrapolation_factor) <= 1.0:
        raise ValueError("geometry_extrapolation_factor must exceed one.")
    if not 0.0 < float(sensor_dropout_fraction) < 1.0:
        raise ValueError(
            "sensor_dropout_fraction must lie strictly between zero and one."
        )
    query_count = int(points) if query_points is None else int(query_points)
    case_count = int(num_cases)
    population_key, material_key, boundary_key = jr.split(jr.key(seed), 3)
    coefficients = jr.normal(population_key, (case_count, 3))
    coefficients = coefficients / jnp.maximum(
        jnp.linalg.norm(coefficients, axis=-1, keepdims=True),
        1e-12,
    )
    material = jr.uniform(
        material_key,
        (case_count,),
        minval=0.2,
        maxval=0.8,
    )
    boundary_values = jr.uniform(
        boundary_key,
        (case_count,),
        minval=-0.2,
        maxval=0.2,
    )
    deformation = jnp.linspace(
        -float(deformation_amplitude),
        float(deformation_amplitude),
        case_count,
    )

    def build(
        *,
        source_count,
        target_count,
        current_deformation,
        current_boundary,
        source_seed,
        query_seed,
        source_mask=None,
    ):
        source_reference, source_boundary = _deformed_elliptic_point_cloud(
            case_count,
            int(source_count),
            seed=int(source_seed),
            boundary_fraction=0.2,
        )
        query_reference, _ = _deformed_elliptic_point_cloud(
            case_count,
            int(target_count),
            seed=int(query_seed),
            boundary_fraction=0.15,
        )
        map_points = jax.vmap(
            lambda points_, deformation_: jax.vmap(
                lambda point: _deformed_elliptic_map(point, deformation_)
            )(points_)
        )
        source_coordinates = map_points(source_reference, current_deformation)
        query_coordinates = map_points(query_reference, current_deformation)
        forcing = jax.vmap(
            lambda points_, deformation_, coefficients_, material_, boundary_: jax.vmap(
                lambda point: _deformed_elliptic_forcing(
                    point,
                    deformation_,
                    coefficients_,
                    material_,
                    boundary_,
                )
            )(points_)
        )(
            source_reference,
            current_deformation,
            coefficients,
            material,
            current_boundary,
        )
        diffusivity = jax.vmap(
            lambda points_, material_: jax.vmap(
                lambda point: _deformed_elliptic_diffusivity(point, material_)
            )(points_)
        )(source_reference, material)
        source_solution = jax.vmap(
            lambda points_, coefficients_, boundary_: jax.vmap(
                lambda point: _deformed_elliptic_solution(point, coefficients_, boundary_)
            )(points_)
        )(source_reference, coefficients, current_boundary)
        target = jax.vmap(
            lambda points_, coefficients_, boundary_: jax.vmap(
                lambda point: _deformed_elliptic_solution(point, coefficients_, boundary_)
            )(points_)
        )(query_reference, coefficients, current_boundary)

        def transformed_jacobians(reference):
            return jax.vmap(
                lambda points_, deformation_: jax.vmap(
                    lambda point: jnp.linalg.det(
                        jax.jacfwd(
                            lambda coordinate: _deformed_elliptic_map(
                                coordinate, deformation_
                            )
                        )(point)
                    )
                )(points_)
            )(reference, current_deformation)

        jacobian_determinants = transformed_jacobians(source_reference)
        query_jacobian_determinants = transformed_jacobians(query_reference)
        minimum_jacobian = float(
            min(jnp.min(jacobian_determinants), jnp.min(query_jacobian_determinants))
        )
        if minimum_jacobian <= 0.0:
            raise ValueError(
                "deformation_amplitude produces a non-positive map Jacobian."
            )
        quadrature = jacobian_determinants / jnp.sum(
            jacobian_determinants, axis=-1, keepdims=True
        )
        query_quadrature = query_jacobian_determinants / jnp.sum(
            query_jacobian_determinants,
            axis=-1,
            keepdims=True,
        )
        common = {
            "coordinates": source_coordinates,
            "quadrature_weights": quadrature,
            "mask": source_mask,
        }
        batch = OperatorBatch(
            inputs={
                "forcing": FunctionSamples(values=forcing, **common),
                "diffusivity": FunctionSamples(values=diffusivity, **common),
                "boundary_value": FunctionSamples(
                    values=source_boundary * source_solution,
                    **common,
                ),
                "boundary_indicator": FunctionSamples(
                    values=source_boundary,
                    **common,
                ),
            },
            queries={
                "query": FunctionSamples(
                    values=None,
                    coordinates=query_coordinates,
                    quadrature_weights=query_quadrature,
                )
            },
            case_axes=("case",),
        )
        return batch, target, minimum_jacobian

    train_batch, train_target, train_minimum_jacobian = build(
        source_count=int(points),
        target_count=query_count,
        current_deformation=deformation,
        current_boundary=boundary_values,
        source_seed=seed + 11,
        query_seed=seed + 17,
    )
    nominal_batch, nominal_target, _ = build(
        source_count=int(points),
        target_count=query_count,
        current_deformation=deformation,
        current_boundary=boundary_values,
        source_seed=seed + 11,
        query_seed=seed + 17,
    )
    independent_query_batch, independent_query_target, _ = build(
        source_count=int(points),
        target_count=query_count + max(4, query_count // 3),
        current_deformation=deformation,
        current_boundary=boundary_values,
        source_seed=seed + 11,
        query_seed=seed + 23,
    )
    resolution_batch, resolution_target, _ = build(
        source_count=int(points) + max(8, int(points) // 2),
        target_count=query_count + max(8, query_count // 2),
        current_deformation=deformation,
        current_boundary=boundary_values,
        source_seed=seed + 29,
        query_seed=seed + 31,
    )
    extrapolated_deformation = (
        jnp.where(jnp.arange(case_count) % 2 == 0, -1.0, 1.0)
        * float(deformation_amplitude)
        * float(geometry_extrapolation_factor)
    )
    extrapolated_batch, extrapolated_target, extrapolated_minimum_jacobian = build(
        source_count=int(points),
        target_count=query_count,
        current_deformation=extrapolated_deformation,
        current_boundary=boundary_values,
        source_seed=seed + 11,
        query_seed=seed + 17,
    )
    source_mask = jr.uniform(jr.key(seed + 37), (case_count, int(points))) >= float(
        sensor_dropout_fraction
    )
    source_mask = source_mask.at[:, 0].set(False)
    source_mask = source_mask.at[:, 1].set(True)
    sensor_batch, sensor_target, _ = build(
        source_count=int(points),
        target_count=query_count,
        current_deformation=deformation,
        current_boundary=boundary_values,
        source_seed=seed + 11,
        query_seed=seed + 17,
        source_mask=source_mask,
    )
    shifted_boundary = boundary_values + jnp.where(
        jnp.arange(case_count) % 2 == 0, -0.35, 0.35
    )
    boundary_batch, boundary_target, _ = build(
        source_count=int(points),
        target_count=query_count,
        current_deformation=deformation,
        current_boundary=shifted_boundary,
        source_seed=seed + 11,
        query_seed=seed + 17,
    )
    case_ids = _case_ids("deformed_elliptic_2d", case_count)
    scenario = OperatorBenchmarkScenario(
        "deformed_elliptic_2d",
        train_batch,
        train_target,
        (
            OperatorBenchmarkEvaluation(
                "nominal",
                nominal_batch,
                nominal_target,
                case_ids=case_ids,
            ),
            OperatorBenchmarkEvaluation(
                "resolution_transfer",
                resolution_batch,
                resolution_target,
                shift="resolution_transfer",
                case_ids=case_ids,
            ),
            OperatorBenchmarkEvaluation(
                "independent_query",
                independent_query_batch,
                independent_query_target,
                shift="independent_query",
                case_ids=case_ids,
            ),
            OperatorBenchmarkEvaluation(
                "geometry_extrapolation",
                extrapolated_batch,
                extrapolated_target,
                shift="geometry_extrapolation",
                case_ids=case_ids,
            ),
            OperatorBenchmarkEvaluation(
                "sensor_dropout",
                sensor_batch,
                sensor_target,
                shift="sensor_dropout",
                case_ids=case_ids,
            ),
            OperatorBenchmarkEvaluation(
                "boundary_condition_shift",
                boundary_batch,
                boundary_target,
                shift="boundary_condition",
                case_ids=case_ids,
            ),
        ),
        seed=int(seed),
        case_ids=case_ids,
        provenance=_generated_provenance("deformed_elliptic_scenario"),
        dimensional_parameters=(
            OperatorParameterRange("domain_length", 1.0, 1.0, "L"),
            OperatorParameterRange("diffusivity", 1.05, 1.6, "L^2 T^-1"),
            OperatorParameterRange(
                "geometry_deformation",
                -float(deformation_amplitude) * float(geometry_extrapolation_factor),
                float(deformation_amplitude) * float(geometry_extrapolation_factor),
                "L",
            ),
            OperatorParameterRange("boundary_value", -0.55, 0.55, "U"),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "deformation_fraction",
                0.0,
                float(deformation_amplitude) * float(geometry_extrapolation_factor),
                "1",
            ),
            OperatorParameterRange("diffusivity_contrast", 1.0, 1.6 / 1.05, "1"),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="exact curvilinear manufactured elliptic solution",
            verification="analytic",
            resolutions=(int(points), query_count),
            relative_error=0.0,
            tolerance=0.0,
        ),
        regimes=(
            "manufactured_elliptic",
            "irregular_geometry",
            "per_case_geometry",
            "independent_query",
            "resolution_transfer",
            "geometry_extrapolation",
            "sensor_dropout",
            "boundary_condition_shift",
        ),
        ladder="geometry_generalization",
        difficulty="hard",
        metadata=(
            ("deformation_amplitude", str(float(deformation_amplitude))),
            (
                "geometry_extrapolation_factor",
                str(float(geometry_extrapolation_factor)),
            ),
            ("sensor_dropout_fraction", str(float(sensor_dropout_fraction))),
            ("minimum_train_jacobian", str(train_minimum_jacobian)),
            (
                "minimum_extrapolated_jacobian",
                str(extrapolated_minimum_jacobian),
            ),
            ("source_points", str(int(points))),
            ("query_points", str(query_count)),
            ("population_seed", str(int(seed))),
        ),
    )
    return scenario


def conservative_ring_transport_scenario(
    *,
    source_points: int = 48,
    query_points: int | None = None,
    support_resolution: int = 16,
    num_cases: int = 12,
    radius_range: tuple[float, float] = (0.45, 0.7),
    center_extent: float = 0.1,
    band_width: float = 0.16,
    advection_time: float = 0.5,
    speed_range: tuple[float, float] = (0.4, 1.0),
    density_variation: float = 0.5,
    maximum_frequency: int = 6,
    geometry_extrapolation_factor: float = 1.35,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    """Mass-preserving transport on independently sampled, supported ring geometries."""

    source_count = int(source_points)
    query_count = source_count if query_points is None else int(query_points)
    support_size = int(support_resolution)
    case_count = int(num_cases)
    if min(source_count, query_count) < 8:
        raise ValueError("Ring transport requires at least eight source/query points.")
    if support_size < 4:
        raise ValueError("support_resolution must be at least four.")
    if case_count < 3:
        raise ValueError("Ring transport requires at least three physical cases.")
    radius_min, radius_max = (float(value) for value in radius_range)
    speed_min, speed_max = (float(value) for value in speed_range)
    if not 0.0 < radius_min < radius_max:
        raise ValueError("radius_range must be strictly positive and increasing.")
    if not 0.0 < speed_min < speed_max:
        raise ValueError("speed_range must be strictly positive and increasing.")
    if float(center_extent) < 0.0 or float(band_width) <= 0.0:
        raise ValueError("center_extent must be non-negative and band_width positive.")
    if float(advection_time) <= 0.0:
        raise ValueError("advection_time must be positive.")
    if not 0.0 < float(density_variation) < 1.0:
        raise ValueError("density_variation must lie strictly between zero and one.")
    if int(maximum_frequency) <= 0:
        raise ValueError("maximum_frequency must be positive.")
    if float(geometry_extrapolation_factor) <= 1.0:
        raise ValueError("geometry_extrapolation_factor must exceed one.")

    resolved_frequency = min(
        int(maximum_frequency),
        max(1, (min(source_count, query_count) - 1) // 2),
    )
    coefficient_key, center_key, radius_key, speed_key = jr.split(jr.key(seed), 4)
    frequencies, coefficients = _periodic_population_coefficients(
        coefficient_key,
        case_count,
        resolved_frequency,
    )
    coefficient_bound = jnp.sum(
        jnp.linalg.norm(coefficients, axis=-1),
        axis=-1,
    )
    density_scale = float(density_variation) / jnp.maximum(
        coefficient_bound,
        1e-12,
    )
    centers = jr.uniform(
        center_key,
        (case_count, 2),
        minval=-float(center_extent),
        maxval=float(center_extent),
    )
    radii = jr.uniform(
        radius_key,
        (case_count,),
        minval=radius_min,
        maxval=radius_max,
    )
    speeds = jr.uniform(
        speed_key,
        (case_count,),
        minval=speed_min,
        maxval=speed_max,
    )
    extrapolation = float(geometry_extrapolation_factor)
    radius_midpoint = 0.5 * (radius_min + radius_max)
    extrapolated_centers = extrapolation * centers
    extrapolated_radii = radius_midpoint + extrapolation * (radii - radius_midpoint)
    minimum_extrapolated_radius = float(jnp.min(extrapolated_radii))
    if minimum_extrapolated_radius <= 0.0:
        raise ValueError("geometry_extrapolation_factor produces a non-positive radius.")

    maximum_center = extrapolation * float(center_extent)
    maximum_radius = max(radius_max, float(jnp.max(extrapolated_radii)))
    support_extent = maximum_center + maximum_radius + float(band_width) + 0.1
    support_axis = jnp.linspace(-support_extent, support_extent, support_size)
    support_x, support_y = jnp.meshgrid(
        support_axis,
        support_axis,
        indexing="ij",
    )
    support_coordinates = jnp.stack((support_x, support_y), axis=-1).reshape((-1, 2))
    support_count = int(support_coordinates.shape[0])
    support_weights = jnp.full(
        (support_count,),
        (2.0 * support_extent) ** 2 / support_count,
    )

    def density(angles, displacement):
        shifted = angles[None, :] - displacement[:, None]
        phase = frequencies[None, :, None] * shifted[:, None, :]
        basis = jnp.stack((jnp.sin(phase), jnp.cos(phase)), axis=-1)
        variation = ein.contract("cmk,cmpk->cp", coefficients, basis)
        return 1.0 + density_scale[:, None] * variation

    def ring_geometry(current_centers, current_radii, count, offset):
        angles = (
            2.0
            * jnp.pi
            * (jnp.arange(int(count), dtype=float) + float(offset))
            / int(count)
        )
        unit = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)
        coordinates = (
            current_centers[:, None, :] + current_radii[:, None, None] * unit[None, :, :]
        )
        weights = jnp.broadcast_to(
            2.0 * jnp.pi * current_radii[:, None] / int(count),
            (case_count, int(count)),
        )
        return angles, coordinates, weights

    def build(
        *,
        current_source_count,
        current_query_count,
        current_centers,
        current_radii,
        current_speeds,
        current_band_width,
        source_offset,
        query_offset,
    ):
        source_angles, source_coordinates, source_weights = ring_geometry(
            current_centers,
            current_radii,
            current_source_count,
            source_offset,
        )
        query_angles, query_coordinates, query_weights = ring_geometry(
            current_centers,
            current_radii,
            current_query_count,
            query_offset,
        )
        source_density = density(source_angles, jnp.zeros_like(current_speeds))
        target = density(
            query_angles,
            current_speeds * float(advection_time),
        )
        relative_support = support_coordinates[None, :, :] - current_centers[:, None, :]
        domain_sdf = jnp.abs(
            jnp.linalg.norm(relative_support, axis=-1) - current_radii[:, None]
        ) - float(current_band_width)
        speed_field = jnp.broadcast_to(
            current_speeds[:, None],
            (case_count, int(current_source_count)),
        )
        batch = OperatorBatch(
            inputs={
                "density": FunctionSamples(
                    values=source_density,
                    coordinates=source_coordinates,
                    quadrature_weights=source_weights,
                ),
                "speed": FunctionSamples(
                    values=speed_field,
                    coordinates=source_coordinates,
                    quadrature_weights=source_weights,
                ),
                "domain_sdf": FunctionSamples(
                    values=domain_sdf,
                    coordinates=support_coordinates,
                    quadrature_weights=support_weights,
                ),
            },
            queries={
                "query": FunctionSamples(
                    values=None,
                    coordinates=query_coordinates,
                    quadrature_weights=query_weights,
                )
            },
            case_axes=("case",),
        )
        source_mass = jnp.sum(source_density * source_weights, axis=-1)
        target_mass = jnp.sum(target * query_weights, axis=-1)
        relative_mass_error = float(
            jnp.max(
                jnp.abs(target_mass - source_mass)
                / jnp.maximum(jnp.abs(source_mass), 1e-12)
            )
        )
        return batch, target, relative_mass_error

    common = {
        "current_centers": centers,
        "current_radii": radii,
        "current_speeds": speeds,
        "current_band_width": float(band_width),
    }
    train_batch, train_target, train_mass_error = build(
        current_source_count=source_count,
        current_query_count=query_count,
        source_offset=0.13,
        query_offset=0.47,
        **common,
    )
    nominal_batch, nominal_target, nominal_mass_error = build(
        current_source_count=source_count,
        current_query_count=query_count,
        source_offset=0.13,
        query_offset=0.47,
        **common,
    )
    resolution_source_count = source_count + max(8, source_count // 2)
    resolution_query_count = query_count + max(8, query_count // 2)
    resolution_batch, resolution_target, resolution_mass_error = build(
        current_source_count=resolution_source_count,
        current_query_count=resolution_query_count,
        source_offset=0.31,
        query_offset=0.71,
        **common,
    )
    independent_query_count = query_count + max(5, query_count // 3)
    independent_batch, independent_target, independent_mass_error = build(
        current_source_count=source_count,
        current_query_count=independent_query_count,
        source_offset=0.13,
        query_offset=0.83,
        **common,
    )
    extrapolated_batch, extrapolated_target, extrapolated_mass_error = build(
        current_source_count=source_count,
        current_query_count=query_count,
        current_centers=extrapolated_centers,
        current_radii=extrapolated_radii,
        current_speeds=speeds,
        current_band_width=float(band_width),
        source_offset=0.13,
        query_offset=0.47,
    )
    support_batch, support_target, support_mass_error = build(
        current_source_count=source_count,
        current_query_count=query_count,
        current_centers=centers,
        current_radii=radii,
        current_speeds=speeds,
        current_band_width=0.65 * float(band_width),
        source_offset=0.13,
        query_offset=0.47,
    )
    shifted_speeds = 1.5 * speeds
    speed_batch, speed_target, speed_mass_error = build(
        current_source_count=source_count,
        current_query_count=query_count,
        current_centers=centers,
        current_radii=radii,
        current_speeds=shifted_speeds,
        current_band_width=float(band_width),
        source_offset=0.13,
        query_offset=0.47,
    )
    mass_errors = (
        train_mass_error,
        nominal_mass_error,
        resolution_mass_error,
        independent_mass_error,
        extrapolated_mass_error,
        support_mass_error,
        speed_mass_error,
    )
    case_ids = _case_ids("conservative_ring_transport_2d", case_count)
    return OperatorBenchmarkScenario(
        "conservative_ring_transport_2d",
        train_batch,
        train_target,
        (
            OperatorBenchmarkEvaluation(
                "nominal",
                nominal_batch,
                nominal_target,
                case_ids=case_ids,
            ),
            OperatorBenchmarkEvaluation(
                "resolution_transfer",
                resolution_batch,
                resolution_target,
                shift="resolution_transfer",
                case_ids=case_ids,
            ),
            OperatorBenchmarkEvaluation(
                "independent_query",
                independent_batch,
                independent_target,
                shift="independent_query",
                case_ids=case_ids,
            ),
            OperatorBenchmarkEvaluation(
                "geometry_extrapolation",
                extrapolated_batch,
                extrapolated_target,
                shift="geometry_extrapolation",
                case_ids=case_ids,
            ),
            OperatorBenchmarkEvaluation(
                "support_extrapolation",
                support_batch,
                support_target,
                shift="support_extrapolation",
                case_ids=case_ids,
            ),
            OperatorBenchmarkEvaluation(
                "speed_extrapolation",
                speed_batch,
                speed_target,
                shift="parameter_extrapolation",
                case_ids=case_ids,
            ),
        ),
        seed=int(seed),
        case_ids=case_ids,
        provenance=_generated_provenance("conservative_ring_transport_scenario"),
        dimensional_parameters=(
            OperatorParameterRange(
                "ring_radius",
                min(radius_min, minimum_extrapolated_radius),
                max(radius_max, float(jnp.max(extrapolated_radii))),
                "L",
            ),
            OperatorParameterRange(
                "ring_center_coordinate",
                -maximum_center,
                maximum_center,
                "L",
            ),
            OperatorParameterRange(
                "support_half_width",
                0.65 * float(band_width),
                float(band_width),
                "L",
            ),
            OperatorParameterRange(
                "angular_speed",
                speed_min,
                1.5 * speed_max,
                "T^-1",
            ),
            OperatorParameterRange(
                "advection_time",
                float(advection_time),
                float(advection_time),
                "T",
            ),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "transport_phase_fraction",
                speed_min * float(advection_time) / (2.0 * jnp.pi),
                1.5 * speed_max * float(advection_time) / (2.0 * jnp.pi),
                "1",
            ),
            OperatorParameterRange(
                "support_width_to_radius",
                0.65 * float(band_width) / radius_max,
                float(band_width) / min(radius_min, minimum_extrapolated_radius),
                "1",
            ),
            OperatorParameterRange(
                "density_variation",
                float(density_variation),
                float(density_variation),
                "1",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="analytic characteristic rotation with exact periodic quadrature",
            verification="analytic",
            resolutions=(
                source_count,
                query_count,
                resolution_source_count,
                resolution_query_count,
            ),
            relative_error=max(mass_errors),
            tolerance=1e-10,
        ),
        regimes=(
            "conservative_transport",
            "irregular_geometry",
            "per_case_geometry",
            "independent_query",
            "resolution_transfer",
            "geometry_extrapolation",
            "support_extrapolation",
            "parameter_extrapolation",
            "signed_distance_support",
        ),
        metadata=(
            ("source_points", str(source_count)),
            ("query_points", str(query_count)),
            ("support_resolution", str(support_size)),
            ("resolved_frequency", str(resolved_frequency)),
            ("advection_time", str(float(advection_time))),
            ("band_width", str(float(band_width))),
            ("population_seed", str(int(seed))),
            ("maximum_relative_mass_error", str(max(mass_errors))),
        ),
        domain_support_key="domain_sdf",
        domain_support_kind="sdf",
        domain_support_threshold=0.0,
        conservation_source_key="density",
    )


def _spectral_project_square(values: np.ndarray, output_size: int, /) -> np.ndarray:
    """Project a periodic square-grid field onto a smaller Fourier grid."""

    array = np.asarray(values)
    input_size = int(array.shape[-1])
    if array.shape[-2:] != (input_size, input_size):
        raise ValueError("Spectral square projection requires equal trailing grid axes.")
    if not 1 < int(output_size) <= input_size:
        raise ValueError("output_size must lie between two and the input grid size.")
    if int(output_size) == input_size:
        return array.copy()
    transformed = np.fft.fftshift(
        np.fft.fft2(array, axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    )
    start = (input_size - int(output_size)) // 2
    stop = start + int(output_size)
    cropped = transformed[..., start:stop, start:stop]
    projected = np.fft.ifft2(
        np.fft.ifftshift(cropped, axes=(-2, -1)),
        axes=(-2, -1),
        norm="ortho",
    ).real
    return projected * (float(output_size) / float(input_size))


def polynomial_poisson_scenario(
    *,
    resolution: int = 16,
    num_cases: int = 8,
    polynomial_degree: int = 2,
    maximum_frequency: int | None = None,
    amplitude_shift: float = 1.5,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    """Periodic polynomial-source Poisson map with alias-free reference targets.

    The learned map is ``v ↦ u``, where ``-Δu = v**p - mean(v**p)`` and ``u``
    has zero mean. Source fields are finite Fourier series. Targets are formed on
    a grid resolving the full polynomial bandwidth, then Fourier-projected onto
    each requested output grid.
    """

    grid_size = int(resolution)
    case_count = int(num_cases)
    degree = int(polynomial_degree)
    if grid_size < 8:
        raise ValueError("resolution must be at least eight.")
    if case_count <= 0:
        raise ValueError("num_cases must be positive.")
    if degree <= 0:
        raise ValueError("polynomial_degree must be positive.")
    if float(amplitude_shift) <= 1.0:
        raise ValueError("amplitude_shift must be greater than one.")

    requested_frequency = (
        max(2, grid_size // 3) if maximum_frequency is None else int(maximum_frequency)
    )
    if requested_frequency <= 0:
        raise ValueError("maximum_frequency must be positive.")
    resolved_frequency = min(requested_frequency, (grid_size - 1) // 2)
    shifted_frequency = min(
        (grid_size - 1) // 2,
        resolved_frequency + max(1, resolved_frequency // 3),
    )
    transfer_resolution = 2 * grid_size
    required_reference = max(
        transfer_resolution,
        2 * (degree * shifted_frequency + 2),
    )
    reference_resolution = (
        required_reference if required_reference % 2 == 0 else required_reference + 1
    )

    reference_axis = np.arange(reference_resolution, dtype=float) / float(
        reference_resolution
    )
    reference_x, reference_y = np.meshgrid(
        reference_axis,
        reference_axis,
        indexing="ij",
    )
    reference_coordinates = jnp.stack(
        (jnp.asarray(reference_x), jnp.asarray(reference_y)),
        axis=-1,
    )
    frequency = (
        2.0
        * np.pi
        * np.fft.fftfreq(
            reference_resolution,
            d=1.0 / reference_resolution,
        )
    )
    frequency_x, frequency_y = np.meshgrid(frequency, frequency, indexing="ij")
    negative_laplacian = frequency_x**2 + frequency_y**2

    def make_population(
        *,
        output_resolution: int,
        population_cases: int,
        population_frequency: int,
        population_amplitude: float,
        population_seed: int,
        label: str,
    ):
        mode_x, mode_y, coefficients = _planar_population_coefficients(
            jr.key(population_seed),
            population_cases,
            population_frequency,
        )
        coefficients = coefficients * float(population_amplitude)
        output_axis = _periodic_axis("x", output_resolution)
        output_axis_y = _periodic_axis("y", output_resolution)
        output_x, output_y = jnp.meshgrid(
            output_axis.nodes,
            output_axis_y.nodes,
            indexing="ij",
        )
        output_coordinates = jnp.stack((output_x, output_y), axis=-1)
        source = _evaluate_planar_population(
            coefficients,
            mode_x,
            mode_y,
            output_coordinates,
        )
        reference_source = np.asarray(
            jax.device_get(
                _evaluate_planar_population(
                    coefficients,
                    mode_x,
                    mode_y,
                    reference_coordinates,
                )
            )
        )
        forcing = reference_source**degree
        forcing = forcing - np.mean(forcing, axis=(-2, -1), keepdims=True)
        forcing_hat = np.fft.fft2(forcing, axes=(-2, -1))
        solution_hat = np.zeros_like(forcing_hat)
        nonzero = negative_laplacian > 0.0
        solution_hat[..., nonzero] = (
            forcing_hat[..., nonzero] / negative_laplacian[nonzero]
        )
        reference_solution = np.fft.ifft2(
            solution_hat,
            axes=(-2, -1),
        ).real
        target = jnp.asarray(
            _spectral_project_square(reference_solution, output_resolution)
        )
        reconstructed = solution_hat * negative_laplacian
        denominator = max(float(np.linalg.norm(forcing_hat)), 1e-15)
        residual = float(np.linalg.norm(reconstructed - forcing_hat) / denominator)
        batch = _grid_batch(
            {
                "source": FunctionSamples(
                    values=source,
                    axes=(output_axis, output_axis_y),
                )
            },
            (output_axis, output_axis_y),
        )
        return (
            batch,
            target,
            residual,
            _case_ids(
                f"polynomial_poisson_2d:{label}:degree_{degree}",
                population_cases,
            ),
        )

    train_batch, train_target, train_residual, train_ids = make_population(
        output_resolution=grid_size,
        population_cases=case_count,
        population_frequency=resolved_frequency,
        population_amplitude=1.0,
        population_seed=seed,
        label="train",
    )
    validation_batch, validation_target, validation_residual, validation_ids = (
        make_population(
            output_resolution=grid_size,
            population_cases=case_count,
            population_frequency=resolved_frequency,
            population_amplitude=1.0,
            population_seed=seed + 1,
            label="validation",
        )
    )
    test_batch, test_target, test_residual, test_ids = make_population(
        output_resolution=grid_size,
        population_cases=case_count,
        population_frequency=resolved_frequency,
        population_amplitude=1.0,
        population_seed=seed + 2,
        label="test",
    )
    amplitude_batch, amplitude_target, amplitude_residual, amplitude_ids = (
        make_population(
            output_resolution=grid_size,
            population_cases=case_count,
            population_frequency=resolved_frequency,
            population_amplitude=float(amplitude_shift),
            population_seed=seed + 3,
            label="amplitude_shift",
        )
    )
    frequency_batch, frequency_target, frequency_residual, frequency_ids = (
        make_population(
            output_resolution=grid_size,
            population_cases=case_count,
            population_frequency=shifted_frequency,
            population_amplitude=1.0,
            population_seed=seed + 4,
            label="frequency_shift",
        )
    )
    transfer_batch, transfer_target, transfer_residual, transfer_ids = make_population(
        output_resolution=transfer_resolution,
        population_cases=case_count,
        population_frequency=resolved_frequency,
        population_amplitude=1.0,
        population_seed=seed + 5,
        label="resolution_transfer",
    )

    evaluations = (
        OperatorBenchmarkEvaluation(
            "in_distribution",
            test_batch,
            test_target,
            case_ids=test_ids,
        ),
        OperatorBenchmarkEvaluation(
            "amplitude_shift",
            amplitude_batch,
            amplitude_target,
            shift="amplitude_extrapolation",
            case_ids=amplitude_ids,
        ),
        OperatorBenchmarkEvaluation(
            "frequency_shift",
            frequency_batch,
            frequency_target,
            shift="frequency_extrapolation",
            case_ids=frequency_ids,
        ),
        OperatorBenchmarkEvaluation(
            "resolution_transfer",
            transfer_batch,
            transfer_target,
            shift="resolution_transfer",
            case_ids=transfer_ids,
        ),
    )
    residuals = (
        train_residual,
        validation_residual,
        test_residual,
        amplitude_residual,
        frequency_residual,
        transfer_residual,
    )
    return OperatorBenchmarkScenario(
        name=f"polynomial_poisson_2d_p{degree}",
        train_batch=train_batch,
        train_target=train_target,
        evaluations=evaluations,
        validation=OperatorBenchmarkEvaluation(
            "validation",
            validation_batch,
            validation_target,
            split="validation",
            case_ids=validation_ids,
        ),
        seed=seed,
        case_ids=train_ids,
        provenance=_generated_provenance("polynomial_poisson_scenario"),
        dimensional_parameters=(
            OperatorParameterRange(
                "domain_period",
                1.0,
                1.0,
                "normalized length",
            ),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "polynomial_degree",
                float(degree),
                float(degree),
                "1",
            ),
            OperatorParameterRange(
                "source_amplitude",
                1.0,
                float(amplitude_shift),
                "1",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method=(
                "oversampled Fourier solution and orthogonal projection of the "
                "band-limited polynomial forcing"
            ),
            verification="discrete_residual",
            resolutions=(reference_resolution,),
            relative_error=max(residuals),
            tolerance=5e-12,
        ),
        regimes=(
            "periodic_tensor_grid",
            "polynomial_nonlinearity",
            "resolution_transfer",
            "amplitude_extrapolation",
            "frequency_extrapolation",
        ),
        metadata=(
            ("polynomial_degree", str(degree)),
            ("source_maximum_frequency", str(resolved_frequency)),
            ("shifted_maximum_frequency", str(shifted_frequency)),
            ("reference_resolution", str(reference_resolution)),
            ("population_seed", str(seed)),
            ("target_construction", "oversampled_fourier_projection"),
            ("sensor_shift_policy", "disabled_for_mechanism_isolation"),
        ),
    )


def irregular_poisson_scenario(
    *,
    points: int = 32,
    num_cases: int = 5,
    geometry_shift: bool = False,
    deformation_amplitude: float = 0.04,
    maximum_frequency: int = 4,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    if float(deformation_amplitude) < 0.0:
        raise ValueError("deformation_amplitude must be non-negative.")
    if int(maximum_frequency) <= 0:
        raise ValueError("maximum_frequency must be positive.")
    resolved_frequency = min(
        int(maximum_frequency),
        max(1, (int(points) - 1) // 2),
    )
    index = jnp.arange(points, dtype=float)
    source_coordinate = jnp.stack(
        (
            (0.5 + index * 0.61803398875) % 1.0,
            (0.25 + index * 0.41421356237) % 1.0,
        ),
        axis=-1,
    )
    mode_x, mode_y, population_coefficients = _planar_population_coefficients(
        jr.key(seed),
        num_cases,
        resolved_frequency,
    )
    weights = jnp.ones((points,)) / points

    def build(current_source):
        current_query = current_source[::-1]
        forcing = _evaluate_planar_population(
            population_coefficients,
            mode_x,
            mode_y,
            current_source,
        )
        displacement = current_query[:, None, :] - current_source[None, :, :]
        distance = jnp.sqrt(jnp.sum(displacement**2, axis=-1) + 1e-3)
        kernel = -jnp.log(distance) / (2.0 * jnp.pi)
        target = ein.contract("qs,cs,s->cq", kernel, forcing, weights)
        batch = OperatorBatch(
            inputs={
                "forcing": FunctionSamples(
                    values=forcing,
                    coordinates=current_source,
                    quadrature_weights=weights,
                )
            },
            queries={"query": FunctionSamples(values=None, coordinates=current_query)},
            case_axes=("case",),
        )
        return batch, target

    batch, target = build(source_coordinate)
    case_ids = _case_ids("irregular_poisson_2d", num_cases)
    evaluations = [
        OperatorBenchmarkEvaluation(
            "permuted_query",
            batch,
            target,
            case_ids=case_ids,
        ),
    ]
    if geometry_shift:
        deformation = float(deformation_amplitude) * jnp.stack(
            (
                jnp.sin(2.0 * jnp.pi * source_coordinate[:, 1]),
                jnp.cos(2.0 * jnp.pi * source_coordinate[:, 0]),
            ),
            axis=-1,
        )
        shifted_batch, shifted_target = build(source_coordinate + deformation)
        evaluations.append(
            OperatorBenchmarkEvaluation(
                "deformed_geometry",
                shifted_batch,
                shifted_target,
                shift="geometry",
                case_ids=case_ids,
            )
        )
    return OperatorBenchmarkScenario(
        "irregular_poisson_2d",
        batch,
        target,
        tuple(evaluations),
        seed=int(seed),
        case_ids=case_ids,
        provenance=_generated_provenance("irregular_poisson_scenario"),
        dimensional_parameters=(
            OperatorParameterRange(
                "kernel_regularization_length",
                float(jnp.sqrt(1e-3)),
                float(jnp.sqrt(1e-3)),
                "L",
            ),
            OperatorParameterRange(
                "geometry_deformation",
                0.0,
                float(deformation_amplitude) if geometry_shift else 0.0,
                "L",
            ),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "deformation_fraction",
                0.0,
                float(deformation_amplitude) if geometry_shift else 0.0,
                "1",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="direct regularized Green-kernel quadrature",
            verification="analytic",
            resolutions=(int(points),),
            relative_error=0.0,
            tolerance=0.0,
        ),
        regimes=(
            "irregular_geometry",
            "independent_query",
            "geometry_extrapolation" if geometry_shift else "fixed_geometry",
            "independent_multi_mode_population",
        ),
        metadata=(
            ("deformation_amplitude", str(float(deformation_amplitude))),
            ("geometry_shift", str(bool(geometry_shift))),
            ("maximum_frequency", str(int(maximum_frequency))),
            ("resolved_frequency", str(int(resolved_frequency))),
            ("population_seed", str(int(seed))),
        ),
    )


def graph_diffusion_scenario(
    *,
    nodes: int = 24,
    test_nodes: int | None = None,
    num_cases: int = 5,
    dt: float = 0.05,
    geometry_shift: bool = False,
    deformation_amplitude: float = 0.12,
    maximum_frequency: int = 6,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    if float(deformation_amplitude) < 0.0:
        raise ValueError("deformation_amplitude must be non-negative.")
    if int(maximum_frequency) <= 0:
        raise ValueError("maximum_frequency must be positive.")
    transfer_nodes = int(nodes if test_nodes is None else min(nodes, test_nodes))
    resolved_frequency = min(
        int(maximum_frequency),
        max(1, (transfer_nodes - 1) // 2),
    )
    frequencies, population_coefficients = _periodic_population_coefficients(
        jr.key(seed),
        num_cases,
        resolved_frequency,
    )

    def build(count, *, deformed=False):
        angle = 2.0 * jnp.pi * jnp.arange(count) / count
        radius = jnp.ones_like(angle)
        if deformed:
            radius = radius + float(deformation_amplitude) * jnp.cos(3.0 * angle)
        coordinates = radius[:, None] * jnp.stack(
            (jnp.cos(angle), jnp.sin(angle)),
            axis=-1,
        )
        values = _evaluate_periodic_population(
            population_coefficients,
            frequencies,
            angle,
        )
        laplacian = (
            jnp.roll(values, 1, axis=-1) - 2.0 * values + jnp.roll(values, -1, axis=-1)
        )
        target = values + dt * count**2 * laplacian / (2.0 * jnp.pi) ** 2
        weights = jnp.full((count,), 2.0 * jnp.pi / count)
        samples = FunctionSamples(
            values=values,
            coordinates=coordinates,
            quadrature_weights=weights,
        )
        batch = OperatorBatch(
            inputs={"state": samples},
            queries={"query": FunctionSamples(values=None, coordinates=coordinates)},
            case_axes=("case",),
        )
        return batch, target

    train_batch, train_target = build(nodes)
    case_ids = _case_ids("graph_diffusion_resolution_transfer", num_cases)
    evaluations = [
        OperatorBenchmarkEvaluation(
            "ring_graph",
            train_batch,
            train_target,
            case_ids=case_ids,
        ),
    ]
    if test_nodes is not None and int(test_nodes) != int(nodes):
        resolution_batch, resolution_target = build(int(test_nodes))
        evaluations.append(
            OperatorBenchmarkEvaluation(
                "higher_resolution",
                resolution_batch,
                resolution_target,
                shift="resolution",
                case_ids=case_ids,
            )
        )
    if geometry_shift:
        geometry_batch, geometry_target = build(nodes, deformed=True)
        evaluations.append(
            OperatorBenchmarkEvaluation(
                "deformed_ring",
                geometry_batch,
                geometry_target,
                shift="geometry",
                case_ids=case_ids,
            )
        )
    return OperatorBenchmarkScenario(
        "graph_diffusion_resolution_transfer",
        train_batch,
        train_target,
        tuple(evaluations),
        seed=int(seed),
        metadata=(
            ("dt", str(float(dt))),
            ("deformation_amplitude", str(float(deformation_amplitude))),
            ("maximum_frequency", str(int(maximum_frequency))),
            ("resolved_frequency", str(int(resolved_frequency))),
            ("population_seed", str(int(seed))),
        ),
        case_ids=case_ids,
        provenance=_generated_provenance("graph_diffusion_scenario"),
        dimensional_parameters=(
            OperatorParameterRange("time_step", float(dt), float(dt), "T"),
            OperatorParameterRange(
                "geometry_deformation",
                0.0,
                float(deformation_amplitude) if geometry_shift else 0.0,
                "L",
            ),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "diffusion_step",
                float(dt),
                float(dt),
                "1",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="explicit cycle-graph finite-difference update",
            verification="discrete_residual",
            resolutions=(
                int(nodes),
                int(nodes if test_nodes is None else test_nodes),
            ),
            relative_error=0.0,
            tolerance=1e-12,
        ),
        regimes=(
            "irregular_geometry",
            "resolution_transfer",
            "geometry_extrapolation" if geometry_shift else "fixed_geometry",
            "independent_multi_mode_population",
        ),
    )


def spherical_diffusion_scenario(
    *,
    bandlimit: int = 12,
    sampling: str = "mw",
    num_cases: int = 6,
    diffusivity: float = 0.02,
    dt: float = 0.1,
    target_steps: int = 1,
    maximum_degree: int = 3,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    if int(bandlimit) <= 1:
        raise ValueError("Spherical bandlimit must exceed one.")
    if float(diffusivity) <= 0.0 or float(dt) <= 0.0:
        raise ValueError("Spherical diffusivity and dt must be positive.")
    if int(target_steps) <= 0:
        raise ValueError("target_steps must be positive.")
    if int(maximum_degree) <= 0:
        raise ValueError("maximum_degree must be positive.")
    plan = SphericalHarmonicPlan(int(bandlimit), sampling=sampling)
    resolved_degree = min(int(maximum_degree), plan.bandlimit - 1)
    theta = plan.theta
    phi = plan.phi
    theta_axis = OperatorAxis(
        "theta",
        theta,
        quadrature_weights=plan.theta_quadrature_weights,
        basis="sphere",
    )
    phi_axis = OperatorAxis(
        "phi",
        phi,
        quadrature_weights=plan.phi_quadrature_weights,
        basis="fourier",
        periodic=True,
    )
    theta_grid, phi_grid = jnp.meshgrid(theta, phi, indexing="ij")
    cosine_theta = jnp.cos(theta_grid)
    sine_theta = jnp.sqrt(jnp.maximum(0.0, 1.0 - cosine_theta**2))

    def associated_legendre(degree: int, order: int):
        polynomial = jnp.ones_like(cosine_theta)
        for factor in range(1, 2 * order, 2):
            polynomial = -factor * sine_theta * polynomial
        if degree == order:
            return polynomial
        next_polynomial = (2 * order + 1) * cosine_theta * polynomial
        if degree == order + 1:
            return next_polynomial
        for current_degree in range(order + 2, degree + 1):
            polynomial, next_polynomial = (
                next_polynomial,
                (
                    (2 * current_degree - 1) * cosine_theta * next_polynomial
                    - (current_degree + order - 1) * polynomial
                )
                / (current_degree - order),
            )
        return next_polynomial

    basis = []
    degrees = []
    for degree in range(1, resolved_degree + 1):
        for order in range(degree + 1):
            polynomial = associated_legendre(degree, order)
            angular_modes = (
                (polynomial,)
                if order == 0
                else (
                    polynomial * jnp.cos(order * phi_grid),
                    polynomial * jnp.sin(order * phi_grid),
                )
            )
            for mode in angular_modes:
                basis.append(mode / jnp.sqrt(jnp.mean(mode**2)))
                degrees.append(degree)
    basis_array = jnp.stack(basis)
    degrees_array = jnp.asarray(degrees)
    coefficients = jr.normal(jr.key(seed), (num_cases, len(basis)))
    coefficients = coefficients / (
        degrees_array.astype(float)[None, :] ** 1.5 * jnp.sqrt(len(basis))
    )
    values = ein.contract("cm,mxy->cxy", coefficients, basis_array)
    attenuation = jnp.exp(
        -float(diffusivity)
        * float(dt)
        * int(target_steps)
        * degrees_array
        * (degrees_array + 1)
    )
    target = ein.contract(
        "cm,m,mxy->cxy",
        coefficients,
        attenuation,
        basis_array,
    )
    batch = _grid_batch(
        {"field": FunctionSamples(values=values, axes=(theta_axis, phi_axis))},
        (theta_axis, phi_axis),
    )
    case_ids = _case_ids("spherical_diffusion", num_cases)
    return OperatorBenchmarkScenario(
        "spherical_diffusion",
        batch,
        target,
        (
            OperatorBenchmarkEvaluation(
                "spherical_grid",
                batch,
                target,
                case_ids=case_ids,
            ),
        ),
        seed=int(seed),
        case_ids=case_ids,
        provenance=_generated_provenance("spherical_diffusion_scenario"),
        dimensional_parameters=(
            OperatorParameterRange(
                "surface_diffusivity",
                float(diffusivity),
                float(diffusivity),
                "L^2 T^-1",
                "log",
            ),
            OperatorParameterRange("time_step", float(dt), float(dt), "T"),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "degree_one_diffusion_time",
                float(2.0 * diffusivity * dt * target_steps),
                float(2.0 * diffusivity * dt * target_steps),
                "1",
                "log",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="analytic multi-degree spherical-harmonic diffusion",
            verification="analytic",
            resolutions=plan.sample_shape,
            relative_error=0.0,
            tolerance=0.0,
        ),
        regimes=(
            "spherical_field",
            "tensor_grid",
            "multi_step" if int(target_steps) > 1 else "single_step",
        ),
        metadata=(
            ("diffusivity", str(float(diffusivity))),
            ("dt", str(float(dt))),
            ("target_steps", str(int(target_steps))),
            ("maximum_degree", str(int(maximum_degree))),
            ("resolved_degree", str(int(resolved_degree))),
            ("bandlimit", str(plan.bandlimit)),
            ("sampling", plan.sampling),
            ("sensor_shift_policy", "disabled"),
            ("population_seed", str(int(seed))),
        ),
    )


def _square_triangle_complex(points: int, /, *, warp: float = 0.0):
    """Build a consistently oriented triangular square mesh with physical metrics."""
    resolved_points = int(points)
    if resolved_points < 3:
        raise ValueError("Square cochain meshes require at least three points per axis.")
    if not 0.0 <= float(warp) < 0.35:
        raise ValueError("Square mesh warp must lie in [0, 0.35).")
    axis = np.linspace(0.0, 1.0, resolved_points)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    if float(warp) > 0.0:
        y = y + float(warp) * np.sin(np.pi * x) * np.sin(np.pi * y)
    vertices = np.stack((x, y), axis=-1).reshape((-1, 2))
    faces = []
    for i in range(resolved_points - 1):
        for j in range(resolved_points - 1):
            lower_left = i * resolved_points + j
            lower_right = (i + 1) * resolved_points + j
            upper_right = (i + 1) * resolved_points + j + 1
            upper_left = i * resolved_points + j + 1
            faces.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    return triangle_mesh_to_cochain_complex(
        vertices,
        np.asarray(faces, dtype=np.int32),
    )


def _annulus_triangle_complex(radial_layers: int, angular_points: int, /):
    """Build an oriented periodic annulus triangulation."""
    resolved_layers = int(radial_layers)
    resolved_angles = int(angular_points)
    if resolved_layers < 1 or resolved_angles < 6:
        raise ValueError(
            "Annulus cochain meshes require one radial layer and six angular points."
        )
    radii = np.linspace(0.45, 1.0, resolved_layers + 1)
    angles = 2.0 * np.pi * np.arange(resolved_angles) / resolved_angles
    vertices = np.asarray(
        [
            (radius * np.cos(angle), radius * np.sin(angle))
            for radius in radii
            for angle in angles
        ],
        dtype=float,
    )
    faces = []
    for radial in range(resolved_layers):
        lower = radial * resolved_angles
        upper = (radial + 1) * resolved_angles
        for angular in range(resolved_angles):
            following = (angular + 1) % resolved_angles
            faces.extend(
                (
                    (
                        lower + angular,
                        upper + angular,
                        upper + following,
                    ),
                    (
                        lower + angular,
                        upper + following,
                        lower + following,
                    ),
                )
            )
    return triangle_mesh_to_cochain_complex(
        vertices,
        np.asarray(faces, dtype=np.int32),
    )


def _cochain_semantics(degree: int) -> CochainFieldSpec:
    return CochainFieldSpec(
        degree,
        cell_orientation="invariant" if int(degree) == 0 else "signed",
        sampling="point_value" if int(degree) == 0 else "cell_integral",
    )


def _cochain_query(name: str) -> OperatorQuerySpec:
    return OperatorQuerySpec(
        name,
        geometry_kind="cell_complex",
        coordinate_components=("x", "y"),
        topology_site="cell",
        quadrature="physical_required",
        fixed_geometry=False,
    )


def _cochain_target(
    values: dict[str, jax.Array],
    queries: dict[str, str],
    batch: OperatorBatch,
    /,
) -> OperatorTargetBatch:
    return OperatorTargetBatch.from_arrays(
        values,
        batch,
        query_names=queries,
    )


def cochain_mixed_darcy_scenario(
    *,
    train_points: int = 5,
    test_points: int = 7,
    num_cases: int = 12,
    reaction: float = 0.2,
    boundary_policy: Literal["absolute", "relative"] = "absolute",
    mesh_warp: float = 0.0,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    """Manufactured mixed pressure/flux operator on metric cochain complexes."""
    if min(int(train_points), int(test_points)) < 3:
        raise ValueError("Cochain Darcy meshes require at least three points per axis.")
    if int(num_cases) <= 0:
        raise ValueError("num_cases must be positive.")
    if float(reaction) <= 0.0:
        raise ValueError("reaction must be positive.")
    if boundary_policy not in ("absolute", "relative"):
        raise ValueError("Unknown cochain boundary policy.")

    fields = (
        OperatorFieldSpec(
            "forcing",
            role="source",
            source_name="forcing",
            cochain=_cochain_semantics(0),
        ),
        OperatorFieldSpec(
            "pressure",
            role="target",
            query_name="vertices",
            cochain=_cochain_semantics(0),
        ),
        OperatorFieldSpec(
            "flux",
            role="target",
            query_name="edges",
            cochain=_cochain_semantics(1),
        ),
    )
    task = OperatorTask(
        "benchmark.cochain_mixed_darcy_2d",
        fields=fields,
        queries=(_cochain_query("vertices"), _cochain_query("edges")),
        problem=OperatorProblemSpec(
            source_query_relation="shared_topology",
            query_is_fixed=False,
            requires_resolution_transfer=True,
        ),
        metadata={
            "boundary_policy": boundary_policy,
            "manufactured": True,
        },
    )
    maximum_frequency = min(3, int(train_points) - 2, int(test_points) - 2)
    mode_pairs = tuple(
        (frequency_x, frequency_y)
        for frequency_x in range(1, maximum_frequency + 1)
        for frequency_y in range(1, maximum_frequency + 1)
    )[:8]
    coefficients = np.asarray(
        jax.device_get(jr.normal(jr.key(seed), (int(num_cases), len(mode_pairs)))),
        dtype=float,
    )

    def build(points: int):
        complex_ir = _square_triangle_complex(points, warp=mesh_warp)
        vertices = np.asarray(complex_ir.coordinates[0], dtype=float)
        x = vertices[:, 0]
        y = vertices[:, 1]
        basis = np.stack(
            tuple(
                np.sin(frequency_x * np.pi * x) * np.sin(frequency_y * np.pi * y)
                for frequency_x, frequency_y in mode_pairs
            )
        )
        pressure = coefficients @ basis
        incidence = complex_ir.incidences[0].scipy_matrix().toarray()
        hodge_zero = np.asarray(complex_ir.hodge_stars[0], dtype=float)
        hodge_one = np.asarray(complex_ir.hodge_stars[1], dtype=float)
        laplacian = ((incidence * hodge_one[None, :]) @ incidence.T) / hodge_zero[:, None]
        forcing = pressure @ laplacian.T + float(reaction) * pressure
        flux = -(pressure @ incidence)
        inputs = {
            "forcing": function_samples_from_cochain(
                complex_ir,
                0,
                values=jnp.asarray(forcing),
                boundary_policy=boundary_policy,
            )
        }
        queries = {
            "vertices": function_samples_from_cochain(
                complex_ir,
                0,
                values=None,
                boundary_policy=boundary_policy,
            ),
            "edges": function_samples_from_cochain(
                complex_ir,
                1,
                values=None,
                boundary_policy=boundary_policy,
            ),
        }
        batch = OperatorBatch(
            inputs=inputs,
            queries=queries,
            case_axes=("case",),
            case_shape=(int(num_cases),),
        )
        target = _cochain_target(
            {
                "pressure": jnp.asarray(pressure),
                "flux": jnp.asarray(flux),
            },
            {"pressure": "vertices", "flux": "edges"},
            batch,
        )
        return batch, target

    train_batch, train_target = build(int(train_points))
    transfer_batch, transfer_target = build(int(test_points))
    scenario_name = "cochain_mixed_darcy_2d"
    case_ids = _case_ids(scenario_name, num_cases)
    spacing_min = 1.0 / max(int(train_points), int(test_points))
    spacing_max = 1.0 / (min(int(train_points), int(test_points)) - 1)
    return OperatorBenchmarkScenario(
        name=scenario_name,
        train_batch=train_batch,
        train_target=train_target,
        evaluations=(
            OperatorBenchmarkEvaluation(
                "topology_refinement",
                transfer_batch,
                transfer_target,
                shift="topology_resolution",
                case_ids=case_ids,
            ),
        ),
        seed=int(seed),
        case_ids=case_ids,
        provenance=_generated_provenance("cochain_mixed_darcy_scenario"),
        dimensional_parameters=(
            OperatorParameterRange(
                "mesh_spacing",
                float(spacing_min),
                float(spacing_max),
                "L",
                "log",
            ),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "reaction_strength",
                float(reaction),
                float(reaction),
                "1",
                "log",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="exact discrete manufactured Hodge-Laplacian system",
            verification="discrete_residual",
            resolutions=(int(train_points), int(test_points)),
            relative_error=0.0,
            tolerance=1e-12,
        ),
        regimes=(
            "cell_complex",
            "mixed_degree",
            "metric_dec",
            "resolution_transfer",
        ),
        metadata=(
            ("boundary_policy", boundary_policy),
            ("mesh_warp", str(float(mesh_warp))),
            ("sensor_shift_policy", "disabled"),
            ("population_seed", str(int(seed))),
        ),
        task=task,
    )


def _annulus_harmonic_template(complex_ir, /) -> np.ndarray:
    subspace = complex_ir.harmonic_subspace
    if subspace is None or subspace.ranks[1] < 1:
        raise ValueError("Annulus benchmark requires a nontrivial degree-one nullspace.")
    incidence = complex_ir.incidences[0].scipy_matrix().toarray()
    coordinates = np.asarray(complex_ir.coordinates[0], dtype=float)
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    angular = np.empty((incidence.shape[1],), dtype=float)
    for edge in range(incidence.shape[1]):
        column = incidence[:, edge]
        tail = int(np.flatnonzero(column < 0.0)[0])
        head = int(np.flatnonzero(column > 0.0)[0])
        difference = angles[head] - angles[tail]
        angular[edge] = (difference + np.pi) % (2.0 * np.pi) - np.pi
    rank = int(subspace.ranks[1])
    basis = np.asarray(subspace.bases[1], dtype=float)[:, :rank]
    metric = np.asarray(complex_ir.hodge_stars[1], dtype=float)
    projected = basis @ (basis.T @ (metric * angular))
    norm = np.sqrt(np.sum(metric * projected * projected))
    if norm <= 1e-12:
        raise ValueError("Annulus angular cochain has no harmonic projection.")
    return projected / norm


def cochain_annulus_harmonic_scenario(
    *,
    train_radial_layers: int = 2,
    train_angular_points: int = 10,
    test_radial_layers: int = 3,
    test_angular_points: int = 14,
    num_cases: int = 12,
    boundary_policy: Literal["absolute", "relative"] = "absolute",
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    """Project mixed annular one-forms onto a metric harmonic subspace."""
    if int(num_cases) <= 0:
        raise ValueError("num_cases must be positive.")
    if boundary_policy not in ("absolute", "relative"):
        raise ValueError("Unknown cochain boundary policy.")
    fields = (
        OperatorFieldSpec(
            "one_form",
            role="source",
            source_name="one_form",
            cochain=_cochain_semantics(1),
        ),
        OperatorFieldSpec(
            "harmonic",
            role="target",
            query_name="edges",
            cochain=_cochain_semantics(1),
        ),
    )
    task = OperatorTask(
        "benchmark.cochain_annulus_harmonic_1form",
        fields=fields,
        queries=(_cochain_query("edges"),),
        problem=OperatorProblemSpec(
            source_query_relation="shared_topology",
            query_is_fixed=False,
            requires_resolution_transfer=True,
        ),
        metadata={
            "boundary_policy": boundary_policy,
            "betti_one": 1,
        },
    )
    rng = np.random.default_rng(int(seed))
    potential_coefficients = rng.normal(size=(int(num_cases), 4))
    face_coefficients = rng.normal(size=(int(num_cases), 4))
    harmonic_amplitudes = rng.uniform(0.5, 1.5, size=(int(num_cases), 1))

    def build(radial_layers: int, angular_points: int):
        bare = _annulus_triangle_complex(radial_layers, angular_points)
        harmonic = compute_harmonic_subspace(
            bare,
            boundary_policy=boundary_policy,
            max_modes=4,
        )
        complex_ir = bare.with_harmonic_subspace(harmonic)
        template = _annulus_harmonic_template(complex_ir)
        vertex_coordinates = np.asarray(complex_ir.coordinates[0], dtype=float)
        face_coordinates = np.asarray(complex_ir.coordinates[2], dtype=float)
        x_vertex = vertex_coordinates[:, 0]
        y_vertex = vertex_coordinates[:, 1]
        x_face = face_coordinates[:, 0]
        y_face = face_coordinates[:, 1]
        vertex_basis = np.stack(
            (
                x_vertex,
                y_vertex,
                x_vertex * y_vertex,
                x_vertex * x_vertex - y_vertex * y_vertex,
            )
        )
        face_basis = np.stack(
            (
                np.ones_like(x_face),
                x_face,
                y_face,
                x_face * y_face,
            )
        )
        potential = potential_coefficients @ vertex_basis
        top_form = face_coefficients @ face_basis
        incidence_zero = complex_ir.incidences[0].scipy_matrix().toarray()
        incidence_one = complex_ir.incidences[1].scipy_matrix().toarray()
        hodge_one = np.asarray(complex_ir.hodge_stars[1], dtype=float)
        hodge_two = np.asarray(complex_ir.hodge_stars[2], dtype=float)
        exact = potential @ incidence_zero
        coexact = ((top_form * hodge_two[None, :]) @ incidence_one.T) / (
            hodge_one[None, :]
        )

        def normalized(values):
            norms = np.sqrt(
                np.sum(hodge_one[None, :] * values * values, axis=1, keepdims=True)
            )
            return values / np.maximum(norms, 1e-12)

        target = harmonic_amplitudes * template[None, :]
        one_form = 0.4 * normalized(exact) + 0.4 * normalized(coexact) + target
        batch = OperatorBatch(
            inputs={
                "one_form": function_samples_from_cochain(
                    complex_ir,
                    1,
                    values=jnp.asarray(one_form),
                    boundary_policy=boundary_policy,
                )
            },
            queries={
                "edges": function_samples_from_cochain(
                    complex_ir,
                    1,
                    values=None,
                    boundary_policy=boundary_policy,
                )
            },
            case_axes=("case",),
            case_shape=(int(num_cases),),
        )
        targets = _cochain_target(
            {"harmonic": jnp.asarray(target)},
            {"harmonic": "edges"},
            batch,
        )
        return batch, targets, int(harmonic.ranks[1])

    train_batch, train_target, train_rank = build(
        int(train_radial_layers),
        int(train_angular_points),
    )
    transfer_batch, transfer_target, transfer_rank = build(
        int(test_radial_layers),
        int(test_angular_points),
    )
    scenario_name = "cochain_annulus_harmonic_projection"
    case_ids = _case_ids(scenario_name, num_cases)
    return OperatorBenchmarkScenario(
        name=scenario_name,
        train_batch=train_batch,
        train_target=train_target,
        evaluations=(
            OperatorBenchmarkEvaluation(
                "topology_refinement",
                transfer_batch,
                transfer_target,
                shift="topology_resolution",
                case_ids=case_ids,
            ),
        ),
        seed=int(seed),
        case_ids=case_ids,
        provenance=_generated_provenance("cochain_annulus_harmonic_scenario"),
        dimensional_parameters=(
            OperatorParameterRange("inner_radius", 0.45, 0.45, "L"),
            OperatorParameterRange("outer_radius", 1.0, 1.0, "L"),
        ),
        nondimensional_parameters=(
            OperatorParameterRange(
                "radial_aspect_ratio",
                0.45,
                0.45,
                "1",
            ),
        ),
        reference_evidence=ReferenceSolverEvidence(
            method="metric-orthogonal DEC harmonic projection",
            verification="discrete_residual",
            resolutions=(int(train_angular_points), int(test_angular_points)),
            relative_error=0.0,
            tolerance=1e-10,
        ),
        regimes=(
            "cell_complex",
            "harmonic_one_form",
            "nontrivial_topology",
            "resolution_transfer",
        ),
        metadata=(
            ("boundary_policy", boundary_policy),
            ("train_harmonic_rank", str(train_rank)),
            ("transfer_harmonic_rank", str(transfer_rank)),
            ("target_intrinsic_rank", str(train_rank)),
            ("sensor_shift_policy", "disabled"),
            ("population_seed", str(int(seed))),
        ),
        task=task,
    )


def split_operator_scenario(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    seed: int = 0,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> OperatorBenchmarkScenario:
    """Split complete physical realizations into deterministic disjoint populations."""
    if len(scenario.train_batch.case_shape) != 1:
        raise ValueError("Scenario splitting requires exactly one case axis.")
    case_count = scenario.train_batch.case_shape[0]
    if case_count < 3:
        raise ValueError("At least three cases are required for train/validation/test.")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("Split fractions must lie strictly between zero and one.")
    case_ids = scenario.case_ids or _case_ids(scenario.name, case_count)
    if len(case_ids) != case_count or len(set(case_ids)) != case_count:
        raise ValueError("Scenario case_ids must uniquely identify every realization.")
    train_count = max(1, int(case_count * train_fraction))
    validation_count = max(1, int(case_count * validation_fraction))
    if train_count + validation_count >= case_count:
        validation_count = 1
        train_count = case_count - 2
    permutation = jr.permutation(jr.key(seed), case_count)
    train_indices = permutation[:train_count]
    validation_indices = permutation[train_count : train_count + validation_count]
    test_indices = permutation[train_count + validation_count :]

    def take_target(target, indices):
        if isinstance(target, OperatorTargetBatch):
            return target.take(indices)
        return jnp.take(target, indices, axis=0)

    def select_ids(values, indices):
        selected = np.asarray(jax.device_get(indices), dtype=np.int64).tolist()
        return tuple(values[int(index)] for index in selected)

    train_ids = select_ids(case_ids, train_indices)
    validation_ids = select_ids(case_ids, validation_indices)
    validation = OperatorBenchmarkEvaluation(
        "validation",
        scenario.train_batch.take(validation_indices),
        take_target(scenario.train_target, validation_indices),
        split="validation",
        case_ids=validation_ids,
    )
    evaluations = []
    for evaluation in scenario.evaluations:
        if evaluation.batch.case_shape and evaluation.batch.case_shape[0] == case_count:
            evaluation_ids = evaluation.case_ids or case_ids
            if len(evaluation_ids) != case_count:
                raise ValueError(
                    f"Evaluation {evaluation.name!r} case_ids do not match its cases."
                )
            evaluations.append(
                replace(
                    evaluation,
                    batch=evaluation.batch.take(test_indices),
                    target=take_target(evaluation.target, test_indices),
                    split="test",
                    case_ids=select_ids(evaluation_ids, test_indices),
                )
            )
        else:
            evaluations.append(evaluation)
    return replace(
        scenario,
        train_batch=scenario.train_batch.take(train_indices),
        train_target=take_target(scenario.train_target, train_indices),
        evaluations=tuple(evaluations),
        validation=validation,
        seed=int(seed),
        case_ids=train_ids,
        metadata=scenario.metadata
        + (
            ("train_cases", str(train_count)),
            ("validation_cases", str(validation_count)),
            ("test_cases", str(len(test_indices))),
        ),
    )


def _replace_sample_values(
    samples: FunctionSamples,
    values,
    /,
    *,
    mask=None,
) -> FunctionSamples:
    return FunctionSamples(
        values=values,
        axes=samples.axes,
        coordinates=samples.coordinates,
        quadrature_weights=samples.quadrature_weights,
        mask=samples.mask if mask is None else mask,
        topology=samples.topology,
    )


def add_input_noise_shift(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    standard_deviation: float = 0.01,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    """Append an evaluation with deterministic additive source noise."""
    if standard_deviation < 0.0:
        raise ValueError("standard_deviation must be non-negative.")
    reference = scenario.evaluations[0]
    keys = iter(jr.split(jr.key(seed), len(reference.batch.inputs)))
    inputs = {}
    for name, samples in reference.batch.inputs.items():
        if samples.values is None:
            inputs[name] = samples
            continue
        values = jnp.asarray(samples.values)
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            inputs[name] = samples
            continue
        scale = jnp.std(values)
        noise = (
            standard_deviation
            * jnp.maximum(scale, 1e-12)
            * jr.normal(
                next(keys),
                values.shape,
                dtype=values.dtype,
            )
        )
        inputs[name] = _replace_sample_values(samples, values + noise)
    noisy_batch = OperatorBatch(
        inputs=inputs,
        queries=reference.batch.queries,
        case_axes=reference.batch.case_axes,
        case_shape=reference.batch.case_shape,
    )
    shifted = replace(
        reference,
        name="input_noise",
        batch=noisy_batch,
        shift="input_noise",
    )
    return replace(scenario, evaluations=scenario.evaluations + (shifted,))


def _sensor_dropout_batch(
    batch: OperatorBatch,
    /,
    *,
    drop_fraction: float,
    seed: int,
    mask_aware: bool,
) -> OperatorBatch:
    if not 0.0 <= drop_fraction < 1.0:
        raise ValueError("drop_fraction must lie in [0, 1).")
    keys = iter(jr.split(jr.key(seed), len(batch.inputs)))
    inputs = {}
    for name, samples in batch.inputs.items():
        if not samples.sample_shape:
            inputs[name] = samples
            continue
        shape = batch.case_shape + samples.sample_shape
        random_retained = jr.uniform(next(keys), shape) >= drop_fraction
        retained = random_retained & samples.mask_array(case_shape=batch.case_shape)
        values = samples.values
        if values is not None:
            array = jnp.asarray(values)
            trailing = array.ndim - len(shape)
            value_mask = retained if mask_aware else random_retained
            array_mask = value_mask.reshape(shape + (1,) * trailing)
            values = jnp.where(array_mask, array, jnp.zeros((), dtype=array.dtype))
        inputs[name] = _replace_sample_values(
            samples,
            values,
            mask=retained if mask_aware else samples.mask,
        )
    return OperatorBatch(
        inputs=inputs,
        queries=batch.queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


def add_sensor_corruption_shift(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    corruption_fraction: float = 0.2,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    """Append raw zero-value corruption without declaring sensors missing."""
    reference = scenario.evaluations[0]
    corruption_batch = _sensor_dropout_batch(
        reference.batch,
        drop_fraction=corruption_fraction,
        seed=seed,
        mask_aware=False,
    )
    shifted = replace(
        reference,
        name="sensor_corruption",
        batch=corruption_batch,
        shift="sensor_corruption",
    )
    return replace(scenario, evaluations=scenario.evaluations + (shifted,))


def add_sensor_dropout_shift(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    drop_fraction: float = 0.2,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    """Append an explicit mask-aware sensor-removal evaluation."""
    reference = scenario.evaluations[0]
    dropout_batch = _sensor_dropout_batch(
        reference.batch,
        drop_fraction=drop_fraction,
        seed=seed,
        mask_aware=True,
    )
    shifted = replace(
        reference,
        name="sensor_dropout",
        batch=dropout_batch,
        shift="sensor_dropout",
    )
    return replace(scenario, evaluations=scenario.evaluations + (shifted,))


def add_sensor_dropout_ladder(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    drop_fractions: tuple[float, ...] = (0.1, 0.3, 0.5),
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    """Append deterministic mask-density evaluations from one reference split."""
    fractions = tuple(float(value) for value in drop_fractions)
    if (
        not fractions
        or len(set(fractions)) != len(fractions)
        or any(not 0.0 <= value < 1.0 for value in fractions)
    ):
        raise ValueError("drop_fractions must contain unique values in [0, 1).")
    reference = scenario.evaluations[0]
    evaluations = []
    for fraction in fractions:
        dropout_batch = _sensor_dropout_batch(
            reference.batch,
            drop_fraction=fraction,
            seed=int(seed),
            mask_aware=True,
        )
        percentage = f"{100.0 * fraction:g}".replace(".", "p")
        evaluations.append(
            replace(
                reference,
                name=f"sensor_dropout_{percentage}pct",
                batch=dropout_batch,
                shift="sensor_dropout",
            )
        )
    return replace(
        scenario,
        evaluations=scenario.evaluations + tuple(evaluations),
        metadata=scenario.metadata
        + (
            (
                "sensor_dropout_ladder",
                ",".join(str(value) for value in fractions),
            ),
        ),
    )


def add_training_sensor_dropout(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    drop_fraction: float = 0.2,
    seed: int = 0,
) -> OperatorBenchmarkScenario:
    """Apply deterministic mask-aware sensor dropout to the training batch only."""
    train_batch = _sensor_dropout_batch(
        scenario.train_batch,
        drop_fraction=drop_fraction,
        seed=seed,
        mask_aware=True,
    )
    return replace(
        scenario,
        train_batch=train_batch,
        metadata=scenario.metadata
        + (
            ("training_augmentation", "sensor_dropout"),
            ("training_sensor_dropout_fraction", str(float(drop_fraction))),
            ("training_sensor_dropout_seed", str(int(seed))),
        ),
    )


def standard_operator_benchmarks(
    *,
    quick: bool = False,
    include_shifts: bool = True,
) -> tuple[OperatorBenchmarkScenario, ...]:
    scale = 1 if quick else 2
    scenarios = (
        periodic_burgers_scenario(
            train_resolution=16 * scale,
            test_resolution=24 * scale,
            num_cases=4 * scale,
            rollout_steps=3,
        ),
        darcy_scenario(resolution=8 + 2 * scale, num_cases=2 * scale),
        navier_stokes_scenario(resolution=8 * scale, num_cases=2 * scale),
        green_function_scenario(
            source_points=10 * scale,
            query_points=15 * scale,
            num_cases=3 * scale,
        ),
        multi_input_diffusion_scenario(
            resolution=12 * scale,
            num_cases=3 * scale,
        ),
        beam_transient_scenario(
            spatial_points=10 * scale,
            time_points=8 * scale,
            num_cases=3 * scale,
        ),
        irregular_poisson_scenario(
            points=16 * scale,
            num_cases=3 * scale,
            geometry_shift=True,
        ),
        graph_diffusion_scenario(
            nodes=12 * scale,
            test_nodes=18 * scale,
            num_cases=3 * scale,
            geometry_shift=True,
        ),
    )
    if not include_shifts:
        return scenarios
    shifted = []
    for index, scenario in enumerate(scenarios):
        scenario = add_input_noise_shift(
            scenario,
            standard_deviation=0.02,
            seed=100 + index,
        )
        scenario = add_sensor_dropout_ladder(
            scenario,
            drop_fractions=(0.3,) if quick else (0.1, 0.3, 0.5),
            seed=200 + 3 * index,
        )
        shifted.append(scenario)
    return tuple(shifted)


__all__ = [
    "OperatorBenchmarkEvaluation",
    "OperatorBenchmarkScenario",
    "add_input_noise_shift",
    "augment_square_group_training",
    "add_sensor_corruption_shift",
    "add_sensor_dropout_ladder",
    "add_sensor_dropout_shift",
    "add_training_sensor_dropout",
    "beam_transient_scenario",
    "causal_relaxation_scenario",
    "conservative_ring_transport_scenario",
    "cochain_annulus_harmonic_scenario",
    "cochain_mixed_darcy_scenario",
    "darcy_scenario",
    "deformed_elliptic_scenario",
    "graph_diffusion_scenario",
    "green_function_scenario",
    "irregular_poisson_scenario",
    "irregular_causal_relaxation_scenario",
    "multi_input_diffusion_scenario",
    "navier_stokes_scenario",
    "periodic_burgers_scenario",
    "polynomial_poisson_scenario",
    "split_operator_scenario",
    "spherical_diffusion_scenario",
    "standard_operator_benchmarks",
]
