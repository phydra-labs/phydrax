#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ..stochastic._bsde import BSDEEvaluation
from ..stochastic._jump_bsde import JumpBSDEEvaluation
from ._bellman import BellmanFilterResult, BellmanSmootherResult
from ._checkpoint import (
    _json_value,
    _read_array_archive,
    _write_array_archive,
    CheckpointCorruptionError,
)
from ._diagnostics import MCMCConvergenceReport
from ._discrepancy_diagnostics import DiscrepancyIdentifiabilityReport
from ._eki import EnsembleKalmanResult
from ._ensemble_filter import EnsembleFilterResult, EnsembleSmootherResult
from ._flow_mcmc import FlowNUTSResult
from ._flow_variational import FlowVariationalResult
from ._kalman import KalmanFilterResult, KalmanSmootherResult
from ._laplace import LaplaceResult
from ._laplax_backend import StructuredLaplaceResult
from ._map import MAPResult
from ._map_candidate_search import MAPCandidateSearchResult
from ._map_search import GaussianProcessMAPSearchResult, MAPSearchResult
from ._mcmc import MCMCResult
from ._nested import nested_sampling_status_name, NestedSamplingResult
from ._particle import (
    ParticleBackwardSimulationResult,
    ParticleBackwardSmootherResult,
    ParticleFilterResult,
    ParticleFisherInformationResult,
    ParticleFisherScoreResult,
    ParticleSmootherResult,
)
from ._particle_genealogical_score import ParticleGenealogicalScoreResult
from ._pathfinder import PathfinderResult
from ._rao_blackwellized import RaoBlackwellizedFilterResult
from ._rao_blackwellized_smoothing import (
    RaoBlackwellizedBackwardSimulationResult,
    RaoBlackwellizedSmootherResult,
)
from ._sgmcmc import SGMCMCResult
from ._sgmcmc_diagnostics import SGMCMCMixingReport
from ._sing import SINGResult
from ._smc import TemperedSMCResult
from ._state_space_amortized import AmortizedStateSpaceVariationalResult
from ._state_space_buffered import BufferedStateSpaceVariationalResult
from ._state_space_variational import StateSpaceVariationalResult
from ._state_space_gp import StateSpaceGaussianProcessResult
from ._variational import VariationalResult


_RESULT_FORMAT = "phydrax-uq-result"


@dataclass(frozen=True)
class UQResultArchive:
    """Immutable metadata and checked NumPy arrays from a portable UQ result."""

    kind: str
    metadata: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    fields: Mapping[str, str]
    trees: Mapping[str, Mapping[str, Any]]
    excluded: tuple[str, ...]

    def array(self, name: str, /) -> np.ndarray:
        """Return one named scalar or dense-array field."""
        try:
            return self.arrays[self.fields[name]]
        except KeyError as error:
            raise KeyError(f"Result archive has no array field {name!r}.") from error

    def tree(self, name: str, /) -> Mapping[str, np.ndarray]:
        """Return one array tree as stable JAX key paths mapped to leaves."""
        try:
            specification = self.trees[name]
        except KeyError as error:
            raise KeyError(f"Result archive has no array tree {name!r}.") from error
        paths = specification.get("paths")
        names = specification.get("arrays")
        if not isinstance(paths, list) or not isinstance(names, list):
            raise CheckpointCorruptionError(
                f"Result tree specification {name!r} is invalid."
            )
        if len(paths) != len(names):
            raise CheckpointCorruptionError(
                f"Result tree specification {name!r} has inconsistent leaves."
            )
        return MappingProxyType(
            {
                str(path): self.arrays[str(array_name)]
                for path, array_name in zip(paths, names, strict=True)
            }
        )


def export_result(result: Any, path: str | Path, /) -> Path:
    """Write a supported UQ result as a pickle-free archive."""
    arrays: dict[str, np.ndarray] = {}
    fields: dict[str, str] = {}
    trees: dict[str, dict[str, Any]] = {}
    kind, metadata, excluded = _adapt_result(result, arrays, fields, trees)
    manifest = {
        "format": _RESULT_FORMAT,
        "result_kind": kind,
        "metadata": _json_value(metadata, path="metadata"),
        "fields": fields,
        "trees": trees,
        "excluded": list(excluded),
    }
    return _write_array_archive(path, manifest=manifest, arrays=arrays)


def read_result_archive(path: str | Path, /) -> UQResultArchive:
    """Read and checksum-validate a portable UQ result archive."""
    manifest, loaded = _read_array_archive(path)
    expected = {
        "format",
        "result_kind",
        "metadata",
        "fields",
        "trees",
        "excluded",
        "arrays",
    }
    missing = expected - set(manifest)
    unknown = set(manifest) - expected
    if missing or unknown:
        raise CheckpointCorruptionError(
            "Result manifest must use the current canonical fields; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}."
        )
    if manifest.get("format") != _RESULT_FORMAT:
        raise CheckpointCorruptionError("Archive is not a Phydrax UQ result.")
    kind = manifest.get("result_kind")
    metadata = manifest.get("metadata")
    fields = manifest.get("fields")
    trees = manifest.get("trees")
    excluded = manifest.get("excluded")
    if (
        not isinstance(kind, str)
        or not isinstance(metadata, dict)
        or not isinstance(fields, dict)
        or not isinstance(trees, dict)
        or not isinstance(excluded, list)
        or any(not isinstance(item, str) for item in excluded)
    ):
        raise CheckpointCorruptionError("Result archive manifest is invalid.")
    arrays: dict[str, np.ndarray] = {}
    for name, value in loaded.items():
        value.setflags(write=False)
        arrays[name] = value
    return UQResultArchive(
        kind=kind,
        metadata=MappingProxyType(metadata),
        arrays=MappingProxyType(arrays),
        fields=MappingProxyType({str(key): str(value) for key, value in fields.items()}),
        trees=MappingProxyType(trees),
        excluded=tuple(excluded),
    )


def to_arviz(result: MCMCResult | FlowNUTSResult | SGMCMCResult, /):
    """Convert chain-preserving posterior draws and honest sampler statistics."""
    if isinstance(result, FlowNUTSResult):
        chain_result = result.mcmc
    elif isinstance(result, (MCMCResult, SGMCMCResult)):
        chain_result = result
    else:
        raise TypeError(
            "to_arviz supports MCMCResult, FlowNUTSResult, and SGMCMCResult only."
        )
    try:
        import arviz as az
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise ImportError("ArviZ is required for to_arviz().") from error

    posterior: dict[str, np.ndarray] = {}
    dimensions: dict[str, list[str]] = {}
    coordinates: dict[str, np.ndarray] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(chain_result.samples)[0]:
        path_string = jax.tree_util.keystr(path) or "<root>"
        name = encode_parameter_name(path_string)
        value = np.asarray(leaf)
        posterior[name] = value
        parameter_dims = []
        for axis, size in enumerate(value.shape[2:]):
            dimension = f"{name}_dim_{axis}"
            parameter_dims.append(dimension)
            coordinates[dimension] = np.arange(size)
        dimensions[name] = parameter_dims

    if isinstance(chain_result, SGMCMCResult):
        sample_stats = {
            "stochastic_gradient_norm": np.asarray(chain_result.gradient_norm),
        }
        if chain_result.log_density is not None:
            sample_stats["lp"] = np.asarray(chain_result.log_density)
        if chain_result.thermostat is not None:
            sample_stats["thermostat"] = np.asarray(chain_result.thermostat)
        if chain_result.momentum_norm is not None:
            sample_stats["momentum_norm"] = np.asarray(chain_result.momentum_norm)
        posterior_attributes = {
            "phydrax_algorithm": chain_result.algorithm,
            "phydrax_approximation": chain_result.approximation,
            "phydrax_chain_method": chain_result.chain_method,
            "phydrax_step_size": chain_result.step_size,
            "phydrax_batch_fraction": chain_result.batch_fraction,
            "phydrax_source_fingerprint": chain_result.source_fingerprint,
            "phydrax_control_variate": chain_result.control_variate is not None,
        }
    else:
        sample_stats = {
            "lp": np.asarray(chain_result.log_density),
            "acceptance_rate": np.asarray(chain_result.acceptance_rate),
            "diverging": np.asarray(chain_result.divergent),
            "energy": np.asarray(chain_result.energy),
            "n_steps": np.asarray(chain_result.num_integration_steps),
            "tree_depth": np.asarray(chain_result.num_trajectory_expansions),
        }
        posterior_attributes = {
            "phydrax_algorithm": chain_result.algorithm,
            "phydrax_chain_method": chain_result.chain_method,
        }
        if isinstance(result, FlowNUTSResult):
            sample_stats.update(
                {
                    "flow_acceptance_rate": np.asarray(result.global_acceptance_rate),
                    "flow_accepted_count": np.asarray(result.global_accepted_count),
                    "flow_mean_log_acceptance_ratio": np.asarray(
                        result.global_mean_log_acceptance_ratio
                    ),
                    "flow_nonfinite_count": np.asarray(result.global_nonfinite_count),
                }
            )
            posterior_attributes["phydrax_flow_nuts_config"] = json.dumps(
                result.config.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
    return az.from_dict(
        {
            "posterior": posterior,
            "sample_stats": sample_stats,
        },
        sample_dims=("chain", "draw"),
        dims=dimensions,
        coords=coordinates,
        attrs={"posterior": posterior_attributes},
    )


def encode_parameter_name(path: str, /) -> str:
    """Encode a JAX key path as a stable xarray variable name."""
    payload = base64.urlsafe_b64encode(path.encode("utf-8")).decode("ascii")
    return "parameter__" + payload.rstrip("=")


def decode_parameter_name(name: str, /) -> str:
    """Recover the JAX key path from an encoded parameter variable name."""
    prefix = "parameter__"
    if not name.startswith(prefix):
        raise ValueError("Encoded parameter name has an invalid prefix.")
    payload = name[len(prefix) :]
    padding = "=" * (-len(payload) % 4)
    try:
        return base64.urlsafe_b64decode(payload + padding).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("Encoded parameter name is invalid.") from error


def _adapt_result(result, arrays, fields, trees):
    if isinstance(result, BellmanFilterResult):
        metadata = _put_bellman_filter_result(result, arrays, fields, prefix="")
        return "bellman_filter", metadata, ("problem",)

    if isinstance(result, BellmanSmootherResult):
        for name, value in (
            ("modes", result.modes),
            ("covariances", result.covariances),
            ("gains", result.gains),
            ("lag_one_covariances", result.lag_one_covariances),
            ("valid", result.valid),
        ):
            _put_field(fields, arrays, name, value)
        metadata = _put_bellman_filter_result(
            result.filter_result, arrays, fields, prefix="filter."
        )
        metadata["smoother_method_id"] = result.method_id
        return "bellman_smoother", metadata, ("filter_result.problem",)

    if isinstance(result, SINGResult):
        for name, value in (
            ("means", result.means),
            ("covariances", result.covariances),
            (
                "transition_cross_covariances",
                result.transition_cross_covariances,
            ),
            ("observation_means", result.observation_means),
            ("observation_covariances", result.observation_covariances),
            (
                "information.diagonal_precision",
                result.state.information.diagonal_precision,
            ),
            (
                "information.transition_precision",
                result.state.information.transition_precision,
            ),
            (
                "information.information_vector",
                result.state.information.information_vector,
            ),
            ("grid.times", result.state.grid.times),
            ("grid.node_valid", result.state.grid.node_valid),
            (
                "grid.observation_node_indices",
                result.state.grid.observation_node_indices,
            ),
            (
                "grid.transition_step_indices",
                result.state.grid.transition_step_indices,
            ),
            ("elbo.per_case_elbo", result.elbo.per_case_elbo),
            ("elbo.total_elbo", result.elbo.total_elbo),
            (
                "elbo.expected_initial_log_density",
                result.elbo.expected_initial_log_density,
            ),
            (
                "elbo.expected_transition_log_density",
                result.elbo.expected_transition_log_density,
            ),
            (
                "elbo.expected_observation_log_density",
                result.elbo.expected_observation_log_density,
            ),
            ("elbo.entropy", result.elbo.entropy),
            ("elbo.valid", result.elbo.valid),
            ("elbo.status", result.elbo.status),
            ("elbo_history", result.elbo_history),
            ("step_size_history", result.step_size_history),
            ("natural_residual_history", result.natural_residual_history),
            ("accepted_history", result.accepted_history),
            ("converged", result.converged),
            ("valid", result.valid),
            ("status", result.status),
            ("state.iteration", result.state.iteration),
        ):
            _put_field(fields, arrays, name, value)
        metadata = {
            "approximation_id": result.approximation_id,
            "model_id": result.model_id,
            "problem_id": result.problem_id,
            "process_id": result.process_id,
            "sequence_id": result.sequence_id,
            "grid_id": result.state.grid.grid_id,
            "information_id": result.state.information.information_id,
            "case_axes": result.case_axes,
            "case_shape": result.case_shape,
            "case_ids": result.case_ids,
            "state_shape": result.state_shape,
            "expectation_method": result.expectation_method,
            "execution_method": result.execution_method,
            "max_iterations": result.max_iterations,
            "num_samples": result.state.num_samples,
            "order": result.state.order,
            "max_dimension": result.state.max_dimension,
            "max_points": result.state.max_points,
            "alpha": result.state.alpha,
            "beta": result.state.beta,
            "kappa": result.state.kappa,
            "rank_tolerance": result.state.information.rank_tolerance,
        }
        return "sing_smoother", metadata, ("state.expectation_key",)

    if isinstance(result, StateSpaceGaussianProcessResult):
        for name, value in (
            ("posterior_times", result.posterior_times),
            ("posterior_mean", result.posterior_mean),
            ("posterior_variance", result.posterior_variance),
            ("predictive_mean", result.predictive_mean),
            ("predictive_variance", result.predictive_variance),
            ("log_marginal_likelihood", result.log_marginal_likelihood),
            ("active_observation_count", result.active_observation_count),
            ("valid", result.valid),
            ("status", result.status),
            ("query_valid", result.query_valid),
            ("train_mask", result.train_mask),
            ("schedule_times", result.schedule_times),
            ("schedule_observation_mask", result.schedule_observation_mask),
            (
                "incremental_log_likelihood",
                result.filter_result.incremental_log_likelihood,
            ),
            ("filter_status", result.filter_result.status),
        ):
            _put_field(fields, arrays, name, value)
        metadata = {
            "state_dimension": result.state_dimension,
            "kernel_id": result.kernel_id,
            "kernel_content_id": result.kernel_content_id,
            "schedule_id": result.schedule_id,
            "method_id": result.method_id,
            "repeated_time_policy": result.repeated_time_policy,
            "precision_evidence": result.precision_evidence.to_dict(),
        }
        return (
            "state_space_gaussian_process",
            metadata,
            ("filter_result", "smoother_result"),
        )

    if isinstance(result, KalmanFilterResult):
        metadata = _put_kalman_filter_result(result, arrays, fields, prefix="")
        return "kalman_filter", metadata, ()

    if isinstance(result, KalmanSmootherResult):
        for name, value in (
            ("means", result.means),
            ("covariances", result.covariances),
            ("gains", result.gains),
            ("valid", result.valid),
        ):
            _put_field(fields, arrays, name, value)
        metadata = _put_kalman_filter_result(
            result.filter_result, arrays, fields, prefix="filter."
        )
        metadata["smoother_execution_method"] = result.execution_method
        metadata["smoother_covariance_form"] = result.covariance_form
        return "kalman_smoother", metadata, ()

    if isinstance(result, ParticleSmootherResult):
        for name, value in (
            ("particles", result.particles),
            ("log_weights", result.log_weights),
            ("means", result.means),
            ("lineage_indices", result.lineage_indices),
            ("horizons", result.horizons),
            ("step_valid", result.step_valid),
            ("valid", result.valid),
            ("status", result.status),
            ("times", result.times),
        ):
            _put_field(fields, arrays, name, value)
        metadata = _put_particle_filter_result(
            result.filter_result, arrays, fields, prefix="filter."
        )
        metadata.update(
            {
                "smoother_method_id": result.method_id,
                "ancestry_gradient": result.ancestry_gradient,
            }
        )
        return "particle_smoother", metadata, ("filter_result.problem",)

    if isinstance(result, ParticleBackwardSmootherResult):
        for name, value in (
            ("particles", result.particles),
            ("log_weights", result.log_weights),
            ("means", result.means),
            ("backward_log_probabilities", result.backward_log_probabilities),
            ("pair_log_weights", result.pair_log_weights),
            ("step_valid", result.step_valid),
            ("valid", result.valid),
            ("status", result.status),
            ("times", result.times),
        ):
            _put_field(fields, arrays, name, value)
        metadata = _put_particle_filter_result(
            result.filter_result, arrays, fields, prefix="filter."
        )
        metadata.update(
            {
                "smoother_method_id": result.method_id,
                "process_id": result.process_id,
                "approximation_id": result.approximation_id,
            }
        )
        return "particle_backward_smoother", metadata, ("filter_result.problem",)

    if isinstance(result, ParticleBackwardSimulationResult):
        _put_field(fields, arrays, "paths", result.paths)
        _put_field(fields, arrays, "particle_indices", result.particle_indices)
        _put_field(fields, arrays, "step_valid", result.step_valid)
        _put_field(fields, arrays, "valid", result.valid)
        smoother = result.smoother
        for name, value in (
            ("smoother.log_weights", smoother.log_weights),
            (
                "smoother.backward_log_probabilities",
                smoother.backward_log_probabilities,
            ),
            ("smoother.pair_log_weights", smoother.pair_log_weights),
        ):
            _put_field(fields, arrays, name, value)
        metadata = _put_particle_filter_result(
            smoother.filter_result, arrays, fields, prefix="filter."
        )
        metadata.update(
            {
                "simulation_method_id": result.method_id,
                "smoother_method_id": smoother.method_id,
                "sample_shape": list(result.sample_shape),
                "ancestry_gradient": result.ancestry_gradient,
                "process_id": smoother.process_id,
                "approximation_id": smoother.approximation_id,
            }
        )
        return (
            "particle_backward_simulation",
            metadata,
            ("smoother.filter_result.problem",),
        )

    if isinstance(result, RaoBlackwellizedFilterResult):
        metadata = _put_rao_blackwellized_filter_result(result, arrays, fields, prefix="")
        return "rao_blackwellized_filter", metadata, ("problem",)

    if isinstance(result, RaoBlackwellizedBackwardSimulationResult):
        for name, value in (
            ("initial_nonlinear_states", result.initial_nonlinear_states),
            ("nonlinear_paths", result.nonlinear_paths),
            ("particle_indices", result.particle_indices),
            ("step_valid", result.step_valid),
            ("valid", result.valid),
        ):
            _put_field(fields, arrays, name, value)
        metadata = _put_rao_blackwellized_filter_result(
            result.filter_result, arrays, fields, prefix="filter."
        )
        metadata.update(
            {
                "simulation_method_id": result.method_id,
                "sample_shape": list(result.sample_shape),
                "ancestry_gradient": result.ancestry_gradient,
                "process_id": result.process_id,
                "approximation_id": result.approximation_id,
            }
        )
        return (
            "rao_blackwellized_backward_simulation",
            metadata,
            ("filter_result.problem",),
        )

    if isinstance(result, RaoBlackwellizedSmootherResult):
        for name, value in (
            ("linear_means", result.linear_means),
            ("linear_covariances", result.linear_covariances),
            ("gains", result.gains),
            ("lag_one_covariances", result.lag_one_covariances),
            ("valid", result.valid),
            ("status", result.status),
            (
                "backward.initial_nonlinear_states",
                result.backward_simulation.initial_nonlinear_states,
            ),
            (
                "backward.nonlinear_paths",
                result.backward_simulation.nonlinear_paths,
            ),
            (
                "backward.particle_indices",
                result.backward_simulation.particle_indices,
            ),
        ):
            _put_field(fields, arrays, name, value)
        metadata = _put_rao_blackwellized_filter_result(
            result.backward_simulation.filter_result,
            arrays,
            fields,
            prefix="filter.",
        )
        metadata.update(
            {
                "smoother_method_id": result.method_id,
                "simulation_method_id": result.backward_simulation.method_id,
                "sample_shape": list(result.sample_shape),
                "ancestry_gradient": result.ancestry_gradient,
                "process_id": result.process_id,
                "approximation_id": result.approximation_id,
            }
        )
        return (
            "rao_blackwellized_smoother",
            metadata,
            ("backward_simulation.filter_result.problem",),
        )

    if isinstance(result, ParticleGenealogicalScoreResult):
        _put_field(fields, arrays, "flat_score", result.flat_score)
        _put_field(fields, arrays, "case_scores", result.case_scores)
        _put_field(fields, arrays, "valid", result.valid)
        _put_flat_inexact_tree(
            trees,
            arrays,
            "model_score",
            result.score,
            expected_size=result.parameter_size,
        )
        metadata = {
            "method_id": result.method_id,
            "ancestry_gradient": result.ancestry_gradient,
            "parameter_size": result.parameter_size,
            "parameter_paths": list(result.parameter_paths),
            "model_id": result.model_id,
            "problem_id": result.problem_id,
            "sequence_id": result.sequence_id,
            "input_id": result.input_id,
            "case_shape": list(result.filter_result.case_shape),
            "case_axes": list(result.filter_result.case_axes),
            "case_ids": list(result.filter_result.case_ids),
        }
        return (
            "particle_genealogical_score",
            metadata,
            ("model_score", "filter_result"),
        )

    if isinstance(result, ParticleFisherScoreResult):
        _put_field(fields, arrays, "flat_score", result.flat_score)
        _put_field(fields, arrays, "case_scores", result.case_scores)
        _put_field(fields, arrays, "valid", result.valid)
        _put_flat_inexact_tree(
            trees,
            arrays,
            "transition_score",
            result.transition_score,
            expected_size=result.parameter_size,
        )
        metadata = {
            "method_id": result.method_id,
            "parameter_size": result.parameter_size,
            "process_id": result.process_id,
            "approximation_id": result.approximation_id,
            "model_id": result.model_id,
            "problem_id": result.problem_id,
            "sequence_id": result.sequence_id,
            "input_id": result.input_id,
            "case_shape": list(result.smoother.case_shape),
            "case_axes": list(result.smoother.case_axes),
            "case_ids": list(result.smoother.case_ids),
        }
        return (
            "particle_fisher_score",
            metadata,
            ("transition_score", "smoother"),
        )

    if isinstance(result, ParticleFisherInformationResult):
        _put_field(fields, arrays, "information", result.information)
        _put_field(fields, arrays, "case_scores", result.case_scores)
        _put_field(fields, arrays, "valid", result.valid)
        _put_flat_inexact_tree(
            trees,
            arrays,
            "transition_score",
            result.score_result.transition_score,
            expected_size=result.parameter_size,
        )
        score = result.score_result
        metadata = {
            "method_id": result.method_id,
            "score_method_id": score.method_id,
            "parameter_size": result.parameter_size,
            "process_id": score.process_id,
            "approximation_id": score.approximation_id,
            "model_id": score.model_id,
            "problem_id": score.problem_id,
            "sequence_id": score.sequence_id,
            "input_id": score.input_id,
            "case_shape": list(score.smoother.case_shape),
            "case_axes": list(score.smoother.case_axes),
            "case_ids": list(score.smoother.case_ids),
        }
        return "particle_fisher_information", metadata, ("score_result",)

    if isinstance(result, ParticleFilterResult):
        metadata = _put_particle_filter_result(result, arrays, fields, prefix="")
        return "particle_filter", metadata, ("problem",)

    if isinstance(result, EnsembleFilterResult):
        metadata = _put_ensemble_filter_result(result, arrays, fields, prefix="")
        return "ensemble_filter", metadata, ("problem",)

    if isinstance(result, EnsembleSmootherResult):
        _put_field(fields, arrays, "ensembles", result.ensembles)
        _put_field(fields, arrays, "valid", result.valid)
        metadata = _put_ensemble_filter_result(
            result.filter_result, arrays, fields, prefix="filter."
        )
        metadata["pseudoinverse_tolerance"] = result.pseudoinverse_tolerance
        return "ensemble_smoother", metadata, ("filter_result.problem",)

    if isinstance(result, JumpBSDEEvaluation):
        metadata = _put_bsde_evaluation(result.base, arrays, fields, prefix="base.")
        for name, value in (
            ("jump_sums", result.jump_sums),
            ("compensator_increments", result.compensator_increments),
            ("compensated_jump_increments", result.compensated_jump_increments),
            ("local_residuals", result.local_residuals),
            ("global_residual", result.global_residual),
            ("valid_paths", result.valid_paths),
        ):
            _put_field(fields, arrays, name, value)
        for label, value in result.event_counts.items():
            _put_field(fields, arrays, f"event_counts.{label}", value)
        for label, value in result.event_status.items():
            _put_field(fields, arrays, f"event_status.{label}", value)
        metadata["problem_id"] = result.problem_id
        metadata["jump_labels"] = list(result.event_counts)
        return "jump_bsde_evaluation", metadata, ("base.paths.realization",)

    if isinstance(result, BSDEEvaluation):
        metadata = _put_bsde_evaluation(result, arrays, fields, prefix="")
        return "bsde_evaluation", metadata, ("paths.realization",)

    if isinstance(result, EnsembleKalmanResult):
        for name in (
            "initial_unconstrained_ensemble",
            "unconstrained_ensemble",
            "initial_ensemble",
            "ensemble",
        ):
            _put_tree(trees, arrays, name, getattr(result, name))
        for name in ("residuals", "temperatures"):
            _put_field(fields, arrays, name, getattr(result, name))
        for name in (
            "temperature_increments",
            "residual_norms",
            "ensemble_spreads",
            "effective_ranks",
            "parameter_update_norms",
        ):
            _put_field(
                fields,
                arrays,
                f"diagnostics.{name}",
                getattr(result.diagnostics, name),
            )
        _put_field(fields, arrays, "root_key", jr.key_data(result.root_key))
        metadata = {
            "ensemble_size": result.ensemble_size,
            "num_steps": result.num_steps,
            "converged": result.converged,
            "termination_reason": result.termination_reason,
            "duration_seconds": result.duration_seconds,
            "sample_memory_bytes": result.sample_memory_bytes,
            "target_ess": result.target_ess,
            "inflation": result.inflation,
            "forward_solve_count": result.diagnostics.forward_solve_count,
            "collapsed": result.diagnostics.collapsed,
            "collapse_step": result.diagnostics.collapse_step,
        }
        return "ensemble_kalman_inversion", metadata, ("problem",)

    if isinstance(result, SGMCMCResult):
        _put_tree(trees, arrays, "samples", result.samples)
        _put_tree(
            trees,
            arrays,
            "unconstrained_samples",
            result.unconstrained_samples,
        )
        _put_tree(trees, arrays, "final_states", result.final_states)
        _put_tree(trees, arrays, "burnin_states", result.burnin_states)
        _put_tree(trees, arrays, "diagnostics.rhat", result.diagnostics.rhat)
        _put_tree(
            trees,
            arrays,
            "diagnostics.bulk_ess",
            result.diagnostics.bulk_ess,
        )
        _put_tree(
            trees,
            arrays,
            "diagnostics.tail_ess",
            result.diagnostics.tail_ess,
        )
        _put_tree(trees, arrays, "diagnostics.mean", result.diagnostics.mean)
        _put_tree(
            trees,
            arrays,
            "diagnostics.standard_deviation",
            result.diagnostics.standard_deviation,
        )
        _put_field(fields, arrays, "gradient_norm", result.gradient_norm)
        _put_field(fields, arrays, "chain_keys", result.chain_keys)
        _put_field(fields, arrays, "root_key", jr.key_data(result.root_key))
        if result.log_density is not None:
            _put_field(fields, arrays, "log_density", result.log_density)
        if result.thermostat is not None:
            _put_field(fields, arrays, "thermostat", result.thermostat)
        if result.momentum_norm is not None:
            _put_field(fields, arrays, "momentum_norm", result.momentum_norm)
        control_metadata = None
        if result.control_variate is not None:
            _put_tree(
                trees,
                arrays,
                "control_variate.center",
                result.control_variate.center,
            )
            _put_tree(
                trees,
                arrays,
                "control_variate.full_gradient",
                result.control_variate.full_gradient,
            )
            control_metadata = {
                "fingerprint": result.control_variate.fingerprint,
                "construction_duration_seconds": (
                    result.control_variate.construction_duration_seconds
                ),
                "construction_gradient_evaluations": (
                    result.control_variate.construction_gradient_evaluations
                ),
            }
        metadata = {
            "algorithm": result.algorithm,
            "approximation": result.approximation,
            "chain_method": result.chain_method,
            "gradient_estimator_id": result.gradient_estimator_id,
            "num_chains": result.num_chains,
            "num_draws": result.num_draws,
            "num_burnin": result.num_burnin,
            "steps_per_sample": result.steps_per_sample,
            "num_updates": result.num_updates,
            "num_gradient_evaluations": result.num_gradient_evaluations,
            "step_size": result.step_size,
            "diffusion": result.diffusion,
            "initial_thermostat": result.initial_thermostat,
            "source_num_factors": result.source_num_factors,
            "batch_capacity": result.batch_capacity,
            "batch_fraction": result.batch_fraction,
            "source_fingerprint": result.source_fingerprint,
            "source_configuration": result.source_configuration,
            "control_variate": control_metadata,
            "compilation_duration_seconds": result.compilation_duration_seconds,
            "burnin_duration_seconds": result.burnin_duration_seconds,
            "sampling_duration_seconds": result.sampling_duration_seconds,
            "duration_seconds": result.duration_seconds,
            "samples_per_second": result.samples_per_second,
            "updates_per_second": result.updates_per_second,
            "gradient_evaluations_per_second": (result.gradient_evaluations_per_second),
            "sample_memory_bytes": result.sample_memory_bytes,
            "mean_update_gradient_norm": result.mean_update_gradient_norm,
            "max_update_gradient_norm": result.max_update_gradient_norm,
        }
        return "sgmcmc", metadata, ("problem",)

    if isinstance(result, AmortizedStateSpaceVariationalResult):
        _put_tree(trees, arrays, "states", result.states)
        _put_array_leaves(trees, arrays, "family_parameters", result.family)
        for name, value in (
            ("log_model", result.log_model),
            ("log_variational", result.log_variational),
            ("diagnostics.steps", result.diagnostics.steps),
            ("diagnostics.elbo", result.diagnostics.elbo),
            ("diagnostics.gradient_norm", result.diagnostics.gradient_norm),
            ("diagnostics.finite", result.diagnostics.finite),
        ):
            _put_field(fields, arrays, name, value)
        _put_field(fields, arrays, "root_key", jr.key_data(result.root_key))
        metadata = {
            "algorithm": "amortized-state-space-reverse-kl",
            "family_id": result.family.family_id,
            "num_draws": result.num_draws,
            "completed_steps": result.diagnostics.completed_steps,
            "duration_seconds": result.duration_seconds,
            "approximation_id": result.approximation_id,
            "hidden_size": result.config.hidden_size,
            "scale_floor": result.config.scale_floor,
            "optimization": result.config.optimization.as_dict(),
        }
        return (
            "amortized_state_space_variational",
            metadata,
            ("problem", "family.static"),
        )

    if isinstance(result, BufferedStateSpaceVariationalResult):
        _put_tree(trees, arrays, "states", result.states)
        _put_array_leaves(trees, arrays, "family_parameters", result.family)
        for name, value in (
            ("log_model", result.log_model),
            ("log_variational", result.log_variational),
            ("diagnostics.steps", result.diagnostics.steps),
            ("diagnostics.target_start", result.diagnostics.target_start),
            ("diagnostics.context_start", result.diagnostics.context_start),
            ("diagnostics.context_end", result.diagnostics.context_end),
            ("diagnostics.elbo", result.diagnostics.elbo),
            ("diagnostics.gradient_norm", result.diagnostics.gradient_norm),
            ("diagnostics.finite", result.diagnostics.finite),
            (
                "window.inclusion_probability",
                result.window_plan.inclusion_probability,
            ),
        ):
            _put_field(fields, arrays, name, value)
        _put_field(fields, arrays, "root_key", jr.key_data(result.root_key))
        metadata = {
            "algorithm": "buffered-state-space-reverse-kl",
            "family_id": result.family.family_id,
            "num_draws": result.num_draws,
            "duration_seconds": result.duration_seconds,
            "approximation_id": result.approximation_id,
            "target_length": result.config.target_length,
            "left_buffer": result.config.left_buffer,
            "right_buffer": result.config.right_buffer,
            "hidden_size": result.config.hidden_size,
            "scale_floor": result.config.scale_floor,
            "optimization": result.config.optimization.as_dict(),
        }
        return (
            "buffered_state_space_variational",
            metadata,
            ("problem", "family.static"),
        )

    if isinstance(result, StateSpaceVariationalResult):
        _put_tree(trees, arrays, "states", result.states)
        _put_array_leaves(
            trees,
            arrays,
            "family_parameters",
            result.family,
        )
        for name, value in (
            ("log_model", result.log_model),
            ("log_variational", result.log_variational),
            ("diagnostics.steps", result.diagnostics.steps),
            ("diagnostics.elbo", result.diagnostics.elbo),
            ("diagnostics.gradient_norm", result.diagnostics.gradient_norm),
            ("diagnostics.finite", result.diagnostics.finite),
        ):
            _put_field(fields, arrays, name, value)
        _put_field(fields, arrays, "root_key", jr.key_data(result.root_key))
        metadata = {
            "algorithm": "state-space-reverse-kl",
            "family_id": result.family.family_id,
            "num_draws": result.num_draws,
            "completed_steps": result.diagnostics.completed_steps,
            "duration_seconds": result.duration_seconds,
            "approximation_id": result.approximation_id,
            "initial_scale": result.config.initial_scale,
            "scale_floor": result.config.scale_floor,
            "optimization": result.config.optimization.as_dict(),
        }
        return (
            "state_space_variational",
            metadata,
            ("problem", "family.static"),
        )

    if isinstance(result, FlowVariationalResult):
        fitted = result.variational
        _put_tree(trees, arrays, "samples", fitted.samples)
        _put_tree(
            trees,
            arrays,
            "unconstrained_samples",
            fitted.unconstrained_samples,
        )
        _put_array_leaves(
            trees,
            arrays,
            "family_parameters",
            fitted.family,
        )
        _put_field(fields, arrays, "log_target", fitted.log_target)
        _put_field(fields, arrays, "log_variational", fitted.log_variational)
        _put_field(fields, arrays, "diagnostics.elbo", fitted.diagnostics.elbo)
        _put_field(
            fields,
            arrays,
            "initialization.elbo",
            result.initialization.diagnostics.elbo,
        )
        metadata = {
            "algorithm": "flow-reverse-kl-variational",
            "family_id": fitted.family.family_id,
            "num_draws": fitted.num_draws,
            "approximation_id": result.approximation_id,
            "duration_seconds": fitted.duration_seconds,
            "sample_memory_bytes": fitted.sample_memory_bytes,
            "family_memory_bytes": fitted.family_memory_bytes,
            "config": result.config.as_dict(),
        }
        return (
            "flow_variational",
            metadata,
            ("variational.problem", "initialization.problem", "family.static"),
        )

    if isinstance(result, VariationalResult):
        _put_tree(trees, arrays, "samples", result.samples)
        _put_tree(
            trees,
            arrays,
            "unconstrained_samples",
            result.unconstrained_samples,
        )
        _put_array_leaves(
            trees,
            arrays,
            "family_parameters",
            result.family,
        )
        for name, value in (
            ("log_target", result.log_target),
            ("log_variational", result.log_variational),
            ("diagnostics.steps", result.diagnostics.steps),
            ("diagnostics.elbo", result.diagnostics.elbo),
            ("diagnostics.gradient_norm", result.diagnostics.gradient_norm),
            ("diagnostics.finite", result.diagnostics.finite),
        ):
            _put_field(fields, arrays, name, value)
        _put_field(fields, arrays, "root_key", jr.key_data(result.root_key))
        metadata = {
            "algorithm": "reverse-kl-variational",
            "family_id": result.family.family_id,
            "num_draws": result.num_draws,
            "completed_steps": result.diagnostics.completed_steps,
            "duration_seconds": result.duration_seconds,
            "optimization_duration_seconds": (result.optimization_duration_seconds),
            "sampling_duration_seconds": result.sampling_duration_seconds,
            "sample_memory_bytes": result.sample_memory_bytes,
            "family_memory_bytes": result.family_memory_bytes,
            "approximation_id": result.approximation_id,
            "config": result.config.as_dict(),
        }
        return "variational", metadata, ("problem", "family.static")

    if isinstance(result, FlowNUTSResult):
        mcmc = result.mcmc
        _put_tree(trees, arrays, "samples", result.samples)
        _put_tree(trees, arrays, "unconstrained_samples", result.unconstrained_samples)
        _put_tree(trees, arrays, "final_states", result.final_states)
        _put_tree(
            trees, arrays, "warmup_states", tuple(item.state for item in result.warmup)
        )
        _put_tree(trees, arrays, "diagnostics.rhat", result.diagnostics.rhat)
        _put_tree(trees, arrays, "diagnostics.bulk_ess", result.diagnostics.bulk_ess)
        _put_tree(trees, arrays, "diagnostics.tail_ess", result.diagnostics.tail_ess)
        _put_tree(trees, arrays, "training_losses", result.training_losses)
        _put_tree(trees, arrays, "validation_losses", result.validation_losses)
        _put_array_leaves(trees, arrays, "flow_parameters", result.flow)
        for name, value in (
            ("log_density", result.log_density),
            ("acceptance_rate", result.acceptance_rate),
            ("divergent", result.divergent),
            ("energy", result.energy),
            ("num_integration_steps", result.num_integration_steps),
            ("num_trajectory_expansions", result.num_trajectory_expansions),
            ("chain_keys", result.chain_keys),
            (
                "adaptation_global_acceptance_rate",
                result.adaptation_global_acceptance_rate,
            ),
            ("adaptation_proposal_ess", result.adaptation_proposal_ess),
            ("adaptation_history_size", result.adaptation_history_size),
            ("global_acceptance_rate", result.global_acceptance_rate),
            ("global_accepted_count", result.global_accepted_count),
            (
                "global_mean_log_acceptance_ratio",
                result.global_mean_log_acceptance_ratio,
            ),
            ("global_nonfinite_count", result.global_nonfinite_count),
        ):
            _put_field(fields, arrays, name, value)
        _put_field(fields, arrays, "root_key", jr.key_data(result.root_key))
        _put_field(
            fields,
            arrays,
            "warmup.step_size",
            jnp.stack([item.step_size for item in result.warmup]),
        )
        _put_field(
            fields,
            arrays,
            "warmup.inverse_mass_matrix",
            jnp.stack([item.inverse_mass_matrix for item in result.warmup]),
        )
        metadata = {
            "algorithm": result.algorithm,
            "chain_method": result.chain_method,
            "num_chains": result.num_chains,
            "num_draws": result.num_draws,
            "num_unique_initial_positions": result.num_unique_initial_positions,
            "duration_seconds": result.duration_seconds,
            "nuts_adaptation_duration_seconds": (result.nuts_adaptation_duration_seconds),
            "flow_adaptation_duration_seconds": (result.flow_adaptation_duration_seconds),
            "flow_training_duration_seconds": list(result.flow_training_duration_seconds),
            "stabilization_duration_seconds": (result.stabilization_duration_seconds),
            "sampling_duration_seconds": result.sampling_duration_seconds,
            "samples_per_second": result.samples_per_second,
            "sample_memory_bytes": result.sample_memory_bytes,
            "flow_parameter_memory_bytes": result.flow_parameter_memory_bytes,
            "history_memory_bytes": result.history_memory_bytes,
            "max_num_doublings": mcmc.max_num_doublings,
            "config": result.config.as_dict(),
            "warmup_duration_seconds": [item.duration_seconds for item in result.warmup],
        }
        return "flow_nuts", metadata, ("mcmc.problem", "flow.static")

    if isinstance(result, MCMCResult):
        _put_tree(trees, arrays, "samples", result.samples)
        _put_tree(trees, arrays, "unconstrained_samples", result.unconstrained_samples)
        _put_tree(trees, arrays, "final_states", result.final_states)
        _put_tree(
            trees, arrays, "warmup_states", tuple(item.state for item in result.warmup)
        )
        _put_tree(trees, arrays, "diagnostics.rhat", result.diagnostics.rhat)
        _put_tree(trees, arrays, "diagnostics.bulk_ess", result.diagnostics.bulk_ess)
        _put_tree(trees, arrays, "diagnostics.tail_ess", result.diagnostics.tail_ess)
        for name in (
            "log_density",
            "acceptance_rate",
            "divergent",
            "energy",
            "num_integration_steps",
            "num_trajectory_expansions",
            "chain_keys",
        ):
            _put_field(fields, arrays, name, getattr(result, name))
        _put_field(fields, arrays, "root_key", jr.key_data(result.root_key))
        _put_field(
            fields,
            arrays,
            "warmup.step_size",
            jnp.stack([item.step_size for item in result.warmup]),
        )
        _put_field(
            fields,
            arrays,
            "warmup.inverse_mass_matrix",
            jnp.stack([item.inverse_mass_matrix for item in result.warmup]),
        )
        if result.causal_diagnostics is not None:
            _put_field(
                fields,
                arrays,
                "causal.converged",
                result.causal_diagnostics.converged,
            )
            _put_field(
                fields,
                arrays,
                "causal.fallback_used",
                result.causal_diagnostics.fallback_used,
            )
            _put_field(
                fields,
                arrays,
                "causal.outer_iterations",
                result.causal_diagnostics.outer_iterations,
            )
            _put_field(
                fields,
                arrays,
                "causal.maximum_residual",
                result.causal_diagnostics.maximum_residual,
            )
            _put_field(
                fields,
                arrays,
                "causal.accepted_nonlinear_steps",
                result.causal_diagnostics.accepted_nonlinear_steps,
            )
            _put_field(
                fields,
                arrays,
                "causal.rejected_nonlinear_steps",
                result.causal_diagnostics.rejected_nonlinear_steps,
            )
            _put_field(
                fields,
                arrays,
                "causal.transition_evaluations",
                result.causal_diagnostics.transition_evaluations,
            )
        metadata = {
            "algorithm": result.algorithm,
            "chain_method": result.chain_method,
            "trajectory_method": result.trajectory_method,
            "causal_config": (
                None if result.causal_config is None else result.causal_config.as_dict()
            ),
            "num_chains": result.num_chains,
            "num_draws": result.num_draws,
            "duration_seconds": result.duration_seconds,
            "adaptation_duration_seconds": result.adaptation_duration_seconds,
            "sampling_duration_seconds": result.sampling_duration_seconds,
            "samples_per_second": result.samples_per_second,
            "sample_memory_bytes": result.sample_memory_bytes,
            "max_num_doublings": result.max_num_doublings,
            "warmup_duration_seconds": [item.duration_seconds for item in result.warmup],
            "warmup_num_integration_steps": [
                item.num_integration_steps for item in result.warmup
            ],
        }
        return "mcmc", metadata, ("problem",)

    if isinstance(result, SGMCMCMixingReport):
        for name, value in (
            ("max_rhat", result.max_rhat),
            ("min_bulk_ess", result.min_bulk_ess),
            ("min_tail_ess", result.min_tail_ess),
            ("max_gradient_norm", result.max_gradient_norm),
        ):
            _put_field(fields, arrays, name, value)
        metadata = result.as_dict()
        metadata["thresholds"] = {
            "max_rhat": result.thresholds.max_rhat,
            "min_bulk_ess": result.thresholds.min_bulk_ess,
            "min_tail_ess": result.thresholds.min_tail_ess,
            "allow_nonfinite_updates": (result.thresholds.allow_nonfinite_updates),
        }
        for name in fields:
            metadata.pop(name, None)
        return "sgmcmc_mixing_report", metadata, ()

    if isinstance(result, MCMCConvergenceReport):
        for name in (
            "max_rhat",
            "min_bulk_ess",
            "min_tail_ess",
            "mean_acceptance_rate",
            "divergence_count",
            "max_integration_steps",
            "max_trajectory_expansions",
            "trajectory_saturation_count",
        ):
            _put_field(fields, arrays, name, getattr(result, name))
        metadata = result.as_dict()
        metadata["thresholds"] = {
            "max_rhat": result.thresholds.max_rhat,
            "min_bulk_ess": result.thresholds.min_bulk_ess,
            "min_tail_ess": result.thresholds.min_tail_ess,
            "allow_divergences": result.thresholds.allow_divergences,
            "allow_trajectory_saturation": (
                result.thresholds.allow_trajectory_saturation
            ),
        }
        for name in fields:
            metadata.pop(name, None)
        return "mcmc_convergence_report", metadata, ()

    if isinstance(result, MAPCandidateSearchResult):
        if result.position is not None:
            _put_tree(trees, arrays, "position", result.position)
        if result.parameters is not None:
            _put_tree(trees, arrays, "parameters", result.parameters)
        _put_field(fields, arrays, "objective", result.objective)
        _put_field(fields, arrays, "log_density", result.log_density)
        metadata = {
            "valid": result.valid,
            "termination_reason": result.termination_reason,
            "flat_index": result.flat_index,
            "product_index": result.product_index,
            "axis_paths": result.axis_paths,
            "product_shape": result.product_shape,
            "candidate_count": result.candidate_count,
            "objective_evaluations": result.objective_evaluations,
            "valid_evaluations": result.valid_evaluations,
            "invalid_evaluations": result.invalid_evaluations,
            "effective_batch_size": result.effective_batch_size,
            "candidate_signature": result.candidate_signature,
            "method_id": result.method_id,
            "search": {"batch_size": result.search.batch_size},
        }
        return "map_candidate_search", metadata, ("problem", "search")

    if isinstance(result, GaussianProcessMAPSearchResult):
        _put_tree(trees, arrays, "position", result.position)
        _put_tree(trees, arrays, "parameters", result.parameters)
        _put_tree(
            trees,
            arrays,
            "evaluated_positions",
            result.evaluated_positions,
        )
        _put_tree(trees, arrays, "lower_bounds", result.lower_bounds)
        _put_tree(trees, arrays, "upper_bounds", result.upper_bounds)
        _put_tree(
            trees,
            arrays,
            "search_kernel",
            result.search.surrogate.kernel,
        )
        for name in (
            "objective",
            "log_density",
            "raw_objectives",
            "valid_evaluations",
            "proposal_kinds",
            "best_objective_history",
            "best_history_valid",
        ):
            _put_field(fields, arrays, name, getattr(result, name))
        _put_field(
            fields,
            arrays,
            "search_noise_scale",
            result.search.surrogate.noise_scale,
        )
        _put_field(
            fields,
            arrays,
            "search_jitter",
            result.search.surrogate.jitter,
        )
        metadata = {
            "valid": result.valid,
            "termination_reason": result.termination_reason,
            "objective_evaluations": result.objective_evaluations,
            "invalid_evaluations": result.invalid_evaluations,
            "fallback_count": result.fallback_count,
            "surrogate_failure_count": result.surrogate_failure_count,
            "design_signature": result.design_signature,
            "method_id": result.method_id,
            "proposal_seconds": result.proposal_seconds,
            "objective_seconds": result.objective_seconds,
            "search": {
                "max_evaluations": result.search.max_evaluations,
                "initial_evaluations": result.search.initial_evaluations,
                "candidate_count": result.search.candidate_count,
                "improvement_margin": result.search.improvement_margin,
                "minimum_separation": result.search.minimum_separation,
                "kernel": type(result.search.surrogate.kernel).__name__,
                "noise_scale_units": "raw_negative_log_density",
                "noise_standardization": "noise_scale / objective_scale",
                "jitter_units": "standardized_covariance",
            },
        }
        return (
            "gaussian_process_map_search",
            metadata,
            ("problem", "search", "key"),
        )

    if isinstance(result, MAPSearchResult):
        _put_tree(trees, arrays, "position", result.position)
        _put_tree(trees, arrays, "parameters", result.parameters)
        _put_tree(
            trees,
            arrays,
            "population_positions",
            result.population_positions,
        )
        _put_tree(trees, arrays, "lower_bounds", result.lower_bounds)
        _put_tree(trees, arrays, "upper_bounds", result.upper_bounds)
        _put_field(fields, arrays, "objective", result.objective)
        _put_field(fields, arrays, "log_density", result.log_density)
        _put_field(
            fields,
            arrays,
            "population_objectives",
            result.population_objectives,
        )
        _put_field(
            fields,
            arrays,
            "best_objective_history",
            result.best_objective_history,
        )
        metadata = {
            "population_converged": result.population_converged,
            "termination_reason": result.termination_reason,
            "generations": result.generations,
            "objective_evaluations": result.objective_evaluations,
            "invalid_evaluations": result.invalid_evaluations,
            "design_signature": result.design_signature,
            "search": {
                "population_size": result.search.population_size,
                "max_generations": result.search.max_generations,
                "strategy": result.search.strategy,
                "differential_weight": result.search.differential_weight,
                "crossover_rate": result.search.crossover_rate,
                "relative_tolerance": result.search.relative_tolerance,
                "absolute_tolerance": result.search.absolute_tolerance,
            },
        }
        return "map_search", metadata, ("problem", "search", "key")

    if isinstance(result, MAPResult):
        for name in ("position", "parameters", "gradient"):
            _put_tree(trees, arrays, name, getattr(result, name))
        for name in (
            "objective",
            "log_density",
            "gradient_norm",
            "objective_history",
        ):
            _put_field(fields, arrays, name, getattr(result, name))
        metadata = {
            "num_steps": result.num_steps,
            "objective_evaluations": result.objective_evaluations,
            "converged": result.converged,
            "termination_reason": result.termination_reason,
            "duration_seconds": result.duration_seconds,
            "initial_compilation_seconds": result.initial_compilation_seconds,
            "initial_evaluation_seconds": result.initial_evaluation_seconds,
            "step_compilation_seconds": result.step_compilation_seconds,
            "optimization_seconds": result.optimization_seconds,
        }
        return "map", metadata, ("problem",)

    if isinstance(result, LaplaceResult):
        for name in ("map_position", "map_parameters", "gradient"):
            _put_tree(trees, arrays, name, getattr(result, name))
        for name in (
            "flat_map_position",
            "gradient_norm",
            "raw_precision",
            "precision",
            "covariance",
            "scale",
            "raw_eigenvalues",
            "eigenvalues",
            "damping",
        ):
            _put_field(fields, arrays, name, getattr(result, name))
        return (
            "laplace",
            {"backend": result.backend, "dimension": result.dimension},
            ("problem", "unravel"),
        )

    if isinstance(result, StructuredLaplaceResult):
        for name in ("map_position", "map_parameters"):
            _put_tree(trees, arrays, name, getattr(result, name))
        _put_array_leaves(trees, arrays, "curvature_estimate", result.curvature_estimate)
        _put_array_leaves(trees, arrays, "posterior_state", result.posterior_state)
        _put_field(fields, arrays, "gradient_norm", result.gradient_norm)
        _put_field(fields, arrays, "prior_precision", result.prior_precision)
        metadata = {
            "curvature": result.curvature,
            "dimension": result.dimension,
            "rank": result.rank,
            "duration_seconds": result.duration_seconds,
            "likelihood_curvature": result.likelihood_curvature,
            "approximate_memory_bytes": result.approximate_memory_bytes,
        }
        return (
            "structured_laplace",
            metadata,
            ("problem", "scale_mv", "covariance_mv", "whitening"),
        )

    if isinstance(result, NestedSamplingResult):
        for name in ("samples", "unconstrained_samples"):
            _put_tree(trees, arrays, name, getattr(result, name))
        _put_array_leaves(trees, arrays, "final_state", result.final_state)
        for name in (
            "log_prior",
            "log_likelihood",
            "birth_log_likelihood",
            "posterior_log_weights",
            "log_prior_volume",
            "live_counts",
            "sample_ids",
            "batch_indices",
            "log_evidence",
            "log_evidence_replicates",
            "log_evidence_shrinkage_std",
            "information",
            "posterior_effective_sample_size",
            "remaining_log_evidence",
            "remaining_evidence_fraction",
            "status",
            "valid",
        ):
            _put_field(fields, arrays, name, getattr(result, name))
        for name in (
            "insertion_ranks",
            "insertion_rank_pvalue",
            "rolling_insertion_rank_pvalues",
            "likelihood_monotonic",
            "constraints_satisfied",
            "initial_finite_fraction",
            "inner_acceptance_rate",
            "expansion_cap_fraction",
            "shrinkage_cap_fraction",
            "zero_movement_fraction",
            "unique_lineage_count",
            "effective_lineage_count",
            "covariance_rank",
            "covariance_condition",
        ):
            _put_field(
                fields,
                arrays,
                f"diagnostics.{name}",
                getattr(result.diagnostics, name),
            )
        _put_field(fields, arrays, "root_key", jr.key_data(result.root_key))
        metadata = {
            "method": result.method,
            "status_name": nested_sampling_status_name(int(result.status)),
            "converged": result.converged,
            "diagnostic_failures": result.diagnostics.failures,
            "duration_seconds": result.duration_seconds,
            "sample_memory_bytes": result.sample_memory_bytes,
            "num_live": result.num_live,
            "num_dead": result.num_dead,
            "num_samples": result.num_samples,
            "num_likelihood_evaluations": result.num_likelihood_evaluations,
            "num_inner_steps": result.num_inner_steps,
            "num_delete": result.num_delete,
        }
        return (
            "nested_sampling",
            metadata,
            ("problem", "final_state.static"),
        )

    if isinstance(result, PathfinderResult):
        for name in ("samples", "unconstrained_samples"):
            _put_tree(trees, arrays, name, getattr(result, name))
        _put_array_leaves(trees, arrays, "state", result.state)
        _put_array_leaves(trees, arrays, "path", result.path)
        for name in ("log_density", "log_approximation_density"):
            _put_field(fields, arrays, name, getattr(result, name))
        _put_field(fields, arrays, "root_key", jr.key_data(result.root_key))
        metadata = {
            "approximation_duration_seconds": result.approximation_duration_seconds,
            "sampling_duration_seconds": result.sampling_duration_seconds,
            "duration_seconds": result.duration_seconds,
            "sample_memory_bytes": result.sample_memory_bytes,
            "optimization_steps": result.optimization_steps,
            "num_samples": result.num_samples,
        }
        return "pathfinder", metadata, ("problem", "state.static", "path.static")

    if isinstance(result, TemperedSMCResult):
        for name in ("samples", "unconstrained_samples", "state"):
            _put_tree(trees, arrays, name, getattr(result, name))
        for name in (
            "final_weights",
            "temperatures",
            "effective_sample_sizes",
            "acceptance_rates",
            "divergence_rates",
            "log_evidence",
        ):
            _put_field(fields, arrays, name, getattr(result, name))
        _put_field(fields, arrays, "root_key", jr.key_data(result.root_key))
        metadata = {
            "duration_seconds": result.duration_seconds,
            "sample_memory_bytes": result.sample_memory_bytes,
            "num_particles": result.num_particles,
            "num_tempering_steps": result.num_tempering_steps,
            "num_unique_initial_particles": result.num_unique_initial_particles,
            "resampling_method": result.resampling_method,
        }
        return "tempered_smc", metadata, ("problem",)

    if isinstance(result, DiscrepancyIdentifiabilityReport):
        for name in (
            "baseline_parameter_bias",
            "fixed_gp_parameter_bias",
            "joint_gp_parameter_bias",
            "nll_improvement",
            "crps_improvement",
            "mean_coverage",
            "max_abs_parameter_gp_correlation",
        ):
            _put_field(fields, arrays, name, getattr(result, name))
        return (
            "discrepancy_identifiability_report",
            {
                "passed": result.passed,
                "failures": result.failures,
                "num_repeats": result.num_repeats,
            },
            (),
        )

    raise TypeError(
        "Unsupported UQ result type for export: "
        f"{type(result).__module__}.{type(result).__qualname__}."
    )


def _put_bellman_filter_result(result, arrays, fields, *, prefix):
    for name, value in (
        ("revised_previous_modes", result.revised_previous_modes),
        ("predicted_modes", result.predicted_modes),
        ("predicted_information", result.predicted_information),
        ("predicted_covariances", result.predicted_covariances),
        ("filtered_modes", result.filtered_modes),
        ("filtered_information", result.filtered_information),
        ("filtered_covariances", result.filtered_covariances),
        ("transition_matrices", result.transition_matrices),
        ("prediction_objectives", result.prediction_objectives),
        ("update_objectives", result.update_objectives),
        ("prediction_gradient_norms", result.prediction_gradient_norms),
        ("update_gradient_norms", result.update_gradient_norms),
        ("prediction_iterations", result.prediction_iterations),
        ("update_iterations", result.update_iterations),
        ("prediction_converged", result.prediction_converged),
        ("update_converged", result.update_converged),
        (
            "predicted_raw_minimum_eigenvalues",
            result.predicted_raw_minimum_eigenvalues,
        ),
        (
            "filtered_raw_minimum_eigenvalues",
            result.filtered_raw_minimum_eigenvalues,
        ),
        (
            "information_gain_minimum_eigenvalues",
            result.information_gain_minimum_eigenvalues,
        ),
        ("observation_log_prob", result.observation_log_prob),
        ("realized_kl_penalties", result.realized_kl_penalties),
        (
            "incremental_pseudo_log_likelihood",
            result.incremental_pseudo_log_likelihood,
        ),
        (
            "cumulative_pseudo_log_likelihood",
            result.cumulative_pseudo_log_likelihood,
        ),
        ("observed_counts", result.observed_counts),
        ("step_valid", result.step_valid),
        ("mode_valid", result.mode_valid),
        ("pseudo_likelihood_valid", result.pseudo_likelihood_valid),
        ("valid", result.valid),
        ("status", result.status),
        ("times", result.times),
        ("final_state.mode", result.final_state.mode),
        ("final_state.information", result.final_state.information),
        ("final_state.covariance", result.final_state.covariance),
        ("final_state.time", result.final_state.time),
        (
            "final_state.pseudo_log_likelihood",
            result.final_state.pseudo_log_likelihood,
        ),
        ("final_state.mode_valid", result.final_state.mode_valid),
        (
            "final_state.pseudo_likelihood_valid",
            result.final_state.pseudo_likelihood_valid,
        ),
        ("final_state.status", result.final_state.status),
    ):
        _put_field(fields, arrays, prefix + name, value)
    return {
        "state_shape": list(result.state_shape),
        "observation_shape": list(result.observation_shape),
        "case_shape": list(result.case_shape),
        "case_axes": list(result.case_axes),
        "case_ids": list(result.case_ids),
        "model_id": result.model_id,
        "problem_id": result.problem_id,
        "sequence_id": result.sequence_id,
        "input_id": result.input_id,
        "execution_method": result.execution_method,
        "curvature_method": result.curvature_method,
        "curvature_damping": result.curvature_damping,
        "optimizer_rtol": result.optimizer_rtol,
        "optimizer_atol": result.optimizer_atol,
        "optimizer_max_steps": result.optimizer_max_steps,
        "max_dimension": result.max_dimension,
        "final_step_index": result.final_state.step_index,
    }


def _put_rao_blackwellized_filter_result(result, arrays, fields, *, prefix):
    for name, value in (
        ("initial_nonlinear_particles", result.initial_nonlinear_particles),
        ("initial_linear_means", result.initial_linear_means),
        ("initial_linear_covariances", result.initial_linear_covariances),
        ("initial_log_weights", result.initial_log_weights),
        (
            "predicted_nonlinear_particles",
            result.predicted_nonlinear_particles,
        ),
        ("predicted_linear_means", result.predicted_linear_means),
        (
            "predicted_linear_covariances",
            result.predicted_linear_covariances,
        ),
        ("posterior_linear_means", result.posterior_linear_means),
        (
            "posterior_linear_covariances",
            result.posterior_linear_covariances,
        ),
        ("posterior_log_weights", result.posterior_log_weights),
        ("nonlinear_particles", result.nonlinear_particles),
        ("linear_means", result.linear_means),
        ("linear_covariances", result.linear_covariances),
        ("log_weights", result.log_weights),
        ("ancestor_indices", result.ancestor_indices),
        ("transition_valid", result.transition_valid),
        ("effective_sample_sizes", result.effective_sample_sizes),
        ("resampled", result.resampled),
        ("incremental_log_likelihood", result.incremental_log_likelihood),
        ("cumulative_log_likelihood", result.cumulative_log_likelihood),
        ("step_valid", result.step_valid),
        ("valid", result.valid),
        ("status", result.status),
        ("times", result.times),
        ("final_state.nonlinear_particles", result.final_state.nonlinear_particles),
        ("final_state.linear_means", result.final_state.linear_means),
        ("final_state.linear_covariances", result.final_state.linear_covariances),
        ("final_state.log_weights", result.final_state.log_weights),
        ("final_state.time", result.final_state.time),
        ("final_state.log_likelihood", result.final_state.log_likelihood),
        ("final_state.valid", result.final_state.valid),
        ("final_state.status", result.final_state.status),
        ("final_state.root_key", jr.key_data(result.final_state.root_key)),
    ):
        _put_field(fields, arrays, prefix + name, value)
    problem = result.problem
    return {
        "nonlinear_state_shape": list(result.nonlinear_state_shape),
        "linear_state_shape": list(result.linear_state_shape),
        "observation_shape": list(result.observation_shape),
        "case_shape": list(result.case_shape),
        "case_axes": list(result.case_axes),
        "case_ids": list(result.case_ids),
        "num_particles": result.num_particles,
        "model_id": problem.model.model_id,
        "problem_id": problem.problem_id,
        "sequence_id": problem.observations.sequence_id,
        "input_id": (
            None if problem.input_signal is None else problem.input_signal.input_id
        ),
        "resampling_method": result.resampling_method,
        "resampling_policy": result.resampling_policy,
        "resampling_threshold": result.resampling_threshold,
        "process_id": problem.model.nonlinear_transition.process_id,
        "approximation_id": problem.model.nonlinear_transition.approximation_id,
    }


def _put_kalman_filter_result(result, arrays, fields, *, prefix):
    for name, value in (
        ("predicted_means", result.predicted_means),
        ("predicted_covariances", result.predicted_covariances),
        ("filtered_means", result.filtered_means),
        ("filtered_covariances", result.filtered_covariances),
        ("transition_matrices", result.transition_matrices),
        ("innovations", result.innovations),
        ("innovation_covariances", result.innovation_covariances),
        (
            "normalized_innovation_squared",
            result.normalized_innovation_squared,
        ),
        ("incremental_log_likelihood", result.incremental_log_likelihood),
        ("cumulative_log_likelihood", result.cumulative_log_likelihood),
        ("observed_counts", result.observed_counts),
        ("step_valid", result.step_valid),
        ("valid", result.valid),
        ("status", result.status),
        ("final_state.mean", result.final_state.mean),
        ("final_state.covariance", result.final_state.covariance),
        ("final_state.time", result.final_state.time),
        ("final_state.log_likelihood", result.final_state.log_likelihood),
        ("final_state.valid", result.final_state.valid),
        ("final_state.status", result.final_state.status),
    ):
        _put_field(fields, arrays, prefix + name, value)
    return {
        "state_shape": list(result.state_shape),
        "observation_shape": list(result.observation_shape),
        "case_shape": list(result.case_shape),
        "case_ids": list(result.case_ids),
        "model_id": result.model_id,
        "problem_id": result.problem_id,
        "sequence_id": result.sequence_id,
        "input_id": result.input_id,
        "covariance_regularization": result.covariance_regularization,
        "execution_method": result.execution_method,
        "covariance_form": result.covariance_form,
        "final_step_index": result.final_state.step_index,
    }


def _put_particle_filter_result(result, arrays, fields, *, prefix):
    for name, value in (
        ("initial_particles", result.initial_particles),
        ("initial_log_weights", result.initial_log_weights),
        ("initial_valid", result.initial_valid),
        ("predicted_particles", result.predicted_particles),
        ("posterior_log_weights", result.posterior_log_weights),
        ("particles", result.particles),
        ("log_weights", result.log_weights),
        ("ancestor_indices", result.ancestor_indices),
        ("transition_valid", result.transition_valid),
        ("effective_sample_sizes", result.effective_sample_sizes),
        ("resampled", result.resampled),
        ("incremental_log_likelihood", result.incremental_log_likelihood),
        ("cumulative_log_likelihood", result.cumulative_log_likelihood),
        ("step_valid", result.step_valid),
        ("valid", result.valid),
        ("status", result.status),
        ("times", result.times),
        ("final_state.particles", result.final_state.particles),
        ("final_state.log_weights", result.final_state.log_weights),
        ("final_state.time", result.final_state.time),
        ("final_state.log_likelihood", result.final_state.log_likelihood),
        ("final_state.valid", result.final_state.valid),
        ("final_state.status", result.final_state.status),
        ("final_state.root_key", jr.key_data(result.final_state.root_key)),
    ):
        _put_field(fields, arrays, prefix + name, value)
    return {
        "state_shape": list(result.state_shape),
        "observation_shape": list(result.observation_shape),
        "case_shape": list(result.case_shape),
        "case_axes": list(result.case_axes),
        "case_ids": list(result.case_ids),
        "num_particles": result.num_particles,
        "model_id": result.model_id,
        "problem_id": result.problem_id,
        "sequence_id": result.sequence_id,
        "input_id": result.input_id,
        "resampling_method": result.resampling_method,
        "resampling_policy": result.resampling_policy,
        "resampling_threshold": result.resampling_threshold,
        "final_step_index": result.final_state.step_index,
    }


def _put_ensemble_filter_result(result, arrays, fields, *, prefix):
    for name, value in (
        ("forecast_ensembles", result.forecast_ensembles),
        ("analysis_ensembles", result.analysis_ensembles),
        ("forecast_observations", result.forecast_observations),
        ("innovations", result.innovations),
        (
            "normalized_innovation_squared",
            result.normalized_innovation_squared,
        ),
        ("incremental_log_likelihood", result.incremental_log_likelihood),
        ("cumulative_log_likelihood", result.cumulative_log_likelihood),
        ("observed_counts", result.observed_counts),
        ("step_valid", result.step_valid),
        ("valid", result.valid),
        ("status", result.status),
        ("times", result.times),
        ("final_state.ensemble", result.final_state.ensemble),
        ("final_state.time", result.final_state.time),
        ("final_state.log_likelihood", result.final_state.log_likelihood),
        ("final_state.valid", result.final_state.valid),
        ("final_state.status", result.final_state.status),
        ("final_state.root_key", jr.key_data(result.final_state.root_key)),
    ):
        _put_field(fields, arrays, prefix + name, value)
    return {
        "state_shape": list(result.state_shape),
        "observation_shape": list(result.observation_shape),
        "case_shape": list(result.case_shape),
        "case_axes": list(result.case_axes),
        "case_ids": list(result.case_ids),
        "ensemble_size": result.ensemble_size,
        "model_id": result.model_id,
        "problem_id": result.problem_id,
        "sequence_id": result.sequence_id,
        "input_id": result.input_id,
        "inflation": result.inflation,
        "covariance_regularization": result.covariance_regularization,
        "final_step_index": result.final_state.step_index,
    }


def _put_bsde_evaluation(result, arrays, fields, *, prefix):
    for name, value in (
        ("values", result.values),
        ("controls", result.controls),
        ("generator_values", result.generator_values),
        ("terminal_residual", result.terminal_residual),
        ("local_residuals", result.local_residuals),
        ("global_residual", result.global_residual),
        ("martingale_increments", result.martingale_increments),
        ("valid_paths", result.valid_paths),
        ("paths.times", result.paths.times),
        ("paths.states", result.paths.states),
        ("paths.wiener_increments", result.paths.wiener_increments),
        ("paths.valid", result.paths.valid),
    ):
        _put_field(fields, arrays, prefix + name, value)
    for label, events in result.paths.jump_events.items():
        for name, value in (
            ("times", events.times),
            ("channels", events.channels),
            ("marks", events.marks),
            ("valid", events.valid),
            ("status", events.status),
        ):
            _put_field(
                fields,
                arrays,
                f"{prefix}paths.jump_events.{label}.{name}",
                value,
            )
        if events.pre_states is not None:
            _put_field(
                fields,
                arrays,
                f"{prefix}paths.jump_events.{label}.pre_states",
                events.pre_states,
            )
            _put_field(
                fields,
                arrays,
                f"{prefix}paths.jump_events.{label}.post_states",
                events.post_states,
            )
    realization = result.paths.realization
    metadata = {
        "quadrature": result.quadrature,
        "control_mode": result.control_mode,
        "sample_shape": list(result.paths.sample_shape),
        "state_shape": list(result.paths.state_shape),
        "noise_shape": list(result.paths.noise_shape),
        "path_id": result.paths.path_id,
        "process_id": result.paths.process_id,
        "jump_labels": list(result.paths.jump_events),
        "realization_type": None,
        "realization_id": None,
        "coupling_id": None,
    }
    if realization is not None:
        metadata.update(
            {
                "realization_type": type(realization).__name__,
                "realization_id": realization.realization_id,
                "coupling_id": realization.coupling_id,
            }
        )
    return metadata


def _put_field(fields, arrays, name, value):
    array_name = f"field/{name}"
    arrays[array_name] = _portable_array(value)
    fields[name] = array_name


def _put_tree(trees, arrays, name, tree):
    path_leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
    if not path_leaves:
        raise ValueError(f"Result array tree {name!r} has no leaves.")
    paths = []
    names = []
    for index, (path, leaf) in enumerate(path_leaves):
        array_name = f"tree/{name}/{index:06d}"
        arrays[array_name] = _portable_array(leaf)
        paths.append(jax.tree_util.keystr(path) or "<root>")
        names.append(array_name)
    trees[name] = {"paths": paths, "arrays": names}


def _put_array_leaves(trees, arrays, name, tree):
    paths = []
    names = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not eqx.is_array(leaf):
            continue
        array_name = f"tree/{name}/{len(names):06d}"
        arrays[array_name] = _portable_array(leaf)
        paths.append(jax.tree_util.keystr(path) or "<root>")
        names.append(array_name)
    if names:
        trees[name] = {"paths": paths, "arrays": names}


def _put_flat_inexact_tree(
    trees,
    arrays,
    name,
    tree,
    *,
    expected_size,
):
    paths = []
    names = []
    leaf_shapes = []
    flat_slices = []
    offset = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not eqx.is_inexact_array(leaf):
            continue
        array_name = f"tree/{name}/{len(names):06d}"
        array = _portable_array(leaf)
        size = int(array.size)
        arrays[array_name] = array
        paths.append(jax.tree_util.keystr(path) or "<root>")
        names.append(array_name)
        leaf_shapes.append(list(array.shape))
        flat_slices.append([offset, offset + size])
        offset += size
    if not names:
        raise ValueError(f"Result inexact array tree {name!r} has no leaves.")
    if offset != int(expected_size):
        raise ValueError(
            f"Result inexact array tree {name!r} has size {offset}, "
            f"expected {expected_size}."
        )
    trees[name] = {
        "paths": paths,
        "arrays": names,
        "leaf_shapes": leaf_shapes,
        "flat_slices": flat_slices,
    }


def _portable_array(value):
    if hasattr(value, "dtype") and str(value.dtype).startswith("key<"):
        value = jr.key_data(value)
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("Portable result archives cannot contain object arrays.")
    return array


__all__ = [
    "UQResultArchive",
    "decode_parameter_name",
    "encode_parameter_name",
    "export_result",
    "read_result_archive",
    "to_arviz",
]
