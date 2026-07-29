#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import base64
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

from ._checkpoint import (
    _json_value,
    _read_array_archive,
    _write_array_archive,
    CheckpointCorruptionError,
)
from ._diagnostics import MCMCConvergenceReport
from ._discrepancy_diagnostics import DiscrepancyIdentifiabilityReport
from ._eki import EnsembleKalmanResult
from ._laplace import LaplaceResult
from ._laplax_backend import StructuredLaplaceResult
from ._map import MAPResult
from ._mcmc import MCMCResult
from ._pathfinder import PathfinderResult
from ._smc import TemperedSMCResult


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


def to_arviz(result: MCMCResult, /):
    """Convert chain-preserving MCMC draws and sampler statistics to ArviZ."""
    if not isinstance(result, MCMCResult):
        raise TypeError("to_arviz currently supports MCMCResult only.")
    try:
        import arviz as az
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise ImportError("ArviZ is required for to_arviz().") from error

    posterior: dict[str, np.ndarray] = {}
    dimensions: dict[str, list[str]] = {}
    coordinates: dict[str, np.ndarray] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(result.samples)[0]:
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

    sample_stats = {
        "lp": np.asarray(result.log_density),
        "acceptance_rate": np.asarray(result.acceptance_rate),
        "diverging": np.asarray(result.divergent),
        "energy": np.asarray(result.energy),
        "n_steps": np.asarray(result.num_integration_steps),
        "tree_depth": np.asarray(result.num_trajectory_expansions),
    }
    inference_data = az.from_dict(
        {
            "posterior": posterior,
            "sample_stats": sample_stats,
        },
        sample_dims=("chain", "draw"),
        dims=dimensions,
        coords=coordinates,
        attrs={
            "posterior": {
                "phydrax_algorithm": result.algorithm,
                "phydrax_chain_method": result.chain_method,
            }
        },
    )
    return inference_data


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
        metadata = {
            "algorithm": result.algorithm,
            "chain_method": result.chain_method,
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
