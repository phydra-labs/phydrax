#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualify built Metal wheels on real Apple Silicon from a fresh environment.

The tool creates a venv with the supplied interpreter, installs exactly the
given Phydrax wheel with its ``metal`` extra resolved to the given companion
wheel, verifies installed-file provenance, probes the actual MPS runtime and
build identity, then runs every required installed-wheel test file serially in
its own process from a copy of ``tests/`` (never the source package) and the
advertised example from a copy outside the checkout::

    uv run python -m tools.qualify_metal_wheels --python /usr/local/bin/python3.13 \\
        --phydrax-wheel dist/phydrax-0.2.2-py3-none-any.whl \\
        --companion-wheel wheelhouse/phydrax_metal-0.2.2-py3-none-macosx_14_0_arm64.whl \\
        --work-dir /tmp/phydrax-metal-qualify --output /tmp/phydrax-metal-qualify.json

Device files start with ``JAX_PLATFORMS=mps`` and ``JAX_ENABLE_X64=1``; host
files start without a platform selection and own their children. A missing,
duplicated, unexpected, skipped, expected-failure or failed node, a device node
without matching ``metal_runtime`` evidence, stale per-file evidence, or a
nonzero child status fails the qualification even if pytest exits zero.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import uuid
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Literal, TypedDict

from tools.build_metal_wheels import BuildFailure, read_lock, source_identity
from tools.check_cuda_extras import wheel_metadata
from tools.check_metal_extras import (
    COMPANION,
    companion_errors,
    COMPANION_PACKAGE,
    CompanionFacts,
    core_metadata_errors,
    record_digest,
    RUNTIME_PLATFORM,
)
from tools.qualify_cuda_wheels import test_requirements


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = Path("examples/metal_scientific_execution.py")
DEVICE_ENVIRONMENT = {"JAX_PLATFORMS": RUNTIME_PLATFORM, "JAX_ENABLE_X64": "1"}
RUNTIME_PROPERTY = "metal_runtime"
CHILD_RUNTIME_PROPERTY = "metal_child_runtime"
INHERITED_ENVIRONMENT = (
    "HOME",
    "TMPDIR",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
)
SYSTEM_PATH = ("/usr/bin", "/bin", "/usr/sbin", "/sbin")
CASE_REPORT = "metal-case-report.json"
EVIDENCE_TEXT_LIMIT = 20_000
DEFAULT_FILE_TIMEOUT = 3600.0
type FileKind = Literal["device", "host"]


@dataclass(frozen=True, slots=True)
class RequiredFile:
    """One installed-wheel test file, its process kind and its exact node set.

    Every node of a device file must carry ``metal_runtime`` evidence. A host
    file's ``child_runtime_tests`` must carry ``metal_child_runtime`` evidence
    from the MPS child process they own.
    """

    kind: FileKind
    tests: tuple[str, ...]
    child_runtime_tests: tuple[str, ...] = ()

    def nodeids(self, path: str, /) -> tuple[str, ...]:
        return tuple(f"{path}::{test}" for test in self.tests)

    def runtime_property(self, test: str, /) -> str | None:
        """The runtime-evidence property one passing node must record, if any."""
        if self.kind == "device":
            return RUNTIME_PROPERTY
        return CHILD_RUNTIME_PROPERTY if test in self.child_runtime_tests else None


def _cases(name: str, *ids: str) -> tuple[str, ...]:
    return tuple(f"{name}[{case}]" for case in ids)


REQUIRED_FILES: Mapping[str, RequiredFile] = {
    "tests/integration/test_metal_import_contract.py": RequiredFile(
        "host",
        (
            "test_backend_diagnostics_import_is_device_inert",
            "test_startup_mps_selection_is_admitted_from_metadata_before_backend_use",
            *_cases(
                "test_conflicting_mps_plugin_refuses_before_backend_use",
                "configure",
                "startup",
            ),
            "test_explicit_cpu_with_companion_preserves_jax_semantics",
            "test_selected_metal_initialization_preserves_configuration_and_callables",
        ),
        child_runtime_tests=(
            "test_selected_metal_initialization_preserves_configuration_and_callables",
        ),
    ),
    "tests/integration/test_metal_runtime.py": RequiredFile(
        "device",
        (
            "test_runtime_identity_matches_installed_companion",
            "test_float32_complex64_and_exact_64_bit_integers_share_one_x64_program",
            *_cases(
                "test_wide_float_is_refused_without_narrowing",
                *(
                    f"{dtype}-{boundary}"
                    for dtype in ("float64", "complex128")
                    for boundary in (
                        "transfer",
                        "constant",
                        "conversion",
                        "intermediate",
                        "result",
                    )
                ),
            ),
            "test_float32_contraction_is_exact_where_representable_and_within_fp32_bound",
            "test_integer_contraction_is_exact_or_explicitly_refused",
            "test_full_width_integer_keys_sort_stably_and_compute_exactly",
            "test_threefry_typed_keys_match_separate_cpu_reference",
            *_cases(
                "test_compiled_guard_in_nested_control_flow",
                "jit-accepted",
                "jit-refused",
                "filter_jit-accepted",
                "filter_jit-refused",
            ),
            *_cases(
                "test_ordinary_gelu_derivatives_match_analytic_reference", "tanh", "erf"
            ),
            "test_ordinary_attention_second_order_matches_separate_cpu_reference",
            "test_dependent_dispatch_completes_with_exact_result",
            "test_execution_error_propagates_to_dependent_results",
            *_cases(
                "test_ordered_host_callbacks_preserve_effects_and_errors",
                "zero-trip",
                "six-steps",
            ),
            "test_bfloat16_host_callback_preserves_dtype_and_strided_result",
            "test_explicit_float32_literal_cast_preserves_rounding_and_ieee_classes",
        ),
    ),
    "tests/integration/test_metal_native_lifecycle.py": RequiredFile(
        "device",
        (
            "test_non_donated_input_survives_repeated_execution",
            "test_donated_input_ownership_is_truthful",
            "test_host_transfer_and_readback_do_not_alias_device_buffers",
            "test_device_copy_is_independent_of_deleted_source",
            "test_zero_copy_export_is_explicit",
            "test_repeated_compilation_releases_callbacks_and_executables",
            "test_allocation_failure_is_reported_and_runtime_recovers",
            "test_unsupported_element_type_refuses_without_aborting",
            "test_process_exit_with_live_and_inflight_work",
        ),
    ),
    "tests/integration/test_metal_pde.py": RequiredFile(
        "device", ("test_float32_mlp_poisson_step_matches_cpu_reference",)
    ),
    "tests/integration/test_metal_spatial.py": RequiredFile(
        "device",
        (
            *_cases(
                "test_metal_morton_integer_codes_keep_every_high_valid_bit",
                "1d-depth63",
                "2d-depth31",
                "3d-depth21",
            ),
            "test_metal_morton_float32_encoding_is_exact_at_full_depth_and_periodic_seams",
            *_cases(
                "test_metal_prepared_float32_nearest_orders_ties_by_full_width_stable_ids",
                "int64",
                "uint64",
            ),
            "test_metal_prepared_float32_nearest_honors_masks_radius_and_periodic_seam",
            "test_metal_prepared_float32_candidate_capacity_overflow_refuses_rows",
            "test_metal_prepared_float32_invalid_geometry_duplicates_and_empty_sets_fail_closed",
            "test_metal_morton_radius_relation_periodic_pairs_boundary_and_pair_overflow",
            "test_metal_prepared_float32_refresh_reuses_rebuilds_and_rolls_back",
            "test_metal_prepared_float32_wrapped_geometry_is_differentiable",
            *_cases(
                "test_metal_sparse_float32_cancellation_keeps_seeded_compensation",
                "fragment-per-event",
                "one-fragment",
            ),
            "test_metal_sparse_float32_ordered_policies_follow_int64_stable_ids",
            "test_metal_sparse_packed_key_groups_order_int64_ids_beyond_float64",
            *_cases(
                "test_metal_sparse_float32_valid_zero_events_keep_jvp_vjp_and_curvature",
                "seeded-fragments-deterministic",
                "seeded-fragments-compensated",
                "one-fragment-deterministic",
                "one-fragment-compensated",
            ),
            "test_metal_sparse_float32_failed_traced_schedule_invalidates_derivatives",
            *_cases(
                "test_metal_fourier_complex64_preserves_representable_modes", "5-8", "8-5"
            ),
            "test_metal_fourier_float32_even_nyquist_splits_and_merges",
            "test_metal_fourier_complex64_multiaxis_phase_shift_matches_direct_evaluation",
            "test_metal_fourier_float32_jit_phase_derivatives_and_adjoint",
        ),
    ),
    "tests/integration/test_metal_models.py": RequiredFile(
        "device",
        (
            "test_float32_mace_energy_forces_and_parameter_gradient_track_cpu_float64",
            "test_float32_mace_second_order_actions_track_cpu_float64",
            *_cases(
                "test_float32_periodic_mace_stress_tracks_cpu_float64_and_replays_bitwise",
                "deterministic",
                "compensated",
            ),
            "test_float32_mace_unknown_species_fails_closed_after_an_accepted_evaluation",
            "test_float32_tabulated_mace_refuses_radii_outside_its_declared_support",
            "test_float32_matrix_free_pcg_converges_with_consumer_true_residual",
            "test_float32_matrix_free_gmres_mathematical_derivatives_match_host_solves",
            "test_float32_iteration_limit_reports_status_and_invalidates_implicit_derivative",
            "test_float32_matrix_free_bicgstab_breakdown_keeps_a_finite_iterate",
            "test_float32_nonfinite_rhs_reports_status_and_invalid_evidence",
            *_cases(
                "test_float32_heun_decay_matches_closed_form_value_and_rate_derivative",
                "two-steps",
                "eight-steps",
                "long-512-steps",
            ),
            "test_float32_heun_oscillator_matches_closed_form_value_and_frequency_derivative",
            "test_float32_heun_step_budget_exhaustion_reports_failure",
        ),
    ),
    "tests/integration/test_metal_index_primitives.py": RequiredFile(
        "device",
        (
            *_cases(
                "test_metal_gather_clamps_full_width_indices_before_narrowing",
                "int64",
                "uint64",
            ),
            *_cases(
                "test_metal_dynamic_slices_clamp_full_width_starts_before_narrowing",
                "int64",
                "uint64",
            ),
            *_cases(
                "test_metal_scatter_partial_windows_match_independent_cpu",
                "int64",
                "uint64",
            ),
            *_cases(
                "test_metal_scatter_batched_interleaved_windows_match_independent_cpu",
                "int64",
                "uint64",
            ),
            *_cases(
                "test_metal_duplicate_scatter_updates_preserve_full_width_values",
                "int64",
                "uint64",
                "complex64",
            ),
            "test_metal_out_of_bounds_nan_scatter_updates_do_not_contaminate_valid_duplicates",
            "test_metal_scatter_min_max_propagate_nan_and_preserve_signed_zero",
            *_cases(
                "test_metal_stable_descending_sort_keeps_full_width_key_payload_ties",
                "int64",
                "uint64",
            ),
            "test_metal_unrecognized_sort_comparator_is_refused_without_single_key_fallback",
            *_cases(
                "test_metal_single_point_scatter_full_width_bounds_and_reducers",
                "False-int64-int64",
                "False-int64-uint64",
                "False-uint64-int64",
                "False-uint64-uint64",
                "False-complex64-int64",
                "False-complex64-uint64",
                "False-float32-int64",
                "False-float32-uint64",
                "True-int64-int64",
                "True-int64-uint64",
                "True-uint64-int64",
                "True-uint64-uint64",
                "True-complex64-int64",
                "True-complex64-uint64",
                "True-float32-int64",
                "True-float32-uint64",
            ),
            *_cases(
                "test_metal_single_interleaved_window_scatter_drop_nan_zero_and_reducers",
                "int64",
                "uint64",
            ),
            "test_metal_single_window_scatter_without_index_components",
            *_cases(
                "test_metal_single_scatter_derivatives",
                "update-1-int64",
                "update-1-uint64",
                "update-3-int64",
                "update-3-uint64",
                "add-1-int64",
                "add-1-uint64",
                "add-3-int64",
                "add-3-uint64",
                "mul-1-int64",
                "mul-1-uint64",
                "mul-3-int64",
                "mul-3-uint64",
            ),
            *_cases(
                "test_metal_fill_gather_preserves_full_width_bounds_and_complete_windows",
                "int64-int64",
                "int64-uint64",
                "uint64-int64",
                "uint64-uint64",
                "float32-int64",
                "float32-uint64",
                "complex64-int64",
                "complex64-uint64",
            ),
            *_cases(
                "test_metal_fill_gather_batched_windows_keep_batch_and_offset_axes",
                "int64",
                "uint64",
            ),
        ),
    ),
    "tests/integration/test_metal_reduction_primitives.py": RequiredFile(
        "device",
        (
            *_cases(
                "test_reduce_extrema_initial_values_and_empty_axes",
                "ordinary-max",
                "ordinary-min",
                "empty-reduced-max",
                "empty-reduced-min",
                "no-axes-max",
                "no-axes-min",
                "scalar-max",
                "scalar-min",
                "empty-output-max",
                "empty-output-min",
            ),
            "test_neutral_sum_and_mean_scientific_axes",
            *_cases(
                "test_reduce_window_padding_dilation_and_initial_values",
                "padded-max",
                "padded-min",
                "padded-sum",
                "padded-prod",
                "cumulative-max",
                "cumulative-min",
                "cumulative-sum",
                "cumulative-prod",
                "dilated-max",
                "dilated-min",
                "dilated-sum",
                "dilated-prod",
                "cropped-two-axis-max",
                "cropped-two-axis-min",
                "cropped-two-axis-sum",
                "cropped-two-axis-prod",
                "empty-input-max",
                "empty-input-min",
                "empty-input-sum",
                "empty-input-prod",
                "scalar-max",
                "scalar-min",
                "scalar-sum",
                "scalar-prod",
            ),
            *_cases(
                "test_reduce_window_xor_exact_padding_and_dilation",
                *(
                    f"{case}-{dtype}"
                    for case in (
                        "padded",
                        "cumulative",
                        "dilated",
                        "cropped-two-axis",
                        "empty-input",
                        "scalar",
                    )
                    for dtype in ("bool", "int8", "uint64")
                ),
            ),
            *_cases("test_reduce_extrema_nan_and_signed_zero", "max", "min"),
            *_cases(
                "test_paired_arg_reduction_full_width_ties_and_nan",
                "int64-max",
                "int64-min",
                "uint64-max",
                "uint64-min",
            ),
            *_cases(
                "test_jax_arg_reduction_declares_int64_and_uses_first_tie", "max", "min"
            ),
            "test_sign_preserves_nan_signed_zero_and_subnormal_sign",
            "test_floating_remainder_extreme_exponents_subnormals_and_special_values",
            *_cases(
                "test_integer_remainder_never_passes_through_float", "int64", "uint64"
            ),
            "test_cbrt_fp32_ulp_bound_and_special_values",
            *_cases(
                "test_long_mixed_magnitude_sum_host64_accuracy",
                "0",
                "500000",
                "1000000",
            ),
            *_cases(
                "test_parallel_sum_arbitrary_axes_declared_fp32",
                "axes0",
                "axes1",
                "axes2",
                "axes3",
            ),
            *_cases(
                "test_associative_integer_reduction_exact_declared_width",
                "axes0-sum-int8",
                "axes0-sum-uint8",
                "axes0-sum-int16",
                "axes0-sum-uint16",
                "axes0-sum-int32",
                "axes0-sum-uint32",
                "axes0-sum-int64",
                "axes0-sum-uint64",
                "axes0-prod-int8",
                "axes0-prod-uint8",
                "axes0-prod-int16",
                "axes0-prod-uint16",
                "axes0-prod-int32",
                "axes0-prod-uint32",
                "axes0-prod-int64",
                "axes0-prod-uint64",
                "axes1-sum-int8",
                "axes1-sum-uint8",
                "axes1-sum-int16",
                "axes1-sum-uint16",
                "axes1-sum-int32",
                "axes1-sum-uint32",
                "axes1-sum-int64",
                "axes1-sum-uint64",
                "axes1-prod-int8",
                "axes1-prod-uint8",
                "axes1-prod-int16",
                "axes1-prod-uint16",
                "axes1-prod-int32",
                "axes1-prod-uint32",
                "axes1-prod-int64",
                "axes1-prod-uint64",
                "axes2-sum-int8",
                "axes2-sum-uint8",
                "axes2-sum-int16",
                "axes2-sum-uint16",
                "axes2-sum-int32",
                "axes2-sum-uint32",
                "axes2-sum-int64",
                "axes2-sum-uint64",
                "axes2-prod-int8",
                "axes2-prod-uint8",
                "axes2-prod-int16",
                "axes2-prod-uint16",
                "axes2-prod-int32",
                "axes2-prod-uint32",
                "axes2-prod-int64",
                "axes2-prod-uint64",
            ),
            *_cases(
                "test_associative_boolean_reduction_exact",
                "axes0-all",
                "axes0-any",
                "axes1-all",
                "axes1-any",
                "axes2-all",
                "axes2-any",
            ),
            *_cases(
                "test_associative_reduction_empty_and_no_axes",
                *(
                    f"{case}-{kind}"
                    for case in ("empty-reduced", "no-axes", "scalar", "empty-output")
                    for kind in ("sum", "prod", "all", "any", "xor")
                ),
            ),
            *_cases(
                "test_xor_reduction_exact_declared_width_and_initial",
                *(
                    f"{initial}-{axes}-{dtype}"
                    for initial in ("neutral", "nonidentity-plugin-specific")
                    for axes in ("axes0", "axes1", "axes2")
                    for dtype in (
                        "bool",
                        "int8",
                        "uint8",
                        "int16",
                        "uint16",
                        "int32",
                        "uint32",
                        "int64",
                        "uint64",
                    )
                ),
            ),
            *_cases(
                "test_xor_reduction_long_axes_and_many_outputs",
                *(
                    f"{layout}-{dtype}"
                    for layout in ("long-axis", "many-outputs")
                    for dtype in ("bool", "int8", "uint64")
                ),
            ),
            "test_array_tree_lineage_uint64_matches_independent_host_mix",
        ),
    ),
    "tests/integration/test_metal_device_regions.py": RequiredFile(
        "device",
        (
            *_cases(
                "test_existing_sparse_ordered_gather_replays_dynamic_values",
                "float32",
                "int64",
                "uint64",
            ),
            *_cases(
                "test_ordered_recurrence_preserves_event_rounding_and_selected_branch",
                "0",
                "1",
                "17",
            ),
            *_cases(
                "test_indexed_device_map_clamps_full_width_starts",
                "int64",
                "uint64",
            ),
            *_cases(
                "test_callbacks_in_selected_branches_keep_general_loop_contract",
                "False",
                "True",
            ),
            "test_dynamic_bounds_remain_correct_general_gpu_loop",
            "test_device_region_reads_strided_inputs_and_parallel_carried_assignments",
        ),
    ),
    "tests/integration/test_metal_contraction_fusion.py": RequiredFile(
        "device",
        (
            *_cases(
                "test_weighted_uvu_dynamic_forward_jvp_and_reverse",
                "batch0-16-dimensions0-1",
                "batch1-8-dimensions1-1",
                "batch2-8-dimensions2-1",
                "batch3-8-dimensions3-1",
                "batch4-8-dimensions4-1",
                "batch5-8-dimensions5-1",
                "batch6-8-dimensions6-1",
                "batch7-8-dimensions7-1",
                "batch8-8-dimensions8-2",
                "batch9-8-dimensions9-1",
            ),
            "test_weighted_uvu_preserves_staged_reduction",
            "test_weighted_uvu_singleton_signed_zero",
        ),
    ),
    "tests/integration/test_metal_native_admission.py": RequiredFile(
        "device",
        (
            *_cases(
                "test_cpu_lapack_custom_call_refused_at_compile",
                *(
                    f"{placement}-{target}"
                    for placement in (
                        "entry",
                        "unselected-cond-branch",
                        "while-nested-switch-branch",
                    )
                    for target in ("mps.eigh", "mps.qr", "mps.svd", "lapack_ssyevd_ffi")
                ),
            ),
            *_cases(
                "test_admitted_gpu_custom_call_executes_in_branches",
                "entry",
                "selected-cond-branch",
            ),
        ),
    ),
    "tests/integration/test_metal_optimized_program.py": RequiredFile(
        "device",
        (
            "test_optimized_program_is_the_inlined_executed_module",
            "test_optimized_program_converts_active_chlo_and_shardy_modules",
            "test_optimized_program_is_identical_before_and_after_execution",
            "test_optimized_program_query_never_runs_host_callbacks",
            "test_optimized_program_concurrent_queries_and_executable_lifetime",
            *_cases(
                "test_optimized_program_publishes_admitted_mhlo_targets",
                "mhlo.sinh",
                "mhlo.erf_inv",
                "mhlo.topk",
            ),
            *_cases(
                "test_optimized_program_namespace_never_admits_unknown_targets",
                "mhlo.not_a_target",
                "mps.mhlo.not_a_target",
                "mps.mhlo.eigh",
            ),
        ),
    ),
}

CASE_PLUGIN = '''"""Per-node outcomes, user properties and module provenance of one Metal file."""

import json
import os
import sys
from pathlib import Path

from _pytest_outcomes import _node_outcome, _NodeCollector


class MetalCaseReport(_NodeCollector):
    def pytest_sessionfinish(self, session, exitstatus):
        checkout = Path(os.environ["PHYDRAX_METAL_CHECKOUT"]).resolve()
        nodes = []
        for nodeid, phases in self.phases.items():
            outcome = _node_outcome(nodeid, phases)
            reports = [phases[when] for when in ("setup", "call", "teardown") if when in phases]
            nodes.append({
                "nodeid": nodeid,
                "outcome": outcome.outcome,
                "message": outcome.message,
                "duration_seconds": outcome.duration_seconds,
                "xfail": any(hasattr(report, "wasxfail") for report in reports),
                "properties": [
                    [str(name), value if isinstance(value, str) else repr(value)]
                    for name, value in reports[-1].user_properties
                ],
            })
        files = {}
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if isinstance(path, str):
                files[name] = Path(path).resolve()
        record = {
            "run": os.environ["PHYDRAX_METAL_QUALIFICATION_RUN"],
            "exit_status": int(exitstatus),
            "collected": list(self.collected),
            "failed_collectors": sorted(set(self.failed_collectors)),
            "nodes": nodes,
            "modules": {
                name: str(files[name])
                for name in ("phydrax", "phydrax_metal", "jax", "jaxlib")
                if name in files
            },
            "checkout_modules": sorted(
                name for name, path in files.items() if path.is_relative_to(checkout)
            ),
            "sys_path": list(sys.path),
        }
        Path(os.environ["PHYDRAX_METAL_CASE_REPORT"]).write_text(
            json.dumps(record, sort_keys=True), encoding="utf-8"
        )


def pytest_configure(config):
    config.pluginmanager.register(MetalCaseReport(), "phydrax-metal-case-report")
'''

HOST_PROBE = """
import json, sys, sysconfig
from importlib import metadata
import phydrax, phydrax_metal
from phydrax import _jax_config
paths = sysconfig.get_paths()
print("PROBE " + json.dumps({
    "versions": {name: metadata.version(name) for name in ("phydrax", "phydrax-metal", "jax", "jaxlib")},
    "modules": {"phydrax": phydrax.__file__, "phydrax_metal": phydrax_metal.__file__},
    "site_packages": sorted({paths["purelib"], paths["platlib"]}),
    "backends_initialized": _jax_config.backends_initialized(),
    "python": list(sys.version_info[:2]),
    "implementation": sys.implementation.name,
    "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
}, sort_keys=True), flush=True)
"""

DEVICE_PROBE = """
import json
import jax
from phydrax import config as pxconfig
from phydrax._jax_config import matmul_precision
from phydrax.backends import metal_environment
report = metal_environment()
facts = report.facts
record = {
    "successful": report.successful,
    "issues": [[issue.code, issue.severity, issue.message] for issue in report.issues],
    "facts": {
        "versions": dict(facts.versions),
        "competing_plugins": list(facts.competing_plugins),
        "selected_platforms": None if facts.selected_platforms is None else list(facts.selected_platforms),
        "build_id": facts.build_id,
        "source_id": facts.source_id,
        "companion_platform": facts.companion_platform,
        "required_jax": facts.required_jax,
        "required_jaxlib": facts.required_jaxlib,
        "minimum_macos": facts.minimum_macos,
        "macos_version": facts.macos_version,
        "device_probed": facts.device_probed,
        "runtime_platform": facts.runtime_platform,
        "runtime_version": facts.runtime_version,
        "device_kind": facts.device_kind,
        "backend_error": facts.backend_error,
    },
}
if report.successful and facts.device_probed:
    from benchmarks._runtime import capture_environment
    record["devices"] = [
        {"platform": device.platform, "platform_version": device.client.platform_version,
         "device_kind": device.device_kind, "device_id": device.id}
        for device in jax.devices()
    ]
    record["x64_enabled"] = bool(pxconfig.x64_enabled())
    record["matmul_precision"] = matmul_precision()
    record["environment"] = capture_environment().to_dict()
print("PROBE " + json.dumps(record, sort_keys=True), flush=True)
"""


class CommandEvidence(TypedDict):
    command: list[str]
    cwd: str
    environment: dict[str, str]
    returncode: int
    timed_out: bool
    stdout: str
    stderr: str


class FileEvidence(TypedDict):
    kind: FileKind
    command: CommandEvidence | None
    report: dict[str, object] | None
    errors: list[str]


class Evidence(TypedDict):
    timestamp: str
    completed_at: str
    run: str
    python: str
    work_dir: str
    wheels: dict[str, str]
    companion: CompanionFacts | None
    host: dict[str, object]
    precision_profile: dict[str, object]
    commands: list[CommandEvidence]
    installed: dict[str, object] | None
    identity: dict[str, object] | None
    expected_runtime: dict[str, object] | None
    files: dict[str, FileEvidence]
    coverage: dict[str, str]
    example: dict[str, object] | None
    errors: list[str]
    successful: bool


PRECISION_PROFILE: dict[str, object] = {
    "x64_enabled": True,
    "matmul_precision": "highest",
    "numerical_dtypes": ["float32", "complex64"],
    "exact_integer_dtypes": ["int64", "uint64"],
    "refused_dtypes": ["float64", "complex128"],
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bounded(text: str) -> str:
    return text if len(text) <= EVIDENCE_TEXT_LIMIT else text[-EVIDENCE_TEXT_LIMIT:]


def clean_environment(venv: Path, kind: FileKind | None) -> dict[str, str]:
    """Allow-listed process environment: no PYTHONPATH, DYLD, JAX, XLA or Phydrax overrides."""
    environment = {
        key: os.environ[key] for key in INHERITED_ENVIRONMENT if key in os.environ
    }
    environment["PATH"] = os.pathsep.join((str(venv / "bin"), *SYSTEM_PATH))
    environment["PYTHONNOUSERSITE"] = "1"
    environment["VIRTUAL_ENV"] = str(venv)
    if kind is not None:
        environment["PHYDRAX_METAL_QUALIFICATION"] = "1"
    if kind == "device":
        environment.update(DEVICE_ENVIRONMENT)
    return environment


def install_environment(venv: Path) -> dict[str, str]:
    """Clean environment plus the caller's PATH and uv index/cache settings."""
    environment = clean_environment(venv, None)
    environment["PATH"] = os.environ.get("PATH", os.pathsep.join(SYSTEM_PATH))
    environment.update(
        (key, value) for key, value in os.environ.items() if key.startswith("UV_")
    )
    return environment


def run(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> CommandEvidence:
    """Run one command in its own session, killing its process group afterwards."""
    with subprocess.Popen(
        list(command),
        cwd=cwd,
        env=None if environment is None else dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        finally:
            # Darwin reports EPERM when only an unreaped zombie remains in the group.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
    # Package-index settings can carry credentials; record their names only.
    recorded = {
        key: "<inherited>" if key.startswith("UV_") else value
        for key, value in (environment or {}).items()
        if key not in INHERITED_ENVIRONMENT
    }
    return {
        "command": list(command),
        "cwd": str(cwd),
        "environment": recorded,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "stdout": _bounded(stdout),
        "stderr": _bounded(stderr),
    }


def probe_record(result: CommandEvidence, /) -> dict[str, object] | None:
    """Return the last ``PROBE`` JSON object a successful probe printed."""
    lines = [line for line in result["stdout"].splitlines() if line.startswith("PROBE ")]
    if result["returncode"] or not lines:
        return None
    try:
        record = json.loads(lines[-1].removeprefix("PROBE "))
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def wheel_record(wheel: Path, /) -> list[tuple[str, str]]:
    """(installed relative path, RECORD digest) for every hashed wheel member."""
    with zipfile.ZipFile(wheel) as archive:
        names = [
            name for name in archive.namelist() if name.endswith(".dist-info/RECORD")
        ]
        if len(names) != 1:
            raise ValueError(f"Expected one RECORD in {wheel.name}, found {names}")
        rows = csv.reader(io.StringIO(archive.read(names[0]).decode("utf-8")))
        entries: list[tuple[str, str]] = []
        for row in rows:
            if len(row) != 3 or not row[1]:
                continue
            first, _, relative = row[0].partition("/")
            if first.endswith(".data"):
                scheme, _, relative = relative.partition("/")
                if scheme not in ("purelib", "platlib"):
                    continue
                entries.append((relative, row[1]))
            else:
                entries.append((row[0], row[1]))
    return entries


def installed_file_errors(wheel: Path, site_packages: Sequence[Path]) -> list[str]:
    """Every hashed wheel file must be installed byte-identically in the venv."""
    errors: list[str] = []
    for relative, digest in wheel_record(wheel):
        candidates = [
            root / relative for root in site_packages if (root / relative).is_file()
        ]
        if not candidates:
            errors.append(f"{wheel.name}: {relative} is not installed")
        elif all(record_digest(path.read_bytes()) != digest for path in candidates):
            errors.append(f"{wheel.name}: installed {relative} differs from the wheel")
    return errors


def host_probe_errors(
    record: Mapping[str, object] | None, *, version: str, jax_version: str, venv: Path
) -> list[str]:
    """Check installed versions, venv-local modules and device-inert root import."""
    if record is None:
        return ["Installed-package probe failed"]
    errors: list[str] = []
    expected = {
        "phydrax": version,
        COMPANION: version,
        "jax": jax_version,
        "jaxlib": jax_version,
    }
    if record.get("versions") != expected:
        errors.append(
            f"Installed versions {record.get('versions')} differ from {expected}"
        )
    modules = record.get("modules")
    if not isinstance(modules, dict) or set(modules) != {"phydrax", COMPANION_PACKAGE}:
        errors.append(f"Installed module paths missing: {modules}")
    else:
        for name, path in modules.items():
            resolved = Path(str(path)).resolve()
            if not resolved.is_relative_to(venv.resolve()) or resolved.is_relative_to(
                ROOT
            ):
                errors.append(f"{name} imported from {resolved}, not the fresh venv")
    if record.get("backends_initialized") is not False:
        errors.append("Importing phydrax and phydrax_metal initialized a JAX backend")
    if record.get("python") != [3, 13]:
        errors.append(f"Venv interpreter is Python {record.get('python')}, not 3.13")
    if (
        record.get("implementation") != "cpython"
        or record.get("free_threaded") is not False
    ):
        errors.append("The Metal runtime requires standard GIL-enabled CPython.")
    return errors


def identity_errors(
    record: Mapping[str, object] | None, *, companion: CompanionFacts, version: str
) -> tuple[list[str], dict[str, object] | None]:
    """Check the probed runtime against the companion wheel's build identity.

    Returns the expected per-case ``metal_runtime`` record when the identity holds.
    """
    if record is None:
        return ["Metal device identity probe failed"], None
    errors: list[str] = []
    facts = record.get("facts")
    if not isinstance(facts, dict):
        return ["Metal identity probe returned no facts"], None
    if record.get("successful") is not True or record.get("issues"):
        errors.append(f"Metal environment report unsuccessful: {record.get('issues')}")
    if facts.get("device_probed") is not True:
        errors.append("Metal environment did not probe a device")
    jax_version = companion["jax_version"]
    platform_version = f"{COMPANION} {version} {companion['source_id']}"
    expected_facts = {
        "versions": {
            "jax": jax_version,
            "jaxlib": jax_version,
            "phydrax": version,
            COMPANION: version,
        },
        "competing_plugins": [],
        "selected_platforms": [RUNTIME_PLATFORM],
        "build_id": companion["build_id"],
        "source_id": companion["source_id"],
        "companion_platform": RUNTIME_PLATFORM,
        "required_jax": jax_version,
        "required_jaxlib": companion["jaxlib_version"],
        "minimum_macos": companion["minimum_macos"],
        "runtime_platform": RUNTIME_PLATFORM,
        "runtime_version": platform_version,
    }
    for key, expected in expected_facts.items():
        if facts.get(key) != expected:
            errors.append(
                f"Runtime fact {key} is {facts.get(key)!r}, expected {expected!r}"
            )
    devices = record.get("devices")
    if (
        not isinstance(devices, list)
        or len(devices) != 1
        or not isinstance(devices[0], dict)
    ):
        return [*errors, f"Expected exactly one visible MPS device: {devices}"], None
    device = devices[0]
    if (
        device.get("platform") != RUNTIME_PLATFORM
        or device.get("platform_version") != platform_version
    ):
        errors.append(
            f"Device {device} is not the installed {platform_version!r} runtime"
        )
    if not isinstance(device.get("device_kind"), str) or device.get(
        "device_kind"
    ) != facts.get("device_kind"):
        errors.append("Device kind disagrees with the Metal environment report")
    if record.get("x64_enabled") is not True:
        errors.append("Metal qualification process does not have x64 enabled")
    if record.get("matmul_precision") != PRECISION_PROFILE["matmul_precision"]:
        errors.append(
            f"Matmul precision {record.get('matmul_precision')!r} is not highest"
        )
    environment = record.get("environment")
    jax_record = environment.get("jax") if isinstance(environment, dict) else None
    if (
        not isinstance(environment, dict)
        or not isinstance(jax_record, dict)
        or jax_record.get("backend") != RUNTIME_PLATFORM
        or jax_record.get("x64_enabled") is not True
        or jax_record.get("version") != jax_version
        or environment.get("jaxlib_version") != companion["jaxlib_version"]
        or environment.get("phydrax_version") != version
        or [entry.get("platform") for entry in jax_record.get("devices", [])]
        != [RUNTIME_PLATFORM]
    ):
        errors.append(
            "Benchmark runtime environment does not describe the selected MPS runtime"
        )
    if errors:
        return errors, None
    return [], {
        "platform": RUNTIME_PLATFORM,
        "platform_version": platform_version,
        "device_kind": device["device_kind"],
        "device_id": device["device_id"],
        "x64_enabled": True,
    }


def file_errors(
    path: str,
    required: RequiredFile,
    report: Mapping[str, object] | None,
    *,
    returncode: int,
    run_id: str,
    expected_runtime: Mapping[str, object] | None,
    venv: Path,
) -> tuple[list[str], dict[str, str]]:
    """Reduce one file process to errors and its observed per-node outcomes."""
    errors: list[str] = []
    coverage: dict[str, str] = {}
    if returncode:
        errors.append(f"{path}: pytest exited with status {returncode}")
    if report is None:
        return [*errors, f"{path}: no case report was produced"], coverage
    if report.get("run") != run_id:
        return [
            *errors,
            f"{path}: case report belongs to another run (stale evidence)",
        ], coverage
    if report.get("exit_status") != 0:
        errors.append(f"{path}: pytest session status {report.get('exit_status')}")
    if report.get("failed_collectors"):
        errors.append(f"{path}: collection failed for {report.get('failed_collectors')}")
    expected = set(required.nodeids(path))
    collected_nodes = report.get("collected")
    collected = (
        [str(nodeid) for nodeid in collected_nodes]
        if isinstance(collected_nodes, list)
        else []
    )
    for nodeid, count in sorted(Counter(collected).items()):
        if count > 1:
            errors.append(f"{path}: node collected {count} times: {nodeid}")
    for nodeid in sorted(set(collected) - expected):
        errors.append(f"{path}: unexpected node outside the required inventory: {nodeid}")
    nodes = report.get("nodes")
    if not isinstance(nodes, list):
        return [*errors, f"{path}: case report has no node list"], coverage
    observed = Counter(
        str(node.get("nodeid")) for node in nodes if isinstance(node, dict)
    )
    for node in nodes:
        if not isinstance(node, dict) or str(node.get("nodeid")) not in expected:
            continue
        nodeid = str(node["nodeid"])
        if observed[nodeid] != 1:
            continue
        coverage[nodeid] = str(node.get("outcome"))
        if node.get("outcome") != "passed":
            errors.append(f"{nodeid} {node.get('outcome')}: {node.get('message')}")
            continue
        if node.get("xfail"):
            errors.append(f"{nodeid} is marked as an expected failure")
            coverage[nodeid] = "expected-failure"
            continue
        property_name = required.runtime_property(nodeid.removeprefix(f"{path}::"))
        if property_name is None:
            continue
        properties = [
            value for name, value in node.get("properties", []) if name == property_name
        ]
        try:
            runtime = json.loads(properties[0]) if len(properties) == 1 else None
        except (TypeError, json.JSONDecodeError):
            runtime = None
        if expected_runtime is None or runtime != dict(expected_runtime):
            errors.append(
                f"{nodeid} lacks matching {property_name} evidence: "
                f"expected {dict(expected_runtime or {})}, got {properties}"
            )
            coverage[nodeid] = "unverified-runtime"
    for nodeid in sorted(expected):
        if observed[nodeid] == 0:
            errors.append(f"Required node missing: {nodeid}")
            coverage[nodeid] = "missing"
        elif observed[nodeid] > 1:
            errors.append(f"Required node reported {observed[nodeid]} times: {nodeid}")
            coverage[nodeid] = "duplicated"
    for nodeid in sorted(set(observed) - expected):
        errors.append(f"{path}: unexpected node outside the required inventory: {nodeid}")
    modules = report.get("modules")
    if not isinstance(modules, dict) or (
        required.kind == "device" and "phydrax" not in modules
    ):
        errors.append(f"{path}: installed phydrax module provenance is missing")
    else:
        for name in ("phydrax", "phydrax_metal"):
            if name in modules and not Path(str(modules[name])).is_relative_to(
                venv.resolve()
            ):
                errors.append(f"{path}: {name} loaded from {modules[name]}, not the venv")
    if report.get("checkout_modules"):
        errors.append(
            f"{path}: modules loaded from the checkout: {report.get('checkout_modules')}"
        )
    return list(dict.fromkeys(errors)), coverage


def example_errors(
    result: CommandEvidence, expected_runtime: Mapping[str, object]
) -> tuple[list[str], dict[str, object] | None]:
    """The installed example must exit zero and print its actual run as JSON last."""
    errors: list[str] = []
    if result["returncode"] or result["timed_out"]:
        errors.append(f"Example exited with status {result['returncode']}")
    lines = result["stdout"].strip().splitlines()
    try:
        output = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        output = None
    if not isinstance(output, dict):
        return [*errors, "Example did not print a JSON object as its last line"], None
    device = output.get("device")
    expected = {
        key: expected_runtime[key]
        for key in ("platform", "platform_version", "device_kind")
    }
    observed = (
        {key: device.get(key) for key in expected} if isinstance(device, dict) else None
    )
    if observed != expected:
        errors.append(f"Example ran on {device}, expected {expected}")
    if output.get("x64_enabled") is not True:
        errors.append("Example did not run with x64 enabled")
    if output.get("status") != "passed":
        errors.append(f"Example reported status {output.get('status')!r}")
    return errors, output


def host_facts() -> dict[str, object]:
    """Exact macOS and Apple Silicon identity of the qualification host."""

    def text(command: Sequence[str]) -> str | None:
        result = subprocess.run(
            list(command), capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    facts: dict[str, object] = {
        "platform": sys.platform,
        "machine": platform.machine(),
        "macos_version": text(("sw_vers", "-productVersion")),
        "macos_build": text(("sw_vers", "-buildVersion")),
        "model": text(("sysctl", "-n", "hw.model")),
        "chip": text(("sysctl", "-n", "machdep.cpu.brand_string")),
        "memory_bytes": text(("sysctl", "-n", "hw.memsize")),
        "logical_cpus": text(("sysctl", "-n", "hw.ncpu")),
        "gpu": None,
    }
    displays = text(("system_profiler", "SPDisplaysDataType", "-json"))
    with contextlib.suppress(TypeError, json.JSONDecodeError, KeyError, IndexError):
        gpu = json.loads(displays or "")["SPDisplaysDataType"][0]
        facts["gpu"] = {"model": gpu.get("sppci_model"), "cores": gpu.get("sppci_cores")}
    return facts


def host_errors(facts: Mapping[str, object], companion: CompanionFacts) -> list[str]:
    errors: list[str] = []
    if (facts.get("platform"), facts.get("machine")) != ("darwin", "arm64"):
        errors.append(f"Qualification host is not Apple Silicon macOS: {facts}")
        return errors
    version = str(facts.get("macos_version") or "")

    def release(value: str) -> tuple[int, ...]:
        return tuple(int(part) for part in value.split(".") if part.isdecimal())

    if not version or release(version) < release(companion["minimum_macos"]):
        errors.append(
            f"macOS {version!r} is below the build minimum {companion['minimum_macos']}"
        )
    gpu = facts.get("gpu")
    if not isinstance(gpu, dict) or not gpu.get("model") or not gpu.get("cores"):
        errors.append("Apple GPU model and core count could not be identified")
    return errors


def write_case_plugin(directory: Path) -> None:
    """Place the per-node outcome plugin and its shared reduction next to a suite."""
    shutil.copy2(
        ROOT / "tools" / "_pytest_outcomes.py", directory / "_pytest_outcomes.py"
    )
    (directory / "phydrax_metal_case_report.py").write_text(CASE_PLUGIN, encoding="utf-8")


def stage_suite(directory: Path) -> None:
    """Copy only the tests, pytest configuration and outcome plugin to a scratch root."""
    shutil.copytree(
        ROOT / "tests",
        directory / "tests",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(ROOT / "pyproject.toml", directory / "pyproject.toml")
    write_case_plugin(directory)


def pytest_command(python: Path, suite: Path, path: str) -> list[str]:
    # macOS /tmp is a symlink; pytest's root must match its physical cwd.
    suite = suite.resolve()
    return [
        str(python),
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-p",
        "phydrax_metal_case_report",
        "-q",
        "-rA",
        "-n",
        "0",
        "-c",
        str(suite / "pyproject.toml"),
        "--rootdir",
        str(suite),
        path,
    ]


def run_file(
    python: Path,
    venv: Path,
    suite: Path,
    path: str,
    required: RequiredFile,
    *,
    run_id: str,
    expected_runtime: Mapping[str, object] | None,
    timeout: float,
) -> tuple[FileEvidence, dict[str, str]]:
    report_path = suite / CASE_REPORT
    report_path.unlink(missing_ok=True)
    environment = clean_environment(venv, required.kind)
    environment.update(
        PHYDRAX_METAL_QUALIFICATION_RUN=run_id,
        PHYDRAX_METAL_CASE_REPORT=str(report_path),
        PHYDRAX_METAL_CHECKOUT=str(ROOT),
    )
    result = run(
        pytest_command(python, suite, path),
        cwd=suite,
        environment=environment,
        timeout=timeout,
    )
    report: dict[str, object] | None = None
    with contextlib.suppress(OSError, json.JSONDecodeError):
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
        report = loaded if isinstance(loaded, dict) else None
    status = -1 if result["timed_out"] else result["returncode"]
    errors, coverage = file_errors(
        path,
        required,
        report,
        returncode=status,
        run_id=run_id,
        expected_runtime=expected_runtime,
        venv=venv,
    )
    return {
        "kind": required.kind,
        "command": result,
        "report": report,
        "errors": errors,
    }, coverage


def _write(path: Path, evidence: Evidence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _static_errors(
    core_wheel: Path, companion_wheel: Path
) -> tuple[list[str], str | None, CompanionFacts | None]:
    metadata = wheel_metadata(core_wheel)
    if isinstance(metadata, str):
        return [metadata], None, None
    version = str(metadata["Version"])
    errors = core_metadata_errors(metadata)
    found, companion = companion_errors(
        companion_wheel, core_version=version, core_metadata=metadata
    )
    return [*errors, *found], version, companion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--python", type=Path, required=True, help="Interpreter for the fresh venv"
    )
    parser.add_argument("--phydrax-wheel", type=Path, required=True)
    parser.add_argument("--companion-wheel", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Absent directory outside the checkout",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON evidence output")
    parser.add_argument(
        "--file-timeout",
        type=float,
        default=DEFAULT_FILE_TIMEOUT,
        help="Seconds before one test-file or example process is killed and failed",
    )
    args = parser.parse_args()
    if not args.file_timeout > 0:
        parser.error("--file-timeout must be positive")
    python: Path = args.python.absolute()
    core_wheel: Path = args.phydrax_wheel.resolve()
    companion_wheel: Path = args.companion_wheel.resolve()
    work_dir: Path = args.work_dir.resolve()
    output: Path = args.output.resolve()
    run_id = uuid.uuid4().hex
    evidence: Evidence = {
        "timestamp": datetime.now(UTC).isoformat(),
        "completed_at": "",
        "run": run_id,
        "python": str(python),
        "work_dir": str(work_dir),
        "wheels": {},
        "companion": None,
        "host": host_facts(),
        "precision_profile": PRECISION_PROFILE,
        "commands": [],
        "installed": None,
        "identity": None,
        "expected_runtime": None,
        "files": {},
        "coverage": {},
        "example": None,
        "errors": [],
        "successful": False,
    }
    errors = evidence["errors"]
    for wheel in (core_wheel, companion_wheel):
        if not wheel.is_file():
            errors.append(f"Wheel not found: {wheel}")
    if work_dir.exists():
        errors.append(
            f"Work directory already exists; qualification needs a fresh one: {work_dir}"
        )
    elif work_dir.resolve().is_relative_to(ROOT):
        errors.append(f"Work directory must be outside the checkout: {work_dir}")
    version = companion = None
    if not errors:
        evidence["wheels"] = {
            path.name: sha256(path) for path in (core_wheel, companion_wheel)
        }
        static, version, companion = _static_errors(core_wheel, companion_wheel)
        errors.extend(static)
        evidence["companion"] = companion
        if companion is not None:
            errors.extend(host_errors(evidence["host"], companion))
            try:
                expected_source, _ = source_identity(read_lock())
            except (BuildFailure, OSError, ValueError) as error:
                errors.append(f"Cannot bind the current native source closure: {error}")
            else:
                if companion["source_id"] != expected_source:
                    errors.append(
                        f"Companion source {companion['source_id']} differs from "
                        f"the current approved source {expected_source}."
                    )
    venv = work_dir / "venv"
    venv_python = venv / "bin" / "python"
    if not errors and version is not None and companion is not None:
        work_dir.mkdir(parents=True)
        wheelhouse = work_dir / "wheelhouse"
        wheelhouse.mkdir()
        shutil.copy2(companion_wheel, wheelhouse / companion_wheel.name)
        steps = (
            [str(python), "-m", "venv", str(venv)],
            [
                "uv",
                "pip",
                "install",
                "--no-config",
                "--no-build",
                "--python",
                str(venv_python),
                "--find-links",
                str(wheelhouse),
                f"phydrax[metal] @ {core_wheel.as_uri()}",
                *test_requirements(core_wheel),
            ],
        )
        for step in steps:
            result = run(step, cwd=work_dir, environment=install_environment(venv))
            evidence["commands"].append(result)
            if result["returncode"]:
                errors.append(f"Command failed: {' '.join(step[:4])}")
                break
    if not errors and version is not None and companion is not None:
        probe_dir = work_dir / "probe"
        (probe_dir / "benchmarks").mkdir(parents=True)
        shutil.copy2(
            ROOT / "benchmarks" / "_runtime.py", probe_dir / "benchmarks" / "_runtime.py"
        )
        host = run(
            [str(venv_python), "-c", HOST_PROBE],
            cwd=probe_dir,
            environment=clean_environment(venv, None),
        )
        evidence["commands"].append(host)
        installed = probe_record(host)
        evidence["installed"] = installed
        errors.extend(
            host_probe_errors(
                installed,
                version=version,
                jax_version=companion["jax_version"],
                venv=venv,
            )
        )
        site_packages = None if installed is None else installed.get("site_packages")
        if isinstance(site_packages, list):
            roots = [Path(str(path)) for path in site_packages]
            for wheel in (core_wheel, companion_wheel):
                errors.extend(installed_file_errors(wheel, roots))
        if not errors:
            device = run(
                [str(venv_python), "-c", DEVICE_PROBE],
                cwd=probe_dir,
                environment=clean_environment(venv, "device"),
                timeout=args.file_timeout,
            )
            evidence["commands"].append(device)
            evidence["identity"] = probe_record(device)
            found, expected_runtime = identity_errors(
                evidence["identity"], companion=companion, version=version
            )
            errors.extend(found)
            evidence["expected_runtime"] = expected_runtime
    expected_runtime = evidence["expected_runtime"]
    if not errors and expected_runtime is not None:
        suite = work_dir / "suite"
        suite.mkdir()
        stage_suite(suite)
        for path, required in REQUIRED_FILES.items():
            record, coverage = run_file(
                venv_python,
                venv,
                suite,
                path,
                required,
                run_id=run_id,
                expected_runtime=expected_runtime,
                timeout=args.file_timeout,
            )
            evidence["files"][path] = record
            evidence["coverage"].update(coverage)
            errors.extend(record["errors"])
            print(f"{path}: {'PASS' if not record['errors'] else 'FAIL'}", flush=True)
        if not (ROOT / EXAMPLE).is_file():
            errors.append(f"Required example is absent: {EXAMPLE}")
        else:
            example_dir = work_dir / "example"
            example_dir.mkdir()
            shutil.copy2(ROOT / EXAMPLE, example_dir / EXAMPLE.name)
            result = run(
                [str(venv_python), EXAMPLE.name],
                cwd=example_dir,
                environment=clean_environment(venv, "device"),
                timeout=args.file_timeout,
            )
            found, example_output = example_errors(result, expected_runtime)
            evidence["example"] = {
                "command": result,
                "output": example_output,
                "errors": found,
            }
            errors.extend(found)
    else:
        for path in (*REQUIRED_FILES, EXAMPLE.as_posix()):
            errors.append(
                f"{path}: not run because installation or runtime identity failed"
            )
    for path in (core_wheel, companion_wheel):
        if path.name in evidence["wheels"] and (
            not path.is_file() or sha256(path) != evidence["wheels"][path.name]
        ):
            errors.append(f"Wheel changed during qualification: {path}")
    evidence["successful"] = (
        not errors
        and set(evidence["coverage"])
        == {
            nodeid
            for path, required in REQUIRED_FILES.items()
            for nodeid in required.nodeids(path)
        }
        and all(outcome == "passed" for outcome in evidence["coverage"].values())
    )
    evidence["completed_at"] = datetime.now(UTC).isoformat()
    _write(output, evidence)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Evidence: {output}")
    return 0 if evidence["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
