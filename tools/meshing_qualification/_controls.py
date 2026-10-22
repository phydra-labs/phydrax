# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import argparse
from collections.abc import Mapping
from math import isfinite
from operator import attrgetter
from pathlib import Path
from typing import Any, final, get_args, Literal, NotRequired, overload, TypedDict

import equinox as eqx

import phydrax as phx
from phydrax import StrictModule
from phydrax.meshing import BisectionCompatibility
from phydrax.typing import parse
from tools._meshing_cases import POLYHEDRAL_CASE_NAMES


type CadCase = Literal[
    "planar-box",
    "quadric-routine",
    "quadric-larger",
    "external-routine",
    "external-combined",
]
type ImplicitCase = Literal[
    "sphere", "sphere-refinement", "offset-ring-torus-anisotropic"
]
type ImageCase = Literal["two-slab", "oblique-junction-transport"]
type DesignCase = Literal["routine-sphere", "larger-sphere", "combined-design-target"]
type OversetProfile = Literal["routine", "larger", "combined"]
type ReconstructionCase = Literal["lobed-sphere", "torus", "rotated-ellipsoid"]

CAD_CASES = get_args(CadCase.__value__)
IMPLICIT_CASES = get_args(ImplicitCase.__value__)
IMAGE_CASES = get_args(ImageCase.__value__)
DESIGN_CASES = get_args(DesignCase.__value__)
OVERSET_PROFILES = get_args(OversetProfile.__value__)
RECONSTRUCTION_CASES = get_args(ReconstructionCase.__value__)


SURFACE_CASES = (
    "sphere",
    "torus",
    "capped_cylinder",
    "annulus",
    "capped_cylinder_features",
    "cylinder_sheet",
    "folded_plates",
)


FAMILY_CASES = (
    "dual-quad",
    "curved-quarter-extrusion-quad",
    "dual-hex",
    "hex-dominant",
    "transfinite-quad",
    "multiblock-quad",
    "sweep-hex",
    "sweep-prism",
    "curved-frame-sweep-hex",
    "multiblock-reversed-hex",
    "multiblock-material-hex",
    "multiblock-cycle-quad",
    "curved-nonsweepable-hex",
    "curved-cavity-hex",
    "curved-material-hex",
    "nonsweepable-dual-hex",
)


HYBRID_PROFILES = (
    "unit-box",
    "curved-periodic-layer-core",
    "original-curved-periodic-refusal",
    "archived-layer-core",
    "concave-box-core-lifecycle",
    "plate5",
    "quadrilateral_plate3",
    "opposing_sloped_plates4",
    "sphere_radius04_subdivision2",
    "outward_box_count3",
    "concave_box_count3",
    "wedge30_rejection",
    "open_box_floor_count4",
    "channel_gap005_rejection",
    "channel_gap005_policies",
    "cad-slab-and-core",
    "cad-entire-solid-slab",
    "cad-outside-controlled-solid",
    "cad-overlapping-wall-slabs",
    "periodic-translation-triangle",
    "periodic-translation-quadrilateral",
    "periodic-partial-triangle",
    "periodic-partial-quadrilateral",
    "periodic-rotation-triangle",
    "periodic-rotation-quadrilateral",
    "periodic-convex-ridge",
    "exact-sweep-nonperiodic-accepted",
    "exact-sweep-nonperiodic-perturbed-cap",
    "exact-sweep-periodic-accepted",
    "exact-sweep-periodic-perturbed-cap",
)


@final
class Scenario(StrictModule):
    """One CLI route, optional mandatory workflow, and benchmark controls."""

    name: str = eqx.field(static=True)
    module: str = eqx.field(static=True)
    workflow: str | None = eqx.field(static=True, default=None)
    driver_arguments: tuple[str, ...] | None = eqx.field(static=True, default=None)
    driver_position: int = eqx.field(static=True, default=-1)
    corpus_position: int = eqx.field(static=True, default=-1)


SCENARIOS = (
    Scenario(
        "native-image-lifecycle",
        "scenarios.image",
        "labeled-image",
        (
            "image_case",
            "image_size",
            "transport_steps",
            "transport_dt",
            "balance_tolerance",
            "oracle_capacity",
        ),
        driver_position=0,
        corpus_position=1,
    ),
    Scenario(
        "native-periodic-lifecycle",
        "scenarios.periodic",
        "periodic-system",
        ("periodic_case",),
        driver_position=3,
        corpus_position=4,
    ),
    Scenario(
        "native-design-lifecycle",
        "scenarios.design",
        "design-learned-loop",
        ("maximum_memory_bytes", "maximum_condition", "design_case"),
        driver_position=4,
        corpus_position=9,
    ),
    Scenario(
        "native-family-lifecycle",
        "scenarios.family",
        "general-quads-hexes",
        ("family_case",),
        driver_position=5,
        corpus_position=6,
    ),
    Scenario(
        "native-hybrid-lifecycle",
        "scenarios.hybrid",
        "hybrid-layers",
        (
            "layer_count",
            "first_thickness",
            "layer_growth",
            "geometry_order",
            "hybrid_profile",
            "hybrid_archive",
            "hybrid_content_id",
            "hybrid_core_size",
            "hybrid_region_densities",
            "hybrid_material_site",
        ),
        driver_position=7,
        corpus_position=5,
    ),
    Scenario(
        "native-implicit-lifecycle",
        "scenarios.implicit",
        "implicit-domain",
        (
            "implicit_case",
            "maximum_memory_bytes",
            "implicit_radius",
            "implicit_fidelity",
            "implicit_maximum_level",
            "implicit_edge_isolation_depth",
            "implicit_root_tolerance",
            "implicit_refinement_rounds",
            "implicit_improvement_passes",
            "implicit_metric_passes",
        ),
        driver_position=8,
        corpus_position=0,
    ),
    Scenario(
        "native-distributed-lifecycle",
        "scenarios.distributed",
        "distributed-lifecycle",
        (
            "distributed_parts",
            "restart_parts",
            "distributed_processes",
            "distributed_source",
            "distributed_halo_width",
            "distributed_halo_capacity",
            "distributed_cavity_capacity",
            "distributed_slack",
            "distributed_checkpoint_root",
            "distributed_host_packet_bytes",
            "distributed_bisection_compatibility",
        ),
        driver_position=9,
        corpus_position=7,
    ),
    Scenario(
        "native-cad-lifecycle",
        "scenarios.cad",
        "native-cad",
        ("cad_format", "cad_case"),
        driver_position=1,
        corpus_position=3,
    ),
    Scenario(
        "native-envelope-lifecycle",
        "scenarios.envelope",
        "imperfect-surface",
        (),
        driver_position=2,
        corpus_position=2,
    ),
    Scenario(
        "native-reconstruction-lifecycle",
        "scenarios.reconstruction",
        "imperfect-surface",
        ("reconstruction_case",),
        driver_position=10,
        corpus_position=10,
    ),
    Scenario(
        "native-overset-lifecycle",
        "scenarios.overset",
        "moving-overset",
        (
            "overset_profile",
            "overset_motion_steps",
            "overset_wall_pairs",
            "overset_donor_pairs",
            "overset_overlap_pairs",
        ),
        driver_position=10,
        corpus_position=8,
    ),
    Scenario("native-polyhedral-lifecycle", "scenarios.polyhedral", "polyhedral-system"),
    Scenario(
        "native-surface-pde",
        "scenarios.surface",
        "planar-features",
        ("surface_case",),
        driver_position=6,
        corpus_position=10,
    ),
    Scenario("native-lifecycle", "scenarios.native_lifecycle"),
    Scenario("comparison-triangle-lifecycle", "scenarios.native_lifecycle"),
    Scenario("native-corpus", "scenarios.native_corpus"),
    Scenario("manifold", "scenarios.providers"),
    Scenario("mmg", "scenarios.providers"),
    Scenario("ftetwild", "scenarios.providers"),
    Scenario("gmsh", "scenarios.gmsh"),
    Scenario("vorocrust", "scenarios.providers"),
    Scenario("poisson", "scenarios.providers"),
    Scenario("cad-partition", "scenarios.cad_partition"),
    Scenario("composite-heat-3d", "scenarios.composite_heat_3d"),
    Scenario("planar-semantic", "scenarios.planar_semantic"),
    Scenario("composite-heat-2d", "scenarios.composite_heat_2d"),
    Scenario("layout-stack", "scenarios.layout_stack"),
    Scenario("layout-observable", "scenarios.layout_observable"),
    Scenario("planar-bands", "scenarios.planar_bands"),
    Scenario("hybrid-layer", "scenarios.hybrid_layer"),
    Scenario("hybrid-diffusion", "scenarios.hybrid_diffusion"),
    Scenario("distribution", "scenarios.distribution"),
    Scenario("forest-amr", "scenarios.forest_amr"),
    Scenario("omega-h", "scenarios.omega_h"),
    Scenario("omega-h-distributed", "scenarios.omega_h"),
    Scenario("metric-adaptation", "scenarios.metric_adaptation"),
    Scenario("gmsh-metric", "scenarios.gmsh_metric"),
    Scenario("boundary-layer", "scenarios.boundary_layer"),
    Scenario("gmsh-boundary-layer", "scenarios.gmsh_boundary_layer"),
    Scenario("supermesh", "scenarios.supermesh"),
    Scenario("remap-p1", "scenarios.remap_p1"),
    Scenario("bisection", "scenarios.bisection"),
    Scenario("device-adaptation", "scenarios.device_adaptation"),
    Scenario("cad-curving", "scenarios.cad_curving"),
    Scenario("native-quad-hex", "scenarios.native_quad_hex_meshing"),
    Scenario("curved-native-transfer", "scenarios.curved_native_transfer"),
    Scenario("predicates", "scenarios.predicates"),
    Scenario("triangulation", "scenarios.triangulation"),
)

SCENARIO_MODULES = {scenario.name: scenario.module for scenario in SCENARIOS}
SCENARIO_WORKFLOWS = {
    scenario.name: scenario.workflow
    for scenario in SCENARIOS
    if scenario.workflow is not None
}
_NATIVE_DRIVER_ARGUMENTS = {
    scenario.name: scenario.driver_arguments
    for scenario in sorted(SCENARIOS, key=attrgetter("driver_position"))
    if scenario.driver_arguments is not None
}
CORPUS_SCENARIOS = tuple(
    scenario
    for scenario in sorted(SCENARIOS, key=attrgetter("corpus_position"))
    if scenario.corpus_position >= 0 and scenario.workflow is not None
)


def _hybrid_material_controls(
    profile: str,
    densities: tuple[float, ...] | list[float] | None,
    site: str | None,
    /,
) -> tuple[tuple[float, ...], str]:
    """Resolve omitted CLI material controls before freezing the request.

    Explicit material values remain the caller's physics; incompatible
    cardinality refuses rather than projecting them into another declaration.
    """
    single_fluid = profile in ("unit-box", "concave-box-core-lifecycle")
    values = ((2.0,) if single_fluid else (2.0, 3.0)) if densities is None else tuple(densities)
    material_site = (
        ("fluid-density" if single_fluid else "two-material-density")
        if site is None
        else site
    )
    count = 1 if single_fluid else 2
    if profile in (
        "unit-box",
        "concave-box-core-lifecycle",
        "curved-periodic-layer-core",
        "archived-layer-core",
    ) and len(values) != count:
        raise ValueError(
            f"qualification.hybrid_region_densities must contain {count} source-region values for {profile!r}; got {values!r}"
        )
    return values, material_site


def authored_resource_envelope(
    scenario: str | None,
    case_name: str,
    controls: Mapping[str, object],
    *,
    larger: bool,
) -> dict[str, Any] | None:
    """Declare finite measured envelopes only for the registered workflow families."""
    limits: dict[str, int | float]
    if scenario == "native-surface-pde" or case_name == "planar-hole-feature":
        limits = {
            "maximum_work_units": 4_000_000,
            "maximum_geometry_queries": 4_000_000,
            "maximum_scratch_bytes": 1_024_000_000,
            "timeout": 300.0,
        }
        rationale = (
            "Capped planar at 2000000 work, 4000000 queries, 512000000 scratch and "
            "300 seconds progressed through scientific carrier fixes into adaptation, "
            "then exhausted 1999999 work with next 2. Queries were 2047558, scratch "
            "359 MB and elapsed time 133 seconds; transfer/cold had not started. "
            "Finite work 4000000 doubled the prior work cap. At work/query caps "
            "4000000 and scratch 512000000, adaptation certification reached "
            "source_affine_chain, requesting 57886604 scratch bytes with 57885430 "
            "remaining, a 1174-byte deficit. Work was 2.81M, queries 2.06M and "
            "elapsed time 146 seconds; later lifecycle had not started. Finite "
            "scratch 1024000000 doubles the prior cap; other caps and science are unchanged."
        )
    elif case_name in POLYHEDRAL_CASE_NAMES:
        limits = {
            "maximum_work_units": 2_000_000,
            "maximum_scratch_bytes": 256_000_000,
            "timeout": 300.0,
        }
        rationale = (
            "Material-L routine at 640000 work, 1280000 queries, 81920000 scratch "
            "and 120 seconds passed source FE/VEM/FV and regeneration, then common-"
            "refinement/conservative remap received native CAPACITY_EXCEEDED. "
            "Target continuation/archive/cold had not started; the exact limiting "
            "axis is not established by that status alone. Finite work 2000000, "
            "scratch 256000000 and whole timeout 300 seconds author diagnostic "
            "headroom for all six polyhedral profiles. Geometry queries retain "
            "their prior capacity-scaled limit (1280000 at capacity 20000); "
            "scientific, entity and data controls are unchanged. This is not "
            "evidence of successful completion."
        )
    elif scenario == "native-periodic-lifecycle":
        periodic_case = controls.get("periodic_case")
        if not isinstance(periodic_case, str):
            return None
        contract = _periodic_qualification_profile_contracts()[periodic_case]
        if contract["expected_admission"] != "positive":
            return None
        limits = {"maximum_scratch_bytes": 256_000_000, "timeout": 300.0}
        rationale = (
            "Retained periodic work evidence 639987 of 640000 predates the current "
            "2048-times-capacity work multiplier and is stale for work-cap policy. "
            "The default work remains 40960000 at capacity 20000 unless explicitly "
            "overridden. The baseline positive exact-algebraic periodic envelope has "
            "scratch 256000000 and whole timeout 300 seconds, above the prior "
            "81920000/120 defaults. Entity/data/scientific controls and "
            "historical binary64 refusal controls are unchanged. This is not "
            "evidence of successful completion."
        )
        if contract["base_case"] == "skew-material-feature":
            limits["maximum_work_units"] = 320_000_000
            limits["maximum_geometry_queries"] = 5_120_000
            limits["maximum_scratch_bytes"] = 512_000_000
            limits["timeout"] = 1200.0
            limits["maximum_recipe_bytes"] = 2_097_152
            rationale += (
                " Material-feature routine at work 160000000 and queries 2560000 "
                "reached its target after three refinements (2480 cells, L2 0.0181), "
                "then coarsen exhausted 159997437 work with next 16384. Queries "
                "were 1.93M; storage/time had headroom, but archive/cold had not "
                "started and lifecycle time was 440 seconds. Only material-feature "
                "routine/larger now author finite work 320000000, queries 5120000 "
                "and whole timeout 1200 seconds. Scratch 512000000 and science "
                "are unchanged; exact lattice profiles are untouched. The scratch "
                "cap retains headroom over the earlier 307950571-byte geometry-"
                "association requirement. Target attainment alone is not evidence "
                "of completed coarsen/archive/cold workflow."
                " At the numerical envelope target and coarsen completed, then "
                "archive recipe 1099585 bytes exceeded the authored 1048576-byte "
                "metadata ceiling by 51009; cold had not started. Only material-"
                "feature routine/larger author finite maximum_recipe_bytes 2097152, "
                "mapped to the existing archive max_manifest_bytes admission owner. "
                "Lattice retains 1048576; numerical and scientific caps are unchanged."
            )
        else:
            rationale += (
                " No current work failure is observed for exact lattice profiles; "
                "their 2048-times-capacity work allowance remains unchanged."
            )
    elif scenario == "native-overset-lifecycle":
        extended = controls.get("overset_profile") in ("larger", "combined")
        limits = {
            "maximum_work_units": 2_000_000 if extended else 1_280_000,
            "maximum_geometry_queries": 2_560_000,
            "maximum_scratch_bytes": 128_000_000,
            "timeout": 180.0,
        }
        rationale = (
            "Moving overset mandatory workflows exhausted the prior finite work/query envelope. "
            "Cold common-refinement polynomial preparation required 93128801 storage "
            "bytes before expansion against 81920000; 128000000 authored scratch bytes "
            "provides 37.4% headroom without changing queries, cavity, or deadline."
        )
        if extended:
            rationale += (
                " Larger warm execution and both archives passed at 1280000 work; "
                "fresh cold FE restore completed, then conservative archive "
                "MeshingGeometryMotionTransition.from_stored_fields to "
                "stored_record_content_id exhausted work before remap. Larger and "
                "combined author 2000000 work; the positive routine remains at 1280000."
            )
    elif scenario == "native-distributed-lifecycle":
        limits = {
            "maximum_work_units": 32_000_000 if larger else 8_000_000,
            "timeout": 600.0,
            "maximum_memory_bytes": (32 if larger else 24) * 1024**3,
        }
        rationale = (
            "Distributed routine at work 8000000, 24 GiB and 300 seconds completed "
            "4-to-2 restore, all 16 roles and continued FE/FV/Euler/RNG/allfield "
            "acceptance, then rearchive binding failed. Producer time was 148.9 "
            "seconds, consumer time 275.4 seconds and scenario time 425 seconds; "
            "memory was healthy. Finite whole timeout 600 seconds provides remaining "
            "workflow headroom; routine work 8000000, routine/larger memory 24/32 GiB and "
            "science are unchanged. This does not resolve the binding failure or "
            "establish successful workflow completion."
        )
        if larger:
            rationale += (
                " Current larger 6-to-3 at work 8000000, 600 seconds and 32 GiB "
                "reached initial FE/FV and first refine, then all ranks exhausted "
                "about 7.99M work with next 6144. Storage/query/cavity/time/memory "
                "were healthy; checkpoint/cold had not started. Only larger "
                "authors finite work 32000000, four times the prior cap for its "
                "two-times resolution/six-rank topology and remaining lifecycle. "
                "Routine remains at 8000000; all other caps and science are unchanged."
            )
    elif scenario == "native-family-lifecycle":
        limits = {
            "maximum_work_units": 128_000_000,
            "maximum_geometry_queries": 32_000_000,
            "maximum_scratch_bytes": 1_536_000_000,
            "maximum_data_bytes": 40_960_000,
            "maximum_memory_bytes": 32 * 1024**3,
            "timeout": 2400.0,
        }
        rationale = (
            "Family durable 2400-second run at MemoryMax=16G completed publication, "
            "independent acceptance, association and all-field preparation, then "
            "the external cgroup recorded oom-kill at finite_volume_owner_preparation_epoch_0 "
            "after 1442 seconds. Work/query/deadline had headroom. Finite execution-group/"
            "process memory 32 GiB (34359738368 bytes) doubles the prior cap; native "
            "scratch 1536000000, retained data 40960000, work/query/deadline and "
            "science are unchanged. Durable incremental stage and failure evidence "
            "remains required before any rerun. This envelope is not evidence of "
            "successful completion; RSS enforcement belongs to the external cgroup."
        )
    elif scenario == "native-design-lifecycle":
        limits = {"maximum_work_units": 480_000, "timeout": 180.0}
        rationale = "Uniform design required more than 330000 work units; larger cold continuation exceeded 120 seconds."
    elif (
        scenario == "native-implicit-lifecycle"
        and controls.get("implicit_case") in (
            "sphere-refinement", "offset-ring-torus-anisotropic",
        )
    ):
        limits = {"timeout": 240.0}
        rationale = (
            "Implicit sphere-refinement at 120 seconds completed both adaptations/"
            "transfers, scalar query and final PDE/error, then received native "
            "TIMEOUT at checkpoint publication. Process time was 122.292 seconds "
            "including 1.319 seconds reporting; archive/cold had not started. "
            "Sphere-refinement authors a finite 240-second whole deadline; "
            "routine sphere remains at 120 seconds. "
            "Reporting time does not extend the scientific clock."
        )
        if controls.get("implicit_case") == "offset-ring-torus-anisotropic":
            limits["maximum_work_units"] = 80_000_000
            limits["timeout"] = 2400.0
            rationale += (
                " Prior torus at work 80000000 and whole timeout 1200 seconds timed "
                "out in its first metric stage at 1145 seconds. The exact workset "
                "optimization has a verified 1.67-1.75x speedup with identical "
                "arrays/evidence/order. Only torus authors finite whole timeout "
                "2400 seconds, combining that measured stage and verified speedup "
                "with remaining workflow headroom. The work 80000000 two-adaptation "
                "envelope and all other scientific/resource controls are unchanged. "
                "This envelope is not evidence of successful completion."
            )
    elif scenario == "native-cad-lifecycle":
        limits = {"timeout": 600.0}
        rationale = (
            "Native CAD cylinder at 240 seconds reached accepted volume and spent "
            "203.668 seconds in exact P2 mapped-cell bisection before timeout; solve, "
            "archive, and cold continuation had not started, with healthy memory. "
            "The finite 600-second whole deadline is 2.5 times the current completion "
            "lower bound, retaining all work/query/scratch and scientific controls."
        )
    elif (
        scenario == "native-image-lifecycle"
        and controls.get("image_case") == "oblique-junction-transport"
    ):
        limits = {"timeout": 360.0}
        rationale = (
            "Oblique6 warm and archive consumed 130.352 seconds; cold ran another "
            "109.686 seconds before the 240-second deadline killed it. Completion "
            "requires more than 240 seconds; 360 seconds leaves at most 229.648 "
            "seconds for cold continuation, with 50% headroom over the current "
            "completion lower bound. Two-slab remains unchanged."
        )
    else:
        return None
    return {
        "requested": dict(limits),
        "effective": dict(limits),
        "rationale": rationale,
        "policy": "finite-authored-whole-workflow-envelope",
        **(
            {
                "execution_group_memory_enforcement": "external-systemd-cgroup-wrapper",
                "external_cgroup_properties": {"MemoryMax": limits["maximum_memory_bytes"]},
            }
            if scenario in ("native-distributed-lifecycle", "native-family-lifecycle") else {}
        ),
        **(
            {
                "rerun_prerequisites": ["durable-incremental-stage-and-failure-evidence"],
            }
            if scenario == "native-family-lifecycle" else {}
        ),
    }


def qualification_timeout(controls: Mapping[str, object], default: float) -> float:
    """Resolve one finite whole-workflow allowance before any launcher starts it."""
    value = controls.get("timeout", default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"qualification.timeout must be finite and positive; got {value!r}")
    return float(value)


def _validate_optional_resource_limit(name: str, value: int | None) -> None:
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError(f"qualification.{name} must be a positive integer; got {value!r}")


def _validate_resource_controls(
    timeout: float,
    maximum_memory_bytes: int,
    maximum_work_units: int | None,
    maximum_geometry_queries: int | None,
    maximum_scratch_bytes: int | None,
    maximum_data_bytes: int | None,
    maximum_recipe_bytes: int | None,
) -> None:
    if not isfinite(timeout) or timeout <= 0.0:
        raise ValueError(f"qualification.timeout must be finite and positive; got {timeout!r}")
    _validate_optional_resource_limit("maximum_memory_bytes", maximum_memory_bytes)
    _validate_optional_resource_limit("maximum_work_units", maximum_work_units)
    _validate_optional_resource_limit("maximum_geometry_queries", maximum_geometry_queries)
    _validate_optional_resource_limit("maximum_scratch_bytes", maximum_scratch_bytes)
    _validate_optional_resource_limit("maximum_data_bytes", maximum_data_bytes)
    _validate_optional_resource_limit("maximum_recipe_bytes", maximum_recipe_bytes)


@final
class QualificationControls(StrictModule):
    scenario: str = eqx.field(static=True, default="manifold")
    executable: str = eqx.field(static=True, default="vc_mesh")
    worker: str | None = eqx.field(static=True, default=None)
    corpus_case: str = eqx.field(static=True, default="planar-hole-feature")
    corpus_profile: tuple[str, ...] | None = eqx.field(static=True, default=None)
    polyhedral_source: str = eqx.field(static=True, default="polyhedral-material-L")
    resolution: int = eqx.field(static=True, default=4)
    capacity: int = eqx.field(static=True, default=20000)
    timeout: float = eqx.field(static=True, default=120.0)
    maximum_work_units: int | None = eqx.field(static=True, default=None)
    maximum_geometry_queries: int | None = eqx.field(static=True, default=None)
    maximum_scratch_bytes: int | None = eqx.field(static=True, default=None)
    maximum_data_bytes: int | None = eqx.field(static=True, default=None)
    maximum_recipe_bytes: int | None = eqx.field(static=True, default=None)
    repeats: int = eqx.field(static=True, default=3)
    target_error: float = eqx.field(static=True, default=0.05)
    adaptation_rounds: int = eqx.field(static=True, default=1)
    cad_curving_fidelity: float | None = eqx.field(static=True, default=None)
    surface_case: str = eqx.field(static=True, default="sphere")
    cad_format: str = eqx.field(static=True, default="step")
    cad_case: CadCase = eqx.field(static=True, default="planar-box")
    implicit_case: ImplicitCase = eqx.field(static=True, default="sphere")
    image_case: ImageCase = eqx.field(static=True, default="two-slab")
    design_case: DesignCase = eqx.field(static=True, default="routine-sphere")
    overset_profile: OversetProfile = eqx.field(static=True, default="routine")
    reconstruction_case: ReconstructionCase = eqx.field(
        static=True, default="lobed-sphere"
    )
    periodic_case: str = eqx.field(static=True, default="skew-lattice-exact-algebraic")
    maximum_memory_bytes: int = eqx.field(static=True, default=8 * 1024**3)
    maximum_condition: float = eqx.field(static=True, default=1000000000000.0)
    overset_motion_steps: int = eqx.field(static=True, default=2)
    overset_wall_pairs: int = eqx.field(static=True, default=500000)
    overset_donor_pairs: int = eqx.field(static=True, default=500000)
    overset_overlap_pairs: int = eqx.field(static=True, default=500000)
    family_case: str = eqx.field(static=True, default="dual-hex")
    layer_count: int = eqx.field(static=True, default=2)
    hybrid_profile: str = eqx.field(static=True, default="unit-box")
    hybrid_archive: str | None = eqx.field(static=True, default=None)
    hybrid_content_id: str | None = eqx.field(static=True, default=None)
    hybrid_region_densities: tuple[float, ...] | None = eqx.field(
        static=True, default=None
    )
    hybrid_material_site: str | None = eqx.field(static=True, default=None)
    hybrid_core_size: float = eqx.field(static=True, default=0.1)
    first_thickness: float = eqx.field(static=True, default=0.1)
    layer_growth: float = eqx.field(static=True, default=1.0)
    geometry_order: int = eqx.field(static=True, default=2)
    implicit_radius: float = eqx.field(static=True, default=0.371)
    implicit_fidelity: float = eqx.field(static=True, default=0.02)
    implicit_maximum_level: int = eqx.field(static=True, default=9)
    implicit_edge_isolation_depth: int = eqx.field(static=True, default=8)
    implicit_root_tolerance: float = eqx.field(static=True, default=1e-10)
    implicit_refinement_rounds: int = eqx.field(static=True, default=8)
    implicit_improvement_passes: int = eqx.field(static=True, default=8)
    implicit_metric_passes: int = eqx.field(static=True, default=16)
    distributed_parts: int = eqx.field(static=True, default=2)
    restart_parts: int = eqx.field(static=True, default=1)
    distributed_processes: int = eqx.field(static=True, default=2)
    distributed_source: str = eqx.field(static=True, default="native-unit-square")
    distributed_halo_width: int = eqx.field(static=True, default=2)
    distributed_halo_capacity: int = eqx.field(static=True, default=20000)
    distributed_cavity_capacity: int = eqx.field(static=True, default=20000)
    distributed_slack: int = eqx.field(static=True, default=2)
    distributed_host_packet_bytes: int = eqx.field(static=True, default=64 * 1024 * 1024)
    distributed_checkpoint_root: str | None = eqx.field(static=True, default=None)
    distributed_bisection_compatibility: BisectionCompatibility = eqx.field(
        static=True, default=BisectionCompatibility.REJECT
    )
    image_size: int = eqx.field(static=True, default=2)
    transport_steps: int = eqx.field(static=True, default=5)
    transport_dt: float = eqx.field(static=True, default=0.01)
    balance_tolerance: float = eqx.field(static=True, default=1e-09)
    oracle_capacity: int = eqx.field(static=True, default=2048)

    def __check_init__(self) -> None:
        _validate_resource_controls(
            self.timeout, self.maximum_memory_bytes,
            self.maximum_work_units, self.maximum_geometry_queries,
            self.maximum_scratch_bytes,
            self.maximum_data_bytes,
            self.maximum_recipe_bytes,
        )

    def to_record(self) -> dict[str, Any]:
        """Serialize host controls at a process or JSON boundary."""
        return {
            "scenario": self.scenario,
            "executable": self.executable,
            "worker": self.worker,
            "corpus_case": self.corpus_case,
            "corpus_profile": self.corpus_profile,
            "polyhedral_source": self.polyhedral_source,
            "resolution": self.resolution,
            "capacity": self.capacity,
            "timeout": self.timeout,
            "maximum_work_units": self.maximum_work_units,
            "maximum_geometry_queries": self.maximum_geometry_queries,
            "maximum_scratch_bytes": self.maximum_scratch_bytes,
            "maximum_data_bytes": self.maximum_data_bytes,
            "maximum_recipe_bytes": self.maximum_recipe_bytes,
            "repeats": self.repeats,
            "target_error": self.target_error,
            "adaptation_rounds": self.adaptation_rounds,
            "cad_curving_fidelity": self.cad_curving_fidelity,
            "surface_case": self.surface_case,
            "cad_format": self.cad_format,
            "cad_case": self.cad_case,
            "implicit_case": self.implicit_case,
            "image_case": self.image_case,
            "design_case": self.design_case,
            "overset_profile": self.overset_profile,
            "reconstruction_case": self.reconstruction_case,
            "periodic_case": self.periodic_case,
            "maximum_memory_bytes": self.maximum_memory_bytes,
            "maximum_condition": self.maximum_condition,
            "overset_motion_steps": self.overset_motion_steps,
            "overset_wall_pairs": self.overset_wall_pairs,
            "overset_donor_pairs": self.overset_donor_pairs,
            "overset_overlap_pairs": self.overset_overlap_pairs,
            "family_case": self.family_case,
            "layer_count": self.layer_count,
            "hybrid_profile": self.hybrid_profile,
            "hybrid_archive": self.hybrid_archive,
            "hybrid_content_id": self.hybrid_content_id,
            "hybrid_region_densities": self.hybrid_region_densities,
            "hybrid_material_site": self.hybrid_material_site,
            "hybrid_core_size": self.hybrid_core_size,
            "first_thickness": self.first_thickness,
            "layer_growth": self.layer_growth,
            "geometry_order": self.geometry_order,
            "implicit_radius": self.implicit_radius,
            "implicit_fidelity": self.implicit_fidelity,
            "implicit_maximum_level": self.implicit_maximum_level,
            "implicit_edge_isolation_depth": self.implicit_edge_isolation_depth,
            "implicit_root_tolerance": self.implicit_root_tolerance,
            "implicit_refinement_rounds": self.implicit_refinement_rounds,
            "implicit_improvement_passes": self.implicit_improvement_passes,
            "implicit_metric_passes": self.implicit_metric_passes,
            "distributed_parts": self.distributed_parts,
            "restart_parts": self.restart_parts,
            "distributed_processes": self.distributed_processes,
            "distributed_source": self.distributed_source,
            "distributed_halo_width": self.distributed_halo_width,
            "distributed_halo_capacity": self.distributed_halo_capacity,
            "distributed_cavity_capacity": self.distributed_cavity_capacity,
            "distributed_slack": self.distributed_slack,
            "distributed_host_packet_bytes": self.distributed_host_packet_bytes,
            "distributed_checkpoint_root": self.distributed_checkpoint_root,
            "distributed_bisection_compatibility": self.distributed_bisection_compatibility.value,
            "image_size": self.image_size,
            "transport_steps": self.transport_steps,
            "transport_dt": self.transport_dt,
            "balance_tolerance": self.balance_tolerance,
            "oracle_capacity": self.oracle_capacity,
        }

    @classmethod
    def from_namespace(cls, value: argparse.Namespace) -> QualificationControls:
        """Convert the parser result once before dispatch."""
        densities, material_site = (
            (
                None
                if value.hybrid_region_densities is None
                else tuple(value.hybrid_region_densities),
                value.hybrid_material_site,
            )
            if value.scenario == "native-corpus"
            else _hybrid_material_controls(
                value.hybrid_profile,
                value.hybrid_region_densities,
                value.hybrid_material_site,
            )
        )
        return cls(
            scenario=value.scenario,
            executable=value.executable,
            worker=value.worker,
            corpus_case=value.corpus_case,
            corpus_profile=None
            if value.corpus_profile is None
            else tuple(value.corpus_profile),
            polyhedral_source=value.polyhedral_source,
            resolution=value.resolution,
            capacity=value.capacity,
            timeout=value.timeout,
            maximum_work_units=value.maximum_work_units,
            maximum_geometry_queries=value.maximum_geometry_queries,
            maximum_scratch_bytes=value.maximum_scratch_bytes,
            maximum_data_bytes=value.maximum_data_bytes,
            maximum_recipe_bytes=value.maximum_recipe_bytes,
            repeats=value.repeats,
            target_error=value.target_error,
            adaptation_rounds=value.adaptation_rounds,
            cad_curving_fidelity=value.cad_curving_fidelity,
            surface_case=value.surface_case,
            cad_format=value.cad_format,
            cad_case=parse(value.cad_case, CadCase, "cad_case"),
            implicit_case=parse(value.implicit_case, ImplicitCase, "implicit_case"),
            image_case=parse(value.image_case, ImageCase, "image_case"),
            design_case=parse(value.design_case, DesignCase, "design_case"),
            overset_profile=parse(
                value.overset_profile, OversetProfile, "overset_profile"
            ),
            reconstruction_case=parse(
                value.reconstruction_case, ReconstructionCase, "reconstruction_case"
            ),
            periodic_case=value.periodic_case,
            maximum_memory_bytes=value.maximum_memory_bytes,
            maximum_condition=value.maximum_condition,
            overset_motion_steps=value.overset_motion_steps,
            overset_wall_pairs=value.overset_wall_pairs,
            overset_donor_pairs=value.overset_donor_pairs,
            overset_overlap_pairs=value.overset_overlap_pairs,
            family_case=value.family_case,
            layer_count=value.layer_count,
            hybrid_profile=value.hybrid_profile,
            hybrid_archive=value.hybrid_archive,
            hybrid_content_id=value.hybrid_content_id,
            hybrid_region_densities=densities,
            hybrid_material_site=material_site,
            hybrid_core_size=value.hybrid_core_size,
            first_thickness=value.first_thickness,
            layer_growth=value.layer_growth,
            geometry_order=value.geometry_order,
            implicit_radius=value.implicit_radius,
            implicit_fidelity=value.implicit_fidelity,
            implicit_maximum_level=value.implicit_maximum_level,
            implicit_edge_isolation_depth=value.implicit_edge_isolation_depth,
            implicit_root_tolerance=value.implicit_root_tolerance,
            implicit_refinement_rounds=value.implicit_refinement_rounds,
            implicit_improvement_passes=value.implicit_improvement_passes,
            implicit_metric_passes=value.implicit_metric_passes,
            distributed_parts=value.distributed_parts,
            restart_parts=value.restart_parts,
            distributed_processes=value.distributed_processes,
            distributed_source=value.distributed_source,
            distributed_halo_width=value.distributed_halo_width,
            distributed_halo_capacity=value.distributed_halo_capacity,
            distributed_cavity_capacity=value.distributed_cavity_capacity,
            distributed_slack=value.distributed_slack,
            distributed_host_packet_bytes=value.distributed_host_packet_bytes,
            distributed_checkpoint_root=value.distributed_checkpoint_root,
            distributed_bisection_compatibility=parse(
                value.distributed_bisection_compatibility,
                BisectionCompatibility,
                "distributed_bisection_compatibility",
            ),
            image_size=value.image_size,
            transport_steps=value.transport_steps,
            transport_dt=value.transport_dt,
            balance_tolerance=value.balance_tolerance,
            oracle_capacity=value.oracle_capacity,
        )

    @classmethod
    def from_record(cls, values: dict[str, Any]) -> QualificationControls:
        """Restore serialized controls; constructors reject unknown fields."""
        values = {
            **values,
            "distributed_bisection_compatibility": parse(
                BisectionCompatibility(values["distributed_bisection_compatibility"]),
                BisectionCompatibility,
                "distributed_bisection_compatibility",
            ),
        }
        return cls(**values)

    def distributed_repository_root(self) -> Path:
        """Resolve the required durable distributed checkpoint destination."""
        value = self.distributed_checkpoint_root
        if value is None:
            raise ValueError(
                f"qualification.distributed_checkpoint_root must name a destination; got {value!r}"
            )
        return Path(value)


@final
class BenchmarkControls(StrictModule):
    resolution: tuple[int, ...] = eqx.field(static=True, default=(8, 16))
    repeats: int = eqx.field(static=True, default=5)
    benchmark_warm_repeats: int = eqx.field(static=True, default=5)
    corpus_case: str = eqx.field(static=True, default="planar-hole-feature")
    corpus_profile: tuple[str, ...] | None = eqx.field(static=True, default=None)
    polyhedral_source: tuple[str, ...] = eqx.field(
        static=True, default=POLYHEDRAL_CASE_NAMES
    )
    capacity: tuple[int, ...] = eqx.field(static=True, default=(20000,))
    timeout: float = eqx.field(static=True, default=120.0)
    maximum_work_units: int | None = eqx.field(static=True, default=None)
    maximum_geometry_queries: int | None = eqx.field(static=True, default=None)
    maximum_scratch_bytes: int | None = eqx.field(static=True, default=None)
    maximum_data_bytes: int | None = eqx.field(static=True, default=None)
    maximum_recipe_bytes: int | None = eqx.field(static=True, default=None)
    target_error: float = eqx.field(static=True, default=0.05)
    adaptation_rounds: int = eqx.field(static=True, default=1)
    comparison: str | None = eqx.field(static=True, default=None)
    cad_format: str = eqx.field(static=True, default="step")
    cad_case: CadCase = eqx.field(static=True, default="planar-box")
    implicit_case: ImplicitCase = eqx.field(static=True, default="sphere")
    image_case: ImageCase = eqx.field(static=True, default="two-slab")
    design_case: DesignCase = eqx.field(static=True, default="routine-sphere")
    overset_profile: OversetProfile = eqx.field(static=True, default="routine")
    reconstruction_case: ReconstructionCase = eqx.field(
        static=True, default="lobed-sphere"
    )
    periodic_case: str = eqx.field(static=True, default="skew-lattice-exact-algebraic")
    maximum_memory_bytes: int = eqx.field(static=True, default=8 * 1024**3)
    maximum_condition: float = eqx.field(static=True, default=1000000000000.0)
    family_case: str = eqx.field(static=True, default="dual-hex")
    surface_case: str = eqx.field(static=True, default="sphere")
    hybrid_profile: str = eqx.field(static=True, default="unit-box")
    hybrid_archive: str | None = eqx.field(static=True, default=None)
    hybrid_content_id: str | None = eqx.field(static=True, default=None)
    hybrid_core_size: float = eqx.field(static=True, default=0.1)
    hybrid_region_densities: tuple[float, ...] | None = eqx.field(
        static=True, default=None
    )
    hybrid_material_site: str | None = eqx.field(static=True, default=None)
    layer_count: int = eqx.field(static=True, default=2)
    first_thickness: float = eqx.field(static=True, default=0.1)
    layer_growth: float = eqx.field(static=True, default=1.0)
    geometry_order: int = eqx.field(static=True, default=2)
    implicit_radius: float = eqx.field(static=True, default=0.371)
    implicit_fidelity: float = eqx.field(static=True, default=0.02)
    implicit_maximum_level: int = eqx.field(static=True, default=9)
    implicit_edge_isolation_depth: int = eqx.field(static=True, default=8)
    implicit_root_tolerance: float = eqx.field(static=True, default=1e-10)
    implicit_refinement_rounds: int = eqx.field(static=True, default=8)
    implicit_improvement_passes: int = eqx.field(static=True, default=8)
    implicit_metric_passes: int = eqx.field(static=True, default=16)
    distributed_parts: int = eqx.field(static=True, default=2)
    restart_parts: int = eqx.field(static=True, default=1)
    distributed_processes: int = eqx.field(static=True, default=2)
    distributed_source: str = eqx.field(static=True, default="native-unit-square")
    distributed_halo_width: int = eqx.field(static=True, default=2)
    distributed_halo_capacity: int = eqx.field(static=True, default=20000)
    distributed_cavity_capacity: int = eqx.field(static=True, default=20000)
    distributed_slack: int = eqx.field(static=True, default=2)
    distributed_host_packet_bytes: int = eqx.field(static=True, default=64 * 1024 * 1024)
    distributed_checkpoint_root: str | None = eqx.field(static=True, default=None)
    distributed_bisection_compatibility: BisectionCompatibility = eqx.field(
        static=True, default=BisectionCompatibility.REJECT
    )
    image_size: int = eqx.field(static=True, default=2)
    transport_steps: int = eqx.field(static=True, default=5)
    transport_dt: float = eqx.field(static=True, default=0.01)
    balance_tolerance: float = eqx.field(static=True, default=1e-09)
    oracle_capacity: int = eqx.field(static=True, default=2048)
    overset_motion_steps: int = eqx.field(static=True, default=2)
    overset_wall_pairs: int = eqx.field(static=True, default=500000)
    overset_donor_pairs: int = eqx.field(static=True, default=500000)
    overset_overlap_pairs: int = eqx.field(static=True, default=500000)
    gmsh_semantic: bool = eqx.field(static=True, default=False)
    case: tuple[str, ...] | None = eqx.field(static=True, default=None)

    def __check_init__(self) -> None:
        _validate_resource_controls(
            self.timeout, self.maximum_memory_bytes,
            self.maximum_work_units, self.maximum_geometry_queries,
            self.maximum_scratch_bytes,
            self.maximum_data_bytes,
            self.maximum_recipe_bytes,
        )

    def to_record(self) -> dict[str, Any]:
        """Serialize host controls at a process or JSON boundary."""
        return {
            "resolution": self.resolution,
            "repeats": self.repeats,
            "benchmark_warm_repeats": self.benchmark_warm_repeats,
            "corpus_case": self.corpus_case,
            "corpus_profile": self.corpus_profile,
            "polyhedral_source": self.polyhedral_source,
            "capacity": self.capacity,
            "timeout": self.timeout,
            "maximum_work_units": self.maximum_work_units,
            "maximum_geometry_queries": self.maximum_geometry_queries,
            "maximum_scratch_bytes": self.maximum_scratch_bytes,
            "maximum_data_bytes": self.maximum_data_bytes,
            "maximum_recipe_bytes": self.maximum_recipe_bytes,
            "target_error": self.target_error,
            "adaptation_rounds": self.adaptation_rounds,
            "comparison": self.comparison,
            "cad_format": self.cad_format,
            "cad_case": self.cad_case,
            "implicit_case": self.implicit_case,
            "image_case": self.image_case,
            "design_case": self.design_case,
            "overset_profile": self.overset_profile,
            "reconstruction_case": self.reconstruction_case,
            "periodic_case": self.periodic_case,
            "maximum_memory_bytes": self.maximum_memory_bytes,
            "maximum_condition": self.maximum_condition,
            "family_case": self.family_case,
            "surface_case": self.surface_case,
            "hybrid_profile": self.hybrid_profile,
            "hybrid_archive": self.hybrid_archive,
            "hybrid_content_id": self.hybrid_content_id,
            "hybrid_core_size": self.hybrid_core_size,
            "hybrid_region_densities": self.hybrid_region_densities,
            "hybrid_material_site": self.hybrid_material_site,
            "layer_count": self.layer_count,
            "first_thickness": self.first_thickness,
            "layer_growth": self.layer_growth,
            "geometry_order": self.geometry_order,
            "implicit_radius": self.implicit_radius,
            "implicit_fidelity": self.implicit_fidelity,
            "implicit_maximum_level": self.implicit_maximum_level,
            "implicit_edge_isolation_depth": self.implicit_edge_isolation_depth,
            "implicit_root_tolerance": self.implicit_root_tolerance,
            "implicit_refinement_rounds": self.implicit_refinement_rounds,
            "implicit_improvement_passes": self.implicit_improvement_passes,
            "implicit_metric_passes": self.implicit_metric_passes,
            "distributed_parts": self.distributed_parts,
            "restart_parts": self.restart_parts,
            "distributed_processes": self.distributed_processes,
            "distributed_source": self.distributed_source,
            "distributed_halo_width": self.distributed_halo_width,
            "distributed_halo_capacity": self.distributed_halo_capacity,
            "distributed_cavity_capacity": self.distributed_cavity_capacity,
            "distributed_slack": self.distributed_slack,
            "distributed_host_packet_bytes": self.distributed_host_packet_bytes,
            "distributed_checkpoint_root": self.distributed_checkpoint_root,
            "distributed_bisection_compatibility": self.distributed_bisection_compatibility.value,
            "image_size": self.image_size,
            "transport_steps": self.transport_steps,
            "transport_dt": self.transport_dt,
            "balance_tolerance": self.balance_tolerance,
            "oracle_capacity": self.oracle_capacity,
            "overset_motion_steps": self.overset_motion_steps,
            "overset_wall_pairs": self.overset_wall_pairs,
            "overset_donor_pairs": self.overset_donor_pairs,
            "overset_overlap_pairs": self.overset_overlap_pairs,
            "gmsh_semantic": self.gmsh_semantic,
            "case": self.case,
        }

    @classmethod
    def from_namespace(cls, value: argparse.Namespace) -> BenchmarkControls:
        """Convert the parser result once before dispatch."""
        return cls(
            resolution=tuple(value.resolution),
            repeats=value.repeats,
            benchmark_warm_repeats=value.benchmark_warm_repeats,
            corpus_case=value.corpus_case,
            corpus_profile=None
            if value.corpus_profile is None
            else tuple(value.corpus_profile),
            polyhedral_source=tuple(value.polyhedral_source),
            capacity=tuple(value.capacity),
            timeout=value.timeout,
            maximum_work_units=value.maximum_work_units,
            maximum_geometry_queries=value.maximum_geometry_queries,
            maximum_scratch_bytes=value.maximum_scratch_bytes,
            maximum_data_bytes=value.maximum_data_bytes,
            maximum_recipe_bytes=value.maximum_recipe_bytes,
            target_error=value.target_error,
            adaptation_rounds=value.adaptation_rounds,
            comparison=value.comparison,
            cad_format=value.cad_format,
            cad_case=parse(value.cad_case, CadCase, "cad_case"),
            implicit_case=parse(value.implicit_case, ImplicitCase, "implicit_case"),
            image_case=parse(value.image_case, ImageCase, "image_case"),
            design_case=parse(value.design_case, DesignCase, "design_case"),
            overset_profile=parse(
                value.overset_profile, OversetProfile, "overset_profile"
            ),
            reconstruction_case=parse(
                value.reconstruction_case, ReconstructionCase, "reconstruction_case"
            ),
            periodic_case=value.periodic_case,
            maximum_memory_bytes=value.maximum_memory_bytes,
            maximum_condition=value.maximum_condition,
            family_case=value.family_case,
            surface_case=value.surface_case,
            hybrid_profile=value.hybrid_profile,
            hybrid_archive=value.hybrid_archive,
            hybrid_content_id=value.hybrid_content_id,
            hybrid_core_size=value.hybrid_core_size,
            hybrid_region_densities=None
            if value.hybrid_region_densities is None
            else tuple(value.hybrid_region_densities),
            hybrid_material_site=value.hybrid_material_site,
            layer_count=value.layer_count,
            first_thickness=value.first_thickness,
            layer_growth=value.layer_growth,
            geometry_order=value.geometry_order,
            implicit_radius=value.implicit_radius,
            implicit_fidelity=value.implicit_fidelity,
            implicit_maximum_level=value.implicit_maximum_level,
            implicit_edge_isolation_depth=value.implicit_edge_isolation_depth,
            implicit_root_tolerance=value.implicit_root_tolerance,
            implicit_refinement_rounds=value.implicit_refinement_rounds,
            implicit_improvement_passes=value.implicit_improvement_passes,
            implicit_metric_passes=value.implicit_metric_passes,
            distributed_parts=value.distributed_parts,
            restart_parts=value.restart_parts,
            distributed_processes=value.distributed_processes,
            distributed_source=value.distributed_source,
            distributed_halo_width=value.distributed_halo_width,
            distributed_halo_capacity=value.distributed_halo_capacity,
            distributed_cavity_capacity=value.distributed_cavity_capacity,
            distributed_slack=value.distributed_slack,
            distributed_host_packet_bytes=value.distributed_host_packet_bytes,
            distributed_checkpoint_root=value.distributed_checkpoint_root,
            distributed_bisection_compatibility=parse(
                value.distributed_bisection_compatibility,
                BisectionCompatibility,
                "distributed_bisection_compatibility",
            ),
            image_size=value.image_size,
            transport_steps=value.transport_steps,
            transport_dt=value.transport_dt,
            balance_tolerance=value.balance_tolerance,
            oracle_capacity=value.oracle_capacity,
            overset_motion_steps=value.overset_motion_steps,
            overset_wall_pairs=value.overset_wall_pairs,
            overset_donor_pairs=value.overset_donor_pairs,
            overset_overlap_pairs=value.overset_overlap_pairs,
            gmsh_semantic=value.gmsh_semantic,
            case=None if value.case is None else tuple(value.case),
        )

    @classmethod
    def from_record(cls, values: dict[str, Any]) -> BenchmarkControls:
        """Restore serialized controls; constructors reject unknown fields."""
        values = {
            **values,
            "distributed_bisection_compatibility": parse(
                BisectionCompatibility(values["distributed_bisection_compatibility"]),
                BisectionCompatibility,
                "distributed_bisection_compatibility",
            ),
        }
        return cls(**values)


def add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared closed source/profile selectors."""
    parser.add_argument("--surface-case", choices=SURFACE_CASES, default="sphere")
    parser.add_argument("--family-case", choices=FAMILY_CASES, default="dual-hex")
    parser.add_argument("--hybrid-profile", choices=HYBRID_PROFILES, default="unit-box")


def add_case_arguments(parser: argparse.ArgumentParser) -> None:
    """Keep qualification and benchmark case selection on one declared contract."""
    defaults = QualificationControls()
    parser.add_argument("--maximum-work-units", type=int, default=None)
    parser.add_argument("--maximum-geometry-queries", type=int, default=None)
    parser.add_argument("--maximum-scratch-bytes", type=int, default=None)
    parser.add_argument("--maximum-data-bytes", type=int, default=None)
    parser.add_argument("--maximum-recipe-bytes", type=int, default=None)
    parser.add_argument("--cad-case", choices=CAD_CASES, default=defaults.cad_case)
    parser.add_argument(
        "--implicit-case", choices=IMPLICIT_CASES, default=defaults.implicit_case
    )
    parser.add_argument("--image-case", choices=IMAGE_CASES, default=defaults.image_case)
    parser.add_argument(
        "--design-case", choices=DESIGN_CASES, default=defaults.design_case
    )
    parser.add_argument(
        "--overset-profile", choices=OVERSET_PROFILES, default=defaults.overset_profile
    )
    parser.add_argument(
        "--reconstruction-case",
        choices=RECONSTRUCTION_CASES,
        default=defaults.reconstruction_case,
    )


def meshing_limit_values(limits: phx.meshing.MeshingLimits) -> dict[str, int | float]:
    """Record the declared admission controls, excluding derived identities."""
    return {
        "maximum_vertices": limits.maximum_vertices,
        "maximum_edges": limits.maximum_edges,
        "maximum_faces": limits.maximum_faces,
        "maximum_cells": limits.maximum_cells,
        "maximum_connectivity_entries": limits.maximum_connectivity_entries,
        "maximum_data_bytes": limits.maximum_data_bytes,
        "maximum_work_units": limits.maximum_work_units,
        "maximum_cavity_cells": limits.maximum_cavity_cells,
        "maximum_geometry_queries": limits.maximum_geometry_queries,
        "maximum_scratch_bytes": limits.maximum_scratch_bytes,
        "maximum_wall_seconds": limits.maximum_wall_seconds,
    }


class _ScaledMeshingLimitArguments(TypedDict, total=False):
    maximum_vertices: int
    maximum_edges: int
    maximum_faces: int
    maximum_cells: int
    maximum_connectivity_entries: int
    maximum_data_bytes: int
    maximum_work_units: int
    maximum_cavity_cells: int
    maximum_geometry_queries: int
    maximum_scratch_bytes: int
    maximum_wall_seconds: float


def capacity_meshing_limits(
    capacity: int,
    timeout: float,
    *,
    vertices: int = 1,
    edges: int | None = None,
    faces: int | None = None,
    cells: int | None = None,
    connectivity_entries: int | None = None,
    data_bytes: int | None = None,
    work_units: int | None = None,
    cavity_cells: int | None = 1,
    geometry_queries: int | None = None,
    scratch_bytes: int | None = None,
    data_bytes_limit: int | None = None,
    scratch_bytes_limit: int | None = None,
    cavity_cells_limit: int | None = None,
    maximum_work_units: int | None = None,
    maximum_geometry_queries: int | None = None,
) -> phx.meshing.MeshingLimits:
    """Lower caller-declared capacity factors without changing admission controls."""
    values: _ScaledMeshingLimitArguments = {
        "maximum_vertices": vertices * capacity,
        "maximum_wall_seconds": timeout,
    }
    if edges is not None:
        values["maximum_edges"] = edges * capacity
    if faces is not None:
        values["maximum_faces"] = faces * capacity
    if cells is not None:
        values["maximum_cells"] = cells * capacity
    if connectivity_entries is not None:
        values["maximum_connectivity_entries"] = connectivity_entries * capacity
    if data_bytes is not None:
        values["maximum_data_bytes"] = data_bytes * capacity
    if work_units is not None:
        values["maximum_work_units"] = work_units * capacity
    if cavity_cells is not None:
        values["maximum_cavity_cells"] = cavity_cells * capacity
    if geometry_queries is not None:
        values["maximum_geometry_queries"] = geometry_queries * capacity
    if scratch_bytes is not None:
        values["maximum_scratch_bytes"] = scratch_bytes * capacity
    if data_bytes_limit is not None:
        values["maximum_data_bytes"] = data_bytes_limit
    if scratch_bytes_limit is not None:
        values["maximum_scratch_bytes"] = scratch_bytes_limit
    if cavity_cells_limit is not None:
        values["maximum_cavity_cells"] = cavity_cells_limit
    for name, value in (
        ("maximum_work_units", maximum_work_units),
        ("maximum_geometry_queries", maximum_geometry_queries),
        ("maximum_scratch_bytes", scratch_bytes_limit),
        ("maximum_data_bytes", data_bytes_limit),
    ):
        _validate_optional_resource_limit(name, value)
    if maximum_work_units is not None:
        values["maximum_work_units"] = maximum_work_units
    if maximum_geometry_queries is not None:
        values["maximum_geometry_queries"] = maximum_geometry_queries
    return phx.meshing.MeshingLimits(**values)


@final
class MeshingLimitControls(StrictModule):
    maximum_vertices: int = eqx.field(static=True, default=10000000)
    maximum_edges: int = eqx.field(static=True, default=50000000)
    maximum_faces: int = eqx.field(static=True, default=50000000)
    maximum_cells: int = eqx.field(static=True, default=20000000)
    maximum_connectivity_entries: int = eqx.field(static=True, default=500000000)
    maximum_data_bytes: int = eqx.field(static=True, default=4000000000)
    maximum_work_units: int = eqx.field(static=True, default=1000000000)
    maximum_cavity_cells: int = eqx.field(static=True, default=1000000)
    maximum_geometry_queries: int = eqx.field(static=True, default=1000000000)
    maximum_scratch_bytes: int = eqx.field(static=True, default=8000000000)
    maximum_wall_seconds: float = eqx.field(static=True, default=3600.0)

    @classmethod
    def from_limits(cls, limits: phx.meshing.MeshingLimits) -> MeshingLimitControls:
        return cls(
            maximum_vertices=limits.maximum_vertices,
            maximum_edges=limits.maximum_edges,
            maximum_faces=limits.maximum_faces,
            maximum_cells=limits.maximum_cells,
            maximum_connectivity_entries=limits.maximum_connectivity_entries,
            maximum_data_bytes=limits.maximum_data_bytes,
            maximum_work_units=limits.maximum_work_units,
            maximum_cavity_cells=limits.maximum_cavity_cells,
            maximum_geometry_queries=limits.maximum_geometry_queries,
            maximum_scratch_bytes=limits.maximum_scratch_bytes,
            maximum_wall_seconds=limits.maximum_wall_seconds,
        )

    def to_limits(self) -> phx.meshing.MeshingLimits:
        """Construct the public limit record, recomputing its content identity."""
        return phx.meshing.MeshingLimits(
            maximum_vertices=self.maximum_vertices,
            maximum_edges=self.maximum_edges,
            maximum_faces=self.maximum_faces,
            maximum_cells=self.maximum_cells,
            maximum_connectivity_entries=self.maximum_connectivity_entries,
            maximum_data_bytes=self.maximum_data_bytes,
            maximum_work_units=self.maximum_work_units,
            maximum_cavity_cells=self.maximum_cavity_cells,
            maximum_geometry_queries=self.maximum_geometry_queries,
            maximum_scratch_bytes=self.maximum_scratch_bytes,
            maximum_wall_seconds=self.maximum_wall_seconds,
        )


@overload
def with_meshing_limits(
    request: phx.meshing.SurfaceMeshingSpec,
    limits: phx.meshing.MeshingLimits,
) -> phx.meshing.SurfaceMeshingSpec: ...


@overload
def with_meshing_limits(
    request: phx.meshing.VolumeMeshingSpec,
    limits: phx.meshing.MeshingLimits,
) -> phx.meshing.VolumeMeshingSpec: ...


def with_meshing_limits(
    request: phx.meshing.SurfaceMeshingSpec | phx.meshing.VolumeMeshingSpec,
    limits: phx.meshing.MeshingLimits,
) -> phx.meshing.SurfaceMeshingSpec | phx.meshing.VolumeMeshingSpec:
    """Rebuild a request with one changed admission, preserving all source controls."""
    match request:
        case phx.meshing.SurfaceMeshingSpec():
            return phx.meshing.SurfaceMeshingSpec(
                request.target,
                request.scope,
                planar_embedding=request.planar_embedding,
                size_controls=request.size_controls,
                protected_features=request.protected_features,
                region_controls=request.region_controls,
                patch_controls=request.patch_controls,
                periodic_constraints=request.periodic_constraints,
                layer_controls=request.layer_controls,
                size_combination=request.size_combination,
                size_compliance=request.size_compliance,
                limits=limits,
                quality_target=request.quality_target,
                background_metric=request.background_metric,
                deterministic=request.deterministic,
            )
        case phx.meshing.VolumeMeshingSpec():
            return phx.meshing.VolumeMeshingSpec(
                request.target,
                request.boundary_scope,
                request.fill_strategy,
                size_controls=request.size_controls,
                protected_features=request.protected_features,
                region_controls=request.region_controls,
                patch_controls=request.patch_controls,
                region_seeds=request.region_seeds,
                hole_seeds=request.hole_seeds,
                layer_controls=request.layer_controls,
                periodic_constraints=request.periodic_constraints,
                size_combination=request.size_combination,
                size_compliance=request.size_compliance,
                limits=limits,
                deterministic=request.deterministic,
            )
        case _:
            raise TypeError(
                f"qualification.request must be a surface or volume specification; got {type(request).__qualname__}"
            )


def _surface_qualification_requirements() -> dict[str, object]:
    """Declare source profiles and their distinct physical PDE obligations."""
    return {
        "required_arguments": {
            "surface_case": {
                "choices": SURFACE_CASES,
                "default": "sphere",
            }
        },
        "evidence_sections": (
            "generation",
            "epochs",
            "source_acceptance",
            "checkpoint",
            "physical_error",
            "mandatory_workflow_completion",
        ),
        "profiles": {
            "sphere": {
                "source": "unit-sphere-seam-and-poles",
                "pde": "zero-mean-Laplace-Beltrami",
            },
            "torus": {
                "source": "original-double-periodic-torus",
                "pde": "toroidal-height-reaction-diffusion",
            },
            "capped_cylinder": {
                "source": "original-shared-periodic-rims",
                "pde": "seam-C1-axial-quartic-reaction-diffusion",
            },
            "capped_cylinder_features": {
                "source": "original-shared-periodic-rims",
                "pde": "seam-C1-axial-quartic-reaction-diffusion",
                "controls": "hard-size-p50-surface-and-rim-curve-deviation",
            },
            "annulus": {
                "source": "original-exact-period-hole",
                "pde": "affine-planar-Dirichlet-diffusion",
            },
            "cylinder_sheet": {
                "source": "original-open-quarter-cylinder",
                "pde": "axial-Dirichlet-diffusion",
            },
            "folded_plates": {
                "source": "original-shared-sharp-interface",
                "pde": "interface-flux-continuous-Dirichlet-diffusion",
            },
        },
    }


def _corpus_contract_choices(
    contract: Mapping[str, Any], argument: str | None
) -> tuple[str, ...]:
    """Read choices from a locally constructed parser contract."""
    choices = (
        contract["parser_choices"]
        if argument is None
        else contract["required_arguments"][argument]["choices"]
    )
    return tuple(choices)


def _cad_qualification_requirements() -> dict[str, object]:
    """Parser and evidence contract; this does not register or execute a case."""
    return {
        "required_arguments": {
            "cad_format": {"choices": ["step", "iges", "brep"]},
            "cad_case": {"choices": CAD_CASES, "default": "planar-box"},
            "resolution": {"minimum": 1},
            "capacity": {"minimum": 1},
            "timeout": {"exclusive_minimum": 0.0, "finite": True},
            "repeats": {"minimum": 1},
            "target_error": {"exclusive_minimum": 0.0, "finite": True},
            "adaptation_rounds": {"minimum": 1},
        },
        "callable": "_cad_qualification_lifecycle",
        "fixture": "native-authored-exact-box-format-roundtrip-and-intersection",
        "evidence_sections": [
            "imports",
            "boolean",
            "query",
            "source",
            "request",
            "native_phases",
            "stages_seconds",
            "epochs",
            "physical_error",
            "time_to_target_error_seconds",
            "peak_process_resident_bytes",
            "device_memory",
            "positive_prerequisite",
        ],
        "registration": "requires-parent-observed-positive-source-transfer-solver-evidence",
        "mandatory_workflow_completion": "incomplete",
        "nonclaims": [
            "Generated format files are not an independently authored external CAD corpus.",
            "The full curved/multiformat/partition/high-order combination remains incomplete.",
            "No optional external engine is a native fallback or a scientific reference.",
        ],
    }


def _overset_qualification_requirements() -> dict[str, object]:
    """Freeze the physical assemblies and their independent consumer obligations."""
    return {
        "callable": "_overset_qualification_lifecycle",
        "required_arguments": {
            "overset_profile": {"choices": OVERSET_PROFILES, "default": "routine"},
            "resolution": {"minimum": 1},
            "capacity": {"minimum": 1},
            "timeout": {"exclusive_minimum": 0.0, "finite": True},
            "repeats": {"minimum": 1},
            "target_error": {"exclusive_minimum": 0.0, "finite": True},
            "overset_motion_steps": {"minimum": 2},
            "overset_wall_pairs": {"minimum": 1},
            "overset_donor_pairs": {"minimum": 1},
            "overset_overlap_pairs": {"minimum": 1},
        },
        "fixture": "native-planar-background-and-protected-square-annulus-bodies",
        "geometry": {
            "background_half_width": 3.0,
            "profiles": {
                "routine": {
                    "fringe_layers": 1,
                    "bodies": [
                        {
                            "name": "moving",
                            "center": [0.0, 0.0],
                            "half_width": 2.25,
                            "cavity_half_width": 0.25,
                            "refinement": 1,
                            "offsets": [[0.55, 0.0], [0.05, 0.0]],
                            "offset_rule": "first-then-repeat-last",
                        },
                    ],
                },
                "larger": {
                    "fringe_layers": 1,
                    "bodies": [
                        {
                            "name": "port",
                            "center": [-1.5, 0.0],
                            "half_width": 1.25,
                            "cavity_half_width": 0.25,
                            "refinement": 2,
                            "offsets": [[0.0, 0.5], [0.0, -0.5]],
                            "offset_rule": "cycle",
                        },
                        {
                            "name": "starboard",
                            "center": [1.5, 0.0],
                            "half_width": 1.25,
                            "cavity_half_width": 0.25,
                            "refinement": 2,
                            "offsets": [[0.0, -0.5], [0.0, 0.5]],
                            "offset_rule": "cycle",
                        },
                    ],
                },
                "combined": {
                    "fringe_layers": 2,
                    "bodies": [
                        {
                            "name": "alpha",
                            "center": [-1.15, -0.3],
                            "half_width": 1.7,
                            "cavity_half_width": 0.3,
                            "refinement": 1,
                            "offsets": [[0.0, 0.45], [0.0, -0.45]],
                            "offset_rule": "cycle",
                        },
                        {
                            "name": "beta",
                            "center": [1.55, 0.5],
                            "half_width": 1.3,
                            "cavity_half_width": 0.2,
                            "refinement": 2,
                            "offsets": [[-0.05, -0.4], [0.05, 0.4]],
                            "offset_rule": "cycle",
                        },
                    ],
                },
            },
        },
        "profiles": {
            "routine": "original-single-annulus-0.55-0.05-translation",
            "larger": "two-refined-bodies-disconnected-holes-opposite-0.5-translations",
            "combined": "mutually-overlapping-mixed-resolution-bodies-two-layer-fringe-narrow-boundary-gaps-donor-competition",
        },
        "routes": {
            "moving_scalar_h1": "actual-P1-FE-interpolation-explicitly-nonconservative",
            "moving_cell_average": "separately-certified-quality-limited-background-ALE-complete-overlap-state-remap",
            "pde_update": "multiplicative-overlapping-Schwarz-composite-Poisson-runtime-lift",
        },
        "manufactured_solution": "u=5+x+y-(x^2+y^2)/4, -Laplace(u)=1, Dirichlet u on background exterior and every protected cavity wall",
        "error_quantity": "independent-manufactured-composite-overset-L2-error",
        "evidence_sections": [
            "parts",
            "native_phases",
            "connectivity",
            "epochs",
            "conservative_profile",
            "refused_overreach_motion",
            "restart",
            "workflow_obligations",
            "physical_error",
            "time_to_target_error_seconds",
            "stages_seconds",
            "peak_process_resident_bytes",
            "device_memory",
            "attempts",
        ],
        "nonclaims": [
            "Partition of unity is not inventory conservation.",
            "The separate background cell-average ALE route does not certify moving FE interpolation.",
            "Wall and exterior reactions are compared with analytic fluxes; the global discrete balance identity is gated.",
            "Profile completion does not complete the full corpus, release, or leadership gates.",
        ],
    }


def _periodic_qualification_profile_contracts() -> dict[str, dict[str, Any]]:
    """Keep historical binary64 exclusions separate from exact-carrier positives."""
    profiles = (
        ("skew-lattice", "qualification-skew-torus", "skew-lattice", False),
        (
            "skew-material-feature",
            "qualification-skew-material-torus",
            "skew-material-feature",
            False,
        ),
        (
            "skew-lattice-exact-algebraic",
            "qualification-skew-torus-exact-algebraic",
            "skew-lattice",
            True,
        ),
        (
            "skew-material-feature-exact-algebraic",
            "qualification-skew-material-torus-exact-algebraic",
            "skew-material-feature",
            True,
        ),
    )
    return {
        name: {
            "source_id": source_id,
            "base_case": base,
            "carrier_representation": "exact-algebraic" if positive else "binary64",
            "expected_admission": "positive" if positive else "refusal",
            "expected_outcome": "passed" if positive else "expected-refusal",
            "full_workflow": positive,
            "contract_scope": (
                "source-preserving-exact-carrier-full-lifecycle"
                if positive
                else "immutable-dyadic-zero-tolerance-statistical-exclusion"
            ),
        }
        for name, source_id, base, positive in profiles
    }


def _periodic_qualification_requirements() -> dict[str, object]:
    """Frozen parser and evidence contract of the periodic lifecycle."""
    return {
        "callable": "_periodic_qualification_lifecycle",
        "profile_contracts": _periodic_qualification_profile_contracts(),
        "required_arguments": {
            "periodic_case": {
                "choices": list(_periodic_qualification_profile_contracts()),
                "default": "skew-lattice-exact-algebraic",
            },
            "resolution": {"minimum": 4, "material_case_requires_even": True},
            "capacity": {"minimum": 1},
            "adaptation_rounds": {"minimum": 1},
            "repeats": {"minimum": 1},
            "timeout": {"exclusive_minimum": 0.0, "finite": True},
            "target_error": {"exclusive_minimum": 0.0, "finite": True},
        },
        "adaptation_rounds_semantics": "complete-refinement-and-coarsening-cycles",
        "evidence_sections": [
            "case",
            "request",
            "generation",
            "exact_source_oracle",
            "profile_obligations",
            "checkpoint",
            "epochs",
            "transitions",
            "native_phases",
            "stages_seconds",
            "physical_error",
            "time_to_target_error_seconds",
            "peak_process_resident_bytes",
        ],
        "nonclaims": [
            "No layer, rotational, curved, or full mandatory corpus completion.",
            "No release or performance leadership inferred from a bounded run.",
            "Historical binary64 zero-tolerance exclusions are never positive completions.",
        ],
    }


class FamilyProfileRecord(TypedDict):
    source: str
    route: phx.meshing.NativeMeshingRoute
    family: str
    dimension: int
    ambient_dimension: NotRequired[int]
    geometry_order: NotRequired[int]
    core_fraction: NotRequired[float]
    family_policy: NotRequired[dict[str, Any]]
    frozen_controls: NotRequired[dict[str, Any]]
    nonsweepability_required: NotRequired[bool]


class FamilyRequirementsRecord(TypedDict):
    scenario: str
    parser_flag: str
    parser_choices: tuple[str, ...]
    parser_default: str
    profiles: dict[str, FamilyProfileRecord]
    controls: dict[str, Any]
    consumer: str
    adaptation: str
    state: str
    nonclaims: list[str]


class HybridProfileContractRecord(TypedDict):
    expected_admission: str
    expected_outcome: str
    contract_scope: str
    full_workflow: bool
    prerequisites: NotRequired[tuple[str, ...]]
    source_controls: NotRequired[dict[str, int | float]]


class HybridRequirementsRecord(TypedDict):
    callable: str
    profile_contracts: dict[str, HybridProfileContractRecord]
    required_arguments: dict[str, dict[str, Any]]
    schedule_constraint: str
    adaptation_rounds_semantics: str
    evidence_sections: list[str]
    mandatory_workflow_completion: str
    nonclaims: list[str]


def _family_qualification_requirements() -> FamilyRequirementsRecord:
    """Declare source/family profiles rather than a Cartesian capability product."""
    profiles: dict[str, FamilyProfileRecord] = {
        "dual-quad": {
            "source": "NativePlanarSource:holed-plate",
            "route": "planar_dual_quad",
            "family": "quadrilateral",
            "dimension": 2,
        },
        "curved-quarter-extrusion-quad": {
            "source": "NativeSurfaceSource:original-quarter-extrusion",
            "route": "parametric_surface",
            "family": "quadrilateral",
            "dimension": 2,
            "ambient_dimension": 3,
            "geometry_order": 3,
        },
        "dual-hex": {
            "source": "NativePlcSource:two-material-cavity",
            "route": "plc_dual_hex",
            "family": "hexahedron",
            "dimension": 3,
        },
        "hex-dominant": {
            "source": "NativePlcSource:two-material-cavity",
            "route": "plc_hex_dominant",
            "family": "hexahedron",
            "dimension": 3,
            "geometry_order": 1,
            "core_fraction": 0.8,
            "family_policy": {
                "required": ("hexahedron",),
                "preferred": (),
                "allowed_transitions": ("pyramid", "tetrahedron"),
                "allow_mixed": True,
            },
            "frozen_controls": {
                "exact_pure_family": False,
                "allow_mixed": True,
                "allowed_transitions": ("pyramid", "tetrahedron"),
                "core_fraction": 0.8,
            },
        },
        "transfinite-quad": {
            "source": "NativeStructuredSource:single-transfinite-block",
            "route": "structured_transfinite",
            "family": "quadrilateral",
            "dimension": 2,
        },
        "multiblock-quad": {
            "source": "NativeStructuredSource:two-glued-transfinite-blocks",
            "route": "structured_transfinite",
            "family": "quadrilateral",
            "dimension": 2,
        },
        "sweep-hex": {
            "source": "NativeSweepSource:declared-square-extrusion",
            "route": "sweep",
            "family": "hexahedron",
            "dimension": 3,
        },
        "sweep-prism": {
            "source": "NativeSweepSource:declared-triangle-extrusion",
            "route": "sweep",
            "family": "prism",
            "dimension": 3,
        },
        "curved-frame-sweep-hex": {
            "source": "NativeSweepSource:authored-curved-profile-absolute-frames",
            "route": "sweep",
            "family": "hexahedron",
            "dimension": 3,
            "geometry_order": 2,
        },
        "multiblock-reversed-hex": {
            "source": "NativeStructuredSource:reversed-logical-face-boxes",
            "route": "structured_transfinite",
            "family": "hexahedron",
            "dimension": 3,
        },
        "multiblock-material-hex": {
            "source": "NativeStructuredSource:reversed-two-material-boxes",
            "route": "structured_transfinite",
            "family": "hexahedron",
            "dimension": 3,
        },
        "multiblock-cycle-quad": {
            "source": "NativeStructuredSource:source-circle-annular-block-cycle",
            "route": "structured_transfinite",
            "family": "quadrilateral",
            "dimension": 2,
        },
        "curved-nonsweepable-hex": {
            "source": "NativeMappedHexSource:curved-six-arm-junction",
            "route": "mapped_frame_grid_hex",
            "family": "hexahedron",
            "dimension": 3,
            "geometry_order": 2,
        },
        "curved-cavity-hex": {
            "source": "NativeMappedHexSource:curved-enclosed-cavity",
            "route": "mapped_balanced_grid_hex",
            "family": "hexahedron",
            "dimension": 3,
            "geometry_order": 2,
        },
        "curved-material-hex": {
            "source": "NativeMappedHexSource:curved-two-material-cavity",
            "route": "mapped_frame_grid_hex",
            "family": "hexahedron",
            "dimension": 3,
            "geometry_order": 2,
        },
        "nonsweepable-dual-hex": {
            "source": "NativePlcSource:unit-tetrahedron-two-enclosed-tetrahedral-cavities",
            "route": "plc_dual_hex",
            "family": "hexahedron",
            "dimension": 3,
            "nonsweepability_required": True,
        },
    }
    for profile in profiles.values():
        profile.setdefault("nonsweepability_required", False)
    return {
        "scenario": "native-family-lifecycle",
        "parser_flag": "--family-case",
        "parser_choices": tuple(profiles),
        "parser_default": "dual-hex",
        "profiles": profiles,
        "controls": {
            "geometry_order": "exact independently authored profile order (1, 2, or 3)",
            "exact_pure_family": True,
            "allow_mixed": False,
            "allowed_transitions": (),
            "size_strength": "hard",
            "size_target": "1/resolution",
            "maximum_edge": "2/resolution",
            "size_relative_tolerance": 0.95,
            "minimum_scaled_jacobian": 1.0e-8,
            "minimum_mean_ratio": 0.01,
            "maximum_aspect_ratio": 100.0,
            "surface_minimum_angle": 0.01,
            "limits": "all MeshingLimits frozen from capacity and timeout",
        },
        "consumer": "geometry-aware scalar Q1 manufactured diffusion",
        "adaptation": "canonical NATIVE_MIXED refine then complete-sibling coarsen per round",
        "state": "actual accepted Q1 solution; 3D canonical Q2 H1/DG1/Hcurl2/Hdiv2 two-component scalar/history moments and complete stationary Euler content, retained each epoch",
        "nonclaims": [
            "The swept box is only a declared sweep profile, never a general nonsweepable all-hex case.",
            "The PL material cavity is routine exact-family evidence, not broad curved all-hex closure.",
            "Mandatory larger curved/nonsweepable, feature/alignment and combined-control corpus remains incomplete.",
            "H1 coarsening interpolates nodal state; no general H1 content-conservation claim.",
        ],
    }


_DISTRIBUTED_GRAPH_IMBALANCE = 1.5


_DISTRIBUTED_ARCHIVE_LIMITS: dict[str, int] = {
    "max_members": 32768,
    "max_manifest_nesting": 128,
    "max_manifest_bytes": 128 * 1024 * 1024,
    "max_central_directory_bytes": 8 * 1024 * 1024,
}


_DISTRIBUTED_REPOSITORY_POLICY: dict[str, int] = {
    "maximum_chunk_bytes": 16384,
    "maximum_metadata_bytes": 128 * 1024 * 1024,
}


_DISTRIBUTED_FIELD_ROLES: tuple[
    tuple[str, str, str, str, str, str, int, bool, str, str | None], ...
] = (
    (
        "temperature",
        "allocated-temperature",
        "coefficients",
        "stage-history",
        "mesh/vertex_ids",
        "vertex",
        0,
        True,
        "1+x+2*y",
        "kelvin",
    ),
    (
        "material-temperature",
        "allocated-material-temperature",
        "material-history",
        "material-history",
        "mesh/vertex_ids",
        "vertex",
        1,
        False,
        "37+3*x-y",
        "kelvin",
    ),
    (
        "essential-values",
        "allocated-essential-values",
        "boundary-values",
        "stage-history",
        "mesh/vertex_ids",
        "vertex",
        2,
        True,
        "1+x+2*y",
        "kelvin",
    ),
    (
        "essential-mask",
        "allocated-essential-mask",
        "boundary-mask",
        "stage-history",
        "mesh/vertex_ids",
        "vertex",
        3,
        True,
        "boundary-dof-mask",
        None,
    ),
    (
        "material-cell-average",
        "allocated-material-cell-average",
        "material-history",
        "material-history",
        "mesh/cell_ids",
        "cell",
        0,
        False,
        "cell-mean(37+3*x-y)",
        "kelvin",
    ),
    (
        "rng-counter",
        "allocated-rng-counter",
        "rng-counter",
        "rng-counter",
        "mesh/cell_ids",
        "cell",
        1,
        False,
        "rademacher(derive_key(wrap_key_data(rng-root-key, threefry2x32), "
        "SampleAddress('phydrax.meshing.qualification.distributed', "
        "'material-history-probe', target=('temperature', 'material-cell-average'), "
        "role='rademacher'), id_hi, id_lo, counter_hi, counter_lo), (8,)); "
        "retained slots keep counters, fresh children start at 0, "
        "coarsened parents resume their retained counter",
        None,
    ),
    (
        "euler-conserved",
        "allocated-euler-conserved",
        "stage-history",
        "stage-history",
        "mesh/cell_ids",
        "cell",
        2,
        False,
        "accepted-Euler-conservative-native-lineage",
        "euler-conserved",
    ),
)


def _distributed_qualification_requirements() -> dict[str, Any]:
    """Freeze placement, source, field, and resource obligations."""

    return {
        "callable": "_distributed_qualification_lifecycle",
        "required_arguments": {
            "resolution": {"minimum": 1, "default": 4},
            "capacity": {"minimum": 1, "default": 20000},
            "timeout": {"exclusive_minimum": 0.0, "finite": True, "default": 120.0},
            "maximum_work_units": {"minimum": 1, "nullable": True},
            "maximum_geometry_queries": {"minimum": 1, "nullable": True},
            "maximum_memory_bytes": {"minimum": 1, "nullable": False},
            "repeats": {"minimum": 1, "default": 3},
            "target_error": {"exclusive_minimum": 0.0, "finite": True, "default": 0.05},
            "adaptation_rounds": {"minimum": 1, "default": 1},
            "distributed_parts": {"minimum": 2, "default": 2},
            "restart_parts": {"minimum": 1, "default": 1},
            "distributed_processes": {"minimum": 2, "default": 2},
            "distributed_source": {
                "choices": ["native-unit-square"],
                "default": "native-unit-square",
            },
            "distributed_bisection_compatibility": {
                "choices": [mode.value for mode in BisectionCompatibility],
                "default": BisectionCompatibility.REJECT.value,
                "meaning": "matching-label preparation charged to the request limits",
            },
            "distributed_halo_width": {"minimum": 1, "default": 2},
            "distributed_halo_capacity": {"minimum": 1, "default": 20000},
            "distributed_cavity_capacity": {"minimum": 1, "default": 20000},
            "distributed_slack": {"minimum": 1, "default": 2},
            "distributed_host_packet_bytes": {"minimum": 1, "default": 64 * 1024 * 1024},
            "distributed_checkpoint_root": {
                "default": None,
                "nullable": False,
                "meaning": "shared POSIX repository written by the producer "
                "processes and read by the fresh restore process",
            },
        },
        "placement": {
            "generation": "one actual native provider result in every producer process",
            "adaptation": "one accepted owner-local partition per actual JAX producer process",
            "restart": "fresh restore process set with a changed owner count",
            "launch": "a single-process invocation launches the producer processes and then restores",
            "unavailable_requested_placement": "failed, never simulated",
            "construction_route": "native host geometry and exact predicates with device-local topology",
        },
        "fields": [
            {
                "name": name,
                "association": association,
                "role": role,
                "raw_history": raw,
                "reference": reference,
                "units": units,
            }
            for name, raw, role, _, _, association, _, _, reference, units in _DISTRIBUTED_FIELD_ROLES
        ],
        "fv_state": {
            "system": "Euler-2D",
            "density": "material-cell-average/38",
            "velocity": [0.0, 0.0],
            "pressure": 1.0,
            "boundary": "slip-wall",
            "initial_condition_scope": "epoch-zero",
            "conserved_components": [
                "density",
                "momentum_x",
                "momentum_y",
                "total_energy",
            ],
            "component_units": ["kg/m^3", "kg/(m^2*s)", "kg/(m^2*s)", "Pa"],
            "continuation": "accepted-state-required",
            "required_transfer": "conservative-and-positive",
        },
        "source": {
            "fixture": "native-unit-square",
            "area": 1.0,
            "perimeter": 4.0,
            "spatial_units": "meter",
            "retained": "actual authored NativePlanarSource",
        },
        "archive_limits": dict(_DISTRIBUTED_ARCHIVE_LIMITS),
        "graph_restart_repartition": {
            "maximum_imbalance": _DISTRIBUTED_GRAPH_IMBALANCE,
            "vertex_weights": "1+cell_id%3",
        },
        "repository_policy": dict(_DISTRIBUTED_REPOSITORY_POLICY),
        "mandatory_stages": [
            "source_preparation",
            "geometry_constraint_preparation",
            "native_generation_publication_certification",
            "owner_local_tile_ownership",
            "initial_fe_halo_preparation",
            "initial_fe_preparation",
            "initial_fe_solve",
            "initial_fields_integral_validation",
            "initial_fv_preparation",
            "initial_fv_halo_preparation",
            "initial_fv_execution_preparation",
            "initial_owner_local_fv",
            "owner_local_refine",
            "owner_local_coarsen",
            "all_field_history_projection",
            "native_graph_restart_repartition",
            "graph_field_history_recovery",
            "process_local_checkpoint",
            "changed_owner_count_restore",
            "source_recertification",
            "changed_owner_count_repack",
            "restored_field_history_recovery",
            "restored_closure_rearchive",
            "continued_fe_solve",
            "continued_owner_local_fv",
        ],
        "required_evidence": [
            "source_id",
            "source_revision",
            "source_binding_id",
            "request_id",
            "control_ids",
            "field_ids",
            "runtime",
            "actual_topology",
            "epochs",
            "native_phases",
            "stages_seconds",
            "communication",
            "migration",
            "checkpoint",
            "restart",
            "stable_ids",
            "conservation",
            "physical_error",
            "time_to_target_error_seconds",
            "compiler",
            "retained_bytes",
            "peak_process_resident_bytes",
            "device_memory",
            "attempts",
            "failure",
        ],
        "nonclaims": [
            "Process-local generation candidates are not a distributed accepted carrier.",
            "A nodal interpolation certificate is not cell inventory conservation.",
            "A routine case does not establish release or distributed performance leadership.",
        ],
    }


_hybrid_qualification_original_profile_names = (
    "plate5",
    "quadrilateral_plate3",
    "opposing_sloped_plates4",
    "sphere_radius04_subdivision2",
    "outward_box_count3",
    "concave_box_count3",
    "wedge30_rejection",
    "open_box_floor_count4",
    "channel_gap005_rejection",
    "channel_gap005_policies",
)


_hybrid_qualification_original_controls = {
    "plate5": "plate(5); geometric(3, 0.01, growth_rate=1.2); ADVANCING",
    "quadrilateral_plate3": "linspace(0,1,4), 9 quadrilaterals; geometric(3,0.01,1.2); ADVANCING",
    "opposing_sloped_plates4": "plate(4), reversed plate(4), upper z=0.03+0.17*x; TERMINATE_LOCALLY; ADVANCING",
    "sphere_radius04_subdivision2": "icosphere(0.4,2), outward; ADVANCING",
    "outward_box_count3": "box((-0.3,-0.3,-0.3),(0.3,0.3,0.3),3,outward=True); ADVANCING",
    "concave_box_count3": "box((0,0,0),(1,1,1),3,outward=False); ADVANCING",
    "wedge30_rejection": "count=3; angle=deg2rad(30); both sheets face inside; ADVANCING default collision",
    "open_box_floor_count4": "box((0,0,0),(1,1,0.5),4,outward=False); remove top; floor-only scope; ADVANCING",
    "channel_gap005_rejection": "opposed plate(4) gap=0.05; ADVANCING default collision",
    "channel_gap005_policies": "opposed plate(4) gap=0.05; REDUCE_THICKNESS, TERMINATE_LOCALLY, MERGE; minimum_thickness_fraction=0.2; ADVANCING",
}


def _hybrid_qualification_original_wall_profiles() -> tuple[str, ...]:
    """Enumerate independent source profiles; do not execute them."""
    return _hybrid_qualification_original_profile_names


def _hybrid_qualification_profiles() -> tuple[str, ...]:
    return (
        "unit-box",
        "curved-periodic-layer-core",
        "original-curved-periodic-refusal",
        "archived-layer-core",
        "concave-box-core-lifecycle",
        *_hybrid_qualification_original_wall_profiles(),
        "cad-slab-and-core",
        "cad-entire-solid-slab",
        "cad-outside-controlled-solid",
        "cad-overlapping-wall-slabs",
        *(
            f"periodic-{periodicity}-{kind}"
            for periodicity in ("translation", "partial", "rotation")
            for kind in ("triangle", "quadrilateral")
        ),
        "periodic-convex-ridge",
        *(
            f"exact-sweep-{periodicity}-{outcome}"
            for periodicity in ("nonperiodic", "periodic")
            for outcome in ("accepted", "perturbed-cap")
        ),
    )


def _hybrid_qualification_profile_contracts() -> dict[str, HybridProfileContractRecord]:
    contracts: dict[str, HybridProfileContractRecord] = {
        profile: {
            "expected_admission": "positive",
            "expected_outcome": "passed",
            "contract_scope": "wall-layer-source",
            "full_workflow": False,
        }
        for profile in _hybrid_qualification_profiles()
    }
    contracts["unit-box"].update(
        contract_scope="native-wall-core-curving-adaptation-H1-diffusion"
    )
    contracts["archived-layer-core"].update(
        contract_scope="original-mixed-lineage-H1Q2-DG1-Hcurl2-Hdiv2-FV-history-PDE-defaultcold",
        full_workflow=True,
        prerequisites=("hybrid_archive", "hybrid_content_id"),
    )
    contracts["concave-box-core-lifecycle"].update(
        contract_scope="original-concave-wall-core-highorder-AMR-H1Q2-DG1-Hcurl2-Hdiv2-FV-history-PDE-defaultcold",
        full_workflow=True,
        prerequisites=(
            "hybrid_core_size",
            "hybrid_region_densities",
            "hybrid_material_site",
        ),
    )
    contracts["curved-periodic-layer-core"].update(
        contract_scope=(
            "corrected-curved-periodic-ADVANCING-layer-core-highorder-AMR-"
            "H1Q2-DG1-Hcurl2-Hdiv2-FV-history-PDE-defaultcold"
        ),
        full_workflow=True,
        source_controls={
            "layer_count": 2,
            "first_thickness": 0.01,
            "growth_rate": 1.0,
            "period_x": 1,
            "geometry_order": 2,
        },
    )
    contracts["concave_box_count3"].update(contract_scope="wall-layer-cap-core-source")
    for profile in (
        "cad-slab-and-core",
        "cad-entire-solid-slab",
        "cad-outside-controlled-solid",
        "cad-overlapping-wall-slabs",
    ):
        contracts[profile].update(contract_scope="native-CAD-exact-partition-source")
    for profile in (
        "exact-sweep-nonperiodic-accepted",
        "exact-sweep-nonperiodic-perturbed-cap",
        "exact-sweep-periodic-accepted",
        "exact-sweep-periodic-perturbed-cap",
    ):
        contracts[profile].update(contract_scope="independent-source-exact-sweep")
    for profile in (
        "original-curved-periodic-refusal",
        "wedge30_rejection",
        "channel_gap005_rejection",
        "cad-outside-controlled-solid",
        "cad-overlapping-wall-slabs",
        "exact-sweep-nonperiodic-perturbed-cap",
        "exact-sweep-periodic-perturbed-cap",
    ):
        contracts[profile].update(
            expected_admission="refusal", expected_outcome="expected-refusal"
        )
    return contracts


def _hybrid_qualification_requirements() -> HybridRequirementsRecord:
    """Bounded caller/parser contract, independent of mandatory-corpus claims."""
    return {
        "callable": "_hybrid_qualification_lifecycle",
        "profile_contracts": _hybrid_qualification_profile_contracts(),
        "required_arguments": {
            "hybrid_profile": {
                "choices": list(_hybrid_qualification_profiles()),
                "default": "unit-box",
            },
            "hybrid_archive": {"required_for": "archived-layer-core"},
            "hybrid_content_id": {"required_for": "archived-layer-core"},
            "hybrid_core_size": {
                "default": 0.1,
                "exclusive_minimum": 0.0,
                "finite": True,
            },
            "hybrid_region_densities": {
                "default": [2.0, 3.0],
                "source_authoritative": True,
                "single_fluid_profiles": ["unit-box", "concave-box-core-lifecycle"],
                "single_fluid_default": [2.0],
            },
            "hybrid_material_site": {
                "default": "two-material-density",
                "source_authoritative": True,
                "single_fluid_default": "fluid-density",
            },
            "resolution": {"minimum": 2},
            "capacity": {"minimum": 1},
            "layer_count": {"minimum": 1, "default": 2},
            "first_thickness": {"exclusive_minimum": 0.0, "finite": True, "default": 0.1},
            "layer_growth": {"minimum": 1.0, "finite": True, "default": 1.0},
            "geometry_order": {"choices": [2, 3, 4, 6, 10], "default": 2},
            "adaptation_rounds": {"minimum": 1},
            "repeats": {"minimum": 1},
            "timeout": {"exclusive_minimum": 0.0, "finite": True},
            "target_error": {"exclusive_minimum": 0.0, "finite": True},
        },
        "schedule_constraint": "sum(first_thickness * layer_growth**k) < 1",
        "adaptation_rounds_semantics": "complete-mixed-refine-and-coarsen-cycles",
        "evidence_sections": [
            "case",
            "request",
            "generation",
            "curving",
            "epochs",
            "transitions",
            "rejection_retention",
            "invariants",
            "native_phases",
            "stages_seconds",
            "physical_error",
            "time_to_target_error_seconds",
            "peak_process_resident_bytes",
            "device_memory",
        ],
        "mandatory_workflow_completion": "incomplete",
        "nonclaims": [
            "The unit-box routine does not qualify nonlinear curved-wall geometry.",
            "The immutable original curved source remains an exact periodic refusal.",
            "The corrected profile does not relabel or repair the original source.",
            "A high-order map certificate is not a corners-only validity check.",
            "No release or performance leadership is inferred.",
        ],
    }


def _implicit_qualification_requirements() -> dict[str, object]:
    """Parser and evidence contract for the analytic restricted volume loop."""
    return {
        "callable": "_implicit_qualification_lifecycle",
        "required_arguments": {
            "implicit_case": {"choices": IMPLICIT_CASES, "default": "sphere"},
            "resolution": {"minimum": 4},
            "capacity": {"minimum": 64},
            "repeats": {"minimum": 1},
            "adaptation_rounds": {"minimum": 1},
            "timeout": {"exclusive_minimum": 0.0, "finite": True},
            "target_error": {"exclusive_minimum": 0.0, "finite": True},
            "maximum_memory_bytes": {"minimum": 1},
            "implicit_radius": {
                "default": 0.371,
                "exclusive_minimum": 0.0,
                "finite": True,
            },
            "implicit_fidelity": {
                "default": 0.02,
                "exclusive_minimum": 0.0,
                "finite": True,
            },
            "implicit_maximum_level": {"default": 9, "minimum": 3, "maximum": 16},
            "implicit_edge_isolation_depth": {"default": 8, "minimum": 0},
            "implicit_root_tolerance": {
                "default": 1.0e-10,
                "exclusive_minimum": 0.0,
                "finite": True,
            },
            "implicit_refinement_rounds": {"default": 8, "minimum": 1},
            "implicit_improvement_passes": {"default": 8, "minimum": 1},
            "implicit_metric_passes": {"default": 16, "minimum": 1},
        },
        "restricted_dual_maximum_depth": 32,
        "restricted_dual_depth_owner": "_construct_implicit_restriction",
        "evidence_sections": [
            "source",
            "discovery",
            "request",
            "generation",
            "epochs",
            "native_phases",
            "stages_seconds",
            "failed_positive_attempts",
            "physical_error",
            "time_to_target_error_seconds",
            "peak_process_resident_bytes",
            "checkpoint",
        ],
        "hard_timeout_owner": "isolated qualification subprocess",
        "nonclaims": [
            "No generic Boolean/neural/periodic implicit-volume admission.",
            "No workset is a published volume or positive completion.",
            "No full mandatory corpus, release, or performance leadership claim.",
        ],
    }


def _design_qualification_requirements() -> dict[str, object]:
    """Expose the design owner's authored cases without duplicating their physics."""
    from phydrax.meshing.qualification._design_oracle import (
        design_qualification_cases,
        DESIGN_QUALIFICATION_DEFAULT_CASE,
    )

    return {
        "callable": "_design_qualification_lifecycle",
        "required_arguments": {
            "design_case": {
                "choices": DESIGN_CASES,
                "default": DESIGN_QUALIFICATION_DEFAULT_CASE,
            },
        },
        "profiles": design_qualification_cases(),
        "nonclaims": [
            "Fixed-epoch derivatives do not differentiate native topology events.",
            "A proposal is not accepted until its native event and physical reanalysis pass.",
        ],
    }


def _reconstruction_qualification_requirements() -> dict[str, object]:
    """Declare the reconstruction cases and the selected repair semantics."""
    return {
        "callable": "_reconstruction_qualification_lifecycle",
        "required_arguments": {
            "reconstruction_case": {
                "choices": RECONSTRUCTION_CASES,
                "default": "lobed-sphere",
            },
        },
        "domain": "underspecified-defective-sampled-closed-surface",
        "research_gate": "underspecified-defective-surface-reconstruction",
        "nonclaims": [
            "Repair selects a represented source; it does not recover a unique unknown original.",
        ],
    }
