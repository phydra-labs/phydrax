# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Unreleased, evidence-gated meshfree capability declarations.

Every support tuple is one exact conjunction of method, boundary, geometry
authority, derivative class, provider, device, declared capacity, precision,
temporal method, reproducibility, and spatial dimension. Evidence for one tuple
never stands in for another, and no declaration here is a release.
"""

from __future__ import annotations

from itertools import product

from ...qualification._registry import CapabilityProfile, SupportTuple


type SupportValue = str | int | bool
type SupportAttributes = dict[str, SupportValue]
type ProfileSpecification = tuple[str, tuple[SupportAttributes, ...], tuple[str, ...]]

_PROVIDER = "phydrax-native"
_SEEDED = "deterministic-seeded"


def meshfree_support(
    *,
    dimension: int,
    method: str,
    boundary: str,
    geometry_authority: str,
    derivative_class: str,
    capacity: int,
    precision: str,
    temporal_method: str = "none",
    provider: str = _PROVIDER,
    device: str = "cpu",
    reproducibility: str = _SEEDED,
) -> SupportAttributes:
    """One exact meshfree support tuple; every coordinate is explicit."""
    if type(dimension) is not int or dimension < 1:
        raise ValueError("dimension must be a positive integer.")
    if type(capacity) is not int or capacity < 1:
        raise ValueError("capacity must be a positive declared point capacity.")
    if precision not in ("float32", "float64", "mixed"):
        raise ValueError("precision must be float32, float64, or mixed.")
    return {
        "dimension": dimension,
        "method": method,
        "boundary": boundary,
        "geometry-authority": geometry_authority,
        "derivative-class": derivative_class,
        "provider": provider,
        "device": device,
        "capacity": capacity,
        "precision": precision,
        "temporal-method": temporal_method,
        "reproducibility": reproducibility,
    }


def _strong_form() -> tuple[SupportAttributes, ...]:
    return tuple(
        meshfree_support(
            dimension=dimension,
            method=method,
            boundary="none",
            geometry_authority="unit-cube-stratified-samples",
            derivative_class="coordinate-orders-1-2",
            capacity=16384,
            precision=precision,
        )
        for dimension, method, precision in product(
            (1, 2, 3), ("gmls", "phs-rbf-fd"), ("float64", "float32")
        )
        # Float32 PHS saddles exceed the float32 fit admission (condition times
        # eps32 at most 1e-3, rounding-level moments) and are refused: unsupported.
        if (method, precision) != ("phs-rbf-fd", "float32")
    )


def _elliptic() -> tuple[SupportAttributes, ...]:
    boundaries = ("dirichlet", "neumann", "robin", "mixed")
    collocated = tuple(
        meshfree_support(
            dimension=dimension,
            method=f"{stencil}-collocated",
            boundary=boundary,
            geometry_authority="cartesian-box-perturbed-samples",
            derivative_class="none",
            capacity=16384,
            precision="float64",
        )
        for dimension, stencil, boundary in product(
            (1, 2, 3), ("phs-rbf-fd", "gmls"), boundaries
        )
    )
    # This support is an explicitly bound native point-primary tensor SBP
    # realization, not arbitrary-cloud constrained GMLS. The prepared native
    # second-order families own derivatives, volume norm and face cubature.
    # Historical unstable constrained-degree-one artifacts are separate failed
    # evidence and cannot qualify this new support identity.
    dissipative = tuple(
        meshfree_support(
            dimension=dimension,
            method="native-tensor-sbp-interior-order-2-dissipative",
            boundary=boundary,
            geometry_authority="native-point-primary-tensor-grid-sbp-norm",
            derivative_class="none",
            capacity=16384,
            precision="float64",
        )
        for dimension, boundary in product((1, 2, 3), boundaries)
    )
    return collocated + dissipative


def _multilevel() -> tuple[SupportAttributes, ...]:
    return tuple(
        meshfree_support(
            dimension=dimension,
            method=f"gmres-{preconditioner}",
            boundary="dirichlet-convex-hull",
            geometry_authority="unit-cube-stratified-samples",
            derivative_class="none",
            capacity=16384,
            precision="float64",
        )
        for dimension, preconditioner in product(
            (2, 3), ("native-ilu", "native-smoothed-aggregation", "meshfree-multilevel")
        )
    )


def _conservative_exterior() -> tuple[SupportAttributes, ...]:
    return tuple(
        meshfree_support(
            dimension=dimension,
            method="edge-moment-metric-exact-signed-and-nonnegative-conic",
            boundary="dirichlet-incomplete-axial-neighborhood",
            geometry_authority="cartesian-nodal-control-volumes",
            derivative_class="fixed-topology-coordinate",
            capacity=4096,
            precision="float64",
        )
        for dimension in (1, 2, 3)
    )


def _surface_operators() -> tuple[SupportAttributes, ...]:
    return tuple(
        meshfree_support(
            dimension=3,
            method=f"surface-gmls-laplace-beltrami-degree-{degree}",
            boundary="closed-surface",
            geometry_authority=f"{surface}-{geometry}",
            derivative_class="none",
            capacity=16384,
            precision="float64",
        )
        for surface, geometry, degree in product(
            ("sphere", "torus"), ("implicit-level-set", "sampled-fit"), (2, 4)
        )
    )


def _moving_surface() -> tuple[SupportAttributes, ...]:
    return (
        meshfree_support(
            dimension=3,
            method="surface-material-ale-reaction-diffusion",
            boundary="closed-surface",
            geometry_authority="prescribed-expanding-sphere",
            derivative_class="none",
            capacity=4096,
            precision="float64",
            temporal_method="additive-imex",
        ),
    )


def _bulk_surface_exchange() -> tuple[SupportAttributes, ...]:
    return (
        meshfree_support(
            dimension=3,
            method="langmuir-bulk-surface-exchange",
            boundary="bulk-surface-interface",
            geometry_authority="sphere-interface-in-box",
            derivative_class="none",
            capacity=4096,
            precision="float64",
            temporal_method="explicit-window-buffer",
        ),
    )


def _learned_flux() -> tuple[SupportAttributes, ...]:
    return tuple(
        meshfree_support(
            dimension=dimension,
            method="learned-monotone-edge-flux-native-newton",
            boundary="dirichlet-incomplete-axial-neighborhood",
            geometry_authority="cartesian-nodal-control-volumes",
            derivative_class="implicit-adjoint",
            capacity=1024,
            precision="float64",
        )
        for dimension in (2, 3)
    )


def _bulk_evolution() -> tuple[SupportAttributes, ...]:
    """Collocation GMLS ADR (fixed and ALE clouds) under native temporal methods."""
    fixed = tuple(
        meshfree_support(
            dimension=2,
            method="gmls-degree3-collocation-adr",
            boundary="periodic",
            geometry_authority="periodic-unit-square-jittered-lattice",
            derivative_class="none",
            capacity=4096,
            precision="float64",
            temporal_method=temporal,
        )
        for temporal in ("ssprk33", "imex-ars-222")
    )
    moving = meshfree_support(
        dimension=2,
        method="gmls-degree3-collocation-ale-adr",
        boundary="periodic",
        geometry_authority="periodic-shell-lattice",
        derivative_class="none",
        capacity=4096,
        precision="float64",
        temporal_method="ssprk33",
    )
    return (*fixed, moving)


def _bulk_transport() -> tuple[SupportAttributes, ...]:
    """Conservative edge-graph transport on lattice control volumes."""

    def graph(method: str, boundary: str, temporal: str) -> SupportAttributes:
        return meshfree_support(
            dimension=2,
            method=method,
            boundary=boundary,
            geometry_authority="cartesian-lattice-control-volumes",
            derivative_class="none",
            capacity=4096,
            precision="float64",
            temporal_method=temporal,
        )

    schemes = ("upwind", "reconstructed", "limited")
    return (
        *(
            graph(f"edge-graph-{scheme}", "inflow-outflow", "ssprk33")
            for scheme in schemes
        ),
        *(
            graph(f"edge-graph-{scheme}", "closed-tangential", "ssprk33")
            for scheme in schemes
        ),
        graph("edge-graph-limited-graph-diffusion", "closed-tangential", "imex-ars-222"),
        graph("edge-graph-fixed-topology-refresh", "lattice-boundary-nodes", "none"),
    )


def _joint_transfer() -> tuple[SupportAttributes, ...]:
    """Audited point transfers; capacity is the controlling route count."""
    accepted = tuple(
        (method, "tensor-simpson-unit-cube")
        for method in (
            "point-transfer-conservative-signed",
            "point-transfer-conservative-positive",
            "point-transfer-joint-nonnegative-constant",
            "point-transfer-joint-signed-degree2",
        )
    )
    refused = (
        (
            "point-transfer-conservative-signed",
            "tensor-simpson-unit-cube-3x-refined-source",
        ),
        (
            "point-transfer-joint-radius-routes",
            "tensor-simpson-unit-cube-with-exterior-target",
        ),
        (
            "point-transfer-joint-nonnegative-constant",
            "tensor-simpson-unit-cube-dilated-target-measure",
        ),
        ("point-transfer-joint-signed-degree2", "tensor-midpoint-unit-cube"),
        ("point-transfer-joint-nonnegative-degree1", "tensor-midpoint-unit-cube"),
    )
    return tuple(
        meshfree_support(
            dimension=dimension,
            method=method,
            boundary="none",
            geometry_authority=geometry,
            derivative_class="frozen-transfer-value",
            capacity=65536,
            precision="float64",
        )
        for dimension in (2, 3)
        for method, geometry in (*accepted, *refused)
    )


def _physical_topology() -> tuple[SupportAttributes, ...]:
    """Committed multiregion events consumed as meshfree epochs; capacity is faces."""
    return tuple(
        meshfree_support(
            dimension=3,
            method=method,
            boundary=boundary,
            geometry_authority=geometry,
            derivative_class="frozen-transfer-value",
            capacity=8192,
            precision="float64",
        )
        for method, boundary, geometry in (
            (
                "multiregion-merge-meshfree-epoch",
                "closed-foam-films",
                "multiregion-icosphere-bubbles",
            ),
            (
                "multiregion-pinch-region-split-meshfree-epoch",
                "fixed-ring-boundary",
                "multiregion-catenoid",
            ),
        )
    )


_K_FORM_METHOD = "gmls-p1-k-form-moment-reconstruction-sparse-hodge"


def _higher_forms() -> tuple[SupportAttributes, ...]:
    """Geometry-authorized complexes; edge/face/cell counts are their capacity."""
    return tuple(
        meshfree_support(
            dimension=dimension,
            method=_K_FORM_METHOD,
            boundary=boundary,
            geometry_authority=authority,
            derivative_class="none",
            capacity=capacity,
            precision="float64",
        )
        for dimension, boundary, authority, capacity in (
            (
                2,
                "absolute-and-relative-boundary-complex",
                "cellmesh-holed-square-domain",
                4096,
            ),
            (
                3,
                "absolute-and-relative-boundary-complex",
                "cellmesh-tunneled-slab-domain",
                1024,
            ),
            (3, "closed-surface", "cellmesh-icosphere-closed-surface", 4096),
        )
    )


def _abstract_clique_forms() -> tuple[SupportAttributes, ...]:
    """Research-only abstract radius-clique complexes: no geometry authority."""
    return (
        meshfree_support(
            dimension=2,
            method="radius-clique-exact-integer-chain",
            boundary="none",
            geometry_authority="abstract-radius-clique-no-geometry",
            derivative_class="none",
            capacity=4096,
            precision="float64",
        ),
    )


def _sensitivities() -> tuple[SupportAttributes, ...]:
    routes = (
        (
            (1, 2, 3),
            "gmls-wendland-c2-smooth-envelope-point-cloud",
            "unit-cube-jittered-lattice",
            "fixed-support-coordinate-jvp-vjp",
            4096,
        ),
        (
            (2, 3),
            "gmls-wendland-c2-smooth-envelope-local-stencil",
            "ball-sources-oblique-cutoff-crossing",
            "smooth-support-cutoff-crossing",
            1024,
        ),
        (
            (1, 2, 3),
            "edge-moment-metric-nonnegative-conic",
            "cartesian-lattice-nodal-control-volumes",
            "strict-active-fixed-set-kkt-jvp",
            4096,
        ),
        (
            (1, 2, 3),
            "conservative-positive-point-transfer-live-history-remap",
            "unit-cube-stratified-supports",
            "frozen-remap-value-tangent-adjoint",
            4096,
        ),
        (
            (1, 2, 3),
            "gmls-knn-local-stencil",
            "axis-equidistant-selection-tie",
            "knn-selection-gap",
            64,
        ),
    )
    return tuple(
        meshfree_support(
            dimension=dimension,
            method=method,
            boundary="dirichlet-lattice-boundary"
            if derivative == "strict-active-fixed-set-kkt-jvp"
            else "none",
            geometry_authority=authority,
            derivative_class=derivative,
            capacity=capacity,
            precision="float64",
        )
        for dimensions, method, authority, derivative, capacity in routes
        for dimension in dimensions
    )


def _distributed() -> tuple[SupportAttributes, ...]:
    """Distributed relations/actions per device class and precision.

    Forced host CPU devices carry functional parity only; GPU and multi-host
    tuples are declared performance rows that need real hardware.
    """
    authority = "unit-cube-stratified-samples"
    gmls = ("gmls-laplacian-shell-knn-halo-bind", "open-box")
    smoother = (
        "inverse-quadratic-neighbor-row-smoother-shell-knn-halo",
        "periodic-all-axes",
    )
    forced = tuple(
        meshfree_support(
            dimension=dimension,
            method=method,
            boundary=boundary,
            geometry_authority=authority,
            derivative_class="linear-action-jvp-vjp",
            capacity=4096,
            precision=precision,
            device="cpu-forced-multi-device",
        )
        for dimension in (2, 3)
        for (method, boundary), precisions in (
            (gmls, ("float32", "float64", "mixed")),
            (smoother, ("float32", "float64")),
        )
        for precision in precisions
    )
    refusals = tuple(
        meshfree_support(
            dimension=dimension,
            method="declared-distributed-capacity-ownership-precision-refusals",
            boundary="open-box",
            geometry_authority=authority,
            derivative_class="none",
            capacity=4096,
            precision=precision,
            device="cpu-forced-multi-device",
        )
        for dimension, precision in product((2, 3), ("float32", "float64"))
    )
    hardware = tuple(
        meshfree_support(
            dimension=dimension,
            method=gmls[0],
            boundary=gmls[1],
            geometry_authority=authority,
            derivative_class="linear-action-jvp-vjp",
            capacity=1048576,
            precision=precision,
            device=device,
        )
        for dimension, device, precision in product(
            (2, 3),
            ("gpu", "multi-device-gpu", "multi-host-cpu", "multi-host-gpu"),
            ("float32", "float64", "mixed"),
        )
    )
    return forced + refusals + hardware


_PLATE = "unit-square-jittered-lattice-trapezoid-measure"


def _incompressible_flow() -> tuple[SupportAttributes, ...]:
    """Periodic exterior-complex flow and bounded generalized Stokes (2-D)."""
    return (
        meshfree_support(
            dimension=2,
            method="exterior-projection-reconstructed-transport-phs3",
            boundary="periodic",
            geometry_authority="periodic-square-lattice-minimum-image-charts",
            derivative_class="pressure-projection-jvp-vjp",
            capacity=1024,
            precision="float64",
            temporal_method="ars-222",
        ),
        meshfree_support(
            dimension=2,
            method="generalized-stokes-pspg-phs3-collocated",
            boundary="dirichlet",
            geometry_authority=_PLATE,
            derivative_class="none",
            capacity=1024,
            precision="float64",
        ),
    )


def _lagrangian_flow() -> tuple[SupportAttributes, ...]:
    """Lagrangian GMLS particle flow per measure, and the declared measure transfer."""
    flows = tuple(
        meshfree_support(
            dimension=2,
            method=f"lagrangian-gmls-weak-projection-{measure}",
            boundary="periodic",
            geometry_authority="periodic-unit-square-cell-lattice",
            derivative_class="none",
            capacity=1024,
            precision="float64",
            temporal_method="explicit-predictor-projection",
        )
        for measure in ("quadrature-volume", "material-mass-sph-wendland-c2")
    )
    transfer = meshfree_support(
        dimension=2,
        method="measure-transfer-conservative-positive-material-to-quadrature",
        boundary="periodic",
        geometry_authority="periodic-unit-square-jittered-cell-lattice",
        derivative_class="none",
        capacity=1024,
        precision="float64",
    )
    return (*flows, transfer)


def _elasticity() -> tuple[SupportAttributes, ...]:
    """Small-strain, neo-Hookean and Herrmann mixed collocation on a plate."""
    # (method, boundary, temporal method, capacity). The traction-face
    # manufactured route is declared to the 65^2 = 4225-point lattice of its
    # four-resolution order study (Q11-elasticity-order); the others stay 1024.
    routes = (
        (
            "small-strain-phs3-collocated-gmres-auxiliary",
            "traction-face-dirichlet",
            "none",
            4225,
        ),
        (
            "small-strain-phs3-collocated-gmres-auxiliary",
            "rollers-traction",
            "none",
            1024,
        ),
        (
            "neo-hookean-incremental-newton-phs3",
            "rollers-traction",
            "incremental-load-steps",
            1024,
        ),
        ("herrmann-mixed-pspg-phs3", "dirichlet-clamped", "none", 1024),
    )
    return tuple(
        meshfree_support(
            dimension=2,
            method=method,
            boundary=boundary,
            geometry_authority=_PLATE,
            derivative_class="none",
            capacity=capacity,
            precision="float64",
            temporal_method=temporal,
        )
        for method, boundary, temporal, capacity in routes
    )


def _surface_stokes() -> tuple[SupportAttributes, ...]:
    return (
        meshfree_support(
            dimension=3,
            method="surface-strain-form-stokes-brinkman-gmls-degree-4",
            boundary="closed-surface",
            geometry_authority="sphere-exact-implicit-normals",
            derivative_class="none",
            capacity=4096,
            precision="float64",
        ),
    )


def _fluid_structure() -> tuple[SupportAttributes, ...]:
    """Physical channel step and the smooth manufactured step (body-forced walls)."""
    return tuple(
        meshfree_support(
            dimension=2,
            method="monolithic-mortar-vector-transmission-phs3-ghost-dense-lu",
            boundary=boundary,
            geometry_authority="polygon-chart-authorized-adjacent-rectangles",
            derivative_class="none",
            capacity=capacity,
            precision="float64",
            temporal_method="implicit-quasi-static-step",
        )
        for boundary, capacity in (
            ("paired-interface-dirichlet-walls", 1024),
            ("paired-interface-manufactured-dirichlet-walls", 2178),
        )
    )


_HYBRID_GEOMETRY = "unit-square-frame-cloud-and-interior-fv-grid"


def _adaptive_refinement() -> tuple[SupportAttributes, ...]:
    """Indicator-driven h-adaptivity through epoch transactions; capacity is points."""
    return (
        meshfree_support(
            dimension=2,
            method="phs-rbf-fd-degree3-probe-residual-dorfler-h-refinement",
            boundary="dirichlet",
            geometry_authority="unit-square-jittered-lattice-projected-boundary-children",
            derivative_class="none",
            capacity=2048,
            precision="float64",
        ),
        meshfree_support(
            dimension=3,
            method="phs-rbf-fd-degree3-vs-degree4-indicator-h-refinement-laplace-beltrami",
            boundary="closed-surface",
            geometry_authority="unit-sphere-implicit-level-set-fibonacci",
            derivative_class="none",
            capacity=1024,
            precision="float64",
        ),
    )


def _learned_correction() -> tuple[SupportAttributes, ...]:
    """Learned metric candidates against the full moment operator; capacity is nodes."""
    return tuple(
        meshfree_support(
            dimension=2,
            method=method,
            boundary="dirichlet-incomplete-axial-neighborhood",
            geometry_authority=geometry,
            derivative_class=derivative,
            capacity=capacity,
            precision="float64",
        )
        for method, geometry, derivative, capacity in (
            (
                "edge-moment-metric-learned-candidate-nonnegative-conic-projection",
                "jittered-cartesian-nodal-control-volumes",
                "projection-tangent",
                256,
            ),
            (
                "edge-moment-metric-learned-candidate-signed-minimum-norm-projection",
                "jittered-cartesian-nodal-control-volumes",
                "projection-tangent",
                256,
            ),
            (
                "edge-moment-metric-learned-log-linear-training",
                "cartesian-nodal-control-volumes",
                "implicit-adjoint",
                256,
            ),
            (
                "edge-geometry-empirical-coverage-refuse",
                "cartesian-nodal-control-volumes-refined-query",
                "none",
                1024,
            ),
        )
    )


def _learned_coupled_law() -> tuple[SupportAttributes, ...]:
    """Coupled O(3) edge laws; capacity is the number of edge frames."""
    return tuple(
        meshfree_support(
            dimension=3,
            method=method,
            boundary="none",
            geometry_authority="random-gaussian-edge-frame-ring",
            derivative_class="jacobian-blocks",
            capacity=1024,
            precision="float64",
        )
        for method in (
            "o3-invariant-convex-monotone-coupled-edge-flux",
            "o3-equivariant-lipschitz-coupled-edge-flux",
        )
    )


def _calibrated_prediction() -> tuple[SupportAttributes, ...]:
    """Split-conformal bands on disjoint complete cases; capacity is hybrid unknowns."""
    return (
        meshfree_support(
            dimension=2,
            method="split-conformal-probe-values-hybrid-overlap",
            boundary="dirichlet",
            geometry_authority=_HYBRID_GEOMETRY,
            derivative_class="none",
            capacity=512,
            precision="float64",
        ),
    )


def _hybrid_schwarz() -> tuple[SupportAttributes, ...]:
    """Meshfree/FV overlap Dirichlet coupling; capacity is cloud points plus cells."""
    return (
        meshfree_support(
            dimension=2,
            method="phs-rbf-fd-degree3-cloud-fv-grid-overlap-dirichlet-gmres-schwarz",
            boundary="dirichlet",
            geometry_authority=_HYBRID_GEOMETRY,
            derivative_class="none",
            capacity=4096,
            precision="float64",
        ),
    )


def _runtime_restart() -> tuple[SupportAttributes, ...]:
    """Durable production runs; reproducibility is the declared replay class."""
    return (
        meshfree_support(
            dimension=2,
            method="gmls-degree2-ale-adr-production-run-support-epoch",
            boundary="periodic",
            geometry_authority="periodic-unit-square-jittered-lattice",
            derivative_class="none",
            capacity=256,
            precision="float64",
            temporal_method="ssprk33",
            reproducibility="bitwise-replay",
        ),
        meshfree_support(
            dimension=2,
            method="gmls-degree3-reaction-diffusion-production-cli",
            boundary="periodic",
            geometry_authority="periodic-unit-square-jittered-lattice",
            derivative_class="none",
            capacity=100,
            precision="float64",
            temporal_method="imex-ars-222",
            reproducibility="bitwise-replay",
        ),
        meshfree_support(
            dimension=2,
            method="gmls-degree2-explicit-diffusion-distributed-production-run",
            boundary="none",
            geometry_authority="unit-square-uniform-random-owner-blocked",
            derivative_class="none",
            capacity=256,
            precision="float64",
            temporal_method="explicit-euler",
            device="cpu-forced-multi-device",
            reproducibility="bitwise-transport-reduction-order-continuation",
        ),
        meshfree_support(
            dimension=3,
            method="surface-material-reaction-diffusion-production-run-live-history",
            boundary="closed-surface",
            geometry_authority="prescribed-expanding-sphere",
            derivative_class="none",
            capacity=256,
            precision="float64",
            temporal_method="moving-surface-fixed-step",
            reproducibility="bitwise-replay",
        ),
        meshfree_support(
            dimension=3,
            method="langmuir-bulk-surface-newton-production-run",
            boundary="bulk-surface-interface",
            geometry_authority="sphere-interface-in-box",
            derivative_class="none",
            capacity=256,
            precision="float64",
            temporal_method="implicit-newton-fixed-step",
            reproducibility="bitwise-replay",
        ),
    )


def _specifications() -> tuple[ProfileSpecification, ...]:
    return (
        (
            "strong-form",
            _strong_form(),
            (
                "analytic",
                "convergence-first-derivative",
                "convergence-laplacian",
                "declared-refusal",
                "row-acceptance",
            ),
        ),
        (
            "elliptic-solve",
            _elliptic(),
            (
                "analytic",
                "boundary-equations",
                "convergence-maximum_solution_error",
                "solve-status",
                "true-residual",
            ),
        ),
        (
            "multilevel",
            _multilevel(),
            (
                "analytic",
                "derivative-solve",
                "fine-system-cost",
                "refresh-residual",
                "transfer-reproduction",
                "true-residual",
            ),
        ),
        (
            "conservative-exterior",
            _conservative_exterior(),
            (
                "analytic",
                "coercivity",
                "conservation",
                "coordinate-derivative",
                "feasibility",
                "moment",
                "nonnegative-refusal",
            ),
        ),
        (
            "surface-operators",
            _surface_operators(),
            ("analytic", "convergence", "geometry-admission", "normal-comparison"),
        ),
        (
            "moving-surface",
            _moving_surface(),
            (
                "conservation",
                "dilution",
                "epoch-remap",
                "repeated-remap",
                "repeated-shift",
                "rollback",
                "step-status",
            ),
        ),
        (
            "bulk-surface-exchange",
            _bulk_surface_exchange(),
            ("analytic", "conservation", "window-lag", "window-status"),
        ),
        (
            "learned-constitutive-flux",
            _learned_flux(),
            (
                "adjoint-status",
                "conservation",
                "coverage",
                "implicit-derivative",
                "law-recovery",
                "newton-refusal",
                "primal-status",
            ),
        ),
        (
            "bulk-evolution",
            _bulk_evolution(),
            (
                "ale-gcl-free-stream",
                "ale-gcl-position",
                "ale-gcl-volume",
                "ale-material-minus-mesh",
                "convergence-spatial",
                "rebase-continuation",
                "rollback",
                "rollout-status",
                "state-held",
                "step-refusal",
                "support-refusal-status",
                "temporal-order",
            ),
        ),
        (
            "bulk-transport",
            _bulk_transport(),
            (
                "bound-preservation",
                "candidate-kept-raw",
                "cfl-certificate",
                "cfl-refusal",
                "conservation-ledger",
                "convergence-l1",
                "mass-conservation",
                "metric-admission",
                "negative-state-status",
                "owner-unchanged",
                "positivity",
                "refresh-refused",
                "rollback",
                "rollout-status",
                "state-held",
                "step-refusal",
                "topology-refusal-status",
            ),
        ),
        (
            "joint-transfer",
            _joint_transfer(),
            (
                "admission",
                "conservation-audit",
                "conservation-independent",
                "constant-reproduction",
                "convergence-remap",
                "coverage-witness",
                "declared-status",
                "dual-identity",
                "moment-reproduction",
                "positivity",
                "transfer-withheld",
                "witness-margin",
                "witness-residual",
            ),
        ),
        (
            "physical-topology",
            _physical_topology(),
            (
                "all-history-remap",
                "authority-agreement",
                "ccd-certified",
                "content-conservation",
                "continuation-published",
                "controller-carried",
                "epoch-published",
                "history-conservation",
                "history-route-refusal",
                "physical-lineage",
                "receipt-ledger",
                "region-removed",
                "region-split-lineage",
                "rollback-derivative-withheld",
                "rollback-failed-history",
                "rollback-refused",
                "rollback-source-returned",
                "rollback-values-unchanged",
                "value-derivative",
                "value-vjp",
            ),
        ),
        (
            "higher-forms",
            _higher_forms(),
            (
                "abstract-clique-refusal",
                "betti-audit-refusal",
                "commutation",
                "d-squared-zero",
                "geometry-admission",
                "geometry-authorized-fidelity",
                "harmonic-absolute",
                "hodge-adjoint",
                "hodge-laplace-consumer",
                "hodge-positivity",
                "hodge-reproduction",
                "measure-audit-refusal",
                "orientation-audit-refusal",
                "patch-capacity-refusal",
                "stokes",
            ),
        ),
        (
            "abstract-clique-forms",
            _abstract_clique_forms(),
            (
                "abstract-fidelity",
                "betti-circle",
                "d-squared-zero",
                "simplex-capacity-refusal",
                "work-bound",
                "work-capacity-refusal",
            ),
        ),
        (
            "sensitivities",
            _sensitivities(),
            (
                "contract-identity",
                "derivative-availability",
                "jvp-finite-difference",
                "refusal-nan-jvp",
                "refusal-status",
                "vjp-duality",
            ),
        ),
        (
            "distributed",
            _distributed(),
            (
                "action-parity",
                "conservation",
                "declared-refusal",
                "duality",
                "halo-exactly-once",
                "knn-completeness",
                "migration",
                "precision-roles",
                "radius-pair-once",
                "reductions",
            ),
        ),
        (
            "incompressible-flow",
            _incompressible_flow(),
            (
                "analytic",
                "convergence",
                "conservation",
                "declared-refusal",
                "divergence",
                "spurious-modes",
            ),
        ),
        (
            "lagrangian-flow",
            _lagrangian_flow(),
            ("conservation", "declared-refusal", "divergence", "sph-interoperability"),
        ),
        (
            "elasticity",
            _elasticity(),
            (
                "analytic",
                "convergence",
                "declared-refusal",
                "energy-work",
                "mixed-stability",
                "rigid-modes",
            ),
        ),
        (
            "surface-stokes",
            _surface_stokes(),
            ("analytic", "convergence", "declared-refusal", "killing-modes", "tangency"),
        ),
        (
            "fluid-structure",
            _fluid_structure(),
            ("force", "interface-certificate", "work-balance"),
        ),
        (
            "adaptive-refinement",
            _adaptive_refinement(),
            (
                "capacity-refusal",
                "epoch-conservation",
                "epoch-stability",
                "epochs-published",
                "equal-points-gain",
                "equal-work-gain",
                "error-monotone",
                "proposals-admitted",
                "rejected-coarsening",
                "rejected-coarsening-rollback",
                "solve-status",
            ),
        ),
        (
            "learned-correction",
            _learned_correction(),
            (
                "accepted-updates",
                "adjoint-status",
                "coercivity",
                "correction-admitted",
                "coverage-control-admitted",
                "coverage-refusal",
                "derivative-availability",
                "failed-step-rejected",
                "implicit-gradient",
                "moment-exactness",
                "positivity-conflict",
                "primal-status",
                "sign-margin",
                "trained-admitted",
                "training-loss-reduction",
            ),
        ),
        (
            "learned-coupled-law",
            _learned_coupled_law(),
            (
                "independent-reversal",
                "jacobian-coercivity",
                "jacobian-symmetry",
                "lipschitz-bound",
                "reversal-parity",
                "rotation-covariance",
                "rotoreflection-covariance",
                "strong-monotonicity",
            ),
        ),
        (
            "calibrated-prediction",
            _calibrated_prediction(),
            (
                "cases-accepted",
                "held-out-coverage",
                "shift-no-guarantee",
                "split-disjoint",
            ),
        ),
        (
            "hybrid-schwarz",
            _hybrid_schwarz(),
            (
                "cloud-order",
                "component-residual",
                "grid-order",
                "hybrid-vs-monolithic",
                "overlap-transfer-rows",
                "schwarz-accepted",
                "schwarz-additive-convergence",
                "schwarz-multiplicative-convergence",
            ),
        ),
        (
            "runtime-restart",
            _runtime_restart(),
            (
                "completed",
                "continuation",
                "foreign-case",
                "history-window-refusal",
                "identity-restart-partition-refused",
                "live-history",
                "memory-evidence",
                "migration-committed",
                "partition-changed",
                "placement-only-roles",
                "refusal-evidence-retained",
                "refused-epoch",
                "refused-newton-evidence",
                "replay-class",
                "replay-classification",
                "resume-from-commit",
                "rng-controller",
                "sigkill-interrupt",
                "stale-capacity",
                "stale-geometry",
                "stale-precision",
                "stale-program",
                "stale-source",
                "store-unchanged",
                "truncated-archive",
            ),
        ),
    )


def meshfree_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Declare candidate/research support; campaign availability is not release."""
    common = (
        "independent-reference",
        "independent-review",
        "public-workflow",
        "resource-envelope",
        "scaling",
    )
    return tuple(
        CapabilityProfile(
            f"meshfree.{name}.profile",
            "phydrax",
            tuple(SupportTuple(f"meshfree.{name}", values) for values in supports),
            required_gates=tuple(sorted({*gates, *common})),
            released=False,
        )
        for name, supports, gates in _specifications()
    )


__all__ = ["meshfree_candidate_profiles", "meshfree_support"]
