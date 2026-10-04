#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Built-in source atlas for the omniphysics program."""

from __future__ import annotations

from ._closure_taxonomy import SourceReuseClass
from ._source_audits import _SOURCE_AUDITS
from ._source_reference import (
    PublicationReference,
    SourceAbsorptionLedger,
    SourceReference,
    SourceReview,
)


_SOURCE_SPECS = (
    (
        "adamantine",
        "https://github.com/adamantine-sim/adamantine",
        "Apache-2.0-WITH-LLVM-exception",
        "permissive",
        ("additive-manufacturing", "moving-heat-source", "material-state"),
    ),
    (
        "exa-ca",
        "https://github.com/LLNL/ExaCA",
        "MIT",
        "permissive",
        ("grain-growth", "thermal-history-transfer"),
    ),
    (
        "bernaise",
        "https://github.com/gautelinga/BERNAISE",
        "MIT",
        "permissive",
        ("electrohydrodynamics", "phase-field", "surface-charge"),
    ),
    (
        "py-stokes",
        "https://github.com/rajeshrinet/pystokes",
        "MIT",
        "permissive",
        ("phoresis", "stokesian-dynamics"),
    ),
    (
        "sfepy",
        "https://github.com/sfepy/sfepy",
        "BSD-3-Clause",
        "permissive",
        ("piezoelectricity", "finite-element"),
    ),
    (
        "ross",
        "https://github.com/petrobras/ross",
        "Apache-2.0",
        "permissive",
        ("rotordynamics", "bearings", "seals"),
    ),
    (
        "pylife",
        "https://github.com/boschresearch/pylife",
        "Apache-2.0",
        "permissive",
        ("fatigue", "load-collectives"),
    ),
    (
        "pybamm",
        "https://github.com/pybamm-team/PyBaMM",
        "BSD-3-Clause",
        "permissive",
        ("electrochemistry", "porous-electrode"),
    ),
    (
        "cantera",
        "https://github.com/Cantera/cantera",
        "BSD-3-Clause",
        "permissive",
        ("surface-chemistry", "thermochemistry"),
    ),
    (
        "openfoam",
        "https://github.com/OpenFOAM/OpenFOAM-dev",
        "GPL-3.0-or-later",
        "strong-copyleft",
        ("multiphase-flow", "combustion", "industrial-cfd"),
    ),
    (
        "additive-foam",
        "https://github.com/ORNL/AdditiveFOAM",
        "GPL-3.0-or-later",
        "strong-copyleft",
        ("additive-manufacturing", "heat-source-calibration"),
    ),
    (
        "laserbeam-foam",
        "https://github.com/laserbeamfoam/LaserbeamFoam",
        "GPL-3.0-or-later",
        "strong-copyleft",
        ("melt-pool", "laser-ray-tracing"),
    ),
    (
        "openfast",
        "https://github.com/OpenFAST/openfast",
        "Apache-2.0",
        "permissive",
        ("wind-energy", "aero-hydro-servo-elastic"),
    ),
    (
        "opengeosys",
        "https://github.com/ufz/ogs",
        "BSD-3-Clause",
        "permissive",
        ("thmc", "geotechnical", "subsurface"),
    ),
    (
        "project-chrono",
        "https://github.com/projectchrono/chrono",
        "BSD-3-Clause",
        "permissive",
        ("multibody", "contact", "vehicles"),
    ),
    (
        "mfix",
        "https://mfix.netl.doe.gov/gitlab/exa/docs",
        "source-available",
        "source-available",
        ("multiphase-reactors", "tfm-dem-pic"),
    ),
    (
        "palace",
        "https://github.com/awslabs/palace",
        "Apache-2.0",
        "permissive",
        ("electromagnetics", "ports", "frequency-domain"),
    ),
    (
        "warpx",
        "https://github.com/BLAST-WarpX/warpx",
        "BSD-3-Clause",
        "permissive",
        ("plasma", "particle-in-cell"),
    ),
    (
        "simvascular",
        "https://github.com/SimVascular/SimVascular",
        "BSD-3-Clause",
        "permissive",
        ("medical-devices", "vascular-flow"),
    ),
    (
        "jsbsim",
        "https://github.com/JSBSim-Team/jsbsim",
        "LGPL-2.1-or-later",
        "weak-copyleft",
        ("flight-dynamics", "aircraft-systems"),
    ),
    (
        "opm-flow",
        "https://github.com/OPM/opm-simulators",
        "GPL-3.0-or-later",
        "strong-copyleft",
        ("reservoir", "wells", "schedules"),
    ),
)


def _source_reference(
    spec: tuple[str, str, str, str, tuple[str, ...]],
) -> SourceReference:
    source_id, _, license, reuse, concepts = spec
    (
        repository_url,
        revision,
        archive_digest,
        license_digest,
        readme_record,
        license_record,
    ) = _SOURCE_AUDITS[source_id]
    reuse_class = SourceReuseClass(reuse)
    return SourceReference.create(
        source_id,
        repository_url,
        revision,
        license,
        reuse_class,
        concepts=concepts,
        archive_digest=archive_digest,
        license_digest=license_digest,
        relevant_documents=(readme_record, license_record),
        code_inspected=False,
        behavior_inspected=False,
        copying_permitted=reuse_class
        in (SourceReuseClass.PERMISSIVE, SourceReuseClass.PUBLIC_DOMAIN),
        provider_only=reuse_class
        in (
            SourceReuseClass.STRONG_COPYLEFT,
            SourceReuseClass.SOURCE_AVAILABLE,
        ),
        technical_reviewer="phydrax-source-metadata-audit-2026-09-20",
        legal_reviewer="automated-license-classification-2026-09-20",
        review_status=SourceReview.TECHNICALLY_REVIEWED,
    )


def _atomistic_sources() -> tuple[SourceReference, ...]:
    """Pinned method/code references; no foundation-weight rights are inferred."""
    return (
        SourceReference.create(
            "symmetrix-xl-engine",
            "https://github.com/bonan-group/symmetrix-xl",
            "da331142729978d27b2a52f6b34e1628593df3d0",
            "MIT",
            SourceReuseClass.PERMISSIVE,
            concepts=(
                "streamed-equivariant-edges",
                "bounded-receiver-replay",
                "radial-spline-projection",
            ),
            archive_digest="deb1a18a23c4e61767e73886a5b6469123114b591aaffe2a7418faa5cf02a78b",
            license_digest="9fb567b39a0908352d37e6af711629f5f586c2d93a8d980fdce603597203c5a1",
            relevant_paths=("libsymmetrix/source", "symmetrix/source/symmetrix"),
            relevant_documents=("LICENSE", "docs/streamed_edge_execution.md"),
            publications=(
                PublicationReference(
                    "Train for Accuracy, Execute at Scale: Architecture-Preserving Inference for Equivariant Atomistic Foundation Models",
                    "https://arxiv.org/abs/2610.01036",
                ),
            ),
            code_inspected=True,
            behavior_inspected=False,
            copying_permitted=False,
            data_rights="No weights or datasets bundled; GPLv2 pair_symmetrix integration excluded.",
            technical_reviewer="phydrax-streamed-atomistic-source-review",
            review_status=SourceReview.TECHNICALLY_REVIEWED,
        ),
        SourceReference.create(
            "mace-code",
            "https://github.com/ACEsuit/mace",
            "0.3.16",
            "MIT",
            SourceReuseClass.PERMISSIVE,
            concepts=(
                "standard-mace-architecture",
                "original-w-fixed-u",
                "trusted-checkpoint-interchange",
            ),
            archive_digest="b80407edf6b2a1ec8523668c2a36852d20927ce1c3c56b70983a9f2dc53233ad",
            license_digest="42137790f854ae2b9d29a0a72a4da3f6fb9a21d820b3f276f23b0575af72e86e",
            relevant_paths=(
                "mace/modules/models.py",
                "mace/modules/blocks.py",
                "mace/modules/symmetric_contraction.py",
            ),
            relevant_documents=("LICENSE.md", "mace_torch-0.3.16.dist-info/METADATA"),
            code_inspected=True,
            copying_permitted=True,
            data_rights="Code MIT does not relicense weights; MP/MPA MIT, OMAT/MH/OFF ASL separately admitted.",
            technical_reviewer="phydrax-streamed-atomistic-source-review",
            review_status=SourceReview.TECHNICALLY_REVIEWED,
        ),
        SourceReference.create(
            "e3nn-code",
            "https://github.com/e3nn/e3nn",
            "0.4.4",
            "MIT",
            SourceReuseClass.PERMISSIVE,
            concepts=("real-o3-basis-gauge", "tensor-product-source-normalization"),
            archive_digest="87d99876abb362a6e07d555d10752ff2e20c2b7731d8928ebe5d9121fb019f84",
            license_digest="2f875c146f567528415acaf19def2531b65e6dba7fb2485f25d86dd4b3912ad7",
            relevant_paths=("e3nn/o3",),
            relevant_documents=("LICENSE", "e3nn-0.4.4.dist-info/METADATA"),
            code_inspected=True,
            copying_permitted=True,
            data_rights="Mathematical/interchange reference; no external checkpoint or dataset bundled.",
            technical_reviewer="phydrax-streamed-atomistic-source-review",
            review_status=SourceReview.TECHNICALLY_REVIEWED,
        ),
    )


def builtin_source_absorption_ledger() -> SourceAbsorptionLedger:
    return SourceAbsorptionLedger.create(
        (*(_source_reference(spec) for spec in _SOURCE_SPECS), *_atomistic_sources())
    )


__all__ = ["builtin_source_absorption_ledger"]
