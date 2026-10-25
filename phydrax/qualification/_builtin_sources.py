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


# Long by declaration, not control flow: source, license and reuse evidence are
# reviewed together in one deterministic ledger. Do not replace these records with
# inferred defaults or a second source registry merely to shorten the function.
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
        SourceReference.create(
            "mace-code-0.3.15",
            "https://github.com/ACEsuit/mace",
            "0.3.15",
            "MIT",
            SourceReuseClass.PERMISSIVE,
            concepts=(
                "maceles-composition",
                "standard-mace-architecture",
                "toolkit-pinned-mace-release",
                "trusted-checkpoint-interchange",
            ),
            archive_digest="sha256:23f9bd7521db56ce91586c110089bdbd79ffd2475fec8fea2cae4780be495049",
            license_digest="sha256:42137790f854ae2b9d29a0a72a4da3f6fb9a21d820b3f276f23b0575af72e86e",
            relevant_paths=(
                "mace/modules/blocks.py",
                "mace/modules/extensions.py",
                "mace/modules/models.py",
            ),
            relevant_documents=(
                "LICENSE.md@sha256:42137790f854ae2b9d29a0a72a4da3f6fb9a21d820b3f276f23b0575af72e86e",
                "PKG-INFO@sha256:82ac1888b0e7b737bc780d73c9aa27f659ad8810f222010dc269f90ab32fc09d",
            ),
            code_inspected=True,
            copying_permitted=True,
            data_rights="Code MIT does not relicense weights; sdist sha256 is the archive identity.",
            technical_reviewer="phydrax-native-atomistic-source-review",
            review_status=SourceReview.TECHNICALLY_REVIEWED,
        ),
        SourceReference.create(
            "nvalchemi-toolkit",
            "https://github.com/NVIDIA/nvalchemi-toolkit",
            "f59e28dc6701583a13bb294e536076c2a7379e98",
            "Apache-2.0",
            SourceReuseClass.PERMISSIVE,
            concepts=(
                "capability-envelope-reference",
                "crystal-structure-prediction-workflow",
                "dftd3-reference-archive-pin",
                "mace-wrapper-energy-output-contract",
                "standard-setting-hall-map",
            ),
            archive_digest="sha256:0dc8a672c5890d5432af0ca56f52818234ab7f410ca464ff1db30ae8744399a6",
            license_digest="sha256:cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
            relevant_paths=(
                "nvalchemi/csp",
                "nvalchemi/csp/_space_group_tables.py",
                "nvalchemi/models/dftd3.py",
            ),
            relevant_documents=(
                "LICENSE@sha256:cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
                "README.md@sha256:3e5c2172bb7e80d1d7c5c0e82986ca91c8672afe4ea69d11069b62b895f9b5e7",
            ),
            code_inspected=True,
            copying_permitted=True,
            data_rights="Apache-2.0 code; derived Hall map carries notice. CSD-derived space-group priors, weights and datasets are not licensed by the code and are not bundled.",
            technical_reviewer="phydrax-native-atomistic-source-review",
            review_status=SourceReview.TECHNICALLY_REVIEWED,
        ),
        SourceReference.create(
            "nvalchemi-toolkit-ops",
            "https://github.com/NVIDIA/nvalchemi-toolkit-ops",
            "10e12340c39fd12a2375d6301768e72d2cd12868",
            "Apache-2.0",
            SourceReuseClass.PERMISSIVE,
            concepts=(
                "conventional-d3-coordination-threshold",
                "fourier-d3-modified-counting",
                "multipole-packed-basis",
                "signed-low-rank-c6-decomposition",
            ),
            archive_digest="git-tree-sha1:e0ee833387dec0f7cce09ed4871c9bce935c11b2",
            license_digest="sha256:1e9f1d1e84fa0fee6c208fd1d119ab52007204b34b7e2391852e68855a9d8c9e",
            relevant_paths=(
                "nvalchemiops/interactions/dispersion",
                "nvalchemiops/interactions/electrostatics",
                "nvalchemiops/torch/interactions/electrostatics/_multipole_moments.py",
            ),
            relevant_documents=(
                "LICENSE@sha256:1e9f1d1e84fa0fee6c208fd1d119ab52007204b34b7e2391852e68855a9d8c9e",
                "README.md@sha256:d3bb79a5c74803d7c8489b3d7c859f2bfd963ac3871ab05ee5a06d0b0712c862",
            ),
            publications=(
                PublicationReference(
                    "A fast summation method for the DFT-D3 dispersion correction",
                    "https://arxiv.org/html/2607.15103",
                ),
            ),
            code_inspected=True,
            copying_permitted=True,
            data_rights="Apache-2.0 code; Warp kernels are never executed or bundled. D3 reference tables remain under their own GPL notice.",
            technical_reviewer="phydrax-native-atomistic-source-review",
            review_status=SourceReview.TECHNICALLY_REVIEWED,
        ),
        SourceReference.create(
            "aimnetcentral",
            "https://github.com/isayevlab/aimnetcentral",
            "b1b0fd8c8b8d8acb8811eb646f9ba7b15d72da55",
            "MIT",
            SourceReuseClass.PERMISSIVE,
            concepts=(
                "aimnet2-architecture",
                "aimnet2-checkpoint-interchange",
                "aimnet2-long-range-composition",
                "ts-dispersion-parameters",
            ),
            archive_digest="sha256:d99b85dd2f6064c800903c2d0792a217933444cc5e0552a6547a22d458c45234",
            license_digest="sha256:ef7b24fb3836c9546b3994037956cf95507b49c65cd622b49320813303704200",
            relevant_paths=(
                "aimnet/models/aimnet2.py",
                "aimnet/modules/aev.py",
                "aimnet/modules/core.py",
                "aimnet/modules/lr.py",
            ),
            relevant_documents=(
                "LICENSE@sha256:ef7b24fb3836c9546b3994037956cf95507b49c65cd622b49320813303704200",
                "README.md@sha256:accd17aef36338e0bf55736e904a966958f37c829354db724af22727c968401c",
            ),
            code_inspected=True,
            copying_permitted=True,
            data_rights="Release 0.2.0 code MIT; pretrained weights carry separate terms and require caller admission.",
            technical_reviewer="phydrax-native-atomistic-source-review",
            review_status=SourceReview.TECHNICALLY_REVIEWED,
        ),
        SourceReference.create(
            "fairchem",
            "https://github.com/FAIR-Chem/fairchem",
            "bb0a062a9a08d8b3ad3995aca5b9765e07804990",
            "MIT",
            SourceReuseClass.PERMISSIVE,
            concepts=(
                "escn-md-backbone",
                "nearest-neighbor-graph-selection",
                "uma-checkpoint-interchange",
                "uma-mixture-of-linear-experts",
            ),
            archive_digest="sha256:d2262d0eef636a6754691e55d5cefdc58a1b2e1f7561239459a1e9fe27af62c8",
            license_digest="sha256:3e10827938aae9e591647722c830065a95b3af2721f0d7f02f373869b40fe81d",
            relevant_paths=(
                "src/fairchem/core/models/uma/escn_md.py",
                "src/fairchem/core/models/uma/escn_md_block.py",
                "src/fairchem/core/models/uma/escn_moe.py",
                "src/fairchem/core/models/uma/outputs.py",
            ),
            relevant_documents=(
                "LICENSE.md@sha256:3e10827938aae9e591647722c830065a95b3af2721f0d7f02f373869b40fe81d",
                "README.md@sha256:5e8a5edbd2035bb65d1e440e586d20aa238a8f91b9515f59a9ed82a6cc499068",
            ),
            code_inspected=True,
            copying_permitted=True,
            data_rights="Code MIT; UMA checkpoints are gated under separate model terms that are not admitted here.",
            technical_reviewer="phydrax-native-atomistic-source-review",
            review_status=SourceReview.TECHNICALLY_REVIEWED,
        ),
        SourceReference.create(
            "les",
            "https://github.com/ChengUCB/les",
            "35b971a650cf30adb84fef7138761e81aa47b3a2",
            "CC-BY-NC-4.0",
            SourceReuseClass.SOURCE_AVAILABLE,
            concepts=("latent-born-charges", "latent-ewald-summation"),
            archive_digest="git-tree-sha1:5c676baef9cde284c4a8b375232c6da4d9594d86",
            license_digest="pyproject.toml@sha256:a9d493333b3ef53be141b4a451303ac1d6b16de03378a8d7a29c50381561d97e",
            relevant_paths=(
                "src/les/les.py",
                "src/les/module/bec.py",
                "src/les/module/ewald.py",
                "src/les/module/ewald_vectorized.py",
            ),
            relevant_documents=(
                "README.md@sha256:b1c53513e3ee393e70b3ca8b40b42a887fb6f970eb8a655e1dc2c2e6e6dfce87",
                "pyproject.toml@sha256:a9d493333b3ef53be141b4a451303ac1d6b16de03378a8d7a29c50381561d97e",
            ),
            publications=(
                PublicationReference(
                    "Latent Ewald summation for machine learning of long-range interactions",
                    "npj Comput. Mater. 11, 80 (2025)",
                ),
            ),
            code_inspected=True,
            copying_permitted=False,
            provider_only=True,
            data_rights="No LICENSE file at the revision; CC-BY-NC-4.0 is declared in pyproject.toml and README.md. No source copied or imported; commercial source use and MACELES weights are not admitted.",
            technical_reviewer="phydrax-native-atomistic-source-review",
            review_status=SourceReview.TECHNICALLY_REVIEWED,
        ),
        SourceReference.create(
            "spglib",
            "https://github.com/spglib/spglib",
            "12355c77fb7c505a55f52cae36341d73b781a065",
            "BSD-3-Clause",
            SourceReuseClass.PERMISSIVE,
            concepts=("hall-symbol-settings", "space-group-operation-oracle"),
            archive_digest="src/spg_database.c@sha256:bf1ab8a74e1d275e20ee4145bd409b62b5295e680599d3a112a41d396bde56bd",
            license_digest="sha256:9c4e602ce15bf48206dad105b666ddaeedba747eeda6465807106bc11f4f587f",
            relevant_paths=("src/spg_database.c",),
            relevant_documents=(
                "COPYING@sha256:9c4e602ce15bf48206dad105b666ddaeedba747eeda6465807106bc11f4f587f",
                "src/spg_database.c@sha256:bf1ab8a74e1d275e20ee4145bd409b62b5295e680599d3a112a41d396bde56bd",
            ),
            code_inspected=True,
            copying_permitted=True,
            data_rights="v2.7.0 Hall table regenerated into exact native operation data with notice; only the database file was admitted, not the full tree.",
            technical_reviewer="phydrax-native-atomistic-source-review",
            review_status=SourceReview.TECHNICALLY_REVIEWED,
        ),
        SourceReference.create(
            "grimme-dftd3",
            "https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/dftd3.tgz",
            "md5:a76c752e587422c239c99109547516d2",
            "GPL-1.0-or-later",
            SourceReuseClass.STRONG_COPYLEFT,
            concepts=("d3-covalent-radii", "d3-r4r2-factors", "d3-reference-c6-tables"),
            archive_digest="sha256:d97cf9758f61aa81fd85425448fbf4a6e8ce07c12e9236739831a3af32880f59",
            license_digest="dftd3.f@sha256:f643e01857d1a6d36d4f839ecb530673c6fc90e86a5a3599a893ba1a61b136f1",
            relevant_paths=("dftd3.f", "pars.f"),
            relevant_documents=(
                "dftd3.f@sha256:f643e01857d1a6d36d4f839ecb530673c6fc90e86a5a3599a893ba1a61b136f1",
                "pars.f@sha256:d3e4abda619d9c481e52f4197f0fb22a952d7feaf13869c53a1206aaa7dfff8f",
            ),
            publications=(
                PublicationReference(
                    "A consistent and accurate ab initio parametrization of density functional dispersion correction (DFT-D) for the 94 elements H-Pu",
                    "doi:10.1063/1.3382344",
                ),
                PublicationReference(
                    "Effect of the damping function in dispersion corrected density functional theory",
                    "doi:10.1002/jcc.21759",
                ),
            ),
            code_inspected=True,
            copying_permitted=False,
            provider_only=True,
            data_rights="GPL notice in dftd3.f; archive MD5 matches the Toolkit/Ops pin. Tables are never bundled or downloaded; callers supply digest-verified files under their own rights.",
            technical_reviewer="phydrax-native-atomistic-source-review",
            review_status=SourceReview.TECHNICALLY_REVIEWED,
        ),
    )


def _periodic_flow_sources() -> tuple[SourceReference, ...]:
    """Strong-copyleft solver reference; Phydrax implements cited publications only."""
    return (
        SourceReference.create(
            "azeban",
            "https://github.com/TobiasRohner/azeban",
            "f34d2225ed849e4d851e3b995687f700eeaa163a",
            "GPL-3.0",
            SourceReuseClass.STRONG_COPYLEFT,
            concepts=(
                "spectral-vanishing-viscosity",
                "statistical-solutions",
                "structure-functions",
                "random-initial-measures",
            ),
            archive_digest="git-tree-sha1:22bf5ba2436d690145f7dadbe900236d32d73230",
            license_digest="sha256:33bc26d4498c183b703d1760ee45b8774d21d6e0178ca393dc715271840c7a21",
            relevant_paths=(
                "doc",
                "experiments",
                "include/azeban/equations",
                "include/azeban/evolution",
                "include/azeban/forcing",
                "include/azeban/operations",
                "scripts/generate_fbm_fourier.py",
                "src/azeban/cuda/operations/cufft_mpi.cpp",
                "src/azeban/equations/incompressible_euler_mpi.cpp",
                "src/azeban/init",
                "src/azeban/operations",
                "src/azeban/simulation.cpp",
            ),
            relevant_documents=(
                "README.md@sha256:73f24879e878c29635c014d445ae5fba6b118cb6d5992dacdba04b9702db4cb7",
                "LICENSE.md@sha256:33bc26d4498c183b703d1760ee45b8774d21d6e0178ca393dc715271840c7a21",
            ),
            publications=(
                PublicationReference(
                    "On the convergence of the spectral viscosity method for the two-dimensional incompressible Euler equations with rough initial data",
                    "https://arxiv.org/abs/1903.12361",
                ),
                PublicationReference(
                    "Statistical solutions of the incompressible Euler equations",
                    "https://arxiv.org/abs/1909.06615",
                ),
                PublicationReference(
                    "Efficient computation of large-scale statistical solutions to incompressible fluid flows",
                    "https://www.sam.math.ethz.ch/sam_reports/reports_final/reports2024/2024-04_rev1.pdf",
                ),
                PublicationReference(
                    "Poseidon: Efficient Foundation Models for PDEs",
                    "https://arxiv.org/abs/2405.19101",
                ),
            ),
            code_inspected=True,
            behavior_inspected=False,
            copying_permitted=False,
            provider_only=True,
            data_rights="No source, constants, configuration, or data copied; formulas and defaults follow the cited publications.",
            technical_reviewer="phydrax-periodic-flow-source-review",
            review_status=SourceReview.TECHNICALLY_REVIEWED,
        ),
    )


def _graph_connectivity_sources() -> tuple[SourceReference, ...]:
    """Inspected connectivity references; formulations are re-derived, not copied."""
    return (
        SourceReference.create(
            "jaxscape",
            "https://github.com/vboussange/jaxscape",
            "5455ca4f0a5ebfe5f27f0b24c672e8e6948c73b4",
            "MIT",
            SourceReuseClass.PERMISSIVE,
            concepts=(
                "effective-resistance",
                "landscape-connectivity",
                "randomized-shortest-paths",
                "resistance-sketching",
            ),
            archive_digest="git-tree-sha1:f5d9cf6b51966b9bdadb7b9e5ee40d1fc200e6a4",
            license_digest="daa2b5d3053360fb529ee4ca62f105bcb45c294a6170d3affeb2c48864792baa",
            relevant_paths=(
                "src/jaxscape/graph.py",
                "src/jaxscape/lcp_distance.py",
                "src/jaxscape/resistance_distance.py",
                "src/jaxscape/rsp_distance.py",
            ),
            relevant_documents=(
                "LICENSE@sha256:daa2b5d3053360fb529ee4ca62f105bcb45c294a6170d3affeb2c48864792baa",
                "README.md@sha256:5323f68f1812928090ae811bb49a8160d24aae430d94848f6c885ae29517db5c",
            ),
            publications=(
                PublicationReference(
                    "Graph sparsification by effective resistances",
                    "doi:10.1137/080734029",
                ),
            ),
            code_inspected=True,
            copying_permitted=True,
            technical_reviewer="phydrax-graph-connectivity-source-review",
            review_status=SourceReview.TECHNICALLY_REVIEWED,
        ),
    )


def builtin_source_absorption_ledger() -> SourceAbsorptionLedger:
    from ._nuclear_fuel_release_matrix import nuclear_fuel_source_references

    return SourceAbsorptionLedger.create(
        (
            *(_source_reference(spec) for spec in _SOURCE_SPECS),
            *_atomistic_sources(),
            *_periodic_flow_sources(),
            *nuclear_fuel_source_references(),
            *_graph_connectivity_sources(),
        )
    )


__all__ = ["builtin_source_absorption_ledger"]
