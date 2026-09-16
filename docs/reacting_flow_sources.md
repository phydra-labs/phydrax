# Reacting-flow source boundaries

The implementation is independent and native. External projects and papers provide equations, architecture vocabulary, or qualification targets; their code, scripts, data, credentials, weights, and benchmark outputs are not redistributed.

| Source | Role | Native boundary |
|---|---|---|
| [RWTH combustion validation framework](https://gitlab.git.nrw/rwth-itv-public/combustion-model-validation-framework-openfoam) | Campaign and observable reference; CC BY-NC | No shell/SLURM/OpenFOAM mutation or data downloader. Embedded plaintext credentials are treated as compromised and are never copied or executed. |
| [Hydrogen engine DNS](https://arxiv.org/abs/2502.16318) | Prospective moving-engine qualification | No code or real-engine DNS claim. |
| [PeleLMeX](https://github.com/AMReX-Combustion/PeleLMeX) | Low-Mach SDC, projection, drift, AMR, and load-balancing reference; BSD-3-Clause | No AMReX/C++ runtime or source adaptation. |
| [CombustionToolbox](https://github.com/CombustionToolbox/combustion_toolbox) | Equilibrium and jump-relation reference; GPL-3.0 | Native optimization and thermodynamics only. |
| [deepFlame](https://github.com/deepmodeling/deepflame-dev) | Learned chemistry and heterogeneous scheduling reference; GPL-3.0 | No OpenFOAM, PyTorch/libtorch, models, or weights. |
| [CEMAFoam](https://github.com/Aalto-CFD/CEMAFoam) | Chemical explosive-mode reference; GPL-3.0 | Native JAX mechanism Jacobian; no pyJac/OpenFOAM source. |
| [Ihme kinetics database](https://github.com/IhmeGroup/Combustion_Kinetics_Database) | Mechanism discovery and comparison catalog | Every mechanism retains its original authorship, license, and manifest. No database bundling. |
| [Flame simulations paper](https://doi.org/10.1016/j.cpc.2018.11.011) | Detailed transport and flame-validation reference | No solver source; all species remain explicit. |
| [DTLreactingFoam paper](https://doi.org/10.1016/j.cpc.2026.110052) | Fitted transport and property-reuse reference | No paper/code adaptation; native error-bounded accepted-state reuse replaces threshold-only copying. |

## Data rule

Mechanisms, transport tables, experimental/DNS fields, and learned artifacts enter only through caller-retained `ReferenceArtifactManifest` values. No network access occurs in numerical or qualification execution. Synthetic cases establish numerical validity, not scientific release.
