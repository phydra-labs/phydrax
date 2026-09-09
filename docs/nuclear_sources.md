# Nuclear and tokamak source boundaries

The native implementation is independent. External projects provide scientific vocabulary, published-method references, interoperability targets, or qualification comparisons. Their inclusion here does not admit their code, data, model weights, or validation claims.

| Source | Role | Native boundary |
|---|---|---|
| [RadSim](https://github.com/LLNL/RadSim) | Gamma source and detector-response reference | No Java/JPype integration; future detector response only |
| [R3BRoot](https://github.com/R3BRootGroup/R3BRoot) | Experimental event/reconstruction reference | ROOT/FairRoot remains external |
| [NPTool](https://github.com/adrien-matta/nptool) | Geant4/ROOT detector workflow reference | No plugin or object-model import |
| [OpenMOC](https://github.com/mit-crpg/OpenMOC) | Deterministic multigroup transport reference | No native MOC in the current support |
| [ARMI](https://github.com/terrapower/armi) | Reactor workflow and statepoint reference | Phydrax lifecycle remains authoritative |
| [KOMODO](https://github.com/imronuke/KOMODO) | Few-group diffusion/kinetics comparison | No Fortran/card-deck integration |
| [ADPRES](https://github.com/imronuke/ADPRES) | Historical coarse-core comparison | Superseded reference only |
| [awesome-nuclear](https://github.com/paulromano/awesome-nuclear) | Discovery catalog | Every linked source requires independent review |
| [Oslo Resources List](https://github.com/oslocyclotronlab/Resources-List) | Nuclear-data discovery catalog | Official databases remain authoritative |
| [3-D fusion/fission animation](https://github.com/Leoverse/3D-Simulation-of-a-Nuclear-Fusion-and-Fission-Reactor) | Educational visualization | Rejected as a physics source |
| [fusion-energy](https://github.com/fusion-energy) | CAD/DAGMC/OpenMC ecosystem reference | Geometry and transport remain governed artifacts |
| [pyPICfusion](https://github.com/LLNL/pyPICfusion) | Weighted fusion-reaction algorithm reference | No GPL source reuse; no reacting PIC claim |
| [PlasmaEvolution](https://github.com/PlasmaControl/PlasmaEvolution) | Shot/profile surrogate reference | No model or checkpoint import |
| [ALARA](https://github.com/svalinn/ALARA) | Activation and irradiation-schedule reference | Native fixed network; ALARA remains an external oracle |
| [KSTAR simulator](https://github.com/jaem-seo/KSTAR_tokamak_simulator) | Machine-surrogate UX reference | No predictive-physics claim |
| [AI tokamak control](https://github.com/jaem-seo/AI_tokamak_control) | Goal/action vocabulary reference | No policy or safety claim |
| [DINA-IMAS](https://github.com/iterorganization/DINA-IMAS) | Free-boundary/circuit/IMAS reference | LGPL implementation remains external |
| [TORAX](https://github.com/google-deepmind/torax) | JAX radial core-transport reference | Native Phydrax transport contract and numerics |
| [METIS](https://github.com/IRFM/METIS) | Reduced scenario reference | Reduced fidelity must remain explicit |
| [FreeGSNKE](https://github.com/FusionComputingLab/freegsnke) | Free-boundary equilibrium/circuit reference | LGPL implementation remains external |
| [Plasma ML catalog](https://github.com/kharitonov-ivan/awesome-ML-for-plasma-physics) | Discovery catalog | No validation or data-rights inference |
| [Fusion open-source catalog](https://github.com/kripnerl/fusion-open-source) | Equilibrium/stellarator discovery catalog | Current native scope remains axisymmetric tokamak |
| [JAR Fusion Core](https://github.com/jackylawck/jar-fusion-core) | Educational UI reference | Rejected as physics or engineering evidence |

## Data authorities

Evaluated or experimental data should cite the authoritative release directly, such as EXFOR, ENSDF/NuDat, RIPL, NIST STAR, or a declared processed nuclear-data library. Catalog membership is not provenance.

## License rule

GPL/LGPL implementation text is not translated into the native code. Permissively licensed code or fixtures may be admitted only with exact notices and file-level rights. A repository license does not automatically cover embedded nuclear data, machine files, geometry, model weights, or third-party benchmark decks.
