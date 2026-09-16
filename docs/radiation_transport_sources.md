# Radiation-transport source boundaries

The native radiation implementation shares no upstream runtime or source code. External works define scientific processes, comparison problems, or workflow vocabulary.

| Source | Role | Native boundary |
|---|---|---|
| [LLNL RadSim](https://github.com/LLNL/RadSim) | Source → transport → detector workflow; MIT | No Java/JPype/provider orchestration. |
| [EGSnrc](https://github.com/nrc-cnrc/EGSnrc) | Photon/electron/positron process and condensed-history reference; AGPL-3.0-or-later | No source translation or linked runtime. The duplicate supplied reference is one source. |
| [MCGPU](https://github.com/DIDSR/MCGPU) | Voxel delta tracking, alias spectrum, scatter-class image, KERMA; US public domain | Independent fixed-capacity JAX implementation; no PENELOPE-derived tables. |
| [XRayMClib/OpenXRayMC](https://github.com/medicalphysics/XRayMClib) | Diagnostic X-ray sources, geometries, KERMA, detector workflow; GPL-3.0 | No C++/Qt/VTK/segmentation runtime or EPICS download. |
| [Eradiate](https://github.com/eradiate/eradiate) | Correlated-k, polarization, scene/sensor composition; LGPL-3.0 | No Mitsuba runtime, scene hierarchy, or bundled databases. |
| [SuperNu](https://github.com/lanl/SuperNu) | IMC/DDMC, census, thick-limit leakage; GPL-3.0 | No Fortran/runtime/atomic data; current native support is bounded one-dimensional packet coupling. |
| [OpenSn](https://github.com/Open-Sn/opensn) | Discrete-ordinates groupsets, sweeps, acceleration; MIT | No C++/PETSc runtime; landed support is one-dimensional slab S_n. |
| [LegPy](https://github.com/JaimeRosado/LegPy) | Educational low-energy cases; GPL-3.0 | Not a production oracle and no source/data reuse. |

## Data rule

Cross sections enter through canonical `DiagnosticPhotonCoefficientTable` values with
`NuclearDataProvenance`; the prepared runtime retains all source identities and
combines mass attenuation only with caller-declared density. Stopping/scattering
powers, emission spectra, atmospheric coefficients, material maps, and external
benchmark outputs likewise require exact manifests and requested-use rights. Core
execution never downloads EPICS, NIST, PENELOPE, detector, or patient data.
Photon-local deposition is explicitly KERMA; absorbed-dose claims require a
separately qualified charged-particle profile.
