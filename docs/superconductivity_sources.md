# Superconductivity source boundaries

The superconductivity ladder is implemented independently against Phydrax geometry, graph, linear algebra, thermodynamics, circuit, and qualification contracts.

| Source | Role | Native boundary |
|---|---|---|
| [SuperScreen](https://github.com/loganbvh/superscreen) | Thin-film London/Pearl, stream functions, fluxoids, inductance; MIT | No Python source adaptation. Native DDG and dense magnetostatic KKT support planar films. |
| [SuperConga](https://gitlab.com/superconga/superconga) | Equilibrium Eilenberger/Riccati, self-consistency, spectroscopy; LGPL-3.0-or-later | No C++/CUDA/frontend code. Native support is spin-degenerate singlet with caller-prepared trajectories. |
| Supplied topological-superconductivity examples | Small real-space BdG examples; MIT | Existing Phydrax BdG/Chern owners remain authoritative; no mutable dense example code is absorbed. |
| [OPENSC2](https://github.com/MAHTEP/OPENSC2) | Cable component, coolant, current-sharing, quench, protection vocabulary; AGPL-3.0 | No Python/GUI/Excel/CoolProp runtime or source adaptation. Native support is a fixed one-dimensional component profile. |

## Material and validation data

Critical-current surfaces, resistivity, calorimetry, coolant properties, cable geometry, experiments, and external simulation outputs are separate governed artifacts. Repository licensing does not grant rights to embedded material curves or benchmark cases. Synthetic London, GL, Riccati, and cable cases establish numerical behavior only; promotion requires independent locked analytic or experimental evidence.
