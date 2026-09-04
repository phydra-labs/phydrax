# Skeletal-muscle cellular electrophysiology

`phydrax.applications.skeletal_muscle.cellular` provides the source-complete
fast-twitch model of Shorten, O'Callaghan, Davidson, and Soboleva (2007). It is
a cellular excitation--calcium--crossbridge model, not a fiber PDE, a CellML
runtime, or a generic muscle force law.

## Source identity and the 56/57-state question

The implementation is independently transcribed from the Physiome Model
Repository workspace changeset
`637da9ef28f7992e40fe79947364a51a38ec818c`, file
`shorten_ocallaghan_davidson_soboleva_2007.cellml`. The exact raw file is
184,890 bytes and has SHA-256
`e14e2aeffeb7b935017414a5ef53c06e43ed6b5fd4d7a92f07e0518b48b413c1`.
The PMR file is licensed under Creative Commons Attribution 3.0 Unported. The
model implements the equations reported in P. R. Shorten, P. O'Callaghan,
J. B. Davidson, and T. K. Soboleva, “A mathematical model of fatigue in
skeletal muscle force contraction,” *Journal of Muscle Research and Cell
Motility* 28 (2007), 293--313, DOI `10.1007/s10974-007-9125-6`.

The authoritative fast-twitch file has **56 differential variables**: 2
membrane voltages, 6 ionic concentrations, 10 membrane gates, 10 Stern--Rios
states, and 28 Razumova calcium, buffer, crossbridge, and phosphate states.
`P_C_SR` is state index 55, the last zero-based slot. The same file has 71
algebraics, 99 independent numeric parameters, and 6 source-derived geometry
constants (105 CellML constant slots after constant folding).

The often-quoted OpenDiHu “57 states and 71 algebraics” belongs to
`new_slow_TK_2014_12_08.cellml`, a derived OpenCMISS-era model that adds an
active-stress route. A stale comment beside OpenDiHu's original Shorten example
also says 57, but the actual template is `CellmlAdapter<56,71>` and its pinned
generated reference says 56 rate/state entries. No dummy 57th state is added
here. The slow-twitch PMR variant is deliberately not exported because it has
not been independently implemented and qualified in this package.

Every final-axis entry is inspectable:

```python
from phydrax.applications.skeletal_muscle.cellular import ShortenFastTwitchModel

model = ShortenFastTwitchModel()
print(model.state_layout.index("Ca_2"))
print(model.state_layout.source_symbol("Ca_2"))  # razumova/Ca_2
print(model.algebraic_layout.index("I_HH"))      # 32
```

The state, parameter, constant, and algebraic layouts each provide `names`,
`units`, `source_symbols`, `index`, `pack`, and `unpack`. State and algebraic
order matches the established libCellML/OpenCOR array order. Parameters are
JAX leaves; the model and source identity remain static.

## Units and signs

The kernel preserves the CellML units rather than applying an implicit SI
conversion:

- time: ms;
- voltage: mV;
- membrane current density: uA/cm2;
- capacitance density: uF/cm2;
- sarcolemmal ionic concentrations: mM;
- calcium, buffers, and crossbridges: uM;
- phosphate: mM.

Sarcolemmal and t-tubule channel currents are positive outward. The stimulus is
positive inward, exactly matching `wal_environment/I_HH`; the source pulse is
left-closed and right-open. `ShortenPulseProtocol()` reproduces nine 150
uA/cm2 pulses of width 0.5 ms beginning every 50 ms from 0 through 400 ms.
Supplying `stimulus_current_uA_per_cm2` replaces the protocol; it is never added
to it.

## Pure evaluation and integration

```python
import numpy as np
from phydrax.applications.skeletal_muscle.cellular import (
    ShortenFastTwitchModel,
    ShortenIntegrationPlan,
)

model = ShortenFastTwitchModel()
y0 = model.initialize()
evaluation = model.evaluate(0.0, y0)
print(evaluation.cytosolic_calcium_uM)
print(evaluation.tension_driver_uM)

times = np.unique(np.r_[np.linspace(0.0, 100.0, 201), 0.5, 50.5])
prepared = ShortenIntegrationPlan(model, times).prepare()
trajectory = prepared.integrate()
```

`evaluate` and `rhs` are pure, fixed-shape, JIT/vmap/JVP-compatible functions
away from source hard branches. Ten first-order membrane gates also expose the
exact frozen-voltage Rush--Larsen update through `exact_gate_update`. Source
comparison shows that exact gates alone do not remove the much faster
Stern--Rios and calcium-buffer reactions, so the prepared complete-cell route
uses PhydraX's differential-solver owner with Diffrax Kvaerno5. The time grid
must contain every stimulus start and end in its support, preventing an
adaptive step from hiding a pulse edge.

A prepared step returns a `ShortenStepCandidate`. `commit()` accepts the
candidate only when the solver succeeds and the complete state is finite,
admissible, and time-aligned. Failure rolls back both time and all 56 state
channels. No partial calcium, gate, or crossbridge update is committed.

The rectangular stimulus, inward-rectifier sign gate, and phosphate
precipitation switch are source-defined hard branches. Values and ordinary
local derivatives away from a branch are supported; the model does not claim a
global derivative across pulse, sign, or precipitation events.

## Force ownership

The source's force-bearing state is `razumova/A_2`, the post-power-stroke
attached-crossbridge concentration. The evaluation reports it as both
`force_bearing_crossbridge_uM` and `tension_driver_uM`. This is a biochemical
tension driver in uM, not force in newtons and not stress in pascals. Converting
it to tissue stress requires one explicitly selected downstream constitutive
owner. It must not be multiplied by D1 terminal relative force, De Groote
force, or another cellular force law. This cellular route owns its calcium,
crossbridge, phosphate-fatigue, and tension-driver response end to end.
