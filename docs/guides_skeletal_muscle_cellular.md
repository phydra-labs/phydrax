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
`new_slow_TK_2014_12_08.cellml`, a derived OpenCMISS-era model that adds a
`dummy` differential state integrating its algebraic stress output. Its
crossbridge distortions `x_1` and `x_2` remain constants; this is not the
non-isometric Heidlauf--Röhrle (2014) distortion-state model.
A stale comment beside OpenDiHu's original Shorten example
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

## Non-isometric and slow-twitch source gates

The selected non-isometric fidelity is
[Heidlauf and Röhrle (2014), DOI 10.3389/fphys.2014.00498](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2014.00498/full),
not the 2013 A2-times-Hill construction. The 2014 paper changes the
crossbridge equations themselves: its Eqs. 3--8 specify a fourth-order
force--length polynomial, two distortion ODEs, cooperative attachment,
distortion-dependent detachment, and resting-baseline-corrected normalized
active stress. The existing 56-state Shorten model is unchanged and accepts
no mechanics feedback.

An immutable-source acquisition found the relevant named files in Benjamin
Maier's [OpenDiHu input dataset, DOI 10.5281/zenodo.4705982](https://zenodo.org/records/4705982).
The [record metadata](https://zenodo.org/api/records/4705982) licenses that
dataset under CC BY 4.0; this is distinct from OpenDiHu's MIT software
license. The following exact downloaded payloads **do not close the 2014
model gate**:

| Acquired file | SHA-256 | Finding |
| --- | --- | --- |
| `WSBM_1457_shorten_model_modified_as_in_heidlauf_2014.cellml` | `3f17b40bba8b6107bf730264fbbb40bdecabf8e4124720cddab0ea871568ddf4` | 58 states, but equations and parameters differ from the paper. |
| `new_slow_TK_2014_12_08.cellml` | `8f63aa38358106b025f259b2c0fddd1c62815433bf52f48537ed7dcb51fc57a9` | 57 states; constant distortions and attachment/detachment rates, plus a stress integral. |
| `new_slow_TK_2014_12_08.c` | `e21d1f4e813807b684a44740f3008e625e6e98210060d0aafa9dc88266eb0202` | Generated 57-state realization of that slow variant, not a 2014 non-isometric oracle. |

The fast WSBM file uses `x_0 = 8 nm`, `nu = 1`, and `theta = 0`, whereas
the paper's fast Table 2 values are `x_0 = 0.05 um`, `nu = 3.4`, and
`theta = 1000`. These are not its only differences:

- Its cooperative terms contain `exp(z - 1)`, where paper Eq. 7 requires
  `exp(z) - 1`. Changing parameter defaults alone does not repair this.
- Its `x_1` balance includes an additional
  `-h_prime * A_2 / A_1 * x_1` contribution relative to paper Eq. 4.
- Its force--length curve is piecewise linear in half-sarcomere length,
  not paper Eq. 3's fourth-order polynomial in full sarcomere length.
- Its active-stress expression divides by `A_2_max * x_0`, with
  `A_2_max = 3 uM`, and omits paper Eq. 6's resting baseline subtraction
  in both numerator and denominator.
- Its imported `stimulation_100Hz.cellml` filename differs from the
  separately deposited `WSBM_1457_stimulation_100Hz.cellml`. A reproducible
  executable would need an explicit, identified import mapping.

The units also cannot be inferred from the filenames. Paper Eq. 4 adds
**half the signed full-sarcomere velocity** to each distortion rate;
negative velocity denotes shortening and positive velocity lengthening.
At the source boundary, metres convert to micrometres by a factor of
1,000,000 and metres/second to micrometres/millisecond by 1,000.
The WSBM file instead declares nanometre distortions and a
nanometre/millisecond velocity. Table 2's dimensionless
distortion-dependence coefficient can be used consistently by writing
Eq. 8 as `g_0 = g_bar * exp(theta * ((x_2 - x_0) / (1 um))**2)`.
This makes the source's numerical micrometre convention explicit; its
value must not silently be reused with unscaled SI or nanometre states.

Alternate source searches covered the complete trees of archived OpenDiHu
at `b444344cf7ed313b26aaa75dd18b5f5933f721b8`, current OpenDiHu at
`be7b1d6862c7d8b29ee83037421d9005b999dfdb`, and OpenCMISS examples at
`3f5a1c744f0493fadaba8649a859e47ddac3e913`. The latter's
`cellModelFiles/shorten_mod_2011_07_04.xml` has 59 differential states,
including the two distortion states and a stress integral, but still
uses constant attachment/detachment rates and a different stress
normalization. Neither that older file nor the separately acquired slow
equilibrium values is a matching 2014 numerical reference.

These mismatches do **not** make the published mathematics unavailable or
require finding an original file with a particular 2014 filename.
Section 2.3 explicitly adopts the original Shorten fast and slow
phenotype parameters without reparametrization. A paper-exact
reconstruction can therefore combine each independently sourced
56-state phenotype with the two distortion states under a **new,
separately identified 58-state model**, applying Eqs. 3--8 and Table 2.
That would not modify the existing 56-state Shorten identity.
The original Shorten initial values can define a declared numerical
initial-value problem, but cannot simply be called the resting
equilibrium of the modified cooperative model: Eq. 7 changes attachment
even at the zero-velocity distortion state `x_1 = 0`, `x_2 = x_0`.

**No Heidlauf--Röhrle 2014 fast or slow model API is currently exported.**
The remaining admission gate is numerical and physiological qualification:
declare and identify the complete initialization, stimulation waveform,
resting equilibration and endpoint-selection procedures, and determine
the normalization endpoints `B(0, 0)` and `B(f_s_max, 0)` reproducibly.
The paper identifies approximately 100 Hz as saturating stimulation
and describes 500 ms preactivation as approximately saturating, but
neither uniquely fixes the endpoint sampling statistic, measurement
time, or resting preparation. A reconstructed implementation needs an independent
executable reference, solver/time-step convergence, and reproduction of
the Fig. 4 shortening/lengthening family and Fig. 5 preactivation and
shortening protocols before admission. No held-out physiological traces
were acquired in this audit, and the rejected implementation assets
are not physiological validation data. Merely patching WSBM defaults or
adding a slow/fast flag would not supply this missing qualification.
