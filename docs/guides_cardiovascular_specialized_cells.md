# Specialized cardiac cellular models

PhydraX provides separate, typed cellular families for atrial working
myocytes, sinoatrial and atrioventricular nodal cells, and Purkinje fibres.
These models do not select a phenotype with a boolean and do not pad every
cell into one ventricular state vector. Each family owns a fixed named
structure-of-arrays (SoA) schema, coefficients, current record, calcium
record, evidence record, and deterministic identity.

## Qualification scope

| Prepared model | Qualified identity | Retained phenotype-defining mechanisms | Scope boundary |
| --- | --- | --- | --- |
| `CourtemancheAtrialModel` | Human working atrial myocyte, Courtemanche–Ramirez–Nattel 1998 informed | fast sodium; atrial `I_to` and `I_Kur`; `I_Kr`, `I_Ks`, `I_K1`; L-type Ca; Na/K pump; Na/Ca exchange; sarcolemmal Ca pump; one SR pool | Reduced 15-state membrane/Ca subsystem, not the full published 21-state cell |
| `ZhangSinoatrialModel` | Rabbit peripheral SAN, Zhang et al. 2000 informed | `I_f`; L- and T-type Ca; rapid/slow and inward-rectifier K; one SR pool | Reduced autonomous pacemaker subsystem, not a central/peripheral parameter sweep |
| `InadaAtrioventricularModel` | Rabbit compact AV node, Inada et al. 2009 informed | compact-node fast Na; L-type Ca; `I_to`; `I_Kr`, `I_K1`; `I_f`; cytosolic Ca removal | Reduced compact N-cell subsystem, not the complete AN–N–NH tissue model |
| `StewartPurkinjeModel` | Human Purkinje fibre, Stewart et al. 2009 informed | fast Na; L-type Ca; `I_to`; `I_Kr`, `I_Ks`, `I_K1`; `I_f`; pump/exchange; one SR pool | Reduced 13-state cellular subsystem, not the full published ion-handling model |

The qualification names are part of each parameter fingerprint. A result
must therefore be described with the exact reduced identity above; it must
not be presented as a simulation of the complete reference model.

## Governing convention

All families use:

- voltage in mV and time in ms;
- concentrations in mM;
- membrane-current density in pA/pF;
- outward-positive named and total ionic currents;
- outward-positive applied current, so `dV/dt = -(I_ion + I_applied)`;
- Hodgkin–Huxley gates with `dx/dt = (x_inf(V) - x) / tau_x(V)`;
- Nernst potentials from explicit fixed intra/extracellular concentrations;
- membrane Ca current converted to mM/ms before SR uptake, leak, release, or
  cytosolic removal is applied.

`rates` returns one typed rate-system record containing `state_rate`,
`currents`, `calcium`, and `evidence`. The individual current fields are
available before summation, and the calcium output keeps membrane current,
current-to-concentration flux, and intracellular flux terms separate. There
is no hidden stimulus.

## Plan, prepare, initialize, evaluate

```python
import jax
import jax.numpy as jnp

from phydrax.applications.cardiovascular.electrophysiology import (
    CourtemancheAtrialParameters,
    ZhangSinoatrialParameters,
)

atrial_plan = CourtemancheAtrialParameters(g_kur_scale=1.0)
atrial = atrial_plan.prepare()
atrial_state = atrial.initialize((128,), dtype=jnp.float32)
atrial_evaluation = jax.jit(atrial.rates)(atrial_state)

san = ZhangSinoatrialParameters().prepare()
san_state = san.initialize((128,), dtype=jnp.float32)
san_evaluation = jax.jit(san.rates)(san_state)

assert atrial_evaluation.currents.total_ionic.shape == (128,)
assert san_evaluation.calcium.net_cytosolic_flux_mM_per_ms.shape == (128,)
```

The parameter object is the host-side plan. `prepare()` freezes its exact
coefficient fingerprint together with the state-layout fingerprint into
`model_id`. `initialize()` then creates only device arrays of the requested
batch shape. Coefficients and topology do not change inside differentiated
execution.

Every coefficient contributes to `parameter_id`; a conductance change
therefore changes both `parameter_id` and the prepared `model_id`.

## Monodomain reaction adapters

The typed SoA API remains the inspectable cellular interface. Explicit
adapters provide the homogeneous final-axis `CardiacReactionModel` contract
consumed by `prepare_reaction` and the monodomain worksets:

```python
import numpy as np

from phydrax.applications.cardiovascular.electrophysiology import (
    CourtemancheAtrialReactionAdapter,
    plan_reaction,
    prepare_reaction,
)

reaction_model = CourtemancheAtrialReactionAdapter(cell_model=atrial)
reaction = prepare_reaction(
    plan_reaction(reaction_model, node_count=128, dtype=np.float64)
)
voltage_mV, local_state = reaction.initialize()
surface_evaluation = reaction_model.evaluate(reaction_model.initialize((128,)))

assert voltage_mV.shape == (128,)
assert local_state.shape == (128, 14)
assert surface_evaluation.current_density_uA_per_mm2.shape[-1] == 12
```

The other explicit routes are `ZhangSinoatrialReactionAdapter`,
`InadaAtrioventricularReactionAdapter`, and
`StewartPurkinjeReactionAdapter`. Each exposes only its model-local final-axis
layout; the adapters do not introduce a cross-phenotype union.

Native pA/pF currents are multiplied by the configured membrane capacitance
to produce physical surface current in µA/mm². The reaction stimulus is
outward-positive and contributes `-stimulus / Cm` to voltage rate.
`PreparedReaction.rates` separately applies its inward-positive volumetric
stimulus after this zero-stimulus evaluation.

Gate steady states and time constants are carried into
`CardiacReactionEvaluation`, so `exact_gate_update` performs the analytic
first-order relaxation and leaves voltage and concentration channels
unchanged. Negative or nonfinite timesteps fail closed. Current components
are final-axis aligned with `current_names`; charge-balance residuals use the
same physical surface units.

The adapter parameter array is an exact device declaration of the already
prepared host-side coefficient plan. Supplying coefficients that differ from
that frozen plan is inadmissible rather than silently changing a static model
inside compiled execution. To tune coefficients, construct a new typed
parameter plan, prepare it, then construct a new adapter and record its new
`model_id`.

## Fixed SoA schemas

The schemas are deliberately independent:

- atrial: `voltage_mV, m, h, j, oa, oi, ua, ui, xr, xs, d, f, f_ca,
  calcium_i_mM, calcium_sr_mM`;
- SAN: `voltage_mV, y_f, d_l, f_l, d_t, f_t, x_r, x_s, calcium_i_mM,
  calcium_sr_mM`;
- AV node: `voltage_mV, m, h, d_l, f_l, r_to, q_to, x_r, y_f,
  calcium_i_mM`;
- Purkinje: `voltage_mV, m, h, j, d, f, x_r, x_s, r_to, s_to, y_f,
  calcium_i_mM, calcium_sr_mM`.

A layout's `pack(state)` produces `(state_size, *batch_shape)` and
`unpack(values)` restores the same typed state. `names`, `state_size`,
`index(name)`, and `layout_id` make checkpoint and coupling mappings
explicit. Packing is an interchange operation; normal model execution keeps
fieldwise SoA arrays and does not allocate a padded cross-phenotype union.

## Phenotype behavior

The supplied initialization fixtures exercise different baseline behavior:

| Model | Initial voltage (mV) | Total ionic current (pA/pF) | Initial `dV/dt` (mV/ms) |
| --- | ---: | ---: | ---: |
| atrial | -81.180 | 0.0006063971 | -0.0006063971 |
| peripheral SAN | -58.000 | -0.1756208912 | 0.1756208912 |
| compact AV node | -60.000 | -0.0019997869 | 0.0019997869 |
| Purkinje | -69.137 | -0.0174153131 | 0.0174153131 |

Thus the atrial fixture is near current balance, while SAN, AV-node, and
Purkinje fixtures expose successively distinct autonomous depolarizing
balances. These values are deterministic implementation-lock fixtures at the
listed initial conditions, not a substitute for protocol-level validation
against experimental traces.

## Admissibility evidence

`admissibility(state)` returns per-cell evidence without synchronizing to the
host. The evidence contains:

- all-state finiteness;
- gate minimum, maximum, and maximum interval violation;
- minimum calcium concentration;
- maximum voltage magnitude;
- an integer fail-closed status bitset; and
- `successful`, which is true only when the status is zero.

The status distinguishes nonfinite values, gates outside `[0, 1]`,
nonpositive calcium, and voltage outside the qualified ±200 mV envelope.
Rates and named currents remain inspectable when evidence fails, but callers
must not commit or checkpoint a candidate with unsuccessful evidence.

A production stepping workflow should therefore be:

1. create and fingerprint the parameter plan;
2. prepare the fixed model and layout;
3. initialize or restore the exact typed state;
4. compute a candidate with the chosen ODE integrator;
5. evaluate currents, calcium output, and admissibility evidence;
6. commit only a successful candidate and record `model_id` plus `layout_id`.

## References

- Courtemanche M, Ramirez RJ, Nattel S. *Ionic mechanisms underlying human
  atrial action potential properties*. American Journal of Physiology 275,
  H301–H321 (1998), PMID 9688927. [Physiome model record](https://models.cellml.org/workspace/courtemanche_ramirez_nattel_1998)
- Zhang H, Holden AV, Kodama I, et al. *Mathematical models of action
  potentials in the periphery and center of the rabbit sinoatrial node*.
  American Journal of Physiology 279, H397–H421 (2000), PMID 10899081.
  [Physiome model record](https://models.cellml.org/exposure/01f6a47881da1925315d1d89d3a8d901/view)
- Inada S, Hancox JC, Zhang H, Boyett MR. *One-dimensional mathematical model
  of the atrioventricular node including atrio-nodal, nodal, and nodal-His
  cells*. Biophysical Journal 97, 2117–2127 (2009), PMID 19843444.
  [Physiome model record](https://models.cellml.org/exposure/d724fc43a0766bd29bedc1ca27f3380d)
- Stewart P, Aslanidi OV, Noble D, Noble PJ, Boyett MR, Zhang H. *Mathematical
  models of the electrical action potential of Purkinje fibre cells*.
  Philosophical Transactions of the Royal Society A 367, 2225–2255 (2009),
  PMID 19414454. [Physiome model record](https://models.cellml.org/exposure/38cf8387b0707f0ef6947f009710aeb5/stewart_aslanidi_noble_noble_boyett_zhang_2009.cellml/view)
