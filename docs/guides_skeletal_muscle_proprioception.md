# Skeletal-muscle proprioception

`MileusnicSpindle2006Plan` implements the three-branch feline muscle-spindle
model of Mileusnic, Brown, Lan, and Loeb, DOI
[`10.1152/jn.00868.2005`](https://doi.org/10.1152/jn.00868.2005).
It is a receptor/transducer model, not a reflex controller.

Inputs are fascicle length normalized by optimal fascicle length, its first and
second time derivatives, and dynamic/static gamma frequencies in pulses per
second. Whole musculotendon length is not a valid substitute. The state carries
bag1/bag2 fusimotor filters and tension/tension-rate for bag1, bag2, and chain
intrafusal branches. Outputs are Ia and group-II rates in impulses per second.
The source force unit is arbitrary and scale-invariant.

The runtime uses the source equations and Table-1 feline parameters with a
fixed-step RK4 transaction. The source did not publish solver settings or initial
numerical states, so `initialize()` computes a declared zero-velocity mechanical
equilibrium. The default maximum step is 0.1 ms. A failed state, input, parameter,
or step proposal rolls back the complete receptor state.

The model was fitted to cat soleus records and validated on cat medial
gastrocnemius data. It omits stiction, initial burst, and movement-history effects,
and it must not be represented as a human-generic spindle.

## Golgi tendon-organ source gate

The companion feline Mileusnic–Loeb model,
DOI [`10.1152/jn.00869.2005`](https://doi.org/10.1152/jn.00869.2005),
is not implemented or exported. Direct inspection of the
[author-hosted paper](https://viterbi.usc.edu/pdfs/gloeb/53681.pdf) resolves the
previous exponent concern: the collagen spring in Eq. 2 is **cubic (3)**;
the force-dependent dashpot in Eq. 6 uses **a = 0.4**. These are different
constitutive terms, not alternative readings of one exponent.

The input contract is a fixed vector of physical tensile forces in **N**, one
per receptor-inserting fiber, together with receptor/fiber/MU identities, fiber
types, collagen areas in **µm²**, and the receptor-specific two-network
partition. Several paper protocols specify usable partitions explicitly; these
do not define a universal per-fiber partition for an arbitrary receptor.
A caller-supplied, content-identified `P_common1` array is therefore mandatory;
no general MG or soleus topology preset is inferred. The source output is the
larger of two continuous
generator rates in impulses/s, not Ib spike events or a reflex controller.

The remaining reproducibility gates are more specific than an exponent:

- **Force-balance interpretation:** Eq. 1 on p. 1791 partitions fiber tension
  between bypass and both common networks. However, “Construction of the
  model” on p. 1792 says that all fiber tension determines bypass length,
  before adjusting the innervated lengths to it. Substituting total fiber
  tension for bypass tension in Eq. 3 is not the exact Eq. 1 equilibrium when
  the common networks carry net force. An exact Eqs. 1–7 reconstruction cannot
  silently be identified with this stated Simulink approximation, nor may an
  accepted force imbalance be presented as conserved mechanics.
- **Initialization and numerics:** the original Simulink model, initialization,
  algebraic-loop treatment, input waveforms, solver and tolerances have not been
  qualified. An independent numerical realization must preserve the chosen
  source equations, establish force-compatible equilibrium, certify algebraic
  residuals and stiff-step convergence, and reproduce the source observations.
- **Scenario and oracle identity:** Fig. 3A's MU-averaged oracle is
  reconstructible as a distinct source representation: the paper explicitly
  replaces the 20-fiber sums in Eqs. 5–7 with 13-MU sums, and Table 1's average
  entries represent 1.6 fibers per MU. Its conceptual 20.8 fibers are not an
  integer per-fiber topology, but are not a missing-input blocker for that
  MU-averaged oracle. Fig. 4 specifies two 2-s tetani separated by 0.5 s and
  the equal, matched 90/10, and opposed 90/10 partitions. Fig. 6 explicitly
  gives five opposing MU-pair partitions, from 90/10 versus 10/90 through
  50/50 versus 50/50 in 10-percentage-point increments.
  A separate, real conflict affects Fig. 5A: Table 1's footnote specifies
  20 fibers from eight FF and six S MUs, whereas the results and figure
  caption specify 14 FF fibers plus seven S fibers from eight FF and five S
  MUs. These conflicting scenarios must not be silently reconciled.
- **Validation and reuse:** rendered Fig. 3–6 are accessible, but raw traces or
  reviewed digitization with axis calibration, uncertainty and reuse rights
  have not been frozen as a qualified oracle. The 0.5-s FF response and
  eight-MU recordings from
  [Gregory–Proske 1979](https://pmc.ncbi.nlm.nih.gov/articles/PMC1279043/)
  were used for parameter fitting, so they are not independent holdout data.
  The paired-MU comparison additionally needs
  [Gregory–Morgan–Proske 1985](https://pubmed.ncbi.nlm.nih.gov/4087039/).

The [USC software download page](https://bme.usc.edu/msms-software-downloads/)
provides MSMS Beta 0.9.4, but acquiring its Windows installer does not establish
that it contains the exact 2006 receptor model or grants reuse of that model.
Neither a replacement private runtime nor a GTO benchmark is presented as
production closure while these gates remain open.

Even after standalone qualification, integrated feedback requires a separately
qualified **terminal per-fiber mechanical-tension projection**. A homogenized
continuum resultant, lumped tendon force, Fuglevand arbitrary force, and
Shorten crossbridge concentration cannot recover those tensions. The GTO remains
an observation-only transducer downstream of the sole mechanical-force owner.
No scalar tendon-force-to-Ib approximation or generic reflex law is substituted.

## Spindle qualification

Run:

```text
python examples/skeletal_muscle_proprioception.py
python tools/skeletal_proprioception_qualification.py
python benchmarks/skeletal_muscle_proprioception.py
```

Differentiation is local to the fixed continuous spindle equations and integration
path. A future event-generating afferent or closed-loop circuit would require a
separate source identity, fixed event capacity, latency contract, and validation.

## Neural event and controller boundaries

The continuous spindle remains valid independently of an afferent spike layer.
[Niu–Nandyala–Sanger 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4255602/)
does not supply numerical Izhikevich `(a, b, c, d)` parameters, initial
membrane/recovery states, or a complete rate-to-current mapping. Its emulation
also adds 5-mV uniform pseudorandom membrane noise generated by an LFSR; it is
not a deterministic, parameter-complete adapter. Reproducing it requires the
author's source, numerical initialization, rate-to-current transform, LFSR
recurrence/streams and reference spike data, rather than conventional neuron
defaults.

[Geyer–Herr 2010](https://doi.org/10.1109/TNSRE.2010.2047592) is a separate
task-specific walking controller, not a generic spinal circuit or an Ib
measurement model. Source-archive reuse permission and frozen reference-output
qualification remain unresolved. Neither an event adapter nor a controller
runtime is exported, and neither changes the existing rate-only spindle.
