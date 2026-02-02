# Surely You're Joking!

This appendix gives intuition and background for Phydrax’s **FeynmaNN** architecture (`phydrax/nn/models/architectures/_feynmann.py`).
The name is a nod to Feynman’s “sum over histories” viewpoint: *many contributions, each carrying a phase, combine by interference.*

This is architectural inspiration, not a claim that the model is literally implementing quantum mechanics.

---

## 1) The path-integral intuition (in one minute)

In Feynman’s path-integral picture, a quantity is written schematically as

\[
\text{(quantity)} \approx \int \exp(i\,\text{action}) \;(\text{contribution})\; d(\text{paths}),
\]

and the key phenomenon is **interference**: phases can make contributions add constructively or cancel.

Another useful picture is **phasors**: each “path” contributes a little arrow in the complex plane. The factor
\(\exp(i\,\text{action})\) is a unit-length rotation of that arrow. The integral (or sum) adds all these arrows.
If the phase varies wildly across paths, the arrows point in many directions and mostly **cancel**; if the phase
is locally coherent, contributions **align** and survive. This “rapidly varying phase cancels out” intuition is the
heart of why path integrals produce sharp, structured behavior without needing any single contribution to be huge.

FeynmaNN borrows exactly this motif: it replaces the continuum of paths by a small set of learnable contributions,
and learns phase modulation so the model can switch between reinforcement and cancellation depending on the current state.

In numerics, integrals are often approximated by **finite sums**. FeynmaNN takes the same “sum of phase-modulated contributions”
motif and turns it into a learnable neural building block.

---

## 2) FeynmaNN in one equation

FeynmaNN builds and updates a complex latent state \(z \in \mathbb{C}^n\) using a repeated “sum-over-paths” block:

\[
z \;\mapsto\; \sum_{k=1}^{K} g_k \, e^{i\,\alpha_k(z)} \, (W_k z + b_k),
\]

where:

- \(K\) is the number of “paths” (think: experts / branches / histories).
- Each path has its own complex affine transform \(W_k z + b_k\).
- \(g_k \ge 0\) are mixing weights with \(\sum_k g_k = 1\).
- \(\alpha_k(z)\in\mathbb{R}\) are learned phases (an “action-like” quantity), typically produced by a small real network that
  looks at \((\Re z, \Im z)\).
- \(e^{i\alpha_k(z)}\) is what makes the sum behave differently from an ordinary mixture: it enables **cancellation**, not just blending.

After each block, FeynmaNN applies a complex nonlinearity (ModReLU) and repeats for `depth` layers, then maps the final
complex latent to the requested output.

---

## 3) What phases buy you: learned interference

Most mixture mechanisms are magnitude-only: they blend contributions. Phases add a second control axis:

- **Constructive interference**: phases align and contributions reinforce.
- **Destructive interference**: phases oppose and contributions cancel.
- **State-dependent combination**: because \(\alpha(z)\) depends on the current latent, which paths reinforce/cancel can change
  across the input domain.

This is a natural inductive bias for targets with oscillatory or interference-like structure (even when the final output is real):
waves, resonances, dispersive responses, multi-scale signals, and fields with sharp phase changes.

---

## 4) “Action” vs “amplitude”: two distinct levers

A useful mental model is that the block separates two roles:

- **Amplitude / contribution**: what each path proposes, via \(W_k z + b_k\).
- **Phase (action-like)**: how proposals should interfere, via \(e^{i\alpha_k(z)}\).

The gates \(g_k\) act like a stable, normalized weighting (a measure-like control), while phases handle reinforcement/cancellation.

The `phase_scale` hyperparameter is the cleanest “how oscillatory are we allowed to be?” dial:
larger values create faster phase variation (more cancellation potential), smaller values make the block behave closer to a conventional
gated mixture.

---

## 5) Real targets vs complex targets

Many PDE/physics problems have real-valued targets even when oscillations matter internally. FeynmaNN supports both:

- **Real output (default)**: use the complex latent as internal representation and read out real features.
- **Complex output (optional)**: keep the model complex through the final projection when the target is genuinely complex.

Either way, the complex latent acts as a compact carrier of phase information that a purely real latent would need to encode indirectly.

---

## 6) Hyperparameters: how to reason about them

A few parameters have especially clear roles:

- `num_paths` (\(K\)):
  - Expressivity of the superposition. Larger \(K\) increases the number of simultaneously-available contributions.
  - Too small: behaves closer to a standard complex network. Too large: more compute/params and potentially harder optimization.

- `phase_scale`:
  - Interference strength. Higher means more oscillatory phase modulation; lower means smoother, more mixture-like behavior.

- `width_action`:
  - Capacity of the phase generator \(\alpha(z)\). If too small, phases are blunt; if too large, phases can become overly chaotic early in training.

- `depth`, `width_size`:
  - Standard capacity knobs, with the twist that depth composes multiple interference steps.

- `learn_gates`:
  - If enabled, the model learns stable mixture weights \(g\). If disabled, it behaves more like a uniform sum-over-paths and relies more on phases.

---

## 7) Beyond Physics

Beyond the physics, the **FeynmaNN** (FeyNN for short) can also be used as a drop-in **new base architecture** alongside **MLP** and **KAN** regardless of application, and can offer substantial improvements.

Try it out!
