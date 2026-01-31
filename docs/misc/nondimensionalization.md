# Nondimensionalization

Nondimensionalization is common in many engineering and physics problems: it improves numerical conditioning, makes parameter scales explicit, and helps compare regimes across systems.

Mathematically, you choose characteristic scales (e.g. length \(L\), time \(T\), and state scale \(U\)) and define dimensionless variables \(\hat{x},\hat{t},\hat{u}\) via

$$
x = L\hat{x},\qquad t = T\hat{t},\qquad u = U\hat{u}.
$$

Derivatives transform by the chain rule, e.g.

$$
\begin{aligned}
\frac{\partial}{\partial t} &= \frac{1}{T}\frac{\partial}{\partial \hat{t}}, \\
\frac{\partial}{\partial x} &= \frac{1}{L}\frac{\partial}{\partial \hat{x}}, \\
\frac{\partial^2}{\partial x^2} &= \frac{1}{L^2}\frac{\partial^2}{\partial \hat{x}^2}.
\end{aligned}
$$

Substitute these into the governing equations and divide through by a reference scale; the remaining coefficients become dimensionless groups (e.g. Reynolds and Péclet numbers) that control the behavior. Convert back by undoing the scaling (\(x=L\hat{x}\), \(t=T\hat{t}\), \(u=U\hat{u}\)).

We recommend using [`unxt`](https://github.com/GalacticDynamics/unxt) for units, conversions, and nondimensionalization workflows.
