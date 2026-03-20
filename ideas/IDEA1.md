Great—here’s a **NeurIPS-style research sketch** for the combined direction:

# R²-VCIP: Robust Reach–Avoid Variational Counterfactual Intervention Planning

## 0) One-line pitch

**Plan individualized intervention sequences that (i) reach a *target set* and (ii) avoid *unsafe regions*, while being *provably robust* to bounded unobserved confounding.**

---

## 1) Problem statement (formal)

### Setting

Discrete-time longitudinal system with history (\bar H_t), actions (A_t), outcomes (Y_t) (possibly vector), horizon (\tau). We plan an intervention sequence (\bar a_{t:,t+\tau-1}).

### Reach–avoid target achievement

Define:

* **Target set** (\mathcal T \subseteq \mathbb R^d) (e.g., (Y_{t+\tau}) within a clinically acceptable range).
* **Safe set** (\mathcal S \subseteq \mathbb R^d) (e.g., BP avoids hypotension/hypertension over the entire horizon).

Define the event:
[
E(\bar a) := \Big{Y_{t+\tau}[\bar a]\in \mathcal T\Big};\cap;\Big{\forall s\in{t,\dots,t+\tau}: Y_s[\bar a]\in \mathcal S\Big}.
]

### Robust objective under unobserved confounding

Let (\mathcal Q(\Gamma)) be a **sequential sensitivity / ambiguity set** encoding bounded departures from ignorability, parameterized by (\Gamma \ge 1) (larger (\Gamma) = more confounding).

We propose:
[
\boxed{
\bar a^* \in \arg\max_{\bar a};;\inf_{q\in \mathcal Q(\Gamma)};\Pr_q\big(E(\bar a)\mid \bar H_t\big)
}
]
This is **robust reach–avoid CTA**.

---

## 2) Modeling backbone: keep VCIP’s latent-state SCM, but change the “goal likelihood”

We retain the latent state (Z_s) idea:

* (Z_s) summarizes system state given history and actions.
* A decoder defines (p_\theta(Y_s \mid Z_s)).
* A transition defines (p_\theta(Z_{s+1}\mid Z_s, a_s)) (or conditioned on covariates).

### Key change: event likelihood, not point likelihood

Instead of targeting a specific (Y_{\text{target}}), we need
[
\Pr_\theta(E(\bar a)\mid \bar H_t)
= \mathbb E_\theta\left[\mathbf 1{Y_{t+\tau}\in \mathcal T}\prod_{s=t}^{t+\tau}\mathbf 1{Y_s\in \mathcal S};\middle|;\bar H_t, \bar a\right].
]

To make it differentiable, use **soft indicators**:

* (g_\mathcal T(y)\approx \mathbf 1{y\in\mathcal T})
* (g_\mathcal S(y)\approx \mathbf 1{y\in\mathcal S})

Example: logistic barriers for interval targets (coordinate-wise):
[
g_{,[\ell,u]}(y) = \sigma(\kappa(y-\ell))\cdot \sigma(\kappa(u-y))
]
((\kappa) controls hardness).

Then define a **reach–avoid score**:
[
\boxed{
J_{\text{RA}}(\bar a;\theta)
:= \log;\mathbb E_\theta\left[g_\mathcal T(Y_{t+\tau})\prod_{s=t}^{t+\tau} g_\mathcal S(Y_s);\middle|;\bar H_t,\bar a\right]
}
]
Compute via reparameterized Monte Carlo over (Z) and (Y) (low-variance with common random numbers across candidate (\bar a)).

---

## 3) Robustness mechanism: a clean sequential sensitivity model + adversarial reweighting

### Ambiguity set via bounded density ratios (sequential)

A tractable, standard choice for (\mathcal Q(\Gamma)) is to bound how much hidden confounding can tilt action assignment relative to a nominal model conditioned on observed history:

For each step (s),
[
w_s(\bar H_s, a_s) := \frac{\pi_q(a_s\mid \bar H_s)}{\pi_0(a_s\mid \bar H_s)}
\quad\text{with}\quad
w_s \in \left[\frac1\Gamma,\Gamma\right], ;;\mathbb E_{\pi_0}[w_s\mid \bar H_s]=1.
]

This induces an adversarially reweighted trajectory measure. The worst-case achievement probability becomes an **infimum over feasible weights**.

### Robust reach–avoid objective (dual / saddle form)

Define a robustified lower bound:
[
\boxed{
J_{\text{RRA}}(\bar a)
:= \inf_{{w_s}\in\mathcal W(\Gamma)};
\log;\mathbb E_{\theta,\pi_0}\left[
\left(\prod_{s=t}^{t+\tau-1} w_s(\bar H_s,a_s)\right)
\cdot g_\mathcal T(Y_{t+\tau})
\cdot \prod_{s=t}^{t+\tau} g_\mathcal S(Y_s)
;\middle|;\bar H_t,\bar a
\right]
}
]

We then plan with:
[
\boxed{
\max_{\bar a}; J_{\text{RRA}}(\bar a)
}
]
This is a **max–min** (planner vs confounding adversary).

---

## 4) Learning + planning algorithm (practical)

### Phase A — Fit the latent model (VCIP-style, but RA-aware)

Train ((\theta,\phi)) on observational data:

* same latent state-space VI training,
* plus an auxiliary “event head” or use the decoder to estimate event surrogates (g_\mathcal T, g_\mathcal S),
* optionally add calibration regularization (reliability of event probabilities matters).

### Phase B — Robust planning at test time (per patient)

Given (\bar H_t), solve:

1. **Inner loop (adversary):** for current (\bar a), update weights (w) by projected gradient steps to minimize (J_{\text{RRA}}).
2. **Outer loop (planner):** update (\bar a) by gradient ascent to maximize (J_{\text{RRA}}).

Implementation details that make it stable:

* parametrize (w_s = \exp(u_s)) and project (u_s\in[-\log\Gamma,\log\Gamma]),
* use a small number of adversary steps per planner step,
* share Monte Carlo samples across steps (variance reduction).

### Optional extension: constraints as hard requirements

If you want **hard** safety constraints, do:
[
\max_{\bar a}; \log \Pr(Y_{t+\tau}\in\mathcal T \mid \cdot)
\quad \text{s.t.}\quad \Pr(\exists s: Y_s\notin\mathcal S \mid \cdot)\le \delta
]
and optimize via a Lagrangian multiplier that is also robustified.

---

## 5) What the “main theorem” could look like

You want one crisp theorem, not 5 small ones. A good flagship statement:

### Theorem (Robust variational reach–avoid lower bound)

Assume:

1. The latent model induces a well-defined trajectory density under (\pi_0),
2. the sensitivity weights satisfy the sequential bounded density ratio constraints (\mathcal W(\Gamma)),
3. the variational approximation error of the latent posterior is bounded by (\varepsilon_{\text{VI}}).

Then for any candidate (\bar a),
[
J_{\text{RRA}}(\bar a)
;\le;
\log \inf_{q\in\mathcal Q(\Gamma)}\Pr_q(E(\bar a)\mid \bar H_t)
;\le;
J_{\text{RRA}}(\bar a) + \varepsilon_{\text{VI}} + \varepsilon_{\text{soft}}(\kappa),
]
where (\varepsilon_{\text{soft}}(\kappa)\to 0) as the soft-indicator hardness (\kappa\to\infty).

Interpretation:

* (J_{\text{RRA}}) is a **certificate**: maximizing it yields sequences with **guaranteed worst-case** reach–avoid probability (up to controlled approximation terms).
* As (\Gamma\to 1), (J_{\text{RRA}}) collapses to the non-robust Set-VCIP objective.

This is the clean story reviewers want: *robustness knob* (\Gamma), *set-target realism*, and *bounded gap*.

---

## 6) Experimental matrix (minimal but strong)

### A) Synthetic: tumor simulator (ground truth counterfactuals)

**Purpose:** demonstrate correctness + robustness under *known* hidden confounding.

Design:

* Add an unobserved (U_s) that affects both action propensity and tumor growth.
* Vary confounding strength and overlap.
* Targets:

  * reach: tumor size (<) threshold by (t+\tau),
  * avoid: toxicity proxy below threshold at all times.

Report:

* worst-case attainment vs (\Gamma) (robustness curve),
* nominal attainment,
* violation rate,
* performance vs horizon (\tau),
* compute budget vs performance.

Baselines:

* VCIP (point target distance),
* Set-VCIP (no robustness),
* policy-learning baselines if feasible (optional),
* sensitivity-agnostic planners.

### B) Semi-synthetic ICU (MIMIC-III/IV or eICU)

**Purpose:** realism under irregularities and partial confounding.

Targets (examples):

* reach: MAP/BP within clinician-defined interval at (t+\tau),
* avoid: hypotension events during horizon,
* optionally multi-objective: lactate down + BP safe.

Evaluation (since true counterfactuals unknown):

* ranking evaluation on candidate sequences (as VCIP does),
* **robust ranking stability**: how rankings change as (\Gamma) increases,
* plausibility: action distribution shift vs clinical patterns,
* stress tests: propensity model misspecification and overlap pruning.

### Key ablations (must-have)

1. remove robustness ((\Gamma=1)),
2. remove avoid constraint (reach-only),
3. hard vs soft indicators,
4. number of adversary steps,
5. calibration regularizer on event probabilities.

---

## 7) What the NeurIPS “contribution bullets” become

1. **New objective:** robust reach–avoid counterfactual intervention planning (set targets + safety).
2. **Method:** a variational latent-state planner with **adversarial sensitivity reweighting** (max–min).
3. **Theory:** certified lower bound on worst-case reach–avoid achievement with explicit approximation terms and robustness knob (\Gamma).
4. **Empirics:** robustness curves under controlled confounding + improved stability on clinical data.

