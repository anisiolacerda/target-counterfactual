Perfect—I'll proceed assuming **reach–avoid is the headline** and use an ICU-style **BP/MAP-in-range** target-set as the running example, while keeping the method general. This builds directly on the VCIP core framing and limitations we summarized earlier. 

---

# NeurIPS Paper Outline (R²-VCIP)

## Title (working)

**R²-VCIP: Robust Reach–Avoid Counterfactual Intervention Planning with Variational Latent Dynamics**

## Abstract (what it should say, in 5 sentences)

1. We study *individualized* counterfactual planning: selecting an intervention sequence to reach a desired clinical state.
2. Unlike prior work targeting point outcomes and assuming ignorability, we target **sets** (clinically acceptable ranges) and enforce **safety** across the horizon.
3. We further introduce **robustness** to unobserved confounding via a sequential sensitivity model, yielding a max–min planning objective.
4. We derive a **robust variational bound** compatible with latent state-space models and enable gradient-based planning through an adversarial reweighting mechanism.
5. Experiments in simulated tumor growth (known counterfactuals) and semi-synthetic ICU setups show improved target attainment, safer trajectories, and graceful degradation under increasing confounding.

---

# 1. Introduction

### Motivation

* Clinical and control applications often require: “get patient into *acceptable range* by time (t+\tau), without entering dangerous states in-between.”
* Real-world treatment assignment is confounded → standard planning under ignorability can recommend unsafe or brittle strategies.

### Gap in VCIP-style planning

* VCIP is strong on **achievement likelihood vs distance-to-target** and long-horizon stability, but:

  1. targets are effectively **point-based** (or treated as such),
  2. **sequential ignorability** is a hard dependency.

### Contributions (bullets)

* **Set-target + safety:** formulate **reach–avoid target achievement** for individualized intervention planning.
* **Robustness:** incorporate a **sequential confounding sensitivity model**; plan for **worst-case** target achievement.
* **Method:** robust variational objective + adversarial reweighting + stable gradient planning.
* **Theory:** certificate-style bound relating robust surrogate to worst-case achievement probability (with soft-indicator approximation).
* **Empirics:** robustness curves vs (\Gamma), improved safety/attainment tradeoffs.

---

# 2. Related Work (tight)

* Counterfactual estimation + planning; DTR/policy learning; model-based RL vs causal planning.
* Reachability / safe RL / reach–avoid control.
* Sensitivity analysis / distributionally robust causal inference (bounded odds ratio / density ratio).
* Latent state-space models for clinical trajectories (and why they are helpful but insufficient without robustness).

---

# 3. Problem Setup

## 3.1 Data & causal setting

* Observational trajectories: ((X_t, A_t, Y_t, V)), history (\bar H_t), horizon (\tau).
* Potential outcomes (Y_s[\bar a]) for intervention sequence (\bar a).

## 3.2 Reach–avoid event

Let (\mathcal T) = target range at terminal time, (\mathcal S) = safe set across horizon. Define:
[
E(\bar a) := {Y_{t+\tau}[\bar a]\in\mathcal T}\cap \bigcap_{s=t}^{t+\tau} {Y_s[\bar a]\in\mathcal S}.
]
Goal (non-robust):
[
\max_{\bar a};\Pr(E(\bar a)\mid \bar H_t).
]

## 3.3 Robust objective under confounding

Introduce ambiguity set (\mathcal Q(\Gamma)) (sequential sensitivity):
[
\boxed{\max_{\bar a};\inf_{q\in\mathcal Q(\Gamma)} \Pr_q(E(\bar a)\mid \bar H_t).}
]
(\Gamma=1) recovers the ignorability-like case; larger (\Gamma) = stronger hidden confounding.

---

# 4. Method

## 4.1 Latent dynamics backbone (keep VCIP’s strengths)

* Latent state (Z_s), transition (p_\theta(Z_{s+1}\mid Z_s, a_s)), emission (p_\theta(Y_s\mid Z_s)).
* Variational posterior (q_\phi(Z_{t:t+\tau}\mid \bar H_t,\bar a)).

## 4.2 Differentiable reach–avoid likelihood

Replace indicators with soft gates:

* (g_{\mathcal T}(y)\approx \mathbf 1{y\in\mathcal T}),
* (g_{\mathcal S}(y)\approx \mathbf 1{y\in\mathcal S}).

Define reach–avoid score:
[
J_{\text{RA}}(\bar a;\theta)
= \log \mathbb E_\theta!\left[g_{\mathcal T}(Y_{t+\tau})\prod_{s=t}^{t+\tau} g_{\mathcal S}(Y_s)\mid \bar H_t,\bar a\right].
]
Use reparameterized Monte Carlo over (Z, Y) (variance reduction: shared randomness across candidate (\bar a)).

## 4.3 Sequential sensitivity via bounded density ratios

Let (\pi_0(a_s\mid \bar H_s)) be a nominal (estimated) behavioral policy. Hidden confounding induces alternative (\pi_q).
Define weights:
[
w_s(\bar H_s,a_s)=\frac{\pi_q(a_s\mid \bar H_s)}{\pi_0(a_s\mid \bar H_s)},
\quad
w_s\in[1/\Gamma,\Gamma],\quad
\mathbb E_{\pi_0}[w_s\mid\bar H_s]=1.
]

## 4.4 Robust reach–avoid objective (max–min)

Robustified score:
[
J_{\text{RRA}}(\bar a)
:= \inf_{{w_s}\in\mathcal W(\Gamma)};
\log \mathbb E_{\theta,\pi_0}!\left[
\left(\prod_{s=t}^{t+\tau-1} w_s\right)
g_{\mathcal T}(Y_{t+\tau})
\prod_{s=t}^{t+\tau} g_{\mathcal S}(Y_s)
\mid \bar H_t,\bar a\right].
]
Planning:
[
\boxed{\max_{\bar a} J_{\text{RRA}}(\bar a).}
]

### Implementation note (stability)

Parameterize (w_s=\exp(u_s)), project (u_s\in[-\log\Gamma,\log\Gamma]). Solve with a few inner adversary steps per planner step.

---

# 5. Theory (main statement you should aim for)

## Theorem (Certificate-style robust bound; target-set + safety)

Under (i) bounded sensitivity weights (\mathcal W(\Gamma)), (ii) well-defined latent trajectory density, (iii) bounded variational gap (\varepsilon_{\text{VI}}), we have for any (\bar a):
[
J_{\text{RRA}}(\bar a)
\le
\log \inf_{q\in\mathcal Q(\Gamma)} \Pr_q(E(\bar a)\mid \bar H_t)
\le
J_{\text{RRA}}(\bar a) + \varepsilon_{\text{VI}} + \varepsilon_{\text{soft}}(\kappa),
]
with (\varepsilon_{\text{soft}}(\kappa)\to 0) as the soft-indicator hardness increases.

**Interpretation:** (J_{\text{RRA}}) is an *optimizable certificate* for worst-case reach–avoid achievement probability (up to controlled approximation).

---

# 6. Algorithm (pseudocode)

```text
Algorithm 1: R²-VCIP (Robust Reach–Avoid VCIP)

Inputs:
  Observational data D, horizon τ
  Target set T, safe set S, softness κ
  Sensitivity Γ ≥ 1
  Steps: N_train, N_plan
  Adversary inner steps K

Phase A — Train latent dynamics model
  Initialize θ, φ
  for iter = 1..N_train:
    Sample minibatch trajectories from D
    Compute variational loss:
      - standard state-space ELBO terms (reconstruction + KL)
      - optional calibration loss for event gates g_T, g_S
    Update θ, φ with gradient descent

Phase B — Robust planning for a new instance with history H_t
  Initialize intervention sequence ā (e.g., from behavior policy or zeros)
  Initialize adversary logits u_s = 0  (so w_s = 1)

  for plan_iter = 1..N_plan:
    # Inner loop: adversary (minimize robust score)
    for k = 1..K:
      w_s = exp(u_s);  project u_s to [-log Γ, log Γ]
      Estimate Monte Carlo objective:
        Ĵ = log E[ (∏_s w_s) * g_T(Y_{t+τ}) * ∏_{s=t}^{t+τ} g_S(Y_s) | H_t, ā ]
      Update u_s by gradient DESCENT on Ĵ

    # Outer loop: planner (maximize robust score)
    w_s = exp(u_s); project as above
    Recompute Ĵ with shared randomness
    Update ā by gradient ASCENT on Ĵ
    Optionally: project ā to action constraints (box / discrete relaxation)

  Return optimized intervention sequence ā*
```

---

# 7. Experiments (minimal matrix that looks “NeurIPS-ready”)

## 7.1 Synthetic (tumor sim with explicit hidden confounder)

* Construct (U_s) affecting both action propensity and dynamics.
* Vary: confounding strength, overlap, horizon (\tau).
* Targets: (Y_{t+\tau}\in\mathcal T) (tumor below threshold), safety: toxicity proxy in (\mathcal S).
  **Report:** attainment, safety violations, robustness curve vs (\Gamma), performance vs (\tau).

## 7.2 Semi-synthetic ICU (MIMIC/eICU-style)

* Define (\mathcal T): MAP/BP in clinician range at (t+\tau); (\mathcal S): avoid hypotension events.
* Since counterfactuals unknown:

  * ranking-based evaluation over candidate sequences,
  * stability of ranking as (\Gamma) increases,
  * plausibility/shift: divergence from behavior policy vs performance.

## Baselines

* VCIP (point target / distance-driven ranking),
* Set-VCIP (reach–avoid without robustness, (\Gamma=1)),
* sensitivity-agnostic safe planning baseline (if available),
* optionally: model-based RL / safe RL baselines (careful: keep comparisons fair).

## Ablations (must-have)

* (\Gamma\in{1, 1.25, 1.5, 2, 3})
* reach-only vs reach–avoid
* hardness (\kappa) (soft vs hard)
* inner adversary steps (K)
* with/without calibration regularizer

---

# 8. What you should do next (no questions needed)

To turn this into an executable plan, I’d recommend locking these *defaults* now:

1. **Outcome (Y):** MAP (or diastolic BP, matching VCIP’s MIMIC setup), scalar.
2. **Target set (\mathcal T):** interval ([65, 75]) (example) at (t+\tau).
3. **Safe set (\mathcal S):** MAP (\ge 60) for all steps (avoid hypotension).
4. **Action space:** start with Bernoulli actions (vasopressor / ventilation), then extend.
