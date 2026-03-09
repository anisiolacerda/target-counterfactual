## Project Schedule

With this goal in mind, the overall steps, currently, are:

- Step 1: Understand the VCIP's codebase located at lightning-hydra-template-main/src/vendor/VCIP.
- Step 2: Using the above understanding of the VCIP's codebase, we will try to replicate the results from the paper. Here, we need to adapt the code to run each result presented at the VCIP paper (available at: literature_review/pdfs/VCIP.pdf).
- Step 2b: ✓ **COMPLETE.** Replicated VCIP results on MIMIC-III real ICU data (feature/mimic-iii-experiments branch). All 5 models × 5 seeds trained on Vast.ai (~19.5 hours). All 3 paper claims verified (VCIP GRP: 0.876→0.992 across tau=2..8; baselines degrade). Analysis notebook: `VCIP/results/mimic/analysis.ipynb`. Remaining: slice discovery for patient subgroup analysis. See `context/MIMIC_DATA_SETUP.md` and `context/MIMIC_EXPERIMENT_MAP.md`.
- Step 2 (Cancer): ✓ **COMPLETE.** Full Cancer replication: 5 models × 5 seeds × 4 gammas (220 runs, 52h on 4× RTX 2060, ~$9.75). All 3 paper claims verified. VCIP GRP: 0.932→0.994 at gamma=4. Ablation confirms confounding adjustment worth ~20 GRP points. Analysis notebook: `VCIP/results/cancer/analysis.ipynb`.
- Step 3: ✓ **MOSTLY COMPLETE.** Weakness analysis notebook (`VCIP/results/weakness_analysis.ipynb`) identifies 6 weaknesses and 4 research directions. Key findings: (W1) ELBO-True correlation rho=0.75-0.90, Top-1 agreement only 19-76%; (W3) single target limitation; (W6) no counterfactual validation on real data. Confounding sensitivity analysis shows VCIP advantage grows with gamma. Remaining: extended ablation (latent dim, LSTM capacity).
