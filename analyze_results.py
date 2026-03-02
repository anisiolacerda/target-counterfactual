"""
Analyze replicated experiment results and compare against VCIP paper.

Results structure:
- rank=True (discrete): case_infos/{seed}/{test}/case_infos_{model}.pkl
  → GRP and RCS metrics for Table 3 (GPR/RCS columns) and Figure 4
- rank=False (continuous): {test}/mse.csv
  → Target distance for Tables 1-2 and Figure 6

Paper reference metrics:
- Table 1: Target distance, gamma=4, tau=1..12, identical strategies (test=False)
- Table 2: Target distance, gamma=4, tau=1..12, distinct strategies (test=True)
- Table 3: Ablation (GPR, RCS at gamma=4; target distance at gamma=1..4)
- Figure 4: GRP/RCS boxplots, gamma=4, tau=2,4,6,8
- Figure 6: Target distance, gamma=1,2,3, tau=1..6
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results_remote/my_outputs/cancer_sim_cont/22")
SEEDS = [10, 101, 1010, 10101, 101010]
MODELS = ["VCIP", "RMSN", "CRN", "CT", "ACTIN"]
# Models with sub-directories (hyperparameter in path)
MODEL_PATHS = {
    "VCIP": "VCIP/train/True",
    "VCIP_ablation": "VCIP_ablation/train/True",
    "RMSN": "RMSN/train/True",
    "RMSN_ab": "RMSN_ab/train/True",
    "CRN": "CRN/train/True",
    "CT": "CT/0.01/train/True",
    "ACTIN": "ACTIN/0.01/train/True",
}


def load_case_infos(gamma, model, seed, test=False):
    """Load pickle file with discrete ranking results."""
    model_path = MODEL_PATHS[model]
    test_str = str(test)
    pkl_path = RESULTS_DIR / f"coeff_{gamma}" / model_path / "case_infos" / str(seed) / test_str / f"case_infos_{model}.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_mse_csv(gamma, model, test=False):
    """Load CSV with continuous optimization (target distance) results."""
    model_path = MODEL_PATHS[model]
    test_str = str(test)
    csv_path = RESULTS_DIR / f"coeff_{gamma}" / model_path / test_str / "mse.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def compute_gpr_rcs(gamma, model, seeds=SEEDS, test=False):
    """Compute GPR and RCS from case_infos for each tau and seed."""
    results = {}
    for seed in seeds:
        data = load_case_infos(gamma, model, seed, test)
        if data is None:
            continue
        model_key = list(data.keys())[0]
        for tau in data[model_key]:
            if tau not in results:
                results[tau] = {"gpr": [], "rcs": []}
            individuals = data[model_key][tau]
            k = len(individuals[0]["model_losses"])  # number of candidates (100)

            # GPR = 1 - avg_rank / k
            ranks = [ind["true_sequence_rank"] for ind in individuals]
            avg_rank = np.mean(ranks)
            gpr = 1.0 - avg_rank / k

            # RCS = average Spearman correlation
            correlations = [ind["correlations"]["model_true"] for ind in individuals
                          if not np.isnan(ind["correlations"]["model_true"])]
            rcs = np.mean(correlations) if correlations else 0.0

            results[tau]["gpr"].append(gpr)
            results[tau]["rcs"].append(rcs)

    return results


def compute_target_distance(gamma, model, seeds=SEEDS, test=False):
    """Compute target distance (mean ± std) from mse.csv."""
    df = load_mse_csv(gamma, model, test)
    if df is None:
        return None
    # Filter to requested seeds
    df = df[df["seed"].isin(seeds)]
    tau_cols = [c for c in df.columns if c.startswith("tau=")]
    result = {}
    for col in tau_cols:
        tau = int(col.split("=")[1])
        values = df[col].values
        result[tau] = {"mean": np.mean(values), "std": np.std(values), "values": values}
    return result


# =============================================================================
# Paper reference values
# =============================================================================

# Table 1: Target distance, gamma=4, identical strategies (test=False)
PAPER_TABLE1 = {
    "ACTIN": {1: 0.42, 2: 0.71, 4: 1.05, 6: 1.30, 8: 1.47, 9: 1.55, 10: 1.59, 11: 1.63, 12: 1.68},
    "CT":    {1: 0.55, 2: 0.88, 4: 1.43, 6: 1.69, 8: 1.87, 9: 2.01, 10: 2.04, 11: 2.10, 12: 2.14},
    "CRN":   {1: 0.38, 2: 0.60, 4: 0.92, 6: 1.19, 8: 1.33, 9: 1.40, 10: 1.49, 11: 1.59, 12: 1.62},
    "RMSN":  {1: 0.30, 2: 0.45, 4: 0.75, 6: 0.98, 8: 1.15, 9: 1.22, 10: 1.28, 11: 1.43, 12: 1.47},
    "VCIP":  {1: 0.29, 2: 0.42, 4: 0.60, 6: 0.75, 8: 0.92, 9: 0.95, 10: 0.99, 11: 1.04, 12: 1.09},
}

# Table 3: Ablation study (GPR, RCS, target distance)
PAPER_TABLE3_GPR = {
    "RMSN":               {2: 0.863, 4: 0.796},
    "RMSN w/o adjustment": {2: 0.797, 4: 0.747},
    "VCIP":               {2: 0.944, 4: 0.972},
    "VCIP w/o adjustment": {2: 0.791, 4: 0.796},
}
PAPER_TABLE3_RCS = {
    "RMSN":               {2: 0.400, 4: 0.251},
    "RMSN w/o adjustment": {2: 0.461, 4: 0.213},
    "VCIP":               {2: 0.772, 4: 0.869},
    "VCIP w/o adjustment": {2: 0.566, 4: 0.595},
}
PAPER_TABLE3_TD = {
    "RMSN":               {1: 0.078, 2: 0.301, 3: 0.599, 4: 0.985},
    "RMSN w/o adjustment": {1: 0.087, 2: 0.280, 3: 0.694, 4: 1.132},
    "VCIP":               {1: 0.101, 2: 0.192, 3: 0.382, 4: 0.746},
    "VCIP w/o adjustment": {1: 0.092, 2: 0.284, 3: 0.756, 4: 0.912},
}


def print_separator(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def analyze_table1():
    """Compare Table 1: Target distance, gamma=4, tau=1..12."""
    print_separator("TABLE 1: Long-range prediction, gamma=4, identical strategies")
    print(f"{'Model':<8}", end="")
    taus = [1, 2, 4, 6, 8, 9, 10, 11, 12]
    for tau in taus:
        print(f"  {'tau='+str(tau):>12}", end="")
    print()
    print("-" * (8 + 14 * len(taus)))

    for model in ["ACTIN", "CT", "CRN", "RMSN", "VCIP"]:
        result = compute_target_distance(4, model, test=False)
        if result is None:
            print(f"{model:<8}  [NO DATA]")
            continue

        print(f"{model:<8}", end="")
        for tau in taus:
            if tau in result:
                m, s = result[tau]["mean"], result[tau]["std"]
                paper_val = PAPER_TABLE1.get(model, {}).get(tau, None)
                diff = ""
                if paper_val is not None:
                    pct = (m - paper_val) / paper_val * 100
                    diff = f"({pct:+.0f}%)"
                print(f"  {m:.2f}±{s:.2f}{diff:>5}", end="")
            else:
                print(f"  {'N/A':>12}", end="")
        print()

    # Print paper values for reference
    print("\nPaper reference (mean only):")
    for model in ["ACTIN", "CT", "CRN", "RMSN", "VCIP"]:
        print(f"{model:<8}", end="")
        for tau in taus:
            val = PAPER_TABLE1.get(model, {}).get(tau, None)
            if val is not None:
                print(f"  {val:>12.2f}", end="")
            else:
                print(f"  {'N/A':>12}", end="")
        print()


def analyze_table3():
    """Compare Table 3: Ablation study."""
    print_separator("TABLE 3: Ablation study, gamma=4")

    # GPR section
    print("\n--- GPR (higher is better) ---")
    print(f"{'Model':<25} {'tau=2 (ours)':>14} {'tau=2 (paper)':>14} {'tau=4 (ours)':>14} {'tau=4 (paper)':>14}")
    model_map = {
        "VCIP": ("VCIP", "VCIP"),
        "VCIP w/o adj": ("VCIP_ablation", "VCIP w/o adjustment"),
        "RMSN": ("RMSN", "RMSN"),
        "RMSN w/o adj": ("RMSN_ab", "RMSN w/o adjustment"),
    }

    for label, (code_name, paper_name) in model_map.items():
        result = compute_gpr_rcs(4, code_name, test=False)
        if not result:
            print(f"{label:<25}  [NO DATA]")
            continue
        gpr_2 = f"{np.mean(result[2]['gpr']):.3f}±{np.std(result[2]['gpr']):.3f}" if 2 in result else "N/A"
        gpr_4 = f"{np.mean(result[4]['gpr']):.3f}±{np.std(result[4]['gpr']):.3f}" if 4 in result else "N/A"
        p2 = f"{PAPER_TABLE3_GPR.get(paper_name, {}).get(2, 'N/A')}"
        p4 = f"{PAPER_TABLE3_GPR.get(paper_name, {}).get(4, 'N/A')}"
        print(f"{label:<25} {gpr_2:>14} {p2:>14} {gpr_4:>14} {p4:>14}")

    # RCS section
    print("\n--- RCS (higher is better) ---")
    print(f"{'Model':<25} {'tau=2 (ours)':>14} {'tau=2 (paper)':>14} {'tau=4 (ours)':>14} {'tau=4 (paper)':>14}")
    for label, (code_name, paper_name) in model_map.items():
        result = compute_gpr_rcs(4, code_name, test=False)
        if not result:
            print(f"{label:<25}  [NO DATA]")
            continue
        rcs_2 = f"{np.mean(result[2]['rcs']):.3f}±{np.std(result[2]['rcs']):.3f}" if 2 in result else "N/A"
        rcs_4 = f"{np.mean(result[4]['rcs']):.3f}±{np.std(result[4]['rcs']):.3f}" if 4 in result else "N/A"
        p2 = f"{PAPER_TABLE3_RCS.get(paper_name, {}).get(2, 'N/A')}"
        p4 = f"{PAPER_TABLE3_RCS.get(paper_name, {}).get(4, 'N/A')}"
        print(f"{label:<25} {rcs_2:>14} {p2:>14} {rcs_4:>14} {p4:>14}")

    # Target distance section
    print("\n--- Target Distance (lower is better) ---")
    print(f"{'Model':<25} {'g=1 (ours)':>12} {'g=1 (paper)':>12} {'g=2 (ours)':>12} {'g=2 (paper)':>12} {'g=3 (ours)':>12} {'g=3 (paper)':>12} {'g=4 (ours)':>12} {'g=4 (paper)':>12}")

    td_model_map = {
        "VCIP": ("VCIP", "VCIP"),
        "VCIP w/o adj": ("VCIP_ablation", "VCIP w/o adjustment"),
        "RMSN": ("RMSN", "RMSN"),
        "RMSN w/o adj": ("RMSN_ab", "RMSN w/o adjustment"),
    }

    for label, (code_name, paper_name) in td_model_map.items():
        print(f"{label:<25}", end="")
        for gamma in [1, 2, 3, 4]:
            result = compute_target_distance(gamma, code_name, test=False)
            if result is None:
                print(f" {'N/A':>12}", end="")
            else:
                # Table 3 uses tau=6 for target distance across gammas
                # Actually, looking at paper: "sequence optimization tasks (with tau=6 evaluated across different gamma values)"
                tau = 6
                if tau in result:
                    m = result[tau]["mean"]
                    s = result[tau]["std"]
                    print(f" {m:.3f}±{s:.3f}", end="")
                else:
                    # Try tau=1 as Table 3 gamma columns seem to be single tau values
                    # Let me check... Table 3 says gamma=1..4 columns, tau=6
                    print(f" {'N/A':>12}", end="")
            paper_val = PAPER_TABLE3_TD.get(paper_name, {}).get(gamma, None)
            if paper_val is not None:
                print(f" {paper_val:>12.3f}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()


def analyze_figure4():
    """Compare Figure 4: GRP/RCS across models, gamma=4, tau=2,4,6,8."""
    print_separator("FIGURE 4: GRP and RCS, gamma=4")

    for metric in ["gpr", "rcs"]:
        print(f"\n--- {metric.upper()} ---")
        print(f"{'Model':<8}", end="")
        for tau in [2, 4, 6, 8]:
            print(f"  {'tau='+str(tau):>14}", end="")
        print()

        for model in ["VCIP", "ACTIN", "CT", "CRN", "RMSN"]:
            result = compute_gpr_rcs(4, model, test=False)
            if not result:
                print(f"{model:<8}  [NO DATA]")
                continue
            print(f"{model:<8}", end="")
            for tau in [2, 4, 6, 8]:
                if tau in result:
                    values = result[tau][metric]
                    m, s = np.mean(values), np.std(values)
                    print(f"  {m:.3f}±{s:.3f}", end="")
                else:
                    print(f"  {'N/A':>14}", end="")
            print()


def analyze_figure6():
    """Compare Figure 6: Target distance, gamma=1,2,3, tau=1..6."""
    print_separator("FIGURE 6: Target distance across gamma=1,2,3")

    for gamma in [1, 2, 3]:
        print(f"\n--- gamma={gamma} ---")
        print(f"{'Model':<8}", end="")
        for tau in range(1, 7):
            print(f"  {'tau='+str(tau):>12}", end="")
        print()

        for model in ["VCIP", "ACTIN", "CT", "CRN", "RMSN"]:
            result = compute_target_distance(gamma, model, test=False)
            if result is None:
                print(f"{model:<8}  [NO DATA]")
                continue
            print(f"{model:<8}", end="")
            for tau in range(1, 7):
                if tau in result:
                    m, s = result[tau]["mean"], result[tau]["std"]
                    print(f"  {m:.3f}±{s:.3f}", end="")
                else:
                    print(f"  {'N/A':>12}", end="")
            print()


def analyze_data_completeness():
    """Check what data we have."""
    print_separator("DATA COMPLETENESS CHECK")

    for gamma in [1, 2, 3, 4]:
        print(f"\ngamma={gamma}:")
        for model in list(MODEL_PATHS.keys()):
            # Check mse.csv
            df = load_mse_csv(gamma, model, test=False)
            csv_status = f"{len(df)} seeds" if df is not None else "MISSING"

            # Check case_infos
            case_count = 0
            for seed in SEEDS:
                if load_case_infos(gamma, model, seed, test=False) is not None:
                    case_count += 1
            case_status = f"{case_count} seeds" if case_count > 0 else "MISSING"

            print(f"  {model:<20} CSV: {csv_status:<12} case_infos: {case_status}")


if __name__ == "__main__":
    analyze_data_completeness()
    analyze_table1()
    analyze_table3()
    analyze_figure4()
    analyze_figure6()
