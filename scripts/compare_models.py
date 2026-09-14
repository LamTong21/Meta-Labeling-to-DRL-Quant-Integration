import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.data_preparer import MultiAssetDataPreparer
from src.evaluation.diagnostics import compute_policy_entropy
from src.evaluation.metrics import (
    calculate_annualized_return,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_turnover_ratio,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Synthesize comparative metrics between Modeling09 and Modeling10.")
    parser.add_argument("--m09-ret", type=str, default="data/processed/modeling09_returns.csv")
    parser.add_argument("--m10-ret", type=str, default="data/processed/modeling10_returns.csv")
    parser.add_argument("--m09-weights", type=str, default="data/processed/modeling09_weights.csv")
    parser.add_argument("--m10-weights", type=str, default="data/processed/modeling10_weights.csv")
    parser.add_argument("--portfolio", type=str, default="data/raw/final_portfolio.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load returns
    s_m09 = pd.read_csv(args.m09_ret, index_col=0, parse_dates=True).squeeze()
    s_m10 = pd.read_csv(args.m10_ret, index_col=0, parse_dates=True).squeeze()
    
    w_m09 = pd.read_csv(args.m09_weights, index_col=0, parse_dates=True)
    w_m10 = pd.read_csv(args.m10_weights, index_col=0, parse_dates=True)

    # Load benchmark VN-Index
    preparer = MultiAssetDataPreparer(portfolio_path=args.portfolio)
    data_dict = preparer.prepare_dataset()
    bench = data_dict[list(data_dict.keys())[0]]["mkt_return"]

    # Intersect overlapping time horizon
    common_idx = s_m09.index.intersection(s_m10.index).intersection(bench.index).sort_values()
    r09 = s_m09.loc[common_idx]
    r10 = s_m10.loc[common_idx]
    rb = bench.loc[common_idx]
    w09 = w_m09.loc[common_idx]
    w10 = w_m10.loc[common_idx]

    # Synthesize Comparative Metrics Matrix
    models = {
        "Modeling09 (Meta-Labeling)": (r09, w09),
        "Modeling10 (PPO DRL)": (r10, w10),
        "Benchmark (VN-Index)": (rb, None),
    }

    summary = {}
    for name, (ret, w) in models.items():
        summary[name] = {
            "Annualized Return": f"{calculate_annualized_return(ret) * 100:.2f}%",
            "Cumulative Return": f"{(np.prod(1.0 + ret) - 1.0) * 100:.2f}%",
            "Annualized Vol": f"{ret.std() * np.sqrt(252) * 100:.2f}%",
            "Sharpe Ratio": f"{calculate_sharpe_ratio(ret, risk_free_rate=0.045):.3f}",
            "Sortino Ratio": f"{calculate_sortino_ratio(ret, risk_free_rate=0.045):.3f}",
            "Max Drawdown": f"{calculate_max_drawdown(ret)['max_drawdown'] * 100:.2f}%",
            "Calmar Ratio": f"{calculate_calmar_ratio(ret):.3f}",
            "Daily Turnover": f"{calculate_turnover_ratio(w) * 100:.2f}%" if w is not None else "N/A",
        }

    comp_df = pd.DataFrame(summary)
    print("\n" + "=" * 90)
    print("PARADIGM COMPARISON: STATISTICAL META-LABELING VS DEEP REINFORCEMENT LEARNING")
    print("=" * 90)
    print(comp_df.to_string())

    # Pathological Cash-Trap Diagnostic (Entropy calculation)
    entropy_m10 = compute_policy_entropy(w10.to_numpy())
    mean_entropy = float(np.mean(entropy_m10))
    cash_trap_ratio = float(np.mean(w10.iloc[:, 0] > 0.95) * 100)

    print("\n" + "-" * 90)
    print("SYSTEMIC DIAGNOSTICS & OPEN PATHOLOGIES (MODELING10):")
    print(f"- Mean Policy Shannon Entropy       : {mean_entropy:.4f}")
    print(f"- Cash Absorbing Saturation (w0 > 95%): {cash_trap_ratio:.2f}% of sessions")
    print("-" * 90)

    comp_df.to_csv("data/processed/paradigm_comparison.csv")
    print("\n[✓] Comparison matrix saved to data/processed/paradigm_comparison.csv")


if __name__ == "__main__":
    main()