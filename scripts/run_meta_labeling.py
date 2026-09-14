import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.data_preparer import MultiAssetDataPreparer
from src.evaluation.backtester import PortfolioBacktester
from src.evaluation.metrics import generate_performance_tearsheet
from src.features.econometric_filters import EconometricFilterEngine
from src.features.information_theory import filter_by_mutual_information
from src.labeling.primary_signal import generate_primary_signal
from src.labeling.triple_barrier import apply_triple_barrier_labeling
from src.models.meta_labeling_model import MetaLabelingEnsemble


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Modeling09 Meta-Labeling pipeline.")
    parser.add_argument("--portfolio", type=str, default="data/raw/final_portfolio.csv", help="Portfolio constituents file")
    parser.add_argument("--estimator", type=str, default="lightgbm", choices=["lightgbm", "xgboost", "random_forest"])
    parser.add_argument("--horizon", type=int, default=5, help="Triple-barrier vertical limit (days)")
    parser.add_argument("--pt-sl", type=float, default=1.5, help="Profit-take and stop-loss vol multiplier")
    parser.add_argument("--cost", type=float, default=0.0015, help="One-way friction cost")
    return parser.parse_args()


def construct_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds stationary econometric and signal processing features."""
    feats = pd.DataFrame(index=df.index)
    feats["log_ret"] = df["log_return"]
    feats["intraday_ret"] = df["intraday_return"]
    feats["overnight_ret"] = df["overnight_return"]
    feats["parkinson_vol"] = df["parkinson_vol"]
    feats["mkt_ret"] = df["mkt_return"]
    
    vol = df["volume"]
    sma_vol = vol.rolling(20, min_periods=1).mean().replace(0, 1.0)
    feats["vol_ratio"] = vol / sma_vol

    # Momentum oscillators
    for window in [5, 10, 20]:
        feats[f"mom_{window}"] = df["close_adj"].pct_change(window)

    return feats.dropna()


def main():
    args = parse_args()
    preparer = MultiAssetDataPreparer(portfolio_path=args.portfolio)
    data_dict = preparer.prepare_dataset()

    engine = EconometricFilterEngine(significance_level=0.05)
    model_weights = {}

    print(f"[*] Training Meta-Labeling models across {len(data_dict)} assets via {args.estimator}...")
    for ticker, df in data_dict.items():
        if len(df) < 200:
            continue

        primary_signals = generate_primary_signal(df, fast_window=10, slow_window=30, method="ma_crossover")
        labeled_df = apply_triple_barrier_labeling(
            df=df,
            primary_signals=primary_signals,
            volatility_col="parkinson_vol",
            horizon=args.horizon,
            pt_sl_ratio=args.pt_sl,
        )

        features = construct_features(df)
        common_idx = labeled_df.index.intersection(features.index)
        if len(common_idx) < 100:
            continue

        X = features.loc[common_idx]
        y = labeled_df.loc[common_idx, "meta_label"].astype(int)
        sigs = labeled_df.loc[common_idx, "primary_signal"].to_numpy()

        # Econometric & Information-Theoretic selection
        X_stat, _ = engine.filter_features(X)
        if X_stat.empty:
            continue
        X_selected, _ = filter_by_mutual_information(X_stat, y, is_classification=True, threshold=0.005)
        if X_selected.empty:
            continue

        # Train and calibrate GBDT
        ensemble = MetaLabelingEnsemble(estimator_type=args.estimator, calibration_method="isotonic")
        ensemble.fit_with_cv_calibration(X_selected, y, n_splits=5)
        _, bet_sizes = ensemble.predict_conviction(X_selected, sigs)

        model_weights[ticker] = pd.Series(bet_sizes, index=common_idx)

    # Consolidate target weights matrix
    weights_df = pd.DataFrame(model_weights).fillna(0.0)
    
    # Backtest under HOSE frictions
    backtester = PortfolioBacktester(data_dict, transaction_cost=args.cost, apply_frictions=True)
    net_returns, executed_weights, fees = backtester.run_weights(weights_df)

    benchmark_rets = data_dict[list(data_dict.keys())[0]]["mkt_return"]
    tearsheet = generate_performance_tearsheet(
        portfolio_returns=net_returns,
        weights_df=executed_weights,
        benchmark_returns=benchmark_rets,
    )

    print("\n" + "=" * 80)
    print(f"MODELING09 (META-LABELING: {args.estimator.upper()}) PERFORMANCE TEARSHEET")
    print("=" * 80)
    print(tearsheet.to_string())

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    net_returns.to_csv(out_dir / "modeling09_returns.csv")
    executed_weights.to_csv(out_dir / "modeling09_weights.csv")
    print(f"\n[✓] Results persisted to {out_dir}/modeling09_*.csv")


if __name__ == "__main__":
    main()