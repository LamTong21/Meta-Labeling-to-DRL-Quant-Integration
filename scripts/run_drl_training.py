import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.data_preparer import MultiAssetDataPreparer
from src.envs.portfolio_env import MultiAssetPortfolioEnv
from src.evaluation.metrics import generate_performance_tearsheet
from src.models.ppo_agent import DRLPortfolioAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Modeling10 PPO Agent.")
    parser.add_argument("--portfolio", type=str, default="data/raw/final_portfolio.csv", help="Portfolio constituents file")
    parser.add_argument("--policy", type=str, default="MlpPolicy", choices=["MlpPolicy", "simplex_mlp", "dirichlet"])
    parser.add_argument("--timesteps", type=int, default=150_000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--lookback", type=int, default=20, help="Observation lookback window")
    parser.add_argument("--save-path", type=str, default="data/processed/models/ppo_model", help="Path to save weights")
    return parser.parse_args()


def main():
    args = parse_args()
    preparer = MultiAssetDataPreparer(portfolio_path=args.portfolio)
    data_dict = preparer.prepare_dataset()

    print(f"[*] Instantiating Vectorized Portfolio Environment (Lookback={args.lookback})...")
    env = MultiAssetPortfolioEnv(
        asset_data=data_dict,
        lookback_window=args.lookback,
        transaction_cost=0.0015,
        lambda_penalty=2.0,
        var_threshold=-0.02,
        apply_frictions=True,
    )

    print(f"[*] Training DRL Agent with policy={args.policy}, timesteps={args.timesteps}...")
    agent = DRLPortfolioAgent(
        env=env,
        policy_type=args.policy,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
    )
    agent.train(total_timesteps=args.timesteps)
    agent.save(args.save_path)
    print(f"[✓] Policy model persisted to {args.save_path}")

    # Deterministic trajectory evaluation
    obs, info = env.reset()
    done = False
    dates, returns, cash_weights, executed_weights = [], [], [], []

    while not done:
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        dates.append(pd.to_datetime(info["date"]))
        returns.append(info["portfolio_return"] - info["turnover_cost"])
        cash_weights.append(info["cash_weight"])
        executed_weights.append(env.current_weights)

    returns_s = pd.Series(returns, index=dates, name="drl_net_return")
    weights_cols = ["CASH"] + env.asset_tickers
    weights_df = pd.DataFrame(executed_weights, index=dates, columns=weights_cols)

    benchmark_rets = data_dict[list(data_dict.keys())[0]]["mkt_return"]
    tearsheet = generate_performance_tearsheet(
        portfolio_returns=returns_s,
        weights_df=weights_df,
        benchmark_returns=benchmark_rets,
    )

    print("\n" + "=" * 80)
    print(f"MODELING10 (DRL PPO: {args.policy}) PERFORMANCE TEARSHEET")
    print("=" * 80)
    print(tearsheet.to_string())

    out_dir = Path("data/processed")
    returns_s.to_csv(out_dir / "modeling10_returns.csv")
    weights_df.to_csv(out_dir / "modeling10_weights.csv")
    print(f"\n[✓] Trajectory records persisted to {out_dir}/modeling10_*.csv")


if __name__ == "__main__":
    main()