from typing import Dict, Union
import numpy as np
import pandas as pd


def calculate_annualized_return(
    returns: pd.Series, 
    trading_days: int = 252
) -> float:
    """Computes compounding annualized return: (1 + R_cum)^(252 / T) - 1."""
    clean_ret = returns.dropna()
    n_days = len(clean_ret)
    if n_days == 0:
        return 0.0

    cumulative_compounded = np.prod(1.0 + clean_ret)
    annualized = (cumulative_compounded ** (trading_days / n_days)) - 1.0
    return float(annualized)


def calculate_max_drawdown(returns: pd.Series) -> Dict[str, float]:
    """Computes historical maximum drawdown and peak-to-trough series."""
    clean_ret = returns.dropna()
    if len(clean_ret) == 0:
        return {"max_drawdown": 0.0, "current_drawdown": 0.0}

    cum_wealth = np.cumprod(1.0 + clean_ret)
    running_max = np.maximum.accumulate(cum_wealth)
    drawdowns = (cum_wealth - running_max) / running_max

    return {
        "max_drawdown": float(np.min(drawdowns)),
        "current_drawdown": float(drawdowns.iloc[-1]),
    }


def calculate_sharpe_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.0, 
    trading_days: int = 252
) -> float:
    """Annualized Sharpe ratio under zero-mean or explicit risk-free rate assumption."""
    clean_ret = returns.dropna()
    excess_ret = clean_ret - (risk_free_rate / trading_days)
    sigma = clean_ret.std()

    if sigma == 0 or np.isnan(sigma):
        return 0.0

    return float((excess_ret.mean() / sigma) * np.sqrt(trading_days))


def calculate_sortino_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.0, 
    trading_days: int = 252
) -> float:
    """Annualized Sortino ratio penalizing solely downside semi-deviation below zero."""
    clean_ret = returns.dropna()
    excess_ret = clean_ret - (risk_free_rate / trading_days)
    downside_returns = clean_ret[clean_ret < 0.0]

    if len(downside_returns) == 0:
        return 0.0

    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    return float((excess_ret.mean() / downside_std) * np.sqrt(trading_days))


def calculate_calmar_ratio(
    returns: pd.Series, 
    trading_days: int = 252
) -> float:
    """Annualized Return over Maximum Absolute Drawdown."""
    ann_return = calculate_annualized_return(returns, trading_days=trading_days)
    max_dd = abs(calculate_max_drawdown(returns)["max_drawdown"])

    if max_dd == 0:
        return 0.0

    return float(ann_return / max_dd)


def calculate_turnover_ratio(weights_df: pd.DataFrame) -> float:
    """Measures mean daily one-way portfolio churn: (1 / 2T) * sum_{t=1}^T ||w_t - w_{t-1}'||_1."""
    deltas = weights_df.diff().abs().dropna()
    daily_two_way_turnover = deltas.sum(axis=1)
    mean_one_way_turnover = float(daily_two_way_turnover.mean() / 2.0)
    return mean_one_way_turnover


def generate_performance_tearsheet(
    portfolio_returns: pd.Series,
    weights_df: Optional[pd.DataFrame] = None,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.045,  # 4.5% statutory benchmark proxy
    trading_days: int = 252,
) -> pd.DataFrame:
    """Generates an institutional summary table comparing performance vectors."""
    metrics = {
        "Annualized Return": f"{calculate_annualized_return(portfolio_returns, trading_days) * 100:.2f}%",
        "Cumulative Return": f"{(np.prod(1.0 + portfolio_returns.dropna()) - 1.0) * 100:.2f}%",
        "Annualized Volatility": f"{portfolio_returns.dropna().std() * np.sqrt(trading_days) * 100:.2f}%",
        "Sharpe Ratio": f"{calculate_sharpe_ratio(portfolio_returns, risk_free_rate, trading_days):.3f}",
        "Sortino Ratio": f"{calculate_sortino_ratio(portfolio_returns, risk_free_rate, trading_days):.3f}",
        "Max Drawdown": f"{calculate_max_drawdown(portfolio_returns)['max_drawdown'] * 100:.2f}%",
        "Calmar Ratio": f"{calculate_calmar_ratio(portfolio_returns, trading_days):.3f}",
    }

    if weights_df is not None:
        metrics["Daily Mean Turnover"] = f"{calculate_turnover_ratio(weights_df) * 100:.2f}%"

    if benchmark_returns is not None:
        aligned_bench = benchmark_returns.loc[portfolio_returns.index].dropna()
        metrics["Benchmark Ann. Return"] = f"{calculate_annualized_return(aligned_bench, trading_days) * 100:.2f}%"
        metrics["Benchmark Sharpe"] = f"{calculate_sharpe_ratio(aligned_bench, risk_free_rate, trading_days):.3f}"
        metrics["Benchmark MaxDD"] = f"{calculate_max_drawdown(aligned_bench)['max_drawdown'] * 100:.2f}%"

    df_sheet = pd.DataFrame.from_dict(metrics, orient="index", columns=["Metric Value"])
    return df_sheet