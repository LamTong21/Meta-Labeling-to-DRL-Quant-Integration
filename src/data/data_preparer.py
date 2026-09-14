import os
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf
from vnstock import Quote[cite: 1, 2]

from src.data.geometry_auditor import audit_candlestick_geometry, normalize_corporate_actions
from src.data.return_decomposer import decompose_returns_and_volatility


class MultiAssetDataPreparer:
    """Orchestrates asset ingestion, candlestick geometry auditing, corporate action normalization,
    and index alignment across multi-asset universes[cite: 1, 2].
    """
    def __init__(
        self,
        portfolio_path: Union[str, Path] = "final_portfolio.csv",
        start_date: str = "2018-01-01",
        end_date: str = "2026-06-01",
        source: str = "VCI",
        cache_dir: Union[str, Path] = "data/raw/data_cache",
    ):
        self.portfolio_path = Path(portfolio_path)
        self.start_date = start_date
        self.end_date = end_date
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.df_portfolio = pd.read_csv(self.portfolio_path)[cite: 1, 2]
        self.tickers = sorted(self.df_portfolio["Ticker"].unique().tolist())[cite: 1, 2]

    def _fetch_history(self, symbol: str) -> pd.DataFrame:
        """Fetches OHLCV bars from local disk cache, vnstock Quote API, or yfinance[cite: 1, 2]."""
        cache_file = self.cache_dir / f"{symbol}.csv"

        if cache_file.exists():
            df = pd.read_csv(cache_file)[cite: 1, 2]
            df["time"] = pd.to_datetime(df["time"])[cite: 1, 2]
            df.set_index("time", inplace=True)[cite: 1, 2]
            return df[cite: 1, 2]

        try:
            if symbol == "VNINDEX":
                q = Quote(symbol="VNINDEX", source=self.source)[cite: 1, 2]
                df = q.history(start=self.start_date, end=self.end_date, interval="1D")[cite: 1, 2]

                if df is not None and not df.empty:
                    df.reset_index(inplace=True)[cite: 1, 2]
                    df.rename(columns={col: str(col).lower() for col in df.columns}, inplace=True)[cite: 1, 2]
                    time_col = "time" if "time" in df.columns else "date"[cite: 1, 2]
                    df.rename(columns={time_col: "time"}, inplace=True)[cite: 1, 2]

                    required_cols = ["time", "open", "high", "low", "close", "volume"][cite: 1, 2]
                    df = df[[col for col in required_cols if col in df.columns]][cite: 1, 2]
                    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)[cite: 1, 2]
                    df.sort_values(by="time", inplace=True)[cite: 1, 2]
                    df.set_index("time", inplace=True)[cite: 1, 2]
                    df.to_csv(cache_file)[cite: 1, 2]
                    return df[cite: 1, 2]
            else:
                yf_symbol = f"{symbol}.VN"[cite: 1, 2]
                ticker_obj = yf.Ticker(yf_symbol)[cite: 1, 2]
                df = ticker_obj.history(start=self.start_date, end=self.end_date, interval="1d")[cite: 1, 2]

                if df is not None and not df.empty:
                    df.reset_index(inplace=True)[cite: 1, 2]
                    df.rename(
                        columns={
                            "Date": "time",
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Volume": "volume",
                        },
                        inplace=True,
                    )[cite: 1, 2]
                    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)[cite: 1, 2]
                    df.sort_values(by="time", inplace=True)[cite: 1, 2]
                    df.set_index("time", inplace=True)[cite: 1, 2]
                    df.to_csv(cache_file)[cite: 1, 2]
                    return df[cite: 1, 2]

        except Exception as err:
            print(f"[!] Lỗi khi truy xuất {symbol}: {err}")[cite: 1, 2]

        return pd.DataFrame()[cite: 1, 2]

    def _clean_and_adjust(self, df: pd.DataFrame, df_mkt: pd.DataFrame) -> pd.DataFrame:
        """Executes full geometric auditing, adjustment, and exogenous return alignment[cite: 1, 2]."""
        df = df[~df.index.duplicated(keep="last")].sort_index()[cite: 1, 2]
        
        # 1. Candlestick physical boundary verification
        df = audit_candlestick_geometry(df)
        
        # 2. Corporate action normalization
        df = normalize_corporate_actions(df)
        
        # 3. Return and Parkinson volatility decomposition
        df = decompose_returns_and_volatility(df, max_calendar_gap_days=7)
        
        # 4. Align exogenous VN-Index dynamics
        if not df_mkt.empty:
            df = df.join(df_mkt, how="left")[cite: 1, 2]
            mkt_price_cols = ["mkt_close", "mkt_volume", "mkt_high", "mkt_low"][cite: 1, 2]
            df[mkt_price_cols] = df[mkt_price_cols].ffill()[cite: 1, 2]
            if "mkt_return" in df.columns:
                df["mkt_return"] = df["mkt_return"].fillna(0.0)[cite: 1, 2]

        return df.dropna(subset=["close_adj", "open_adj"])[cite: 1, 2]

    def prepare_dataset(self) -> Dict[str, pd.DataFrame]:
        """Runs the complete ingestion and validation pipeline across all portfolio constituents[cite: 1, 2]."""
        print("[*] Đang nạp dữ liệu Benchmark VN-INDEX...")[cite: 1, 2]
        df_mkt = self._fetch_history("VNINDEX")[cite: 1, 2]

        if not df_mkt.empty:
            df_mkt.index = pd.to_datetime(df_mkt.index).normalize()[cite: 1, 2]
            df_mkt = df_mkt[~df_mkt.index.duplicated(keep="last")].sort_index()[cite: 1, 2]
            df_mkt = df_mkt[["close", "volume", "high", "low"]].rename(
                columns={
                    "close": "mkt_close",
                    "volume": "mkt_volume",
                    "high": "mkt_high",
                    "low": "mkt_low",
                }
            )[cite: 1, 2]
            df_mkt["mkt_return"] = np.log(df_mkt["mkt_close"] / df_mkt["mkt_close"].shift(1))[cite: 1, 2]
        else:
            print("[!] Cảnh báo: Không thể tải dữ liệu VNINDEX benchmark.")[cite: 1, 2]

        clean_dict: Dict[str, pd.DataFrame] = {}
        for idx, ticker in enumerate(self.tickers, 1):[cite: 1, 2]
            print(f"[{idx:02d}/{len(self.tickers):02d}] Chuẩn bị dữ liệu: {ticker}...")[cite: 1, 2]
            df_raw = self._fetch_history(ticker)[cite: 1, 2]
            if not df_raw.empty and len(df_raw) > 50:[cite: 1, 2]
                clean_df = self._clean_and_adjust(df_raw, df_mkt)[cite: 1, 2]
                clean_dict[ticker] = clean_df[cite: 1, 2]
            else:
                print(f"  [!] Bỏ qua {ticker} (dữ liệu không đủ).")[cite: 1, 2]
            time.sleep(0.5)

        print(f"\n[✓] Hoàn tất Data Engine: {len(clean_dict)}/{len(self.tickers)} mã sẵn sàng.")[cite: 1, 2]
        return clean_dict[cite: 1, 2]import os
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf
from vnstock import Quote[cite: 1, 2]

from src.data.geometry_auditor import audit_candlestick_geometry, normalize_corporate_actions
from src.data.return_decomposer import decompose_returns_and_volatility


class MultiAssetDataPreparer:
    """Orchestrates asset ingestion, candlestick geometry auditing, corporate action normalization,
    and index alignment across multi-asset universes[cite: 1, 2].
    """
    def __init__(
        self,
        portfolio_path: Union[str, Path] = "final_portfolio.csv",
        start_date: str = "2018-01-01",
        end_date: str = "2026-06-01",
        source: str = "VCI",
        cache_dir: Union[str, Path] = "data/raw/data_cache",
    ):
        self.portfolio_path = Path(portfolio_path)
        self.start_date = start_date
        self.end_date = end_date
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.df_portfolio = pd.read_csv(self.portfolio_path)[cite: 1, 2]
        self.tickers = sorted(self.df_portfolio["Ticker"].unique().tolist())[cite: 1, 2]

    def _fetch_history(self, symbol: str) -> pd.DataFrame:
        """Fetches OHLCV bars from local disk cache, vnstock Quote API, or yfinance[cite: 1, 2]."""
        cache_file = self.cache_dir / f"{symbol}.csv"

        if cache_file.exists():
            df = pd.read_csv(cache_file)[cite: 1, 2]
            df["time"] = pd.to_datetime(df["time"])[cite: 1, 2]
            df.set_index("time", inplace=True)[cite: 1, 2]
            return df[cite: 1, 2]

        try:
            if symbol == "VNINDEX":
                q = Quote(symbol="VNINDEX", source=self.source)[cite: 1, 2]
                df = q.history(start=self.start_date, end=self.end_date, interval="1D")[cite: 1, 2]

                if df is not None and not df.empty:
                    df.reset_index(inplace=True)[cite: 1, 2]
                    df.rename(columns={col: str(col).lower() for col in df.columns}, inplace=True)[cite: 1, 2]
                    time_col = "time" if "time" in df.columns else "date"[cite: 1, 2]
                    df.rename(columns={time_col: "time"}, inplace=True)[cite: 1, 2]

                    required_cols = ["time", "open", "high", "low", "close", "volume"][cite: 1, 2]
                    df = df[[col for col in required_cols if col in df.columns]][cite: 1, 2]
                    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)[cite: 1, 2]
                    df.sort_values(by="time", inplace=True)[cite: 1, 2]
                    df.set_index("time", inplace=True)[cite: 1, 2]
                    df.to_csv(cache_file)[cite: 1, 2]
                    return df[cite: 1, 2]
            else:
                yf_symbol = f"{symbol}.VN"[cite: 1, 2]
                ticker_obj = yf.Ticker(yf_symbol)[cite: 1, 2]
                df = ticker_obj.history(start=self.start_date, end=self.end_date, interval="1d")[cite: 1, 2]

                if df is not None and not df.empty:
                    df.reset_index(inplace=True)[cite: 1, 2]
                    df.rename(
                        columns={
                            "Date": "time",
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Volume": "volume",
                        },
                        inplace=True,
                    )[cite: 1, 2]
                    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)[cite: 1, 2]
                    df.sort_values(by="time", inplace=True)[cite: 1, 2]
                    df.set_index("time", inplace=True)[cite: 1, 2]
                    df.to_csv(cache_file)[cite: 1, 2]
                    return df[cite: 1, 2]

        except Exception as err:
            print(f"[!] Lỗi khi truy xuất {symbol}: {err}")[cite: 1, 2]

        return pd.DataFrame()[cite: 1, 2]

    def _clean_and_adjust(self, df: pd.DataFrame, df_mkt: pd.DataFrame) -> pd.DataFrame:
        """Executes full geometric auditing, adjustment, and exogenous return alignment[cite: 1, 2]."""
        df = df[~df.index.duplicated(keep="last")].sort_index()[cite: 1, 2]
        
        # 1. Candlestick physical boundary verification
        df = audit_candlestick_geometry(df)
        
        # 2. Corporate action normalization
        df = normalize_corporate_actions(df)
        
        # 3. Return and Parkinson volatility decomposition
        df = decompose_returns_and_volatility(df, max_calendar_gap_days=7)
        
        # 4. Align exogenous VN-Index dynamics
        if not df_mkt.empty:
            df = df.join(df_mkt, how="left")[cite: 1, 2]
            mkt_price_cols = ["mkt_close", "mkt_volume", "mkt_high", "mkt_low"][cite: 1, 2]
            df[mkt_price_cols] = df[mkt_price_cols].ffill()[cite: 1, 2]
            if "mkt_return" in df.columns:
                df["mkt_return"] = df["mkt_return"].fillna(0.0)[cite: 1, 2]

        return df.dropna(subset=["close_adj", "open_adj"])[cite: 1, 2]

    def prepare_dataset(self) -> Dict[str, pd.DataFrame]:
        """Runs the complete ingestion and validation pipeline across all portfolio constituents[cite: 1, 2]."""
        print("[*] Đang nạp dữ liệu Benchmark VN-INDEX...")[cite: 1, 2]
        df_mkt = self._fetch_history("VNINDEX")[cite: 1, 2]

        if not df_mkt.empty:
            df_mkt.index = pd.to_datetime(df_mkt.index).normalize()[cite: 1, 2]
            df_mkt = df_mkt[~df_mkt.index.duplicated(keep="last")].sort_index()[cite: 1, 2]
            df_mkt = df_mkt[["close", "volume", "high", "low"]].rename(
                columns={
                    "close": "mkt_close",
                    "volume": "mkt_volume",
                    "high": "mkt_high",
                    "low": "mkt_low",
                }
            )[cite: 1, 2]
            df_mkt["mkt_return"] = np.log(df_mkt["mkt_close"] / df_mkt["mkt_close"].shift(1))[cite: 1, 2]
        else:
            print("[!] Cảnh báo: Không thể tải dữ liệu VNINDEX benchmark.")[cite: 1, 2]

        clean_dict: Dict[str, pd.DataFrame] = {}
        for idx, ticker in enumerate(self.tickers, 1):[cite: 1, 2]
            print(f"[{idx:02d}/{len(self.tickers):02d}] Chuẩn bị dữ liệu: {ticker}...")[cite: 1, 2]
            df_raw = self._fetch_history(ticker)[cite: 1, 2]
            if not df_raw.empty and len(df_raw) > 50:[cite: 1, 2]
                clean_df = self._clean_and_adjust(df_raw, df_mkt)[cite: 1, 2]
                clean_dict[ticker] = clean_df[cite: 1, 2]
            else:
                print(f"  [!] Bỏ qua {ticker} (dữ liệu không đủ).")[cite: 1, 2]
            time.sleep(0.5)

        print(f"\n[✓] Hoàn tất Data Engine: {len(clean_dict)}/{len(self.tickers)} mã sẵn sàng.")[cite: 1, 2]
        return clean_dict[cite: 1, 2]