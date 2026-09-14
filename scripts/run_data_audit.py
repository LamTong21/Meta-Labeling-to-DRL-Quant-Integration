import argparse
from pathlib import Path
from src.data.data_preparer import MultiAssetDataPreparer
from src.evaluation.diagnostics import evaluate_missingness_ledger, raw_data_profile


def parse_args():
    parser = argparse.ArgumentParser(description="Audit OHLCV physical candlestick integrity and missingness ledger.")
    parser.add_argument("--portfolio", type=str, default="data/raw/final_portfolio.csv", help="Path to portfolio constituents CSV")
    parser.add_argument("--start", type=str, default="2018-07-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-06-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--source", type=str, default="KBS", help="Data source for VNINDEX quote API")
    parser.add_argument("--cache-dir", type=str, default="data/raw/data_cache", help="Disk cache directory")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[*] Initializing MultiAssetDataPreparer from {args.portfolio}...")
    preparer = MultiAssetDataPreparer(
        portfolio_path=args.portfolio,
        start_date=args.start,
        end_date=args.end,
        source=args.source,
        cache_dir=args.cache_dir,
    )
    
    clean_dict = preparer.prepare_dataset()
    
    # Audit diagnostic profile on representative constituent tickers
    sample_tickers = [t for t in ["ABT", "BMP", "FPT", "VNM"] if t in clean_dict]
    for ticker in sample_tickers:
        raw_data_profile(clean_dict[ticker], symbol=ticker)

    # Generate universe-wide Missing Value Ledger
    print("\n" + "=" * 80)
    print("UNIVERSE MISSINGNESS & INTEGRITY LEDGER")
    print("=" * 80)
    ledger_df = evaluate_missingness_ledger(clean_dict)
    print(ledger_df.to_string(index=False))

    output_path = Path("data/processed/missingness_ledger.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_df.to_csv(output_path, index=False)
    print(f"\n[✓] Ledger persisted successfully to {output_path}")


if __name__ == "__main__":
    main()