#!/usr/bin/env python3
"""
Calculate and plot Jensen-Shannon divergence between adjacent years.

This measures how much the market share distribution changes from year to year.
High divergence = major shifts in market composition
Low divergence = stable market composition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import warnings
import json
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

warnings.filterwarnings('ignore')
import logging
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Use same configuration as main script
START_YEAR = 1962
END_YEAR = None
TOP_N_COMPANIES = 10
MIN_COMPANIES_PER_YEAR = 8

def get_ticker_universe(ticker_file='tickers.json'):
    ticker_path = Path(ticker_file)
    if not ticker_path.exists():
        raise FileNotFoundError(f"Ticker file not found: {ticker_file}")

    with open(ticker_path, 'r') as f:
        ticker_data = json.load(f)

    tickers = []
    for sector, subcategories in ticker_data.items():
        for subcategory, ticker_list in subcategories.items():
            tickers.extend(ticker_list)

    return sorted(list(set(tickers)))

def fetch_market_cap_data(tickers, start_year, end_year):
    if end_year is None:
        end_year = datetime.now().year

    all_data = []
    failed_tickers = {}

    print(f"Fetching data for {len(tickers)} tickers from {start_year} to {end_year}...")
    print("This may take a few minutes.\n")

    for i, ticker in enumerate(tickers, 1):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(tickers)} tickers processed...")

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(
                start=f"{start_year}-01-01",
                end=f"{end_year+1}-01-01",
                interval="1mo"
            )

            if hist.empty:
                failed_tickers[ticker] = "No historical price data available"
                continue

            info = stock.info
            shares_outstanding = info.get('sharesOutstanding', None)

            if shares_outstanding is None:
                failed_tickers[ticker] = "Shares outstanding not available"
                continue

            for date, row in hist.iterrows():
                close_price = row['Close']

                if pd.notna(close_price) and close_price > 0:
                    market_cap = close_price * shares_outstanding

                    all_data.append({
                        'date': date,
                        'ticker': ticker,
                        'price': close_price,
                        'market_cap': market_cap
                    })

        except Exception as e:
            failed_tickers[ticker] = f"Error: {str(e)}"
            continue

    print(f"\nData fetching complete. {len(all_data)} data points collected.")

    df = pd.DataFrame(all_data)
    return df, failed_tickers

def calculate_annual_top_n(df, n):
    if df.empty:
        return pd.DataFrame()

    print(f"\nCalculating top {n} companies by market cap for each year...")

    df['year'] = pd.to_datetime(df['date']).dt.year
    df_year_end = df.groupby(['year', 'ticker']).last().reset_index()
    df_year_end['rank'] = df_year_end.groupby('year')['market_cap'].rank(
        ascending=False,
        method='min'
    )

    top_n = df_year_end[df_year_end['rank'] <= n].copy()
    top_n = top_n.sort_values(['year', 'rank'])
    top_n['market_cap_billions'] = (top_n['market_cap'] / 1e9).round(2)

    result = top_n[['year', 'rank', 'ticker', 'market_cap_billions', 'price']]

    print(f"Calculated top {n} for {len(result['year'].unique())} years")
    print(f"Year range: {result['year'].min():.0f} - {result['year'].max():.0f}")

    return result

def filter_complete_years(df, min_companies_per_year):
    print(f"\nFiltering to years with at least {min_companies_per_year} companies...")

    year_counts = df.groupby('year').size()
    full_years = year_counts[year_counts >= min_companies_per_year].index
    df_filtered = df[df['year'].isin(full_years)].copy()

    print(f"  Kept {len(full_years)} years with complete data")
    print(f"  Year range: {df_filtered['year'].min():.0f} - {df_filtered['year'].max():.0f}")

    return df_filtered

def calculate_market_share_percentages(df):
    print("\nCalculating market share percentages...")

    total_by_year = df.groupby('year')['market_cap_billions'].sum()
    df_with_pct = df.copy()
    df_with_pct['pct'] = df_with_pct.apply(
        lambda row: 100 * row['market_cap_billions'] / total_by_year[row['year']],
        axis=1
    )

    pivot = df_with_pct.pivot(index='year', columns='ticker', values='pct').fillna(0)
    col_sums = pivot.sum().sort_values(ascending=False)
    pivot = pivot[col_sums.index]

    print(f"  Companies in dataset: {len(pivot.columns)}")

    return pivot

def calculate_js_divergence(pivot):
    """
    Calculate Jensen-Shannon divergence between adjacent years.

    JS divergence is a symmetrized and smoothed version of KL divergence.
    It measures how different two probability distributions are.

    Range: 0 to 1
    - 0 = identical distributions
    - 1 = completely different distributions
    """
    print("\nCalculating Jensen-Shannon divergence between adjacent years...")

    years = sorted(pivot.index)
    js_divergences = []
    year_pairs = []

    for i in range(len(years) - 1):
        year1 = years[i]
        year2 = years[i + 1]

        # Get distributions for both years (as percentages)
        dist1 = pivot.loc[year1].values / 100.0  # Convert to probabilities
        dist2 = pivot.loc[year2].values / 100.0

        # Ensure distributions sum to 1 (they should, but just in case)
        dist1 = dist1 / dist1.sum()
        dist2 = dist2 / dist2.sum()

        # Calculate JS divergence
        js_div = jensenshannon(dist1, dist2, base=2)  # base=2 gives bits

        js_divergences.append(js_div)
        year_pairs.append(f"{year1}-{year2}")

        print(f"  {year1} -> {year2}: JS divergence = {js_div:.4f}")

    return years[1:], js_divergences, year_pairs

def plot_js_divergence(years, js_divergences, output_file='js_divergence.png'):
    """
    Create a line plot of JS divergence vs year.
    """
    print(f"\nCreating JS divergence plot...")

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot JS divergence as line with markers
    ax.plot(years, js_divergences, 'o-', linewidth=2, markersize=6, color='#2E86AB')

    # Highlight major changes (>0.3 is significant)
    significant_threshold = 0.3
    for year, js_div in zip(years, js_divergences):
        if js_div > significant_threshold:
            ax.axvline(x=year, color='red', alpha=0.3, linestyle='--', linewidth=1)
            ax.text(year, js_div + 0.02, f'{year}\n{js_div:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   color='red')

    # Add horizontal line at threshold
    ax.axhline(y=significant_threshold, color='red', alpha=0.2,
              linestyle=':', linewidth=1, label=f'Significant change threshold ({significant_threshold})')

    # Labels and title
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Jensen-Shannon Divergence', fontsize=14, fontweight='bold')
    ax.set_title('Market Composition Changes Over Time\n' +
                'Jensen-Shannon Divergence Between Adjacent Years',
                fontsize=16, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, alpha=0.3, axis='both')

    # Legend
    ax.legend(loc='upper left', fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    print(f"  Saving figure to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    print(f"Successfully saved: {output_file}")

    return fig

def main():
    print("=" * 70)
    print("JENSEN-SHANNON DIVERGENCE CALCULATOR")
    print("Measuring market composition changes year-over-year")
    print("=" * 70)
    print()

    # Get ticker universe
    print("Step 1: Getting ticker universe...")
    tickers = get_ticker_universe()
    print(f"  {len(tickers)} tickers to analyze")
    print()

    # Fetch historical data
    print("Step 2: Fetching historical data from Yahoo Finance...")
    df, failed_tickers = fetch_market_cap_data(tickers, START_YEAR, END_YEAR)

    if df.empty:
        print("\nERROR: No data fetched.")
        return

    print(f"  Fetched {len(df)} data points")
    print()

    # Calculate top N companies per year
    print(f"Step 3: Calculating top {TOP_N_COMPANIES} companies per year...")
    top_n = calculate_annual_top_n(df, TOP_N_COMPANIES)
    print()

    # Filter to complete years
    print("Step 4: Filtering to complete years...")
    top_n_filtered = filter_complete_years(top_n, MIN_COMPANIES_PER_YEAR)
    print()

    # Calculate market share percentages
    print("Step 5: Calculating market share percentages...")
    pivot = calculate_market_share_percentages(top_n_filtered)
    print()

    # Calculate JS divergence
    print("Step 6: Calculating Jensen-Shannon divergence...")
    years, js_divergences, year_pairs = calculate_js_divergence(pivot)
    print()

    # Find top 5 biggest changes
    print("\nTop 5 Biggest Market Composition Changes:")
    print("=" * 70)
    sorted_changes = sorted(zip(year_pairs, js_divergences), key=lambda x: x[1], reverse=True)
    for i, (year_pair, js_div) in enumerate(sorted_changes[:5], 1):
        print(f"  {i}. {year_pair}: JS divergence = {js_div:.4f}")
    print()

    # Create visualization
    print("Step 7: Creating visualization...")
    fig = plot_js_divergence(years, js_divergences)

    print()
    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print()
    print("Output file: js_divergence.png")
    print()

if __name__ == "__main__":
    main()
