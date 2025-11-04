#!/usr/bin/env python3
"""
Generate Market Share Stacked Bar Chart from Scratch

This script generates market share visualizations from Yahoo Finance data:
1. Loads ticker list from tickers.json configuration file
2. Fetches historical market cap data from Yahoo Finance (yfinance)
3. Calculates top 10 companies by market cap for each year
4. Creates a stacked bar chart visualization

Input:
    - tickers.json: Configuration file with ticker symbols organized by sector

Output:
    - market_share_stacked.png: Stacked bar chart showing percentage of total
      top 10 market cap held by each company, year by year
    - failed_tickers.log: Log file listing any tickers that failed to fetch
      with reasons (only created if there are failures)

Requirements:
    - Internet connection (to fetch data from Yahoo Finance)
    - Python packages: pandas, numpy, matplotlib, yfinance

Note: The data fetching process may take several minutes (5-10 minutes) as it
      needs to query Yahoo Finance for historical data for ~170 companies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import warnings
import json
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Suppress yfinance warnings and messages
import logging
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# =============================================================================
# CONFIGURATION PARAMETERS - DATA FETCHING
# =============================================================================

# Year range for historical data
START_YEAR = 1970
END_YEAR = None  # None = current year

# Number of top companies to include per year
TOP_N_COMPANIES = 10

# =============================================================================
# CONFIGURATION PARAMETERS - VISUALIZATION
# =============================================================================

# Output file path
OUTPUT_PNG = 'market_share_stacked.png'

# Minimum number of companies required per year to include that year in analysis
# This filters out years with incomplete data
MIN_COMPANIES_PER_YEAR = 10

# Figure dimensions in inches (width, height)
FIGURE_WIDTH = 20
FIGURE_HEIGHT = 10

# DPI (dots per inch) for output image - higher = better quality but larger file
OUTPUT_DPI = 300

# Bar width as fraction of x-axis unit (0.8 means small gaps between bars)
BAR_WIDTH = 0.8

# Threshold for labeling segments: only label if segment is this % or larger
# This prevents clutter from tiny segments
LABEL_THRESHOLD_PCT = 5.0

# Threshold for using white text: segments larger than this use white text
# Smaller segments use black text for better contrast
WHITE_TEXT_THRESHOLD_PCT = 10.0

# Font sizes for various elements
LABEL_FONT_SIZE = 8
AXIS_LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 16
LEGEND_FONT_SIZE = 10

# Number of top companies to include in legend (to avoid overcrowding)
LEGEND_TOP_N = 20

# Y-axis tick interval (percentage points)
Y_AXIS_TICK_INTERVAL = 10

# =============================================================================
# TICKER UNIVERSE
# =============================================================================

def get_ticker_universe(ticker_file='tickers.json'):
    """
    Load comprehensive list of major US companies from JSON configuration file.

    This includes current and historical companies across multiple sectors:
    - Technology (current and historical/delisted)
    - Finance (current and historical, including 2008 crisis casualties)
    - Healthcare/Pharma
    - Consumer/Retail
    - Industrial/Energy
    - Automotive
    - Telecom
    - Airlines
    - Other historical companies

    Parameters:
        ticker_file (str): Path to JSON file containing ticker symbols
                          Default: 'tickers.json' in current directory

    Returns:
        list: Sorted list of unique ticker symbols

    Raises:
        FileNotFoundError: If ticker file doesn't exist
        json.JSONDecodeError: If ticker file is not valid JSON
    """
    ticker_path = Path(ticker_file)

    if not ticker_path.exists():
        raise FileNotFoundError(
            f"Ticker file not found: {ticker_file}\n"
            f"Expected location: {ticker_path.absolute()}"
        )

    # Load ticker data from JSON file
    with open(ticker_path, 'r') as f:
        ticker_data = json.load(f)

    # Flatten nested structure into single list
    tickers = []
    for sector, subcategories in ticker_data.items():
        for subcategory, ticker_list in subcategories.items():
            tickers.extend(ticker_list)

    # Return sorted unique list
    return sorted(list(set(tickers)))

# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_market_cap_data(tickers, start_year, end_year):
    """
    Fetch historical market cap data from Yahoo Finance for given tickers.

    For each ticker, this function:
    1. Downloads monthly historical price data
    2. Gets current shares outstanding
    3. Calculates approximate market cap (price × shares)

    Note: This uses current shares outstanding as an approximation for all
          historical periods. For truly accurate historical data, you would
          need historical shares outstanding from quarterly reports.

    Parameters:
        tickers (list): List of ticker symbols to fetch
        start_year (int): Starting year for data
        end_year (int or None): Ending year (None = current year)

    Returns:
        tuple: (pandas.DataFrame, dict)
            - DataFrame with columns: date, ticker, price, market_cap
              Contains monthly data points for each ticker
            - Dictionary with failed tickers and reasons
    """
    if end_year is None:
        end_year = datetime.now().year

    all_data = []
    failed_tickers = {}

    print(f"Fetching data for {len(tickers)} tickers from {start_year} to {end_year}...")
    print("This may take 5-10 minutes. Please be patient.\n")

    for i, ticker in enumerate(tickers, 1):
        # Progress indicator every 10 tickers
        if i % 10 == 0:
            print(f"Progress: {i}/{len(tickers)} tickers processed...")

        try:
            # Create yfinance Ticker object
            stock = yf.Ticker(ticker)

            # Get historical price data (monthly intervals)
            # Monthly data is sufficient for yearly analysis and faster to fetch
            hist = stock.history(
                start=f"{start_year}-01-01",
                end=f"{end_year+1}-01-01",
                interval="1mo"
            )

            # Skip if no data available
            if hist.empty:
                failed_tickers[ticker] = "No historical price data available"
                continue

            # Get current shares outstanding from company info
            # Note: This is a simplification - ideally we'd have historical shares
            info = stock.info
            shares_outstanding = info.get('sharesOutstanding', None)

            # Skip if shares outstanding not available
            if shares_outstanding is None:
                failed_tickers[ticker] = "Shares outstanding not available"
                continue

            # Calculate market cap for each historical price point
            # Market cap = Price × Shares Outstanding
            for date, row in hist.iterrows():
                close_price = row['Close']

                # Only include valid prices
                if pd.notna(close_price) and close_price > 0:
                    market_cap = close_price * shares_outstanding

                    all_data.append({
                        'date': date,
                        'ticker': ticker,
                        'price': close_price,
                        'market_cap': market_cap
                    })

        except Exception as e:
            # Skip tickers that error out (delisted, API issues, etc.)
            failed_tickers[ticker] = f"Error: {str(e)}"
            continue

    print(f"\nData fetching complete. {len(all_data)} data points collected.")
    print(f"Successfully fetched: {len(tickers) - len(failed_tickers)}/{len(tickers)} tickers")
    print(f"Failed: {len(failed_tickers)} tickers (details will be saved to log file)")

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    return df, failed_tickers

# =============================================================================
# DATA PROCESSING
# =============================================================================

def calculate_annual_top_n(df, n):
    """
    Calculate top N companies by market cap for each year.

    For each year:
    1. Takes the last available data point (year-end approximation)
    2. Ranks companies by market cap
    3. Selects top N companies

    Parameters:
        df (pandas.DataFrame): DataFrame with date, ticker, market_cap columns
        n (int): Number of top companies to select per year

    Returns:
        pandas.DataFrame: DataFrame with columns: year, rank, ticker,
                         market_cap_billions, price
                         Contains top N companies for each year
    """
    if df.empty:
        return pd.DataFrame()

    print(f"\nCalculating top {n} companies by market cap for each year...")

    # Extract year from date
    df['year'] = pd.to_datetime(df['date']).dt.year

    # For each year-ticker combination, take the last available data point
    # This approximates year-end values
    df_year_end = df.groupby(['year', 'ticker']).last().reset_index()

    # Rank companies by market cap within each year
    # rank=1 means largest market cap, rank=2 means second largest, etc.
    df_year_end['rank'] = df_year_end.groupby('year')['market_cap'].rank(
        ascending=False,  # Higher market cap = lower rank number
        method='min'       # Ties get the same rank
    )

    # Filter to keep only top N companies per year
    top_n = df_year_end[df_year_end['rank'] <= n].copy()

    # Sort by year and rank for readability
    top_n = top_n.sort_values(['year', 'rank'])

    # Convert market cap from dollars to billions for easier reading
    top_n['market_cap_billions'] = (top_n['market_cap'] / 1e9).round(2)

    # Select and order columns
    result = top_n[['year', 'rank', 'ticker', 'market_cap_billions', 'price']]

    print(f"Calculated top {n} for {len(result['year'].unique())} years")
    print(f"Year range: {result['year'].min():.0f} - {result['year'].max():.0f}")

    return result

def filter_complete_years(df, min_companies_per_year):
    """
    Filter dataset to include only years with complete coverage.

    Some years may have incomplete data (fewer than N companies) due to:
    - Limited data availability for early years
    - Delisted companies without historical data
    - Data quality issues

    This function filters out years with insufficient data.

    Parameters:
        df (pandas.DataFrame): DataFrame with year column
        min_companies_per_year (int): Minimum companies required per year

    Returns:
        pandas.DataFrame: Filtered dataframe with only complete years
    """
    print(f"\nFiltering to years with at least {min_companies_per_year} companies...")

    # Count how many companies we have for each year
    year_counts = df.groupby('year').size()

    # Identify years that meet the minimum threshold
    full_years = year_counts[year_counts >= min_companies_per_year].index

    # Filter the dataframe
    df_filtered = df[df['year'].isin(full_years)].copy()

    print(f"  Kept {len(full_years)} years with complete data")
    print(f"  Year range: {df_filtered['year'].min():.0f} - {df_filtered['year'].max():.0f}")

    return df_filtered

# =============================================================================
# MARKET SHARE CALCULATION
# =============================================================================

def calculate_market_share_percentages(df):
    """
    Calculate each company's market share as percentage of total for that year.

    For each year:
    1. Calculate total market cap (sum of all top N companies)
    2. Calculate each company's percentage of that total
    3. Pivot to create a matrix (years × companies)

    Parameters:
        df (pandas.DataFrame): DataFrame with columns: year, ticker, market_cap_billions

    Returns:
        pandas.DataFrame: Pivot table with years as rows, tickers as columns,
                         and market share percentages as values
    """
    print("\nCalculating market share percentages...")

    # Calculate the total market cap for each year
    # This is the sum of all top N companies' market caps
    total_by_year = df.groupby('year')['market_cap_billions'].sum()

    # Create a copy to add percentage column
    df_with_pct = df.copy()

    # Calculate percentage for each company-year combination
    # Percentage = (Company's market cap / Year's total) × 100
    df_with_pct['pct'] = df_with_pct.apply(
        lambda row: 100 * row['market_cap_billions'] / total_by_year[row['year']],
        axis=1
    )

    # Pivot the data into a matrix format
    # Rows = years, Columns = tickers, Values = percentages
    # fillna(0) handles companies that didn't exist in certain years
    pivot = df_with_pct.pivot(index='year', columns='ticker', values='pct').fillna(0)

    # Sort columns by total appearance across all years
    # This keeps the most significant companies first and ensures consistent colors
    col_sums = pivot.sum().sort_values(ascending=False)
    pivot = pivot[col_sums.index]

    print(f"  Companies in dataset: {len(pivot.columns)}")
    print(f"  Most significant: {', '.join(pivot.columns[:5].tolist())}")

    return pivot

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_stacked_bar_chart(pivot, output_file, figure_size, dpi, bar_width,
                            label_threshold, white_text_threshold, label_font_size,
                            axis_label_font_size, title_font_size, legend_font_size,
                            legend_top_n, y_tick_interval):
    """
    Create and save a stacked bar chart of market share percentages.

    The chart shows one bar per year, with each bar divided into colored segments
    representing each company's market share percentage.

    Parameters:
        pivot (pandas.DataFrame): Pivot table with years as index, companies as columns
        output_file (str): Path where the PNG image will be saved
        figure_size (tuple): (width, height) in inches
        dpi (int): Resolution in dots per inch
        bar_width (float): Width of bars as fraction of x-axis unit
        label_threshold (float): Minimum percentage to show label on segment
        white_text_threshold (float): Minimum percentage to use white text
        label_font_size (int): Font size for bar labels
        axis_label_font_size (int): Font size for axis labels
        title_font_size (int): Font size for title
        legend_font_size (int): Font size for legend
        legend_top_n (int): Number of companies to include in legend
        y_tick_interval (int): Spacing between y-axis ticks

    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    print(f"\nCreating stacked bar chart...")

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=figure_size)

    # Get arrays of years and companies
    years = pivot.index.values
    companies = pivot.columns
    n_companies = len(companies)

    # Create color map: assign a unique color to each company
    # tab20c provides 20 distinct colors, will cycle if more companies
    colors = plt.cm.tab20c(np.linspace(0, 1, n_companies))

    # Initialize array to track cumulative height for stacking
    # Each company's bar segment sits on top of the previous companies
    bottom = np.zeros(len(years))

    # Store bar objects and labels for legend
    bars = []
    labels = []

    # Draw bars for each company
    print(f"  Drawing bars for {n_companies} companies...")
    for i, ticker in enumerate(companies):
        # Get this company's percentage values for all years
        values = pivot[ticker].values

        # Create stacked bar segment for this company
        bar = ax.bar(
            years,                    # x-coordinates (years)
            values,                   # heights (percentages)
            bottom=bottom,            # starting height (top of previous layer)
            label=ticker,             # legend label
            color=colors[i],          # assigned color
            edgecolor='white',        # white border for separation
            linewidth=0.5,            # thin border
            width=bar_width           # bar width
        )

        # Store for legend
        bars.append(bar)
        labels.append(ticker)

        # Update bottom for next layer
        # Next company's bar starts where this one ends
        bottom += values

    # Add text labels on bar segments
    print("  Adding labels to bars...")
    for year_idx, year in enumerate(years):
        # Track cumulative height to position labels correctly
        cumulative = 0

        for ticker_idx, ticker in enumerate(companies):
            # Get the percentage value for this company-year
            value = pivot.loc[year, ticker]

            # Only label segments that are large enough to read
            if value >= label_threshold:
                # Calculate vertical position (middle of segment)
                y_pos = cumulative + value / 2

                # Choose text color based on segment size
                # White text on large segments, black on small ones
                text_color = 'white' if value > white_text_threshold else 'black'

                # Add text label showing ticker and percentage
                ax.text(
                    year, y_pos,                           # position
                    f'{ticker}\n{value:.1f}%',            # text
                    ha='center', va='center',              # alignment
                    fontsize=label_font_size,              # size
                    fontweight='bold',                     # weight
                    color=text_color                       # color
                )

            # Update cumulative height for next company
            cumulative += value

    # Configure axes and labels
    print("  Configuring axes and labels...")

    # X-axis label
    ax.set_xlabel('Year', fontsize=axis_label_font_size, fontweight='bold')

    # Y-axis label
    ax.set_ylabel('Percentage of Total Top 10 Market Cap (%)',
                  fontsize=axis_label_font_size, fontweight='bold')

    # Title (two lines)
    ax.set_title(
        'Market Share Distribution Among Top 10 Companies\n' +
        'Stacked by Company (% of total top 10 market cap)',
        fontsize=title_font_size,
        fontweight='bold',
        pad=20
    )

    # Y-axis range and ticks
    ax.set_ylim(0, 100)  # 0-100% range
    ax.set_yticks(range(0, 101, y_tick_interval))

    # Create legend with most frequent companies only
    # Including all companies would clutter the legend
    print("  Adding legend...")

    col_sums = pivot.sum().sort_values(ascending=False)
    top_companies = col_sums.head(legend_top_n).index

    # Filter to top companies
    handles = [bars[i] for i, ticker in enumerate(companies) if ticker in top_companies]
    legend_labels = [ticker for ticker in companies if ticker in top_companies]

    # Add legend
    ax.legend(
        handles, legend_labels,
        loc='upper left',
        bbox_to_anchor=(1.01, 1),  # position outside plot area
        fontsize=legend_font_size,
        ncol=1
    )

    # Add grid for easier reading
    ax.grid(True, alpha=0.3, axis='y', zorder=0)

    # Adjust layout to prevent cutoff
    plt.tight_layout()

    # Save figure
    print(f"  Saving figure to {output_file}...")
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')

    print(f"Successfully saved: {output_file}")

    return fig

# =============================================================================
# ERROR LOGGING
# =============================================================================

def write_failed_tickers_log(failed_tickers, output_file='failed_tickers.log'):
    """
    Write failed tickers and their failure reasons to a log file.

    Parameters:
        failed_tickers (dict): Dictionary mapping ticker symbols to failure reasons
        output_file (str): Path to log file. Default: 'failed_tickers.log'

    Returns:
        None
    """
    if not failed_tickers:
        print("  No failed tickers to log.")
        return

    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FAILED TICKERS LOG\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total failed: {len(failed_tickers)}\n\n")
        f.write("-" * 70 + "\n\n")

        # Sort by ticker symbol for easy reading
        for ticker in sorted(failed_tickers.keys()):
            reason = failed_tickers[ticker]
            f.write(f"{ticker:10} | {reason}\n")

    print(f"  Failed tickers log saved to: {output_file}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function that orchestrates the entire process:
    1. Get list of tickers to analyze
    2. Fetch historical data from Yahoo Finance
    3. Calculate top N companies per year
    4. Filter to complete years
    5. Calculate market share percentages
    6. Create and save visualization
    """
    print("=" * 70)
    print("MARKET SHARE STACKED BAR CHART GENERATOR")
    print("Generating from scratch using Yahoo Finance data")
    print("=" * 70)
    print()

    # Step 1: Get ticker universe
    print("Step 1: Getting ticker universe...")
    tickers = get_ticker_universe()
    print(f"  {len(tickers)} tickers to analyze")
    print(f"  Sample: {', '.join(tickers[:10])}")
    print()

    # Step 2: Fetch historical data from Yahoo Finance
    print("Step 2: Fetching historical data from Yahoo Finance...")
    print("  WARNING: This will take 5-10 minutes. Please be patient.")
    df, failed_tickers = fetch_market_cap_data(tickers, START_YEAR, END_YEAR)

    # Write failed tickers log
    write_failed_tickers_log(failed_tickers)

    if df.empty:
        print("\nERROR: No data fetched. Possible causes:")
        print("  - No internet connection")
        print("  - Yahoo Finance API issues")
        print("  - All tickers failed to fetch")
        print(f"  Check failed_tickers.log for details")
        return

    print(f"  Fetched {len(df)} data points")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print()

    # Step 3: Calculate top N companies per year
    print(f"Step 3: Calculating top {TOP_N_COMPANIES} companies per year...")
    top_n = calculate_annual_top_n(df, TOP_N_COMPANIES)
    print()

    # Step 4: Filter to complete years
    print("Step 4: Filtering to complete years...")
    top_n_filtered = filter_complete_years(top_n, MIN_COMPANIES_PER_YEAR)
    print()

    # Step 5: Calculate market share percentages
    print("Step 5: Calculating market share percentages...")
    pivot = calculate_market_share_percentages(top_n_filtered)
    print()

    # Step 6: Create visualization
    print("Step 6: Creating visualization...")
    fig = create_stacked_bar_chart(
        pivot,
        OUTPUT_PNG,
        figure_size=(FIGURE_WIDTH, FIGURE_HEIGHT),
        dpi=OUTPUT_DPI,
        bar_width=BAR_WIDTH,
        label_threshold=LABEL_THRESHOLD_PCT,
        white_text_threshold=WHITE_TEXT_THRESHOLD_PCT,
        label_font_size=LABEL_FONT_SIZE,
        axis_label_font_size=AXIS_LABEL_FONT_SIZE,
        title_font_size=TITLE_FONT_SIZE,
        legend_font_size=LEGEND_FONT_SIZE,
        legend_top_n=LEGEND_TOP_N,
        y_tick_interval=Y_AXIS_TICK_INTERVAL
    )

    print()
    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print()
    print(f"Output file: {OUTPUT_PNG}")
    if failed_tickers:
        print(f"Failed tickers log: failed_tickers.log ({len(failed_tickers)} failed)")
    print()

if __name__ == "__main__":
    main()
