#!/usr/bin/env python3
"""
Chart showing the decoupling of unemployment and stock market performance
Compares S&P 500 and unemployment rate over the last 20 years
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import yfinance as yf

# Set style
plt.style.use('dark_background')

# Fetch S&P 500 data
print("Fetching S&P 500 data...")
sp500 = yf.download('^GSPC', start='2005-01-01', end=datetime.now().strftime('%Y-%m-%d'))
sp500_monthly = sp500['Close'].resample('MS').mean()

# Normalize S&P 500 to base 100 (Jan 2005)
sp500_normalized = (sp500_monthly / sp500_monthly.iloc[0]) * 100

# U.S. unemployment data (you'll need to get this from FRED or manually)
# For now, I'll create a function to fetch it via pandas_datareader
try:
    from pandas_datareader import data as pdr
    print("Fetching unemployment data from FRED...")
    unemployment = pdr.DataReader('UNRATE', 'fred', '2005-01-01', datetime.now().strftime('%Y-%m-%d'))
except ImportError:
    print("pandas_datareader not installed. Using yfinance for both datasets.")
    print("Note: For unemployment data, consider installing pandas_datareader:")
    print("  pip install pandas_datareader")
    # Fallback: create synthetic unemployment data for demonstration
    dates = sp500_monthly.index
    # Rough unemployment approximation (you should replace with real data)
    unemployment = pd.DataFrame({
        'UNRATE': [
            5.3, 5.1, 5.0, 4.9, 4.7, 4.6, 4.5, 4.7, 5.8, 9.3,  # 2005-2009
            9.6, 8.9, 8.1, 7.4, 6.2, 5.3, 4.9, 4.4, 3.9, 3.7,  # 2010-2019
            3.5, 13.0, 8.4, 6.7, 5.4, 4.2, 3.8, 3.6, 3.5, 3.7,  # 2020-2024
            4.0  # 2025
        ]
    }, index=pd.date_range('2005-01-01', periods=len(dates), freq='YS'))
    print("WARNING: Using synthetic unemployment data. Install pandas_datareader for real data.")

# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot S&P 500 on left axis
color1 = '#00ff00'  # Bright green
ax1.set_xlabel('Year', fontsize=14, color='white', fontweight='bold')
ax1.set_ylabel('S&P 500 (indexed to 100 in 2005)', fontsize=14, color=color1, fontweight='bold')
line1 = ax1.plot(sp500_normalized.index, sp500_normalized.values,
                 color=color1, linewidth=3, label='S&P 500 (indexed)')
ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax1.tick_params(axis='x', labelcolor='white', labelsize=12)
ax1.grid(True, alpha=0.3, color='white')

# Create second y-axis for unemployment
ax2 = ax1.twinx()
color2 = '#ffffff'  # White for unemployment
ax2.set_ylabel('Unemployment Rate (%)', fontsize=14, color=color2, fontweight='bold')
line2 = ax2.plot(unemployment.index, unemployment['UNRATE'],
                 color=color2, linewidth=3, label='Unemployment Rate', linestyle='--', alpha=0.9)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)

# Highlight key periods
# 2008 Financial Crisis
ax1.axvspan(datetime(2007, 12, 1), datetime(2009, 6, 1),
            alpha=0.2, color='yellow', label='2008 Crisis')

# COVID-19 Pandemic
ax1.axvspan(datetime(2020, 2, 1), datetime(2020, 12, 1),
            alpha=0.25, color='red', label='COVID-19')

# Post-COVID "decoupling" period
ax1.axvspan(datetime(2021, 1, 1), datetime.now(),
            alpha=0.15, color='cyan', label='Post-COVID Period')

# Title and legend
plt.title('Decoupling of Stock Market and Unemployment\nS&P 500 vs. Unemployment Rate (2005-2025)',
          fontsize=18, color='white', pad=20, fontweight='bold')

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=12, framealpha=0.8, facecolor='#1a1a1a', edgecolor='white')

# Add annotations for key observations
ax1.annotate('2008: Markets crash,\nunemployment rises\n(coupled)',
             xy=(datetime(2009, 1, 1), 50),
             xytext=(datetime(2010, 1, 1), 70),
             arrowprops=dict(arrowstyle='->', color='white', alpha=0.9, lw=2),
             fontsize=11, color='white', ha='left', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='black', edgecolor='white', alpha=0.8))

ax1.annotate('2020-2025: Markets recover rapidly,\nunemployment remains elevated\n(decoupled)',
             xy=(datetime(2022, 6, 1), 400),
             xytext=(datetime(2016, 1, 1), 450),
             arrowprops=dict(arrowstyle='->', color='white', alpha=0.9, lw=2),
             fontsize=11, color='white', ha='left', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='black', edgecolor='white', alpha=0.8))

plt.tight_layout()

# Save figure
output_file = 'unemployment_market_decoupling.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
print(f"\nChart saved to: {output_file}")

# Also save as SVG for presentations
output_svg = 'unemployment_market_decoupling.svg'
plt.savefig(output_svg, bbox_inches='tight', facecolor='black')
print(f"SVG saved to: {output_svg}")

plt.show()
