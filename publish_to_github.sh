#!/bin/bash

# =============================================================================
# GitHub Repository Creation and Push Script
# =============================================================================
# This script will:
#   1. Initialize a git repository in the current directory
#   2. Create a public GitHub repository using gh CLI
#   3. Add and commit all files
#   4. Push to the remote repository
#
# EDIT THE VARIABLES BELOW BEFORE RUNNING
# =============================================================================

# Repository name (will appear as github.com/YOUR_USERNAME/REPO_NAME)
REPO_NAME="market-share-visualization"

# Repository description (shows up on GitHub)
REPO_DESCRIPTION="Python script that generates market share visualization from Yahoo Finance data with clean separation of data and code"

# Commit message for the initial commit
COMMIT_MESSAGE="Initial commit: Market share stacked bar chart generator

- Python script that fetches historical market cap data from Yahoo Finance
- Ticker list separated into tickers.json configuration file for easy editing
- Generates market_share_stacked.png visualization
- Failed ticker logging to failed_tickers.log for debugging
- Heavily commented code with no magic numbers
- All parameters configurable at top of script
- Clean separation of data (JSON) from code (Python)
- Includes example output image
- Takes 5-10 minutes to run (downloads ~55 years of data for ~100 companies)"

# Repository visibility (public or private)
REPO_VISIBILITY="public"

# =============================================================================
# EXECUTION (you can leave this section as-is)
# =============================================================================

set -e  # Exit on any error

echo "=========================================="
echo "GitHub Repository Setup"
echo "=========================================="
echo ""
echo "Repository name: $REPO_NAME"
echo "Visibility: $REPO_VISIBILITY"
echo "Description: $REPO_DESCRIPTION"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
else
    echo "Git repository already initialized."
fi

# Add all files to staging
echo "Adding files to git..."
git add .

# Create initial commit
echo "Creating commit..."
git commit -m "$COMMIT_MESSAGE"

# Create GitHub repository using gh CLI
echo "Creating GitHub repository..."
gh repo create "$REPO_NAME" \
    --"$REPO_VISIBILITY" \
    --description "$REPO_DESCRIPTION" \
    --source=. \
    --push

echo ""
echo "=========================================="
echo "SUCCESS!"
echo "=========================================="
echo ""
echo "Repository created and pushed to GitHub"
echo "View it at: https://github.com/$(gh api user -q .login)/$REPO_NAME"
echo ""
