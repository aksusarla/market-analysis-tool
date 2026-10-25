#!/usr/bin/env bash
# Download (or update) NSE end-of-day data: ~420 MB, 3,700+ symbols, daily since 1995.
set -euo pipefail
mkdir -p data
if [ -d data/eod2_data/.git ]; then
  git -C data/eod2_data pull --ff-only
else
  git clone --depth 1 --filter=blob:none --sparse https://github.com/BennyThadikaran/eod2_data.git data/eod2_data
  git -C data/eod2_data sparse-checkout set daily
fi
echo "Data ready in data/eod2_data"
