#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <url> <password>"
    exit 1
fi

URL=$1
PASSWORD=$2


uv run playwright install-deps  # This might require sudo.
uv run playwright install chromium

uv run --python 3.14 download_scripts/download_mvimgnet2.py \
  --url "$URL" \
  --pwd "$PASSWORD"