#!/usr/bin/env bash
set -e
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/main.py --minutes ${1:-2} --arrival ${2:-600} --cycle ${3:-60}
