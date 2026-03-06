#!/usr/bin/env bash
# Run the pipeline using the project venv (avoids NumPy 2 / Anaconda conflicts)
cd "$(dirname "$0")"
.venv/bin/python -m app.main "$@"
