#!/bin/bash
set -e

if [ "$1" = "api" ]; then
    exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
elif [ "$1" = "dashboard" ]; then
    exec streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
elif [ "$1" = "both" ]; then
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
    exec streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
else
    exec "$@"
fi
