#!/bin/bash
echo "Starting Monroe Archives API..."
uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}
