#!/bin/bash
# AlloyGen 2.0 — Start both backend and frontend dev servers

trap 'kill 0' EXIT

cd "$(dirname "$0")"

echo "Starting backend on :8005..."
uvicorn backend.app:app --reload --port 8005 &

echo "Starting frontend on :5178..."
cd frontend && npm run dev &

wait
