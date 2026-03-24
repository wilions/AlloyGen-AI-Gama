#!/bin/bash
# Start both backend and frontend dev servers

trap 'kill 0' EXIT

echo "Starting backend on :8005..."
cd "$(dirname "$0")"
uvicorn backend.main:app --reload --port 8005 &

echo "Starting frontend on :5178..."
cd frontend && npm run dev &

wait
