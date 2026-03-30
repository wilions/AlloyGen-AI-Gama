# AlloyGen 2.0 — Architecture

## Overview

AlloyGen 2.0 is an intelligent alloy design platform combining multi-LLM chat, 34 ML models (including deep learning and uncertainty-aware models), alloy-specific physics featurization, SHAP explainability, multi-objective inverse design, and active learning — all through a real-time WebSocket chat interface.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, async Python |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS |
| Database | SQLAlchemy async + aiosqlite (SQLite) |
| Auth | JWT (python-jose) + bcrypt (passlib) |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch |
| Visualization | Recharts (bars, sparklines), Plotly (Pareto, heatmaps, ternary) |
| LLM | OpenAI SDK (supports Gemini, OpenAI, Claude via base_url switching) |

## Directory Structure

```
AlloyGen-2.0/
├── backend/
│   ├── app.py                    # FastAPI app factory with all routers
│   ├── main.py                   # Entrypoint
│   ├── config.py                 # Environment-based configuration
│   ├── pipeline.py               # ML pipeline orchestration
│   ├── errors.py                 # Custom exception classes
│   ├── utils.py                  # Column matching utilities
│   ├── auth/                     # JWT authentication
│   ├── database/                 # Async SQLAlchemy (5 tables)
│   ├── llm/                      # Multi-LLM provider registry
│   ├── agents/                   # 8 specialized agents
│   ├── ml/                       # ML modules (featurizers, UQ, deep learning, SHAP, optimization, active learning)
│   └── routes/                   # REST + WebSocket endpoints
├── frontend/
│   ├── src/
│   │   ├── api/                  # Auth API + JWT fetch wrapper
│   │   ├── context/              # AuthContext + ChatContext
│   │   ├── hooks/                # useWebSocket, useFileUpload
│   │   ├── pages/                # LoginPage, DashboardPage
│   │   └── components/
│   │       ├── charts/           # 7 chart types (Recharts + Plotly)
│   │       └── PeriodicTableInput.tsx
├── data/                         # SQLite database
├── models/                       # Saved .joblib model files
└── dev.sh                        # Start backend (:8005) + frontend (:5178)
```

## Data Flow

```
User Message → WebSocket → Pipeline Orchestrator
  ├── RequirementsAgent (LLM) → extract targets + task type
  ├── DataPrepAgent (LLM + featurizers) → clean + alloy physics features
  ├── ModelSearchAgent (deterministic) → select models
  ├── TrainingAgent (ML) → train 21+ models + SHAP + uncertainty
  └── PredictionAgent (LLM + ML) → predictions with UQ + SHAP
```

## Key Differentiators

| Feature | Description |
|---------|-------------|
| Alloy Physics Features | VEC, δ, ΔS_mix, ΔH_mix, Ω — auto-detected from composition columns |
| Uncertainty Quantification | GP (Matern 5/2) + ensemble uncertainty with confidence levels |
| Deep Learning | CrabNet-style attention, DeepMLP with residual, TabNet |
| SHAP Explainability | Global importance + per-prediction waterfall |
| Multi-LLM | Gemini, OpenAI, Claude via single SDK |
| Multi-Objective Design | Reparameterized composition + multivariate TPE + Pareto front |
| Active Learning | Campaign-based GP surrogates with Expected Improvement |
| Literature Extraction | LLM extracts composition-property data from paper text |
| Interactive Periodic Table | Element picker with composition sum tracking |
| Inline Charts | 7 chart types rendered inline in chat messages |
