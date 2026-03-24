# AlloyGen AI (Gama) — Architecture

## System Overview

```mermaid
graph TB
    subgraph Frontend["Frontend (React + Vite :5174)"]
        UI[Chat UI<br/>Glass Morphism]
        WS_Client[WebSocket Client<br/>useWebSocket hook]
        FileUp[File Upload<br/>useFileUpload hook]
        State[ChatContext<br/>useReducer + localStorage]

        UI --> State
        UI --> WS_Client
        UI --> FileUp
    end

    subgraph Backend["Backend (FastAPI :8001)"]
        WS_Server[WebSocket Endpoint<br/>/ws/session_id]
        Upload[POST /upload]
        Batch[POST /batch-predict]
        Download[GET /download/path]
        Sessions[(Session Store<br/>in-memory dict)]
        Pipeline[Pipeline Runner<br/>pipeline.py]

        WS_Server --> Sessions
        WS_Server --> Pipeline
        Upload --> Sessions
    end

    subgraph Agents["Agent Layer"]
        Req[Requirements<br/>Agent]
        Prep[DataPrep<br/>Agent]
        Search[OnlineSearch<br/>Agent]
        ModelS[ModelSearch<br/>Agent]
        Train[Training<br/>Agent]
        Pred[Prediction<br/>Agent]
        Inv[InverseDesign<br/>Agent]
    end

    subgraph ML["ML Layer"]
        Models["Model Registry<br/>20+ algorithms"]
        MultiOut[Multi-Output<br/>Wrapper]
        Optuna[Optuna<br/>Optimizer]
        Weights["Target Weighting<br/>inv-variance × precision"]
    end

    subgraph External["External Services"]
        Gemini[Gemini API<br/>Flash + Pro models]
        Disk[(File System<br/>uploads/ models/)]
    end

    WS_Client <-->|"WebSocket JSON"| WS_Server
    FileUp -->|"multipart/form-data"| Upload
    UI -->|"CSV upload"| Batch

    Pipeline --> Req & Prep & Search & ModelS & Train
    WS_Server --> Pred & Inv

    Req & Prep & Search --> Gemini
    Pred --> Gemini
    Inv --> Gemini

    Train --> Models
    Models --> MultiOut
    Inv --> Optuna
    Optuna --> Weights

    Train --> Disk
    Pred --> Disk
    Inv --> Disk
    Upload --> Disk
```

## Pipeline Flow

```mermaid
flowchart LR
    A[Requirements<br/>Gathering] --> B[Data<br/>Reorganization]
    B --> C[Online Data<br/>Search]
    C --> D[Model<br/>Search]
    D --> E[Training &<br/>Selection]
    E --> F[Prediction<br/>Ready]

    style A fill:#58a6ff,color:#000
    style F fill:#3fb950,color:#000
```

### Post-Pipeline Capabilities

```mermaid
flowchart TB
    Ready[Prediction Ready] --> FP[Forward Prediction]
    Ready --> ID[Inverse Design]
    Ready --> BP[Batch Prediction]

    FP --> Extract["LLM extracts features<br/>from natural language"]
    Extract --> Predict["model.predict()"]
    Predict --> Validate["Extrapolation &<br/>plausibility checks"]

    ID --> Step1["Step 1: Extract constraints<br/>via LLM (JSON mode)"]
    Step1 --> Step2["Step 2: User selects<br/>elements to optimize"]
    Step2 --> Optimize["Optuna optimization<br/>200 trials"]
    Optimize --> Rank["Rank top 5<br/>candidates"]

    BP --> Align["Align CSV columns<br/>to training features"]
    Align --> BulkPred["Bulk model.predict()"]
    BulkPred --> CSV["Output CSV with<br/>predicted columns"]
```

## Inverse Design Weighting (Multi-Output)

```mermaid
flowchart TB
    subgraph Weighting["Per-Target Weight Calculation"]
        IV["Inverse Variance<br/>w_var = 1 / var(target)"]
        CP["Constraint Precision<br/>w_prec = data_range / constraint_span"]
        Combine["w = w_var × w_prec"]
        Norm["Mean-normalise<br/>so avg weight = 1.0"]

        IV --> Combine
        CP --> Combine
        Combine --> Norm
    end

    subgraph Penalty["Weighted Penalty Function"]
        Violation["violation = predicted - bound"]
        Scale["Normalise by data range"]
        Weight["Multiply by target weight"]
        Sum["total = Σ weight × (violation / range)²"]

        Violation --> Scale --> Weight --> Sum
    end

    Norm --> Weight
```

**Weight examples:**
- Tight constraint `hardness 49-51` (span=2) over data range 500 → high `w_prec`
- Low-variance target → high `w_var`
- One-sided constraint (only min or max) → neutral `w_prec = 1.0`

## Prediction Safety

```mermaid
flowchart LR
    Input[User Input] --> LLM["LLM feature<br/>extraction"]
    LLM --> Check1{"Input features<br/>within training<br/>range?"}
    Check1 -->|No| Warn1["⚠️ Extrapolation<br/>warning"]
    Check1 -->|Yes| Predict
    Warn1 --> Predict[model.predict]
    Predict --> Check2{"Prediction within<br/>plausible range?"}
    Check2 -->|No| Warn2["⚠️ Plausibility<br/>warning"]
    Check2 -->|Yes| Output[Show result]
    Warn2 --> Output
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, TypeScript, Tailwind CSS 4, Vite 8 |
| Backend | FastAPI, Python 3.9+, WebSocket |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna |
| LLM | Gemini API (Flash for extraction, Pro for analysis) |
| State | In-memory sessions (backend), localStorage (frontend) |

## File Structure

```
Alloy Gen Chatbot/
├── backend/
│   ├── main.py              # FastAPI app, WebSocket, endpoints
│   ├── pipeline.py           # Session dataclass, pipeline runner
│   ├── config.py             # Environment config (Gemini keys, limits)
│   ├── errors.py             # Custom exceptions
│   └── agents/
│       ├── base_agent.py     # BaseAgent with LLM chat helper
│       ├── requirements_agent.py
│       ├── data_prep_agent.py
│       ├── online_search_agent.py
│       ├── model_search_agent.py
│       ├── training_agent.py
│       ├── prediction_agent.py    # + extrapolation/plausibility checks
│       └── inverse_design_agent.py # + weighted multi-output optimization
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── index.css         # Glass morphism theme
│   │   ├── types.ts
│   │   ├── constants.ts      # Pipeline step definitions
│   │   ├── context/ChatContext.tsx
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts
│   │   │   └── useFileUpload.ts
│   │   └── components/
│   │       ├── Header.tsx
│   │       ├── Sidebar.tsx
│   │       ├── ChatPanel.tsx      # + welcome state
│   │       ├── ChatMessage.tsx    # + avatars, timestamps
│   │       ├── PipelineTracker.tsx # + progress line
│   │       ├── FileUpload.tsx
│   │       ├── TypingIndicator.tsx
│   │       ├── ReconnectBanner.tsx
│   │       └── ToastContainer.tsx
│   └── vite.config.ts        # Proxy /ws → :8001
├── models/                    # Saved .joblib models
├── uploads/                   # Temporary uploaded files
├── dev.sh                     # Start both servers
└── requirements.txt
```
