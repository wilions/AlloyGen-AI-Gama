from __future__ import annotations

import json
import logging
import os

import joblib
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.agents import (
    RequirementsAgent,
    DataPrepAgent,
    OnlineSearchAgent,
    ModelSearchAgent,
    TrainingAgent,
    PredictionAgent,
    InverseDesignAgent,
)
from backend.pipeline import send_ws, run_pipeline
from backend.routes._helpers import (
    sessions,
    sanitize_id,
    get_or_create_session,
    touch,
    is_inverse_design_request,
    GREETING,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    session_id = sanitize_id(session_id)
    await websocket.accept()

    session = get_or_create_session(session_id)

    agents = {
        "requirements": RequirementsAgent(),
        "prep": DataPrepAgent(),
        "online_search": OnlineSearchAgent(),
        "model_search": ModelSearchAgent(),
        "train": TrainingAgent(),
        "prediction": PredictionAgent(),
        "inverse_design": InverseDesignAgent(),
    }

    if not session.history:
        session.history.append({"role": "assistant", "content": GREETING})

    await send_ws(websocket, "message", GREETING, session.state)

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            user_msg = payload.get("message", "")
            msg_type = payload.get("type", "message")
            touch(session)

            if msg_type == "cancel":
                session.request_cancel()
                continue

            if msg_type == "select_model":
                model_path = payload.get("model_path", "")
                if not model_path or not os.path.exists(model_path):
                    await send_ws(websocket, "message", "Model not found.", session.state)
                    continue
                try:
                    model_data = joblib.load(model_path)
                    session.model_path = model_path
                    session.targets = model_data.get("targets", [])
                    session.task_types = [model_data.get("task_type", "regression")]
                    session.state = "prediction"
                    targets_display = ", ".join(f"**{t}**" for t in session.targets)
                    model_name = model_data.get("best_model_name", "Unknown")
                    score = model_data.get("score", 0)
                    msg = (
                        f"Loaded saved model: **{model_name}** (score: {score:.4f})\n"
                        f"Targets: {targets_display}\n\n"
                        "You can now:\n"
                        "1. **Forward prediction** — provide alloy composition to predict properties\n"
                        "2. **Inverse design** — describe desired properties to find optimal compositions\n"
                        "3. **Batch prediction** — upload a CSV for bulk predictions"
                    )
                    session.history.append({"role": "assistant", "content": msg})
                    await send_ws(websocket, "message", msg, "prediction")
                except Exception as e:
                    await send_ws(websocket, "message", f"Failed to load model: {e}", session.state)
                continue

            if msg_type == "reset":
                for path in (session.data_path, session.clean_data_path):
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                from backend.pipeline import Session
                sessions[session_id] = Session()
                session = sessions[session_id]
                session.history.append({"role": "assistant", "content": GREETING})
                await send_ws(websocket, "message", GREETING, session.state)
                continue

            if user_msg:
                session.history.append({"role": "user", "content": user_msg})

            try:
                if session.state == "requirements":
                    if user_msg.startswith("[SYSTEM]:"):
                        if session.data_summary:
                            summary_msg = (
                                f"{session.data_summary}\n\n"
                                "Which column(s)/property(ies) would you like to predict? "
                                "You can specify **multiple targets** (e.g. 'tensile strength, elongation, and hardness') "
                                "and they'll all be predicted simultaneously by a single multi-output model.\n\n"
                                "Also, is this a **regression** (predicting a number) or "
                                "**classification** (predicting a category) task?"
                            )
                        else:
                            summary_msg = (
                                "Your file has been uploaded. "
                                "Which property(ies) would you like to predict?"
                            )
                        session.awaiting_target = True
                        session.history.append({"role": "assistant", "content": summary_msg})
                        await send_ws(websocket, "message", summary_msg, session.state)

                    elif session.awaiting_target:
                        response, targets, task_types, _, ready = await agents[
                            "requirements"
                        ].process(session.history, data_summary=session.data_summary)
                        session.history.append({"role": "assistant", "content": response})
                        await send_ws(websocket, "message", response, session.state)

                        if ready and targets:
                            session.targets = targets
                            session.task_types = task_types
                            session.has_dataset = True
                            session.ready_for_pipeline = True
                            session.awaiting_target = False

                    else:
                        response, targets, task_types, has_dataset, ready = await agents[
                            "requirements"
                        ].process(session.history)
                        session.history.append({"role": "assistant", "content": response})
                        await send_ws(websocket, "message", response, session.state)

                    if session.ready_for_pipeline and session.data_path:
                        await run_pipeline(session, websocket, agents)

                elif session.state == "prediction":
                    if user_msg.startswith("[SYSTEM]:BATCH:"):
                        batch_msg = "Processing batch predictions..."
                        await send_ws(websocket, "status", batch_msg, session.state)
                        summary, output_path = await agents["prediction"].batch_predict(
                            session.model_path, session.data_path
                        )
                        await send_ws(websocket, "message", summary, session.state)

                    elif session.pending_inverse_constraints is not None:
                        await send_ws(websocket, "status", "Running inverse design optimization...", session.state)
                        data_path = session.clean_data_path or session.data_path
                        response = await agents["inverse_design"].parse_elements_and_run(
                            user_msg, session.model_path, data_path,
                            session.pending_inverse_constraints,
                            session.pending_inverse_fixed or {},
                        )
                        session.pending_inverse_constraints = None
                        session.pending_inverse_fixed = None
                        session.history.append({"role": "assistant", "content": response})
                        await send_ws(websocket, "message", response, session.state)

                    elif is_inverse_design_request(user_msg):
                        data_path = session.clean_data_path or session.data_path
                        response, constraints, fixed = await agents["inverse_design"].extract_constraints(
                            user_msg, session.model_path, data_path
                        )
                        if constraints:
                            session.pending_inverse_constraints = constraints
                            session.pending_inverse_fixed = fixed
                        session.history.append({"role": "assistant", "content": response})
                        await send_ws(websocket, "message", response, session.state)

                    else:
                        response = await agents["prediction"].process(
                            user_msg, session.model_path, session.clean_data_path or session.data_path
                        )
                        session.history.append({"role": "assistant", "content": response})
                        await send_ws(websocket, "message", response, session.state)

                else:
                    await send_ws(
                        websocket, "message",
                        f"Processing in background... ({session.state})",
                        session.state,
                    )
            except Exception as e:
                logger.exception("Error processing message in state '%s'", session.state)
                error_msg = f"An error occurred: {e}\n\nPlease try again or reset the session."
                session.history.append({"role": "assistant", "content": error_msg})
                await send_ws(websocket, "error", error_msg, session.state)

    except WebSocketDisconnect:
        logger.info("Client %s disconnected", session_id)
