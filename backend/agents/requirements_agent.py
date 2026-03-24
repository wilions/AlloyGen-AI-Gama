from __future__ import annotations
import json
import logging
from backend.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RequirementsAgent(BaseAgent):
    async def process(
        self, history: list[dict], data_summary: str | None = None
    ) -> tuple[str, list[str] | None, list[str] | None, bool | None, bool]:
        """
        Extract ML requirements from conversation.
        Returns (response, targets_list, task_types_list, has_dataset, ready).
        Supports multiple targets in one message (e.g. "predict tensile strength, elongation, and hardness").
        """
        if data_summary:
            prompt = (
                "You are a metallurgical and mechanical expert assistant. "
                "The user has uploaded a dataset. Here is a summary of their data:\n\n"
                f"{data_summary}\n\n"
                "The user is now telling you which column(s)/property(ies) they want to predict. "
                "They may specify ONE or MULTIPLE targets.\n\n"
                "Based on their message, identify:\n"
                "1. The exact target column name(s) from the dataset columns listed above.\n"
                "2. For each target, the ML task type: 'regression' if the target column is numeric, "
                "'classification' if it is categorical.\n\n"
                "If you can determine the target(s), output JSON exactly like:\n"
                '{"message": "Your conversational response confirming their choice(s)", '
                '"ready": true, "targets": [{"target": "exact_column_name", "task_type": "regression"}, ...]}\n\n'
                "The targets array can have one or many entries.\n"
                "If the user's message is unclear or doesn't match any column, ask for clarification:\n"
                '{"message": "Question asking for clarification", "ready": false}'
            )
        else:
            prompt = (
                "You are a metallurgical and mechanical expert assistant. Your goal is to help the user "
                "define their machine learning goal based on their dataset. "
                "The user may want to predict ONE or MULTIPLE target properties. "
                "You need to identify: "
                "1. The target variable(s) (e.g. 'Tensile Strength', 'Hardness', etc) they want to predict. "
                "2. The ML task type for each (regression or classification). "
                "3. Whether the user has a dataset to upload or explicitly states they do not have one. "
                "If you have this information (1 and 2), output your response in JSON format exactly like: "
                '{"message": "Your conversational response", "ready": true, '
                '"targets": [{"target": "target_name", "task_type": "regression"}, ...], '
                '"has_dataset": true} '
                "The targets array can have one or many entries. "
                "If you need more info from the user about the target or task type, output: "
                '{"message": "Question asking for more info", "ready": false}'
            )

        messages = [{"role": "system", "content": prompt}]
        for msg in history[-5:]:
            messages.append({"role": msg["role"], "content": str(msg["content"])})

        raw = await self._chat(messages, json_mode=True)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON: %s", raw[:200])
            return (
                "I had trouble understanding that. Could you please rephrase which "
                "column(s) you'd like to predict and whether it's regression or classification?",
                None, None, None, False,
            )

        if result.get("ready"):
            targets_raw = result.get("targets", [])
            # Backward compat: if old format with single target/task_type
            if not targets_raw and result.get("target"):
                targets_raw = [{"target": result["target"], "task_type": result.get("task_type", "regression")}]

            targets = [t["target"] for t in targets_raw]
            task_types = [t["task_type"] for t in targets_raw]
            return (
                result.get("message", ""),
                targets,
                task_types,
                result.get("has_dataset", True),
                True,
            )
        return result.get("message", ""), None, None, None, False
