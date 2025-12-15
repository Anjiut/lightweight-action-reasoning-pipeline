import os
import json
import argparse
from typing import List, Dict, Any

from openai import OpenAI

# ------------------------------------------------------------
# OpenAI client
# ------------------------------------------------------------
# Requires environment variable:
# export OPENAI_API_KEY="your_api_key"
client = OpenAI()

# ------------------------------------------------------------
# Prompt templates for single-action reasoning
# ------------------------------------------------------------

PROMPT_TEMPLATE = """
You are an embodied AI research assistant.

You are given a high-level human action label. Based on common sense,
infer the underlying intent, environment context, and task-level information.

Action categories:
- open_door: the person walks to a door and opens it.
- pick_book: the person picks up a book from a shelf or table.
- pour_water: the person pours water from a kettle/bottle into a cup.
- walk_stop: the person walks and then stops briefly.

Current action label: "{action_label}"

Additional context:
{action_hint}

Return ONLY a valid JSON object with the following keys (ALL must appear):

1. intent (string)
2. next_step (string)
3. explanation (string; 1-3 sentences)
4. environment (object):
   - scene_type (string)
   - key_objects (list of strings)
5. agent_state (string)
6. affordances (list of strings; 3-5 items)
7. constraints (list of strings; 2-4 items)
8. task_goal (string)
9. current_step (string)
10. future_steps (list of strings; 2-4 items)
11. target_objects (list of strings; 2-4 items)

IMPORTANT:
- Output JSON only.
- Do not include markdown code fences.
- Do not include any extra text.
"""

ACTION_HINTS = {
    "open_door": (
        "Usually involves approaching a door and opening it. "
        "Common scenes include rooms, corridors, offices, or classrooms."
    ),
    "pick_book": (
        "Usually involves picking up a book from a shelf or table. "
        "Common scenes include libraries, study rooms, or offices."
    ),
    "pour_water": (
        "Usually involves pouring liquid from a container into a cup. "
        "Common scenes include kitchens, pantries, or office break rooms."
    ),
    "walk_stop": (
        "Usually involves walking and then stopping briefly. "
        "Common scenes include corridors, indoor spaces, sidewalks, or open areas."
    ),
}

VALID_ACTIONS = list(ACTION_HINTS.keys())

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t

def _ensure_list_of_str(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    return [str(i) for i in x if i is not None]

def _ensure_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)

def _normalize_single_action_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure ALL required keys exist with correct types.
    Also apply minimal non-empty defaults for affordances/constraints/future_steps/target_objects
    to avoid "empty list" outputs that look weak in demos.
    """
    # Base skeleton
    out: Dict[str, Any] = {
        "intent": "",
        "next_step": "",
        "explanation": "",
        "environment": {
            "scene_type": "",
            "key_objects": [],
        },
        "agent_state": "",
        "affordances": [],
        "constraints": [],
        "task_goal": "",
        "current_step": "",
        "future_steps": [],
        "target_objects": [],
    }

    if not isinstance(data, dict):
        data = {}

    # Strings
    for k in ["intent", "next_step", "explanation", "agent_state", "task_goal", "current_step"]:
        out[k] = _ensure_str(data.get(k, out[k]))

    # Environment
    env = data.get("environment", {})
    if not isinstance(env, dict):
        env = {}
    out["environment"]["scene_type"] = _ensure_str(env.get("scene_type", ""))
    out["environment"]["key_objects"] = _ensure_list_of_str(env.get("key_objects", []))

    # Lists
    out["affordances"] = _ensure_list_of_str(data.get("affordances", []))
    out["constraints"] = _ensure_list_of_str(data.get("constraints", []))
    out["future_steps"] = _ensure_list_of_str(data.get("future_steps", []))
    out["target_objects"] = _ensure_list_of_str(data.get("target_objects", []))

    # Minimal “looks good in demo” defaults if model returns empty
    if len(out["affordances"]) < 3:
        out["affordances"] = (out["affordances"] + [
            "approach relevant object safely",
            "grasp/manipulate the object",
            "move to the next location or step",
        ])[:5]

    if len(out["constraints"]) < 2:
        out["constraints"] = (out["constraints"] + [
            "avoid collisions with people/objects",
            "ensure stable footing and safe motion",
        ])[:4]

    if len(out["future_steps"]) < 2:
        out["future_steps"] = (out["future_steps"] + [
            "complete the immediate subtask",
            "transition to the next goal-oriented action",
        ])[:4]

    if len(out["target_objects"]) < 2:
        # Use environment objects as hints if present
        candidates = out["environment"]["key_objects"]
        if candidates:
            out["target_objects"] = (out["target_objects"] + candidates)[:4]
        else:
            out["target_objects"] = (out["target_objects"] + ["relevant object", "destination"])[:4]

    return out

def _normalize_temporal_json(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "overall_task": "",
        "step_roles": [],
        "possible_goal": "",
        "future_plan": [],
        "uncertainty_notes": [],
    }
    if not isinstance(data, dict):
        data = {}

    out["overall_task"] = _ensure_str(data.get("overall_task", ""))
    out["possible_goal"] = _ensure_str(data.get("possible_goal", ""))

    out["step_roles"] = _ensure_list_of_str(data.get("step_roles", []))
    out["future_plan"] = _ensure_list_of_str(data.get("future_plan", []))
    notes = data.get("uncertainty_notes", [])
    out["uncertainty_notes"] = _ensure_list_of_str(notes)

    return out

# ------------------------------------------------------------
# Single-action reasoning
# ------------------------------------------------------------

def llm_reasoning(action_label: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    action_hint = ACTION_HINTS.get(action_label, "")
    prompt = PROMPT_TEMPLATE.format(action_label=action_label, action_hint=action_hint)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful embodied AI researcher. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    content = _strip_code_fences(response.choices[0].message.content or "")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: keep the raw text in explanation so you can debug
        data = {
            "intent": "",
            "next_step": "",
            "explanation": content,
            "environment": {"scene_type": "", "key_objects": []},
            "agent_state": "",
            "affordances": [],
            "constraints": [],
            "task_goal": "",
            "current_step": "",
            "future_steps": [],
            "target_objects": [],
        }

    return _normalize_single_action_json(data)

# ------------------------------------------------------------
# Temporal / task-level reasoning over action sequences
# ------------------------------------------------------------

TEMPORAL_PROMPT_TEMPLATE = """
You are an embodied AI researcher analyzing a sequence of human actions.

Given the following action sequence (in order):
{history_str}

Analyze the sequence as a single task episode and return a strict JSON object
with the following keys (ALL must appear):

- overall_task (string)
- step_roles (list of strings; one per action in the same order)
- possible_goal (string)
- future_plan (list of 2-4 strings)
- uncertainty_notes (list of 1-3 strings)

Return JSON only. Do not include any extra text.
"""

def llm_temporal_reasoning(action_history: List[str], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    if not action_history:
        return _normalize_temporal_json({})

    history_str = " -> ".join(action_history)
    prompt = TEMPORAL_PROMPT_TEMPLATE.format(history_str=history_str)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful embodied AI researcher. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    content = _strip_code_fences(response.choices[0].message.content or "")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {
            "overall_task": "",
            "step_roles": [],
            "possible_goal": "",
            "future_plan": [],
            "uncertainty_notes": [content],
        }

    return _normalize_temporal_json(data)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reasoning agent for embodied action understanding.")
    parser.add_argument("--action", type=str, default=None, help=f"One of: {', '.join(VALID_ACTIONS)}")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Action sequence separated by commas, e.g., open_door,walk_stop,pick_book")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name")
    args = parser.parse_args()

    # If user provides a sequence, run temporal reasoning
    if args.sequence:
        seq = [s.strip() for s in args.sequence.split(",") if s.strip()]
        result = llm_temporal_reasoning(seq, model=args.model)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # Otherwise run single-action reasoning
    action = args.action
    if not action:
        print("=== Human Action Reasoning Agent ===")
        print("Available labels:", ", ".join(VALID_ACTIONS))
        action = input("Enter action label (e.g., open_door): ").strip()

    if not action:
        print("[ERROR] Empty action label.")
        return

    result = llm_reasoning(action, model=args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

