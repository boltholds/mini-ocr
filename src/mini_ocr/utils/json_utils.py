from __future__ import annotations

import json
import re
from typing import Any


def loads_json_relaxed(content: str) -> dict[str, Any]:
    """Parse LLM JSON output with small, explicit repairs.

    This helper accepts plain JSON, fenced JSON blocks, and responses with a
    single JSON object surrounded by prose. It also repairs common invalid
    backslash escapes produced by local LLMs while preserving valid JSON escapes.
    """
    text = (content or "").strip()
    if not text:
        raise json.JSONDecodeError("empty JSON content", text, 0)

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        obj = match.group(0)
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', obj)
            return json.loads(repaired)
