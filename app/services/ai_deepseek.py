from __future__ import annotations

"""
Thin client for calling a DeepSeek-compatible OpenAI-style API.

This module is intentionally minimal and does NOT change any cleaning logic.
It gives you a single entrypoint you can call from elsewhere in the app when
you want AI-assisted reasoning (e.g., for proposing categorical normalizations
for unknown domains).

Expected setup (one of):
1. Run deepseek-coder behind an OpenAI-compatible HTTP endpoint (e.g. vLLM,
   local gateway, or provider) and set:
   - DEEPSEEK_API_BASE  (e.g. http://localhost:8000/v1)
   - DEEPSEEK_API_KEY   (dummy if your server does not enforce auth)
   - DEEPSEEK_MODEL     (e.g. deepseek-coder or whatever your server exposes)

2. Or point DEEPSEEK_API_BASE/KEY/MODEL at any other OpenAI-compatible endpoint
   that serves a coding-capable model.
"""

import json
import os
from typing import Any, Dict, List, Optional

import requests


class DeepSeekConfigError(RuntimeError):
    """Raised when DeepSeek configuration is missing or invalid."""


def _get_config() -> tuple[str, str, str]:
    api_base = os.getenv("DEEPSEEK_API_BASE")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-coder")
    if not api_base or not api_key:
        raise DeepSeekConfigError(
            "DeepSeek configuration missing. Set DEEPSEEK_API_BASE and DEEPSEEK_API_KEY "
            "to point at an OpenAI-compatible endpoint serving deepseek-coder."
        )
    return api_base.rstrip("/"), api_key, model


def _chat(
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """
    Call the DeepSeek-compatible chat/completions endpoint and return the content string.
    """
    api_base, api_key, model = _get_config()
    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Unexpected DeepSeek response format: {data}") from exc


def suggest_categorical_mappings(
    column_name: str,
    sample_values: List[str],
    *,
    max_categories: int = 40,
) -> Dict[str, str]:
    """
    Ask deepseek-coder to propose canonical labels for a categorical column.

    Returns a mapping raw_value -> canonical_value. Only high-confidence mappings
    should be applied automatically; others can be surfaced as suggestions.
    """
    if not sample_values:
        return {}

    unique_values = sorted({v for v in sample_values if v is not None})
    if len(unique_values) > max_categories:
        unique_values = unique_values[:max_categories]

    examples = "\n".join(f"- {v}" for v in unique_values)
    system = (
        "You are a data cleaning assistant. Given the distinct values of a single "
        "categorical column, propose a canonical label for each value. Preserve the "
        "semantics (do NOT merge different real categories), but fix casing, spacing, "
        "and obvious typos. Respond ONLY with a JSON object mapping each raw value to "
        "its canonical value."
    )
    user = (
        f"Column name: {column_name}\n"
        f"Distinct values (subset):\n{examples}\n\n"
        "Return JSON only, like:\n"
        '{ "raw_value": "Canonical Value", "...": "..." }'
    )

    content = _chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=512,
    )

    # Try to parse a JSON object out of the response. If parsing fails, return empty.
    # This keeps the caller logic simple and safe.
    try:
        # Allow responses that contain extra text around the JSON by finding the first
        # JSON object in the content.
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
            return {}
        json_str = content[first_brace : last_brace + 1]
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}

