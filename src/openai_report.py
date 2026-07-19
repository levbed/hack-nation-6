from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .report import assert_aggregate_payload_safe


DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
DEFAULT_MAX_OUTPUT_TOKENS = 4000

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "executive_summary": {"type": "string"},
        "key_findings": {"type": "array", "items": {"type": "string"}},
        "evidence_strength": {"type": "string"},
        "data_quality_flags": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "responsible_use": {"type": "string"},
    },
    "required": [
        "title",
        "executive_summary",
        "key_findings",
        "evidence_strength",
        "data_quality_flags",
        "limitations",
        "responsible_use",
    ],
    "additionalProperties": False,
}


def load_aggregate_summary(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Benchmark summary must be a JSON object.")
    assert_aggregate_payload_safe(payload)
    return payload


def summarize_with_openai(
    payload: dict[str, Any],
    model: str = DEFAULT_OPENAI_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    client: Any | None = None,
) -> dict[str, Any]:
    assert_aggregate_payload_safe(payload)
    if client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for the OpenAI summary command.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ValueError("Install dependencies with: python -m pip install -r requirements.txt") from exc
        client = OpenAI()

    response = client.responses.create(
        model=model,
        store=False,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": "low"},
        instructions=(
            "You are a cautious scientific benchmark analyst. Interpret only the supplied aggregate metrics. "
            "Do not make diagnostic, fertility, treatment, causal, or participant-level claims. Distinguish a "
            "bootstrap-supported difference from an inconclusive comparison. State when added modalities do not "
            "improve over history. Never imply that this exploratory benchmark is clinically validated."
        ),
        input=json.dumps(payload, sort_keys=True, allow_nan=False),
        text={
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "cyclebench_report",
                "strict": True,
                "schema": REPORT_SCHEMA,
            }
        },
    )
    status = getattr(response, "status", None)
    if status not in (None, "completed"):
        details = getattr(response, "incomplete_details", None)
        if hasattr(details, "model_dump"):
            details = details.model_dump()
        if isinstance(details, dict):
            reason = details.get("reason", "unknown")
        else:
            reason = getattr(details, "reason", "unknown")
        raise RuntimeError(
            f"OpenAI response did not complete (status={status}, reason={reason}, "
            f"max_output_tokens={max_output_tokens})."
        )

    output_text = getattr(response, "output_text", "")
    if not output_text:
        raise RuntimeError("OpenAI returned no structured report text.")
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        request_id = getattr(response, "_request_id", None)
        request_suffix = f" Request ID: {request_id}." if request_id else ""
        raise RuntimeError(
            f"OpenAI returned malformed structured output at character {exc.pos}."
            f"{request_suffix}"
        ) from exc
    usage = getattr(response, "usage", None)
    if usage is not None and hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    result = {
        "model": model,
        "max_output_tokens": max_output_tokens,
        "request_id": getattr(response, "_request_id", None),
        "usage": usage,
        "report": parsed,
    }
    assert_aggregate_payload_safe(result)
    return result


def write_openai_outputs(
    json_path: str | Path,
    markdown_path: str | Path,
    result: dict[str, Any],
) -> None:
    assert_aggregate_payload_safe(result)
    json_destination = Path(json_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    report = result["report"]
    lines = [
        f"# {report['title']}",
        "",
        report["executive_summary"],
        "",
        "## Key Findings",
        "",
        *[f"- {item}" for item in report["key_findings"]],
        "",
        "## Evidence Strength",
        "",
        report["evidence_strength"],
        "",
        "## Data Quality Flags",
        "",
        *[f"- {item}" for item in report["data_quality_flags"]],
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
        "## Responsible Use",
        "",
        report["responsible_use"],
        "",
        f"Generated with `{result['model']}` from aggregate benchmark statistics only.",
        "",
    ]
    markdown_destination = Path(markdown_path)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.write_text("\n".join(lines))
