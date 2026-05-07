#!/usr/bin/env python3
"""Local compression runner for miner/miner.py.

This accepts JSON, JSONL, or plain text input. If the input file is normal text,
the entire file is passed to miner.main() as the task string.

Examples:
    python3 test.py --input input.txt --ratio 0.2
    python3 test.py --input input.json --output output.json --ratio 0.07
    python3 test.py --input miner/sample_tasks/cot_compression_tasks.jsonl --index 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output.json"
MINER_PATH = ROOT / "miner" / "miner.py"


def load_miner():
    spec = importlib.util.spec_from_file_location("local_soma_miner", MINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load miner from {MINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise RuntimeError("miner/miner.py must define main(task, compression_ratio)")
    return module


def extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        if any(key in payload for key in ("messages", "conversation", "events")):
            return json.dumps(payload, ensure_ascii=False)
        if "file_name" in payload and isinstance(payload.get("text"), str):
            return json.dumps(payload, ensure_ascii=False)
        if any(
            key in payload
            for key in (
                "scenario_title",
                "problem_statement",
                "environment_context",
                "input_request",
                "agent_expected_actions",
                "edge_cases_and_risks",
                "success_criteria",
            )
        ):
            return json.dumps(payload, ensure_ascii=False)
        for key in ("source_text", "text", "task", "prompt", "context"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(payload, ensure_ascii=False)
    if isinstance(payload, list):
        return json.dumps(payload, ensure_ascii=False)
    return str(payload)


def load_input(path: Path, index: int) -> str:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()
    if not stripped:
        return ""

    # JSON object/string/array.
    if stripped[0] in "[{":
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            if isinstance(payload, list):
                if not payload:
                    return ""
                if index < 0 or index >= len(payload):
                    raise IndexError(f"--index {index} out of range for {len(payload)} items")
                return extract_text(payload[index])
            return extract_text(payload)

    # JSONL: use the selected line if every non-empty line is JSON-like enough.
    lines = [line for line in raw.splitlines() if line.strip()]
    if lines and all(line.lstrip().startswith(("{", "[", '"')) for line in lines[: min(5, len(lines))]):
        if index < 0 or index >= len(lines):
            raise IndexError(f"--index {index} out of range for {len(lines)} JSONL lines")
        try:
            return extract_text(json.loads(lines[index]))
        except json.JSONDecodeError:
            pass

    # Plain text fallback.
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Compress a local task file with miner/miner.py")
    parser.add_argument("--input", type=Path, required=True, help="JSON, JSONL, or plain text task file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument("--ratio", type=float, default=0.2, help="Compression ratio in (0, 1]")
    parser.add_argument("--index", type=int, default=0, help="Item/line index for JSON arrays or JSONL")
    args = parser.parse_args()

    if args.ratio <= 0 or args.ratio > 1:
        raise ValueError("--ratio must be in (0, 1]")
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    miner = load_miner()
    source_text = load_input(args.input, args.index)
    compressed_text = miner.main(source_text, args.ratio)

    source_tokens = miner.token_count(source_text) if hasattr(miner, "token_count") else None
    compressed_tokens = (
        miner.token_count(compressed_text) if hasattr(miner, "token_count") else None
    )
    token_limit = int(source_tokens * args.ratio) if source_tokens is not None else None

    result = {
        "input": str(args.input),
        "index": args.index,
        "compression_ratio": args.ratio,
        "source_tokens": source_tokens,
        "compressed_tokens": compressed_tokens,
        "token_limit": token_limit,
        "within_budget": (
            compressed_tokens <= token_limit
            if compressed_tokens is not None and token_limit is not None
            else None
        ),
        "compressed_text": compressed_text,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {args.output}")
    if source_tokens is not None and compressed_tokens is not None:
        print(f"tokens: {compressed_tokens}/{token_limit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
