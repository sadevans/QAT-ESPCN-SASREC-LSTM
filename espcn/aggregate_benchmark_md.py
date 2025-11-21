from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_records(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return []


def fmt(value, default: str = "-") -> str:
    if value is None:
        return default
    return str(value)


def to_markdown(records: List[Dict[str, Any]]) -> str:
    columns = [
        "model",
        "quant_method",
        "psnr_y",
        "ssim",
        "throughput_samples_per_sec",
        "avg_latency_ms",
        "median_latency_ms",
        "model_size_mb",
    ]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for rec in records:
        row = [fmt(rec.get(col)) for col in columns]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ESPCN benchmark JSON to Markdown table.")
    parser.add_argument(
        "--input",
        type=str,
        default="results/espcn_quant_benchmark.json",
        help="Path to benchmark JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/espcn_quant_benchmark.md",
        help="Path to output Markdown file.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Input JSON not found: {in_path}")

    records = load_records(in_path)
    md = to_markdown(records)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)


if __name__ == "__main__":
    main()


