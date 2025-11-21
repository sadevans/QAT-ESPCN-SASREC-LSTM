
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CPU metrics across SASRec models.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["results/*sasrec*_cpu_eval.json"],
        help="Glob(s) pointing to CPU evaluation JSON files.",
    )
    parser.add_argument(
        "--baseline",
        default="sasrec_fp32",
        help="Model name used as the NDCG/latency reference.",
    )
    parser.add_argument(
        "--csv-out",
        default="results/cpu_metrics_summary_sasrec.csv",
        help="Where to store the CSV summary.",
    )
    parser.add_argument(
        "--md-out",
        default="results/cpu_metrics_summary_sasrec.md",
        help="Where to store the Markdown table.",
    )
    return parser.parse_args()


def collect_files(patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        for match in glob.glob(pattern):
            path = Path(match)
            if path.is_file():
                files.append(path)
    if not files:
        print(f"[warn] No input files found for patterns: {patterns}", file=sys.stderr)
        return []
    return sorted(files)


def load_record(path: Path) -> Dict[str, Union[float, str]]:
    payload = json.loads(path.read_text())
    run_name = payload.get("run_name", path.stem.replace("_cpu_eval", ""))
    
    quant_method = payload.get("quant_method", "unknown")
    if quant_method == "unknown":
        lower_name = run_name.lower()
        if "lsq" in lower_name:
            quant_method = "LSQ"
        elif "apot" in lower_name:
            quant_method = "APoT"
        elif "qdrop" in lower_name:
            quant_method = "QDrop"
        elif "adaround" in lower_name:
            quant_method = "AdaRound"
        elif "fp32" in lower_name:
            quant_method = "FP32"
        else:
            quant_method = "Other"

    ndcg = float(payload.get("ndcg", 0.0))
    hit = float(payload.get("hit", 0.0))
    
    throughput = float(payload.get("throughput_samples_per_sec", 0.0))
    avg_latency = float(payload.get("avg_latency_ms", 0.0))
    median_latency = float(payload.get("median_latency_ms", avg_latency))

    return {
        "model": run_name,
        "quant_method": quant_method,
        "ndcg": ndcg,
        "hit": hit,
        "throughput_sps": throughput,
        "avg_latency_ms": avg_latency,
        "median_latency_ms": median_latency,
    }


def annotate_against_baseline(records: List[Dict[str, Union[float, str]]], baseline_name: str):
    baseline = next((rec for rec in records if rec["model"] == baseline_name), None)
    
    if baseline is None:
        # Try partial match if exact match fails
        baseline = next((rec for rec in records if baseline_name in str(rec["model"])), None)
        
    if baseline is None:
        print(f"[warn] baseline '{baseline_name}' not found; skipping delta columns.", file=sys.stderr)
        for rec in records:
            rec["ndcg_delta"] = None
            rec["throughput_speedup"] = None
            rec["latency_ratio"] = None
        return

    for rec in records:
        rec["ndcg_delta"] = float(rec["ndcg"]) - float(baseline["ndcg"])
        
        base_throughput = float(baseline["throughput_sps"])
        curr_throughput = float(rec["throughput_sps"])
        if base_throughput > 0:
            rec["throughput_speedup"] = curr_throughput / base_throughput
        else:
            rec["throughput_speedup"] = 0.0
            
        base_latency = float(baseline["avg_latency_ms"])
        curr_latency = float(rec["avg_latency_ms"])
        if curr_latency > 0:
            rec["latency_ratio"] = base_latency / curr_latency
        else:
            rec["latency_ratio"] = 0.0


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(records: Sequence[Dict[str, Union[float, str]]], path: Path, columns: Sequence[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for rec in records:
            row = []
            for col in columns:
                value = rec.get(col)
                if value is None:
                    row.append("")
                else:
                    row.append(f"{value}")
            handle.write(",".join(row) + "\n")


def fmt_value(column: str, value) -> str:
    if value is None:
        return "-"
    numeric_formats = {
        "ndcg": "{:.4f}",
        "hit": "{:.4f}",
        "throughput_sps": "{:.2f}",
        "avg_latency_ms": "{:.3f}",
        "median_latency_ms": "{:.3f}",
        "ndcg_delta": "{:+.4f}",
        "throughput_speedup": "{:.2f}",
        "latency_ratio": "{:.2f}",
    }
    if column in numeric_formats:
        try:
            return numeric_formats[column].format(float(value))
        except (TypeError, ValueError):
            return "-"
    return str(value)


def render_table(records: Sequence[Dict[str, Union[float, str]]], columns: Sequence[str]) -> str:
    if not records:
        return "No records to display."
        
    widths = {col: len(col) for col in columns}
    for rec in records:
        for col in columns:
            widths[col] = max(widths[col], len(fmt_value(col, rec.get(col))))

    def render_row(values: Iterable[str]) -> str:
        parts = []
        for col, value in zip(columns, values):
            parts.append(value.ljust(widths[col]))
        return " | ".join(parts)

    lines = [render_row(columns)]
    lines.append(" | ".join("-" * widths[col] for col in columns))
    for rec in records:
        lines.append(render_row(fmt_value(col, rec.get(col)) for col in columns))
    return "\n".join(lines)


def write_markdown(records: Sequence[Dict[str, Union[float, str]]], path: Path, columns: Sequence[str]) -> None:
    ensure_parent(path)
    if not records:
        path.write_text("No records found.")
        return
        
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for rec in records:
        values = [fmt_value(col, rec.get(col)) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    files = collect_files(args.inputs)
    if not files:
        print("No result files found. Exiting.")
        return

    records = [load_record(path) for path in files]
    annotate_against_baseline(records, args.baseline)
    
    columns = [
        "model",
        "quant_method",
        "ndcg",
        "hit",
        "throughput_sps",
        "avg_latency_ms",
        "ndcg_delta",
        "throughput_speedup",
        "latency_ratio",
    ]
    
    csv_path = Path(args.csv_out)
    write_csv(records, csv_path, columns)
    
    md_path = Path(args.md_out)
    write_markdown(records, md_path, columns)
    
    print(render_table(records, columns))
    print(f"\nSaved CSV to {csv_path} and Markdown to {md_path}")


if __name__ == "__main__":
    main()

