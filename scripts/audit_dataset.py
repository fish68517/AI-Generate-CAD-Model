"""Audit a local CADmium JSONL sample and create beginner-friendly plots."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/samples/test_sample.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/data_audit"))
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    uids = [row.get("uid") for row in rows]
    category_counts = Counter(row.get("category", "missing") for row in rows)
    primitive_counts: Counter = Counter()
    operation_counts: Counter = Counter()
    annotation_chars: list[int] = []
    json_chars: list[int] = []
    parse_errors: list[dict] = []
    schema_errors: list[dict] = []
    parsed_values: list[dict | None] = []

    for row in rows:
        annotation_chars.append(len(row.get("annotation", "")))
        json_chars.append(len(row.get("json_desc", "")))
        try:
            value = json.loads(row["json_desc"])
            parsed_values.append(value)
        except Exception as exc:
            parsed_values.append(None)
            parse_errors.append({"uid": row.get("uid"), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not isinstance(value.get("parts"), dict):
            schema_errors.append({"uid": row.get("uid"), "error": "parts 不是对象"})
            continue
        for part in value["parts"].values():
            if not part:
                continue
            operation_counts[part.get("extrusion", {}).get("operation", "missing")] += 1
            for face in part.get("sketch", {}).values():
                for loop in face.values():
                    for key in loop:
                        primitive_counts[key.split("_", 1)[0]] += 1

    token_report = {"enabled": not args.skip_tokenizer}
    token_lengths: list[int] = []
    if not args.skip_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        for row in rows:
            messages = [
                {"role": "system", "content": "You are a CAD designer."},
                {"role": "user", "content": row["annotation"]},
                {"role": "assistant", "content": row["json_desc"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            token_lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        token_report.update(
            {
                "tokenizer": args.tokenizer,
                "max_length": args.max_length,
                "min": min(token_lengths),
                "max": max(token_lengths),
                "mean": round(sum(token_lengths) / len(token_lengths), 2),
                "truncated_count": sum(length > args.max_length for length in token_lengths),
                "truncated_ratio": round(sum(length > args.max_length for length in token_lengths) / len(token_lengths), 4),
            }
        )

    report = {
        "input": str(args.input.resolve()),
        "row_count": len(rows),
        "unique_uid_count": len(set(uids)),
        "duplicate_uid_count": len(uids) - len(set(uids)),
        "empty_annotation_count": sum(not row.get("annotation") for row in rows),
        "empty_json_count": sum(not row.get("json_desc") for row in rows),
        "json_parse_success_count": len(rows) - len(parse_errors),
        "schema_error_count": len(schema_errors),
        "category_counts": dict(category_counts),
        "primitive_counts": dict(primitive_counts),
        "operation_counts": dict(operation_counts),
        "annotation_chars": {"min": min(annotation_chars), "max": max(annotation_chars), "mean": round(sum(annotation_chars) / len(annotation_chars), 2)},
        "json_chars": {"min": min(json_chars), "max": max(json_chars), "mean": round(sum(json_chars) / len(json_chars), 2)},
        "tokenization": token_report,
        "parse_errors": parse_errors,
        "schema_errors": schema_errors,
    }
    (args.output_dir / "data_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(category_counts.keys(), category_counts.values(), color="#3b82f6")
    axes[0].set_title("Sample categories")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(primitive_counts.keys(), primitive_counts.values(), color="#10b981")
    axes[1].set_title("CAD primitives")
    lengths = token_lengths if token_lengths else json_chars
    axes[2].hist(lengths, bins=min(10, max(3, len(lengths) // 2)), color="#f59e0b", edgecolor="white")
    axes[2].axvline(args.max_length, color="red", linestyle="--", label=f"limit={args.max_length}")
    axes[2].set_title("Token length" if token_lengths else "JSON character length")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "data_audit.png", dpi=180)
    plt.close(fig)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not parse_errors and not schema_errors and len(rows) == len(set(uids)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
