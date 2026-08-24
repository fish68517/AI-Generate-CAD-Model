"""Create deterministic raw and tokenized small-sample LoRA splits.

Unlike the public tokenizer, padding positions are explicitly masked with -100.
This script is suitable for preparing and auditing data locally before upload;
it does not start training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        if not all(key in row for key in ("uid", "annotation", "json_desc")):
            raise ValueError("每条记录必须包含 uid、annotation 和 json_desc")
    if len({row["uid"] for row in rows}) != len(rows):
        raise ValueError("输入中存在重复 uid")
    return rows


def stable_category_order(rows: list[dict], seed: int) -> list[dict]:
    """Hash-sort within categories and interleave categories for balanced prefixes."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row.get("category", "unknown"), []).append(row)
    for category, values in groups.items():
        values.sort(
            key=lambda row: hashlib.sha256(f"{seed}|{category}|{row['uid']}".encode()).hexdigest()
        )
    ordered: list[dict] = []
    categories = sorted(groups)
    while any(groups.values()):
        for category in categories:
            if groups[category]:
                ordered.append(groups[category].pop(0))
    return ordered


def split_rows(
    rows: list[dict], train_size: int, validation_size: int, test_size: int, seed: int
) -> dict[str, list[dict]]:
    requested = train_size + validation_size + test_size
    if requested > len(rows):
        raise ValueError(f"请求 {requested} 条，但输入只有 {len(rows)} 条")
    ordered = stable_category_order(rows, seed)
    train_end = train_size
    validation_end = train_end + validation_size
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end : validation_end + test_size],
    }


def tokenize_row(row: dict, tokenizer, system_message: str, max_length: int) -> tuple[dict, dict]:
    prompt_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": row["annotation"]},
    ]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": row["json_desc"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    raw_prompt_tokens = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    raw_full_tokens = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
    encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        add_special_tokens=False,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    prompt_boundary = min(raw_prompt_tokens, max_length)
    labels = list(input_ids)
    for index in range(prompt_boundary):
        labels[index] = -100
    for index, attended in enumerate(attention_mask):
        if not attended:
            labels[index] = -100
    supervised_tokens = sum(value != -100 for value in labels)
    output = {
        "uid": row["uid"],
        "prompt": full_text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    audit = {
        "uid": row["uid"],
        "category": row.get("category", "unknown"),
        "raw_prompt_tokens": raw_prompt_tokens,
        "raw_full_tokens": raw_full_tokens,
        "truncated": raw_full_tokens > max_length,
        "supervised_tokens": supervised_tokens,
        "padding_tokens": attention_mask.count(0),
        "padding_labels_not_masked": sum(
            not attended and label != -100 for attended, label in zip(attention_mask, labels)
        ),
    }
    return output, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/samples/test_sample.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cloud_smoke"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--tokenizer", default="models/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--train-size", type=int, default=10)
    parser.add_argument("--validation-size", type=int, default=5)
    parser.add_argument("--test-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = split_rows(
        load_rows(args.input),
        args.train_size,
        args.validation_size,
        args.test_size,
        args.seed,
    )
    manifest = {
        "input": str(args.input.resolve()),
        "seed": args.seed,
        "max_length": args.max_length,
        "tokenizer": args.tokenizer,
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "splits": {},
        "data_leakage_warning": (
            "默认输入是固定 test_sample.jsonl，因此这些切分只用于工程 smoke test；"
            "不得用于论文指标或正式训练报告。正式 LoRA 必须从 CADmium-ds train/validation 切分生成。"
        ),
    }
    for split_name, split_values in splits.items():
        raw_path = args.output_dir / f"{split_name}.jsonl"
        raw_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_values),
            encoding="utf-8",
        )
        manifest["splits"][split_name] = {
            "uids": [row["uid"] for row in split_values],
            "category_counts": {
                category: sum(row.get("category", "unknown") == category for row in split_values)
                for category in sorted({row.get("category", "unknown") for row in split_values})
            },
            "raw_jsonl": str(raw_path.resolve()),
        }

    if not args.raw_only:
        from datasets import Dataset
        from transformers import AutoTokenizer

        sys.path.insert(0, str((args.repo_root.resolve() / "CADmium")))
        from cadmium.src.utils.prompts import SYSTEM_MESSAGE  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        for split_name, split_values in splits.items():
            tokenized = []
            audits = []
            for row in split_values:
                output, audit = tokenize_row(row, tokenizer, SYSTEM_MESSAGE, args.max_length)
                tokenized.append(output)
                audits.append(audit)
            parquet_path = args.output_dir / f"{split_name}.parquet"
            Dataset.from_list(tokenized).to_parquet(parquet_path)
            audit_path = args.output_dir / f"{split_name}_token_audit.json"
            audit_path.write_text(json.dumps(audits, indent=2, ensure_ascii=False), encoding="utf-8")
            supervised = [item["supervised_tokens"] for item in audits]
            manifest["splits"][split_name].update(
                {
                    "parquet": str(parquet_path.resolve()),
                    "token_audit": str(audit_path.resolve()),
                    "truncated_count": sum(item["truncated"] for item in audits),
                    "zero_supervised_count": sum(value == 0 for value in supervised),
                    "padding_labels_not_masked": sum(
                        item["padding_labels_not_masked"] for item in audits
                    ),
                    "supervised_tokens_min": min(supervised),
                    "supervised_tokens_mean": round(statistics.mean(supervised), 2),
                    "supervised_tokens_max": max(supervised),
                }
            )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    has_bad_labels = any(
        info.get("padding_labels_not_masked", 0) or info.get("zero_supervised_count", 0)
        for info in manifest["splits"].values()
    )
    return 1 if has_bad_labels else 0


if __name__ == "__main__":
    raise SystemExit(main())
