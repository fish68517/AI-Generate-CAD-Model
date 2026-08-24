"""Resumable CADmium base+adapter batch inference for the cloud GPU stage.

The script is intentionally importable and supports ``--dry-run`` on the local
CPU machine.  Real inference requires a CUDA PyTorch build and at least 24 GB
VRAM for the planned unquantized 1.5B validation run.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("生成文本中没有完整 JSON 对象")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("生成 JSON 顶层不是对象")
    return value


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    required = {"uid", "annotation"}
    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"第 {index} 行缺少字段：{sorted(missing)}")
    uids = [row["uid"] for row in rows]
    if len(set(uids)) != len(uids):
        raise ValueError("输入中存在重复 uid")
    return rows[:limit] if limit is not None else rows


def completed_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        completed.add(json.loads(line)["uid"])
    return completed


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def build_summary(records: list[dict], total_input: int) -> dict:
    attempted = len(records)
    parse_success = sum(bool(record.get("json_parse_success")) for record in records)
    generation_times = [float(record["generation_seconds"]) for record in records if "generation_seconds" in record]
    return {
        "total_input": total_input,
        "attempted": attempted,
        "generation_success": sum(bool(record.get("generation_success")) for record in records),
        "json_parse_success": parse_success,
        "json_parse_rate": parse_success / attempted if attempted else 0.0,
        "input_truncated": sum(bool(record.get("input_truncated")) for record in records),
        "mean_generation_seconds": statistics.mean(generation_times) if generation_times else None,
        "p50_generation_seconds": percentile(generation_times, 0.50),
        "p95_generation_seconds": percentile(generation_times, 0.95),
        "status_note": "原始回答与失败记录均计入分母；几何重建和论文指标在后续步骤计算。",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/samples/test_sample.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/gpu_infer_1p5b_20"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter", default="chandar-lab/CADmium-1.5B")
    parser.add_argument("--base-revision")
    parser.add_argument("--adapter-revision")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 1:
        raise ValueError("当前实现为保证逐 uid 断点记录，仅支持 --batch-size 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.input, args.limit)
    predictions_path = args.output_dir / "predictions.jsonl"
    existing_uids = completed_uids(predictions_path)
    pending = [row for row in rows if row["uid"] not in existing_uids]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "base_model": args.base_model,
        "adapter": args.adapter,
        "base_revision": args.base_revision,
        "adapter_revision": args.adapter_revision,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "total_rows": len(rows),
        "already_completed": len(existing_uids & {row['uid'] for row in rows}),
        "pending_rows": len(pending),
        "python": platform.python_version(),
        "dry_run": args.dry_run,
    }
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.dry_run:
        manifest["validated_uids"] = [row["uid"] for row in rows]
        manifest["result"] = "输入、uid、断点目录和参数验证通过；未加载模型，未写 predictions.jsonl。"
        (args.output_dir / "dry_run_summary.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    import sys

    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError(f"真实批量推理需要 CUDA；当前 torch={torch.__version__}, CUDA={torch.version.cuda}")
    if torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
        raise RuntimeError("检测到显存不足 20 GiB；本阶段正式批量验证要求至少 24GB 标称显存")

    sys.path.insert(0, str((args.repo_root.resolve() / "CADmium")))
    from cadmium.src.utils.prompts import SYSTEM_MESSAGE  # type: ignore

    set_seed(args.seed)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    base_common = {
        "revision": args.base_revision,
        "local_files_only": args.local_files_only,
    }
    base_common = {key: value for key, value in base_common.items() if value is not None}
    adapter_common = {
        "revision": args.adapter_revision,
        "local_files_only": args.local_files_only,
    }
    adapter_common = {key: value for key, value in adapter_common.items() if value is not None}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left", **base_common)
    model_kwargs = {"device_map": {"": 0}, "low_cpu_mem_usage": True, **base_common}
    if args.quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    adapter_config_path = Path(args.adapter) / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    else:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            args.adapter,
            "adapter_config.json",
            revision=args.adapter_revision,
            local_files_only=args.local_files_only,
        )
        adapter_config = json.loads(Path(downloaded).read_text(encoding="utf-8"))
    supported = set(inspect.signature(LoraConfig).parameters)
    compatible = {key: value for key, value in adapter_config.items() if key in supported and key != "peft_type"}
    model = PeftModel.from_pretrained(
        model, args.adapter, config=LoraConfig(**compatible), **adapter_common
    )
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    for row in pending:
        record = {"uid": row["uid"], "category": row.get("category", "unknown")}
        try:
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": row["annotation"]},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            raw_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            )
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, encoded["input_ids"].shape[1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            record.update(
                {
                    "generation_success": True,
                    "raw_input_tokens": raw_tokens,
                    "input_tokens": int(encoded["input_ids"].shape[1]),
                    "input_truncated": raw_tokens > args.max_input_tokens,
                    "generated_tokens": int(new_tokens.shape[0]),
                    "generation_seconds": round(time.perf_counter() - started, 4),
                    "raw_response": response,
                }
            )
            try:
                record["generated_json"] = extract_json(response)
                record["json_parse_success"] = True
            except Exception as exc:
                record["json_parse_success"] = False
                record["json_parse_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            record.update(
                {
                    "generation_success": False,
                    "json_parse_success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        append_jsonl(predictions_path, record)

    records = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line]
    selected_uids = {row["uid"] for row in rows}
    records = [record for record in records if record["uid"] in selected_uids]
    summary = build_summary(records, len(rows))
    summary.update(
        {
            "gpu": torch.cuda.get_device_name(0),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["attempted"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
