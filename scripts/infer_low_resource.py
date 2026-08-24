"""Low-resource CADmium inference probe with explicit base + LoRA loading."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import psutil

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("生成文本中没有完整 JSON 对象")
    return json.loads(text[start : end + 1])


def load_system_message(repo_root: Path) -> str:
    import sys

    sys.path.insert(0, str(repo_root / "CADmium"))
    from cadmium.src.utils.prompts import SYSTEM_MESSAGE  # type: ignore

    return SYSTEM_MESSAGE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="chandar-lab/CADmium-1.5B")
    parser.add_argument("--base-model", default=None, help="可选的本地基座目录；省略时读取 adapter_config")
    parser.add_argument("--prompt", default="Create a rectangular prism with length 0.75, width 0.5 and height 0.2.")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--mode", choices=("metadata", "dry-run", "cpu-fp32", "cpu-int8", "cuda-4bit"), default="dry-run")
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/inference"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.prompt_file is not None:
        args.prompt = args.prompt_file.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    started = time.perf_counter()
    attempt = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapter": args.adapter,
        "mode": args.mode,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "prompt": args.prompt,
        "success": False,
    }
    output_path = args.output_dir / f"attempt_{args.mode}.json"
    try:
        adapter_source = Path(args.adapter)
        if adapter_source.exists():
            adapter_path = adapter_source / "adapter_config.json"
        else:
            adapter_path = Path(hf_hub_download(args.adapter, "adapter_config.json"))
        adapter_config = json.loads(adapter_path.read_text(encoding="utf-8"))
        base_model = args.base_model or adapter_config["base_model_name_or_path"]
        attempt["adapter_config"] = adapter_config
        attempt["base_model"] = base_model

        from transformers import AutoTokenizer

        # The published adapter contains a tokenizer.json produced by a newer
        # tokenizers build than the paper-era stack. The base tokenizer is the
        # authoritative compatible tokenizer and was also used for training.
        tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
        system_message = load_system_message(args.repo_root.resolve())
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": args.prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        raw_input_tokens = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        )
        attempt["raw_input_tokens"] = raw_input_tokens
        attempt["input_tokens"] = int(encoded["input_ids"].shape[1])
        attempt["input_truncated"] = raw_input_tokens > args.max_input_tokens
        (args.output_dir / "input_prompt.txt").write_text(prompt_text, encoding="utf-8")

        if args.mode in {"metadata", "dry-run"}:
            attempt["success"] = True
            attempt["result"] = "adapter 元数据与 prompt/tokenizer 链路验证通过，未加载 1.5B 基座权重。"
        else:
            import torch
            import inspect
            from peft import LoraConfig, PeftModel
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            if args.mode == "cuda-4bit":
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        f"当前 PyTorch={torch.__version__}，CUDA runtime={torch.version.cuda}，无法进行 CUDA 4bit 推理"
                    )
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    quantization_config=quantization_config,
                    device_map={"": 0},
                    low_cpu_mem_usage=True,
                )
                device = torch.device("cuda:0")
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float32,
                    device_map={"": "cpu"},
                    low_cpu_mem_usage=True,
                )
                device = torch.device("cpu")

            # Published adapter_config.json can gain fields from newer PEFT
            # versions (for example corda_config). Filter metadata against the
            # installed paper-era LoraConfig instead of editing downloaded data.
            supported = set(inspect.signature(LoraConfig).parameters)
            compatible_adapter_config = {
                key: value
                for key, value in adapter_config.items()
                if key in supported and key not in {"peft_type"}
            }
            lora_config = LoraConfig(**compatible_adapter_config)
            attempt["ignored_adapter_config_keys"] = sorted(set(adapter_config) - supported)
            model = PeftModel.from_pretrained(model, args.adapter, config=lora_config)
            if args.mode == "cpu-int8":
                model = model.merge_and_unload()
                model = torch.ao.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            model.eval()
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generation_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, encoded["input_ids"].shape[1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            generation_seconds = time.perf_counter() - generation_started
            (args.output_dir / f"response_{args.mode}.txt").write_text(response, encoding="utf-8")
            attempt.update(
                {
                    "generated_tokens": int(new_tokens.shape[0]),
                    "generation_seconds": round(generation_seconds, 3),
                    "tokens_per_second": round(float(new_tokens.shape[0]) / generation_seconds, 4),
                    "response": response,
                }
            )
            try:
                parsed = extract_json(response)
                (args.output_dir / f"generated_{args.mode}.json").write_text(
                    json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                attempt["json_parse_success"] = True
                try:
                    import sys

                    sys.path.insert(0, str((args.repo_root / "scripts").resolve()))
                    from reconstruct_json import reconstruct_record

                    attempt["reconstruction"] = reconstruct_record(
                        f"inference_{args.mode}",
                        parsed,
                        args.output_dir / "reconstruction",
                        category="model_prediction",
                        export_html=True,
                    )
                except Exception as reconstruction_exc:
                    attempt["reconstruction"] = {
                        "status": "S3",
                        "success": False,
                        "error": f"{type(reconstruction_exc).__name__}: {reconstruction_exc}",
                    }
            except Exception as parse_exc:
                attempt["json_parse_success"] = False
                attempt["json_parse_error"] = f"{type(parse_exc).__name__}: {parse_exc}"
            attempt["success"] = True
            attempt["resident_memory_mb"] = round(psutil.Process(os.getpid()).memory_info().rss / 1024**2, 1)
    except Exception as exc:
        attempt["error_type"] = type(exc).__name__
        attempt["error"] = str(exc)
        attempt["traceback"] = traceback.format_exc(limit=8)
    finally:
        attempt["total_seconds"] = round(time.perf_counter() - started, 3)
        output_path.write_text(json.dumps(attempt, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(attempt, indent=2, ensure_ascii=False))
        print(f"\n推理记录已保存：{output_path.resolve()}")
    return 0 if attempt["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
